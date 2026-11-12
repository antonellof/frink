//! `--model-draft` on the server: a second, smaller checkpoint that
//! drafts tokens for speculative decoding, as llama-server's `-md`.
//!
//! The weights are loaded once and shared ([`frink_models::DraftWeights`]),
//! and every request gets a [`frink_models::DraftModelSpeculator`] with
//! KV of its own. The server verifies a draft by agreement: a drafted
//! token is kept only when it is the token the target's own sampler
//! draws there (`crate::sampling_loop`). So the output is the output
//! without a drafter, whatever the drafter proposes, and a good drafter
//! only saves forward passes.
//!
//! A DFlash draft (llama.cpp's `dflash` architecture) is picked from
//! the file as `frink run -md` picks it. It reads the target's layer
//! inputs, so the private loop feeds it the features of every position
//! it commits, starting with the prompt's; a prefix-cache hit has none
//! to give, and the two are refused together.
//!
//! Speculation runs on the private decode loop and in the continuous
//! batch, where every row's block is verified in one forward
//! (`crate::serving::batch` and its `speculate`). Where the server could
//! not honour the flag, it refuses at startup rather than accepting it
//! and serving without it. The paged store rolls a rejected draft back by rewinding its tables,
//! except under a sliding window, where that request decodes without
//! drafting. A recurrent target is served: its caches are checkpointed
//! before each verify batch and restored and replayed on a rejection.

use std::sync::Arc;

use anyhow::{bail, Context};
use frink_models::Decoder;

use crate::model::LoadedModel;

/// The draft checkpoint's path (`--model-draft`, `-md`).
pub(crate) const DRAFT_ENV: &str = "FRINK_MODEL_DRAFT";
/// Tokens drafted per verification step (`--draft-max`).
pub(crate) const DRAFT_MAX_ENV: &str = "FRINK_DRAFT_MAX";
/// The drafter's own probability below which it stops a block
/// (`--draft-p-min`).
pub(crate) const DRAFT_P_MIN_ENV: &str = "FRINK_DRAFT_P_MIN";
/// llama.cpp's defaults at the pin (`common_params_speculative_draft`),
/// which `frink run` uses too, so the two front ends draft alike.
pub(crate) const DEFAULT_DRAFT_MAX: usize = 3;
pub(crate) const DEFAULT_DRAFT_P_MIN: f32 = 0.0;

/// The weights a draft is made of.
#[derive(Clone)]
pub(crate) enum DraftKind {
    /// An autoregressive checkpoint of the target's family.
    Model(frink_models::DraftWeights),
    /// A DFlash / DFlash2 block draft.
    DFlash(Arc<frink_models::dflash::DFlashDraft>),
}

/// A loaded drafter and how it drafts.
#[derive(Clone)]
pub(crate) struct ServerDraft {
    pub(crate) kind: DraftKind,
    pub(crate) max: usize,
    pub(crate) p_min: f32,
}

impl ServerDraft {
    /// One request's drafter, with KV of its own. `None` only for a
    /// target this drafter was not checked against ([`Self::check`]
    /// runs at startup and on every model swap, so a served model has
    /// always passed it).
    pub(crate) fn speculator(
        &self,
        target: &frink_models::ModelConfig,
        sampling: frink_models::sampling::SamplingParams,
        seed: u64,
    ) -> Option<frink_models::DraftModelSpeculator> {
        let DraftKind::Model(weights) = &self.kind else {
            return None;
        };
        frink_models::DraftModelSpeculator::from_weights(
            weights, target, sampling, seed, self.max, self.p_min,
        )
        .ok()
    }

    /// One request's DFlash drafter, empty until the prompt's features
    /// reach it. `None` for an autoregressive draft, or a target this
    /// draft was not checked against.
    pub(crate) fn dflash<'t>(
        &self,
        target: &'t Decoder,
    ) -> Option<frink_models::dflash::DFlashDrafter<'t>> {
        let DraftKind::DFlash(draft) = &self.kind else {
            return None;
        };
        frink_models::dflash::DFlashDrafter::new(Arc::clone(draft), target, self.max, self.p_min)
            .ok()
    }

    /// Whether this drafter can serve `target`. A DFlash draft must fit
    /// the target's widths, token table and layers. A model draft needs
    /// the same vocabulary size, and its KV in host memory, where a
    /// rejected block can be rolled back: one warm-up proposal finds
    /// out, as `frink run -md` does, since a backend that keeps KV on
    /// the device leaves the host caches empty.
    pub(crate) fn check(&self, target: &Decoder) -> anyhow::Result<()> {
        use frink_models::speculative::Drafter;
        let weights = match &self.kind {
            DraftKind::DFlash(draft) => {
                frink_models::dflash::DFlashDrafter::new(Arc::clone(draft), target, 1, 0.0)?;
                return Ok(());
            }
            DraftKind::Model(weights) => weights,
        };
        let target = &target.config;
        let mut drafter = frink_models::DraftModelSpeculator::from_weights(
            weights,
            target,
            frink_models::sampling::SamplingParams::default(),
            0,
            self.max,
            self.p_min,
        )?;
        let _ = drafter.propose(&[0], &[], 1);
        if !drafter.keeps_host_kv() {
            bail!(
                "--model-draft needs the draft model's KV cache in host memory, and this \
                 backend keeps it on the device, so the drafter cannot roll back the positions \
                 the target rejects. Start the server with --device cpu, or without \
                 --model-draft"
            );
        }
        Ok(())
    }
}

/// The drafter `FRINK_MODEL_DRAFT` names, loaded for `target`, or
/// `None` when no draft is configured. See [`configure`].
pub(crate) fn from_env(
    target: &LoadedModel,
    continuous_batching: bool,
    paged_kv: bool,
    prefix_cache: bool,
) -> anyhow::Result<Option<ServerDraft>> {
    let target = generic_decoder(target);
    configure(
        std::env::var(DRAFT_ENV).ok(),
        std::env::var(DRAFT_MAX_ENV).ok(),
        std::env::var(DRAFT_P_MIN_ENV).ok(),
        target,
        Deployment {
            continuous_batching,
            paged_kv,
            prefix_cache,
        },
    )
}

/// The decoder a drafter is checked against: the generic GGUF one, or
/// `None` for a model on a dedicated engine, which no drafter serves.
pub(crate) fn generic_decoder(loaded: &LoadedModel) -> Option<&Decoder> {
    match loaded {
        LoadedModel::Gguf(g) => Some(&g.decoder),
        _ => None,
    }
}

/// The parts of the deployment a drafter can be refused for.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Deployment {
    pub(crate) continuous_batching: bool,
    pub(crate) paged_kv: bool,
    pub(crate) prefix_cache: bool,
}

/// A model swapped in through `/admin/models/load` is held to the
/// drafter the server started with, as the first model was: a model
/// it cannot serve refuses the swap rather than being served without
/// the speculation the server was started for.
pub(crate) fn check_swap(
    draft: Option<&ServerDraft>,
    target: Option<&Decoder>,
    id: &str,
) -> anyhow::Result<()> {
    let Some(draft) = draft else {
        return Ok(());
    };
    match target {
        Some(target) => draft
            .check(target)
            .with_context(|| format!("'{id}' cannot be served with this server's --model-draft")),
        None => bail!(
            "--model-draft drafts for the generic GGUF decoder, and '{id}' runs on a dedicated \
             engine"
        ),
    }
}

/// Refuses, naming the conflict, every deployment that would accept
/// the flag and not draft: continuous batching, the paged store, and a
/// model that runs on a dedicated engine (`target` is `None`). Then
/// loads and checks the drafter.
pub(crate) fn configure(
    path: Option<String>,
    max: Option<String>,
    p_min: Option<String>,
    target: Option<&Decoder>,
    deployment: Deployment,
) -> anyhow::Result<Option<ServerDraft>> {
    let Some(path) = path else {
        return Ok(None);
    };
    let Some(target) = target else {
        bail!(
            "--model-draft drafts for the generic GGUF decoder, and the loaded model runs on \
             a dedicated engine"
        );
    };
    let max = match max {
        Some(v) => v
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .with_context(|| {
                format!("{DRAFT_MAX_ENV} (--draft-max) must be at least 1, got {v:?}")
            })?,
        None => DEFAULT_DRAFT_MAX,
    };
    let p_min = match p_min {
        Some(v) => v
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|p| (0.0..=1.0).contains(p))
            .with_context(|| {
                format!("{DRAFT_P_MIN_ENV} (--draft-p-min) must be between 0 and 1, got {v:?}")
            })?,
        None => DEFAULT_DRAFT_P_MIN,
    };
    let is_dflash = frink_gguf::GgufFile::open(&path)
        .map(|f| frink_models::dflash::is_dflash_draft(&f))
        .unwrap_or(false);
    let kind = if is_dflash {
        // A restored prefix brings K/V and no layer inputs, so the draft
        // would have nothing to read for the cached positions.
        // The target's layer inputs are tapped on the private loop's
        // forwards only.
        if deployment.continuous_batching {
            bail!(
                "a DFlash --model-draft reads the target's layer inputs, which the continuous \
                 batcher does not tap yet: drop --cont-batching / --parallel, or use an \
                 autoregressive draft"
            );
        }
        if deployment.paged_kv {
            bail!(
                "a DFlash --model-draft reads the target's layer inputs, which the paged KV \
                 store's forwards do not tap yet. Unset FRINK_PAGED_KV_BLOCKS, or use an \
                 autoregressive draft"
            );
        }
        if deployment.prefix_cache {
            bail!(
                "a DFlash --model-draft reads the target's features at every prompt position, \
                 and a prefix-cache hit restores K/V without them. Unset \
                 FRINK_PREFIX_CACHE_ENTRIES, or use an autoregressive draft"
            );
        }
        let file =
            frink_gguf::GgufFile::open(&path).with_context(|| format!("--model-draft {path}"))?;
        DraftKind::DFlash(Arc::new(
            frink_models::dflash::DFlashDraft::load(&file)
                .with_context(|| format!("--model-draft {path}"))?,
        ))
    } else {
        DraftKind::Model(
            frink_models::draft_model::load(&path, &target.config)
                .with_context(|| format!("--model-draft {path}"))?,
        )
    };
    let draft = ServerDraft { kind, max, p_min };
    draft.check(target)?;
    tracing::info!(
        "{} {path}: up to {max} tokens per verification step",
        if is_dflash {
            "DFlash draft"
        } else {
            "draft model"
        }
    );
    Ok(Some(draft))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> Decoder {
        let mut cfg = frink_models::config::test_dense_fixture();
        cfg.vocab_size = 256;
        Decoder::new_random_small(cfg, 2, 256)
    }

    fn refusal(
        max: Option<&str>,
        p_min: Option<&str>,
        target: Option<&Decoder>,
        cb: bool,
        paged: bool,
    ) -> String {
        configure(
            Some("/no/such/draft.gguf".into()),
            max.map(str::to_string),
            p_min.map(str::to_string),
            target,
            Deployment {
                continuous_batching: cb,
                paged_kv: paged,
                prefix_cache: false,
            },
        )
        .err()
        .map(|e| format!("{e:#}"))
        .expect("refused")
    }

    /// No flag is no drafter. Every deployment that would accept the
    /// flag and not draft is refused naming why, and a bad number names
    /// its flag, before any file is opened.
    #[test]
    fn a_draft_the_server_cannot_honour_is_refused_at_startup() {
        let t = target();
        assert!(configure(
            None,
            None,
            None,
            Some(&t),
            Deployment {
                continuous_batching: true,
                paged_kv: true,
                prefix_cache: true
            }
        )
        .unwrap()
        .is_none());
        // Continuous batching speculates: what fails is the missing file.
        assert!(refusal(None, None, Some(&t), true, false).contains("/no/such/draft.gguf"));
        // The paged store speculates: what fails is the missing file.
        assert!(refusal(None, None, Some(&t), false, true).contains("/no/such/draft.gguf"));
        assert!(refusal(None, None, None, false, false).contains("dedicated engine"));
        assert!(refusal(Some("0"), None, Some(&t), false, false).contains("--draft-max"));
        assert!(refusal(None, Some("1.5"), Some(&t), false, false).contains("--draft-p-min"));
        assert!(refusal(None, None, Some(&t), false, false).contains("/no/such/draft.gguf"));
    }

    fn fixture(name: &str) -> String {
        format!(
            "{}/../frink-models/tests/fixtures/{name}_tiny.gguf",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn fixture_decoder(name: &str) -> Decoder {
        let path = fixture(name);
        let file = frink_gguf::GgufFile::open(&path).expect("fixture opens");
        let mut cfg = frink_models::ModelConfig::from_gguf(&file).expect("fixture config");
        cfg.kv_dtype = frink_core::kv_store::KvDtype::F32;
        Decoder::from_gguf(&path, cfg).expect("fixture loads")
    }

    /// A DFlash draft is picked from the file, held to the target it
    /// reads, and refused beside the prefix cache, whose hits carry no
    /// layer inputs.
    #[test]
    fn a_dflash_draft_is_picked_from_the_file_and_held_to_its_target() {
        let t = fixture_decoder("dflash_target");
        let configured = |prefix_cache, paged_kv| {
            configure(
                Some(fixture("dflash2_draft")),
                Some("4".into()),
                None,
                Some(&t),
                Deployment {
                    prefix_cache,
                    paged_kv,
                    ..Deployment::default()
                },
            )
        };
        let draft = configured(false, false)
            .expect("loads")
            .expect("configured");
        assert!(matches!(draft.kind, DraftKind::DFlash(_)));
        assert!(draft.dflash(&t).is_some());
        assert!(draft.speculator(&t.config, Default::default(), 0).is_none());
        let refused = format!("{:#}", configured(true, false).err().expect("refused"));
        assert!(refused.contains("FRINK_PREFIX_CACHE_ENTRIES"), "{refused}");
        let refused = format!("{:#}", configured(false, true).err().expect("refused"));
        assert!(refused.contains("FRINK_PAGED_KV_BLOCKS"), "{refused}");
        let refused = format!(
            "{:#}",
            configure(
                Some(fixture("dflash2_draft")),
                None,
                None,
                Some(&t),
                Deployment {
                    continuous_batching: true,
                    ..Deployment::default()
                },
            )
            .err()
            .expect("refused")
        );
        assert!(refused.contains("--cont-batching"), "{refused}");

        // A target of another width is refused, at startup and on a swap.
        let other = target();
        assert!(format!("{:#}", draft.check(&other).unwrap_err()).contains("does not fit"));
        assert!(format!(
            "{:#}",
            check_swap(Some(&draft), Some(&other), "m").unwrap_err()
        )
        .contains("'m'"));
    }

    /// A drafter whose vocabulary differs from the target's cannot
    /// serve it, and one that matches can.
    #[test]
    fn check_holds_the_drafter_to_the_target() {
        let t = target();
        let draft = ServerDraft {
            kind: DraftKind::Model(frink_models::DraftWeights::new(
                frink_models::Decoder::new_random_small(t.config.clone(), 2, 256),
            )),
            max: 4,
            p_min: 0.0,
        };
        draft.check(&t).expect("same vocabulary, host KV");
        let mut other_cfg = t.config.clone();
        other_cfg.vocab_size = 512;
        let other = Decoder::new_random_small(other_cfg, 2, 512);
        assert!(format!("{:#}", draft.check(&other).unwrap_err()).contains("vocabulary"));

        // The same rule on a model swap, and a swap with no drafter
        // configured is never refused.
        check_swap(Some(&draft), Some(&t), "m").expect("matching swap");
        let swap = format!(
            "{:#}",
            check_swap(Some(&draft), Some(&other), "m").unwrap_err()
        );
        assert!(
            swap.contains("'m'") && swap.contains("vocabulary"),
            "{swap}"
        );
        assert!(
            format!("{:#}", check_swap(Some(&draft), None, "m").unwrap_err())
                .contains("dedicated engine")
        );
        check_swap(None, None, "m").expect("no drafter, no refusal");
    }
}
