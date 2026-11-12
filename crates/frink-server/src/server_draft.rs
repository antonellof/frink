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
//! Speculation runs on the private decode loop. Where the server could
//! not honour the flag, it refuses at startup rather than accepting it
//! and serving without it: continuous batching does not speculate yet,
//! and neither the paged store nor a recurrent model can roll a
//! rejected draft back.

use anyhow::{bail, Context};
use frink_models::ModelConfig;

use crate::model::LoadedModel;

/// The draft checkpoint's path (`--model-draft`, `-md`).
pub(crate) const DRAFT_ENV: &str = "FRINK_MODEL_DRAFT";
/// Tokens drafted per verification step (`--draft-max`).
pub(crate) const DRAFT_MAX_ENV: &str = "FRINK_DRAFT_MAX";
/// The drafter's own probability below which it stops a block
/// (`--draft-p-min`).
pub(crate) const DRAFT_P_MIN_ENV: &str = "FRINK_DRAFT_P_MIN";
/// `frink run`'s defaults, so the two front ends draft alike.
pub(crate) const DEFAULT_DRAFT_MAX: usize = 5;
pub(crate) const DEFAULT_DRAFT_P_MIN: f32 = 0.75;

/// A loaded drafter and how it drafts.
#[derive(Clone)]
pub(crate) struct ServerDraft {
    pub(crate) weights: frink_models::DraftWeights,
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
        target: &ModelConfig,
        sampling: frink_models::sampling::SamplingParams,
        seed: u64,
    ) -> Option<frink_models::DraftModelSpeculator> {
        frink_models::DraftModelSpeculator::from_weights(
            &self.weights,
            target,
            sampling,
            seed,
            self.max,
            self.p_min,
        )
        .ok()
    }

    /// Whether this drafter can serve `target`: the same vocabulary
    /// size, and draft KV in host memory, where a rejected block can be
    /// rolled back. One warm-up proposal finds out, as `frink run -md`
    /// does: a backend that keeps KV on the device leaves the host
    /// caches empty.
    pub(crate) fn check(&self, target: &ModelConfig) -> anyhow::Result<()> {
        use frink_models::speculative::Drafter;
        let mut drafter = frink_models::DraftModelSpeculator::from_weights(
            &self.weights,
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
) -> anyhow::Result<Option<ServerDraft>> {
    let target = generic_config(target);
    configure(
        std::env::var(DRAFT_ENV).ok(),
        std::env::var(DRAFT_MAX_ENV).ok(),
        std::env::var(DRAFT_P_MIN_ENV).ok(),
        target,
        continuous_batching,
        paged_kv,
    )
}

/// The config a drafter is checked against: the generic GGUF decoder's,
/// or `None` for a model on a dedicated engine, which no drafter serves.
pub(crate) fn generic_config(loaded: &LoadedModel) -> Option<&ModelConfig> {
    match loaded {
        LoadedModel::Gguf(g) => Some(&g.decoder.config),
        _ => None,
    }
}

/// A model swapped in through `/admin/models/load` is held to the
/// drafter the server started with, as the first model was: a model
/// it cannot serve refuses the swap rather than being served without
/// the speculation the server was started for.
pub(crate) fn check_swap(
    draft: Option<&ServerDraft>,
    target: Option<&ModelConfig>,
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
    target: Option<&ModelConfig>,
    continuous_batching: bool,
    paged_kv: bool,
) -> anyhow::Result<Option<ServerDraft>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if continuous_batching {
        bail!(
            "--model-draft speculates on the private decode loop, and continuous batching does \
             not speculate yet. On Metal it is on by default: add --no-cont-batching"
        );
    }
    if paged_kv {
        bail!(
            "--model-draft cannot run on the paged KV store, which cannot roll a rejected draft \
             back. Unset FRINK_PAGED_KV_BLOCKS, or start without --model-draft"
        );
    }
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
    let weights = frink_models::draft_model::load(&path, target)
        .with_context(|| format!("--model-draft {path}"))?;
    let draft = ServerDraft {
        weights,
        max,
        p_min,
    };
    draft.check(target)?;
    tracing::info!("draft model {path}: up to {max} tokens per verification step");
    Ok(Some(draft))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(
        max: Option<&str>,
        p_min: Option<&str>,
        target: Option<&ModelConfig>,
        cb: bool,
        paged: bool,
    ) -> String {
        configure(
            Some("/no/such/draft.gguf".into()),
            max.map(str::to_string),
            p_min.map(str::to_string),
            target,
            cb,
            paged,
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
        let cfg = frink_models::config::test_dense_fixture();
        assert!(configure(None, None, None, Some(&cfg), true, true)
            .unwrap()
            .is_none());
        assert!(refusal(None, None, Some(&cfg), true, false).contains("continuous batching"));
        assert!(refusal(None, None, Some(&cfg), false, true).contains("paged KV"));
        assert!(refusal(None, None, None, false, false).contains("dedicated engine"));
        assert!(refusal(Some("0"), None, Some(&cfg), false, false).contains("--draft-max"));
        assert!(refusal(None, Some("1.5"), Some(&cfg), false, false).contains("--draft-p-min"));
        assert!(refusal(None, None, Some(&cfg), false, false).contains("/no/such/draft.gguf"));
    }

    /// A drafter whose vocabulary differs from the target's cannot
    /// serve it, and one that matches can.
    #[test]
    fn check_holds_the_drafter_to_the_target() {
        let mut cfg = frink_models::config::test_dense_fixture();
        cfg.vocab_size = 256;
        let draft = ServerDraft {
            weights: frink_models::DraftWeights::new(frink_models::Decoder::new_random_small(
                cfg.clone(),
                2,
                256,
            )),
            max: 4,
            p_min: 0.0,
        };
        draft.check(&cfg).expect("same vocabulary, host KV");
        let mut other = cfg.clone();
        other.vocab_size = 512;
        assert!(format!("{:#}", draft.check(&other).unwrap_err()).contains("vocabulary"));

        // The same rule on a model swap, and a swap with no drafter
        // configured is never refused.
        check_swap(Some(&draft), Some(&cfg), "m").expect("matching swap");
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
