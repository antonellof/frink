//! The sampler knobs an OpenAI request body carries, resolved to
//! [`SamplingParams`] in exactly one place.
//!
//! `/v1/chat/completions` and `/v1/completions` cannot share a request
//! struct -- their bodies are genuinely different shapes -- so each
//! names its own wire fields. What they must NOT each own is the
//! mapping: which knobs exist, and what each one means when the caller
//! said nothing. That copy had already drifted. `/v1/completions`
//! hardcoded `top_k: 0` and `repetition_penalty: 1.0` while the chat
//! route read both off the request, so four real sampler fields were
//! dropped by serde on one route and honoured on the other -- the same
//! defect as `logit_bias`, four more times.
//!
//! So the knobs are one struct and the defaults are one function. A knob
//! added here is added to both routes at once, or to neither.

use frink_models::dry::{DryRequest, DryVocab, DryVocabMissing};
use frink_models::sampler_order::SamplerOrder;
use frink_models::sampling::SamplingParams;

/// How many recent tokens the repetition/presence/frequency penalties
/// look at. llama.cpp's `penalty_last_n` default (`common/common.h:238`);
/// the OpenAI wire has no field for it on either route.
const DEFAULT_PENALTY_LAST_N: usize = 64;

/// One request's sampler knobs, each `None` when the caller said
/// nothing about it.
#[derive(Debug, Default, Clone)]
pub(crate) struct SamplingKnobs {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    /// llama.cpp's `--min-p`: keep only candidates at least this
    /// fraction as likely as the most likely one.
    ///
    /// `None` resolves to `0.0` (off), NOT to llama.cpp's CLI default of
    /// 0.05. A server that quietly truncated every request nobody
    /// configured would be a behaviour change no caller asked for, and
    /// an HTTP client is not running llama.cpp's command line. The
    /// number lives on the CLI flag, where the person who typed it can
    /// see it.
    pub(crate) min_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    /// llama.cpp's `typical_p` / `typ_p`. `None` resolves to `1.0`,
    /// which is off and is also upstream's default.
    pub(crate) typical_p: Option<f32>,
    /// llama.cpp's `top_n_sigma`. `None` resolves to `-1.0`, off.
    pub(crate) top_n_sigma: Option<f32>,
    /// llama.cpp's `xtc_probability`. `None` resolves to `0.0`, off.
    pub(crate) xtc_probability: Option<f32>,
    /// llama.cpp's `xtc_threshold`. `None` resolves to upstream's
    /// `0.1`, which is NOT itself a disabling value -- the probability
    /// is what switches XTC off -- so a request that sets only the
    /// threshold still gets no XTC.
    pub(crate) xtc_threshold: Option<f32>,
    /// llama.cpp's `dry_multiplier`. `None` resolves to `0.0`, off.
    pub(crate) dry_multiplier: Option<f32>,
    /// llama.cpp's `dry_base`, default 1.75.
    pub(crate) dry_base: Option<f32>,
    /// llama.cpp's `dry_allowed_length`, default 2.
    pub(crate) dry_allowed_length: Option<i32>,
    /// llama.cpp's `dry_penalty_last_n`, default `-1` (the context).
    pub(crate) dry_penalty_last_n: Option<i32>,
    /// llama.cpp's `dry_sequence_breakers`. `None` is llama.cpp's own
    /// four defaults; an explicit empty list is "no breakers", which is
    /// what `--dry-sequence-breaker none` asks for on the command line.
    pub(crate) dry_sequence_breakers: Option<Vec<String>>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    /// llama.cpp's `repeat_last_n`: how many recent tokens the three
    /// penalties look back over. `0` disables them entirely.
    ///
    /// Only llama.cpp's native `/completion` wire has a field for this;
    /// neither OpenAI route does, so both leave it `None` and get
    /// [`DEFAULT_PENALTY_LAST_N`]. It lives here rather than on that one
    /// route because it is a sampler knob, and the whole point of this
    /// struct is that no route owns its own copy of what a knob means.
    pub(crate) penalty_last_n: Option<usize>,
    /// llama.cpp's `samplers`: the ORDER the chain runs in.
    ///
    /// Already validated -- every route parses it through
    /// [`crate::unsupported_sampling::parse_sampler_order`], which
    /// refuses an unknown or unimplemented sampler BY NAME before the
    /// request reaches here. `None` is a request that said nothing,
    /// which resolves to frink's default chain and therefore samples
    /// exactly what it did before this field existed.
    pub(crate) sampler_order: Option<SamplerOrder>,
}

/// The sampler fields llama.cpp's chain gained over OpenAI's schema,
/// as ONE `#[serde(flatten)]`ed struct rather than nine fields repeated
/// in three request bodies.
///
/// `/v1/chat/completions`, `/v1/completions` and llama.cpp's native
/// `/completion` are genuinely different shapes and cannot share a
/// request struct -- but they can share this. Nine fields written out
/// three times is nine chances to add one to two routes, and that exact
/// defect has already shipped here twice: `logit_bias` declared on one
/// route and not the other, and four sampler fields hardcoded on
/// `/v1/completions` while the chat route read them.
///
/// The names are llama.cpp's server's own, so a client written against
/// upstream works here unchanged.
#[derive(Debug, Default, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub(crate) struct ExtraSamplerFields {
    pub(crate) typical_p: Option<f32>,
    pub(crate) top_n_sigma: Option<f32>,
    pub(crate) xtc_probability: Option<f32>,
    pub(crate) xtc_threshold: Option<f32>,
    pub(crate) dry_multiplier: Option<f32>,
    pub(crate) dry_base: Option<f32>,
    pub(crate) dry_allowed_length: Option<i32>,
    pub(crate) dry_penalty_last_n: Option<i32>,
    pub(crate) dry_sequence_breakers: Option<Vec<String>>,
}

impl ExtraSamplerFields {
    /// Copy these onto a [`SamplingKnobs`].
    ///
    /// **Exhaustive destructure, no `..`**, for the same reason
    /// [`SamplingKnobs::resolve`] has one: a field added to the wire
    /// struct and not copied here is an unused variable, and
    /// `cargo clippy -- -D warnings` is a gate. A caller's field would
    /// otherwise be deserialized, discarded, and answered with a 200.
    pub(crate) fn apply(&self, knobs: &mut SamplingKnobs) {
        let ExtraSamplerFields {
            typical_p,
            top_n_sigma,
            xtc_probability,
            xtc_threshold,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            dry_sequence_breakers,
        } = self;
        knobs.typical_p = *typical_p;
        knobs.top_n_sigma = *top_n_sigma;
        knobs.xtc_probability = *xtc_probability;
        knobs.xtc_threshold = *xtc_threshold;
        knobs.dry_multiplier = *dry_multiplier;
        knobs.dry_base = *dry_base;
        knobs.dry_allowed_length = *dry_allowed_length;
        knobs.dry_penalty_last_n = *dry_penalty_last_n;
        knobs.dry_sequence_breakers = dry_sequence_breakers.clone();
    }
}

/// The model facts the sampler needs that a request body cannot carry:
/// the vocabulary DRY's sequence breakers are tokenised against, and the
/// context size `dry_penalty_last_n = -1` resolves to.
///
/// Passed rather than looked up, because the route already holds the
/// pinned `ActiveModel` and a second lookup could see a different model
/// after `/admin/models/load`.
#[derive(Clone, Copy)]
pub(crate) struct SamplerModel<'a> {
    /// `None` for a checkpoint with no real vocabulary (the
    /// synthetic-weight fallback), which makes DRY a refusal rather
    /// than a sampler that silently ignores its breakers.
    pub(crate) vocab: Option<&'a dyn DryVocab>,
    /// llama.cpp's `n_ctx_train`. `usize::MAX` when this server could
    /// not price a ceiling for the model: that reads as "no ceiling
    /// smaller than the sequence itself", which is what upstream's
    /// clamp degenerates to, and is derived rather than a chosen
    /// constant that would silently truncate someone's window.
    pub(crate) context_size: usize,
}

impl SamplerModel<'_> {
    /// No vocabulary and no ceiling.
    ///
    /// For tests that resolve knobs without loading a checkpoint, and
    /// it is deliberately the one value that makes DRY a REFUSAL rather
    /// than a silently breaker-less sampler -- so a test that wants DRY
    /// has to say which vocabulary it wants it against.
    #[cfg(test)]
    pub(crate) fn absent() -> Self {
        SamplerModel {
            vocab: None,
            context_size: usize::MAX,
        }
    }
}

impl SamplingKnobs {
    /// Fill every gap the request left with this server's default.
    ///
    /// **The destructure is exhaustive ON PURPOSE, with no `..`.** A
    /// knob added to this struct and not read here is an `unused
    /// variable` warning, and `cargo clippy -- -D warnings` is a gate in
    /// this repo, so it does not compile. That is the whole defence
    /// against the defect this module's header describes: a field the
    /// wire accepts, serde deserializes, and the sampler never sees.
    pub(crate) fn resolve(
        &self,
        model: SamplerModel<'_>,
    ) -> Result<SamplingParams, DryVocabMissing> {
        let SamplingKnobs {
            temperature,
            top_p,
            min_p,
            top_k,
            typical_p,
            top_n_sigma,
            xtc_probability,
            xtc_threshold,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_penalty_last_n,
            dry_sequence_breakers,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            penalty_last_n,
            sampler_order,
        } = self;
        let defaults = DryRequest::default();
        let dry = DryRequest {
            multiplier: dry_multiplier.unwrap_or(defaults.multiplier),
            base: dry_base.unwrap_or(defaults.base),
            allowed_length: dry_allowed_length.unwrap_or(defaults.allowed_length),
            penalty_last_n: dry_penalty_last_n.unwrap_or(defaults.penalty_last_n),
            sequence_breakers: dry_sequence_breakers
                .clone()
                .unwrap_or(defaults.sequence_breakers),
        };
        Ok(SamplingParams {
            temperature: temperature.unwrap_or(0.0),
            top_p: top_p.unwrap_or(1.0),
            min_p: min_p.unwrap_or(0.0),
            top_k: top_k.unwrap_or(0),
            typical_p: typical_p.unwrap_or(1.0),
            top_n_sigma: top_n_sigma.unwrap_or(-1.0),
            xtc_probability: xtc_probability.unwrap_or(0.0),
            xtc_threshold: xtc_threshold.unwrap_or(0.1),
            dry: dry.resolve(model.vocab, model.context_size)?,
            repetition_penalty: repetition_penalty.unwrap_or(1.0),
            penalty_last_n: penalty_last_n.unwrap_or(DEFAULT_PENALTY_LAST_N),
            presence_penalty: presence_penalty.unwrap_or(0.0),
            frequency_penalty: frequency_penalty.unwrap_or(0.0),
            sampler_order: sampler_order.unwrap_or_default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty request is greedy decoding with every filter off: the
    /// same "do nothing the caller did not ask for" baseline
    /// `SamplingParams::default` is.
    #[test]
    fn an_empty_request_resolves_to_the_do_nothing_baseline() {
        let resolved = SamplingKnobs::default()
            .resolve(crate::sampling_knobs::SamplerModel::absent())
            .expect("no dry");
        let baseline = SamplingParams::default();
        assert_eq!(resolved.temperature, baseline.temperature);
        assert_eq!(resolved.top_p, baseline.top_p);
        assert_eq!(resolved.min_p, baseline.min_p);
        assert_eq!(resolved.top_k, baseline.top_k);
        assert_eq!(resolved.repetition_penalty, baseline.repetition_penalty);
        assert_eq!(resolved.penalty_last_n, baseline.penalty_last_n);
    }

    /// The penalty window is a knob like any other: absent means this
    /// server's default, and `0` means the caller asked for the
    /// penalties to be switched off, which is not the same thing.
    #[test]
    fn the_penalty_window_is_honoured_including_the_value_that_disables_it() {
        assert_eq!(
            SamplingKnobs::default()
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .penalty_last_n,
            DEFAULT_PENALTY_LAST_N
        );
        let asked = SamplingKnobs {
            penalty_last_n: Some(0),
            ..SamplingKnobs::default()
        };
        assert_eq!(
            asked
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .penalty_last_n,
            0
        );
        let wide = SamplingKnobs {
            penalty_last_n: Some(4096),
            ..SamplingKnobs::default()
        };
        assert_eq!(
            wide.resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .penalty_last_n,
            4096
        );
    }

    /// A request that said nothing about `samplers` gets the chain
    /// frink has always run, and one that did gets the chain it asked
    /// for, in that order.
    ///
    /// The default half is the one that matters: every existing client
    /// omits this field, so a default that resolved to anything but
    /// `SamplerOrder::default()` would change the output of every
    /// request already in flight.
    #[test]
    fn an_unset_sampler_order_is_the_chain_this_server_already_ran() {
        assert_eq!(
            SamplingKnobs::default()
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .sampler_order,
            SamplerOrder::default()
        );
        let asked = SamplingKnobs {
            sampler_order: Some(
                "penalties;temperature;top_k"
                    .parse()
                    .expect("a chain frink implements"),
            ),
            ..SamplingKnobs::default()
        };
        assert_eq!(
            asked
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .sampler_order
                .to_string(),
            "penalties;temperature;top_k"
        );
    }

    /// Specifically: a server that shipped llama.cpp's 0.05 would
    /// truncate the distribution of every request nobody configured.
    #[test]
    fn min_p_defaults_to_off_rather_than_to_llama_cpps_cli_number() {
        assert_eq!(
            SamplingKnobs::default()
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .min_p,
            0.0
        );
        assert_eq!(
            SamplingKnobs {
                min_p: Some(0.05),
                ..SamplingKnobs::default()
            }
            .resolve(crate::sampling_knobs::SamplerModel::absent())
            .expect("no dry")
            .min_p,
            0.05
        );
    }

    /// A vocabulary just large enough for DRY to tokenise a breaker
    /// against, so a test can switch DRY on without loading a model.
    struct TinyVocab;

    impl DryVocab for TinyVocab {
        fn n_tokens(&self) -> usize {
            4
        }
        fn detokenize(&self, token: usize) -> String {
            ["a", "b", "\n", "c"][token].to_string()
        }
        fn tokenize(&self, text: &str) -> Vec<usize> {
            text.chars()
                .filter_map(|c| match c {
                    'a' => Some(0),
                    'b' => Some(1),
                    '\n' => Some(2),
                    'c' => Some(3),
                    _ => None,
                })
                .collect()
        }
    }

    fn tiny_model() -> SamplerModel<'static> {
        SamplerModel {
            vocab: Some(&TinyVocab),
            context_size: 4096,
        }
    }

    /// One live value per extra wire field, all set at once so the
    /// `dry_*` fields have a non-zero multiplier to matter under.
    fn live_fields() -> ExtraSamplerFields {
        ExtraSamplerFields {
            typical_p: Some(0.9),
            top_n_sigma: Some(1.5),
            xtc_probability: Some(0.3),
            xtc_threshold: Some(0.2),
            dry_multiplier: Some(0.8),
            dry_base: Some(1.5),
            dry_allowed_length: Some(3),
            dry_penalty_last_n: Some(128),
            dry_sequence_breakers: Some(vec!["\n".to_string()]),
        }
    }

    /// **Every field the wire accepts changes what the sampler does.**
    ///
    /// This is the assertion for the defect ADR 0004 names: a wire
    /// struct and a sampler that silently disagree about which fields
    /// exist, six parameters accepted and ignored. Serde deserializing a
    /// field is not evidence that anything reads it.
    ///
    /// The coverage list is DERIVED, not restated: the field names come
    /// out of `serde_json::to_value` on a fully populated
    /// [`ExtraSamplerFields`], so a field added to that struct and not
    /// probed here turns this red rather than passing unnoticed. And
    /// each probe is checked through
    /// [`crate::response_cache::sampling_key`], which is the exhaustive
    /// destructure of `SamplingParams` -- so "reaches the sampler" and
    /// "is in the cache key" are one assertion, and neither can be
    /// satisfied without the other.
    #[test]
    fn every_extra_wire_sampler_field_changes_the_resolved_sampler() {
        let live = live_fields();
        let key_of = |fields: &ExtraSamplerFields| {
            let mut knobs = SamplingKnobs::default();
            fields.apply(&mut knobs);
            crate::response_cache::sampling_key(
                &knobs
                    .resolve(tiny_model())
                    .expect("the tiny vocab serves DRY"),
            )
        };
        let baseline = key_of(&live);

        let probes: Vec<(&str, ExtraSamplerFields)> = vec![
            (
                "typical_p",
                ExtraSamplerFields {
                    typical_p: Some(0.5),
                    ..live.clone()
                },
            ),
            (
                "top_n_sigma",
                ExtraSamplerFields {
                    top_n_sigma: Some(2.5),
                    ..live.clone()
                },
            ),
            (
                "xtc_probability",
                ExtraSamplerFields {
                    xtc_probability: Some(0.7),
                    ..live.clone()
                },
            ),
            (
                "xtc_threshold",
                ExtraSamplerFields {
                    xtc_threshold: Some(0.4),
                    ..live.clone()
                },
            ),
            (
                "dry_multiplier",
                ExtraSamplerFields {
                    dry_multiplier: Some(1.6),
                    ..live.clone()
                },
            ),
            (
                "dry_base",
                ExtraSamplerFields {
                    dry_base: Some(2.0),
                    ..live.clone()
                },
            ),
            (
                "dry_allowed_length",
                ExtraSamplerFields {
                    dry_allowed_length: Some(5),
                    ..live.clone()
                },
            ),
            (
                "dry_penalty_last_n",
                ExtraSamplerFields {
                    dry_penalty_last_n: Some(64),
                    ..live.clone()
                },
            ),
            (
                "dry_sequence_breakers",
                ExtraSamplerFields {
                    dry_sequence_breakers: Some(vec!["a".to_string()]),
                    ..live.clone()
                },
            ),
        ];

        let probed: std::collections::BTreeSet<String> =
            probes.iter().map(|(name, _)| name.to_string()).collect();
        let declared: std::collections::BTreeSet<String> = serde_json::to_value(&live)
            .expect("serialises")
            .as_object()
            .expect("an object")
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            probed, declared,
            "a field on the wire is not probed here, so nothing would notice \
             if the sampler stopped reading it"
        );

        for (name, fields) in probes {
            assert_ne!(
                key_of(&fields),
                baseline,
                "`{name}` is accepted on the wire and changes nothing in the \
                 resolved sampler"
            );
        }
    }

    /// The three routes that take these fields resolve them the same
    /// way, asserted through the real request types.
    ///
    /// A shared `#[serde(flatten)]` struct only helps if every route
    /// both DECLARES it and calls `apply`. `/v1/completions` once
    /// hardcoded four sampler fields the chat route read off the
    /// request; this is the assertion that would have caught it.
    #[test]
    fn every_route_resolves_the_extra_sampler_fields_the_same_way() {
        let extras = serde_json::json!({
            "typical_p": 0.9,
            "top_n_sigma": 1.5,
            "xtc_probability": 0.3,
            "xtc_threshold": 0.2,
            "dry_multiplier": 0.8,
            "dry_base": 1.5,
            "dry_allowed_length": 3,
            "dry_penalty_last_n": 128,
            "dry_sequence_breakers": ["\n"],
        });
        let with = |mut body: serde_json::Value| {
            for (k, v) in extras.as_object().expect("an object") {
                body[k] = v.clone();
            }
            body
        };
        let chat: crate::ChatCompletionRequest = serde_json::from_value(with(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        })))
        .expect("chat request");
        let completions: crate::openai_extra::CompletionsRequest =
            serde_json::from_value(with(serde_json::json!({"prompt": "hi"})))
                .expect("completions request");
        let native: crate::completion::CompletionRequest =
            serde_json::from_value(with(serde_json::json!({"prompt": "hi"})))
                .expect("completion request");

        let key = |knobs: SamplingKnobs| {
            crate::response_cache::sampling_key(&knobs.resolve(tiny_model()).expect("dry"))
        };
        let chat_key = key(chat.sampling_knobs().expect("chat knobs"));
        let completions_key = key(completions.sampling_knobs().expect("completions knobs"));
        let native_key = key(native.sampling_knobs().expect("native knobs"));
        assert_eq!(chat_key, completions_key, "chat and /v1/completions differ");
        assert_eq!(chat_key, native_key, "chat and /completion differ");

        // And the values really are the ones sent, not three copies of
        // the defaults agreeing with each other.
        let resolved = chat
            .sampling_knobs()
            .expect("knobs")
            .resolve(tiny_model())
            .expect("dry");
        assert_eq!(resolved.typical_p, 0.9);
        assert_eq!(resolved.top_n_sigma, 1.5);
        assert_eq!(resolved.xtc_probability, 0.3);
        assert_eq!(resolved.xtc_threshold, 0.2);
        assert!(resolved.dry.is_enabled());
        assert_eq!(resolved.dry.multiplier(), 0.8);
        assert_eq!(resolved.dry.base(), 1.5);
        assert_eq!(resolved.dry.allowed_length(), 3);
        assert_eq!(resolved.dry.penalty_last_n(), 128);
        assert_eq!(resolved.dry.breakers().raw(), ["\n".to_string()]);
    }

    /// A request that says nothing about the new samplers resolves to
    /// values that make each of them a NO-OP, so llama.cpp's full
    /// default chain leaves an unconfigured request exactly where it
    /// was before these four samplers existed.
    #[test]
    fn an_unconfigured_request_leaves_every_new_sampler_switched_off() {
        let resolved = SamplingKnobs::default()
            .resolve(SamplerModel::absent())
            .expect("no dry means no vocabulary is needed");
        assert_eq!(resolved.typical_p, 1.0);
        assert_eq!(resolved.top_n_sigma, -1.0);
        assert_eq!(resolved.xtc_probability, 0.0);
        assert_eq!(resolved.xtc_threshold, 0.1);
        assert!(!resolved.dry.is_enabled());
        assert!(!resolved.xtc_can_fire());
        assert!(resolved.greedy_equals_raw_argmax());
    }

    /// DRY asked for against a checkpoint with no vocabulary is a
    /// REFUSAL, not DRY with no sequence breakers.
    ///
    /// A breaker-less DRY looks like working DRY and penalises across
    /// every boundary the caller named. The refusal is reachable: it
    /// fires for exactly the models `Model::has_real_vocabulary` says
    /// have none.
    #[test]
    fn dry_without_a_vocabulary_is_refused_rather_than_run_without_breakers() {
        let mut knobs = SamplingKnobs::default();
        ExtraSamplerFields {
            dry_multiplier: Some(0.8),
            ..Default::default()
        }
        .apply(&mut knobs);
        knobs
            .resolve(SamplerModel::absent())
            .expect_err("no vocabulary, so no breakers, so no DRY");

        // The same request with the breakers explicitly emptied IS
        // served: there is nothing left to tokenise.
        let mut none = SamplingKnobs::default();
        ExtraSamplerFields {
            dry_multiplier: Some(0.8),
            dry_sequence_breakers: Some(Vec::new()),
            ..Default::default()
        }
        .apply(&mut none);
        let resolved = none
            .resolve(SamplerModel::absent())
            .expect("no breakers needs no vocabulary");
        assert!(resolved.dry.is_enabled());
        assert!(resolved.dry.breakers().is_empty());
    }
}
