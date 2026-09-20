//! The sampling a **checkpoint recommends for itself**, and the
//! precedence that resolves it against what one request asked for.
//!
//! Split out of `sampling.rs`: this is a policy about where a number
//! comes from, not part of the sampler that consumes it.

use super::SamplingParams;

/// The sampling a **checkpoint recommends for itself**, one `Option`
/// per field so that "this model says nothing about top_p" stays
/// distinguishable from "this model recommends top_p = 1.0". This is
/// the `sampling_defaults='model'` convention, ported from FreeToken
/// `python/freetoken/utils/hf.py:92 load_generation_sampling`.
///
/// Every field is `None` for a checkpoint that recommends nothing,
/// which is the overwhelming majority, and
/// [`RecommendedSampling::resolve`] then reproduces frink's existing
/// defaults exactly -- a recommendation may only fill a gap the request
/// left, never override it.
///
/// Why this exists at all: reasoning checkpoints are tuned for a
/// specific sampler (Qwen3.5 ships temperature 1.0, top_k 20, top_p
/// 0.95) and ship those numbers with the weights. Served under a
/// generic greedy-or-0.8 default they fall into repetition loops --
/// fluent output that never terminates -- which reads as a broken model
/// rather than as a serving default nobody read off the file.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RecommendedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

/// The sampling fields **one request** actually specified. `None` means
/// the request said nothing about that field, so the checkpoint's
/// recommendation (and then the framework default) may speak for it.
///
/// Collapsing this to a plain [`SamplingParams`] at the wire boundary
/// -- `temperature: req.temperature.unwrap_or(0.0)` -- is what destroys
/// the distinction: a request that omitted `temperature` becomes
/// indistinguishable from one that explicitly asked for greedy, and no
/// recommendation can ever apply.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RequestedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

impl RecommendedSampling {
    /// True when the checkpoint recommended nothing at all, i.e.
    /// [`Self::resolve`] is guaranteed to return the framework defaults
    /// for any request. Useful for telling an operator whether
    /// "model defaults" had anything to act on.
    pub fn is_empty(&self) -> bool {
        *self == RecommendedSampling::default()
    }

    /// Precedence, exactly as FreeToken's `resolve_sampling.pick`
    /// (`python/freetoken/server/generation.py:170`) applies it: the
    /// **request's** own value, else the **checkpoint's**
    /// recommendation, else the **framework** default carried by
    /// `framework` (frink's `SamplingParams::default()` unless a caller
    /// has its own).
    ///
    /// The penalty fields are taken from `framework` untouched: nothing
    /// in the reference reads a recommended penalty, and inventing one
    /// here would be this function changing generation on its own.
    ///
    /// Getting the order wrong in either direction is a silent
    /// behaviour change: recommendation-over-request makes a client's
    /// explicit `temperature: 0` unreachable on a model that recommends
    /// 1.0, and framework-over-recommendation is the greedy repetition
    /// loop this whole path exists to avoid.
    pub fn resolve(
        &self,
        requested: RequestedSampling,
        framework: SamplingParams,
    ) -> SamplingParams {
        SamplingParams {
            temperature: requested
                .temperature
                .or(self.temperature)
                .unwrap_or(framework.temperature),
            top_p: requested.top_p.or(self.top_p).unwrap_or(framework.top_p),
            top_k: requested.top_k.or(self.top_k).unwrap_or(framework.top_k),
            ..framework
        }
    }

    /// The recommendation in a HuggingFace-style `generation_config.json`
    /// body.
    ///
    /// Two rules, both from the reference
    /// (`hf.py:92 load_generation_sampling`):
    ///
    /// * `do_sample: false` means the checkpoint recommends **greedy**,
    ///   which is returned as `temperature = 0` and *nothing else* --
    ///   the top_k/top_p in such a file describe a sampler the model
    ///   asks not to be used.
    /// * otherwise only the keys **actually present** are returned. An
    ///   absent key stays `None`; filling it with a house default (the
    ///   naive reading, and what HF's own `GenerationConfig` object does
    ///   for you) would turn silence into a recommendation and let a
    ///   file that says only `temperature: 0.6` also pin top_p to 1.0,
    ///   overriding the server's own default for a value the checkpoint
    ///   never expressed.
    ///
    /// A file that does not parse, or is not a JSON object, recommends
    /// nothing -- a malformed sidecar must not be able to change how a
    /// model is sampled.
    pub fn from_generation_config(json: &str) -> Self {
        let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(json)
        else {
            return RecommendedSampling::default();
        };
        if map.get("do_sample").and_then(|v| v.as_bool()) == Some(false) {
            return RecommendedSampling {
                temperature: Some(0.0),
                ..RecommendedSampling::default()
            };
        }
        RecommendedSampling {
            temperature: map
                .get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            top_p: map.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
            top_k: map
                .get("top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }

    /// [`Self::from_generation_config`] for the `generation_config.json`
    /// beside a checkpoint's weights (an HF-format model directory).
    ///
    /// A directory with no such file recommends nothing, exactly like a
    /// GGUF with no `general.sampling.*` keys: the absence of a
    /// recommendation is the normal case and must never be an error that
    /// stops a model from loading.
    pub fn from_model_dir(dir: &std::path::Path) -> Self {
        match std::fs::read_to_string(dir.join("generation_config.json")) {
            Ok(text) => Self::from_generation_config(&text),
            Err(_) => RecommendedSampling::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the keys the file actually carries become a recommendation.
    ///
    /// **This test fails if an absent key is filled with a house
    /// default** (the naive reading, and what HF's own
    /// `GenerationConfig` object does): `top_p` and `top_k` would come
    /// back as `Some(1.0)` / `Some(0)` and would then override whatever
    /// the server itself defaults to, for values this checkpoint never
    /// expressed.
    #[test]
    fn an_absent_generation_config_key_stays_absent_rather_than_taking_a_default() {
        let recommended = RecommendedSampling::from_generation_config(r#"{"temperature": 0.6}"#);
        assert_eq!(recommended.temperature, Some(0.6));
        assert_eq!(recommended.top_p, None, "top_p was not in the file");
        assert_eq!(recommended.top_k, None, "top_k was not in the file");
        // An explicit JSON null is silence too (the reference's
        // `if val is not None`).
        let nulled = RecommendedSampling::from_generation_config(r#"{"top_p": null}"#);
        assert_eq!(nulled, RecommendedSampling::default());
    }

    /// A reasoning checkpoint's full recommendation survives intact --
    /// the case the whole path exists for (Qwen3.5: temp 1.0, top_k 20,
    /// top_p 0.95).
    #[test]
    fn every_generation_config_key_present_is_recommended() {
        let recommended = RecommendedSampling::from_generation_config(
            r#"{"do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}"#,
        );
        assert_eq!(
            recommended,
            RecommendedSampling {
                temperature: Some(1.0),
                top_p: Some(0.95),
                top_k: Some(20),
            }
        );
    }

    /// `do_sample: false` recommends greedy, expressed as temperature 0
    /// and *nothing else*: the top_k/top_p such a file also carries
    /// describe a sampler it is asking not to be used, so returning them
    /// would filter a distribution the model wants collapsed to its
    /// argmax.
    #[test]
    fn do_sample_false_recommends_greedy_and_no_other_field() {
        let recommended = RecommendedSampling::from_generation_config(
            r#"{"do_sample": false, "temperature": 0.7, "top_k": 50, "top_p": 0.9}"#,
        );
        assert_eq!(recommended.temperature, Some(0.0));
        assert_eq!(recommended.top_p, None);
        assert_eq!(recommended.top_k, None);
    }

    /// A sidecar that does not parse must not be able to change how the
    /// model is sampled.
    #[test]
    fn a_malformed_generation_config_recommends_nothing() {
        for text in ["", "not json", "[1, 2, 3]", "null"] {
            assert!(
                RecommendedSampling::from_generation_config(text).is_empty(),
                "{text:?} must recommend nothing"
            );
        }
    }

    /// Precedence: the request wins over the checkpoint, and the
    /// checkpoint only fills what the request left unset. An explicit
    /// `temperature: 0` from a client must stay reachable on a model
    /// that recommends 1.0.
    #[test]
    fn a_request_outranks_the_recommendation_which_outranks_the_framework_default() {
        let recommended = RecommendedSampling {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
        };
        let resolved = recommended.resolve(
            RequestedSampling {
                temperature: Some(0.0),
                ..RequestedSampling::default()
            },
            SamplingParams::default(),
        );
        assert_eq!(resolved.temperature, 0.0, "the request asked for greedy");
        assert_eq!(resolved.top_p, 0.95, "the request said nothing about top_p");
        assert_eq!(resolved.top_k, 20, "the request said nothing about top_k");
        // Penalties are never recommended, only carried through.
        assert_eq!(resolved.repetition_penalty, 1.0);
    }

    /// A checkpoint that recommends nothing must leave frink's existing
    /// behaviour bit-identical: greedy, unfiltered, exactly
    /// `SamplingParams::default()`.
    #[test]
    fn a_checkpoint_that_recommends_nothing_leaves_the_framework_defaults_alone() {
        let resolved = RecommendedSampling::default()
            .resolve(RequestedSampling::default(), SamplingParams::default());
        let default = SamplingParams::default();
        assert_eq!(resolved.temperature, default.temperature);
        assert_eq!(resolved.top_p, default.top_p);
        assert_eq!(resolved.top_k, default.top_k);
    }

    /// A model directory with no `generation_config.json` recommends
    /// nothing rather than failing: the absence of a recommendation is
    /// the normal case for most checkpoints.
    #[test]
    fn a_model_directory_without_a_generation_config_recommends_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "frink_test_no_generation_config_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(RecommendedSampling::from_model_dir(&dir).is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The sidecar is read from the directory beside the weights, the
    /// same place HF's `GenerationConfig.from_pretrained` looks.
    #[test]
    fn a_model_directory_generation_config_is_read_from_beside_the_weights() {
        let dir = std::env::temp_dir().join(format!(
            "frink_test_generation_config_dir_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.6, "top_p": 0.95}"#,
        )
        .unwrap();
        let recommended = RecommendedSampling::from_model_dir(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(recommended.temperature, Some(0.6));
        assert_eq!(recommended.top_p, Some(0.95));
        assert_eq!(recommended.top_k, None);
    }
}
