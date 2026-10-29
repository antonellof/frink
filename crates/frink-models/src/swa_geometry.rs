//! The ways a sliding-window layer can differ from a full-attention
//! layer in llama.cpp beyond its mask and its RoPE base: a second ROTARY
//! WIDTH, which frink now honours; a second HEAD WIDTH, which it
//! refuses by name; and YaRN switched off, which it refuses by name.
//!
//! 1. **A second rotary width.** The base loader seeds `n_rot_swa` from
//!    `n_rot_full` and lets `rope.dimension_count_swa` override it
//!    (`llama-model.cpp:1222-1223`), and `llama_hparams::n_rot(il)` is
//!    `is_swa(il) ? n_rot_swa : n_rot_full` (`llama-hparams.cpp:85-91`).
//!    It is a TWO-VALUED field, not an array -- measured over all 155
//!    graphs when the per-layer shape seam was built. Two things set the
//!    sliding width apart from the full one:
//!
//!    * the key, written by `conversion/laguna.py:115` for Laguna-XS.2
//!      (full layers 64 of 128 dims, sliding layers all 128,
//!      `laguna.cpp:43-45`) and by `conversion/gemma.py:673-674,700`
//!      for Gemma-4, which is on its own engine;
//!    * `step35.cpp:9`, `n_rot_full = n_rot_full / 2`, run AFTER the
//!      base loader seeded `n_rot_swa`, so the sliding layers keep the
//!      file's width and the full ones rotate half of it, with NO key
//!      saying so (the converter only asserts the checkpoint's
//!      `partial_rotary_factors` are 1.0 / 0.5, `step3.py:170`).
//!
//!    [`rotary_widths`] resolves both into `ModelConfig::rope_dim` (the
//!    full layers') and `ModelConfig::rope_dim_swa` (the sliding
//!    layers'), and `ModelConfig::layer_rope` hands each layer its own.
//!    The fused Metal launches take one width for every layer and are
//!    fenced off such a model (`Decoder::metal_can_serve_model`).
//!
//! 2. **A second head width.** `attention.key_length_swa` and
//!    `attention.value_length_swa` (`llama-model.cpp:1216-1219`) give
//!    the sliding layers their own K and V widths. frink carries ONE
//!    `head_dim` in every cache geometry and attention kernel, so a
//!    file whose sliding layers differ is refused naming the key
//!    ([`swa_geometry_refusal`]). Gemma-4 writes these; it is on its own
//!    engine.
//!
//! 3. **YaRN switched off on the sliding layers.** Three graphs on the
//!    generic path rope their sliding layers with `freq_scale = 1`,
//!    `ext_factor = 0`, `attn_factor = 1` while their full layers use
//!    the model's scaling: `olmo2.cpp:120-134` (Olmo-3), `mellum.cpp:
//!    128-142`, `laguna.cpp:48,181-193`. Measured by grepping every
//!    `ggml_rope_ext` call under `src/models/` for a zeroed
//!    `ext_factor`; `deepseek4.cpp` is the fourth and is on its own
//!    engine. frink folds the frequency half of YaRN into
//!    [`crate::config::RopeFreqs`], which already keeps the sliding
//!    layers unscaled for these three (`swa_rope_scale_follows_model`
//!    is false for all of them), but `rope_attn_factor` -- YaRN's
//!    `mscale` -- is one value for the whole model, so a file with a
//!    window AND a scaling is refused. A window with no scaling, or a
//!    scaling with no window, reduces both branches to the same RoPE
//!    and is served. [`swa_layers_unscaled_rope`] is the table; it used
//!    to be `arch == "olmo2"` in the loader, one row of three.
//!
//! 4. **No rope factors on the sliding layers.** `step35.cpp:247` is
//!    `rope_factors = is_swa ? nullptr : model.get_rope_factors(...)`:
//!    a `rope_freqs.weight` (Step-3.5-Flash's llama3 scaling) reaches
//!    the full layers only, and the sliding layers divide by nothing.
//!    Measured: `grep -l "is_swa ? nullptr" src/models/*.cpp` is
//!    `step35.cpp` and `gemma4-assistant.cpp` (its own engine); at the
//!    `604cc430` pin `lfm3.cpp:113` spells the same rule as
//!    `hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs`.
//!    [`swa_layers_drop_rope_factors`] is the table, and the loader
//!    sizes `RopeFreqs::full` to the full layers' width and
//!    `RopeFreqs::swa` to all ones at the sliding layers' width for it.
//!    Any OTHER architecture with two rotary widths and per-band
//!    divisors is refused by name ([`two_widths_with_factors_refusal`]),
//!    because one divisor vector cannot serve two widths and nothing
//!    upstream says which layers would take which.

/// The llama.cpp lines where `arch` ropes its sliding layers with the
/// scaling switched off, or `None` when its sliding layers inherit the
/// model's scaling like everyone else's.
pub fn swa_layers_unscaled_rope(arch: &str) -> Option<&'static str> {
    crate::models::ArchFacts::of(arch).swa.layers_unscaled_rope
}

/// [`swa_layers_unscaled_rope`] from this table alone (`olmo2` is in
/// `crate::models::olmo2`).
pub(crate) fn swa_layers_unscaled_rope_from_table(arch: &str) -> Option<&'static str> {
    match arch {
        "mellum" => Some("mellum.cpp:128-142"),
        "laguna" => Some("laguna.cpp:48,181-193"),
        _ => None,
    }
}

/// The llama.cpp line where `arch` halves the FULL-attention layers'
/// rotary width after the sliding width was seeded from the unhalved
/// value, or `None`. `step35.cpp:9` alone (measured: `grep -n
/// "n_rot_full / 2" src/models/*.cpp`).
pub fn full_layers_rotate_half(arch: &str) -> Option<&'static str> {
    match arch {
        "step35" => Some("step35.cpp:9"),
        _ => None,
    }
}

/// The llama.cpp line where `arch` reads `attention.sliding_window`
/// as a REQUIRED key (`ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,
/// hparams.n_swa)` with no `false`), or `None` where the key is
/// optional. A file without it fails to load upstream; here it would
/// have loaded as a windowless model that rotates no layer at all
/// (`crate::rope_layers::RopeLayers::SlidingOnly` with nothing sliding),
/// which is a different graph from the one llama.cpp refuses to build.
/// Measured: `grep -n "LLM_KV_ATTENTION_SLIDING_WINDOW," src/models/*.cpp`
/// without a trailing `false` on the generic path. Answers through the
/// registry (`kolibri1` lives in `crate::models`).
pub fn window_required(arch: &str) -> Option<&'static str> {
    crate::models::ArchFacts::of(arch).swa.window_required
}

/// The legacy table behind [`window_required`], for
/// `ArchFacts::from_tables`.
pub(crate) fn window_required_from_table(arch: &str) -> Option<&'static str> {
    match arch {
        "exaone-moe" => Some("exaone-moe.cpp:13"),
        "lfm3" => Some("lfm3.cpp:18"),
        _ => None,
    }
}

/// The llama.cpp line where `arch` reads `rope.freq_base_swa` as a
/// REQUIRED key, or `None` where the key is optional (and the sliding
/// layers' base falls back as `capability::swa_rope_base_follows_model`
/// says). A file without it fails to load upstream; here it would have
/// rotated the sliding layers at the fallback base. Measured: `grep -n
/// "LLM_KV_ROPE_FREQ_BASE_SWA" src/models/*.cpp` without a trailing
/// `false` is `dots3note.cpp:33`, `lfm3.cpp:20` and `muse-spark.cpp:7`,
/// and only `lfm3` is on the generic path.
pub fn swa_rope_base_required(arch: &str) -> Option<&'static str> {
    match arch {
        "lfm3" => Some("lfm3.cpp:20"),
        _ => None,
    }
}

/// The llama.cpp line where `arch` passes NO rope factors to its
/// sliding layers, or `None` when the sliding layers take the same
/// factors tensor as the full ones.
pub fn swa_layers_drop_rope_factors(arch: &str) -> Option<&'static str> {
    match arch {
        "step35" => Some("step35.cpp:247"),
        // `rope_factors = hparams.is_swa(il) ? nullptr : rope_freqs`,
        // landed after the 2026-09-19 pin. The converter writes the
        // tensor as the FULL layers' partial-RoPE mask
        // (`conversion/lfm3.py:101-111`), so a sliding layer that took
        // it would rotate a fraction of its head.
        "lfm3" => Some("lfm3.cpp:113"),
        _ => None,
    }
}

/// The three SWA-geometry keys as the file declares them, beside the
/// full-attention values they would override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwaGeometry {
    /// `{arch}.rope.dimension_count_swa`.
    pub rope_dim_swa: Option<u64>,
    /// `{arch}.attention.key_length_swa`.
    pub key_length_swa: Option<u64>,
    /// `{arch}.attention.value_length_swa`.
    pub value_length_swa: Option<u64>,
    /// The full-attention rotary width, `rope.dimension_count` or the
    /// head width when the key is absent -- what `n_rot_swa` defaults
    /// to (`llama-model.cpp:1222`).
    pub rope_dim_full: u64,
    /// The full-attention head width, what both `_swa` head keys
    /// default to (`llama-model.cpp:1216-1217`).
    pub head_dim: u64,
}

/// The two rotary widths a file resolves to, in `ModelConfig`'s
/// spelling: `None` is "the whole head" for `full` and "the same as
/// `full`" for `swa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotaryWidths {
    pub full: Option<usize>,
    pub swa: Option<usize>,
}

/// `n_rot_full` and `n_rot_swa` as llama.cpp resolves them
/// (`llama-model.cpp:1200-1223`, then the architecture's
/// `load_arch_hparams`), for a model with at least one sliding layer.
///
/// The caller checks that, because on a model with no sliding layer
/// the `_swa` value is dead metadata upstream too (`n_rot(il)` never
/// takes that branch) -- and for `step35`, whose every real export
/// slides on most layers, the halving applies regardless.
pub fn rotary_widths(arch: &str, g: SwaGeometry) -> RotaryWidths {
    let head_dim = g.head_dim as usize;
    let seeded = g.rope_dim_full as usize;
    // The sliding width is seeded from the file's full width BEFORE
    // `load_arch_hparams` runs (`llama-model.cpp:1222`), so step35's
    // halving at `:9` reaches only the full layers.
    let swa_raw = g.rope_dim_swa.map(|w| w as usize).unwrap_or(seeded);
    let full_raw = if full_layers_rotate_half(arch).is_some() {
        seeded / 2
    } else {
        seeded
    };
    // The whole head is `None`, whichever number said so.
    let narrow = |w: usize| (w > 0 && w < head_dim).then_some(w);
    let full = narrow(full_raw);
    // The sliding width is a SECOND width only when it really differs;
    // a restatement of the full one is `None` so the two-width fences
    // do not fire on it.
    let swa = (narrow(swa_raw) != full).then_some(swa_raw);
    RotaryWidths { full, swa }
}

/// Why this file cannot be served, or `None` when every declared
/// head-width key restates the full-attention value (which llama.cpp
/// treats exactly as the key being absent).
///
/// Only meaningful for a model with at least one sliding layer; the
/// caller checks that, because a key nothing reads is not a hazard.
pub fn swa_geometry_refusal(arch: &str, g: SwaGeometry) -> Option<String> {
    let (key, declared, full) = [
        ("attention.key_length_swa", g.key_length_swa, g.head_dim),
        ("attention.value_length_swa", g.value_length_swa, g.head_dim),
    ]
    .into_iter()
    .find_map(|(key, declared, full)| declared.filter(|v| *v != full).map(|v| (key, v, full)))?;
    Some(format!(
        "`{arch}.{key}` = {declared} while the full-attention layers use {full}: llama.cpp \
         gives the sliding layers their own head width (llama-model.cpp:1215-1219) and frink \
         carries one for the whole model, so honouring the file would cache half the layers \
         at the wrong width"
    ))
}

/// Why a model with two rotary widths AND per-band divisors cannot be
/// served, or `None` when the architecture says which layers take the
/// divisors ([`swa_layers_drop_rope_factors`]).
pub fn two_widths_with_factors_refusal(arch: &str, widths: RotaryWidths) -> Option<String> {
    if widths.swa.is_none() || swa_layers_drop_rope_factors(arch).is_some() {
        return None;
    }
    Some(format!(
        "the sliding layers rotate {:?} dims and the full layers {:?} (`n_rot(il)`, \
         llama-hparams.cpp:85-91), and the file declares per-band RoPE divisors (a \
         `rope_freqs.weight` tensor or a linear / YaRN scaling): one divisor vector cannot \
         serve two widths, and no line of `{arch}`'s graph says which layers take it -- \
         `step35.cpp:247` does, and is the only generic-path graph that does",
        widths.swa, widths.full
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> SwaGeometry {
        SwaGeometry {
            rope_dim_swa: None,
            key_length_swa: None,
            value_length_swa: None,
            rope_dim_full: 64,
            head_dim: 128,
        }
    }

    /// A head-width key restating the full value is what an absent key
    /// means to llama.cpp, so it is served; a key that differs is
    /// refused naming that key and both numbers. The rotary key is no
    /// longer a refusal here at all.
    #[test]
    fn a_swa_head_width_equal_to_the_full_one_is_served_and_a_different_one_names_the_key() {
        assert_eq!(swa_geometry_refusal("laguna", geometry()), None);
        let mut same = geometry();
        same.rope_dim_swa = Some(128);
        same.key_length_swa = Some(128);
        same.value_length_swa = Some(128);
        assert_eq!(swa_geometry_refusal("laguna", same), None);

        let mut kv = geometry();
        kv.value_length_swa = Some(256);
        let msg = swa_geometry_refusal("gemma4", kv).expect("refused");
        assert!(msg.contains("value_length_swa"), "{msg}");
        let mut k = geometry();
        k.key_length_swa = Some(64);
        let msg = swa_geometry_refusal("gemma4", k).expect("refused");
        assert!(
            msg.contains("`gemma4.attention.key_length_swa` = 64"),
            "{msg}"
        );
    }

    /// The four ways the two widths come out: the key (Laguna-XS.2),
    /// the halving (step35), both restating one width, and a key that
    /// merely says "whole head" on a whole-head model.
    #[test]
    fn rotary_widths_resolve_as_llama_cpp_resolves_n_rot_full_and_n_rot_swa() {
        // Laguna-XS.2: full 64 of 128, sliding all 128 (the key says so).
        let mut xs2 = geometry();
        xs2.rope_dim_swa = Some(128);
        assert_eq!(
            rotary_widths("laguna", xs2),
            RotaryWidths {
                full: Some(64),
                swa: Some(128)
            }
        );
        // step35: the file says 128 (the whole head), the graph halves
        // the full layers, the sliding ones keep 128.
        let mut s35 = geometry();
        s35.rope_dim_full = 128;
        assert_eq!(
            rotary_widths("step35", s35),
            RotaryWidths {
                full: Some(64),
                swa: Some(128)
            }
        );
        // The same file under any other architecture: one width, whole
        // head, nothing per layer.
        assert_eq!(
            rotary_widths("gemma3", s35),
            RotaryWidths {
                full: None,
                swa: None
            }
        );
        // A key restating the full width is no second width.
        let mut same = geometry();
        same.rope_dim_swa = Some(64);
        assert_eq!(
            rotary_widths("laguna", same),
            RotaryWidths {
                full: Some(64),
                swa: None
            }
        );
        // No key, partial rotary everywhere (Phi-3 with a window).
        assert_eq!(
            rotary_widths("phi3", geometry()),
            RotaryWidths {
                full: Some(64),
                swa: None
            }
        );
    }

    /// Two widths plus divisors is refused unless the graph drops the
    /// divisors on its sliding layers; one width with divisors is fine.
    #[test]
    fn two_widths_with_divisors_is_refused_except_where_the_graph_drops_them() {
        let two = RotaryWidths {
            full: Some(64),
            swa: Some(128),
        };
        let msg = two_widths_with_factors_refusal("laguna", two).expect("refused");
        assert!(
            msg.contains("Some(128)") && msg.contains("step35.cpp:247"),
            "{msg}"
        );
        assert_eq!(two_widths_with_factors_refusal("step35", two), None);
        assert_eq!(
            two_widths_with_factors_refusal(
                "laguna",
                RotaryWidths {
                    full: Some(64),
                    swa: None
                }
            ),
            None
        );
    }

    /// The three rows that zero YaRN on their sliding layers, each
    /// citing the lines, and a plain-SWA architecture answering `None`;
    /// the two one-row tables likewise.
    #[test]
    fn the_per_arch_tables_cite_lines_for_exactly_their_rows() {
        for arch in ["olmo2", "mellum", "laguna"] {
            let lines = swa_layers_unscaled_rope(arch).unwrap_or_else(|| panic!("{arch}"));
            assert!(lines.contains(".cpp:"), "{arch}: {lines}");
        }
        for arch in ["gemma3", "gpt-oss", "exaone4", "afmoe", "llama", "step35"] {
            assert_eq!(swa_layers_unscaled_rope(arch), None, "{arch}");
        }
        assert_eq!(full_layers_rotate_half("step35"), Some("step35.cpp:9"));
        assert_eq!(
            swa_layers_drop_rope_factors("step35"),
            Some("step35.cpp:247")
        );
        for arch in ["laguna", "gemma3", "llama", "mimo2"] {
            assert_eq!(full_layers_rotate_half(arch), None, "{arch}");
            assert_eq!(swa_layers_drop_rope_factors(arch), None, "{arch}");
        }
        // `lfm3` drops the factors without halving the width, and is
        // the one generic-path graph that requires the sliding base.
        assert_eq!(swa_layers_drop_rope_factors("lfm3"), Some("lfm3.cpp:113"));
        assert_eq!(full_layers_rotate_half("lfm3"), None);
        assert_eq!(swa_rope_base_required("lfm3"), Some("lfm3.cpp:20"));
        assert_eq!(window_required("lfm3"), Some("lfm3.cpp:18"));
        for arch in ["gemma3", "step35", "lfm2", "cohere2"] {
            assert_eq!(swa_rope_base_required(arch), None, "{arch}");
        }
    }
}
