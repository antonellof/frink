//! The NTK-alpha RoPE base rescale, for the architectures that apply it.
//!
//! `{arch}.rope.scaling.alpha` (`LLM_KV_ROPE_SCALING_ALPHA`) is read for
//! EVERY architecture in llama.cpp's generic hparams loader
//! (`llama-model.cpp:1186`, optional, default `0.0f`), but only two
//! architectures do anything with it: `hunyuan-vl` and the
//! `hunyuan-dense` that inherits its `load_arch_hparams`
//! (`models.h:1830-1832`). `src/models/hunyuan-vl.cpp:8-12` is the whole
//! of it:
//!
//! ```text
//! // XDRoPE / NTK-aware scaling: base = rope_theta * alpha^(dim / (dim - 2))
//! if (hparams.rope_scaling_alpha > 0.0f) {
//!     const int dim = hparams.n_embd_head_k();
//!     hparams.rope_freq_base_train = hparams.rope_freq_base_train
//!         * powf(hparams.rope_scaling_alpha, (float)dim / (float)(dim - 2));
//! }
//! ```
//!
//! **Why this is a list and not a generic rule.** The key is generic and
//! the behaviour is not. Applying the rescale wherever the key appears
//! would change the RoPE base of every other architecture that carries
//! it, which is the "two structures that must agree about one thing"
//! shape this repo keeps paying for -- llama.cpp reads the key in one
//! place and applies it in another, and only the second place is
//! per-architecture.
//!
//! **What a real `hunyuan-dense` checkpoint actually carries.** Not this
//! key. `conversion/hunyuan.py:254-281` (`HunYuanModel`, the
//! `HUNYUAN_DENSE` converter) does the same arithmetic in Python --
//! `scaled_base = base * (alpha ** (dim / (dim - 2)))` at :270 -- and
//! writes the ALREADY-SCALED value through `add_rope_freq_base`, with no
//! alpha key at all. The converter line that writes
//! `add_rope_scaling_alpha` is :356, and it is in `HunyuanVLTextModel`,
//! whose `model_arch` is `HUNYUAN_VL`, a different GGUF architecture
//! string and a different (still-refusing) frink row.
//!
//! So for every converter-produced `hunyuan-dense` file this function
//! returns the base unchanged, and the rescale exists for the case
//! llama.cpp will still honour: a file that does carry the key. That is
//! not a gate that cannot fire -- it is arithmetic that must agree with
//! llama.cpp's when the key is there, and
//! `tests/one_match_arm_graphs.rs` drives it out of a fixture that
//! carries `hunyuan-dense.rope.scaling.alpha` and compares against
//! libllama's own logits.

/// The architectures still in the legacy tables whose
/// `load_arch_hparams` applies `{arch}.rope.scaling.alpha` to the
/// trained RoPE base.
///
/// `hunyuan-vl` is here for completeness of the reading even though it
/// resolves to a deferred multimodal row rather than the generic
/// decoder: leaving it out would make the list disagree with
/// `models.h:1830`, where `hunyuan-dense` inherits the behaviour FROM
/// it. `hunyuan-dense` states it in `crate::models::hunyuan_dense`.
pub const NTK_ALPHA_RESCALED_ROPE_BASE: &[&str] = &["hunyuan-vl"];

/// [`NTK_ALPHA_RESCALED_ROPE_BASE`] alone, for
/// `ArchFacts::from_tables`.
pub(crate) fn rescales_rope_base_from_table(arch: &str) -> bool {
    NTK_ALPHA_RESCALED_ROPE_BASE.contains(&arch)
}

/// llama.cpp's `hunyuan-vl.cpp:8-12`, for the base this architecture
/// should rotate at.
///
/// Returns `base` unchanged when the architecture does not apply the
/// rescale, when the key is absent, or when the declared alpha is not
/// positive -- llama.cpp's own `> 0.0f` guard, and the reason a default
/// of `0.0` is a no-op rather than a collapse to zero frequency.
///
/// `head_dim` is `hparams.n_embd_head_k()`, which frink carries as
/// `ModelConfig::head_dim`. A head_dim of 2 or less would divide by zero
/// or negate the exponent; llama.cpp has no guard because no attention
/// head is that narrow, and this returns the base unchanged rather than
/// producing an infinity.
pub fn ntk_alpha_scaled_rope_base(
    arch: &str,
    base: f32,
    head_dim: usize,
    alpha: Option<f32>,
) -> f32 {
    ntk_alpha_scaled_rope_base_for(
        crate::models::ArchFacts::of(arch).scaling.rope_ntk_alpha,
        base,
        head_dim,
        alpha,
    )
}

/// [`ntk_alpha_scaled_rope_base`] given the row
/// (`ScalingFacts::rope_ntk_alpha`), for the config loader.
pub(crate) fn ntk_alpha_scaled_rope_base_for(
    rescales: bool,
    base: f32,
    head_dim: usize,
    alpha: Option<f32>,
) -> f32 {
    if !rescales {
        return base;
    }
    let Some(alpha) = alpha.filter(|a| *a > 0.0) else {
        return base;
    };
    if head_dim <= 2 {
        return base;
    }
    // f32 throughout, and in llama.cpp's order: the exponent is a float
    // division of two ints, then `powf`. Doing the division in f64 moves
    // the last bits of the base and, at a base of ~1e6, the last bits of
    // every RoPE angle with it.
    let exponent = head_dim as f32 / (head_dim as f32 - 2.0);
    base * alpha.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula, against the converter's own Python for the same
    /// inputs.
    ///
    /// `conversion/hunyuan.py:266-270` defaults alpha to 50 and computes
    /// `base * (alpha ** (dim / (dim - 2)))`. For HunYuan-7B's real
    /// shape (head_dim 128, base 10000, alpha 50) that is
    /// 10000 * 50^(128/126) = 10000 * 53.235... The assertion is on the
    /// arithmetic, not on a value copied out of a run.
    #[test]
    fn the_rescale_is_base_times_alpha_to_the_dim_over_dim_minus_two() {
        let got = ntk_alpha_scaled_rope_base("hunyuan-dense", 10_000.0, 128, Some(50.0));
        let want = 10_000.0f32 * 50.0f32.powf(128.0 / 126.0);
        assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        assert!(got > 10_000.0, "alpha > 1 must EXPAND the base, got {got}");
    }

    /// `hunyuan-dense` states the row in `crate::models`, `hunyuan-vl`
    /// is still in the legacy table, and both answer through the same
    /// public lookup.
    #[test]
    fn both_rows_rescale_whichever_place_states_them() {
        assert!(rescales_rope_base_from_table("hunyuan-vl"));
        assert!(!rescales_rope_base_from_table("hunyuan-dense"));
        assert!(
            crate::models::ArchFacts::of("hunyuan-dense")
                .scaling
                .rope_ntk_alpha
        );
        assert!(
            crate::models::ArchFacts::of("hunyuan-vl")
                .scaling
                .rope_ntk_alpha
        );
        for arch in ["hunyuan-dense", "hunyuan-vl"] {
            assert!(ntk_alpha_scaled_rope_base(arch, 10_000.0, 128, Some(50.0)) > 10_000.0);
        }
    }

    /// An architecture that is not on the list keeps its base even when
    /// its file carries the key.
    ///
    /// This is the half that matters: llama.cpp reads the key
    /// generically and applies it in exactly two graphs, so a frink
    /// that applied it generically would rotate every `qwen3`,
    /// `deepseek` and `llama` checkpoint carrying the key at a base
    /// llama.cpp never uses.
    #[test]
    fn an_architecture_that_does_not_apply_the_rescale_keeps_its_base() {
        for arch in ["llama", "qwen3", "hunyuan-moe"] {
            assert_eq!(
                ntk_alpha_scaled_rope_base(arch, 10_000.0, 128, Some(50.0)),
                10_000.0,
                "{arch} must not be rescaled"
            );
        }
    }

    /// llama.cpp's `> 0.0f` guard, both halves.
    #[test]
    fn a_missing_or_non_positive_alpha_is_a_no_op() {
        for alpha in [None, Some(0.0), Some(-1.0)] {
            assert_eq!(
                ntk_alpha_scaled_rope_base("hunyuan-dense", 10_000.0, 128, alpha),
                10_000.0,
                "alpha {alpha:?} must leave the base alone"
            );
        }
    }

    /// An alpha below 1 CONTRACTS the base, and the function does not
    /// quietly clamp it.
    ///
    /// Worth pinning because the NTK-aware recipe is described
    /// everywhere as "extend the context by raising the base", and a
    /// reader could add a `max(1.0)` that llama.cpp does not have.
    #[test]
    fn an_alpha_below_one_contracts_the_base_rather_than_being_clamped() {
        let got = ntk_alpha_scaled_rope_base("hunyuan-dense", 10_000.0, 8, Some(0.25));
        assert!(got < 10_000.0, "got {got}");
        assert!((got - 10_000.0f32 * 0.25f32.powf(8.0 / 6.0)).abs() < 1e-3);
    }

    /// A head_dim of 2 would divide by zero; the base survives instead.
    #[test]
    fn a_two_wide_head_returns_the_base_rather_than_an_infinity() {
        let got = ntk_alpha_scaled_rope_base("hunyuan-dense", 10_000.0, 2, Some(50.0));
        assert_eq!(got, 10_000.0);
        assert!(got.is_finite());
    }
}
