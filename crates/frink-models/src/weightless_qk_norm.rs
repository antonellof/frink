//! Llama 4's `Llama4TextL2Norm`: a per-head RMSNorm with NO weight on
//! Q and K, AFTER RoPE, on the layers that rotate.
//!
//! `llama4.cpp:182-188`:
//!
//! ```text
//! if (use_rope && hparams.use_kq_norm) {
//!     Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
//!     Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
//! }
//! ```
//!
//! `Qcur` is `{n_embd_head, n_head, n_tokens}` there, so the norm is
//! over each head; no `attn_q_norm` / `attn_k_norm` tensor is created
//! (`:63-95`), so a file carrying one is refused as unread. `:43` sets
//! `use_kq_norm = type != LLM_TYPE_17B_128E`, and `:38-40` derive the
//! type from `n_expert` ALONE: 128 experts (Maverick) is the one shape
//! without the norm, and everything else has it (Scout's 16, and any
//! count the switch calls `UNKNOWN`). A zero count never reaches the
//! graph (`:49-51` throw; `crate::moe_interleave::experts_required`).
//!
//! # Reach
//!
//! `grep -n 'ggml_rms_norm(ctx0, Qcur' src/models/*.cpp` over all
//! 155 graphs: `llama4.cpp:184` and `llama.cpp:164`, the latter under
//! the same `hparams.use_kq_norm`, which NOTHING but `llama4.cpp:43`
//! assigns (`grep -rn use_kq_norm src/`), so it is dead on the `llama`
//! graph. One reachable graph, so the fact is a `bool` on
//! `ModelConfig` -- there is no second shape to name -- read at the
//! post-RoPE QK-norm hook against `ModelConfig::layer_rotates`, and
//! refused by every fused Metal launch through
//! `Decoder::metal_can_serve_model`.
//!
//! Note what it is NOT: `crate::capability::QkNormStyle::PerHeadScalar`
//! is talkie's per-head RMS with a learned scalar gain on Q and a
//! weightless K; this is weightless on BOTH, and per layer.

/// True when `arch` with `n_experts` routed experts norms Q and K this
/// way (`llama4.cpp:43`: every Llama 4 but the 128-expert one).
///
/// The rule is `crate::models::llama4::weightless_qk_norm`; every other
/// architecture has none ([`no_weightless_qk_norm`]).
pub fn weightless_qk_norm(arch: &str, n_experts: usize) -> bool {
    (crate::models::ArchFacts::of(arch).rules.weightless_qk_norm)(arch, n_experts)
}

/// `HparamRules::LEGACY`'s: no graph outside `crate::models` norms Q and
/// K this way.
pub fn no_weightless_qk_norm(_arch: &str, _n_experts: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maverick_is_the_one_shape_without_it() {
        assert!(weightless_qk_norm("llama4", 16));
        assert!(weightless_qk_norm("llama4", 64));
        assert!(!weightless_qk_norm("llama4", 128));
        assert!(!weightless_qk_norm("llama", 16));
    }
}
