//! WHERE the routing weight multiplies a routed expert: its output, or
//! its input.
//!
//! `llama-graph.cpp:1947`:
//!
//! ```text
//! const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN
//! ```
//!
//! and `:2086-2090` multiply the (repeated) input by the selected
//! weights before `build_lora_mm_id`, where `:2246-2249` multiply the
//! expert OUTPUT for everyone else. So Llama 4 computes
//! `down(silu(gate(w x)) * up(w x))` where every other graph computes
//! `w * down(silu(gate(x)) * up(x))`, and since SwiGLU is not
//! homogeneous the two differ on every token (the fixture's logits
//! move by 0.86 between them, measured). Reach: `grep -n LLM_ARCH_LLAMA4
//! src/llama-graph.cpp` is this line and `:1999`, which selects the
//! top-k on the raw logits instead of the sigmoid probabilities -- a
//! monotone map, so the same experts -- and nothing else. One graph of
//! 140, so the fact is a `bool` on `MoeLayerConfig`.
//!
//! It is served by scaling the INPUT ROW per slot and carrying a
//! weight of 1 through the output sum, at the two host sites that
//! gather a routed expert's input (`Decoder::cpu_moe_serial_experts`
//! for the row body, `Decoder::moe_ffn_batch` for the batched one);
//! the shared-activation kernel `cpu_moe_topk_parallel_slots` quantises
//! ONE input for every slot and declines the model, and every fused
//! Metal MoE launch is behind `Decoder::metal_can_serve_model`, which
//! refuses Llama 4 on its chunked window.

use std::borrow::Cow;

/// `true` when `arch` weights the input (module doc).
///
/// A row of the architecture's `crate::models` definition: `llama4`
/// (`llama-graph.cpp:1947,2086-2090`) is the one graph with it.
pub fn weight_before_ffn(arch: &str) -> bool {
    crate::models::ArchFacts::of(arch)
        .moe
        .routed_weight_before_ffn
}

/// One slot's expert input and the weight its output carries: `(x, w)`
/// where the weight sits on the output, `(w * x, 1)` where it sits on
/// the input.
pub fn routed_slot(x: &[f32], w: f32, before_ffn: bool) -> (Cow<'_, [f32]>, f32) {
    if before_ffn {
        (Cow::Owned(x.iter().map(|v| v * w).collect()), 1.0)
    } else {
        (Cow::Borrowed(x), w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weight_moves_from_the_output_to_the_input() {
        let x = [1.0f32, -2.0, 4.0];
        let (input, w) = routed_slot(&x, 0.5, false);
        assert_eq!((&*input, w), (&x[..], 0.5));
        let (input, w) = routed_slot(&x, 0.5, true);
        assert_eq!((&*input, w), (&[0.5f32, -1.0, 2.0][..], 1.0));
        assert!(weight_before_ffn("llama4") && !weight_before_ffn("mimo2"));
    }
}
