//! **THE PER-HEAD LAYERNORM ON Q AND K** -- `attn_q_norm` of shape
//! `{n_embd_head_k, n_head}`, a DISTINCT weight per head, applied as
//! `LLM_NORM` (mean-subtracting LayerNorm, no bias) over each head, and
//! which of llama.cpp's graphs build it.
//!
//! # What it is
//!
//! `build_norm(Qcur, attn_q_norm, NULL, LLM_NORM, il)` over a `Qcur` of
//! `{n_embd_head, n_head, n_tokens}` normalises each head's
//! `n_embd_head` values -- mean subtracted, divided by the standard
//! deviation -- and multiplies by that head's OWN row of the weight.
//! Frink's QK norm (`capability::QkNormStyle`) is an RMSNorm in every
//! variant, and its per-head form shares ONE `head_dim`-long weight
//! across the heads; a `{head_dim, n_head}` weight is `n_head *
//! head_dim` long, which the loader's length rule would read as
//! `WholeVector` -- one RMS over the whole projection -- and the fused
//! Metal attention infers the same thing from the same length. Two
//! wrongs that agree with each other, which is why this is a refusal by
//! name and not a length case.
//!
//! # Reach -- MEASURED
//!
//! Over all 155 `src/models/*.cpp` (2026-09-12): `grep -A3
//! "build_norm(Qcur" | grep "LLM_NORM,"` is five graphs. Two of them
//! (`bert.cpp:124`, `mpt.cpp:110`) norm the WHOLE projection with a
//! `{n_embd}` weight and a bias, a different op on non-generic rows.
//! The three that create the weight `{n_embd_head_k, n_head}` and norm
//! per head are [`PER_HEAD_LAYER_NORM_QK`]: `stablelm.cpp:34-35,84-97`
//! (OPTIONAL; StableLM-2-12B), `command-r.cpp:28-31,80,87` (REQUIRED
//! at `n_layer >= 64`, i.e. Command-R+), `chameleon.cpp:33-35,91-102`
//! (REQUIRED, with an optional bias). Applied BEFORE RoPE in all three.
//!
//! # What this module does today
//!
//! Refuses the `stablelm` shape by name, from a fixture llama.cpp runs
//! (`tests/fixtures/stablelm_qknorm_tiny.gguf`; libllama's logits differ
//! from the plain file's by 8.73, measured). `command-r` and
//! `chameleon` are refused or deferred before any tensor is read, so
//! their rows are recorded facts for the seam that serves the op: a
//! fourth `QkNormStyle` carrying the per-head weight table, a LayerNorm
//! body in `decoder/qk_norm.rs`, and `metal_attn_view` refusing it
//! until the kernel has a mean-subtracting branch.

/// The three graphs, with the lines, and whether the tensor is
/// required.
pub const PER_HEAD_LAYER_NORM_QK: &[(&str, bool, &str)] = &[
    // `stablelm` (optional) states its row in `crate::models::stablelm`
    // (#403 step 7a).
    // `command-r` (required) is in `crate::models::command_r`.
    ("chameleon", true, "src/models/chameleon.cpp:33-35,91-102"),
    // Not per head: `{n_embd}` with a bias, one LayerNorm over the whole
    // projection (`mpt.cpp:52-56,106-115`, optional; MPT-7B has none,
    // some fine-tunes set `qk_ln`). The same refusal for the same
    // reason: a LayerNorm where frink's QK norm is an RMSNorm.
    ("mpt", false, "src/models/mpt.cpp:52-56,106-115"),
];

/// Whether `arch` applies its `attn_q_norm` / `attn_k_norm` as the
/// per-head LayerNorm.
pub fn uses_per_head_layer_norm_qk(arch: &str) -> bool {
    per_head_layer_norm_lines(arch).is_some()
}

/// The refusal reason for layer `l` of `arch` when it carries a Q or K
/// norm tensor and the architecture applies it as the per-head
/// LayerNorm; `None` when it carries neither or the architecture's QK
/// norm is an RMSNorm.
pub fn per_head_layer_norm_refusal(arch: &str, l: usize, has_q_or_k_norm: bool) -> Option<String> {
    per_head_layer_norm_refusal_for(per_head_layer_norm_lines(arch), l, has_q_or_k_norm)
}

/// `arch`'s [`PER_HEAD_LAYER_NORM_QK`] row: the llama.cpp lines.
pub fn per_head_layer_norm_lines(arch: &str) -> Option<&'static str> {
    crate::models::ArchFacts::of(arch)
        .tensors
        .per_head_layer_norm_qk
}

/// [`per_head_layer_norm_lines`] from [`PER_HEAD_LAYER_NORM_QK`] alone.
pub(crate) fn per_head_layer_norm_lines_from_table(arch: &str) -> Option<&'static str> {
    PER_HEAD_LAYER_NORM_QK
        .iter()
        .find(|(n, _, _)| *n == arch)
        .map(|(_, _, lines)| *lines)
}

/// [`per_head_layer_norm_refusal`] given the row (`ArchFacts`).
pub fn per_head_layer_norm_refusal_for(
    row: Option<&str>,
    l: usize,
    has_q_or_k_norm: bool,
) -> Option<String> {
    let lines = row?;
    if !has_q_or_k_norm {
        return None;
    }
    Some(format!(
        "`blk.{l}.attn_q_norm.weight` / `attn_k_norm.weight` are a per-head LayerNorm here: \
         `{{n_embd_head_k, n_head}}`, a distinct weight per head, applied as LLM_NORM (mean \
         subtracted) over each head before RoPE ({lines}). frink's QK norm is an RMSNorm \
         whose per-head weight is shared by every head, and a weight this long would be read \
         as one RMS over the whole projection; libllama's logits for this shape differ from \
         the same file without the tensors by 8.73 (measured, \
         tests/fixtures/stablelm_qknorm_tiny.gguf), so it stops \
         (`frink_models::qk_layer_norm`)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is a registered architecture, and no generic-path row
    /// in the table is also in the RMSNorm per-head tables: one
    /// architecture, one QK norm function.
    #[test]
    fn every_row_is_registered_and_has_one_qk_norm_function() {
        for (arch, _, lines) in PER_HEAD_LAYER_NORM_QK {
            assert!(
                crate::capability::resolve_profile(arch).is_some(),
                "`{arch}` ({lines}) is not a registered architecture"
            );
            assert!(
                !crate::capability::uses_per_head_scalar_qk_gain(arch),
                "`{arch}` cannot be both the scalar gain and the LayerNorm"
            );
        }
    }

    /// The refusal fires on the tensor, not on the name alone: a
    /// StableLM-2-1.6B file (no QK norm) is not refused for a tensor it
    /// does not have, and a Qwen3 file (RMS per head) is never refused
    /// here.
    #[test]
    fn the_refusal_needs_both_the_architecture_and_the_tensor() {
        assert!(per_head_layer_norm_refusal("stablelm", 0, false).is_none());
        let reason = per_head_layer_norm_refusal("stablelm", 3, true).expect("refused");
        assert!(reason.contains("blk.3.attn_q_norm.weight"), "{reason}");
        assert!(reason.contains("stablelm.cpp:34-35,84-97"), "{reason}");
        assert!(per_head_layer_norm_refusal("qwen3", 0, true).is_none());
        assert!(uses_per_head_layer_norm_qk("command-r"));
        assert!(!uses_per_head_layer_norm_qk("qwen3"));
    }
}
