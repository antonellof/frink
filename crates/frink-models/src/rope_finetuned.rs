//! `{arch}.rope.scaling.finetuned` as a SWITCH, for the one family that
//! reads it that way.
//!
//! `LLM_KV_ROPE_SCALING_FINETUNED` is a boolean llama.cpp otherwise only
//! prints. Granite is the exception, and says so in a comment:
//!
//! ```text
//! // src/models/granite.cpp:33-35
//! // Granite uses rope_finetuned as a switch for rope, so default to true
//! bool rope_finetuned = true;
//! ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
//! hparams.rope_finetuned = rope_finetuned;
//! ```
//!
//! and the graph then builds no `inp_pos` at all and skips both
//! `ggml_rope_ext` calls when it is false (`granite.cpp:130-133`,
//! `:206-219`; `granite-hybrid.cpp:14-17,178-185` the same). That is
//! the ALiBi class of divergence: a file that declares it would run
//! through a decoder rotating every Q and K head of every layer,
//! fluently, from positions the checkpoint does not encode that way.
//!
//! Until 2026-09-14 frink had no way to express "this architecture,
//! unrotated" and REFUSED the file. It has one since `gpt2` closed:
//! [`crate::rope_layers::RopeLayers::Never`], the rule that rotates
//! nothing, and [`unrotated`] is what selects it. The refusal's fixture
//! (`granite_norope_tiny.gguf`) has a libllama golden now
//! (`tests/granite_family_graphs.rs`).
//!
//! **Which files carry it.** `conversion/granite.py:253`, inside
//! `GraniteHybridModel`, writes it for EVERY Granite-4.0 hybrid export,
//! `false` unless the checkpoint is Bamba or has no Mamba layer at all
//! -- so every real Granite-4.0-H file runs its attention layers
//! WITHOUT RoPE (NoPE), and this switch is what a served `granitehybrid`
//! needs before its logits can match. `GraniteModel` and
//! `GraniteMoeModel` never write the key, so every converter-produced
//! `granite` / `granitemoe` file takes the `true` default and rotates;
//! and `granite-moe.cpp` does not read it at all, so a `granitemoe`
//! file that declares it false rotates too.

/// Architectures whose graph builds no positions and applies no RoPE
/// when `{arch}.rope.scaling.finetuned` is false. `granite` states it
/// in `crate::models::granite`; `granitemoe` runs the same graph but its
/// hparam loader never reads the key, so it is NOT gated
/// (`crate::models::granite_moe`, #439).
pub const ROPE_GATED_ON_FINETUNED: &[&str] = &["granitehybrid", "granite-hybrid"];

/// True when `arch` gates its RoPE on the key and the file declares it
/// false: the model rotates NOTHING (`RopeLayers::Never`).
///
/// `declared` is the file's value, `None` when the key is absent --
/// which is llama.cpp's `true` default for these rows, i.e. rotate.
pub fn unrotated(arch: &str, declared: Option<bool>) -> bool {
    unrotated_for(
        crate::models::ArchFacts::of(arch)
            .topology
            .rope_gated_on_finetuned,
        declared,
    )
}

/// [`unrotated`] given the row (`ArchFacts`).
pub fn unrotated_for(gated: bool, declared: Option<bool>) -> bool {
    gated && declared == Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch fires on exactly the value llama.cpp branches on, and
    /// on nothing else.
    ///
    /// The absent case is the one worth pinning: `None` is llama.cpp's
    /// `true` default for these rows, so treating a missing key as "no
    /// RoPE" would unrotate every real Granite-3 checkpoint there is.
    #[test]
    fn only_an_explicit_false_switches_rotation_off() {
        for arch in ROPE_GATED_ON_FINETUNED {
            assert!(unrotated(arch, Some(false)), "{arch}: an explicit false");
            assert!(
                !unrotated(arch, Some(true)),
                "{arch} rotates when the file says so"
            );
            assert!(
                !unrotated(arch, None),
                "{arch} must take llama.cpp's `true` default when the key is absent"
            );
        }
    }

    /// An architecture that merely CARRIES the key is not gated by it.
    ///
    /// llama.cpp reads `rope_finetuned` for every architecture and
    /// branches on it in two graphs. Honouring it everywhere the key
    /// appears would unrotate legacy `llama` files converted by
    /// `examples/convert_legacy_llama.py:915`, which writes it and means
    /// nothing by it.
    #[test]
    fn an_architecture_that_does_not_gate_its_rope_is_unaffected() {
        for arch in ["llama", "qwen3", "mamba2"] {
            assert!(
                !unrotated(arch, Some(false)),
                "{arch} does not read this key as a switch"
            );
        }
    }
}
