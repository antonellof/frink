//! The check that every tensor in a GGUF was consumed by the loader, or
//! is one llama.cpp also leaves unread.

use frink_gguf::ShardedGguf;

use super::LoadError;

/// Tensor-name prefixes a text-generation load legitimately never
/// reads. Everything here is consumed by a *different* code path, not by
/// nothing: multimodal projector planes belong to `mmproj`, and the
/// per-shard split bookkeeping is metadata, not weights.
pub(super) const IGNORED_TENSOR_PREFIXES: &[&str] =
    &["mm.", "v.", "mmproj.", "resampler.", "audio."];

/// Fails the load when the checkpoint carries tensors this build never
/// looked at.
///
/// A tensor nobody reads is not a harmless extra: it is a term of the
/// real graph that ours is missing. gpt-oss ships `blk.N.attn_sinks`
/// and frink has no attention-sink code anywhere, so the file loads,
/// runs at full speed, and emits a different distribution than the model
/// it claims to be; the newer MoE recipes ship `ffn_exp_probs_b` the
/// same way. Both are silent today, and both are exactly what the
/// architecture registry cannot catch, because the architecture *string*
/// is one frink does support -- it is the checkpoint that carries more
/// than the registry entry promises.
///
/// This is deliberately the last check in the load: by here every loader
/// arm has had its chance to ask for what it needs, so what is left over
/// is what nothing in this build knows about.
///
/// `FRINK_ALLOW_UNKNOWN_TENSORS=1` downgrades it to a warning, for the
/// case where a human has decided the missing term does not matter (a
/// bias tensor of zeros, an auxiliary head that never runs). The default
/// is refusal: a wrong answer is worse than no answer.
pub fn assert_every_tensor_consumed(file: &ShardedGguf) -> Result<(), LoadError> {
    let mut left: Vec<String> = file
        .unconsumed_tensors()
        .into_iter()
        .filter(|n| !IGNORED_TENSOR_PREFIXES.iter().any(|p| n.starts_with(p)))
        .collect();
    if left.is_empty() {
        return Ok(());
    }
    left.sort();
    let shown = left.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
    let listing = if left.len() > 8 {
        format!("{shown}, … (+{} more)", left.len() - 8)
    } else {
        shown
    };
    if matches!(
        std::env::var("FRINK_ALLOW_UNKNOWN_TENSORS").ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    ) {
        eprintln!(
            "frink: WARNING -- {} tensor(s) in this checkpoint are never read \
             ({listing}); output may be wrong (FRINK_ALLOW_UNKNOWN_TENSORS=1)",
            left.len()
        );
        return Ok(());
    }
    Err(LoadError::UnconsumedTensors(left.len(), listing))
}
