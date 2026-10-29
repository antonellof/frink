//! The one place this crate encodes a compute dispatch, and the counters
//! that measure what a decode token costs the host.
//!
//! GitHub issue #149 measured that 26-29% of Metal decode wall time was
//! not GPU time, and read that as host-side command encoding:
//! `dispatchThreadgroups`, `emitComputeProgramVariantAndArguments`,
//! `memoryBarrierWithResources`. Counting was the first step, and the
//! count retired that reading in two stages: PR #156 removed 13% of the
//! dispatches for 2.3% of the host time, and `crate::timing` then
//! clocked the encode phase directly at 0.15 ms per token on
//! Llama-3.2-1B and 0.4 ms on Gemma-2-2B, about 2% of wall. Most of
//! the 26% was a second, untimed command buffer executing on the GPU.
//! The counters stay because a claim about encode work still needs
//! them to be checked.
//!
//! So every `dispatchThreadgroups_threadsPerThreadgroup` in the crate
//! goes through [`dispatch_counted`], and every barrier through
//! [`note_barrier`]. Two counters that must agree with what is actually
//! encoded is the repo's dominant bug shape, so nothing is allowed to
//! bypass them: [`no_encode_site_bypasses_the_counter`] scans the crate's
//! own source and fails if a raw call reappears.
//!
//! Read the counters with [`metal_encode_stats`], reset them with
//! [`metal_encode_stats_reset`]. `FRINK_METAL_GPU_TIMING=1` prints them
//! per token beside the GPU-versus-wall time.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLSize};
use std::sync::atomic::{AtomicU64, Ordering};

static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);
static BARRIER_COUNT: AtomicU64 = AtomicU64::new(0);
static BEGIN_OP_COUNT: AtomicU64 = AtomicU64::new(0);

/// What one encode pass cost the host, in the two units that matter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EncodeStats {
    /// Compute dispatches encoded.
    pub dispatches: u64,
    /// `memoryBarrierWithResources` calls emitted.
    pub barriers: u64,
    /// Hazard checks performed, i.e. ops offered to the tracker. The
    /// `barriers / begin_ops` ratio is 1.00 for a fully serialised pass
    /// and lower when dispatches overlap.
    pub begin_ops: u64,
}

/// Snapshot the process-wide counters.
pub fn metal_encode_stats() -> EncodeStats {
    EncodeStats {
        dispatches: DISPATCH_COUNT.load(Ordering::Relaxed),
        barriers: BARRIER_COUNT.load(Ordering::Relaxed),
        begin_ops: BEGIN_OP_COUNT.load(Ordering::Relaxed),
    }
}

pub fn metal_encode_stats_reset() {
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
    BARRIER_COUNT.store(0, Ordering::Relaxed);
    BEGIN_OP_COUNT.store(0, Ordering::Relaxed);
}

/// Encode one compute dispatch, counting it.
///
/// The counting is the point: see the module docs. Every call site in
/// the crate uses this instead of the raw Metal method.
#[inline]
pub(crate) fn dispatch_counted(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
) {
    DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
}

/// Record one emitted barrier, returning the new running total.
///
/// Called from the two barrier primitives in [`crate::gpu`], which are
/// the only places the crate touches Metal's barrier API, so this counts
/// barriers from the hazard tracker AND the hand-placed ones in the MoE
/// encode paths.
#[inline]
pub(crate) fn note_barrier() -> u64 {
    BARRIER_COUNT.fetch_add(1, Ordering::Relaxed) + 1
}

/// Record one hazard check offered to the tracker (barrier or not).
#[inline]
pub(crate) fn note_begin_op() {
    BEGIN_OP_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    /// A dispatch that does not go through [`super::dispatch_counted`] is
    /// invisible to the per-token accounting, and the accounting is the
    /// only evidence that a fusion removed work. A counter beside the
    /// thing it counts, with nothing forcing them to agree, is this
    /// repo's dominant defect shape, so make the disagreement a red test
    /// rather than a wrong number in a PR body.
    #[test]
    fn no_encode_site_bypasses_the_counter() {
        // Every `.rs` in this crate, found by walking `src/` rather than
        // listed by hand. The hand list this replaced named 16 of the 27
        // files, and six raw dispatches in the four it missed (`gdn.rs`,
        // `gdn_chunk.rs`, `gdn_head.rs`, `hadamard.rs`) were invisible to
        // the counter, which is the counter frink-models reads to prove
        // a CPU-pinned process encoded no Metal work at all.
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut sources: Vec<(String, String)> = Vec::new();
        let mut dirs = vec![src_dir.clone()];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).expect("src/ is readable") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    let name = path
                        .strip_prefix(&src_dir)
                        .expect("under src/")
                        .to_string_lossy()
                        .into_owned();
                    let text = std::fs::read_to_string(&path).expect("source is readable");
                    sources.push((name, text));
                }
            }
        }
        assert!(
            sources.iter().any(|(n, _)| n == "gpu.rs") && sources.len() > 20,
            "the walk found {} files; it is not reading this crate",
            sources.len()
        );
        // This file holds the wrapper (the one legitimate caller of the
        // raw dispatch method) and the patterns this test searches for,
        // so it is not itself a subject.
        let subjects = || sources.iter().filter(|(name, _)| name != "dispatch.rs");

        let mut raw = Vec::new();
        for (name, src) in subjects() {
            for (i, line) in src.lines().enumerate() {
                if line.contains(".dispatchThreadgroups_threadsPerThreadgroup(")
                    || line.contains(".dispatchThreads_threadsPerThreadgroup(")
                {
                    raw.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            raw.is_empty(),
            "these dispatches bypass dispatch_counted, so the per-token \
             dispatch count under-reports what the host actually encodes: {raw:#?}"
        );

        // And the same for barriers. `memory_barrier_buffers` /
        // `memory_barrier_resources` in gpu.rs are the crate's only
        // callers of Metal's barrier API, and they count what they emit.
        let mut raw_barriers = Vec::new();
        for (name, src) in subjects() {
            for (i, line) in src.lines().enumerate() {
                if !line.contains(".memoryBarrierWith") {
                    continue;
                }
                let inside_the_primitive = name == "gpu.rs"
                    && (line.contains("encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers)")
                        || line.contains("encoder.memoryBarrierWithResources_count("));
                if !inside_the_primitive {
                    raw_barriers.push(format!("{name}:{}: {}", i + 1, line.trim()));
                }
            }
        }
        assert!(
            raw_barriers.is_empty(),
            "these barriers bypass the counting primitives in gpu.rs and are \
             therefore invisible to the per-token barrier count: {raw_barriers:#?}"
        );
    }
}
