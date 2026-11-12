//! **THE SAME PHYSICAL LAYERS RUN MORE THAN ONCE** -- Nanbeige's
//! `num_loops`, as one value a `ModelConfig` carries and one rule for
//! which logical layer runs which weights and where the loop norm sits.
//!
//! # What it is
//!
//! `src/models/nanbeige.cpp:6-12` read `{arch}.num_loops` (default 1)
//! and `{arch}.skip_loop_final_norm` (default false). With `n_loops >
//! 1`, `:19-31` set `n_layer_all = n_layer_phys * n_loops` and COPY the
//! per-layer head / kv-head / ff / swa arrays from physical layer `i`
//! to every logical slot `i + j * n_phys`; `:47-66` create tensors for
//! the `n_phys` physical layers only and `:69-73` alias
//! `layers[i + j * n_phys] = layers[i]`. So the graph (`:94-176`) walks
//! `n_layer_all` logical layers, each with its OWN KV cache and its own
//! row in every per-layer table, over `n_phys` sets of weights. After
//! the last logical layer of every pass but the final one (`:167-175`:
//! `(il + 1) % n_phys == 0 && (il + 1) < n_layer`), the running
//! residual is normed with `output_norm` -- the lm_head's norm, the
//! same tensor -- unless `skip_loop_final_norm`. Everything inside a
//! pass is plain Llama.
//!
//! # Reach -- MEASURED
//!
//! `grep -l 'n_loops\|n_layer_phys' src/models/*.cpp` over all 155
//! graphs (2026-09-12) is `nanbeige.cpp`; `LLM_KV_NUM_LOOPS` and
//! `LLM_KV_SKIP_LOOP_FINAL_NORM` are read nowhere else. So
//! [`LOOP_READERS`] has one row and the keys are dead metadata on every
//! other architecture, as upstream (the `yarn_log_multiplier` rule,
//! `crate::yarn_magnitude`).
//!
//! # What frink does with it
//!
//! The weights are shared and the KV is not, and the seam says exactly
//! that rather than copying weights: `Decoder::layers` stays the
//! PHYSICAL vector the loader filled, `ModelConfig::n_layers` is the
//! LOGICAL count (so every KV cache, per-layer table and budget is
//! sized per logical layer, as `n_layer_all` sizes them upstream), and
//! `Decoder::layer_for(l)` is the ONE mapping from a logical index to
//! its weights, `l % n_phys`. The three host bodies iterate logical
//! indices and ask it; `LayerLoops::loop_norm_after(l)` says where the
//! loop norm goes and `Decoder::final_norm` is the tensor that goes
//! there. The per-layer shape arrays are replicated at load, as
//! `:24-26` replicate them.
//!
//! Every fused Metal launch indexes `layers[l]` and `metal_kvs[l]` with
//! one `l`, so `metal_can_serve_model` refuses a looped model; the host
//! bodies serve it. A LoRA adapter attaches to physical layers by
//! tensor name and so reaches every logical layer that shares them,
//! which is what `build_lora_mm` on an aliased `layers[il]` does too.

use crate::LoadError;
use frink_gguf::TensorSource;

/// Architectures whose `load_arch_hparams` reads `num_loops`, with the
/// line.
pub const LOOP_READERS: &[(&str, &str)] = &[
    ("nanbeige", "src/models/nanbeige.cpp:6-31,167-175"),
    ("hrm_text", "src/models/hrm-text.cpp:10-23,47-87,183-196"),
];

/// A model whose logical layers are `n_loops` passes over `n_phys`
/// physical ones. Only ever constructed with `n_loops >= 2`: a file
/// declaring 1 (or nothing) is a plain model and carries `None`.
/// The norm a pass boundary applies to the residual, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopNorm {
    /// `output_norm`, the model's own weighted final norm
    /// (`nanbeige.cpp:167-175`).
    Output,
    /// A WEIGHTLESS RMS: `hrm-text.cpp:162` closes every stack with
    /// `build_norm(cur, nullptr, nullptr, LLM_NORM_RMS, ...)`, and the
    /// architecture has no `output_norm` tensor at all -- the last
    /// stack's norm IS the final one.
    Weightless,
}

/// Which of HRM-Text's two residual streams a finished stack updates.
///
/// `hrm-text.cpp:183-196`: each H cycle runs `l_cycles` LOW stacks and
/// then one HIGH stack, every stack reading `zH + zL` and replacing one
/// of the two with its output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HrmStream {
    /// A low-cycle pass: its output becomes `zL`.
    Low,
    /// The last pass of an H cycle: its output becomes `zH`.
    High,
}

/// A model whose logical layers are several passes over fewer physical
/// ones.
///
/// Two shapes, because two graphs upstream do it and they differ in
/// every detail but that: which physical layer a pass runs, what norm
/// sits at a pass boundary, and whether the passes share ONE residual
/// stream (`nanbeige`) or recombine TWO (`hrm_text`). Parameterising
/// one enum rather than adding a second field for each difference is
/// what keeps the three host bodies asking one question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerLoops {
    /// `nanbeige.cpp:6-31`: `n_loops` passes over `n_phys` layers, one
    /// residual stream, `output_norm` between passes.
    Repeat {
        /// Physical layers: the blocks the file holds tensors for.
        n_phys: usize,
        /// Passes over them; `>= 2`.
        n_loops: usize,
        /// `{arch}.skip_loop_final_norm`: no norm between passes.
        skip_loop_final_norm: bool,
    },
    /// `hrm-text.cpp:47-87,183-196`: TWO stacks of `lps` layers each --
    /// the file holds `2 * lps` blocks -- replayed over
    /// `h_cycles * (l_cycles + 1)` passes, with a weightless norm after
    /// every pass and the two residual streams recombined at every
    /// stack boundary.
    Hrm {
        /// `{arch}.hrm.layers_per_stack`.
        lps: usize,
        /// `{arch}.hrm.h_cycles`.
        h_cycles: usize,
        /// `{arch}.hrm.l_cycles`.
        l_cycles: usize,
    },
}

impl LayerLoops {
    /// llama.cpp's `n_layer_all`, the count every per-layer thing --
    /// the KV caches above all -- is sized by.
    pub fn logical_layers(&self) -> usize {
        match *self {
            Self::Repeat {
                n_phys, n_loops, ..
            } => n_phys * n_loops,
            // `hrm-text.cpp:22-23` asserts exactly this against
            // `block_count`.
            Self::Hrm {
                lps,
                h_cycles,
                l_cycles,
            } => lps * h_cycles * (l_cycles + 1),
        }
    }

    /// The physical layer logical layer `l` runs.
    pub fn physical(&self, l: usize) -> usize {
        match *self {
            Self::Repeat { n_phys, .. } => l % n_phys,
            // `hrm-text.cpp:57-68`: the first low pass creates blocks
            // `[0, lps)` and the first high pass blocks `[lps, 2*lps)`;
            // every later pass ALIASES one of the two.
            Self::Hrm { lps, l_cycles, .. } => {
                let pass = l / lps;
                let is_high = pass % (l_cycles + 1) == l_cycles;
                (if is_high { lps } else { 0 }) + l % lps
            }
        }
    }

    /// Physical layers: how many blocks the file holds tensors for.
    pub fn physical_layers(&self) -> usize {
        match *self {
            Self::Repeat { n_phys, .. } => n_phys,
            Self::Hrm { lps, .. } => 2 * lps,
        }
    }

    /// The norm applied to the residual AFTER logical layer `l`, if
    /// any.
    ///
    /// `nanbeige` norms at the end of every pass but the LAST (its own
    /// `output_norm` follows that one anyway) and only when the file
    /// does not skip it; `hrm_text` norms at the end of EVERY stack,
    /// including the last, because it has no other final norm.
    pub fn loop_norm_after(&self, l: usize) -> Option<LoopNorm> {
        match *self {
            Self::Repeat {
                n_phys,
                skip_loop_final_norm,
                ..
            } => (!skip_loop_final_norm
                && (l + 1).is_multiple_of(n_phys)
                && l + 1 < self.logical_layers())
            .then_some(LoopNorm::Output),
            Self::Hrm { lps, .. } => (l + 1).is_multiple_of(lps).then_some(LoopNorm::Weightless),
        }
    }

    /// Does logical layer `l` START a stack, i.e. does the residual it
    /// reads come from recombining the two streams?
    pub fn stack_starts_at(&self, l: usize) -> bool {
        match *self {
            Self::Repeat { .. } => false,
            Self::Hrm { lps, .. } => l.is_multiple_of(lps),
        }
    }

    /// Which stream the pass ending at logical layer `l` writes, or
    /// `None` when `l` does not end a stack.
    pub fn stream_after(&self, l: usize) -> Option<HrmStream> {
        match *self {
            Self::Repeat { .. } => None,
            Self::Hrm { lps, l_cycles, .. } => {
                if !(l + 1).is_multiple_of(lps) {
                    return None;
                }
                let pass = l / lps;
                Some(if pass % (l_cycles + 1) == l_cycles {
                    HrmStream::High
                } else {
                    HrmStream::Low
                })
            }
        }
    }
}

/// Reads the two keys for a file: `Some` only for an architecture whose
/// graph loops AND a `num_loops` above 1.
pub fn read_layer_loops(
    file: &impl TensorSource,
    arch: &str,
    n_phys: usize,
) -> Result<Option<LayerLoops>, LoadError> {
    if !LOOP_READERS.iter().any(|(name, _)| *name == arch) {
        return Ok(None);
    }
    let key = |k: &str| format!("{arch}.{k}");
    if arch == "hrm_text" {
        // `hrm-text.cpp:10-12` reads all three REQUIRED and `:17-19`
        // asserts each is above zero; `:22-23` then asserts the layer
        // count IS `lps * h * (l + 1)`, which the caller checks below
        // against `block_count`.
        let read = |k: &str| -> Result<usize, LoadError> {
            file.metadata_u64(&key(k))
                .map(|v| v as usize)
                .ok_or_else(|| LoadError::MissingHparam(key(k)))
        };
        let (lps, h_cycles, l_cycles) = (
            read("hrm.layers_per_stack")?,
            read("hrm.h_cycles")?,
            read("hrm.l_cycles")?,
        );
        if lps == 0 || h_cycles == 0 || l_cycles == 0 {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "hrm.layers_per_stack / h_cycles / l_cycles are {lps} / {h_cycles} / \
                     {l_cycles}; llama.cpp asserts each is above zero (hrm-text.cpp:17-19)"
                ),
            ));
        }
        // Checked, because the three come off the wire and the product is
        // only ever compared with `block_count`: an overflow is a file
        // that cannot be the schedule `block_count` says.
        let logical = lps
            .checked_mul(h_cycles)
            .and_then(|n| n.checked_mul(l_cycles.checked_add(1)?));
        if logical != Some(n_phys) {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "block_count is {n_phys} but hrm.layers_per_stack * hrm.h_cycles * \
                     (hrm.l_cycles + 1) is {}; llama.cpp asserts they are equal \
                     (hrm-text.cpp:22-23)",
                    logical.map_or("past usize::MAX".to_string(), |n| n.to_string())
                ),
            ));
        }
        let schedule = LayerLoops::Hrm {
            lps,
            h_cycles,
            l_cycles,
        };
        // `hrm-text.cpp:22-23`: the GGUF block count IS the expanded
        // slot count, checked above with the product unwrapped.
        debug_assert_eq!(schedule.logical_layers(), n_phys);
        return Ok(Some(schedule));
    }
    let n_loops = file.metadata_u64(&key("num_loops")).unwrap_or(1) as usize;
    if n_loops == 0 {
        // `GGML_ASSERT(n_loops_u >= 1)`, nanbeige.cpp:8.
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            "num_loops is 0; llama.cpp asserts it is at least 1".to_string(),
        ));
    }
    if n_loops == 1 {
        return Ok(None);
    }
    // `GGML_ASSERT(n_layer_phys * n_loops_u <= LLAMA_MAX_LAYERS)`,
    // nanbeige.cpp:16: every logical layer is a KV slot, so an unbounded
    // loop count would size one per loop.
    let max = crate::mtp_blocks::LLAMA_MAX_LAYERS;
    if n_phys.checked_mul(n_loops).is_none_or(|n| n > max) {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "num_loops = {n_loops} over {n_phys} layers is more than llama.cpp's \
                 LLAMA_MAX_LAYERS ({max}) logical layers (nanbeige.cpp:16)"
            ),
        ));
    }
    let skip_loop_final_norm = file
        .metadata_bool(&key("skip_loop_final_norm"))
        .unwrap_or(false);
    Ok(Some(LayerLoops::Repeat {
        n_phys,
        n_loops,
        skip_loop_final_norm,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two physical layers run twice: logical 0,1,2,3 are physical
    /// 0,1,0,1; the loop norm sits after logical 1 and NOT after
    /// logical 3, which the final norm follows.
    #[test]
    fn the_schedule_and_the_loop_norm_follow_nanbeige_cpp() {
        let loops = LayerLoops::Repeat {
            n_phys: 2,
            n_loops: 2,
            skip_loop_final_norm: false,
        };
        assert_eq!(loops.logical_layers(), 4);
        assert_eq!(
            (0..4).map(|l| loops.physical(l)).collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert_eq!(
            (0..4)
                .map(|l| loops.loop_norm_after(l).is_some())
                .collect::<Vec<_>>(),
            [false, true, false, false]
        );
        assert_eq!(loops.loop_norm_after(1), Some(LoopNorm::Output));
        // A looped model reads ONE stream, so nothing recombines.
        assert!((0..4).all(|l| !loops.stack_starts_at(l)));
        assert!((0..4).all(|l| loops.stream_after(l).is_none()));
        let skipped = LayerLoops::Repeat {
            n_phys: 2,
            n_loops: 2,
            skip_loop_final_norm: true,
        };
        assert!((0..4).all(|l| skipped.loop_norm_after(l).is_none()));
        // Three passes over three layers: norms after 2 and 5, not 8.
        let three = LayerLoops::Repeat {
            n_phys: 3,
            n_loops: 3,
            skip_loop_final_norm: false,
        };
        assert_eq!(
            (0..9)
                .filter(|&l| three.loop_norm_after(l).is_some())
                .collect::<Vec<_>>(),
            [2, 5]
        );
    }

    /// HRM-Text's schedule, read off `hrm-text.cpp:57-68,183-196`:
    /// two stacks of `lps` blocks, replayed over `h * (l + 1)` passes,
    /// a weightless norm after EVERY pass, and the stream each pass
    /// writes.
    #[test]
    fn the_hrm_schedule_aliases_two_stacks_and_names_the_stream_each_pass_writes() {
        // lps = 2, h = 2, l = 2: passes are LOW LOW HIGH LOW LOW HIGH,
        // twelve logical layers over four physical ones.
        let hrm = LayerLoops::Hrm {
            lps: 2,
            h_cycles: 2,
            l_cycles: 2,
        };
        assert_eq!(hrm.logical_layers(), 12);
        assert_eq!(hrm.physical_layers(), 4);
        assert_eq!(
            (0..12).map(|l| hrm.physical(l)).collect::<Vec<_>>(),
            // LOW(0,1) LOW(0,1) HIGH(2,3) LOW(0,1) LOW(0,1) HIGH(2,3)
            [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3]
        );
        // Every stack boundary norms, including the last: the
        // architecture has no `output_norm` tensor.
        assert_eq!(
            (0..12)
                .filter(|&l| hrm.loop_norm_after(l) == Some(LoopNorm::Weightless))
                .collect::<Vec<_>>(),
            [1, 3, 5, 7, 9, 11]
        );
        assert_eq!(
            (0..12)
                .filter(|&l| hrm.stack_starts_at(l))
                .collect::<Vec<_>>(),
            [0, 2, 4, 6, 8, 10]
        );
        assert_eq!(
            (0..12)
                .filter_map(|l| hrm.stream_after(l))
                .collect::<Vec<_>>(),
            [
                HrmStream::Low,
                HrmStream::Low,
                HrmStream::High,
                HrmStream::Low,
                HrmStream::Low,
                HrmStream::High,
            ]
        );
    }

    #[test]
    fn every_reader_is_an_audited_generic_row() {
        for (arch, line) in LOOP_READERS {
            let profile = crate::capability::resolve_profile(arch)
                .unwrap_or_else(|| panic!("`{arch}` ({line}) is not a registered architecture"));
            assert!(matches!(
                profile.path,
                crate::capability::ArchPath::GenericGqa { .. }
            ));
            assert!(crate::capability::AUDITED_GENERIC_GQA.contains(arch));
        }
    }
}
