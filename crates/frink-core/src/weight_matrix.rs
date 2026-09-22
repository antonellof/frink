//! `WeightMatrix`: a weight matrix that may live either as plain f32
//! (small dims, embeddings, synthetic test weights) or as raw
//! Q8_0/Q4_0 block bytes loaded straight from a GGUF file, with no f32
//! expansion at load time. This is what lets frink load a
//! multi-billion-parameter checkpoint without first blowing it up 4x
//! in RAM: the loader (frink-models) hands tensors over still
//! quantized, and every matmul call here dispatches to the fused
//! dequant+dot kernels in frink-quant.

use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

use frink_gguf::GgmlType;

use crate::tensor::Tensor;

pub mod gpu_backend;
pub mod hadamard;
pub mod lora;
mod repack_cache;

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
use gpu_backend::BackendDispatch;
use gpu_backend::{with_gpu_backend_caps, with_gpu_backends, BackendCaps, Cuda, Metal};
pub use lora::{LoraDelta, LoraScale, LoraShapeError, LoraStack};
pub use repack_cache::MapId;
use repack_cache::{
    get_or_repack_q4_0x4, get_or_repack_q4k, get_or_repack_q5k, get_or_repack_q6k,
    get_or_repack_q8x4,
};

/// Backing storage for a quantized weight matrix's raw bytes: either an
/// owned buffer (synthetic/test weights, or any tensor that had to be
/// copied for some other reason) or a zero-copy view into a shared
/// memory-mapped GGUF file. This is the fix for the "loader read
/// everything into a fresh Vec<u8>" inefficiency: a real checkpoint's
/// resident memory should be the mmap itself, not a second copy of it,
/// which is how llama.cpp's mmap-based loader both avoid
/// doubling a multi-hundred-gigabyte checkpoint's memory footprint.
pub enum WeightBytes {
    Owned(Vec<u8>),
    Mapped {
        mmap: Arc<memmap2::Mmap>,
        range: Range<usize>,
    },
    /// A sub-range of a shared, lease-style buffer (e.g. one matrix
    /// inside an `frink_core::expert_store::ExpertLease`'s combined
    /// gate/up/down bytes). Holding the `Arc` here is exactly what
    /// makes the store's lease pinning structural: as long as any
    /// `WeightMatrix` built over these bytes is alive, the cache entry's
    /// strong count stays >1 and eviction cannot reuse it.
    Shared {
        buf: Arc<Vec<u8>>,
        range: Range<usize>,
    },
}

impl WeightBytes {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            WeightBytes::Owned(v) => v,
            WeightBytes::Mapped { mmap, range } => &mmap[range.clone()],
            WeightBytes::Shared { buf, range } => &buf[range.clone()],
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The identity a repack cache may key on, or `None` for bytes that
    /// must never be cached by address.
    ///
    /// This *replaces* an `address_is_stable() -> bool`, and the boolean
    /// was the bug: a yes/no answer cannot say whether the mapping that
    /// made the address meaningful is still alive, so the cache went on
    /// trusting an address after the mapping behind it was gone. See
    /// [`MapId`] for the ABA that produces and how the `Weak` closes it.
    ///
    /// `Shared` stays `None`, and for a different reason that a `Weak`
    /// would NOT fix: it is a lease over an expert store's *recycled*
    /// buffer, so the allocation stays alive and keeps its address while
    /// its CONTENTS are replaced by another expert's. Identity is stable
    /// there and still means nothing. That produced fluent garbage on
    /// OLMoE with expert streaming on, while the raw weight bytes
    /// compared equal, because the corruption was in the CACHE and not
    /// in the weights.
    ///
    /// `Owned` stays `None` too: a freed `Vec`'s address is reused, and
    /// nothing holds a handle that could witness the free.
    pub fn map_id(&self) -> Option<MapId> {
        match self {
            WeightBytes::Mapped { mmap, range } => Some(MapId::of(mmap, range.start)),
            WeightBytes::Owned(_) | WeightBytes::Shared { .. } => None,
        }
    }

    /// True if this is a zero-copy mmap view rather than an owned
    /// heap allocation -- useful for tests/diagnostics asserting that
    /// the loader actually took the zero-copy path.
    pub fn is_mapped(&self) -> bool {
        matches!(self, WeightBytes::Mapped { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantKind {
    Q8_0,
    Q4_0,
    /// The dominant real-world GGUF quantization formats (most
    /// published checkpoints ship as Q4_K_M or similar K-quant mixes,
    /// not the legacy Q4_0/Q8_0 formats above). See
    /// `frink_quant`'s module docs for the block layout and
    /// independent Python cross-validation.
    Q4K,
    Q5K,
    Q6K,
    /// The two more-aggressive K-quant tiers, used in Q2_K/Q3_K_M/
    /// Q3_K_L-style quant mixes (the far more common Q4_K_M/Q5_K_M
    /// mixes only combine with Q6_K, already covered above). See
    /// `frink_quant`'s module docs and independent Python
    /// cross-validation.
    Q2K,
    Q3K,
    /// Legacy, largely-obsolete-for-new-releases formats, still
    /// occasionally encountered. See `frink_quant`'s module docs;
    /// byte layouts verified against real `ggml-common.h` source.
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_1,
    /// Non-linear ("codebook") quants: a 4-bit index maps through a
    /// shared 16-entry signed lookup table instead of a linear
    /// `nibble*scale+min` transform. See `frink_quant`'s module docs
    /// and independent Python cross-validation.
    IQ4NL,
    IQ4XS,
    /// The codebook-grid low-bit formats used throughout published
    /// "Dynamic" low-bit GGUFs of large MoE models (grid-table
    /// magnitudes + shared sign patterns; scalar kernels only so far).
    /// See `frink_quant`'s module docs and the ggml-cross-validated
    /// independent Python reference.
    IQ1S,
    IQ2XXS,
    IQ3XXS,
    /// The second codebook-grid tier (ggml tags 17/21/22/29), which the
    /// published `UD-*` recipes reach for when the `_XXS` tier is too
    /// lossy -- IQ3_S especially, since it is most of what an `IQ3_M`
    /// mix contains. Scalar kernels only; goldens are the real compiled
    /// ggml dequantizers' own output, asserted bit-exactly.
    IQ2XS,
    IQ2S,
    IQ3S,
    IQ1M,
    /// GGUF *block*-MXFP4 (17-byte interleaved blocks, ggml tag 39) --
    /// not the same layout as `WeightMatrix::Mxfp4`'s two-buffer
    /// safetensors form, though the math is identical. Scalar kernel
    /// only so far.
    Mxfp4Gguf,
    /// ggml `TQ1_0` and PrismML `PTQ1_0`: base-3 trits, 256 and 128 to
    /// a block (`frink_quant::ternary`). Bonsai 2 27B is PTQ1_0.
    Tq1_0,
    Ptq1_0,
}

impl QuantKind {
    /// The name a GPU kernel table indexes this storage by, or `None`
    /// for a storage no matvec kernel is written for.
    ///
    /// Exhaustive on purpose, with no `_` arm: a kind added to the enum
    /// has to answer here, and a caller asks the BACKEND's own table
    /// (`frink_metal::gpu::matvec_launch_meta`) whether that name has
    /// a kernel. A second hand-written match of kinds to kernels is how
    /// `Q5_0` and `PTQ1_0` came to be refused by every fused Metal path
    /// in `frink-models` while the Metal table served both.
    pub fn metal_kind_name(self) -> Option<&'static str> {
        match self {
            QuantKind::Q8_0 => Some("Q8_0"),
            QuantKind::Q4_0 => Some("Q4_0"),
            QuantKind::Q5_0 => Some("Q5_0"),
            QuantKind::Q4K => Some("Q4_K"),
            QuantKind::Q5K => Some("Q5_K"),
            QuantKind::Q6K => Some("Q6_K"),
            QuantKind::IQ4XS => Some("IQ4_XS"),
            QuantKind::Ptq1_0 => Some("PTQ1_0"),
            // No Metal matvec kernel: scalar host paths only.
            QuantKind::Q2K
            | QuantKind::Q3K
            | QuantKind::Q4_1
            | QuantKind::Q5_1
            | QuantKind::Q8_1
            | QuantKind::IQ4NL
            | QuantKind::IQ1S
            | QuantKind::IQ2XXS
            | QuantKind::IQ3XXS
            | QuantKind::IQ2XS
            | QuantKind::IQ2S
            | QuantKind::IQ3S
            | QuantKind::IQ1M
            | QuantKind::Mxfp4Gguf
            | QuantKind::Tq1_0 => None,
        }
    }
}

impl QuantKind {
    /// Every variant, so exhaustiveness can be *tested* rather than
    /// trusted. The kernel-coverage tests below iterate this; adding a
    /// variant without adding it here fails to compile (the match in
    /// [`Self::name`] is exhaustive and this list is checked against it).
    pub const ALL: &'static [QuantKind] = &[
        QuantKind::Q8_0,
        QuantKind::Q4_0,
        QuantKind::Q4K,
        QuantKind::Q5K,
        QuantKind::Q6K,
        QuantKind::Q2K,
        QuantKind::Q3K,
        QuantKind::Q4_1,
        QuantKind::Q5_0,
        QuantKind::Q5_1,
        QuantKind::Q8_1,
        QuantKind::IQ4NL,
        QuantKind::IQ4XS,
        QuantKind::IQ1S,
        QuantKind::IQ2XXS,
        QuantKind::IQ3XXS,
        QuantKind::IQ2XS,
        QuantKind::IQ2S,
        QuantKind::IQ3S,
        QuantKind::IQ1M,
        QuantKind::Mxfp4Gguf,
        QuantKind::Tq1_0,
        QuantKind::Ptq1_0,
    ];

    /// The GGUF-facing name. Also the key
    /// [`frink_metal::gpu::matvec_launch_meta`] is looked up by, which
    /// is why it is one function and not a `Debug` impl.
    pub fn name(self) -> &'static str {
        match self {
            QuantKind::Q8_0 => "Q8_0",
            QuantKind::Q4_0 => "Q4_0",
            QuantKind::Q4K => "Q4_K",
            QuantKind::Q5K => "Q5_K",
            QuantKind::Q6K => "Q6_K",
            QuantKind::Q2K => "Q2_K",
            QuantKind::Q3K => "Q3_K",
            QuantKind::Q4_1 => "Q4_1",
            QuantKind::Q5_0 => "Q5_0",
            QuantKind::Q5_1 => "Q5_1",
            QuantKind::Q8_1 => "Q8_1",
            QuantKind::IQ4NL => "IQ4_NL",
            QuantKind::IQ4XS => "IQ4_XS",
            QuantKind::IQ1S => "IQ1_S",
            QuantKind::IQ2XXS => "IQ2_XXS",
            QuantKind::IQ3XXS => "IQ3_XXS",
            QuantKind::IQ2XS => "IQ2_XS",
            QuantKind::IQ2S => "IQ2_S",
            QuantKind::IQ3S => "IQ3_S",
            QuantKind::IQ1M => "IQ1_M",
            QuantKind::Mxfp4Gguf => "MXFP4",
            QuantKind::Tq1_0 => "TQ1_0",
            QuantKind::Ptq1_0 => "PTQ1_0",
        }
    }
}

/// Which quant kinds have a **Metal matvec** kernel, as the kernel name
/// [`frink_metal::gpu::matvec_launch_meta`] resolves.
///
/// The table itself is [`Metal::matvec_kernel`]; this is the name the
/// rest of the tree already imports, kept so the single source of truth
/// moving did not become 30 edits in crates owned by someone else.
pub fn metal_matvec_kind_name(kind: QuantKind) -> Option<&'static str> {
    Metal::matvec_kernel(kind)
}

/// Which quant kinds have a **Metal batched simdgroup GEMM**
/// (`*_mul_mm_sg`), the prefill path. Delegates to
/// [`Metal::gemm_supported`].
pub fn metal_mul_mm_kind_supported(kind: QuantKind) -> bool {
    Metal::gemm_supported(kind)
}

/// Maps a GGUF tensor's on-disk dtype to the [`QuantKind`] a
/// [`WeightMatrix`] uses to pick a fused dequant+dot kernel, or `None`
/// for a dtype with no quantized kernel (F32, or one not implemented at
/// all).
///
/// **The single source of truth for that question**, for the same
/// reason [`metal_mul_mm_kind_supported`] is for its own: this table
/// used to be copied into six GGUF loaders, and the copies drifted.
/// Three of them (`loader`, `glm52_gguf_loader`, `kimi_gguf_loader`)
/// listed 21 dtypes while the other three (`mla_gguf_loader`,
/// `gemma4_gguf_loader`, `hybrid_gguf_loader`) listed 17 -- missing
/// `IQ1_S`, `IQ2_XXS`, `IQ3_XXS` and `MXFP4`. A miss is not a slow
/// path, it is `LoadError::UnsupportedDtype`, so a DeepSeek-MLA
/// checkpoint quantized to `IQ2_XXS` -- an ordinary combination for a
/// model that large -- was refused outright while the identical quant
/// loaded fine on the generic path.
pub fn quant_kind_for(dtype: GgmlType) -> Option<QuantKind> {
    match dtype {
        GgmlType::Q8_0 => Some(QuantKind::Q8_0),
        GgmlType::Q4_0 => Some(QuantKind::Q4_0),
        GgmlType::Q4K => Some(QuantKind::Q4K),
        GgmlType::Q5K => Some(QuantKind::Q5K),
        GgmlType::Q6K => Some(QuantKind::Q6K),
        GgmlType::Q2K => Some(QuantKind::Q2K),
        GgmlType::Q3K => Some(QuantKind::Q3K),
        GgmlType::Q4_1 => Some(QuantKind::Q4_1),
        GgmlType::Q5_0 => Some(QuantKind::Q5_0),
        GgmlType::Q5_1 => Some(QuantKind::Q5_1),
        GgmlType::Q8_1 => Some(QuantKind::Q8_1),
        GgmlType::IQ4NL => Some(QuantKind::IQ4NL),
        GgmlType::IQ4XS => Some(QuantKind::IQ4XS),
        GgmlType::IQ2XS => Some(QuantKind::IQ2XS),
        GgmlType::IQ2S => Some(QuantKind::IQ2S),
        GgmlType::IQ3S => Some(QuantKind::IQ3S),
        GgmlType::IQ1M => Some(QuantKind::IQ1M),
        GgmlType::IQ1S => Some(QuantKind::IQ1S),
        GgmlType::IQ2XXS => Some(QuantKind::IQ2XXS),
        GgmlType::IQ3XXS => Some(QuantKind::IQ3XXS),
        GgmlType::MXFP4 => Some(QuantKind::Mxfp4Gguf),
        GgmlType::TQ1_0 => Some(QuantKind::Tq1_0),
        GgmlType::PTQ1_0 => Some(QuantKind::Ptq1_0),
        _ => None,
    }
}

/// Which quant kinds have a **CUDA batched GEMM** (`mul_mm`), the
/// prefill path. Delegates to [`Cuda::gemm_supported`], which is where
/// the "UNRUN ON HARDWARE" caveat is written down.
pub fn cuda_mul_mm_kind_supported(kind: QuantKind) -> bool {
    Cuda::gemm_supported(kind)
}

/// Which quant kinds have a **CUDA matvec** kernel, the decode path.
/// Delegates to [`Cuda::matvec_kernel`], whose `Option<&str>` is the
/// shape Metal needs and CUDA does not — the `bool` is this wrapper.
pub fn cuda_matvec_kind_supported(kind: QuantKind) -> bool {
    Cuda::matvec_kernel(kind).is_some()
}

/// Which quant kinds take the CPU integer `vec_dot` path (activation
/// quantized to Q8/Q8_K, int8xint8 dots) rather than the much slower f32
/// dequant-dot. `cols` matters: the K-quant kernels need a whole number
/// of 256-element super-blocks, the legacy ones 32-element blocks.
pub fn cpu_int_dot_kind_supported(kind: QuantKind, cols: usize) -> bool {
    match kind {
        QuantKind::Q8_0 | QuantKind::Q4_0 => cols.is_multiple_of(32),
        QuantKind::Q4K | QuantKind::Q5K | QuantKind::Q6K => cols.is_multiple_of(256),
        _ => false,
    }
}

/// The backend dense matmuls will actually use in this process, decided
/// by the same cached env/probe reads dispatch uses. CUDA wins when both
/// are compiled in — and it wins here because it is first in
/// [`gpu_backend::with_gpu_backends`], the single ordered list
/// [`WeightMatrix::apply_gpu`] also expands, rather than because that
/// order is written out a second time.
pub fn active_backend() -> crate::kernel_registry::Backend {
    #[allow(unused_macros)]
    macro_rules! first_enabled {
        ($b:ty) => {
            if <$b as BackendDispatch>::dense_enabled() {
                return <$b as BackendCaps>::ID;
            }
        };
    }
    with_gpu_backends!(first_enabled);
    crate::kernel_registry::Backend::Cpu
}

/// Minimum multiply-accumulates a rayon task should carry before it is
/// worth its own scheduling. Chosen by measurement, not derivation.
///
/// **This is a rayon-only mitigation and it is unreachable on
/// [`crate::par::Backend::Spin`].** It exists to stop rayon splitting a
/// matvec into tasks too small to repay a fork-join; the persistent pool
/// has no fork-join to repay, so it chunks by pool width alone (see the
/// `MIN_TASK_MACS` section of [`crate::par`]). Issue #27 asks for this
/// constant to be deleted rather than retuned, and on the pool's path it
/// is: [`WeightMatrix::min_rows_per_task`] returns before reading it
/// whenever [`crate::par::backend`] picked the pool for this operation.
/// It survives on the fork-join path, which is still every operation
/// below [`crate::par::policy::SPIN_MIN_OP_MACS`], because removing it
/// there re-opens the 13-16x small-model regression recorded on
/// [`WeightMatrix::min_rows_per_task`].
const MIN_TASK_MACS: usize = 1 << 16;

/// Whether dense [`WeightMatrix::apply`] / [`WeightMatrix::apply_batch`]
/// should try Metal first (when built with `--features metal`).
///
/// - `FRINK_METAL=0|false|off|cpu` — force CPU
/// - `FRINK_METAL=1|true|on|metal` — force Metal attempt
/// - unset / `auto` — Metal when [`frink_metal::gpu::probe`] finds a device
///
/// Decision is cached for the process lifetime (env read once).
#[cfg(feature = "metal")]
pub fn metal_dense_enabled() -> bool {
    Metal::dense_enabled()
}

/// Whether dense [`WeightMatrix::apply`] should try CUDA first (when
/// built with `--features cuda`).
///
/// - `FRINK_CUDA=0|false|off|cpu` — force skip CUDA dense
/// - `FRINK_CUDA=1|true|on|cuda` — force CUDA attempt
/// - unset / `auto` — CUDA when a device probe succeeds
#[cfg(feature = "cuda")]
pub fn cuda_dense_enabled() -> bool {
    Cuda::dense_enabled()
}

/// Whether CPU Q8_0 / Q4_0 / Q4_K / Q5_K / Q6_K matvec should quantize the
/// activation to int8 and use the integer `vec_dot` path. Q4_K
/// additionally lazy-repacks into interleaved `block_q4_Kx8` for 8-wide
/// GEMV; Q8_0 into `block_q8_0x4` and Q4_0 into `block_q4_0x4` for
/// 4-wide GEMV.
///
/// Off by default *as a library*, and turned on by both binaries (see
/// `frink_core::threads`'s siblings in `frink-cli`/`frink-server`,
/// which set `FRINK_CPU_INT_DOT=1` unless the caller already chose).
/// The split is deliberate: this is what llama.cpp's CPU backend does
/// unconditionally -- quantize the activation to Q8, run integer
/// `vec_dot` -- and it is worth 28% of CPU decode on Host B
/// (Qwen2.5-0.5B Q8_0, `-ngl 0 -t 6`: 58.0 -> 80.5 tok/s). But it also
/// perturbs results below the f32 reference's precision, and this
/// crate's golden cross-validation against the independent NumPy
/// reference asserts exact agreement. So the *inference product*
/// defaults to fast and the *library default* stays reference-exact.
///
/// **This is the master switch, not the dispatch rule.** It says whether
/// the tier is on at all; whether a given piece of work should take it
/// is [`cpu_int_dot_for`], which also asks whether this host has the
/// kernel for that workload's shape. Production dispatch calls that one.
pub fn cpu_int_dot_enabled() -> bool {
    #[cfg(test)]
    {
        // The env var is read once into a `OnceLock`, so a test cannot
        // flip it after any other test has already observed it. Without
        // an override, `cargo test` runs with int-dot *off* and every
        // interleaved/i8mm batch kernel below is dead code in CI --
        // which is how the whole repack tier went untested end to end.
        // See [`tests::ForceIntDot`].
        match INT_DOT_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Acquire) {
            0 => return false,
            1 => return true,
            _ => {}
        }
    }
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FRINK_CPU_INT_DOT").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Test-only forcing of [`cpu_int_dot_enabled`]: `-1` unset, `0` off,
/// `1` on. A global atomic rather than a thread-local because the paths
/// it gates run on Rayon workers, which do not inherit thread-locals
/// from the test thread.
#[cfg(test)]
static INT_DOT_TEST_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Sets `FRINK_CPU_INT_DOT=1` unless the caller already expressed a
/// preference. Call from a binary's startup, before any worker threads
/// exist. See [`cpu_int_dot_enabled`] for why the default lives here
/// rather than in the getter.
///
/// # Safety
/// Must be called while the process is still single-threaded, since it
/// mutates the process environment.
pub unsafe fn default_cpu_int_dot_on() {
    if std::env::var_os("FRINK_CPU_INT_DOT").is_none() && int_dot_is_a_win_here() {
        unsafe { std::env::set_var("FRINK_CPU_INT_DOT", "1") };
    }
}

/// Whether the int-dot path is faster than the f32 one on THIS
/// architecture.
///
/// It is not universally faster, and the default said it was. The
/// interleaved int8 kernels this path selects were written for
/// aarch64: `i8mm` SMMLA tiers, interleave-8 NEON GEMV, the Q8_K repack.
/// x86_64 has none of that, so on x86 the switch selects a scalar
/// integer loop AND bypasses the AVX2 f32 dot that does exist
/// (`dot_q4_k_f32` and friends are gated on `avx2` + `fma`).
///
/// Measured 2026-09-04 on an idle 32-core Ryzen 9 7945HX (Zen 4, with
/// `avx512_vnni` that nothing here uses), `tg64`, int-dot on against
/// off:
///
/// | model | on (was the default) | off |
/// |---|---|---|
/// | Llama-3.2-1B Q4_K_M | 10.08 | **48.94** |
/// | Llama-3.2-1B Q6_K | 4.11 | **36.23** |
/// | Llama-3.2-3B Q4_K_M | 4.73 | **19.30** |
///
/// So the default cost x86 between 4x and 8.8x of decode, and it is
/// most of why `benchmarks/RESULTS.md` had no x86 row worth showing
/// (#127). Prefill is unaffected (95.3 against 89.4 on the 1B), which
/// is consistent: prefill goes through the batched GEMM rather than
/// this dot.
///
/// **That measurement is per WORKLOAD, and the flag was per process.**
/// It says the matvec half of the tier loses on x86 and says nothing
/// against the batch half; the batch half simply had no x86 kernel to
/// try, which is #152. Now that it does, the rule is
/// [`int_dot_tier_here`] and this function is only its "is any half
/// worth turning on by default" summary.
///
/// This is a DEFAULT, not a gate: `FRINK_CPU_INT_DOT=1` still turns it
/// on anywhere.
fn int_dot_is_a_win_here() -> bool {
    let tier = int_dot_tier_here();
    tier.matvec || tier.batch_gemm
}

/// Which shape of work a call site is asking the repacked integer tier
/// for. Not a hint: the two are different kernels and, on x86, different
/// answers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntDotShape {
    /// One activation against the whole matrix — `apply`, `apply_cpu_q8`,
    /// the MoE per-expert dots. Decode, and the `nrc == 1` GEMV kernels.
    Matvec,
    /// A batch of activations at once, through the interleaved `×4`
    /// GEMMs. Prefill.
    BatchGemm,
}

/// **The** predicate for "does this work take the repacked integer
/// tier". Every call site asks this and none restates it.
///
/// Two things have to be true: `FRINK_CPU_INT_DOT` is on (the master
/// switch, [`cpu_int_dot_enabled`]), and this host has kernels worth
/// taking for `shape` ([`int_dot_tier_here`]).
///
/// Splitting by shape is the whole point. The tier used to be one
/// process-wide flag over two unrelated kernel families, so x86 had to
/// choose between a batched GEMM it wanted and a matvec that cost it 4x
/// to 8.8x of decode — and chose neither.
pub fn cpu_int_dot_for(shape: IntDotShape) -> bool {
    cpu_int_dot_enabled() && int_dot_tier_here().covers(shape)
}

/// Which halves of the repacked integer tier are worth taking on this
/// host.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct IntDotTier {
    matvec: bool,
    batch_gemm: bool,
}

impl IntDotTier {
    /// Exhaustive on purpose, with no `_` arm: a third workload shape
    /// must state its own answer rather than inherit one.
    fn covers(self, shape: IntDotShape) -> bool {
        match shape {
            IntDotShape::Matvec => self.matvec,
            IntDotShape::BatchGemm => self.batch_gemm,
        }
    }
}

/// The per-host, per-workload rule, in one place.
///
/// - **aarch64**: the matvec half always — the interleave-8 NEON GEMV
///   and the i8mm SMMLA GEMMs are the kernels this tier was written for,
///   worth ~28% of decode and 15x of prefill (`FRINK_CPU_INT_DOT=0`
///   takes Llama-3.2-1B Q4_K_M pp512 from 420.34 to 27.81 tok/s, #152).
///   The batch half asks the kernels, which on a host with `i8mm` is the
///   quad GEMM and on one with only `dotprod` is the width-4 `sdot` GEMM.
/// - **x86_64**: the batch half only, and only when the AVX2 `×4` GEMMs
///   are actually present. The matvec half stays off because it was
///   MEASURED to lose — see the table above — and nothing in this change
///   touches the kernel it loses to.
/// - anywhere else: neither, because neither has a kernel.
///
/// `batch_gemm` is not a written-down claim about any architecture; it
/// asks `frink_quant` whether a SIMD batch GEMM exists at the width
/// this host packs with. A host cannot be told the tier is a win while
/// its kernel is missing, and an x86 host without AVX2 gets the same
/// answer a RISC-V one does.
///
/// It asks [`frink_quant::batch_gemm_is_accelerated`] and NOT
/// `interleaved_gemm_is_accelerated`: the latter is about the
/// interleave-8 quad kernels, which is the right question for whether to
/// PREPARE a quad and the wrong one for whether the tier buys anything.
/// A pre-i8mm aarch64 host — every M1 Mac, every A14-and-earlier iPhone,
/// and the Cortex-A55-class cores that are still most of the Android
/// fleet — runs a width-4 `dotprod` GEMM for Q4_K, Q5_K, Q8_0 and Q4_0,
/// so the narrower predicate reports "no SIMD GEMM" on a host that is
/// running one. Q6_K has no width-4 GEMM on purpose (its scalar Kx8 GEMM
/// measured slower than the per-row NEON dot), and the per-kind
/// `q*_gemm_uses_acts_x4` entry points are what keep that distinction.
///
/// # On the `cfg!` in here
///
/// `par::policy` warns against exactly this shape — "an
/// architecture-conditional default is what `FRINK_CPU_INT_DOT` was" —
/// and it is right that an *unmeasured* one is how this went wrong.
/// This one is the measurement: the x86 matvec row above is a real
/// before/after on a quiet host, and the x86 batch row is gated on a
/// runtime probe rather than a guess. The two predicates also answer
/// different questions and must not be merged: `policy::backend` picks
/// the SCHEDULER by work size; this picks the KERNEL by workload shape.
/// Whether the Q5_K batched matmul takes the Kx8 path: every aarch64
/// host (the i8mm quad GEMM, the width-4 `sdot` GEMM, or the scalar Kx8
/// body, as the Q4_K arm takes it unconditionally), and any other host
/// whose `x4` GEMM has a SIMD kernel at this width (AVX2 since #159).
///
/// Asked of the kernels rather than written beside them, the shape
/// `int_dot_tier_here` has: this read `cfg!(target_arch = "aarch64")`
/// alone until 2026-09-15, which sent every x86 Q5_K prefill through
/// the per-row GEMM -- 44.2 against llama.cpp's 378.7 tok/s on
/// Llama-3.2-1B Q5_K_M (8.56x) on a Ryzen 5950X, beside Q4_K and Q6_K
/// at 1.19x on the same host, because those two arms asked the kernels
/// and this one asked the architecture. With the predicate: 417.1
/// tok/s, 0.91x. Phi-4-mini's `attn_qkv` is Q5_K too.
#[inline]
fn q5k_batch_takes_kx8(interleave: usize) -> bool {
    cfg!(target_arch = "aarch64") || frink_quant::q5_kx8_gemm_uses_acts_x4(interleave)
}

fn int_dot_tier_here() -> IntDotTier {
    #[cfg(target_arch = "aarch64")]
    {
        IntDotTier {
            matvec: true,
            batch_gemm: frink_quant::batch_gemm_is_accelerated(frink_quant::preferred_interleave()),
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        IntDotTier {
            matvec: false,
            batch_gemm: frink_quant::batch_gemm_is_accelerated(frink_quant::preferred_interleave()),
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        IntDotTier {
            matvec: false,
            batch_gemm: false,
        }
    }
}

/// A batch of activations quantized once for reuse across several
/// [`WeightMatrix::apply_batch_with_acts`] calls that read the same input
/// (q/k/v on one normed batch; gate/up on another). Build with
/// [`WeightMatrix::quantize_batch_acts`]. Q8_0/Q4_0 matrices consume
/// [`BatchActs::Q8`]; the K-quants consume [`BatchActs::Q8K`].
///
/// `tiles` carries the *interleaved* activation quads the i8mm GEMMs read
/// (llama.cpp's `wdata` after `ggml_quantize_mat_q8_K_4x8`), not just the
/// per-position quantization. Sharing stops at the same place the
/// quantization does: q/k/v build one set between them instead of three,
/// gate/up one instead of two. It is empty on hosts with no i8mm kernel,
/// where preparing a quad buys nothing.
///
/// `cols` is recorded so a set built for one width can never be handed to
/// a matrix of another. The tiles are chunked four positions wide for
/// every kind (`Q8K_ACTS_X4_NC`, and `Q4_KX8_GEMM_NC` / `Q5_KX8_GEMM_NC`
/// are the same 4), which is why one set serves Q4_K, Q5_K and Q6_K --
/// and, in the [`BatchActs::Q8`] variant, both Q8_0 and Q4_0.
pub enum BatchActs {
    Q8 {
        acts: Vec<frink_quant::Q8Activations>,
        tiles: Vec<frink_quant::Q8ActsX4>,
        cols: usize,
    },
    Q8K {
        acts: Vec<frink_quant::Q8KActivations>,
        tiles: Vec<frink_quant::Q8KActsX4>,
        cols: usize,
    },
}

// Sharing one quad set across kinds is only sound while every `x4`
// consumer chunks the batch the same way. If one of these widths is ever
// retuned on its own, the quads a Q4_K gate builds stop lining up with
// what a Q5_K sibling indexes, and the failure is a wrong answer rather
// than a panic -- so it fails the build instead.
const _: () = {
    assert!(frink_quant::Q4_KX8_GEMM_NC == frink_quant::Q8K_ACTS_X4_NC);
    assert!(frink_quant::Q5_KX8_GEMM_NC == frink_quant::Q8K_ACTS_X4_NC);
};

pub enum WeightMatrix {
    F32(Tensor),
    Quantized {
        data: WeightBytes,
        rows: usize,
        cols: usize,
        kind: QuantKind,
    },
    /// MXFP4 (OCP Microscaling 4-bit float, Kimi K3's real routed-expert
    /// format): unlike every `Quantized` kind above, which store one
    /// interleaved block buffer per row, Kimi K3's real checkpoint
    /// stores the packed 4-bit codes and per-group E8M0 scales as two
    /// *separate* tensors (confirmed against a real shard header, see
    /// `frink_quant`'s MXFP4 module docs) -- so this variant holds two
    /// independently zero-copy-mappable buffers instead of `Quantized`'s
    /// single `data` buffer. `apply`/`apply_batch` dispatch to
    /// `frink_quant::dot_mxfp4_row_f32`, which reads directly from
    /// these buffers without ever materializing a dequantized f32 copy
    /// of the whole matrix -- the same zero-copy-mmap-plus-fused-dot
    /// discipline as every `Quantized` kind, letting a real MXFP4
    /// checkpoint's resident memory stay close to its on-disk size
    /// instead of the ~8x larger eager-f32-dequant footprint.
    Mxfp4 {
        packed: WeightBytes,
        scale: WeightBytes,
        rows: usize,
        cols: usize,
    },
    /// `base` with one or more LoRA adapters attached: every product
    /// this matrix computes is `W x + Σ_i s_i · B_i (A_i x)`, the
    /// low-rank term added inside the same method that computed `W x`
    /// (see [`lora`]). `base` is never itself `Adapted`:
    /// [`Self::attach_lora`] pushes onto the existing stack instead.
    ///
    /// A fourth variant rather than a field on the other three, so that
    /// every place that reaches PAST the methods for raw bytes -- a
    /// fused Metal stack, a simdgroup-GEMM descriptor, a Q8 row dot --
    /// has to say what it does with an adapter, and the answer written
    /// into each of them is `None`: those callers fall back to the
    /// methods, which serve the delta, rather than run the base weights
    /// and drop it.
    Adapted {
        base: Box<WeightMatrix>,
        lora: LoraStack,
    },
    /// A weight PrismML folded a Hadamard rotation into
    /// (`hadamard::HadamardFold`): the product is right only after the
    /// activation has been transformed the same way, so every `apply`
    /// here transforms and then runs `base`, and every fused launch
    /// view answers `None` for it as it does for `Adapted`. The row
    /// lookup (`FoldSite::RowLookup`) restores a stored row instead.
    Folded {
        base: Box<WeightMatrix>,
        fold: std::sync::Arc<hadamard::HadamardFold>,
    },
}

impl WeightMatrix {
    /// Attaches one adapter's `(A, B)` pair. A second adapter on the
    /// same weight joins the first's stack; the base is boxed exactly
    /// once.
    pub fn attach_lora(&mut self, delta: LoraDelta) {
        assert_eq!(delta.rows(), self.rows(), "LoRA delta rows");
        assert_eq!(delta.cols(), self.cols(), "LoRA delta cols");
        if let WeightMatrix::Adapted { lora, .. } = self {
            lora.push(delta);
            return;
        }
        let placeholder = WeightMatrix::F32(Tensor::new(Vec::new(), vec![0, 0]));
        let base = std::mem::replace(self, placeholder);
        *self = WeightMatrix::Adapted {
            base: Box::new(base),
            lora: LoraStack::new(delta),
        };
    }

    /// The adapters on this matrix, if any.
    pub fn lora(&self) -> Option<&LoraStack> {
        match self {
            WeightMatrix::Adapted { lora, .. } => Some(lora),
            _ => None,
        }
    }

    /// Wraps `self` in a Hadamard fold (`hadamard::HadamardFold`).
    /// The fold's width, when it has signs, must be this matrix's
    /// column count.
    pub fn fold_hadamard(&mut self, fold: std::sync::Arc<hadamard::HadamardFold>) {
        if let Some(w) = fold.width() {
            assert_eq!(
                w,
                self.cols(),
                "Hadamard sign vector width vs matrix columns"
            );
        }
        let placeholder = WeightMatrix::F32(Tensor::new(Vec::new(), vec![0, 0]));
        let base = std::mem::replace(self, placeholder);
        *self = WeightMatrix::Folded {
            base: Box::new(base),
            fold,
        };
    }

    /// The fold on this matrix, if any.
    pub fn hadamard_fold(&self) -> Option<&std::sync::Arc<hadamard::HadamardFold>> {
        match self {
            WeightMatrix::Folded { fold, .. } => Some(fold),
            _ => None,
        }
    }

    /// The matrix a GPU launch should read, and the rotation its INPUT
    /// needs before it does.
    ///
    /// One accessor rather than two, because a caller that took the
    /// base and forgot the fold would run transformed weights against
    /// an untransformed activation: a wrong answer that nothing fails
    /// on. `(self, None)` for every matrix that is not folded.
    pub fn launch_parts(
        &self,
    ) -> (
        &WeightMatrix,
        Option<&std::sync::Arc<hadamard::HadamardFold>>,
    ) {
        match self {
            WeightMatrix::Folded { base, fold } => (base, Some(fold)),
            _ => (self, None),
        }
    }

    /// The weights under any adapter: `self` when there is none. NOT
    /// under a fold: a folded matrix's base answers a different
    /// product, so `base()` of a `Folded` is the `Folded` itself and a
    /// caller that wants the bytes goes through `hadamard_fold()`.
    pub fn base(&self) -> &WeightMatrix {
        match self {
            WeightMatrix::Adapted { base, .. } => base,
            _ => self,
        }
    }

    /// Raw quantized byte length, or 0 for a float matrix. For
    /// comparing two backings of the same weight.
    pub fn bytes_len(&self) -> usize {
        match self {
            WeightMatrix::Quantized { data, .. } => data.len(),
            WeightMatrix::Adapted { base, .. } | WeightMatrix::Folded { base, .. } => {
                base.bytes_len()
            }
            _ => 0,
        }
    }

    /// Do two matrices hold the same quantized bytes?
    ///
    /// Exists to answer one question: when a streamed expert and a
    /// resident one disagree about a model's output, is the difference
    /// in the WEIGHTS or downstream of them?
    pub fn bytes_eq(&self, other: &WeightMatrix) -> bool {
        match (self.base(), other.base()) {
            (WeightMatrix::Quantized { data: a, .. }, WeightMatrix::Quantized { data: b, .. }) => {
                a.as_slice() == b.as_slice()
            }
            _ => false,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.rows(),
            WeightMatrix::Quantized { rows, .. } => *rows,
            WeightMatrix::Mxfp4 { rows, .. } => *rows,
            WeightMatrix::Adapted { base, .. } | WeightMatrix::Folded { base, .. } => base.rows(),
        }
    }

    /// The block format, or `None` for the two non-block storages
    /// (`F32`, safetensors-pair `Mxfp4`). This is the key every
    /// kernel-availability table is indexed by.
    pub fn quant_kind(&self) -> Option<QuantKind> {
        match self {
            WeightMatrix::Quantized { kind, .. } => Some(*kind),
            WeightMatrix::F32(_) | WeightMatrix::Mxfp4 { .. } => None,
            WeightMatrix::Adapted { base, .. } | WeightMatrix::Folded { base, .. } => {
                base.quant_kind()
            }
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.cols(),
            WeightMatrix::Quantized { cols, .. } => *cols,
            WeightMatrix::Mxfp4 { cols, .. } => *cols,
            WeightMatrix::Adapted { base, .. } | WeightMatrix::Folded { base, .. } => base.cols(),
        }
    }

    fn block_bytes_per_row(&self, kind: QuantKind, cols: usize) -> usize {
        match kind {
            QuantKind::Q8_0 => {
                (cols / frink_quant::Q8_0_BLOCK_ELEMS) * frink_quant::Q8_0_BLOCK_BYTES
            }
            QuantKind::Q4_0 => {
                (cols / frink_quant::Q4_0_BLOCK_ELEMS) * frink_quant::Q4_0_BLOCK_BYTES
            }
            QuantKind::Q4K => {
                (cols / frink_quant::Q4_K_BLOCK_ELEMS) * frink_quant::Q4_K_BLOCK_BYTES
            }
            QuantKind::Q5K => {
                (cols / frink_quant::Q5_K_BLOCK_ELEMS) * frink_quant::Q5_K_BLOCK_BYTES
            }
            QuantKind::Q6K => {
                (cols / frink_quant::Q6_K_BLOCK_ELEMS) * frink_quant::Q6_K_BLOCK_BYTES
            }
            QuantKind::Q2K => {
                (cols / frink_quant::Q2_K_BLOCK_ELEMS) * frink_quant::Q2_K_BLOCK_BYTES
            }
            QuantKind::Q3K => {
                (cols / frink_quant::Q3_K_BLOCK_ELEMS) * frink_quant::Q3_K_BLOCK_BYTES
            }
            QuantKind::Q4_1 => {
                (cols / frink_quant::Q4_1_BLOCK_ELEMS) * frink_quant::Q4_1_BLOCK_BYTES
            }
            QuantKind::Q5_0 => {
                (cols / frink_quant::Q5_0_BLOCK_ELEMS) * frink_quant::Q5_0_BLOCK_BYTES
            }
            QuantKind::Q5_1 => {
                (cols / frink_quant::Q5_1_BLOCK_ELEMS) * frink_quant::Q5_1_BLOCK_BYTES
            }
            QuantKind::Q8_1 => {
                (cols / frink_quant::Q8_1_BLOCK_ELEMS) * frink_quant::Q8_1_BLOCK_BYTES
            }
            QuantKind::IQ4NL => {
                (cols / frink_quant::IQ4_NL_BLOCK_ELEMS) * frink_quant::IQ4_NL_BLOCK_BYTES
            }
            QuantKind::IQ4XS => {
                (cols / frink_quant::IQ4_XS_BLOCK_ELEMS) * frink_quant::IQ4_XS_BLOCK_BYTES
            }
            QuantKind::IQ1S => {
                (cols / frink_quant::IQ1_S_BLOCK_ELEMS) * frink_quant::IQ1_S_BLOCK_BYTES
            }
            QuantKind::IQ2XXS => {
                (cols / frink_quant::IQ2_XXS_BLOCK_ELEMS) * frink_quant::IQ2_XXS_BLOCK_BYTES
            }
            QuantKind::IQ3XXS => {
                (cols / frink_quant::IQ3_XXS_BLOCK_ELEMS) * frink_quant::IQ3_XXS_BLOCK_BYTES
            }
            QuantKind::IQ2XS => {
                (cols / frink_quant::IQ2_XS_BLOCK_ELEMS) * frink_quant::IQ2_XS_BLOCK_BYTES
            }
            QuantKind::IQ2S => {
                (cols / frink_quant::IQ2_S_BLOCK_ELEMS) * frink_quant::IQ2_S_BLOCK_BYTES
            }
            QuantKind::IQ3S => {
                (cols / frink_quant::IQ3_S_BLOCK_ELEMS) * frink_quant::IQ3_S_BLOCK_BYTES
            }
            QuantKind::IQ1M => {
                (cols / frink_quant::IQ1_M_BLOCK_ELEMS) * frink_quant::IQ1_M_BLOCK_BYTES
            }
            QuantKind::Mxfp4Gguf => {
                (cols / frink_quant::MXFP4_GGUF_BLOCK_ELEMS) * frink_quant::MXFP4_GGUF_BLOCK_BYTES
            }
            QuantKind::Tq1_0 => {
                (cols / frink_quant::ternary::TQ1_0_BLOCK_ELEMS)
                    * frink_quant::ternary::TQ1_0_BLOCK_BYTES
            }
            QuantKind::Ptq1_0 => {
                (cols / frink_quant::ternary::PTQ1_0_BLOCK_ELEMS)
                    * frink_quant::ternary::PTQ1_0_BLOCK_BYTES
            }
        }
    }

    /// A reasonable minimum number of rows for one rayon task to
    /// process, to avoid rayon's work-stealing splitter fragmenting a
    /// matmul into tasks so small that scheduling/synchronization
    /// overhead dominates the real per-row work (a fused dequant+dot,
    /// not free). This is a real, measured fix, not speculative
    /// tuning: naive per-row splitting (rayon's default) caused a
    /// 13-16x throughput regression on a host configured with far more
    /// rayon threads than a small model's matrices have useful
    /// parallelism for (observed directly on a shared-core rented
    /// host, where auto-detected high thread counts collapsed
    /// throughput ~13-16x on a small model). Aims for ~4 tasks per thread
    /// -- enough that rayon's work-stealing can still load-balance
    /// across threads that finish early, without going all the way
    /// down to one task per row.
    ///
    /// Floor of 8 avoids Rayon thrash on tiny mats (SmolLM2 attn_kv
    /// has 192 rows → without a floor, ~48 one-row tasks on 10 cores).
    ///
    /// The floor is also **work-aware**, which matters for decode. A row
    /// count alone says nothing about how much arithmetic a task carries:
    /// SmolLM2's 576-wide projections split into ~24 tasks of ~14K MACs
    /// each, far too little to pay for a fork-join. Measured on this host
    /// (both engines back to back, thread count as the only variable):
    /// frink scales 1.40x / 2.93x from 1 to 6 threads on TinyLlama /
    /// Mistral-7B where llama.cpp scales 1.99x / 4.39x, and the deficit
    /// grows as the model shrinks -- the signature of tasks too small to
    /// amortise their own scheduling, not of slow kernels (frink is
    /// *ahead* of llama at one thread on Mistral-7B).
    ///
    /// [`crate::par::with_op_work`] supplies the elements-per-row so a
    /// task can be required to carry at least [`MIN_TASK_MACS`]
    /// multiply-accumulates. Zero (unset) keeps the old row-only
    /// behaviour, so any call site that has not opted in is unchanged.
    ///
    /// Nothing here needs to ask which scheduler won this operation.
    /// What this returns is a `min_len`, and `min_len` is read only by
    /// the fork-join arm of [`crate::par`] -- the persistent pool's arm
    /// chunks by width alone, which
    /// `par::tests::the_spin_arm_chunks_by_pool_width_with_no_work_threshold`
    /// asserts. A second `Backend::Spin` check here was written and
    /// removed: deleting it changed no result, which is the definition
    /// of a gate that cannot fire.
    fn min_rows_per_task(rows: usize) -> usize {
        let threads = crate::par::num_threads();
        let by_threads = (rows / (threads * 4)).max(8.min(rows.max(1)));
        let per_row = crate::par::macs_per_row();
        if per_row == 0 {
            return by_threads;
        }
        let need = MIN_TASK_MACS.div_ceil(per_row.max(1));
        by_threads.max(need.min(rows.max(1)))
    }

    /// Run `body(g, t0, t1)` for every row-group `g` and activation-tile
    /// range `[t0, t1)` of a llama-style 2D chunk grid over
    /// (row-groups × batch tiles).
    ///
    /// This is the port of `ggml_compute_forward_mul_mat`'s chunking
    /// (`ggml-cpu.c`): ~16 rows / 16 batch positions per chunk, and if
    /// that grid is smaller than `4 × threads`, re-chunk by thread along
    /// the larger dimension. llama walks the grid with an atomic
    /// `current_chunk` because its threadpool has no scheduler; Rayon
    /// already work-steals, so handing it the same chunks (`min_len 1`)
    /// gets the same load balancing. The point is the *batch* dimension:
    /// splitting only by rows leaves a 192-row projection with ~3 tasks
    /// no matter how many positions are in flight.
    fn par_chunked_groups(
        n_groups: usize,
        group_rows: usize,
        n_tiles: usize,
        tile_batch: usize,
        body: impl Fn(usize, usize, usize) + Sync,
    ) {
        if n_groups == 0 || n_tiles == 0 {
            return;
        }
        let nth = crate::par::num_threads();
        const CHUNK_ELEMS: usize = 16;
        let g_per_chunk = (CHUNK_ELEMS / group_rows).max(1);
        let t_per_chunk = (CHUNK_ELEMS / tile_batch).max(1);
        let mut nchunk_g = n_groups.div_ceil(g_per_chunk);
        let mut nchunk_t = n_tiles.div_ceil(t_per_chunk);
        if nchunk_g * nchunk_t < nth * 4 {
            // llama's fallback: one chunk per thread along the larger dim.
            if n_groups * group_rows > n_tiles * tile_batch {
                nchunk_g = nth.min(n_groups);
                nchunk_t = 1;
            } else {
                nchunk_g = 1;
                nchunk_t = nth.min(n_tiles);
            }
        }
        let dg = n_groups.div_ceil(nchunk_g);
        let dt = n_tiles.div_ceil(nchunk_t);
        crate::par::indices(nchunk_g * nchunk_t, 1, |chunk| {
            let g0 = (chunk % nchunk_g) * dg;
            let g1 = (g0 + dg).min(n_groups);
            let t0 = (chunk / nchunk_g) * dt;
            let t1 = (t0 + dt).min(n_tiles);
            for g in g0..g1 {
                body(g, t0, t1);
            }
        });
    }

    /// Resolve the Q8_0-format activations (and the interleaved quads, if
    /// any) an [`Self::apply_batch_with_acts`] arm should read.
    ///
    /// Returns the shared batch when it matches this matrix -- same
    /// positions, same width -- and otherwise quantizes into `owned` and
    /// returns that with no quads, so the caller builds its own. A
    /// mismatched `shared` is silently ignored rather than trusted, which
    /// is what keeps a mixed-width projection group correct.
    ///
    /// The returned quads are only ever the *shared* ones. The empty slice
    /// therefore means "nobody prepared these for you", not "this host has
    /// no i8mm kernel" -- the caller still decides that with
    /// `q8_0x4_gemm_uses_acts_x4`.
    fn q8_acts<'a>(
        shared: Option<&'a BatchActs>,
        x_batch: &[f32],
        batch_size: usize,
        cols: usize,
        owned: &'a mut Vec<frink_quant::Q8Activations>,
    ) -> (
        &'a [frink_quant::Q8Activations],
        &'a [frink_quant::Q8ActsX4],
    ) {
        if let Some(BatchActs::Q8 {
            acts,
            tiles,
            cols: c,
        }) = shared
        {
            if acts.len() == batch_size && *c == cols {
                return (acts, tiles);
            }
        }
        *owned = (0..batch_size)
            .into_par_iter()
            .map(|b| frink_quant::quantize_activations_q8(&x_batch[b * cols..(b + 1) * cols]))
            .collect();
        (owned, &[])
    }

    /// [`Self::q8_acts`] for the Q8_K format the K-quants consume.
    fn q8k_acts<'a>(
        shared: Option<&'a BatchActs>,
        x_batch: &[f32],
        batch_size: usize,
        cols: usize,
        owned: &'a mut Vec<frink_quant::Q8KActivations>,
    ) -> (
        &'a [frink_quant::Q8KActivations],
        &'a [frink_quant::Q8KActsX4],
    ) {
        if let Some(BatchActs::Q8K {
            acts,
            tiles,
            cols: c,
        }) = shared
        {
            if acts.len() == batch_size && *c == cols {
                return (acts, tiles);
            }
        }
        *owned = (0..batch_size)
            .into_par_iter()
            .map(|b| frink_quant::quantize_activations_q8_k(&x_batch[b * cols..(b + 1) * cols]))
            .collect();
        (owned, &[])
    }

    /// Prefer serial when the mat is too small for fork-join to pay off.
    fn prefer_serial_matvec(rows: usize, cols: usize) -> bool {
        // ~256k f32-equivalent ops: below this, Rayon overhead dominates
        // on Host B-class cores for Q8/Q4 decode GEMVs.
        rows.saturating_mul(cols) < 256_000
    }

    fn dot(kind: QuantKind, row: &[u8], x: &[f32]) -> f32 {
        match kind {
            QuantKind::Q8_0 => frink_quant::dot_q8_0_f32(row, x),
            QuantKind::Q4_0 => frink_quant::dot_q4_0_f32(row, x),
            QuantKind::Q4K => frink_quant::dot_q4_k_f32(row, x),
            QuantKind::Q5K => frink_quant::dot_q5_k_f32(row, x),
            QuantKind::Q6K => frink_quant::dot_q6_k_f32(row, x),
            QuantKind::Q2K => frink_quant::dot_q2_k_f32(row, x),
            QuantKind::Q3K => frink_quant::dot_q3_k_f32(row, x),
            QuantKind::Q4_1 => frink_quant::dot_q4_1_f32(row, x),
            QuantKind::Q5_0 => frink_quant::dot_q5_0_f32(row, x),
            QuantKind::Q5_1 => frink_quant::dot_q5_1_f32(row, x),
            QuantKind::Q8_1 => frink_quant::dot_q8_1_f32(row, x),
            QuantKind::IQ4NL => frink_quant::dot_iq4_nl_f32(row, x),
            QuantKind::IQ4XS => frink_quant::dot_iq4_xs_f32(row, x),
            QuantKind::IQ1S => frink_quant::dot_iq1_s_f32(row, x),
            QuantKind::IQ2XXS => frink_quant::dot_iq2_xxs_f32(row, x),
            QuantKind::IQ3XXS => frink_quant::dot_iq3_xxs_f32(row, x),
            QuantKind::IQ2XS => frink_quant::dot_iq2_xs_f32(row, x),
            QuantKind::IQ2S => frink_quant::dot_iq2_s_f32(row, x),
            QuantKind::IQ3S => frink_quant::dot_iq3_s_f32(row, x),
            QuantKind::IQ1M => frink_quant::dot_iq1_m_f32(row, x),
            QuantKind::Mxfp4Gguf => frink_quant::dot_mxfp4_gguf_f32(row, x),
            QuantKind::Tq1_0 => {
                frink_quant::ternary::dot_trits_f32(row, frink_quant::ternary::TQ1_0, x)
            }
            QuantKind::Ptq1_0 => {
                frink_quant::ternary::dot_trits_f32(row, frink_quant::ternary::PTQ1_0, x)
            }
        }
    }

    /// Per-kind full-buffer dequantization -- the row-lookup counterpart
    /// of `dot`'s fused per-kind dispatch below.
    fn dequant(kind: QuantKind, bytes: &[u8]) -> Vec<f32> {
        let out = match kind {
            QuantKind::Q8_0 => frink_quant::dequant_q8_0(bytes),
            QuantKind::Q4_0 => frink_quant::dequant_q4_0(bytes),
            QuantKind::Q4K => frink_quant::dequant_q4_k(bytes),
            QuantKind::Q5K => frink_quant::dequant_q5_k(bytes),
            QuantKind::Q6K => frink_quant::dequant_q6_k(bytes),
            QuantKind::Q2K => frink_quant::dequant_q2_k(bytes),
            QuantKind::Q3K => frink_quant::dequant_q3_k(bytes),
            QuantKind::Q4_1 => frink_quant::dequant_q4_1(bytes),
            QuantKind::Q5_0 => frink_quant::dequant_q5_0(bytes),
            QuantKind::Q5_1 => frink_quant::dequant_q5_1(bytes),
            QuantKind::Q8_1 => frink_quant::dequant_q8_1(bytes),
            QuantKind::IQ4NL => frink_quant::dequant_iq4_nl(bytes),
            QuantKind::IQ4XS => frink_quant::dequant_iq4_xs(bytes),
            QuantKind::IQ1S => frink_quant::dequant_iq1_s(bytes),
            QuantKind::IQ2XXS => frink_quant::dequant_iq2_xxs(bytes),
            QuantKind::IQ3XXS => frink_quant::dequant_iq3_xxs(bytes),
            QuantKind::IQ2XS => frink_quant::dequant_iq2_xs(bytes),
            QuantKind::IQ2S => frink_quant::dequant_iq2_s(bytes),
            QuantKind::IQ3S => frink_quant::dequant_iq3_s(bytes),
            QuantKind::IQ1M => frink_quant::dequant_iq1_m(bytes),
            QuantKind::Mxfp4Gguf => frink_quant::dequant_mxfp4_gguf(bytes),
            QuantKind::Tq1_0 => {
                frink_quant::ternary::dequant_trits(bytes, frink_quant::ternary::TQ1_0)
            }
            QuantKind::Ptq1_0 => {
                frink_quant::ternary::dequant_trits(bytes, frink_quant::ternary::PTQ1_0)
            }
        };
        out.expect("row byte length is block-aligned by construction (block_bytes_per_row)")
    }

    /// Dequantizes exactly one row to f32, without touching any other
    /// row's bytes. This is what makes a *quantized* embedding table
    /// usable directly: token lookup reads `row_bytes` bytes and
    /// dequantizes `cols` values, instead of the whole vocabulary
    /// tensor ever being widened to f32 (which for a large-vocab model
    /// is a multi-GB allocation that exists only to be indexed one row
    /// at a time).
    pub fn dequant_row(&self, r: usize) -> Vec<f32> {
        assert!(r < self.rows(), "row {r} out of range ({})", self.rows());
        match self {
            WeightMatrix::F32(t) => t.row(r).to_vec(),
            WeightMatrix::Quantized {
                data, cols, kind, ..
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, *cols);
                let bytes = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                let out = Self::dequant(*kind, bytes);
                debug_assert_eq!(out.len(), *cols);
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                cols,
                ..
            } => {
                let packed_per_row = cols / 2;
                let scales_per_row = cols / frink_quant::MXFP4_GROUP_SIZE;
                let p = &packed.as_slice()[r * packed_per_row..(r + 1) * packed_per_row];
                let sc = &scale.as_slice()[r * scales_per_row..(r + 1) * scales_per_row];
                frink_quant::dequant_mxfp4_row(p, sc)
                    .expect("row slices are group-aligned by construction")
            }
            WeightMatrix::Adapted { base, lora } => {
                let mut row = base.dequant_row(r);
                lora.add_row_to(r, &mut row);
                row
            }
            // A stored row of a lookup table comes back to the primal
            // basis; a row of an input-folded weight stays folded, which
            // is what a caller splitting or widening the matrix needs.
            WeightMatrix::Folded { base, fold } => {
                let mut row = base.dequant_row(r);
                if fold.site == hadamard::FoldSite::RowLookup {
                    fold.restore_row(&mut row);
                }
                row
            }
        }
    }

    /// Whether batching this matrix during prefill beats running the
    /// fused per-position dense-FFN launch once per token.
    ///
    /// Measured, not assumed. Every kind with a simdgroup GEMM
    /// (`*_mul_mm_sg`) batches: Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, IQ4_XS.
    /// The remaining IQ codebook kinds have no GEMM, and their batched
    /// *matvec* loses to the fused per-position launch — IQ4_XS
    /// regressed 72.1 -> 33.2 on Llama-3.2-1B while it was in that
    /// state — so they keep the per-position path until a GEMM exists
    /// for them too.
    /// This matrix as a Metal simdgroup-GEMM descriptor, or `None` if
    /// its quant kind has no GEMM (so it must stay on the matvec path).
    /// Lets several matmuls be encoded into one command buffer instead
    /// of one launch each.
    #[cfg(feature = "metal")]
    pub fn mul_mm_sg_launch(&self) -> Option<frink_metal::gpu::MulMmSgLaunch<'_>> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        let kind_name = match kind {
            QuantKind::Q8_0 => "Q8_0",
            QuantKind::Q4_0 => "Q4_0",
            QuantKind::Q5_0 => "Q5_0",
            QuantKind::Q4K => "Q4_K",
            QuantKind::Q5K => "Q5_K",
            QuantKind::Q6K => "Q6_K",
            QuantKind::IQ4XS => "IQ4_XS",
            QuantKind::Ptq1_0 => "PTQ1_0",
            _ => return None,
        };
        let (fn_name, block_bytes, block_elems) = frink_metal::gpu::mul_mm_sg_meta(kind_name)?;
        Some(frink_metal::gpu::MulMmSgLaunch {
            weights: data.as_slice(),
            rows: *rows,
            row_bytes: self.block_bytes_per_row(*kind, *cols),
            fn_name,
            block_bytes,
            block_elems,
        })
    }

    /// This matrix as the CUDA batched GEMM takes it, or `None` for an
    /// adapted matrix (a LoRA delta lives in `WeightMatrix::Adapted` and
    /// no raw-bytes launch serves it) or a kind with no `mul_mm` row.
    /// The CUDA twin of [`Self::mul_mm_sg_launch`], and the ONE
    /// constructor of [`frink_cuda::prefill::MulMmWeights`]: the row
    /// byte count comes from the same `block_bytes_per_row` the matvec
    /// seam is held to, so a kind added there is a kind added here.
    #[cfg(feature = "cuda")]
    pub fn cuda_mul_mm_view(&self) -> Option<frink_cuda::prefill::MulMmWeights<'_>> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if !cuda_mul_mm_kind_supported(*kind) {
            return None;
        }
        let mm_kind = frink_cuda::mul_mm::kind_by_name(kind.name())?;
        Some(frink_cuda::prefill::MulMmWeights {
            kind: mm_kind,
            data: data.as_slice(),
            rows: *rows,
            cols: *cols,
            row_bytes: self.block_bytes_per_row(*kind, *cols),
        })
    }

    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub fn prefers_gpu_batch(&self) -> bool {
        !matches!(
            self.base(),
            WeightMatrix::Quantized {
                kind: QuantKind::IQ4NL
                    | QuantKind::IQ1S
                    | QuantKind::IQ2XXS
                    | QuantKind::IQ3XXS
                    | QuantKind::IQ2XS
                    | QuantKind::IQ2S
                    | QuantKind::IQ3S
                    | QuantKind::IQ1M,
                ..
            }
        )
    }

    /// Computes `W @ x` for a single activation vector `x` of length
    /// `self.cols()`, returning a vector of length `self.rows()`.
    /// Parallelized over output rows with rayon, same decomposition as
    /// `matmul_f32`.
    ///
    /// With `--features metal` / `--features cuda`, when the matching
    /// dense GPU env selects a device (see [`metal_dense_enabled`] /
    /// [`cuda_dense_enabled`]), quantized kinds that have a GPU kernel
    /// go through [`Self::apply_gpu`] first so dense Llama-class
    /// decode uses the GPU instead of only MoE expert placement.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );
        crate::activation_tap::observe(self, x, 1);
        if let WeightMatrix::Adapted { base, lora } = self {
            let mut out = base.apply(x);
            lora.add_to(x, &mut out);
            return out;
        }
        if let WeightMatrix::Folded { base, fold } = self {
            return base.apply(&fold.transform_input(x));
        }
        #[cfg(feature = "cuda")]
        {
            if cuda_dense_enabled() {
                if let Some(out) = self.apply_gpu(x) {
                    return out;
                }
            }
        }
        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled() {
                if let Some(out) = self.apply_gpu(x) {
                    return out;
                }
            }
        }
        self.apply_cpu(x)
    }

    /// [`Self::apply`] followed by `softcap_inplace(.., softcap)`, as
    /// ONE operation: Gemma-2's lm_head with its `final_logit_softcap`.
    ///
    /// On Metal the cap runs as an epilogue in the matvec's own command
    /// buffer (`frink_metal::gpu::MatvecEpilogue`), so the host never
    /// walks the 256k logits before sampling them: that walk was
    /// 0.65 ms per token, more than the whole encode phase (PR #202).
    /// Everywhere else, and whenever the Metal launch is refused or
    /// fails, it is the host multiply-tanh it always was. Either way
    /// the caller gets capped logits and never has to remember the cap.
    pub fn apply_softcapped(&self, x: &[f32], softcap: f32) -> Vec<f32> {
        #[cfg(feature = "metal")]
        if metal_dense_enabled() {
            if let WeightMatrix::Quantized {
                data,
                rows,
                cols,
                kind,
            } = self
            {
                if let Some(kind_name) = Metal::matvec_kernel(*kind) {
                    crate::activation_tap::observe(self, x, 1);
                    let row_bytes = self.block_bytes_per_row(*kind, *cols);
                    let epilogue = frink_metal::gpu::MatvecEpilogue {
                        softcap: Some(softcap),
                    };
                    match frink_metal::gpu::launch_matvec_kind_with(
                        kind_name,
                        data.as_slice(),
                        x,
                        *rows,
                        row_bytes,
                        epilogue,
                    ) {
                        Some(Ok(out)) => return out,
                        Some(Err(e)) => {
                            eprintln!(
                                "frink: Metal softcapped matvec failed, falling back to CPU: {e}"
                            );
                        }
                        None => {}
                    }
                }
            }
        }
        let mut out = self.apply(x);
        crate::matmul::softcap_inplace(&mut out, softcap);
        out
    }

    /// CPU-only matvec (NEON/AVX/scalar via `frink-quant`). Used by
    /// [`Self::apply`] after Metal miss/disable, and by GPU parity tests
    /// that must not recurse into [`Self::apply_gpu`].
    /// Applies three independent matrices to the same activation,
    /// overlapping their parallel regions instead of running them one
    /// after another.
    ///
    /// Decode opens one rayon fork-join per weight matrix -- roughly
    /// seven per layer -- and the measured CPU decode deficit is
    /// scheduling, not kernels (frink scales 1.40x/2.93x from 1 to 6
    /// threads where llama.cpp scales 1.99x/4.39x, while *beating* llama
    /// at one thread). q/k/v share an input and are independent, so
    /// their regions can coexist and let rayon's work-stealing fill
    /// threads that would otherwise idle at the tail of each one.
    ///
    /// Under [`crate::par::Backend::Spin`] the three run one after the
    /// other instead: each already spreads across the whole persistent
    /// pool, and the reason to overlap them was to hide a fork-join that
    /// the persistent pool does not pay. That choice lives in
    /// [`crate::par::join3`], not here, so it cannot drift from the one
    /// in `frink-moe`'s gate/up pair.
    ///
    /// On a GPU backend the three go into ONE command buffer
    /// ([`Self::apply_gpu_multi`]: one upload of `x`, one wait) and fall
    /// back to three sequential launches when the fused path declines
    /// (a kind without a kernel, folds that differ). Each round trip a
    /// per-matvec model pays is ~0.2 ms of latency beyond the kernel
    /// (measured on an M2 Pro with `FRINK_METAL_GPU_TIMING`), which
    /// is what this saves.
    pub fn apply_three(a: &Self, b: &Self, c: &Self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if Self::gpu_dense_active() {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if let Some(mut outs) = Self::apply_gpu_multi(&[a, b, c], x) {
                let c = outs.pop().unwrap();
                let b = outs.pop().unwrap();
                let a = outs.pop().unwrap();
                return (a, b, c);
            }
            return (a.apply(x), b.apply(x), c.apply(x));
        }
        crate::par::join3(|| a.apply(x), || b.apply(x), || c.apply(x))
    }

    /// Two projections of one activation: [`Self::apply_three`] for a
    /// pair (a gated delta-net's `qkv` and `z`, a GLU's gate and up).
    pub fn apply_pair(a: &Self, b: &Self, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if Self::gpu_dense_active() {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if let Some(mut outs) = Self::apply_gpu_multi(&[a, b], x) {
                let b = outs.pop().unwrap();
                let a = outs.pop().unwrap();
                return (a, b);
            }
            return (a.apply(x), b.apply(x));
        }
        crate::par::join2(|| a.apply(x), || b.apply(x))
    }

    /// Whether a dense matvec would go to an accelerator right now.
    fn gpu_dense_active() -> bool {
        #[cfg(feature = "metal")]
        let gpu = metal_dense_enabled();
        #[cfg(not(feature = "metal"))]
        let gpu = false;
        #[cfg(feature = "cuda")]
        let gpu = gpu || cuda_dense_enabled();
        gpu
    }

    pub fn apply_cpu(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );
        // Decode: one activation, so this operation is `rows x cols`
        // MACs and a task's share of it is (rows in task) x cols.
        // Publishing the shape is what lets both the scheduler choice
        // and the task floor be work-aware rather than row-count-aware.
        crate::par::with_op_work(self.rows(), x.len(), || self.apply_cpu_inner(x))
    }

    fn apply_cpu_inner(&self, x: &[f32]) -> Vec<f32> {
        match self {
            WeightMatrix::F32(t) => {
                let xt = Tensor::new(x.to_vec(), vec![1, x.len()]);
                crate::matmul::matmul_f32(&xt, t).data
            }
            WeightMatrix::Quantized {
                data,
                rows,
                cols,
                kind,
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, *cols);
                let mut out = vec![0f32; *rows];
                // FRINK_CPU_INT_DOT=1: quantize the shared activation once,
                // then every row dot is int8×int8 → i32 (llama.cpp CPU matmul).
                // Q8_0/Q4_0 use 32-elem Q8_0 acts; Q4_K/Q5_K/Q6_K use Q8_K.
                if cpu_int_dot_for(IntDotShape::Matvec) {
                    match *kind {
                        QuantKind::Q8_0 if x.len().is_multiple_of(32) => {
                            let act = frink_quant::quantize_activations_q8(x);
                            let n_groups = *rows / frink_quant::Q8_0X4_NROWS;
                            let serial = Self::prefer_serial_matvec(*rows, *cols);
                            // Probed once per matvec, not once per row-group:
                            // `q*_interleave` reads a CPU feature bit, and LLVM
                            // cannot hoist that relaxed atomic load out of the
                            // caller's loop. `is_aarch64_feature_detected!` ran
                            // 131k times in one Mistral-7B projection before the
                            // last one of these was hoisted.
                            let interleave = frink_quant::q8_0x4_interleave();
                            if n_groups > 0 {
                                let packed = get_or_repack_q8x4(data, *rows, *cols);
                                if serial {
                                    for (g, chunk) in out[..n_groups * frink_quant::Q8_0X4_NROWS]
                                        .chunks_mut(frink_quant::Q8_0X4_NROWS)
                                        .enumerate()
                                    {
                                        frink_quant::gemv_q8_0x4_group(
                                            &packed, g, &act, *cols, interleave, chunk,
                                        );
                                    }
                                } else {
                                    crate::par::chunks_mut(
                                        &mut out[..n_groups * frink_quant::Q8_0X4_NROWS],
                                        frink_quant::Q8_0X4_NROWS,
                                        Self::min_rows_per_task(n_groups).max(1),
                                        |g, chunk| {
                                            frink_quant::gemv_q8_0x4_group(
                                                &packed, g, &act, *cols, interleave, chunk,
                                            );
                                        },
                                    );
                                }
                                let data_slice = data.as_slice();
                                let tail_len = *rows - n_groups * frink_quant::Q8_0X4_NROWS;
                                if tail_len > 0 {
                                    let tail = &mut out[n_groups * frink_quant::Q8_0X4_NROWS..];
                                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                                        for (i, o) in tail.iter_mut().enumerate() {
                                            let r = n_groups * frink_quant::Q8_0X4_NROWS + i;
                                            let row =
                                                &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                            *o = frink_quant::dot_q8_0_q8(row, &act);
                                        }
                                    } else {
                                        let min_len = Self::min_rows_per_task(tail_len);
                                        crate::par::items_mut(tail, min_len, |i, o| {
                                            let r = n_groups * frink_quant::Q8_0X4_NROWS + i;
                                            let row =
                                                &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                            *o = frink_quant::dot_q8_0_q8(row, &act);
                                        });
                                    }
                                }
                                return out;
                            }
                            if serial {
                                for (r, o) in out.iter_mut().enumerate() {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_q8_0_q8(row, &act);
                                }
                            } else {
                                crate::par::items_mut(
                                    &mut out,
                                    Self::min_rows_per_task(*rows),
                                    |r, o| {
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        *o = frink_quant::dot_q8_0_q8(row, &act);
                                    },
                                );
                            }
                            return out;
                        }
                        QuantKind::Q4_0 if x.len().is_multiple_of(32) => {
                            let act = frink_quant::quantize_activations_q8(x);
                            let n_groups = *rows / frink_quant::Q4_0X4_NROWS;
                            let serial = Self::prefer_serial_matvec(*rows, *cols);
                            // Probed once per matvec, not once per row-group:
                            // `q*_interleave` reads a CPU feature bit, and LLVM
                            // cannot hoist that relaxed atomic load out of the
                            // caller's loop. `is_aarch64_feature_detected!` ran
                            // 131k times in one Mistral-7B projection before the
                            // last one of these was hoisted.
                            let interleave = frink_quant::q4_0x4_interleave();
                            if n_groups > 0 {
                                let packed = get_or_repack_q4_0x4(data, *rows, *cols);
                                if serial {
                                    for (g, chunk) in out[..n_groups * frink_quant::Q4_0X4_NROWS]
                                        .chunks_mut(frink_quant::Q4_0X4_NROWS)
                                        .enumerate()
                                    {
                                        frink_quant::gemv_q4_0x4_group(
                                            &packed, g, &act, *cols, interleave, chunk,
                                        );
                                    }
                                } else {
                                    crate::par::chunks_mut(
                                        &mut out[..n_groups * frink_quant::Q4_0X4_NROWS],
                                        frink_quant::Q4_0X4_NROWS,
                                        Self::min_rows_per_task(n_groups).max(1),
                                        |g, chunk| {
                                            frink_quant::gemv_q4_0x4_group(
                                                &packed, g, &act, *cols, interleave, chunk,
                                            );
                                        },
                                    );
                                }
                                let data_slice = data.as_slice();
                                let tail_len = *rows - n_groups * frink_quant::Q4_0X4_NROWS;
                                if tail_len > 0 {
                                    let tail = &mut out[n_groups * frink_quant::Q4_0X4_NROWS..];
                                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                                        for (i, o) in tail.iter_mut().enumerate() {
                                            let r = n_groups * frink_quant::Q4_0X4_NROWS + i;
                                            let row =
                                                &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                            *o = frink_quant::dot_q4_0_q8(row, &act);
                                        }
                                    } else {
                                        let min_len = Self::min_rows_per_task(tail_len);
                                        crate::par::items_mut(tail, min_len, |i, o| {
                                            let r = n_groups * frink_quant::Q4_0X4_NROWS + i;
                                            let row =
                                                &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                            *o = frink_quant::dot_q4_0_q8(row, &act);
                                        });
                                    }
                                }
                                return out;
                            }
                            if serial {
                                for (r, o) in out.iter_mut().enumerate() {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_q4_0_q8(row, &act);
                                }
                            } else {
                                crate::par::items_mut(
                                    &mut out,
                                    Self::min_rows_per_task(*rows),
                                    |r, o| {
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        *o = frink_quant::dot_q4_0_q8(row, &act);
                                    },
                                );
                            }
                            return out;
                        }
                        QuantKind::Q4K if x.len().is_multiple_of(256) => {
                            let act = frink_quant::quantize_activations_q8_k(x);
                            let n_groups = *rows / frink_quant::Q4_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = frink_quant::q4_kx8_interleave();
                                let packed = get_or_repack_q4k(data, *rows, *cols);
                                crate::par::chunks_mut(
                                    &mut out[..n_groups * frink_quant::Q4_KX8_NROWS],
                                    frink_quant::Q4_KX8_NROWS,
                                    Self::min_rows_per_task(n_groups).max(1),
                                    |g, chunk| {
                                        frink_quant::gemv_q4_kx8_group(
                                            &packed, g, &act, *cols, interleave, chunk,
                                        );
                                    },
                                );
                                let data_slice = data.as_slice();
                                crate::par::items_mut(
                                    &mut out[n_groups * frink_quant::Q4_KX8_NROWS..],
                                    Self::min_rows_per_task(
                                        *rows - n_groups * frink_quant::Q4_KX8_NROWS,
                                    ),
                                    |i, o| {
                                        let r = n_groups * frink_quant::Q4_KX8_NROWS + i;
                                        let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                        *o = frink_quant::dot_q4_k_q8(row, &act);
                                    },
                                );
                                return out;
                            }
                            crate::par::items_mut(
                                &mut out,
                                Self::min_rows_per_task(*rows),
                                |r, o| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_q4_k_q8(row, &act);
                                },
                            );
                            return out;
                        }
                        QuantKind::Q5K if x.len().is_multiple_of(256) => {
                            let act = frink_quant::quantize_activations_q8_k(x);
                            let n_groups = *rows / frink_quant::Q5_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = frink_quant::q5_kx8_interleave();
                                let packed = get_or_repack_q5k(data, *rows, *cols);
                                crate::par::chunks_mut(
                                    &mut out[..n_groups * frink_quant::Q5_KX8_NROWS],
                                    frink_quant::Q5_KX8_NROWS,
                                    Self::min_rows_per_task(n_groups).max(1),
                                    |g, chunk| {
                                        frink_quant::gemv_q5_kx8_group(
                                            &packed, g, &act, *cols, interleave, chunk,
                                        );
                                    },
                                );
                                let data_slice = data.as_slice();
                                crate::par::items_mut(
                                    &mut out[n_groups * frink_quant::Q5_KX8_NROWS..],
                                    Self::min_rows_per_task(
                                        *rows - n_groups * frink_quant::Q5_KX8_NROWS,
                                    ),
                                    |i, o| {
                                        let r = n_groups * frink_quant::Q5_KX8_NROWS + i;
                                        let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                        *o = frink_quant::dot_q5_k_q8(row, &act);
                                    },
                                );
                                return out;
                            }
                            crate::par::items_mut(
                                &mut out,
                                Self::min_rows_per_task(*rows),
                                |r, o| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_q5_k_q8(row, &act);
                                },
                            );
                            return out;
                        }
                        // IQ4_XS over Q8_K activations, llama.cpp's
                        // `ggml_vec_dot_iq4_xs_q8_K` (`frink_quant::
                        // iq4_xs_q8`): the same int8 lane the K-quants
                        // take, so decode and prefill agree on the
                        // activation quantization.
                        QuantKind::IQ4XS if x.len().is_multiple_of(256) => {
                            let act = frink_quant::quantize_activations_q8_k(x);
                            crate::par::items_mut(
                                &mut out,
                                Self::min_rows_per_task(*rows),
                                |r, o| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_iq4_xs_q8_k(row, &act);
                                },
                            );
                            return out;
                        }
                        QuantKind::Q6K if x.len().is_multiple_of(256) => {
                            let act = frink_quant::quantize_activations_q8_k(x);
                            let n_groups = *rows / frink_quant::Q6_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = frink_quant::q6_kx8_interleave();
                                let packed = get_or_repack_q6k(data, *rows, *cols);
                                crate::par::chunks_mut(
                                    &mut out[..n_groups * frink_quant::Q6_KX8_NROWS],
                                    frink_quant::Q6_KX8_NROWS,
                                    Self::min_rows_per_task(n_groups).max(1),
                                    |g, out8| {
                                        frink_quant::gemv_q6_kx8_group(
                                            &packed, g, &act, *cols, interleave, out8,
                                        );
                                    },
                                );
                                crate::par::items_mut(
                                    &mut out[n_groups * frink_quant::Q6_KX8_NROWS..],
                                    Self::min_rows_per_task(
                                        *rows - n_groups * frink_quant::Q6_KX8_NROWS,
                                    ),
                                    |i, o| {
                                        let r = n_groups * frink_quant::Q6_KX8_NROWS + i;
                                        let row =
                                            &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                        *o = frink_quant::dot_q6_k_q8(row, &act);
                                    },
                                );
                                return out;
                            }
                            crate::par::items_mut(
                                &mut out,
                                Self::min_rows_per_task(*rows),
                                |r, o| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    *o = frink_quant::dot_q6_k_q8(row, &act);
                                },
                            );
                            return out;
                        }
                        _ => {}
                    }
                }
                crate::par::items_mut(&mut out, Self::min_rows_per_task(*rows), |r, o| {
                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                    *o = Self::dot(*kind, row, x);
                });
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                rows,
                cols,
            } => {
                let packed_row_bytes = cols / 2;
                let scale_row_bytes = cols / frink_quant::MXFP4_GROUP_SIZE;
                let mut out = vec![0f32; *rows];
                crate::par::items_mut(&mut out, Self::min_rows_per_task(*rows), |r, o| {
                    let prow = &packed.as_slice()[r * packed_row_bytes..(r + 1) * packed_row_bytes];
                    let srow = &scale.as_slice()[r * scale_row_bytes..(r + 1) * scale_row_bytes];
                    *o = frink_quant::dot_mxfp4_row_f32(prow, srow, x);
                });
                out
            }
            WeightMatrix::Adapted { base, lora } => {
                let mut out = base.apply_cpu_inner(x);
                lora.add_to(x, &mut out);
                out
            }
            WeightMatrix::Folded { base, fold } => base.apply_cpu_inner(&fold.transform_input(x)),
        }
    }

    /// INT_DOT matvec against a pre-quantized Q8_0 activation (shared gate/up).
    ///
    /// Publishes this operation's shape for exactly the same reason
    /// [`Self::apply_cpu`] does, and it matters more here: the dense FFN
    /// gate and up projections are the widest matvecs in a decode step,
    /// so they are the ones the scheduler rule is deciding about.
    pub fn apply_cpu_q8(&self, act: &frink_quant::Q8Activations) -> Option<Vec<f32>> {
        crate::par::with_op_work(self.rows(), self.cols(), || self.apply_cpu_q8_inner(act))
    }

    /// [`Self::apply_cpu_q8`] with the operation's shape already
    /// published. Split only so the publish wraps every return path.
    fn apply_cpu_q8_inner(&self, act: &frink_quant::Q8Activations) -> Option<Vec<f32>> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0)
            || !cpu_int_dot_for(IntDotShape::Matvec)
        {
            return None;
        }
        if act.q.len() != *cols || !cols.is_multiple_of(32) {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let mut out = vec![0f32; *rows];
        let kind = *kind;
        let bytes = data.as_slice();
        // Q8_0×4 / Q4_0×4 interleaved GEMV — same paths as `apply_cpu` so
        // dense FFN gate+up hit the fast kernels, not per-row int dots.
        if matches!(kind, QuantKind::Q8_0) {
            let n_groups = *rows / frink_quant::Q8_0X4_NROWS;
            if n_groups > 0 {
                let packed = get_or_repack_q8x4(data, *rows, *cols);
                let serial = Self::prefer_serial_matvec(*rows, *cols);
                // Probed once per matvec, not once per row-group:
                // `q*_interleave` reads a CPU feature bit, and LLVM
                // cannot hoist that relaxed atomic load out of the
                // caller's loop. `is_aarch64_feature_detected!` ran
                // 131k times in one Mistral-7B projection before the
                // last one of these was hoisted.
                let interleave = frink_quant::q8_0x4_interleave();
                let body = |g: usize, chunk: &mut [f32]| {
                    frink_quant::gemv_q8_0x4_group(&packed, g, act, *cols, interleave, chunk);
                };
                if serial {
                    for (g, chunk) in out[..n_groups * frink_quant::Q8_0X4_NROWS]
                        .chunks_mut(frink_quant::Q8_0X4_NROWS)
                        .enumerate()
                    {
                        body(g, chunk);
                    }
                } else {
                    crate::par::chunks_mut(
                        &mut out[..n_groups * frink_quant::Q8_0X4_NROWS],
                        frink_quant::Q8_0X4_NROWS,
                        Self::min_rows_per_task(n_groups).max(1),
                        |g, chunk| body(g, chunk),
                    );
                }
                let tail_len = *rows - n_groups * frink_quant::Q8_0X4_NROWS;
                if tail_len > 0 {
                    let tail = &mut out[n_groups * frink_quant::Q8_0X4_NROWS..];
                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                        for (i, o) in tail.iter_mut().enumerate() {
                            let r = n_groups * frink_quant::Q8_0X4_NROWS + i;
                            *o = frink_quant::dot_q8_0_q8(
                                &bytes[r * row_bytes..(r + 1) * row_bytes],
                                act,
                            );
                        }
                    } else {
                        let min_len = Self::min_rows_per_task(tail_len);
                        crate::par::items_mut(tail, min_len, |i, o| {
                            let r = n_groups * frink_quant::Q8_0X4_NROWS + i;
                            *o = frink_quant::dot_q8_0_q8(
                                &bytes[r * row_bytes..(r + 1) * row_bytes],
                                act,
                            );
                        });
                    }
                }
                return Some(out);
            }
        }
        if matches!(kind, QuantKind::Q4_0) {
            let n_groups = *rows / frink_quant::Q4_0X4_NROWS;
            if n_groups > 0 {
                let packed = get_or_repack_q4_0x4(data, *rows, *cols);
                let serial = Self::prefer_serial_matvec(*rows, *cols);
                // Probed once per matvec, not once per row-group:
                // `q*_interleave` reads a CPU feature bit, and LLVM
                // cannot hoist that relaxed atomic load out of the
                // caller's loop. `is_aarch64_feature_detected!` ran
                // 131k times in one Mistral-7B projection before the
                // last one of these was hoisted.
                let interleave = frink_quant::q4_0x4_interleave();
                let body = |g: usize, chunk: &mut [f32]| {
                    frink_quant::gemv_q4_0x4_group(&packed, g, act, *cols, interleave, chunk);
                };
                if serial {
                    for (g, chunk) in out[..n_groups * frink_quant::Q4_0X4_NROWS]
                        .chunks_mut(frink_quant::Q4_0X4_NROWS)
                        .enumerate()
                    {
                        body(g, chunk);
                    }
                } else {
                    crate::par::chunks_mut(
                        &mut out[..n_groups * frink_quant::Q4_0X4_NROWS],
                        frink_quant::Q4_0X4_NROWS,
                        Self::min_rows_per_task(n_groups).max(1),
                        |g, chunk| body(g, chunk),
                    );
                }
                let tail_len = *rows - n_groups * frink_quant::Q4_0X4_NROWS;
                if tail_len > 0 {
                    let tail = &mut out[n_groups * frink_quant::Q4_0X4_NROWS..];
                    if serial || Self::prefer_serial_matvec(tail_len, *cols) {
                        for (i, o) in tail.iter_mut().enumerate() {
                            let r = n_groups * frink_quant::Q4_0X4_NROWS + i;
                            *o = frink_quant::dot_q4_0_q8(
                                &bytes[r * row_bytes..(r + 1) * row_bytes],
                                act,
                            );
                        }
                    } else {
                        let min_len = Self::min_rows_per_task(tail_len);
                        crate::par::items_mut(tail, min_len, |i, o| {
                            let r = n_groups * frink_quant::Q4_0X4_NROWS + i;
                            *o = frink_quant::dot_q4_0_q8(
                                &bytes[r * row_bytes..(r + 1) * row_bytes],
                                act,
                            );
                        });
                    }
                }
                return Some(out);
            }
        }
        if Self::prefer_serial_matvec(*rows, *cols) {
            for (r, o) in out.iter_mut().enumerate() {
                let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
                *o = match kind {
                    QuantKind::Q8_0 => frink_quant::dot_q8_0_q8(row, act),
                    QuantKind::Q4_0 => frink_quant::dot_q4_0_q8(row, act),
                    _ => unreachable!(),
                };
            }
            return Some(out);
        }
        crate::par::items_mut(&mut out, Self::min_rows_per_task(*rows), |r, o| {
            let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
            *o = match kind {
                QuantKind::Q8_0 => frink_quant::dot_q8_0_q8(row, act),
                QuantKind::Q4_0 => frink_quant::dot_q4_0_q8(row, act),
                _ => unreachable!(),
            };
        });
        Some(out)
    }

    /// Two contiguous rows × one Q8 act (shared act loads). Q4_0 uses
    /// [`frink_quant::dot_q4_0_q8_2row`]; Q8_0 falls back to two singles.
    pub fn dot_pair_cpu_q8(
        &self,
        row: usize,
        act: &frink_quant::Q8Activations,
    ) -> Option<(f32, f32)> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0)
            || !cpu_int_dot_for(IntDotShape::Matvec)
        {
            return None;
        }
        if act.q.len() != *cols || !cols.is_multiple_of(32) || row + 1 >= *rows {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let bytes = data.as_slice();
        let r0 = &bytes[row * row_bytes..(row + 1) * row_bytes];
        let r1 = &bytes[(row + 1) * row_bytes..(row + 2) * row_bytes];
        Some(match *kind {
            QuantKind::Q4_0 => frink_quant::dot_q4_0_q8_2row(r0, r1, act),
            QuantKind::Q8_0 => (
                frink_quant::dot_q8_0_q8(r0, act),
                frink_quant::dot_q8_0_q8(r1, act),
            ),
            _ => unreachable!(),
        })
    }

    /// Single-row INT_DOT against pre-quantized Q8_0 acts (llama `mul_mat_id`
    /// inner loop). Returns `None` if this matrix is not Q4_0/Q8_0 INT_DOT.
    pub fn dot_row_cpu_q8(&self, row: usize, act: &frink_quant::Q8Activations) -> Option<f32> {
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        if row >= *rows
            || !matches!(*kind, QuantKind::Q8_0 | QuantKind::Q4_0)
            || !cpu_int_dot_for(IntDotShape::Matvec)
            || act.q.len() != *cols
            || !cols.is_multiple_of(32)
        {
            return None;
        }
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        let bytes = &data.as_slice()[row * row_bytes..(row + 1) * row_bytes];
        Some(match *kind {
            QuantKind::Q8_0 => frink_quant::dot_q8_0_q8(bytes, act),
            QuantKind::Q4_0 => frink_quant::dot_q4_0_q8(bytes, act),
            _ => unreachable!(),
        })
    }

    /// Computes `W @ X` for a *batch* of activation vectors at once:
    /// `x_batch` is `batch_size` rows of `self.cols()` elements each,
    /// flattened row-major; returns `batch_size` rows of
    /// `self.rows()` elements each, flattened row-major (`[batch,
    /// rows]`, matching the layout `Tensor`/`Decoder` expect for
    /// chaining into further matmuls).
    ///
    /// This is not just a convenience wrapper: for a quantized matrix,
    /// each weight row's bytes are read from memory *once* and dotted
    /// against every activation in the batch, instead of once per
    /// `apply` call. For a memory-bandwidth-bound quantized matmul --
    /// which fused Q8_0/Q4_0 dot products are, since the whole point of
    /// keeping weights quantized is that reading them is the
    /// bottleneck, not the arithmetic -- processing `batch_size`
    /// positions this way costs roughly the same *memory traffic* as
    /// processing one position, not `batch_size` times as much. This
    /// is the same reason speculative-decoding verification and batched
    /// prefill are faster per-token than sequential single-token decode
    /// on real hardware: it turns `batch_size` separate reads of the
    /// same weights into one.
    ///
    /// With Metal dense enabled, dispatches a single batched Metal
    /// command buffer — Q4_0/Q4_K/Q6_K/Q8_0 reuse the weights through a
    /// simdgroup `mul_mm` at `batch_size >= 4`; every other kind, and
    /// every smaller batch, uses
    /// [`frink_metal::gpu::launch_matvec_batch`]. Falls back to
    /// per-row [`Self::apply`] if the batch launch fails.
    pub fn apply_batch(&self, x_batch: &[f32], batch_size: usize) -> Vec<f32> {
        self.apply_batch_with_acts(x_batch, batch_size, None)
    }

    /// Quantize `x_batch` once, in the activation format this matrix's
    /// INT_DOT batch path consumes, for sharing across every projection
    /// that reads the same input (q/k/v on one normed batch; gate/up on
    /// another). Returns `None` when [`Self::apply_batch`] would not use
    /// quantized activations for this matrix — GPU dispatch, INT_DOT off,
    /// unsupported kind or width — so callers can pass the result straight
    /// to [`Self::apply_batch_with_acts`] unconditionally.
    pub fn quantize_batch_acts(&self, x_batch: &[f32], batch_size: usize) -> Option<BatchActs> {
        if let WeightMatrix::Adapted { base, .. } = self {
            // The activations the BASE consumes; the delta reads the
            // f32 batch itself.
            return base.quantize_batch_acts(x_batch, batch_size);
        }
        if let WeightMatrix::Folded { .. } = self {
            // The base consumes the TRANSFORMED batch, which no sibling
            // matrix shares unless it shares the fold; `apply_batch`
            // quantizes its own.
            return None;
        }
        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled()
                && matches!(
                    self,
                    WeightMatrix::Quantized { kind, .. } if Self::metal_kind_supported(*kind)
                )
            {
                return None;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if cuda_dense_enabled() && matches!(self, WeightMatrix::Quantized { .. }) {
                return None;
            }
        }
        let WeightMatrix::Quantized { cols, kind, .. } = self else {
            return None;
        };
        if !cpu_int_dot_for(IntDotShape::BatchGemm) || x_batch.len() != batch_size * cols {
            return None;
        }
        let cols = *cols;
        match kind {
            QuantKind::Q8_0 | QuantKind::Q4_0 if cols.is_multiple_of(32) => {
                let acts: Vec<_> = (0..batch_size)
                    .into_par_iter()
                    .map(|b| {
                        frink_quant::quantize_activations_q8(&x_batch[b * cols..(b + 1) * cols])
                    })
                    .collect();
                // Q8_0 and Q4_0 agree on both the interleave width and the
                // predicate, so one tile set serves either consumer.
                let tiles =
                    if frink_quant::q8_0x4_gemm_uses_acts_x4(frink_quant::q8_0x4_interleave()) {
                        acts.par_chunks(frink_quant::Q8K_ACTS_X4_NC)
                            .map(|chunk| frink_quant::prepare_q8_acts_x4(chunk, cols))
                            .collect()
                    } else {
                        Vec::new()
                    };
                Some(BatchActs::Q8 { acts, tiles, cols })
            }
            QuantKind::Q4K | QuantKind::Q5K | QuantKind::Q6K if cols.is_multiple_of(256) => {
                let acts: Vec<_> = (0..batch_size)
                    .into_par_iter()
                    .map(|b| {
                        frink_quant::quantize_activations_q8_k(&x_batch[b * cols..(b + 1) * cols])
                    })
                    .collect();
                // All three K-quants share the predicate and the quad
                // width, so the set a Q4_K gate builds is exactly what a
                // Q5_K or Q6_K sibling would have built for itself.
                let tiles =
                    if frink_quant::q4_kx8_gemm_uses_acts_x4(frink_quant::q4_kx8_interleave()) {
                        acts.par_chunks(frink_quant::Q8K_ACTS_X4_NC)
                            .map(|chunk| frink_quant::prepare_q8_k_acts_x4(chunk, cols))
                            .collect()
                    } else {
                        Vec::new()
                    };
                Some(BatchActs::Q8K { acts, tiles, cols })
            }
            _ => None,
        }
    }

    /// [`Self::apply_batch`], optionally reusing a shared pre-quantized
    /// activation batch from [`Self::quantize_batch_acts`]. A `shared`
    /// value whose format or length does not match this matrix is simply
    /// ignored (the activations are re-quantized locally), so mixed-kind
    /// projection groups stay correct.
    /// Two matrices of one batch, with a folded rotation applied ONCE
    /// when they share it.
    ///
    /// `gate` and `up` read the same batch, and so do a gated
    /// delta-net's `qkv` and `z`; each call to
    /// [`Self::apply_batch_with_acts`] rotates the batch for itself, so
    /// a folded pair transformed the same `[rows][n_embd]` block twice.
    /// The butterfly is the second-largest cost of a Bonsai prefill
    /// step (`sample`, beside the recurrence), and half of it was that
    /// repeat.
    ///
    /// Falls back to two independent calls whenever the pair does not
    /// share one fold, which is every unfolded model: there the cost
    /// being saved does not exist.
    pub fn apply_batch_pair_with_acts(
        a: &Self,
        b: &Self,
        x_batch: &[f32],
        batch_size: usize,
        shared: Option<&BatchActs>,
    ) -> (Vec<f32>, Vec<f32>) {
        if let (
            WeightMatrix::Folded {
                base: base_a,
                fold: fold_a,
            },
            WeightMatrix::Folded {
                base: base_b,
                fold: fold_b,
            },
        ) = (a, b)
        {
            if std::sync::Arc::ptr_eq(fold_a, fold_b) && batch_size > 0 {
                let transformed = fold_a.transform_rows(x_batch, batch_size);
                // `shared` describes the UNROTATED batch, so it cannot
                // be handed to a matrix reading the rotated one.
                return (
                    base_a.apply_batch_with_acts(&transformed, batch_size, None),
                    base_b.apply_batch_with_acts(&transformed, batch_size, None),
                );
            }
        }
        (
            a.apply_batch_with_acts(x_batch, batch_size, shared),
            b.apply_batch_with_acts(x_batch, batch_size, shared),
        )
    }

    pub fn apply_batch_with_acts(
        &self,
        x_batch: &[f32],
        batch_size: usize,
        shared: Option<&BatchActs>,
    ) -> Vec<f32> {
        let cols = self.cols();
        assert_eq!(
            x_batch.len(),
            batch_size * cols,
            "x_batch length must be batch_size * cols"
        );
        if batch_size == 0 {
            return Vec::new();
        }
        crate::activation_tap::observe(self, x_batch, batch_size);
        if let WeightMatrix::Adapted { base, lora } = self {
            let mut out = base.apply_batch_with_acts(x_batch, batch_size, shared);
            lora.add_batch_to(x_batch, batch_size, &mut out);
            return out;
        }
        if let WeightMatrix::Folded { base, fold } = self {
            let transformed = fold.transform_rows(x_batch, batch_size);
            return base.apply_batch_with_acts(&transformed, batch_size, None);
        }

        /// Raw pointer to this function's `[batch][rows]` output, shared
        /// across rayon tasks.
        ///
        /// Parallelism is over weight rows, but a row's `batch_size` output
        /// slots (`out[b * rows + r]` for every `b`) interleave with every
        /// other row's, so they cannot be handed out as disjoint `&mut`
        /// chunks. Each task writes only the rows it owns, which keeps the
        /// writes race-free; this wrapper just carries the pointer across
        /// the `Send`/`Sync` boundary. Writing straight into the final
        /// layout kills what used to be here: a `[rows][batch]` staging vec
        /// (zeroed every call) plus a serial rows × batch transpose after
        /// the parallel section had already finished.
        #[derive(Clone, Copy)]
        struct BatchOut(*mut f32);
        unsafe impl Send for BatchOut {}
        unsafe impl Sync for BatchOut {}
        impl BatchOut {
            /// Safety: `idx` in bounds, and concurrent tasks never pass
            /// the same `idx` (they own disjoint row sets).
            #[inline]
            unsafe fn set(self, idx: usize, v: f32) {
                *self.0.add(idx) = v;
            }
        }

        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled()
                && matches!(
                    self,
                    WeightMatrix::Quantized { kind, .. } if Self::metal_kind_supported(*kind)
                )
            {
                if let Some(out) = self.apply_gpu_batch(x_batch, batch_size) {
                    return out;
                }
                // The kind is Metal-supported, so reaching here means a
                // launch failed and the batch degrades to `batch_size`
                // separate `apply` calls -- each its own command buffer,
                // commit and wait.
                crate::kernel_registry::miss(
                    crate::kernel_registry::Lookup::new(
                        crate::kernel_registry::Backend::Metal,
                        crate::kernel_registry::op::GEMM_PREFILL,
                        self.quant_kind(),
                    ),
                    "N x apply (one command buffer each)",
                );
                let rows = self.rows();
                let mut out = vec![0f32; batch_size * rows];
                for b in 0..batch_size {
                    let y = self.apply(&x_batch[b * cols..(b + 1) * cols]);
                    out[b * rows..(b + 1) * rows].copy_from_slice(&y);
                }
                return out;
            } else if metal_dense_enabled() {
                // Metal is on but this matrix has no Metal kernel at
                // all, so the whole GEMM runs on the CPU. For a
                // quantized weight that is the IQ4_XS shape exactly; for
                // an F32 one it is the documented host GEMM.
                let look = crate::kernel_registry::Lookup::new(
                    crate::kernel_registry::Backend::Metal,
                    crate::kernel_registry::op::GEMM_PREFILL,
                    self.quant_kind(),
                );
                if self.quant_kind().is_some() {
                    crate::kernel_registry::miss(look, "CPU apply_batch");
                } else {
                    crate::kernel_registry::miss_by_design(look, "CPU f32 GEMM");
                }
            }
        }

        // CUDA has a batched GEMM for every kind in
        // `frink_cuda::mul_mm::KINDS` (`cuda_mul_mm_kind_supported`),
        // and NO PART OF IT HAS RUN ON A GPU. Every other kind still
        // takes the per-position matvec loop below, which is the arm
        // that has -- for the six kinds that predate 2026-09-09.
        //
        // That loop is why this arm exists at all: without it a batched
        // prefill fell through to the CPU branch and never touched the
        // GPU -- measured on an RTX 4090, SmolLM2 `pp512` ran at 28
        // tok/s against llama.cpp's 57466. Per-position matvec is still
        // the wrong shape for a wide prefill, but it is the GPU rather
        // than 26 idle SMs, and the fallback now records a
        // `GEMM_PREFILL` miss instead of degrading silently.
        #[cfg(feature = "cuda")]
        {
            // The batched GEMM first, when the kind has one and the
            // batch is wide enough to pay for it. Below that threshold a
            // single token stays on the matvec kernels, which are the
            // arm that has actually run on a GPU.
            if cuda_dense_enabled() {
                if let WeightMatrix::Quantized { data, kind, .. } = self {
                    if cuda_mul_mm_kind_supported(*kind)
                        && frink_cuda::mul_mm::worth_a_gemm(batch_size)
                    {
                        let mm_kind = frink_cuda::mul_mm::kind_by_name(kind.name())
                            .expect("cuda_mul_mm_kind_supported agreed");
                        let row_bytes = self.block_bytes_per_row(*kind, cols);
                        match frink_cuda::mul_mm_launch::launch_mul_mm(
                            mm_kind,
                            data.as_slice(),
                            x_batch,
                            self.rows(),
                            cols,
                            batch_size,
                            row_bytes,
                        ) {
                            Ok(out) => return out,
                            Err(_) => {
                                // The kind HAS a GEMM, so reaching here is a
                                // launch failure rather than an unsupported
                                // kind, and the batch degrades to per-position
                                // matvecs. This call site used to be the one
                                // SILENT fallback in the registry's table.
                                crate::kernel_registry::miss(
                                    crate::kernel_registry::Lookup::new(
                                        crate::kernel_registry::Backend::Cuda,
                                        crate::kernel_registry::op::GEMM_PREFILL,
                                        self.quant_kind(),
                                    ),
                                    "N x matvec (the GEMM launch failed)",
                                );
                            }
                        }
                    }
                }
            }
            if cuda_dense_enabled()
                && matches!(self, WeightMatrix::Quantized { .. })
                && self.apply_gpu(&x_batch[..cols]).is_some()
            {
                let rows = self.rows();
                let mut out = vec![0f32; batch_size * rows];
                for b in 0..batch_size {
                    match self.apply_gpu(&x_batch[b * cols..(b + 1) * cols]) {
                        Some(y) => out[b * rows..(b + 1) * rows].copy_from_slice(&y),
                        None => {
                            let y = self.apply(&x_batch[b * cols..(b + 1) * cols]);
                            out[b * rows..(b + 1) * rows].copy_from_slice(&y);
                        }
                    }
                }
                return out;
            }
        }

        match self {
            WeightMatrix::F32(t) => {
                let xt = Tensor::new(x_batch.to_vec(), vec![batch_size, cols]);
                crate::matmul::matmul_f32(&xt, t).data
            }
            WeightMatrix::Quantized {
                data,
                rows,
                cols: _,
                kind,
            } => {
                let row_bytes = self.block_bytes_per_row(*kind, cols);
                // Written directly in the [batch, rows] layout the function
                // returns: each parallel task owns a disjoint set of rows
                // `r` and scatters `out[b * rows + r]` for every `b`
                // through `BatchOut`.
                let mut out = vec![0f32; batch_size * rows];
                let out_w = BatchOut(out.as_mut_ptr());

                // Prefill INT_DOT: quantize each activation once, then
                // reuse Q8 packs across all weight rows (llama CPU path).
                if cpu_int_dot_for(IntDotShape::BatchGemm) {
                    match *kind {
                        QuantKind::Q8_0 if cols.is_multiple_of(32) => {
                            let mut acts_owned = Vec::new();
                            let (acts, shared_tiles) =
                                Self::q8_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            let n_groups = *rows / frink_quant::Q8_0X4_NROWS;
                            if n_groups > 0 {
                                let packed = get_or_repack_q8x4(data, *rows, cols);
                                let nrows_g = frink_quant::Q8_0X4_NROWS;
                                let interleave = frink_quant::q8_0x4_interleave();
                                if frink_quant::q8_0x4_gemm_uses_acts_x4(interleave) {
                                    // i8mm: interleave each quad of
                                    // activations once per matmul (llama.cpp
                                    // `ggml_quantize_mat_q8_0_4x8` into
                                    // `wdata`); every row-group reuses it.
                                    let nc = frink_quant::Q8K_ACTS_X4_NC;
                                    let tiles_owned: Vec<frink_quant::Q8ActsX4>;
                                    let act_tiles: &[frink_quant::Q8ActsX4] =
                                        if shared_tiles.is_empty() {
                                            tiles_owned = acts
                                                .par_chunks(nc)
                                                .map(|chunk| {
                                                    frink_quant::prepare_q8_acts_x4(chunk, cols)
                                                })
                                                .collect();
                                            &tiles_owned
                                        } else {
                                            shared_tiles
                                        };
                                    // One runtime i8mm probe per matmul, not
                                    // one per (row-group x quad); see
                                    // `frink_quant::AccelX4`.
                                    let accel = frink_quant::AccelX4::detect();
                                    Self::par_chunked_groups(
                                        n_groups,
                                        nrows_g,
                                        act_tiles.len(),
                                        nc,
                                        |g, t0, t1| {
                                            let mut tmp = [0f32;
                                                frink_quant::Q8_0X4_NROWS
                                                    * frink_quant::Q8K_ACTS_X4_NC];
                                            for (t, tile) in act_tiles[t0..t1].iter().enumerate() {
                                                let t = t0 + t;
                                                let n = tile.na;
                                                let tmp = &mut tmp[..nrows_g * n];
                                                frink_quant::gemm_q8_0x4_group_x4_on(
                                                    &packed, g, tile, cols, interleave, accel, tmp,
                                                );
                                                for j in 0..n {
                                                    let col = (t * nc + j) * rows + g * nrows_g;
                                                    for r in 0..nrows_g {
                                                        unsafe {
                                                            out_w.set(col + r, tmp[r * n + j]);
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    );
                                } else {
                                    // GEMM, not a GEMV per position: the
                                    // batched kernel writes a `[row][batch]`
                                    // span, and the group's weight vectors
                                    // stay in registers across a tile of
                                    // activations. The span is then scattered
                                    // into the [batch][rows] output right
                                    // here, in parallel.
                                    let span = frink_quant::Q8_0X4_GEMM_NC;
                                    let n_tiles = batch_size.div_ceil(span);
                                    Self::par_chunked_groups(
                                        n_groups,
                                        nrows_g,
                                        n_tiles,
                                        span,
                                        |g, t0, t1| {
                                            let b0 = t0 * span;
                                            let b1 = (t1 * span).min(batch_size);
                                            let n = b1 - b0;
                                            let mut group = vec![0f32; nrows_g * n];
                                            frink_quant::gemm_q8_0x4_group(
                                                &packed,
                                                g,
                                                &acts[b0..b1],
                                                cols,
                                                interleave,
                                                &mut group,
                                            );
                                            for (bi, b) in (b0..b1).enumerate() {
                                                for r in 0..nrows_g {
                                                    unsafe {
                                                        out_w.set(
                                                            b * rows + g * nrows_g + r,
                                                            group[r * n + bi],
                                                        );
                                                    }
                                                }
                                            }
                                        },
                                    );
                                }
                                let data_slice = data.as_slice();
                                let tail = *rows - n_groups * frink_quant::Q8_0X4_NROWS;
                                crate::par::indices(tail, Self::min_rows_per_task(tail), |i| {
                                    let r = n_groups * frink_quant::Q8_0X4_NROWS + i;
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q8_0_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            } else {
                                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q8_0_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            }
                            return out;
                        }
                        QuantKind::Q4_0 if cols.is_multiple_of(32) => {
                            let mut acts_owned = Vec::new();
                            let (acts, shared_tiles) =
                                Self::q8_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            let n_groups = *rows / frink_quant::Q4_0X4_NROWS;
                            if n_groups > 0 {
                                let packed = get_or_repack_q4_0x4(data, *rows, cols);
                                let nrows_g = frink_quant::Q4_0X4_NROWS;
                                let interleave = frink_quant::q4_0x4_interleave();
                                if frink_quant::q4_0x4_gemm_uses_acts_x4(interleave) {
                                    // i8mm: same once-per-matmul activation
                                    // quad hoist as the Q8_0 arm above.
                                    let nc = frink_quant::Q8K_ACTS_X4_NC;
                                    let tiles_owned: Vec<frink_quant::Q8ActsX4>;
                                    let act_tiles: &[frink_quant::Q8ActsX4] =
                                        if shared_tiles.is_empty() {
                                            tiles_owned = acts
                                                .par_chunks(nc)
                                                .map(|chunk| {
                                                    frink_quant::prepare_q8_acts_x4(chunk, cols)
                                                })
                                                .collect();
                                            &tiles_owned
                                        } else {
                                            shared_tiles
                                        };
                                    let accel = frink_quant::AccelX4::detect();
                                    Self::par_chunked_groups(
                                        n_groups,
                                        nrows_g,
                                        act_tiles.len(),
                                        nc,
                                        |g, t0, t1| {
                                            let mut tmp = [0f32;
                                                frink_quant::Q4_0X4_NROWS
                                                    * frink_quant::Q8K_ACTS_X4_NC];
                                            for (t, tile) in act_tiles[t0..t1].iter().enumerate() {
                                                let t = t0 + t;
                                                let n = tile.na;
                                                let tmp = &mut tmp[..nrows_g * n];
                                                frink_quant::gemm_q4_0x4_group_x4_on(
                                                    &packed, g, tile, cols, interleave, accel, tmp,
                                                );
                                                for j in 0..n {
                                                    let col = (t * nc + j) * rows + g * nrows_g;
                                                    for r in 0..nrows_g {
                                                        unsafe {
                                                            out_w.set(col + r, tmp[r * n + j]);
                                                        }
                                                    }
                                                }
                                            }
                                        },
                                    );
                                } else {
                                    // GEMM, not a GEMV per position: the
                                    // batched kernel writes a `[row][batch]`
                                    // span, and the group's weight vectors
                                    // stay in registers across a tile of
                                    // activations. The span is then scattered
                                    // into the [batch][rows] output right
                                    // here, in parallel.
                                    let span = frink_quant::Q8_0X4_GEMM_NC;
                                    let n_tiles = batch_size.div_ceil(span);
                                    Self::par_chunked_groups(
                                        n_groups,
                                        nrows_g,
                                        n_tiles,
                                        span,
                                        |g, t0, t1| {
                                            let b0 = t0 * span;
                                            let b1 = (t1 * span).min(batch_size);
                                            let n = b1 - b0;
                                            let mut group = vec![0f32; nrows_g * n];
                                            frink_quant::gemm_q4_0x4_group(
                                                &packed,
                                                g,
                                                &acts[b0..b1],
                                                cols,
                                                interleave,
                                                &mut group,
                                            );
                                            for (bi, b) in (b0..b1).enumerate() {
                                                for r in 0..nrows_g {
                                                    unsafe {
                                                        out_w.set(
                                                            b * rows + g * nrows_g + r,
                                                            group[r * n + bi],
                                                        );
                                                    }
                                                }
                                            }
                                        },
                                    );
                                }
                                let data_slice = data.as_slice();
                                let tail = *rows - n_groups * frink_quant::Q4_0X4_NROWS;
                                crate::par::indices(tail, Self::min_rows_per_task(tail), |i| {
                                    let r = n_groups * frink_quant::Q4_0X4_NROWS + i;
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q4_0_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            } else {
                                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q4_0_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            }
                            return out;
                        }
                        QuantKind::Q4K if cols.is_multiple_of(256) => {
                            let mut acts_owned = Vec::new();
                            let (acts, shared_tiles) =
                                Self::q8k_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            let n_groups = *rows / frink_quant::Q4_KX8_NROWS;
                            if n_groups > 0 {
                                let interleave = frink_quant::q4_kx8_interleave();
                                let packed = get_or_repack_q4k(data, *rows, cols);
                                let nc = frink_quant::Q4_KX8_GEMM_NC;
                                // On the i8mm path, interleave each quad of
                                // activations once per matmul (llama.cpp
                                // `ggml_quantize_mat_q8_K_4x8` into `wdata`);
                                // the kernel used to redo it per row-group.
                                // A `shared` batch has already paid for this
                                // on behalf of every sibling projection. The
                                // predicate is asked first either way: it,
                                // not the donor, decides whether this matrix
                                // has an x4 kernel at all.
                                let tiles_owned: Vec<frink_quant::Q8KActsX4>;
                                let act_tiles: &[frink_quant::Q8KActsX4] =
                                    if !frink_quant::q4_kx8_gemm_uses_acts_x4(interleave) {
                                        &[]
                                    } else if !shared_tiles.is_empty() {
                                        shared_tiles
                                    } else {
                                        tiles_owned = acts
                                            .par_chunks(nc)
                                            .map(|chunk| {
                                                frink_quant::prepare_q8_k_acts_x4(chunk, cols)
                                            })
                                            .collect();
                                        &tiles_owned
                                    };
                                let accel = frink_quant::AccelX4::detect();
                                let n_tiles = batch_size.div_ceil(nc);
                                Self::par_chunked_groups(
                                    n_groups,
                                    frink_quant::Q4_KX8_NROWS,
                                    n_tiles,
                                    nc,
                                    |g, t0, t1| {
                                        let mut tile = [0f32;
                                            frink_quant::Q4_KX8_NROWS * frink_quant::Q4_KX8_GEMM_NC];
                                        for t in t0..t1 {
                                            let chunk =
                                                &acts[t * nc..((t + 1) * nc).min(batch_size)];
                                            let n = chunk.len();
                                            let tile = &mut tile[..frink_quant::Q4_KX8_NROWS * n];
                                            if act_tiles.is_empty() {
                                                frink_quant::gemm_q4_kx8_group(
                                                    &packed, g, chunk, cols, interleave, tile,
                                                );
                                            } else {
                                                frink_quant::gemm_q4_kx8_group_x4_on(
                                                    &packed,
                                                    g,
                                                    &act_tiles[t],
                                                    cols,
                                                    interleave,
                                                    accel,
                                                    tile,
                                                );
                                            }
                                            for j in 0..n {
                                                let col = (t * nc + j) * rows
                                                    + g * frink_quant::Q4_KX8_NROWS;
                                                for r in 0..frink_quant::Q4_KX8_NROWS {
                                                    unsafe {
                                                        out_w.set(col + r, tile[r * n + j]);
                                                    }
                                                }
                                            }
                                        }
                                    },
                                );
                                let data_slice = data.as_slice();
                                let tail = *rows - n_groups * frink_quant::Q4_KX8_NROWS;
                                crate::par::indices(tail, Self::min_rows_per_task(tail), |i| {
                                    let r = n_groups * frink_quant::Q4_KX8_NROWS + i;
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q4_k_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            } else {
                                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q4_k_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            }
                            return out;
                        }
                        QuantKind::Q5K if cols.is_multiple_of(256) => {
                            let mut acts_owned = Vec::new();
                            let (acts, shared_tiles) =
                                Self::q8k_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            // The Kx8 batch path: every aarch64 host (i8mm,
                            // dotprod, or the scalar Kx8 body, as the Q4_K arm
                            // takes it), and any other host whose `x4` GEMM
                            // has a SIMD kernel at this width (AVX2 since
                            // #159). This read `cfg!(target_arch = "aarch64")`
                            // alone until 2026-09-15, which sent every x86
                            // Q5_K prefill through the per-row GEMM below:
                            // 44.2 against llama.cpp's 378.7 tok/s on
                            // Llama-3.2-1B Q5_K_M (8.56x) on a Ryzen 5950X,
                            // beside Q4_K at 1.19x and Q6_K at 1.19x on the
                            // same host, because those two arms asked the
                            // kernels and this one asked the architecture.
                            let interleave = frink_quant::q5_kx8_interleave();
                            let use_kx8 = q5k_batch_takes_kx8(interleave);
                            let n_groups = if use_kx8 {
                                *rows / frink_quant::Q5_KX8_NROWS
                            } else {
                                0
                            };
                            if n_groups > 0 {
                                let packed = get_or_repack_q5k(data, *rows, cols);
                                let nc = frink_quant::Q5_KX8_GEMM_NC;
                                // On the i8mm path, interleave each quad of
                                // activations once per matmul; the kernel
                                // consumes it for every row-group. A `shared`
                                // batch has already paid for it. Predicate
                                // first, as in the Q4_K arm.
                                let tiles_owned: Vec<frink_quant::Q8KActsX4>;
                                let act_tiles: &[frink_quant::Q8KActsX4] =
                                    if !frink_quant::q5_kx8_gemm_uses_acts_x4(interleave) {
                                        &[]
                                    } else if !shared_tiles.is_empty() {
                                        shared_tiles
                                    } else {
                                        tiles_owned = acts
                                            .par_chunks(nc)
                                            .map(|chunk| {
                                                frink_quant::prepare_q8_k_acts_x4(chunk, cols)
                                            })
                                            .collect();
                                        &tiles_owned
                                    };
                                let accel = frink_quant::AccelX4::detect();
                                let n_tiles = batch_size.div_ceil(nc);
                                Self::par_chunked_groups(
                                    n_groups,
                                    frink_quant::Q5_KX8_NROWS,
                                    n_tiles,
                                    nc,
                                    |g, t0, t1| {
                                        let mut tile = [0f32;
                                            frink_quant::Q5_KX8_NROWS * frink_quant::Q5_KX8_GEMM_NC];
                                        for t in t0..t1 {
                                            let chunk =
                                                &acts[t * nc..((t + 1) * nc).min(batch_size)];
                                            let n = chunk.len();
                                            let tile = &mut tile[..frink_quant::Q5_KX8_NROWS * n];
                                            if act_tiles.is_empty() {
                                                frink_quant::gemm_q5_kx8_group(
                                                    &packed, g, chunk, cols, interleave, tile,
                                                );
                                            } else {
                                                frink_quant::gemm_q5_kx8_group_x4_on(
                                                    &packed,
                                                    g,
                                                    &act_tiles[t],
                                                    cols,
                                                    interleave,
                                                    accel,
                                                    tile,
                                                );
                                            }
                                            for j in 0..n {
                                                let col = (t * nc + j) * rows
                                                    + g * frink_quant::Q5_KX8_NROWS;
                                                for r in 0..frink_quant::Q5_KX8_NROWS {
                                                    unsafe {
                                                        out_w.set(col + r, tile[r * n + j]);
                                                    }
                                                }
                                            }
                                        }
                                    },
                                );
                                let data_slice = data.as_slice();
                                let tail = *rows - n_groups * frink_quant::Q5_KX8_NROWS;
                                crate::par::indices(tail, Self::min_rows_per_task(tail), |i| {
                                    let r = n_groups * frink_quant::Q5_KX8_NROWS + i;
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q5_k_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            } else {
                                let data_slice = data.as_slice();
                                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    let nc = frink_quant::Q5_K_GEMM_NC;
                                    for (t, chunk) in acts.chunks(nc).enumerate() {
                                        let n = chunk.len();
                                        let mut tmp = [0f32; frink_quant::Q5_K_GEMM_NC];
                                        frink_quant::gemm_q5_k_q8_row(row, chunk, &mut tmp[..n]);
                                        for (j, v) in tmp[..n].iter().enumerate() {
                                            unsafe {
                                                out_w.set((t * nc + j) * rows + r, *v);
                                            }
                                        }
                                    }
                                });
                            }
                            return out;
                        }
                        QuantKind::Q6K if cols.is_multiple_of(256) => {
                            let mut acts_owned = Vec::new();
                            let (acts, shared_tiles) =
                                Self::q8k_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            // Kx8 batch path only where the i8mm GEMM
                            // exists (the scalar Kx8 GEMM measured slower
                            // than the per-row NEON dot on Phi ffn_down,
                            // so everything else keeps the row path).
                            let interleave = frink_quant::q6_kx8_interleave();
                            let use_kx8 = frink_quant::q6_kx8_gemm_uses_acts_x4(interleave);
                            let n_groups = if use_kx8 {
                                *rows / frink_quant::Q6_KX8_NROWS
                            } else {
                                0
                            };
                            if n_groups > 0 {
                                let packed = get_or_repack_q6k(data, *rows, cols);
                                // Quads of 4 (the i8mm tile shape), not
                                // [`Q6_KX8_GEMM_NC`].
                                let nc = frink_quant::Q8K_ACTS_X4_NC;
                                let tiles_owned: Vec<frink_quant::Q8KActsX4>;
                                let act_tiles: &[frink_quant::Q8KActsX4] = if shared_tiles
                                    .is_empty()
                                {
                                    tiles_owned = acts
                                        .par_chunks(nc)
                                        .map(|chunk| frink_quant::prepare_q8_k_acts_x4(chunk, cols))
                                        .collect();
                                    &tiles_owned
                                } else {
                                    shared_tiles
                                };
                                let accel = frink_quant::AccelX4::detect();
                                let n_tiles = batch_size.div_ceil(nc);
                                Self::par_chunked_groups(
                                    n_groups,
                                    frink_quant::Q6_KX8_NROWS,
                                    n_tiles,
                                    nc,
                                    |g, t0, t1| {
                                        let mut tile = [0f32;
                                            frink_quant::Q6_KX8_NROWS * frink_quant::Q8K_ACTS_X4_NC];
                                        for t in t0..t1 {
                                            let chunk =
                                                &acts[t * nc..((t + 1) * nc).min(batch_size)];
                                            let n = chunk.len();
                                            let tile = &mut tile[..frink_quant::Q6_KX8_NROWS * n];
                                            frink_quant::gemm_q6_kx8_group_x4_on(
                                                &packed,
                                                g,
                                                &act_tiles[t],
                                                cols,
                                                interleave,
                                                accel,
                                                tile,
                                            );
                                            for j in 0..n {
                                                let col = (t * nc + j) * rows
                                                    + g * frink_quant::Q6_KX8_NROWS;
                                                for r in 0..frink_quant::Q6_KX8_NROWS {
                                                    unsafe {
                                                        out_w.set(col + r, tile[r * n + j]);
                                                    }
                                                }
                                            }
                                        }
                                    },
                                );
                                let data_slice = data.as_slice();
                                let tail = *rows - n_groups * frink_quant::Q6_KX8_NROWS;
                                crate::par::indices(tail, Self::min_rows_per_task(tail), |i| {
                                    let r = n_groups * frink_quant::Q6_KX8_NROWS + i;
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    for (b, act) in acts.iter().enumerate() {
                                        unsafe {
                                            out_w.set(
                                                b * rows + r,
                                                frink_quant::dot_q6_k_q8(row, act),
                                            );
                                        }
                                    }
                                });
                            } else {
                                let data_slice = data.as_slice();
                                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                    let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                    let nc = frink_quant::Q6_K_GEMM_NC;
                                    for (t, chunk) in acts.chunks(nc).enumerate() {
                                        let mut tmp = [0f32; frink_quant::Q6_K_GEMM_NC];
                                        let n = chunk.len();
                                        frink_quant::gemm_q6_k_q8_row(row, chunk, &mut tmp[..n]);
                                        for (j, v) in tmp[..n].iter().enumerate() {
                                            unsafe {
                                                out_w.set((t * nc + j) * rows + r, *v);
                                            }
                                        }
                                    }
                                });
                            }
                            return out;
                        }
                        // IQ4_XS: quantize the activations to Q8_K ONCE
                        // per matmul and run the int8 dot per (row,
                        // activation). No Kx8 tier, so the row's nibbles
                        // are still decoded per activation, as llama.cpp's
                        // own IQ4_XS prefill decodes them; what the f32
                        // fallback below paid on top was an f32 FMA per
                        // element and a per-activation f32 read of the
                        // row, measured 4.45x behind llama.cpp on a Ryzen
                        // 9 3900X (2026-09-15) where every K-quant on the
                        // same host was 1.0x to 1.4x.
                        QuantKind::IQ4XS if cols.is_multiple_of(256) => {
                            let mut acts_owned = Vec::new();
                            let (acts, _) =
                                Self::q8k_acts(shared, x_batch, batch_size, cols, &mut acts_owned);
                            let data_slice = data.as_slice();
                            crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                                let row = &data_slice[r * row_bytes..(r + 1) * row_bytes];
                                for (b, act) in acts.iter().enumerate() {
                                    unsafe {
                                        out_w.set(
                                            b * rows + r,
                                            frink_quant::dot_iq4_xs_q8_k(row, act),
                                        );
                                    }
                                }
                            });
                            return out;
                        }
                        QuantKind::Q5K | QuantKind::Q6K => {}
                        _ => {}
                    }
                }

                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                    let row = &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
                    for b in 0..batch_size {
                        let x = &x_batch[b * cols..(b + 1) * cols];
                        unsafe {
                            out_w.set(b * rows + r, Self::dot(*kind, row, x));
                        }
                    }
                });
                out
            }
            WeightMatrix::Mxfp4 {
                packed,
                scale,
                rows,
                cols: _,
            } => {
                let packed_row_bytes = cols / 2;
                let scale_row_bytes = cols / frink_quant::MXFP4_GROUP_SIZE;
                let mut out = vec![0f32; batch_size * rows];
                let out_w = BatchOut(out.as_mut_ptr());
                crate::par::indices(*rows, Self::min_rows_per_task(*rows), |r| {
                    let prow = &packed.as_slice()[r * packed_row_bytes..(r + 1) * packed_row_bytes];
                    let srow = &scale.as_slice()[r * scale_row_bytes..(r + 1) * scale_row_bytes];
                    for b in 0..batch_size {
                        let x = &x_batch[b * cols..(b + 1) * cols];
                        unsafe {
                            out_w.set(b * rows + r, frink_quant::dot_mxfp4_row_f32(prow, srow, x));
                        }
                    }
                });
                out
            }
            WeightMatrix::Adapted { .. } | WeightMatrix::Folded { .. } => {
                unreachable!("handled before dispatch")
            }
        }
    }

    /// Bytes actually resident in memory for this matrix -- the number
    /// that matters for "can this model's weights fit in RAM/VRAM at
    /// all," as opposed to the always-4x-larger f32-expanded size.
    pub fn resident_bytes(&self) -> usize {
        match self {
            WeightMatrix::F32(t) => t.len() * 4,
            WeightMatrix::Quantized { data, .. } => data.len(),
            WeightMatrix::Mxfp4 { packed, scale, .. } => packed.len() + scale.len(),
            WeightMatrix::Adapted { base, lora } => base.resident_bytes() + lora.resident_bytes(),
            WeightMatrix::Folded { base, .. } => base.resident_bytes(),
        }
    }

    /// Dispatches a single matvec through a real GPU kernel when a GPU
    /// feature is compiled in (`cuda` and/or `metal`) and this matrix
    /// is one of the five GPU-accelerated quant kinds (Q8_0, Q4_0,
    /// Q4_K, Q5_K, Q6_K). Returns `None` for every other case (no GPU
    /// feature, `F32`/`Mxfp4`/`Mxfp4Gguf`, or a `Quantized` kind other
    /// than the five below), so the caller falls back to `apply()` on
    /// the CPU -- this is a real dispatch decision
    /// (`frink_moe::run_expert_placed` uses it exactly this way), not
    /// a stub. Metal weight buffers are process-resident after the first
    /// upload (`frink_metal::gpu` weight cache); activations still
    /// upload per call. When both `cuda` and `metal` are enabled, CUDA
    /// is tried first and Metal is the fallback.
    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    pub fn apply_gpu(&self, x: &[f32]) -> Option<Vec<f32>> {
        assert_eq!(
            x.len(),
            self.cols(),
            "activation length must match matrix column count"
        );
        if let WeightMatrix::Adapted { base, lora } = self {
            let mut out = base.apply_gpu(x)?;
            lora.add_to(x, &mut out);
            return Some(out);
        }
        if let WeightMatrix::Folded { base, fold } = self {
            // The rotation goes into the matvec's own command buffer
            // when Metal can serve it, so the host neither runs the
            // butterfly nor ships a second vector. Every other backend
            // takes the host transform, which is the definition both
            // sides are checked against.
            #[cfg(feature = "metal")]
            if metal_dense_enabled() {
                if let Some(plan) = fold.metal_plan(x.len()) {
                    if let Some(out) = base.apply_gpu_folded(x, Some(&plan)) {
                        return Some(out);
                    }
                }
            }
            return base.apply_gpu(&fold.transform_input(x));
        }

        // F32 stays on CPU in apply_gpu: a lone small router matvec is
        // faster as host GEMV than a Metal sync. F32 Metal launches are
        // used when fused into MoE resident decode (encode_matvec).
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            // Deliberate, and recorded rather than hidden: an MoE
            // router is a lone small F32 matvec that costs more to ship
            // to the GPU than to compute on the host.
            let backend = active_backend();
            if backend.is_accelerator() {
                crate::kernel_registry::miss_by_design(
                    crate::kernel_registry::Lookup::new(
                        backend,
                        crate::kernel_registry::op::MATVEC,
                        None,
                    ),
                    "host GEMV",
                );
            }
            return None;
        };
        let row_bytes = self.block_bytes_per_row(*kind, *cols);

        // One body per backend, expanded over the one ordered list, in
        // place of the two hand-kept `match kind` tables this used to
        // hold -- which differed in arity, in error type, and (silently)
        // by one entry. A third backend adds no code here.
        #[allow(unused_macros)]
        macro_rules! try_matvec {
            ($b:ty) => {
                if let Some(result) = <$b as BackendDispatch>::launch_matvec(
                    *kind,
                    data.as_slice(),
                    x,
                    *rows,
                    row_bytes,
                ) {
                    match result {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!(
                                "frink: {} matvec dispatch failed, {}: {e}",
                                <$b as BackendCaps>::NAME,
                                <$b as BackendDispatch>::MATVEC_FALLBACK
                            );
                        }
                    }
                }
            };
        }
        with_gpu_backends!(try_matvec);

        // Reached only on a miss or a launch error, i.e. only when the
        // caller is about to run the whole matvec on the host anyway --
        // so recording it here costs nothing measurable and is the only
        // signal that a GPU run is quietly not one.
        let backend = active_backend();
        if backend.is_accelerator() {
            crate::kernel_registry::miss(
                crate::kernel_registry::Lookup::new(
                    backend,
                    crate::kernel_registry::op::MATVEC,
                    Some(*kind),
                ),
                "CPU apply_cpu",
            );
        }
        None
    }

    /// Runs several independent matvecs that share the same activation
    /// `x` in one GPU dispatch (one upload of `x`, one wait). Tries
    /// CUDA first (when `cuda_dense_enabled()`), then Metal (when
    /// `metal_dense_enabled()`). Intended for Q/K/V (and similar)
    /// projections. Returns `None` if no GPU backend is enabled, any
    /// matrix lacks a GPU kernel, or all fused launches fail — caller
    /// should fall back to sequential [`Self::apply`].
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn apply_gpu_multi(mats: &[&WeightMatrix], x: &[f32]) -> Option<Vec<Vec<f32>>> {
        Self::apply_gpu_multi_folded(mats, x, None)
    }

    /// One matrix through the fused path, with an optional device-side
    /// Hadamard prologue. The single-matrix caller of
    /// [`Self::apply_gpu_multi_folded`], so a folded projection and a
    /// folded q/k/v triple take the same launch.
    #[cfg(feature = "metal")]
    pub(crate) fn apply_gpu_folded(
        &self,
        x: &[f32],
        fold: Option<&frink_metal::hadamard::FoldPlan<'_>>,
    ) -> Option<Vec<f32>> {
        Self::apply_gpu_multi_folded(&[self], x, fold).map(|mut outs| outs.pop().unwrap())
    }

    /// [`Self::apply_gpu_multi`] with PrismML's folded rotation applied
    /// to `x` on the device before the matvecs read it.
    ///
    /// `fold` is Metal-only: the rotation kernel is
    /// `frink_metal::hadamard`, and a CUDA or Vulkan caller takes the
    /// host butterfly rather than a rotation nothing there implements.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(crate) fn apply_gpu_multi_folded(
        mats: &[&WeightMatrix],
        x: &[f32],
        #[cfg(feature = "metal")] fold: Option<&frink_metal::hadamard::FoldPlan<'_>>,
        #[cfg(not(feature = "metal"))] fold: Option<&()>,
    ) -> Option<Vec<Vec<f32>>> {
        // Off Metal there is no rotation kernel, so `None` is the only
        // value this can take and nothing below reads it. Bound here
        // rather than renamed, because the parameter is part of one
        // signature both builds call.
        #[cfg(not(feature = "metal"))]
        let _ = &fold;
        if mats.is_empty() {
            return None;
        }
        assert_eq!(
            x.len(),
            mats[0].cols(),
            "activation length must match matrix column count"
        );
        if mats.iter().any(|m| m.lora().is_some()) {
            // The fused launch runs over the bases; each adapter's
            // delta is added to its own output on the host.
            let bases: Vec<&WeightMatrix> = mats.iter().map(|m| m.base()).collect();
            let mut outs = Self::apply_gpu_multi(&bases, x)?;
            for (m, out) in mats.iter().zip(outs.iter_mut()) {
                if let Some(lora) = m.lora() {
                    lora.add_to(x, out);
                }
            }
            return Some(outs);
        }
        if let Some(fold) = mats[0].hadamard_fold() {
            // One fold shared by every matrix (q/k/v split from one folded
            // fused projection): transform once, launch the bases.
            // Different folds, or a fold on only some, is not a shared
            // activation any more, and the caller runs them one by one.
            if !mats.iter().all(|m| {
                m.hadamard_fold()
                    .is_some_and(|f| std::sync::Arc::ptr_eq(f, fold))
            }) {
                return None;
            }
            let bases: Vec<&WeightMatrix> = mats
                .iter()
                .map(|m| match m {
                    WeightMatrix::Folded { base, .. } => base.as_ref(),
                    _ => unreachable!("every matrix carries the fold"),
                })
                .collect();
            let xt = fold.transform_input(x);
            return Self::apply_gpu_multi(&bases, &xt);
        }
        if mats.iter().any(|m| m.hadamard_fold().is_some()) {
            return None;
        }

        // Try CUDA first if enabled.
        #[cfg(feature = "cuda")]
        if cuda_dense_enabled() {
            let mut launches = Vec::with_capacity(mats.len());
            for m in mats {
                assert_eq!(m.cols(), mats[0].cols());
                let WeightMatrix::Quantized {
                    data,
                    rows,
                    cols,
                    kind,
                } = m
                else {
                    return None;
                };
                // One table, in `frink-cuda`, exactly as the Metal arm
                // below asks `matvec_launch_meta`. This match was
                // written out here and again in
                // `apply_gpu_dense_ffn_swiglu`, three copies of one
                // five-row list with nothing holding them together --
                // and a kind added to the capability table but not to a
                // copy loses its fused launch silently, which is the
                // failure this file has paid for twice.
                let (kernel_src, module_name, fn_name) =
                    frink_cuda::gpu::matvec_launch_meta(kind.name())?;
                let row_bytes = m.block_bytes_per_row(*kind, *cols);
                let n_blocks_per_row = row_bytes / Self::block_bytes_for_kind(*kind);
                launches.push(frink_cuda::gpu::MatvecLaunch {
                    kernel_src,
                    module_name,
                    fn_name,
                    // Borrow mmap/owned storage — never to_vec() (breaks
                    // resident_cuda_weights pointer cache; re-uploads GB).
                    weights: data.as_slice(),
                    rows: *rows,
                    row_bytes,
                    n_blocks_per_row,
                });
            }
            match frink_cuda::gpu::launch_matvec_multi(x, &launches) {
                Ok(outs) => return Some(outs),
                Err(e) => {
                    eprintln!("frink: CUDA multi-matvec failed, trying next backend: {e}");
                }
            }
        }

        // Try Metal if CUDA didn't return or failed.
        #[cfg(feature = "metal")]
        if metal_dense_enabled() {
            let mut launches = Vec::with_capacity(mats.len());
            let mut held: Vec<(&[u8], usize, usize, &'static str)> = Vec::with_capacity(mats.len());
            for m in mats {
                assert_eq!(m.cols(), mats[0].cols());
                let WeightMatrix::Quantized {
                    data,
                    rows,
                    cols,
                    kind,
                } = m
                else {
                    return None;
                };
                // The one Metal kind table (`Metal::matvec_kernel`).
                // This was a hand-written six-row match that lacked
                // Q5_0 and PTQ1_0, so every gate/up and q/k/v pair of
                // those kinds ran as separate launches: correct, and
                // one GPU round trip per layer slower than the single
                // path for the kinds the copy happened to name.
                let kind_name = Metal::matvec_kernel(*kind)?;
                let row_bytes = m.block_bytes_per_row(*kind, *cols);
                held.push((data.as_slice(), *rows, row_bytes, kind_name));
            }
            for (weights, rows, row_bytes, kind_name) in &held {
                let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                    frink_metal::gpu::matvec_launch_meta(kind_name)?;
                launches.push(frink_metal::gpu::MatvecLaunch {
                    kernel_src: src,
                    fn_name,
                    block_bytes,
                    block_elems,
                    weights,
                    rows: *rows,
                    row_bytes: *row_bytes,
                    rows_per_tg,
                });
            }
            match frink_metal::gpu::launch_matvec_fused_folded(
                x,
                &launches,
                frink_metal::gpu::MatvecEpilogue::default(),
                fold,
            ) {
                Ok(outs) => return Some(outs),
                Err(e) => {
                    eprintln!("frink: Metal fused matvec failed, falling back to CPU: {e}");
                }
            }
        }

        None
    }

    /// Dense SwiGLU FFN on GPU with device-resident activations:
    /// one upload of `x`, gate+up+silu×up+down on device, one download.
    /// Tries CUDA first when enabled, then Metal. Returns `None` if
    /// no GPU path applies — caller falls back to [`Self::apply`] /
    /// multi-matvec.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub fn apply_gpu_dense_ffn_swiglu(
        gate: &WeightMatrix,
        up: &WeightMatrix,
        down: &WeightMatrix,
        x: &[f32],
    ) -> Option<Vec<f32>> {
        #[cfg(feature = "cuda")]
        {
            if cuda_dense_enabled() {
                fn cuda_launch(m: &WeightMatrix) -> Option<frink_cuda::gpu::MatvecLaunch<'_>> {
                    let WeightMatrix::Quantized {
                        data,
                        rows,
                        cols,
                        kind,
                    } = m
                    else {
                        return None;
                    };
                    // The second of the two copies this used to hold.
                    // See the note in `apply_gpu_multi`.
                    let (kernel_src, module_name, fn_name) =
                        frink_cuda::gpu::matvec_launch_meta(kind.name())?;
                    let row_bytes = m.block_bytes_per_row(*kind, *cols);
                    let n_blocks_per_row = row_bytes / WeightMatrix::block_bytes_for_kind(*kind);
                    Some(frink_cuda::gpu::MatvecLaunch {
                        kernel_src,
                        module_name,
                        fn_name,
                        weights: data.as_slice(),
                        rows: *rows,
                        row_bytes,
                        n_blocks_per_row,
                    })
                }
                if let (Some(g), Some(u), Some(d)) =
                    (cuda_launch(gate), cuda_launch(up), cuda_launch(down))
                {
                    assert_eq!(gate.cols(), x.len());
                    assert_eq!(up.cols(), x.len());
                    assert_eq!(down.cols(), gate.rows());
                    match frink_cuda::gpu::launch_dense_ffn_swiglu(&g, &u, &d, x) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!("frink: CUDA dense FFN fuse failed, trying next: {e}");
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        {
            if metal_dense_enabled() {
                fn metal_launch(m: &WeightMatrix) -> Option<frink_metal::gpu::MatvecLaunch<'_>> {
                    let WeightMatrix::Quantized {
                        data,
                        rows,
                        cols: _,
                        kind,
                        // Through the fold, which travels as a plan beside
                        // the launch rather than as a transformed vector.
                        // `base()` unwraps a LoRA only, deliberately: an
                        // adapter is arithmetic this kernel cannot do.
                    } = (match m {
                        WeightMatrix::Folded { base, .. } => base.as_ref(),
                        other => other,
                    })
                    else {
                        return None;
                    };
                    // The one Metal kind table again: this was the
                    // THIRD hand-written copy of it, and it lacked
                    // Q5_0 and PTQ1_0, so those kinds ran their FFN as
                    // three submissions with a host SwiGLU between.
                    let kind_name = Metal::matvec_kernel(*kind)?;
                    let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
                        frink_metal::gpu::matvec_launch_meta(kind_name)?;
                    // A zero-row matrix has no rows to stride over, so
                    // there is no meaningful row size; `checked_div`
                    // says that once instead of splitting it across a
                    // guard and a bare division.
                    let row_bytes = data.as_slice().len().checked_div(*rows).unwrap_or(0);
                    Some(frink_metal::gpu::MatvecLaunch {
                        kernel_src: src,
                        fn_name,
                        block_bytes,
                        block_elems,
                        weights: data.as_slice(),
                        rows: *rows,
                        row_bytes,
                        rows_per_tg,
                    })
                }
                // A folded FFN reads TWO rotated activations: `x`
                // before gate/up, and the SwiGLU output before `down`.
                // They are different widths and therefore different
                // folds, so both travel with the launch rather than
                // one standing in for the other.
                let fold_x = gate.hadamard_fold().and_then(|f| f.metal_plan(x.len()));
                let fold_act = down.hadamard_fold().and_then(|f| f.metal_plan(gate.rows()));
                let folds_agree = match (gate.hadamard_fold(), up.hadamard_fold()) {
                    (Some(g), Some(u)) => std::sync::Arc::ptr_eq(g, u) && fold_x.is_some(),
                    (None, None) => true,
                    _ => false,
                } && (down.hadamard_fold().is_none() == fold_act.is_none());
                if let (Some(g), Some(u), Some(d), true) = (
                    metal_launch(gate),
                    metal_launch(up),
                    metal_launch(down),
                    folds_agree,
                ) {
                    assert_eq!(gate.cols(), x.len());
                    assert_eq!(up.cols(), x.len());
                    assert_eq!(down.cols(), gate.rows());
                    match frink_metal::gpu::launch_dense_ffn_swiglu_folded(
                        &g,
                        &u,
                        &d,
                        x,
                        fold_x.as_ref(),
                        fold_act.as_ref(),
                    ) {
                        Ok(out) => return Some(out),
                        Err(e) => {
                            eprintln!("frink: Metal dense FFN fuse failed, falling back: {e}");
                        }
                    }
                }
            }
        }
        None
    }

    /// Runs one weight matrix against `batch_size` activations in a
    /// single Metal command buffer (shared resident weights, one
    /// upload of `x_batch`, one GPU wait). `x_batch` / return layout
    /// match [`Self::apply_batch`]: `[batch, cols]` → `[batch, rows]`.
    /// Returns `None` if Metal dense is off, the kind lacks a Metal
    /// kernel, or the launch fails.
    ///
    /// `batch_size >= 4` takes the weight-reuse `mul_mm` path where the
    /// kind has one; everything else falls through to
    /// [`frink_metal::gpu::launch_matvec_batch`].
    #[cfg(feature = "metal")]
    pub fn apply_gpu_batch(&self, x_batch: &[f32], batch_size: usize) -> Option<Vec<f32>> {
        if !metal_dense_enabled() || batch_size == 0 {
            return None;
        }
        if let WeightMatrix::Adapted { base, lora } = self {
            let mut out = base.apply_gpu_batch(x_batch, batch_size)?;
            lora.add_batch_to(x_batch, batch_size, &mut out);
            return Some(out);
        }
        if let WeightMatrix::Folded { base, fold } = self {
            // A BATCH keeps the host rotation, and that is a
            // measurement rather than an omission: the same device
            // prologue that pays for itself on a decode token
            // (`apply_gpu`) cost 6% of Bonsai prefill, because
            // `transform_rows` is already parallel across cores and the
            // kernel serialises into the GEMM's own command buffer
            // instead. `pp128` 33.96 -> 32.04 tok/s on an M2 Pro, three
            // reps each, which is why `apply_gpu_batch` takes no fold.
            let transformed = fold.transform_rows(x_batch, batch_size);
            return base.apply_gpu_batch(&transformed, batch_size);
        }
        let WeightMatrix::Quantized {
            data,
            rows,
            cols,
            kind,
        } = self
        else {
            return None;
        };
        let Some(kind_name) = metal_matvec_kind_name(*kind) else {
            crate::kernel_registry::miss(
                crate::kernel_registry::Lookup::new(
                    crate::kernel_registry::Backend::Metal,
                    crate::kernel_registry::op::GEMM_PREFILL,
                    Some(*kind),
                ),
                "CPU apply_batch",
            );
            return None;
        };
        let (src, fn_name, block_bytes, block_elems, rows_per_tg) =
            frink_metal::gpu::matvec_launch_meta(kind_name)?;
        let row_bytes = self.block_bytes_per_row(*kind, *cols);
        // Weight-reuse simdgroup GEMM for prefill batch >= 4: each 64x32
        // output tile reads its weight slice once into threadgroup
        // memory instead of once per token. The batched matvec below is
        // the fallback it replaces -- correct, but it re-reads the whole
        // matrix for every token, which is why Metal `pp512` was 14-99x
        // behind llama.cpp. Threshold 4 (was 8) covers shorter prompts
        // without changing the decode path (batch_size == 1 is matvec).
        //
        // ONE launcher keyed by the kind's name, read off the same table
        // `gemm_supported` is derived from. This was a seven-arm match
        // naming each kind's wrapper, and the arms lagged the table:
        // Q5_0 and PTQ1_0 had a GEMM the table admitted and the match
        // did not reach, so their prefill ran N matvecs while the kernel
        // registry recorded a GEMM hit.
        if batch_size >= 4 {
            if !metal_mul_mm_kind_supported(*kind) {
                // Observation only: a kind with a matvec kernel but no
                // simdgroup GEMM still runs on Metal, as `batch`
                // separate matvecs over the same weights. That is the
                // shape that cost IQ4_XS 13.7x, and it is invisible in
                // the output.
                crate::kernel_registry::miss(
                    crate::kernel_registry::Lookup::new(
                        crate::kernel_registry::Backend::Metal,
                        crate::kernel_registry::op::GEMM_PREFILL,
                        Some(*kind),
                    ),
                    "Metal N x matvec batch",
                );
            }
            match frink_metal::gpu::launch_mul_mm_sg(
                kind.name(),
                data.as_slice(),
                x_batch,
                *rows,
                row_bytes,
                batch_size,
            ) {
                Ok(Some(out)) => return Some(out),
                Ok(None) => {}
                Err(e) => {
                    eprintln!(
                        "frink: Metal {} simdgroup mul_mm failed, batched fallback: {e}",
                        kind.name()
                    );
                }
            }
            // The older non-simdgroup GEMMs, kept for the two kinds that
            // have one, between the simdgroup kernel and the matvec.
            let older = match kind {
                QuantKind::Q4_0 => Some(frink_metal::gpu::launch_q4_0_mul_mm(
                    data.as_slice(),
                    x_batch,
                    *rows,
                    row_bytes,
                    batch_size,
                )),
                QuantKind::Q4K => Some(frink_metal::gpu::launch_q4_k_mul_mm(
                    data.as_slice(),
                    x_batch,
                    *rows,
                    row_bytes,
                    batch_size,
                )),
                _ => None,
            };
            match older {
                Some(Ok(out)) => return Some(out),
                Some(Err(e)) => {
                    eprintln!(
                        "frink: Metal {} mul_mm failed, matvec fallback: {e}",
                        kind.name()
                    );
                }
                None => {}
            }
        }
        let launch = frink_metal::gpu::MatvecLaunch {
            kernel_src: src,
            fn_name,
            block_bytes,
            block_elems,
            weights: data.as_slice(),
            rows: *rows,
            row_bytes,
            rows_per_tg,
        };
        match frink_metal::gpu::launch_matvec_batch(&launch, x_batch, batch_size) {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("frink: Metal batch matvec failed, falling back: {e}");
                None
            }
        }
    }

    /// Delegates to [`metal_matvec_kind_name`]. Kept as a method because
    /// the call sites read better, but it must never grow a list of its
    /// own again — a second copy of this list is what sent IQ4_XS
    /// batched prefill to the CPU.
    #[cfg(feature = "metal")]
    fn metal_kind_supported(kind: QuantKind) -> bool {
        metal_matvec_kind_name(kind).is_some()
    }

    /// Eagerly resolve, and record, every kernel lookup this matrix's
    /// dispatch paths will make later, without dispatching anything.
    ///
    /// Call once per weight while the model is being built, with `role`
    /// naming the tensor (`"attn_q"`, `"ffn_down"`, ...). The predicates
    /// consulted here are the *same functions* the hot path consults, so
    /// the recorded prediction cannot drift from the decision. See
    /// [`crate::kernel_registry`] for why this exists and
    /// [`crate::kernel_registry::seal`] for what is done with it.
    ///
    /// Observation only: nothing here influences a later dispatch.
    #[track_caller]
    pub fn probe_kernels(&self, role: &'static str) {
        if !crate::kernel_registry::enabled() {
            return;
        }
        self.probe_kernels_into(
            crate::kernel_registry::global(),
            role,
            std::panic::Location::caller(),
        );
    }

    /// [`Self::probe_kernels`] against an explicit registry and call
    /// site, so tests can probe into an instance of their own instead of
    /// the process-wide one.
    pub fn probe_kernels_into(
        &self,
        reg: &crate::kernel_registry::Registry,
        role: &'static str,
        loc: &'static std::panic::Location<'static>,
    ) {
        self.probe_kernels_for(reg, active_backend(), role, loc)
    }

    /// [`Self::probe_kernels_into`] against an explicit backend rather
    /// than [`active_backend`]. Lets a test on a CPU-only build ask what
    /// a Metal or CUDA run would resolve -- which is the only way the
    /// kernel-coverage tests can run under plain
    /// `cargo test --workspace`, where every GPU feature is off.
    pub fn probe_kernels_for(
        &self,
        reg: &crate::kernel_registry::Registry,
        backend: crate::kernel_registry::Backend,
        role: &'static str,
        loc: &'static std::panic::Location<'static>,
    ) {
        use crate::kernel_registry::{op, Backend, Lookup, Outcome};

        let kind = self.quant_kind();
        let cols = self.cols();
        let look = |op: &'static str| Lookup {
            backend,
            op,
            role,
            kind,
        };

        // Whether the accelerator, if one is selected, can run this
        // matrix at all -- and if so, whether prefill gets a real GEMM
        // or `batch` matvecs over the same weights.
        //
        // Read off `BackendCaps` over the ungated backend table rather
        // than from a `match backend` written out here. The two are
        // NOT interchangeable: the hand-written match had a `Backend::
        // Cpu => (false, false)` arm and no `_`, so it was exhaustive
        // by luck -- a third variant broke it, which is the good case.
        // A fourth backend added while a `_` arm existed would have
        // silently reported "no kernels" for a backend that had them.
        //
        // Ungated on purpose: a CPU-only build must be able to ask what
        // CUDA would resolve, which is what every kernel-coverage test
        // below does. `BackendDispatch` is unavailable here for exactly
        // that reason.
        //
        // The predicate sets are per backend and genuinely different:
        // CUDA has a batched GEMM for a SUBSET of the kinds it has
        // matvecs for (Q8_0, Q4_0) and decomposes the rest into
        // per-position matvecs; Vulkan has one matvec and no GEMM at
        // all. `GEMM_FALLBACK` is what each of those decompositions is
        // actually called.
        let (matvec, gemm, gemm_fallback) = {
            let mut found = (false, false, "");
            macro_rules! caps_of {
                ($b:ty) => {
                    if backend == <$b as BackendCaps>::ID {
                        found = (
                            kind.is_some_and(|k| <$b as BackendCaps>::matvec_kernel(k).is_some()),
                            kind.is_some_and(<$b as BackendCaps>::gemm_supported),
                            <$b as BackendCaps>::GEMM_FALLBACK,
                        );
                    }
                };
            }
            with_gpu_backend_caps!(caps_of);
            found
        };

        if backend.is_accelerator() {
            reg.record_build_at(
                loc,
                look(op::MATVEC),
                match kind {
                    // An accelerator kernel exists for this format.
                    _ if matvec => Outcome::Hit,
                    // No kernel: the whole matvec runs on the host.
                    Some(_) => Outcome::slow_path("CPU apply_cpu"),
                    // F32 has no quantized kernel by construction, and a
                    // lone small F32 matvec (an MoE router) is host work
                    // on purpose -- see `apply_gpu`.
                    None => Outcome::by_design("host GEMV"),
                },
            );
            reg.record_build_at(
                loc,
                look(op::GEMM_PREFILL),
                match (gemm, matvec, kind) {
                    (true, ..) => Outcome::Hit,
                    // A matvec but no GEMM. What that costs is per
                    // backend -- Metal re-reads the whole weight matrix
                    // once per position but stays on the GPU (the 13.7x
                    // shape), CUDA does the same through a different
                    // entry point, and Vulkan has no batch path at all
                    // so the prefill lands on the host -- so the name
                    // comes from the backend instead of from an arm
                    // here that a new variant would fall through.
                    (false, true, _) => Outcome::slow_path(gemm_fallback),
                    (false, false, Some(_)) => Outcome::slow_path("CPU apply_batch"),
                    (false, false, None) => Outcome::by_design("CPU f32 GEMM"),
                },
            );
        }

        // The host path is what every accelerator miss lands on, so
        // record its tier too: integer vec_dot, or the much slower f32
        // dequant-dot.
        if !matvec || !gemm {
            let int_dot = cpu_int_dot_for(IntDotShape::Matvec)
                && kind.is_some_and(|k| cpu_int_dot_kind_supported(k, cols));
            reg.record_build_at(
                loc,
                Lookup {
                    backend: Backend::Cpu,
                    op: op::MATVEC,
                    role,
                    kind,
                },
                match kind {
                    _ if int_dot => Outcome::Hit,
                    // A quantized weight with no integer vec_dot kernel
                    // dequantizes to f32 first: a much slower engine,
                    // and invisible in the output.
                    Some(_) => Outcome::slow_path("f32 dequant-dot"),
                    None => Outcome::by_design("f32 GEMM"),
                },
            );
        }
    }

    /// The block size (in bytes) for exactly the quant kinds
    /// `apply_gpu` dispatches to a real CUDA or Vulkan kernel for -- a
    /// small, deliberately partial mirror of `block_bytes_per_row`'s
    /// per-kind match.
    ///
    /// Partial means this `unreachable!()` is reachable by a mistake:
    /// widening `Cuda::matvec_kernel` without adding the row here
    /// turns a decode into a panic in a rayon worker rather than a
    /// fallback. `every_cuda_or_vulkan_matvec_kind_has_a_block_size`
    /// calls it for every claimed kind so that lands as a red test
    /// instead.
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    pub(crate) fn block_bytes_for_kind(kind: QuantKind) -> usize {
        match kind {
            QuantKind::Q8_0 => frink_quant::Q8_0_BLOCK_BYTES,
            QuantKind::Q4_0 => frink_quant::Q4_0_BLOCK_BYTES,
            QuantKind::Q5_0 => frink_quant::Q5_0_BLOCK_BYTES,
            QuantKind::Q4K => frink_quant::Q4_K_BLOCK_BYTES,
            QuantKind::Q5K => frink_quant::Q5_K_BLOCK_BYTES,
            QuantKind::Q6K => frink_quant::Q6_K_BLOCK_BYTES,
            QuantKind::Q2K => frink_quant::Q2_K_BLOCK_BYTES,
            QuantKind::Q3K => frink_quant::Q3_K_BLOCK_BYTES,
            QuantKind::IQ4NL => frink_quant::IQ4_NL_BLOCK_BYTES,
            QuantKind::IQ4XS => frink_quant::IQ4_XS_BLOCK_BYTES,
            QuantKind::Mxfp4Gguf => frink_quant::MXFP4_GGUF_BLOCK_BYTES,
            _ => unreachable!(
                "apply_gpu only calls this for the CUDA/Vulkan-dispatchable kinds, not {kind:?}"
            ),
        }
    }
}
#[cfg(test)]
mod tests {

    /// The task floor is **work-aware**, which is the whole reason
    /// [`crate::par::with_op_work`] exists: a row count alone cannot
    /// tell a 64-wide matrix from a 256-wide one, and rayon splitting
    /// the narrow one by rows alone is the measured 13-16x small-model
    /// regression.
    ///
    /// Both shapes here sit under [`crate::par::policy::SPIN_MIN_OP_MACS`]
    /// so both are decided by the fork-join arm, which is the only arm
    /// that reads a `min_len` at all.
    ///
    /// Sabotage: drop the `MIN_TASK_MACS` term from `min_rows_per_task`
    /// and this goes red, because both shapes then collapse onto the
    /// same row-count floor.
    #[test]
    fn the_task_floor_demands_more_rows_of_a_narrower_matrix() {
        if crate::par::policy::pinned().is_some() {
            return; // pinned: not the arm this floor belongs to
        }
        let rows = 4096usize;
        let narrow = crate::par::with_op_work(rows, 64, || WeightMatrix::min_rows_per_task(rows));
        let wider = crate::par::with_op_work(rows, 256, || WeightMatrix::min_rows_per_task(rows));
        assert_eq!(narrow, MIN_TASK_MACS.div_ceil(64));
        assert!(
            narrow > wider,
            "a 64-wide row carries a quarter of a 256-wide row's work, so a \
             task must hold four times as many of them: {narrow} vs {wider}"
        );
    }

    /// The four dtypes the drifted copies were missing.
    ///
    /// Three of the six loaders stopped at IQ1_M, so `IQ1_S`,
    /// `IQ2_XXS`, `IQ3_XXS` and `MXFP4` mapped to `None` there -- and a
    /// `None` is `LoadError::UnsupportedDtype`, not a slower path. A
    /// DeepSeek-MLA checkpoint at `IQ2_XXS`, an ordinary quant for a
    /// model that size, was refused outright while the same quant
    /// loaded on the generic path. One table is what stops that
    /// recurring.
    #[test]
    fn the_four_dtypes_the_duplicated_tables_disagreed_about_all_map() {
        assert_eq!(quant_kind_for(GgmlType::IQ1S), Some(QuantKind::IQ1S));
        assert_eq!(quant_kind_for(GgmlType::IQ2XXS), Some(QuantKind::IQ2XXS));
        assert_eq!(quant_kind_for(GgmlType::IQ3XXS), Some(QuantKind::IQ3XXS));
        assert_eq!(quant_kind_for(GgmlType::MXFP4), Some(QuantKind::Mxfp4Gguf));
    }

    /// Every dtype with a CPU dequant kernel must be reachable through
    /// this map, or the kernel exists and no loader can ever hand it a
    /// tensor. Checked against the two backend tables rather than a
    /// hand-written list, so adding a kernel without a mapping fails
    /// here instead of at a user's load.
    #[test]
    fn every_dtype_with_a_gpu_kernel_is_reachable_through_the_map() {
        let mapped: Vec<QuantKind> = [
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::IQ4XS,
        ]
        .into_iter()
        .map(|d| quant_kind_for(d).expect("a dtype with a GPU kernel must map"))
        .collect();
        for kind in mapped {
            assert!(
                metal_mul_mm_kind_supported(kind) || cuda_matvec_kind_supported(kind),
                "{kind:?} was listed as having a GPU kernel"
            );
        }
    }

    /// The CUDA capability predicates and the *launch* table must name
    /// the same set, for every kind.
    ///
    /// `Cuda::matvec_kernel` and `Cuda::gemm_supported` are DERIVED
    /// from `frink-cuda`'s own kernel tables now, so the two pairs
    /// that used to be checked here cannot disagree -- those tests were
    /// deleted rather than left comparing a table to itself, which
    /// reads as coverage and is not.
    ///
    /// This one still matters. [`cuda_matvec_launch`] is a table of
    /// FUNCTION POINTERS, which only exist under `--features cuda`, so
    /// it cannot be derived from a table of strings. Over-claiming in
    /// the capability predicate sends a decode to a launcher that does
    /// not exist; under-claiming leaves a kernel nothing calls. The
    /// dispatch seam only `debug_assert!`s the agreement at the moment
    /// a matmul happens to run, which in release is no check at all.
    #[cfg(feature = "cuda")]
    #[test]
    fn every_cuda_matvec_kind_has_a_launcher() {
        use super::gpu_backend::cuda_matvec_launch;
        for &kind in QuantKind::ALL {
            assert_eq!(
                cuda_matvec_kind_supported(kind),
                cuda_matvec_launch(kind).is_some(),
                "{kind:?}: the capability table and the launch table disagree"
            );
        }
    }

    /// `block_bytes_for_kind` is deliberately partial, so every kind
    /// CUDA or Vulkan claims a matvec for has to be one of its arms.
    ///
    /// Calling it IS the assertion: the arm it lacks is an
    /// `unreachable!()`, and reaching that in a rayon worker is a panic
    /// rather than the fallback the seam promises. `Metal` is
    /// deliberately not checked -- it claims IQ4_XS, asks
    /// `frink_metal::gpu::matvec_launch_meta` for its block size, and
    /// never touches this function.
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[test]
    fn every_cuda_or_vulkan_matvec_kind_has_a_block_size() {
        use super::gpu_backend::{BackendCaps, Cuda, Vulkan};
        for &kind in QuantKind::ALL {
            if Cuda::matvec_kernel(kind).is_none() && Vulkan::matvec_kernel(kind).is_none() {
                continue;
            }
            let block_bytes = WeightMatrix::block_bytes_for_kind(kind);
            assert!(
                block_bytes > 0,
                "{kind:?}: a claimed matvec kind needs a real block size"
            );
        }
    }

    /// `block_bytes_for_kind` and `block_bytes_per_row` are two
    /// functions that must agree about one format's geometry, and the
    /// matvec seam DIVIDES one by the other.
    ///
    /// `Cuda::launch_matvec` derives `n_blocks_per_row` as
    /// `block_bytes_per_row(kind, cols) / block_bytes_for_kind(kind)`
    /// and hands it to a kernel that strides the row by a byte count
    /// written as a literal in CUDA C. If the two disagreed by so much
    /// as one byte the division would silently truncate, the kernel
    /// would read fewer blocks than the row holds, and every output
    /// would be a partial dot product -- plausible numbers, no error,
    /// no panic, and nothing in the suite red.
    ///
    /// Both are also held to `frink-cuda`'s own `MulMmKind` row, which
    /// is where that CUDA C literal comes from, so all three agree or
    /// this fails.
    ///
    /// Nothing checked any of it. That was survivable while the two
    /// tables were edited together by one person on one day; five kinds
    /// joined on 2026-09-09 and each needed a row in both.
    ///
    /// Sabotage: give any kind the wrong constant in either function
    /// and this names it.
    ///
    /// Gated like its neighbour: `block_bytes_for_kind` itself only
    /// exists when a backend that calls it is compiled in.
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[test]
    fn the_two_block_size_functions_agree_for_every_gpu_kind() {
        use super::gpu_backend::{BackendCaps, Cuda, Vulkan};
        for &kind in QuantKind::ALL {
            if Cuda::matvec_kernel(kind).is_none() && Vulkan::matvec_kernel(kind).is_none() {
                continue;
            }
            let block_bytes = WeightMatrix::block_bytes_for_kind(kind);
            let mm = frink_cuda::mul_mm::kind_by_name(kind.name())
                .unwrap_or_else(|| panic!("{kind:?}: claims a GPU matvec with no mul_mm row"));
            assert_eq!(
                block_bytes, mm.block_bytes,
                "{kind:?}: frink-core's block size is not the one the kernel strides by"
            );

            // `block_bytes_per_row` takes `&self` but reads only its
            // arguments, so any matrix of the right kind will do.
            let probe = WeightMatrix::Quantized {
                data: WeightBytes::Owned(Vec::new()),
                rows: 1,
                cols: mm.block_elems,
                kind,
            };
            // Three, four and five whole blocks: a per-row function
            // that had dropped the multiply would still pass at one.
            for blocks in 3..=5usize {
                let cols = mm.block_elems * blocks;
                let row_bytes = probe.block_bytes_per_row(kind, cols);
                assert_eq!(
                    row_bytes,
                    blocks * block_bytes,
                    "{kind:?}: block_bytes_per_row({cols}) is not {blocks} x {block_bytes}"
                );
                assert_eq!(
                    row_bytes / block_bytes,
                    blocks,
                    "{kind:?}: the n_blocks_per_row the matvec seam derives is wrong"
                );
            }
        }
    }

    /// F32 and F16 are not quantized, so `None` is the right answer and
    /// not a gap: the loader builds a plain `WeightMatrix::F32` for
    /// them rather than reporting an unsupported dtype.
    #[test]
    fn an_unquantized_dtype_maps_to_nothing() {
        assert_eq!(quant_kind_for(GgmlType::F32), None);
        assert_eq!(quant_kind_for(GgmlType::F16), None);
    }
    use super::*;

    /// Forces [`cpu_int_dot_enabled`] for the lifetime of the guard, so a
    /// test can drive the quantized-activation batch kernels (the
    /// interleaved `block_q*_Kx8` / `block_q*_0x4` repack tier and the
    /// NEON i8mm GEMMs behind it) that every shipped binary turns on via
    /// `default_cpu_int_dot_on` but `cargo test` otherwise leaves off.
    ///
    /// The override is process-global, so the guard serializes on a
    /// mutex: two tests forcing opposite values concurrently would
    /// otherwise see each other's setting.
    pub(super) struct ForceIntDot {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    /// Hoisted out of `new` so the guard's OWN test can hold it for the
    /// whole of its body.
    ///
    /// It could not before, and that was a real flake rather than a
    /// theoretical one: the test asserted the override was cleared
    /// AFTER its guard dropped, which is after the lock was released,
    /// so any other test taking the guard in between stored its value
    /// first and the assertion read that. Seen once in a full-workspace
    /// run and green on its own and on rerun, which is the signature.
    static INT_DOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    impl ForceIntDot {
        pub(super) fn new(on: bool) -> Self {
            let lock = INT_DOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            Self::set(on);
            Self { _lock: lock }
        }

        /// The store the guard makes. Separate so the guard's test can
        /// drive it under a lock it holds itself, rather than restating
        /// the constant in a second place.
        fn set(on: bool) {
            INT_DOT_TEST_OVERRIDE.store(i8::from(on), std::sync::atomic::Ordering::Release);
        }

        /// The store the guard makes on drop.
        fn clear() {
            INT_DOT_TEST_OVERRIDE.store(-1, std::sync::atomic::Ordering::Release);
        }
    }

    impl Drop for ForceIntDot {
        fn drop(&mut self) {
            // Before `_lock` drops: a struct's fields drop after its
            // `Drop` body, so the override is cleared while this still
            // holds the lock and no other guard can be between.
            Self::clear();
        }
    }

    /// The guard has to actually move the getter, in both directions --
    /// otherwise every test built on it silently exercises one path
    /// twice, which is exactly the hole it exists to close.
    #[test]
    fn force_int_dot_moves_the_getter_and_restores_it() {
        // Held for the WHOLE body, which is the fix: the old version
        // read the process-global after its guard had released the
        // lock, so a concurrent test's store could be what it read.
        let _lock = INT_DOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        ForceIntDot::set(true);
        assert!(cpu_int_dot_enabled(), "forcing on must enable int dot");
        ForceIntDot::set(false);
        assert!(!cpu_int_dot_enabled(), "forcing off must disable int dot");

        ForceIntDot::clear();
        assert_eq!(
            INT_DOT_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Acquire),
            -1,
            "clearing must restore the unforced getter"
        );
        assert!(
            cpu_int_dot_enabled() == cpu_int_dot_enabled(),
            "the unforced getter must be readable"
        );
    }

    /// `dequant_row` must reproduce exactly the values a full-buffer
    /// dequantization of the same row produces, for every storage
    /// variant -- and read only that row's bytes (each row here has
    /// distinct values, so an off-by-one-row slice fails loudly).
    #[test]
    fn dequant_row_matches_full_dequant_per_row() {
        // F32 variant.
        let rows = 3;
        let cols = 64;
        let f32_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let m = WeightMatrix::F32(Tensor::new(f32_data.clone(), vec![rows, cols]));
        for r in 0..rows {
            assert_eq!(m.dequant_row(r), &f32_data[r * cols..(r + 1) * cols]);
        }

        // Quantized (Q8_0) variant: quantize each row independently and
        // compare dequant_row against dequantizing that row's bytes.
        let mut packed = Vec::new();
        for r in 0..rows {
            packed.extend(make_q8_0_row(&f32_data[r * cols..(r + 1) * cols]));
        }
        let row_bytes = packed.len() / rows;
        let q = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed.clone()),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        for r in 0..rows {
            let expected =
                frink_quant::dequant_q8_0(&packed[r * row_bytes..(r + 1) * row_bytes]).unwrap();
            assert_eq!(q.dequant_row(r), expected, "Q8_0 row {r}");
        }

        // Mxfp4 (two-buffer) variant: arbitrary valid bytes, compare
        // against the row-level reference dequantizer directly.
        let cols = 64;
        let packed: Vec<u8> = pseudo_bytes(7, rows * cols / 2);
        let scales: Vec<u8> = pseudo_bytes(11, rows * cols / 32);
        let m = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed.clone()),
            scale: WeightBytes::Owned(scales.clone()),
            rows,
            cols,
        };
        for r in 0..rows {
            let expected = frink_quant::dequant_mxfp4_row(
                &packed[r * cols / 2..(r + 1) * cols / 2],
                &scales[r * cols / 32..(r + 1) * cols / 32],
            )
            .unwrap();
            assert_eq!(m.dequant_row(r), expected, "Mxfp4 row {r}");
        }
    }

    /// A quantized matrix used as an embedding table: `dequant_row`
    /// then a dot product must agree with `apply` against a one-hot...
    /// no -- more directly, with the fused `dot` of that row, proving
    /// row lookup and matmul read identical bytes.
    #[test]
    fn dequant_row_agrees_with_fused_dot_on_the_same_row() {
        let rows = 4;
        let cols = 64;
        let f32_data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let mut packed = Vec::new();
        for r in 0..rows {
            packed.extend(make_q8_0_row(&f32_data[r * cols..(r + 1) * cols]));
        }
        let q = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.031).cos()).collect();
        let applied = q.apply(&x);
        // With `FRINK_CPU_INT_DOT` on, `apply` quantizes the ACTIVATION to
        // int8 as well, so the two sides no longer differ only by float
        // summation order and a fixed 1e-4 is not the right bar -- it fired
        // at 6.5e-3 on a result of 5.25, which is the activation error, not
        // a byte disagreement. The worst case is derivable rather than
        // guessed: `quantize_activations_q8` rounds to `d = amax/127`, so
        // each element moves by at most `d/2`, and the dot's error is
        // bounded by that times the row's L1 norm.
        let bound = |row: &[f32]| {
            if !cpu_int_dot_for(IntDotShape::Matvec) {
                return 1e-4;
            }
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let l1: f32 = row.iter().map(|w| w.abs()).sum();
            (amax / 127.0 / 2.0) * l1
        };
        for (r, &got) in applied.iter().enumerate() {
            let row = q.dequant_row(r);
            let via_row: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            let bound = bound(&row);
            assert!(
                (got - via_row).abs() < bound,
                "row {r}: apply={got} via dequant_row={via_row} (bound {bound:e})"
            );
        }
    }

    fn make_q8_0_row(values: &[f32]) -> Vec<u8> {
        frink_quant::quantize_q8_0(values)
    }

    /// Deterministic byte generator for MXFP4 test fixtures (no
    /// quantizer exists in `frink_quant` -- MXFP4 is only ever a
    /// real, already-quantized checkpoint format, never produced by
    /// frink -- so tests build arbitrary-but-valid-shaped bytes
    /// directly, same convention as `frink-models::kimi_loader`'s
    /// tests).
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Clamped to a realistic E8M0 scale range -- see
    /// `frink-models::kimi_loader`'s identical helper for why (byte
    /// 255 is OCP-spec-reserved for NaN, and bytes above ~252 can
    /// legitimately overflow f32::MAX when combined with E2M1's max
    /// magnitude; neither is representative of a real trained weight).
    fn pseudo_mxfp4_scale_bytes(seed: u32, len: usize) -> Vec<u8> {
        pseudo_bytes(seed, len)
            .into_iter()
            .map(|b| b % 180)
            .collect()
    }

    #[test]
    fn f32_and_mxfp4_paths_agree() {
        let rows = 2;
        let cols = 64; // 2 MXFP4 groups of 32 per row
        let packed = pseudo_bytes(1, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(2, rows * (cols / frink_quant::MXFP4_GROUP_SIZE));
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.3).collect();

        // Independent reference: dequantize each row to plain f32 (the
        // already-tested `dequant_mxfp4_row`), then use the ordinary
        // F32 matmul path.
        let mut f32_weights = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let prow = &packed[r * (cols / 2)..(r + 1) * (cols / 2)];
            let srow = &scale[r * (cols / frink_quant::MXFP4_GROUP_SIZE)
                ..(r + 1) * (cols / frink_quant::MXFP4_GROUP_SIZE)];
            f32_weights.extend(frink_quant::dequant_mxfp4_row(prow, srow).unwrap());
        }
        let f32_matrix = WeightMatrix::F32(Tensor::new(f32_weights, vec![rows, cols]));
        let f32_out = f32_matrix.apply(&x);

        let mxfp4_matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };
        let mxfp4_out = mxfp4_matrix.apply(&x);

        assert_eq!(f32_out.len(), rows);
        assert_eq!(mxfp4_out.len(), rows);
        for (f, m) in f32_out.iter().zip(mxfp4_out.iter()) {
            assert!((f - m).abs() < 1e-3, "f32={f} mxfp4={m}");
        }
    }

    #[test]
    fn mxfp4_apply_batch_matches_sequential_apply_calls() {
        let rows = 3;
        let cols = 64;
        let packed = pseudo_bytes(3, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(4, rows * (cols / frink_quant::MXFP4_GROUP_SIZE));
        let matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };

        let batch_size = 4;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 13) as f32) * 0.02 - 0.15)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            assert_eq!(
                sequential, from_batch,
                "batch row {b} disagrees with sequential apply()"
            );
        }
    }

    #[test]
    fn mxfp4_resident_bytes_matches_the_packed_plus_scale_byte_count_not_eager_f32() {
        let rows = 2;
        let cols = 64;
        let packed = pseudo_bytes(5, rows * (cols / 2));
        let scale = pseudo_mxfp4_scale_bytes(6, rows * (cols / frink_quant::MXFP4_GROUP_SIZE));
        let packed_len = packed.len();
        let scale_len = scale.len();
        let matrix = WeightMatrix::Mxfp4 {
            packed: WeightBytes::Owned(packed),
            scale: WeightBytes::Owned(scale),
            rows,
            cols,
        };

        assert_eq!(matrix.resident_bytes(), packed_len + scale_len);
        // Real MXFP4 packs 2 values/byte plus 1 scale byte per 32
        // values -- resident_bytes should be far below the 4-bytes-
        // per-value eager-f32 footprint.
        let eager_f32_bytes = rows * cols * 4;
        assert!(
            matrix.resident_bytes() * 4 < eager_f32_bytes,
            "expected MXFP4 resident bytes well under 1/4 of eager f32: got {} vs {}",
            matrix.resident_bytes(),
            eager_f32_bytes
        );
    }

    #[test]
    fn f32_and_quantized_paths_agree_within_quant_error() {
        // 1 row, 32 cols, values chosen to keep Q8_0 error small.
        let weights: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.2).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05 - 0.8).collect();

        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![1, 32]));
        let f32_out = f32_matrix.apply(&x);

        let packed = make_q8_0_row(&weights);
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        let quant_out = quant_matrix.apply(&x);

        assert_eq!(f32_out.len(), 1);
        assert_eq!(quant_out.len(), 1);
        assert!(
            (f32_out[0] - quant_out[0]).abs() < 0.05,
            "f32={} quant={}",
            f32_out[0],
            quant_out[0]
        );
    }

    #[test]
    fn quantized_resident_bytes_is_smaller_than_f32() {
        let weights = vec![0.1f32; 64]; // 2 rows x 32 cols
        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![2, 32]));

        let mut packed = Vec::new();
        for chunk in weights.chunks(32) {
            packed.extend(frink_quant::quantize_q8_0(chunk));
        }
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 2,
            cols: 32,
            kind: QuantKind::Q8_0,
        };

        assert_eq!(f32_matrix.resident_bytes(), 64 * 4); // 256 bytes
        assert_eq!(quant_matrix.resident_bytes(), 2 * 34); // 68 bytes
        assert!(quant_matrix.resident_bytes() < f32_matrix.resident_bytes());
        // Q8_0 should be close to the theoretical ~4x reduction vs f32.
        let ratio = f32_matrix.resident_bytes() as f32 / quant_matrix.resident_bytes() as f32;
        assert!(ratio > 3.5, "expected ~4x reduction, got {ratio}x");
    }

    #[test]
    fn rows_and_cols_report_correctly_for_both_variants() {
        let f32_matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        assert_eq!(f32_matrix.rows(), 2);
        assert_eq!(f32_matrix.cols(), 3);

        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(vec![0u8; 34]),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        assert_eq!(quant_matrix.rows(), 1);
        assert_eq!(quant_matrix.cols(), 32);
    }

    #[test]
    #[should_panic]
    fn apply_panics_on_activation_length_mismatch() {
        let f32_matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        f32_matrix.apply(&[1.0, 2.0]); // wrong length (needs 3)
    }

    #[test]
    fn apply_batch_with_batch_size_one_matches_apply() {
        // Pinned, not inherited. This asserts `apply` and `apply_batch`
        // are BIT-identical, which is only true while both take the same
        // kernel -- and since #152 they do not on x86, where the batch
        // half of the int-dot tier is taken and the matvec half is not.
        // The override is process-global, so without the guard a
        // concurrent test holding it on decides this one's result.
        let _int_dot = ForceIntDot::new(false);
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.13).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02 - 0.3).collect();

        let f32_matrix = WeightMatrix::F32(Tensor::new(weights.clone(), vec![1, 32]));
        let single = f32_matrix.apply(&x);
        let batched = f32_matrix.apply_batch(&x, 1);
        assert_eq!(single, batched);

        let packed = frink_quant::quantize_q8_0(&weights);
        let quant_matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows: 1,
            cols: 32,
            kind: QuantKind::Q8_0,
        };
        let single_q = quant_matrix.apply(&x);
        let batched_q = quant_matrix.apply_batch(&x, 1);
        assert_eq!(single_q, batched_q);
    }

    #[test]
    fn apply_batch_matches_sequential_apply_calls_for_each_row_f32() {
        let rows = 3;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let matrix = WeightMatrix::F32(Tensor::new(weights, vec![rows, cols]));

        let batch_size = 4;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 13) as f32) * 0.03 - 0.2)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            assert_eq!(
                sequential, from_batch,
                "batch row {b} disagrees with sequential apply()"
            );
        }
    }

    #[test]
    fn apply_batch_matches_sequential_apply_calls_for_each_row_quantized() {
        let rows = 3;
        let cols = 32;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.07)
            .collect();
        let mut packed = Vec::new();
        for row in weights.chunks(cols) {
            packed.extend(frink_quant::quantize_q8_0(row));
        }
        let matrix = WeightMatrix::Quantized {
            data: WeightBytes::Owned(packed),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };

        let batch_size = 5;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| ((i % 11) as f32) * 0.04 - 0.25)
            .collect();

        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);

        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            assert_batch_row_matches(QuantKind::Q8_0, "", b, &sequential, from_batch);
        }
    }

    /// Minimal f16 encode for small positive normals (test fixtures only).
    pub(super) fn f16_le(x: f32) -> [u8; 2] {
        let bits = x.to_bits();
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = (bits >> 13) & 0x3ff;
        (((exp as u16) << 10) | mant as u16).to_le_bytes()
    }

    /// Deterministic pseudo-random quantized matrix: every byte pattern is
    /// a valid weight block, only the f16 scale fields need sane values.
    /// Compare one row of `apply_batch` against `apply`, scaled by the
    /// magnitude of the row rather than of each element.
    ///
    /// The element-wise denominator (`err / s.abs().max(1.0)`) is wrong
    /// for a dot product over random data: the sums cancel, so a result
    /// that lands near zero turns a normal rounding difference into a
    /// relative error of 30%. Measured on Metal, the divergence is a
    /// uniform 5.5e-4 of the row's own scale across every quant kind
    /// and batch index, and up to 2.9e-1 of the individual result. The
    /// first number describes the arithmetic; the second describes
    /// which results happened to cancel.
    ///
    /// This matters because `apply_batch` is not `apply` on a GPU
    /// build: `apply_batch` dispatches to Metal while `apply` stays on
    /// the CPU, so this compares two backends. The bound stays tight on
    /// CPU, where both sides are the same code and must agree closely.
    fn assert_batch_row_matches(
        kind: QuantKind,
        ctx: &str,
        b: usize,
        sequential: &[f32],
        from_batch: &[f32],
    ) {
        let scale = sequential
            .iter()
            .fold(0.0f32, |a, v| a.max(v.abs()))
            .max(1.0);
        // A GPU build compares Metal against the CPU; a CPU build
        // compares the CPU against itself -- UNLESS this host takes only
        // one half of the int-dot tier, in which case `apply` and
        // `apply_batch` are not the same arithmetic at all.
        //
        // That is x86 since #152: the batch half runs the AVX2
        // interleaved GEMM over an int8-quantized activation while the
        // matvec half stays on the f32 AVX2 dot, because the int8 matvec
        // measured 4x to 8.8x slower there. The gap between the two
        // sides is then the ACTIVATION quantization floor -- each element
        // of `x` moves by up to `d/2` at `d = amax/127` -- not float
        // summation order, and a 1e-4 bar describes the wrong thing.
        //
        // Measured across every shape in these tests on a linux/amd64
        // container with real AVX2 (2026-09-09): worst 7.9e-3 of the row
        // scale. 6e-2 keeps a 7.6x margin, the same discipline as
        // `int_dot_batch_matches_dequant_dot_reference`, and is still far
        // inside a mis-pack, which decorrelates the two outputs entirely.
        let mixed = cpu_int_dot_for(IntDotShape::Matvec) != cpu_int_dot_for(IntDotShape::BatchGemm);
        let bound = if cfg!(any(feature = "metal", feature = "cuda")) {
            5e-3
        } else if mixed {
            6e-2
        } else {
            1e-4
        };
        for (r, (s, got)) in sequential.iter().zip(from_batch.iter()).enumerate() {
            let err = (s - got).abs() / scale;
            assert!(
                err < bound,
                "{kind:?} {ctx} batch {b} row {r}: apply()={s} apply_batch={got} \
                 (err {err:e} of row scale {scale}, bound {bound:e})"
            );
        }
    }

    fn synth_quant_matrix(kind: QuantKind, rows: usize, cols: usize) -> WeightMatrix {
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        let mut data = Vec::new();
        match kind {
            QuantKind::Q8_0 | QuantKind::Q4_0 => {
                let qs = if kind == QuantKind::Q8_0 { 32 } else { 16 };
                for _ in 0..rows * (cols / 32) {
                    data.extend_from_slice(&f16_le(0.02 + f32::from(next()) * 0.0004));
                    for _ in 0..qs {
                        data.push(next());
                    }
                }
            }
            QuantKind::Q4K | QuantKind::Q5K => {
                let body = if kind == QuantKind::Q4K {
                    12 + 128
                } else {
                    12 + 32 + 128
                };
                for _ in 0..rows * (cols / 256) {
                    data.extend_from_slice(&f16_le(0.01 + f32::from(next()) * 0.0002));
                    data.extend_from_slice(&f16_le(0.005 + f32::from(next()) * 0.0001));
                    for _ in 0..body {
                        data.push(next());
                    }
                }
            }
            QuantKind::Q6K => {
                for _ in 0..rows * (cols / 256) {
                    for _ in 0..128 + 64 + 16 {
                        data.push(next());
                    }
                    data.extend_from_slice(&f16_le(0.01 + f32::from(next()) * 0.0002));
                }
            }
            _ => unreachable!("synth_quant_matrix: unsupported kind"),
        }
        WeightMatrix::Quantized {
            data: WeightBytes::Owned(data),
            rows,
            cols,
            kind,
        }
    }
    /// One `apply_batch` vs per-row `apply` sweep, parameterized by shape
    /// so the shape tests below differ only in the numbers they pass.
    fn assert_apply_batch_matches_apply(
        kind: QuantKind,
        rows: usize,
        cols: usize,
        batch_size: usize,
        seed: usize,
    ) {
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| (((i * 31 + seed) % 97) as f32) * 0.021 - 1.0)
            .collect();
        let matrix = synth_quant_matrix(kind, rows, cols);
        let batched = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(batched.len(), batch_size * rows);
        let ctx = format!(
            "rows {rows} cols {cols} batch_size {batch_size} int_dot {}",
            cpu_int_dot_for(IntDotShape::BatchGemm)
        );
        for b in 0..batch_size {
            let x = &x_batch[b * cols..(b + 1) * cols];
            let sequential = matrix.apply(x);
            let from_batch = &batched[b * rows..(b + 1) * rows];
            // Delegates rather than restating the bound. The first
            // version of this helper compared each element against
            // `s.abs().max(1.0)`, which is a bare 1e-4 ABSOLUTE bound
            // for any row whose value is small -- and a dot product of
            // 512 terms that cancels to -0.76 carries the rounding of
            // the terms, not of the result. It passed on aarch64 and
            // failed on x86_64 CI at 1.07e-4, on one row out of 17094.
            // `assert_batch_row_matches` already divides by the row
            // vector's own scale, which is the invariant that makes the
            // comparison meaningful, and it is now the only place the
            // tolerance is written down.
            assert_batch_row_matches(kind, &ctx, b, &sequential, from_batch);
        }
    }

    const BATCH_SHAPE_KINDS: [QuantKind; 5] = [
        QuantKind::Q8_0,
        QuantKind::Q4_0,
        QuantKind::Q4K,
        QuantKind::Q5K,
        QuantKind::Q6K,
    ];

    /// `apply_batch` writes straight into the `[batch][rows]` output from
    /// parallel tasks (no staging transpose); the shapes here force every
    /// write pattern: full row-groups, a tail of leftover rows, and both
    /// full and partial activation tiles.
    ///
    /// Run under both settings of [`cpu_int_dot_enabled`]. With int-dot
    /// off, `apply_batch` dequantizes and the repack tier is skipped
    /// entirely; with it on -- which is what every shipped binary does,
    /// via `default_cpu_int_dot_on` -- the interleaved `block_q*_Kx8` /
    /// `block_q*_0x4` kernels and, on an i8mm host, the SMMLA GEMMs are
    /// the code under test. `cargo test` leaves the env var unset, so
    /// without [`ForceIntDot`] only the first of those two ever ran.
    #[test]
    fn apply_batch_matches_apply_across_kinds_with_groups_and_tail() {
        for int_dot in [false, true] {
            let _g = ForceIntDot::new(int_dot);
            for kind in BATCH_SHAPE_KINDS {
                // 19 rows = 2x8-row groups + 3 tail (4x4-row groups + 3
                // for Q8_0/Q4_0); 6 activations = one full 4-tile + a
                // partial one.
                assert_apply_batch_matches_apply(kind, 19, 512, 6, 7);
            }
        }
    }

    /// Shapes too small to fill one interleaved row-group, which the
    /// tests around this one never reach: they use `rows` big enough that
    /// `n_groups > 0` for every kind. At `rows < 8` the K-quant arms take
    /// their `else` branch (per-row `gemm_q*_k_q8_row`) with the repack
    /// path completely bypassed, and at `rows < 4` the Q8_0/Q4_0 arms do
    /// the same with `dot_q*_q8`. `rows = 5` is the mixed case: one full
    /// `block_q*_0x4` group plus a 1-row tail for Q8_0/Q4_0, zero groups
    /// for the Kx8 kinds. `rows = 1` is the single-row case.
    ///
    /// `cols = 256` is also the minimum K for a K-quant -- a single
    /// super-block, so every kernel's block loop runs exactly one trip.
    /// `batch_size = 1` is the single-column case: one activation in the
    /// quad, `na = 1` with three zero-padded lanes in `Q8KActsX4` /
    /// `Q8ActsX4`.
    #[test]
    fn apply_batch_matches_apply_for_sub_tile_shapes() {
        for int_dot in [false, true] {
            let _g = ForceIntDot::new(int_dot);
            for kind in BATCH_SHAPE_KINDS {
                for rows in [1, 2, 3, 5, 7] {
                    for batch_size in [1, 2, 5] {
                        assert_apply_batch_matches_apply(kind, rows, 256, batch_size, 13);
                    }
                }
            }
        }
    }

    /// `apply_batch` under int-dot against an f32 dequantize-and-dot
    /// reference that never touches the packed buffer.
    ///
    /// Every other batch test compares `apply_batch` against `apply`,
    /// which under int-dot is the packed **GEMV** against the packed
    /// **GEMM** -- two kernels reading the *same* interleaved bytes. That
    /// catches a bad kernel but is structurally blind to a bad
    /// `pack_q*_matrix_x*`: both sides read the same wrong bytes and
    /// agree. `dequant_row` is the only reference in the tree that
    /// re-derives the weights from the canonical GGUF blocks, so it is
    /// the only one that can see a mis-interleave.
    ///
    /// The bound is the Q8/Q8_K *activation* quantization floor, not the
    /// kernel's, and it is scaled by the RMS of the reference outputs
    /// rather than per element: these synthetic weights are uniform
    /// random bytes, so individual dots cancel to near zero and a
    /// per-element relative bound would be meaningless. Worst deviation
    /// measured across every shape below, on an M2 Pro (i8mm), is 0.016 x
    /// RMS; 0.12 keeps a 7x margin. Coarse on purpose -- a mis-pack
    /// decorrelates the output from the reference entirely (measured at
    /// 2.07 x RMS for a one-row shift in the Q5_K `qh` interleave), an
    /// order of magnitude past this bound.
    #[test]
    fn int_dot_batch_matches_dequant_dot_reference() {
        let _g = ForceIntDot::new(true);
        // The batch half needs a SIMD `x4` GEMM, so a host without
        // one (an x86 box with no AVX2, Rosetta included) has no
        // packed path to test. Skipping is honest; asserting would
        // make the suite red for a host that is behaving correctly.
        assert!(cpu_int_dot_enabled(), "forcing on must enable int dot");
        if !cpu_int_dot_for(IntDotShape::BatchGemm) {
            return;
        }
        for kind in BATCH_SHAPE_KINDS {
            // Rows straddle both tile widths: below the tile, one short
            // of it, exactly it, one past it, and multi-group with a
            // tail. Batch straddles the 4-wide activation quad. cols 256
            // is the minimum K for a K-quant (one super-block).
            for rows in [1, 3, 5, 7, 8, 9, 19] {
                for cols in [256, 512] {
                    for batch_size in [1, 3, 4, 9] {
                        assert_int_dot_matches_dequant_dot(kind, rows, cols, batch_size, 23);
                    }
                }
            }
        }
        // Q8_0/Q4_0 alone can go down to a single 32-element block.
        for kind in [QuantKind::Q8_0, QuantKind::Q4_0] {
            for rows in [1, 3, 4, 5, 11] {
                for batch_size in [1, 3, 4, 9] {
                    assert_int_dot_matches_dequant_dot(kind, rows, 32, batch_size, 29);
                }
            }
        }
    }

    fn assert_int_dot_matches_dequant_dot(
        kind: QuantKind,
        rows: usize,
        cols: usize,
        batch_size: usize,
        seed: usize,
    ) {
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| (((i * 37 + seed) % 89) as f32) * 0.019 - 0.8)
            .collect();
        let matrix = synth_quant_matrix(kind, rows, cols);
        let got = matrix.apply_batch(&x_batch, batch_size);
        assert_eq!(got.len(), batch_size * rows);

        let mut want = vec![0f32; batch_size * rows];
        for r in 0..rows {
            let w = matrix.dequant_row(r);
            assert_eq!(w.len(), cols);
            for b in 0..batch_size {
                let x = &x_batch[b * cols..(b + 1) * cols];
                want[b * rows + r] = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let err = (g - w).abs();
            assert!(
                err < 0.12 * rms.max(1e-3),
                "{kind:?} rows {rows} cols {cols} batch_size {batch_size} [flat {i}]: \
                 int-dot={g} dequant-dot={w} (err {err}, rms {rms})"
            );
        }
    }

    /// Large enough that `par_chunked_groups` builds a real 2D chunk grid
    /// (32 row-groups × 17 activation tiles) instead of falling back to
    /// one-chunk-per-thread — every (group, tile-range) seam in the
    /// chunked scatter is crossed. The smaller cross-kind test above
    /// covers the fallback path. Both int-dot settings, for the same
    /// reason as that test.
    #[test]
    fn apply_batch_chunked_grid_matches_apply() {
        for int_dot in [false, true] {
            let _g = ForceIntDot::new(int_dot);
            for kind in BATCH_SHAPE_KINDS {
                // 259 rows = 32 groups of 8 + 3 tail (64 of 4 + 3 for
                // Q8_0/Q4_0); 66 activations = 16 full 4-tiles + a
                // partial one.
                assert_apply_batch_matches_apply(kind, 259, 512, 66, 5);
            }
        }
    }

    /// Sharing one quantized activation batch across projections must be
    /// invisible in the results: a matching `BatchActs` produces exactly
    /// what `apply_batch` produces (same quantization, same interleaved
    /// quads, same kernels), and a mismatched variant is ignored rather
    /// than misused.
    #[test]
    fn apply_batch_with_shared_acts_matches_apply_batch() {
        // Shared quads are built under one setting and consumed under
        // another if a concurrent test flips the global mid-run; pin it
        // on, which is also the setting that gives this test something
        // to compare.
        let _int_dot = ForceIntDot::new(true);
        let rows = 19;
        let cols = 512;
        let batch_size = 6;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| (((i * 29 + 11) % 89) as f32) * 0.023 - 1.0)
            .collect();
        for kind in [
            QuantKind::Q8_0,
            QuantKind::Q4_0,
            QuantKind::Q4K,
            QuantKind::Q6K,
        ] {
            let matrix = synth_quant_matrix(kind, rows, cols);
            let baseline = matrix.apply_batch(&x_batch, batch_size);

            let shared = matrix.quantize_batch_acts(&x_batch, batch_size);
            let with_shared = matrix.apply_batch_with_acts(&x_batch, batch_size, shared.as_ref());
            assert_eq!(
                baseline, with_shared,
                "{kind:?}: shared acts changed the result"
            );

            let wrong = match kind {
                QuantKind::Q8_0 | QuantKind::Q4_0 => BatchActs::Q8K {
                    acts: Vec::new(),
                    tiles: Vec::new(),
                    cols,
                },
                _ => BatchActs::Q8 {
                    acts: Vec::new(),
                    tiles: Vec::new(),
                    cols,
                },
            };
            let with_wrong = matrix.apply_batch_with_acts(&x_batch, batch_size, Some(&wrong));
            assert_eq!(
                baseline, with_wrong,
                "{kind:?}: mismatched shared acts were not ignored"
            );
        }
    }

    /// The interleaved quads now ride along with the activations, so the
    /// guard that decides whether a `shared` batch is usable has to cover
    /// them too -- and that guard is the one thing here that is not gated
    /// on `FRINK_CPU_INT_DOT`, so it is tested directly.
    ///
    /// A stale set is not a panic. The quads are indexed by super-block, so
    /// a batch prepared at another width either reads past its own end or
    /// silently dots the wrong columns; both surface as a wrong answer.
    /// What must happen instead is a local re-quantization with no quads,
    /// which is what the fresh-fallback assertions below pin.
    #[test]
    fn shared_acts_are_reused_only_at_the_matching_length_and_width() {
        let cols = 512;
        let batch_size = 7;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| (((i * 37 + 5) % 83) as f32) * 0.019 - 0.9)
            .collect();

        let acts: Vec<_> = (0..batch_size)
            .map(|b| frink_quant::quantize_activations_q8_k(&x_batch[b * cols..(b + 1) * cols]))
            .collect();
        let tiles: Vec<_> = acts
            .chunks(frink_quant::Q8K_ACTS_X4_NC)
            .map(|c| frink_quant::prepare_q8_k_acts_x4(c, cols))
            .collect();
        let n_tiles = tiles.len();
        let shared = BatchActs::Q8K { acts, tiles, cols };

        let mut owned = Vec::new();
        let (got, quads) =
            WeightMatrix::q8k_acts(Some(&shared), &x_batch, batch_size, cols, &mut owned);
        assert_eq!(got.len(), batch_size);
        assert_eq!(
            quads.len(),
            n_tiles,
            "matching batch did not reuse its quads"
        );
        assert!(owned.is_empty(), "matching batch was re-quantized anyway");

        // Same positions, another width: refuse and re-quantize.
        let mut owned = Vec::new();
        let (got, quads) =
            WeightMatrix::q8k_acts(Some(&shared), &x_batch, batch_size, 256, &mut owned);
        assert!(quads.is_empty(), "quads from another width were accepted");
        assert_eq!(got.len(), batch_size);
        assert_eq!(got[0].n_blocks(), 1, "fallback did not quantize at 256");

        // Same width, another position count: refuse and re-quantize.
        let mut owned = Vec::new();
        let (got, quads) =
            WeightMatrix::q8k_acts(Some(&shared), &x_batch[..cols], 1, cols, &mut owned);
        assert!(quads.is_empty(), "quads for another batch were accepted");
        assert_eq!(got.len(), 1);

        // The Q8_0 half of the same guard.
        let acts: Vec<_> = (0..batch_size)
            .map(|b| frink_quant::quantize_activations_q8(&x_batch[b * cols..(b + 1) * cols]))
            .collect();
        let tiles: Vec<_> = acts
            .chunks(frink_quant::Q8K_ACTS_X4_NC)
            .map(|c| frink_quant::prepare_q8_acts_x4(c, cols))
            .collect();
        let n_tiles = tiles.len();
        let shared = BatchActs::Q8 { acts, tiles, cols };

        let mut owned = Vec::new();
        let (got, quads) =
            WeightMatrix::q8_acts(Some(&shared), &x_batch, batch_size, cols, &mut owned);
        assert_eq!(got.len(), batch_size);
        assert_eq!(
            quads.len(),
            n_tiles,
            "matching batch did not reuse its quads"
        );

        let mut owned = Vec::new();
        let (got, quads) =
            WeightMatrix::q8_acts(Some(&shared), &x_batch, batch_size, 256, &mut owned);
        assert!(quads.is_empty(), "quads from another width were accepted");
        assert_eq!(got[0].n_blocks(), 8, "fallback did not quantize at 256");
    }

    /// Whatever a projection would have built for itself, a sibling's
    /// shared batch must hand it the same thing. Q4_K, Q5_K and Q6_K read
    /// one Q8_K quad set between them, and Q8_0 and Q4_0 one Q8_0 set, so
    /// the donor's kind must not show through.
    ///
    /// Gated the same way the path itself is: with `FRINK_CPU_INT_DOT`
    /// off (the library default) `quantize_batch_acts` returns `None` and
    /// no projection consumes quads at all, so this asserts against the
    /// INT_DOT build. Run the suite both ways.
    #[test]
    fn shared_quads_are_what_each_consumer_would_have_built_itself() {
        // The early return below reads a process-global, so it has to be
        // pinned or a neighbour can turn the tier off between the check
        // and the assertions it guards.
        let _int_dot = ForceIntDot::new(true);
        if !cpu_int_dot_for(IntDotShape::BatchGemm) {
            return;
        }
        let rows = 24;
        let cols = 512;
        let batch_size = 7;
        let x_batch: Vec<f32> = (0..batch_size * cols)
            .map(|i| (((i * 37 + 5) % 83) as f32) * 0.019 - 0.9)
            .collect();

        for (donor, consumers) in [
            (QuantKind::Q4K, &[QuantKind::Q5K, QuantKind::Q6K][..]),
            (QuantKind::Q8_0, &[QuantKind::Q4_0][..]),
        ] {
            let shared = synth_quant_matrix(donor, rows, cols)
                .quantize_batch_acts(&x_batch, batch_size)
                .expect("INT_DOT is on and this kind/width is eligible");
            for kind in consumers {
                let matrix = synth_quant_matrix(*kind, rows, cols);
                let baseline = matrix.apply_batch(&x_batch, batch_size);
                let shared_out = matrix.apply_batch_with_acts(&x_batch, batch_size, Some(&shared));
                assert_eq!(
                    baseline, shared_out,
                    "{kind:?} consuming {donor:?} quads changed the result"
                );
            }
        }
    }

    #[test]
    fn apply_batch_with_zero_batch_size_returns_empty() {
        let matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
        let out = matrix.apply_batch(&[], 0);
        assert!(out.is_empty());
    }

    #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
    mod gpu_dispatch {
        use super::*;

        /// `apply_gpu` must return `None` for `F32` -- and, crucially,
        /// without ever touching the CUDA driver at all (this runs on
        /// every CI machine, none of which have a GPU): the `let ...
        /// else { return None }` pattern match happens before any
        /// `frink_cuda` call, so this is a real, meaningful assertion
        /// about dispatch behavior, not a stub.
        #[test]
        fn apply_gpu_returns_none_for_f32() {
            let matrix = WeightMatrix::F32(Tensor::new(vec![0.0; 6], vec![2, 3]));
            assert!(matrix.apply_gpu(&[0.0, 0.0, 0.0]).is_none());
        }

        #[test]
        fn apply_gpu_returns_none_for_mxfp4() {
            let matrix = WeightMatrix::Mxfp4 {
                packed: WeightBytes::Owned(vec![0u8; 32]),
                scale: WeightBytes::Owned(vec![0u8; 2]),
                rows: 1,
                cols: 64,
            };
            assert!(matrix.apply_gpu(&vec![0.0; 64]).is_none());
        }

        /// A `Quantized` matrix whose `kind` has no GPU kernel on any
        /// compiled backend must fall back to `None`, not panic on the
        /// `unreachable!()` in `block_bytes_for_kind` -- proving the
        /// two match arms (`apply_gpu`'s launch table,
        /// `block_bytes_for_kind`'s partial one) stay in sync.
        ///
        /// The probe was `Q2_K` until 2026-09-09, when Q2_K gained a
        /// CUDA matvec and a GEMM and stopped being unsupported. `Q4_1`
        /// has neither on any backend and is the hole now. Moving it
        /// found a real defect rather than being bookkeeping: with the
        /// `cuda` feature on and no driver present, the first real
        /// dispatch through `Cuda::launch_matvec` aborted the process
        /// inside `cudarc`'s library loader, which that arm's
        /// `Result` could never have reported.
        #[test]
        fn apply_gpu_returns_none_for_an_unsupported_quant_kind() {
            let matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(vec![0u8; frink_quant::Q4_1_BLOCK_BYTES]),
                rows: 1,
                cols: frink_quant::Q4_1_BLOCK_ELEMS,
                kind: QuantKind::Q4_1,
            };
            assert!(matrix
                .apply_gpu(&[0.0; frink_quant::Q4_1_BLOCK_ELEMS])
                .is_none());
        }

        #[test]
        #[ignore = "requires real GPU hardware (CUDA or Metal) -- run with --ignored"]
        fn apply_gpu_matches_apply_for_q8_0_on_real_hardware() {
            let weights: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.05).collect();
            let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
            let packed = frink_quant::quantize_q8_0(&weights);
            let matrix = WeightMatrix::Quantized {
                data: WeightBytes::Owned(packed),
                rows: 1,
                cols: 64,
                kind: QuantKind::Q8_0,
            };

            let cpu = matrix.apply_cpu(&x);
            let gpu = matrix
                .apply_gpu(&x)
                .expect("Q8_0 must dispatch to a real GPU kernel");
            assert_eq!(cpu.len(), gpu.len());
            for (c, g) in cpu.iter().zip(gpu.iter()) {
                assert!((c - g).abs() < 1e-2, "cpu={c} gpu={g}");
            }
        }
    }

    // ---- kernel-lookup registry coverage -------------------------------
    //
    // These are the tests that would have caught the IQ4_XS silent CPU
    // prefill at `cargo test` time instead of via a 13.7x benchmark row.

    /// A quantized matrix of `kind` with `cols` columns, filled with
    /// arbitrary bytes -- the probe reads only shape and kind, never the
    /// weights, so the contents are irrelevant.
    fn shaped(kind: QuantKind, rows: usize, cols: usize) -> WeightMatrix {
        let per_row = match kind {
            QuantKind::Q8_0 => cols / 32 * 34,
            _ => cols,
        };
        WeightMatrix::Quantized {
            data: WeightBytes::Owned(vec![0u8; rows * per_row.max(1)]),
            rows,
            cols,
            kind,
        }
    }

    /// `QuantKind::ALL` must actually list every variant. `name()` is
    /// exhaustive by the compiler, so distinct names prove distinct
    /// variants; the count pins that none was dropped from the list.
    #[test]
    fn quant_kind_all_lists_every_variant_exactly_once() {
        let mut names: Vec<&str> = QuantKind::ALL.iter().map(|k| k.name()).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "QuantKind::ALL has a duplicate");
        assert_eq!(
            total, 23,
            "a QuantKind variant was added without updating ALL"
        );
    }

    /// The invariant that keeps prefill honest: every kind with a Metal
    /// matvec also has a Metal batched GEMM. Break it and the kind still
    /// "runs on Metal" -- as `batch` separate matvecs over the same
    /// weights, which is exactly the shape that put IQ4_XS 13.7x behind
    /// with no symptom other than a slow benchmark.
    #[test]
    fn every_metal_matvec_kind_also_has_a_metal_gemm() {
        for &k in QuantKind::ALL {
            assert_eq!(
                metal_matvec_kind_name(k).is_some(),
                metal_mul_mm_kind_supported(k),
                "{}: matvec and mul_mm kernel tables disagree -- one of the two \
                 is a silent slow path",
                k.name()
            );
        }
    }

    /// The kind tables are pure lookups over the name, so a kind that
    /// claims a kernel must name itself the way the Metal launch meta
    /// table is keyed.
    #[test]
    fn metal_kind_names_match_the_quant_kind_names() {
        for &k in QuantKind::ALL {
            if let Some(name) = metal_matvec_kind_name(k) {
                assert_eq!(name, k.name());
            }
        }
    }

    /// THE registry test: a kind with no accelerator kernel, probed
    /// while the model is built, must be recorded as a miss and must be
    /// a seal-time violation -- not silently absorbed by a fallback.
    ///
    /// Runs on any build: the backend is passed explicitly, so it does
    /// not need `--features metal` to ask what Metal would resolve.
    #[test]
    fn a_deliberately_unsupported_kind_trips_the_registry() {
        use crate::kernel_registry::{Backend, Outcome};

        let reg = crate::kernel_registry::Registry::new();
        let loc = std::panic::Location::caller();

        // Supported: Q4_K has both a Metal matvec and a Metal GEMM.
        shaped(QuantKind::Q4K, 64, 256).probe_kernels_for(&reg, Backend::Metal, "ffn_down", loc);
        // Unsupported: no Metal kernel of any kind for IQ2_XXS.
        shaped(QuantKind::IQ2XXS, 64, 256).probe_kernels_for(&reg, Backend::Metal, "ffn_up", loc);

        let report = reg.seal();
        let violations = &report.violations;
        assert_eq!(
            violations.len(),
            2,
            "expected matvec + gemm misses for IQ2_XXS only, got: {:?}",
            report
                .entries
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
        );
        assert!(
            violations
                .iter()
                .all(|v| v.key.kind == Some(QuantKind::IQ2XXS)),
            "Q4_K must not be flagged"
        );
        assert!(
            violations.iter().any(|v| matches!(
                v.outcome,
                Outcome::Miss { fallback, .. } if fallback == "CPU apply_batch"
            )),
            "the report must name the fallback that will actually run"
        );
        let rendered = report.render_violations();
        assert!(rendered.contains("IQ2_XXS"), "{rendered}");
        assert!(rendered.contains("weight_matrix.rs"), "{rendered}");

        // And the host tier it lands on is recorded too: IQ2_XXS has no
        // integer vec_dot either, so it is f32 dequant-dot.
        assert!(
            report.entries.iter().any(|e| e.key.backend == Backend::Cpu
                && e.key.kind == Some(QuantKind::IQ2XXS)
                && matches!(e.outcome, Outcome::Miss { fallback, .. } if fallback == "f32 dequant-dot")),
            "{:?}",
            report.entries.iter().map(|e| e.to_string()).collect::<Vec<_>>()
        );
    }

    /// A supported kind on a selected accelerator produces no violation
    /// at all -- otherwise the signal is noise and gets ignored.
    #[test]
    fn a_fully_supported_model_seals_clean() {
        use crate::kernel_registry::Backend;

        let reg = crate::kernel_registry::Registry::new();
        let loc = std::panic::Location::caller();
        for kind in [QuantKind::Q4K, QuantKind::Q6K, QuantKind::Q8_0] {
            shaped(kind, 64, 256).probe_kernels_for(&reg, Backend::Metal, "ffn_down", loc);
        }
        let report = reg.seal();
        assert!(report.violations.is_empty(), "{}", report.render());
    }

    /// A kind CUDA cannot run at all must be RECORDED as leaving the
    /// GPU, by name, rather than left to a comment in
    /// `apply_batch_with_acts`.
    ///
    /// This test used to probe `Q4K` and expect the fallback
    /// `"CUDA per-position matvec"`, which is what a kind gets when it
    /// has a matvec but no GEMM. **That combination no longer exists on
    /// CUDA.** The K-quants gained a GEMM on 2026-09-04, motivated by
    /// Llama-3.2-3B Q4_K_M running pp512 at 4.88 tok/s against
    /// llama.cpp's 1586.80, and the invariant below now forbids the
    /// combination from coming back.
    ///
    /// So the probe moved to a kind with neither kernel, and the
    /// expected fallback moved with it: with no matvec to loop over
    /// there is no per-position loop, and the whole matmul leaves for
    /// the host.
    ///
    /// It has moved three times. `Q5_0` was that kind until
    /// 2026-09-05, when it gained both; `Q2_K` was until 2026-09-09,
    /// when it and Q3_K did. `Q4_1` is the hole now, and the next row
    /// of the coverage table in `docs/plans/cpu-cuda-parity.md` §6 --
    /// which is the point: the test names a real hole and stops
    /// compiling a comment. When Q4_1 lands, this probe moves again.
    #[test]
    fn a_kind_cuda_cannot_run_is_recorded_as_leaving_the_gpu() {
        use crate::kernel_registry::{op, Backend, Outcome};

        let reg = crate::kernel_registry::Registry::new();
        let loc = std::panic::Location::caller();
        shaped(QuantKind::Q4_1, 64, 256).probe_kernels_for(&reg, Backend::Cuda, "ffn_down", loc);
        let report = reg.seal();
        assert!(
            report.entries.iter().any(|e| e.key.backend == Backend::Cuda
                && e.key.op == op::GEMM_PREFILL
                && matches!(
                    e.outcome,
                    Outcome::Miss { fallback, .. } if fallback == "CPU apply_batch"
                )),
            "{}",
            report.render()
        );
    }

    /// CUDA's matvec set and its GEMM set are now the same, and that is
    /// worth pinning: a kind that can be decoded on the GPU but not
    /// prefilled there is the shape that cost 325x, and it went
    /// unnoticed because a fallback still answers correctly.
    ///
    /// If a future kind gains a matvec without a GEMM, this fails and
    /// names it, rather than a benchmark noticing months later.
    #[test]
    fn a_cuda_kind_with_a_matvec_also_has_a_gemm() {
        for kind in QuantKind::ALL {
            if cuda_matvec_kind_supported(*kind) {
                assert!(
                    cuda_mul_mm_kind_supported(*kind),
                    "{kind:?} can be decoded on CUDA but not prefilled there, \
                     which decomposes a prefill into one matvec launch per position"
                );
            }
        }
    }

    /// An F32 weight has no quantized kernel by construction; the probe
    /// records the host GEMV but must not call it a violation, or every
    /// MoE router would fail a strict run.
    #[test]
    fn an_f32_weight_is_recorded_without_being_a_violation() {
        use crate::kernel_registry::Backend;

        let reg = crate::kernel_registry::Registry::new();
        let m = WeightMatrix::F32(Tensor::new(vec![0.0; 64 * 32], vec![64, 32]));
        m.probe_kernels_for(
            &reg,
            Backend::Metal,
            "moe_router",
            std::panic::Location::caller(),
        );
        let report = reg.seal();
        assert!(!report.misses.is_empty());
        assert!(report.violations.is_empty(), "{}", report.render());
    }
}

#[cfg(test)]
mod int_dot_default_tests {
    use super::{IntDotShape, IntDotTier};

    /// The int-dot rule follows the kernels that exist, per workload,
    /// not the wish that every architecture had every kernel.
    ///
    /// Taking the MATVEC half where the interleaved kernels do not exist
    /// selects a scalar integer loop and skips the AVX2 f32 dot that
    /// does, which measured 4x to 8.8x of x86 decode (#127). Adding AVX2
    /// GEMMs (#152) does not change that: they are batch kernels, and
    /// the matvec half of x86 is still the f32 dot's.
    #[test]
    fn the_matvec_half_is_taken_only_where_its_kernels_are() {
        assert_eq!(
            super::int_dot_tier_here().matvec,
            cfg!(target_arch = "aarch64"),
            "the matvec half is aarch64's (i8mm, interleave-8 NEON) and nowhere else; \
             x86 measured 4x to 8.8x slower with it on"
        );
    }

    /// The BATCH half is not a `cfg!` claim: it asks the kernels.
    ///
    /// A host may only be told the batch tier is a win if `frink_quant`
    /// reports a SIMD batch GEMM at the width this host packs with. That
    /// is what stops the two structures — the list of architectures
    /// believed to have kernels, and the kernels — from drifting apart,
    /// which is how the 4x-to-8.8x regression happened in the first
    /// place.
    ///
    /// The probe is [`frink_quant::batch_gemm_is_accelerated`] and not
    /// `interleaved_gemm_is_accelerated`, because those are different
    /// questions and this one asked the narrower of the two. The
    /// interleave-8 predicate is about the quad kernels; a pre-i8mm
    /// aarch64 host runs a width-4 `dotprod` GEMM for four of the five
    /// kinds, so it has a SIMD batch GEMM while the interleave-8
    /// predicate says it does not.
    #[test]
    fn the_batch_half_is_taken_only_where_a_simd_gemm_answers_for_it() {
        assert_eq!(
            super::int_dot_tier_here().batch_gemm,
            frink_quant::batch_gemm_is_accelerated(frink_quant::preferred_interleave())
                && cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
            "the batch half must agree with the kernel probe, not with a written-down list"
        );
    }

    /// The Q5_K batch gate asks the kernels: wherever the Q5_K `x4`
    /// GEMM has a SIMD kernel the Kx8 path is taken, whatever the
    /// architecture. A `cfg!` alone here is the defect this test exists
    /// for (8.56x on x86 prefill, 2026-09-15).
    #[test]
    fn the_q5k_batch_path_is_taken_wherever_its_simd_gemm_exists() {
        let interleave = frink_quant::q5_kx8_interleave();
        if frink_quant::q5_kx8_gemm_uses_acts_x4(interleave) {
            assert!(super::q5k_batch_takes_kx8(interleave));
        }
        // And on a host with no such kernel, the gate agrees with the
        // Q4_K arm's rule, which is "aarch64 always".
        assert!(super::q5k_batch_takes_kx8(interleave) || !cfg!(target_arch = "aarch64"));
    }

    /// The two probes are not interchangeable, and this pins the
    /// difference so neither can quietly be swapped for the other.
    ///
    /// On a host with a width-4 SIMD batch GEMM and no interleave-8 one
    /// — every pre-i8mm aarch64 host, which is every M1 Mac, every
    /// A14-and-earlier iPhone, and the Cortex-A55-class cores that are
    /// still most of the Android fleet — the wider predicate says yes
    /// and the narrower says no. Reading the narrower as "is the batch
    /// tier a win" is what made this test fail on an M1 while the code
    /// was running a SIMD GEMM the whole time.
    #[test]
    fn the_batch_probe_is_wider_than_the_interleave_8_one() {
        for width in [4usize, 8] {
            assert!(
                frink_quant::batch_gemm_is_accelerated(width)
                    || !frink_quant::interleaved_gemm_is_accelerated(width),
                "the batch probe must answer yes wherever the interleave-8 one does"
            );
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            assert!(
                frink_quant::batch_gemm_is_accelerated(4),
                "a dotprod host runs the width-4 sdot GEMM for Q4_K/Q5_K/Q8_0/Q4_0"
            );
            assert!(
                !frink_quant::interleaved_gemm_is_accelerated(4),
                "the interleave-8 predicate is about the quad kernels only"
            );
        }
    }

    /// `int_dot_is_a_win_here` — the thing `default_cpu_int_dot_on`
    /// consults — is the OR of the two halves, so a host with only the
    /// batch half still gets the env default it needs to reach it.
    #[test]
    fn the_default_is_on_when_either_half_is_a_win() {
        let tier = super::int_dot_tier_here();
        assert_eq!(
            super::int_dot_is_a_win_here(),
            tier.matvec || tier.batch_gemm
        );
    }

    /// `covers` must actually separate the two shapes, in both
    /// directions — otherwise every call site below asks a question with
    /// one answer and the split is decoration.
    #[test]
    fn covers_answers_per_shape_rather_than_per_host() {
        let matvec_only = IntDotTier {
            matvec: true,
            batch_gemm: false,
        };
        let batch_only = IntDotTier {
            matvec: false,
            batch_gemm: true,
        };
        assert!(matvec_only.covers(IntDotShape::Matvec));
        assert!(!matvec_only.covers(IntDotShape::BatchGemm));
        assert!(!batch_only.covers(IntDotShape::Matvec));
        assert!(batch_only.covers(IntDotShape::BatchGemm));
    }
}
