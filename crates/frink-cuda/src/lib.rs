//! frink-cuda: hardware capability detection (always compiled, always
//! tested) plus an optional, feature-gated CUDA execution path.
//!
//! Build without any GPU support (the default): `cargo build -p frink-cuda`.
//! Build with the CUDA scaffolding included: `cargo build -p frink-cuda --features cuda`.
//!
//! The `cuda` feature compiles cleanly in this development sandbox
//! (which has neither a CUDA toolkit nor a GPU) because `cudarc` is
//! configured for dynamic loading -- the driver and NVRTC libraries
//! are `dlopen`'d at runtime, not linked at build time. That means
//! "this crate compiles with `--features cuda`" is a true, checked
//! fact. It does **not** mean the CUDA kernels in `gpu.rs` have ever
//! executed successfully; see that module's docs for exactly what has
//! and has not been verified.

pub mod capability;
pub mod coalesced_twin;
/// Syncs and copies per decode token / prefill step (`FRINK_CUDA_COUNT`).
pub mod counters;
/// One-row (decode) attention split across the keys: kernel text and
/// its host twin, always compiled so the host check can run it.
pub mod decode_attn;
/// Scalar twins of the q8_1 quantizer and the dp4a matvecs.
pub mod dp4a_twin;
/// The embedding gather (`get_rows`): kernel text built from each
/// kind's `mul_mm` dequant, and its scalar twin. Always compiled, so
/// `frink-core`'s kernel matrix reads the set on a CPU-only build.
pub mod get_rows;
/// The IQ grid tables as CUDA C, generated from `frink_quant::iq_tables`.
pub mod iq_grids;
/// Host wall time of a CUDA step by phase (`FRINK_CUDA_TIME=1`).
pub mod phases;
/// The routed-expert kernels (`mul_mv_id`, `mul_mm_id`): text, the
/// routing map and the twins.
pub mod moe_id;

/// The per-kind CUDA matvec (decode) kernel table. Always compiled,
/// for the same reason [`mul_mm`] is: it is kernel text plus three
/// strings per row, and `frink-core` derives its CUDA decode
/// capability from it on builds that do not link `cudarc`.
pub mod matvec_kinds;

/// The int8 prefill GEMM (llama.cpp's MMQ) and its q8_1 activations.
pub mod mmq;
/// The `mul_mm` kernel source and its per-quant-kind dispatch table.
/// Always compiled: it is CUDA C *text* plus a scalar twin, neither of
/// which needs `cudarc`, so the default `cargo test -p frink-cuda` run
/// on a GPU-less host still exercises the arithmetic the kernel encodes.
/// Only the launch path ([`mul_mm_launch`]) is feature-gated.
pub mod mul_mm;
/// One module per quant-format family, holding the [`mul_mm::KINDS`]
/// rows. Split out of `mul_mm.rs` so that adding a format is a new file
/// rather than a new section of a file nobody wants to open.
pub mod mul_mm_kinds;
pub mod mul_mm_ref;
/// The tensor-core body for `mul_mm` (text, always compiled; launched
/// by `mul_mm_launch` on `sm_80` and up).
pub mod mul_mm_tc;
/// Prefill attention on the tensor cores: kernel text, launch geometry
/// and its host twin, always compiled so the host check can run it. Its
/// `launch` module (behind `cuda`) runs it on `sm_80` and up.
pub mod prefill_attn_tc;

#[cfg(feature = "cuda")]
pub mod attn;
/// The dp4a decode matvecs on the device, the default decode path.
#[cfg(feature = "cuda")]
mod dp4a_launch;
#[cfg(feature = "cuda")]
mod gen_launch;
#[cfg(feature = "cuda")]
pub mod get_rows_launch;
/// The gated activations of a fused FFN (SwiGLU, GeGLU, gpt-oss's).
#[cfg(feature = "cuda")]
mod glu;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod graph;
/// The decode matvec launch path: plan, enqueue, multi and fused FFN.
#[cfg(feature = "cuda")]
mod matvec_launch;
#[cfg(feature = "cuda")]
pub mod moe_id_launch;
#[cfg(feature = "cuda")]
pub mod mul_mm_launch;
/// One dense prefill layer resident on the device (#259).
#[cfg(feature = "cuda")]
pub mod prefill;
/// Process-resident device copies of host weight buffers.
#[cfg(feature = "cuda")]
mod resident;
/// The counted door every bus copy and host wait goes through.
#[cfg(feature = "cuda")]
mod xfer;

pub use capability::{HardwareProfile, SimdCaps};
