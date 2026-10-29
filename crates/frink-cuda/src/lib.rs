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

/// The per-kind CUDA matvec (decode) kernel table. Always compiled,
/// for the same reason [`mul_mm`] is: it is kernel text plus three
/// strings per row, and `frink-core` derives its CUDA decode
/// capability from it on builds that do not link `cudarc`.
pub mod matvec_kinds;

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

#[cfg(feature = "cuda")]
pub mod attn;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod graph;
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
