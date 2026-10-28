//! frink-metal: Apple Silicon GPU capability detection (always
//! compiled, always tested) plus an optional, feature-gated Metal
//! compute execution path.
//!
//! Build without any GPU support (the default): `cargo build -p frink-metal`.
//! Build with the Metal scaffolding included (macOS only):
//! `cargo build -p frink-metal --features metal`.
//!
//! Unlike `frink-cuda`'s CUDA path (which had to be verified against
//! rented hardware since this development machine has no NVIDIA GPU),
//! the `metal` feature's kernels have been run directly on the real
//! Apple Silicon GPU this project is developed on -- see `gpu.rs`'s
//! module docs for exactly what's been verified and against what.

pub mod capability;

#[cfg(feature = "metal")]
pub mod gpu;

/// PrismML's folded Hadamard rotation as a device kernel.
#[cfg(feature = "metal")]
pub mod hadamard;

/// The gated delta-net recurrence as a device kernel.
#[cfg(feature = "metal")]
pub mod gdn;

/// The gated delta-net recurrence a CHUNK of rows at a time, which is
/// what makes it worth putting on the device at all.
#[cfg(feature = "metal")]
pub mod gdn_chunk;

/// The HEAD of a recurrent layer -- convolution, l2 norms, gates --
/// which is what the host still does between the QKV projection and
/// the recurrence, and so what keeps a layer from being one submission.
#[cfg(feature = "metal")]
pub mod gdn_head;

/// A recurrent layer's whole branch in ONE command buffer: the head,
/// the recurrence, the gated norm and the output projection.
#[cfg(feature = "metal")]
pub mod gdn_branch;

/// Shared-storage buffers reused across launches, because allocating
/// them per launch is host time the GPU ledger cannot see.
#[cfg(feature = "metal")]
pub(crate) mod scratch_pool;

/// PrismML's PTQ1_0 trit format on Metal (`ternary::PTQ1_0Dequant`).
pub mod ternary;

#[cfg(feature = "metal")]
pub mod attn;

/// The KV wire format: dtypes, append and dequant kernels. Split from
/// `attn` so the store and the softmax are separate concepts.
#[cfg(feature = "metal")]
pub mod kv_wire;

#[cfg(feature = "metal")]
pub mod rope;

#[cfg(feature = "metal")]
mod fa_vec_decode;

#[cfg(feature = "metal")]
pub mod decode_dense;

#[cfg(feature = "metal")]
pub mod resident_act;

#[cfg(feature = "metal")]
mod resident_cache;

/// One Shared buffer holding a pool of routed-expert slots, aliased by
/// the weight cache so unified memory holds each streamed expert once.
#[cfg(feature = "metal")]
pub mod expert_slab;

#[cfg(feature = "metal")]
pub mod greedy_fold;

#[cfg(feature = "metal")]
mod dispatch;

#[cfg(feature = "metal")]
mod mem_ranges;

#[cfg(feature = "metal")]
pub(crate) mod timing;

#[cfg(feature = "metal")]
pub(crate) mod kernel_timing;

#[cfg(all(test, feature = "metal"))]
mod kernel_bench;

#[cfg(feature = "metal")]
mod moe_ids;

#[cfg(feature = "metal")]
pub use dispatch::{metal_encode_stats, metal_encode_stats_reset, EncodeStats};

#[cfg(feature = "metal")]
pub mod elem;

#[cfg(feature = "metal")]
pub mod norm;

#[cfg(feature = "metal")]
pub mod embd;

pub use capability::MetalProfile;
