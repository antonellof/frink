//! The per-quant-kind rows of the CUDA `mul_mm` dispatch table, one
//! module per format family.
//!
//! [`crate::mul_mm`] holds the GEMM body, the geometry constants and the
//! emitter; everything that varies per format lives here. Each row is a
//! [`MulMmKind`](crate::mul_mm::MulMmKind): a CUDA C `frink_dequant_sub`
//! and, beside it in the same struct, the Rust twin of that function.
//!
//! Split by family rather than one file per kind because the families
//! are what actually share code: the legacy formats share a 32-element
//! block and `nl == 2`, the K-quants share a 256-element super-block
//! and (for two of them) the 6-bit scale/min unpack, and the codebook
//! formats share the `__constant__` table seam that the affine kinds do
//! not use at all.
//!
//! The table itself ([`crate::mul_mm::KINDS`]) stays in `mul_mm`, so
//! there is exactly one list and a row added here without a row there
//! compiles to dead code that clippy names.

pub mod codebook;
pub mod iq;
pub mod kquant;
pub mod legacy;
pub mod ternary;

pub use codebook::{IQ4_NL, IQ4_XS, MXFP4};
pub use iq::{IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS};
pub use kquant::{Q2_K, Q3_K, Q4_K, Q5_K, Q6_K};
pub use legacy::{Q4_0, Q4_1, Q5_0, Q5_1, Q8_0};
pub use ternary::{PTQ1_0, TQ1_0};
