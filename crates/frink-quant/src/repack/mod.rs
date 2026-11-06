//! Interleaved Q4_K / Q5_K × Q8_K and Q8_0 × Q8 GEMV (llama.cpp repack layouts).
//!
//! - Q4_K: packs 8 rows into `block_q4_Kx8` (`make_block_q4_Kx8`).
//! - Q5_K: packs 8 rows into `block_q5_Kx8` (`make_block_q5_Kx8`).
//! - Q8_0: packs 4 rows into `block_q8_0x4` (`make_block_q8_0x4`) with
//!   4-byte interleave for NEON SDOT `ggml_gemv_q8_0_4x4_q8_0`.
//! - Q4_0: packs 4 rows into `block_q4_0x4` (`make_block_q4_0x4`) with
//!   4-byte interleave + XOR `0x88888888` for `ggml_gemv_q4_0_4x4_q8_0`.
//!
//! Gated on `FRINK_CPU_INT_DOT`, which `frink` and `frink-server`
//! turn on by default (`=0` opts out); off in the library so golden
//! cross-validation stays reference-exact.

mod common;
mod q4_0x4;
mod q4_kx8;
mod q5_kx8;
mod q6_kx8;
mod q8_0x4;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::sdot;

#[cfg(test)]
mod tests;

pub use common::*;
pub use q4_0x4::*;
pub use q4_kx8::*;
pub use q5_kx8::*;
pub use q6_kx8::*;
pub use q8_0x4::*;
