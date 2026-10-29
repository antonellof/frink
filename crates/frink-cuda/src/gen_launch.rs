//! `launch_<kind>_matvec` for every matvec generated from a GEMM dequant
//! ([`crate::matvec_kinds::generic`]), one table line each. Re-exported
//! from [`crate::gpu`] beside the hand-written kinds' launchers, which
//! `frink-core`'s CUDA launch table names.
//!
//! **NEVER RUN ON A GPU.** See [`crate::matvec_kinds::generic`].

use crate::gpu::{launch_matvec_by_kind, CudaError};

macro_rules! generated_launchers {
    ($($kind:literal, $launch:ident),* $(,)?) => {$(
        #[doc = concat!("Launches the ", $kind, " matvec, generated from its GEMM dequant.")]
        pub fn $launch(
            weights: &[u8],
            x: &[f32],
            rows: usize,
            row_bytes: usize,
            n_blocks_per_row: usize,
        ) -> Result<Vec<f32>, CudaError> {
            launch_matvec_by_kind($kind, weights, x, rows, row_bytes, n_blocks_per_row)
        }
    )*};
}

generated_launchers!(
    "Q4_1",
    launch_q4_1_matvec,
    "Q5_1",
    launch_q5_1_matvec,
    "TQ1_0",
    launch_tq1_0_matvec,
    "PTQ1_0",
    launch_ptq1_0_matvec,
    "IQ2_XXS",
    launch_iq2_xxs_matvec,
    "IQ2_XS",
    launch_iq2_xs_matvec,
    "IQ2_S",
    launch_iq2_s_matvec,
    "IQ3_XXS",
    launch_iq3_xxs_matvec,
    "IQ3_S",
    launch_iq3_s_matvec,
    "IQ1_S",
    launch_iq1_s_matvec,
    "IQ1_M",
    launch_iq1_m_matvec,
);
