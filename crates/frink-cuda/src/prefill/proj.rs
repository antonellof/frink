//! One quantized projection of a resident layer, and the one decision
//! of which kernel multiplies it: the matvec for a single row, the
//! GEMM for a batch.
//!
//! The stack serves both a prefill batch and a decode token, and the
//! two differ in exactly this. A GEMM tile spends all but one of its
//! columns on nothing at `batch == 1` (`crate::mul_mm::worth_a_gemm`),
//! and the matvec kernels are the decode arm that has run on hardware.

use super::MulMmWeights;
use crate::gpu::{enqueue_matvec, CudaError, MatvecLaunch, ResidentCudaWeights};
use crate::mul_mm::MulMmKind;
use crate::mul_mm_launch::enqueue_mul_mm;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// A projection as the stack takes it: the GEMM's view of the bytes,
/// which every projection has (it carries the shape the layer checks),
/// and the matvec's, which a single-row launch needs.
///
/// Both views are built by `frink-core` from one `WeightMatrix`
/// (`cuda_mul_mm_view`, `cuda_matvec_launch`); [`Self::validate`]
/// checks they describe the same bytes, so a caller cannot hand the
/// stack a matvec of one tensor and the shape of another.
#[derive(Clone, Copy)]
pub struct ProjWeights<'a> {
    pub gemm: MulMmWeights<'a>,
    pub matvec: Option<MatvecLaunch<'a>>,
}

impl<'a> ProjWeights<'a> {
    /// A projection the stack can only run as a GEMM.
    pub fn gemm_only(gemm: MulMmWeights<'a>) -> Self {
        Self { gemm, matvec: None }
    }

    pub fn rows(&self) -> usize {
        self.gemm.rows
    }

    pub fn cols(&self) -> usize {
        self.gemm.cols
    }

    pub fn kind(&self) -> &'static MulMmKind {
        self.gemm.kind
    }

    /// Whether a batch of `batch` rows multiplies by the matvec. THE
    /// rule; nothing else in the stack restates it.
    pub fn uses_matvec(batch: usize) -> bool {
        batch == 1
    }

    pub(super) fn validate(&self, batch: usize) -> Result<(), CudaError> {
        if !Self::uses_matvec(batch) {
            return self.gemm.validate(batch);
        }
        let Some(mv) = &self.matvec else {
            return Err(CudaError::Unsupported(format!(
                "resident stack: a one-row step needs a matvec for this {} projection",
                self.gemm.kind.name
            )));
        };
        let same_bytes = std::ptr::eq(mv.weights.as_ptr(), self.gemm.data.as_ptr())
            && mv.weights.len() == self.gemm.data.len();
        if !same_bytes
            || mv.rows != self.gemm.rows
            || mv.row_bytes != self.gemm.row_bytes
            || mv.rows.checked_mul(mv.row_bytes) != Some(mv.weights.len())
            || mv.n_blocks_per_row.checked_mul(self.gemm.kind.block_bytes) != Some(mv.row_bytes)
            || mv.n_blocks_per_row.checked_mul(self.gemm.kind.block_elems) != Some(self.gemm.cols)
        {
            return Err(CudaError::Unsupported(
                "resident stack: the matvec and GEMM views of a projection disagree".to_string(),
            ));
        }
        Ok(())
    }

    /// Multiplies `x` (`batch` rows of `cols`) on the device. Returns
    /// the output and the resident weight buffer, which the caller
    /// holds until the stream has finished.
    pub(super) fn enqueue(
        &self,
        dev: &Arc<CudaDevice>,
        x: &CudaSlice<f32>,
        batch: usize,
    ) -> Result<(CudaSlice<f32>, Arc<ResidentCudaWeights>), CudaError> {
        match (&self.matvec, Self::uses_matvec(batch)) {
            (Some(mv), true) => enqueue_matvec(dev, mv, x),
            (None, true) => Err(CudaError::Unsupported(
                "resident stack: a one-row step needs a matvec".to_string(),
            )),
            (_, false) => {
                let m = &self.gemm;
                enqueue_mul_mm(dev, m.kind, m.data, x, m.rows, m.cols, batch, m.row_bytes)
            }
        }
    }
}
