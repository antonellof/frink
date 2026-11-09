//! What a resident stack holds alive until its one download, and the
//! two ways a layer adds to it.

use super::ProjWeights;
use crate::gpu::{resident_cuda_f32, CudaError};
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Every resident buffer a launch reads, held until the stack's one
/// download has synchronised the stream: the kernels read them
/// asynchronously, and a buffer must not be released under a kernel
/// that is still reading it.
#[derive(Default)]
pub(super) struct Held {
    pub(super) weights: Vec<Arc<crate::gpu::ResidentCudaWeights>>,
    pub(super) vectors: Vec<Arc<crate::resident::ResidentCuda<f32>>>,
    /// Anything else a kernel reads: a layer's expert tables.
    pub(super) keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
}

impl Held {
    /// A long-lived f32 vector (norm weight, bias, RoPE divisors) on
    /// the device, held: uploaded on first use, never per call.
    pub(super) fn vector(
        &mut self,
        dev: &Arc<CudaDevice>,
        v: &[f32],
    ) -> Result<Arc<crate::resident::ResidentCuda<f32>>, CudaError> {
        let r = resident_cuda_f32(dev, v)?;
        self.vectors.push(r.clone());
        Ok(r)
    }

    /// One projection on the device, its weights held.
    pub(super) fn proj(
        &mut self,
        dev: &Arc<CudaDevice>,
        m: &ProjWeights<'_>,
        x: &CudaSlice<f32>,
        batch: usize,
    ) -> Result<CudaSlice<f32>, CudaError> {
        let (out, w) = m.enqueue(dev, x, batch)?;
        self.weights.push(w);
        Ok(out)
    }
}
