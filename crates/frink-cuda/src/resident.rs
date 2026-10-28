//! Device-resident copies of host buffers that live for the process:
//! the quantized weights every matvec and GEMM reads, keyed by host
//! address and length and checked against a sample of their bytes.
//!
//! Moved out of `gpu.rs` unchanged; `gpu.rs` keeps the launch plumbing.

use crate::gpu::CudaError;

pub(crate) struct ResidentCudaWeights {
    pub(crate) slice: cudarc::driver::CudaSlice<u8>,
    #[allow(dead_code)] // Kept for diagnostics / future eviction logic.
    nbytes: usize,
    /// What the bytes at the key looked like when they were uploaded.
    /// A host buffer can be freed and its address handed to a different
    /// buffer of the same length, and `(pointer, len)` cannot tell the
    /// two apart: the tensor-core `mul_mm` hardware test multiplied one
    /// fixture's weights by another fixture's activations for exactly
    /// that reason (2026-09-17), and a merged LoRA or any other
    /// short-lived matrix could do the same in a server. Production
    /// weights are one mmap for the life of the process, which is why
    /// nothing had noticed. This is the same identity hazard
    /// `frink-metal/src/resident_act.rs` (#166) closed for
    /// activations, closed the cheap way for weights: a sample of the
    /// bytes, compared on every hit.
    fingerprint: WeightFingerprint,
}

/// A cheap sample of a weight buffer: its first and last 64 bytes and
/// 64 from the middle. Not a hash of the whole tensor (that is
/// gigabytes per token); enough that a different buffer at the same
/// address is caught unless it agrees on 192 sampled bytes.
#[derive(PartialEq, Eq, Clone, Debug)]
struct WeightFingerprint {
    head: [u8; 64],
    mid: [u8; 64],
    tail: [u8; 64],
}

impl WeightFingerprint {
    fn of(bytes: &[u8]) -> Self {
        fn take(bytes: &[u8], at: usize) -> [u8; 64] {
            let mut out = [0u8; 64];
            let at = at.min(bytes.len());
            let n = (bytes.len() - at).min(64);
            out[..n].copy_from_slice(&bytes[at..at + n]);
            out
        }
        Self {
            head: take(bytes, 0),
            mid: take(bytes, bytes.len() / 2),
            tail: take(bytes, bytes.len().saturating_sub(64)),
        }
    }
}

// SAFETY: slices live on the process-wide shared CudaDevice and are
// only read by kernels; same sharing model as Metal's ResidentWeightBuffer.
unsafe impl Send for ResidentCudaWeights {}
unsafe impl Sync for ResidentCudaWeights {}

type CudaWeightCacheKey = (usize, usize);
type CudaWeightCacheMap =
    std::collections::HashMap<CudaWeightCacheKey, std::sync::Arc<ResidentCudaWeights>>;

static CUDA_WEIGHT_CACHE: std::sync::Mutex<Option<CudaWeightCacheMap>> =
    std::sync::Mutex::new(None);

pub(crate) fn resident_cuda_weights(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    weights: &[u8],
) -> Result<std::sync::Arc<ResidentCudaWeights>, CudaError> {
    let key = (weights.as_ptr() as usize, weights.len());
    let fingerprint = WeightFingerprint::of(weights);
    let mut guard = CUDA_WEIGHT_CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some(cached) = cache.get(&key) {
        if cached.fingerprint == fingerprint {
            return Ok(cached.clone());
        }
        // Same address and length, different bytes: the buffer this
        // entry was uploaded from is gone. Re-upload rather than serve
        // another tensor's weights.
        cache.remove(&key);
    }
    let slice = dev
        .htod_copy(weights.to_vec())
        .map_err(|e| CudaError::Launch(format!("{e:?}")))?;
    let cached = std::sync::Arc::new(ResidentCudaWeights {
        slice,
        nbytes: weights.len(),
        fingerprint,
    });
    cache.insert(key, cached.clone());
    Ok(cached)
}

#[cfg(test)]
mod fingerprint_tests {
    use super::WeightFingerprint;

    #[test]
    fn a_buffer_that_differs_anywhere_sampled_has_a_different_fingerprint() {
        let base: Vec<u8> = (0..1000u32).map(|i| (i * 7) as u8).collect();
        let f = WeightFingerprint::of(&base);
        assert_eq!(f, WeightFingerprint::of(&base.clone()));
        for at in [0usize, 63, 500, 531, 936, 999] {
            let mut other = base.clone();
            other[at] ^= 0xff;
            assert_ne!(f, WeightFingerprint::of(&other), "byte {at} is sampled");
        }
        // Short buffers are sampled whole.
        let short = vec![1u8, 2, 3];
        assert_ne!(
            WeightFingerprint::of(&short),
            WeightFingerprint::of(&[1u8, 2, 4])
        );
    }
}
