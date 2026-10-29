//! Device-resident copies of host buffers that live for the process,
//! keyed by host address and length and checked against a sample of
//! their bytes: the quantized weights every matvec and GEMM reads, and
//! the small f32 vectors beside them (norm weights, biases, RoPE
//! divisors) that the resident decode and prefill stacks read once per
//! layer.
//!
//! The f32 half exists because those stacks used to upload every norm
//! weight on every call: two to seven host-to-device copies per layer
//! per token for vectors that never change. ONE cache function serves
//! both element types, over two maps, so the identity check below is
//! the same check for a 12 KB norm as for a 50 MB projection.

use crate::gpu::CudaError;
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) struct ResidentCuda<T> {
    pub(crate) slice: CudaSlice<T>,
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

/// The quantized-weight instance, under the name every caller uses.
pub(crate) type ResidentCudaWeights = ResidentCuda<u8>;

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
unsafe impl<T> Send for ResidentCuda<T> {}
unsafe impl<T> Sync for ResidentCuda<T> {}

type CacheMap<T> = HashMap<(usize, usize), Arc<ResidentCuda<T>>>;

static CUDA_WEIGHT_CACHE: Mutex<Option<CacheMap<u8>>> = Mutex::new(None);
static CUDA_F32_CACHE: Mutex<Option<CacheMap<f32>>> = Mutex::new(None);

/// The bytes of an f32 slice, for the fingerprint.
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and every bit pattern is a valid u8;
    // the view covers exactly `size_of_val(v)` bytes of `v`, which
    // outlives the returned borrow.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// The one body both caches run.
fn resident_in<T: DeviceRepr + Unpin + Clone>(
    map: &Mutex<Option<CacheMap<T>>>,
    dev: &Arc<CudaDevice>,
    data: &[T],
    bytes: &[u8],
) -> Result<Arc<ResidentCuda<T>>, CudaError> {
    let key = (data.as_ptr() as usize, data.len());
    let fingerprint = WeightFingerprint::of(bytes);
    // A panic elsewhere while holding the lock leaves the map whole
    // (every edit is one insert or remove), so a poisoned lock is still
    // a usable cache.
    let mut guard = map.lock().unwrap_or_else(|p| p.into_inner());
    let cache = guard.get_or_insert_with(HashMap::new);
    if let Some(cached) = cache.get(&key) {
        if cached.fingerprint == fingerprint {
            return Ok(cached.clone());
        }
        // Same address and length, different bytes: the buffer this
        // entry was uploaded from is gone. Re-upload rather than serve
        // another tensor's weights.
        cache.remove(&key);
    }
    let slice = crate::xfer::htod(dev, "resident upload", data)?;
    let cached = Arc::new(ResidentCuda { slice, fingerprint });
    cache.insert(key, cached.clone());
    Ok(cached)
}

/// A quantized weight buffer on the device, uploaded on first use.
pub(crate) fn resident_cuda_weights(
    dev: &Arc<CudaDevice>,
    weights: &[u8],
) -> Result<Arc<ResidentCudaWeights>, CudaError> {
    resident_in(&CUDA_WEIGHT_CACHE, dev, weights, weights)
}

/// A long-lived f32 vector (a norm weight, a bias, a RoPE divisor
/// table) on the device, uploaded on first use. The caller must hold
/// the returned `Arc` until the stream has finished with it.
pub(crate) fn resident_cuda_f32(
    dev: &Arc<CudaDevice>,
    v: &[f32],
) -> Result<Arc<ResidentCuda<f32>>, CudaError> {
    resident_in(&CUDA_F32_CACHE, dev, v, f32_bytes(v))
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
