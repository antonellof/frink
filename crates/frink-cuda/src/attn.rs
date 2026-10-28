//! CUDA GQA decode (B=1 online-softmax). Structural mirror of
//! `frink-metal::attn::gqa_decode` — one threadgroup per query head,
//! online rescale over the causal KV cache. Weights/QKV projections
//! still enter via `gpu::launch_matvec*`; this only covers the
//! attention reduction itself.
//!
//! Status: compiles under `--features cuda`; numerical parity is
//! gated on hardware (`#[ignore]` tests). The decode kernel is
//! warp-parallel (one warp per head: warp-reduced q·k, per-lane `acc`
//! slices). Wired into `Decoder::forward_token` behind `FRINK_CUDA_GQA=1`
//! (host fallback on error). [`CudaKvBuffers`] keeps K/V resident across
//! tokens so each call only uploads the new Q (+ optional K/V append)
//! instead of the full cache — one step toward one-sync-per-token.
//! CUDA-graph capture lives in [`crate::graph`].

use super::gpu::{ensure_module_loaded, shared_device, CudaError};
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Mutex;

/// Online-softmax GQA decode: `q` [n_heads * head_dim],
/// `k_cache`/`v_cache` [seq_len * n_kv_heads * head_dim].
pub const GQA_DECODE_KERNEL_SRC: &str = r#"
// NVRTC compiles without the C headers, so `INFINITY` is not defined:
// it is a <math.h> macro, and this kernel used it for the online-softmax
// running maximum. The kernel therefore never compiled, and
// `FRINK_CUDA_GQA=1` failed at launch for anyone who set it. Found by
// running the `#[ignore]`d hardware tests on a real GPU for the first
// time, 2026-09-04. The bit pattern is the same value without the
// header dependency.
__device__ __forceinline__ float frink_inf() {
    return __int_as_float(0x7f800000);
}

extern "C" __global__ void gqa_decode(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    float* out,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int seq_len
) {
    int h = blockIdx.x;
    if (h >= n_heads || seq_len <= 0) return;
    int lane = threadIdx.x;      // 0..W-1 (one warp per head)
    int W = blockDim.x;          // == 32
    int group_size = n_heads / max(n_kv_heads, 1);
    int kv_h = h / max(group_size, 1);
    float scale = rsqrtf((float)head_dim);
    const float* q_h = q + h * head_dim;

    float acc[8];
    int n_local = 0;
    for (int d = lane; d < head_dim; d += W) { acc[n_local++] = 0.f; }

    float m = -frink_inf();
    float s = 0.f;
    const unsigned mask = 0xffffffffu;

    for (int t = 0; t < seq_len; t++) {
        const float* k_t = k_cache + (t * n_kv_heads + kv_h) * head_dim;
        float pdot = 0.f;
        for (int d = lane; d < head_dim; d += W) pdot += q_h[d] * k_t[d];
        for (int off = W / 2; off > 0; off >>= 1) {
            pdot += __shfl_down_sync(mask, pdot, off);
        }
        float dot = __shfl_sync(mask, pdot, 0);
        float score = dot * scale;
        float m2 = fmaxf(m, score);
        float a = (m == -frink_inf()) ? 0.f : expf(m - m2);
        float b = expf(score - m2);
        s = s * a + b;
        const float* v_t = v_cache + (t * n_kv_heads + kv_h) * head_dim;
        int li = 0;
        for (int d = lane; d < head_dim; d += W) {
            acc[li] = acc[li] * a + b * v_t[d];
            li++;
        }
        m = m2;
    }
    float inv = (s > 0.f) ? (1.f / s) : 0.f;
    float* out_h = out + h * head_dim;
    int li = 0;
    for (int d = lane; d < head_dim; d += W) {
        out_h[d] = acc[li] * inv;
        li++;
    }
}
"#;

/// Process-wide resident K/V buffers for dense CUDA decode. Capacity is
/// fixed at construction; appends grow a host staging mirror and refresh
/// the device buffers (full-prefix HtoD). Avoids re-reading the host
/// `KvCache` layout from the decoder and is the stepping stone to a true
/// device-side kv_append kernel.
pub struct CudaKvBuffers {
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    k_host: Vec<f32>,
    v_host: Vec<f32>,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    seq_len: usize,
}

// SAFETY: buffers live on the shared CudaDevice; append + decode run
// on the default stream from one decode thread at a time (server uses
// spawn_blocking per request).
unsafe impl Send for CudaKvBuffers {}
unsafe impl Sync for CudaKvBuffers {}

impl CudaKvBuffers {
    pub fn new(n_kv_heads: usize, head_dim: usize, capacity: usize) -> Result<Self, CudaError> {
        let dev = shared_device()?;
        let elems = capacity
            .checked_mul(n_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| CudaError::Launch("CudaKvBuffers size overflow".into()))?;
        let k = dev
            .alloc_zeros::<f32>(elems)
            .map_err(|e| CudaError::Launch(format!("kv k alloc: {e:?}")))?;
        let v = dev
            .alloc_zeros::<f32>(elems)
            .map_err(|e| CudaError::Launch(format!("kv v alloc: {e:?}")))?;
        Ok(Self {
            k,
            v,
            k_host: Vec::with_capacity(elems),
            v_host: Vec::with_capacity(elems),
            n_kv_heads,
            head_dim,
            capacity,
            seq_len: 0,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
        self.k_host.clear();
        self.v_host.clear();
    }

    /// Append one token's K and V (`n_kv_heads * head_dim` each).
    pub fn append(&mut self, k_tok: &[f32], v_tok: &[f32]) -> Result<(), CudaError> {
        let row = self.n_kv_heads * self.head_dim;
        if k_tok.len() != row || v_tok.len() != row {
            return Err(CudaError::Launch(
                "CudaKvBuffers append length mismatch".into(),
            ));
        }
        if self.seq_len >= self.capacity {
            return Err(CudaError::Launch("CudaKvBuffers capacity exhausted".into()));
        }
        self.k_host.extend_from_slice(k_tok);
        self.v_host.extend_from_slice(v_tok);
        self.seq_len += 1;
        let dev = shared_device()?;
        // Refresh device prefix (capacity-sized buffers; only prefix used).
        let mut k_full = self.k_host.clone();
        k_full.resize(self.capacity * row, 0.0);
        let mut v_full = self.v_host.clone();
        v_full.resize(self.capacity * row, 0.0);
        self.k = crate::xfer::htod_owned(&dev, "kv k refresh", k_full)?;
        self.v = crate::xfer::htod_owned(&dev, "kv v refresh", v_full)?;
        Ok(())
    }
}

static LAYER_KV: Mutex<Option<Vec<CudaKvBuffers>>> = Mutex::new(None);

/// Ensure per-layer resident KV exists for `n_layers` (clears on shape change).
pub fn ensure_layer_kv(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    capacity: usize,
) -> Result<(), CudaError> {
    let mut guard = LAYER_KV.lock().unwrap();
    let needs_new = match guard.as_ref() {
        None => true,
        Some(v) => {
            v.len() != n_layers
                || v.first().is_none_or(|b| {
                    b.n_kv_heads != n_kv_heads || b.head_dim != head_dim || b.capacity != capacity
                })
        }
    };
    if needs_new {
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(CudaKvBuffers::new(n_kv_heads, head_dim, capacity)?);
        }
        *guard = Some(layers);
    }
    Ok(())
}

/// Clear all layer KV seq lens (new sequence).
pub fn clear_layer_kv() {
    if let Some(layers) = LAYER_KV.lock().unwrap().as_mut() {
        for b in layers.iter_mut() {
            b.clear();
        }
    }
}

/// Run GQA against resident KV. When `host_seq` matches the buffer's
/// length after appending the latest token, uses the device cache. If the
/// resident seq drifts from the host cache, falls back to a full-cache
/// upload via [`launch_gqa_decode`].
#[allow(clippy::too_many_arguments)] // mirror launch_gqa_decode + layer + host_seq
pub fn launch_gqa_decode_resident(
    layer: usize,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    host_seq: usize,
) -> Result<Vec<f32>, CudaError> {
    let row = n_kv_heads * head_dim;
    if host_seq == 0 {
        return Ok(vec![0.0; n_heads * head_dim]);
    }
    if k_cache.len() < host_seq * row || v_cache.len() < host_seq * row {
        return Err(CudaError::Launch(
            "gqa resident cache length mismatch".into(),
        ));
    }
    let k_tok = &k_cache[(host_seq - 1) * row..host_seq * row];
    let v_tok = &v_cache[(host_seq - 1) * row..host_seq * row];

    let mut guard = LAYER_KV.lock().unwrap();
    let Some(layers) = guard.as_mut() else {
        return launch_gqa_decode(q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, host_seq);
    };
    let buf = layers
        .get_mut(layer)
        .ok_or_else(|| CudaError::Launch(format!("CudaKvBuffers missing layer {layer}")))?;

    if buf.seq_len + 1 != host_seq {
        // Drift (mixed host/GPU path, capacity rebuild, etc.): resync
        // from the host cache rather than appending blindly.
        buf.clear();
        for t in 0..host_seq {
            let ks = &k_cache[t * row..(t + 1) * row];
            let vs = &v_cache[t * row..(t + 1) * row];
            buf.append(ks, vs)?;
        }
    } else {
        buf.append(k_tok, v_tok)?;
    }
    let seq_len = buf.seq_len;
    let dev = shared_device()?;
    ensure_module_loaded(
        &dev,
        GQA_DECODE_KERNEL_SRC,
        "frink_gqa_decode",
        "gqa_decode",
    )?;
    let func = dev
        .get_func("frink_gqa_decode", "gqa_decode")
        .ok_or_else(|| CudaError::Launch("gqa_decode func missing after load".into()))?;
    let d_q = crate::xfer::htod(&dev, "gqa q upload", q)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(n_heads * head_dim)
        .map_err(|e| CudaError::Launch(format!("gqa out alloc: {e:?}")))?;
    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &d_q,
                &buf.k,
                &buf.v,
                &mut d_out,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("gqa_decode launch: {e:?}")))?;
    }
    crate::xfer::dtoh(&dev, "gqa download", &d_out)
}

/// Launches [`GQA_DECODE_KERNEL_SRC`]. Requires `head_dim <= 256`.
pub fn launch_gqa_decode(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<Vec<f32>, CudaError> {
    if head_dim == 0 || head_dim > 256 {
        return Err(CudaError::Launch(format!(
            "gqa_decode head_dim={head_dim} out of range (1..=256)"
        )));
    }
    if q.len() != n_heads * head_dim {
        return Err(CudaError::Launch("gqa_decode q length mismatch".into()));
    }
    if k_cache.len() != seq_len * n_kv_heads * head_dim
        || v_cache.len() != seq_len * n_kv_heads * head_dim
    {
        return Err(CudaError::Launch("gqa_decode KV length mismatch".into()));
    }
    if seq_len == 0 {
        return Ok(vec![0.0; n_heads * head_dim]);
    }

    let dev = shared_device()?;
    ensure_module_loaded(
        &dev,
        GQA_DECODE_KERNEL_SRC,
        "frink_gqa_decode",
        "gqa_decode",
    )?;
    let func = dev
        .get_func("frink_gqa_decode", "gqa_decode")
        .ok_or_else(|| CudaError::Launch("gqa_decode func missing after load".into()))?;
    let d_q = crate::xfer::htod(&dev, "gqa q upload", q)?;
    let d_k = crate::xfer::htod(&dev, "gqa k upload", k_cache)?;
    let d_v = crate::xfer::htod(&dev, "gqa v upload", v_cache)?;
    let mut d_out = dev
        .alloc_zeros::<f32>(n_heads * head_dim)
        .map_err(|e| CudaError::Launch(format!("gqa out alloc: {e:?}")))?;

    let cfg = LaunchConfig {
        grid_dim: (n_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        func.launch(
            cfg,
            (
                &d_q,
                &d_k,
                &d_v,
                &mut d_out,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
            ),
        )
        .map_err(|e| CudaError::Launch(format!("gqa_decode launch: {e:?}")))?;
    }
    crate::xfer::dtoh(&dev, "gqa download", &d_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_layer_kv_shape_helpers_are_callable() {
        // No GPU required: just ensure the static starts empty and clear is safe.
        clear_layer_kv();
    }

    #[test]
    #[ignore = "needs NVIDIA GPU"]
    fn gqa_decode_runs_on_hardware() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 3;
        let q = vec![0.01f32; n_heads * head_dim];
        let k = vec![0.02f32; seq_len * n_kv_heads * head_dim];
        let v = vec![0.03f32; seq_len * n_kv_heads * head_dim];
        let out =
            launch_gqa_decode(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len).expect("gpu gqa");
        assert_eq!(out.len(), n_heads * head_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
