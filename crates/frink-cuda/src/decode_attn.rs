//! Attention for ONE query row over a long K/V history: the decode
//! case, split across the keys so a token at position 4000 is not one
//! warp walking 4000 keys in series.
//!
//! The prefill kernel (`prefill::kernels::CAUSAL_GQA_PREFILL_KERNEL_SRC`)
//! gives each (query row, head) one warp and walks its keys in order,
//! which is the right shape for 512 query rows and the wrong one for a
//! single row: at `n_q = 1` Llama-3.2-3B launches 24 warps on a card
//! that holds a few thousand, and each one's loop is a chain of
//! dependent loads, reductions and `expf`s as long as the history. This
//! is llama.cpp's answer for the same case (`fattn-common.cuh`,
//! `flash_attn_combine_results` at the pin, `parallel_blocks` in
//! `launch_fattn`): split the keys into chunks, give every (head, chunk)
//! its own block, and merge the chunks' partial softmaxes in a second,
//! tiny kernel.
//!
//! * `decode_attn_partial_f32`, grid `(n_heads, n_chunks)`, four warps:
//!   warp `w` of chunk `c` takes keys `lo + c*CHUNK + w`, then every
//!   fourth, with the prefill kernel's online softmax and the same
//!   `float4` lane slices of Q and V. The four warps' `(m, l, acc)`
//!   are merged in shared memory and written UNNORMALISED to one
//!   scratch buffer ([`scratch_len`]): `acc` at `[(h * n_chunks + c) *
//!   head_dim]`, then the `(m, l)` pairs at `n_heads * n_chunks *
//!   head_dim + 2 * (h * n_chunks + c)`.
//! * `decode_attn_combine_f32`, grid `n_heads`, `head_dim` threads:
//!   `out = sum_c e^(m_c - M) acc_c / sum_c e^(m_c - M) l_c`.
//!
//! The host twin [`decode_attn_twin`] performs the same f32 operations
//! in the same order, including the shuffle butterfly, so with FP
//! contraction off (`tools/decode_attn_host_check`) the kernel text and
//! the twin agree bit for bit; a GPU contracts, so the hardware test
//! compares with a tolerance.
//!
//! Always compiled: it is text and a twin, and the host check runs it
//! on a build with no CUDA feature.

/// Keys per chunk: one block's share of the history.
pub const CHUNK: usize = 64;
/// Warps per partial block.
pub const WARPS: usize = 4;

/// `(module, entry point)` of the two kernels. One source, two NVRTC
/// modules, because a module registers only the entry points it was
/// loaded with.
pub const PARTIAL: (&str, &str) = ("frink_decode_attn_partial", "decode_attn_partial_f32");
pub const COMBINE: (&str, &str) = ("frink_decode_attn_combine", "decode_attn_combine_f32");

/// The keys a query at position `seq_len - 1` reads: `lo..seq_len`,
/// `lo` the sliding window's start (`seq_len.saturating_sub(w)`, the
/// host's rule) or 0.
pub fn key_range(seq_len: usize, window: Option<usize>) -> (usize, usize) {
    let lo = match window {
        Some(w) if w > 0 && seq_len > w => seq_len - w,
        _ => 0,
    };
    (lo, seq_len)
}

/// Floats of scratch the pair needs: the accumulators, then the
/// `(m, l)` pairs.
pub fn scratch_len(n_heads: usize, n_chunks: usize, head_dim: usize) -> usize {
    n_heads * n_chunks * (head_dim + 2)
}

/// How many chunks the partial kernel's grid has for `n_keys` keys;
/// at least one, so an empty history still writes a (zero) row.
pub fn n_chunks(n_keys: usize) -> usize {
    n_keys.div_ceil(CHUNK).max(1)
}

pub const DECODE_ATTN_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float frink_inf_da() {
    return __int_as_float(0x7f800000);
}

#define FX_DA_CHUNK 64
#define FX_DA_WARPS 4

// q [n_heads, head_dim]; k_all / v_all [>= seq_len, n_kv_heads, head_dim];
// part: [n_heads, n_chunks, head_dim] accumulators, then
// [n_heads, n_chunks, 2] (m, l) pairs, in one scratch buffer.
extern "C" __global__ void decode_attn_partial_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    float* __restrict__ part,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int lo,
    int seq_len,
    int n_chunks,
    float scale,
    float softcap
) {
    __shared__ float sm_m[FX_DA_WARPS];
    __shared__ float sm_l[FX_DA_WARPS];
    __shared__ float sm_acc[FX_DA_WARPS * 256];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int h = blockIdx.x;
    const int c = blockIdx.y;
    if (h >= n_heads || c >= n_chunks) return;
    float* part_acc = part;
    float* part_ml = part + (size_t)n_heads * n_chunks * head_dim;
    const int group_size = n_heads / max(n_kv_heads, 1);
    const int kv_h = h / max(group_size, 1);
    const int d4 = head_dim >> 2;
    const int n_slices = (d4 + 31) >> 5;
    const float4* q_h = (const float4*)(q + (size_t)h * head_dim);
    const unsigned mask = 0xffffffffu;

    float4 qv[2];
    float4 acc[2];
    for (int s = 0; s < 2; s++) {
        const int i = lane + 32 * s;
        if (s < n_slices && i < d4) {
            qv[s] = q_h[i];
        } else {
            qv[s].x = 0.f; qv[s].y = 0.f; qv[s].z = 0.f; qv[s].w = 0.f;
        }
        acc[s].x = 0.f; acc[s].y = 0.f; acc[s].z = 0.f; acc[s].w = 0.f;
    }
    float m = -frink_inf_da();
    float l = 0.f;

    const int c_lo = lo + c * FX_DA_CHUNK;
    int c_hi = c_lo + FX_DA_CHUNK;
    if (c_hi > seq_len) c_hi = seq_len;
    for (int t = c_lo + warp; t < c_hi; t += FX_DA_WARPS) {
        const float4* k_t = (const float4*)(k_all + ((size_t)t * n_kv_heads + kv_h) * head_dim);
        float pdot = 0.f;
        for (int s = 0; s < 2; s++) {
            const int i = lane + 32 * s;
            if (s < n_slices && i < d4) {
                const float4 kv = k_t[i];
                pdot += qv[s].x * kv.x + qv[s].y * kv.y + qv[s].z * kv.z + qv[s].w * kv.w;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            pdot += __shfl_xor_sync(mask, pdot, off);
        }
        float score = pdot * scale;
        if (softcap > 0.f) score = softcap * tanhf(score / softcap);
        const float m2 = fmaxf(m, score);
        const float a = (m == -frink_inf_da()) ? 0.f : expf(m - m2);
        const float b = expf(score - m2);
        l = l * a + b;
        const float4* v_t = (const float4*)(v_all + ((size_t)t * n_kv_heads + kv_h) * head_dim);
        for (int s = 0; s < 2; s++) {
            const int i = lane + 32 * s;
            if (s < n_slices && i < d4) {
                const float4 vv = v_t[i];
                acc[s].x = acc[s].x * a + b * vv.x;
                acc[s].y = acc[s].y * a + b * vv.y;
                acc[s].z = acc[s].z * a + b * vv.z;
                acc[s].w = acc[s].w * a + b * vv.w;
            }
        }
        m = m2;
    }

    if (lane == 0) {
        sm_m[warp] = m;
        sm_l[warp] = l;
    }
    for (int s = 0; s < 2; s++) {
        const int i = lane + 32 * s;
        if (s < n_slices && i < d4) {
            float* dst = sm_acc + warp * 256 + 4 * i;
            dst[0] = acc[s].x; dst[1] = acc[s].y; dst[2] = acc[s].z; dst[3] = acc[s].w;
        }
    }
    __syncthreads();

    // Merge the warps: every thread reads all four (m, l), and owns
    // channels tid and tid + 128 of the chunk's accumulator.
    float big = sm_m[0];
    for (int w = 1; w < FX_DA_WARPS; w++) big = fmaxf(big, sm_m[w]);
    float f[FX_DA_WARPS];
    for (int w = 0; w < FX_DA_WARPS; w++) {
        f[w] = (sm_m[w] == -frink_inf_da()) ? 0.f : expf(sm_m[w] - big);
    }
    const size_t slot = (size_t)h * n_chunks + c;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.f;
        for (int w = 0; w < FX_DA_WARPS; w++) sum += f[w] * sm_acc[w * 256 + d];
        part_acc[slot * head_dim + d] = sum;
    }
    if (threadIdx.x == 0) {
        float ls = 0.f;
        for (int w = 0; w < FX_DA_WARPS; w++) ls += f[w] * sm_l[w];
        part_ml[slot * 2] = big;
        part_ml[slot * 2 + 1] = ls;
    }
}

// out [n_heads, head_dim]. One block per head, one thread per channel.
extern "C" __global__ void decode_attn_combine_f32(
    const float* __restrict__ part,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int n_chunks
) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;
    const float* part_acc = part;
    const float* part_ml = part + (size_t)n_heads * n_chunks * head_dim;
    const float* ml = part_ml + (size_t)h * n_chunks * 2;
    float big = ml[0];
    for (int c = 1; c < n_chunks; c++) big = fmaxf(big, ml[c * 2]);
    float num = 0.f;
    float den = 0.f;
    for (int c = 0; c < n_chunks; c++) {
        const float mc = ml[c * 2];
        const float fc = (mc == -frink_inf_da()) ? 0.f : expf(mc - big);
        num += fc * part_acc[((size_t)h * n_chunks + c) * head_dim + d];
        den += fc * ml[c * 2 + 1];
    }
    out[(size_t)h * head_dim + d] = (den > 0.f) ? num / den : 0.f;
}
"#;

/// The launch geometry, as the enqueue and the host check use it.
pub mod geometry {
    /// `(grid, block)` of the partial kernel.
    pub fn partial(n_heads: usize, n_chunks: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (
            (n_heads as u32, n_chunks as u32, 1),
            ((super::WARPS * 32) as u32, 1, 1),
        )
    }
    /// `(grid, block)` of the combine kernel.
    pub fn combine(n_heads: usize, head_dim: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        ((n_heads as u32, 1, 1), (head_dim as u32, 1, 1))
    }
}

/// The shapes the kernel takes: `head_dim` a multiple of 4, at most
/// 256 (two `float4` per lane, and the shared accumulator's width).
pub fn head_dim_ok(head_dim: usize) -> bool {
    head_dim > 0 && head_dim <= 256 && head_dim.is_multiple_of(4)
}

/// The kernel pair's arithmetic on the host, in its order.
///
/// `q` is `[n_heads, head_dim]`, `k` and `v` hold at least `seq_len`
/// rows of `[n_kv_heads, head_dim]`. Returns `[n_heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn decode_attn_twin(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    window: Option<usize>,
    scale: f32,
    softcap: Option<f32>,
) -> Vec<f32> {
    assert!(head_dim_ok(head_dim), "head_dim {head_dim}");
    let softcap = softcap.filter(|c| *c > 0.0).unwrap_or(0.0);
    let (lo, hi) = key_range(seq_len, window);
    let chunks = n_chunks(hi - lo);
    let group = (n_heads / n_kv_heads.max(1)).max(1);
    let d4 = head_dim / 4;
    let mut out = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let mut part_acc = vec![0f32; chunks * head_dim];
        let mut part_ml = vec![0f32; chunks * 2];
        for c in 0..chunks {
            let c_lo = lo + c * CHUNK;
            let c_hi = (c_lo + CHUNK).min(hi);
            let mut sm_m = [0f32; WARPS];
            let mut sm_l = [0f32; WARPS];
            let mut sm_acc = vec![0f32; WARPS * 256];
            for w in 0..WARPS {
                let mut m = f32::NEG_INFINITY;
                let mut l = 0f32;
                // acc[lane][s] as [lane * 8 + s * 4 + j].
                let mut acc = [[0f32; 8]; 32];
                let mut t = c_lo + w;
                while t < c_hi {
                    let k_t = &k[(t * n_kv_heads + kv_h) * head_dim..][..head_dim];
                    let v_t = &v[(t * n_kv_heads + kv_h) * head_dim..][..head_dim];
                    // Each lane's partial dot over its float4 slices.
                    let mut pd = [0f32; 32];
                    for (lane, p) in pd.iter_mut().enumerate() {
                        for s in 0..2 {
                            let i = lane + 32 * s;
                            if i < d4 {
                                let (qq, kk) = (&q_h[4 * i..4 * i + 4], &k_t[4 * i..4 * i + 4]);
                                *p += qq[0] * kk[0] + qq[1] * kk[1] + qq[2] * kk[2] + qq[3] * kk[3];
                            }
                        }
                    }
                    // The xor butterfly: every lane ends with the same
                    // sum, in the same association.
                    let mut off = 16;
                    while off > 0 {
                        let prev = pd;
                        for (lane, p) in pd.iter_mut().enumerate() {
                            *p = prev[lane] + prev[lane ^ off];
                        }
                        off >>= 1;
                    }
                    let mut score = pd[0] * scale;
                    if softcap > 0.0 {
                        score = softcap * (score / softcap).tanh();
                    }
                    let m2 = m.max(score);
                    let a = if m == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (m - m2).exp()
                    };
                    let b = (score - m2).exp();
                    l = l * a + b;
                    for (lane, acc) in acc.iter_mut().enumerate() {
                        for s in 0..2 {
                            let i = lane + 32 * s;
                            if i < d4 {
                                for j in 0..4 {
                                    acc[s * 4 + j] = acc[s * 4 + j] * a + b * v_t[4 * i + j];
                                }
                            }
                        }
                    }
                    m = m2;
                    t += WARPS;
                }
                sm_m[w] = m;
                sm_l[w] = l;
                for (lane, acc) in acc.iter().enumerate() {
                    for s in 0..2 {
                        let i = lane + 32 * s;
                        if i < d4 {
                            for j in 0..4 {
                                sm_acc[w * 256 + 4 * i + j] = acc[s * 4 + j];
                            }
                        }
                    }
                }
            }
            let mut big = sm_m[0];
            for &mw in &sm_m[1..] {
                big = big.max(mw);
            }
            let f: Vec<f32> = sm_m
                .iter()
                .map(|&mw| {
                    if mw == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (mw - big).exp()
                    }
                })
                .collect();
            for d in 0..head_dim {
                let mut sum = 0f32;
                for w in 0..WARPS {
                    sum += f[w] * sm_acc[w * 256 + d];
                }
                part_acc[c * head_dim + d] = sum;
            }
            let mut ls = 0f32;
            for w in 0..WARPS {
                ls += f[w] * sm_l[w];
            }
            part_ml[c * 2] = big;
            part_ml[c * 2 + 1] = ls;
        }
        let mut big = part_ml[0];
        for c in 1..chunks {
            big = big.max(part_ml[c * 2]);
        }
        for d in 0..head_dim {
            let mut num = 0f32;
            let mut den = 0f32;
            for c in 0..chunks {
                let mc = part_ml[c * 2];
                let fc = if mc == f32::NEG_INFINITY {
                    0.0
                } else {
                    (mc - big).exp()
                };
                num += fc * part_acc[c * head_dim + d];
                den += fc * part_ml[c * 2 + 1];
            }
            out[h * head_dim + d] = if den > 0.0 { num / den } else { 0.0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-pass softmax attention, the definition the twin must agree
    /// with: `frink_core::attention::causal_gqa_attention_row` for one
    /// query, naively.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        hd: usize,
        seq_len: usize,
        window: Option<usize>,
        scale: f32,
        softcap: Option<f32>,
    ) -> Vec<f32> {
        let (lo, hi) = key_range(seq_len, window);
        let group = n_heads / n_kv_heads;
        let mut out = vec![0f32; n_heads * hd];
        for h in 0..n_heads {
            let kv_h = h / group;
            let qh = &q[h * hd..(h + 1) * hd];
            let s: Vec<f64> = (lo..hi)
                .map(|t| {
                    let kt = &k[(t * n_kv_heads + kv_h) * hd..][..hd];
                    let mut x =
                        qh.iter().zip(kt).map(|(a, b)| (a * b) as f64).sum::<f64>() * scale as f64;
                    if let Some(c) = softcap {
                        x = c as f64 * (x / c as f64).tanh();
                    }
                    x
                })
                .collect();
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            for (ti, t) in (lo..hi).enumerate() {
                let vt = &v[(t * n_kv_heads + kv_h) * hd..][..hd];
                for d in 0..hd {
                    out[h * hd + d] += (e[ti] / z * vt[d] as f64) as f32;
                }
            }
        }
        out
    }

    fn wave(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.173 + seed).sin()).collect()
    }

    /// The twin is what the kernel text is checked against bit for bit
    /// on the host, so it has to be RIGHT, not just consistent: held
    /// against the plain two-pass definition across one chunk, many
    /// chunks with a ragged tail, a window that starts mid-chunk, a
    /// softcap, GQA groups and both head widths the lane slices take.
    #[test]
    fn the_twin_is_attention_across_chunks_windows_softcaps_and_head_widths() {
        for (n_heads, n_kv, hd) in [
            (4usize, 2usize, 64usize),
            (3, 1, 128),
            (2, 2, 256),
            (2, 1, 12),
        ] {
            for (seq_len, window, softcap) in [
                (1usize, None, None),
                (5, None, None),
                (64, None, None),
                (65, None, None),
                (300, None, None),
                (300, Some(100), None),
                (130, None, Some(30.0)),
            ] {
                let q = wave(n_heads * hd, 0.3);
                let k = wave(seq_len * n_kv * hd, 0.7);
                let v = wave(seq_len * n_kv * hd, 1.1);
                let scale = (hd as f32).sqrt().recip();
                let got = decode_attn_twin(
                    &q, &k, &v, n_heads, n_kv, hd, seq_len, window, scale, softcap,
                );
                let want = naive(
                    &q, &k, &v, n_heads, n_kv, hd, seq_len, window, scale, softcap,
                );
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (g - w).abs() <= 2e-5 * w.abs().max(1.0),
                        "hd={hd} seq={seq_len} window={window:?} cap={softcap:?} [{i}]: twin {g} want {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_key_range_is_the_hosts_window_rule() {
        assert_eq!(key_range(10, None), (0, 10));
        assert_eq!(key_range(10, Some(4)), (6, 10));
        assert_eq!(key_range(3, Some(4)), (0, 3));
        assert_eq!(key_range(10, Some(0)), (0, 10));
        assert_eq!(n_chunks(0), 1);
        assert_eq!(n_chunks(CHUNK), 1);
        assert_eq!(n_chunks(CHUNK + 1), 2);
    }

    /// The constants the kernel text spells as `#define`s are the ones
    /// the launch and the twin use: a chunk or warp count changed in
    /// one place and not the other indexes the scratch wrongly and
    /// returns numbers rather than failing.
    #[test]
    fn the_kernel_text_spells_the_same_chunk_and_warp_count() {
        assert!(DECODE_ATTN_KERNEL_SRC.contains(&format!("#define FX_DA_CHUNK {CHUNK}\n")));
        assert!(DECODE_ATTN_KERNEL_SRC.contains(&format!("#define FX_DA_WARPS {WARPS}\n")));
        assert!(DECODE_ATTN_KERNEL_SRC.contains(&format!("void {}(", PARTIAL.1)));
        assert!(DECODE_ATTN_KERNEL_SRC.contains(&format!("void {}(", COMBINE.1)));
        assert_eq!(geometry::partial(3, 7), ((3, 7, 1), (128, 1, 1)));
    }
}
