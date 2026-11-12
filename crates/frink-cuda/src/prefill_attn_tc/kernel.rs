//! The tensor-core prefill attention as CUDA C text. The `FX_*`
//! geometry is `#define`d by [`super::Variant::kernel_src`], one unit
//! per head width, and the PTX helpers come from
//! [`crate::mul_mm_tc::TC_PTX_SRC`].

/// The dynamic shared-memory declaration, named so the host check can
/// put a host buffer in its place and fail loudly when it changes.
pub const SMEM_DECL: &str = "extern __shared__ __align__(16) unsigned short fx_tc_smem[];";

/// The kernel body.
///
/// Block `(x, y)` is token tile `bx = gridDim.x - 1 - x`, the last
/// tokens (which see the most keys) launching first: 64 rows, row `r`
/// being token `bx * 64/G + r / G` and query head `r % G` of head group
/// `y % n_hg` of KV head `y / n_hg`, so every row of the block reads the
/// same K/V tiles (`super::block_row`). Warp `w` owns rows `16w .. 16w +
/// 15`, one `m16n8k16` row tile.
///
/// Per tile of `FX_BC` keys: K and V are converted to f16 into shared
/// memory, S = Q K^T accumulates in f32 on the tensor cores, the
/// softcap and the computed causal / window predicate apply in f32,
/// the online softmax runs in f32 (two rows per lane, the max reduced
/// across the lane's quad), P is packed to f16 straight from the S
/// fragments (the C layout of `m16n8k16` is its A layout), and O += P V
/// accumulates in f32 with V's fragments from `ldmatrix.trans`. A warp
/// skips a tile none of its rows sees, and a tile every one of its rows
/// sees in full takes no predicate. The sink folds in after the last
/// key, then O / l is written as f32.
pub const BODY_SRC: &str = r#"
#define FX_KS (FX_D / 16)
#define FX_NT (FX_BC / 8)
#define FX_DT (FX_D / 8)
#define FX_PK (FX_BC / 16)
#define FX_Q_STAGE (FX_ROWS * (FX_D / 4) / FX_THREADS)
#define FX_KV_STAGE (FX_BC * (FX_D / 4) / FX_THREADS)

__device__ __forceinline__ float frink_inf_tc() {
    return __int_as_float(0x7f800000);
}

// Four floats as four f16 into shared memory, round to nearest even.
__device__ __forceinline__ void frink_st4h(unsigned short* p, float a, float b, float c, float d) {
    uint2 h;
    h.x = frink_pack2h(a, b);
    h.y = frink_pack2h(c, d);
    *(uint2*)p = h;
}

extern "C" __global__ void FX_FN_NAME(
    const float* __restrict__ q,
    const float* __restrict__ k_all,
    const float* __restrict__ v_all,
    float* __restrict__ out,
    const float* __restrict__ sinks,  // one logit per query head, or null
    int n_q,
    int heads,       // n_heads | n_kv_heads << 16
    int start_pos,
    int window,      // 0: none
    float qscale,    // the scale, over the softcap when there is one
    float softcap,   // 0: none
    int group        // G, query heads per block: 1, 2, 4 or 8
) {
    extern __shared__ __align__(16) unsigned short fx_tc_smem[];
    const int n_heads = heads & 0xffff;
    const int n_kv_heads = heads >> 16;
    const int gqa = n_heads / n_kv_heads;
    const int n_hg = (gqa + group - 1) / group;
    const int by = (int)blockIdx.y;
    const int kvh = by / n_hg;
    const int hg0 = (by - kvh * n_hg) * group;
    // The heaviest token tiles first, so the last wave is light ones.
    const int j0 = (int)(gridDim.x - 1 - blockIdx.x) * (FX_ROWS / group);
    const int last_j = min(j0 + FX_ROWS / group, n_q) - 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tg = lane & 3;
    const int lm = lane >> 3;
    const int lr8 = lane & 7;

    // Q lives in shared memory for the whole kernel (FX_Q_REGS 0) or
    // only until each warp has its fragments in registers, in the space
    // the K tile takes next.
    unsigned short* const q_s = fx_tc_smem;
    unsigned short* const k_s = fx_tc_smem + (FX_Q_REGS ? 0 : FX_ROWS * FX_LD);
    unsigned short* const v_s = k_s + FX_BC * FX_LD;

    // The block's rows of Q, times qscale in f32, as f16. A row past
    // the batch or past the KV head's query heads is zeros and is never
    // stored.
#pragma unroll 4
    for (int it = 0; it < FX_Q_STAGE; it++) {
        const int i = tid + it * FX_THREADS;
        const int r = i / (FX_D / 4);
        const int c = i - r * (FX_D / 4);
        const int j = j0 + r / group;
        const int hi = hg0 + r % group;
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (j < n_q && hi < gqa) {
            x = ((const float4*)(q + ((size_t)j * n_heads + kvh * gqa + hi) * FX_D))[c];
        }
        frink_st4h(q_s + r * FX_LD + 4 * c, x.x * qscale, x.y * qscale, x.z * qscale, x.w * qscale);
    }
    __syncthreads();

    // A fragments of the warp's rows, by ldmatrix: lane l addresses row
    // (l & 7) of matrix (l >> 3), rows 0-7 and 8-15 at k 0-7, then the
    // same rows at k 8-15.
#if FX_Q_REGS
    unsigned int qf[FX_KS][4];
#pragma unroll
    for (int ks = 0; ks < FX_KS; ks++) {
        frink_ldsm_x4(qf[ks], q_s + (16 * warp + (lm & 1) * 8 + lr8) * FX_LD + 16 * ks + (lm >> 1) * 8);
    }
#endif

    // Positions: the block's tokens sit at start_pos + j0 .. start_pos +
    // last_j, the warp's real ones at w_lo .. w_hi, and the lane's two
    // rows (g and g + 8 of the warp's sixteen) at p_a and p_b.
    const int w_first = j0 + (16 * warp) / group;
    const bool w_idle = w_first > last_j;
    const int w_lo = start_pos + w_first;
    const int w_hi = start_pos + min(j0 + (16 * warp + 15) / group, last_j);
    const int p_a = start_pos + j0 + (16 * warp + g) / group;
    const int p_b = start_pos + j0 + (16 * warp + g + 8) / group;
    // Keys any row of the block sees: from the first row's window start
    // to the last row's own position.
    const int k_hi = start_pos + last_j + 1;
    const int k_lo = window > 0 ? max(0, start_pos + j0 - window + 1) : 0;

    // The running max starts at -FLT_MAX / 2, not -inf: a row with no
    // key yet keeps alpha = expf(0) = 1 and every p = 0, never a NaN.
    float m_r[2] = {-__int_as_float(0x7effffff), -__int_as_float(0x7effffff)};
    // This lane's share of each row's sum. The quad adds its four at
    // the end.
    float l_r[2] = {0.f, 0.f};
    float o[FX_DT][4];
#pragma unroll
    for (int dt = 0; dt < FX_DT; dt++) {
        o[dt][0] = 0.f; o[dt][1] = 0.f; o[dt][2] = 0.f; o[dt][3] = 0.f;
    }

    for (int t0 = k_lo; t0 < k_hi; t0 += FX_BC) {
        // Every warp is past its reads of the previous tile (the first
        // time, of the staged Q).
        __syncthreads();
        // K and V rows of the tile as f16. A key past k_hi is zeros. Four
        // trips in flight: eight float4 loads a thread, without holding
        // a whole tile in registers beside O.
#pragma unroll 4
        for (int it = 0; it < FX_KV_STAGE; it++) {
            const int i = tid + it * FX_THREADS;
            const int jj = i / (FX_D / 4);
            const int c = i - jj * (FX_D / 4);
            const int kt = t0 + jj;
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
            if (kt < k_hi) {
                const size_t at = ((size_t)kt * n_kv_heads + kvh) * FX_D;
                a = ((const float4*)(k_all + at))[c];
                b = ((const float4*)(v_all + at))[c];
            }
            frink_st4h(k_s + jj * FX_LD + 4 * c, a.x, a.y, a.z, a.w);
            frink_st4h(v_s + jj * FX_LD + 4 * c, b.x, b.y, b.z, b.w);
        }
        __syncthreads();

        // Warp-uniform: no row of the warp sees a key of the tile, or
        // every row sees all of them.
        const int t_last = t0 + FX_BC - 1;
        if (w_idle || t0 > w_hi || (window > 0 && w_lo - t_last >= window)) continue;
        const bool full = t_last <= w_lo && (window == 0 || w_hi - t0 < window);

        // S = Q K^T. K is stored [key][channel], which is the n-major
        // B layout: matrices 0 and 1 are keys 0-7 at k 0-7 and 8-15 (one
        // n8 fragment), 2 and 3 the next eight keys.
        float s[FX_NT][4];
#pragma unroll
        for (int nt = 0; nt < FX_NT; nt++) {
            s[nt][0] = 0.f; s[nt][1] = 0.f; s[nt][2] = 0.f; s[nt][3] = 0.f;
        }
#pragma unroll
        for (int ks = 0; ks < FX_KS; ks++) {
            unsigned int a[4];
#if FX_Q_REGS
            a[0] = qf[ks][0]; a[1] = qf[ks][1]; a[2] = qf[ks][2]; a[3] = qf[ks][3];
#else
            frink_ldsm_x4(a, q_s + (16 * warp + (lm & 1) * 8 + lr8) * FX_LD + 16 * ks + (lm >> 1) * 8);
#endif
#pragma unroll
            for (int nt = 0; nt < FX_NT; nt += 2) {
                unsigned int b[4];
                frink_ldsm_x4(b, k_s + (8 * nt + (lm >> 1) * 8 + lr8) * FX_LD + 16 * ks + (lm & 1) * 8);
                frink_mma16816(s[nt], a, &b[0]);
                frink_mma16816(s[nt + 1], a, &b[2]);
            }
        }

        // C fragment: s[nt][e] is row g (e < 2) or g + 8 at key
        // t0 + 8 nt + 2 tg + (e & 1). Softcap, predicate, row max.
        float mx[2] = {-frink_inf_tc(), -frink_inf_tc()};
#pragma unroll
        for (int nt = 0; nt < FX_NT; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float x = s[nt][e];
                if (softcap > 0.f) x = softcap * tanhf(x);
                if (!full) {
                    const int kp = t0 + 8 * nt + 2 * tg + (e & 1);
                    const int qp = e < 2 ? p_a : p_b;
                    if (kp > qp || (window > 0 && qp - kp >= window)) x = -frink_inf_tc();
                }
                s[nt][e] = x;
                mx[e >> 1] = fmaxf(mx[e >> 1], x);
            }
        }
        float alpha[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], 2));
            const float mn = fmaxf(m_r[i], mx[i]);
            alpha[i] = expf(m_r[i] - mn);
            m_r[i] = mn;
        }
        float ls[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < FX_NT; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float p = expf(s[nt][e] - m_r[e >> 1]);
                s[nt][e] = p;
                ls[e >> 1] += p;
            }
        }
        l_r[0] = l_r[0] * alpha[0] + ls[0];
        l_r[1] = l_r[1] * alpha[1] + ls[1];
#pragma unroll
        for (int dt = 0; dt < FX_DT; dt++) {
            o[dt][0] *= alpha[0]; o[dt][1] *= alpha[0];
            o[dt][2] *= alpha[1]; o[dt][3] *= alpha[1];
        }

        // O += P V. The S fragments of key tiles 2kk and 2kk + 1 are the
        // A fragment of k step kk as they stand. V is stored [key]
        // [channel], and ldmatrix.trans hands each lane keys 2tg and
        // 2tg + 1 of channel g, the k-major B layout: matrices 0 and 1
        // are keys 0-7 and 8-15 of channels 0-7, 2 and 3 of channels 8-15.
#pragma unroll
        for (int kk = 0; kk < FX_PK; kk++) {
            unsigned int pa[4];
            pa[0] = frink_pack2h(s[2 * kk][0], s[2 * kk][1]);
            pa[1] = frink_pack2h(s[2 * kk][2], s[2 * kk][3]);
            pa[2] = frink_pack2h(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[3] = frink_pack2h(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dt = 0; dt < FX_DT; dt += 2) {
                unsigned int b[4];
                frink_ldsm_x4_trans(b, v_s + (16 * kk + (lm & 1) * 8 + lr8) * FX_LD + 8 * dt + (lm >> 1) * 8);
                frink_mma16816(o[dt], pa, &b[0]);
                frink_mma16816(o[dt + 1], pa, &b[2]);
            }
        }
    }

    // The quad's four shares of each row's sum, every lane of the warp
    // taking part.
#pragma unroll
    for (int i = 0; i < 2; i++) {
        l_r[i] += __shfl_xor_sync(0xffffffffu, l_r[i], 1);
        l_r[i] += __shfl_xor_sync(0xffffffffu, l_r[i], 2);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = 16 * warp + g + 8 * i;
        const int j = j0 + r / group;
        const int hi = hg0 + r % group;
        if (j >= n_q || hi >= gqa) continue;
        const int h = kvh * gqa + hi;
        // A sink (gpt-oss) is one more logit with no value, folded in
        // after every key as the host's `online_attn_accumulate` folds
        // it: the max and the sum move, O is rescaled.
        float sc = 1.f;
        float li = l_r[i];
        if (sinks != nullptr) {
            const float sk = sinks[h];
            const float mn = fmaxf(m_r[i], sk);
            sc = expf(m_r[i] - mn);
            li = li * sc + expf(sk - mn);
        }
        const float inv = li > 0.f ? sc / li : 0.f;
        float* orow = out + ((size_t)j * n_heads + h) * FX_D + 2 * tg;
#pragma unroll
        for (int dt = 0; dt < FX_DT; dt++) {
            *(float2*)(orow + 8 * dt) = make_float2(o[dt][2 * i] * inv, o[dt][2 * i + 1] * inv);
        }
    }
}
"#;
