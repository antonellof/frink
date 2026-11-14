//! The Mamba selective-state-space step, as ggml computes it, for both
//! generations.
//!
//! Two kernels and a state. `ggml_ssm_conv` (`ggml-cpu/ops.cpp:9557-9608`)
//! is a causal depthwise convolution of width `d_conv` over the
//! projected rows, with the previous `d_conv - 1` rows as the state;
//! `ggml_ssm_scan` (`:9627-9850`) is the recurrence
//!
//! ```text
//! dt'      = softplus(dt_h)                     (ggml-impl.h:107-109)
//! dA       = exp(dt' * A)                       Mamba-2: one scalar per head (:9682-9724)
//!                                               Mamba-1: one per state element (:9779-9820)
//! S[h,d,:] = S[h,d,:] * dA + B[g,:] * (x[h,d] * dt')
//! y[h,d]   = S[h,d,:] . C[g,:]
//! ```
//!
//! with `g = h / (n_head / n_group)` (`repeat_interleave`) and a float
//! accumulator. The state is `[n_head][head_dim][d_state]` with the
//! state index fastest, ggml's `{d_state, head_dim, n_head}`. Mamba-1
//! (`build_mamba_layer`) is the same kernel with `n_head = d_inner`,
//! `head_dim = 1`, `n_group = 1` and A `{d_state, d_inner}`; the
//! kernel branches on `src3->ne[0] == 1` and so does [`Decay`].
//!
//! This file is the arithmetic only: no weights, no norms, no
//! projections. `frink_models::mamba2` owns those and the residual
//! topology; [`crate::recurrent_state::RecurrentState`] owns the two
//! buffers between tokens.

/// `log(1 + exp(x))`, in ggml's precision (`ggml_compute_softplus_f32`).
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// One token of the causal conv: `out[c] = sum_i taps[c][i] * window[i][c]`,
/// where `window` is the previous `d_conv - 1` rows (the state, oldest
/// first) followed by this token's `x`, and the state is then shifted
/// by one row with `x` appended.
///
/// `state` is `[(d_conv - 1)][width]`, `taps` is `[width][d_conv]`
/// (ggml `{d_conv, width}`: channel `c`'s taps contiguous, oldest input
/// on tap 0), `x` and `out` are `[width]`.
pub fn conv_step(state: &mut [f32], taps: &[f32], d_conv: usize, x: &[f32], out: &mut [f32]) {
    let width = x.len();
    assert_eq!(state.len(), (d_conv - 1) * width);
    assert_eq!(taps.len(), width * d_conv);
    assert_eq!(out.len(), width);
    for c in 0..width {
        let t = &taps[c * d_conv..(c + 1) * d_conv];
        // ops.cpp:9598-9603: a float accumulator over the window in
        // order, the newest input last.
        let mut acc = 0.0f32;
        for (i, tap) in t.iter().enumerate().take(d_conv - 1) {
            acc += state[i * width + c] * tap;
        }
        acc += x[c] * t[d_conv - 1];
        out[c] = acc;
    }
    // Shift: drop the oldest row, append `x`.
    if d_conv > 1 {
        state.copy_within(width.., 0);
        state[(d_conv - 2) * width..].copy_from_slice(x);
    }
}

/// `rows` tokens of [`conv_step`], then `silu(out + bias)` on each output
/// (`mamba-base.cpp:66-70`, `:222-226`), parallel over channels.
///
/// Channels are independent across the whole batch (each reads only its
/// own column of the state and its own taps), so one parallel region
/// walks every row of a channel with its window in registers. Each
/// output is the same float sum [`conv_step`] makes, in the same order.
///
/// `x` row `r` is `width` inputs at `r * x_stride`; `out` is
/// `[rows][width]`. A serial row loop here was a sixth of a Granite-4.0-H
/// prompt.
#[allow(clippy::too_many_arguments)]
pub fn conv_silu_rows(
    state: &mut [f32],
    taps: &[f32],
    bias: &[f32],
    d_conv: usize,
    x: &[f32],
    x_stride: usize,
    rows: usize,
    out: &mut [f32],
) {
    let width = bias.len();
    let hist = d_conv - 1;
    assert_eq!(state.len(), hist * width);
    assert_eq!(taps.len(), width * d_conv);
    assert_eq!(out.len(), rows * width);
    assert!(rows == 0 || x.len() >= (rows - 1) * x_stride + width);
    if rows == 0 {
        return;
    }
    // Blocks of channels, so a task reads and writes runs of a row
    // rather than one float per cache line. A block holds its `rows`
    // output rows, then its new history rows, each `BLOCK` wide.
    const BLOCK: usize = 64;
    let n_blocks = width.div_ceil(BLOCK);
    let per = (rows + hist) * BLOCK;
    let mut by_block = vec![0.0f32; n_blocks * per];
    let state_in: &[f32] = state;
    crate::par::chunks_mut(&mut by_block, per, 1, |b, blk| {
        let c0 = b * BLOCK;
        let bw = BLOCK.min(width - c0);
        // The block's windows, `[hist][BLOCK]`, oldest row first.
        let mut window = vec![0.0f32; hist * BLOCK];
        for i in 0..hist {
            window[i * BLOCK..i * BLOCK + bw]
                .copy_from_slice(&state_in[i * width + c0..i * width + c0 + bw]);
        }
        for r in 0..rows {
            let xr = &x[r * x_stride + c0..r * x_stride + c0 + bw];
            for (j, &xc) in xr.iter().enumerate() {
                let c = c0 + j;
                let t = &taps[c * d_conv..(c + 1) * d_conv];
                // ops.cpp:9598-9603, as `conv_step`.
                let mut acc = 0.0f32;
                for (i, tap) in t.iter().enumerate().take(hist) {
                    acc += window[i * BLOCK + j] * tap;
                }
                acc += xc * t[hist];
                let v = acc + bias[c];
                blk[r * BLOCK + j] = v / (1.0 + (-v).exp());
                for i in 1..hist {
                    window[(i - 1) * BLOCK + j] = window[i * BLOCK + j];
                }
                if hist > 0 {
                    window[(hist - 1) * BLOCK + j] = xc;
                }
            }
        }
        blk[rows * BLOCK..].copy_from_slice(&window);
    });
    for (b, blk) in by_block.chunks_exact(per).enumerate() {
        let c0 = b * BLOCK;
        let bw = BLOCK.min(width - c0);
        for r in 0..rows {
            out[r * width + c0..r * width + c0 + bw]
                .copy_from_slice(&blk[r * BLOCK..r * BLOCK + bw]);
        }
        for i in 0..hist {
            let src = &blk[(rows + i) * BLOCK..(rows + i) * BLOCK + bw];
            state[i * width + c0..i * width + c0 + bw].copy_from_slice(src);
        }
    }
}

/// The stored `A`, in the two shapes the kernel takes.
#[derive(Debug, Clone, Copy)]
pub enum Decay<'a> {
    /// Mamba-2: `[n_head]`, `{1, n_head}` in ggml; `dA` is one scalar
    /// per head.
    PerHead(&'a [f32]),
    /// Mamba-1: `[n_head][d_state]`, `{d_state, n_head}` in ggml; `dA`
    /// is one scalar per state element.
    PerState(&'a [f32]),
}

/// The geometry one scan step needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanDims {
    pub n_head: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub n_group: usize,
}

impl ScanDims {
    /// Floats in one sequence's SSM state.
    pub fn state_len(self) -> usize {
        self.n_head * self.head_dim * self.d_state
    }
}

/// One token of the Mamba-2 scan, in place on `state`
/// (`[n_head][head_dim][d_state]`).
///
/// `x` is `[n_head][head_dim]`, `dt` is `[n_head]` (BEFORE softplus,
/// the bias already added), `a` is the stored `ssm_a` (already
/// negative) in either shape, `b` and `c` are `[n_group][d_state]`,
/// `y` is `[n_head][head_dim]`. It is [`scan_rows`] at one row, the
/// body both share.
#[allow(clippy::too_many_arguments)] // the seven operands ggml_ssm_scan takes, plus the dims
pub fn scan_step(
    dims: ScanDims,
    state: &mut [f32],
    x: &[f32],
    dt: &[f32],
    a: Decay<'_>,
    b: &[f32],
    c: &[f32],
    y: &mut [f32],
) {
    let (n_x, n_bc) = (dims.n_head * dims.head_dim, dims.n_group * dims.d_state);
    assert_eq!(x.len(), n_x);
    assert_eq!(b.len(), n_bc);
    assert_eq!(c.len(), n_bc);
    let mut xbc = Vec::with_capacity(n_x + 2 * n_bc);
    xbc.extend_from_slice(x);
    xbc.extend_from_slice(b);
    xbc.extend_from_slice(c);
    let batch = ScanRows {
        xbc: &xbc,
        stride: n_x + 2 * n_bc,
        dt,
        rows: 1,
    };
    scan_rows(dims, state, batch, a, y);
}

/// `rows` consecutive tokens of ONE sequence, as [`scan_rows`] reads them.
///
/// Row `r` of `xbc` starts at `r * stride` and is `[x | B | C]`: `x` is
/// `[n_head][head_dim]`, `B` and `C` are `[n_group][d_state]` each, the
/// order Mamba-2's conv output already holds them in
/// (`mamba-base.cpp:231-236`). `dt` is `[rows][n_head]`, BEFORE softplus.
#[derive(Debug, Clone, Copy)]
pub struct ScanRows<'a> {
    pub xbc: &'a [f32],
    pub stride: usize,
    pub dt: &'a [f32],
    pub rows: usize,
}

/// Rows from which a batch's scan goes to the device when one is on: a
/// decode token's scan is cheaper on the host than a submission.
pub const DEVICE_SCAN_MIN_ROWS: usize = 8;

/// [`scan_rows`] on a sequence's own state, on the device when one
/// serves the batch (`frink_metal::ssm_scan`, a prompt on Metal), and
/// on the host otherwise or if the launch fails. The state is the
/// page-aligned [`AlignedF32`] a `RecurrentState` holds, which is what
/// lets the kernel advance it in place.
///
/// [`AlignedF32`]: crate::recurrent_state::AlignedF32
pub fn scan_rows_state(
    dims: ScanDims,
    state: &mut crate::recurrent_state::AlignedF32,
    batch: ScanRows<'_>,
    a: Decay<'_>,
    y: &mut [f32],
) {
    #[cfg(feature = "metal")]
    if batch.rows >= DEVICE_SCAN_MIN_ROWS && crate::metal_dense_enabled() {
        let shape = frink_metal::ssm_scan::ScanShape {
            n_head: dims.n_head,
            head_dim: dims.head_dim,
            d_state: dims.d_state,
            n_group: dims.n_group,
        };
        let (a_rows, per_state) = match a {
            Decay::PerHead(a) => (a, false),
            Decay::PerState(a) => (a, true),
        };
        assert_eq!(state.len(), dims.state_len());
        // SAFETY: `state` is an `AlignedF32`, page-aligned with a
        // whole-page allocation of `alloc_bytes`, and the `&mut` held
        // across the call keeps every other reader and writer off it
        // until the launch, which waits for the GPU, returns.
        let ran = unsafe {
            frink_metal::ssm_scan::launch_scan_rows(
                shape,
                state.as_ptr(),
                state.alloc_bytes(),
                batch.xbc,
                batch.stride,
                batch.dt,
                batch.rows,
                a_rows,
                per_state,
                y,
            )
        };
        if ran.is_ok() {
            return;
        }
    }
    scan_rows(dims, state, batch, a, y);
}

/// The scan over `rows` tokens of one sequence, in place on `state`
/// (`[n_head][head_dim][d_state]`); `y` is `[rows][n_head][head_dim]`.
///
/// Heads are independent across the whole batch: each owns its
/// `head_dim x d_state` slice of the state and reads its own `x` and
/// `dt`. So ONE parallel region runs every row of a head, which is how
/// `ops.cpp:10014-10019` divides the work between threads too, and
/// within a head the rows run in order, exactly as `rows` single steps
/// would. A serial loop over heads here was most of a Granite-4.0-H or
/// Falcon-H1 prompt.
pub fn scan_rows(
    dims: ScanDims,
    state: &mut [f32],
    batch: ScanRows<'_>,
    a: Decay<'_>,
    y: &mut [f32],
) {
    let ScanDims {
        n_head,
        head_dim,
        d_state,
        n_group,
    } = dims;
    let ScanRows {
        xbc,
        stride,
        dt,
        rows,
    } = batch;
    let n_x = n_head * head_dim;
    let n_bc = n_group * d_state;
    assert_eq!(state.len(), dims.state_len());
    assert!(stride >= n_x + 2 * n_bc, "a row holds x, B and C");
    assert!(rows == 0 || xbc.len() >= (rows - 1) * stride + n_x + 2 * n_bc);
    assert_eq!(dt.len(), rows * n_head);
    match a {
        Decay::PerHead(a) => assert_eq!(a.len(), n_head),
        Decay::PerState(a) => assert_eq!(a.len(), n_head * d_state),
    }
    assert_eq!(y.len(), rows * n_x);
    assert_eq!(n_head % n_group, 0, "ops.cpp:9659");
    if rows == 0 || n_x == 0 || d_state == 0 {
        return;
    }
    let heads_per_group = n_head / n_group;
    // Each head's outputs as `[n_head][rows][head_dim]`, so a task owns
    // one contiguous run of them; a single row is already in `y`'s order.
    let mut by_head = if rows == 1 {
        Vec::new()
    } else {
        vec![0.0f32; rows * n_x]
    };
    let out: &mut [f32] = if rows == 1 { &mut *y } else { &mut by_head };
    // Enough heads per task that a Mamba-1 layer (thousands of heads of
    // one channel and sixteen states) is not one task per head.
    let min_len = (4096 / (head_dim * d_state)).max(1);
    crate::par::chunks_mut2_by(
        state,
        out,
        head_dim * d_state,
        rows * head_dim,
        min_len,
        |h, st, yh| {
            let g = h / heads_per_group;
            for r in 0..rows {
                let row = &xbc[r * stride..r * stride + n_x + 2 * n_bc];
                let bg = &row[n_x + g * d_state..n_x + (g + 1) * d_state];
                let cg = &row[n_x + n_bc + g * d_state..n_x + n_bc + (g + 1) * d_state];
                let dt_sp = softplus(dt[r * n_head + h]);
                for d in 0..head_dim {
                    let x_dt = row[h * head_dim + d] * dt_sp;
                    let s = &mut st[d * d_state..(d + 1) * d_state];
                    yh[r * head_dim + d] = match a {
                        // ops.cpp:10042: dA outside the state-wise loop.
                        Decay::PerHead(a) => per_head_row(s, (dt_sp * a[h]).exp(), x_dt, bg, cg),
                        // ops.cpp:10169-10181, the scalar arm on every
                        // non-SVE host: one `expf(dt_soft_plus * A)` per
                        // state element and a single running sum ("can't
                        // really use GGML_SIMD here").
                        Decay::PerState(a) => {
                            let ah = &a[h * d_state..(h + 1) * d_state];
                            let mut sum = 0.0f32;
                            for k in 0..d_state {
                                let v = s[k] * (dt_sp * ah[k]).exp() + bg[k] * x_dt;
                                sum += v * cg[k];
                                s[k] = v;
                            }
                            sum
                        }
                    };
                }
            }
        },
    );
    // Back to `[rows][n_head][head_dim]`, element by element: a Mamba-1
    // head is ONE channel, and a slice copy per element was most of the
    // scan's time on a Mamba prompt.
    if rows > 1 {
        for (r, y_row) in y.chunks_exact_mut(n_x).enumerate() {
            for (h, y_head) in y_row.chunks_exact_mut(head_dim).enumerate() {
                let src = (h * rows + r) * head_dim;
                for (d, v) in y_head.iter_mut().enumerate() {
                    *v = by_head[src + d];
                }
            }
        }
    }
}

/// ggml's f32 step on an ARM host (`GGML_F32_STEP`, `simd-mappings.h:
/// 354`): four vectors of four lanes.
const F32_STEP: usize = 16;

/// One state row of the Mamba-2 scan, `s = s * dA + B * x_dt`, and its
/// dot with `C`, in the order `ops.cpp:10085-10128` computes it on an
/// ARM host (the NEON arm of `GGML_SIMD`; Apple Silicon has no SVE):
///
/// - sixteen running sums, each `vfmaq_f32(sum, s, C)` (`GGML_F32x4_FMA`,
///   a FUSED multiply-add), over every full step of sixteen states;
/// - folded as `GGML_F32x4_REDUCE` folds them with `GGML_F32_ARR` 4:
///   vector 0 += vector 2, vector 1 += vector 3, vector 0 += vector 1,
///   then `vaddvq_f32`'s pairwise lane sum `(l0 + l1) + (l2 + l3)`;
/// - then the scalar tail past the last full step, onto that sum.
///
/// The state update is a multiply, a multiply and an add, unfused, as
/// the intrinsics spell it. Every published Mamba-2 checkpoint has a
/// `d_state` that is a multiple of sixteen, so the tail is for fixtures.
fn per_head_row(s: &mut [f32], da: f32, x_dt: f32, b: &[f32], c: &[f32]) -> f32 {
    let n = s.len();
    let np = n & !(F32_STEP - 1);
    let mut acc = [0.0f32; F32_STEP];
    for i in (0..np).step_by(F32_STEP) {
        let si = &mut s[i..i + F32_STEP];
        let (bi, ci) = (&b[i..i + F32_STEP], &c[i..i + F32_STEP]);
        for l in 0..F32_STEP {
            let v = si[l] * da + bi[l] * x_dt;
            acc[l] = v.mul_add(ci[l], acc[l]);
            si[l] = v;
        }
    }
    let mut sum = 0.0f32;
    if np > 0 {
        // Vector `j` is lanes `4j..4j + 4`.
        for l in 0..4 {
            acc[l] += acc[8 + l];
            acc[4 + l] += acc[12 + l];
        }
        for l in 0..4 {
            acc[l] += acc[4 + l];
        }
        sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    for k in np..n {
        let v = s[k] * da + b[k] * x_dt;
        sum += v * c[k];
        s[k] = v;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softplus_is_ggml_s() {
        assert!((softplus(0.0) - 2f32.ln()).abs() < 1e-7);
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(-30.0)).abs() < 1e-6);
    }

    /// Two channels, taps `[1, 2, 3]` and `[0, 0, 1]`: the newest input
    /// on the LAST tap, the state shifting one row per step.
    #[test]
    fn conv_step_puts_the_newest_input_on_the_last_tap_and_shifts() {
        let taps = [1.0, 2.0, 3.0, 0.0, 0.0, 1.0];
        let mut state = vec![0.0f32; 2 * 2];
        let mut out = [0.0f32; 2];
        conv_step(&mut state, &taps, 3, &[1.0, 5.0], &mut out);
        assert_eq!(out, [3.0, 5.0]);
        assert_eq!(state, vec![0.0, 0.0, 1.0, 5.0]);
        conv_step(&mut state, &taps, 3, &[2.0, 6.0], &mut out);
        assert_eq!(out, [3.0 * 2.0 + 2.0 * 1.0, 6.0]);
        conv_step(&mut state, &taps, 3, &[1.0, 7.0], &mut out);
        assert_eq!(out, [3.0 + 4.0 + 1.0, 7.0]);
        assert_eq!(state, vec![2.0, 6.0, 1.0, 7.0]);
    }

    /// One head, one channel, one state: the recurrence by hand.
    #[test]
    fn scan_step_is_the_recurrence() {
        let dims = ScanDims {
            n_head: 1,
            head_dim: 1,
            d_state: 2,
            n_group: 1,
        };
        let mut state = vec![0.0f32; 2];
        let mut y = [0.0f32];
        let a = Decay::PerHead(&[-1.0f32]);
        // dt = 0 -> softplus = ln 2; dA = exp(-ln 2) = 0.5.
        scan_step(
            dims,
            &mut state,
            &[2.0],
            &[0.0],
            a,
            &[1.0, 3.0],
            &[1.0, 1.0],
            &mut y,
        );
        let x_dt = 2.0 * 2f32.ln();
        assert!((state[0] - x_dt).abs() < 1e-6 && (state[1] - 3.0 * x_dt).abs() < 1e-6);
        assert!((y[0] - 4.0 * x_dt).abs() < 1e-5);
        scan_step(
            dims,
            &mut state,
            &[0.0],
            &[0.0],
            a,
            &[1.0, 3.0],
            &[1.0, 0.0],
            &mut y,
        );
        assert!((state[0] - 0.5 * x_dt).abs() < 1e-6, "decayed by dA");
        assert!((y[0] - 0.5 * x_dt).abs() < 1e-6, "C selects state 0");
    }

    /// Mamba-1's per-state decay: two states of one head decay
    /// differently.
    #[test]
    fn per_state_decay_is_one_exp_per_state_element() {
        let dims = ScanDims {
            n_head: 1,
            head_dim: 1,
            d_state: 2,
            n_group: 1,
        };
        let mut state = vec![1.0f32; 2];
        let mut y = [0.0f32];
        // dt = 30 -> softplus = 30; dA = exp(30 * A).
        let a = Decay::PerState(&[0.0, -(2f32.ln()) / 30.0]);
        scan_step(
            dims,
            &mut state,
            &[0.0],
            &[30.0],
            a,
            &[0.0, 0.0],
            &[1.0, 1.0],
            &mut y,
        );
        assert!(
            (state[0] - 1.0).abs() < 1e-6 && (state[1] - 0.5).abs() < 1e-5,
            "{state:?}"
        );
        assert!((y[0] - 1.5).abs() < 1e-5);
    }

    /// Groups: head `h` reads B/C group `h / (n_head / n_group)`.
    #[test]
    fn heads_read_their_group_s_b_and_c() {
        let dims = ScanDims {
            n_head: 4,
            head_dim: 1,
            d_state: 1,
            n_group: 2,
        };
        let mut state = vec![0.0f32; 4];
        let mut y = [0.0f32; 4];
        scan_step(
            dims,
            &mut state,
            &[1.0; 4],
            &[30.0; 4],
            Decay::PerHead(&[0.0; 4]),
            &[1.0, 10.0],
            &[1.0, 1.0],
            &mut y,
        );
        // softplus(30) = 30, dA = 1: state = B[g] * 30.
        assert_eq!(y, [30.0, 30.0, 300.0, 300.0]);
    }

    fn rnd(seed: &mut u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((*seed >> 9) as f32 / (1u32 << 23) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// A batch of rows is the same arithmetic as that many single steps,
    /// to the bit, in both decay shapes and with groups and a tail past
    /// the last full step of sixteen states.
    #[test]
    fn scan_rows_is_that_many_steps() {
        let dims = ScanDims {
            n_head: 4,
            head_dim: 3,
            d_state: 20,
            n_group: 2,
        };
        let (n_x, n_bc, rows) = (12, 40, 5);
        let mut seed = 7u32;
        let xbc = rnd(&mut seed, rows * (n_x + 2 * n_bc));
        let dt = rnd(&mut seed, rows * dims.n_head);
        let a_head: Vec<f32> = rnd(&mut seed, 4).iter().map(|v| -v.abs()).collect();
        let a_state: Vec<f32> = rnd(&mut seed, 80).iter().map(|v| -v.abs()).collect();
        for a in [Decay::PerHead(&a_head), Decay::PerState(&a_state)] {
            let init = rnd(&mut seed, dims.state_len());
            let mut batched = init.clone();
            let mut y_batched = vec![0.0f32; rows * n_x];
            let batch = ScanRows {
                xbc: &xbc,
                stride: n_x + 2 * n_bc,
                dt: &dt,
                rows,
            };
            scan_rows(dims, &mut batched, batch, a, &mut y_batched);
            let mut stepped = init;
            let mut y_stepped = Vec::new();
            for r in 0..rows {
                let row = &xbc[r * (n_x + 2 * n_bc)..(r + 1) * (n_x + 2 * n_bc)];
                let mut y = vec![0.0f32; n_x];
                scan_step(
                    dims,
                    &mut stepped,
                    &row[..n_x],
                    &dt[r * 4..(r + 1) * 4],
                    a,
                    &row[n_x..n_x + n_bc],
                    &row[n_x + n_bc..],
                    &mut y,
                );
                y_stepped.extend(y);
            }
            assert_eq!(y_batched, y_stepped);
            assert_eq!(batched, stepped);
        }
    }

    /// The Mamba-2 dot is folded as ggml's NEON arm folds it, not as a
    /// running sum: sixteen fused partial sums, reduced vector 0 + 2,
    /// 1 + 3, then 0 + 1, then the lanes pairwise, then the tail. The
    /// values put a large term and its negation in lanes the two orders
    /// combine differently, so a running sum gets a different answer.
    #[test]
    fn the_mamba2_dot_folds_as_ggml_neon_does() {
        let n = 18;
        let mut s = vec![0.0f32; n];
        // dA 0, x_dt 1: the new state is B, so the dot is B . C.
        let mut b = vec![0.0f32; n];
        b[0] = 1.0e8;
        b[1] = 1.0;
        b[8] = -1.0e8;
        b[16] = 3.0;
        b[17] = 0.5;
        let c = vec![1.0f32; n];
        let got = per_head_row(&mut s, 0.0, 1.0, &b, &c);
        // Lane 0 is vector 0, lane 8 is vector 2: they cancel first, and
        // lane 1 survives. Then the tail, 3 and 0.5, in order.
        assert_eq!(got, ((0.0f32 + 1.0) + 3.0) + 0.5);
        // A running sum loses lane 1 to the 1e8 it is added to.
        let running: f32 = b.iter().zip(&c).fold(0.0f32, |acc, (x, y)| acc + x * y);
        assert_ne!(got, running, "the fixture must tell the two orders apart");
        assert_eq!(s, b, "the state is updated in place");
    }

    /// The batched conv is `rows` single steps and their bias and SiLU,
    /// to the bit, outputs and state both, over strided input rows and
    /// more channels than one parallel task takes.
    #[test]
    fn conv_silu_rows_is_that_many_steps() {
        let (width, d_conv, rows, x_stride) = (150usize, 4usize, 6usize, 170usize);
        let mut seed = 5u32;
        let taps = rnd(&mut seed, width * d_conv);
        let bias = rnd(&mut seed, width);
        let x = rnd(&mut seed, rows * x_stride);
        let init = rnd(&mut seed, (d_conv - 1) * width);
        let mut batched = init.clone();
        let mut out = vec![0.0f32; rows * width];
        conv_silu_rows(
            &mut batched,
            &taps,
            &bias,
            d_conv,
            &x,
            x_stride,
            rows,
            &mut out,
        );
        let mut stepped = init;
        let mut want = Vec::new();
        for r in 0..rows {
            let mut o = vec![0.0f32; width];
            let row = &x[r * x_stride..r * x_stride + width];
            conv_step(&mut stepped, &taps, d_conv, row, &mut o);
            want.extend(o.iter().zip(&bias).map(|(v, b)| {
                let v = v + b;
                v / (1.0 + (-v).exp())
            }));
        }
        assert_eq!(out, want);
        assert_eq!(batched, stepped);
    }
}
