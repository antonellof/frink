//! The gated delta-net step (Qwen3-Next / Qwen3.5 linear attention), as
//! llama.cpp's autoregressive graph computes it
//! (`delta-net-base.cpp:289-365`, `build_delta_net_autoregressive`).
//!
//! Per token, per V head `h` reading K head `kh`:
//!
//! ```text
//! q'      = q / sqrt(S)                                   :319-321
//! S       = S * exp(g_h)                                  :339-340   g_h <= 0, one scalar per head
//! pred[j] = sum_i S[j][i] * k[i]                          :343-345   ("sk", the state's guess at v)
//! d[j]    = (v[j] - pred[j]) * beta_h                     :348-350
//! S[j][i] += k[i] * d[j]                                  :357-361
//! o[j]    = sum_i S[j][i] * q'[i]                         :362-363
//! ```
//!
//! with the state `[n_v_heads][S][S]` as `S[j][i]`, `i` the key index
//! innermost (ggml's `{S_v, S_v, H_v}` with `ne0` the key dim: `:344`
//! multiplies `k` along `ne0`, and the output at `:362` multiplies `q`
//! along it too). The chunked prefill kernel (`:16-287`) is the same
//! recurrence in a different summation order.
//!
//! Which K head a V head reads is the caller's: `Qwen3.5` tiles
//! (`h % n_k_heads`, `llama-model.cpp:524-526`, the converter having
//! reordered V heads for `ggml_repeat`), `Qwen3-Next` groups
//! (`h / (n_v / n_k)`). [`HeadMap`] names the two so the wrong one
//! cannot be assumed.
//!
//! This file is the arithmetic only; `frink_models::gdn` owns the
//! projections, the conv, the norms and the gates around it.

/// How V head `h` finds its K head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadMap {
    /// `h % n_k_heads` (Qwen3.5: `llama-model.cpp:526`, "k0_v0, k1_v1,
    /// k0_v2, k1_v3").
    Tiled,
    /// `h / (n_v_heads / n_k_heads)` (Qwen3-Next: `:525`, "k0_v0, k0_v1,
    /// k1_v2, k1_v3").
    Grouped,
}

impl HeadMap {
    pub fn k_head(self, v_head: usize, n_k_heads: usize, n_v_heads: usize) -> usize {
        match self {
            HeadMap::Tiled => v_head % n_k_heads,
            HeadMap::Grouped => v_head / (n_v_heads / n_k_heads),
        }
    }
}

/// The geometry one step needs. `head_dim` is both the key and the
/// value width: the autoregressive graph asserts `S_k == S_v` (`:307`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaDims {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_dim: usize,
    pub map: HeadMap,
}

impl DeltaDims {
    /// Floats in one sequence's state.
    pub fn state_len(self) -> usize {
        self.n_v_heads * self.head_dim * self.head_dim
    }
}

/// `x / max(||x||, eps)`, ggml's `ggml_l2_norm` (ops.cpp:4185-4210): the
/// sum of squares in double, the divisor clamped by `eps` from below.
pub fn l2_normalize(x: &mut [f32], eps: f32) {
    let sum: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let scale = 1.0 / (sum as f32).sqrt().max(eps);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// One token of the delta rule, in place on `state`
/// (`[n_v_heads][head_dim][head_dim]`, value index outer, key index
/// inner).
///
/// `q` and `k` are `[n_k_heads][head_dim]` (already l2-normed; `q` is
/// scaled by `1/sqrt(head_dim)` HERE, `:319`), `v` is
/// `[n_v_heads][head_dim]`, `g` is `[n_v_heads]` (the log decay, the
/// `exp` taken here, `:339`), `beta` is `[n_v_heads]` (after the
/// sigmoid), `out` is `[n_v_heads][head_dim]`.
#[allow(clippy::too_many_arguments)] // the six operands ggml_gated_delta_net takes, plus the dims and the output
pub fn delta_step(
    dims: DeltaDims,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    out: &mut [f32],
) {
    let DeltaDims {
        n_k_heads,
        n_v_heads,
        head_dim: s,
        map,
    } = dims;
    assert_eq!(state.len(), dims.state_len());
    assert_eq!(q.len(), n_k_heads * s);
    assert_eq!(k.len(), n_k_heads * s);
    assert_eq!(v.len(), n_v_heads * s);
    assert_eq!(g.len(), n_v_heads);
    assert_eq!(beta.len(), n_v_heads);
    assert_eq!(out.len(), n_v_heads * s);
    assert!(n_k_heads > 0 && n_v_heads.is_multiple_of(n_k_heads), ":308");
    let scale = 1.0 / (s as f32).sqrt();
    // Heads are independent (each owns an `S x S` state and `S`
    // outputs), so they run as one parallel region over the head axis;
    // a serial version of this loop was 15% of a Bonsai-2-27B decode
    // step (48 layers x 48 heads x 128 x 128 state floats per token).
    // The two inner reductions are written over four accumulators so
    // the compiler can vectorise them (a single-accumulator float sum
    // cannot be reordered without fast-math).
    crate::par::chunks_mut2_by(state, out, s * s, s, 1, |h, st, oh| {
        let kh = map.k_head(h, n_k_heads, n_v_heads);
        let (qh, kk) = (&q[kh * s..(kh + 1) * s], &k[kh * s..(kh + 1) * s]);
        let vh = &v[h * s..(h + 1) * s];
        let decay = g[h].exp();
        let mut d = vec![0.0f32; s];
        // :340 then :343-350: decay, the state's prediction, the error
        // scaled by beta.
        for j in 0..s {
            let row = &mut st[j * s..(j + 1) * s];
            let pred = decay_and_dot(row, kk, decay);
            d[j] = (vh[j] - pred) * beta[h];
        }
        // :357-363: the rank-one update, then the read-out with the
        // scaled query.
        for j in 0..s {
            let row = &mut st[j * s..(j + 1) * s];
            oh[j] = update_and_dot(row, kk, d[j], qh) * scale;
        }
    });
}

/// `row *= decay`, then `row . k`, over four lanes of accumulation.
#[inline]
fn decay_and_dot(row: &mut [f32], k: &[f32], decay: f32) -> f32 {
    let mut acc = [0.0f32; 4];
    let (rb, rt) = row.as_chunks_mut::<4>();
    let (kb, kt) = k.as_chunks::<4>();
    for (r, kk) in rb.iter_mut().zip(kb) {
        for l in 0..4 {
            r[l] *= decay;
            acc[l] += r[l] * kk[l];
        }
    }
    let mut tail = 0.0f32;
    for (r, kk) in rt.iter_mut().zip(kt) {
        *r *= decay;
        tail += *r * *kk;
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

/// `row += k * d`, then `row . q`, over four lanes of accumulation.
#[inline]
fn update_and_dot(row: &mut [f32], k: &[f32], d: f32, q: &[f32]) -> f32 {
    let mut acc = [0.0f32; 4];
    let (rb, rt) = row.as_chunks_mut::<4>();
    let (kb, kt) = k.as_chunks::<4>();
    let (qb, qt) = q.as_chunks::<4>();
    for ((r, kk), qq) in rb.iter_mut().zip(kb).zip(qb) {
        for l in 0..4 {
            r[l] += kk[l] * d;
            acc[l] += r[l] * qq[l];
        }
    }
    let mut tail = 0.0f32;
    for ((r, kk), qq) in rt.iter_mut().zip(kt).zip(qt) {
        *r += *kk * d;
        tail += *r * *qq;
    }
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device step against this one at B=1, printed rather than
    /// asserted: run with `--nocapture`.
    ///
    /// This is the number the three losses in
    /// `1ebda4fc:docs/plans/gdn-resident-state.md` came down to, and it was
    /// never measured on its own -- only end to end, where it read as
    /// "the kernel is not better than six cores" without saying why.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn device_delta_step_against_host_throughput() {
        use frink_metal::gdn::{DeltaShape, HeadMapKind};

        let (n_k, n_v, s) = (4usize, 48usize, 128usize);
        let dims = DeltaDims {
            n_k_heads: n_k,
            n_v_heads: n_v,
            head_dim: s,
            map: HeadMap::Tiled,
        };
        let shape = DeltaShape {
            n_k_heads: n_k,
            n_v_heads: n_v,
            head_dim: s,
            map: HeadMapKind::Tiled,
        };
        let mut seed = 11u32;
        let mut draw = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / 8388608.0 - 1.0) * scale
                })
                .collect()
        };
        let state0 = draw(dims.state_len(), 0.5);
        let q = draw(n_k * s, 1.0);
        let k = draw(n_k * s, 1.0);
        let v = draw(n_v * s, 1.0);
        let g: Vec<f32> = draw(n_v, 3.0).iter().map(|x| -x.abs()).collect();
        let beta: Vec<f32> = draw(n_v, 1.0).iter().map(|x| 0.5 + 0.25 * x).collect();
        let reps = 20;

        let mut host_state = state0.clone();
        let mut out = vec![0.0f32; n_v * s];
        delta_step(dims, &mut host_state, &q, &k, &v, &g, &beta, &mut out);
        let t = std::time::Instant::now();
        for _ in 0..reps {
            delta_step(dims, &mut host_state, &q, &k, &v, &g, &beta, &mut out);
        }
        let host_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let mut device_state = state0.clone();
        frink_metal::gdn::launch_delta_step(shape, &mut device_state, &q, &k, &v, &g, &beta)
            .expect("warm");
        let t = std::time::Instant::now();
        for _ in 0..reps {
            frink_metal::gdn::launch_delta_step(shape, &mut device_state, &q, &k, &v, &g, &beta)
                .expect("the kernel launches");
        }
        let device_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        // The state is read and written once, which is the floor either
        // side is working against.
        let gb = 2.0 * dims.state_len() as f64 * 4.0 / 1e9;
        eprintln!(
            "one row, {n_v}x{s}: host {host_us:.0} us ({:.1} GB/s), device {device_us:.0} us \
             ({:.1} GB/s, uploads and readback included)",
            gb / (host_us / 1e6),
            gb / (device_us / 1e6),
        );
    }

    /// The device kernel IS this recurrence: same decay, same
    /// prediction, same beta-scaled error, same rank-one update, same
    /// scaled read-out, and the same state left behind. Run on the M2
    /// Pro; a kernel that disagreed would move a Bonsai token's logits
    /// without moving any test that does not launch it.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_device_delta_step_matches_this_one() {
        use frink_metal::gdn::{DeltaShape, HeadMapKind};

        for (n_k, n_v, s, map, kind) in [
            (
                4usize,
                48usize,
                128usize,
                HeadMap::Tiled,
                HeadMapKind::Tiled,
            ),
            (4, 48, 128, HeadMap::Grouped, HeadMapKind::Grouped),
            (2, 2, 4, HeadMap::Tiled, HeadMapKind::Tiled),
        ] {
            let dims = DeltaDims {
                n_k_heads: n_k,
                n_v_heads: n_v,
                head_dim: s,
                map,
            };
            let mut seed = 12345u32;
            let mut draw = |n: usize, scale: f32| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        ((seed >> 8) as f32 / 8388608.0 - 1.0) * scale
                    })
                    .collect()
            };
            let state0 = draw(dims.state_len(), 0.5);
            let q = draw(n_k * s, 1.0);
            let k = draw(n_k * s, 1.0);
            let v = draw(n_v * s, 1.0);
            // The decay is `exp(g)`, so g is negative in a real layer.
            let g: Vec<f32> = draw(n_v, 1.0).iter().map(|x| -x.abs()).collect();
            let beta: Vec<f32> = draw(n_v, 1.0).iter().map(|x| 0.5 + 0.25 * x).collect();

            let mut host_state = state0.clone();
            let mut host_out = vec![0.0f32; n_v * s];
            delta_step(dims, &mut host_state, &q, &k, &v, &g, &beta, &mut host_out);

            let mut device_state = state0.clone();
            let shape = DeltaShape {
                n_k_heads: n_k,
                n_v_heads: n_v,
                head_dim: s,
                map: kind,
            };
            let device_out = frink_metal::gdn::launch_delta_step(
                shape,
                &mut device_state,
                &q,
                &k,
                &v,
                &g,
                &beta,
            )
            .expect("the kernel launches");

            let tol = 2e-4;
            for (i, (a, b)) in device_out.iter().zip(host_out.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{n_v}x{s} {map:?} out[{i}]: device={a} host={b}"
                );
            }
            for (i, (a, b)) in device_state.iter().zip(host_state.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * b.abs().max(1.0),
                    "{n_v}x{s} {map:?} state[{i}]: device={a} host={b}"
                );
            }
        }
    }

    /// What the recurrence achieves on this host, printed rather than
    /// asserted: run with `--nocapture`. Bonsai's shape is 48 value
    /// heads of 128, a 3.1 MB state per layer.
    ///
    /// It reads 36 GB/s of state on an M2 Pro, which is the memory
    /// floor for reading and writing that state once per row, not a
    /// missing vectorisation: rewriting the two passes as one read pass
    /// plus a streaming update -- algebraically exact, since
    /// `S_new . q` expands to `decay (S_old . q) + d (k . q)` -- moved
    /// it to 197 us from 175, because both passes already hit the same
    /// 512-byte row while it is hot in L1.
    ///
    /// What DOES move it is reading the state once per CHUNK of rows
    /// instead of once per row (`1ebda4fc:docs/plans/gdn-resident-state.md`).
    #[test]
    #[ignore = "a measurement, not an assertion; run with --nocapture"]
    fn delta_step_throughput_probe() {
        let dims = DeltaDims {
            n_k_heads: 4,
            n_v_heads: 48,
            head_dim: 128,
            map: HeadMap::Tiled,
        };
        let mut state = vec![0.01f32; dims.state_len()];
        let q: Vec<f32> = (0..4 * 128).map(|i| (i as f32 * 0.01).sin()).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..48 * 128).map(|i| (i as f32 * 0.02).cos()).collect();
        let g = vec![-0.1f32; 48];
        let beta = vec![0.5f32; 48];
        let mut out = vec![0.0f32; 48 * 128];
        for _ in 0..8 {
            delta_step(dims, &mut state, &q, &k, &v, &g, &beta, &mut out);
        }
        let n = 200;
        let t = std::time::Instant::now();
        for _ in 0..n {
            delta_step(dims, &mut state, &q, &k, &v, &g, &beta, &mut out);
        }
        let per = t.elapsed().as_secs_f64() / n as f64;
        let flops = 48.0 * 2.0 * 2.0 * 128.0 * 128.0;
        let bytes = (dims.state_len() * 4 * 2) as f64;
        eprintln!(
            "delta_step {:.1} us/call, {:.1} GFLOP/s, {:.1} GB/s of state",
            per * 1e6,
            flops / per / 1e9,
            bytes / per / 1e9
        );
    }

    #[test]
    fn l2_normalize_clamps_the_divisor() {
        let mut v = [3.0f32, 4.0];
        l2_normalize(&mut v, 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = [0.0f32, 0.0];
        l2_normalize(&mut z, 1e-6);
        assert_eq!(z, [0.0, 0.0]);
    }

    /// One head, `S = 1`: the scalar recurrence by hand.
    #[test]
    fn the_scalar_recurrence() {
        let dims = DeltaDims {
            n_k_heads: 1,
            n_v_heads: 1,
            head_dim: 1,
            map: HeadMap::Tiled,
        };
        let mut st = vec![2.0f32];
        let mut out = [0.0f32];
        // decay exp(0) = 1; pred = 2 * k(1) = 2; d = (v(5) - 2) * beta(0.5) = 1.5;
        // S = 2 + 1 * 1.5 = 3.5; o = 3.5 * q(1) * 1.
        delta_step(
            dims,
            &mut st,
            &[1.0],
            &[1.0],
            &[5.0],
            &[0.0],
            &[0.5],
            &mut out,
        );
        assert!((st[0] - 3.5).abs() < 1e-6 && (out[0] - 3.5).abs() < 1e-6);
    }

    /// Two K heads, four V heads: tiled reads `h % 2`, grouped `h / 2`.
    #[test]
    fn the_two_head_maps_differ_and_are_the_documented_ones() {
        assert_eq!(HeadMap::Tiled.k_head(3, 2, 4), 1);
        assert_eq!(HeadMap::Grouped.k_head(3, 2, 4), 1);
        assert_eq!(HeadMap::Tiled.k_head(1, 2, 4), 1);
        assert_eq!(HeadMap::Grouped.k_head(1, 2, 4), 0);
        let mk = |map| DeltaDims {
            n_k_heads: 2,
            n_v_heads: 4,
            head_dim: 1,
            map,
        };
        let (q, k) = ([1.0f32, 1.0], [1.0f32, 10.0]);
        let v = [1.0f32; 4];
        let mut out_t = [0.0f32; 4];
        let mut out_g = [0.0f32; 4];
        delta_step(
            mk(HeadMap::Tiled),
            &mut [0.0; 4],
            &q,
            &k,
            &v,
            &[0.0; 4],
            &[1.0; 4],
            &mut out_t,
        );
        delta_step(
            mk(HeadMap::Grouped),
            &mut [0.0; 4],
            &q,
            &k,
            &v,
            &[0.0; 4],
            &[1.0; 4],
            &mut out_g,
        );
        // S = k * v after one step (from zero): o = k * q.
        assert_eq!(out_t, [1.0, 10.0, 1.0, 10.0]);
        assert_eq!(out_g, [1.0, 1.0, 10.0, 10.0]);
    }
}
