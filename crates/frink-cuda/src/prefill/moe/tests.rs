//! The routed FFN against a host twin: a decode token through the
//! matvecs and a prefill batch through the grouped GEMMs, both routed by
//! the caller on the device's logits. The device tests are `#[ignore]`d.

use super::super::tests::{
    assert_close, host_causal_gqa, host_rms_norm_rows, q8_0_proj, small_q8_0, wave,
};
use super::super::*;
use crate::mul_mm::Q8_0;
use crate::mul_mm_ref::mul_mm_reference;

/// A routed layer through the stack at one token: the router's logits
/// come back to the caller's routing, which picks two of four experts,
/// and the layer's output is the host twin's, the experts summed from
/// zero in the routing's order and then added to the residual.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn a_routed_layer_matches_its_host_twin_routing_on_the_device_logits() {
    routed_layer_matches_its_host_twin(1);
}

/// The same layer over a prefill batch: every token's logits come back
/// in one download and are routed in token order, and the experts run
/// as grouped GEMMs over the tokens each one was given.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up); not yet run"]
fn a_routed_batch_matches_its_host_twin_through_the_grouped_gemms() {
    routed_layer_matches_its_host_twin(6);
}

/// Router margins the fixture keeps for every token: the second pick
/// beats the third by at least this, so the device's f16 operands and
/// the host's f32 cannot route a token differently.
const ROUTE_MARGIN: f32 = 0.05;

/// The routed fixture's per-token logits and top-2 softmax picks, on the
/// host. Kept apart from the device so the margins can be checked
/// without one (`the_routed_fixture_routes_with_a_margin`).
struct RoutedTwin {
    logits: Vec<Vec<f32>>,
    picks: Vec<Vec<(usize, f32)>>,
    hidden: Vec<f32>,
}

fn top2_softmax(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut ids: Vec<usize> = (0..logits.len()).collect();
    ids.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
    let top = &ids[..2];
    let m = logits[top[0]];
    let e: Vec<f32> = top.iter().map(|&i| (logits[i] - m).exp()).collect();
    let s: f32 = e.iter().sum();
    top.iter().zip(&e).map(|(&i, &x)| (i, x / s)).collect()
}

/// The routed layer's fixture: four Q8_0 experts, an f32 router whose
/// rows are distinct waves (so the tokens of a batch route differently),
/// and the host twin's output.
struct RoutedFixture {
    hidden: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    attn: [&'static [u8]; 4],
    experts: Vec<(&'static [u8], &'static [u8], &'static [u8])>,
    router: Vec<f32>,
    attn_norm_w: Vec<f32>,
    ffn_norm_w: Vec<f32>,
    hidden_in: Vec<f32>,
    eps: f32,
}

impl RoutedFixture {
    fn new(batch: usize) -> Self {
        let (hidden, n_heads, n_kv_heads, head_dim, ffn, n_experts) =
            (64usize, 2usize, 1usize, 32usize, 96usize, 4usize);
        let q_w = n_heads * head_dim;
        let kv_w = n_kv_heads * head_dim;
        let attn = [
            small_q8_0(q_w, hidden, 1),
            small_q8_0(kv_w, hidden, 2),
            small_q8_0(kv_w, hidden, 3),
            small_q8_0(hidden, q_w, 4),
        ];
        let experts = (0..n_experts as u32)
            .map(|e| {
                (
                    small_q8_0(ffn, hidden, 10 + 3 * e),
                    small_q8_0(ffn, hidden, 11 + 3 * e),
                    small_q8_0(hidden, ffn, 12 + 3 * e),
                )
            })
            .collect();
        let router: Vec<f32> = (0..n_experts)
            .flat_map(|e| wave(hidden, 0.9 * e as f32 + 0.3))
            .map(|v| 0.4 * v)
            .collect();
        let scale = |seed| {
            wave(hidden, seed)
                .iter()
                .map(|v| 1.0 + 0.1 * v)
                .collect::<Vec<_>>()
        };
        let hidden_in = (0..batch)
            .flat_map(|t| wave(hidden, 0.05 + 1.7 * t as f32))
            .collect();
        Self {
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn,
            attn,
            experts,
            router,
            attn_norm_w: scale(1.1),
            ffn_norm_w: scale(1.3),
            hidden_in,
            eps: 1e-5,
        }
    }

    fn args(&self) -> AttnArgs {
        AttnArgs {
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            start_pos: 0,
            window: None,
            scale: (self.head_dim as f32).sqrt().recip(),
            softcap: None,
        }
    }

    fn twin(&self, batch: usize) -> RoutedTwin {
        let (hidden, ffn) = (self.hidden, self.ffn);
        let q_w = self.n_heads * self.head_dim;
        let kv_w = self.n_kv_heads * self.head_dim;
        let mm = |data: &[u8], rows, cols, x: &[f32], n: usize| {
            mul_mm_reference(&Q8_0, data, x, rows, cols, n, cols / 32 * 34).unwrap()
        };
        let [qd, kd, vd, od] = self.attn;
        let normed = host_rms_norm_rows(&self.hidden_in, &self.attn_norm_w, hidden, self.eps);
        let q = mm(qd, q_w, hidden, &normed, batch);
        let k = mm(kd, kv_w, hidden, &normed, batch);
        let v = mm(vd, kv_w, hidden, &normed, batch);
        let attn = host_causal_gqa(&q, &k, &v, batch, &self.args());
        let o = mm(od, hidden, q_w, &attn, batch);
        let mut h: Vec<f32> = self.hidden_in.iter().zip(&o).map(|(a, b)| a + b).collect();
        let normed2 = host_rms_norm_rows(&h, &self.ffn_norm_w, hidden, self.eps);
        let mut logits = Vec::new();
        let mut picks = Vec::new();
        for (t, x) in normed2.chunks(hidden).enumerate() {
            let lg: Vec<f32> = self
                .router
                .chunks(hidden)
                .map(|r| r.iter().zip(x).map(|(w, x)| w * x).sum())
                .collect();
            let pk = top2_softmax(&lg);
            let mut sum = vec![0f32; hidden];
            for &(eid, w) in &pk {
                let (gd, ud, dd) = self.experts[eid];
                let g = mm(gd, ffn, hidden, x, 1);
                let u = mm(ud, ffn, hidden, x, 1);
                let act: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                let d = mm(dd, hidden, ffn, &act, 1);
                for (s, y) in sum.iter_mut().zip(&d) {
                    *s += y * w;
                }
            }
            for (a, b) in h[t * hidden..(t + 1) * hidden].iter_mut().zip(&sum) {
                *a += b;
            }
            logits.push(lg);
            picks.push(pk);
        }
        RoutedTwin {
            logits,
            picks,
            hidden: h,
        }
    }
}

/// The routed fixture is one the device cannot route differently, and
/// at a batch it gives the experts different tokens: checked on the
/// host, so a fixture change is caught without a GPU.
#[test]
fn the_routed_fixture_routes_with_a_margin() {
    for batch in [1usize, 6] {
        let twin = RoutedFixture::new(batch).twin(batch);
        for lg in &twin.logits {
            let mut s = lg.clone();
            s.sort_by(|a, b| b.total_cmp(a));
            assert!(s[1] - s[2] >= ROUTE_MARGIN, "token margin {lg:?}");
        }
        if batch > 1 {
            let sets: std::collections::BTreeSet<Vec<usize>> = twin
                .picks
                .iter()
                .map(|p| {
                    let mut ids: Vec<usize> = p.iter().map(|&(e, _)| e).collect();
                    ids.sort_unstable();
                    ids
                })
                .collect();
            assert!(sets.len() >= 2, "every token routed alike: {sets:?}");
        }
    }
}

fn routed_layer_matches_its_host_twin(batch: usize) {
    let fx = RoutedFixture::new(batch);
    let twin = fx.twin(batch);
    let (hidden, ffn) = (fx.hidden, fx.ffn);
    let q_w = fx.n_heads * fx.head_dim;
    let kv_w = fx.n_kv_heads * fx.head_dim;
    let [qd, kd, vd, od] = fx.attn;
    let layer = PrefillDenseLayerCuda {
        attn_norm_w: &fx.attn_norm_w,
        ffn_norm_w: &fx.ffn_norm_w,
        q: q8_0_proj(qd, q_w, hidden),
        k: q8_0_proj(kd, kv_w, hidden),
        v: q8_0_proj(vd, kv_w, hidden),
        o: q8_0_proj(od, hidden, q_w),
        ffn: StackFfn::Moe {
            router: RouterCuda::F32 {
                data: &fx.router,
                rows: fx.experts.len(),
            },
            ffn_dim: ffn,
        },
        glu: GluCuda::Silu,
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: None,
    };
    let params = PrefillParams {
        n_heads: fx.n_heads,
        n_kv_heads: fx.n_kv_heads,
        head_dim: fx.head_dim,
        rms_eps: fx.eps,
        attn_scale: fx.args().scale,
        attn_softcap: None,
        window: None,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    };
    let mut asked = 0usize;
    let mut device_route =
        |i: usize, got: &[f32]| -> Result<Vec<(ExpertProjCuda<'static>, f32)>, CudaError> {
            assert_eq!(i, 0);
            // Tokens are routed in order, once each.
            assert_close(got, &twin.logits[asked], 1e-3, "router logits");
            asked += 1;
            Ok(top2_softmax(got)
                .into_iter()
                .map(|(eid, w)| {
                    let (gd, ud, dd) = fx.experts[eid];
                    (
                        ExpertProjCuda {
                            gate: q8_0_proj(gd, ffn, hidden),
                            up: q8_0_proj(ud, ffn, hidden),
                            down: q8_0_proj(dd, hidden, ffn),
                        },
                        w,
                    )
                })
                .collect())
        };
    let out = launch_stack(
        StackInput::Hidden(&fx.hidden_in),
        &[(&layer, &params)],
        batch,
        StackKv::UploadPrefix,
        Some(&mut device_route),
    )
    .unwrap();
    assert_eq!(asked, batch, "the routing is asked once per token");
    assert_close(&out.hidden, &twin.hidden, 1e-2, "routed layer hidden");

    // Without a routing the same run is refused before the device.
    let err = launch_prefill_dense_stack(
        StackInput::Hidden(&fx.hidden_in),
        &[(&layer, &params)],
        batch,
        StackKv::UploadPrefix,
    )
    .err()
    .expect("refused");
    assert!(format!("{err:?}").contains("no routing"), "{err:?}");
}

/// A routed layer's router is held to its layer before the device, at
/// a decode token and in a batch alike.
#[test]
fn a_routed_layer_s_router_is_held_to_its_layer() {
    let hidden = 64usize;
    let router = vec![0f32; 4 * hidden];
    let short = vec![0f32; 4 * hidden - 1];
    let norm = vec![1f32; hidden];
    let w = small_q8_0(hidden, hidden, 1);
    let layer_with = |data: &'static [f32]| PrefillDenseLayerCuda {
        attn_norm_w: Box::leak(norm.clone().into_boxed_slice()),
        ffn_norm_w: Box::leak(norm.clone().into_boxed_slice()),
        q: q8_0_proj(w, hidden, hidden),
        k: q8_0_proj(w, hidden, hidden),
        v: q8_0_proj(w, hidden, hidden),
        o: q8_0_proj(w, hidden, hidden),
        ffn: StackFfn::Moe {
            router: RouterCuda::F32 { data, rows: 4 },
            ffn_dim: 96,
        },
        glu: GluCuda::Silu,
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: None,
    };
    let params = PrefillParams {
        n_heads: 2,
        n_kv_heads: 2,
        head_dim: 32,
        rms_eps: 1e-5,
        attn_scale: 0.17,
        attn_softcap: None,
        window: None,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    };
    let good = layer_with(Box::leak(router.into_boxed_slice()));
    let bad = layer_with(Box::leak(short.into_boxed_slice()));
    for batch in [1usize, 4] {
        assert!(check_layer_shapes(&good, &params, batch, hidden).is_ok());
        let err = check_layer_shapes(&bad, &params, batch, hidden).unwrap_err();
        assert!(format!("{err:?}").contains("router"), "{err:?}");
    }
}
