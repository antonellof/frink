//! The routed FFN against a host twin: a decode token through the
//! matvecs and a prefill batch through the grouped GEMMs, both routed by
//! the caller on the device's logits. The device tests are `#[ignore]`d.

use super::super::tests::{
    assert_close, host_causal_gqa_sinks, host_rms_norm_rows, host_swiglu_oai, q8_0_proj,
    q8_1_round, small_q8_0, wave, BATCH_GEMM_TOL,
};
use super::super::*;
use crate::mul_mm::Q8_0;
use crate::mul_mm_ref::mul_mm_reference;
use crate::prefill::moe_device::{DeviceGating, DeviceRouting, ExpertSource};

/// A routed layer through the stack at one token: the router's logits
/// come back to the caller's routing, which picks two of four experts,
/// and the layer's output is the host twin's, the experts summed from
/// zero in the routing's order and then added to the residual.
#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn a_routed_layer_matches_its_host_twin_routing_on_the_device_logits() {
    routed_layer_matches_its_host_twin(1, false);
}

/// The same layer over a prefill batch: every token's logits come back
/// in one download and are routed in token order, and the experts run
/// as grouped GEMMs over the tokens each one was given.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up). Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn a_routed_batch_matches_its_host_twin_through_the_grouped_gemms() {
    routed_layer_matches_its_host_twin(6, false);
}

/// A gpt-oss layer at one token: attention sinks in the decode
/// kernel's combine, `wo`'s bias, the router's bias in the caller's
/// routing, each expert's three biases and the clamped SwiGLU on the
/// per-expert path.
#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn a_gpt_oss_layer_matches_its_host_twin_at_one_token() {
    routed_layer_matches_its_host_twin(1, true);
}

/// The same gpt-oss layer over a prefill batch: the sinks in the tiled
/// prefill attention, and each pair's expert biases through the grouped
/// GEMMs' bias tables.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up). Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn a_gpt_oss_batch_matches_its_host_twin_through_the_grouped_gemms() {
    routed_layer_matches_its_host_twin(6, true);
}

/// Router margins the fixture keeps for every token: the second pick
/// beats the third by at least this, so the device's f16 operands and
/// the host's f32 cannot route a token differently.
const ROUTE_MARGIN: f32 = 0.05;

/// How far a batch's device logits may sit from the host's, relative to
/// the larger of the logit and the token's RMS (`assert_close`). The
/// twin rounds the batch GEMMs' inputs through q8_1 as the device does,
/// so this is f32 accumulation order and an occasional run rounded the
/// other way.
const BATCH_LOGIT_TOL: f32 = 5e-3;

/// The routed fixture's per-token logits and top-2 softmax picks, on the
/// host. Kept apart from the device so the margins can be checked
/// without one (`the_routed_fixture_routes_with_a_margin`).
struct RoutedTwin {
    /// The router's logits, as the device computes them.
    logits: Vec<Vec<f32>>,
    /// What the routing sees: the logits plus gpt-oss's router bias.
    routed: Vec<Vec<f32>>,
    picks: Vec<Vec<(usize, f32)>>,
    hidden: Vec<f32>,
}

/// One of gpt-oss's additions, for [`RoutedFixture::twin_without`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum OaiPart {
    Sinks,
    OBias,
    ExpertBias,
    Clamp,
}

/// gpt-oss's additions to the routed layer, leaked so the routing can
/// lend them for the stack's lifetime.
struct OaiExtras {
    sinks: &'static [f32],
    o_bias: &'static [f32],
    router_bias: &'static [f32],
    expert_bias: Vec<ExpertBiasCuda<'static>>,
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
    oai: Option<OaiExtras>,
}

impl RoutedFixture {
    fn new(batch: usize, oai: bool) -> Self {
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
        let leak = |v: Vec<f32>| -> &'static [f32] { Box::leak(v.into_boxed_slice()) };
        let oai = oai.then(|| OaiExtras {
            sinks: leak((0..n_heads).map(|h| 0.4 - 0.7 * h as f32).collect()),
            o_bias: leak(wave(hidden, 2.3).iter().map(|v| 0.1 * v).collect()),
            router_bias: leak(vec![0.05, -0.1, 0.15, 0.0]),
            expert_bias: (0..n_experts)
                .map(|e| ExpertBiasCuda {
                    gate: leak(wave(ffn, 3.1 + e as f32).iter().map(|v| 0.2 * v).collect()),
                    up: leak(wave(ffn, 4.7 + e as f32).iter().map(|v| 0.2 * v).collect()),
                    down: leak(
                        wave(hidden, 5.9 + e as f32)
                            .iter()
                            .map(|v| 0.05 * v)
                            .collect(),
                    ),
                })
                .collect(),
        });
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
            oai,
        }
    }

    fn glu(&self) -> GluCuda {
        if self.oai.is_some() {
            GluCuda::SwigluOai
        } else {
            GluCuda::Silu
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
        self.twin_without(batch, None)
    }

    /// The twin with one of gpt-oss's additions left out, to show the
    /// fixture can tell each from its absence.
    fn twin_without(&self, batch: usize, left_out: Option<OaiPart>) -> RoutedTwin {
        let (hidden, ffn) = (self.hidden, self.ffn);
        let keep = |p: OaiPart| left_out != Some(p);
        let q_w = self.n_heads * self.head_dim;
        let kv_w = self.n_kv_heads * self.head_dim;
        // A batch's projections are the int8 GEMM's, which takes its
        // input through q8_1; a decode token's are f32 matvecs.
        let mm = |data: &[u8], rows, cols, x: &[f32], n: usize| {
            let x = if batch > 1 {
                q8_1_round(x, cols)
            } else {
                x.to_vec()
            };
            mul_mm_reference(&Q8_0, data, &x, rows, cols, n, cols / 32 * 34).unwrap()
        };
        let [qd, kd, vd, od] = self.attn;
        let normed = host_rms_norm_rows(&self.hidden_in, &self.attn_norm_w, hidden, self.eps);
        let q = mm(qd, q_w, hidden, &normed, batch);
        let k = mm(kd, kv_w, hidden, &normed, batch);
        let v = mm(vd, kv_w, hidden, &normed, batch);
        let oai = self.oai.as_ref();
        let sinks = oai.filter(|_| keep(OaiPart::Sinks)).map(|o| o.sinks);
        let attn = host_causal_gqa_sinks(&q, &k, &v, batch, &self.args(), sinks);
        let mut o = mm(od, hidden, q_w, &attn, batch);
        if let Some(oai) = oai.filter(|_| keep(OaiPart::OBias)) {
            for row in o.chunks_mut(hidden) {
                for (x, b) in row.iter_mut().zip(oai.o_bias) {
                    *x += b;
                }
            }
        }
        let mut h: Vec<f32> = self.hidden_in.iter().zip(&o).map(|(a, b)| a + b).collect();
        let normed2 = host_rms_norm_rows(&h, &self.ffn_norm_w, hidden, self.eps);
        let mut logits = Vec::new();
        let mut routed = Vec::new();
        let mut picks = Vec::new();
        let add = |x: &mut [f32], b: &[f32]| x.iter_mut().zip(b).for_each(|(x, b)| *x += b);
        for (t, x) in normed2.chunks(hidden).enumerate() {
            let lg: Vec<f32> = self
                .router
                .chunks(hidden)
                .map(|r| r.iter().zip(x).map(|(w, x)| w * x).sum())
                .collect();
            let rt = self.route_logits(&lg);
            let pk = top2_softmax(&rt);
            let mut sum = vec![0f32; hidden];
            for &(eid, w) in &pk {
                let (gd, ud, dd) = self.experts[eid];
                let bias = oai
                    .filter(|_| keep(OaiPart::ExpertBias))
                    .map(|o| o.expert_bias[eid]);
                let mut g = mm(gd, ffn, hidden, x, 1);
                let mut u = mm(ud, ffn, hidden, x, 1);
                if let Some(b) = bias {
                    add(&mut g, b.gate);
                    add(&mut u, b.up);
                }
                let act: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| match oai.filter(|_| keep(OaiPart::Clamp)) {
                        Some(_) => host_swiglu_oai(g, u),
                        None => g / (1.0 + (-g).exp()) * u,
                    })
                    .collect();
                let mut d = mm(dd, hidden, ffn, &act, 1);
                if let Some(b) = bias {
                    add(&mut d, b.down);
                }
                for (s, y) in sum.iter_mut().zip(&d) {
                    *s += y * w;
                }
            }
            for (a, b) in h[t * hidden..(t + 1) * hidden].iter_mut().zip(&sum) {
                *a += b;
            }
            logits.push(lg);
            routed.push(rt);
            picks.push(pk);
        }
        RoutedTwin {
            logits,
            routed,
            picks,
            hidden: h,
        }
    }

    /// The logits the routing sees: gpt-oss adds its router's bias, as
    /// `Decoder::gpt_oss_route` does.
    fn route_logits(&self, logits: &[f32]) -> Vec<f32> {
        match &self.oai {
            Some(o) => logits
                .iter()
                .zip(o.router_bias)
                .map(|(l, b)| l + b)
                .collect(),
            None => logits.to_vec(),
        }
    }
}

/// The routed fixture is one the device cannot route differently, and
/// at a batch it gives the experts different tokens: checked on the
/// host, so a fixture change is caught without a GPU.
#[test]
fn the_routed_fixture_routes_with_a_margin() {
    for (batch, oai) in [(1usize, false), (6, false), (1, true), (6, true)] {
        let twin = RoutedFixture::new(batch, oai).twin(batch);
        for lg in &twin.routed {
            let mut s = lg.clone();
            s.sort_by(|a, b| b.total_cmp(a));
            assert!(s[1] - s[2] >= ROUTE_MARGIN, "token margin {lg:?}");
            // Two logits each moved by the device's bound cannot swap.
            let rms = (lg.iter().map(|v| v * v).sum::<f32>() / lg.len() as f32).sqrt();
            let bound = BATCH_LOGIT_TOL * s[1].abs().max(s[2].abs()).max(rms);
            assert!(
                s[1] - s[2] > 2.0 * bound,
                "margin inside the device bound {lg:?}"
            );
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

/// Each of gpt-oss's additions moves the gpt-oss twin's output by many
/// times the device test's tolerance, so a stack that dropped one (a
/// sink ignored, a bias not added, SwiGLU run unclamped) fails it.
#[test]
fn the_gpt_oss_fixture_tells_each_addition_from_its_absence() {
    for batch in [1usize, 6] {
        let fx = RoutedFixture::new(batch, true);
        let full = fx.twin(batch).hidden;
        let rms = (full.iter().map(|v| v * v).sum::<f32>() / full.len() as f32).sqrt();
        for part in [
            OaiPart::Sinks,
            OaiPart::OBias,
            OaiPart::ExpertBias,
            OaiPart::Clamp,
        ] {
            let without = fx.twin_without(batch, Some(part)).hidden;
            let moved = full
                .iter()
                .zip(&without)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                moved > 10.0 * 1e-2 * rms,
                "batch {batch}: leaving out {part:?} moves the output by {moved} (rms {rms})"
            );
        }
    }
}

fn routed_layer_matches_its_host_twin(batch: usize, oai: bool) {
    let fx = RoutedFixture::new(batch, oai);
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
            device: None,
        },
        glu: fx.glu(),
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda {
            sinks: fx.oai.as_ref().map(|o| o.sinks),
            o_bias: fx.oai.as_ref().map(|o| o.o_bias),
            ..AttnExtrasCuda::default()
        },
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
            let tol = if batch == 1 { 1e-3 } else { BATCH_LOGIT_TOL };
            assert_close(got, &twin.logits[asked], tol, "router logits");
            asked += 1;
            Ok(top2_softmax(&fx.route_logits(got))
                .into_iter()
                .map(|(eid, w)| {
                    let (gd, ud, dd) = fx.experts[eid];
                    (
                        ExpertProjCuda {
                            gate: q8_0_proj(gd, ffn, hidden),
                            up: q8_0_proj(ud, ffn, hidden),
                            down: q8_0_proj(dd, hidden, ffn),
                            bias: fx.oai.as_ref().map(|o| o.expert_bias[eid]),
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
    let tol = if batch == 1 { 1e-2 } else { BATCH_GEMM_TOL };
    assert_close(&out.hidden, &twin.hidden, tol, "routed layer hidden");

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
            device: None,
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

/// The device routing's arithmetic, `frink_moe`'s three rules as they
/// are written there (`route_top_k_softmax`, its renormalised form, and
/// `route_top_k_softmax_weight`), for the kernel test below. A tie goes
/// to the lower id, which a stable sort by descending value gives.
fn host_route(logits: &[f32], bias: Option<&[f32]>, k: usize, mode: i32) -> Vec<(usize, f32)> {
    let l: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &x)| x + bias.map_or(0.0, |b| b[i]))
        .collect();
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
    let top = &idx[..k];
    if mode == 2 {
        let e: Vec<f32> = top.iter().map(|&i| (l[i] - l[top[0]]).exp()).collect();
        let s: f32 = e.iter().sum();
        return top.iter().zip(&e).map(|(&i, &x)| (i, x / s)).collect();
    }
    let max = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = l.iter().map(|&x| (x - max).exp()).sum();
    let mut w: Vec<f32> = top.iter().map(|&i| (l[i] - max).exp() / sum).collect();
    if mode == 1 {
        let ts: f32 = w.iter().sum();
        w.iter_mut().for_each(|x| *x /= ts);
    }
    top.iter().copied().zip(w).collect()
}

/// The routing kernel against the host's rules: every mode, expert
/// counts that fill and leave idle a lane's eight slots, `k` up to the
/// kernel's 16, a router bias, and several tokens at once.
#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an L40S (sm_89, Hugging Face Jobs)"]
fn the_device_routing_picks_the_host_s_experts_and_weights() {
    use super::super::enqueue::enqueue_moe_topk;
    let dev = crate::gpu::shared_device().unwrap();
    for (n, k, batch) in [
        (128usize, 8usize, 3usize),
        (32, 4, 1),
        (256, 16, 2),
        (45, 2, 5),
    ] {
        for mode in [0, 1, 2] {
            for biased in [false, true] {
                let logits: Vec<f32> = (0..batch * n)
                    .map(|i| ((i as f32) * 0.37).sin() * 3.0 + ((i * 7 % 13) as f32) * 0.01)
                    .collect();
                let bias: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.11).cos() * 0.5).collect();
                let d_l = crate::xfer::htod(&dev, "test logits", &logits).unwrap();
                let d_b = crate::xfer::htod(&dev, "test bias", &bias).unwrap();
                let mut ids = dev.alloc_zeros::<i32>(batch * k).unwrap();
                let w = enqueue_moe_topk(
                    &dev,
                    &d_l,
                    biased.then_some(&d_b),
                    n,
                    k,
                    mode,
                    batch,
                    &mut ids.slice_mut(..),
                )
                .unwrap();
                let got_ids = crate::xfer::dtoh(&dev, "test ids", &ids).unwrap();
                let got_w = crate::xfer::dtoh(&dev, "test w", &w).unwrap();
                for t in 0..batch {
                    let want = host_route(
                        &logits[t * n..(t + 1) * n],
                        biased.then_some(&bias[..]),
                        k,
                        mode,
                    );
                    for (s, &(id, wt)) in want.iter().enumerate() {
                        let what = format!("n={n} k={k} mode={mode} bias={biased} t={t} slot {s}");
                        assert_eq!(got_ids[t * k + s] as usize, id, "{what}");
                        assert!(
                            (got_w[t * k + s] - wt).abs() <= 1e-5 * wt.max(1e-3),
                            "{what}"
                        );
                    }
                }
            }
        }
    }
}

/// The fixture's experts as the device's routing tables take them.
struct FixtureExperts<'f>(&'f RoutedFixture);

impl ExpertSource for FixtureExperts<'_> {
    fn experts(&self, _layer: usize) -> Vec<ExpertProjCuda<'_>> {
        let (hidden, ffn) = (self.0.hidden, self.0.ffn);
        self.0
            .experts
            .iter()
            .enumerate()
            .map(|(e, &(gd, ud, dd))| ExpertProjCuda {
                gate: q8_0_proj(gd, ffn, hidden),
                up: q8_0_proj(ud, ffn, hidden),
                down: q8_0_proj(dd, hidden, ffn),
                bias: self.0.oai.as_ref().map(|o| o.expert_bias[e]),
            })
            .collect()
    }
}

/// A decode token routed on the device matches the host twin, plain and
/// gpt-oss alike, without asking the caller's routing, and the experts
/// it chose come back in the host routing's order.
#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an L40S (sm_89, Hugging Face Jobs)"]
fn a_device_routed_token_matches_its_host_twin_without_the_host_routing() {
    device_routed_matches_its_host_twin(1);
}

/// The same over a prefill batch: the tokens grouped by expert on the
/// device and every projection one grouped GEMM, with no host routing
/// either, every token's experts back in order.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up). Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn a_device_routed_batch_matches_its_host_twin_grouped_on_the_device() {
    device_routed_matches_its_host_twin(6);
}

fn device_routed_matches_its_host_twin(batch: usize) {
    for oai in [false, true] {
        let fx = RoutedFixture::new(batch, oai);
        let twin = fx.twin(batch);
        let (hidden, ffn) = (fx.hidden, fx.ffn);
        let q_w = fx.n_heads * fx.head_dim;
        let kv_w = fx.n_kv_heads * fx.head_dim;
        let [qd, kd, vd, od] = fx.attn;
        let source = FixtureExperts(&fx);
        let (e0, e_last) = (fx.experts[0].0, fx.experts[fx.experts.len() - 1].2);
        let device = DeviceRouting {
            // Unique per form: the cache is keyed by it.
            key: if oai { 0xD00D_0001 } else { 0xD00D_0000 },
            probe: [e0, e_last],
            n_experts: fx.experts.len(),
            k: 2,
            // Two of four renormalised is the twin's softmax over the two
            // chosen; gpt-oss's rule adds its router bias first.
            gating: if oai {
                DeviceGating::SoftmaxWeight
            } else {
                DeviceGating::Softmax { norm: true }
            },
            router_bias: fx.oai.as_ref().map(|o| o.router_bias),
            source: &source,
            layer: 0,
        };
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
                device: Some(device),
            },
            glu: fx.glu(),
            post_attn_norm: None,
            post_ffn_norm: None,
            extras: AttnExtrasCuda {
                sinks: fx.oai.as_ref().map(|o| o.sinks),
                o_bias: fx.oai.as_ref().map(|o| o.o_bias),
                ..AttnExtrasCuda::default()
            },
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
        let mut host_route =
            |_: usize, _: &[f32]| -> Result<Vec<(ExpertProjCuda<'static>, f32)>, CudaError> {
                asked += 1;
                Err(CudaError::Unsupported(
                    "the device routes this layer".to_string(),
                ))
            };
        let out = launch_stack(
            StackInput::Hidden(&fx.hidden_in),
            &[(&layer, &params)],
            batch,
            StackKv::UploadPrefix,
            Some(&mut host_route),
        )
        .unwrap();
        assert_eq!(
            asked, 0,
            "batch={batch} oai={oai}: the host routing was asked"
        );
        let want_ids: Vec<usize> = twin
            .picks
            .iter()
            .flat_map(|p| p.iter().map(|&(e, _)| e))
            .collect();
        assert_eq!(
            out.device_routed,
            vec![Some(want_ids)],
            "batch={batch} oai={oai}"
        );
        let tol = if batch == 1 { 1e-2 } else { BATCH_GEMM_TOL };
        assert_close(&out.hidden, &twin.hidden, tol, "device-routed layer hidden");
    }
}
