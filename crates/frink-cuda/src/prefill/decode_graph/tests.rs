//! The decode graph against the eager stack: the rule on the host, and
//! on a device every token of a replayed run held to the eager run's
//! bits. The device tests are `#[ignore]`d.

use super::*;
use crate::gpu::shared_device;
use crate::mul_mm::Q8_0;
use crate::prefill::kv_mirror::grown_capacity;
use crate::prefill::tests::{q8_0_proj, small_q8_0, wave};
use crate::prefill::{
    launch_stack_in, AttnExtrasCuda, ExpertProjCuda, GluCuda, LayerRopeCuda, MulMmWeights,
    RopeLayoutCuda,
};

/// A run captures after one eager token that could be captured, replays
/// only the key it captured, and a key whose capture failed stays eager
/// until the key changes.
#[test]
fn a_run_captures_after_one_capturable_eager_token_and_replays_only_its_key() {
    let (a, b) = (vec![1u64, 2, 3], vec![1u64, 2, 4]);
    assert_eq!(plan(SlotKey::Cold, &a), Plan::Eager);
    assert_eq!(after_eager(SlotKey::Cold, &a, true), Settle::Warm);
    // A layer routed on the host waits mid-token: never captured.
    assert_eq!(after_eager(SlotKey::Cold, &a, false), Settle::Refused);
    assert_eq!(plan(SlotKey::Warm(&a), &a), Plan::Capture);
    assert_eq!(plan(SlotKey::Warm(&a), &b), Plan::Eager);
    assert_eq!(plan(SlotKey::Ready(&a), &a), Plan::Replay);
    // A mirror grew, or the run took other layers: eager, then warm.
    assert_eq!(plan(SlotKey::Ready(&a), &b), Plan::Eager);
    assert_eq!(after_eager(SlotKey::Ready(&a), &b, true), Settle::Warm);
    // A failed capture is not retried every token...
    assert_eq!(plan(SlotKey::Refused(&a), &a), Plan::Eager);
    assert_eq!(after_eager(SlotKey::Refused(&a), &a, true), Settle::Refused);
    // ...but a new key gets its chance.
    assert_eq!(after_eager(SlotKey::Refused(&a), &b, true), Settle::Warm);
}

/// How the rule runs a sequence decoding `tokens` tokens after `prefix`
/// rows, its key changing only when the mirror's capacity does: eager
/// the first token and the first after each growth, captured the next,
/// replayed the rest. What the device tests expect.
fn expected_uses(prefix: usize, tokens: usize) -> Vec<GraphUse> {
    let mut cap = grown_capacity(0, prefix + 1).unwrap();
    let mut since = 0usize;
    (0..tokens)
        .map(|t| {
            let grown = grown_capacity(cap, prefix + t + 1).unwrap();
            if grown != cap {
                cap = grown;
                since = 0;
            }
            since += 1;
            match since {
                1 => GraphUse::Eager,
                2 => GraphUse::Captured,
                _ => GraphUse::Replayed,
            }
        })
        .collect()
}

/// The device tests' expectation is the rule's: simulated through
/// [`plan`] and [`after_eager`] with the capacity as the key, the same
/// sequence comes out.
#[test]
fn the_expected_sequence_is_what_the_rule_does_as_a_mirror_grows() {
    let (prefix, tokens) = (186, 84);
    let mut cap = grown_capacity(0, prefix + 1).unwrap();
    let mut slot: (u8, Vec<u64>) = (0, Vec::new());
    let mut ran = Vec::new();
    for t in 0..tokens {
        cap = grown_capacity(cap, prefix + t + 1).unwrap();
        let key = vec![cap as u64];
        let view = match slot.0 {
            0 => SlotKey::Cold,
            1 => SlotKey::Warm(&slot.1),
            2 => SlotKey::Refused(&slot.1),
            _ => SlotKey::Ready(&slot.1),
        };
        let (used, next) = match plan(view, &key) {
            Plan::Replay => (GraphUse::Replayed, 3),
            Plan::Capture => (GraphUse::Captured, 3),
            Plan::Eager => match after_eager(view, &key, true) {
                Settle::Warm => (GraphUse::Eager, 1),
                Settle::Refused => (GraphUse::Eager, 2),
            },
        };
        slot = (next, key);
        ran.push(used);
    }
    let want = expected_uses(prefix, tokens);
    assert_eq!(ran, want);
    // The fixture crosses a chunk boundary inside the first graph's life
    // (seq_len 193) and a capacity doubling (256 to 512).
    assert_eq!(want[0], GraphUse::Eager);
    assert_eq!(want[1], GraphUse::Captured);
    assert_eq!(want[256 - prefix], GraphUse::Eager);
    assert_eq!(want[257 - prefix], GraphUse::Captured);
    assert_eq!(
        want.iter().filter(|u| **u == GraphUse::Replayed).count(),
        tokens - 4
    );
}

/// What one token of a run returned.
struct Token {
    hidden: Vec<f32>,
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    routed: Vec<Option<Vec<usize>>>,
    graph: GraphUse,
}

/// A run's per-token input: a fresh hidden row of the width, or a token
/// id an embedding table is gathered at.
pub(in crate::prefill) enum Feed {
    Hidden(usize),
    Gather {
        table: MulMmWeights<'static>,
        vocab: usize,
    },
}

/// Decodes `tokens` tokens through `layers` (their params `base`, the
/// history and position filled per token) after a host history of
/// `prefix` rows a layer, on fresh mirrors, eagerly or with graphs.
fn decode(
    layers: &[&PrefillDenseLayerCuda<'static>],
    base: &[PrefillParams<'static>],
    feed: &Feed,
    prefix: usize,
    tokens: usize,
    graphs: bool,
) -> Vec<Token> {
    let n = layers.len();
    let width = |l: usize| base[l].n_kv_heads * base[l].head_dim;
    let mut hk: Vec<Vec<f32>> = (0..n)
        .map(|l| wave(prefix * width(l), 3.1 + 0.6 * l as f32))
        .collect();
    let mut hv: Vec<Vec<f32>> = (0..n)
        .map(|l| wave(prefix * width(l), 3.3 + 0.6 * l as f32))
        .collect();
    let mut mirrors: Vec<Option<CudaKvMirror>> = (0..n).map(|_| None).collect();
    let mut asked = 0usize;
    let mut run = Vec::with_capacity(tokens);
    for t in 0..tokens {
        let pos = prefix + t;
        let token = {
            let params: Vec<PrefillParams<'_>> = base
                .iter()
                .enumerate()
                .map(|(l, b)| PrefillParams {
                    prefix_k: &hk[l],
                    prefix_v: &hv[l],
                    start_pos: pos,
                    ..*b
                })
                .collect();
            let launch: Vec<_> = layers.iter().zip(&params).map(|(l, p)| (*l, p)).collect();
            let mut slots: Vec<&mut Option<CudaKvMirror>> = mirrors.iter_mut().collect();
            let row;
            let ids;
            let input = match feed {
                Feed::Hidden(w) => {
                    row = wave(*w, 0.05 + 0.37 * t as f32);
                    StackInput::Hidden(&row)
                }
                Feed::Gather { table, vocab } => {
                    ids = [((pos * 7 + 3) % vocab) as u32];
                    StackInput::Gather {
                        table: *table,
                        ids: &ids,
                    }
                }
            };
            let mut route =
                |_: usize, _: &[f32]| -> Result<Vec<(ExpertProjCuda<'static>, f32)>, CudaError> {
                    asked += 1;
                    Err(CudaError::Unsupported(
                        "the device routes every layer here".to_string(),
                    ))
                };
            let s = launch_stack_in(
                input,
                &launch,
                1,
                StackKv::Mirrors(&mut slots),
                Some(&mut route),
                graphs,
            )
            .unwrap_or_else(|e| panic!("graphs={graphs} token {t}: {e}"));
            Token {
                hidden: s.hidden.clone(),
                kv: s.kv_rows_owned(),
                routed: s.device_routed.clone(),
                graph: s.graph,
            }
        };
        for l in 0..n {
            hk[l].extend_from_slice(&token.kv[l].0);
            hv[l].extend_from_slice(&token.kv[l].1);
        }
        run.push(token);
    }
    assert_eq!(asked, 0, "graphs={graphs}: the host routing was asked");
    run
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Decodes the same tokens eagerly and through graph replay, each on
/// fresh mirrors, and holds every token's hidden row, K and V rows and
/// routed experts to the eager run's bit for bit, and the graph run to
/// the rule's sequence of eager, captured and replayed tokens.
pub(in crate::prefill) fn replay_matches_eager(
    layers: &[&PrefillDenseLayerCuda<'static>],
    base: &[PrefillParams<'static>],
    feed: &Feed,
    prefix: usize,
    tokens: usize,
) {
    let dev = shared_device().expect("a CUDA device");
    assert!(
        crate::graph::device_supports(&dev),
        "the graph needs sm_70 or newer with stream-ordered allocation"
    );
    let eager = decode(layers, base, feed, prefix, tokens, false);
    let graph = decode(layers, base, feed, prefix, tokens, true);
    for (t, (e, g)) in eager.iter().zip(&graph).enumerate() {
        assert_eq!(e.graph, GraphUse::Eager, "token {t} of the eager run");
        let at = format!("token {t} at {} ({:?})", prefix + t, g.graph);
        assert_eq!(bits(&g.hidden), bits(&e.hidden), "{at}: hidden");
        for (l, ((gk, gv), (ek, ev))) in g.kv.iter().zip(&e.kv).enumerate() {
            assert_eq!(bits(gk), bits(ek), "{at}: layer {l} K");
            assert_eq!(bits(gv), bits(ev), "{at}: layer {l} V");
        }
        assert_eq!(g.routed, e.routed, "{at}: routed experts");
    }
    let uses: Vec<GraphUse> = graph.iter().map(|t| t.graph).collect();
    assert_eq!(uses, expected_uses(prefix, tokens));
}

/// From position 186 for 84 tokens: the first graph's life crosses the
/// one-row attention's chunk boundary at 192 keys, and the mirror grows
/// from 256 rows to 512 at position 256, which captures again.
pub(in crate::prefill) const PREFIX: usize = 186;
pub(in crate::prefill) const TOKENS: usize = 84;

/// Two dense layers, Llama's and a windowed Qwen3-like one (NEOX RoPE,
/// per-head QK norms, a post-attention norm, a window narrower than the
/// history), fed by the embedding gather, replayed from a graph and held
/// to the eager run token by token.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn a_dense_run_replayed_from_a_graph_is_the_eager_run_to_the_bit() {
    let (hidden, n_heads, n_kv_heads, head_dim, ffn, vocab) =
        (64usize, 2usize, 1usize, 32usize, 96usize, 50usize);
    let (q_w, kv_w) = (n_heads * head_dim, n_kv_heads * head_dim);
    let leak = |v: Vec<f32>| -> &'static [f32] { Box::leak(v.into_boxed_slice()) };
    let norm = |seed| leak(wave(hidden, seed).iter().map(|v| 1.0 + 0.1 * v).collect());
    let head_norm = |seed| leak(wave(head_dim, seed).iter().map(|v| 1.0 + 0.1 * v).collect());
    let ff = leak((0..head_dim / 2).map(|i| 1.0 + i as f32 * 0.1).collect());
    let layer = |seed: u32, windowed: bool| PrefillDenseLayerCuda {
        attn_norm_w: norm(1.1 + seed as f32),
        ffn_norm_w: norm(1.3 + seed as f32),
        q: q8_0_proj(small_q8_0(q_w, hidden, seed), q_w, hidden),
        k: q8_0_proj(small_q8_0(kv_w, hidden, seed + 1), kv_w, hidden),
        v: q8_0_proj(small_q8_0(kv_w, hidden, seed + 2), kv_w, hidden),
        o: q8_0_proj(small_q8_0(hidden, q_w, seed + 3), hidden, q_w),
        ffn: StackFfn::Dense(ExpertProjCuda {
            gate: q8_0_proj(small_q8_0(ffn, hidden, seed + 4), ffn, hidden),
            up: q8_0_proj(small_q8_0(ffn, hidden, seed + 5), ffn, hidden),
            down: q8_0_proj(small_q8_0(hidden, ffn, seed + 6), hidden, ffn),
            bias: None,
        }),
        glu: GluCuda::Silu,
        post_attn_norm: windowed.then(|| norm(2.1)),
        post_ffn_norm: None,
        extras: AttnExtrasCuda {
            q_norm: windowed.then(|| head_norm(2.3)),
            k_norm: windowed.then(|| head_norm(2.5)),
            ..AttnExtrasCuda::default()
        },
        rope: Some(LayerRopeCuda {
            theta: if windowed { 1_000_000.0 } else { 500_000.0 },
            freq_factors: (!windowed).then_some(ff),
            rot_dim: head_dim,
            layout: if windowed {
                RopeLayoutCuda::Neox
            } else {
                RopeLayoutCuda::Norm
            },
            mscale: 1.0,
        }),
    };
    let (full, windowed) = (layer(60, false), layer(70, true));
    let params = |window| PrefillParams {
        n_heads,
        n_kv_heads,
        head_dim,
        rms_eps: 1e-5,
        attn_scale: (head_dim as f32).sqrt().recip(),
        attn_softcap: None,
        window,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    };
    let table = MulMmWeights {
        kind: &Q8_0,
        data: small_q8_0(vocab, hidden, 80),
        rows: vocab,
        cols: hidden,
        row_bytes: hidden / 32 * 34,
    };
    replay_matches_eager(
        &[&full, &windowed],
        &[params(None), params(Some(100))],
        &Feed::Gather { table, vocab },
        PREFIX,
        TOKENS,
    );
}
