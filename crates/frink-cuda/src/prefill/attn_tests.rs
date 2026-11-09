//! The attention kernels the stack runs, each against the host
//! definition: the two prefill kernels over a batch of query rows (the
//! SIMT one and the tensor-core one) and the split-K decode kernel over
//! one, with windows, softcaps and gpt-oss's attention sinks. The
//! device tests are `#[ignore]`d.

use super::enqueue::{
    enqueue_causal_gqa_prefill, enqueue_causal_gqa_prefill_simt, enqueue_decode_attn, AttnArgs,
};
use super::tests::{assert_close, host_causal_gqa_sinks, wave};
use crate::gpu::{compute_capability_major, shared_device};
use crate::prefill_attn_tc::launch::{
    enqueue_prefill_attn_tc_with_group, forced_simt, PrefillAttnBody,
};
use crate::prefill_attn_tc::{f16_error_bound, GROUPS};
use crate::xfer;

/// The tensor-core kernel against the definition, at every group size
/// and through the stack's one door, within the error its f16 operands
/// can put on an output (`prefill_attn_tc::f16_error_bound`, derived
/// from the rounding of Q, K, V and P and held to the definition on the
/// host by `the_twin_is_attention_within_the_f16_bound`). The kernel
/// text itself matches its twin bit for bit on a CPU
/// (`tools/prefill_attn_tc_host_check/run.sh`). This test is what that
/// check cannot see: NVRTC, the real `mma.sync` and `ldmatrix`, the
/// shared memory opt-in and the scheduler.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up); not yet run"]
fn tensor_core_prefill_attention_matches_the_host_within_its_f16_bound() {
    let dev = shared_device().unwrap();
    assert!(
        compute_capability_major(&dev) >= crate::mul_mm_tc::MIN_CC_MAJOR,
        "the tensor-core prefill attention needs sm_80 or newer"
    );
    let up = |x: &[f32]| xfer::htod(&dev, "attn tc test", x).unwrap();
    for (n_q, n_heads, n_kv_heads, head_dim, start_pos, window, softcap) in [
        // GQA 1, one block, a batch that is no multiple of 16.
        (13, 4, 4, 64, 0, None, None),
        // GQA 2 after a prefix.
        (77, 8, 4, 64, 9, None, None),
        // GQA 4, Llama-3's ratio, over several key tiles.
        (100, 8, 2, 128, 0, None, None),
        // GQA 8, and a window that binds after a prefix.
        (70, 8, 1, 128, 33, Some(23), None),
        // GQA 7, whose rows pad at G = 8, and a softcap that binds.
        (45, 7, 1, 128, 20, None, Some(1.5)),
        // 256 wide, which opts in to 66 KiB: a window and a softcap
        // together (a warp skips tiles at G = 1), and neither.
        (150, 4, 2, 256, 9, Some(57), Some(2.0)),
        (33, 2, 1, 256, 0, None, None),
        // A long prefix: many key tiles per block.
        (200, 2, 1, 128, 300, None, None),
    ] {
        let total = start_pos + n_q;
        let q = wave(n_q * n_heads * head_dim, 0.2);
        let k = wave(total * n_kv_heads * head_dim, 0.4);
        let v = wave(total * n_kv_heads * head_dim, 0.6);
        let args = AttnArgs {
            n_heads,
            n_kv_heads,
            head_dim,
            start_pos,
            window,
            scale: (head_dim as f32).sqrt().recip(),
            softcap,
        };
        let want_body = if forced_simt() {
            PrefillAttnBody::Simt
        } else {
            PrefillAttnBody::TensorCore
        };
        assert_eq!(PrefillAttnBody::select(&dev, &args), want_body);
        let bound = f16_error_bound(&q, &k, &v, n_q, &args);
        let (d_q, d_k, d_v) = (up(&q), up(&k), up(&v));
        for sink in [None, Some(-1.0f32), Some(3.0)] {
            let sinks: Option<Vec<f32>> =
                sink.map(|s| (0..n_heads).map(|h| s + 0.5 * h as f32).collect());
            let want = host_causal_gqa_sinks(&q, &k, &v, n_q, &args, sinks.as_deref());
            let d_s = sinks.as_deref().map(up);
            let what = format!(
                "attn tc n_q={n_q} heads={n_heads}/{n_kv_heads} hd={head_dim} start={start_pos} \
                 window={window:?} softcap={softcap:?} sink={sink:?}"
            );
            let check = |got: Vec<f32>, how: &str| {
                assert_eq!(got.len(), want.len(), "{what} {how}: length");
                let err = got
                    .iter()
                    .zip(&want)
                    .map(|(g, w)| (g - w).abs())
                    .fold(0f32, f32::max);
                eprintln!("{what} {how}: worst error {err:.3e} of a bound {bound:.3e}");
                assert!(
                    err <= bound,
                    "{what} {how}: error {err} over the f16 bound {bound}"
                );
            };
            for group in GROUPS {
                let out = enqueue_prefill_attn_tc_with_group(
                    &dev,
                    &d_q,
                    &d_k,
                    &d_v,
                    n_q,
                    &args,
                    d_s.as_ref(),
                    group,
                )
                .unwrap();
                check(
                    xfer::dtoh(&dev, "attn tc out", &out).unwrap(),
                    &format!("G={group}"),
                );
            }
            let out = enqueue_causal_gqa_prefill(&dev, &d_q, &d_k, &d_v, n_q, &args, d_s.as_ref())
                .unwrap();
            check(
                xfer::dtoh(&dev, "attn tc out", &out).unwrap(),
                "through the stack's door",
            );
        }
    }
}

#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn causal_gqa_prefill_matches_the_host_with_a_prefix_a_window_a_softcap_and_sinks() {
    let dev = shared_device().unwrap();
    let up = |x: &[f32]| xfer::htod(&dev, "attn test", x).unwrap();
    // One block; several blocks of rows whose keys span tiles; a head
    // of 256 (two float4s a lane) under GQA 2:1; and a window that
    // starts inside a tile and leaves whole tiles out of later rows.
    for (n_q, n_heads, n_kv_heads, head_dim, start_pos, window, softcap) in [
        (13, 4, 2, 64, 0, None, None),
        (13, 4, 2, 64, 7, None, None),
        (13, 4, 2, 64, 7, Some(5), None),
        (13, 4, 2, 64, 0, None, Some(50.0)),
        (77, 8, 2, 128, 0, None, None),
        (77, 8, 2, 128, 45, Some(40), None),
        (40, 2, 1, 256, 3, None, Some(30.0)),
        (150, 4, 4, 96, 0, Some(33), None),
    ] {
        // No sinks, then gpt-oss's: one logit per head, below some rows'
        // scores and above others'.
        for sink in [None, Some(-1.0f32), Some(3.0)] {
            let total = start_pos + n_q;
            let q = wave(n_q * n_heads * head_dim, 0.2);
            let k = wave(total * n_kv_heads * head_dim, 0.4);
            let v = wave(total * n_kv_heads * head_dim, 0.6);
            let args = AttnArgs {
                n_heads,
                n_kv_heads,
                head_dim,
                start_pos,
                window,
                scale: (head_dim as f32).sqrt().recip(),
                softcap,
            };
            let sinks: Option<Vec<f32>> =
                sink.map(|s| (0..n_heads).map(|h| s + 0.5 * h as f32).collect());
            let want = host_causal_gqa_sinks(&q, &k, &v, n_q, &args, sinks.as_deref());
            let d_q = up(&q);
            let d_k = up(&k);
            let d_v = up(&v);
            let d_s = sinks.as_deref().map(up);
            // The SIMT body by name: the stack's door takes the tensor
            // cores on sm_80 and up, held to an f16 bound above.
            let out =
                enqueue_causal_gqa_prefill_simt(&dev, &d_q, &d_k, &d_v, n_q, &args, d_s.as_ref())
                    .unwrap();
            let got = xfer::dtoh(&dev, "attn test out", &out).unwrap();
            assert_close(
                &got,
                &want,
                1e-4,
                &format!(
                    "attn n_q={n_q} hd={head_dim} start={start_pos} window={window:?} \
                     softcap={softcap:?} sink={sink:?}"
                ),
            );
        }
    }
}

#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn decode_attn_matches_its_twin_across_chunks_a_window_a_softcap_and_sinks() {
    let dev = shared_device().unwrap();
    let up = |x: &[f32]| xfer::htod(&dev, "attn test", x).unwrap();
    for (n_heads, n_kv_heads, head_dim) in [(24usize, 8usize, 128usize), (4, 2, 64), (2, 1, 256)] {
        for (seq_len, window, softcap) in [
            (1usize, None, None),
            (63, None, None),
            (64, None, None),
            (65, None, None),
            (777, None, None),
            (777, Some(300), None),
            (130, None, Some(50.0)),
        ] {
            for sink in [None, Some(-1.0f32), Some(3.0)] {
                let q = wave(n_heads * head_dim, 0.2);
                let k = wave(seq_len * n_kv_heads * head_dim, 0.4);
                let v = wave(seq_len * n_kv_heads * head_dim, 0.6);
                let scale = (head_dim as f32).sqrt().recip();
                let sinks: Option<Vec<f32>> =
                    sink.map(|s| (0..n_heads).map(|h| s + 0.5 * h as f32).collect());
                let want = crate::decode_attn::decode_attn_twin(
                    &q,
                    &k,
                    &v,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    window,
                    scale,
                    softcap,
                    sinks.as_deref(),
                );
                let args = AttnArgs {
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    start_pos: seq_len - 1,
                    window,
                    scale,
                    softcap,
                };
                let d_q = up(&q);
                let d_k = up(&k);
                let d_v = up(&v);
                let d_s = sinks.as_deref().map(up);
                let out = enqueue_decode_attn(&dev, &d_q, &d_k, &d_v, &args, d_s.as_ref()).unwrap();
                let got = xfer::dtoh(&dev, "attn test out", &out).unwrap();
                let what = format!(
                "decode attn h={n_heads} hd={head_dim} seq={seq_len} w={window:?} sink={sink:?}"
            );
                assert_close(&got, &want, 1e-4, &what);
                // And against the definition, through the prefill kernel's
                // host loop at one query row.
                let naive = host_causal_gqa_sinks(&q, &k, &v, 1, &args, sinks.as_deref());
                assert_close(&got, &naive, 1e-4, &format!("{what} vs naive"));
            }
        }
    }
}
