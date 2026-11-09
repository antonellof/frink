//! The two attention kernels the stack runs, each against the host
//! definition: the tiled prefill kernel over a batch of query rows and
//! the split-K decode kernel over one, with windows, softcaps and
//! gpt-oss's attention sinks. The device tests are `#[ignore]`d.

use super::enqueue::{enqueue_causal_gqa_prefill, enqueue_decode_attn, AttnArgs};
use super::tests::{assert_close, host_causal_gqa_sinks, wave};
use crate::gpu::shared_device;
use crate::xfer;

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
            let out = enqueue_causal_gqa_prefill(&dev, &d_q, &d_k, &d_v, n_q, &args, d_s.as_ref())
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
