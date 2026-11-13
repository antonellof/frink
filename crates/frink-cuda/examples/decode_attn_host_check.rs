//! Emits everything `tools/decode_attn_host_check/run.sh` needs to run
//! the *real* one-row attention CUDA C (`frink_cuda::decode_attn`) on a
//! host CPU and compare it bit for bit against
//! `frink_cuda::decode_attn::decode_attn_twin`: the kernel text, a
//! fixture per case, the twin's output, and a manifest.
//!
//! Not a substitute for a GPU: it cannot see NVRTC reject the text or
//! a race a real scheduler would expose. It does check that the
//! translation unit compiles, that both kernels index the scratch and
//! the K/V rows the way the twin does, at every chunk boundary the
//! cases cross, and that the butterfly and the two merges are the ones
//! the twin models. Each case runs twice: as a token enqueued as it runs
//! launches it (the position by value, exactly its chunks), and as a
//! replayed CUDA graph does (the position from a device word, the grid
//! [`replay_slots`] wide for the history's capacity), so the two forms
//! are held to the same twin bit for bit.
//!
//! Run it through `tools/decode_attn_host_check/run.sh`, not directly.

use frink_cuda::decode_attn::{
    decode_attn_twin, key_range, n_chunks, replay_slots, DECODE_ATTN_KERNEL_SRC,
};

/// `(n_heads, n_kv_heads, head_dim, seq_len, window, softcap, sinks)`:
/// one chunk, exactly one chunk, a ragged second chunk, many chunks, a
/// window that starts mid-chunk, a softcap, both lane-slice widths, a
/// head width that leaves lanes idle, and attention sinks, one below
/// every score and one above (gpt-oss's window of 128 among them).
type Case = (
    usize,
    usize,
    usize,
    usize,
    Option<usize>,
    Option<f32>,
    Option<f32>,
);
const CASES: &[Case] = &[
    (4, 2, 64, 1, None, None, None),
    (4, 2, 64, 64, None, None, None),
    (4, 2, 64, 65, None, None, None),
    (3, 1, 128, 300, None, None, None),
    (3, 1, 128, 300, Some(100), None, None),
    (2, 2, 256, 130, None, Some(30.0), None),
    (2, 1, 12, 70, None, None, None),
    (8, 1, 64, 300, Some(128), None, Some(-3.0)),
    (4, 2, 64, 65, None, None, Some(4.0)),
];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: decode_attn_host_check <output-dir>");
    std::fs::write(format!("{dir}/decode_attn.cu"), DECODE_ATTN_KERNEL_SRC).unwrap();
    let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    let wave = |n: usize, seed: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * 0.173 + seed).sin()).collect()
    };
    let mut manifest = String::new();
    for (i, &(n_heads, n_kv, hd, seq_len, window, softcap, sink)) in CASES.iter().enumerate() {
        let q = wave(n_heads * hd, 0.3 + i as f32);
        let k = wave(seq_len * n_kv * hd, 0.7);
        let v = wave(seq_len * n_kv * hd, 1.1);
        let scale = (hd as f32).sqrt().recip();
        // One logit per head around the case's level, so the heads'
        // sinks differ.
        let sinks: Option<Vec<f32>> =
            sink.map(|s| (0..n_heads).map(|h| s + 0.25 * h as f32).collect());
        let want = decode_attn_twin(
            &q,
            &k,
            &v,
            n_heads,
            n_kv,
            hd,
            seq_len,
            window,
            scale,
            softcap,
            sinks.as_deref(),
        );
        let (lo, hi) = key_range(seq_len, window);
        let chunks = n_chunks(hi - lo);
        // The capacity a KV mirror holding `seq_len` rows has: a power
        // of two from 256, as `kv_mirror::grown_capacity` grows it.
        let cap_rows = seq_len.next_power_of_two().max(256);
        let slots = replay_slots(cap_rows, window);
        assert!(
            slots >= chunks,
            "case {i}: {slots} slots for {chunks} chunks"
        );
        let tag = format!("case{i}");
        std::fs::write(format!("{dir}/{tag}.q"), bytes(&q)).unwrap();
        std::fs::write(format!("{dir}/{tag}.k"), bytes(&k)).unwrap();
        std::fs::write(format!("{dir}/{tag}.v"), bytes(&v)).unwrap();
        std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
        // Empty when the case has no sinks: the C side passes null.
        std::fs::write(
            format!("{dir}/{tag}.sinks"),
            bytes(sinks.as_deref().unwrap_or(&[])),
        )
        .unwrap();
        // Floats as their bits, so the C side reads exactly the value
        // the twin used.
        let cap = softcap.filter(|c| *c > 0.0).unwrap_or(0.0);
        manifest.push_str(&format!(
            "{tag} {n_heads} {n_kv} {hd} {hi} {} {chunks} {slots} {} {}\n",
            window.unwrap_or(0),
            scale.to_bits(),
            cap.to_bits()
        ));
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
