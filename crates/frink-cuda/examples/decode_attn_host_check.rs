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
//! the twin models.
//!
//! Run it through `tools/decode_attn_host_check/run.sh`, not directly.

use frink_cuda::decode_attn::{decode_attn_twin, key_range, n_chunks, DECODE_ATTN_KERNEL_SRC};

/// `(n_heads, n_kv_heads, head_dim, seq_len, window, softcap)`: one
/// chunk, exactly one chunk, a ragged second chunk, many chunks, a
/// window that starts mid-chunk, a softcap, both lane-slice widths, and
/// a head width that leaves lanes idle.
type Case = (usize, usize, usize, usize, Option<usize>, Option<f32>);
const CASES: &[Case] = &[
    (4, 2, 64, 1, None, None),
    (4, 2, 64, 64, None, None),
    (4, 2, 64, 65, None, None),
    (3, 1, 128, 300, None, None),
    (3, 1, 128, 300, Some(100), None),
    (2, 2, 256, 130, None, Some(30.0)),
    (2, 1, 12, 70, None, None),
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
    for (i, &(n_heads, n_kv, hd, seq_len, window, softcap)) in CASES.iter().enumerate() {
        let q = wave(n_heads * hd, 0.3 + i as f32);
        let k = wave(seq_len * n_kv * hd, 0.7);
        let v = wave(seq_len * n_kv * hd, 1.1);
        let scale = (hd as f32).sqrt().recip();
        let want = decode_attn_twin(
            &q, &k, &v, n_heads, n_kv, hd, seq_len, window, scale, softcap,
        );
        let (lo, hi) = key_range(seq_len, window);
        let chunks = n_chunks(hi - lo);
        let tag = format!("case{i}");
        std::fs::write(format!("{dir}/{tag}.q"), bytes(&q)).unwrap();
        std::fs::write(format!("{dir}/{tag}.k"), bytes(&k)).unwrap();
        std::fs::write(format!("{dir}/{tag}.v"), bytes(&v)).unwrap();
        std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
        // Floats as their bits, so the C side reads exactly the value
        // the twin used.
        let cap = softcap.filter(|c| *c > 0.0).unwrap_or(0.0);
        manifest.push_str(&format!(
            "{tag} {n_heads} {n_kv} {hd} {lo} {hi} {chunks} {} {}\n",
            scale.to_bits(),
            cap.to_bits()
        ));
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
