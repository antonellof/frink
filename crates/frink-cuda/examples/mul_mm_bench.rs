//! Times the batched quantized GEMM on a real CUDA device, on the
//! projection shapes of the dense models the benchmark suite serves,
//! and prints TFLOPS per kind and shape. Run on a GPU host:
//!
//!   cargo run --release -p frink-cuda --features cuda --example mul_mm_bench
//!   cargo run --release -p frink-cuda --features cuda --example mul_mm_bench -- Q4_K 512
//!
//! The arguments are an optional kind name and an optional batch
//! (default 512, `pp512`'s). The weights are pseudo-random bytes: the
//! timing does not depend on their values, and the correctness of
//! every body is the hardware tests' (`mul_mm_launch`), not this.
//! `FRINK_CUDA_MUL_MM=simt` times the SIMT body instead.

use frink_cuda::mul_mm::{kind_by_name, KINDS};
use frink_cuda::mul_mm_launch::time_mul_mm;

/// `(rows, cols)`: Llama-3.1-8B's Q/O, K/V, gate/up and down, and
/// Llama-3.2-3B's Q/O and down.
const SHAPES: &[(usize, usize)] = &[
    (4096, 4096),
    (1024, 4096),
    (14336, 4096),
    (4096, 14336),
    (3072, 3072),
    (3072, 8192),
];

fn main() {
    let mut args = std::env::args().skip(1);
    let only = args.next();
    let batch: usize = args
        .next()
        .map(|b| b.parse().expect("batch"))
        .unwrap_or(512);
    let kinds: Vec<_> = match only.as_deref() {
        Some(name) => vec![kind_by_name(name).expect("a kind in mul_mm::KINDS")],
        None => ["Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "IQ4_XS"]
            .iter()
            .filter_map(|n| KINDS.iter().find(|k| k.name == *n))
            .collect(),
    };
    println!(
        "{:8} {:>6} {:>6} {:>6} {:>10} {:>8}",
        "kind", "rows", "cols", "batch", "ms", "TFLOPS"
    );
    for kind in kinds {
        for &(rows, cols) in SHAPES {
            if !cols.is_multiple_of(kind.block_elems) {
                continue;
            }
            let row_bytes = cols / kind.block_elems * kind.block_bytes;
            let mut state = 0x9e37_79b9u32;
            let weights: Vec<u8> = (0..rows * row_bytes)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect();
            let x: Vec<f32> = (0..batch * cols)
                .map(|i| ((i as f32) * 0.019).cos())
                .collect();
            let s = time_mul_mm(kind, &weights, &x, rows, cols, batch, row_bytes, 20)
                .expect("the GEMM runs on this device");
            let tflops = 2.0 * (rows * cols * batch) as f64 / s / 1e12;
            println!(
                "{:8} {rows:>6} {cols:>6} {batch:>6} {:>10.3} {tflops:>8.1}",
                kind.name,
                s * 1e3
            );
        }
    }
}
