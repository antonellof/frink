//! Times the decode matvec on a real CUDA device, on the projection
//! shapes of the dense models the benchmark suite serves, and prints
//! the weight bandwidth each reaches. Run on a GPU host:
//!
//!   cargo run --release -p frink-cuda --features cuda --example matvec_bench
//!   FRINK_CUDA_MATVEC_WPR=4 cargo run --release ... --example matvec_bench
//!
//! The geometry is the environment's (`FRINK_CUDA_MATVEC`,
//! `FRINK_CUDA_MATVEC_WPR`), read once per process, so a sweep runs
//! the example once per setting. Under `FRINK_CUDA_MATVEC=dp4a` each
//! timed launch quantizes its activation too, which a lone projection
//! pays and a shared one (Q/K/V, gate/up) pays once. The weights are pseudo-random bytes:
//! the timing does not depend on their values, and correctness is the
//! hardware tests' (`gpu.rs`), not this.

use frink_cuda::gpu::{matvec_launch_meta, time_matvec, MatvecLaunch};
use frink_cuda::mul_mm::kind_by_name;

/// `(rows, cols)`: Llama-3.1-8B's Q/O, K/V, gate/up, down and LM head.
const SHAPES: &[(usize, usize)] = &[
    (4096, 4096),
    (1024, 4096),
    (14336, 4096),
    (4096, 14336),
    (128256, 4096),
];

fn main() {
    println!(
        "{:6} {:>7} {:>6} {:>9} {:>8}",
        "kind", "rows", "cols", "us", "GB/s"
    );
    for name in ["Q4_K", "Q5_K", "Q6_K", "Q8_0"] {
        let geometry = kind_by_name(name).expect("a GEMM kind with the block geometry");
        let (kernel_src, module_name, fn_name) = matvec_launch_meta(name).expect("a matvec");
        for &(rows, cols) in SHAPES {
            let n_blocks_per_row = cols / geometry.block_elems;
            let row_bytes = n_blocks_per_row * geometry.block_bytes;
            let mut state = 0x2545_f491u32;
            let weights: Vec<u8> = (0..rows * row_bytes)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (state >> 16) as u8
                })
                .collect();
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
            let launch = MatvecLaunch {
                kernel_src,
                module_name,
                fn_name,
                weights: &weights,
                rows,
                row_bytes,
                n_blocks_per_row,
            };
            let s = time_matvec(&launch, &x, 50).expect("the matvec runs on this device");
            println!(
                "{name:6} {rows:>7} {cols:>6} {:>9.1} {:>8.1}",
                s * 1e6,
                weights.len() as f64 / s / 1e9
            );
        }
    }
}
