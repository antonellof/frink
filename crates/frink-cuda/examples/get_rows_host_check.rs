//! Emits everything `tools/get_rows_host_check/run.sh` needs to execute
//! the *real* get_rows CUDA C on a host CPU and compare it bit for bit
//! against the scalar twin (`frink_cuda::get_rows::get_rows_reference`):
//! one `.cu` per kind, a fixture per shape, the twin's output, and a
//! manifest.
//!
//! Not a substitute for a GPU: it cannot see NVRTC reject the text or a
//! launch configuration the device refuses. It does check that the
//! unit compiles as C++, and that its index arithmetic and its dequant
//! address and decode what the twin does, against the kernel text.
//!
//! Run it through `tools/get_rows_host_check/run.sh`, not directly.

use frink_cuda::get_rows::{get_rows_reference, kernel_src, names, THREADS};
use frink_cuda::mul_mm::KINDS;

/// A table of 13 rows, gathered in an order that repeats one id and
/// ends on the last row. Widths of one block, of a partial grid block
/// of threads, and of more than one.
const N_ROWS: usize = 13;
const IDS: &[u32] = &[12, 0, 5, 5, 11, 1];
const WIDTHS: &[usize] = &[256, 1280, 2304];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: get_rows_host_check <output-dir>");
    let mut manifest = String::new();
    for k in KINDS {
        std::fs::write(format!("{dir}/{}.cu", k.name), kernel_src(k)).unwrap();
        for &n_cols in WIDTHS {
            let row_bytes = (n_cols / k.block_elems) * k.block_bytes;
            // Arbitrary bytes, so degenerate f16 scales (NaN/Inf) are
            // exercised too; the two sides must agree on those.
            let mut state = 777u32 ^ (n_cols as u32);
            let table: Vec<u8> = (0..N_ROWS * row_bytes)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (state >> 16) as u8
                })
                .collect();
            let want = get_rows_reference(k, &table, N_ROWS, IDS, n_cols, row_bytes).unwrap();
            let tag = format!("{}_{n_cols}", k.name);
            let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(format!("{dir}/{tag}.w"), &table).unwrap();
            std::fs::write(
                format!("{dir}/{tag}.ids"),
                IDS.iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
            manifest.push_str(&format!(
                "{} {} {tag} {} {n_cols} {row_bytes} {THREADS}\n",
                k.name,
                names(k).1,
                IDS.len()
            ));
        }
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
