//! Emits everything `tools/matvec_host_check/run.sh` needs to execute
//! the *real* coalesced matvec CUDA C on a host CPU, at both widths, and
//! compare it bit for bit against the scalar twins in
//! `frink_cuda::coalesced_twin`: one `.cu` per (kind, width), a fixture
//! per shape, the twin's expected output, and a manifest.
//!
//! Not a substitute for a GPU. It cannot see NVRTC reject the text, a
//! misaligned vector load fault, or a race a real warp scheduler would
//! expose. It does check that the translation unit compiles as C++,
//! that its index arithmetic addresses what the twin addresses at every
//! width, and that the shuffle reduction is the one the twin models,
//! against the kernel text itself rather than a Rust paraphrase of it.
//!
//! Run it through `tools/matvec_host_check/run.sh`, not directly.

use frink_cuda::coalesced_twin::twin_for;
use frink_cuda::matvec_kinds::coalesced::{MatvecWidth, COALESCED, WARPS_PER_BLOCK};
use frink_cuda::matvec_kinds::{generated_kind, generic, KINDS};

/// Rows not a multiple of the warps per block, so the tail warp of the
/// last block returns early. Block counts below, at and above a wide
/// warp's blocks per trip, and a 3072-wide K-quant row.
const ROWS: usize = 37;
const BLOCKS: &[usize] = &[1, 3, 4, 5, 12];

/// Offsets of each block's f16 header fields. Their exponent's top bit
/// is cleared so every scale is finite: a fixture that is mostly NaN
/// agrees with anything.
fn f16_fields(name: &str) -> &'static [usize] {
    match name {
        "Q4_K" | "Q5_K" => &[0, 2],
        "Q6_K" => &[208],
        "Q8_0" => &[0],
        "Q4_1" | "Q5_1" => &[0, 2],
        "TQ1_0" => &[52],
        "PTQ1_0" => &[26],
        "IQ2_XXS" | "IQ2_XS" | "IQ2_S" | "IQ3_XXS" | "IQ3_S" | "IQ1_S" => &[0],
        // IQ1_M's scale is assembled from scale-word nibbles; byte 55
        // holds its sign and top exponent bit, so clearing byte 55's two
        // top bits (`at + 1` below) keeps it finite.
        "IQ1_M" => &[54],
        other => panic!("{other}: no header layout for the host check"),
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: matvec_host_check <output-dir>");
    let mut manifest = String::new();
    for ck in COALESCED {
        let mv = KINDS
            .iter()
            .find(|k| k.fn_name == ck.base_fn)
            .expect("every coalesced kind routes from a MatvecKind");
        let mm = frink_cuda::mul_mm::kind_by_name(mv.name).expect("block geometry");
        let twin = twin_for(ck.entry).expect("every coalesced kernel has a twin");
        for width in [MatvecWidth::Narrow, MatvecWidth::Wide] {
            let lanes = ck.lanes(width);
            let unit = format!("{}_{lanes}", mv.name);
            std::fs::write(format!("{dir}/{unit}.cu"), ck.source(width)).unwrap();
            for &n_blocks in BLOCKS {
                let row_bytes = n_blocks * mm.block_bytes;
                let cols = n_blocks * mm.block_elems;
                let mut state = 12345u32 ^ (n_blocks as u32).wrapping_mul(2_654_435_761);
                let mut weights: Vec<u8> = (0..ROWS * row_bytes)
                    .map(|_| {
                        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                        (state >> 16) as u8
                    })
                    .collect();
                for blk in 0..ROWS * n_blocks {
                    for at in f16_fields(mv.name) {
                        weights[blk * mm.block_bytes + at + 1] &= 0x3F;
                    }
                }
                let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();
                let want: Vec<f32> = (0..ROWS)
                    .map(|r| {
                        twin(
                            &weights[r * row_bytes..(r + 1) * row_bytes],
                            &x,
                            n_blocks,
                            lanes,
                        )
                    })
                    .collect();

                let tag = format!("{unit}_{n_blocks}");
                let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
                std::fs::write(format!("{dir}/{tag}.w"), &weights).unwrap();
                std::fs::write(format!("{dir}/{tag}.x"), bytes(&x)).unwrap();
                std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
                manifest.push_str(&format!(
                    "{unit} {} {tag} {ROWS} {row_bytes} {n_blocks} {WARPS_PER_BLOCK}\n",
                    ck.entry
                ));
            }
        }
    }
    // The matvecs generated from a GEMM dequant (`matvec_kinds::generic`):
    // block-per-row, so the manifest's warp count is 0 and `main.cpp`
    // launches one 256-thread block per row. Scales finite, as above.
    for mv in KINDS {
        let Some(mm) = generated_kind(mv) else {
            continue;
        };
        let unit = format!("{}_generic", mv.name);
        std::fs::write(format!("{dir}/{unit}.cu"), mv.source()).unwrap();
        // One block, a row shorter than the threads, and one long enough
        // that every thread wraps (over 256 sub-blocks).
        for n_blocks in [
            1usize,
            5,
            (generic::THREADS * 16 * 2).div_ceil(mm.block_elems) + 3,
        ] {
            let row_bytes = n_blocks * mm.block_bytes;
            let cols = n_blocks * mm.block_elems;
            let mut state = 999u32 ^ (n_blocks as u32).wrapping_mul(2_654_435_761);
            let mut weights: Vec<u8> = (0..ROWS * row_bytes)
                .map(|_| {
                    state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (state >> 16) as u8
                })
                .collect();
            for blk in 0..ROWS * n_blocks {
                for at in f16_fields(mv.name) {
                    weights[blk * mm.block_bytes + at + 1] &= 0x3F;
                }
            }
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();
            let want: Vec<f32> = (0..ROWS)
                .map(|r| {
                    generic::matvec_twin(
                        mm,
                        &weights[r * row_bytes..(r + 1) * row_bytes],
                        &x,
                        n_blocks,
                    )
                })
                .collect();
            let tag = format!("{unit}_{n_blocks}");
            let bytes = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(format!("{dir}/{tag}.w"), &weights).unwrap();
            std::fs::write(format!("{dir}/{tag}.x"), bytes(&x)).unwrap();
            std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
            manifest.push_str(&format!(
                "{unit} {} {tag} {ROWS} {row_bytes} {n_blocks} 0\n",
                mv.fn_name
            ));
        }
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
