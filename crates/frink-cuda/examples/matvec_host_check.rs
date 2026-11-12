//! Emits everything `tools/matvec_host_check/run.sh` needs to execute
//! the *real* coalesced matvec CUDA C on a host CPU, at both widths, and
//! compare it bit for bit against the scalar twins in
//! `frink_cuda::coalesced_twin`: one `.cu` per (kind, width), a fixture
//! per shape, the twin's expected output, and a manifest. The q8_1
//! quantizer and the dp4a matvecs get the same against
//! `frink_cuda::dp4a_twin`, in a manifest of their own
//! (`dp4a_manifest.txt`, run by `dp4a_main.cpp`).
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
        // An E8M0 scale byte, not an f16 field: kept finite in `main`.
        "MXFP4" => &[],
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

/// `rows` rows of `n_blocks` pseudo-random blocks of kind `name`, every
/// f16 header field finite and MXFP4's E8M0 byte near 2^0, both sides
/// of the bias.
fn fixture_weights(name: &str, block_bytes: usize, blocks: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut weights: Vec<u8> = (0..blocks * block_bytes)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (state >> 16) as u8
        })
        .collect();
    for blk in 0..blocks {
        for at in f16_fields(name) {
            weights[blk * block_bytes + at + 1] &= 0x3F;
        }
        if name == "MXFP4" {
            let e = &mut weights[blk * block_bytes];
            *e = 120 + *e % 16;
        }
    }
    weights
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The q8_1 quantizer and every dp4a matvec, at every geometry the
/// launch can pick and in the routed-expert form, into
/// `dp4a_manifest.txt` (see `run.sh`). The dot fixtures' q8_1 inputs are
/// the twin quantizer's bytes, which the quantizer's own lines hold to
/// the kernel's.
fn emit_dp4a(dir: &str) {
    use frink_cuda::dp4a_twin::{dp4a_matvec_row, quantize_q8_1};
    use frink_cuda::matvec_kinds::dp4a::{
        quantize_source, Dp4aGeometry, DP4A, Q8_1_BLOCK_BYTES, QUANTIZE_ENTRY, QUANTIZE_THREADS,
    };
    let mut manifest = String::new();

    let unit = "q8_1_quantize";
    std::fs::write(format!("{dir}/{unit}.cu"), quantize_source()).unwrap();
    // A zero block (d = 0, no division), exact halves at amax 127, a
    // block whose f16 d underflows to zero, one whose f16 sum overflows,
    // and a long vector of mixed scales over several threadblocks.
    let mut edge = vec![0f32; 32];
    edge.extend((0..32).map(|i| match i {
        0 => 127.0,
        1 => -127.0,
        2 => 2.5,
        3 => -2.5,
        4 => 0.5,
        _ => i as f32 - 16.0,
    }));
    edge.extend((0..32).map(|i| ((i as f32) * 0.7).sin() * 1e-6));
    edge.extend((0..32).map(|i| 3000.0 + i as f32));
    let long: Vec<f32> = (0..4096 + 96)
        .map(|i| ((i as f32) * 0.019).cos() * (1.0 + ((i / 32) % 9) as f32).powi(3))
        .collect();
    for (tag, x) in [("edge", edge), ("long", long)] {
        let tag = format!("{unit}_{tag}");
        std::fs::write(format!("{dir}/{tag}.x"), f32_bytes(&x)).unwrap();
        std::fs::write(format!("{dir}/{tag}.want"), quantize_q8_1(&x)).unwrap();
        manifest.push_str(&format!(
            "quant {unit} {QUANTIZE_ENTRY} {tag} {} {QUANTIZE_THREADS}\n",
            x.len()
        ));
    }

    let geometries = [
        (1, 1, false),
        (2, 1, false),
        (2, 2, false),
        (4, 1, false),
        (4, 4, false),
        (4, 1, true),
        (4, 4, true),
    ];
    for kind in DP4A {
        let mv = KINDS
            .iter()
            .find(|k| k.fn_name == kind.base_fn)
            .expect("every dp4a kind routes from a MatvecKind");
        // Rows of one block, of fewer blocks than one trip of four warps,
        // and of several trips with a ragged tail.
        let block_counts: &[usize] = if kind.qk == 256 {
            &[1, 3, 6, 9, 17]
        } else {
            &[1, 5, 31, 70, 130]
        };
        for (nwarps, rows_per_block, multi) in geometries {
            let g = Dp4aGeometry {
                nwarps,
                rows: rows_per_block,
            };
            let m = if multi { "_multi" } else { "" };
            let unit = format!("{}_dp4a_w{nwarps}_r{rows_per_block}{m}", mv.name);
            std::fs::write(format!("{dir}/{unit}.cu"), kind.source(g, multi)).unwrap();
            for &n_blocks in block_counts {
                let row_bytes = n_blocks * kind.block_bytes;
                let n_q8 = n_blocks * kind.q8_per_block();
                let shapes: &[(usize, usize)] = if multi {
                    &[(2, 0), (2, n_q8)]
                } else {
                    &[(1, 0)]
                };
                for &(slots, y_stride) in shapes {
                    let seed = 777 ^ (n_blocks as u32).wrapping_mul(2_654_435_761);
                    let weights =
                        fixture_weights(mv.name, kind.block_bytes, slots * ROWS * n_blocks, seed);
                    let inputs = if y_stride == 0 { 1 } else { slots };
                    let x: Vec<f32> = (0..inputs * n_q8 * 32)
                        .map(|i| ((i as f32) * 0.019).cos() * (1.0 + ((i / 32) % 5) as f32))
                        .collect();
                    let y = quantize_q8_1(&x);
                    let mut want = Vec::with_capacity(slots * ROWS);
                    for s in 0..slots {
                        let ys = &y[s * y_stride * Q8_1_BLOCK_BYTES..];
                        for r in 0..ROWS {
                            let at = (s * ROWS + r) * row_bytes;
                            want.push(dp4a_matvec_row(
                                kind,
                                &weights[at..at + row_bytes],
                                ys,
                                n_blocks,
                                nwarps,
                            ));
                        }
                    }
                    let tag = format!("{unit}_{n_blocks}_{slots}_{y_stride}");
                    std::fs::write(format!("{dir}/{tag}.w"), &weights).unwrap();
                    std::fs::write(format!("{dir}/{tag}.y"), &y).unwrap();
                    std::fs::write(format!("{dir}/{tag}.want"), f32_bytes(&want)).unwrap();
                    manifest.push_str(&format!(
                        "dot {unit} {} {tag} {ROWS} {row_bytes} {n_blocks} {} {rows_per_block} {slots} {y_stride}\n",
                        kind.entry,
                        g.threads()
                    ));
                }
            }
        }
    }
    std::fs::write(format!("{dir}/dp4a_manifest.txt"), manifest).unwrap();
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
            std::fs::write(format!("{dir}/{unit}.cu"), ck.source(width, 1)).unwrap();
            for &n_blocks in BLOCKS {
                let row_bytes = n_blocks * mm.block_bytes;
                let cols = n_blocks * mm.block_elems;
                let seed = 12345u32 ^ (n_blocks as u32).wrapping_mul(2_654_435_761);
                let weights = fixture_weights(mv.name, mm.block_bytes, ROWS * n_blocks, seed);
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
            let seed = 999u32 ^ (n_blocks as u32).wrapping_mul(2_654_435_761);
            let weights = fixture_weights(mv.name, mm.block_bytes, ROWS * n_blocks, seed);
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
    emit_dp4a(&dir);
}
