//! The int8 GEMM and its quantizer on a device, against their twins.
//! Every test here is `#[ignore]`d: run them with
//! `cargo test -p frink-cuda --features cuda -- --ignored` on real
//! hardware and write down what happened in the `ignore` text.

use super::kernel::{self, MmqKind};
use super::quantize::{self, Layout};
use crate::gpu::shared_device;
use crate::xfer;

/// Each 32-value run at its own amplitude (1x to 16x), so a scale taken
/// from the wrong run is off by a whole factor.
fn wave(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * (1u32 << ((i / 32) % 5)) as f32)
        .collect()
}

/// The quantizer's blocks are its twin's, bit for bit: both layouts, a
/// gather, a K that is not a whole number of blocks, padding columns.
#[test]
#[ignore = "requires real CUDA hardware. Passed 2026-11-09 on an A10G (sm_86, Hugging Face Jobs)"]
fn the_quantizer_writes_its_twin_s_blocks() {
    let dev = shared_device().unwrap();
    for (k, rows, ncols, ncols_pad, gathered) in [
        (128usize, 3usize, 3usize, 64usize, false),
        (2880, 5, 4, 64, true),
        (4096, 70, 70, 128, false),
    ] {
        let x = wave(rows * k, 0.1);
        let gather: Vec<i32> = (0..ncols as i32)
            .map(|c| (c * 3 + 1) % rows as i32)
            .collect();
        for layout in [Layout::D4, Layout::Ds4] {
            let d_x = xfer::htod(&dev, "q x", &x).unwrap();
            let d_g = xfer::htod(&dev, "q gather", &gather).unwrap();
            let y = quantize::enqueue_quantize(
                &dev,
                layout,
                &d_x,
                gathered.then_some(&d_g),
                k,
                ncols,
                ncols_pad,
            )
            .unwrap();
            let got = xfer::dtoh(&dev, "q y", &y).unwrap();
            let want = quantize::quantize_twin(
                &x,
                gathered.then_some(&gather[..]),
                k,
                ncols,
                ncols_pad,
                layout,
            );
            assert_eq!(got.len(), want.len());
            if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
                panic!(
                    "{layout:?} k={k} ncols={ncols}: byte {i}: got {} want {}",
                    got[i], want[i]
                );
            }
        }
    }
}

/// The int8 GEMM is its twin's for every kind, within the f32 rounding
/// of one accumulation order (one thread owns each output and walks K in
/// order, so only the FMA contraction can differ), and close to the f32
/// reference within the q8_1 rounding: a row count and a K that are not
/// whole tiles, a batch past one tile and one under it.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up). Passed 2026-11-09 on an A100 (sm_80, Hugging Face Jobs)"]
fn the_int8_gemm_matches_its_twin_and_the_reference() {
    let dev = shared_device().unwrap();
    for (kind, mm, shapes) in [
        (
            MmqKind::Q8_0,
            &crate::mul_mm::Q8_0,
            &[(135usize, 96usize, 37usize), (256, 2880, 130), (33, 128, 1)][..],
        ),
        (
            MmqKind::Q4_K,
            &crate::mul_mm::Q4_K,
            &[(135, 256, 37), (256, 4096, 130), (33, 512, 1)][..],
        ),
        (
            MmqKind::Q6_K,
            &crate::mul_mm::Q6_K,
            &[(135, 256, 37), (256, 4096, 130), (33, 512, 1)][..],
        ),
        (
            MmqKind::MXFP4,
            &crate::mul_mm::MXFP4,
            &[(135, 96, 37), (256, 2880, 130), (33, 128, 1)][..],
        ),
    ] {
        for (&(n_rows, n_cols, batch), cols) in shapes
            .iter()
            .flat_map(|s| [(s, kernel::COLS), (s, kernel::WIDE_COLS)])
        {
            let (be, bb) = kind.block();
            let row_bytes = n_cols / be * bb;
            let w = crate::mul_mm_ref::fixtures::weights(mm, n_rows, n_cols, 91);
            let x = wave(batch * n_cols, 0.4);
            let d_x = xfer::htod(&dev, "mmq x", &x).unwrap();
            let (out, held) = super::enqueue_mmq_cols(
                &dev, kind, &w, &d_x, n_rows, n_cols, batch, row_bytes, cols,
            )
            .unwrap();
            let got = xfer::dtoh(&dev, "mmq out", &out).unwrap();
            drop(held);
            let ncols_pad = kernel::ncols_pad(batch);
            let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, kind.layout());
            let twin = match kind {
                MmqKind::Q8_0 => super::mmq_q8_0_twin(&w, &y, n_rows, n_cols, batch, ncols_pad),
                MmqKind::Q4_K => super::mmq_q4_k_twin(&w, &y, n_rows, n_cols, batch, ncols_pad),
                MmqKind::Q6_K => super::mmq_q6_k_twin(&w, &y, n_rows, n_cols, batch, ncols_pad),
                MmqKind::MXFP4 => super::mmq_mxfp4_twin(&w, &y, n_rows, n_cols, batch, ncols_pad),
            };
            let reference =
                crate::mul_mm_ref::mul_mm_reference(mm, &w, &x, n_rows, n_cols, batch, row_bytes)
                    .unwrap();
            let rms =
                (reference.iter().map(|v| v * v).sum::<f32>() / reference.len() as f32).sqrt();
            let l1 = crate::mul_mm_ref::product_l1(mm, &w, &x, n_rows, n_cols, batch, row_bytes)
                .unwrap();
            for (i, (((g, t), r), l)) in got.iter().zip(&twin).zip(&reference).zip(&l1).enumerate()
            {
                let what = format!("{kind:?} {n_rows}x{n_cols}x{batch} tile {cols} [{i}]");
                assert!(
                    (g - t).abs() <= 1e-5 * t.abs().max(rms),
                    "{what}: GPU {g} twin {t}"
                );
                // q8_1 rounds each activation by up to 1/254 of its run's
                // amax, and DS4's half-rounded scales and sums move the
                // terms Q4_K's min cancels: the twin lands within 0.15% of
                // the products' L1 of the reference on these fixtures, so
                // 0.5% bounds it with room and still catches a wrong scale.
                assert!(
                    (g - r).abs() <= 5e-3 * l + 1e-6 * rms,
                    "{what}: GPU {g} reference {r} (l1 {l})"
                );
            }
        }
    }
}

/// The grouped form equals the plain form run once per group on that
/// group's gathered rows, bit for bit: an empty group, a group of one,
/// one wider than a tile of tokens, tokens shared between groups.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up). Passed 2026-11-09 on an A100 (sm_80, Hugging Face Jobs)"]
fn the_grouped_int8_gemm_equals_one_gemm_per_group() {
    let dev = shared_device().unwrap();
    let tokens = 90usize;
    let groups: [Vec<i32>; 4] = [
        vec![],
        vec![17],
        (0..70).map(|i| (i * 7 % tokens) as i32).collect(),
        (0..19).map(|i| 89 - i * 3).collect(),
    ];
    for (kind, mm, n_cols) in [
        (MmqKind::Q8_0, &crate::mul_mm::Q8_0, 2880usize),
        (MmqKind::Q4_K, &crate::mul_mm::Q4_K, 512),
        (MmqKind::MXFP4, &crate::mul_mm::MXFP4, 2880),
    ] {
        let n_rows = 135usize;
        let (be, bb) = kind.block();
        let row_bytes = n_cols / be * bb;
        let weights: Vec<Vec<u8>> = (0..groups.len() as u32)
            .map(|g| crate::mul_mm_ref::fixtures::weights(mm, n_rows, n_cols, 600 + g))
            .collect();
        let x = wave(tokens * n_cols, 0.7);
        let mut group_off = vec![0i32];
        let mut gather = Vec::new();
        for g in &groups {
            gather.extend_from_slice(g);
            group_off.push(gather.len() as i32);
        }
        let pairs = gather.len();
        let ncols_pad = (pairs + kernel::COLS - 1).div_ceil(kernel::COLS) * kernel::COLS;
        let d_x = xfer::htod(&dev, "g x", &x).unwrap();
        let d_gather = xfer::htod(&dev, "g gather", &gather).unwrap();
        let d_off = xfer::htod(&dev, "g off", &group_off).unwrap();
        let y = quantize::enqueue_quantize(
            &dev,
            kind.layout(),
            &d_x,
            Some(&d_gather),
            n_cols,
            pairs,
            ncols_pad,
        )
        .unwrap();
        let held: Vec<_> = weights
            .iter()
            .map(|w| crate::gpu::resident_cuda_weights(&dev, w).unwrap())
            .collect();
        let ptrs: Vec<u64> = held
            .iter()
            .map(|w| *cudarc::driver::DevicePtr::device_ptr(&w.slice))
            .collect();
        let d_tab = xfer::htod(&dev, "g tab", &ptrs).unwrap();
        // SAFETY: the table points at `held`, alive past the download; the
        // offsets end at `pairs` with no group past 70; `y` was quantized
        // at `ncols_pad >= pairs + 63`.
        let out = unsafe {
            super::enqueue_mmq_grouped(
                &dev,
                kind,
                &d_tab.slice(..),
                groups.len(),
                &d_off,
                &y,
                ncols_pad,
                pairs,
                70,
                n_rows,
                n_cols,
                row_bytes,
            )
        }
        .unwrap();
        let got = xfer::dtoh(&dev, "g out", &out).unwrap();
        for (g, rows) in groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let xg: Vec<f32> = rows
                .iter()
                .flat_map(|&t| {
                    x[t as usize * n_cols..(t as usize + 1) * n_cols]
                        .iter()
                        .copied()
                })
                .collect();
            let d_xg = xfer::htod(&dev, "g xg", &xg).unwrap();
            let (want, w) = super::enqueue_mmq(
                &dev,
                kind,
                &weights[g],
                &d_xg,
                n_rows,
                n_cols,
                rows.len(),
                row_bytes,
            )
            .unwrap();
            let want = xfer::dtoh(&dev, "g want", &want).unwrap();
            drop(w);
            let lo = group_off[g] as usize * n_rows;
            let have = &got[lo..lo + rows.len() * n_rows];
            for (i, (a, b)) in have.iter().zip(&want).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{kind:?} group {g} [{i}]: grouped {a} plain {b}"
                );
            }
        }
        drop(held);
    }
}
