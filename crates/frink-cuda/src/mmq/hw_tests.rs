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
#[ignore = "requires real CUDA hardware; not yet run"]
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

/// The Q8_0 int8 GEMM is its twin's, within the f32 rounding of two
/// accumulation orders (none: one thread owns each output and walks K
/// in order), and close to the f32 reference within the q8_1 rounding:
/// a row count and a K that are not whole tiles, a batch past one tile
/// and one under it.
#[test]
#[ignore = "requires real CUDA hardware (sm_80 and up); not yet run"]
fn the_q8_0_int8_gemm_matches_its_twin_and_the_reference() {
    let dev = shared_device().unwrap();
    for (n_rows, n_cols, batch) in [(135usize, 96usize, 37usize), (256, 2880, 130), (33, 128, 1)] {
        let row_bytes = n_cols / 32 * 34;
        let w = crate::mul_mm_ref::fixtures::weights(&crate::mul_mm::Q8_0, n_rows, n_cols, 91);
        let x = wave(batch * n_cols, 0.4);
        let d_x = xfer::htod(&dev, "mmq x", &x).unwrap();
        let (out, held) = super::enqueue_mmq(
            &dev,
            MmqKind::Q8_0,
            &w,
            &d_x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        )
        .unwrap();
        let got = xfer::dtoh(&dev, "mmq out", &out).unwrap();
        drop(held);
        let ncols_pad = kernel::ncols_pad(batch);
        let y = quantize::quantize_twin(&x, None, n_cols, batch, ncols_pad, Layout::D4);
        let twin = super::mmq_q8_0_twin(&w, &y, n_rows, n_cols, batch, ncols_pad);
        let reference = crate::mul_mm_ref::mul_mm_reference(
            &crate::mul_mm::Q8_0,
            &w,
            &x,
            n_rows,
            n_cols,
            batch,
            row_bytes,
        )
        .unwrap();
        let rms = (reference.iter().map(|v| v * v).sum::<f32>() / reference.len() as f32).sqrt();
        for (i, ((g, t), r)) in got.iter().zip(&twin).zip(&reference).enumerate() {
            let what = format!("{n_rows}x{n_cols}x{batch} [{i}]");
            assert!(
                (g - t).abs() <= 1e-5 * t.abs().max(rms),
                "{what}: GPU {g} twin {t}"
            );
            // q8_1 rounds each activation by up to half a step, 1/254 of
            // its run's amax: a few parts in a thousand of the row's scale.
            assert!((g - r).abs() <= 2e-2 * rms, "{what}: GPU {g} reference {r}");
        }
    }
}
