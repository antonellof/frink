//! The dp4a launch path: its routing rule on the host, and on a device
//! the quantizer and every kernel against their twins, the dp4a result
//! against the f32 one within the q8_1 rounding, and the routed form
//! against one launch per expert.
//!
//!   cargo test -p frink-cuda --features cuda -- --include-ignored dp4a

use super::*;
use crate::dp4a_twin::fixtures::{activations, dequant, dequant_q8_1, q8_1_bound, weights};
use crate::dp4a_twin::{dp4a_matvec_row, quantize_q8_1};
use crate::matvec_kinds::dp4a::DP4A;

/// The routing rule's shape half: whole q8_1 blocks, every slot's row
/// inside the input. A short input would have the kernel read past the
/// quantized buffer; a ragged stride would start a slot mid-block.
///
/// Sabotage: `last <= x_len` to `last <= x_len + 32` routes the short input.
#[test]
fn dp4a_routes_only_whole_q8_1_inputs_that_cover_every_row() {
    let q4k = dp4a_kind("q4_k_matvec").expect("Q4_K");
    let q80 = dp4a_kind("q8_0_matvec").expect("Q8_0");
    assert!(
        dp4a_route_for(q4k, 16, 4096, 0, 1).is_some(),
        "one row of 16 blocks"
    );
    assert!(
        dp4a_route_for(q4k, 16, 4064, 0, 1).is_none(),
        "input shorter than the row"
    );
    assert!(
        dp4a_route_for(q80, 3, 100, 0, 1).is_none(),
        "not whole q8_1 blocks"
    );
    assert!(
        dp4a_route_for(q80, 3, 96, 0, 4).is_some(),
        "four slots, one input"
    );
    assert!(
        dp4a_route_for(q80, 3, 4 * 96, 96, 4).is_some(),
        "four slots, four inputs"
    );
    assert!(
        dp4a_route_for(q80, 3, 4 * 96 - 32, 96, 4).is_none(),
        "the last slot overruns"
    );
    assert!(
        dp4a_route_for(q80, 3, 4 * 112, 112, 4).is_none(),
        "a stride of 3.5 blocks"
    );
    assert!(dp4a_route_for(q80, 0, 96, 0, 1).is_none(), "no blocks");
    assert!(dp4a_route_for(q80, 3, 96, 0, 0).is_none(), "no slots");
}

fn launch_of<'a>(kind: &Dp4aKind, w: &'a [u8], rows: usize, n_blocks: usize) -> MatvecLaunch<'a> {
    let mv = crate::matvec_kinds::KINDS
        .iter()
        .find(|m| m.fn_name == kind.base_fn)
        .expect("routed from a MatvecKind");
    MatvecLaunch {
        kernel_src: mv.source(),
        module_name: mv.module_name,
        fn_name: mv.fn_name,
        weights: w,
        rows,
        row_bytes: n_blocks * kind.block_bytes,
        n_blocks_per_row: n_blocks,
    }
}

/// `sum |w_i * xq_i|` over a row: the scale a contracted (FMA) device
/// result may differ from the twin's by, a few f32 ulps of it.
fn term_scale(kind: &Dp4aKind, row: &[u8], y: &[u8]) -> f64 {
    dequant(kind, row)
        .iter()
        .zip(dequant_q8_1(y))
        .map(|(a, b)| (f64::from(*a) * f64::from(b)).abs())
        .sum()
}

/// The device quantizer against its twin, byte for byte: an all-zero
/// block, exact halves, a block whose f16 `d` underflows, one whose f16
/// sum overflows, and a long vector over several threadblocks. Nothing
/// here contracts (no multiply feeds an add), so equality is the bar.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn the_q8_1_quantizer_matches_its_twin_byte_for_byte() {
    let dev = crate::gpu::shared_device().expect("device");
    let mut edge = vec![0f32; 32];
    edge.extend((0..32).map(|i| match i {
        0 => 127.0,
        1 => -127.0,
        2 => 2.5,
        3 => -2.5,
        _ => i as f32 - 16.0,
    }));
    edge.extend((0..32).map(|i| ((i as f32) * 0.7).sin() * 1e-6));
    edge.extend((0..32).map(|i| 3000.0 + i as f32));
    for x in [edge, activations(14336, 0.1), activations(32, 2.0)] {
        let d_x = dev.htod_copy(x.clone()).expect("upload");
        let y = enqueue_quantize_q8_1(&dev, &d_x).expect("quantize");
        let got = dev.dtoh_sync_copy(&y).expect("download");
        let want = quantize_q8_1(&x);
        let first = got.iter().zip(&want).position(|(a, b)| a != b);
        assert_eq!(got.len(), want.len());
        assert!(
            first.is_none(),
            "n={}: first differing byte {first:?}",
            x.len()
        );
    }
}

/// Every dp4a kernel, at every geometry the launch can pick, against
/// its twin on a device. The host check runs the same text on a CPU;
/// this is what proves NVRTC takes it, that `dp4a`, `prmt` and the f16
/// `cvt`s are the instructions the portable branches stand for, and
/// that the loads' alignment holds. The device contracts into FMAs, so
/// the bar is a few ulps of the row's term scale, not equality.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn every_dp4a_matvec_matches_its_twin_at_every_geometry() {
    let dev = crate::gpu::shared_device().expect("device");
    let rows = 37usize;
    for kind in DP4A {
        let counts: &[usize] = if kind.qk == 256 {
            &[1, 3, 6, 17]
        } else {
            &[1, 5, 31, 130]
        };
        for &n_blocks in counts {
            let w = weights(kind, rows * n_blocks, 40 + n_blocks as u32);
            let x = activations(n_blocks * kind.qk, n_blocks as f32);
            let y = quantize_q8_1(&x);
            let d_y = dev.htod_copy(y.clone()).expect("upload");
            let launch = launch_of(kind, &w, rows, n_blocks);
            for (nwarps, r) in [(1, 1), (2, 1), (2, 2), (4, 1), (4, 4)] {
                let g = Dp4aGeometry { nwarps, rows: r };
                let (d_out, _w) = enqueue_dp4a_at(&dev, kind, &launch, &d_y, g)
                    .unwrap_or_else(|e| panic!("{} {g:?}: {e:?}", kind.entry));
                let got = dev.dtoh_sync_copy(&d_out).expect("download");
                let rb = launch.row_bytes;
                for (i, v) in got.iter().enumerate() {
                    let row = &w[i * rb..(i + 1) * rb];
                    let want = dp4a_matvec_row(kind, row, &y, n_blocks, nwarps);
                    let tol = 1e-5 * term_scale(kind, row, &y) + 1e-6;
                    assert!(
                        f64::from((v - want).abs()) <= tol,
                        "{} {g:?} n={n_blocks} row {i}: device {v} twin {want} (tol {tol:e})",
                        kind.entry
                    );
                }
            }
        }
    }
}

/// The dp4a matvec against the f32 coalesced matvec of the same kind on
/// the same device: they differ by the q8_1 rounding of the activation
/// and by nothing else, so within `dp4a_twin::fixtures::q8_1_bound`
/// (about 2^-7.8 of each block's largest activation, per unit of weight
/// magnitude). Decode shapes: a 4096-wide K-quant row, a 4096-wide
/// Q8_0 row, gpt-oss's 2880-wide MXFP4 row.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn the_dp4a_matvec_is_within_the_q8_1_rounding_of_the_f32_one() {
    use crate::matvec_kinds::coalesced::{coalesced_kind, wpr_for, MatvecWidth};
    let dev = crate::gpu::shared_device().expect("device");
    let rows = 64usize;
    for kind in DP4A {
        let n_blocks = match kind.qk {
            256 => 16,
            _ if kind.block_bytes == 17 => 90,
            _ => 128,
        };
        let w = weights(kind, rows * n_blocks, 7);
        let x = activations(n_blocks * kind.qk, 0.7);
        let d_x = dev.htod_copy(x.clone()).expect("upload");
        let launch = launch_of(kind, &w, rows, n_blocks);
        let input = MatvecInput::new(&d_x);
        let (d_q, _wq) = enqueue_dp4a(&dev, kind, &launch, &input).expect("dp4a");
        let ck = coalesced_kind(kind.base_fn).expect("a coalesced kernel");
        let wpr = wpr_for(rows, n_blocks, 32 / ck.lanes(MatvecWidth::Wide));
        let (d_f, _wf) =
            crate::matvec_launch::enqueue_matvec_at(&dev, &launch, &d_x, MatvecWidth::Wide, wpr)
                .expect("f32");
        let (q, f) = (
            dev.dtoh_sync_copy(&d_q).unwrap(),
            dev.dtoh_sync_copy(&d_f).unwrap(),
        );
        let rb = launch.row_bytes;
        for i in 0..rows {
            let wr = dequant(kind, &w[i * rb..(i + 1) * rb]);
            let bound = q8_1_bound(&wr, &x);
            let gap = f64::from((q[i] - f[i]).abs());
            assert!(
                gap <= bound,
                "{} row {i}: dp4a {} f32 {} gap {gap:e} > {bound:e}",
                kind.entry,
                q[i],
                f[i]
            );
        }
    }
}

/// Several experts in one routed launch equal one launch each, bit for
/// bit: a slot is the single kernel's arithmetic at the single kernel's
/// geometry on its own weights and input. Once with one input for every
/// slot (gate, up) and once with an input each (down). The input is
/// quantized once for all of them.
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn several_experts_in_one_dp4a_launch_equal_one_launch_each() {
    use cudarc::driver::DevicePtr;
    let dev = crate::gpu::shared_device().expect("device");
    let (rows, k) = (37usize, 3usize);
    for kind in DP4A {
        let n_blocks = if kind.qk == 256 { 12 } else { 90 };
        let cols = n_blocks * kind.qk;
        let ws: Vec<Vec<u8>> = (0..k as u32)
            .map(|e| weights(kind, rows * n_blocks, 60 + e))
            .collect();
        let launches: Vec<_> = ws
            .iter()
            .map(|w| launch_of(kind, w, rows, n_blocks))
            .collect();
        let held: Vec<_> = ws
            .iter()
            .map(|w| crate::gpu::resident_cuda_weights(&dev, w).expect("resident"))
            .collect();
        let table: Vec<u64> = held.iter().map(|h| *h.slice.device_ptr()).collect();
        let d_table = dev.htod_copy(table).unwrap();
        let shape = MultiShape::of(&launches[0]);
        for x_stride in [0, cols] {
            let x = activations(k * cols, 1.3);
            let d_x = dev.htod_copy(x.clone()).unwrap();
            let input = MatvecInput::new(&d_x);
            assert!(!input.quantized());
            // SAFETY: the table's `k` pointers are `held`'s resident
            // weights of this kind and shape, alive past the download.
            let d_all = unsafe {
                enqueue_dp4a_multi_table(
                    &dev,
                    kind,
                    &shape,
                    &d_table.slice(..),
                    k,
                    &input,
                    x_stride,
                )
            }
            .expect("multi");
            assert!(input.quantized());
            let all = dev.dtoh_sync_copy(&d_all).unwrap();
            for (e, launch) in launches.iter().enumerate() {
                let xe = dev
                    .htod_copy(x[e * x_stride..e * x_stride + cols].to_vec())
                    .unwrap();
                let (d_one, _w) =
                    enqueue_dp4a(&dev, kind, launch, &MatvecInput::new(&xe)).expect("one");
                let one = dev.dtoh_sync_copy(&d_one).unwrap();
                for (r, (a, b)) in all[e * rows..(e + 1) * rows].iter().zip(&one).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "{} expert {e} row {r} stride {x_stride}: multi {a} one {b}",
                        kind.entry
                    );
                }
            }
        }
    }
}

/// Under `FRINK_CUDA_MATVEC=dp4a` the ordinary matvec entry routes each
/// dp4a kind to its dp4a kernel (bit-equal to calling it directly) and
/// every other kind to its f32 kernel, and one input serves several
/// matvecs. Without the switch it says so and checks nothing, since the
/// switch is read once per process:
///
///   FRINK_CUDA_MATVEC=dp4a cargo test -p frink-cuda --features cuda -- --ignored the_switch
#[test]
#[ignore = "requires real CUDA hardware and FRINK_CUDA_MATVEC=dp4a; not yet run"]
fn the_switch_routes_the_dp4a_kinds_through_one_quantized_input() {
    if !dp4a_enabled() {
        eprintln!("FRINK_CUDA_MATVEC is not dp4a in this process: nothing to check");
        return;
    }
    let dev = crate::gpu::shared_device().expect("device");
    let rows = 37usize;
    for kind in DP4A {
        let n_blocks = if kind.qk == 256 { 16 } else { 128 };
        let w = weights(kind, rows * n_blocks, 3);
        let launch = launch_of(kind, &w, rows, n_blocks);
        let d_x = dev.htod_copy(activations(n_blocks * kind.qk, 0.2)).unwrap();
        let input = MatvecInput::new(&d_x);
        let (a, _wa) =
            crate::matvec_launch::enqueue_matvec_in(&dev, &launch, &input).expect("routed");
        assert!(
            input.quantized(),
            "{}: the switch did not route to dp4a",
            kind.entry
        );
        let first = *cudarc::driver::DevicePtr::device_ptr(input.q8(&dev).unwrap());
        let (b, _wb) = enqueue_dp4a(&dev, kind, &launch, &input).expect("direct");
        assert_eq!(
            first,
            *cudarc::driver::DevicePtr::device_ptr(input.q8(&dev).unwrap())
        );
        let (a, b) = (
            dev.dtoh_sync_copy(&a).unwrap(),
            dev.dtoh_sync_copy(&b).unwrap(),
        );
        assert!(
            a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{}",
            kind.entry
        );
    }
}
