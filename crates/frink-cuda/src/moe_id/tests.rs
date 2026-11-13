use super::*;
use crate::mul_mm::{KINDS, SUB};
use crate::mul_mm_ref::fixtures;

/// A plane of `n_experts` experts in `kind`, each `rows` x `cols`.
pub(crate) fn plane(
    kind: &'static MulMmKind,
    n_experts: usize,
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    fixtures::weights(kind, n_experts * rows, cols, 6007)
}

fn acts(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.017 + seed).sin()).collect()
}

/// The f64 dequantize-then-dot of one routed row, the independent
/// oracle the twins are held to, and the sum of the terms' magnitudes,
/// which is what f32 rounding error scales with (random K-quant scales
/// make large terms that cancel).
fn dot_f64(kind: &MulMmKind, row: &[u8], x: &[f32]) -> (f64, f64) {
    let (mut acc, mut mag) = (0f64, 0f64);
    for s in 0..x.len() / SUB {
        let at = (s / kind.nl()) * kind.block_bytes;
        let mut reg = [0f32; SUB];
        (kind.dequant_twin)(&row[at..at + kind.block_bytes], s % kind.nl(), &mut reg);
        for (i, r) in reg.iter().enumerate() {
            let t = f64::from(*r) * f64::from(x[s * SUB + i]);
            acc += t;
            mag += t.abs();
        }
    }
    (acc, mag)
}

/// Rows of one expert are in slot order, the activation row is the
/// token (gate, up) or the slot (down), and only routed experts appear.
#[test]
fn the_map_groups_slots_by_expert_in_slot_order() {
    // Three tokens, top 2: slots 0..6 route to experts 3,1 / 1,0 / 3,1.
    let ids = [3u32, 1, 1, 0, 3, 1];
    let m = IdMap::build(&ids, 2, 5, true);
    assert_eq!(m.eids, vec![0, 1, 3]);
    assert_eq!(m.counts, vec![1, 3, 2]);
    assert_eq!(m.offsets, vec![0, 1, 4]);
    assert_eq!(m.map_d, vec![3, 1, 2, 5, 0, 4]);
    assert_eq!(m.map_b, vec![1, 0, 1, 2, 0, 2]);
    assert_eq!(m.max_count(), 3);
    let down = IdMap::build(&ids, 2, 5, false);
    assert_eq!(down.map_b, down.map_d);
}

/// Both twins are the routed product, for every kind: each slot's
/// output is its expert's rows against its activation row, held to an
/// f64 dequantize-then-dot. Experts repeat and one is never routed.
///
/// Sabotage: offset the expert by `e + 1` in `IdPlane::expert` and
/// this names the kind.
#[test]
fn both_twins_are_the_routed_product_for_every_kind() {
    let (n_experts, top_k, n_tokens) = (4usize, 2usize, 3usize);
    let ids = [2u32, 0, 0, 3, 2, 2];
    for k in KINDS {
        let cols = k.block_elems.max(64).next_multiple_of(crate::mul_mm::BK);
        let cols = cols.next_multiple_of(k.block_elems);
        let rows = 5;
        let data = plane(k, n_experts, rows, cols);
        let p = IdPlane {
            kind: k,
            data: &data,
            n_experts,
            expert_stride: rows * (cols / k.block_elems) * k.block_bytes,
            rows,
            cols,
        };
        p.validate(&ids).unwrap();
        let x = acts(n_tokens * cols, 0.3);
        let mv = mv_id_reference(&p, &ids, &x, top_k);
        let mm = mm_id_reference(
            &p,
            &IdMap::build(&ids, top_k, n_experts, true),
            &x,
            ids.len(),
        );
        let rb = p.row_bytes();
        for (slot, &e) in ids.iter().enumerate() {
            let xr = &x[(slot / top_k) * cols..(slot / top_k + 1) * cols];
            for r in 0..rows {
                let w = &p.expert(e as usize)[r * rb..(r + 1) * rb];
                let (want, mag) = dot_f64(k, w, xr);
                let tol = 1e-5 * mag.max(1.0);
                for (what, got) in [
                    ("mv_id", mv[slot * rows + r]),
                    ("mm_id", mm[slot * rows + r]),
                ] {
                    assert!(
                        f64::from(got).is_nan() && want.is_nan()
                            || (f64::from(got) - want).abs() <= tol,
                        "{} {what} slot {slot} row {r}: {got} vs {want}",
                        k.name
                    );
                }
            }
        }
    }
}

/// Every unit defines its id entry once and leaves no placeholder; the
/// dense GEMM unit is unchanged in shape (no id parameters), and no two
/// kinds or kernels share an NVRTC name.
#[test]
fn every_id_unit_defines_its_entry_once() {
    let mut seen = std::collections::HashSet::new();
    for k in KINDS {
        let (mvm, mve) = mv_id_names(k);
        let (mmm, mme) = mm_id_names(k);
        assert_eq!(
            mv_id_src(k).matches(&format!("void {mve}(")).count(),
            1,
            "{}",
            k.name
        );
        let mm = mm_id_src(k);
        assert_eq!(mm.matches(&format!("void {mme}(")).count(), 1, "{}", k.name);
        assert!(
            mm.contains("map_b[c]") && mm.contains("experts[3 * z + 2]"),
            "{}",
            k.name
        );
        assert!(!mm.contains("FX_ID_"), "{}", k.name);
        let dense = crate::mul_mm::kernel_src(k);
        assert!(
            !dense.contains("map_b") && !dense.contains("FX_ID_"),
            "{}",
            k.name
        );
        for n in [mvm, mve, mmm, mme] {
            assert!(seen.insert(n), "{n} is not unique");
        }
        assert_ne!(
            mvm,
            crate::matvec_kinds::kind_by_name(k.name)
                .unwrap()
                .module_name
        );
        assert!(supported(k.name).is_some());
    }
    assert!(supported("Q8_1").is_none());
}

/// A plane the kernels cannot index is refused by name.
#[test]
fn validate_names_what_it_refuses() {
    let k = &crate::mul_mm::Q8_0;
    let data = vec![0u8; 2 * 4 * 2 * 34];
    let p = IdPlane {
        kind: k,
        data: &data,
        n_experts: 2,
        expert_stride: 4 * 2 * 34,
        rows: 4,
        cols: 64,
    };
    assert!(p.validate(&[0, 1]).is_ok());
    assert!(p.validate(&[2]).unwrap_err().contains("expert 2"));
    assert!(IdPlane { cols: 48, ..p }.validate(&[0]).is_err());
    assert!(IdPlane {
        expert_stride: 100,
        ..p
    }
    .validate(&[0])
    .is_err());
    assert!(IdPlane { n_experts: 3, ..p }.validate(&[0]).is_err());
}
