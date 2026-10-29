use super::*;
use crate::weight_matrix::gpu_backend::{with_gpu_backend_caps, BackendCaps};
use crate::weight_matrix::{cpu_int_dot_for, IntDotShape};

/// Host-fallback cells per (backend, op), and the CPU's `f32` cells.
///
/// This is the scoreboard for north-star point 3. Closing a kernel
/// turns this red on purpose: lower the number in the same change and
/// regenerate `docs/KERNELS.md`, so the diff says which cell closed.
const PINNED: &[(Backend, Op, usize)] = &[
    (Backend::Cpu, Op::Matvec, 17),
    (Backend::Cpu, Op::Gemm, 17),
    (Backend::Cpu, Op::GetRows, 0),
    (Backend::Cuda, Op::Matvec, 12),
    (Backend::Cuda, Op::Gemm, 12),
    (Backend::Cuda, Op::GetRows, 23),
    (Backend::Metal, Op::Matvec, 4),
    (Backend::Metal, Op::Gemm, 4),
    (Backend::Metal, Op::GetRows, 4),
    (Backend::Vulkan, Op::Matvec, 22),
    (Backend::Vulkan, Op::Gemm, 23),
    (Backend::Vulkan, Op::GetRows, 23),
];

/// The gap a host fallback or a CPU `f32` cell stands for.
fn gap(backend: Backend, cell: Cell) -> bool {
    if backend.is_accelerator() {
        cell.is_host()
    } else {
        cell == Cell::F32Dot
    }
}

#[test]
fn host_fallback_counts_are_pinned() {
    let m = KernelMatrix::derive();
    let mut got = Vec::new();
    for &b in Backend::ALL {
        for o in Op::ALL {
            got.push((b, o, m.count(b, o, |c| gap(b, c))));
        }
    }
    assert_eq!(
        got.len(),
        PINNED.len(),
        "a backend or op was added; pin its row"
    );
    for row in &got {
        assert!(
            PINNED.contains(row),
            "kernel coverage changed: {:?} is now {}. Update PINNED and run \
             `frink caps --kernels --write docs/KERNELS.md`.\nfull: {got:?}",
            (row.0, row.1),
            row.2,
        );
    }
}

/// `docs/KERNELS.md` is the rendering of the dispatch, not a copy that
/// can drift from it.
#[test]
fn the_checked_in_doc_is_the_dispatch() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/KERNELS.md");
    let on_disk = std::fs::read_to_string(path).expect("docs/KERNELS.md is checked in");
    let derived = KernelMatrix::derive().markdown();
    assert!(
        on_disk == derived,
        "docs/KERNELS.md disagrees with the dispatch. Regenerate it with \
         `frink caps --kernels --write docs/KERNELS.md`."
    );
}

/// The matrix tells host from device by the fallback's name, so every
/// backend's `GEMM_FALLBACK` must be a registered host path or name its
/// own backend. Without this a new host path spelled any other way
/// would be counted as staying on the GPU.
#[test]
fn every_gemm_fallback_is_a_host_path_or_names_its_backend() {
    macro_rules! check {
        ($b:ty) => {
            let f = <$b as BackendCaps>::GEMM_FALLBACK;
            assert!(
                host_fallback::is_host(f) || f.starts_with(<$b as BackendCaps>::NAME),
                "{}: GEMM_FALLBACK {f:?} is neither in host_fallback::ALL nor on the device",
                <$b as BackendCaps>::NAME
            );
        };
    }
    with_gpu_backend_caps!(check);
}

/// Every accelerator cell agrees with the backend's capability table.
/// The matrix reads the probe; this reads the table the probe reads, so
/// a probe that stopped consulting it fails here.
#[test]
fn accelerator_cells_are_the_capability_tables() {
    let m = KernelMatrix::derive();
    macro_rules! check {
        ($b:ty) => {
            let b = <$b as BackendCaps>::ID;
            for &k in QuantKind::ALL {
                let matvec = <$b as BackendCaps>::matvec_kernel(k).is_some();
                let gemm = <$b as BackendCaps>::gemm_supported(k);
                assert_eq!(
                    m.cell(b, Op::Matvec, k) == Cell::Kernel,
                    matvec,
                    "{b} matvec {k:?}"
                );
                assert_eq!(
                    m.cell(b, Op::Gemm, k) == Cell::Kernel,
                    gemm,
                    "{b} gemm {k:?}"
                );
                if !matvec {
                    assert!(m.cell(b, Op::Gemm, k).is_host(), "{b} gemm {k:?}");
                }
            }
        };
    }
    with_gpu_backend_caps!(check);
}

/// A weight of `kind` whose bytes cycle through `0x11..=0x17`.
///
/// Every byte in that range read as the high half of an f16 has an
/// exponent field of 4 or 5, so every scale in every format is finite;
/// every grid index is in range; and the bytes vary, so no format
/// cancels to zero. A constant fill does cancel: Q4_K with `d == dmin`
/// and equal scales and mins dequantizes every weight to exactly 0, and
/// both paths then agree to the bit.
fn filled(kind: QuantKind, rows: usize, cols: usize) -> WeightMatrix {
    let probe = WeightMatrix::Quantized {
        data: WeightBytes::Owned(Vec::new()),
        rows: 1,
        cols,
        kind,
    };
    let row_bytes = probe.block_bytes_per_row(kind, cols);
    WeightMatrix::Quantized {
        data: WeightBytes::Owned(
            (0..rows * row_bytes)
                .map(|i| 0x11 + (i % 7) as u8)
                .collect(),
        ),
        rows,
        cols,
        kind,
    }
}

/// Activations that do not survive Q8 quantization exactly, so the
/// integer path and the f32 path cannot agree to the bit.
fn lossy_acts(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 + 1e-3
        })
        .collect()
}

fn differs(a: &[f32], b: &[f32]) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
}

/// [`cpu_int_dot_kind_supported`] is a hand list beside the dispatch
/// arms in `apply_cpu` and `apply_batch`. This runs both for every kind
/// and asks which path they took: the f32 fallthrough is
/// `WeightMatrix::dot` per row, bit for bit, and the integer arms
/// quantize the activation, so they cannot match it.
///
/// It caught IQ4_XS: both entry points have its `iq4_xs_q8` arm and the
/// predicate did not list it.
///
/// The tier is forced on, as every shipped binary has it
/// (`default_cpu_int_dot_on`). Each half then runs only where this
/// host's kernels take it (`cpu_int_dot_for`): on x86 the matvec half
/// is off by measurement and only the batch half is checked.
#[test]
fn the_cpu_int_dot_table_is_the_dispatch() {
    if crate::weight_matrix::active_backend() != Backend::Cpu {
        return; // a GPU build would answer for the GPU
    }
    let _int_dot = crate::weight_matrix::tests::ForceIntDot::new(true);
    let (rows, cols, batch) = (8usize, PROBE_COLS, 3usize);
    let mut checked = 0;
    for &kind in QuantKind::ALL {
        let w = filled(kind, rows, cols);
        let row_bytes = w.block_bytes_per_row(kind, cols);
        let WeightMatrix::Quantized { data, .. } = &w else {
            unreachable!()
        };
        let row = |r: usize| &data.as_slice()[r * row_bytes..(r + 1) * row_bytes];
        let claimed = cpu_int_dot_kind_supported(kind, cols);

        if cpu_int_dot_for(IntDotShape::Matvec) {
            let x = lossy_acts(cols, 7);
            let f32_path: Vec<f32> = (0..rows)
                .map(|r| WeightMatrix::dot(kind, row(r), &x))
                .collect();
            let took_int = differs(&w.apply_cpu(&x), &f32_path);
            assert_eq!(
                took_int,
                claimed,
                "{kind:?} matvec: cpu_int_dot_kind_supported says {claimed}, apply_cpu \
                 took the {} path",
                if took_int { "integer" } else { "f32" }
            );
            checked += 1;
        }
        if cpu_int_dot_for(IntDotShape::BatchGemm) {
            let xb = lossy_acts(batch * cols, 11);
            let mut f32_path = vec![0f32; batch * rows];
            for b in 0..batch {
                for r in 0..rows {
                    f32_path[b * rows + r] =
                        WeightMatrix::dot(kind, row(r), &xb[b * cols..(b + 1) * cols]);
                }
            }
            let took_int = differs(&w.apply_batch(&xb, batch), &f32_path);
            assert_eq!(
                took_int,
                claimed,
                "{kind:?} batch: cpu_int_dot_kind_supported says {claimed}, apply_batch \
                 took the {} path",
                if took_int { "integer" } else { "f32" }
            );
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("the_cpu_int_dot_table_is_the_dispatch: no integer tier on this host");
    }
}

/// The Metal gather column is the decode path's table, and no other
/// backend claims one.
#[test]
fn get_rows_is_the_metal_gather_table_and_host_elsewhere() {
    let m = KernelMatrix::derive();
    for &k in QuantKind::ALL {
        let gathers = frink_metal::embd_kinds::EmbdKind::from_kind_name(k.name()).is_some();
        assert_eq!(
            m.cell(Backend::Metal, Op::GetRows, k) == Cell::Kernel,
            gathers
        );
        assert_eq!(m.cell(Backend::Cpu, Op::GetRows, k), Cell::Kernel);
        assert!(m.cell(Backend::Cuda, Op::GetRows, k).is_host());
        assert!(m.cell(Backend::Vulkan, Op::GetRows, k).is_host());
    }
}
