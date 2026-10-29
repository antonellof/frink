//! The kernel coverage matrix: for every quant kind, backend and op,
//! whether a kernel runs it or the work leaves for the host (#405,
//! ADR 0010).
//!
//! # Where the answers come from
//!
//! Nothing here is a table of its own.
//!
//! - **Accelerator matvec and GEMM** cells are what
//!   [`WeightMatrix::probe_kernels_for`] records for a synthetic matrix
//!   of each kind. That is the probe the loader runs on every weight
//!   and the one whose misses the load-time `NO KERNEL` lines print, so
//!   the matrix and the log cannot disagree. The probe reads
//!   [`crate::weight_matrix::gpu_backend::BackendCaps`], which is the
//!   dispatch's own capability table (CUDA's is derived from
//!   `frink_cuda`'s kernel tables; Metal's and Vulkan's are held to
//!   their launch tables by the `gpu_backend` tests).
//! - **CPU matvec and GEMM** cells are
//!   [`cpu_int_dot_kind_supported`], the predicate the probe records the
//!   host tier with. It is a hand list beside the dispatch arms, so
//!   `the_cpu_int_dot_table_is_the_dispatch` below runs both CPU entry
//!   points for every kind and fails when they disagree.
//! - **Metal `get_rows`** is `frink_metal::embd_kinds`, the set the
//!   decode path's embedding gather looks itself up in. No other
//!   backend has a gather kernel; their embedding row is
//!   [`WeightMatrix::dequant_row`] on the host.
//!
//! `docs/KERNELS.md` is [`KernelMatrix::markdown`], checked in, and a
//! test fails when the two differ. Another test pins the host-fallback
//! count per backend and op, so closing a cell shows up as a diff.

use crate::kernel_registry::{host_fallback, op, Backend, Outcome, Registry};
use crate::weight_matrix::{cpu_int_dot_kind_supported, QuantKind, WeightBytes, WeightMatrix};

/// Width of the synthetic matrix each kind is probed with. A whole
/// number of 256-element super-blocks, so every kind's block geometry
/// and every int-dot width rule is satisfied: the matrix answers for a
/// well-formed tensor, which is the question.
const PROBE_COLS: usize = 1024;

/// The op classes the matrix covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Op {
    /// One activation against the weight matrix: decode.
    Matvec,
    /// A batch of activations: prefill.
    Gemm,
    /// One row of the token embedding, dequantized: every token.
    GetRows,
}

impl Op {
    pub const ALL: [Op; 3] = [Op::Matvec, Op::Gemm, Op::GetRows];

    pub fn name(self) -> &'static str {
        match self {
            Op::Matvec => "matvec",
            Op::Gemm => "gemm",
            Op::GetRows => "get_rows",
        }
    }

    /// What the op is for, for table headers.
    pub fn purpose(self) -> &'static str {
        match self {
            Op::Matvec => "decode",
            Op::Gemm => "prefill",
            Op::GetRows => "embedding",
        }
    }
}

/// What one (backend, op, kind) resolves to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cell {
    /// A kernel for this kind and op runs on this backend.
    Kernel,
    /// No kernel of this shape, but the work stays on the backend: one
    /// matvec per position over the same weights. Named by the fallback.
    Loop(&'static str),
    /// The work runs on the host CPU and the backend is idle for it.
    Host(&'static str),
    /// CPU only: no integer `vec_dot` for this kind, so the weight is
    /// dequantized into an f32 dot. The CPU column's slow tier.
    F32Dot,
    /// Accelerators only: the kind is ggml's activation format with no
    /// weight path ([`QuantKind::is_activation_format`]), so there is no
    /// reference kernel to match and no weight to expect. Not a fallback.
    NotApplicable,
}

impl Cell {
    /// The short mark the markdown tables print.
    pub fn mark(self) -> &'static str {
        match self {
            Cell::Kernel => "kernel",
            Cell::Loop(_) => "loop",
            Cell::Host(_) => "HOST",
            Cell::F32Dot => "f32",
            Cell::NotApplicable => "n/a",
        }
    }

    pub fn is_host(self) -> bool {
        matches!(self, Cell::Host(_))
    }
}

/// The host path an embedding row takes on a backend with no gather.
pub const HOST_GET_ROWS: &str = "dequant_row";

/// Every (backend, op, kind) cell, derived from the dispatch.
#[derive(Clone, Debug)]
pub struct KernelMatrix {
    cells: Vec<(Backend, Op, QuantKind, Cell)>,
}

impl KernelMatrix {
    /// Ask the dispatch, once per cell.
    pub fn derive() -> Self {
        let mut cells = Vec::new();
        for &backend in Backend::ALL {
            for &kind in QuantKind::ALL {
                let (matvec, gemm) = matmul_cells(backend, kind);
                cells.push((backend, Op::Matvec, kind, matvec));
                cells.push((backend, Op::Gemm, kind, gemm));
                cells.push((backend, Op::GetRows, kind, get_rows_cell(backend, kind)));
            }
        }
        KernelMatrix { cells }
    }

    pub fn cell(&self, backend: Backend, op: Op, kind: QuantKind) -> Cell {
        self.cells
            .iter()
            .find(|c| c.0 == backend && c.1 == op && c.2 == kind)
            .map(|c| c.3)
            .expect("derive() fills every (backend, op, kind)")
    }

    /// How many kinds on `backend` give `op` a cell matching `pred`.
    pub fn count(&self, backend: Backend, op: Op, pred: impl Fn(Cell) -> bool) -> usize {
        self.cells
            .iter()
            .filter(|c| c.0 == backend && c.1 == op && pred(c.3))
            .count()
    }

    /// The headline number: kinds whose `op` leaves `backend` for the host.
    pub fn host_fallbacks(&self, backend: Backend, op: Op) -> usize {
        self.count(backend, op, Cell::is_host)
    }

    /// `docs/KERNELS.md`.
    pub fn markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("# Kernel coverage\n\n");
        s.push_str(
            "Generated by `frink caps --kernels --write docs/KERNELS.md` from the\n\
             dispatch tables (`frink_core::kernel_matrix`); do not edit by hand.\n\
             `cargo test -p frink-core kernel_matrix` fails when this file and the\n\
             dispatch disagree. Tracked in #405, decided in\n\
             [ADR 0010](adr/0010-coverage-before-speed.md).\n\n",
        );
        s.push_str(
            "- `kernel`: a kernel for this kind and op runs on the backend.\n\
             - `loop`: no kernel of this shape; the work stays on the backend as\n  \
             one matvec per position.\n\
             - `HOST`: the work runs on the host CPU and the backend is idle for it.\n  \
             The output is still correct, which is why it reads as a slow row.\n\
             - `f32` (CPU only): no integer `vec_dot` for this kind, so the weight\n  \
             is dequantized into an f32 dot.\n\
             - `n/a` (accelerators only): ggml defines the kind only as the\n  \
             activation side of an integer dot (`Q8_1`: no `to_float`, no kernel\n  \
             on any backend, no quantizer writes it), so there is no reference\n  \
             kernel to match. Not counted as a host fallback; frink's CPU still\n  \
             reads such a weight.\n\n",
        );

        s.push_str("## Host fallbacks\n\n");
        s.push_str("Kinds out of ");
        s.push_str(&QuantKind::ALL.len().to_string());
        s.push_str(" whose op leaves the backend for the host (`f32` on CPU).\n\n");
        s.push_str("| backend |");
        for o in Op::ALL {
            s.push_str(&format!(" {} ({}) |", o.name(), o.purpose()));
        }
        s.push_str("\n|---|");
        for _ in Op::ALL {
            s.push_str("---|");
        }
        s.push('\n');
        for &b in Backend::ALL {
            s.push_str(&format!("| {b} |"));
            for o in Op::ALL {
                let n = if b.is_accelerator() {
                    self.host_fallbacks(b, o)
                } else {
                    self.count(b, o, |c| c == Cell::F32Dot)
                };
                s.push_str(&format!(" {n} |"));
            }
            s.push('\n');
        }
        s.push('\n');

        for o in Op::ALL {
            s.push_str(&format!("## {} ({})\n\n", o.name(), o.purpose()));
            s.push_str("| kind |");
            for &b in Backend::ALL {
                s.push_str(&format!(" {b} |"));
            }
            s.push_str("\n|---|");
            for _ in Backend::ALL {
                s.push_str("---|");
            }
            s.push('\n');
            for &k in QuantKind::ALL {
                s.push_str(&format!("| {} |", k.name()));
                for &b in Backend::ALL {
                    s.push_str(&format!(" {} |", self.cell(b, o, k).mark()));
                }
                s.push('\n');
            }
            s.push('\n');
        }

        s.push_str("## Reading the CPU column\n\n");
        s.push_str(
            "The CPU column reads the kind table, not one host's tier. Which half\n\
             of the integer tier a host takes is decided at run time\n\
             (`FRINK_CPU_INT_DOT`, and the x86 matvec half is off by measurement), so\n\
             a `kernel` cell is the best path the build has for that kind.\n\n",
        );
        s.push_str("## Reading the get_rows column\n\n");
        s.push_str(
            "Metal's gather runs inside the fused decode stack, for a model with no\n\
             embedding scale or norm. Everywhere else the embedding row is\n\
             `WeightMatrix::dequant_row` on the host, one row per token.\n",
        );
        s
    }
}

/// The matvec and GEMM cells for one kind on one backend.
fn matmul_cells(backend: Backend, kind: QuantKind) -> (Cell, Cell) {
    if backend.is_accelerator() && kind.is_activation_format() {
        return (Cell::NotApplicable, Cell::NotApplicable);
    }
    if !backend.is_accelerator() {
        let tier = if cpu_int_dot_kind_supported(kind, PROBE_COLS) {
            Cell::Kernel
        } else {
            Cell::F32Dot
        };
        return (tier, tier);
    }
    let reg = Registry::new();
    probe_matrix(kind).probe_kernels_for(
        &reg,
        backend,
        "kernel_matrix",
        std::panic::Location::caller(),
    );
    let entries = reg.entries();
    let read = |want: &str| {
        let e = entries
            .iter()
            .find(|e| e.key.backend == backend && e.key.op == want)
            .unwrap_or_else(|| panic!("the probe records {want} for {backend} {kind:?}"));
        match e.outcome {
            Outcome::Hit => Cell::Kernel,
            Outcome::Miss { fallback, .. } if host_fallback::is_host(fallback) => {
                Cell::Host(fallback)
            }
            Outcome::Miss { fallback, .. } => Cell::Loop(fallback),
        }
    };
    (read(op::MATVEC), read(op::GEMM_PREFILL))
}

/// The embedding gather. Exhaustive over [`Backend`] with no `_` arm, so
/// a new backend has to say whether it gathers on the device.
fn get_rows_cell(backend: Backend, kind: QuantKind) -> Cell {
    if backend.is_accelerator() && kind.is_activation_format() {
        return Cell::NotApplicable;
    }
    match backend {
        Backend::Cpu => Cell::Kernel,
        Backend::Metal => match frink_metal::embd_kinds::EmbdKind::from_kind_name(kind.name()) {
            Some(_) => Cell::Kernel,
            None => Cell::Host(HOST_GET_ROWS),
        },
        Backend::Cuda | Backend::Vulkan => Cell::Host(HOST_GET_ROWS),
    }
}

/// A weight of `kind` the probe can be asked about. The probe reads only
/// the kind and the width, so the bytes are left empty.
fn probe_matrix(kind: QuantKind) -> WeightMatrix {
    WeightMatrix::Quantized {
        data: WeightBytes::Owned(Vec::new()),
        rows: 1,
        cols: PROBE_COLS,
        kind,
    }
}

#[cfg(test)]
mod tests;
