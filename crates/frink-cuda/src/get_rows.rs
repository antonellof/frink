//! `get_rows`: the embedding gather on CUDA (#405). One kernel per
//! quant kind, each built from that kind's `frink_dequant_sub`, the
//! same function its GEMM's A-tile loader calls.
//!
//! # One dequant per kind
//!
//! The kind's dequant is written once, in its [`MulMmKind`] row
//! ([`crate::mul_mm_kinds`]). [`kernel_src`] splices the same preamble
//! the GEMM's [`crate::mul_mm::kernel_src_with_body`] emits (f16 decode,
//! K-scale unpack, codebook or grid tables, the dequant) ahead of a
//! gather body, so a kind with a GEMM has a gather by construction, and
//! [`kind_by_name`] answers from [`crate::mul_mm::KINDS`] rather than a
//! list of its own. ggml's `k_get_rows` (`ggml-cuda/getrows.cu`) is the
//! same shape: a gather is the kind's dequantize, written to the row a
//! token id names.
//!
//! # Exact, not close
//!
//! A gather does no arithmetic past the dequant, so its output must be
//! the host's `WeightMatrix::dequant_row` bit for bit (ADR 0009), up to
//! the sign of a zero (Q4_0's dequant follows ggml's CUDA order and
//! writes `+0` where the CPU writes `-0`; see the test's `same_value`): a
//! decode that gathers on the device and one that embeds on the host
//! read the same first hidden state.
//! `every_kind_gathers_frink_quants_dequant_bit_for_bit` holds the
//! scalar twin ([`get_rows_reference`]) to `frink_quant`'s dequant for
//! every kind, and `tools/get_rows_host_check/run.sh` executes this
//! file's CUDA C on the host against the twin.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No GPU has run these. Run
//! `cargo test -p frink-cuda --features cuda -- --ignored` on a device
//! before any doc calls the gather measured.

use crate::mul_mm::{kernel_src_with_body, MulMmKind, SUB};

/// Threads per gather block: one thread per 16-element sub-block, so a
/// 4096-wide row is four blocks of 64.
pub const THREADS: usize = 64;

/// CUDA caps `gridDim.y`, which carries the token, at 65535.
pub const MAX_IDS: usize = 65_535;

/// The gather body, identical for every kind. `table` is `n_rows`
/// quantized rows of `row_bytes` each; `dst` is `n_ids` rows of
/// `n_cols` f32, row `r` the dequantized row `ids[r]`. Thread `sub` of
/// block row `r` writes elements `[16 sub, 16 sub + 16)` of that row.
/// The ids are checked against the table on the host before a launch
/// (`validate`), so the kernel trusts them.
const BODY_SRC: &str = r#"
extern "C" __global__ void FX_FN_NAME(
    const unsigned char* __restrict__ table,
    const unsigned int* __restrict__ ids,
    float* __restrict__ dst,
    int n_ids,
    int n_cols,
    int row_bytes
) {
    const int r = blockIdx.y;
    const int sub = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_ids || sub >= n_cols / FX_SUB) {
        return;
    }
    const unsigned char* rp = table + (size_t)ids[r] * (size_t)row_bytes;
    float reg[FX_SUB];
    frink_dequant_sub(
        rp + (size_t)(sub / FX_NL) * (size_t)FX_BLOCK_BYTES,
        sub % FX_NL,
        reg);
    float* o = dst + (size_t)r * (size_t)n_cols + (size_t)sub * FX_SUB;
#pragma unroll
    for (int i = 0; i < FX_SUB; i++) {
        o[i] = reg[i];
    }
}
"#;

/// The kinds CUDA gathers: exactly the kinds with a dequant, which are
/// the GEMM's. `None` means the embedding row is dequantized on the
/// host.
pub fn kind_by_name(name: &str) -> Option<&'static MulMmKind> {
    crate::mul_mm::kind_by_name(name)
}

/// The NVRTC module and entry point for `kind`'s gather
/// (`frink_get_rows_q4_k`, `get_rows_q4_k`), leaked once per kind so
/// the load-once cache can key on `&'static str`.
pub fn names(kind: &MulMmKind) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Names = HashMap<&'static str, (&'static str, &'static str)>;
    static NAMES: OnceLock<Mutex<Names>> = OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    *map.entry(kind.name).or_insert_with(|| {
        let stem = kind.name.to_ascii_lowercase();
        (
            Box::leak(format!("frink_get_rows_{stem}").into_boxed_str()),
            Box::leak(format!("get_rows_{stem}").into_boxed_str()),
        )
    })
}

/// The complete translation unit for `kind`'s gather.
pub fn kernel_src(kind: &MulMmKind) -> String {
    let body = BODY_SRC.replace("FX_FN_NAME", names(kind).1);
    kernel_src_with_body(kind, &body)
}

/// The launch grid: sub-blocks along x, ids along y.
pub fn grid_dims(n_ids: usize, n_cols: usize) -> (usize, usize) {
    ((n_cols / SUB).div_ceil(THREADS), n_ids)
}

/// Why a gather was refused, named so the caller falls back to the
/// host row and can say why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GetRowsUnsupported {
    /// `n_cols` is not a whole number of `kind` blocks.
    ColsNotBlockAligned { n_cols: usize, kind: &'static str },
    /// `row_bytes` is not `n_cols` worth of blocks.
    RowBytesMismatch { row_bytes: usize, expected: usize },
    /// The table holds fewer than `n_rows * row_bytes` bytes.
    TableTooSmall { got: usize, want: usize },
    /// An id names a row past the table.
    IdOutOfRange { id: u32, n_rows: usize },
    /// No ids, or more than one grid can carry.
    IdCount(usize),
}

impl std::fmt::Display for GetRowsUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ColsNotBlockAligned { n_cols, kind } => {
                write!(f, "get_rows: {n_cols} columns is not whole {kind} blocks")
            }
            Self::RowBytesMismatch {
                row_bytes,
                expected,
            } => write!(
                f,
                "get_rows: row_bytes {row_bytes}, the shape needs {expected}"
            ),
            Self::TableTooSmall { got, want } => {
                write!(f, "get_rows: table is {got} bytes, needs {want}")
            }
            Self::IdOutOfRange { id, n_rows } => {
                write!(f, "get_rows: id {id} past a {n_rows}-row table")
            }
            Self::IdCount(n) => write!(f, "get_rows: {n} ids (1..={MAX_IDS})"),
        }
    }
}

impl std::error::Error for GetRowsUnsupported {}

/// The checks the kernel's index arithmetic assumes, shared by the
/// launch and the twin so they cannot check different things.
pub fn validate(
    kind: &MulMmKind,
    table_len: usize,
    n_rows: usize,
    ids: &[u32],
    n_cols: usize,
    row_bytes: usize,
) -> Result<(), GetRowsUnsupported> {
    if n_cols == 0 || !n_cols.is_multiple_of(kind.block_elems) {
        return Err(GetRowsUnsupported::ColsNotBlockAligned {
            n_cols,
            kind: kind.name,
        });
    }
    let expected = (n_cols / kind.block_elems) * kind.block_bytes;
    if row_bytes != expected {
        return Err(GetRowsUnsupported::RowBytesMismatch {
            row_bytes,
            expected,
        });
    }
    let want = n_rows * row_bytes;
    if table_len < want {
        return Err(GetRowsUnsupported::TableTooSmall {
            got: table_len,
            want,
        });
    }
    if ids.is_empty() || ids.len() > MAX_IDS {
        return Err(GetRowsUnsupported::IdCount(ids.len()));
    }
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= n_rows) {
        return Err(GetRowsUnsupported::IdOutOfRange { id, n_rows });
    }
    Ok(())
}

/// Scalar twin of the gather: the same sub-block walk, through the
/// kind's `dequant_twin`, into the same output layout.
pub fn get_rows_reference(
    kind: &MulMmKind,
    table: &[u8],
    n_rows: usize,
    ids: &[u32],
    n_cols: usize,
    row_bytes: usize,
) -> Result<Vec<f32>, GetRowsUnsupported> {
    validate(kind, table.len(), n_rows, ids, n_cols, row_bytes)?;
    let nl = kind.nl();
    let mut out = vec![0f32; ids.len() * n_cols];
    for (r, &id) in ids.iter().enumerate() {
        let rp = &table[id as usize * row_bytes..(id as usize + 1) * row_bytes];
        for sub in 0..n_cols / SUB {
            let at = (sub / nl) * kind.block_bytes;
            let mut reg = [0f32; SUB];
            (kind.dequant_twin)(&rp[at..at + kind.block_bytes], sub % nl, &mut reg);
            out[r * n_cols + sub * SUB..r * n_cols + (sub + 1) * SUB].copy_from_slice(&reg);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mul_mm::KINDS;
    use crate::mul_mm_ref::fixtures;

    type Dequant = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;

    /// `frink_quant`'s dequant for every kind in [`KINDS`]. No catch-all:
    /// a kind added to the table without one fails by name.
    fn cpu_dequant(name: &str) -> Dequant {
        match name {
            "Q8_0" => frink_quant::dequant_q8_0,
            "Q4_0" => frink_quant::dequant_q4_0,
            "Q5_0" => frink_quant::dequant_q5_0,
            "Q2_K" => frink_quant::dequant_q2_k,
            "Q3_K" => frink_quant::dequant_q3_k,
            "Q4_K" => frink_quant::dequant_q4_k,
            "Q5_K" => frink_quant::dequant_q5_k,
            "Q6_K" => frink_quant::dequant_q6_k,
            "IQ4_NL" => frink_quant::dequant_iq4_nl,
            "IQ4_XS" => frink_quant::dequant_iq4_xs,
            "MXFP4" => frink_quant::dequant_mxfp4_gguf,
            other => panic!("{other}: no frink_quant dequant to hold the gather to"),
        }
    }

    /// Bit equality, except that `+0` and `-0` agree and two NaNs
    /// agree. The sign of a zero is the one difference the dequants may
    /// show: Q4_0's is `d * q + (-8 d)` (llama's Metal and CUDA
    /// `convert.cu` order, `+0` at `q = 8`) where `frink_quant` and
    /// ggml's CPU write `(q - 8) * d` (`-0` when `d < 0`). A zero adds
    /// and squares the same with either sign, so the hidden state that
    /// follows is the same bits.
    fn same_value(g: f32, w: f32) -> bool {
        g.to_bits() == w.to_bits() || g == w || (g.is_nan() && w.is_nan())
    }

    /// The gather of every kind is the host's dequant of the same row,
    /// bit for bit up to the sign of a zero ([`same_value`]), which is what lets the decode take the device row in
    /// place of `dequant_row`. Ids out of order, repeated, and naming
    /// the last row.
    ///
    /// Sabotage: change any kind's `dequant_twin` arithmetic (or swap
    /// two rows' ids in the twin) and this names the kind and element.
    #[test]
    fn every_kind_gathers_frink_quants_dequant_bit_for_bit() {
        let n_rows = 9;
        let ids = [8u32, 0, 3, 3, 7];
        for k in KINDS {
            let n_cols = 2 * k.block_elems.max(64);
            let row_bytes = (n_cols / k.block_elems) * k.block_bytes;
            let table = fixtures::weights(k, n_rows, n_cols, 977);
            let got = get_rows_reference(k, &table, n_rows, &ids, n_cols, row_bytes).unwrap();
            let dequant = cpu_dequant(k.name);
            for (r, &id) in ids.iter().enumerate() {
                let row = &table[id as usize * row_bytes..(id as usize + 1) * row_bytes];
                let want = dequant(row).unwrap();
                for (i, (g, w)) in got[r * n_cols..(r + 1) * n_cols]
                    .iter()
                    .zip(&want)
                    .enumerate()
                {
                    assert!(
                        same_value(*g, *w),
                        "{} id {id} element {i}: gather {g}, frink_quant {w}",
                        k.name
                    );
                }
            }
        }
    }

    /// Every unit defines its entry point, once, and the dequant it
    /// calls; names never collide across kinds or with the GEMM's.
    #[test]
    fn every_kind_emits_its_own_gather_unit() {
        let mut seen = std::collections::HashSet::new();
        for k in KINDS {
            let (module, entry) = names(k);
            let src = kernel_src(k);
            assert_eq!(
                src.matches(&format!("void {entry}(")).count(),
                1,
                "{}",
                k.name
            );
            assert!(src.contains("void frink_dequant_sub("), "{}", k.name);
            assert!(src.contains(k.dequant_src), "{}", k.name);
            assert!(!src.contains("FX_FN_NAME"), "{}", k.name);
            assert!(seen.insert(module) && seen.insert(entry), "{}", k.name);
            assert_ne!(module, k.module_name);
            assert_eq!(names(k), (module, entry), "the leak is cached");
        }
    }

    /// Every refusal names what it refuses, and a well-formed call
    /// passes.
    #[test]
    fn validate_names_what_it_refuses() {
        let k = &crate::mul_mm::Q4_K;
        let rb = k.block_bytes;
        assert_eq!(validate(k, 4 * rb, 4, &[0, 3], 256, rb), Ok(()));
        assert!(matches!(
            validate(k, 4 * rb, 4, &[0], 128, rb),
            Err(GetRowsUnsupported::ColsNotBlockAligned { .. })
        ));
        assert!(matches!(
            validate(k, 4 * rb, 4, &[0], 256, rb + 2),
            Err(GetRowsUnsupported::RowBytesMismatch { .. })
        ));
        assert!(matches!(
            validate(k, 3 * rb, 4, &[0], 256, rb),
            Err(GetRowsUnsupported::TableTooSmall { .. })
        ));
        assert_eq!(
            validate(k, 4 * rb, 4, &[4], 256, rb),
            Err(GetRowsUnsupported::IdOutOfRange { id: 4, n_rows: 4 })
        );
        assert_eq!(
            validate(k, 4 * rb, 4, &[], 256, rb),
            Err(GetRowsUnsupported::IdCount(0))
        );
    }

    /// The grid covers every sub-block of a row, with no block wholly
    /// past it.
    #[test]
    fn the_grid_covers_the_row() {
        for n_cols in [32usize, 256, 1024, 4096, 4128] {
            let (gx, gy) = grid_dims(3, n_cols);
            assert_eq!(gy, 3);
            assert!(gx * THREADS >= n_cols / SUB);
            assert!((gx - 1) * THREADS < n_cols / SUB);
        }
    }
}
