//! The routed-expert kernels (`mul_mv_id`, `mul_mm_id`) on CUDA, for
//! every kind with a dequant: kernel text, the routing map, and the
//! scalar twins. Always compiled, so the host check and the fence the
//! decoder asks run on a CPU-only build.
//!
//! # Built from the kind's one dequant
//!
//! - `mul_mv_id` is the entry `<kind>_moe_mv_id` in the generated
//!   matvec unit ([`crate::matvec_kinds::generic`]): block `(row, slot)`
//!   offsets the weights to `ids[slot]`'s expert and runs the same
//!   `frink_matvec_row` the decode matvec runs. ggml's `mul_mat_vec_q`
//!   with `ids` (`mmvq.cu`) is the same shape.
//! - `mul_mm_id` is the dense GEMM body ([`crate::mul_mm::gemm_unit`])
//!   with an indirection: `blockIdx.z` is one active expert, its tile
//!   columns are that expert's routed rows (`map_b` picks each one's
//!   activation row, `map_d` its output row), and its weights are the
//!   expert's slice of the plane. ggml's `mul_mat_id` gathers the rows
//!   per expert the same way before `mmq` (`mmid.cu`).
//!
//! Both read the kind's `frink_dequant_sub`, so the dequant exists once
//! per kind for the matvec, the GEMM, the gather and both id kernels,
//! and [`supported`] is the GEMM table: every kind CUDA runs.
//!
//! # Exact twins
//!
//! [`mv_id_reference`] is the generic matvec's twin per slot, and
//! [`mm_id_reference`] is the dense GEMM's twin over each expert's
//! gathered rows: the GEMM accumulates each output element over K in
//! the same order whatever tile its column lands in, so gathering
//! columns cannot change a bit. `tools/moe_id_host_check/run.sh`
//! executes both kernels' CUDA C on the host against these.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No GPU has run these.

use crate::matvec_kinds::generic;
use crate::mul_mm::{GemmIdParts, MulMmKind};

/// An expert must gather at least this many rows (routed slots) for a
/// step to take `mul_mm_id`; below it every slot runs `mul_mv_id`.
/// Metal's switch point (`mul_mm_id` once an expert gathers eight
/// tokens).
pub const MM_ID_MIN_ROWS: usize = 8;

/// Whether CUDA has both id kernels for `kind`: exactly the kinds with
/// a GEMM row, since both kernels are built from its dequant. The
/// decoder's fence: a pack in any other kind runs one matvec per expert.
pub fn supported(kind: &str) -> Option<&'static MulMmKind> {
    crate::mul_mm::kind_by_name(kind)
}

/// NVRTC module and entry for `kind`'s `mul_mv_id`. The entry lives in
/// the generated matvec unit; it is loaded under its own module key so
/// the decode matvec's module (loaded with its one entry) is untouched.
pub fn mv_id_names(kind: &MulMmKind) -> (&'static str, &'static str) {
    leaked(kind, "mv", || {
        let stem = kind.name.to_ascii_lowercase();
        (format!("frink_moe_mv_id_{stem}"), generic::id_entry(kind))
    })
}

/// NVRTC module and entry for `kind`'s `mul_mm_id`.
pub fn mm_id_names(kind: &MulMmKind) -> (&'static str, &'static str) {
    leaked(kind, "mm", || {
        let stem = kind.name.to_ascii_lowercase();
        (
            format!("frink_moe_mm_id_{stem}"),
            format!("{stem}_moe_mm_id"),
        )
    })
}

fn leaked(
    kind: &MulMmKind,
    which: &'static str,
    make: impl FnOnce() -> (String, String),
) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Names = HashMap<(&'static str, &'static str), (&'static str, &'static str)>;
    static NAMES: OnceLock<Mutex<Names>> = OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    *map.entry((kind.name, which)).or_insert_with(|| {
        let (m, f) = make();
        (
            Box::leak(m.into_boxed_str()) as &str,
            Box::leak(f.into_boxed_str()) as &str,
        )
    })
}

/// The unit holding `kind`'s `mul_mv_id`: the generated matvec unit.
pub fn mv_id_src(kind: &MulMmKind) -> &'static str {
    generic::matvec_src(kind)
}

/// The routed-expert half of the GEMM body. Block `z` is active expert
/// `z`; a block whose column tile is past that expert's rows returns
/// before the first barrier, uniformly for the whole block.
const MM_ID_PARTS: GemmIdParts = GemmIdParts {
    params: ",
    const int* __restrict__ map_b,
    const int* __restrict__ map_d,
    const int* __restrict__ experts,
    unsigned long long expert_stride",
    prologue: "    const int z = blockIdx.z;
    batch = experts[3 * z + 2];
    if (r1 >= batch) {
        return;
    }
    src0 += (size_t)experts[3 * z] * (size_t)expert_stride;
    map_b += experts[3 * z + 1];
    map_d += experts[3 * z + 1];
",
    b_row: "map_b[c]",
    d_row: "map_d[c]",
};

/// The unit holding `kind`'s `mul_mm_id`, with the GEMM's geometry
/// defines.
pub fn mm_id_src(kind: &MulMmKind) -> String {
    format!(
        "{}{}",
        crate::mul_mm::gemm_defines(),
        crate::mul_mm::gemm_unit(kind, mm_id_names(kind).1, Some(&MM_ID_PARTS))
    )
}

/// Which routed rows each active expert gathers. Built on the host
/// from the routing decision; the kernels read it as-is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdMap {
    /// The active experts, ascending.
    pub eids: Vec<i32>,
    /// Where each active expert's rows start in `map_b` / `map_d`.
    pub offsets: Vec<i32>,
    /// How many rows each active expert gathers.
    pub counts: Vec<i32>,
    /// Activation row of each gathered row.
    pub map_b: Vec<i32>,
    /// Output row (the slot) of each gathered row.
    pub map_d: Vec<i32>,
}

impl IdMap {
    /// `ids` holds one expert per slot, `top_k` slots per token. With
    /// `by_token` the activation row is the slot's token (gate, up);
    /// without it, the slot itself (down, which reads per-slot rows).
    /// Rows of one expert are in slot order.
    pub fn build(ids: &[u32], top_k: usize, n_experts: usize, by_token: bool) -> Self {
        let mut per: Vec<Vec<usize>> = vec![Vec::new(); n_experts];
        for (slot, &e) in ids.iter().enumerate() {
            per[e as usize].push(slot);
        }
        let mut m = IdMap {
            eids: Vec::new(),
            offsets: Vec::new(),
            counts: Vec::new(),
            map_b: Vec::with_capacity(ids.len()),
            map_d: Vec::with_capacity(ids.len()),
        };
        for (e, slots) in per.iter().enumerate() {
            if slots.is_empty() {
                continue;
            }
            m.eids.push(e as i32);
            m.offsets.push(m.map_b.len() as i32);
            m.counts.push(slots.len() as i32);
            for &s in slots {
                m.map_b.push(if by_token {
                    (s / top_k) as i32
                } else {
                    s as i32
                });
                m.map_d.push(s as i32);
            }
        }
        m
    }

    /// `(expert, offset, count)` per active expert, flattened: the one
    /// array the kernel reads them from (a launch takes at most twelve
    /// arguments).
    pub fn experts(&self) -> Vec<i32> {
        (0..self.eids.len())
            .flat_map(|z| [self.eids[z], self.offsets[z], self.counts[z]])
            .collect()
    }

    /// The most rows any one expert gathers.
    pub fn max_count(&self) -> usize {
        self.counts.iter().copied().max().unwrap_or(0) as usize
    }
}

/// One routed projection: a plane of `n_experts` matrices of `rows` x
/// `cols`, each `expert_stride` bytes, in `kind`'s format.
#[derive(Debug, Clone, Copy)]
pub struct IdPlane<'a> {
    pub kind: &'static MulMmKind,
    pub data: &'a [u8],
    pub n_experts: usize,
    pub expert_stride: usize,
    pub rows: usize,
    pub cols: usize,
}

impl IdPlane<'_> {
    /// Bytes per weight row.
    pub fn row_bytes(&self) -> usize {
        (self.cols / self.kind.block_elems) * self.kind.block_bytes
    }

    /// The shape agreement the kernels' index arithmetic assumes.
    pub fn validate(&self, ids: &[u32]) -> Result<(), String> {
        let k = self.kind;
        if self.rows == 0 || self.cols == 0 || !self.cols.is_multiple_of(k.block_elems) {
            return Err(format!(
                "moe id: {} columns is not whole {} blocks",
                self.cols, k.name
            ));
        }
        if !self.cols.is_multiple_of(crate::mul_mm::BK) {
            return Err(format!(
                "moe id: {} columns is not a multiple of the K tile",
                self.cols
            ));
        }
        if self.expert_stride < self.rows * self.row_bytes() {
            return Err(format!(
                "moe id: an expert stride of {} bytes for {} rows of {}",
                self.expert_stride,
                self.rows,
                self.row_bytes()
            ));
        }
        if self.data.len() < self.n_experts * self.expert_stride {
            return Err(format!(
                "moe id: a plane of {} bytes for {} experts of {}",
                self.data.len(),
                self.n_experts,
                self.expert_stride
            ));
        }
        if let Some(&e) = ids.iter().find(|&&e| e as usize >= self.n_experts) {
            return Err(format!("moe id: expert {e} of {}", self.n_experts));
        }
        Ok(())
    }

    fn expert(&self, e: usize) -> &[u8] {
        &self.data[e * self.expert_stride..e * self.expert_stride + self.rows * self.row_bytes()]
    }
}

/// Scalar twin of `mul_mv_id`: `out[slot * rows + row]` is the generic
/// matvec twin of expert `ids[slot]`'s row against activation row
/// `slot / x_div`.
pub fn mv_id_reference(p: &IdPlane<'_>, ids: &[u32], x: &[f32], x_div: usize) -> Vec<f32> {
    let rb = p.row_bytes();
    let n_blocks = p.cols / p.kind.block_elems;
    let mut out = vec![0f32; ids.len() * p.rows];
    for (slot, &e) in ids.iter().enumerate() {
        let w = p.expert(e as usize);
        let xr = &x[(slot / x_div) * p.cols..(slot / x_div + 1) * p.cols];
        for row in 0..p.rows {
            out[slot * p.rows + row] =
                generic::matvec_twin(p.kind, &w[row * rb..(row + 1) * rb], xr, n_blocks);
        }
    }
    out
}

/// Scalar twin of `mul_mm_id`: per active expert, the dense GEMM twin
/// over its gathered activation rows, scattered to their output rows.
/// `n_out` is the number of output rows (slots).
pub fn mm_id_reference(p: &IdPlane<'_>, map: &IdMap, x: &[f32], n_out: usize) -> Vec<f32> {
    let rb = p.row_bytes();
    let mut out = vec![0f32; n_out * p.rows];
    for z in 0..map.eids.len() {
        let (o, n) = (map.offsets[z] as usize, map.counts[z] as usize);
        let mut gathered = Vec::with_capacity(n * p.cols);
        for &b in &map.map_b[o..o + n] {
            gathered.extend_from_slice(&x[b as usize * p.cols..(b as usize + 1) * p.cols]);
        }
        let w = p.expert(map.eids[z] as usize);
        let y = crate::mul_mm_ref::mul_mm_reference(p.kind, w, &gathered, p.rows, p.cols, n, rb)
            .expect("validated shape");
        for (i, &d) in map.map_d[o..o + n].iter().enumerate() {
            out[d as usize * p.rows..(d as usize + 1) * p.rows]
                .copy_from_slice(&y[i * p.rows..(i + 1) * p.rows]);
        }
    }
    out
}

/// `silu(gate) * up`, the device kernel's formula (`gpu::SILU_MUL_
/// KERNEL_SRC`), for the twin of a whole routed SwiGLU step.
pub fn silu_mul_reference(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

#[cfg(test)]
mod tests;
