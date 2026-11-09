//! Every `frink-cuda` kernel compiled for AMD, as text.
//!
//! The census is what the `hiprtc` gate compiles for every served target
//! (`rtc::tests::every_census_kernel_compiles_for_every_served_target`),
//! so a kernel added to `frink-cuda` that HIP cannot compile fails CI
//! rather than the first AMD user. Built from `frink-cuda`'s own tables,
//! so a new quant kind joins it without an edit here.
//!
//! Left out, deliberately: the tensor-core `mul_mm` body
//! (`mul_mm_tc`, `mma.sync` / `ldmatrix`), which is NVIDIA's instruction
//! set. AMD prefills through the generic `mul_mm` until a WMMA (RDNA3+)
//! or MFMA (CDNA) body exists. The kernels `frink-cuda` keeps behind its
//! `cuda` feature (attention, prefill, elementwise) are added by the
//! gate test itself, which sees them through a dev-dependency.

use frink_cuda::matvec_kinds::coalesced::{MatvecWidth, COALESCED};
use frink_cuda::matvec_kinds::dp4a::{Dp4aGeometry, DP4A};

/// One translation unit to compile.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Unique, for messages.
    pub name: String,
    /// The `extern "C" __global__` entry points, read off the source.
    pub entries: Vec<String>,
    /// The `frink-cuda` text, without the HIP prelude.
    pub source: String,
}

/// The `extern "C" __global__` entry points `source` defines, in order.
pub fn entries(source: &str) -> Vec<String> {
    let marker = "__global__ void ";
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(i) = rest.find(marker) {
        rest = &rest[i + marker.len()..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push(name);
        }
    }
    out
}

impl Kernel {
    /// A census entry for `source`, with its entry points read off it.
    pub fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let source = source.into();
        Self {
            name: name.into(),
            entries: entries(&source),
            source,
        }
    }
}

/// The always-compiled part of the census: every decode matvec (both
/// widths of the coalesced ones, at one and four warps per row), the
/// q8_1 quantizer and the dp4a matvecs, every GEMM, every embedding
/// gather and the split-K decode attention.
pub fn kernels() -> Vec<Kernel> {
    let mut out = Vec::new();
    for k in frink_cuda::matvec_kinds::KINDS {
        out.push(Kernel::new(
            format!("matvec {}", k.name),
            k.source().to_string(),
        ));
    }
    // One warp per row, and several: the shared-memory reduction a row
    // of more than one warp takes is its own code path.
    for k in COALESCED {
        for (w, tag) in [(MatvecWidth::Narrow, "narrow"), (MatvecWidth::Wide, "wide")] {
            for wpr in [1usize, 4] {
                out.push(Kernel::new(
                    format!("coalesced {} {tag} x{wpr}", k.entry),
                    k.source(w, wpr),
                ));
            }
            // The routed-expert form reads its weights through a table.
            out.push(Kernel::new(
                format!("coalesced {} {tag} routed", k.entry),
                k.multi_source(w, 4),
            ));
        }
    }
    // The integer-dot matvecs compile their portable branches here (no
    // `__CUDA_ARCH__`): one row a threadblock, `nwarps` rows (small k),
    // and the routed form.
    out.push(Kernel::new(
        "quantize_q8_1",
        frink_cuda::matvec_kinds::dp4a::quantize_source(),
    ));
    for k in DP4A {
        for (rows, multi, tag) in [(1, false, "x1"), (4, false, "x4"), (1, true, "routed")] {
            let g = Dp4aGeometry { nwarps: 4, rows };
            out.push(Kernel::new(
                format!("dp4a {} {tag}", k.entry),
                k.source(g, multi),
            ));
        }
    }
    for k in frink_cuda::mul_mm::KINDS {
        out.push(Kernel::new(
            format!("mul_mm {}", k.name),
            frink_cuda::mul_mm::kernel_src(k),
        ));
        out.push(Kernel::new(
            format!("get_rows {}", k.name),
            frink_cuda::get_rows::kernel_src(k),
        ));
    }
    out.push(Kernel::new(
        "decode_attn",
        frink_cuda::decode_attn::DECODE_ATTN_KERNEL_SRC.to_string(),
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_census_covers_every_table_and_no_tensor_core_body() {
        let ks = kernels();
        let n_mm = frink_cuda::mul_mm::KINDS.len();
        assert_eq!(
            ks.len(),
            frink_cuda::matvec_kinds::KINDS.len()
                + 6 * COALESCED.len()
                + 1
                + 3 * DP4A.len()
                + 2 * n_mm
                + 1
        );
        let mut names: Vec<&str> = ks.iter().map(|k| k.name.as_str()).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "a census name repeats");
        for k in &ks {
            assert!(
                !k.source.contains("mma.sync"),
                "{} carries NVIDIA tensor-core code",
                k.name
            );
            assert!(
                !k.entries.is_empty(),
                "{} defines no __global__ entry point",
                k.name
            );
        }
        // The table's own entry name is among what the source defines.
        for (k, kind) in ks.iter().zip(frink_cuda::matvec_kinds::KINDS) {
            assert!(k.entries.iter().any(|e| e == kind.fn_name), "{}", k.name);
        }
    }
}
