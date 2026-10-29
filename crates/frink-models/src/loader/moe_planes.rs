//! The load-time validation that a layer's experts sit in contiguous
//! planes the Metal `mul_mm_sg` MoE kernels can index.

use frink_core::weight_matrix::WeightMatrix;
use frink_moe::ExpertWeights;

#[cfg(feature = "metal")]
use crate::decoder::MoePackedQ4Planes;

/// When every routed expert is mmap-backed with a Metal simdgroup-GEMM
/// kind and back-to-back slices, record the combined gate/up/down planes
/// for Metal packed MoE. Gate/up/down may differ in kind (e.g. Q4_K /
/// Q4_K / Q8_0) but must be uniform across experts per role.
#[cfg(feature = "metal")]
pub(super) fn try_build_moe_packed_q4_planes(
    experts: &[ExpertWeights],
) -> Option<MoePackedQ4Planes> {
    use frink_core::weight_matrix::{QuantKind, WeightBytes};
    use std::sync::Arc;

    if experts.is_empty() {
        return None;
    }

    fn mapped_sg(m: &WeightMatrix) -> Option<(WeightBytes, usize, &'static str)> {
        match m {
            WeightMatrix::Quantized {
                data: WeightBytes::Mapped { mmap, range },
                rows,
                kind,
                ..
            } => {
                let kind_str = match kind {
                    QuantKind::Q4_0 => "Q4_0",
                    QuantKind::Q5_0 => "Q5_0",
                    QuantKind::Q4K => "Q4_K",
                    QuantKind::Q5K => "Q5_K",
                    QuantKind::Q6K => "Q6_K",
                    QuantKind::Q8_0 => "Q8_0",
                    QuantKind::IQ4XS => "IQ4_XS",
                    QuantKind::Q2K => "Q2_K",
                    QuantKind::Q3K => "Q3_K",
                    QuantKind::IQ4NL => "IQ4_NL",
                    QuantKind::Mxfp4Gguf => "MXFP4",
                    QuantKind::IQ2XXS => "IQ2_XXS",
                    QuantKind::IQ2XS => "IQ2_XS",
                    QuantKind::IQ2S => "IQ2_S",
                    QuantKind::IQ3XXS => "IQ3_XXS",
                    QuantKind::IQ3S => "IQ3_S",
                    QuantKind::IQ1S => "IQ1_S",
                    QuantKind::IQ1M => "IQ1_M",
                    QuantKind::Q4_1 => "Q4_1",
                    QuantKind::Q5_1 => "Q5_1",
                    QuantKind::Tq1_0 => "TQ1_0",
                    _ => return None,
                };
                let _ = frink_metal::gpu::mul_mm_sg_meta(kind_str)?;
                Some((
                    WeightBytes::Mapped {
                        mmap: Arc::clone(mmap),
                        range: range.clone(),
                    },
                    *rows,
                    kind_str,
                ))
            }
            _ => None,
        }
    }

    let (gate0, ffn_rows, gate_kind) = mapped_sg(&experts[0].gate)?;
    let (up0, up_rows, up_kind) = mapped_sg(&experts[0].up)?;
    let (down0, hidden_rows, down_kind) = mapped_sg(&experts[0].down)?;
    if up_rows != ffn_rows {
        return None;
    }
    let WeightBytes::Mapped {
        mmap: gate_mmap,
        range: gate0_range,
    } = &gate0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: up_mmap,
        range: up0_range,
    } = &up0
    else {
        return None;
    };
    let WeightBytes::Mapped {
        mmap: down_mmap,
        range: down0_range,
    } = &down0
    else {
        return None;
    };

    let gate_stride = gate0_range.len();
    let up_stride = up0_range.len();
    let down_stride = down0_range.len();
    if gate_stride == 0 || up_stride == 0 || down_stride == 0 {
        return None;
    }

    let n = experts.len();
    for (i, ex) in experts.iter().enumerate().skip(1) {
        let (g, fr, gk) = mapped_sg(&ex.gate)?;
        let (u, ur, uk) = mapped_sg(&ex.up)?;
        let (d, hr, dk) = mapped_sg(&ex.down)?;
        if gk != gate_kind || uk != up_kind || dk != down_kind {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &g else {
            return None;
        };
        if fr != ffn_rows {
            return None;
        }
        if !Arc::ptr_eq(mmap, gate_mmap)
            || range.len() != gate_stride
            || range.start != gate0_range.start + i * gate_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &u else {
            return None;
        };
        if ur != ffn_rows
            || !Arc::ptr_eq(mmap, up_mmap)
            || range.len() != up_stride
            || range.start != up0_range.start + i * up_stride
        {
            return None;
        }
        let WeightBytes::Mapped { mmap, range } = &d else {
            return None;
        };
        if hr != hidden_rows
            || !Arc::ptr_eq(mmap, down_mmap)
            || range.len() != down_stride
            || range.start != down0_range.start + i * down_stride
        {
            return None;
        }
    }

    Some(MoePackedQ4Planes::new(
        WeightBytes::Mapped {
            mmap: Arc::clone(gate_mmap),
            range: gate0_range.start..gate0_range.start + n * gate_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(up_mmap),
            range: up0_range.start..up0_range.start + n * up_stride,
        },
        WeightBytes::Mapped {
            mmap: Arc::clone(down_mmap),
            range: down0_range.start..down0_range.start + n * down_stride,
        },
        gate_stride,
        up_stride,
        down_stride,
        n,
        ffn_rows,
        hidden_rows,
        gate_kind,
        up_kind,
        down_kind,
    ))
}
