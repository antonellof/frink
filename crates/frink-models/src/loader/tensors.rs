//! Reading one tensor out of a GGUF: finding it, widening a 1-D tensor
//! to f32 through the one shared helper, loading a matrix as a
//! `WeightMatrix` that stays quantized on the mmap, and slicing rows or
//! experts out of a packed tensor.

use frink_core::tensor::Tensor;
use frink_core::weight_matrix::quant_kind_for;
use frink_core::weight_matrix::{WeightBytes, WeightMatrix};
use frink_gguf::{GgmlType, GgufError, TensorInfo, TensorSource};
use std::sync::Arc;

use super::LoadError;

pub(crate) fn find_info<'a>(
    file: &'a impl TensorSource,
    name: &str,
) -> Result<&'a TensorInfo, LoadError> {
    file.find_tensor(name)
        .ok_or_else(|| LoadError::Gguf(GgufError::TensorNotFound(name.to_string())))
}

pub(crate) fn load_f32_vec_optional(
    file: &impl TensorSource,
    name: &str,
) -> Result<Option<Vec<f32>>, LoadError> {
    if file.find_tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(load_f32_vec(file, name)?))
}

/// Slice `n` rows starting at `start` out of a quantized matrix without
/// dequantizing: every `Quantized` kind stores one interleaved block
/// buffer per row (fixed `row_bytes`), so a row range is a contiguous
/// byte range. Mapped sources stay zero-copy (sub-range of the same
/// mmap); other backings get an owned copy. Returns `None` for non-
/// quantized matrices (F32 / MXFP4) -- callers fall back to dequant.
pub(crate) fn slice_quantized_rows(
    m: &WeightMatrix,
    start: usize,
    n: usize,
) -> Option<WeightMatrix> {
    // A folded fused projection splits into folded parts sharing ONE
    // fold: the input transform is the same for q, k and v, and
    // `apply_gpu_multi` recognises the shared `Arc`.
    if let WeightMatrix::Folded { base, fold } = m {
        let mut part = slice_quantized_rows(base, start, n)?;
        part.fold_hadamard(fold.clone());
        return Some(part);
    }
    let WeightMatrix::Quantized {
        data,
        rows,
        cols,
        kind,
    } = m
    else {
        return None;
    };
    let total = data.len();
    if *rows == 0 || total % *rows != 0 || start + n > *rows {
        return None;
    }
    let row_bytes = total / *rows;
    let (b0, b1) = (start * row_bytes, (start + n) * row_bytes);
    let bytes = match data {
        WeightBytes::Mapped { mmap, range } => WeightBytes::Mapped {
            mmap: mmap.clone(),
            range: range.start + b0..range.start + b1,
        },
        other => WeightBytes::Owned(other.as_slice()[b0..b1].to_vec()),
    };
    Some(WeightMatrix::Quantized {
        data: bytes,
        rows: n,
        cols: *cols,
        kind: *kind,
    })
}

/// Widen a raw plain-float tensor (`F32` / `F16` / `BF16`) to `f32`.
///
/// The three unquantized element types are handled identically at every
/// call site (eager widening to an owned buffer -- none of them has a
/// block structure a fused dot kernel could exploit), and each of the
/// seven GGUF loaders used to inline the same two-way match. F16 had no
/// arm in any of them, which made every `*-f16.gguf` a hard
/// `UnsupportedDtype` even though the type was parsed and sized.
pub(crate) fn widen_plain_float(
    dtype: GgmlType,
    raw: &[u8],
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    match dtype {
        GgmlType::F32 => {
            let mut out = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes(*chunk));
            }
            Ok(out)
        }
        GgmlType::F16 => frink_quant::dequant_f16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::F16)),
        GgmlType::BF16 => frink_quant::dequant_bf16(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::BF16)),
        // MXFP4 is accepted as a weight matrix and as an MoE expert
        // tensor, and `WeightMatrix::dequant` already calls this
        // dequantizer, so refusing it here made a 1-D MXFP4 norm or
        // bias a hard load error on a checkpoint whose 2-D tensors of
        // the same type load fine. That contradicted this function's
        // own contract, which is to widen whatever the loaders accept.
        GgmlType::MXFP4 => frink_quant::dequant_mxfp4_gguf(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::MXFP4)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

pub(crate) fn load_f32_vec(file: &impl TensorSource, name: &str) -> Result<Vec<f32>, LoadError> {
    let info = find_info(file, name)?;
    let raw = file.tensor_bytes(name)?;
    match info.dtype {
        // MXFP4 rides with the plain floats because `widen_plain_float`
        // is where its arm already lives -- routing it here rather than
        // giving this table its own `dequant_mxfp4_gguf` call keeps ONE
        // MXFP4 arm in this file instead of two that can drift.
        //
        // It has to be in *both* tables' reach, and it was in neither's:
        // `load_weight_matrix` accepts MXFP4 as a 2-D weight and
        // `load_moe_expert_matrices` accepts it as an expert tensor, so
        // a checkpoint whose norms happen to be MXFP4 failed here with
        // `UnsupportedDtype` while its far larger tensors of the exact
        // same dtype loaded fine.
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::MXFP4 => {
            widen_plain_float(info.dtype, raw, name)
        }
        GgmlType::Q8_0 => frink_quant::dequant_q8_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_0)),
        GgmlType::Q4_0 => frink_quant::dequant_q4_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_0)),
        GgmlType::Q4K => frink_quant::dequant_q4_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4K)),
        GgmlType::Q5K => frink_quant::dequant_q5_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5K)),
        GgmlType::Q6K => frink_quant::dequant_q6_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q6K)),
        GgmlType::Q2K => frink_quant::dequant_q2_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q2K)),
        GgmlType::Q3K => frink_quant::dequant_q3_k(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q3K)),
        GgmlType::Q4_1 => frink_quant::dequant_q4_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q4_1)),
        GgmlType::Q5_0 => frink_quant::dequant_q5_0(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_0)),
        GgmlType::Q5_1 => frink_quant::dequant_q5_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q5_1)),
        GgmlType::Q8_1 => frink_quant::dequant_q8_1(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::Q8_1)),
        GgmlType::IQ4NL => frink_quant::dequant_iq4_nl(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4NL)),
        GgmlType::IQ4XS => frink_quant::dequant_iq4_xs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ4XS)),
        // The codebook-grid tiers. Rare on the 1-D tensors this
        // function widens (norms and biases are almost always F32),
        // but a dtype frink can decode should never be rejected here
        // just because the *other* dispatch table below knows it --
        // that split is how a supported format turns into a load
        // failure on the one checkpoint that uses it.
        GgmlType::IQ1S => frink_quant::dequant_iq1_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ1S)),
        GgmlType::IQ1M => frink_quant::dequant_iq1_m(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ1M)),
        GgmlType::IQ2XXS => frink_quant::dequant_iq2_xxs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2XXS)),
        GgmlType::IQ2XS => frink_quant::dequant_iq2_xs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2XS)),
        GgmlType::IQ2S => frink_quant::dequant_iq2_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ2S)),
        GgmlType::IQ3XXS => frink_quant::dequant_iq3_xxs(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ3XXS)),
        GgmlType::IQ3S => frink_quant::dequant_iq3_s(raw)
            .map_err(|_| LoadError::UnsupportedDtype(name.to_string(), GgmlType::IQ3S)),
        other => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
    }
}

/// Loads a 2D weight matrix, keeping Q8_0/Q4_0 tensors quantized (raw
/// bytes copied out, never dequantized) and only expanding truly F32
/// tensors. This is the memory- and bandwidth-saving path: for a
/// multi-billion-parameter checkpoint the difference between this and
/// "dequant everything on load" is the difference between fitting in
/// RAM and not.
pub(crate) fn load_weight_matrix(
    file: &impl TensorSource,
    name: &str,
) -> Result<WeightMatrix, LoadError> {
    let mut m = load_weight_matrix_unfolded(file, name)?;
    // A PrismML checkpoint folds a Hadamard rotation into the listed
    // weights (`crate::hadamard_fold`); the matrix carries the
    // activation-side transform so every `apply` undoes it.
    if let Some(fold) = crate::hadamard_fold::fold_for(file, name, m.cols())? {
        m.fold_hadamard(fold);
    }
    Ok(m)
}

pub(super) fn load_weight_matrix_unfolded(
    file: &impl TensorSource,
    name: &str,
) -> Result<WeightMatrix, LoadError> {
    let info = find_info(file, name)?;
    // GGUF's on-disk `ne[]` shape array is fastest-varying-dimension-first
    // (ggml convention), i.e. `[in_features, out_features]` for a 2D
    // weight matrix -- the *reverse* of the row-major `[rows, cols]` =
    // `[out_features, in_features]` order `WeightMatrix`/`matmul_f32`
    // need. Reversed here once so every consumer below gets the correct
    // orientation. Before this reversal existed, every 2D tensor in an
    // externally-produced GGUF file was silently loaded transposed -- a
    // real bug found by running a real downloaded checkpoint
    // (TinyLlama-1.1B-Chat, e.g. `attn_k.weight`'s real raw shape is
    // `[2048, 256]` = `[hidden_dim, kv_dim]` = `[in, out]`) -- found
    // as a real transposition bug affecting every externally-produced
    // GGUF file, caught by serving a real downloaded checkpoint.
    let shape: Vec<usize> = info.shape.iter().rev().map(|&d| d as usize).collect();
    // A ggml tensor's `ne[]` is always four long and trailing 1s are
    // implicit, so a GGUF writer is free to store a `[in, 1]` matrix
    // with `n_dims = 1`. llama.cpp reads it back as a matrix anyway --
    // `check_tensor_dims` compares each requested dimension against
    // `cur->ne[i]` and requires 1 for the dimensions the file does not
    // carry -- so a single-output projection is a 2-D weight there and
    // must be one here. Refusing it instead made a real checkpoint
    // unloadable: `cross-encoder/ms-marco-MiniLM-L6-v2` writes
    // `cls.output.weight` as `[384]`, i.e. the 1x384 relevance head
    // that `/v1/rerank` exists to run, and the whole route died at load
    // with "expected 2D".
    let (rows, cols) = match shape.as_slice() {
        [r, c] => (*r, *c),
        [c] => (1, *c),
        other => {
            return Err(LoadError::UnsupportedDtype(
                format!("{name} (expected 2D, got shape {other:?})"),
                info.dtype,
            ))
        }
    };
    // `shape` is what the file said; `[rows, cols]` is what the matrix
    // is. They differ exactly in the 1-D case above, and the `Tensor`
    // must carry the matrix shape or `apply` reads it as a vector.
    let shape = vec![rows, cols];

    match info.dtype {
        // BF16 has no block/scale structure to keep quantized-in-place
        // the way Q4_0/Q8_0/K-quants do -- there's no fused dot kernel
        // that would make sense for a plain narrowed float, so it's
        // eagerly widened to an owned f32 Tensor exactly like F32
        // tensors already are.
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let data = load_f32_vec(file, name)?;
            Ok(WeightMatrix::F32(Tensor::new(data, shape)))
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                frink_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                Ok(WeightMatrix::Quantized {
                    data: WeightBytes::Mapped { mmap, range },
                    rows,
                    cols,
                    kind,
                })
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}

/// Splits a packed 3D MoE expert tensor `blk.N.ffn_{gate,up,down}_exps.weight`
/// (shape `[n_experts, out_dim, in_dim]`) into per-expert `WeightMatrix`es,
/// slicing raw bytes directly (quantized tensors stay quantized; block
/// boundaries never cross expert boundaries since `in_dim` is a whole
/// number of quantization blocks). Matches llama.cpp/ik_llama.cpp layout
/// confirmed on real OLMoE and Qwen2-MoE GGUF checkpoints.
pub(crate) fn split_expert_tensor(
    file: &impl TensorSource,
    name: &str,
    n_experts: usize,
) -> Result<Vec<WeightMatrix>, LoadError> {
    let info = find_info(file, name)?;
    // Real raw shape is `[in_dim, out_dim, n_experts]` (ggml's
    // fastest-first `ne[]` order -- see `load_weight_matrix`'s doc
    // comment for the confirmed 2D case this generalizes from). `n_experts`
    // is the slowest-varying (last, i.e. outermost/most-major) dimension,
    // so each expert's `out_dim*in_dim` block is contiguous with experts
    // back-to-back in the mmap.
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let raw = file.tensor_bytes(name)?;

    match info.dtype {
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
            let all = crate::loader::widen_plain_float(info.dtype, raw, name)?;
            let per_expert = out_dim * in_dim;
            Ok((0..n_experts)
                .map(|e| {
                    WeightMatrix::F32(Tensor::new(
                        all[e * per_expert..(e + 1) * per_expert].to_vec(),
                        vec![out_dim, in_dim],
                    ))
                })
                .collect())
        }
        other => match quant_kind_for(other) {
            Some(kind) => {
                let (mmap, full_range) = file.tensor_mapped_range(name)?;
                #[cfg(feature = "metal")]
                frink_metal::gpu::register_weight_mmap(Arc::clone(&mmap));
                let bytes_per_expert = raw.len() / n_experts;
                Ok((0..n_experts)
                    .map(|e| WeightMatrix::Quantized {
                        data: WeightBytes::Mapped {
                            mmap: Arc::clone(&mmap),
                            range: (full_range.start + e * bytes_per_expert)
                                ..(full_range.start + (e + 1) * bytes_per_expert),
                        },
                        rows: out_dim,
                        cols: in_dim,
                        kind,
                    })
                    .collect())
            }
            None => Err(LoadError::UnsupportedDtype(name.to_string(), other)),
        },
    }
}
