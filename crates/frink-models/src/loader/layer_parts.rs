//! Per-layer parts with a loader of their own: gpt-oss's side table,
//! attention sinks, and a dense FFN as one expert.

use frink_core::tensor::Tensor;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;
use frink_moe::ExpertWeights;

use crate::config::ModelConfig;

use super::tensors::{
    load_f32_vec, load_f32_vec_optional, load_weight_matrix, slice_quantized_rows,
};
use super::LoadError;

/// Like `load_f32_vec`, but for tensors that only exist on some
/// checkpoints (e.g. `attn_q_norm`/`attn_k_norm` -- OLMoE-style
/// per-projection QK-RMSNorm applied to the full q_proj/k_proj output
/// before RoPE, confirmed against `OlmoeAttention.forward` in
/// `transformers/models/olmoe/modeling_olmoe.py`: `q_norm(q_proj(x))`,
/// `k_norm(k_proj(x))`, both plain RMSNorm over the whole projected
/// width, not per-head). Absent for every other preset/fixture this
/// loader already handles -- `None` there is correct, not a missing
/// feature.
/// Loads the four gpt-oss-only side-table tensors for one layer, and
/// checks that the fifth, the attention sinks, was loaded onto the
/// layer's [`AttnWeights`] by the generic tensor-presence read.
///
/// Every one of them is **required**: a gpt-oss checkpoint that is
/// missing any of these is not a gpt-oss checkpoint frink can run, and
/// quietly substituting zeros would reintroduce exactly the
/// silently-wrong-graph failure this path exists to remove. The lengths
/// are asserted against the config for the same reason -- a bias of the
/// wrong width would otherwise be applied to a `zip`-truncated prefix
/// and produce a plausible, wrong answer.
///
/// Shapes follow `src/models/openai-moe.cpp::load_arch_tensors`:
/// `attn_sinks {n_head}` (`:44`, flags `0`, so REQUIRED there too),
/// `attn_output.bias {n_embd}`, `ffn_gate_inp.bias {n_expert}`,
/// `ffn_{gate,up}_exps.bias {n_ff_exp, n_expert}`,
/// `ffn_down_exps.bias {n_embd, n_expert}`. GGUF stores the fastest
/// dimension first, so the 2-D bias tensors arrive expert-major and
/// split by simple chunking.
pub(super) fn load_gpt_oss_layer(
    file: &impl TensorSource,
    l: usize,
    config: &ModelConfig,
    sinks_loaded: bool,
) -> Result<crate::decoder::GptOssLayer, LoadError> {
    let n_experts = config.moe.n_experts;
    let ff = config.moe.expert_ffn_dim;

    let want = |name: &str, got: usize, expect: usize| -> Result<(), LoadError> {
        if got == expect {
            Ok(())
        } else {
            Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!("{name} has {got} elements, expected {expect}"),
            ))
        }
    };

    if !sinks_loaded {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "blk.{l}.attn_sinks.weight is missing; gpt-oss requires it \
                 (src/models/openai-moe.cpp:44) and llama.cpp refuses the file without it"
            ),
        ));
    }
    // `attn_output.bias` is `AttnWeights::o_bias` now, read by
    // `crate::proj_bias` (gpt-oss is a REQUIRED row of its table).
    let router_bias = load_f32_vec(file, &format!("blk.{l}.ffn_gate_inp.bias"))?;
    want(
        &format!("blk.{l}.ffn_gate_inp.bias"),
        router_bias.len(),
        n_experts,
    )?;

    let gate_b = load_f32_vec(file, &format!("blk.{l}.ffn_gate_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_gate_exps.bias"),
        gate_b.len(),
        n_experts * ff,
    )?;
    let up_b = load_f32_vec(file, &format!("blk.{l}.ffn_up_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_up_exps.bias"),
        up_b.len(),
        n_experts * ff,
    )?;
    let down_b = load_f32_vec(file, &format!("blk.{l}.ffn_down_exps.bias"))?;
    want(
        &format!("blk.{l}.ffn_down_exps.bias"),
        down_b.len(),
        n_experts * config.hidden_dim,
    )?;

    let expert_bias = (0..n_experts)
        .map(|e| frink_moe::ExpertBias {
            gate: gate_b[e * ff..(e + 1) * ff].to_vec(),
            up: up_b[e * ff..(e + 1) * ff].to_vec(),
            down: down_b[e * config.hidden_dim..(e + 1) * config.hidden_dim].to_vec(),
        })
        .collect();

    Ok(crate::decoder::GptOssLayer {
        router_bias,
        expert_bias,
    })
}

/// `blk.N.attn_sinks.weight` when the file carries it, checked to be
/// one logit per query head of THIS layer (`{n_head}` in every graph
/// that creates it: `openai-moe.cpp:44`, `mimo2.cpp:58`).
///
/// Optional here because that is what the tensor's consumers make it:
/// `build_attn_mha` takes a nullable `sinks` and `mimo2.cpp:58` creates
/// it `TENSOR_NOT_REQUIRED`. gpt-oss, which requires it, checks the
/// result where its side table loads.
pub(super) fn load_attn_sinks(
    file: &impl TensorSource,
    l: usize,
    n_heads: usize,
) -> Result<Option<Vec<f32>>, LoadError> {
    let name = format!("blk.{l}.attn_sinks.weight");
    let Some(sinks) = load_f32_vec_optional(file, &name)? else {
        return Ok(None);
    };
    if sinks.len() != n_heads {
        return Err(LoadError::UnsupportedFeature(
            name,
            format!(
                "attention sinks are one logit per query head; this layer has {n_heads} heads \
                 and the tensor {} entries",
                sinks.len()
            ),
        ));
    }
    Ok(Some(sinks))
}

/// Dense-layer FFN tensors: standard gate/up/down, Phi-3 fused
/// `ffn_up` with `2 * ffn_dim` rows and no separate gate, the UNGATED
/// two-matrix FFN (`FfnActivation::ReluSqr`), or nothing at all for an
/// FFN-free layer (`ffn_dim == 0`).
///
/// `ffn_dim` is THIS layer's width (`ModelConfig::layer_shape`), which
/// is the model's for every architecture but the per-layer ones.
pub(super) fn load_dense_expert(
    file: &impl TensorSource,
    layer: usize,
    config: &ModelConfig,
    ffn_dim: usize,
) -> Result<ExpertWeights, LoadError> {
    if ffn_dim == 0 {
        return Ok(crate::layer_shapes::absent_ffn(config.hidden_dim));
    }
    let gate_name = format!("blk.{layer}.ffn_gate.weight");
    let up_name = format!("blk.{layer}.ffn_up.weight");
    let down_name = format!("blk.{layer}.ffn_down.weight");
    if config.ffn_is_ungated() {
        // `arcee.cpp:39-40` and `apertus.cpp:45-46` create `ffn_up` and
        // `ffn_down` and no gate; a file carrying one describes a graph
        // this architecture does not compute, and would otherwise be
        // left as an unread tensor with a less specific message.
        if file.find_tensor(&gate_name).is_some() {
            return Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!(
                    "{gate_name} is present but this architecture's FFN is ungated \
                     ({:?}: LLM_FFN_RELU_SQR under LLM_FFN_SEQ with a null gate, \
                     arcee.cpp:123-128, or ggml_xielu over ffn_up alone, apertus.cpp:129-142)",
                    config.ffn_activation
                ),
            ));
        }
        let up = load_weight_matrix(file, &up_name)?;
        if up.rows() != ffn_dim {
            return Err(LoadError::UnsupportedFeature(
                config.name.to_string(),
                format!(
                    "{up_name} has {} rows; the ungated FFN expects feed_forward_length = \
                     {ffn_dim}",
                    up.rows()
                ),
            ));
        }
        // The alias: the same tensor read again. A zero-copy view of the
        // same bytes for a quantized mmapped file; an owned widening for
        // an F32/F16 one. See `FfnActivation::ReluSqr` for why the pair
        // is aliased rather than the struct given an `Option`.
        return Ok(ExpertWeights {
            gate: load_weight_matrix(file, &up_name)?,
            up,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    if file.find_tensor(&gate_name).is_some() {
        return Ok(ExpertWeights {
            gate: load_weight_matrix(file, &gate_name)?,
            up: load_weight_matrix(file, &up_name)?,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    // Phi-3 fused SwiGLU: up is [hidden, 2*ff], first half gate, second up.
    let fused = load_weight_matrix(file, &up_name)?;
    let ff = ffn_dim;
    if fused.rows() != 2 * ff {
        return Err(LoadError::UnsupportedFeature(
            config.name.to_string(),
            format!(
                "{up_name} has {} rows without a companion ffn_gate; \
                 expected fused SwiGLU with 2*ffn_dim = {} rows",
                fused.rows(),
                2 * ff
            ),
        ));
    }
    let cols = fused.cols();
    // Quantized fused gate+up: split by rows, no dequant (Metal-capable).
    if let (Some(gate), Some(up)) = (
        slice_quantized_rows(&fused, 0, ff),
        slice_quantized_rows(&fused, ff, ff),
    ) {
        return Ok(ExpertWeights {
            gate,
            up,
            down: load_weight_matrix(file, &down_name)?,
        });
    }
    let mut full = Vec::with_capacity(fused.rows() * cols);
    for r in 0..fused.rows() {
        full.extend_from_slice(&fused.dequant_row(r));
    }
    let gate = WeightMatrix::F32(Tensor::new(full[..ff * cols].to_vec(), vec![ff, cols]));
    let up = WeightMatrix::F32(Tensor::new(full[ff * cols..].to_vec(), vec![ff, cols]));
    Ok(ExpertWeights {
        gate,
        up,
        down: load_weight_matrix(file, &down_name)?,
    })
}
