//! Loads `frink-models::kimi_decoder` weights from a real Kimi K3
//! **GGUF** checkpoint (as opposed to `kimi_loader`, which reads the
//! real safetensors checkpoint) -- via `frink_gguf::TensorSource`, the
//! same trait `frink-models::loader`'s GQA-model GGUF path uses.
//!
//! Every tensor name/shape here is confirmed against real, inspectable
//! upstream source, not guessed: `ggml-org/llama.cpp#26185` ("model:
//! add Kimi-K3 text model", open/unmerged as of 2026-08-01), whose
//! `src/models/kimi-k3.cpp` (`create_tensor(tn(...))` calls) gives the
//! exact real tensor names and shapes, and whose
//! `conversion/kimi_k3.py`/`gguf-py/gguf/constants.py` confirm how the
//! real HF checkpoint's tensors map onto them. See
//! docs/MODELS.md's "Kimi K3 GGUF loader" section for the full
//! citation trail. This loader has NOT been run against a real K3 GGUF
//! file (the smallest real quant, `unsloth/Kimi-K3-GGUF`'s UD-IQ1_S, is
//! 594GB) -- it is real, inspectable code built from real upstream
//! evidence, tested here against small synthetic on-disk fixtures.
//!
//! One real, non-obvious fact this loader has to account for:
//! **the block-residual weights are stored differently in GGUF than in
//! the real safetensors checkpoint.** Safetensors carries
//! `self_attention_res_norm.weight`/`self_attention_res_proj.weight`
//! (and the `mlp_res_*`/`output_attn_res_*` equivalents) as two
//! separate `[hidden_dim]` vectors each; `conversion/kimi_k3.py`'s
//! `_try_fuse_res` fuses each pair into a **single** `[hidden_dim]`
//! GGUF tensor at conversion time (`blk.{bid}.attn_res_score`,
//! `blk.{bid}.ffn_res_score`, `output_res_score`), since
//! `_apply_attn_res` in the real `modeling_kimi_linear.py` only ever
//! uses the elementwise product of the two factors, never either alone.
//! `frink_models::block_residual::apply_attn_res_prescored` takes that
//! product directly; this loader reads the fused GGUF tensor into the
//! existing `KimiDecoderLayerWeights::self_attention_res_norm_weight`/
//! `mlp_res_norm_weight`/`KimiDecoderWeights::output_attn_res_norm_weight`
//! fields (each paired with a constant-`1.0` "proj" vector), so
//! `norm_weight * proj_weight` reproduces the real fused score exactly
//! without changing `kimi_decoder`'s existing math or struct shape.

use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::{GgmlType, TensorSource};

use crate::config::LayerAttentionKind;
use crate::kda::KdaAttnWeights;
use crate::kimi_decoder::{
    DenseMlpWeights, KimiDecoderLayerWeights, KimiDecoderWeights, KimiLayerAttention, KimiLayerFfn,
};
use crate::latent_moe::{KimiExpertBacking, KimiExpertWeights, KimiLatentMoeWeights};
use crate::loader::LoadError;
use crate::loader::{load_f32_vec, load_weight_matrix, split_expert_tensor};
use crate::mla::{MlaAttnWeights, MlaKvB, MlaQProj};

/// Real per-layer hyperparameters needed to load any layer from a real
/// Kimi K3 GGUF file -- the GGUF counterpart of
/// `kimi_loader::KimiRealHparams`, plus `n_expert_latent` (GGUF's
/// `{arch}.expert_latent_length`, absent from the safetensors loader
/// since that path reads `routed_expert_hidden_size` from a different
/// source: the real HF `config.json`).
pub struct KimiGgufHparams {
    pub hidden_dim: usize,
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub mla_num_heads: usize,
    pub mla_q_lora_rank: usize,
    pub mla_kv_lora_rank: usize,
    pub mla_qk_nope_head_dim: usize,
    pub mla_qk_rope_head_dim: usize,
    pub mla_v_head_dim: usize,
    pub dense_intermediate_dim: usize,
    pub moe_hidden_dim: usize,
    pub moe_intermediate_dim: usize,
    pub n_experts: usize,
    pub num_shared_experts: usize,
}

/// Loads one KDA-attention layer's weights from a real Kimi K3 GGUF
/// file. Real tensor names/shapes confirmed against
/// `ggml-org/llama.cpp#26185`'s `src/models/kimi-k3.cpp`
/// (`create_tensor_qkv` for `attn_q`/`attn_k`/`attn_v`, then
/// `ssm_conv1d_{q,k,v}`, `ssm_f_a`/`ssm_f_b`, `ssm_beta`, `ssm_a`,
/// `ssm_dt.bias`, `ssm_g`, `ssm_norm`, `attn_output`).
fn load_kda_attn(
    file: &impl TensorSource,
    layer_idx: usize,
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
) -> Result<KdaAttnWeights, LoadError> {
    let l = layer_idx;
    let projection_size = num_heads * head_dim;

    // Real on-disk conv weight shape is {d_conv, 1, d_inner, 1} (ggml
    // ne[] order, reversed from numpy) -- already exactly
    // [projection_size, kernel_size] flattened once the two size-1 axes
    // are dropped, same as the safetensors path's raw-bytes read.
    let q_conv_weight = load_f32_vec(file, &format!("blk.{l}.ssm_conv1d_q.weight"))?;
    let k_conv_weight = load_f32_vec(file, &format!("blk.{l}.ssm_conv1d_k.weight"))?;
    let v_conv_weight = load_f32_vec(file, &format!("blk.{l}.ssm_conv1d_v.weight"))?;
    assert!(
        q_conv_weight.len().is_multiple_of(projection_size),
        "blk.{l}.ssm_conv1d_q.weight: {} elements not a multiple of projection_size={projection_size}",
        q_conv_weight.len()
    );

    // Real GGUF ssm_a is plain [n_head] (unlike the safetensors
    // checkpoint's padded-to-128 A_log -- see kimi_loader's module doc
    // comment): conversion writes -exp(A_log) directly (kimi_k3.py's
    // `name.endswith(".A_log")` branch), so no further transform needed
    // here, just read it as-is.
    let a_log = load_f32_vec(file, &format!("blk.{l}.ssm_a"))?;
    assert_eq!(
        a_log.len(),
        num_heads,
        "blk.{l}.ssm_a: {} elements, expected num_heads={num_heads}",
        a_log.len()
    );

    Ok(KdaAttnWeights {
        q_proj: load_weight_matrix(file, &format!("blk.{l}.attn_q.weight"))?,
        k_proj: load_weight_matrix(file, &format!("blk.{l}.attn_k.weight"))?,
        v_proj: load_weight_matrix(file, &format!("blk.{l}.attn_v.weight"))?,
        q_conv_weight,
        k_conv_weight,
        v_conv_weight,
        a_log,
        f_a_proj: load_weight_matrix(file, &format!("blk.{l}.ssm_f_a.weight"))?,
        f_b_proj: load_weight_matrix(file, &format!("blk.{l}.ssm_f_b.weight"))?,
        // Real tensor name is `ssm_dt.bias` (conversion renames the HF
        // `dt_bias` -> `dt_proj.bias`, which `tn(LLM_TENSOR_SSM_DT,
        // "bias", i)` maps to `blk.{bid}.ssm_dt.bias`).
        dt_bias: load_f32_vec(file, &format!("blk.{l}.ssm_dt.bias"))?,
        b_proj: load_weight_matrix(file, &format!("blk.{l}.ssm_beta.weight"))?,
        g_proj: load_weight_matrix(file, &format!("blk.{l}.ssm_g.weight"))?,
        o_norm_weight: load_f32_vec(file, &format!("blk.{l}.ssm_norm.weight"))?,
        o_proj: {
            // Real shape {d_inner, n_embd} (ggml ne[]) -> row-major
            // [hidden_dim, projection_size] once `load_weight_matrix`
            // reverses it -- checked against both hparams explicitly,
            // since a mismatch here means this checkpoint's real KDA
            // head/hidden dims disagree with what `hp` was built with.
            let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;
            assert_eq!(
                o_proj.rows(),
                hidden_dim,
                "blk.{l}.attn_output.weight row count"
            );
            assert_eq!(
                o_proj.cols(),
                projection_size,
                "blk.{l}.attn_output.weight col count"
            );
            o_proj
        },
    })
}

/// Loads one Gated-MLA-attention layer's weights from a real Kimi K3
/// GGUF file. Real tensor names/shapes confirmed against
/// `src/models/kimi-k3.cpp` (`attn_q_a`/`attn_q_b` when
/// `attn_q_a_norm` exists -- always true for K3's real
/// `q_lora_rank`=1536 -- `attn_kv_a_mqa`, `attn_k_b`/`attn_v_b` (the
/// `TENSOR_NOT_REQUIRED` combined `attn_kv_b` alternative isn't used by
/// this checkpoint's real conversion, which always splits it -- see
/// `conversion/kimi_k3.py`'s `kv_b_proj` split), `attn_gate`,
/// `attn_output`).
#[allow(clippy::too_many_arguments)]
fn load_mla_attn(
    file: &impl TensorSource,
    layer_idx: usize,
    num_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    hidden_dim: usize,
) -> Result<MlaAttnWeights, LoadError> {
    let l = layer_idx;
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    // Real on-disk k_b/v_b shapes are {qk_nope_head_dim, kv_lora_rank,
    // n_head} / {kv_lora_rank, n_embd_head_v, n_head} (3D, per-head).
    // This loader reads the combined `attn_kv_b` (the shape
    // `kimi_loader::load_mla_attn` builds from the safetensors
    // checkpoint's single combined `kv_b_proj`) into `MlaKvB::Combined`;
    // `MlaKvB::Split` exists now (the DeepSeek loader fills it from the
    // 3D pair and `mla_forward_token` runs the absorbed form on it), and
    // taking it here is a change to make against a Kimi golden, not by
    // analogy. `find_info` fails loudly if `attn_kv_b` is absent rather
    // than loading transposed or mismatched data.
    let q_a_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_a.weight"))?;
    assert_eq!(
        q_a_proj.rows(),
        q_lora_rank,
        "blk.{l}.attn_q_a.weight row count"
    );
    assert_eq!(
        q_a_proj.cols(),
        hidden_dim,
        "blk.{l}.attn_q_a.weight col count"
    );

    let q_b_proj = load_weight_matrix(file, &format!("blk.{l}.attn_q_b.weight"))?;
    assert_eq!(
        q_b_proj.rows(),
        num_heads * q_head_dim,
        "blk.{l}.attn_q_b.weight row count"
    );
    assert_eq!(
        q_b_proj.cols(),
        q_lora_rank,
        "blk.{l}.attn_q_b.weight col count"
    );

    let kv_a_proj_with_mqa = load_weight_matrix(file, &format!("blk.{l}.attn_kv_a_mqa.weight"))?;
    assert_eq!(
        kv_a_proj_with_mqa.rows(),
        kv_lora_rank + qk_rope_head_dim,
        "blk.{l}.attn_kv_a_mqa.weight row count"
    );

    let o_proj = load_weight_matrix(file, &format!("blk.{l}.attn_output.weight"))?;
    assert_eq!(
        o_proj.rows(),
        hidden_dim,
        "blk.{l}.attn_output.weight row count"
    );
    assert_eq!(
        o_proj.cols(),
        num_heads * v_head_dim,
        "blk.{l}.attn_output.weight col count"
    );

    Ok(MlaAttnWeights {
        q: MlaQProj::LowRank {
            a: q_a_proj,
            norm: load_f32_vec(file, &format!("blk.{l}.attn_q_a_norm.weight"))?,
            b: q_b_proj,
        },
        kv_a_proj_with_mqa,
        kv_a_layernorm: load_f32_vec(file, &format!("blk.{l}.attn_kv_a_norm.weight"))?,
        kv_b: MlaKvB::Combined(load_weight_matrix(
            file,
            &format!("blk.{l}.attn_kv_b.weight"),
        )?),
        o_proj,
        g_proj: Some(load_weight_matrix(
            file,
            &format!("blk.{l}.attn_gate.weight"),
        )?),
    })
}

/// Loads the dense leading layer's feed-forward block from a real Kimi
/// K3 GGUF file (real tensor names `blk.{bid}.ffn_{gate,down,up}`).
fn load_dense_mlp(
    file: &impl TensorSource,
    layer_idx: usize,
) -> Result<DenseMlpWeights, LoadError> {
    let l = layer_idx;
    Ok(DenseMlpWeights {
        gate_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_gate.weight"))?,
        up_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_up.weight"))?,
        down_proj: load_weight_matrix(file, &format!("blk.{l}.ffn_down.weight"))?,
    })
}

/// Loads one full latent-MoE layer from a real Kimi K3 GGUF file. Real
/// tensor names confirmed against `src/models/kimi-k3.cpp`:
/// `ffn_gate_inp` (router), `exp_probs_b` (aux-loss-free bias, note:
/// no `ffn_` prefix -- confirmed against `gguf-py/gguf/constants.py`'s
/// `MODEL_TENSOR_NAMES[FFN_EXP_PROBS_B]` = `"blk.{bid}.exp_probs_b"`),
/// `ffn_{gate,down,up}_exps` (packed 3D routed experts, real shape
/// `[n_embd_latent, n_ff_exp, n_expert]` -- latent-space, not
/// `hidden_dim`), `ffn_routed_{down,up,norm}` (the latent
/// down/up-projection this architecture adds), `ffn_{gate,down,up}_shexp`
/// (shared experts, full `hidden_dim`).
fn load_latent_moe(
    file: &impl TensorSource,
    layer_idx: usize,
    hidden_dim: usize,
    moe_hidden_dim: usize,
    n_experts: usize,
) -> Result<KimiLatentMoeWeights, LoadError> {
    let l = layer_idx;

    let gate_exps = split_expert_tensor(file, &format!("blk.{l}.ffn_gate_exps.weight"), n_experts)?;
    let down_exps = split_expert_tensor(file, &format!("blk.{l}.ffn_down_exps.weight"), n_experts)?;
    let up_exps = split_expert_tensor(file, &format!("blk.{l}.ffn_up_exps.weight"), n_experts)?;
    for w in gate_exps.iter().chain(up_exps.iter()) {
        assert_eq!(
            w.cols(),
            moe_hidden_dim,
            "blk.{l}.ffn_{{gate,up}}_exps.weight col count"
        );
    }
    for w in down_exps.iter() {
        assert_eq!(
            w.rows(),
            moe_hidden_dim,
            "blk.{l}.ffn_down_exps.weight row count"
        );
    }
    let experts: Vec<KimiExpertWeights> = gate_exps
        .into_iter()
        .zip(down_exps)
        .zip(up_exps)
        .map(|((w1, w2), w3)| KimiExpertWeights { w1, w2, w3 })
        .collect();

    // Real raw ne[] is {n_embd, n_embd_latent} for down / {n_embd_latent,
    // n_embd} for up (`src/models/kimi-k3.cpp`'s real `create_tensor`
    // calls) -- reversed by `load_weight_matrix` into row-major
    // [moe_hidden_dim, hidden_dim] / [hidden_dim, moe_hidden_dim],
    // matching `KimiLatentMoeWeights::down_proj`/`up_proj`'s own doc
    // comment exactly (down projects hidden->latent, up projects
    // latent->hidden).
    let down_proj = load_weight_matrix(file, &format!("blk.{l}.ffn_routed_down.weight"))?;
    assert_eq!(
        down_proj.rows(),
        moe_hidden_dim,
        "blk.{l}.ffn_routed_down.weight row count"
    );
    assert_eq!(
        down_proj.cols(),
        hidden_dim,
        "blk.{l}.ffn_routed_down.weight col count"
    );
    let up_proj = load_weight_matrix(file, &format!("blk.{l}.ffn_routed_up.weight"))?;
    assert_eq!(
        up_proj.rows(),
        hidden_dim,
        "blk.{l}.ffn_routed_up.weight row count"
    );
    assert_eq!(
        up_proj.cols(),
        moe_hidden_dim,
        "blk.{l}.ffn_routed_up.weight col count"
    );

    Ok(KimiLatentMoeWeights {
        router_weight: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_inp.weight"))?,
        e_score_correction_bias: load_f32_vec(file, &format!("blk.{l}.exp_probs_b.bias"))?,
        down_proj,
        up_proj,
        routed_expert_norm_weight: Some(load_f32_vec(
            file,
            &format!("blk.{l}.ffn_routed_norm.weight"),
        )?),
        experts: KimiExpertBacking::Resident(experts),
        shared_expert: KimiExpertWeights {
            w1: load_weight_matrix(file, &format!("blk.{l}.ffn_gate_shexp.weight"))?,
            w2: load_weight_matrix(file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
            w3: load_weight_matrix(file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
        },
    })
}

/// Reads a fused block-residual score tensor (`blk.{bid}.attn_res_score`/
/// `ffn_res_score`/`output_res_score` -- see module doc comment) into
/// `(norm_weight, proj_weight)` such that `norm_weight * proj_weight`
/// (elementwise) reproduces the real fused score exactly: the fused
/// tensor goes into `norm_weight` unchanged, `proj_weight` is a
/// constant all-ones vector. Lets this loader populate
/// `KimiDecoderLayerWeights`'s existing two-factor fields (and
/// `kimi_decoder`'s existing `apply_attn_res` call sites) without
/// changing either.
fn load_fused_res_score(
    file: &impl TensorSource,
    name: &str,
    hidden_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), LoadError> {
    let fused = load_f32_vec(file, name)?;
    assert_eq!(
        fused.len(),
        hidden_dim,
        "'{name}' has {} elements, expected hidden_dim={hidden_dim}",
        fused.len()
    );
    Ok((fused, vec![1.0; hidden_dim]))
}

/// Loads any one real Kimi K3 layer (KDA or Gated-MLA attention, dense
/// or latent-MoE FFN) from a real GGUF file, dispatching on `kind`/
/// `is_dense` -- the GGUF counterpart of `kimi_loader::load_kimi_layer`.
pub fn load_kimi_gguf_layer(
    file: &impl TensorSource,
    hp: &KimiGgufHparams,
    kind: LayerAttentionKind,
    is_dense: bool,
    layer_idx: usize,
) -> Result<KimiDecoderLayerWeights, LoadError> {
    let l = layer_idx;
    let input_layernorm_weight = load_f32_vec(file, &format!("blk.{l}.attn_norm.weight"))?;
    let post_attention_layernorm_weight = load_f32_vec(file, &format!("blk.{l}.ffn_norm.weight"))?;

    let (self_attention_res_norm_weight, self_attention_res_proj_weight) = load_fused_res_score(
        file,
        &format!("blk.{l}.attn_res_score.weight"),
        hp.hidden_dim,
    )?;
    let (mlp_res_norm_weight, mlp_res_proj_weight) = load_fused_res_score(
        file,
        &format!("blk.{l}.ffn_res_score.weight"),
        hp.hidden_dim,
    )?;

    let attn = match kind {
        LayerAttentionKind::KimiKda => KimiLayerAttention::Kda(Box::new(load_kda_attn(
            file,
            l,
            hp.kda_num_heads,
            hp.kda_head_dim,
            hp.hidden_dim,
        )?)),
        LayerAttentionKind::KimiMla => KimiLayerAttention::Mla(Box::new(load_mla_attn(
            file,
            l,
            hp.mla_num_heads,
            hp.mla_q_lora_rank,
            hp.mla_kv_lora_rank,
            hp.mla_qk_nope_head_dim,
            hp.mla_qk_rope_head_dim,
            hp.mla_v_head_dim,
            hp.hidden_dim,
        )?)),
        LayerAttentionKind::Gqa => {
            panic!("load_kimi_gguf_layer is only for KimiHybrid (KDA/Gated-MLA) layers")
        }
    };

    let ffn = if is_dense {
        KimiLayerFfn::Dense(Box::new(load_dense_mlp(file, l)?))
    } else {
        KimiLayerFfn::Moe(Box::new(load_latent_moe(
            file,
            l,
            hp.hidden_dim,
            hp.moe_hidden_dim,
            hp.n_experts,
        )?))
    };

    Ok(KimiDecoderLayerWeights {
        input_layernorm_weight,
        attn,
        post_attention_layernorm_weight,
        ffn,
        self_attention_res_norm_weight,
        self_attention_res_proj_weight,
        mlp_res_norm_weight,
        mlp_res_proj_weight,
    })
}

/// Loads a complete `KimiDecoderWeights` from a real Kimi K3 GGUF file
/// -- the GGUF counterpart of `kimi_loader::load_kimi_checkpoint`. Real
/// top-level tensor names: `token_embd.weight`, `output.weight`,
/// `output_norm.weight`, `output_res_score.weight` (confirmed against
/// `src/models/kimi-k3.cpp`'s real
/// `create_tensor(tn(LLM_TENSOR_OUTPUT_RES_SCORE, "weight"), ...)` call
/// -- `gguf-py/gguf/constants.py`'s `MODEL_TENSOR_NAMES` table gives the
/// bare base name `"output_res_score"`, but `tn(...)`'s `"weight"`
/// argument still appends the usual `.weight` suffix, same as every
/// other tensor).
pub fn load_kimi_gguf_checkpoint(
    file: &impl TensorSource,
    model_cfg: &crate::config::ModelConfig,
    hp: &KimiGgufHparams,
) -> Result<KimiDecoderWeights, LoadError> {
    let embedding_matrix = load_weight_matrix(file, "token_embd.weight")?;
    let embedding = match embedding_matrix {
        WeightMatrix::F32(t) => t,
        _ => {
            // Kimi K3's real embedding stays BF16 (never routed-expert
            // MXFP4), which `load_weight_matrix` already widens to F32
            // above -- a quantized embedding here would mean this
            // checkpoint's conversion diverged from what's been
            // confirmed, so fail loudly rather than silently drop
            // precision or panic deeper in `KimiDecoderWeights::embedding`'s
            // consumers.
            return Err(LoadError::UnsupportedDtype(
                "token_embd.weight (expected F32/BF16 for Kimi K3)".to_string(),
                GgmlType::Other(0),
            ));
        }
    };

    let (output_attn_res_norm_weight, output_attn_res_proj_weight) =
        load_fused_res_score(file, "output_res_score.weight", hp.hidden_dim)?;

    let mut layers = Vec::with_capacity(model_cfg.n_layers);
    for l in 0..model_cfg.n_layers {
        layers.push(load_kimi_gguf_layer(
            file,
            hp,
            model_cfg.layer_attention_kind(l),
            model_cfg.layer_is_dense(l),
            l,
        )?);
    }

    Ok(KimiDecoderWeights {
        embedding,
        layers,
        output_attn_res_norm_weight,
        output_attn_res_proj_weight,
        final_norm_weight: load_f32_vec(file, "output_norm.weight")?,
        output_head: load_weight_matrix(file, "output.weight")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let bits = v.to_bits();
            out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        out
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// One tensor to write into a synthetic GGUF fixture: real name,
    /// ggml dtype tag (0 = F32, 30 = BF16), semantic `[rows, cols, ...]`
    /// shape (this builder reverses it to ggml's `ne[]` fastest-first
    /// convention itself, matching every real GGUF writer/reader in
    /// this codebase), and raw little-endian bytes.
    struct FixtureTensor {
        name: String,
        dtype_tag: u32,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn bf16_tensor(name: impl Into<String>, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        FixtureTensor {
            name: name.into(),
            dtype_tag: 30,
            shape,
            bytes: bf16_bytes(&values),
        }
    }

    fn f32_tensor(name: impl Into<String>, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        FixtureTensor {
            name: name.into(),
            dtype_tag: 0,
            shape,
            bytes: f32_bytes(&values),
        }
    }

    /// Builds a real, parseable on-disk GGUF file from a flat list of
    /// tensors -- a general-purpose fixture writer (only F32/BF16
    /// dtypes, matching what this loader actually needs to test; no
    /// quantized-tensor support, unlike `loader.rs`'s per-quant-kind
    /// single-tensor builders, since none of the K3 hparams under test
    /// here are quantized-format-sensitive).
    fn build_gguf(arch: &str, tensors: &[FixtureTensor]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count: just general.architecture

        let write_string = |buf: &mut Vec<u8>, s: &str| {
            buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
        };
        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
        write_string(&mut buf, arch);

        // Tensor info block: name, n_dims, ne[] (reversed from the
        // semantic shape passed in), dtype tag, byte offset (relative
        // to the aligned data section, back-to-back with 32-byte
        // padding after each tensor -- matches GgufFile::parse's
        // `data_start` + per-tensor `offset` read).
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            write_string(&mut buf, &t.name);
            buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
            for &d in t.shape.iter().rev() {
                buf.write_u64::<LittleEndian>(d).unwrap();
            }
            buf.write_u32::<LittleEndian>(t.dtype_tag).unwrap();
            offsets.push(offset);
            buf.write_u64::<LittleEndian>(offset).unwrap();
            let padded = t.bytes.len().div_ceil(32) * 32;
            offset += padded as u64;
        }

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            let want_len = data_start + off as usize;
            while buf.len() < want_len {
                buf.push(0);
            }
            buf.extend_from_slice(&t.bytes);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    struct Dims {
        hidden_dim: usize,
        kda_num_heads: usize,
        kda_head_dim: usize,
        mla_num_heads: usize,
        mla_q_lora_rank: usize,
        mla_kv_lora_rank: usize,
        mla_qk_nope_head_dim: usize,
        mla_qk_rope_head_dim: usize,
        mla_v_head_dim: usize,
        dense_intermediate_dim: usize,
        moe_hidden_dim: usize,
        moe_intermediate_dim: usize,
        n_experts: usize,
        num_shared_experts: usize,
    }

    /// Appends one real layer's full tensor set (KDA+dense, or
    /// MLA+MoE), matching every real tensor name/shape this loader's
    /// `load_kimi_gguf_layer` expects.
    fn push_layer_tensors(
        tensors: &mut Vec<FixtureTensor>,
        l: usize,
        kind: LayerAttentionKind,
        is_dense: bool,
        d: &Dims,
    ) {
        let h = d.hidden_dim;
        tensors.push(bf16_tensor(
            format!("blk.{l}.attn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(bf16_tensor(
            format!("blk.{l}.ffn_norm.weight"),
            vec![h as u64],
            vec![1.0; h],
        ));
        tensors.push(bf16_tensor(
            format!("blk.{l}.attn_res_score.weight"),
            vec![h as u64],
            vec![0.01; h],
        ));
        tensors.push(bf16_tensor(
            format!("blk.{l}.ffn_res_score.weight"),
            vec![h as u64],
            vec![0.01; h],
        ));

        match kind {
            LayerAttentionKind::KimiKda => {
                let proj = d.kda_num_heads * d.kda_head_dim;
                for name in ["attn_q", "attn_k", "attn_v"] {
                    tensors.push(bf16_tensor(
                        format!("blk.{l}.{name}.weight"),
                        vec![proj as u64, h as u64],
                        vec![0.02; proj * h],
                    ));
                }
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_conv1d_q.weight"),
                    vec![proj as u64, 4],
                    vec![0.1; proj * 4],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_conv1d_k.weight"),
                    vec![proj as u64, 4],
                    vec![0.1; proj * 4],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_conv1d_v.weight"),
                    vec![proj as u64, 4],
                    vec![0.1; proj * 4],
                ));
                tensors.push(f32_tensor(
                    format!("blk.{l}.ssm_a"),
                    vec![d.kda_num_heads as u64],
                    vec![-0.5; d.kda_num_heads],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_f_a.weight"),
                    vec![d.kda_head_dim as u64, h as u64],
                    vec![0.02; d.kda_head_dim * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_f_b.weight"),
                    vec![proj as u64, d.kda_head_dim as u64],
                    vec![0.02; proj * d.kda_head_dim],
                ));
                tensors.push(f32_tensor(
                    format!("blk.{l}.ssm_dt.bias"),
                    vec![proj as u64],
                    vec![0.0; proj],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_beta.weight"),
                    vec![d.kda_num_heads as u64, h as u64],
                    vec![0.02; d.kda_num_heads * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_g.weight"),
                    vec![proj as u64, h as u64],
                    vec![0.02; proj * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.ssm_norm.weight"),
                    vec![d.kda_head_dim as u64],
                    vec![1.0; d.kda_head_dim],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_output.weight"),
                    vec![h as u64, proj as u64],
                    vec![0.02; h * proj],
                ));
            }
            LayerAttentionKind::KimiMla => {
                let q_head_dim = d.mla_qk_nope_head_dim + d.mla_qk_rope_head_dim;
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_q_a.weight"),
                    vec![d.mla_q_lora_rank as u64, h as u64],
                    vec![0.02; d.mla_q_lora_rank * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_q_a_norm.weight"),
                    vec![d.mla_q_lora_rank as u64],
                    vec![1.0; d.mla_q_lora_rank],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_q_b.weight"),
                    vec![
                        (d.mla_num_heads * q_head_dim) as u64,
                        d.mla_q_lora_rank as u64,
                    ],
                    vec![0.02; d.mla_num_heads * q_head_dim * d.mla_q_lora_rank],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_kv_a_mqa.weight"),
                    vec![
                        (d.mla_kv_lora_rank + d.mla_qk_rope_head_dim) as u64,
                        h as u64,
                    ],
                    vec![0.02; (d.mla_kv_lora_rank + d.mla_qk_rope_head_dim) * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_kv_a_norm.weight"),
                    vec![d.mla_kv_lora_rank as u64],
                    vec![1.0; d.mla_kv_lora_rank],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_kv_b.weight"),
                    vec![
                        (d.mla_num_heads * (d.mla_qk_nope_head_dim + d.mla_v_head_dim)) as u64,
                        d.mla_kv_lora_rank as u64,
                    ],
                    vec![
                        0.02;
                        d.mla_num_heads
                            * (d.mla_qk_nope_head_dim + d.mla_v_head_dim)
                            * d.mla_kv_lora_rank
                    ],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_gate.weight"),
                    vec![(d.mla_num_heads * d.mla_v_head_dim) as u64, h as u64],
                    vec![0.02; d.mla_num_heads * d.mla_v_head_dim * h],
                ));
                tensors.push(bf16_tensor(
                    format!("blk.{l}.attn_output.weight"),
                    vec![h as u64, (d.mla_num_heads * d.mla_v_head_dim) as u64],
                    vec![0.02; h * d.mla_num_heads * d.mla_v_head_dim],
                ));
            }
            LayerAttentionKind::Gqa => unreachable!("fixture only builds Kimi layers"),
        }

        if is_dense {
            for name in ["ffn_gate", "ffn_up"] {
                tensors.push(bf16_tensor(
                    format!("blk.{l}.{name}.weight"),
                    vec![d.dense_intermediate_dim as u64, h as u64],
                    vec![0.02; d.dense_intermediate_dim * h],
                ));
            }
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_down.weight"),
                vec![h as u64, d.dense_intermediate_dim as u64],
                vec![0.02; h * d.dense_intermediate_dim],
            ));
        } else {
            let m = d.moe_hidden_dim;
            let ff = d.moe_intermediate_dim;
            let n = d.n_experts;
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_gate_inp.weight"),
                vec![n as u64, h as u64],
                vec![0.02; n * h],
            ));
            tensors.push(f32_tensor(
                format!("blk.{l}.exp_probs_b.bias"),
                vec![n as u64],
                vec![0.0; n],
            ));
            // `split_expert_tensor` (unlike `load_weight_matrix`) reads
            // `info.shape` directly with no reversal -- the real raw
            // on-disk ne[] IS [n_embd_latent, n_ff_exp, n_expert] for
            // gate/up (real `create_tensor` call literally writes that
            // ne[] order) and [n_ff_exp, n_embd_latent, n_expert] for
            // down. This builder's `bf16_tensor` reverses its `shape`
            // arg once to produce the written raw ne[], so the argument
            // here must be the REVERSE of the real ne[] shown above.
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_gate_exps.weight"),
                vec![n as u64, ff as u64, m as u64],
                vec![0.02; m * ff * n],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_down_exps.weight"),
                vec![n as u64, m as u64, ff as u64],
                vec![0.02; ff * m * n],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_up_exps.weight"),
                vec![n as u64, ff as u64, m as u64],
                vec![0.02; m * ff * n],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_routed_down.weight"),
                vec![m as u64, h as u64],
                vec![0.02; m * h],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_routed_up.weight"),
                vec![h as u64, m as u64],
                vec![0.02; h * m],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_routed_norm.weight"),
                vec![m as u64],
                vec![1.0; m],
            ));
            let shexp_dim = ff * d.num_shared_experts;
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_gate_shexp.weight"),
                vec![shexp_dim as u64, h as u64],
                vec![0.02; shexp_dim * h],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_down_shexp.weight"),
                vec![h as u64, shexp_dim as u64],
                vec![0.02; h * shexp_dim],
            ));
            tensors.push(bf16_tensor(
                format!("blk.{l}.ffn_up_shexp.weight"),
                vec![shexp_dim as u64, h as u64],
                vec![0.02; shexp_dim * h],
            ));
        }
    }

    fn build_synthetic_gguf_checkpoint() -> (
        std::path::PathBuf,
        frink_gguf::GgufFile,
        crate::config::ModelConfig,
        KimiGgufHparams,
    ) {
        let d = Dims {
            hidden_dim: 8,
            kda_num_heads: 2,
            kda_head_dim: 3,
            mla_num_heads: 1,
            mla_q_lora_rank: 4,
            mla_kv_lora_rank: 4,
            mla_qk_nope_head_dim: 2,
            mla_qk_rope_head_dim: 2,
            mla_v_head_dim: 2,
            dense_intermediate_dim: 5,
            moe_hidden_dim: 6,
            moe_intermediate_dim: 4,
            n_experts: 2,
            num_shared_experts: 1,
        };
        let vocab_size = 6;

        // 2 real layers: 0 = dense+KDA (matches Kimi K3's real layer
        // 0), 1 = MoE+MLA -- covering every real attention/FFN
        // combination `load_kimi_gguf_layer` must dispatch correctly.
        let model_cfg = crate::config::ModelConfig {
            // Kimi's stack has no per-layer RoPE gate; llama.cpp writes
            // no `use_rope` for it (`crate::rope_layers`).
            rope_layers: crate::rope_layers::RopeLayers::All,
            layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
            name: "synthetic-kimi-gguf-test",
            n_layers: 2,
            n_mtp_blocks: 0,
            hidden_dim: d.hidden_dim,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
            v_head_dim: None,
            vocab_size,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            post_norm_eps: 1e-5,
            sliding_window: None,
            moe: frink_moe::MoeLayerConfig {
                expert_weights_scale: 1.0,
                routed_weight_before_ffn: false,
                n_experts: d.n_experts,
                n_experts_active: d.n_experts,
                n_shared_experts: d.num_shared_experts,
                hidden_dim: d.hidden_dim,
                expert_ffn_dim: d.moe_intermediate_dim,
                gating: frink_moe::GatingFunction::Sigmoid,
                norm_topk_prob: true,
                expert_group_count: None,
                expert_group_used_count: None,
            },
            n_dense_leading_layers: 1,
            moe_interleave_step: None,
            norm_function: crate::norm::NormFunction::Rms,
            attention: crate::config::AttentionKind::KimiHybrid(
                crate::config::KimiHybridAttention {
                    kda_layers: vec![1],
                    full_attn_layers: vec![2],
                    mla: crate::config::MlaConfig {
                        num_heads: d.mla_num_heads,
                        q_lora_rank: d.mla_q_lora_rank,
                        kv_lora_rank: d.mla_kv_lora_rank,
                        qk_nope_head_dim: d.mla_qk_nope_head_dim,
                        qk_rope_head_dim: d.mla_qk_rope_head_dim,
                        v_head_dim: d.mla_v_head_dim,
                        use_output_gate: true,
                        rope: None,
                    },
                    kda: crate::config::KdaConfig {
                        num_heads: d.kda_num_heads,
                        head_dim: d.kda_head_dim,
                        short_conv_kernel_size: 4,
                        gate_lower_bound: -5.0,
                        use_full_rank_gate: true,
                    },
                },
            ),
            rope_freqs: None,
            rope_attn_factor: 1.0,
            rope_dim: None,
            rope_dim_swa: None,
            rope_freqs_long: None,
            rope_freqs_short: None,
            rope_orig_ctx: None,
            rope_layout: crate::config::RopeLayout::Neox,
            qk_norm_style: crate::capability::QkNormStyle::WholeVector,
            swa_layers: crate::swa_layers::SwaLayers::All,

            attn_logit_softcap: None,
            final_logit_softcap: None,
            embedding_scale: None,
            residual_scale: None,
            normed_residual_scale: None,
            clamp_kqv: None,
            attn_temperature: None,
            router_input: crate::router_input::RouterInput::NormedFfnInput,
            block_sub_norms: false,
            parallel_residual: false,
            learned_positions: false,
            attn_value_scale: None,
            alibi_max_bias: None,
            layer_loops: None,
            skip_stream: false,
            parallel_ssm: false,
            swa_chunked: false,
            weightless_qk_norm: false,
            logit_multiplier: None,
            attention_scale: None,
            rope_theta_swa: None,
            ffn_activation: crate::config::FfnActivation::Swiglu,
            best_effort_fields: &["synthetic test config, not a real preset"],
        };

        let mut tensors: Vec<FixtureTensor> = Vec::new();
        push_layer_tensors(&mut tensors, 0, LayerAttentionKind::KimiKda, true, &d);
        push_layer_tensors(&mut tensors, 1, LayerAttentionKind::KimiMla, false, &d);

        tensors.push(bf16_tensor(
            "token_embd.weight",
            vec![vocab_size as u64, d.hidden_dim as u64],
            vec![0.02; vocab_size * d.hidden_dim],
        ));
        tensors.push(bf16_tensor(
            "output.weight",
            vec![vocab_size as u64, d.hidden_dim as u64],
            vec![0.02; vocab_size * d.hidden_dim],
        ));
        tensors.push(bf16_tensor(
            "output_norm.weight",
            vec![d.hidden_dim as u64],
            vec![1.0; d.hidden_dim],
        ));
        tensors.push(bf16_tensor(
            "output_res_score.weight",
            vec![d.hidden_dim as u64],
            vec![0.01; d.hidden_dim],
        ));

        let bytes = build_gguf("kimi-k3", &tensors);
        let path =
            std::env::temp_dir().join(format!("frink_kimi_gguf_test_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = frink_gguf::GgufFile::open(&path).expect("synthetic GGUF must parse");

        let hp = KimiGgufHparams {
            hidden_dim: d.hidden_dim,
            kda_num_heads: d.kda_num_heads,
            kda_head_dim: d.kda_head_dim,
            short_conv_kernel_size: 4,
            mla_num_heads: d.mla_num_heads,
            mla_q_lora_rank: d.mla_q_lora_rank,
            mla_kv_lora_rank: d.mla_kv_lora_rank,
            mla_qk_nope_head_dim: d.mla_qk_nope_head_dim,
            mla_qk_rope_head_dim: d.mla_qk_rope_head_dim,
            mla_v_head_dim: d.mla_v_head_dim,
            dense_intermediate_dim: d.dense_intermediate_dim,
            moe_hidden_dim: d.moe_hidden_dim,
            moe_intermediate_dim: d.moe_intermediate_dim,
            n_experts: d.n_experts,
            num_shared_experts: d.num_shared_experts,
        };

        (path, file, model_cfg, hp)
    }

    #[test]
    fn load_kimi_gguf_checkpoint_assembles_every_real_layer_kind_and_runs_end_to_end() {
        let (path, file, model_cfg, hp) = build_synthetic_gguf_checkpoint();
        let weights = load_kimi_gguf_checkpoint(&file, &model_cfg, &hp)
            .expect("must assemble a complete synthetic GGUF checkpoint");
        std::fs::remove_file(&path).ok();

        assert_eq!(weights.layers.len(), 2);
        assert!(matches!(weights.layers[0].ffn, KimiLayerFfn::Dense(_)));
        assert!(matches!(weights.layers[0].attn, KimiLayerAttention::Kda(_)));
        assert!(matches!(weights.layers[1].ffn, KimiLayerFfn::Moe(_)));
        assert!(matches!(weights.layers[1].attn, KimiLayerAttention::Mla(_)));
        assert_eq!(weights.embedding.rows(), model_cfg.vocab_size);
        assert_eq!(weights.embedding.cols(), hp.hidden_dim);
        assert_eq!(weights.output_head.rows(), model_cfg.vocab_size);
        assert_eq!(weights.final_norm_weight.len(), hp.hidden_dim);

        // Run a real forward pass through the fully-assembled checkpoint
        // to confirm every piece composes correctly end to end, not
        // just that each layer loads (mirrors
        // `kimi_loader`'s own `load_kimi_checkpoint_assembles_every_real_layer_kind`).
        let mla_cfg = crate::config::MlaConfig {
            num_heads: hp.mla_num_heads,
            q_lora_rank: hp.mla_q_lora_rank,
            kv_lora_rank: hp.mla_kv_lora_rank,
            qk_nope_head_dim: hp.mla_qk_nope_head_dim,
            qk_rope_head_dim: hp.mla_qk_rope_head_dim,
            v_head_dim: hp.mla_v_head_dim,
            use_output_gate: true,
            rope: None,
        };
        let kda_cfg = crate::config::KdaConfig {
            num_heads: hp.kda_num_heads,
            head_dim: hp.kda_head_dim,
            short_conv_kernel_size: hp.short_conv_kernel_size,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };
        let decoder_cfg = crate::kimi_decoder::KimiDecoderConfig {
            attn_res_block_size: 12,
            rms_norm_eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            moe: crate::latent_moe::KimiMoeConfig {
                n_experts_active: hp.n_experts,
                moe_renormalize: true,
                routed_scaling_factor: 1.0,
                situ_beta: 4.0,
                situ_linear_beta: 25.0,
                rms_norm_eps: 1e-5,
            },
        };
        let mut state = crate::kimi_decoder::KimiDecodeState::new(&weights, &kda_cfg);
        let logits = crate::kimi_decoder::kimi_forward_token(
            &weights,
            &decoder_cfg,
            &mla_cfg,
            &kda_cfg,
            0,
            &mut state,
        );
        assert_eq!(logits.len(), model_cfg.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
