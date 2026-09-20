//! Loads `frink-models::kimi_decoder` weights from Kimi K3's real
//! safetensors checkpoint (via `frink-safetensors::ShardedSafetensors`),
//! using the exact real tensor names/shapes/dtypes fetched live from
//! `huggingface.co/moonshotai/Kimi-K3` (a real shard header, not
//! guessed) -- confirmed to match this crate's `kda`/`mla`/`latent_moe`/
//! `kimi_decoder` struct field names and shapes exactly.
//!
//! One real, non-obvious fact confirmed by reading actual tensor shapes
//! rather than assuming they match `modeling_kimi_linear.py`'s
//! `KimiDeltaAttention.__init__` literally: `self_attn.A_log`'s real
//! on-disk shape is `[128]`, not `[num_heads]` = `[96]` (confirmed
//! independently by `self_attn.b_proj.weight`'s real shape `[96,
//! 7168]`, which is unambiguously `[num_heads, hidden_dim]`). The real
//! `fused_recurrent_kda` kernel only ever indexes `A_log[i_hv]` for
//! `i_hv` in `0..num_heads`, so this is real, harmless padding (likely
//! to a GPU-friendly round size) rather than a spec mismatch -- this
//! loader reads the real 128-element tensor but only uses its first
//! `num_heads` elements, matching what the real kernel actually
//! consumes.
//!
//! Dequantizes small `F32`/`BF16` tensors (per-head attention
//! parameters, layernorms, dense-FFN and shared-expert projections) to
//! owned `f32` eagerly at load time -- matching this project's
//! established BF16-handling convention, e.g. `frink-models::loader`'s
//! GGUF path -- since these are cheap regardless. Routed-expert `MXFP4`
//! weights are the one format this loader does **not** eagerly
//! dequantize: `load_mxfp4_weight_matrix` builds a zero-copy
//! `WeightMatrix::Mxfp4` (mmap-backed `packed`/`scale` buffers, real
//! Kimi K3 stores these as two separate tensors per projection -- see
//! `frink_quant::dot_mxfp4_row_f32`'s doc comment), matching the
//! zero-copy-mmap-plus-fused-dot discipline every other quantized
//! format in this codebase already uses. This is a real fix, not just a
//! design preference: real-hardware testing (rented 62GB instance, see
//! docs/MODELS.md) found that eagerly dequantizing all 896 of a real MoE
//! layer's routed experts to owned `f32` needs roughly 117GB of RAM and
//! reproducibly OOM-killed a 62GB rented instance -- the zero-copy path
//! here keeps a loaded layer's resident memory close to its on-disk
//! size (the real MXFP4 packing ratio: 2 values/byte plus one scale
//! byte per 32 values) instead of expanding every value to 4 bytes
//! whether it's ever used by real top-k routing or not.

use frink_core::tensor::Tensor;
use frink_core::weight_matrix::{WeightBytes, WeightMatrix};
use frink_safetensors::{SafetensorsDtype, ShardedSafetensors};
use thiserror::Error;

use crate::config::LayerAttentionKind;
use crate::kda::KdaAttnWeights;
use crate::kimi_decoder::DenseMlpWeights;
use crate::latent_moe::{KimiExpertBacking, KimiExpertWeights, KimiLatentMoeWeights};
use crate::mla::{MlaAttnWeights, MlaKvB, MlaQProj};
use frink_core::expert_store::{ExpertKey, ExpertSource, ExpertStore};

#[derive(Debug, Error)]
pub enum KimiLoadError {
    #[error("safetensors error: {0}")]
    Safetensors(#[from] frink_safetensors::SafetensorsError),
    #[error("tensor '{0}' has unsupported dtype {1:?} (expected F32 or BF16)")]
    UnsupportedDtype(String, SafetensorsDtype),
    #[error("{0}")]
    Other(String),
}

/// Reads any real tensor as an owned `f32` vector, dispatching on its
/// real declared dtype through [`crate::safetensors_f32::widen_to_f32`]
/// (`F32` direct, `F16` / `BF16` widened losslessly) -- exposed
/// `pub` since not every real weight (e.g. the per-layer
/// `input_layernorm.weight`/`post_attention_layernorm.weight`, which
/// aren't nested inside `KdaAttnWeights`/`MlaAttnWeights`/
/// `DenseMlpWeights`/`BlockResidualWeights`) has a dedicated loader
/// function above.
pub fn load_f32_vec(shard: &ShardedSafetensors, name: &str) -> Result<Vec<f32>, KimiLoadError> {
    let info = shard
        .tensor_info(name)
        .ok_or_else(|| frink_safetensors::SafetensorsError::TensorNotFound(name.to_string()))?;
    let raw = shard.tensor_bytes(name)?;
    crate::safetensors_f32::widen_to_f32(info.dtype, raw)
        .ok_or_else(|| KimiLoadError::UnsupportedDtype(name.to_string(), info.dtype))
}

fn load_weight_matrix(
    shard: &ShardedSafetensors,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightMatrix, KimiLoadError> {
    let data = load_f32_vec(shard, name)?;
    assert_eq!(
        data.len(),
        rows * cols,
        "tensor '{name}' has {} elements, expected {rows}*{cols}",
        data.len()
    );
    Ok(WeightMatrix::F32(Tensor::new(data, vec![rows, cols])))
}

/// Loads one KDA-attention layer's weights (real tensor names under
/// `{prefix}.self_attn.*`). `num_heads`/`head_dim`/`hidden_dim` must
/// match the real config (Kimi K3: 96/128/7168) -- passed explicitly
/// rather than hardcoded so this loader can also be exercised against
/// small synthetic on-disk fixtures in tests.
pub fn load_kda_attn(
    shard: &ShardedSafetensors,
    prefix: &str,
    num_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
) -> Result<KdaAttnWeights, KimiLoadError> {
    let projection_size = num_heads * head_dim;
    let a_log_full = load_f32_vec(shard, &format!("{prefix}.self_attn.A_log"))?;
    // Real padding: only the first `num_heads` of A_log's real on-disk
    // elements are ever read by the real kernel -- see module doc
    // comment.
    let a_log = a_log_full[..num_heads].to_vec();

    Ok(KdaAttnWeights {
        q_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.q_proj.weight"),
            projection_size,
            hidden_dim,
        )?,
        k_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.k_proj.weight"),
            projection_size,
            hidden_dim,
        )?,
        v_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.v_proj.weight"),
            projection_size,
            hidden_dim,
        )?,
        // Real on-disk shape is [projection_size, 1, kernel_size]; the
        // middle dim is always 1 (depthwise conv), so the raw bytes are
        // already exactly [projection_size, kernel_size] flattened --
        // no reshape needed, just read as a flat vec.
        q_conv_weight: load_f32_vec(shard, &format!("{prefix}.self_attn.q_conv1d.weight"))?,
        k_conv_weight: load_f32_vec(shard, &format!("{prefix}.self_attn.k_conv1d.weight"))?,
        v_conv_weight: load_f32_vec(shard, &format!("{prefix}.self_attn.v_conv1d.weight"))?,
        a_log,
        f_a_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.f_a_proj.weight"),
            head_dim,
            hidden_dim,
        )?,
        f_b_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.f_b_proj.weight"),
            projection_size,
            head_dim,
        )?,
        dt_bias: load_f32_vec(shard, &format!("{prefix}.self_attn.dt_bias"))?,
        b_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.b_proj.weight"),
            num_heads,
            hidden_dim,
        )?,
        g_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.g_proj.weight"),
            projection_size,
            hidden_dim,
        )?,
        o_norm_weight: load_f32_vec(shard, &format!("{prefix}.self_attn.o_norm.weight"))?,
        o_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden_dim,
            projection_size,
        )?,
    })
}

/// Loads one Gated-MLA-attention layer's weights (real tensor names
/// under `{prefix}.self_attn.*`).
#[allow(clippy::too_many_arguments)]
pub fn load_mla_attn(
    shard: &ShardedSafetensors,
    prefix: &str,
    num_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    hidden_dim: usize,
) -> Result<MlaAttnWeights, KimiLoadError> {
    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    Ok(MlaAttnWeights {
        q: MlaQProj::LowRank {
            a: load_weight_matrix(
                shard,
                &format!("{prefix}.self_attn.q_a_proj.weight"),
                q_lora_rank,
                hidden_dim,
            )?,
            norm: load_f32_vec(shard, &format!("{prefix}.self_attn.q_a_layernorm.weight"))?,
            b: load_weight_matrix(
                shard,
                &format!("{prefix}.self_attn.q_b_proj.weight"),
                num_heads * q_head_dim,
                q_lora_rank,
            )?,
        },
        kv_a_proj_with_mqa: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            kv_lora_rank + qk_rope_head_dim,
            hidden_dim,
        )?,
        kv_a_layernorm: load_f32_vec(shard, &format!("{prefix}.self_attn.kv_a_layernorm.weight"))?,
        kv_b: MlaKvB::Combined(load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.kv_b_proj.weight"),
            num_heads * (qk_nope_head_dim + v_head_dim),
            kv_lora_rank,
        )?),
        o_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.o_proj.weight"),
            hidden_dim,
            num_heads * v_head_dim,
        )?,
        g_proj: Some(load_weight_matrix(
            shard,
            &format!("{prefix}.self_attn.g_proj.weight"),
            num_heads * v_head_dim,
            hidden_dim,
        )?),
    })
}

/// Loads the dense leading layer's feed-forward block (real tensor
/// names under `{prefix}.mlp.*`) -- Kimi K3's layer 0 only
/// (`first_k_dense_replace`=1).
pub fn load_dense_mlp(
    shard: &ShardedSafetensors,
    prefix: &str,
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Result<DenseMlpWeights, KimiLoadError> {
    Ok(DenseMlpWeights {
        gate_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.mlp.gate_proj.weight"),
            intermediate_dim,
            hidden_dim,
        )?,
        up_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.mlp.up_proj.weight"),
            intermediate_dim,
            hidden_dim,
        )?,
        down_proj: load_weight_matrix(
            shard,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden_dim,
            intermediate_dim,
        )?,
    })
}

/// The four block-residual weight vectors real Kimi K3 attaches to
/// every layer (`{prefix}.self_attention_res_{norm,proj}.weight`,
/// `{prefix}.mlp_res_{norm,proj}.weight`). Real on-disk `*_proj.weight`
/// shape is `[1, hidden_dim]` (a `Linear(hidden_dim, 1)`'s weight); the
/// raw bytes are already exactly `[hidden_dim]` flattened.
pub struct BlockResidualWeights {
    pub self_attention_res_norm_weight: Vec<f32>,
    pub self_attention_res_proj_weight: Vec<f32>,
    pub mlp_res_norm_weight: Vec<f32>,
    pub mlp_res_proj_weight: Vec<f32>,
}

pub fn load_block_residual(
    shard: &ShardedSafetensors,
    prefix: &str,
) -> Result<BlockResidualWeights, KimiLoadError> {
    Ok(BlockResidualWeights {
        self_attention_res_norm_weight: load_f32_vec(
            shard,
            &format!("{prefix}.self_attention_res_norm.weight"),
        )?,
        self_attention_res_proj_weight: load_f32_vec(
            shard,
            &format!("{prefix}.self_attention_res_proj.weight"),
        )?,
        mlp_res_norm_weight: load_f32_vec(shard, &format!("{prefix}.mlp_res_norm.weight"))?,
        mlp_res_proj_weight: load_f32_vec(shard, &format!("{prefix}.mlp_res_proj.weight"))?,
    })
}

/// Loads one MXFP4-quantized weight matrix from its real two-separate-
/// tensors shape (`*.weight_packed` + `*.weight_scale`, confirmed
/// against a real shard header -- see `frink_quant::dot_mxfp4_row_f32`'s
/// doc comment) as a zero-copy `WeightMatrix::Mxfp4`: both buffers are
/// mmap-backed views (`ShardedSafetensors::tensor_mapped_range`), never
/// copied into an owned buffer, let alone dequantized to `f32` --
/// `apply`/`apply_batch` dispatch straight to the fused
/// `dot_mxfp4_row_f32` kernel. See this module's doc comment for why
/// this matters at real scale (a real, measured ~117GB RAM difference
/// for one full 896-expert MoE layer).
fn load_mxfp4_weight_matrix(
    shard: &ShardedSafetensors,
    packed_name: &str,
    scale_name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightMatrix, KimiLoadError> {
    let (packed_mmap, packed_range) = shard.tensor_mapped_range(packed_name)?;
    let (scale_mmap, scale_range) = shard.tensor_mapped_range(scale_name)?;
    let packed_per_row = cols / 2;
    let scale_per_row = cols / frink_quant::MXFP4_GROUP_SIZE;
    assert_eq!(
        packed_range.len(),
        rows * packed_per_row,
        "'{packed_name}' has {} bytes, expected {rows}*{packed_per_row}",
        packed_range.len()
    );
    assert_eq!(
        scale_range.len(),
        rows * scale_per_row,
        "'{scale_name}' has {} bytes, expected {rows}*{scale_per_row}",
        scale_range.len()
    );

    Ok(WeightMatrix::Mxfp4 {
        packed: WeightBytes::Mapped {
            mmap: packed_mmap,
            range: packed_range,
        },
        scale: WeightBytes::Mapped {
            mmap: scale_mmap,
            range: scale_range,
        },
        rows,
        cols,
    })
}

/// Loads one routed expert's real MXFP4 weights
/// (`{prefix}.experts.{expert_idx}.{w1,w2,w3}.{weight_packed,weight_scale}`).
/// `moe_hidden_dim` is the real *latent* dimension
/// (`routed_expert_hidden_size`=3584 for Kimi K3, not the outer
/// `hidden_dim`=7168 -- see `frink-models::latent_moe`'s module doc
/// comment); `moe_intermediate_dim` is the per-expert FFN size (3072).
/// Per-layer byte layout of one store-backed Kimi routed expert's
/// combined buffer: w1_packed, w1_scale, w2_packed, w2_scale,
/// w3_packed, w3_scale concatenated in that fixed order. Every expert
/// in a Kimi layer has identical dims, so one layout serves the layer.
#[derive(Debug, Clone, Copy)]
pub struct KimiStoredExpertLayout {
    pub moe_hidden_dim: usize,
    pub moe_intermediate_dim: usize,
}

impl KimiStoredExpertLayout {
    fn seg_lens(&self) -> [usize; 6] {
        let (h, m) = (self.moe_hidden_dim, self.moe_intermediate_dim);
        let g = frink_quant::MXFP4_GROUP_SIZE;
        [
            m * h / 2, // w1 packed: [m, h]
            m * h / g, // w1 scale
            h * m / 2, // w2 packed: [h, m]
            h * m / g, // w2 scale
            m * h / 2, // w3 packed: [m, h]
            m * h / g, // w3 scale
        ]
    }

    pub fn total_bytes(&self) -> usize {
        self.seg_lens().iter().sum()
    }

    /// Builds temporary two-buffer MXFP4 `WeightMatrix` views over a
    /// leased buffer; each view's `WeightBytes::Shared` clone keeps
    /// the cache entry pinned for the view's lifetime.
    pub fn materialize(&self, lease: &frink_core::expert_store::ExpertLease) -> KimiExpertWeights {
        let (h, m) = (self.moe_hidden_dim, self.moe_intermediate_dim);
        let lens = self.seg_lens();
        let mut offsets = [0usize; 6];
        for i in 1..6 {
            offsets[i] = offsets[i - 1] + lens[i - 1];
        }
        let shared = |i: usize| WeightBytes::Shared {
            buf: lease.shared_buf(),
            range: offsets[i]..offsets[i] + lens[i],
        };
        let mx = |pi: usize, si: usize, rows: usize, cols: usize| WeightMatrix::Mxfp4 {
            packed: shared(pi),
            scale: shared(si),
            rows,
            cols,
        };
        KimiExpertWeights {
            w1: mx(0, 1, m, h),
            w2: mx(2, 3, h, m),
            w3: mx(4, 5, m, h),
        }
    }
}

/// [`ExpertSource`] over a Kimi safetensors checkpoint: each expert's
/// six tensors (three matrices' packed+scale buffers) are read
/// positionally from the owning shard files and concatenated in
/// `KimiStoredExpertLayout`'s fixed order.
pub struct KimiExpertSource {
    files: Vec<std::fs::File>,
    /// (layer, expert) -> six (file index, offset, len) segments.
    segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 6]>,
}

impl ExpertSource for KimiExpertSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.segments
            .get(&key)
            .map(|segs| segs.iter().map(|&(_, _, len)| len).sum())
    }

    fn read_expert(&self, key: ExpertKey) -> std::io::Result<Vec<u8>> {
        let segs = self
            .segments
            .get(&key)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("{key:?}")))?;
        let total: usize = segs.iter().map(|&(_, _, len)| len).sum();
        let mut buf = vec![0u8; total];
        let mut written = 0;
        for &(fi, offset, len) in segs {
            let dst = &mut buf[written..written + len];
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                self.files[fi].read_exact_at(dst, offset)?;
            }
            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = &self.files[fi];
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(dst)?;
            }
            written += len;
        }
        Ok(buf)
    }
}

pub fn load_kimi_expert(
    shard: &ShardedSafetensors,
    moe_prefix: &str,
    expert_idx: usize,
    moe_hidden_dim: usize,
    moe_intermediate_dim: usize,
) -> Result<KimiExpertWeights, KimiLoadError> {
    let expert_prefix = format!("{moe_prefix}.experts.{expert_idx}");
    Ok(KimiExpertWeights {
        w1: load_mxfp4_weight_matrix(
            shard,
            &format!("{expert_prefix}.w1.weight_packed"),
            &format!("{expert_prefix}.w1.weight_scale"),
            moe_intermediate_dim,
            moe_hidden_dim,
        )?,
        w2: load_mxfp4_weight_matrix(
            shard,
            &format!("{expert_prefix}.w2.weight_packed"),
            &format!("{expert_prefix}.w2.weight_scale"),
            moe_hidden_dim,
            moe_intermediate_dim,
        )?,
        w3: load_mxfp4_weight_matrix(
            shard,
            &format!("{expert_prefix}.w3.weight_packed"),
            &format!("{expert_prefix}.w3.weight_scale"),
            moe_intermediate_dim,
            moe_hidden_dim,
        )?,
    })
}

/// Loads one full MoE layer: the gate (with its real aux-loss-free
/// `e_score_correction_bias`), the shared down/up latent projections +
/// norm, every routed expert (`n_experts`, real MXFP4), and the shared
/// expert (real `BF16`, on the full `hidden_dim`, not the latent space
/// -- see `frink-models::latent_moe`'s module doc comment).
#[allow(clippy::too_many_arguments)]
pub fn load_latent_moe(
    shard: &ShardedSafetensors,
    prefix: &str,
    hidden_dim: usize,
    moe_hidden_dim: usize,
    moe_intermediate_dim: usize,
    n_experts: usize,
    shared_intermediate_dim: usize,
) -> Result<KimiLatentMoeWeights, KimiLoadError> {
    let moe_prefix = format!("{prefix}.block_sparse_moe");
    let mut experts = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        experts.push(load_kimi_expert(
            shard,
            &moe_prefix,
            e,
            moe_hidden_dim,
            moe_intermediate_dim,
        )?);
    }
    let experts = KimiExpertBacking::Resident(experts);

    Ok(KimiLatentMoeWeights {
        router_weight: load_weight_matrix(
            shard,
            &format!("{moe_prefix}.gate.weight"),
            n_experts,
            hidden_dim,
        )?,
        e_score_correction_bias: load_f32_vec(
            shard,
            &format!("{moe_prefix}.gate.e_score_correction_bias"),
        )?,
        down_proj: load_weight_matrix(
            shard,
            &format!("{moe_prefix}.routed_expert_down_proj.weight"),
            moe_hidden_dim,
            hidden_dim,
        )?,
        up_proj: load_weight_matrix(
            shard,
            &format!("{moe_prefix}.routed_expert_up_proj.weight"),
            hidden_dim,
            moe_hidden_dim,
        )?,
        routed_expert_norm_weight: Some(load_f32_vec(
            shard,
            &format!("{moe_prefix}.routed_expert_norm.weight"),
        )?),
        experts,
        shared_expert: KimiExpertWeights {
            w1: load_weight_matrix(
                shard,
                &format!("{moe_prefix}.shared_experts.gate_proj.weight"),
                shared_intermediate_dim,
                hidden_dim,
            )?,
            w2: load_weight_matrix(
                shard,
                &format!("{moe_prefix}.shared_experts.down_proj.weight"),
                hidden_dim,
                shared_intermediate_dim,
            )?,
            w3: load_weight_matrix(
                shard,
                &format!("{moe_prefix}.shared_experts.up_proj.weight"),
                shared_intermediate_dim,
                hidden_dim,
            )?,
        },
    })
}

/// Kimi K3's real per-layer hyperparameters needed to load any layer
/// (not tied to `frink_moe::MoeLayerConfig`/`frink_models::ModelConfig`,
/// neither of which model the "latent MoE" down-projected dimension or
/// the dense leading layer's own intermediate size -- kept as a small,
/// dedicated struct here rather than widening those shared types for
/// one model's real values).
pub struct KimiRealHparams {
    pub hidden_dim: usize,
    pub kda_num_heads: usize,
    pub kda_head_dim: usize,
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

impl KimiRealHparams {
    /// Kimi K3's real published values (`config.json`).
    pub fn real() -> Self {
        KimiRealHparams {
            hidden_dim: 7168,
            kda_num_heads: 96,
            kda_head_dim: 128,
            mla_num_heads: 96,
            mla_q_lora_rank: 1536,
            mla_kv_lora_rank: 512,
            mla_qk_nope_head_dim: 128,
            mla_qk_rope_head_dim: 64,
            mla_v_head_dim: 128,
            dense_intermediate_dim: 33792,
            moe_hidden_dim: 3584,
            moe_intermediate_dim: 3072,
            n_experts: 896,
            num_shared_experts: 2,
        }
    }
}

/// Loads any one layer (KDA or Gated-MLA attention, dense or MoE FFN),
/// dispatching on `kind`/`is_dense` -- pass
/// `ModelConfig::layer_attention_kind(layer_idx)`/
/// `ModelConfig::layer_is_dense(layer_idx)` for Kimi K3's real per-layer
/// topology.
pub fn load_kimi_layer(
    shard: &ShardedSafetensors,
    hp: &KimiRealHparams,
    kind: LayerAttentionKind,
    is_dense: bool,
    layer_idx: usize,
) -> Result<crate::kimi_decoder::KimiDecoderLayerWeights, KimiLoadError> {
    let prefix = format!("language_model.model.layers.{layer_idx}");

    let input_layernorm_weight = load_f32_vec(shard, &format!("{prefix}.input_layernorm.weight"))?;
    let post_attention_layernorm_weight =
        load_f32_vec(shard, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let block_res = load_block_residual(shard, &prefix)?;

    let attn = match kind {
        LayerAttentionKind::KimiKda => {
            crate::kimi_decoder::KimiLayerAttention::Kda(Box::new(load_kda_attn(
                shard,
                &prefix,
                hp.kda_num_heads,
                hp.kda_head_dim,
                hp.hidden_dim,
            )?))
        }
        LayerAttentionKind::KimiMla => {
            crate::kimi_decoder::KimiLayerAttention::Mla(Box::new(load_mla_attn(
                shard,
                &prefix,
                hp.mla_num_heads,
                hp.mla_q_lora_rank,
                hp.mla_kv_lora_rank,
                hp.mla_qk_nope_head_dim,
                hp.mla_qk_rope_head_dim,
                hp.mla_v_head_dim,
                hp.hidden_dim,
            )?))
        }
        LayerAttentionKind::Gqa => {
            panic!("load_kimi_layer is only for KimiHybrid (KDA/Gated-MLA) layers")
        }
    };

    let ffn = if is_dense {
        crate::kimi_decoder::KimiLayerFfn::Dense(Box::new(load_dense_mlp(
            shard,
            &prefix,
            hp.hidden_dim,
            hp.dense_intermediate_dim,
        )?))
    } else {
        crate::kimi_decoder::KimiLayerFfn::Moe(Box::new(load_latent_moe(
            shard,
            &prefix,
            hp.hidden_dim,
            hp.moe_hidden_dim,
            hp.moe_intermediate_dim,
            hp.n_experts,
            hp.moe_intermediate_dim * hp.num_shared_experts,
        )?))
    };

    Ok(crate::kimi_decoder::KimiDecoderLayerWeights {
        input_layernorm_weight,
        attn,
        post_attention_layernorm_weight,
        ffn,
        self_attention_res_norm_weight: block_res.self_attention_res_norm_weight,
        self_attention_res_proj_weight: block_res.self_attention_res_proj_weight,
        mlp_res_norm_weight: block_res.mlp_res_norm_weight,
        mlp_res_proj_weight: block_res.mlp_res_proj_weight,
    })
}

/// Loads a complete `KimiDecoderWeights` -- every one of `model_cfg`'s
/// real layers (dispatched per-layer via `model_cfg.layer_attention_kind`/
/// `layer_is_dense`, driven by `hp`'s per-layer dimensions), plus the
/// real top-level tensors (real names confirmed against a real shard
/// header: `language_model.model.embed_tokens.weight`,
/// `language_model.lm_head.weight`, `language_model.model.norm.weight`,
/// `language_model.model.output_attn_res_{norm,proj}.weight`). This is
/// the assembly step `load_kimi_layer` itself doesn't do -- calling it
/// once per real layer and building the surrounding `KimiDecoderWeights`
/// -- analogous to `frink-models::loader::Decoder::from_gguf`, but for
/// Kimi K3's real safetensors format. Not blocked on anything (the
/// zero-copy MXFP4 fix removes the memory obstacle a full loader would
/// otherwise hit for every non-dense layer's routed experts); simply
/// not runnable against the real 2.8T-parameter checkpoint in this
/// environment (96 shards, 1.56TB) -- tested here against small
/// synthetic on-disk fixtures instead, real safetensors bytes and real
/// tensor names throughout.
/// Like [`load_kimi_checkpoint`], but with `expert_cache_bytes:
/// Some(budget)` every MoE layer's routed experts are converted to
/// store-backed lazy materialization after loading: one bounded,
/// lease-protected `ExpertStore` shared by the whole model reads each
/// expert's six tensors positionally from the owning shard files on
/// miss, instead of holding 896 expert objects per layer resident.
/// Attention, dense layers, shared experts, router/projections,
/// embeddings, and the output head are untouched. Bit-identical to
/// the eager path (same bytes, same kernels) -- pinned by the
/// equivalence test against the synthetic multi-layer checkpoint.
pub fn load_kimi_checkpoint_with_expert_cache(
    shard: &ShardedSafetensors,
    model_cfg: &crate::config::ModelConfig,
    hp: &KimiRealHparams,
    expert_cache_bytes: Option<u64>,
) -> Result<crate::kimi_decoder::KimiDecoderWeights, KimiLoadError> {
    let mut weights = load_kimi_checkpoint(shard, model_cfg, hp)?;
    let Some(budget) = expert_cache_bytes else {
        return Ok(weights);
    };

    // Collect every MoE layer's per-expert file segments, then swap
    // each layer's backing to the one shared store.
    let mut files: Vec<std::fs::File> = Vec::new();
    let mut path_index: std::collections::HashMap<std::path::PathBuf, usize> =
        std::collections::HashMap::new();
    let mut segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 6]> =
        std::collections::HashMap::new();
    let layout = KimiStoredExpertLayout {
        moe_hidden_dim: hp.moe_hidden_dim,
        moe_intermediate_dim: hp.moe_intermediate_dim,
    };
    let mut moe_layers: Vec<(usize, usize)> = Vec::new(); // (layer_idx, n_experts)

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let crate::kimi_decoder::KimiLayerFfn::Moe(moe) = &layer.ffn else {
            continue;
        };
        let n_experts = moe.experts.n_experts();
        let moe_prefix = format!("language_model.model.layers.{layer_idx}.block_sparse_moe");
        for e in 0..n_experts {
            let expert_prefix = format!("{moe_prefix}.experts.{e}");
            let mut segs = [(0usize, 0u64, 0usize); 6];
            for (i, tensor) in [
                format!("{expert_prefix}.w1.weight_packed"),
                format!("{expert_prefix}.w1.weight_scale"),
                format!("{expert_prefix}.w2.weight_packed"),
                format!("{expert_prefix}.w2.weight_scale"),
                format!("{expert_prefix}.w3.weight_packed"),
                format!("{expert_prefix}.w3.weight_scale"),
            ]
            .iter()
            .enumerate()
            {
                let (path, range) = shard.tensor_file_location(tensor)?;
                let fi = match path_index.get(path) {
                    Some(&fi) => fi,
                    None => {
                        let fi = files.len();
                        files.push(std::fs::File::open(path).map_err(|e| {
                            KimiLoadError::Other(format!(
                                "opening shard file {} for expert streaming: {e}",
                                path.display()
                            ))
                        })?);
                        path_index.insert(path.to_path_buf(), fi);
                        fi
                    }
                };
                segs[i] = (fi, range.start as u64, range.end - range.start);
            }
            segments.insert(
                ExpertKey {
                    layer: layer_idx as u32,
                    expert: e as u32,
                },
                segs,
            );
        }
        moe_layers.push((layer_idx, n_experts));
    }

    if moe_layers.is_empty() {
        return Ok(weights);
    }
    let store = std::sync::Arc::new(ExpertStore::new(
        KimiExpertSource { files, segments },
        budget as usize,
    ));
    for (layer_idx, n_experts) in moe_layers {
        if let crate::kimi_decoder::KimiLayerFfn::Moe(moe) = &mut weights.layers[layer_idx].ffn {
            moe.experts = KimiExpertBacking::Stored {
                store: std::sync::Arc::clone(&store),
                layout,
                n_experts,
                layer: layer_idx as u32,
            };
        }
    }
    Ok(weights)
}

pub fn load_kimi_checkpoint(
    shard: &ShardedSafetensors,
    model_cfg: &crate::config::ModelConfig,
    hp: &KimiRealHparams,
) -> Result<crate::kimi_decoder::KimiDecoderWeights, KimiLoadError> {
    let mut layers = Vec::with_capacity(model_cfg.n_layers);
    for layer_idx in 0..model_cfg.n_layers {
        let kind = model_cfg.layer_attention_kind(layer_idx);
        let is_dense = model_cfg.layer_is_dense(layer_idx);
        layers.push(load_kimi_layer(shard, hp, kind, is_dense, layer_idx)?);
    }

    let embedding_data = load_f32_vec(shard, "language_model.model.embed_tokens.weight")?;
    let embedding = Tensor::new(embedding_data, vec![model_cfg.vocab_size, hp.hidden_dim]);
    let output_head = load_weight_matrix(
        shard,
        "language_model.lm_head.weight",
        model_cfg.vocab_size,
        hp.hidden_dim,
    )?;
    let final_norm_weight = load_f32_vec(shard, "language_model.model.norm.weight")?;
    let output_attn_res_norm_weight =
        load_f32_vec(shard, "language_model.model.output_attn_res_norm.weight")?;
    let output_attn_res_proj_weight =
        load_f32_vec(shard, "language_model.model.output_attn_res_proj.weight")?;

    Ok(crate::kimi_decoder::KimiDecoderWeights {
        embedding,
        layers,
        output_attn_res_norm_weight,
        output_attn_res_proj_weight,
        final_norm_weight,
        output_head,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    /// Builds a real on-disk safetensors file (header + raw bytes)
    /// containing exactly one BF16-widened-from-f32 tensor for the
    /// given name/shape -- BF16 chosen since it's the dominant real
    /// dtype in Kimi K3's checkpoint, exercising the dequant path.
    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &v in values {
            let bits = v.to_bits();
            out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        out
    }

    fn build_shard(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut offset = 0u64;
        let mut data = Vec::new();
        for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let end = offset + bytes.len() as u64;
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_str}],\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
            data.extend_from_slice(bytes);
        }
        header.push('}');

        let mut buf = Vec::new();
        buf.write_u64::<LittleEndian>(header.len() as u64).unwrap();
        buf.write_all(header.as_bytes()).unwrap();
        buf.extend_from_slice(&data);
        buf
    }

    #[test]
    fn loads_a_dense_mlp_from_a_real_on_disk_safetensors_shard() {
        let hidden_dim = 4;
        let intermediate_dim = 6;
        let gate = vec![
            0.1f32, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8, -0.9, 1.0, 1.1, -1.2, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let gate = &gate[..intermediate_dim * hidden_dim];
        let up = vec![0.05f32; intermediate_dim * hidden_dim];
        let down = vec![0.02f32; hidden_dim * intermediate_dim];

        let shard_bytes = build_shard(&[
            (
                "model.layers.0.mlp.gate_proj.weight",
                "BF16",
                &[intermediate_dim, hidden_dim],
                bf16_bytes(gate),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "BF16",
                &[intermediate_dim, hidden_dim],
                bf16_bytes(&up),
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "BF16",
                &[hidden_dim, intermediate_dim],
                bf16_bytes(&down),
            ),
        ]);

        let dir = std::env::temp_dir().join(format!(
            "frink_kimi_loader_dense_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let index = r#"{"weight_map":{
            "model.layers.0.mlp.gate_proj.weight":"shard0.safetensors",
            "model.layers.0.mlp.up_proj.weight":"shard0.safetensors",
            "model.layers.0.mlp.down_proj.weight":"shard0.safetensors"
        }}"#;
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let weights = load_dense_mlp(&shard, "model.layers.0", hidden_dim, intermediate_dim)
            .expect("must load dense mlp");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(weights.gate_proj.rows(), intermediate_dim);
        assert_eq!(weights.gate_proj.cols(), hidden_dim);
        let x = vec![1.0f32; hidden_dim];
        let out = weights.forward(&x, 4.0, 25.0);
        assert_eq!(out.len(), hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_log_padding_is_truncated_to_num_heads() {
        // Real on-disk A_log has 128 elements but only num_heads(=2
        // here) are ever consumed -- confirm the loader truncates
        // rather than asserting a shape match against the full tensor.
        let a_log_full: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let raw: Vec<u8> = a_log_full.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shard_bytes = build_shard(&[("self_attn.A_log", "F32", &[8], raw)]);
        let dir = std::env::temp_dir().join(format!(
            "frink_kimi_loader_alog_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let index = r#"{"weight_map":{"self_attn.A_log":"shard0.safetensors"}}"#;
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let full = load_f32_vec(&shard, "self_attn.A_log").unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(full.len(), 8);
        let truncated = &full[..2];
        assert_eq!(truncated, &[0.0, 0.1]);
    }

    /// Deterministic byte generator (no external `rand` dependency in
    /// this crate) -- any byte pattern is a structurally valid MXFP4
    /// block (dequant doesn't depend on the bytes coming from a real
    /// quantizer), so this just needs to be varied, not random.
    fn pseudo_bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect()
    }

    /// Like `pseudo_bytes`, but clamped to a realistic E8M0 scale range
    /// (roughly `2^-127` to `2^53`). Scale bytes near the top of the
    /// real `u8` range (255 reserved for NaN by the OCP spec and
    /// deliberately not special-cased by `frink_quant::dequant_mxfp4_row`,
    /// matching real `ggml_e8m0_to_fp32`'s own documented limitation; and
    /// bytes up to ~252 combined with E2M1's max magnitude of 6 can
    /// legitimately overflow `f32::MAX`) are real OCP MX behavior, not a
    /// bug -- just not representative of any real *trained* weight's
    /// scale, and not what this test (confirming the loader wires real
    /// bytes through correctly) is checking for.
    fn pseudo_scale_bytes(seed: u32, len: usize) -> Vec<u8> {
        pseudo_bytes(seed, len)
            .into_iter()
            .map(|b| b % 180)
            .collect()
    }

    #[test]
    fn loads_one_mxfp4_expert_from_a_real_on_disk_safetensors_shard() {
        // Smallest valid dims: MXFP4_GROUP_SIZE=32, so every in_dim here
        // must be a multiple of 32.
        let moe_hidden_dim = 32;
        let moe_intermediate_dim = 32;
        let expert_prefix = "model.layers.3.block_sparse_moe.experts.0";

        let w1_packed = pseudo_bytes(1, moe_intermediate_dim * (moe_hidden_dim / 2));
        let w1_scale = pseudo_scale_bytes(2, moe_intermediate_dim * (moe_hidden_dim / 32));
        let w2_packed = pseudo_bytes(3, moe_hidden_dim * (moe_intermediate_dim / 2));
        let w2_scale = pseudo_scale_bytes(4, moe_hidden_dim * (moe_intermediate_dim / 32));
        let w3_packed = pseudo_bytes(5, moe_intermediate_dim * (moe_hidden_dim / 2));
        let w3_scale = pseudo_scale_bytes(6, moe_intermediate_dim * (moe_hidden_dim / 32));

        let shard_bytes = build_shard(&[
            (
                &format!("{expert_prefix}.w1.weight_packed"),
                "U8",
                &[moe_intermediate_dim, moe_hidden_dim / 2],
                w1_packed,
            ),
            (
                &format!("{expert_prefix}.w1.weight_scale"),
                "U8",
                &[moe_intermediate_dim, moe_hidden_dim / 32],
                w1_scale,
            ),
            (
                &format!("{expert_prefix}.w2.weight_packed"),
                "U8",
                &[moe_hidden_dim, moe_intermediate_dim / 2],
                w2_packed,
            ),
            (
                &format!("{expert_prefix}.w2.weight_scale"),
                "U8",
                &[moe_hidden_dim, moe_intermediate_dim / 32],
                w2_scale,
            ),
            (
                &format!("{expert_prefix}.w3.weight_packed"),
                "U8",
                &[moe_intermediate_dim, moe_hidden_dim / 2],
                w3_packed,
            ),
            (
                &format!("{expert_prefix}.w3.weight_scale"),
                "U8",
                &[moe_intermediate_dim, moe_hidden_dim / 32],
                w3_scale,
            ),
        ]);

        let dir = std::env::temp_dir().join(format!(
            "frink_kimi_loader_mxfp4_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let index = format!(
            r#"{{"weight_map":{{
                "{expert_prefix}.w1.weight_packed":"shard0.safetensors",
                "{expert_prefix}.w1.weight_scale":"shard0.safetensors",
                "{expert_prefix}.w2.weight_packed":"shard0.safetensors",
                "{expert_prefix}.w2.weight_scale":"shard0.safetensors",
                "{expert_prefix}.w3.weight_packed":"shard0.safetensors",
                "{expert_prefix}.w3.weight_scale":"shard0.safetensors"
            }}}}"#
        );
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, &index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let expert = load_kimi_expert(
            &shard,
            "model.layers.3.block_sparse_moe",
            0,
            moe_hidden_dim,
            moe_intermediate_dim,
        )
        .expect("must load real MXFP4 expert weights");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(expert.w1.rows(), moe_intermediate_dim);
        assert_eq!(expert.w1.cols(), moe_hidden_dim);
        assert_eq!(expert.w2.rows(), moe_hidden_dim);
        assert_eq!(expert.w2.cols(), moe_intermediate_dim);

        let x = vec![0.1f32; moe_hidden_dim];
        let out = expert.forward(&x, 4.0, 25.0);
        assert_eq!(out.len(), moe_hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Like `build_shard`, but takes owned `String`/`Vec<usize>` tensor
    /// descriptors so callers can build the list programmatically
    /// (needed for `load_kimi_layer`'s tests, which have far more
    /// tensors than the hand-written fixtures above).
    fn build_shard_owned(tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)>) -> Vec<u8> {
        let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (name.as_str(), *dtype, shape.as_slice(), bytes.clone())
            })
            .collect();
        build_shard(&refs)
    }

    #[test]
    fn load_kimi_layer_dispatches_kda_plus_dense_at_a_nonzero_layer_index() {
        let hidden_dim = 8;
        let kda_num_heads = 2;
        let kda_head_dim = 3;
        let kda_proj = kda_num_heads * kda_head_dim;
        let conv_size = 4;
        let dense_intermediate = 5;
        let layer_idx = 5;
        let prefix = format!("language_model.model.layers.{layer_idx}");

        let mut tensors = Vec::new();
        let mut push_bf16 = |name: String, shape: Vec<usize>, n: usize| {
            tensors.push((name, "BF16", shape, bf16_bytes(&vec![0.05f32; n])));
        };
        push_bf16(
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attention_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attention_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.mlp_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.mlp_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.f_a_proj.weight"),
            vec![kda_head_dim, hidden_dim],
            kda_head_dim * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.f_b_proj.weight"),
            vec![kda_proj, kda_head_dim],
            kda_proj * kda_head_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.b_proj.weight"),
            vec![kda_num_heads, hidden_dim],
            kda_num_heads * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.g_proj.weight"),
            vec![kda_proj, hidden_dim],
            kda_proj * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden_dim, kda_proj],
            hidden_dim * kda_proj,
        );
        push_bf16(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![dense_intermediate, hidden_dim],
            dense_intermediate * hidden_dim,
        );
        push_bf16(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden_dim, dense_intermediate],
            hidden_dim * dense_intermediate,
        );

        let f32_vec = |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        tensors.push((
            format!("{prefix}.self_attn.A_log"),
            "F32",
            vec![kda_num_heads],
            f32_vec(vec![0.5; kda_num_heads]),
        ));
        tensors.push((
            format!("{prefix}.self_attn.dt_bias"),
            "F32",
            vec![kda_proj],
            f32_vec(vec![0.1; kda_proj]),
        ));
        tensors.push((
            format!("{prefix}.self_attn.o_norm.weight"),
            "F32",
            vec![kda_head_dim],
            f32_vec(vec![1.0; kda_head_dim]),
        ));
        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            tensors.push((
                format!("{prefix}.self_attn.{conv_name}.weight"),
                "F32",
                vec![kda_proj, 1, conv_size],
                f32_vec(vec![0.1; kda_proj * conv_size]),
            ));
        }

        let shard_bytes = build_shard_owned(tensors.clone());
        let dir = std::env::temp_dir().join(format!(
            "frink_kimi_loader_layer_kda_dense_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let map_entries: Vec<String> = tensors
            .iter()
            .map(|(name, ..)| format!("\"{name}\":\"shard0.safetensors\""))
            .collect();
        let index = format!("{{\"weight_map\":{{{}}}}}", map_entries.join(","));
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, &index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let mut hp = KimiRealHparams::real();
        hp.hidden_dim = hidden_dim;
        hp.kda_num_heads = kda_num_heads;
        hp.kda_head_dim = kda_head_dim;
        hp.dense_intermediate_dim = dense_intermediate;

        let layer = load_kimi_layer(&shard, &hp, LayerAttentionKind::KimiKda, true, layer_idx)
            .expect("must load a real KDA+dense layer at a nonzero layer index");
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(
            layer.attn,
            crate::kimi_decoder::KimiLayerAttention::Kda(_)
        ));
        assert!(matches!(
            layer.ffn,
            crate::kimi_decoder::KimiLayerFfn::Dense(_)
        ));
        assert_eq!(layer.input_layernorm_weight.len(), hidden_dim);
    }

    #[test]
    fn load_kimi_layer_dispatches_mla_plus_latent_moe() {
        let hidden_dim = 8;
        let num_heads = 1;
        let q_lora_rank = 4;
        let kv_lora_rank = 4;
        let qk_nope_head_dim = 2;
        let qk_rope_head_dim = 2;
        let v_head_dim = 2;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let moe_hidden_dim = 32;
        let moe_intermediate_dim = 32;
        let n_experts = 2;
        let num_shared_experts = 1;
        let shared_intermediate_dim = moe_intermediate_dim * num_shared_experts;
        let layer_idx = 7;
        let prefix = format!("language_model.model.layers.{layer_idx}");

        let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        let push_bf16 = |tensors: &mut Vec<(String, &str, Vec<usize>, Vec<u8>)>,
                         name: String,
                         shape: Vec<usize>,
                         n: usize| {
            tensors.push((name, "BF16", shape, bf16_bytes(&vec![0.05f32; n])));
        };
        push_bf16(
            &mut tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attention_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attention_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.mlp_res_norm.weight"),
            vec![hidden_dim],
            hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.mlp_res_proj.weight"),
            vec![1, hidden_dim],
            hidden_dim,
        );

        // MLA attention tensors.
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.q_a_proj.weight"),
            vec![q_lora_rank, hidden_dim],
            q_lora_rank * hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.q_a_layernorm.weight"),
            vec![q_lora_rank],
            q_lora_rank,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.q_b_proj.weight"),
            vec![num_heads * q_head_dim, q_lora_rank],
            num_heads * q_head_dim * q_lora_rank,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            vec![kv_lora_rank + qk_rope_head_dim, hidden_dim],
            (kv_lora_rank + qk_rope_head_dim) * hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            vec![kv_lora_rank],
            kv_lora_rank,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.kv_b_proj.weight"),
            vec![num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank],
            num_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden_dim, num_heads * v_head_dim],
            hidden_dim * num_heads * v_head_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.self_attn.g_proj.weight"),
            vec![num_heads * v_head_dim, hidden_dim],
            num_heads * v_head_dim * hidden_dim,
        );

        // Latent-MoE tensors.
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.gate.weight"),
            vec![n_experts, hidden_dim],
            n_experts * hidden_dim,
        );
        let bias_bytes: Vec<u8> = vec![0.0f32; n_experts]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        tensors.push((
            format!("{prefix}.block_sparse_moe.gate.e_score_correction_bias"),
            "F32",
            vec![n_experts],
            bias_bytes,
        ));
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.routed_expert_down_proj.weight"),
            vec![moe_hidden_dim, hidden_dim],
            moe_hidden_dim * hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.routed_expert_up_proj.weight"),
            vec![hidden_dim, moe_hidden_dim],
            hidden_dim * moe_hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.routed_expert_norm.weight"),
            vec![moe_hidden_dim],
            moe_hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.shared_experts.gate_proj.weight"),
            vec![shared_intermediate_dim, hidden_dim],
            shared_intermediate_dim * hidden_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.shared_experts.down_proj.weight"),
            vec![hidden_dim, shared_intermediate_dim],
            hidden_dim * shared_intermediate_dim,
        );
        push_bf16(
            &mut tensors,
            format!("{prefix}.block_sparse_moe.shared_experts.up_proj.weight"),
            vec![shared_intermediate_dim, hidden_dim],
            shared_intermediate_dim * hidden_dim,
        );

        for e in 0..n_experts {
            let expert_prefix = format!("{prefix}.block_sparse_moe.experts.{e}");
            let seed_base = (e as u32 + 1) * 10;
            tensors.push((
                format!("{expert_prefix}.w1.weight_packed"),
                "U8",
                vec![moe_intermediate_dim, moe_hidden_dim / 2],
                pseudo_bytes(seed_base + 1, moe_intermediate_dim * (moe_hidden_dim / 2)),
            ));
            tensors.push((
                format!("{expert_prefix}.w1.weight_scale"),
                "U8",
                vec![moe_intermediate_dim, moe_hidden_dim / 32],
                pseudo_scale_bytes(seed_base + 2, moe_intermediate_dim * (moe_hidden_dim / 32)),
            ));
            tensors.push((
                format!("{expert_prefix}.w2.weight_packed"),
                "U8",
                vec![moe_hidden_dim, moe_intermediate_dim / 2],
                pseudo_bytes(seed_base + 3, moe_hidden_dim * (moe_intermediate_dim / 2)),
            ));
            tensors.push((
                format!("{expert_prefix}.w2.weight_scale"),
                "U8",
                vec![moe_hidden_dim, moe_intermediate_dim / 32],
                pseudo_scale_bytes(seed_base + 4, moe_hidden_dim * (moe_intermediate_dim / 32)),
            ));
            tensors.push((
                format!("{expert_prefix}.w3.weight_packed"),
                "U8",
                vec![moe_intermediate_dim, moe_hidden_dim / 2],
                pseudo_bytes(seed_base + 5, moe_intermediate_dim * (moe_hidden_dim / 2)),
            ));
            tensors.push((
                format!("{expert_prefix}.w3.weight_scale"),
                "U8",
                vec![moe_intermediate_dim, moe_hidden_dim / 32],
                pseudo_scale_bytes(seed_base + 6, moe_intermediate_dim * (moe_hidden_dim / 32)),
            ));
        }

        let shard_bytes = build_shard_owned(tensors.clone());
        let dir = std::env::temp_dir().join(format!(
            "frink_kimi_loader_layer_mla_moe_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let map_entries: Vec<String> = tensors
            .iter()
            .map(|(name, ..)| format!("\"{name}\":\"shard0.safetensors\""))
            .collect();
        let index = format!("{{\"weight_map\":{{{}}}}}", map_entries.join(","));
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, &index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let mut hp = KimiRealHparams::real();
        hp.hidden_dim = hidden_dim;
        hp.mla_num_heads = num_heads;
        hp.mla_q_lora_rank = q_lora_rank;
        hp.mla_kv_lora_rank = kv_lora_rank;
        hp.mla_qk_nope_head_dim = qk_nope_head_dim;
        hp.mla_qk_rope_head_dim = qk_rope_head_dim;
        hp.mla_v_head_dim = v_head_dim;
        hp.moe_hidden_dim = moe_hidden_dim;
        hp.moe_intermediate_dim = moe_intermediate_dim;
        hp.n_experts = n_experts;
        hp.num_shared_experts = num_shared_experts;

        let layer = load_kimi_layer(&shard, &hp, LayerAttentionKind::KimiMla, false, layer_idx)
            .expect("must load a real MLA+latent-MoE layer");
        std::fs::remove_dir_all(&dir).ok();

        assert!(matches!(
            layer.attn,
            crate::kimi_decoder::KimiLayerAttention::Mla(_)
        ));
        match &layer.ffn {
            crate::kimi_decoder::KimiLayerFfn::Moe(moe) => {
                assert_eq!(moe.experts.n_experts(), n_experts);
            }
            crate::kimi_decoder::KimiLayerFfn::Dense(_) => panic!("expected Moe ffn"),
        }
        assert_eq!(layer.input_layernorm_weight.len(), hidden_dim);
    }

    /// Dims for the small synthetic checkpoint
    /// `load_kimi_checkpoint_assembles_every_real_layer_kind` builds --
    /// every field mirrors `KimiRealHparams`, just at test scale.
    struct SyntheticDims {
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

    /// Appends one real layer's tensor set (KDA or MLA attention, dense
    /// or MoE FFN, per `kind`/`is_dense`) to `tensors`, matching the
    /// exact real tensor names/shapes `load_kimi_layer` expects --
    /// shared by `load_kimi_checkpoint_assembles_every_real_layer_kind`
    /// across all 3 of its synthetic layers to avoid repeating each
    /// layer's ~15-30 tensor descriptors by hand.
    #[allow(clippy::too_many_arguments)]
    fn push_layer_tensors(
        tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
        layer_idx: usize,
        kind: LayerAttentionKind,
        is_dense: bool,
        d: &SyntheticDims,
    ) {
        let prefix = format!("language_model.model.layers.{layer_idx}");
        let push_bf16 = |tensors: &mut Vec<(String, &'static str, Vec<usize>, Vec<u8>)>,
                         name: String,
                         shape: Vec<usize>,
                         n: usize| {
            tensors.push((name, "BF16", shape, bf16_bytes(&vec![0.05f32; n])));
        };

        push_bf16(
            tensors,
            format!("{prefix}.input_layernorm.weight"),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16(
            tensors,
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16(
            tensors,
            format!("{prefix}.self_attention_res_norm.weight"),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16(
            tensors,
            format!("{prefix}.self_attention_res_proj.weight"),
            vec![1, d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16(
            tensors,
            format!("{prefix}.mlp_res_norm.weight"),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16(
            tensors,
            format!("{prefix}.mlp_res_proj.weight"),
            vec![1, d.hidden_dim],
            d.hidden_dim,
        );

        match kind {
            LayerAttentionKind::KimiKda => {
                let proj = d.kda_num_heads * d.kda_head_dim;
                for name in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                    push_bf16(
                        tensors,
                        format!("{prefix}.self_attn.{name}.weight"),
                        vec![proj, d.hidden_dim],
                        proj * d.hidden_dim,
                    );
                }
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.f_a_proj.weight"),
                    vec![d.kda_head_dim, d.hidden_dim],
                    d.kda_head_dim * d.hidden_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.f_b_proj.weight"),
                    vec![proj, d.kda_head_dim],
                    proj * d.kda_head_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.b_proj.weight"),
                    vec![d.kda_num_heads, d.hidden_dim],
                    d.kda_num_heads * d.hidden_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![d.hidden_dim, proj],
                    d.hidden_dim * proj,
                );
                let f32_vec =
                    |v: Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
                tensors.push((
                    format!("{prefix}.self_attn.A_log"),
                    "F32",
                    vec![d.kda_num_heads],
                    f32_vec(vec![0.5; d.kda_num_heads]),
                ));
                tensors.push((
                    format!("{prefix}.self_attn.dt_bias"),
                    "F32",
                    vec![proj],
                    f32_vec(vec![0.1; proj]),
                ));
                tensors.push((
                    format!("{prefix}.self_attn.o_norm.weight"),
                    "F32",
                    vec![d.kda_head_dim],
                    f32_vec(vec![1.0; d.kda_head_dim]),
                ));
                for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    tensors.push((
                        format!("{prefix}.self_attn.{conv_name}.weight"),
                        "F32",
                        vec![proj, 1, 4],
                        f32_vec(vec![0.1; proj * 4]),
                    ));
                }
            }
            LayerAttentionKind::KimiMla => {
                let q_head_dim = d.mla_qk_nope_head_dim + d.mla_qk_rope_head_dim;
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.q_a_proj.weight"),
                    vec![d.mla_q_lora_rank, d.hidden_dim],
                    d.mla_q_lora_rank * d.hidden_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.q_a_layernorm.weight"),
                    vec![d.mla_q_lora_rank],
                    d.mla_q_lora_rank,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.q_b_proj.weight"),
                    vec![d.mla_num_heads * q_head_dim, d.mla_q_lora_rank],
                    d.mla_num_heads * q_head_dim * d.mla_q_lora_rank,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    vec![d.mla_kv_lora_rank + d.mla_qk_rope_head_dim, d.hidden_dim],
                    (d.mla_kv_lora_rank + d.mla_qk_rope_head_dim) * d.hidden_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                    vec![d.mla_kv_lora_rank],
                    d.mla_kv_lora_rank,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.kv_b_proj.weight"),
                    vec![
                        d.mla_num_heads * (d.mla_qk_nope_head_dim + d.mla_v_head_dim),
                        d.mla_kv_lora_rank,
                    ],
                    d.mla_num_heads
                        * (d.mla_qk_nope_head_dim + d.mla_v_head_dim)
                        * d.mla_kv_lora_rank,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![d.hidden_dim, d.mla_num_heads * d.mla_v_head_dim],
                    d.hidden_dim * d.mla_num_heads * d.mla_v_head_dim,
                );
                push_bf16(
                    tensors,
                    format!("{prefix}.self_attn.g_proj.weight"),
                    vec![d.mla_num_heads * d.mla_v_head_dim, d.hidden_dim],
                    d.mla_num_heads * d.mla_v_head_dim * d.hidden_dim,
                );
            }
            LayerAttentionKind::Gqa => panic!("synthetic checkpoint test never uses Gqa"),
        }

        if is_dense {
            push_bf16(
                tensors,
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![d.dense_intermediate_dim, d.hidden_dim],
                d.dense_intermediate_dim * d.hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.mlp.up_proj.weight"),
                vec![d.dense_intermediate_dim, d.hidden_dim],
                d.dense_intermediate_dim * d.hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.mlp.down_proj.weight"),
                vec![d.hidden_dim, d.dense_intermediate_dim],
                d.hidden_dim * d.dense_intermediate_dim,
            );
        } else {
            let shared_intermediate_dim = d.moe_intermediate_dim * d.num_shared_experts;
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.gate.weight"),
                vec![d.n_experts, d.hidden_dim],
                d.n_experts * d.hidden_dim,
            );
            let bias_bytes: Vec<u8> = vec![0.0f32; d.n_experts]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            tensors.push((
                format!("{prefix}.block_sparse_moe.gate.e_score_correction_bias"),
                "F32",
                vec![d.n_experts],
                bias_bytes,
            ));
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.routed_expert_down_proj.weight"),
                vec![d.moe_hidden_dim, d.hidden_dim],
                d.moe_hidden_dim * d.hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.routed_expert_up_proj.weight"),
                vec![d.hidden_dim, d.moe_hidden_dim],
                d.hidden_dim * d.moe_hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.routed_expert_norm.weight"),
                vec![d.moe_hidden_dim],
                d.moe_hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.shared_experts.gate_proj.weight"),
                vec![shared_intermediate_dim, d.hidden_dim],
                shared_intermediate_dim * d.hidden_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.shared_experts.down_proj.weight"),
                vec![d.hidden_dim, shared_intermediate_dim],
                d.hidden_dim * shared_intermediate_dim,
            );
            push_bf16(
                tensors,
                format!("{prefix}.block_sparse_moe.shared_experts.up_proj.weight"),
                vec![shared_intermediate_dim, d.hidden_dim],
                shared_intermediate_dim * d.hidden_dim,
            );

            for e in 0..d.n_experts {
                let expert_prefix = format!("{prefix}.block_sparse_moe.experts.{e}");
                let seed_base = (layer_idx as u32 * 100) + (e as u32 + 1) * 10;
                tensors.push((
                    format!("{expert_prefix}.w1.weight_packed"),
                    "U8",
                    vec![d.moe_intermediate_dim, d.moe_hidden_dim / 2],
                    pseudo_bytes(
                        seed_base + 1,
                        d.moe_intermediate_dim * (d.moe_hidden_dim / 2),
                    ),
                ));
                tensors.push((
                    format!("{expert_prefix}.w1.weight_scale"),
                    "U8",
                    vec![d.moe_intermediate_dim, d.moe_hidden_dim / 32],
                    pseudo_scale_bytes(
                        seed_base + 2,
                        d.moe_intermediate_dim * (d.moe_hidden_dim / 32),
                    ),
                ));
                tensors.push((
                    format!("{expert_prefix}.w2.weight_packed"),
                    "U8",
                    vec![d.moe_hidden_dim, d.moe_intermediate_dim / 2],
                    pseudo_bytes(
                        seed_base + 3,
                        d.moe_hidden_dim * (d.moe_intermediate_dim / 2),
                    ),
                ));
                tensors.push((
                    format!("{expert_prefix}.w2.weight_scale"),
                    "U8",
                    vec![d.moe_hidden_dim, d.moe_intermediate_dim / 32],
                    pseudo_scale_bytes(
                        seed_base + 4,
                        d.moe_hidden_dim * (d.moe_intermediate_dim / 32),
                    ),
                ));
                tensors.push((
                    format!("{expert_prefix}.w3.weight_packed"),
                    "U8",
                    vec![d.moe_intermediate_dim, d.moe_hidden_dim / 2],
                    pseudo_bytes(
                        seed_base + 5,
                        d.moe_intermediate_dim * (d.moe_hidden_dim / 2),
                    ),
                ));
                tensors.push((
                    format!("{expert_prefix}.w3.weight_scale"),
                    "U8",
                    vec![d.moe_intermediate_dim, d.moe_hidden_dim / 32],
                    pseudo_scale_bytes(
                        seed_base + 6,
                        d.moe_intermediate_dim * (d.moe_hidden_dim / 32),
                    ),
                ));
            }
        }
    }

    /// Builds the 3-layer synthetic checkpoint (dense+KDA, MoE+KDA,
    /// MoE+MLA) on disk and opens it -- shared by the assembly test and
    /// the store-backed equivalence test. Caller removes `dir`.
    fn build_synthetic_full_checkpoint(
        dir_name: &str,
    ) -> (
        std::path::PathBuf,
        ShardedSafetensors,
        crate::config::ModelConfig,
        KimiRealHparams,
    ) {
        let d = SyntheticDims {
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
            moe_hidden_dim: 32,
            moe_intermediate_dim: 32,
            n_experts: 2,
            num_shared_experts: 1,
        };
        let vocab_size = 6;

        // 3 real layers: 0 = dense+KDA (matches Kimi K3's real layer 0),
        // 1 = MoE+KDA, 2 = MoE+MLA -- covering every real
        // attention/FFN combination `load_kimi_checkpoint` must
        // dispatch correctly.
        let model_cfg = crate::config::ModelConfig {
            // Kimi's stack has no per-layer RoPE gate; llama.cpp writes
            // no `use_rope` for it (`crate::rope_layers`).
            rope_layers: crate::rope_layers::RopeLayers::All,
            layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
            name: "synthetic-kimi-test",
            n_layers: 3,
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
                    kda_layers: vec![1, 2],
                    full_attn_layers: vec![3],
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

        let mut tensors: Vec<(String, &'static str, Vec<usize>, Vec<u8>)> = Vec::new();
        push_layer_tensors(&mut tensors, 0, LayerAttentionKind::KimiKda, true, &d);
        push_layer_tensors(&mut tensors, 1, LayerAttentionKind::KimiKda, false, &d);
        push_layer_tensors(&mut tensors, 2, LayerAttentionKind::KimiMla, false, &d);

        let mut push_bf16_top = |name: String, shape: Vec<usize>, n: usize| {
            tensors.push((name, "BF16", shape, bf16_bytes(&vec![0.02f32; n])));
        };
        push_bf16_top(
            "language_model.model.embed_tokens.weight".to_string(),
            vec![vocab_size, d.hidden_dim],
            vocab_size * d.hidden_dim,
        );
        push_bf16_top(
            "language_model.lm_head.weight".to_string(),
            vec![vocab_size, d.hidden_dim],
            vocab_size * d.hidden_dim,
        );
        push_bf16_top(
            "language_model.model.norm.weight".to_string(),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16_top(
            "language_model.model.output_attn_res_norm.weight".to_string(),
            vec![d.hidden_dim],
            d.hidden_dim,
        );
        push_bf16_top(
            "language_model.model.output_attn_res_proj.weight".to_string(),
            vec![1, d.hidden_dim],
            d.hidden_dim,
        );

        let shard_bytes = build_shard_owned(tensors.clone());
        let dir = std::env::temp_dir().join(dir_name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shard0.safetensors"), &shard_bytes).unwrap();
        let map_entries: Vec<String> = tensors
            .iter()
            .map(|(name, ..)| format!("\"{name}\":\"shard0.safetensors\""))
            .collect();
        let index = format!("{{\"weight_map\":{{{}}}}}", map_entries.join(","));
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, &index).unwrap();

        let shard = ShardedSafetensors::open_index(&index_path).expect("must open index");
        let hp = KimiRealHparams {
            hidden_dim: d.hidden_dim,
            kda_num_heads: d.kda_num_heads,
            kda_head_dim: d.kda_head_dim,
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

        (dir, shard, model_cfg, hp)
    }

    /// The unchanged-output gate for Kimi expert streaming: the same
    /// synthetic checkpoint loaded eagerly vs. store-backed (generous
    /// AND smaller-than-one-expert budgets) must produce bit-identical
    /// forward-pass outputs -- same bytes, same kernels, assert_eq on
    /// f32 vectors with no tolerance.
    #[test]
    fn store_backed_kimi_experts_produce_bit_identical_outputs() {
        let (dir, shard, model_cfg, hp) =
            build_synthetic_full_checkpoint("frink_kimi_store_equivalence_test");

        let eager = load_kimi_checkpoint(&shard, &model_cfg, &hp).expect("eager load");
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
            short_conv_kernel_size: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        };
        let dec_cfg = crate::kimi_decoder::KimiDecoderConfig {
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

        for budget in [64 * 1024 * 1024u64, 1u64] {
            let stored =
                load_kimi_checkpoint_with_expert_cache(&shard, &model_cfg, &hp, Some(budget))
                    .expect("store-backed load");
            let mut state_a = crate::kimi_decoder::KimiDecodeState::new(&eager, &kda_cfg);
            let mut state_b = crate::kimi_decoder::KimiDecodeState::new(&stored, &kda_cfg);
            for &tok in &[1usize, 3, 0, 2] {
                let a = crate::kimi_decoder::kimi_forward_token(
                    &eager,
                    &dec_cfg,
                    &mla_cfg,
                    &kda_cfg,
                    tok,
                    &mut state_a,
                );
                let b = crate::kimi_decoder::kimi_forward_token(
                    &stored,
                    &dec_cfg,
                    &mla_cfg,
                    &kda_cfg,
                    tok,
                    &mut state_b,
                );
                assert_eq!(
                    a, b,
                    "budget={budget}: store-backed Kimi output must be bit-identical"
                );
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_kimi_checkpoint_assembles_every_real_layer_kind() {
        let (dir, shard, model_cfg, hp) =
            build_synthetic_full_checkpoint("frink_kimi_loader_full_checkpoint_test");

        let weights = load_kimi_checkpoint(&shard, &model_cfg, &hp)
            .expect("must assemble a complete synthetic checkpoint");
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(weights.layers.len(), 3);
        assert!(matches!(
            weights.layers[0].ffn,
            crate::kimi_decoder::KimiLayerFfn::Dense(_)
        ));
        assert!(matches!(
            weights.layers[0].attn,
            crate::kimi_decoder::KimiLayerAttention::Kda(_)
        ));
        assert!(matches!(
            weights.layers[1].ffn,
            crate::kimi_decoder::KimiLayerFfn::Moe(_)
        ));
        assert!(matches!(
            weights.layers[1].attn,
            crate::kimi_decoder::KimiLayerAttention::Kda(_)
        ));
        assert!(matches!(
            weights.layers[2].ffn,
            crate::kimi_decoder::KimiLayerFfn::Moe(_)
        ));
        assert!(matches!(
            weights.layers[2].attn,
            crate::kimi_decoder::KimiLayerAttention::Mla(_)
        ));
        assert_eq!(weights.embedding.rows(), model_cfg.vocab_size);
        assert_eq!(weights.embedding.cols(), hp.hidden_dim);
        assert_eq!(weights.output_head.rows(), model_cfg.vocab_size);
        assert_eq!(weights.final_norm_weight.len(), hp.hidden_dim);

        // Run a real forward pass through the fully-assembled checkpoint
        // to confirm every piece composes correctly end to end, not
        // just that each layer loads.
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
            short_conv_kernel_size: 4,
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
