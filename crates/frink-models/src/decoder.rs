//! Generic decoder-only transformer forward pass, assembled from a
//! ModelConfig. Each layer is: RMSNorm -> GQA attention (+RoPE) ->
//! residual -> RMSNorm -> MoE FFN (router + routed experts + shared
//! experts) -> residual. This is the standard decoder block shape
//! shared by the LLaMA/DeepSeek/GLM/Kimi family of open-weight models.
//!
//! Weight loading from a real GGUF checkpoint lives in `loader`
//! (`Decoder::from_gguf`); `Decoder::new_random` builds
//! correctly-shaped, randomly initialized weights so the full pipeline
//! -- embedding lookup, N decoder layers, output head -- can be
//! exercised end to end with real assertions about shapes, finiteness,
//! and determinism, without requiring a multi-hundred-gigabyte
//! checkpoint to be present.

mod attn_block;
#[cfg(feature = "cuda")]
mod cuda_prefill;
#[cfg(feature = "metal")]
mod device_attention;
mod entry;
mod ffn_block;
#[cfg(feature = "metal")]
mod fused_attention;
#[cfg(feature = "metal")]
mod fused_recurrent;
#[cfg(any(feature = "metal", feature = "cuda"))]
mod fused_view;
pub mod kv_window;
mod lm_head;
mod qk_norm;
mod qkv_bias;
mod recurrent_block;
mod rope;

use std::sync::atomic::{AtomicU64, Ordering};

use crate::skip_stream::SkipStream;
pub(crate) use attn_block::KvStep;
use frink_core::attention::{
    causal_gqa_attention_prefill_shared_kv_split, causal_gqa_attention_row,
    causal_gqa_attention_softcap,
};
use frink_core::cache::{KvCache, PagedKvCache, PagedStoreExhausted, SharedPagedKv};
use frink_core::matmul::rms_norm;
pub use kv_window::{KvWindowPolicy, KV_WINDOW_ENV};
#[cfg(feature = "metal")]
use lm_head::FoldedLmHead;
use lm_head::Logits;
use rayon::prelude::*;

/// `Decoder::alibi_slopes` from the config, the ONE derivation, so a
/// constructor cannot carry a bias without its slopes.
pub(crate) fn config_alibi_slopes(config: &ModelConfig) -> Option<Vec<f32>> {
    config
        .alibi_max_bias
        .and_then(|b| frink_core::alibi::slopes(config.n_heads, b))
}

/// Whether the CUDA `gqa_decode` kernel should serve the per-token GQA
/// reduction (`FRINK_CUDA_GQA=1`). Off by default and only compiled with
/// `--features cuda`; the host path is byte-identical when unset.
#[cfg(feature = "cuda")]
fn cuda_gqa_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("FRINK_CUDA_GQA").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}
use frink_core::tensor::Tensor;
use frink_core::weight_matrix::WeightMatrix;
use frink_moe::{
    combine_expert_outputs, route_top_k, run_expert, run_expert_placed, ExpertPlacement,
    ExpertWeights, GluAct, PlacementPlan,
};

use crate::config::ModelConfig;
use crate::norm::NormOp;
use crate::scalar_multipliers::residual_add;

pub struct AttnWeights {
    pub q_proj: WeightMatrix, // [n_heads*head_dim, hidden_dim]
    pub k_proj: WeightMatrix, // [n_kv_heads*head_dim, hidden_dim]
    pub v_proj: WeightMatrix, // [n_kv_heads*head_dim, hidden_dim]
    pub o_proj: WeightMatrix, // [hidden_dim, n_heads*head_dim]
    /// The PRE-attention norm, or [`NormOp::None`] for the
    /// post-norm-only topology (`olmo2` / `exaone4`), which projects
    /// Q/K/V straight off the raw residual. See [`crate::norm`].
    pub norm_weight: NormOp,
    /// OLMoE-style QK-RMSNorm (`attn_q_norm`/`attn_k_norm` GGUF tensors),
    /// applied to the *whole* q_proj/k_proj output (width `n_heads*head_dim`
    /// / `n_kv_heads*head_dim`) before RoPE -- confirmed against
    /// `OlmoeAttention.forward` in `transformers/models/olmoe/modeling_olmoe.py`
    /// (`q_norm(q_proj(x))`, `k_norm(k_proj(x))`, both plain whole-vector
    /// RMSNorm, not per-head). `None` for every model that doesn't ship
    /// these tensors -- absent, not zero/identity-weighted, so existing
    /// presets/fixtures are byte-for-byte unaffected.
    ///
    /// Qwen3 / Gemma3 ship the same tensor names with length `head_dim`
    /// (per-head). Which style is used is selected by
    /// [`ModelConfig::qk_norm_style`] (refined at load from weight length).
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// Qwen2/Qwen2-MoE-family QKV attention bias (`attn_{q,k,v}.bias`
    /// GGUF tensors, real `config.qkv_bias`), added elementwise to the
    /// corresponding projection's output before QK-norm/RoPE -- confirmed
    /// against the real `transformers` source
    /// (`Qwen2MoeAttention.__init__`: `q_proj = nn.Linear(..., bias=
    /// config.qkv_bias)`, same for `k_proj`/`v_proj`; `o_proj` has no
    /// bias). Found as a real, previously-unhandled architecture gap:
    /// frink's generic GGUF loader silently ignored these real tensors
    /// entirely, producing fluent-but-wrong output on a real downloaded
    /// Qwen1.5-MoE checkpoint (same failure class as OLMoE's missing
    /// QK-norm). `None` for every model that doesn't ship these tensors.
    pub q_bias: Option<Vec<f32>>,
    pub k_bias: Option<Vec<f32>>,
    pub v_bias: Option<Vec<f32>>,
    /// Gemma 2+/3 post-attention RMSNorm (`blk.N.post_attention_norm.weight`
    /// / llama.cpp `attn_post_norm`). Applied to attention output before
    /// the residual add. `None` for Llama/Qwen/OLMoE.
    pub post_attn_norm: Option<Vec<f32>>,
    /// Gemma 2+/3 post-FFN RMSNorm (`blk.N.post_ffw_norm.weight`).
    pub post_ffn_norm: Option<Vec<f32>>,
    /// The learned output gate, `blk.N.attn_gate.weight`, applied to the
    /// attention output before `o_proj` (`afmoe`, `laguna`, `step35`).
    /// See [`crate::attn_gate`] for the two axes it varies on and the
    /// one input it always reads. `None` for every architecture whose
    /// graph has no such op; a file carrying the tensor on one of those
    /// is refused as unconsumed rather than gated.
    pub output_gate: Option<crate::attn_gate::AttnGate>,
    /// `blk.N.attn_sinks.weight`, one learned logit per query head that
    /// joins every softmax and contributes nothing to the output
    /// (`ggml_soft_max_add_sinks`, `llama-graph.cpp:2600`).
    ///
    /// Used to live on the gpt-oss side table alone, which spelled the
    /// rule as "arch is gpt-oss". Four llama.cpp graphs pass this
    /// tensor into `build_attn` -- `openai-moe.cpp:115`,
    /// `mimo2.cpp:177`, `dflash.cpp`, `deepseek4.cpp` -- through the
    /// SAME `build_attn_mha` path, so the rule is "the tensor is
    /// present". gpt-oss requires it (`openai-moe.cpp:44`) and
    /// `loader.rs` still refuses a gpt-oss file without one; the fused
    /// Metal launches refuse any layer that has one, by the exhaustive
    /// destructure in `Decoder::metal_attn_view`.
    pub sinks: Option<Vec<f32>>,
    /// BitNet's `blk.N.attn_sub_norm.weight`, `[hidden_dim]`: an RMSNorm
    /// on the attention output BEFORE `o_proj` (`bitnet.cpp:101-106`),
    /// the other side of that matmul from `post_attn_norm`. Loaded
    /// only for a model whose `ModelConfig::block_sub_norms` says so
    /// (`crate::sub_norms`); the fused Metal launches refuse the model
    /// through `metal_can_serve_model` and the layer through the
    /// exhaustive destructure in `Decoder::metal_attn_view`.
    pub attn_sub_norm: Option<Vec<f32>>,
    /// `blk.N.attn_output.scale`, the `{1}` companion `build_lora_mm`
    /// multiplies the attention branch by right after `wo`
    /// (`llama-graph.cpp:1492-1494`; `talkie` writes it on every export,
    /// `crate::weight_scales`). The fused Metal launches refuse a layer
    /// that has one through the destructure in `metal_attn_view`.
    pub o_scale: Option<f32>,
    /// `blk.N.attn_output.bias`, added right after `wo` (and after
    /// `o_scale`, the order `build_attn` has). Loaded for the
    /// architectures whose graph creates the tensor
    /// (`crate::proj_bias::ATTN_OUT_BIAS_CREATORS`), which includes
    /// gpt-oss, whose bias used to live on its side table; `None`
    /// everywhere else, where a present tensor is refused as unread.
    /// The fused Metal launches refuse a layer that has one through
    /// the destructure in `metal_attn_view`.
    pub o_bias: Option<Vec<f32>>,
    /// LFM2's short convolution, `Some` on exactly the layers whose
    /// shape is `AttnShape::ShortConv` (`crate::shortconv`); the four
    /// projections above are empty on such a layer. The host bodies
    /// branch on the SHAPE and reach this field through it; the fused
    /// Metal launches refuse the model (a short-conv model is never
    /// uniform) and the layer (the destructure in `metal_attn_view`).
    pub shortconv: Option<crate::shortconv::ShortConv>,
    /// The state-space block (`crate::ssm_block`): `Some` on exactly the
    /// layers whose shape is `AttnShape::Mamba1` / `Mamba2`, and on
    /// every GQA layer of a `ModelConfig::parallel_ssm` model
    /// (`crate::mamba2::PARALLEL_WITH_ATTENTION`); the same rules as
    /// `shortconv` otherwise.
    pub ssm: Option<crate::ssm_block::SsmBlock>,
    /// `attn_q` is `2 * n_heads * head_dim` wide, each head's `[q, gate]`
    /// interleaved, and `sigmoid(gate)` multiplies the attention output
    /// before `wo` (`crate::attn_gate::Q_INTERLEAVED_GATE_ARCHS`). The
    /// projection stays one matrix; the three host bodies split its
    /// output. The fused Metal launches refuse the layer.
    pub q_gate_interleaved: bool,
}

/// How a layer's routed experts are held. `Resident` is the original
/// always-in-memory form (owned f32 or zero-copy mmap views).
/// `Stored` holds only byte-range layouts; each use acquires the
/// expert's bytes from a bounded, lease-protected
/// `frink_core::expert_store::ExpertStore` shared by every layer
/// (one global byte budget), builds temporary `WeightMatrix` views
/// over the leased buffer (`WeightBytes::Shared`, which pins the
/// cache entry for the views' lifetime), and drops them after the
/// expert runs. Dequantized math over identical bytes is identical,
/// so the two backings are bit-equivalent by construction -- pinned
/// by an integration test against the MoE fixture.
pub enum ExpertBacking {
    Resident(Vec<ExpertWeights>),
    Stored {
        store:
            std::sync::Arc<frink_core::expert_store::ExpertStore<crate::loader::GgufExpertSource>>,
        layouts: Vec<crate::loader::StoredExpertLayout>,
        layer: u32,
    },
}

impl ExpertBacking {
    pub fn n_experts(&self) -> usize {
        match self {
            ExpertBacking::Resident(v) => v.len(),
            ExpertBacking::Stored { layouts, .. } => layouts.len(),
        }
    }
}

pub struct MoeWeights {
    pub router: WeightMatrix, // [n_experts, hidden_dim]
    pub experts: ExpertBacking,
    pub shared_experts: Vec<ExpertWeights>,
    /// Qwen2-MoE-specific: when present, the shared experts' combined
    /// output is scaled by `sigmoid(shared_expert_gate . x)` before
    /// being added to the routed output, instead of added unconditionally
    /// -- confirmed against the real `transformers` source
    /// (`Qwen2MoeSparseMoeBlock.forward`: `shared_expert_output =
    /// F.sigmoid(self.shared_expert_gate(hidden_states)) *
    /// shared_expert_output`) and llama.cpp's real `qwen2moe.cpp`
    /// (`ffn_gate_inp_shexp` dotted against the hidden state, sigmoid,
    /// multiplied into the shared-expert branch before the final add).
    /// Real on-disk shape is `[hidden_dim]` (a `Linear(hidden_dim, 1,
    /// bias=false)`'s weight, flattened -- ggml's real `create_tensor`
    /// call declares it as `{n_embd}`, not a 2D matrix), so this is a
    /// plain owned vector dotted with the normed hidden state directly,
    /// not a `WeightMatrix`. `None` for every other architecture
    /// (DeepSeek-V3's shared experts, for one real confirmed contrast,
    /// add unconditionally with no gate at all).
    pub shared_expert_gate: Option<Vec<f32>>,
    /// The PRE-FFN norm, or [`NormOp::None`] for the post-norm-only
    /// topology (`olmo2` / `exaone4`), which runs the FFN on the raw
    /// post-attention residual. See [`crate::norm`].
    pub norm_weight: NormOp,
    /// DeepSeek-V3's aux-loss-free expert-selection bias, on disk as
    /// `blk.{N}.exp_probs_b.bias` (llama.cpp's `LLM_TENSOR_FFN_EXP_PROBS_B`
    /// -- note the on-disk name has no `ffn_` prefix, `llama-arch.cpp:416`).
    /// It is added to the *selection* score only: the top-k is taken over
    /// `gating(logit) + bias[expert]`, while each winner's combine weight
    /// comes from the unbiased `gating(logit)`
    /// (`build_moe_ffn`: "leave probs unbiased as it's later used to get
    /// expert weights"). Biasing the weight too would silently skew every
    /// routed contribution away from what the router learned.
    ///
    /// `None` for every checkpoint that does not ship the tensor. When it
    /// *is* present, the GPU MoE fast paths refuse the layer rather than
    /// route without it -- their kernels have no bias input.
    pub exp_probs_bias: Option<Vec<f32>>,
    /// BitNet's `blk.N.ffn_sub_norm.weight`, `[ffn_dim]`: an RMSNorm on
    /// `silu(gate) * up` BEFORE `down` (`bitnet.cpp:135-140`), inside
    /// the dense FFN. `Some` only for a model whose
    /// `ModelConfig::block_sub_norms` says so (`crate::sub_norms`), and
    /// only on a dense layer: no MoE graph has this site. Read by
    /// `Decoder::run_dense_expert` and `Decoder::dense_ffn_batch`, the
    /// two dense FFN bodies; the fused kernels never see it because
    /// `metal_can_serve_model` refuses the model.
    pub ffn_sub_norm: Option<Vec<f32>>,
    /// `blk.N.ffn_down.scale`, the `{1}` companion multiplied onto the
    /// dense FFN's output right after `down` (`crate::weight_scales`);
    /// refused on a routed layer, whose experts carry their own.
    pub down_scale: Option<f32>,
    /// The dense FFN's `blk.N.ffn_{up,gate,down}.bias`
    /// (`crate::proj_bias`), on a dense layer whose architecture's
    /// graph creates them and whose file carries at least one; `None`
    /// otherwise. Applied by `run_dense_expert` and `dense_ffn_batch`,
    /// whose fused Metal launch has no bias site and is fenced on it.
    pub dense_bias: Option<frink_moe::DenseBias>,
    /// Arctic's `blk.N.ffn_norm_exps.weight`, `[hidden_dim]`: the SECOND
    /// per-layer norm, applied to the layer INPUT to make the routed
    /// branch's operand (`arctic.cpp:45,136-139`;
    /// `RouterInput::NormedLayerInput`). REQUIRED on a routed layer of
    /// such a model, `None` everywhere else. Read by
    /// `Decoder::router_operand`, the one constructor of the operand.
    pub exps_norm: Option<Vec<f32>>,
    /// The factor on `ffn_out + moe_out` for a layer whose dense FFN is
    /// summed with its experts (`crate::parallel_dense_ffn`): `Some`
    /// only when the loader filled `shared_experts` from the dense
    /// names AND the row scales the sum (`grok.cpp:180`, `sqrt(2)/2`).
    /// Applied by the FFN bodies to the whole branch output before the
    /// post-FFN norm.
    pub parallel_sum_scale: Option<f32>,
    /// `Some` when this layer is a PARALLEL residual, `x + attn(norm(x))
    /// + ffn(norm(x))`, and under which norm the FFN reads the layer
    /// input (`crate::parallel_residual`): the vector attention read
    /// (`SharedNorm`, and then `norm_weight` is `NormOp::None` because
    /// there is no tensor) or its own `ffn_norm` of the layer input
    /// (`TwoNorms`). `None` is the sequential layer, `ffn_norm(h)` over
    /// the post-attention residual. Read by ONE constructor,
    /// `Decoder::branch_inputs`, before attention runs.
    pub parallel: Option<crate::parallel_residual::ParallelNorm>,
    /// How many times each routed expert (index into `experts`) has been
    /// selected by `route_top_k` across every `forward_token`/
    /// `forward_batch` call so far. Real observed hotness, not a
    /// placeholder -- feeds `placement_plan` below, which is what
    /// `PlacementPlan::from_budget` needs to prioritize actually-hot
    /// experts for GPU residency instead of guessing by index.
    pub activation_counts: Vec<AtomicU64>,
    /// Verified-at-load contiguous expert planes for Metal MoE
    /// (`mul_mm_sg` gather/id). Built in `loader` when every routed expert
    /// is mmap-backed with a simdgroup-GEMM quant (Q4_0 / Q4_K / Q8_0 / …)
    /// and back-to-back gate/up/down slices. Gate/up/down kinds may differ
    /// (Qwen1.5-MoE: Q4_K gate/up + Q8_0 down). `None` for store-backed,
    /// F32, or non-contiguous layouts.
    #[cfg(feature = "metal")]
    pub packed_q4: Option<MoePackedQ4Planes>,
}

/// Load-time validated contiguous expert tensor planes (any `mul_mm_sg` quant).
#[cfg(feature = "metal")]
pub struct MoePackedQ4Planes {
    gate: frink_core::weight_matrix::WeightBytes,
    up: frink_core::weight_matrix::WeightBytes,
    down: frink_core::weight_matrix::WeightBytes,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    n_experts: usize,
    ffn_rows: usize,
    hidden_rows: usize,
    gate_row_bytes: usize,
    down_row_bytes: usize,
    gate_kind: &'static str,
    up_kind: &'static str,
    down_kind: &'static str,
}

#[cfg(feature = "metal")]
impl MoePackedQ4Planes {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gate: frink_core::weight_matrix::WeightBytes,
        up: frink_core::weight_matrix::WeightBytes,
        down: frink_core::weight_matrix::WeightBytes,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
        n_experts: usize,
        ffn_rows: usize,
        hidden_rows: usize,
        gate_kind: &'static str,
        up_kind: &'static str,
        down_kind: &'static str,
    ) -> Self {
        Self {
            gate,
            up,
            down,
            gate_stride,
            up_stride,
            down_stride,
            n_experts,
            ffn_rows,
            hidden_rows,
            gate_row_bytes: gate_stride / ffn_rows,
            down_row_bytes: down_stride / hidden_rows,
            gate_kind,
            up_kind,
            down_kind,
        }
    }

    pub fn view(&self) -> frink_metal::gpu::MoePackedQ4<'_> {
        frink_metal::gpu::MoePackedQ4 {
            gate: self.gate.as_slice(),
            up: self.up.as_slice(),
            down: self.down.as_slice(),
            gate_stride: self.gate_stride,
            up_stride: self.up_stride,
            down_stride: self.down_stride,
            n_experts: self.n_experts,
            ffn_rows: self.ffn_rows,
            hidden_rows: self.hidden_rows,
            gate_row_bytes: self.gate_row_bytes,
            down_row_bytes: self.down_row_bytes,
            gate_kind: self.gate_kind,
            up_kind: self.up_kind,
            down_kind: self.down_kind,
        }
    }
}

impl MoeWeights {
    pub fn n_experts(&self) -> usize {
        self.experts.n_experts()
    }

    /// This routed expert's weight byte footprint, from resident
    /// matrices or the stored layout -- identical numbers either way,
    /// so residency planning is backing-independent.
    pub fn expert_bytes(&self, e: usize) -> usize {
        match &self.experts {
            ExpertBacking::Resident(v) => {
                let ex = &v[e];
                ex.gate.resident_bytes() + ex.up.resident_bytes() + ex.down.resident_bytes()
            }
            ExpertBacking::Stored { layouts, .. } => layouts[e].total_bytes(),
        }
    }

    /// Runs `f` against expert `e`'s weights, materializing them from
    /// the store first when this layer is store-backed. The lease (and
    /// therefore the cache entry's pin) lives exactly as long as `f`'s
    /// borrow.
    pub fn with_expert<R>(&self, e: usize, f: impl FnOnce(&ExpertWeights) -> R) -> R {
        match &self.experts {
            ExpertBacking::Resident(v) => f(&v[e]),
            ExpertBacking::Stored {
                store,
                layouts,
                layer,
            } => {
                let lease = store
                    .acquire(frink_core::expert_store::ExpertKey {
                        layer: *layer,
                        expert: e as u32,
                    })
                    .unwrap_or_else(|err| {
                        panic!(
                            "expert store read failed for layer {layer} expert {e}: {err} \
                             (checkpoint file unreadable mid-decode)"
                        )
                    });
                let tmp = layouts[e].materialize(&lease);
                f(&tmp)
            }
        }
    }

    /// The one expert a DENSE layer runs, recorded as the host body
    /// records it, so a fused layer's hotness counters do not depend on
    /// which backend ran it (`crate::fused_layer`).
    #[cfg(feature = "metal")]
    pub(crate) fn record_activations_dense(&self) {
        self.record_activations(&[0]);
    }

    fn record_activations(&self, expert_ids: &[usize]) {
        for &eid in expert_ids {
            if let Some(counter) = self.activation_counts.get(eid) {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// A real VRAM-budget-and-hotness-driven placement plan for this
    /// layer's routed experts, built from each expert's actual resident
    /// byte size (`WeightMatrix::resident_bytes()` summed across its
    /// gate/up/down matrices, so it reflects the real quantization
    /// format in use, not an estimate) and the activation counts
    /// observed so far. See `frink_moe::PlacementPlan::from_budget`.
    pub fn placement_plan(&self, vram_budget_bytes: u64) -> PlacementPlan {
        let sizes: Vec<usize> = (0..self.n_experts())
            .map(|e| self.expert_bytes(e))
            .collect();
        let counts: Vec<u64> = self
            .activation_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let has_observations = counts.iter().any(|&c| c > 0);
        PlacementPlan::from_budget(
            &sizes,
            has_observations.then_some(counts.as_slice()),
            vram_budget_bytes,
        )
    }
}

pub struct LayerWeights {
    pub attn: AttnWeights,
    pub moe: MoeWeights,
    /// Talkie's `blk.N.layer_output_scale.weight`, the scalar its
    /// embedding skip stream is multiplied by before joining this
    /// layer's output (`talkie.cpp:123-126`, `crate::skip_stream`).
    /// `Some` only on a `ModelConfig::skip_stream` model.
    pub out_scale: Option<f32>,
}

/// The per-layer weights the gpt-oss graph carries and the generic GQA
/// layer structs do not.
///
/// Held as a side table on [`Decoder`] rather than as new `Option`
/// fields on [`AttnWeights`]/[`MoeWeights`] for two reasons. The first
/// is mechanical: those two structs have thirty construction sites
/// across seven loaders and every dedicated engine, and none of them
/// will ever set these. The second is the point of the exercise — a
/// checkpoint either has the whole gpt-oss graph or none of it, so
/// `Decoder::gpt_oss.is_some()` is a single, checkable predicate for
/// "this model needs the gpt-oss path", which is what the CPU-only and
/// paged-attention refusals below key off. Scattering four independent
/// `Option`s would make "half the graph is wired" representable, and
/// that state is precisely the silent-wrong-answer bug this work exists
/// to remove.
///
/// The attention sinks used to be the fifth field here and are
/// [`AttnWeights::sinks`] now, because they are NOT gpt-oss-only:
/// `mimo2.cpp:58,177` passes the same tensor into the same
/// `build_attn_mha`. What keeps the whole-graph rule intact is that
/// the loader refuses a gpt-oss file whose sinks are absent.
pub struct GptOssLayer {
    /// `blk.N.ffn_gate_inp.bias`, added to the router logits.
    /// (`attn_output.bias` used to be here too and is
    /// [`AttnWeights::o_bias`] now: `crate::proj_bias` fills that slot
    /// for every graph that creates the tensor, gpt-oss among them.)
    pub router_bias: Vec<f32>,
    /// `blk.N.ffn_{gate,up,down}_exps.bias`, one entry per expert.
    pub expert_bias: Vec<frink_moe::ExpertBias>,
}

/// gpt-oss side table: one entry per layer, in layer order.
pub struct GptOssWeights {
    pub layers: Vec<GptOssLayer>,
}

pub struct Decoder {
    pub config: ModelConfig,
    /// `[vocab_size, hidden_dim]`. A `WeightMatrix` rather than an
    /// eagerly-widened f32 `Tensor`, so a quantized `token_embd.weight`
    /// stays quantized on disk/mmap and token lookup dequantizes one
    /// row at a time (`WeightMatrix::dequant_row`) -- a large-vocab
    /// model's embedding table is multi-GB in f32 and only ever read
    /// row-wise.
    pub embedding: WeightMatrix,
    /// The learned position table, `[context_length, hidden_dim]`, for
    /// a graph that adds one to the embeddings (`crate::position_embd`;
    /// `gpt2`, `starcoder`), read by [`Self::embed_token`] and nothing
    /// else. `None` for every architecture that encodes position by
    /// rotation or not at all.
    pub position_embd: Option<WeightMatrix>,
    /// The norm on the token embeddings before layer 0
    /// (`norm_sites::EMBEDDING_NORM_ARCHITECTURES`: `bloom`'s
    /// `token_embd_norm`, a biased LayerNorm), or [`NormOp::None`] for
    /// every other architecture. Applied in [`Self::embed_token`], the
    /// one embedding site; the GPU embedding gather has no norm and is
    /// not taken for a model that has one.
    pub embedding_norm: NormOp,
    /// HRM-Text's learned LOW stream (`hrm.z_l_init`,
    /// `hrm-text.cpp:46`, one `[n_embd]` row), `None` for every other
    /// architecture. `crate::hrm` is the state the bodies carry.
    pub hrm_z_l_init: Option<Vec<f32>>,
    pub layers: Vec<LayerWeights>,
    /// The norm before the LM head.
    ///
    /// A [`NormOp`] rather than a `Vec<f32>` because `olmo` is the first
    /// architecture whose final norm is not an RMSNorm: `olmo.cpp:128-130`
    /// is `build_norm(cur, NULL, NULL, LLM_NORM, -1)`, and `olmo.cpp:15-36`
    /// creates no `output_norm` tensor for it to weight. The fused Metal
    /// stacks that fold `final_norm + lm_head + argmax` had
    /// `Some(&self.final_norm)` written into them unconditionally; now
    /// they ask [`NormOp::rms_weights`] and fall back to the host body
    /// when there is nothing to hand over.
    pub final_norm: NormOp,
    pub output_head: WeightMatrix, // [vocab_size, hidden_dim]
    /// `output.bias`, `[vocab_size]`, added to the logits right after
    /// the head (`crate::proj_bias::OUTPUT_BIAS_CREATORS`: `phi2` and
    /// `phimoe` REQUIRE it, `qwen2` creates it optional). Applied in
    /// `decoder::lm_head::Logits`, the one place the head's
    /// post-projection transforms run; a head with a bias is never
    /// folded into a fused Metal decode stack (`FoldedLmHead::permit`).
    pub output_bias: Option<Vec<f32>>,
    /// ALiBi's per-head slopes, `[n_heads]`, `Some` exactly when
    /// `ModelConfig::alibi_max_bias` is (`frink_core::alibi::slopes`),
    /// handed to every host attention kernel as its additive per-key
    /// bias. Derived from the config at construction so the two cannot
    /// disagree; no fused GPU path serves a model that has them.
    pub alibi_slopes: Option<Vec<f32>>,
    /// Real VRAM budget for GPU-resident routed experts.
    /// `None` (both constructors below
    /// set it) means every expert always runs on CPU -- the exact
    /// behavior this field's absence had before it existed. `Some(bytes)`
    /// makes each forward call build ONE global `ResidencyPlan`
    /// (`Decoder::residency_plan`) across every layer's actual
    /// resident expert sizes and observed activation counts against
    /// this single budget -- the budget is never re-spent per layer --
    /// dispatching device-placed routed experts through
    /// `frink_moe::run_expert_placed` (a real CUDA kernel when the
    /// `cuda` feature is compiled in and the expert's quant kind has
    /// one; a correct CPU fallback otherwise, so setting this on a
    /// non-`cuda` build is harmless, just never GPU-accelerated).
    /// Shared experts and a dense layer's sole expert always run on
    /// CPU regardless -- every token activates them, so there's no
    /// routing decision to offload the way routed-expert placement is.
    /// Rebuilding the plan on every forward call is real but not yet
    /// performance-tuned; a real, disclosed limit, not a correctness
    /// gap.
    pub gpu_vram_budget_bytes: Option<u64>,
    /// `Some` only for the gpt-oss family. See [`GptOssWeights`]. When
    /// set, every layer runs the gpt-oss CPU graph (attention sinks,
    /// alternating SWA, biased router + experts, `swiglu_oai`), GPU
    /// offload is refused at load time, and the paged-KV decode path is
    /// refused at call time — neither implements sinks, and answering
    /// with a different distribution is the failure this replaces.
    pub gpt_oss: Option<GptOssWeights>,
    /// Does this architecture norm Q and K AFTER RoPE rather than
    /// before? `maincoder` and `hunyuan-moe` do; see [`qk_norm`] for
    /// the llama.cpp lines and for why no GGUF key can answer this.
    /// Set by the loader from the architecture string; refused by
    /// `layer_supports_metal_attn`, because no fused kernel can express
    /// the order.
    pub qk_norm_after_rope: bool,
    /// Per-layer Metal-resident KV for fused decode/prefill attention
    /// (`FRINK_METAL_ATTN`). Lazily allocated. After
    /// [`frink_metal::attn::launch_decode_dense_stack`], Metal KV is
    /// authoritative for the next decode step; host [`KvCache`] may lag
    /// until [`Self::sync_metal_attn_kv_to_host`] or a CPU fallback.
    /// Prefill / prefix restore still upload host → Metal when lengths
    /// diverge for other reasons.
    #[cfg(feature = "metal")]
    pub(crate) metal_attn_kv: std::sync::Mutex<Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
    /// Load-time execution plan (family, fused-op caps, SWA/RoPE
    /// policy). Built once; hot path must not re-resolve architecture
    /// strings. See [`crate::execution_plan`].
    pub execution_plan: crate::execution_plan::ExecutionPlan,
    /// May a windowed layer's host [`KvCache`] drop rows that have
    /// fallen behind its window? Off unless `FRINK_KV_WINDOW` says
    /// otherwise; see [`kv_window`] for what else turns it off. A field
    /// rather than a cached global so a test can run both arms in one
    /// process and compare tokens.
    pub kv_window: KvWindowPolicy,
    /// LoRA adapters attached after load, by id. Each one's deltas
    /// live INSIDE the `WeightMatrix` values above
    /// (`WeightMatrix::Adapted`); this list is what a server lists and
    /// rescales. See [`crate::lora_attach`].
    pub lora_adapters: Vec<crate::lora_attach::LoraAttached>,
    /// Cache key hit → fused caps last used for that geometry (enables
    /// decode/prefill plan reuse without rebuilding residency).
    pub plan_cache: std::sync::Mutex<
        std::collections::HashMap<
            crate::execution_plan::PlanGeometry,
            crate::execution_plan::FusedOpCaps,
        >,
    >,
}

/// Simple deterministic pseudo-random generator so tests are
/// reproducible without pulling in an external `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * 0.1).collect()
    }
}

/// Where a batch of independent sequences keeps its KV.
///
/// `forward_multi_seq` batches the projections across sequences but
/// must attend per sequence, because each has its own length and its
/// own history. That per-sequence step is the ONLY place the batched
/// path touches a cache, which is why paging it is a parameter here
/// rather than a second copy of a 300-line function -- the lesson the
/// paged decode path taught by losing five model features one at a
/// time to exactly that kind of copy.
pub enum MultiSeqKv<'a> {
    Contiguous(&'a mut [Vec<KvCache>]),
    Paged {
        caches: &'a mut [Vec<PagedKvCache>],
        stores: &'a SharedPagedKv,
    },
}

impl MultiSeqKv<'_> {
    /// Sequences in the batch.
    pub fn len(&self) -> usize {
        match self {
            MultiSeqKv::Contiguous(c) => c.len(),
            MultiSeqKv::Paged { caches, .. } => caches.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Layers each sequence carries, for the shape assertion.
    fn layers_per_seq(&self, seq: usize) -> usize {
        match self {
            MultiSeqKv::Contiguous(c) => c[seq].len(),
            MultiSeqKv::Paged { caches, .. } => caches[seq].len(),
        }
    }

    /// Sequence `b`'s layer-`l` cache as the one-row step the attention
    /// and short-conv bodies take. The only place this enum's arms are
    /// matched for a cache, so paging changes where rows live and
    /// nothing else.
    fn step(&mut self, b: usize, l: usize) -> KvStep<'_> {
        match self {
            // `Batched`, not `Decode`: the CUDA resident per-layer KV
            // holds ONE sequence's history, and this path never seeds
            // it. See `KvStep::Batched`.
            MultiSeqKv::Contiguous(caches) => KvStep::Batched(&mut caches[b][l]),
            MultiSeqKv::Paged { caches, stores } => KvStep::Paged {
                cache: &mut caches[b][l],
                stores,
            },
        }
    }
}

impl Decoder {
    /// Eagerly resolve every kernel lookup this model's dispatch paths
    /// will make, and record it in
    /// [`frink_core::kernel_registry`] before anything runs.
    ///
    /// Call once, at the end of loading, immediately before
    /// [`frink_core::kernel_registry::seal`]. Nothing here dispatches
    /// or decides anything: it asks the same predicates the hot path
    /// asks and writes the answers down, so a kernel that is missing
    /// becomes a startup line instead of an unexplained benchmark row.
    ///
    /// Routed experts held in an [`ExpertBacking::Stored`] layer are not
    /// probed -- they exist only as byte ranges until a token routes to
    /// them, and materialising every expert here would defeat the
    /// bounded expert store. Their kinds are the same as the resident
    /// case, and a dispatch-site miss still trips the sealed registry.
    pub fn probe_kernels(&self) {
        use frink_core::kernel_registry as reg;

        if !reg::enabled() {
            return;
        }
        self.embedding.probe_kernels("token_embd");
        self.output_head.probe_kernels("output_head");
        for layer in &self.layers {
            layer.attn.q_proj.probe_kernels("attn_q");
            layer.attn.k_proj.probe_kernels("attn_k");
            layer.attn.v_proj.probe_kernels("attn_v");
            layer.attn.o_proj.probe_kernels("attn_o");
            layer.moe.router.probe_kernels("moe_router");
            for e in &layer.moe.shared_experts {
                e.gate.probe_kernels("shexp_gate");
                e.up.probe_kernels("shexp_up");
                e.down.probe_kernels("shexp_down");
            }
            if let ExpertBacking::Resident(experts) = &layer.moe.experts {
                for e in experts {
                    e.gate.probe_kernels("ffn_gate");
                    e.up.probe_kernels("ffn_up");
                    e.down.probe_kernels("ffn_down");
                }
            }
        }
        // The generic decoder has a real batched prefill
        // (`forward_hidden_batch`), so a `pp512` here is one GEMM per
        // projection, not 512 matvecs. Recorded as a hit so that an
        // engine which lacks it stands out as a miss rather than as an
        // absence.
        reg::record_build(
            reg::Lookup::new(
                frink_core::weight_matrix::active_backend(),
                reg::op::ENGINE_PREFILL_BATCH,
                None,
            )
            .with_role("generic_decoder"),
            reg::Outcome::Hit,
        );
    }

    /// Builds a decoder with correctly-shaped, randomly initialized
    /// weights for `config`, but overrides `n_layers` and `vocab_size`
    /// with small test-scale numbers so it can actually be allocated and
    /// run inside a CI sandbox. Use this to validate the forward-pass
    /// plumbing only, never to draw conclusions about real model
    /// quality.
    pub fn new_random_small(config: ModelConfig, n_layers: usize, vocab_size: usize) -> Self {
        let mut rng = Lcg::new(42);
        let mut config = config;
        config.n_layers = n_layers;
        config.vocab_size = vocab_size;
        let hidden = config.hidden_dim;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;

        let embedding = WeightMatrix::F32(Tensor::new(
            rng.vec(vocab_size * hidden),
            vec![vocab_size, hidden],
        ));

        let wm = |data: Vec<f32>, shape: Vec<usize>| WeightMatrix::F32(Tensor::new(data, shape));

        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let attn = AttnWeights {
                q_proj: wm(
                    rng.vec(n_heads * head_dim * hidden),
                    vec![n_heads * head_dim, hidden],
                ),
                k_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                v_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                o_proj: wm(
                    rng.vec(hidden * n_heads * head_dim),
                    vec![hidden, n_heads * head_dim],
                ),
                norm_weight: NormOp::Rms(vec![1.0; hidden]),
                q_norm: None,
                k_norm: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                post_attn_norm: None,
                post_ffn_norm: None,
                output_gate: None,
                sinks: None,
                attn_sub_norm: None,
                o_scale: None,
                o_bias: None,
                shortconv: None,
                ssm: None,
                q_gate_interleaved: false,
            };

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) get a single-expert, no-shared-expert
            // dense-equivalent FFN regardless of this model's global
            // MoE topology, matching the DeepSeek-2/3-family
            // convention found in ik_llama.cpp's source.
            let is_dense_layer = config.layer_is_dense(layer_idx);
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let n_shared = if is_dense_layer {
                0
            } else {
                config.moe.n_shared_experts
            };
            let ffn_dim = config.moe.expert_ffn_dim;
            let make_expert = |rng: &mut Lcg| ExpertWeights {
                gate: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                up: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                down: WeightMatrix::F32(Tensor::new(
                    rng.vec(hidden * ffn_dim),
                    vec![hidden, ffn_dim],
                )),
            };
            let experts: Vec<ExpertWeights> =
                (0..n_experts).map(|_| make_expert(&mut rng)).collect();
            let shared_experts = (0..n_shared).map(|_| make_expert(&mut rng)).collect();
            let activation_counts = (0..experts.len()).map(|_| AtomicU64::new(0)).collect();

            let moe = MoeWeights {
                exp_probs_bias: None,
                ffn_sub_norm: None,
                down_scale: None,
                exps_norm: None,
                parallel_sum_scale: None,
                parallel: None,
                dense_bias: None,
                router: wm(rng.vec(n_experts * hidden), vec![n_experts, hidden]),
                experts: ExpertBacking::Resident(experts),
                shared_experts,
                shared_expert_gate: None,
                norm_weight: NormOp::Rms(vec![1.0; hidden]),
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4: None,
            };

            layers.push(LayerWeights {
                attn,
                moe,
                out_scale: None,
            });
        }

        let final_norm = NormOp::Rms(vec![1.0; hidden]);
        let output_head = wm(rng.vec(vocab_size * hidden), vec![vocab_size, hidden]);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            crate::capability::DecoderFamily::StandardGqa,
            crate::capability::MemoryKind::KvGqa,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        let alibi_slopes = config_alibi_slopes(&config);
        Decoder {
            config,
            embedding,
            position_embd: None,
            embedding_norm: NormOp::None,
            hrm_z_l_init: None,
            layers,
            final_norm,
            output_head,
            output_bias: None,
            alibi_slopes,
            gpu_vram_budget_bytes: None,
            // Synthetic-weights constructor: no checkpoint, no gpt-oss.
            gpt_oss: None,
            // Synthetic-weights constructor: the preset families it
            // serves all norm before RoPE.
            qk_norm_after_rope: false,
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            kv_window: KvWindowPolicy::from_env(),
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            lora_adapters: Vec::new(),
        }
    }

    /// Metal's launch description for `m`, or `None` when no kernel
    /// serves its storage. Delegates to `crate::metal_launch`, which is
    /// where both this file and `crate::gdn` read it from.
    #[cfg(feature = "metal")]
    fn metal_matvec_launch<'a>(m: &'a WeightMatrix) -> Option<frink_metal::gpu::MatvecLaunch<'a>> {
        crate::metal_launch::matvec(m)
    }
    /// HRM-Text's two streams for a decode of `rows` rows, or `None`
    /// for every other architecture (`crate::hrm`).
    ///
    /// Called by every body that walks layers, so the state is built
    /// in one place from one rule; a body that forgot it would run the
    /// stacks on a single stream and answer fluently.
    pub(crate) fn hrm_streams(&self, embedded: &[f32]) -> Option<crate::hrm::HrmStreams> {
        let init = self.hrm_z_l_init.as_ref()?;
        crate::hrm::hrm_schedule(self.config.layer_loops)?;
        Some(crate::hrm::HrmStreams::new(embedded, init))
    }

    /// The residual layer `l` reads: `zH + zL` at a stack boundary,
    /// and whatever the previous layer produced everywhere else.
    pub(crate) fn hrm_stack_input(
        &self,
        streams: Option<&crate::hrm::HrmStreams>,
        l: usize,
        hidden: &mut Vec<f32>,
    ) {
        let Some(streams) = streams else { return };
        let Some(loops) = self.config.layer_loops else {
            return;
        };
        if loops.stack_starts_at(l) {
            *hidden = streams.stack_input();
        }
    }

    /// Store a finished stack's output in the stream it writes.
    ///
    /// Runs AFTER the FFN body's weightless pass norm
    /// (`Decoder::apply_loop_norm`), because `hrm-text.cpp:162` norms
    /// inside `build_stack` and `:186,193` store what it returned.
    pub(crate) fn hrm_store(
        &self,
        streams: Option<&mut crate::hrm::HrmStreams>,
        l: usize,
        hidden: &[f32],
    ) {
        let Some(streams) = streams else { return };
        let Some(loops) = self.config.layer_loops else {
            return;
        };
        if let Some(stream) = loops.stream_after(l) {
            streams.store(stream, hidden);
        }
    }

    /// The per-model facts no fused Metal kernel implements, as ONE
    /// predicate the four Metal eligibility checks share.
    ///
    /// Five today. `residual_scale`: every fused launch that folds a
    /// residual add in -- the dense decode stack, the resident MoE
    /// decode stack, and both prefill stacks -- adds the branch output
    /// to the stream on device, with no uniform for a multiplier, so a
    /// Granite layer served by any of them would be scaled by the host
    /// bodies and not by the GPU. `clamp_kqv`: the fused launches apply
    /// the QKV bias inside their kernels (`AttnExtras`) and clamp
    /// nothing, so a DBRX or clamped-OLMo layer served by any of them
    /// would run unclamped projections while the host bodies clamp
    /// (`decoder/qkv_bias.rs`). An FFN activation the kernels cannot
    /// spell: every fused dense launch takes a `gelu: bool`, so the
    /// ungated ReLU-squared FFN (`GluAct::Reglu`, arcee) has no value to
    /// pass and every site that asks `fused_kernel_gelu_flag` gets
    /// `None` -- this predicate keeps the whole model off the stacks so
    /// that a layer is never half-served. And per-layer shapes: the
    /// stacks size Q/K/V and the Metal KV plane from ONE head count
    /// (`n_heads` is a launch argument, `MetalKvBuffers` one geometry),
    /// so a model whose layers disagree (`crate::layer_shapes`, deci /
    /// openelm) stays on the host bodies, which read each layer's own.
    /// And the per-position attention temperature
    /// (`crate::attn_temperature`): no fused launch takes a per-token Q
    /// scale, so a Ministral-3 layer served by any of them would attend
    /// at temperature 1 while the host bodies step it with position.
    /// Either way it is the same weights answering differently
    /// depending on which backend took the token, the exact failure
    /// `attention_scale` is fenced off for next door.
    ///
    /// It is one function rather than four spellings because the GPU
    /// router's eligibility check has already drifted four ways in this
    /// file -- prefill tested three conditions, fused decode two, and
    /// the whole-stack decode NONE.
    ///
    /// `lora_attached` is the fifth fact, and the first that is not a
    /// property of the file: a LoRA delta lives inside a
    /// `WeightMatrix::Adapted` and is served by that type's methods,
    /// while every fused stack takes its weights as raw bytes through
    /// `metal_matvec_launch` / `mul_mm_sg_launch` (both answer `None`
    /// for an adapted matrix). Fencing the whole model here, too, is
    /// what keeps a model with an adapter on one layer from being served
    /// half by a stack and half by the host.
    ///
    /// Named for the backend that had fused stacks first. The CUDA
    /// resident prefill layer (`decoder/cuda_prefill.rs`, #259) asks the
    /// same question through `fused_view`, because the facts are
    /// properties of the model and not of the kernel language.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn metal_can_serve_model(config: &ModelConfig, lora_attached: bool) -> bool {
        !lora_attached
            && config.residual_scale.is_none()
            && config.clamp_kqv.is_none()
            && config.attn_temperature.is_none()
            // BitNet's two inner norms (`crate::sub_norms`): no fused
            // kernel norms between attention and `wo`, or between the
            // activation and `down`.
            && !config.block_sub_norms
            // Every fused launch bakes the pre-FFN norm over the
            // POST-ATTENTION residual into its kernel; a parallel layer
            // norms the layer INPUT (`crate::parallel_residual`).
            && !config.parallel_residual
            // The GPU embedding gather has no add and no stack sees
            // `pos` for it (`crate::position_embd`).
            && !config.learned_positions
            // Every fused launch runs attention alone on its layer; a
            // parallel Mamba-2 block (`crate::mamba2`) has no kernel.
            && !config.parallel_ssm
            // Every fused launch takes ONE sliding window per layer; a
            // chunked layer's window is per query
            // (`crate::chunked_swa`), and no fused kernel norms Q and K
            // without a weight after RoPE (`crate::weightless_qk_norm`).
            && !config.swa_chunked
            && !config.weightless_qk_norm
            // No fused kernel adds a per-key bias (`crate::alibi`).
            && config.alibi_max_bias.is_none()
            // Every fused launch takes ONE epsilon for the whole layer;
            // `muse-glimmer.cpp:63` runs its post-norms at a literal
            // 1e-8 and its pre-norms at the model's
            // (`crate::norm::POST_NORM_EPS_LITERAL`), so a model whose
            // two epsilons differ stays on the host rather than having
            // its post-norms silently run at the wrong one.
            && config.post_norm_eps() == config.rms_norm_eps
            // Every fused launch takes ONE head width for K and V (the
            // KV buffers, the attention tile, the `wo` fold); MiMo-V2's
            // split widths stay on the host (`crate::kv_head_dims`).
            && !config.kv_head_dims_split()
            // No fused kernel scales the branch after its `wo` fold
            // (`crate::attn_value_scale`).
            && config.attn_value_scale.is_none()
            // Every fused launch indexes `layers[l]` and `metal_kvs[l]`
            // with ONE `l`; a looped model's logical layers outnumber
            // its weights (`crate::layer_loops`).
            && config.layer_loops.is_none()
            // Talkie: an embedding norm no gather kernel applies, a
            // second residual no stack carries, a per-head SCALAR Q gain
            // and a weightless K norm no `AttnExtras` spells
            // (`crate::skip_stream`, `QkNormStyle::PerHeadScalar`).
            && !config.skip_stream
            && config.qk_norm_style != crate::capability::QkNormStyle::PerHeadScalar
            // `model_ffn_act` is `None` for an activation that varies
            // by layer (xIELU's parameters), which no fused kernel
            // takes; `fused_kernel_gelu_flag` is `None` for one no
            // kernel spells.
            && config
                .model_ffn_act()
                .and_then(GluAct::fused_kernel_gelu_flag)
                .is_some()
            && config.layer_shapes.is_uniform()
            // Each fused launch takes ONE rotary width (`MetalRope::
            // rot_dim`); a model whose sliding layers rotate a
            // different width from its full ones (`rope_dim_swa`,
            // Step-3.5 and Laguna-XS.2) stays on the host bodies, which
            // read each layer's own through `layer_rope`.
            && !config.rope_dim_varies_by_layer()
    }

    /// The blocked prefill attention, on the GPU when this shape is one
    /// the Metal flash kernel serves and on the Rayon host kernel
    /// otherwise, which is the ONE place that choice is made.
    ///
    /// It is reached only by layers the FUSED Metal attention block
    /// cannot take -- a gated softmax attention, a V width that differs
    /// from K's, a projection with no Metal launch -- so the layer's
    /// Q, K and V are on the host either way and there is nothing to
    /// fuse; what is left is whether the `n_q x n_kv` score matrix is
    /// built by six cores or by the GPU. On Bonsai's 16 attention
    /// layers that matrix is the largest single host cost of a prefill
    /// (`dot_f32`, `pv_tile` and `qk_tile` are 11178 of a sampled
    /// prefill's top-of-stack against 1504 for the next thing).
    ///
    /// The refusals are the kernel's, not a guess: it takes no sliding
    /// window, no ALiBi slopes, no attention sink and one head width
    /// for K and V, and a shape it does not serve falls through to the
    /// host body below rather than being approximated.
    #[allow(clippy::too_many_arguments)]
    fn prefill_attention_blocked(
        &self,
        q_batch: &[f32],
        cache_k: &[f32],
        cache_v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        batch_size: usize,
        base_seq_len: usize,
        softcap: Option<f32>,
        window: Option<usize>,
    ) -> Vec<f32> {
        #[cfg(feature = "metal")]
        if window.is_none()
            && self.alibi_slopes.is_none()
            && v_head_dim == head_dim
            && frink_core::weight_matrix::metal_dense_enabled()
        {
            if let Ok(out) = frink_metal::attn::launch_gqa_prefill_host_ex(
                q_batch,
                cache_k,
                cache_v,
                n_heads,
                n_kv_heads,
                head_dim,
                batch_size,
                base_seq_len,
                softcap,
            ) {
                return out;
            }
        }
        causal_gqa_attention_prefill_shared_kv_split(
            q_batch,
            cache_k,
            cache_v,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            batch_size,
            base_seq_len,
            softcap,
            window,
            self.alibi_slopes.as_deref(),
        )
    }

    /// True when this layer can use the fused Metal attention block
    /// (Norm or NeoX RoPE, quantized projections; QKV bias + QK-norm
    /// via [`frink_metal::attn::AttnExtras`]).
    #[cfg(feature = "metal")]
    fn layer_supports_metal_attn(&self, layer: &LayerWeights) -> bool {
        // The backend-neutral questions are `fused_view`'s; what is
        // left is whether Metal has a launch for every projection.
        self.layer_supports_fused_attn(layer)
            && Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// Layer features only the fused dense stack implements — the
    /// per-layer Metal launches would silently skip them (wrong output).
    #[cfg(feature = "metal")]
    fn layer_needs_metal_stack(&self, layer: &LayerWeights, layer_idx: usize) -> bool {
        layer.attn.post_attn_norm.is_some()
            || layer.attn.post_ffn_norm.is_some()
            || self.config.layer_sliding_window(layer_idx).is_some()
            || !self.config.layer_ffn_acts(layer_idx).all_swiglu()
            // `Some(base)` when this layer rotates at the model's own
            // base, so this ONE comparison covers both stack-only RoPE
            // facts: a per-layer base (Gemma-3's sliding layers) and a
            // layer that does not rotate at all (`crate::rope_layers`,
            // EXAONE-4 32B's full-attention layers). A second predicate
            // beside it is exactly the drift this file keeps paying for.
            || self.config.layer_rope_theta(layer_idx) != Some(self.config.rope_theta)
    }

    /// BOTH halves of layer `il`'s RoPE, in the shape the Metal
    /// launches take.
    ///
    /// One conversion, every Metal call site, because
    /// `ModelConfig::layer_rope` returning a pair and the launches
    /// taking a pair is only worth anything while nothing in between
    /// gets to take one half and default the other. That is exactly
    /// what the fused stacks used to do: a per-layer `rope_theta` beside
    /// ONE `freq_factors` slice for the whole run, which refused
    /// Gemma-3 4B/12B/27B off the fused path entirely rather than rope
    /// five layers in six at the wrong scale.
    #[cfg(feature = "metal")]
    fn metal_layer_rope(&self, layer_idx: usize) -> Option<frink_metal::attn::LayerRope<'_>> {
        // Exhaustive on purpose: the third field is the rotary width,
        // which the Metal kernels take as ONE uniform for every layer
        // (`MetalRope::rot_dim`, from `metal_rope`). A layer whose width
        // differs from the model's must never reach a launch, and
        // `metal_can_serve_model` refuses such a model up front; this
        // is the check that the fence held, not a second fence.
        let crate::config::LayerRopeParams {
            theta,
            freq_factors,
            rot_dim,
        } = self.config.layer_rope(layer_idx)?;
        assert_eq!(
            rot_dim,
            self.config.rope_dim.filter(|w| *w < self.config.head_dim),
            "layer {layer_idx} rotates a different width from the model's; \
             `metal_can_serve_model` must have refused this model"
        );
        Some(frink_metal::attn::LayerRope {
            theta,
            freq_factors,
        })
    }

    /// [`Self::fused_attn_extras`] -- the ONE exhaustive destructure of
    /// `AttnWeights`, in `fused_view` -- in Metal's spelling.
    #[cfg(feature = "metal")]
    fn metal_attn_view<'a>(
        &self,
        layer: &'a LayerWeights,
    ) -> Option<frink_metal::attn::AttnExtras<'a>> {
        let fused_view::FusedAttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        } = Self::fused_attn_extras(layer)?;
        Some(frink_metal::attn::AttnExtras {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            attn_logit_softcap: self.config.attn_logit_softcap,
        })
    }

    /// Optional QKV bias / QK-norm ops for the Metal attn paths, for a
    /// layer [`Self::layer_supports_metal_attn`] has admitted. The
    /// `expect` can only fire when a launch site runs without asking
    /// that predicate, which is the drift this file's eligibility
    /// checks exist to stop, and a panic there beats a silent answer
    /// without the gate or the sinks.
    #[cfg(feature = "metal")]
    fn metal_attn_extras<'a>(&self, layer: &'a LayerWeights) -> frink_metal::attn::AttnExtras<'a> {
        self.metal_attn_view(layer)
            .expect("layer_supports_metal_attn admits this layer, so its weights have a Metal view")
    }

    /// GPU expert residency only when Metal attention stays on-device
    /// (when the Metal dense+attn path is active). Avoids CPU-attention
    /// ↔ GPU-expert activation ping-pong on Metal MoE.
    #[cfg(feature = "metal")]
    fn expert_residency_plan(&self, use_metal_attn: bool) -> Option<frink_moe::ResidencyPlan> {
        if frink_core::metal_dense_enabled()
            && frink_metal::attn::metal_attn_enabled()
            && !use_metal_attn
        {
            return None;
        }
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    #[cfg(not(feature = "metal"))]
    fn expert_residency_plan(&self, _use_metal_attn: bool) -> Option<frink_moe::ResidencyPlan> {
        self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b))
    }

    /// Map this checkpoint's RoPE onto the Metal kernel uniforms:
    /// pairing convention, rotary width (`n_rot`), and ggml `rope_yarn`'s
    /// `mscale`. The last two are what
    /// [`Decoder::apply_rope_head_theta`] and
    /// [`Decoder::apply_rope_attn_factor`] do on the CPU side, so the
    /// two backends stay one graph.
    #[cfg(feature = "metal")]
    fn metal_rope(&self) -> frink_metal::attn::MetalRope {
        use crate::config::RopeLayout;
        let layout = match self.config.rope_layout {
            RopeLayout::Norm => frink_metal::attn::MetalRopeLayout::Norm,
            RopeLayout::Neox => frink_metal::attn::MetalRopeLayout::Neox,
        };
        frink_metal::attn::MetalRope {
            layout,
            rot_dim: self
                .config
                .rope_dim
                .filter(|rot| *rot < self.config.head_dim),
            attn_factor: self.config.rope_attn_factor,
        }
    }

    /// Dense FFN (single expert) with Metal-capable gate/up/down.
    #[cfg(feature = "metal")]
    fn layer_supports_metal_dense_ffn(layer: &LayerWeights) -> bool {
        Self::is_dense_layer(layer)
            // No fused FFN kernel scales after `down` (`crate::weight_scales`)
            // or adds a bias anywhere (`crate::proj_bias`).
            && layer.moe.down_scale.is_none()
            && layer.moe.dense_bias.is_none()
            && layer.moe.with_expert(0, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            })
    }

    /// Dense layer eligible for the one-CB `mul_mm_sg` prefill stack.
    /// QKV bias / QK-norm are applied on-GPU via [`AttnExtras`] (same as
    /// decode); SWA fit is checked separately.
    #[cfg(feature = "metal")]
    fn metal_prefill_dense_layer_eligible(
        layer: &LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> bool {
        Self::fused_prefill_dense_layer_eligible(layer, config, lora_attached)
    }

    #[cfg(feature = "metal")]
    fn metal_prefill_dense_swa_fits(
        &self,
        layer_idx: usize,
        start_pos: usize,
        batch_size: usize,
    ) -> bool {
        match self.config.layer_sliding_window(layer_idx) {
            Some(window) => start_pos + batch_size <= window,
            None => true,
        }
    }

    /// Routed-expert FFN for the fused prefill stack, or `None` when this
    /// layer must keep the host-routed path (`launch_moe_prefill_q4_0`).
    ///
    /// Note: routing happens on the GPU here, so prefill no longer feeds
    /// `record_activations`. Expert hotness for `inspect-plan` comes from
    /// decode, which still routes on the host.
    #[cfg(feature = "metal")]
    fn metal_prefill_moe<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> Option<frink_metal::gpu::PrefillMoeMetal<'a>> {
        if !Self::metal_can_serve_model(config, lora_attached)
            || Self::is_dense_layer(layer)
            || !layer.moe.shared_experts.is_empty()
            || !config.model_ffn_act().is_some_and(GluAct::is_swiglu)
            // See `gpu_router_matches_host_routing`: the GPU router
            // takes router weights and nothing else.
            || !Self::gpu_router_matches_host_routing(layer, config)
        {
            return None;
        }
        let frink_core::weight_matrix::WeightMatrix::F32(router) = &layer.moe.router else {
            return None;
        };
        let packed = Self::moe_packed_q4(&layer.moe)?;
        let moe = frink_metal::gpu::PrefillMoeMetal {
            router_w: &router.data,
            top_k: config.moe.n_experts_active,
            renormalize: config.moe.norm_topk_prob,
            packed,
        };
        moe.is_supported().then_some(moe)
    }

    /// FFN half of a fused-prefill-stack layer: dense `mul_mm_sg` launches
    /// or (MoE) the routed-expert description.
    #[cfg(feature = "metal")]
    fn metal_prefill_ffn<'a>(
        layer: &'a LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> Option<frink_metal::attn::PrefillFfnMetal<'a>> {
        if let Some(moe) = Self::metal_prefill_moe(layer, config, lora_attached) {
            return Some(frink_metal::attn::PrefillFfnMetal::Moe(moe));
        }
        if !Self::is_dense_layer(layer) {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        Some(frink_metal::attn::PrefillFfnMetal::Dense {
            gate: ex.gate.mul_mm_sg_launch()?,
            up: ex.up.mul_mm_sg_launch()?,
            down: ex.down.mul_mm_sg_launch()?,
        })
    }

    /// Length of a consecutive run of Metal prefill-stack layers from
    /// `start`, or `None` when fewer than two layers qualify.
    #[cfg(feature = "metal")]
    fn metal_prefill_dense_stack_run_len(
        &self,
        start: usize,
        start_pos: usize,
        batch_size: usize,
        kv_caches: &[KvCache],
        metal_kvs: Option<&[frink_metal::attn::MetalKvBuffers]>,
    ) -> Option<usize> {
        // See `layer_supports_metal_attn`: gpt-oss stays on CPU.
        if self.gpt_oss.is_some() {
            return None;
        }
        let metal_kvs = metal_kvs?;
        let mut run = 0usize;
        for li in start..self.layers.len() {
            let layer = &self.layers[li];
            let cache = &kv_caches[li];
            if !self.metal_prefill_dense_swa_fits(li, start_pos, batch_size) {
                break;
            }
            // POSITIONS: compared against `start_pos`, and against
            // Metal's own count of the same sequence.
            if metal_kvs[li].seq_len != cache.positions() || start_pos != cache.positions() {
                break;
            }
            let ok = layer.attn.q_proj.mul_mm_sg_launch().is_some()
                && layer.attn.k_proj.mul_mm_sg_launch().is_some()
                && layer.attn.v_proj.mul_mm_sg_launch().is_some()
                && layer.attn.o_proj.mul_mm_sg_launch().is_some()
                && Self::metal_prefill_ffn(layer, &self.config, self.lora_attached()).is_some();
            if !ok {
                break;
            }
            run += 1;
        }
        (run >= 2).then_some(run)
    }

    /// Try [`frink_metal::attn::launch_prefill_dense_stack`] for
    /// `run_len` layers starting at `start`. Advances host + Metal KV
    /// on success.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    fn try_metal_prefill_dense_stack(
        &self,
        start: usize,
        run_len: usize,
        hidden_batch: &[f32],
        start_pos: usize,
        batch_size: usize,
        n_heads: usize,
        metal_kvs: &mut [frink_metal::attn::MetalKvBuffers],
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Option<Vec<f32>> {
        // `None` for an activation no fused kernel implements: the
        // stack does not launch, rather than running it as GELU.
        let gelu = self
            .config
            .model_ffn_act()
            .and_then(GluAct::fused_kernel_gelu_flag)?;
        let mut prefill_layers = Vec::with_capacity(run_len);
        for li in start..start + run_len {
            let layer = &self.layers[li];
            let ffn = Self::metal_prefill_ffn(layer, &self.config, self.lora_attached())?;
            if matches!(ffn, frink_metal::attn::PrefillFfnMetal::Dense { .. }) {
                layer.moe.record_activations(&[0]);
            }
            let (q, k, v, o) = (
                layer.attn.q_proj.mul_mm_sg_launch()?,
                layer.attn.k_proj.mul_mm_sg_launch()?,
                layer.attn.v_proj.mul_mm_sg_launch()?,
                layer.attn.o_proj.mul_mm_sg_launch()?,
            );
            prefill_layers.push(frink_metal::attn::PrefillDenseLayerMetal {
                // `?`, not a `&`: the kernel applies the RMSNorm itself
                // and the post-norm-only topology has no weight to give
                // it, so the stack declines and the host body runs. See
                // `crate::norm`.
                attn_norm_w: layer.attn.norm_weight.rms_weights()?,
                ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
                q,
                k,
                v,
                o,
                ffn,
                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                extras: self.metal_attn_extras(layer),
                rope: self.metal_layer_rope(li),
                layer_idx: li as u32,
            });
        }
        let kvs = &mut metal_kvs[start..start + run_len];
        let h_out = frink_metal::attn::launch_prefill_dense_stack(
            hidden_batch,
            &prefill_layers,
            kvs,
            n_heads,
            batch_size,
            self.metal_rope(),
            start_pos,
            self.config.rms_norm_eps,
            gelu,
            self.config.attn_logit_softcap,
        )
        .ok()?;
        for (mkv, cache) in kvs.iter().zip(&mut kv_caches[start..start + run_len]) {
            Self::advance_host_kv_after_metal_prefill(
                mkv,
                cache,
                batch_size,
                host_kv_authoritative,
            );
        }
        Some(h_out)
    }

    /// True when a plain top-k softmax over the raw router logits picks
    /// the SAME experts with the SAME weights that
    /// [`Self::route_for_layer`] would.
    ///
    /// Every Metal MoE path either routes on the GPU (which implements
    /// exactly that plain top-k and takes no other input) or, in
    /// `launch_moe_decode_pre`'s case, used to re-implement it on the
    /// host. `route_for_layer` has three arms this does not: grouped
    /// routing, a per-expert router bias (`exp_probs_bias`), and
    /// `expert_weights_scale`. A checkpoint carrying any of them routes
    /// to DIFFERENT experts with DIFFERENT weights depending on which
    /// backend served the token -- not an error, a different model.
    ///
    /// One predicate rather than the four hand-copied `.is_none()` lists
    /// this used to be, because those lists had already drifted three
    /// ways: the prefill sites checked all three conditions, the fused
    /// decode layer checked two of them, and the whole-stack decode
    /// checked none. Mirror `route_for_layer` arm for arm when either
    /// changes.
    // Read by the three Metal MoE eligibility predicates, and by
    // `the_gpu_router_predicate_admits_only_routing_it_reproduces`. A
    // CPU-only build has no Metal path to gate, so it is dead there.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    fn gpu_router_matches_host_routing(layer: &LayerWeights, config: &ModelConfig) -> bool {
        matches!(config.moe.gating, frink_moe::GatingFunction::Softmax)
            // Every GPU router reads `normed2`; a model whose router
            // reads the raw layer input (`crate::router_input`) would
            // route on the wrong tensor at full speed.
            && config.router_input == crate::router_input::RouterInput::NormedFfnInput
            // Conservative on purpose: `route_for_layer` only takes its
            // grouped arm for `n_groups > 1`, but a checkpoint that
            // declares the key at all is one this kernel was never
            // checked against.
            && config.moe.expert_group_count.is_none()
            && layer.moe.exp_probs_bias.is_none()
            && config.moe.expert_weights_scale == 1.0
    }

    /// MoE layer eligible for resident Metal decode (attn+router+experts
    /// without host residual ping-pong). Requires SwiGLU, no shared
    /// experts, Resident expert backing, Metal router/QKV/O, and a
    /// routing decision the GPU router reproduces exactly.
    #[cfg(feature = "metal")]
    fn layer_supports_metal_moe_resident(
        layer: &LayerWeights,
        config: &ModelConfig,
        lora_attached: bool,
    ) -> bool {
        Self::metal_can_serve_model(config, lora_attached)
            && !Self::is_dense_layer(layer)
            && layer.moe.shared_experts.is_empty()
            && Self::gpu_router_matches_host_routing(layer, config)
            && config.model_ffn_act().is_some_and(GluAct::is_swiglu)
            // Streamed experts are eligible too. They were excluded
            // while the fused launch could not hold all of top-k at
            // once; it can now, by materialising each expert into an
            // owned view that carries its own pin on the store entry.
            && matches!(
                layer.moe.experts,
                ExpertBacking::Resident(_) | ExpertBacking::Stored { .. }
            )
            && Self::metal_matvec_launch(&layer.moe.router).is_some()
            && Self::metal_matvec_launch(&layer.attn.q_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.k_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.v_proj).is_some()
            && Self::metal_matvec_launch(&layer.attn.o_proj).is_some()
    }

    /// One Metal CB for all top-k routed experts (weighted sum). Returns
    /// `None` if any expert lacks a Metal launch (caller falls back).
    #[cfg(feature = "metal")]
    fn try_metal_moe_topk(
        layer: &LayerWeights,
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
    ) -> Option<Vec<f32>> {
        if decision.expert_ids.is_empty() {
            return Some(vec![0f32; normed2.len()]);
        }
        // Build launches while holding each expert briefly; collect owned
        // weight refs via with_expert into temporary MatvecLaunch list.
        let mut launches: Vec<frink_metal::gpu::MoeExpertLaunch<'_>> =
            Vec::with_capacity(decision.expert_ids.len());
        // Lifetime: MatvecLaunch borrows WeightMatrix bytes that live in
        // layer.moe for the duration of this call. Collect via a scoped
        // approach — we need all launches alive together.
        // Use indices + rebuild inside a single with_experts loop.
        struct Pending {
            eid: usize,
            weight: f32,
        }
        let pending: Vec<Pending> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| Pending { eid, weight: w })
            .collect();

        // Validate all experts have Metal launches first.
        for p in &pending {
            let ok = layer.moe.with_expert(p.eid, |ex| {
                Self::metal_matvec_launch(&ex.gate).is_some()
                    && Self::metal_matvec_launch(&ex.up).is_some()
                    && Self::metal_matvec_launch(&ex.down).is_some()
            });
            if !ok {
                return None;
            }
        }

        // A `MatvecLaunch` borrows the expert's bytes, so every expert
        // in the batch has to stay alive until the command buffer is
        // encoded. Resident experts live in a `Vec` and can simply be
        // indexed. Streamed experts used to fall back to the CPU here,
        // on the reasoning that `with_expert` lends one at a time so
        // all of top-k could not be held at once.
        //
        // That was true of `with_expert` and not of the store beneath
        // it: `StoredExpertLayout::materialize` returns an OWNED
        // `ExpertWeights` whose `WeightBytes::Shared` clones the
        // lease's `Arc`, so each one carries its own pin and the store
        // cannot evict it while the view is alive. Materialising all of
        // top-k into a vector that outlives the launches is therefore
        // sound, and the vector is what holds the pins.
        //
        // The fallback was not a small loss. Expert streaming is how a
        // model larger than memory runs at all, so refusing the fused
        // path here meant that turning streaming on silently disabled
        // the Metal MoE kernels: exactly the configuration where the
        // GPU matters most ran on the CPU instead.
        let streamed: Vec<ExpertWeights>;
        match &layer.moe.experts {
            ExpertBacking::Resident(experts) => {
                for p in &pending {
                    let ex = &experts[p.eid];
                    launches.push(frink_metal::gpu::MoeExpertLaunch {
                        gate: Self::metal_matvec_launch(&ex.gate)?,
                        up: Self::metal_matvec_launch(&ex.up)?,
                        down: Self::metal_matvec_launch(&ex.down)?,
                        weight: p.weight,
                    });
                }
            }
            ExpertBacking::Stored {
                store,
                layouts,
                layer: layer_idx,
            } => {
                // Ask for the whole batch before touching any of it, so
                // a miss on the last expert cannot evict the first: the
                // store is bounded, and top-k reads are what compete
                // for it.
                let keys: Vec<_> = pending
                    .iter()
                    .map(|p| frink_core::expert_store::ExpertKey {
                        layer: *layer_idx,
                        expert: p.eid as u32,
                    })
                    .collect();
                store.prefetch(&keys);

                let mut held = Vec::with_capacity(pending.len());
                for (p, key) in pending.iter().zip(keys) {
                    // A read failure here is not fatal: the CPU path
                    // reads the same bytes and will report it. Falling
                    // back beats panicking mid-decode.
                    let lease = store.acquire(key).ok()?;
                    held.push(layouts[p.eid].materialize(&lease));
                }
                streamed = held;

                for (p, ex) in pending.iter().zip(streamed.iter()) {
                    launches.push(frink_metal::gpu::MoeExpertLaunch {
                        gate: Self::metal_matvec_launch(&ex.gate)?,
                        up: Self::metal_matvec_launch(&ex.up)?,
                        down: Self::metal_matvec_launch(&ex.down)?,
                        weight: p.weight,
                    });
                }
            }
        }

        match frink_metal::gpu::launch_moe_topk_swiglu(normed2, &launches) {
            Ok(out) => Some(out),
            Err(e) => {
                eprintln!("frink: Metal MoE top-k fuse failed, falling back: {e}");
                None
            }
        }
    }

    /// Contiguous Q4_0 expert planes for llama-style `mul_mv_id` MoE.
    #[cfg(feature = "metal")]
    fn moe_packed_q4(moe: &MoeWeights) -> Option<frink_metal::gpu::MoePackedQ4<'_>> {
        moe.packed_q4.as_ref().map(MoePackedQ4Planes::view)
    }

    /// Prefill MoE FFN on Metal: host route over T, then one packed-id CB
    /// (`launch_moe_prefill_q4_0`). Shared experts (if any) run as dense
    /// batch FFN on the host/GPU path afterwards — not through `mul_mm_id`.
    /// Returns FFN outs `[T, H]` or `None`.
    #[cfg(feature = "metal")]
    fn try_metal_moe_prefill_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        let acts = config.layer_ffn_acts(layer_idx);
        if batch_size == 0
            || !frink_core::metal_dense_enabled()
            || !acts.routed.is_swiglu()
            // The GPU router kernels take router weights and nothing
            // else: no `exp_probs_b` input, no `expert_weights_scale`
            // uniform, no groups. See `gpu_router_matches_host_routing`.
            || !Self::gpu_router_matches_host_routing(layer, config)
        {
            return None;
        }
        let ExpertBacking::Resident(_) = &layer.moe.experts else {
            return None;
        };
        let packed = Self::moe_packed_q4(&layer.moe)?;
        if !frink_metal::gpu::moe_packed_mul_mv_id_supported(
            packed.gate_kind,
            packed.up_kind,
            packed.down_kind,
        ) {
            return None;
        }
        let top_k = config.moe.n_experts_active;
        if top_k == 0 || top_k > 8 || packed.hidden_rows != hidden_dim {
            return None;
        }
        let n_experts = layer.moe.n_experts().max(1);
        let mut ids = Vec::with_capacity(batch_size * top_k);
        let mut route = Vec::with_capacity(batch_size * top_k);
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = route_top_k(logits, top_k, config.moe.gating, config.moe.norm_topk_prob);
            layer.moe.record_activations(&decision.expert_ids);
            if decision.expert_ids.len() != top_k {
                return None;
            }
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                ids.push(eid as i32);
                route.push(w);
            }
        }
        let mut out = match frink_metal::gpu::launch_moe_prefill_q4_0(
            normed2_batch,
            batch_size,
            &packed,
            &ids,
            &route,
            top_k,
        ) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("frink: Metal MoE prefill failed, CPU fallback: {e}");
                return None;
            }
        };
        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut out,
            // The shared experts are `build_ffn`'s site, not the routed
            // experts' the fence above checked; on the host either way.
            acts.dense,
        );
        Some(out)
    }

    /// Shared expert as dense batch FFN (llama qwen2moe: not through
    /// `mul_mat_id`). Optional sigmoid gate scales per token.
    fn accumulate_shared_experts_batch(
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        hidden_dim: usize,
        acc: &mut [f32],
        act: GluAct,
    ) {
        for shex in &layer.moe.shared_experts {
            // Prefer one Metal FFN CB (gate∥up→SiLU→down) over three
            // `apply_batch` round-trips — Qwen shexp is 4× routed width.
            #[cfg(feature = "metal")]
            let down = if frink_core::metal_dense_enabled() && batch_size >= 4 {
                match (
                    shex.gate.mul_mm_sg_launch(),
                    shex.up.mul_mm_sg_launch(),
                    shex.down.mul_mm_sg_launch(),
                    // The launch's last argument selects GELU over
                    // SiLU inside the kernel; hardcoding `false` here
                    // ran a shared expert as SwiGLU on a GeGLU model,
                    // and `!is_swiglu()` would run a third activation
                    // as GELU. `None` keeps the host path.
                    act.fused_kernel_gelu_flag(),
                ) {
                    (Some(g), Some(u), Some(d), Some(gelu)) => {
                        frink_metal::gpu::launch_dense_ffn_swiglu_batch(
                            &g,
                            &u,
                            &d,
                            normed2_batch,
                            batch_size,
                            gelu,
                        )
                        .ok()
                    }
                    _ => None,
                }
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let down: Option<Vec<f32>> = None;
            // Without `metal` the binding above is a literal `None`; the
            // fallback is the only arm and clippy flags the unwrap.
            #[cfg_attr(not(feature = "metal"), allow(clippy::unnecessary_literal_unwrap))]
            let down = down.unwrap_or_else(|| {
                let ffn_acts = shex.gate.quantize_batch_acts(normed2_batch, batch_size);
                // One rotation for the pair when they share a fold
                // (`apply_batch_pair_with_acts`); two independent calls
                // otherwise, which is every unfolded model.
                let (gate, up) = WeightMatrix::apply_batch_pair_with_acts(
                    &shex.gate,
                    &shex.up,
                    normed2_batch,
                    batch_size,
                    ffn_acts.as_ref(),
                );
                let activated = act.apply(&gate, &up);
                shex.down.apply_batch(&activated, batch_size)
            });
            if let Some(gate_w) = &layer.moe.shared_expert_gate {
                for b in 0..batch_size {
                    let x = &normed2_batch[b * hidden_dim..(b + 1) * hidden_dim];
                    let logit: f32 = gate_w.iter().zip(x.iter()).map(|(g, v)| g * v).sum();
                    let scale = 1.0 / (1.0 + (-logit).exp());
                    let out = &down[b * hidden_dim..(b + 1) * hidden_dim];
                    let row = &mut acc[b * hidden_dim..(b + 1) * hidden_dim];
                    for (a, &o) in row.iter_mut().zip(out.iter()) {
                        *a += scale * o;
                    }
                }
            } else {
                for (a, &o) in acc.iter_mut().zip(down.iter()) {
                    *a += o;
                }
            }
        }
    }

    /// Phase-2 of resident MoE decode: experts on GPU `x2`, add into GPU `h`.
    #[cfg(feature = "metal")]
    fn try_metal_moe_experts_resident(
        layer: &LayerWeights,
        decision: &frink_moe::RoutingDecision,
    ) -> Option<()> {
        if decision.expert_ids.is_empty() {
            return Some(());
        }
        let pending: Vec<(usize, f32)> = decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| (eid, w))
            .collect();
        // Bail on the backing BEFORE validating the experts. The
        // validation loop below calls `with_expert`, which for
        // `Stored` backing acquires a lease and so can read from the
        // checkpoint file. Refusing afterwards meant a streamed layer
        // paid one read per top-k expert and then threw all of them
        // away, before `try_metal_moe_topk` read the same experts
        // again. This path stays resident-only for now, but it must
        // decline for free.
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        for &(eid, _) in &pending {
            let ex = &experts[eid];
            if Self::metal_matvec_launch(&ex.gate).is_none()
                || Self::metal_matvec_launch(&ex.up).is_none()
                || Self::metal_matvec_launch(&ex.down).is_none()
            {
                return None;
            }
        }
        let mut launches = Vec::with_capacity(pending.len());
        for &(eid, weight) in &pending {
            let ex = &experts[eid];
            launches.push(frink_metal::gpu::MoeExpertLaunch {
                gate: Self::metal_matvec_launch(&ex.gate)?,
                up: Self::metal_matvec_launch(&ex.up)?,
                down: Self::metal_matvec_launch(&ex.down)?,
                weight,
            });
        }
        match frink_metal::attn::launch_moe_decode_experts(&launches) {
            Ok(()) => Some(()),
            Err(e) => {
                eprintln!("frink: Metal MoE experts failed, falling back: {e}");
                None
            }
        }
    }

    /// Advance the host [`KvCache`] over the positions a Metal prefill
    /// kernel just wrote to the device.
    ///
    /// Two ways to do that, and which one is right depends on whether
    /// anyone will READ the host rows.
    ///
    /// The contiguous path never does: Metal stays authoritative from
    /// prefill through decode, so [`KvCache::advance_len`]'s zero fill
    /// is a placeholder that only has to keep `seq_len` in step for the
    /// sync checks, and skipping the download is the whole point.
    ///
    /// The PAGED path does. `forward_batch_last_paged` scatters these
    /// rows into the page store, and a caller that reads placeholders
    /// gets a prompt the model never saw -- which is exactly how paged
    /// KV on Metal came to answer fluent nonsense while paged-on-CPU
    /// and contiguous-on-Metal were each correct. So it asks for the
    /// real rows and pays one download per layer, against a gather and
    /// a scatter it was already paying.
    ///
    /// Done here, per layer, immediately after the launch, rather than
    /// once at the end: the Metal KV buffers are dropped outright when
    /// a later layer's launch fails, and rows nobody downloaded before
    /// that are simply gone.
    #[cfg(feature = "metal")]
    fn advance_host_kv_after_metal_prefill(
        mkv: &frink_metal::attn::MetalKvBuffers,
        cache: &mut KvCache,
        batch_size: usize,
        host_kv_authoritative: bool,
    ) {
        if host_kv_authoritative {
            Self::catch_up_host_kv_from_metal(mkv, cache);
            debug_assert_eq!(cache.positions(), mkv.seq_len);
        } else {
            cache
                .advance_len(batch_size)
                .expect("unbounded/planned KvCache growth is infallible");
        }
    }

    /// Append host [`KvCache`] positions that Metal already holds but host
    /// skipped (dense-stack fast path). No-op when `cache.seq_len` is caught up.
    #[cfg(feature = "metal")]
    fn catch_up_host_kv_from_metal(mkv: &frink_metal::attn::MetalKvBuffers, cache: &mut KvCache) {
        // ROWS on both sides: this fills the host buffer with rows
        // Metal already holds, and `push` below advances positions with
        // them. Neither store evicts, so the two agree; when one learns
        // to (#61) this is a place that has to say which it meant.
        if cache.rows() >= mkv.seq_len {
            return;
        }
        let start = cache.rows();
        let n = mkv.seq_len - start;
        let (k, v) = mkv.tokens_host(start, n);
        let per = cache.n_kv_heads * cache.head_dim;
        for i in 0..n {
            let off = i * per;
            cache
                .push(&k[off..off + per], &v[off..off + per])
                .expect("unbounded/planned KvCache growth is infallible");
        }
    }

    /// Pull every layer's Metal-ahead suffix into `kv_caches` (prefix-cache
    /// Poison-tolerant lock for the shared Metal KV arena. A panicked
    /// holder must not permanently brick every later decode.
    #[cfg(feature = "metal")]
    fn lock_metal_attn_kv(
        mutex: &std::sync::Mutex<Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
    ) -> std::sync::MutexGuard<'_, Option<Vec<frink_metal::attn::MetalKvBuffers>>> {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// True once a fused Metal attention launch has allocated this
    /// decoder's per-layer Metal KV, i.e. once a token has actually been
    /// served by a fused launch rather than by the host bodies.
    ///
    /// For tests that switch Metal ON and need to know whether the
    /// switch reached a kernel: a model the eligibility predicates keep
    /// on the host answers `false` forever, and a test asserting logits
    /// alone could not tell that from a launch that happened to agree.
    #[cfg(feature = "metal")]
    pub fn metal_attn_kv_allocated(&self) -> bool {
        Self::lock_metal_attn_kv(&self.metal_attn_kv).is_some()
    }

    /// store, continuous-batch / CPU readers). Safe no-op without Metal KV.
    #[cfg(feature = "metal")]
    pub fn sync_metal_attn_kv_to_host(&self, kv_caches: &mut [KvCache]) {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        let guard = Self::lock_metal_attn_kv(&self.metal_attn_kv);
        let Some(metal_kvs) = guard.as_ref() else {
            return;
        };
        if metal_kvs.len() != kv_caches.len() {
            return;
        }
        for (mkv, cache) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
            Self::catch_up_host_kv_from_metal(mkv, cache);
        }
    }

    /// GQA decode reduction for one token. Uses the CUDA `gqa_decode`
    /// kernel when built with `--features cuda` and `FRINK_CUDA_GQA=1`
    /// (falling back to the host path on any launch error), else the
    /// portable [`causal_gqa_attention`]. With residency enabled the
    /// K/V append stays in [`frink_cuda::attn::CudaKvBuffers`] so only
    /// Q crosses the bus per call (plus a prefix refresh on append).
    #[allow(clippy::too_many_arguments)]
    fn gqa_attention(
        &self,
        layer: usize,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        {
            if cuda_gqa_enabled()
                && self.config.layer_shapes.is_uniform()
                && self.alibi_slopes.is_none()
            {
                match frink_cuda::attn::launch_gqa_decode_resident(
                    layer, q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                ) {
                    Ok(out) => return out,
                    Err(e) => {
                        eprintln!(
                            "frink: CUDA GQA resident decode failed, trying full upload: {e}"
                        );
                    }
                }
                match frink_cuda::attn::launch_gqa_decode(
                    q, k, v, n_heads, n_kv_heads, head_dim, seq_len,
                ) {
                    Ok(out) => return out,
                    Err(e) => {
                        eprintln!("frink: CUDA GQA decode failed, host fallback: {e}");
                    }
                }
            }
        }
        let _ = layer;
        causal_gqa_attention_softcap(
            q,
            k,
            v,
            n_heads,
            n_kv_heads,
            head_dim,
            seq_len,
            self.config.attn_logit_softcap,
        )
    }

    /// The body of [`Self::forward_token`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    fn forward_token_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Vec<f32> {
        // Clear stale dense-stack activation TLS. MoE scratch buffers are
        // reused across tokens (re-seeded); cleared after lm_head below.
        #[cfg(feature = "metal")]
        frink_metal::gpu::clear_resident_activation();

        assert_eq!(kv_caches.len(), self.config.n_layers);
        // Read only by the Metal arms below: the host layer body moved
        // into `attn_block` / `ffn_block_row`, which read the geometry
        // off `self.config` themselves.
        #[cfg(feature = "metal")]
        let hidden_dim = self.config.hidden_dim;
        #[cfg(feature = "metal")]
        let head_dim = self.config.head_dim;
        #[cfg(feature = "metal")]
        let n_heads = self.config.n_heads;
        #[cfg(feature = "metal")]
        let n_kv_heads = self.config.n_kv_heads;

        #[cfg(feature = "metal")]
        let metal_embd_kind = {
            let metal_path = frink_core::metal_dense_enabled()
                && frink_metal::attn::metal_attn_enabled()
                && self
                    .layers
                    .iter()
                    .all(|l| self.layer_supports_metal_attn(l))
                && self.layers.iter().all(Self::layer_supports_metal_dense_ffn);
            // Gemma scales the embedding row (`embedding_scale`) — the GPU
            // gather has no scale op, so dequant + scale on the host; nor
            // has it a norm (`embedding_norm`) or a position table (the
            // latter already fenced through `metal_can_serve_model`).
            if metal_path
                && self.config.embedding_scale.is_none()
                && matches!(self.embedding_norm, NormOp::None)
            {
                Self::metal_matvec_launch(&self.embedding)
                    .and_then(|l| frink_metal::embd::EmbdKind::from_fn_name(l.fn_name))
            } else {
                None
            }
        };
        // `metal_embd_kind` is only `Some` when `embedding_scale` is
        // `None` (the GPU gather has no scale op), so the empty vector
        // this leaves behind is one the scale would not have touched.
        #[cfg(feature = "metal")]
        let mut hidden = if metal_embd_kind.is_some() {
            Vec::new()
        } else {
            self.embed_token(token_id, pos)
        };
        #[cfg(not(feature = "metal"))]
        let mut hidden = self.embed_token(token_id, pos);
        #[cfg(feature = "cuda")]
        if cuda_gqa_enabled()
            && self.config.layer_shapes.is_uniform()
            && self.alibi_slopes.is_none()
        {
            // Fixed capacity so ensure_layer_kv does not recreate (and
            // wipe) mid-sequence as pos grows. ONE geometry for every
            // layer, which is why a per-layer-shape model never seeds it
            // (`gqa_attention` skips the resident hook on the same
            // predicate).
            const CUDA_KV_CAP: usize = 4096;
            if let Err(e) = frink_cuda::attn::ensure_layer_kv(
                self.layers.len(),
                self.config.n_kv_heads,
                self.config.head_dim,
                CUDA_KV_CAP,
            ) {
                eprintln!("frink: CUDA KV residency init failed: {e}");
            }
            if pos == 0 {
                frink_cuda::attn::clear_layer_kv();
            }
        }

        #[cfg(feature = "metal")]
        let use_metal_attn = frink_core::metal_dense_enabled()
            && frink_metal::attn::metal_attn_enabled()
            && self
                .layers
                .iter()
                .all(|l| self.layer_supports_metal_attn(l));

        #[cfg(not(feature = "metal"))]
        let use_metal_attn = false;

        let residency = self.expert_residency_plan(use_metal_attn);

        #[cfg(feature = "metal")]
        let mut metal_kv_guard: Option<
            std::sync::MutexGuard<'_, Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
        > = if use_metal_attn {
            Some(Self::lock_metal_attn_kv(&self.metal_attn_kv))
        } else {
            None
        };

        #[cfg(feature = "metal")]
        if let Some(guard) = metal_kv_guard.as_mut() {
            let need = self.layers.len();
            let cap = kv_caches
                .iter()
                // POSITIONS: sized against `pos`, which is a position.
                .map(|c| c.positions().max(pos + 1).saturating_add(256))
                .max()
                .unwrap_or(512)
                .max(512)
                .max(pos + 1);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    if v.len() != need || v.iter().any(|m| m.capacity() < pos + 1) {
                        // Growing / reshaping: preserve Metal-ahead tokens on host first.
                        if v.len() == need {
                            for (m, c) in v.iter().zip(kv_caches.iter_mut()) {
                                Self::catch_up_host_kv_from_metal(m, c);
                            }
                        }
                        true
                    } else if v.iter().all(|m| m.seq_len == pos) {
                        // Metal already holds tokens [0, pos). Host may lag
                        // after dense-stack decode — do not re-upload from host.
                        false
                    } else {
                        // Stale Metal (new request / prefix restore): rebuild from host.
                        true
                    }
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match frink_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    // Sync from host after CPU prefill / prefix restore / capacity grow.
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
                        // ROWS: this uploads `c.k` / `c.v` themselves, so the
                        // count must describe those buffers.
                        if c.rows() > 0 && m.upload_from_host(&c.k, &c.v, c.rows()).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        **guard = Some(bufs);
                    } else {
                        **guard = None;
                    }
                } else {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let mut metal_stack_done = false;
        #[cfg(feature = "metal")]
        let mut final_norm_done_in_stack = false;
        // OLMoE: all MoE layers in one CB (llama graph style).
        #[cfg(feature = "metal")]
        if use_metal_attn
            && self.layers.iter().enumerate().all(|(i, l)| {
                // Both halves are required. `layer_supports_metal_moe_resident`
                // answers "is this an MoE layer the GPU router can serve",
                // and says nothing about the four features
                // `MoeLayerMetal` has no fields for: a per-layer
                // `rope_theta`, a sliding `window`, `post_attn_norm`
                // and `post_ffn_norm`. `DenseLayerMetal` carries all
                // four and `launch_decode_dense_stack` implements
                // them; the MoE stack does neither, and nothing here
                // refused, so a windowed or sandwich-normed MoE
                // checkpoint would have answered as a different
                // model with no error.
                //
                // The per-layer path already pairs these two checks
                // (see the `metal_moe_resident` branch in the decode
                // loop). Only the whole-stack path was missing it.
                // Latent today because OLMoE and Qwen3-MoE ship none
                // of the four, which is exactly how `attention_scale`
                // stayed latent.
                Self::layer_supports_metal_moe_resident(l, &self.config, self.lora_attached())
                    && !self.layer_needs_metal_stack(l, i)
            })
            && !self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                if let Some(metal_kvs) = guard.as_mut() {
                    if metal_kvs.iter().all(|m| m.seq_len == pos) {
                        let mut moe_layers = Vec::with_capacity(self.layers.len());
                        let mut ok = true;
                        for layer in &self.layers {
                            let ExpertBacking::Resident(_) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let Some(packed) = Self::moe_packed_q4(&layer.moe) else {
                                ok = false;
                                break;
                            };
                            let (Some(q), Some(k), Some(v), Some(o), Some(r), Some(an), Some(fnw)) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&layer.moe.router),
                                // The kernel bakes both RMSNorms in; the
                                // post-norm-only topology has neither.
                                layer.attn.norm_weight.rms_weights(),
                                layer.moe.norm_weight.rms_weights(),
                            ) else {
                                ok = false;
                                break;
                            };
                            moe_layers.push(frink_metal::attn::MoeLayerMetal {
                                attn_norm_w: an,
                                ffn_norm_w: fnw,
                                q,
                                k,
                                v,
                                o,
                                router: r,
                                packed,
                                extras: self.metal_attn_extras(layer),
                            });
                        }
                        if ok {
                            // Greedy: fold lm_head+argmax into the stack like the
                            // dense path does, and download one u32 instead of a
                            // hidden vector.
                            let greedy_gpu = frink_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            // One value carries both "lm_head runs in the
                            // stack" and "the stack returns an argmax id",
                            // so the second cannot drift off the first.
                            // See `decoder::lm_head`.
                            let folded = FoldedLmHead::permit(
                                greedy_gpu,
                                &self.final_norm,
                                self.output_bias.as_deref(),
                                lm_head_gpu_launch,
                            );
                            // The MoE stack used to be handed
                            // `Some(&self.final_norm)` unconditionally,
                            // which is unrepresentable now: a model whose
                            // final norm has no RMS weights leaves the
                            // norm to the host body below, and
                            // `final_norm_done_in_stack` is read off the
                            // SAME value rather than hardcoded `true`.
                            let final_norm_w = self.final_norm.rms_weights();
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales embd on host; GPU gather has no scale.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(frink_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            if embd_gather.is_none() && hidden.is_empty() {
                                hidden = self.embedding.dequant_row(token_id);
                                if let Some(scale) = self.config.embedding_scale {
                                    for v in hidden.iter_mut() {
                                        *v *= scale;
                                    }
                                }
                            }
                            let seed = if embd_gather.is_some() {
                                frink_metal::attn::moe_decode_ensure(hidden_dim)
                            } else {
                                frink_metal::attn::moe_decode_seed(&hidden)
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match seed.and_then(|_| {
                                // A stack whose layer 0 does not rotate
                                // cannot ride this launch; refuse rather
                                // than rotate, and the CPU body serves
                                // the token.
                                let stack_rope = self
                                    .metal_layer_rope(0)
                                    .ok_or(frink_metal::gpu::MetalError::CommandFailed)?;
                                frink_metal::attn::launch_moe_decode_stack(
                                    hidden_ref,
                                    &moe_layers,
                                    metal_kvs,
                                    self.config.moe.n_experts_active,
                                    self.config.moe.norm_topk_prob,
                                    n_heads,
                                    self.metal_rope(),
                                    // ONE `LayerRope` for the whole
                                    // stack, sound because no layer here
                                    // `layer_needs_metal_stack`: none
                                    // slides, none has a base of its
                                    // own, and none is unrotated -- that
                                    // predicate now covers all three
                                    // through the same accessor, so the
                                    // stack-wide answer and the
                                    // per-layer one cannot disagree.
                                    stack_rope,
                                    pos,
                                    self.config.rms_norm_eps,
                                    final_norm_w,
                                    folded.as_ref().map(FoldedLmHead::launch),
                                    folded.as_ref().is_some_and(FoldedLmHead::argmax_only),
                                    true,
                                    embd_gather.as_ref(),
                                )
                            }) {
                                Ok((out, per_layer_ids)) => {
                                    for (layer, ids) in self.layers.iter().zip(per_layer_ids.iter())
                                    {
                                        if !ids.is_empty() {
                                            layer.moe.record_activations(ids);
                                        }
                                    }
                                    if let Some(folded) = folded.as_ref() {
                                        #[cfg(feature = "metal")]
                                        frink_metal::gpu::clear_resident_activation();
                                        // Softcaps anything vocabulary-shaped;
                                        // passes a 1-element argmax id through.
                                        return folded.interpret(
                                            out,
                                            self.output_head.rows(),
                                            self.config.final_logit_softcap,
                                            self.config.logit_multiplier,
                                        );
                                    }
                                    hidden = out;
                                    final_norm_done_in_stack = final_norm_w.is_some();
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "frink: Metal MoE stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metal")]
        if !metal_stack_done
            && use_metal_attn
            && self.layers.iter().all(Self::layer_supports_metal_dense_ffn)
        {
            if let Some(guard) = metal_kv_guard.as_mut() {
                let mut clear_metal_after_stack = false;
                if let Some(metal_kvs) = guard.as_mut() {
                    let seq_ok = metal_kvs.iter().all(|m| m.seq_len == pos);
                    if seq_ok {
                        // Build launches only for resident dense experts (Llama path).
                        let mut dense_layers = Vec::with_capacity(self.layers.len());
                        // The stack's activation uniform, or no stack
                        // at all for an activation it cannot spell.
                        let stack_gelu = self
                            .config
                            .model_ffn_act()
                            .and_then(GluAct::fused_kernel_gelu_flag);
                        let mut ok = stack_gelu.is_some();
                        for (li, layer) in self.layers.iter().enumerate() {
                            let ExpertBacking::Resident(experts) = &layer.moe.experts else {
                                ok = false;
                                break;
                            };
                            let ex = &experts[0];
                            let (
                                Some(q),
                                Some(k),
                                Some(v),
                                Some(o),
                                Some(g),
                                Some(u),
                                Some(d),
                                Some(an),
                                Some(fnw),
                            ) = (
                                Self::metal_matvec_launch(&layer.attn.q_proj),
                                Self::metal_matvec_launch(&layer.attn.k_proj),
                                Self::metal_matvec_launch(&layer.attn.v_proj),
                                Self::metal_matvec_launch(&layer.attn.o_proj),
                                Self::metal_matvec_launch(&ex.gate),
                                Self::metal_matvec_launch(&ex.up),
                                Self::metal_matvec_launch(&ex.down),
                                // The kernel bakes both RMSNorms in; the
                                // post-norm-only topology has neither.
                                layer.attn.norm_weight.rms_weights(),
                                layer.moe.norm_weight.rms_weights(),
                            )
                            else {
                                ok = false;
                                break;
                            };
                            dense_layers.push(frink_metal::attn::DenseLayerMetal {
                                attn_norm_w: an,
                                ffn_norm_w: fnw,
                                q,
                                k,
                                v,
                                o,
                                gate: g,
                                up: u,
                                down: d,
                                extras: self.metal_attn_extras(layer),
                                rope: self.metal_layer_rope(li),
                                window: self.config.layer_sliding_window(li),
                                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                                post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                            });
                        }
                        if ok {
                            // Greedy GPU argmax-in-stack (1×u32 download) when
                            // generate marked this thread for temperature<=0.
                            // Otherwise host lm_head after the hidden download,
                            // which measured ~2x the tok/s of a full-vocab one.
                            let greedy_gpu = frink_metal::attn::metal_greedy_argmax_active();
                            let lm_head_gpu_launch = Self::metal_matvec_launch(&self.output_head);
                            // See `decoder::lm_head`: folding lm_head into
                            // the stack and the stack returning an argmax id
                            // are one decision, held in one value.
                            let folded = FoldedLmHead::permit(
                                greedy_gpu,
                                &self.final_norm,
                                self.output_bias.as_deref(),
                                lm_head_gpu_launch,
                            );
                            // Pass final_norm_w when: (1) lm_head runs in stack (folded),
                            // OR (2) lm_head will route to GPU after stack (lm_head_gpu_launch
                            // but no fold) so we can skip download→reupload via TLS.
                            //
                            // `rms_weights()` is what makes the second
                            // case safe for a non-RMS final norm: it is
                            // `None` there, so the stack does not norm
                            // and the host body does.
                            let final_norm_w = if folded.is_some() || lm_head_gpu_launch.is_some() {
                                self.final_norm.rms_weights()
                            } else {
                                None
                            };
                            let embd_launch = Self::metal_matvec_launch(&self.embedding);
                            // Gemma scales the embedding row on the host
                            // (`hidden` already carries sqrt(hidden_dim));
                            // the GPU gather has no scale op — skip it.
                            let embd_gather = if self.config.embedding_scale.is_some() {
                                None
                            } else {
                                match (metal_embd_kind, embd_launch.as_ref()) {
                                    (Some(kind), Some(launch)) => {
                                        Some(frink_metal::attn::EmbdGatherMetal {
                                            kind,
                                            weights: launch.weights,
                                            rows: launch.rows,
                                            row_bytes: launch.row_bytes,
                                            n_cols: hidden_dim,
                                            token_id,
                                        })
                                    }
                                    _ => None,
                                }
                            };
                            let hidden_ref: &[f32] =
                                if embd_gather.is_some() { &[] } else { &hidden };
                            match frink_metal::attn::launch_decode_dense_stack(
                                hidden_ref,
                                &dense_layers,
                                metal_kvs,
                                n_heads,
                                self.metal_rope(),
                                pos,
                                self.config.rms_norm_eps,
                                final_norm_w,
                                folded.as_ref().map(FoldedLmHead::launch),
                                folded.as_ref().is_some_and(FoldedLmHead::argmax_only),
                                embd_gather.as_ref(),
                                stack_gelu.expect("`ok` is false without it"),
                            ) {
                                Ok(out) => {
                                    // Metal KV advanced in-place. Skip host
                                    // last_token_host+push — host may lag until
                                    // sync_metal_attn_kv_to_host / CPU fallback.
                                    // Dense stack has no MoE routing; skip
                                    // per-layer hotness atomics on the hot path.
                                    if let Some(folded) = folded.as_ref() {
                                        // Skip host final_norm/lm_head. Clear TLS.
                                        #[cfg(feature = "metal")]
                                        frink_metal::gpu::clear_resident_activation();
                                        // `interpret` is what keeps
                                        // `final_logit_softcap` applied: the id
                                        // shape passes through, anything
                                        // vocabulary-shaped gets capped.
                                        return folded.interpret(
                                            out,
                                            self.output_head.rows(),
                                            self.config.final_logit_softcap,
                                            self.config.logit_multiplier,
                                        );
                                    }
                                    // Stack downloaded hidden (possibly normalized if
                                    // final_norm_w was Some). Track whether host should
                                    // skip final_norm.
                                    final_norm_done_in_stack = final_norm_w.is_some();
                                    hidden = out;
                                    metal_stack_done = true;
                                }
                                Err(e) => {
                                    eprintln!(
                                        "frink: Metal dense stack failed, per-layer fallback: {e}"
                                    );
                                    if hidden.is_empty() {
                                        hidden = self.embedding.dequant_row(token_id);
                                        if let Some(scale) = self.config.embedding_scale {
                                            for v in hidden.iter_mut() {
                                                *v *= scale;
                                            }
                                        }
                                    }
                                    // Preserve any prior Metal-ahead tokens on host
                                    // before dropping the device buffers.
                                    for (m, c) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
                                        Self::catch_up_host_kv_from_metal(m, c);
                                    }
                                    clear_metal_after_stack = true;
                                }
                            }
                        }
                    }
                }
                if clear_metal_after_stack {
                    **guard = None;
                }
            }
        }

        #[cfg(feature = "metal")]
        let run_cpu_layers = !metal_stack_done;
        #[cfg(not(feature = "metal"))]
        let run_cpu_layers = true;

        // When true, residual lives in Metal MoE scratch — host `hidden` is stale.
        #[cfg(feature = "metal")]
        let mut metal_moe_resident = false;

        #[cfg(feature = "metal")]
        if run_cpu_layers && hidden.is_empty() && !metal_moe_resident {
            // GPU embedding gather or a skipped Metal dense stack can leave
            // `hidden` empty; CPU fallback must not call rms_norm on it.
            hidden = self.embed_token(token_id, pos);
        }

        // The skip source: the (normed) embedding, before any layer
        // touches `hidden` (`crate::skip_stream`). Captured here and
        // not at `embed_token`, because the Metal arms above leave
        // `hidden` empty on purpose -- and a skip-stream model never
        // reaches them (`metal_can_serve_model`).
        let skip_rows = self.config.skip_stream.then(|| hidden.clone());
        let mut hrm = self.hrm_streams(&hidden);
        if run_cpu_layers {
            // Indexed rather than iterated, because a RUN of consecutive
            // recurrent layers is submitted together
            // (`frink_metal::gdn_branch::GdnRun`) and needs several
            // caches at once. `fused_through` is how many layers that
            // consumed; every `continue` below is a `for`, so it cannot
            // spin.
            // `fused_through` is only ever written under `metal`; the
            // binding is unconditional so the loop reads the same either
            // way.
            #[allow(unused_mut, unused_assignments)]
            let mut fused_through = 0usize;
            // The Q/K/V a recurrent run encoded for the attention layer
            // that follows it, carried one iteration so that layer does
            // not project them again in a submission of its own.
            #[cfg(feature = "metal")]
            let mut pending_qkv: crate::decoder::fused_recurrent::PendingQkv = None;
            #[allow(clippy::needless_range_loop)]
            for l in 0..kv_caches.len() {
                self.hrm_stack_input(hrm.as_ref(), l, &mut hidden);
                if l < fused_through {
                    continue;
                }
                // Consecutive layers the fused launch serves whole pass
                // their residual stream to each other on the device and
                // wait ONCE, which is the submission count the decode
                // gap is made of (`docs/plans/gdn-resident-state.md`).
                #[cfg(feature = "metal")]
                if let Some(end) =
                    self.fused_recurrent_run(l, &mut hidden, kv_caches, &mut pending_qkv)
                {
                    fused_through = end;
                    continue;
                }
                let cache = &mut kv_caches[l];
                let layer = self.layer_for(l);
                // --- attention block ---
                #[cfg(feature = "metal")]
                if metal_moe_resident
                    && (!Self::layer_supports_metal_moe_resident(
                        layer,
                        &self.config,
                        self.lora_attached(),
                    ) || self.layer_needs_metal_stack(layer, l))
                {
                    if let Some(h) = frink_metal::attn::moe_decode_take_hidden() {
                        hidden = h;
                    }
                    metal_moe_resident = false;
                }

                // The router's operand, captured where llama.cpp reads it:
                // BEFORE attention (`smallthinker.cpp:111`). A resident
                // Metal stack never serves that shape
                // (`gpu_router_matches_host_routing`), so `hidden` is not
                // stale for the one architecture that reads it here.
                let inputs = self.branch_inputs(layer, &hidden, 1);
                #[cfg(feature = "metal")]
                let normed = if metal_moe_resident {
                    // Residual is on-device; host rms_norm would use stale hidden.
                    Vec::new()
                } else {
                    layer
                        .attn
                        .norm_weight
                        .apply(&hidden, self.config.rms_norm_eps)
                };
                #[cfg(not(feature = "metal"))]
                let normed = layer
                    .attn
                    .norm_weight
                    .apply(&hidden, self.config.rms_norm_eps);

                #[cfg(feature = "metal")]
                {
                    let mut did_metal_attn = false;
                    let mut did_metal_dense = false;
                    let mut did_metal_moe = false;
                    let mut clear_metal_kv = false;
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // Metal-authoritative: host may lag after dense-stack skip.
                            // Stack-only features (SWA / sandwich norms / GeGLU /
                            // per-layer theta) are NOT encoded by the per-layer
                            // launches — those layers must go to CPU here.
                            if metal_kvs[l].seq_len == pos
                                && !self.layer_needs_metal_stack(layer, l)
                            {
                                // The two pre-norm weights join the four
                                // projection launches in ONE gate, because every
                                // fused launch below bakes the RMSNorm into its
                                // kernel and there is no weight to bake for the
                                // post-norm-only topology (`olmo2` / `exaone4`).
                                // `NormOp::rms_weights` returning `None` sends
                                // the whole layer to the host body, which reads
                                // the raw residual the way llama.cpp does.
                                // `metal_layer_rope` joins the six because
                                // these four launches all rope
                                // unconditionally, and a layer llama.cpp
                                // does not rotate must reach the host
                                // body instead. It is the same `None`
                                // `layer_needs_metal_stack` reads two
                                // lines up, from the same accessor, so
                                // the two cannot disagree about which
                                // layers those are.
                                if let (
                                    Some(q_l),
                                    Some(k_l),
                                    Some(v_l),
                                    Some(o_l),
                                    Some(attn_norm_w),
                                    Some(ffn_norm_w),
                                    Some(layer_rope),
                                ) = (
                                    Self::metal_matvec_launch(&layer.attn.q_proj),
                                    Self::metal_matvec_launch(&layer.attn.k_proj),
                                    Self::metal_matvec_launch(&layer.attn.v_proj),
                                    Self::metal_matvec_launch(&layer.attn.o_proj),
                                    layer.attn.norm_weight.rms_weights(),
                                    layer.moe.norm_weight.rms_weights(),
                                    self.metal_layer_rope(l),
                                ) {
                                    // Full dense layer on one CB when FFN is Metal-capable.
                                    if Self::layer_supports_metal_dense_ffn(layer) {
                                        let dense_ok = layer.moe.with_expert(0, |ex| {
                                        let (Some(g_l), Some(u_l), Some(d_l)) = (
                                            Self::metal_matvec_launch(&ex.gate),
                                            Self::metal_matvec_launch(&ex.up),
                                            Self::metal_matvec_launch(&ex.down),
                                        ) else {
                                            return false;
                                        };
                                        match frink_metal::attn::launch_decode_dense_layer(
                                            &hidden,
                                            attn_norm_w,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            ffn_norm_w,
                                            &g_l,
                                            &u_l,
                                            &d_l,
                                            n_heads,
                                            self.metal_rope(),
                                            layer_rope,
                                            pos,
                                            self.config.rms_norm_eps,
                                            &self.metal_attn_extras(layer),
                                        ) {
                                            Ok(new_h) => {
                                                // Catch up any dense-stack lag + this token.
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                layer.moe.record_activations(&[0]);
                                                hidden = new_h;
                                                true
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                    "frink: Metal dense layer failed, CPU fallback: {e}"
                                                );
                                                false
                                            }
                                        }
                                    });
                                        if dense_ok {
                                            did_metal_dense = true;
                                            did_metal_attn = true;
                                        } else if metal_kvs[l].seq_len != cache.rows() {
                                            // Dense path may have advanced Metal KV before failing.
                                            Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                                            clear_metal_kv = true;
                                        }
                                    }

                                    // Resident MoE: attn+router on GPU, host top-k only,
                                    // then batched experts — no hidden download/upload.
                                    if !did_metal_dense
                                        && !clear_metal_kv
                                        && Self::layer_supports_metal_moe_resident(
                                            layer,
                                            &self.config,
                                            self.lora_attached(),
                                        )
                                    {
                                        if let Some(router_l) =
                                            Self::metal_matvec_launch(&layer.moe.router)
                                        {
                                            let seed_ok = if metal_moe_resident {
                                                true
                                            } else {
                                                match frink_metal::attn::moe_decode_seed(&hidden) {
                                                    Ok(()) => {
                                                        metal_moe_resident = true;
                                                        true
                                                    }
                                                    Err(e) => {
                                                        eprintln!(
                                                            "frink: Metal MoE seed failed: {e}"
                                                        );
                                                        false
                                                    }
                                                }
                                            };
                                            if seed_ok {
                                                // Prefer one-CB fused path (GPU top-k + packed experts).
                                                // See
                                                // `layer_supports_metal_moe_resident`:
                                                // the fused decode kernel
                                                // routes on the GPU and
                                                // has no `exp_probs_b` /
                                                // `expert_weights_scale`
                                                // input either.
                                                let fused_ok = match &layer.moe.experts {
                                                    ExpertBacking::Resident(_) => {
                                                        if let Some(packed) =
                                                            Self::moe_packed_q4(&layer.moe)
                                                        {
                                                            match frink_metal::attn::launch_moe_decode_layer_fused(
                                                                attn_norm_w,
                                                                &q_l,
                                                                &k_l,
                                                                &v_l,
                                                                &o_l,
                                                                &mut metal_kvs[l],
                                                                ffn_norm_w,
                                                                &router_l,
                                                                &packed,
                                                                self.config.moe.n_experts_active,
                                                                self.config.moe.norm_topk_prob,
                                                                n_heads,
                                                                self.metal_rope(),
                                                                layer_rope,
                                                                pos,
                                                                self.config.rms_norm_eps,
                                                                &self.metal_attn_extras(layer),
                                                            ) {
                                                                Ok(ids) => {
                                                                    layer.moe.record_activations(&ids);
                                                                    did_metal_moe = true;
                                                                    did_metal_attn = true;
                                                                    true
                                                                }
                                                                Err(e) => {
                                                                    eprintln!(
                                                                        "frink: Metal MoE fused layer failed: {e}"
                                                                    );
                                                                    false
                                                                }
                                                            }
                                                        } else {
                                                            false
                                                        }
                                                    }
                                                    _ => false,
                                                };

                                                if !fused_ok {
                                                    match frink_metal::attn::launch_moe_decode_pre(
                                                        attn_norm_w,
                                                        &q_l,
                                                        &k_l,
                                                        &v_l,
                                                        &o_l,
                                                        &mut metal_kvs[l],
                                                        ffn_norm_w,
                                                        &router_l,
                                                        n_heads,
                                                        self.metal_rope(),
                                                        layer_rope,
                                                        pos,
                                                        self.config.rms_norm_eps,
                                                        &self.metal_attn_extras(layer),
                                                    ) {
                                                        Ok(logits) => {
                                                            // Routing happens HERE, on the host,
                                                            // so there is no kernel limitation to
                                                            // excuse a second router: call the
                                                            // one every other host path calls.
                                                            let decision = Self::route_for_layer(
                                                                layer,
                                                                &logits,
                                                                &self.config,
                                                            );
                                                            layer.moe.record_activations(
                                                                &decision.expert_ids,
                                                            );
                                                            if let Some(()) = Self::try_metal_moe_experts_resident(
                                                            layer,
                                                            &decision,
                                                        ) {
                                                            did_metal_moe = true;
                                                            did_metal_attn = true;
                                                        } else if let Some(h) =
                                                            frink_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                            metal_moe_resident = false;
                                                            // KV already advanced; finish FFN on host.
                                                            let normed2 = layer
                                                                .moe
                                                                .norm_weight
                                                                .apply(
                                                                &hidden,
                                                                self.config.rms_norm_eps,
                                                            );
                                                            let ffn_out = Self::combine_ffn_outputs_for_position(
                                                                l,
                                                                layer,
                                                                &normed2,
                                                                &normed2,
                                                                &logits,
                                                                &self.config,
                                                                hidden_dim,
                                                                residency.as_ref().map(|p| p.layer_plan(self.physical_index(l))),
                                                            );
                                                            residual_add(
                                                                &mut hidden,
                                                                &ffn_out,
                                                                self.config.residual_scale,
                                                            );
                                                            did_metal_attn = true;
                                                            did_metal_moe = true; // skip second FFN
                                                        }
                                                        }
                                                        Err(e) => {
                                                            eprintln!(
                                                            "frink: Metal MoE pre failed, fallback: {e}"
                                                        );
                                                            if let Some(h) =
                                                            frink_metal::attn::moe_decode_take_hidden()
                                                        {
                                                            hidden = h;
                                                        }
                                                            metal_moe_resident = false;
                                                            if metal_kvs[l].seq_len != cache.rows()
                                                            {
                                                                Self::catch_up_host_kv_from_metal(
                                                                    &metal_kvs[l],
                                                                    cache,
                                                                );
                                                                clear_metal_kv = true;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if !did_metal_dense && !did_metal_moe && !clear_metal_kv {
                                        match frink_metal::attn::launch_decode_attn_block(
                                            &normed,
                                            &q_l,
                                            &k_l,
                                            &v_l,
                                            &o_l,
                                            &mut metal_kvs[l],
                                            n_heads,
                                            self.metal_rope(),
                                            layer_rope,
                                            pos,
                                            &self.metal_attn_extras(layer),
                                            self.config.rms_norm_eps,
                                        ) {
                                            Ok(projected) => {
                                                // Keep Metal KV authoritative — skip per-layer
                                                // host catch-up (dense-stack style). Host is
                                                // flushed on CPU fallback / prefix sync.
                                                residual_add(
                                                    &mut hidden,
                                                    &projected,
                                                    self.config.residual_scale,
                                                );
                                                did_metal_attn = true;
                                            }
                                            Err(e) => {
                                                eprintln!(
                                                "frink: Metal attn block failed, CPU fallback: {e}"
                                            );
                                                Self::catch_up_host_kv_from_metal(
                                                    &metal_kvs[l],
                                                    cache,
                                                );
                                                clear_metal_kv = true;
                                            }
                                        }
                                    }
                                }
                            } else if metal_kvs[l].seq_len > cache.rows() {
                                // Leaving Metal path: host must see full KV for CPU attn.
                                Self::catch_up_host_kv_from_metal(&metal_kvs[l], cache);
                            }
                        }
                        if clear_metal_kv {
                            **guard = None;
                        }
                    }
                    if did_metal_attn {
                        if !did_metal_dense && !did_metal_moe {
                            self.ffn_block_row(
                                l,
                                layer,
                                &mut hidden,
                                None,
                                residency
                                    .as_ref()
                                    .map(|p| p.layer_plan(self.physical_index(l))),
                                inputs,
                                skip_rows.as_deref().map(|rows| SkipStream { rows }),
                            );
                        }
                        continue;
                    }
                }

                let oai = self
                    .gpt_oss
                    .as_ref()
                    .map(|g| &g.layers[self.physical_index(l)]);
                // A recurrent layer whose whole shape the fused Metal
                // launch serves runs END TO END in one submission --
                // branch, residual, norm, FFN, residual -- and this
                // loop moves to the next layer. Everything it does not
                // serve it refuses (`crate::fused_layer`,
                // `decoder::fused_recurrent`), and the two bodies below
                // run exactly as before.
                #[cfg(feature = "metal")]
                if let Some(out) =
                    self.fused_recurrent_layer(l, layer, &normed, &hidden, &mut cache.recurrent)
                {
                    hidden = out;
                    cache
                        .advance_len(1)
                        .expect("unbounded/planned KvCache growth is infallible");
                    continue;
                }
                // An ATTENTION layer whose tail the fused launch serves
                // hands `wo` over to it, so `wo`, the residual add, the
                // FFN norm and the FFN are ONE command buffer instead
                // of two (`crate::decoder::fused_attention`). The
                // attention itself still runs on the host: the KV lives
                // there.
                #[cfg(feature = "metal")]
                let mut deferred: Option<Vec<f32>> = None;
                #[cfg(feature = "metal")]
                let mut ready: Option<attn_block::AttnReady> = None;
                #[cfg(feature = "metal")]
                let residual_in = hidden.clone();
                #[cfg(feature = "metal")]
                let tail = if self.fused_attention_tail_eligible(l, layer) {
                    attn_block::AttnTail::Defer {
                        branch: &mut deferred,
                        // The attention itself goes to the device when
                        // this layer's shape allows it; otherwise only
                        // the tail is fused and the host attends.
                        ready: self
                            .device_attention_shape_ok(l, layer)
                            .then_some(&mut ready),
                    }
                } else {
                    attn_block::AttnTail::apply()
                };
                #[cfg(not(feature = "metal"))]
                let tail = attn_block::AttnTail::apply();
                // The projections the preceding run encoded for THIS
                // layer, if it did: then nothing here projects again.
                #[cfg(feature = "metal")]
                let precomputed = match pending_qkv.take() {
                    Some((at, qkv)) if at == l => Some(qkv),
                    // A run encoded a head for a different layer than
                    // the one we reached, which only a refusal in
                    // between can cause: drop it rather than use it.
                    _ => None,
                };
                #[cfg(not(feature = "metal"))]
                let precomputed = None;
                let projected = self.attn_block_tail(
                    l,
                    layer,
                    &normed,
                    pos,
                    KvStep::Decode(&mut *cache),
                    tail,
                    precomputed,
                );
                // The attention layer's inputs came back instead of its
                // output: run the whole layer on the device, at the head
                // of the next recurrent run when there is one.
                #[cfg(feature = "metal")]
                if let Some(r) = ready {
                    let gate = r.gate.as_deref();
                    if let Some(end) = self.device_attention_layer_then_run(
                        l,
                        layer,
                        &r.q,
                        &r.k,
                        &r.v,
                        gate,
                        &mut hidden,
                        kv_caches,
                        &mut pending_qkv,
                    ) {
                        fused_through = end;
                        continue;
                    }
                    let cache = &mut kv_caches[l];
                    if let Some(out) = self.device_attention_layer(
                        l,
                        layer,
                        cache,
                        &r.q,
                        &r.k,
                        &r.v,
                        gate,
                        &residual_in,
                    ) {
                        cache
                            .push(&r.k, &r.v)
                            .expect("unbounded/planned KvCache growth is infallible");
                        hidden = out;
                        continue;
                    }
                    // Both device shapes refused after the layer had
                    // already been prepared, so finish it on the host
                    // from the very same inputs.
                    let mut attn_out = self.push_and_attend_row(
                        KvStep::Decode(&mut kv_caches[l]),
                        l,
                        layer,
                        &r.k,
                        &r.v,
                        &r.q,
                    );
                    if let Some(g) = gate {
                        crate::attn_gate::apply_interleaved_gate(&mut attn_out, g);
                    }
                    let projected = self.project_attn_rows(layer, &attn_out, 1);
                    residual_add(&mut hidden, &projected, self.config.residual_scale);
                    self.ffn_block_row(
                        l,
                        layer,
                        &mut hidden,
                        oai,
                        residency
                            .as_ref()
                            .map(|p| p.layer_plan(self.physical_index(l))),
                        inputs,
                        skip_rows.as_deref().map(|rows| SkipStream { rows }),
                    );
                    continue;
                }
                #[cfg(feature = "metal")]
                if let Some(branch) = deferred {
                    // The tail feeds the recurrent layers after it, so
                    // it rides in THEIR command buffer when there are
                    // any: one wait for the four layers instead of two.
                    if let Some(end) = self.fused_attention_tail_then_run(
                        l,
                        layer,
                        &branch,
                        &mut hidden,
                        kv_caches,
                        &mut pending_qkv,
                    ) {
                        fused_through = end;
                        continue;
                    }
                    if let Some(out) = self.fused_attention_tail(l, layer, &branch, &hidden) {
                        hidden = out;
                        continue;
                    }
                    // The launch refused after the predicate said yes,
                    // which only a device error does: finish the layer
                    // on the host from the branch it handed back.
                    let p = self.project_attn_rows(layer, &branch, 1);
                    residual_add(&mut hidden, &p, self.config.residual_scale);
                }
                if let Some(projected) = projected {
                    residual_add(&mut hidden, &projected, self.config.residual_scale);
                }
                self.ffn_block_row(
                    l,
                    layer,
                    &mut hidden,
                    oai,
                    residency
                        .as_ref()
                        .map(|p| p.layer_plan(self.physical_index(l))),
                    inputs,
                    skip_rows.as_deref().map(|rows| SkipStream { rows }),
                );
                self.hrm_store(hrm.as_mut(), l, &hidden);
            }
        } // run_cpu_layers

        #[cfg(feature = "metal")]
        if metal_moe_resident {
            if let Some(h) = frink_metal::attn::moe_decode_take_hidden() {
                hidden = h;
            }
        }

        // If Metal stack already ran final_norm, hidden is normalized; else
        // normalize here.
        #[cfg(feature = "metal")]
        let final_normed = if final_norm_done_in_stack {
            hidden.clone()
        } else {
            self.final_norm.apply(&hidden, self.config.rms_norm_eps)
        };
        #[cfg(not(feature = "metal"))]
        let final_normed = self.final_norm.apply(&hidden, self.config.rms_norm_eps);

        let logits = self.logits_from_normed(&final_normed);
        // Clear dense-stack activation TLS after lm_head (may have consumed it).
        // Keep MoE scratch buffers alive across tokens — `moe_decode_seed`
        // overwrites `h` each token; clearing here forced full realloc.
        #[cfg(feature = "metal")]
        frink_metal::gpu::clear_resident_activation();
        logits
    }

    /// The body of [`Self::forward_token_paged`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    fn forward_token_paged_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        assert_eq!(stores.layer_count(), self.config.n_layers);
        // All layers advance or none do. Pushing per layer with `?` and
        // failing at layer 3 of 4 leaves layers 0..2 holding a position
        // the rest do not, and nothing downstream reports it: the next
        // step simply attends over a shorter history in the tail
        // layers. Reserving one position everywhere first turns that
        // into a clean refusal.
        //
        // The guards span the check AND the push for the same reason
        // the prefill path holds them: otherwise another request takes
        // the blocks in between.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, 1) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            // Reserve by taking the blocks now, so the per-layer pushes
            // below cannot fail. `PagedKvCache::reserve` grows the block
            // table without advancing `seq_len`, leaving each push a
            // pure write into a block this sequence already owns.
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, 1)
                    .expect("checked against free_block_count under this same guard");
            }
        }

        let mut hidden = self.embed_token(token_id, pos);
        let skip_rows = self.config.skip_stream.then(|| hidden.clone());
        let mut hrm = self.hrm_streams(&hidden);
        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for (l, cache) in kv_caches.iter_mut().enumerate() {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden);
            let layer = self.layer_for(l);
            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden, 1);
            let normed = layer
                .attn
                .norm_weight
                .apply(&hidden, self.config.rms_norm_eps);

            // The same body the contiguous path runs, with the paged
            // backing as its one parameter. It used to be a copy, and
            // the copy had silently dropped `attention_scale`,
            // `post_attn_norm`, `post_ffn_norm`, gpt-oss's `o_bias` and
            // `gpt_oss_ffn` -- five features that each produce a
            // plausible distribution rather than an error.
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);
            if let Some(projected) = self.attn_block(
                l,
                layer,
                &normed,
                pos,
                KvStep::Paged {
                    cache: &mut *cache,
                    stores,
                },
            ) {
                residual_add(&mut hidden, &projected, self.config.residual_scale);
            }
            self.ffn_block_row(
                l,
                layer,
                &mut hidden,
                oai,
                residency
                    .as_ref()
                    .map(|p| p.layer_plan(self.physical_index(l))),
                inputs,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden);
        }

        let final_normed = self.final_norm.apply(&hidden, self.config.rms_norm_eps);
        Ok(self.logits_from_normed(&final_normed))
    }

    /// The shared expert store's live counters, when this model runs
    /// with store-backed (streamed) routed experts -- `None` for fully
    /// resident models. Every store-backed layer shares one store, so
    /// the first one found speaks for the whole model.
    pub fn expert_store_stats(&self) -> Option<frink_core::expert_store::ExpertStoreStats> {
        self.layers.iter().find_map(|l| match &l.moe.experts {
            ExpertBacking::Stored { store, .. } => Some(store.stats()),
            ExpertBacking::Resident(_) => None,
        })
    }

    /// Builds one global device-residency plan across ALL layers'
    /// routed experts against the single configured VRAM budget --
    /// every `(layer, expert)` candidate competes in one hotness-
    /// ordered pass and the running byte total is shared, so the
    /// budget cannot be re-spent per layer (the accounting bug the
    /// earlier per-layer `placement_plan` calls had: N layers would
    /// plan N x the configured bytes). Dense layers contribute no
    /// candidates (their sole expert always runs on CPU). Rebuilt per
    /// forward call so it tracks observed hotness; not yet
    /// performance-tuned, a disclosed limit.
    fn residency_plan(&self, vram_budget_bytes: u64) -> frink_moe::ResidencyPlan {
        let mut sizes_per_layer: Vec<Vec<usize>> = Vec::with_capacity(self.layers.len());
        let mut counts_per_layer: Vec<Vec<u64>> = Vec::with_capacity(self.layers.len());
        let mut any_observed = false;
        for layer in &self.layers {
            if Self::is_dense_layer(layer) {
                sizes_per_layer.push(Vec::new());
                counts_per_layer.push(Vec::new());
                continue;
            }
            sizes_per_layer.push(
                (0..layer.moe.n_experts())
                    .map(|e| layer.moe.expert_bytes(e))
                    .collect(),
            );
            let counts: Vec<u64> = layer
                .moe
                .activation_counts
                .iter()
                .map(|c| c.load(Ordering::Relaxed))
                .collect();
            any_observed |= counts.iter().any(|&c| c > 0);
            counts_per_layer.push(counts);
        }
        PlacementPlan::plan_layers_against_global_budget(
            &sizes_per_layer,
            any_observed.then_some(counts_per_layer.as_slice()),
            vram_budget_bytes,
        )
    }

    /// True if this layer has nothing to route: exactly one expert and
    /// no shared experts, the shape every non-MoE model (and every
    /// DeepSeek-style "leading dense layer") loads as. Top-1 selection
    /// out of one expert always picks it, and its weight is always
    /// exactly 1.0 regardless of gating function (softmax over one
    /// logit is trivially 1.0; sigmoid-then-renormalize divides the
    /// selected score by itself) -- so skipping the router matmul,
    /// `route_top_k`'s sort/exp/renormalize work, and
    /// `combine_expert_outputs`'s Vec-wrapping for this case is not an
    /// approximation, it produces the exact same result.
    fn is_dense_layer(layer: &LayerWeights) -> bool {
        layer.moe.n_experts() == 1 && layer.moe.shared_experts.is_empty()
    }

    /// llama.cpp `mul_mat_id` style: shared Q8 act + one flat parallel
    /// region over `(slot, row_pair)` for gate∥up (2-row SDOT), then
    /// SwiGLU, then per-slot down. Three regions, none nested, all
    /// through `frink_core::par` so they follow whichever scheduler
    /// `FRINK_CPU_POOL` selected.
    fn cpu_moe_topk_parallel_slots(
        experts: &[ExpertWeights],
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
        hidden_dim: usize,
        act: GluAct,
    ) -> Option<Vec<(Vec<f32>, f32)>> {
        if !frink_core::weight_matrix::cpu_int_dot_for(
            frink_core::weight_matrix::IntDotShape::Matvec,
        ) || !normed2.len().is_multiple_of(32)
        {
            return None;
        }
        let n_slots = decision.expert_ids.len();
        if n_slots == 0 {
            return Some(Vec::new());
        }
        for &eid in &decision.expert_ids {
            let ex = experts.get(eid)?;
            if ex.gate.rows() == 0
                || ex.up.rows() != ex.gate.rows()
                || ex.down.rows() != hidden_dim
                || ex.gate.cols() != normed2.len()
                || ex.up.cols() != normed2.len()
                || ex.down.cols() != ex.gate.rows()
            {
                return None;
            }
            if !matches!(
                &ex.gate,
                WeightMatrix::Quantized {
                    kind: frink_core::QuantKind::Q4_0 | frink_core::QuantKind::Q8_0,
                    ..
                }
            ) || !matches!(
                &ex.up,
                WeightMatrix::Quantized {
                    kind: frink_core::QuantKind::Q4_0 | frink_core::QuantKind::Q8_0,
                    ..
                }
            ) {
                return None;
            }
        }
        let ffn_rows = experts[decision.expert_ids[0]].gate.rows();
        // Even ffn_rows: par_chunks_mut(2) never crosses a slot boundary.
        if !ffn_rows.is_multiple_of(2) {
            return None;
        }
        let q8 = frink_quant::quantize_activations_q8(normed2);
        let eids = &decision.expert_ids;
        let mut gate = vec![0f32; n_slots * ffn_rows];
        let mut up = vec![0f32; n_slots * ffn_rows];
        frink_core::par::chunks_mut2(&mut gate, &mut up, 2, 1, |p, gc, uc| {
            let row0 = p * 2;
            let slot = row0 / ffn_rows;
            let r = row0 % ffn_rows;
            let ex = &experts[eids[slot]];
            if let (Some((g0, g1)), Some((u0, u1))) = (
                ex.gate.dot_pair_cpu_q8(r, &q8),
                ex.up.dot_pair_cpu_q8(r, &q8),
            ) {
                gc[0] = g0;
                gc[1] = g1;
                uc[0] = u0;
                uc[1] = u1;
            } else {
                gc[0] = ex.gate.dot_row_cpu_q8(r, &q8).unwrap_or(0.0);
                gc[1] = ex.gate.dot_row_cpu_q8(r + 1, &q8).unwrap_or(0.0);
                uc[0] = ex.up.dot_row_cpu_q8(r, &q8).unwrap_or(0.0);
                uc[1] = ex.up.dot_row_cpu_q8(r + 1, &q8).unwrap_or(0.0);
            }
        });
        let mut activated = vec![0f32; n_slots * ffn_rows];
        // The combine here is always parallel (decode's `n_slots *
        // ffn_rows` sits under `frink_core::matmul`'s own fork
        // threshold), which is why this does not just call `act.apply`.
        // `GluAct::combine` is that function one element at a time --
        // this used to match the variant here and multiply `f(gate) *
        // up` itself, a second spelling of the activation that a
        // parameterised variant (xIELU reads `up` alone) could not fit.
        frink_core::par::items_mut(&mut activated, 1, |idx, a| {
            *a = act.combine(gate[idx], up[idx])
        });
        let mut outs: Vec<(Vec<f32>, f32)> = decision
            .weights
            .iter()
            .map(|&w| (vec![0f32; hidden_dim], w))
            .collect();
        frink_core::par::items_mut(&mut outs, 1, |slot, (out, _)| {
            let ex = &experts[eids[slot]];
            let act_slot = &activated[slot * ffn_rows..(slot + 1) * ffn_rows];
            if act_slot.len().is_multiple_of(32) {
                let down_q8 = frink_quant::quantize_activations_q8(act_slot);
                if let Some(d) = ex.down.apply_cpu_q8(&down_q8) {
                    *out = d;
                    return;
                }
            }
            *out = ex.down.apply(act_slot);
        });
        Some(outs)
    }

    /// Fallback: serial top-k with shared Q8 act (pre-mul_mat_id path).
    ///
    /// `weight_before_ffn` (`crate::routed_weight_site`): each slot
    /// reads its own scaled input and the shared activation is not
    /// built.
    fn cpu_moe_serial_experts(
        layer: &LayerWeights,
        normed2: &[f32],
        decision: &frink_moe::RoutingDecision,
        plan: Option<&PlacementPlan>,
        act: GluAct,
        weight_before_ffn: bool,
    ) -> Vec<(Vec<f32>, f32)> {
        let shared_act = if !weight_before_ffn
            && frink_core::weight_matrix::cpu_int_dot_for(
                frink_core::weight_matrix::IntDotShape::Matvec,
            )
            && normed2.len().is_multiple_of(32)
            && plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true)
        {
            Some(frink_quant::quantize_activations_q8(normed2))
        } else {
            None
        };
        decision
            .expert_ids
            .iter()
            .zip(decision.weights.iter())
            .map(|(&eid, &w)| {
                let placement = plan
                    .map(|p| p.placement_for(eid))
                    .unwrap_or(ExpertPlacement::Cpu);
                let (input, w) =
                    crate::routed_weight_site::routed_slot(normed2, w, weight_before_ffn);
                let out = layer.moe.with_expert(eid, |ex| {
                    if let Some(ref q8) = shared_act {
                        if let (Some(gate), Some(up)) =
                            (ex.gate.apply_cpu_q8(q8), ex.up.apply_cpu_q8(q8))
                        {
                            let activated = act.apply(&gate, &up);
                            return ex.down.apply(&activated);
                        }
                    }
                    run_expert_placed(&input, ex, placement, act)
                });
                (out, w)
            })
            .collect()
    }

    /// Runs one position's normalized hidden state through this
    /// layer's MoE FFN block, given already-computed router logits for
    /// that position, returning the combined output to add back into
    /// the residual stream. Shared by `forward_token` (router computed
    /// via a single `apply` call, since there's only one position) and
    /// `forward_batch`'s per-position loop (router computed via one
    /// batched `apply_batch` call up front, sliced per position here --
    /// see `forward_batch`'s doc comment for why that batching matters
    /// and must not be lost by calling this per position instead).
    /// `gpu_vram_budget_bytes`: see `Decoder::gpu_vram_budget_bytes`'s
    /// doc comment -- `None` dispatches every routed expert through
    /// `run_expert_placed` with `ExpertPlacement::Cpu`, which is
    /// exactly `run_expert`'s own behavior, so this is a real
    /// zero-behavior-change default, not just "probably fine."
    /// One token's routing decision for one MoE layer.
    ///
    /// Three shapes, in the order llama.cpp's `build_moe_ffn` decides
    /// them: grouped selection when the checkpoint declares expert
    /// groups; the biased/scaled port when the layer carries
    /// `exp_probs_b` or the model carries a non-unit
    /// `expert_weights_scale`; otherwise the plain top-k this decoder has
    /// always used. The last arm is kept rather than folded into
    /// `route_top_k_biased` so that every checkpoint without those two
    /// features routes through byte-identical code to before.
    ///
    /// `exp_probs_b` together with expert groups is refused at load
    /// (`loader.rs`), so that combination cannot reach here.
    fn route_for_layer(
        layer: &LayerWeights,
        router_logits: &[f32],
        config: &ModelConfig,
    ) -> frink_moe::RoutingDecision {
        match (
            config.moe.expert_group_count,
            config.moe.expert_group_used_count,
        ) {
            (Some(n_groups), Some(k_per_group)) if n_groups > 1 && k_per_group > 0 => {
                frink_moe::route_top_k_grouped(
                    router_logits,
                    n_groups,
                    k_per_group,
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                )
            }
            _ if layer.moe.exp_probs_bias.is_some() || config.moe.expert_weights_scale != 1.0 => {
                frink_moe::route_top_k_biased(
                    router_logits,
                    layer.moe.exp_probs_bias.as_deref(),
                    config.moe.n_experts_active,
                    config.moe.gating,
                    config.moe.norm_topk_prob,
                    config.moe.expert_weights_scale,
                )
            }
            _ => route_top_k(
                router_logits,
                config.moe.n_experts_active,
                config.moe.gating,
                config.moe.norm_topk_prob,
            ),
        }
    }

    /// `normed2` is what the DENSE half (the shared experts) reads;
    /// `routed_input` is what the routed experts read -- the same slice
    /// for every architecture but Arctic, whose routed branch reads
    /// `ffn_norm_exps(inpSA)` (`crate::router_input`). Two arguments
    /// rather than one so a caller cannot hand the experts the wrong
    /// vector without saying so.
    #[allow(clippy::too_many_arguments)]
    fn combine_ffn_outputs_for_position(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2: &[f32],
        routed_input: &[f32],
        router_logits: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
    ) -> Vec<f32> {
        let decision = Self::route_for_layer(layer, router_logits, config);
        let acts = config.layer_ffn_acts(layer_idx);
        let act = acts.routed;
        layer.moe.record_activations(&decision.expert_ids);
        // Best-effort warm of the routed experts for this layer into
        // the store cache (SSD streaming overlap). Resident-backed
        // layers skip this entirely.
        if let ExpertBacking::Stored {
            store,
            layer: layer_id,
            ..
        } = &layer.moe.experts
        {
            let keys: Vec<frink_core::expert_store::ExpertKey> = decision
                .expert_ids
                .iter()
                .map(|&eid| frink_core::expert_store::ExpertKey {
                    layer: *layer_id,
                    expert: eid as u32,
                })
                .collect();
            store.prefetch(&keys);
        }

        // Metal: fuse all top-k experts into one CB (one wait) when every
        // routed expert has Metal matvec launches. Shared experts (rare
        // for OLMoE) still run on the host after.
        // `launch_moe_topk_swiglu` is SwiGLU-only, so a GeGLU MoE layer
        // keeps the host path rather than taking a kernel that computes
        // a different activation.
        #[cfg(feature = "metal")]
        if frink_core::metal_dense_enabled()
            && act.is_swiglu()
            && layer.moe.shared_experts.is_empty()
        {
            if let Some(fused) = Self::try_metal_moe_topk(layer, routed_input, &decision) {
                return fused;
            }
        }

        let routed_outputs: Vec<(Vec<f32>, f32)> = {
            // llama.cpp mul_mat_id: one shared Q8 act + flat (slot,row)
            // parallel over all top-k experts (not serial expert loops each
            // with their own rayon fork-join).
            let all_cpu = plan
                .map(|p| {
                    decision
                        .expert_ids
                        .iter()
                        .all(|&eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu))
                })
                .unwrap_or(true);
            // The parallel-slot kernel quantises ONE input for every
            // slot; a model that weights the input per slot
            // (`crate::routed_weight_site`) takes the serial site.
            let before = config.moe.routed_weight_before_ffn;
            if let (true, false, ExpertBacking::Resident(experts)) =
                (all_cpu, before, &layer.moe.experts)
            {
                if let Some(outs) = Self::cpu_moe_topk_parallel_slots(
                    experts,
                    routed_input,
                    &decision,
                    hidden_dim,
                    act,
                ) {
                    outs
                } else {
                    Self::cpu_moe_serial_experts(layer, routed_input, &decision, plan, act, before)
                }
            } else {
                Self::cpu_moe_serial_experts(layer, routed_input, &decision, plan, act, before)
            }
        };
        // Shared experts fire on every token regardless of routing, so
        // there's no offload decision to make for them the way there
        // is for routed experts -- always CPU, matching `run_expert`.
        let mut shared_outputs: Vec<Vec<f32>> = layer
            .moe
            .shared_experts
            .iter()
            .map(|e| run_expert(normed2, e, acts.dense))
            .collect();
        // Qwen2-MoE-specific: see `MoeWeights::shared_expert_gate`'s doc
        // comment. Scaling here (before `combine_expert_outputs`, which
        // is architecture-agnostic and knows nothing about this gate)
        // keeps the gate a decoder-level detail, not a frink-moe API
        // change.
        if let Some(gate) = &layer.moe.shared_expert_gate {
            let gate_logit: f32 = gate.iter().zip(normed2.iter()).map(|(g, x)| g * x).sum();
            let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
            for out in shared_outputs.iter_mut() {
                for x in out.iter_mut() {
                    *x *= gate_value;
                }
            }
        }

        combine_expert_outputs(&routed_outputs, &shared_outputs, hidden_dim)
    }

    /// The dense FFN for a whole batch of positions in three batched
    /// matmuls (gate, up, down) instead of three per position.
    ///
    /// This is the counterpart of what `forward_hidden_batch` already
    /// did for Q/K/V and the router, and it is where a dense model's
    /// prefill time actually goes: `WeightMatrix::apply_batch` reads
    /// each weight row once and dots it against every position, rather
    /// than re-reading the whole FFN for each one.
    ///
    /// `None` for anything that is not a plain dense layer -- MoE
    /// routing is per position by construction, so those keep the
    /// sequential path.
    ///
    /// On a GPU backend the per-position alternative is one *fused*
    /// gate+up+SiLU+down launch (`apply_gpu_dense_ffn_swiglu`), so this
    /// used to be gated off there: three separate batched launches lost
    /// to it while `apply_batch` was still a batched *matvec*.
    ///
    /// That stopped being true once the simdgroup GEMM landed, and the
    /// old gate turned out to be the dominant cost of Metal prefill --
    /// a 512-token prompt ran the FFN one position at a time, 512 x
    /// n_layers fused launches, which a profile put at 90% of prefill
    /// while the GEMM it bypassed accounted for 21%.
    ///
    /// Decode (`batch_size == 1`) still takes the fused per-position
    /// launch, which is the right shape there.
    fn dense_ffn_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        batch_size: usize,
        config: &ModelConfig,
    ) -> Option<Vec<f32>> {
        let act = config.layer_ffn_acts(layer_idx).dense;
        // Match the GPU `mul_mm` threshold: below it the per-call launch
        // overhead outweighs the weight reuse.
        if !Self::is_dense_layer(layer) || batch_size < 4 {
            return None;
        }
        // On a GPU backend this only wins when the weights have a real
        // batched GEMM; otherwise `apply_batch` is a batched matvec and
        // loses to the fused per-position launch.
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            let gpu_dense = frink_core::weight_matrix::metal_dense_enabled();
            #[cfg(not(feature = "metal"))]
            let gpu_dense = false;
            #[cfg(feature = "cuda")]
            let gpu_dense = gpu_dense || frink_core::weight_matrix::cuda_dense_enabled();
            if gpu_dense {
                let all_gemm = layer.moe.with_expert(0, |ex| {
                    ex.gate.prefers_gpu_batch()
                        && ex.up.prefers_gpu_batch()
                        && ex.down.prefers_gpu_batch()
                });
                if !all_gemm {
                    return None;
                }
            }
        }
        layer.moe.record_activations(&[0]);
        // One command buffer for the whole FFN when every matrix has a
        // simdgroup GEMM: gate and up feed the activation and the down
        // projection without the intermediates ever touching the host.
        // Three separate launches cost three round trips per layer plus
        // four copies of a `batch x ffn_dim` tensor. Not for a layer
        // with a norm between the activation and `down`
        // (`ffn_sub_norm`): the kernel has no such site.
        // ...nor a bias at any of its three sites (`crate::proj_bias`).
        #[cfg(feature = "metal")]
        if let (true, Some(gelu), None, None) = (
            frink_core::weight_matrix::metal_dense_enabled(),
            act.fused_kernel_gelu_flag(),
            layer.moe.ffn_sub_norm.as_ref(),
            layer.moe.dense_bias.as_ref(),
        ) {
            let fused = layer.moe.with_expert(0, |ex| {
                let (g, u, d) = (
                    ex.gate.mul_mm_sg_launch()?,
                    ex.up.mul_mm_sg_launch()?,
                    ex.down.mul_mm_sg_launch()?,
                );
                frink_metal::gpu::launch_dense_ffn_swiglu_batch(
                    &g,
                    &u,
                    &d,
                    normed2_batch,
                    batch_size,
                    gelu,
                )
                .ok()
            });
            if let Some(out) = fused {
                return Some(out);
            }
        }
        Some(layer.moe.with_expert(0, |ex| {
            let ffn_acts = ex.gate.quantize_batch_acts(normed2_batch, batch_size);
            let (mut gate, mut up) = WeightMatrix::apply_batch_pair_with_acts(
                &ex.gate,
                &ex.up,
                normed2_batch,
                batch_size,
                ffn_acts.as_ref(),
            );
            // `build_ffn`: `up_b` / `gate_b` before the activation
            // (`crate::proj_bias`). On an ungated layer `gate` is an
            // alias of `up` and only `up_b` exists; `act.apply` reads
            // `up` alone for it, so the gate copy going unbiased is not
            // read.
            if let Some(bias) = &layer.moe.dense_bias {
                bias.add_pre_activation(&mut gate, &mut up, batch_size);
            }
            let mut activated = act.apply(&gate, &up);
            // bitnet.cpp:135-140, per row, on the same vector the row
            // body norms in `frink_moe::run_expert_sub_normed`.
            if let Some(w) = &layer.moe.ffn_sub_norm {
                let width = w.len();
                activated = activated
                    .chunks(width)
                    .flat_map(|row| rms_norm(row, w, config.rms_norm_eps))
                    .collect();
            }
            let mut down = ex.down.apply_batch(&activated, batch_size);
            if let Some(bias) = &layer.moe.dense_bias {
                bias.add_post_down(&mut down, batch_size);
            }
            down
        }))
    }

    /// CPU MoE prefill: bucket tokens by expert, then one
    /// `apply_batch` per expert with tokens instead of per-token
    /// `combine_ffn_outputs_for_position`. Shared experts append via
    /// [`Self::accumulate_shared_experts_batch`]. `None` when gates fail
    /// (small batch, dense, Metal preferred, non-resident, or any
    /// GPU-placed expert). Both gated activations are served here --
    /// the combine goes through [`GluAct`], so GeGLU no longer falls out
    /// to the per-position path.
    /// `routed_batch` is what the experts read, `normed2_batch` what the
    /// shared experts read: the same rows for every architecture but
    /// Arctic (see `combine_ffn_outputs_for_position`).
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn_batch(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2_batch: &[f32],
        routed_batch: &[f32],
        router_logits_batch: &[f32],
        batch_size: usize,
        config: &ModelConfig,
        plan: Option<&PlacementPlan>,
    ) -> Option<Vec<f32>> {
        let hidden_dim = config.hidden_dim;
        if batch_size < 32 || Self::is_dense_layer(layer) {
            return None;
        }
        // Metal prefill owns MoE when dense Metal is on
        // (`try_metal_moe_prefill_batch`); do not steal the path.
        #[cfg(feature = "metal")]
        if frink_core::metal_dense_enabled() {
            return None;
        }
        let acts = config.layer_ffn_acts(layer_idx);
        let act = acts.routed;
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let n_experts = experts.len();
        let all_cpu = plan
            .map(|p| (0..n_experts).all(|eid| matches!(p.placement_for(eid), ExpertPlacement::Cpu)))
            .unwrap_or(true);
        if !all_cpu || n_experts == 0 {
            return None;
        }

        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_experts];
        for b in 0..batch_size {
            let logits = &router_logits_batch[b * n_experts..(b + 1) * n_experts];
            let decision = Self::route_for_layer(layer, logits, config);
            layer.moe.record_activations(&decision.expert_ids);
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                buckets[eid].push((b, w));
            }
        }

        let mut acc = vec![0f32; batch_size * hidden_dim];
        for (eid, toks) in buckets.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let mut gathered = vec![0f32; n * hidden_dim];
            // Each gathered row is one (token, slot): the slot's input
            // and the weight its output carries come from the one
            // helper the row body uses (`crate::routed_weight_site`).
            let mut out_weights = Vec::with_capacity(n);
            for (i, &(tok, w)) in toks.iter().enumerate() {
                let (input, w) = crate::routed_weight_site::routed_slot(
                    &routed_batch[tok * hidden_dim..(tok + 1) * hidden_dim],
                    w,
                    config.moe.routed_weight_before_ffn,
                );
                gathered[i * hidden_dim..(i + 1) * hidden_dim].copy_from_slice(&input);
                out_weights.push(w);
            }
            let ex = &experts[eid];
            let ffn_acts = ex.gate.quantize_batch_acts(&gathered, n);
            let gate = ex
                .gate
                .apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
            let up = ex.up.apply_batch_with_acts(&gathered, n, ffn_acts.as_ref());
            let activated = act.apply(&gate, &up);
            let down = ex.down.apply_batch(&activated, n);
            for (i, (&(tok, _), &w)) in toks.iter().zip(out_weights.iter()).enumerate() {
                let out = &down[i * hidden_dim..(i + 1) * hidden_dim];
                let row = &mut acc[tok * hidden_dim..(tok + 1) * hidden_dim];
                for (a, &o) in row.iter_mut().zip(out.iter()) {
                    *a += w * o;
                }
            }
        }

        Self::accumulate_shared_experts_batch(
            layer,
            normed2_batch,
            batch_size,
            hidden_dim,
            &mut acc,
            acts.dense,
        );
        Some(acc)
    }

    /// gpt-oss's MoE FFN for one position.
    ///
    /// A separate function rather than another branch inside
    /// `combine_ffn_outputs_for_position` on purpose: that path carries
    /// expert-store prefetch, residency placement, a Metal top-k fusion
    /// and a batched parallel-slot kernel, and every one of them would
    /// need its own gpt-oss variant to stay honest. This is the whole
    /// gpt-oss FFN in one readable block, checked end-to-end against
    /// llama.cpp, and slow — routed experts run serially. It is the
    /// correct-first shape; making it fast is a separate change with its
    /// own A/B, not something to smuggle in under a correctness fix.
    ///
    /// Ported from `llama-graph.cpp::build_moe_ffn` with
    /// `gating_op = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`,
    /// `type_op = LLM_FFN_SWIGLU_OAI_MOE`, `norm_w = false`,
    /// `w_scale = 1`, all four bias tensors present.
    fn gpt_oss_ffn(
        layer: &LayerWeights,
        oai: &GptOssLayer,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut router_logits = layer.moe.router.apply(normed2);
        for (x, b) in router_logits.iter_mut().zip(oai.router_bias.iter()) {
            *x += b;
        }
        // Selection on the raw biased logits, softmax over the winners
        // only -- see `route_top_k_softmax_weight`.
        let decision =
            frink_moe::route_top_k_softmax_weight(&router_logits, config.moe.n_experts_active);
        layer.moe.record_activations(&decision.expert_ids);

        let mut out = vec![0f32; hidden_dim];
        for (slot, &eid) in decision.expert_ids.iter().enumerate() {
            let w = decision.weights[slot];
            let expert_out = layer.moe.with_expert(eid, |ex| {
                frink_moe::run_expert_oai(
                    normed2,
                    ex,
                    &oai.expert_bias[eid],
                    frink_moe::SWIGLU_OAI_ALPHA,
                    frink_moe::SWIGLU_OAI_LIMIT,
                )
            });
            for (o, e) in out.iter_mut().zip(expert_out.iter()) {
                *o += w * e;
            }
        }
        out
    }

    /// `forward_token`'s MoE FFN block for one position: the dense
    /// fast path (see `is_dense_layer`) or the full router+combine path
    /// with the router computed inline via a single-position `apply`.
    fn run_ffn_block(
        layer_idx: usize,
        layer: &LayerWeights,
        normed2: &[f32],
        config: &ModelConfig,
        hidden_dim: usize,
        plan: Option<&PlacementPlan>,
        operand: ffn_block::RouterOperand,
    ) -> Vec<f32> {
        if Self::is_dense_layer(layer) {
            layer.moe.record_activations(&[0]);
            // One expert, run exactly the way a routed one is. The GeGLU
            // arm used to be spelled out here and nowhere else, which is
            // precisely how the routed paths ended up SwiGLU-only.
            let act = config.layer_ffn_acts(layer_idx).dense;
            return Self::run_dense_expert(layer, normed2, act, config.rms_norm_eps);
        }
        // What the router reads is the caller's fact (`crate::router_input`):
        // the normed FFN input here, logits computed before attention, or
        // Arctic's normed layer input, which its experts read too.
        let (router_logits, routed_input): (Vec<f32>, &[f32]) = match &operand {
            ffn_block::RouterOperand::FfnInput => (layer.moe.router.apply(normed2), normed2),
            ffn_block::RouterOperand::Precomputed(logits) => (logits.clone(), normed2),
            ffn_block::RouterOperand::BranchInput(x) => (layer.moe.router.apply(x), x.as_slice()),
        };
        Self::combine_ffn_outputs_for_position(
            layer_idx,
            layer,
            normed2,
            routed_input,
            &router_logits,
            config,
            hidden_dim,
            plan,
        )
    }

    /// One token's embedding row, scaled if this checkpoint scales it.
    ///
    /// `embedding_scale` is `sqrt(hidden_dim)` on the Gemma family and
    /// `None` everywhere else, so a path that dequantizes the row and
    /// forgets the multiply is wrong on exactly one family and right on
    /// every other -- which is why it survived as a drift for as long as
    /// it did. The lookup and the scale live in one function so a caller
    /// cannot obtain the row without it.
    /// The embedding row, scaled by `embedding_scale` where the
    /// architecture has one, and -- for a skip-stream model
    /// (`crate::skip_stream`) -- RMS-normed without a weight, as
    /// `talkie.cpp:50` norms `inpL` before layer 0. The ONE embedding
    /// site; what it returns is both layer 0's input and the skip
    /// source.
    ///
    /// `pos` is the token's position, which a learned position table
    /// (`crate::position_embd`) indexes: `gpt2.cpp:74-77` add row `pos`
    /// after `build_inp_embd`'s scale and before anything else.
    fn embed_token(&self, token_id: usize, pos: usize) -> Vec<f32> {
        let mut row = self.embedding.dequant_row(token_id);
        if let Some(scale) = self.config.embedding_scale {
            for v in row.iter_mut() {
                *v *= scale;
            }
        }
        if let Some(table) = &self.position_embd {
            assert!(
                pos < table.rows(),
                "position {pos} is past the {}-row learned position table \
                 (`position_embd.weight`); the trained context is the model's limit",
                table.rows()
            );
            for (v, p) in row.iter_mut().zip(table.dequant_row(pos)) {
                *v += p;
            }
        }
        // `bloom.cpp:77-80`: the embedding normed before layer 0. No
        // graph has both a position table and this norm, so their
        // order is not a graph's; it is the order the two arrived in.
        if !matches!(self.embedding_norm, NormOp::None) {
            row = self.embedding_norm.apply(&row, self.config.rms_norm_eps);
        }
        if self.config.skip_stream {
            row = crate::norm::rms_norm_no_params(&row, self.config.rms_norm_eps);
        }
        row
    }

    /// [`Self::embed_token`] for a whole batch: `[batch, hidden]`,
    /// flattened row-major, row `b` at `position(b)`.
    fn embed_tokens(&self, tokens: &[usize], position: impl Fn(usize) -> usize) -> Vec<f32> {
        tokens
            .iter()
            .enumerate()
            .flat_map(|(b, &t)| self.embed_token(t, position(b)))
            .collect()
    }

    /// The `output_head` half of a single-position forward: project the
    /// final-normed hidden state and softcap the result if this
    /// checkpoint softcaps it.
    ///
    /// The counterpart to [`Self::logits_from_flat_hidden`] for the
    /// one-row case, and held here for the same reason: Gemma-2 caps its
    /// final logits at 30.0, so a path that projects and returns without
    /// capping produces a different distribution -- not an error, just a
    /// quietly wrong one.
    fn logits_from_normed(&self, final_normed: &[f32]) -> Vec<f32> {
        Logits::project(
            &self.output_head,
            final_normed,
            self.output_bias.as_deref(),
            self.config.final_logit_softcap,
            self.config.logit_multiplier,
        )
        .into_vec()
    }

    /// The `output_head` half of [`Self::forward_batch`], split out so
    /// the hidden-state-returning variant cannot drift from it (a
    /// second copy of the softcap would be a silent quality bug).
    fn logits_from_flat_hidden(&self, flat: Vec<f32>, batch_size: usize) -> Vec<Vec<f32>> {
        let vocab_size = self.output_head.rows();
        let logits_batch = Logits::from_output_head(
            self.output_head.apply_batch(&flat, batch_size),
            self.output_bias.as_deref(),
            self.config.final_logit_softcap,
            self.config.logit_multiplier,
        );
        logits_batch
            .as_slice()
            .chunks(vocab_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// [`Self::forward_batch_last`], plus the choice of whether the host
    /// caches have to hold the real K/V when it returns. See
    /// [`Self::advance_host_kv_after_metal_prefill`] for why that is a
    /// choice at all.
    fn forward_batch_last_inner(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Vec<f32> {
        let hiddens =
            self.forward_hidden_batch_inner(tokens, start_pos, kv_caches, host_kv_authoritative);
        let Some(last) = hiddens.last() else {
            return Vec::new();
        };
        self.logits_from_normed(last)
    }

    /// The body of [`Self::forward_batch_last_paged`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    fn forward_batch_last_paged_on_worker(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Result<Vec<f32>, PagedStoreExhausted> {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        assert_eq!(stores.layer_count(), self.config.n_layers);
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Reserve every layer up front, under guards spanning the check
        // AND the take. Each layer has its own store, so one having
        // room says nothing about the next -- and under concurrency,
        // checking and then taking as separate steps lets another
        // request slip in between and leave this one half-written.
        //
        // Reserving before the forward rather than after also means a
        // request that cannot fit is refused before it burns a prefill.
        {
            let mut guards = stores.write_all();
            for (cache, store) in kv_caches.iter().zip(guards.iter()) {
                if cache.blocks_needed_for(store, tokens.len()) > store.free_block_count() {
                    return Err(PagedStoreExhausted);
                }
            }
            for (cache, store) in kv_caches.iter_mut().zip(guards.iter_mut()) {
                cache
                    .reserve(store, tokens.len())
                    .expect("checked against free_block_count under this same guard");
            }
        }

        // Gather under read guards, one layer at a time: the forward
        // below is the expensive part and holds nothing.
        let mut scratch: Vec<KvCache> = kv_caches
            .iter()
            .enumerate()
            .map(|(l, cache)| cache.to_contiguous(&stores.read(l)))
            .collect();

        // `host_kv_authoritative`: the scatter below READS these caches,
        // and a Metal prefill otherwise leaves them holding
        // `advance_len` placeholders while the real K/V sits on the
        // device. Copying those placeholders into the page store is
        // what made paged KV on Metal answer fluent nonsense from a
        // prompt the model never attended over.
        let logits = self.forward_batch_last_inner(tokens, start_pos, &mut scratch, true);

        // Scatter into blocks this sequence already owns. Nothing here
        // can fail, which is the point of reserving above.
        for (l, (cache, gathered)) in kv_caches.iter_mut().zip(&scratch).enumerate() {
            let mut store = stores.write(l);
            let width = store.n_kv_heads() * store.head_dim();
            let base = cache.seq_len() * width;
            cache
                .append_contiguous(
                    &mut store,
                    &gathered.k[base..],
                    &gathered.v[base..],
                    tokens.len(),
                )
                .expect("blocks reserved above are still held by this sequence");
            // A recurrent layer's state moved in the scratch copy and
            // is not rows (`frink_core::recurrent_state`).
            cache.recurrent = gathered.recurrent.clone();
        }
        Ok(logits)
    }

    /// [`Self::forward_hidden_batch`] with one extra promise the public
    /// signature cannot express.
    ///
    /// `host_kv_authoritative` says whether the caller will READ
    /// `kv_caches` afterwards. Metal prefill normally leaves K/V on the
    /// device and fills the host rows with a `advance_len` placeholder,
    /// which is correct only because the contiguous decode path then
    /// reads the device buffers too. `forward_batch_last_paged` reads
    /// the host rows -- it copies them into the page store -- so it
    /// passes `true` and pays for the download.
    fn forward_hidden_batch_inner(
        &self,
        tokens: &[usize],
        start_pos: usize,
        kv_caches: &mut [KvCache],
        host_kv_authoritative: bool,
    ) -> Vec<Vec<f32>> {
        // Read only by the Metal arms below; a CPU-only build fills the
        // host cache with real rows on every path and has nothing to
        // choose between.
        let _ = host_kv_authoritative;
        assert_eq!(kv_caches.len(), self.config.n_layers);
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let v_head_dim = self.config.v_head_dim();
        // The Metal KV plane holds ONE geometry, and `use_metal_attn`
        // below is false for a per-layer-shape model
        // (`metal_can_serve_model`), so the widest layer's count is
        // every layer's wherever this is read.
        #[cfg(feature = "metal")]
        let n_kv_heads = self.config.n_kv_heads;

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens, |b| start_pos + b);
        let skip_rows = self.config.skip_stream.then(|| hidden_batch.clone());

        #[cfg(feature = "metal")]
        let use_metal_attn = frink_core::metal_dense_enabled()
            && frink_metal::attn::metal_attn_enabled()
            && self
                .layers
                .iter()
                .all(|l| self.layer_supports_metal_attn(l));

        #[cfg(not(feature = "metal"))]
        let use_metal_attn = false;

        let residency = self.expert_residency_plan(use_metal_attn);

        #[cfg(feature = "metal")]
        let mut metal_kv_guard: Option<
            std::sync::MutexGuard<'_, Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
        > = if use_metal_attn {
            Some(Self::lock_metal_attn_kv(&self.metal_attn_kv))
        } else {
            None
        };

        #[cfg(feature = "metal")]
        if let Some(guard) = metal_kv_guard.as_mut() {
            let need = self.layers.len();
            let need_cap = start_pos
                .saturating_add(batch_size)
                .saturating_add(256)
                .max(512);
            let reset = match guard.as_ref() {
                None => true,
                Some(v) => {
                    v.len() != need
                        || v.iter().any(|m| m.capacity() < need_cap)
                        || v.iter()
                            .zip(kv_caches.iter())
                            // ROWS: Metal holds rows, and this asks whether the
                            // host buffer matches them.
                            .any(|(m, c)| m.seq_len != c.rows())
                }
            };
            if reset {
                let mut bufs = Vec::with_capacity(need);
                for _ in 0..need {
                    match frink_metal::attn::MetalKvBuffers::with_capacity(
                        n_kv_heads, head_dim, need_cap,
                    ) {
                        Ok(b) => bufs.push(b),
                        Err(_) => {
                            **guard = None;
                            break;
                        }
                    }
                }
                if bufs.len() == need {
                    let mut ok = true;
                    for (m, c) in bufs.iter_mut().zip(kv_caches.iter()) {
                        if c.rows() > 0 && m.upload_from_host(&c.k, &c.v, c.rows()).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        **guard = Some(bufs);
                    } else {
                        **guard = None;
                    }
                } else {
                    **guard = None;
                }
            }
        }

        let n_layers = self.config.n_layers;
        let mut hrm = self.hrm_streams(&hidden_batch);
        let mut l = 0usize;
        // Labelled for the Metal arm inside the `'attention` block below,
        // whose `continue` must name the loop it leaves.
        #[allow(unused_labels)]
        'layers: while l < n_layers {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden_batch);
            let layer = self.layer_for(l);
            // THIS layer's head counts. Zero for the two attention-less
            // shapes, which leave the loop below before a width is used;
            // the Metal arms only run on a uniform model
            // (`metal_can_serve_model`), where every layer's counts are
            // the config's.
            let shape = self.config.layer_shape(l);
            let (n_heads, n_kv_heads) = (shape.attention.n_heads(), shape.attention.n_kv_heads());
            let q_width = n_heads * head_dim;
            let kv_width = n_kv_heads * head_dim;
            let v_width = n_kv_heads * v_head_dim;
            let out_width = n_heads * v_head_dim;

            // Multi-layer dense prefill: one CB, activations stay on GPU.
            #[cfg(feature = "metal")]
            if use_metal_attn && batch_size >= 4 {
                if let Some(guard) = metal_kv_guard.as_mut() {
                    if let Some(metal_kvs) = guard.as_mut() {
                        if let Some(run_len) = self.metal_prefill_dense_stack_run_len(
                            l,
                            start_pos,
                            batch_size,
                            kv_caches,
                            Some(metal_kvs.as_slice()),
                        ) {
                            if let Some(h_out) = self.try_metal_prefill_dense_stack(
                                l,
                                run_len,
                                &hidden_batch,
                                start_pos,
                                batch_size,
                                n_heads,
                                metal_kvs,
                                kv_caches,
                                host_kv_authoritative,
                            ) {
                                hidden_batch = h_out;
                                l += run_len;
                                continue;
                            }
                        }
                    }
                }
            }

            // A run of dense layers on the device, the hidden batch
            // resident across them, instead of seven round trips per
            // layer (#259). Declines before touching any cache; the
            // host body below then runs the layer.
            #[cfg(feature = "cuda")]
            if let Some((h_out, run_len)) = self.try_cuda_prefill_dense_stack(
                l,
                &hidden_batch,
                start_pos,
                batch_size,
                kv_caches,
            ) {
                hidden_batch = h_out;
                l += run_len;
                continue;
            }

            let cache = &mut kv_caches[l];

            // One-CB dense prefill (RMSNorm→QKV GEMM→attn→O→FFN) when every
            // projection has mul_mm_sg and the layer has no QKV bias / QK-norm.
            #[cfg(feature = "metal")]
            if use_metal_attn
                && batch_size >= 4
                && Self::metal_prefill_dense_layer_eligible(
                    layer,
                    &self.config,
                    self.lora_attached(),
                )
            {
                let swa_fits = self.metal_prefill_dense_swa_fits(l, start_pos, batch_size);
                if swa_fits {
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // POSITIONS: compared against `start_pos`.
                            if metal_kvs[l].seq_len == cache.positions()
                                && start_pos == cache.positions()
                            {
                                layer.moe.record_activations(&[0]);
                                let fused = layer.moe.with_expert(0, |ex| {
                                    let (q, k, v, o) = (
                                        layer.attn.q_proj.mul_mm_sg_launch()?,
                                        layer.attn.k_proj.mul_mm_sg_launch()?,
                                        layer.attn.v_proj.mul_mm_sg_launch()?,
                                        layer.attn.o_proj.mul_mm_sg_launch()?,
                                    );
                                    let ffn = frink_metal::attn::PrefillFfnMetal::Dense {
                                        gate: ex.gate.mul_mm_sg_launch()?,
                                        up: ex.up.mul_mm_sg_launch()?,
                                        down: ex.down.mul_mm_sg_launch()?,
                                    };
                                    let gelu = self
                                        .config
                                        .model_ffn_act()
                                        .and_then(GluAct::fused_kernel_gelu_flag)?;
                                    let prefill_layer = frink_metal::attn::PrefillDenseLayerMetal {
                                        // See `try_metal_prefill_dense_stack`:
                                        // no weight, no fused launch.
                                        attn_norm_w: layer.attn.norm_weight.rms_weights()?,
                                        ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
                                        q,
                                        k,
                                        v,
                                        o,
                                        ffn,
                                        post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                                        post_ffn_norm: layer.attn.post_ffn_norm.as_deref(),
                                        extras: self.metal_attn_extras(layer),
                                        rope: self.metal_layer_rope(l),
                                        layer_idx: l as u32,
                                    };
                                    frink_metal::attn::launch_prefill_dense_layer(
                                        &hidden_batch,
                                        &prefill_layer,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope(),
                                        start_pos,
                                        self.config.rms_norm_eps,
                                        gelu,
                                        self.config.attn_logit_softcap,
                                    )
                                    .ok()
                                });
                                if let Some(h_out) = fused {
                                    Self::advance_host_kv_after_metal_prefill(
                                        &metal_kvs[l],
                                        cache,
                                        batch_size,
                                        host_kv_authoritative,
                                    );
                                    hidden_batch = h_out;
                                    l += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden_batch, batch_size);
            let normed_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| layer.attn.norm_weight.apply(h, self.config.rms_norm_eps))
                .flatten()
                .collect();
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);

            // The GQA body, or the two shapes that have none of it
            // (`crate::layer_shapes::AttnShape`): a labelled block so the
            // FFN half below is reached by all three without a copy of
            // it, and so the Metal arms inside keep their `continue`s.
            'attention: {
                match shape.attention {
                    // deci.cpp:107-109: the residual passes straight through.
                    crate::layer_shapes::AttnShape::Absent => break 'attention,
                    // deci.cpp:115-118: `attn_norm` then `wo`, nothing else.
                    crate::layer_shapes::AttnShape::Linear => {
                        let projected = layer.attn.o_proj.apply_batch(&normed_batch, batch_size);
                        residual_add(&mut hidden_batch, &projected, self.config.residual_scale);
                        break 'attention;
                    }
                    // lfm2.cpp:197 / granite-hybrid.cpp:163: the rows are
                    // consecutive positions of one sequence, on this
                    // layer's cache.
                    crate::layer_shapes::AttnShape::ShortConv
                    | crate::layer_shapes::AttnShape::Mamba2
                    | crate::layer_shapes::AttnShape::Mamba1
                    | crate::layer_shapes::AttnShape::Plamo2Ssm
                    | crate::layer_shapes::AttnShape::Gdn => {
                        let out = self.recurrent_block(
                            l,
                            layer,
                            &normed_batch,
                            batch_size,
                            KvStep::Batched(cache),
                        );
                        residual_add(&mut hidden_batch, &out, self.config.residual_scale);
                        break 'attention;
                    }
                    crate::layer_shapes::AttnShape::Gqa { .. } => {}
                }

                // falcon-h1.cpp:156-160: the parallel Mamba-2 block over the
                // same normed rows, on this layer's cache, before the push.
                let parallel_ssm = self.parallel_ssm_rows(
                    l,
                    layer,
                    &normed_batch,
                    batch_size,
                    &mut cache.recurrent,
                );
                // One shared activation-quant pass for q/k/v (plan 1e): the
                // three projections read the same normed batch, so quantize it
                // once instead of once per projection. A kind mismatch inside
                // the group just re-quantizes locally.
                let qkv_acts = layer
                    .attn
                    .q_proj
                    .quantize_batch_acts(&normed_batch, batch_size);
                let mut q_batch = layer.attn.q_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                // qwen35.cpp:191-199: the gate rides in `wq`; split it
                // off before anything reads a Q width.
                let q_gate = layer.attn.q_gate_interleaved.then(|| {
                    let (q, gate) = crate::attn_gate::split_interleaved_q_gate(
                        &q_batch, batch_size, n_heads, head_dim,
                    );
                    q_batch = q;
                    gate
                });
                let mut k_batch = layer.attn.k_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                let mut v_batch = layer.attn.v_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                drop(qkv_acts);

                self.apply_qkv_bias_and_clamp(
                    layer,
                    &mut q_batch,
                    &mut k_batch,
                    &mut v_batch,
                    q_width,
                    kv_width,
                    v_width,
                );

                self.apply_qk_norms_pre_rope(layer, &mut q_batch, &mut k_batch, q_width, kv_width);
                // Host-side `mscale`, applied before either backend ropes.
                // The Metal branch below therefore hands its kernels
                // `attn_factor_applied_by_caller()` — folding it into cos/sin
                // there as well would square it.
                self.apply_rope_attn_factor(&mut q_batch, &mut k_batch, l);

                #[cfg(feature = "metal")]
                {
                    let mut did_metal_prefill = false;
                    // The Metal prefill kernel is full-causal: only safe on a
                    // SWA layer while every causal position is still inside
                    // the window. Longer prompts fall back to CPU attention.
                    let swa_fits = match self.config.layer_sliding_window(l) {
                        Some(window) => start_pos + batch_size <= window,
                        None => true,
                    };
                    // Metal prefill applies attn softcap in FA-vec / legacy GQA.
                    if let Some(guard) = metal_kv_guard.as_mut() {
                        if let Some(metal_kvs) = guard.as_mut() {
                            // POSITIONS: compared against `start_pos`.
                            // `layer_rope` is `None` where llama.cpp does
                            // not rotate this layer at all
                            // (`crate::rope_layers`); the Metal prefill
                            // block always ropes, so such a layer takes the
                            // CPU body below rather than a rotation the
                            // checkpoint never trained.
                            if let (true, Some(layer_rope)) = (
                                metal_kvs[l].seq_len == cache.positions()
                                    && start_pos == cache.positions()
                                    && swa_fits,
                                self.metal_layer_rope(l),
                            ) {
                                let prefill_res = {
                                    frink_metal::attn::launch_prefill_attn_block(
                                        &q_batch,
                                        &k_batch,
                                        &v_batch,
                                        &mut metal_kvs[l],
                                        n_heads,
                                        batch_size,
                                        self.metal_rope().attn_factor_applied_by_caller(),
                                        layer_rope,
                                        start_pos,
                                        self.config.attn_logit_softcap,
                                        false,
                                    )
                                    .map(
                                        |(attn_out_batch, _, _)| {
                                            Self::advance_host_kv_after_metal_prefill(
                                                &metal_kvs[l],
                                                cache,
                                                batch_size,
                                                host_kv_authoritative,
                                            );
                                            let projected_batch = layer
                                                .attn
                                                .o_proj
                                                .apply_batch(&attn_out_batch, batch_size);
                                            let projected_batch =
                                                if let Some(post) = &layer.attn.post_attn_norm {
                                                    projected_batch
                                                        .chunks(hidden_dim)
                                                        .flat_map(|row| {
                                                            rms_norm(
                                                                row,
                                                                post,
                                                                self.config.rms_norm_eps,
                                                            )
                                                        })
                                                        .collect::<Vec<_>>()
                                                } else {
                                                    projected_batch
                                                };
                                            residual_add(
                                                &mut hidden_batch,
                                                &projected_batch,
                                                self.config.residual_scale,
                                            );
                                            true
                                        },
                                    )
                                };
                                match prefill_res {
                                    Ok(true) => {
                                        did_metal_prefill = true;
                                    }
                                    Ok(false) => {}
                                    Err(e) => {
                                        eprintln!(
                                            "frink: Metal prefill attn failed, CPU fallback: {e}"
                                        );
                                        **guard = None;
                                    }
                                }
                            }
                        }
                    }
                    if did_metal_prefill {
                        self.ffn_block_batch(
                            l,
                            layer,
                            &mut hidden_batch,
                            batch_size,
                            oai,
                            residency
                                .as_ref()
                                .map(|p| p.layer_plan(self.physical_index(l))),
                            inputs,
                            ffn_block::BatchedFfnKernels::Prefill,
                            skip_rows.as_deref().map(|rows| SkipStream { rows }),
                        );
                        l += 1;
                        continue 'layers;
                    }
                }

                // RoPE per token is independent; parallelize for CPU pp512.
                q_batch
                    .par_chunks_mut(q_width)
                    .zip(k_batch.par_chunks_mut(kv_width))
                    .enumerate()
                    .for_each(|(b, (q_row, k_row))| {
                        let pos = start_pos + b;
                        for h in 0..n_heads {
                            self.apply_rope_head_layer(
                                &mut q_row[h * head_dim..(h + 1) * head_dim],
                                pos,
                                l,
                            );
                        }
                        for h in 0..n_kv_heads {
                            self.apply_rope_head_layer(
                                &mut k_row[h * head_dim..(h + 1) * head_dim],
                                pos,
                                l,
                            );
                        }
                    });
                // `maincoder` / `hunyuan-moe` norm HERE instead. Reachable
                // only on the host path, which is why
                // `layer_supports_metal_attn` refuses the layer outright
                // rather than letting the Metal arms above consume a batch
                // that has not been normed yet.
                self.apply_qk_norms_post_rope(
                    layer,
                    l,
                    &mut q_batch,
                    &mut k_batch,
                    q_width,
                    kv_width,
                );
                // Elementwise, so the whole Q batch in one call. Like the
                // multi-sequence path, this body did not apply it at all
                // until the decoration audit. It is placed AFTER the Metal
                // arms above deliberately: none of the seven fused launches
                // has an `attention_scale` uniform, and Q never returns to
                // the host inside `launch_prefill_dense_layer` /
                // `launch_prefill_dense_stack` for it to be scaled. The
                // refusal that keeps those arms out of reach when
                // `attention_scale` is set is in `layer_supports_metal_attn`.
                self.apply_attention_scale(&mut q_batch);
                // Same placement and same fence, per row: row `b` of this
                // batch is position `start_pos + b`, which is what the
                // RoPE loop above used for it.
                self.apply_attn_temperature(l, &mut q_batch, q_width, |b| start_pos + b);

                // ROWS, not positions: it is added to `b + 1` below to give
                // each query in the batch the length of the KV it attends
                // over, which is a count of resident rows.
                let base_seq_len = cache.rows();
                for b in 0..batch_size {
                    cache
                        .push(
                            &k_batch[b * kv_width..(b + 1) * kv_width],
                            &v_batch[b * v_width..(b + 1) * v_width],
                        )
                        .expect("unbounded/planned KvCache growth is infallible");
                }

                // Prefill attention over the just-written KV prefix. Parallel
                // over query positions — the serial loop was a dominant CPU
                // pp512 bottleneck (each query still attends only its causal
                // prefix; K/V slices are immutable after the pushes above).
                let cache_k = &cache.k;
                let cache_v = &cache.v;
                let softcap = self.config.attn_logit_softcap;
                let batch_window = self.config.batch_window(l, start_pos, batch_size);
                // A layer with sinks takes the per-query path, windowed
                // or not: the blocked kernel has no sink term. So does a
                // chunked layer whose queries straddle a chunk boundary
                // (`crate::chunked_swa`): the blocked kernel takes ONE
                // window. Everything
                // else goes through the blocked kernel, which is Rayon over
                // `[query-block x head]` against one shared KV buffer,
                // windowed or not. SWA layers used to take a per-query
                // `causal_gqa_attention_windowed_softcap` instead, which is
                // `online_attn_accumulate`: two scalar `exp` and a
                // head_dim-wide rescale per KV position, with the head axis
                // serial inside each task. On Gemma-3-1B (22 of 26 layers
                // are SWA) that arm was 19.6% of non-idle CPU `pp512`
                // samples while doing the *same* KV work as this one - at
                // `pp512` the 512-wide window covers the whole prompt.
                let sinks = layer.attn.sinks.as_deref();
                let blocked = match (batch_window, sinks) {
                    (crate::config::BatchWindow::Uniform(window), None) => Some(window),
                    (crate::config::BatchWindow::Uniform(_), Some(_))
                    | (crate::config::BatchWindow::PerQuery, _) => None,
                };
                let mut attn_out_batch = match blocked {
                    Some(window) => self.prefill_attention_blocked(
                        &q_batch,
                        cache_k,
                        cache_v,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        v_head_dim,
                        batch_size,
                        base_seq_len,
                        softcap,
                        window,
                    ),
                    None => {
                        let mut out = vec![0f32; batch_size * out_width];
                        out.par_chunks_mut(out_width)
                            .enumerate()
                            .for_each(|(b, dest)| {
                                let seq_len_b = base_seq_len + b + 1;
                                let attn_out = causal_gqa_attention_row(
                                    &q_batch[b * q_width..(b + 1) * q_width],
                                    &cache_k[..seq_len_b * kv_width],
                                    &cache_v[..seq_len_b * v_width],
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    v_head_dim,
                                    seq_len_b,
                                    // Row `b` is position `start_pos + b`,
                                    // as the RoPE loop above had it.
                                    self.config.layer_window_for_query(l, start_pos + b),
                                    sinks,
                                    // The sink arm carries no softcap, as
                                    // `push_and_attend_row` has it.
                                    if sinks.is_some() { None } else { softcap },
                                    self.alibi_slopes.as_deref(),
                                );
                                dest.copy_from_slice(&attn_out);
                            });
                        out
                    }
                };

                // Every query in this batch has now been answered, so the
                // rows behind the window are rows nothing will read again
                // (#61). This is why eviction is not inside `KvCache::push`:
                // `base_seq_len` above was captured BEFORE the batch's
                // pushes and every query's KV length is derived from it, so
                // a drop between the push loop and here would attend the
                // whole prompt over shifted keys.
                //
                // Per layer rather than after the stack, and that is where
                // most of the prefill saving is: a windowed layer hands its
                // prompt rows back before the next layer allocates its own,
                // so a 32k prompt holds ONE layer's full history at a time
                // instead of every windowed layer's at once.
                self.evict_layer_kv(l, cache);

                let mut projected_batch = self.attn_out_to_residual_rows(
                    layer,
                    &normed_batch,
                    &mut attn_out_batch,
                    batch_size,
                    q_gate.as_deref(),
                );
                Self::add_parallel_ssm(&mut projected_batch, parallel_ssm);
                residual_add(
                    &mut hidden_batch,
                    &projected_batch,
                    self.config.residual_scale,
                );
            } // 'attention

            // --- FFN block ---
            self.ffn_block_batch(
                l,
                layer,
                &mut hidden_batch,
                batch_size,
                oai,
                residency
                    .as_ref()
                    .map(|p| p.layer_plan(self.physical_index(l))),
                inputs,
                ffn_block::BatchedFfnKernels::Prefill,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden_batch);
            l += 1;
        }

        hidden_batch
            .chunks(hidden_dim)
            .map(|h| self.final_norm.apply(h, self.config.rms_norm_eps))
            .collect()
    }

    /// Appends one position to sequence `b`'s layer-`l` KV, then
    /// attends over everything that sequence holds.
    ///
    /// The only place `forward_multi_seq_kv` touches a cache, and so
    /// the only place the backing matters.
    ///
    /// Selects sequence `b`'s layer-`l` cache and hands it to
    /// [`Decoder::push_and_attend_row`], the one attend body the whole
    /// crate shares. This used to spell that body out a second time; the
    /// contiguous arm of the copy differed from `forward_token`'s by
    /// exactly one call (the CUDA resident hook), which is the kind of
    /// difference nobody notices until it is a wrong answer.
    #[allow(clippy::too_many_arguments)] // one per thing the step needs
    fn push_and_attend(
        &self,
        kv: &mut MultiSeqKv<'_>,
        b: usize,
        l: usize,
        layer: &LayerWeights,
        k: &[f32],
        v: &[f32],
        q: &[f32],
    ) -> Vec<f32> {
        self.push_and_attend_row(kv.step(b, l), l, layer, k, v, q)
    }

    /// The body of [`Self::forward_multi_seq_kv`], already running on a
    /// CPU-pool worker. See `entry.rs` for why the split exists.
    fn forward_multi_seq_kv_on_worker(
        &self,
        tokens: &[usize],
        positions: &[usize],
        kv: &mut MultiSeqKv<'_>,
    ) -> Vec<Vec<f32>> {
        assert_eq!(tokens.len(), positions.len());
        assert_eq!(tokens.len(), kv.len());
        let batch_size = tokens.len();
        if batch_size == 0 {
            return Vec::new();
        }
        for seq in 0..batch_size {
            assert_eq!(kv.layers_per_seq(seq), self.config.n_layers);
        }

        let hidden_dim = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let v_head_dim = self.config.v_head_dim();

        // [batch, hidden], flattened row-major.
        let mut hidden_batch: Vec<f32> = self.embed_tokens(tokens, |b| positions[b]);
        let skip_rows = self.config.skip_stream.then(|| hidden_batch.clone());
        let mut hrm = self.hrm_streams(&hidden_batch);

        let residency = self.gpu_vram_budget_bytes.map(|b| self.residency_plan(b));

        for l in 0..self.config.n_layers {
            self.hrm_stack_input(hrm.as_ref(), l, &mut hidden_batch);
            let layer = self.layer_for(l);
            // THIS layer's head counts; see the prefill body.
            let shape = self.config.layer_shape(l);
            let (n_heads, n_kv_heads) = (shape.attention.n_heads(), shape.attention.n_kv_heads());
            // --- attention block ---
            let inputs = self.branch_inputs(layer, &hidden_batch, batch_size);
            let normed_batch: Vec<f32> = hidden_batch
                .par_chunks(hidden_dim)
                .map(|h| layer.attn.norm_weight.apply(h, self.config.rms_norm_eps))
                .flatten()
                .collect();
            let oai = self
                .gpt_oss
                .as_ref()
                .map(|g| &g.layers[self.physical_index(l)]);

            'attention: {
                match shape.attention {
                    crate::layer_shapes::AttnShape::Absent => break 'attention,
                    crate::layer_shapes::AttnShape::Linear => {
                        let projected = layer.attn.o_proj.apply_batch(&normed_batch, batch_size);
                        residual_add(&mut hidden_batch, &projected, self.config.residual_scale);
                        break 'attention;
                    }
                    // lfm2.cpp:197 / granite-hybrid.cpp:163: each row is
                    // ONE position of its own sequence, so each runs on
                    // its own cache.
                    crate::layer_shapes::AttnShape::ShortConv
                    | crate::layer_shapes::AttnShape::Mamba2
                    | crate::layer_shapes::AttnShape::Mamba1
                    | crate::layer_shapes::AttnShape::Plamo2Ssm
                    | crate::layer_shapes::AttnShape::Gdn => {
                        let mut out = Vec::with_capacity(batch_size * hidden_dim);
                        for b in 0..batch_size {
                            let step = kv.step(b, l);
                            out.extend(self.recurrent_block(
                                l,
                                layer,
                                &normed_batch[b * hidden_dim..(b + 1) * hidden_dim],
                                1,
                                step,
                            ));
                        }
                        residual_add(&mut hidden_batch, &out, self.config.residual_scale);
                        break 'attention;
                    }
                    crate::layer_shapes::AttnShape::Gqa { .. } => {}
                }

                // falcon-h1.cpp:156-160: the parallel Mamba-2 block, each
                // row on its own sequence's cache, before the push.
                let parallel_ssm = layer.attn.ssm.as_ref().map(|_| {
                    let mut out = Vec::with_capacity(batch_size * hidden_dim);
                    for b in 0..batch_size {
                        let mut step = kv.step(b, l);
                        out.extend(
                            self.parallel_ssm_rows(
                                l,
                                layer,
                                &normed_batch[b * hidden_dim..(b + 1) * hidden_dim],
                                1,
                                step.recurrent_slot(),
                            )
                            .expect("the layer has the block"),
                        );
                    }
                    out
                });
                // One shared activation-quant pass for q/k/v (plan 1e): the
                // three projections read the same normed batch, so quantize it
                // once instead of once per projection. A kind mismatch inside
                // the group just re-quantizes locally.
                let qkv_acts = layer
                    .attn
                    .q_proj
                    .quantize_batch_acts(&normed_batch, batch_size);
                let mut q_batch = layer.attn.q_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                // qwen35.cpp:191-199: the gate rides in `wq`; split it
                // off before anything reads a Q width.
                let q_gate = layer.attn.q_gate_interleaved.then(|| {
                    let (q, gate) = crate::attn_gate::split_interleaved_q_gate(
                        &q_batch, batch_size, n_heads, head_dim,
                    );
                    q_batch = q;
                    gate
                });
                let mut k_batch = layer.attn.k_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                let mut v_batch = layer.attn.v_proj.apply_batch_with_acts(
                    &normed_batch,
                    batch_size,
                    qkv_acts.as_ref(),
                );
                drop(qkv_acts);

                let q_width = n_heads * head_dim;
                let kv_width = n_kv_heads * head_dim;
                let v_width = n_kv_heads * v_head_dim;
                let out_width = n_heads * v_head_dim;

                self.apply_qkv_bias_and_clamp(
                    layer,
                    &mut q_batch,
                    &mut k_batch,
                    &mut v_batch,
                    q_width,
                    kv_width,
                    v_width,
                );

                self.apply_qk_norms_pre_rope(layer, &mut q_batch, &mut k_batch, q_width, kv_width);
                self.apply_rope_attn_factor(&mut q_batch, &mut k_batch, l);

                for b in 0..batch_size {
                    let pos = positions[b];
                    let q_row = &mut q_batch[b * q_width..(b + 1) * q_width];
                    for h in 0..n_heads {
                        self.apply_rope_head_layer(
                            &mut q_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                    let k_row = &mut k_batch[b * kv_width..(b + 1) * kv_width];
                    for h in 0..n_kv_heads {
                        self.apply_rope_head_layer(
                            &mut k_row[h * head_dim..(h + 1) * head_dim],
                            pos,
                            l,
                        );
                    }
                }
                self.apply_qk_norms_post_rope(
                    layer,
                    l,
                    &mut q_batch,
                    &mut k_batch,
                    q_width,
                    kv_width,
                );
                // Applied to the whole Q batch at once because it is
                // elementwise. This path did not apply it at all until the
                // decoration audit: `attention_scale` reached only
                // `forward_token`'s CPU arm and `forward_token_paged`, so a
                // checkpoint carrying one answered at one temperature when
                // decoded alone and another when batched with its neighbours.
                self.apply_attention_scale(&mut q_batch);
                self.apply_attn_temperature(l, &mut q_batch, q_width, |b| positions[b]);

                let mut attn_out_batch = vec![0f32; batch_size * out_width];
                for b in 0..batch_size {
                    let attn_out = self.push_and_attend(
                        kv,
                        b,
                        l,
                        layer,
                        &k_batch[b * kv_width..(b + 1) * kv_width],
                        &v_batch[b * v_width..(b + 1) * v_width],
                        &q_batch[b * q_width..(b + 1) * q_width],
                    );
                    attn_out_batch[b * out_width..(b + 1) * out_width].copy_from_slice(&attn_out);
                }

                let mut projected_batch = self.attn_out_to_residual_rows(
                    layer,
                    &normed_batch,
                    &mut attn_out_batch,
                    batch_size,
                    q_gate.as_deref(),
                );
                Self::add_parallel_ssm(&mut projected_batch, parallel_ssm);
                residual_add(
                    &mut hidden_batch,
                    &projected_batch,
                    self.config.residual_scale,
                );
            } // 'attention

            // --- FFN block --- per row, as this body has always run it;
            // see `BatchedFfnKernels::PerRow`.
            self.ffn_block_batch(
                l,
                layer,
                &mut hidden_batch,
                batch_size,
                oai,
                residency
                    .as_ref()
                    .map(|p| p.layer_plan(self.physical_index(l))),
                inputs,
                ffn_block::BatchedFfnKernels::PerRow,
                skip_rows.as_deref().map(|rows| SkipStream { rows }),
            );
            self.hrm_store(hrm.as_mut(), l, &hidden_batch);
        }

        let final_normed_batch: Vec<f32> = hidden_batch
            .par_chunks(hidden_dim)
            .map(|h| self.final_norm.apply(h, self.config.rms_norm_eps))
            .flatten()
            .collect();
        self.logits_from_flat_hidden(final_normed_batch, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::glm_5_2;
    use frink_core::cache::PagedKvStore;

    /// Small config used purely to keep the test fast: same
    /// architecture *shape* (GQA ratio, MoE topology) as GLM-5.2, but
    /// with tiny dims so the whole thing runs in milliseconds.
    fn tiny_test_config() -> ModelConfig {
        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        cfg
    }

    /// A GeGLU model's ROUTED experts must run GeGLU.
    ///
    /// `run_ffn_block` used to consult `ffn_activation` only in its dense
    /// arm; `combine_ffn_outputs_for_position` and everything under it
    /// was unconditionally SwiGLU, so a GeGLU MoE would have produced
    /// fluent, wrong logits with nothing in the tree to notice. That is
    /// not hypothetical: llama.cpp's `grok` passes `LLM_FFN_GELU` to
    /// `build_moe_ffn` (`.scratch/llama.cpp/src/models/grok.cpp`), and
    /// `grok` sits on `ArchPath::GenericGqa` in `capability.rs`.
    ///
    /// The reference is written out here in plain loops -- its own GELU
    /// and SiLU, not `frink_core`'s -- so it cannot agree with the code
    /// under test by sharing its bug. The second assertion is the one
    /// that makes this a test rather than a smoke check: the SwiGLU
    /// answer must be visibly different, so an implementation that
    /// ignores the activation cannot pass.
    #[test]
    fn a_geglu_moe_layer_runs_geglu_in_its_routed_experts_not_swiglu() {
        let mut cfg = tiny_test_config();
        cfg.ffn_activation = crate::config::FfnActivation::Gelu;
        let decoder = Decoder::new_random_small(cfg, 2, 8);
        let hidden_dim = decoder.config.hidden_dim;
        let layer = &decoder.layers[1];
        assert!(
            !Decoder::is_dense_layer(layer),
            "this test is about the ROUTED path; layer 1 must be a real MoE layer"
        );

        // Larger than the usual unit inputs on purpose: GELU and SiLU
        // are close near zero, and a reference that cannot tell them
        // apart cannot catch the bug this test exists for.
        let normed2: Vec<f32> = (0..hidden_dim)
            .map(|i| (i as f32 * 0.37).sin() * 12.0)
            .collect();

        let gelu = |x: f32| {
            let t = (0.797_884_6f32 * (x + 0.044_715 * x * x * x)).tanh();
            0.5 * x * (1.0 + t)
        };
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let expert_ref = |ex: &ExpertWeights, f: &dyn Fn(f32) -> f32| -> Vec<f32> {
            let g = ex.gate.apply(&normed2);
            let u = ex.up.apply(&normed2);
            let a: Vec<f32> = g.iter().zip(u.iter()).map(|(&g, &u)| f(g) * u).collect();
            ex.down.apply(&a)
        };

        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            panic!("new_random_small builds resident experts");
        };
        let router_logits = layer.moe.router.apply(&normed2);
        let decision = Decoder::route_for_layer(layer, &router_logits, &decoder.config);
        let block_ref = |f: &dyn Fn(f32) -> f32| -> Vec<f32> {
            let mut out = vec![0f32; hidden_dim];
            for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
                for (o, e) in out.iter_mut().zip(expert_ref(&experts[eid], f).iter()) {
                    *o += w * e;
                }
            }
            assert!(
                layer.moe.shared_expert_gate.is_none(),
                "tiny_test_config's shared experts are ungated; reference assumes it"
            );
            for shex in &layer.moe.shared_experts {
                for (o, e) in out.iter_mut().zip(expert_ref(shex, f).iter()) {
                    *o += e;
                }
            }
            out
        };
        let expected_geglu = block_ref(&gelu);
        let expected_swiglu = block_ref(&silu);

        let got = Decoder::run_ffn_block(
            1,
            layer,
            &normed2,
            &decoder.config,
            hidden_dim,
            None,
            ffn_block::RouterOperand::FfnInput,
        );
        assert_eq!(got.len(), hidden_dim);
        for (i, (a, b)) in got.iter().zip(expected_geglu.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * b.abs().max(1.0),
                "routed GeGLU FFN element {i}: got {a}, expected {b}"
            );
        }
        assert!(
            expected_geglu
                .iter()
                .zip(expected_swiglu.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "GeGLU and SwiGLU must differ measurably on this input, or this test \
             could not detect a routed expert that silently ran SwiGLU"
        );
    }

    #[test]
    fn forward_pass_produces_finite_logits_of_correct_shape() {
        let vocab = 10;
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, vocab);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        let logits = decoder.forward_token(3, 0, &mut caches);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must not contain NaN/Inf"
        );
    }

    /// `gpu_vram_budget_bytes` must be a real zero-behavior-change
    /// default at `None`, and a *real placement plan that places
    /// nothing* (a zero VRAM budget, so `PlacementPlan::from_budget`
    /// fits no expert at all) must produce byte-identical output to
    /// `None` too -- proving the new plumbing (building a plan,
    /// looking up each routed expert's placement, dispatching through
    /// `run_expert_placed`) doesn't change results when nothing is
    /// actually GPU-placed, without needing real CUDA hardware to
    /// check (that hardware-dependent half is
    /// `frink-moe`'s/`frink-core`'s own `#[ignore]`d tests).
    #[test]
    fn gpu_vram_budget_bytes_with_nothing_placed_matches_the_default() {
        let mut decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let mut caches_default: Vec<KvCache> = decoder.config.new_kv_caches();
        let default_logits = decoder.forward_token(3, 0, &mut caches_default);

        decoder.gpu_vram_budget_bytes = Some(0);
        let mut caches_zero_budget: Vec<KvCache> = decoder.config.new_kv_caches();
        let zero_budget_logits = decoder.forward_token(3, 0, &mut caches_zero_budget);

        assert_eq!(
            default_logits, zero_budget_logits,
            "a placement plan that places nothing on GPU must match the None default exactly"
        );
    }

    /// Qwen2-MoE's real shared-expert sigmoid gate
    /// (`MoeWeights::shared_expert_gate`): exact math check by mutating
    /// `layer.moe.shared_expert_gate` in place on an already-built
    /// decoder (no need to reconstruct a `LayerWeights`/`MoeWeights`
    /// from scratch) and comparing against a hand-derived expectation:
    /// the *only* thing the gate changes is the shared experts' own
    /// contribution, scaled by `sigmoid(gate . x)` -- so
    /// `gated_shared_output == ungated_shared_output * sigmoid_value`
    /// exactly, computed independently here via `run_expert` on the
    /// same layer's shared expert.
    #[test]
    fn shared_expert_gate_scales_shared_output_by_sigmoid_of_the_gate_logit() {
        let cfg = tiny_test_config();
        let mut decoder = Decoder::new_random_small(cfg, 2, 8);
        let hidden_dim = decoder.config.hidden_dim;
        assert_eq!(
            decoder.layers[1].moe.shared_experts.len(),
            1,
            "test assumes tiny_test_config's real MoE layer has exactly one shared expert"
        );

        let normed2: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let gate_vec: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.13 - 0.5).collect();

        // Independently compute what the shared expert alone produces,
        // and what sigmoid(gate . x) should scale it by -- this is the
        // ground truth the gated code path must reproduce exactly.
        let shared_out_raw = run_expert(
            &normed2,
            &decoder.layers[1].moe.shared_experts[0],
            decoder.config.layer_ffn_acts(1).dense,
        );
        let gate_logit: f32 = gate_vec
            .iter()
            .zip(normed2.iter())
            .map(|(g, x)| g * x)
            .sum();
        let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
        let expected_gated_shared: Vec<f32> =
            shared_out_raw.iter().map(|x| x * gate_value).collect();

        // Run the real FFN combine path twice (gate absent, then
        // present) and recover each run's shared-only contribution by
        // subtracting the routed contribution, which the gate never
        // touches and is identical between the two runs (same router,
        // same experts, same input).
        let router_logits = decoder.layers[1].moe.router.apply(&normed2);
        let ungated_total = Decoder::combine_ffn_outputs_for_position(
            1,
            &decoder.layers[1],
            &normed2,
            &normed2,
            &router_logits,
            &decoder.config,
            hidden_dim,
            None,
        );
        decoder.layers[1].moe.shared_expert_gate = Some(gate_vec);
        let gated_total = Decoder::combine_ffn_outputs_for_position(
            1,
            &decoder.layers[1],
            &normed2,
            &normed2,
            &router_logits,
            &decoder.config,
            hidden_dim,
            None,
        );

        for (i, ((u, g), expected_shared)) in ungated_total
            .iter()
            .zip(gated_total.iter())
            .zip(expected_gated_shared.iter())
            .enumerate()
        {
            let routed_contribution = u - shared_out_raw[i];
            let gated_shared_recovered = g - routed_contribution;
            assert!(
                (gated_shared_recovered - expected_shared).abs() < 1e-4,
                "index {i}: recovered gated shared output {gated_shared_recovered} != expected {expected_shared} (sigmoid({gate_logit})={gate_value})"
            );
        }
    }

    #[test]
    fn kv_cache_grows_by_one_position_per_layer_per_step() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 3, 5);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        decoder.forward_token(0, 0, &mut caches);
        decoder.forward_token(1, 1, &mut caches);
        decoder.forward_token(2, 2, &mut caches);

        for cache in &caches {
            assert_eq!(cache.positions(), 3);
        }
    }

    #[test]
    fn same_token_same_position_is_deterministic() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches_a: Vec<KvCache> = decoder.config.new_kv_caches();
        let mut caches_b: Vec<KvCache> = decoder.config.new_kv_caches();

        let out_a = decoder.forward_token(4, 0, &mut caches_a);
        let out_b = decoder.forward_token(4, 0, &mut caches_b);
        assert_eq!(out_a, out_b, "identical input state must yield identical output (no hidden randomness in the forward pass)");
    }

    #[test]
    fn multi_step_decode_stays_finite_across_positions() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        for pos in 0..16 {
            let logits = decoder.forward_token(pos % 8, pos, &mut caches);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "position {pos}: logits must stay finite across an extended decode run"
            );
        }
    }

    /// `forward_token_paged` must produce bit-identical output to
    /// `forward_token` across a multi-step decode (each layer's paged
    /// store sized generously so no layer ever exhausts its blocks) --
    /// the block-table indirection is a storage-layout detail, not a
    /// math change.
    #[test]
    fn forward_token_paged_matches_forward_token_bit_identical() {
        paged_matches_contiguous(tiny_test_config());
    }

    /// Every arm of the attention dispatch, not just the plain one.
    ///
    /// The paged path used to implement only full causal attention, and
    /// `forward_token_paged` asserted rather than run gpt-oss, because a
    /// missing sink term would have changed the distribution silently.
    /// Now that it mirrors all three arms, each one has to be held to
    /// the same bar the plain arm always was: BIT-identical, not close.
    ///
    /// A sliding window and a softcap are both driven from the config
    /// here, so a future edit that wires one arm and forgets another
    /// fails on the arm it forgot rather than on a model nobody tests.
    #[test]
    fn every_paged_attention_arm_is_bit_identical_to_its_contiguous_twin() {
        let windowed = || {
            let mut cfg = tiny_test_config();
            // Smaller than the decode length below, so the window really
            // drops positions rather than degenerating to full causal.
            cfg.sliding_window = Some(2);
            cfg.swa_layers = crate::swa_layers::SwaLayers::All;
            cfg
        };
        let softcapped = || {
            let mut cfg = tiny_test_config();
            // Small enough that `sc * tanh(s / sc)` actually compresses.
            // A realistic 30.0 is numerically indistinguishable from no
            // cap at these tiny weights, so a test using it would pass
            // whether or not the arm was wired -- checked by breaking
            // the arm on purpose and watching it still pass.
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        let both = || {
            let mut cfg = windowed();
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        // Alternating window/full layers: the per-layer arm choice has
        // to be honoured per layer, not decided once for the model.
        let alternating = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
            cfg
        };

        for cfg in [windowed(), softcapped(), both(), alternating()] {
            paged_matches_contiguous(cfg);
        }
    }

    /// Five MORE model features the paged path had lost the same way
    /// the first five went: by being a copy of the contiguous loop that
    /// nothing forced to stay in step.
    ///
    /// Found by running Gemma-2-2B through paged KV and watching it
    /// answer differently from the same model on the same backend with
    /// a contiguous cache -- on CPU, with no GPU involved at all. None
    /// of the arm tests above could see it, because `tiny_test_config`
    /// sets none of these and `new_random_small` builds every layer
    /// without the two sandwich norms.
    ///
    /// - `attention_scale`: Gemma scales Q itself and asks the kernel
    ///   for a score scale of 1.0, so the built-in `1/sqrt(head_dim)`
    ///   has to be compensated for. Missing, the model answers at a
    ///   different temperature.
    /// - `post_attn_norm` / `post_ffn_norm`: Gemma-2's sandwich norms,
    ///   applied to each branch before it rejoins the residual.
    /// - gpt-oss's `o_bias`, and its own FFN (`gpt_oss_ffn`, which
    ///   biases the router and runs the clamped OAI SwiGLU) instead of
    ///   the generic one.
    ///
    /// Every one of them produces a plausible distribution rather than
    /// an error, which is exactly why they are pinned rather than
    /// trusted. Values are chosen so each really bites: a scale of 1.0
    /// or an all-ones norm would let this pass either way.
    #[test]
    fn the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies() {
        // Gemma's query pre-attention scalar, well away from the
        // kernel's own 1/sqrt(head_dim).
        let mut scaled = tiny_test_config();
        scaled.attention_scale = Some(0.37);
        paged_matches_contiguous_with(scaled, |_| {});

        // Sandwich norms, one at a time and then together, so a wired
        // half is not covered for by the other.
        for (attn, ffn) in [(true, false), (false, true), (true, true)] {
            paged_matches_contiguous_with(tiny_test_config(), with_sandwich_norms(attn, ffn));
        }

        // gpt-oss: the O bias and the OAI FFN, which the paged path was
        // substituting the generic router+SwiGLU for.
        paged_matches_contiguous_with(tiny_test_config(), with_gpt_oss_graph);
    }

    /// The same feature list as
    /// [`the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies`],
    /// checked against `forward_hidden_batch_inner` instead.
    ///
    /// Necessary because `forward_token` and `forward_token_paged` now
    /// share ONE body (`Decoder::attn_block`) that differs only in its
    /// `KvStep`, so the paged test can no longer see a decoration
    /// dropped from that body -- deleting `post_attn_norm` or gpt-oss's
    /// `o_bias` from it leaves the whole suite green, which was measured
    /// rather than assumed. `forward_hidden_batch_inner` is deliberately
    /// NOT collapsed into the same body, so it is the independent
    /// ground truth that keeps these features pinned.
    #[test]
    fn the_batched_path_keeps_every_per_layer_feature_the_token_path_applies() {
        for (attn, ffn) in [(true, false), (false, true), (true, true)] {
            batched_matches_contiguous_with(tiny_test_config(), with_sandwich_norms(attn, ffn));
        }
        batched_matches_contiguous_with(tiny_test_config(), with_gpt_oss_graph);
    }

    /// Gemma-2's two sandwich norms, as a switch both parity helpers
    /// take, so the paged and batched tests cannot drift over WHICH
    /// features they claim to cover.
    ///
    /// Per-layer values, so a path that applied layer 0's norm
    /// everywhere would still fail.
    fn with_sandwich_norms(attn: bool, ffn: bool) -> impl Fn(&mut Decoder) {
        move |d: &mut Decoder| {
            let hidden = d.config.hidden_dim;
            for (i, layer) in d.layers.iter_mut().enumerate() {
                let w: Vec<f32> = (0..hidden)
                    .map(|j| 0.5 + (i * hidden + j) as f32 * 0.01)
                    .collect();
                if attn {
                    layer.attn.post_attn_norm = Some(w.clone());
                }
                if ffn {
                    layer.attn.post_ffn_norm = Some(w);
                }
            }
        }
    }

    /// The whole gpt-oss graph: attention sinks on every layer's
    /// attention weights, and the side table -- the O bias, the router
    /// bias and the per-expert biases `gpt_oss_ffn` reads.
    fn with_gpt_oss_graph(d: &mut Decoder) {
        let hidden = d.config.hidden_dim;
        let n_heads = d.config.n_heads;
        let n_experts = d.config.moe.n_experts;
        let ffn = d.config.moe.expert_ffn_dim;
        let n_layers = d.layers.len();
        for (l, layer) in d.layers.iter_mut().enumerate() {
            layer.attn.sinks = Some((0..n_heads).map(|h| 0.1 + (l + h) as f32 * 0.05).collect());
            layer.attn.o_bias = Some((0..hidden).map(|j| 0.02 * (j as f32 - 8.0)).collect());
        }
        d.gpt_oss = Some(GptOssWeights {
            layers: (0..n_layers)
                .map(|_| GptOssLayer {
                    router_bias: (0..n_experts).map(|e| 0.03 * e as f32).collect(),
                    expert_bias: (0..n_experts)
                        .map(|e| frink_moe::ExpertBias {
                            gate: vec![0.01 * (e + 1) as f32; ffn],
                            up: vec![-0.02 * (e + 1) as f32; ffn],
                            down: vec![0.005 * (e + 1) as f32; hidden],
                        })
                        .collect(),
                })
                .collect(),
        });
    }

    /// [`paged_matches_contiguous_with`] for `forward_batch` against
    /// sequential `forward_token`.
    ///
    /// Not bit-identity: batched prefill runs the blocked three-pass
    /// softmax while decode keeps the online accumulator, so the two
    /// agree to a tolerance rather than to the bit -- the same reason
    /// `decoder_via_engine_trait_matches_forward_batch_ground_truth`
    /// gives. 1e-5 is four orders below the ~1e-1 a dropped decoration
    /// moves these logits by.
    fn batched_matches_contiguous_with(config: ModelConfig, prepare: impl Fn(&mut Decoder)) {
        let n_layers = 2;
        let vocab = 10;
        let tokens = [3usize, 5, 7, 2, 9, 1];

        let mut seq_decoder = Decoder::new_random_small(config.clone(), n_layers, vocab);
        prepare(&mut seq_decoder);
        let mut seq_caches: Vec<KvCache> = seq_decoder.config.new_kv_caches();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| seq_decoder.forward_token(t, pos, &mut seq_caches))
            .collect();

        // Same seed -> identical weights before `prepare`, and `prepare`
        // is deterministic, so this is a like-for-like comparison.
        let mut batch_decoder = Decoder::new_random_small(config, n_layers, vocab);
        prepare(&mut batch_decoder);
        let mut batch_caches: Vec<KvCache> = batch_decoder.config.new_kv_caches();
        let batched = batch_decoder.forward_batch(&tokens, 0, &mut batch_caches);

        assert_eq!(sequential.len(), batched.len());
        for (pos, (a, b)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "position {pos}: logit count");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-5,
                    "position {pos}, logit {i}: token path={x} batched={y}"
                );
            }
        }
    }

    /// The two rules that live OUTSIDE the layer loop, which the arm
    /// test above cannot reach.
    ///
    /// The paged path had drifted from the contiguous one at both ends
    /// of the stack, and neither drift was visible to any existing test
    /// because `tiny_test_config` sets neither field:
    ///
    /// - it called `embedding.dequant_row` directly instead of scaling
    ///   the row by `embedding_scale`, so every Gemma token entered the
    ///   stack `sqrt(hidden_dim)` times too small;
    /// - it returned `output_head.apply(..)` raw instead of applying
    ///   `final_logit_softcap`, so Gemma-2's 30.0 cap never ran.
    ///
    /// Both produce a plausible distribution rather than an error, which
    /// is the whole reason to pin them: a wrong answer that still looks
    /// like an answer is what a parity test is for. Values here are
    /// chosen so each one actually bites -- a scale of 1.0 or a cap far
    /// above the logit range would let this pass either way.
    #[test]
    fn the_paged_path_scales_embeddings_and_softcaps_logits_like_the_contiguous_one() {
        let scaled = || {
            let mut cfg = tiny_test_config();
            cfg.embedding_scale = Some(7.5);
            cfg
        };
        let capped = || {
            let mut cfg = tiny_test_config();
            // Small enough that `sc * tanh(x / sc)` really compresses at
            // this model's logit magnitudes, on the same reasoning as
            // the attention softcap above.
            cfg.final_logit_softcap = Some(0.05);
            cfg
        };
        let both = || {
            let mut cfg = scaled();
            cfg.final_logit_softcap = Some(0.05);
            cfg
        };

        for cfg in [scaled(), capped(), both()] {
            paged_matches_contiguous(cfg);
        }
    }

    /// Paged prefill must agree with contiguous prefill, and must leave
    /// the KV in a state a paged DECODE can continue from.
    ///
    /// The second half is the one worth having. `forward_batch_last`
    /// returns only the last row's logits, so a gather/scatter that
    /// mangled the KV -- wrote the rows in the wrong order, dropped the
    /// part-full tail block, mis-sized a copy -- could still return the
    /// right logits for THIS call and only surface on the next token.
    /// Decoding four more tokens after the prefill is what makes the
    /// stored KV observable, so both paths are compared over the whole
    /// continuation rather than at the seam.
    ///
    /// A block size of 2 against a 5-token prompt is deliberate: it
    /// leaves the tail block part-full, which is the case
    /// `blocks_needed_for` exists for and the one a `n / block_size`
    /// reservation would get wrong.
    fn paged_prefill_matches_contiguous(config: ModelConfig) {
        let n_layers = 2;
        let decoder = Decoder::new_random_small(config, n_layers, 10);
        let prompt = [3usize, 1, 4, 1, 5];
        let continuation = [9usize, 2, 6, 5];

        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let mut plain = vec![decoder.forward_batch_last(&prompt, 0, &mut caches)];
        for (i, &tok) in continuation.iter().enumerate() {
            plain.push(decoder.forward_token(tok, prompt.len() + i, &mut caches));
        }

        let mut paged_caches: Vec<PagedKvCache> =
            (0..n_layers).map(|_| PagedKvCache::new()).collect();
        let stores = SharedPagedKv::from_stores(
            (0..n_layers)
                .map(|_| {
                    PagedKvStore::new(
                        /* block_size = */ 2,
                        /* total_blocks = */ 16,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );
        let mut paged = vec![decoder
            .forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores)
            .expect("store sized generously, must not exhaust")];
        for (i, &tok) in continuation.iter().enumerate() {
            paged.push(
                decoder
                    .forward_token_paged(tok, prompt.len() + i, &mut paged_caches, &stores)
                    .expect("store sized generously, must not exhaust"),
            );
        }

        assert_eq!(
            paged_caches[0].seq_len(),
            prompt.len() + continuation.len(),
            "paged prefill must advance seq_len by exactly the batch size"
        );
        assert_eq!(plain.len(), paged.len());
        for (step, (a, b)) in plain.iter().zip(paged.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "step {step}: logit count");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "step {step}: paged prefill + decode must be bit-identical to contiguous"
                );
            }
        }
    }

    /// Every arm again, this time through the prefill entry point. The
    /// gather is shared, but the kernel the gathered buffer reaches is
    /// the BLOCKED prefill one rather than the per-query decode one, so
    /// arm coverage here is not implied by the decode tests above.
    #[test]
    fn paged_prefill_is_bit_identical_across_every_arm() {
        let windowed = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_layers = crate::swa_layers::SwaLayers::All;
            cfg
        };
        let scaled_and_capped = || {
            let mut cfg = tiny_test_config();
            cfg.embedding_scale = Some(7.5);
            cfg.final_logit_softcap = Some(0.05);
            cfg.attn_logit_softcap = Some(0.05);
            cfg
        };
        let alternating = || {
            let mut cfg = tiny_test_config();
            cfg.sliding_window = Some(2);
            cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
            cfg
        };

        for cfg in [
            tiny_test_config(),
            windowed(),
            scaled_and_capped(),
            alternating(),
        ] {
            paged_prefill_matches_contiguous(cfg);
        }
    }

    /// A prefill the stores cannot hold refuses having written NOTHING
    /// -- checked on the case that actually needs the up-front loop.
    ///
    /// Each layer owns its own store, so layer 0 having room says
    /// nothing about layer 1. `append_contiguous` already refuses
    /// rather than half-writing a single layer, so a test whose layers
    /// are sized alike passes with the cross-layer reservation deleted
    /// -- it would be asserting a property it never exercises. Here
    /// layer 0 has room for the whole prompt and layer 1 does not, so
    /// without the up-front check layer 0 is written, layer 1 refuses,
    /// and the sequence ends up with its layers at DIFFERENT lengths.
    /// No caller can recover from that, and nothing downstream would
    /// report it: the next decode step simply attends over a shorter
    /// history in one layer than the others.
    ///
    /// Verified by deleting the reservation loop and watching this fail
    /// on `layer 1 must be untouched`.
    /// Three requests sharing one set of per-layer stores must get
    /// exactly what they would get alone.
    ///
    /// This is the property the RwLock exists for, and it cannot be
    /// asserted single-threaded. Every request writes only blocks it
    /// owns, so sharing changes where rows live and nothing else --
    /// bit-identical, not close. A store that let one request's rows
    /// land in another's blocks shows up here and nowhere else.
    #[test]
    fn concurrent_decodes_against_one_shared_store_match_running_them_alone() {
        use std::sync::Arc;

        let decoder = Arc::new(Decoder::new_random_small(tiny_test_config(), 2, 10));
        let prompts: [&[usize]; 3] = [&[3, 1, 4], &[1, 5, 9], &[2, 6, 5]];
        let continuation = [7usize, 8, 3];

        // Each request run alone, against its own store, is the answer
        // sharing must not change.
        let solo: Vec<Vec<Vec<f32>>> = prompts
            .iter()
            .map(|prompt| {
                let stores = SharedPagedKv::new(
                    2,
                    4,
                    32,
                    decoder.config.n_kv_heads,
                    decoder.config.head_dim,
                );
                let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
                run_one(&decoder, prompt, &continuation, &mut caches, &stores)
            })
            .collect();

        // The same three, concurrently, sharing ONE set of per-layer
        // stores. Every request writes only blocks it owns, so the
        // answers must be identical -- not close, identical. A store
        // that let one request's rows land in another's blocks would
        // show up here and nowhere else.
        let shared = Arc::new(SharedPagedKv::new(
            2,
            4,
            96,
            decoder.config.n_kv_heads,
            decoder.config.head_dim,
        ));
        let together: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
            let handles: Vec<_> = prompts
                .iter()
                .map(|prompt| {
                    let decoder = Arc::clone(&decoder);
                    let shared = Arc::clone(&shared);
                    scope.spawn(move || {
                        let mut caches: Vec<PagedKvCache> =
                            (0..2).map(|_| PagedKvCache::new()).collect();
                        run_one(&decoder, prompt, &continuation, &mut caches, &shared)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        for (r, (alone, concurrent)) in solo.iter().zip(together.iter()).enumerate() {
            assert_eq!(alone.len(), concurrent.len(), "request {r}: step count");
            for (step, (a, b)) in alone.iter().zip(concurrent.iter()).enumerate() {
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "request {r} step {step}: sharing a store changed the answer"
                    );
                }
            }
        }
    }

    /// Prefill then decode, returning every step's logits.
    fn run_one(
        decoder: &Decoder,
        prompt: &[usize],
        continuation: &[usize],
        caches: &mut [PagedKvCache],
        stores: &SharedPagedKv,
    ) -> Vec<Vec<f32>> {
        let mut out = vec![decoder
            .forward_batch_last_paged(prompt, 0, caches, stores)
            .expect("sized generously")];
        for (i, &tok) in continuation.iter().enumerate() {
            out.push(
                decoder
                    .forward_token_paged(tok, prompt.len() + i, caches, stores)
                    .expect("sized generously"),
            );
        }
        out
    }

    /// A decode step the stores cannot hold advances NO layer.
    ///
    /// This was a real defect until the reservation moved into
    /// `forward_token_paged`: it pushed per layer with `?`, so a store
    /// exhausting at layer 1 of 2 left layer 0 holding a position layer
    /// 1 did not. Nothing downstream reports that -- the next step just
    /// attends over a shorter history in the tail layers -- and the
    /// prefill path had the guard while decode never did.
    ///
    /// Layer 0 is given room and layer 1 none, so the bug is reachable:
    /// with the reservation removed, layer 0 advances and layer 1
    /// refuses.
    #[test]
    fn a_decode_step_the_stores_cannot_hold_advances_no_layer() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
        // Block size 1 so "one more position" always needs a block.
        // Layer 0 gets two, layer 1 exactly one: the prompt fills layer
        // 1 completely, so the decode step below cannot fit there.
        let stores = SharedPagedKv::from_stores(
            [2usize, 1]
                .into_iter()
                .map(|blocks| {
                    PagedKvStore::new(
                        1,
                        blocks,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );

        decoder
            .forward_batch_last_paged(&[1usize], 0, &mut caches, &stores)
            .expect("one position fits in both layers");
        assert_eq!(caches[0].seq_len(), 1);
        assert_eq!(caches[1].seq_len(), 1);

        let result = decoder.forward_token_paged(2, 1, &mut caches, &stores);
        assert!(result.is_err(), "layer 1 has no block left");
        assert_eq!(
            caches[0].seq_len(),
            1,
            "layer 0 must not advance past a layer that could not"
        );
        assert_eq!(caches[1].seq_len(), 1);
    }

    #[test]
    fn a_prefill_the_stores_cannot_hold_refuses_before_writing_any_layer() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let prompt = [1usize, 2, 3, 4, 5, 6];
        let mut paged_caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
        // Layer 0 fits the prompt with room to spare; layer 1's two
        // blocks of 2 hold 4 positions against a prompt of 6.
        let stores = SharedPagedKv::from_stores(
            [8usize, 2]
                .into_iter()
                .map(|blocks| {
                    PagedKvStore::new(
                        2,
                        blocks,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );

        let result = decoder.forward_batch_last_paged(&prompt, 0, &mut paged_caches, &stores);
        assert!(result.is_err(), "layer 1's store cannot hold the prompt");
        for (i, cache) in paged_caches.iter().enumerate() {
            assert_eq!(cache.seq_len(), 0, "layer {i} must be untouched");
            assert!(cache.block_table().is_empty(), "layer {i} holds no block");
        }
        for (i, expected) in [8usize, 2].into_iter().enumerate() {
            assert_eq!(stores.free_blocks(i), expected, "layer {i} leaked no block");
        }
    }

    /// Chunked prefill: two calls appending into the same sequence must
    /// equal one call over the concatenation.
    ///
    /// This is the case the part-full tail block breaks if
    /// `to_contiguous` or the reservation is wrong, and it is how the
    /// serving path actually prefills long prompts.
    #[test]
    fn two_paged_prefill_chunks_equal_one_call_over_the_whole_prompt() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 10);
        let prompt = [3usize, 1, 4, 1, 5, 9, 2];
        let split = 3;

        let run = |chunks: &[&[usize]]| {
            let mut caches: Vec<PagedKvCache> = (0..2).map(|_| PagedKvCache::new()).collect();
            let stores = SharedPagedKv::from_stores(
                (0..2)
                    .map(|_| {
                        PagedKvStore::new(2, 16, decoder.config.n_kv_heads, decoder.config.head_dim)
                    })
                    .collect(),
            );
            let mut pos = 0;
            let mut last = Vec::new();
            for chunk in chunks {
                last = decoder
                    .forward_batch_last_paged(chunk, pos, &mut caches, &stores)
                    .expect("sized generously");
                pos += chunk.len();
            }
            last
        };

        let whole = run(&[&prompt]);
        let chunked = run(&[&prompt[..split], &prompt[split..]]);
        assert_eq!(whole.len(), chunked.len());
        for (x, y) in whole.iter().zip(chunked.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "a chunked prefill must equal one call over the same tokens"
            );
        }
    }

    fn paged_matches_contiguous(config: ModelConfig) {
        paged_matches_contiguous_with(config, |_| {});
    }

    /// [`paged_matches_contiguous`] for the features that live on the
    /// WEIGHTS rather than in the config, and so cannot be switched on
    /// by handing a different `ModelConfig` in.
    fn paged_matches_contiguous_with(config: ModelConfig, prepare: impl FnOnce(&mut Decoder)) {
        let n_layers = 2;
        let mut decoder = Decoder::new_random_small(config, n_layers, 10);
        prepare(&mut decoder);
        let decoder = decoder;

        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let steps = [3usize, 5, 7, 2, 9, 1];
        let mut plain_logits = Vec::new();
        for (pos, &tok) in steps.iter().enumerate() {
            plain_logits.push(decoder.forward_token(tok, pos, &mut caches));
        }

        let block_size = 2;
        let mut paged_caches: Vec<PagedKvCache> =
            (0..n_layers).map(|_| PagedKvCache::new()).collect();
        let stores = SharedPagedKv::from_stores(
            (0..n_layers)
                .map(|_| {
                    PagedKvStore::new(
                        block_size,
                        /* total_blocks = */ 16,
                        decoder.config.n_kv_heads,
                        decoder.config.head_dim,
                    )
                })
                .collect(),
        );
        let mut paged_logits = Vec::new();
        for (pos, &tok) in steps.iter().enumerate() {
            paged_logits.push(
                decoder
                    .forward_token_paged(tok, pos, &mut paged_caches, &stores)
                    .expect("store sized generously, must not exhaust"),
            );
        }

        assert_eq!(plain_logits.len(), paged_logits.len());
        for (a, b) in plain_logits.iter().zip(paged_logits.iter()) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "paged decode must be bit-identical to contiguous decode"
                );
            }
        }
    }

    /// The single most important correctness property of
    /// `forward_batch`: batching positions together for shared matmuls
    /// must produce EXACTLY the same result as processing them one at
    /// a time with `forward_token`, since causal masking guarantees
    /// position `i` only ever sees positions `<= i`. If this test
    /// fails, `forward_batch` is not a safe drop-in replacement for
    /// sequential decode, which would make speculative decoding built
    /// on top of it produce silently wrong output.
    /// How far a batched forward may sit from the per-token one.
    ///
    /// MEASURED rather than chosen: over these fixtures the largest
    /// deviation is 2.98e-8, single-ulp f32 on logits around 0.05,
    /// and it comes from the reduction order of a batched matmul
    /// against a per-token one. The bound is loose enough for that and
    /// for another machine's reduction order, and tight enough that a
    /// real drift in the batched path cannot hide under it.
    ///
    /// It was 1e-3 in four separate literals, five orders above the
    /// thing they were nominally checking, and two of the tests using
    /// it had "exactly" in their names.
    const BATCH_VS_SEQUENTIAL_ULP: f32 = 1e-6;

    /// Batched and sequential agree to about one f32 ulp, NOT exactly.
    ///
    /// The name used to say "exactly" and the assertion allowed 1e-3,
    /// which was wrong in both directions at once: the two paths are
    /// not bit-identical (the sibling tests that really are say
    /// `bit_identical`, and mean it), and the real difference is five
    /// orders tighter than the bound that was nominally checking it.
    /// Measured over this fixture: the largest deviation is 1.5e-8,
    /// single-ulp on values around 0.05, arising from the reduction
    /// order of a batched matmul against a per-token one.
    ///
    /// The bound is 1e-6 now: loose enough for that ulp and for a
    /// different machine's reduction order, tight enough that an
    /// actual drift in the batched path cannot hide under it. This
    /// matters beyond hygiene -- the server verifies speculated tokens
    /// against logits from THIS function, so a near-tie inside this
    /// margin is the one case where a speculated token can disagree
    /// with what sequential decoding would have drawn.
    #[test]
    fn forward_batch_matches_sequential_forward_token_to_one_ulp() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        // A second decoder built with the same seed produces identical
        // weights (Decoder::new_random_small is deterministic), so
        // this is a fair like-for-like comparison against a fresh
        // cache rather than reusing decoder_a's now-mutated cache.
        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(seq_logits.len(), batch_logits.len());
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-6,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    /// `forward_batch_last` exists to skip the vocabulary projection for
    /// every position but the last, so the one thing that must hold is
    /// that the row it *does* produce is the same row `forward_batch`
    /// would have produced. It must also leave the KV cache in the same
    /// state -- prefill's whole purpose -- which is checked by decoding
    /// one more token from each cache and comparing.
    #[test]
    fn forward_batch_last_matches_the_final_row_of_forward_batch() {
        let cfg = tiny_test_config();
        let vocab = 16;
        let tokens = vec![1usize, 4, 7, 2, 9];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let all_rows = decoder_a.forward_batch(&tokens, 0, &mut caches_a);

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        let last = decoder_b.forward_batch_last(&tokens, 0, &mut caches_b);

        let expected = all_rows.last().expect("one row per prompt token");
        assert_eq!(last.len(), expected.len());
        for (i, (a, b)) in expected.iter().zip(last.iter()).enumerate() {
            assert!(
                a == b,
                "logit {i}: forward_batch={a} forward_batch_last={b}"
            );
        }

        // Same KV state: the next token's logits must agree too.
        let next_a = decoder_a.forward_token(3, tokens.len(), &mut caches_a);
        let next_b = decoder_b.forward_token(3, tokens.len(), &mut caches_b);
        for (i, (a, b)) in next_a.iter().zip(next_b.iter()).enumerate() {
            assert!(a == b, "post-prefill decode logit {i}: {a} vs {b}");
        }

        // Empty prompt is the degenerate case both paths must survive.
        let mut caches_c: Vec<KvCache> = decoder_b.config.new_kv_caches();
        assert!(decoder_b
            .forward_batch_last(&[], 0, &mut caches_c)
            .is_empty());
    }

    /// `forward_multi_seq`'s core correctness property: batching N
    /// independent sequences (different token histories, different
    /// current positions, different KV caches) together must produce
    /// EXACTLY the same per-sequence output as running each sequence
    /// through `forward_token` alone, one step at a time. This is what
    /// makes continuous batching safe -- no sequence's attention may
    /// ever be perturbed by another sequence sharing its batched
    /// matmul step.
    /// The PAGED batch step must equal the contiguous one, bit for bit.
    ///
    /// Continuous batching and paging are independent choices, so a
    /// deployment can have either, both or neither; if they disagree,
    /// the answer depends on two switches nobody thinks of as changing
    /// the model. Every sequence here is at a different position with a
    /// different length, which is the case the batched path exists for
    /// and the one where a shared-KV mistake would surface.
    #[test]
    fn a_paged_multi_seq_step_is_bit_identical_to_the_contiguous_one() {
        for cfg in [
            tiny_test_config(),
            {
                let mut c = tiny_test_config();
                c.sliding_window = Some(2);
                c.swa_layers = crate::swa_layers::SwaLayers::All;
                c
            },
            {
                let mut c = tiny_test_config();
                c.embedding_scale = Some(7.5);
                c.final_logit_softcap = Some(0.05);
                c.attn_logit_softcap = Some(0.05);
                c
            },
        ] {
            let n_layers = 2;
            let decoder = Decoder::new_random_small(cfg, n_layers, 10);
            let histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];
            let next = [6usize, 1, 2];

            // Contiguous: build each sequence's history, then one step.
            let mut contiguous: Vec<Vec<KvCache>> = histories
                .iter()
                .map(|h| {
                    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
                    for (pos, &tok) in h.iter().enumerate() {
                        decoder.forward_token(tok, pos, &mut caches);
                    }
                    caches
                })
                .collect();
            let positions: Vec<usize> = histories.iter().map(|h| h.len()).collect();
            let want = decoder.forward_multi_seq(&next, &positions, &mut contiguous);

            // Paged: same histories through the paged decode path, then
            // one batched step over the shared store.
            let stores = SharedPagedKv::new(
                n_layers,
                /* block_size = */ 2,
                /* blocks_per_layer = */ 64,
                decoder.config.n_kv_heads,
                decoder.config.head_dim,
            );
            let mut paged: Vec<Vec<PagedKvCache>> = histories
                .iter()
                .map(|h| {
                    let mut caches: Vec<PagedKvCache> =
                        (0..n_layers).map(|_| PagedKvCache::new()).collect();
                    for (pos, &tok) in h.iter().enumerate() {
                        decoder
                            .forward_token_paged(tok, pos, &mut caches, &stores)
                            .expect("sized generously");
                    }
                    caches
                })
                .collect();
            let got = decoder.forward_multi_seq_kv(
                &next,
                &positions,
                &mut MultiSeqKv::Paged {
                    caches: &mut paged,
                    stores: &stores,
                },
            );

            assert_eq!(want.len(), got.len());
            for (s, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(a.len(), b.len(), "sequence {s}: logit count");
                for (x, y) in a.iter().zip(b.iter()) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "sequence {s}: paged batching changed the answer"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_multi_seq_matches_independent_forward_token_per_sequence() {
        let cfg = tiny_test_config();
        let vocab = 8;
        // 3 independent sequences, deliberately different lengths/
        // histories/current tokens, so no two sequences are at the
        // same position when batched together.
        let seq_histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut independent_logits: Vec<Vec<f32>> = Vec::new();
        for history in seq_histories.iter() {
            let mut caches: Vec<KvCache> = decoder_a.config.new_kv_caches();
            let mut logits = Vec::new();
            for (pos, &tok) in history.iter().enumerate() {
                logits = decoder_a.forward_token(tok, pos, &mut caches);
            }
            independent_logits.push(logits);
        }

        // Same seed -> identical weights, fresh caches for a fair
        // comparison (mirrors forward_batch_matches_sequential_forward_token_exactly).
        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut per_seq_caches: Vec<Vec<KvCache>> = seq_histories
            .iter()
            .map(|_| decoder_b.config.new_kv_caches())
            .collect();

        // Feed every sequence's prefix (all but its last token)
        // through forward_multi_seq one shared step at a time, then
        // do a final batched step for the last token of every
        // sequence so all three arrive at their final position in
        // the same batched call -- exercising genuinely different
        // per-sequence positions/histories within one batch, not just
        // parallel identical-length sequences.
        let max_len = seq_histories.iter().map(|h| h.len()).max().unwrap();
        let mut batched_logits: Vec<Vec<f32>> = vec![Vec::new(); seq_histories.len()];
        for step in 0..max_len {
            let mut tokens = Vec::new();
            let mut positions = Vec::new();
            let mut active: Vec<usize> = Vec::new();
            for (s, history) in seq_histories.iter().enumerate() {
                if step < history.len() {
                    tokens.push(history[step]);
                    positions.push(step);
                    active.push(s);
                }
            }
            if tokens.is_empty() {
                continue;
            }
            let mut active_caches: Vec<Vec<KvCache>> = active
                .iter()
                .map(|&s| std::mem::take(&mut per_seq_caches[s]))
                .collect();
            let step_logits = decoder_b.forward_multi_seq(&tokens, &positions, &mut active_caches);
            for ((&s, caches), logits) in active.iter().zip(active_caches).zip(step_logits) {
                per_seq_caches[s] = caches;
                batched_logits[s] = logits;
            }
        }

        assert_eq!(batched_logits.len(), independent_logits.len());
        for (s, (seq_logits, batch_logits)) in independent_logits
            .iter()
            .zip(batched_logits.iter())
            .enumerate()
        {
            assert_eq!(seq_logits.len(), batch_logits.len());
            for (i, (a, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "sequence {s}, logit {i}: independent={a} batched={b}"
                );
            }
        }
    }

    /// The gap the decoration audit found, from the side the existing
    /// guard could not see.
    ///
    /// `the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies`
    /// sets `attention_scale` and compares `forward_token` against
    /// `forward_token_paged` -- the two bodies that AGREED. It never
    /// compared them against `forward_hidden_batch_inner`, which applied
    /// the scale nowhere, so a Gemma-shaped checkpoint would answer at
    /// one temperature when decoded a token at a time and at another
    /// when its prompt was prefilled. Not an error; a plausible
    /// distribution from the wrong model.
    ///
    /// The first assertion is the one that makes this a guard rather
    /// than an assertion: 0.37 is well away from the kernel's own
    /// `1/sqrt(head_dim)`, so if setting it does not move the logits
    /// then both sides are ignoring it and the comparison below proves
    /// nothing.
    /// The Metal attention kernels infer Q/K norm style from the weight
    /// LENGTH; the host branches on `ModelConfig::qk_norm_style`. Two
    /// mechanisms for one decision, so they have to agree.
    ///
    /// They do, and not by luck: `loader.rs`'s `refined_qk_norm` DERIVES
    /// the enum from the same length rule, and refuses to load anything
    /// that matches neither width. This pins that, because the failure
    /// would be silent and would land on audited architectures --
    /// OLMoE is whole-vector, Qwen3 and Gemma-3 are per-head, and all
    /// three are in `AUDITED_GENERIC_GQA`, so an inference that assumed
    /// one style would answer wrong on the others at full speed.
    ///
    /// Raised by the decoration audit as unverifiable from the host
    /// side, which is exactly why it is written down here rather than
    /// left as a comment on one of the two sides.
    #[test]
    fn the_metal_qk_norm_length_rule_is_the_one_the_loader_derives_the_style_from() {
        use crate::capability::QkNormStyle;
        let head_dim = 8usize;
        let n_heads = 4usize;

        // The rule `frink-metal/src/attn.rs` applies, transcribed.
        let metal_says_per_head = |len: usize| len == head_dim;
        // The rule `loader.rs::refined_qk_norm` applies, transcribed.
        let loader_style = |len: usize| -> Option<QkNormStyle> {
            if len == head_dim {
                Some(QkNormStyle::PerHead)
            } else if len == n_heads * head_dim {
                Some(QkNormStyle::WholeVector)
            } else {
                None
            }
        };

        for len in [head_dim, n_heads * head_dim] {
            let style = loader_style(len).expect("both widths load");
            assert_eq!(
                metal_says_per_head(len),
                style == QkNormStyle::PerHead,
                "length {len} loads as {style:?} but Metal would infer the other style"
            );
        }

        // A width neither side handles must be refused at load rather
        // than reaching a kernel that would pick a branch anyway.
        assert!(
            loader_style(head_dim + 1).is_none(),
            "an unrecognised norm width must be a load error, not a coin flip"
        );

        // The one ambiguous case, and it is harmless: with a single
        // head the two widths coincide, so both rules take their PerHead
        // branch and per-head RMS over one head IS whole-vector RMS.
        let single_head = |len: usize| len == head_dim;
        assert!(single_head(head_dim));
        assert_eq!(
            loader_style(head_dim),
            Some(QkNormStyle::PerHead),
            "with n_heads == 1 both widths are head_dim, and both sides must land \
             on the same branch rather than one falling through"
        );
    }

    #[test]
    fn the_batched_path_applies_attention_scale_like_the_contiguous_one() {
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];
        let scaled = || {
            let mut cfg = tiny_test_config();
            // Far from the kernel's own 1/sqrt(head_dim) on purpose:
            // at this model's scale a scalar near 1 moves the logits by
            // ~2e-4, which is below the noise a tolerance test can see.
            cfg.attention_scale = Some(8.0);
            cfg
        };
        let fresh_caches = |d: &Decoder| -> Vec<KvCache> { d.config.new_kv_caches() };

        // Same seed -> identical weights, so the only difference between
        // these three decoders is the config field under test.
        let seq_decoder = Decoder::new_random_small(scaled(), 2, vocab);
        let mut seq_caches = fresh_caches(&seq_decoder);
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| seq_decoder.forward_token(t, pos, &mut seq_caches))
            .collect();

        let batch_decoder = Decoder::new_random_small(scaled(), 2, vocab);
        let mut batch_caches = fresh_caches(&batch_decoder);
        let batched = batch_decoder.forward_batch(&tokens, 0, &mut batch_caches);

        let plain_decoder = Decoder::new_random_small(tiny_test_config(), 2, vocab);
        let mut plain_caches = fresh_caches(&plain_decoder);
        let unscaled = plain_decoder.forward_batch(&tokens, 0, &mut plain_caches);
        assert!(
            batched
                .iter()
                .zip(unscaled.iter())
                .any(|(s, u)| s.iter().zip(u.iter()).any(|(a, b)| (a - b).abs() > 1e-3)),
            "attention_scale must change the batched answer, or this test cannot fail"
        );

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            assert_eq!(seq_logits.len(), batch_logits.len());
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < 1e-5,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    /// [`the_batched_path_applies_attention_scale_like_the_contiguous_one`]
    /// for the fourth host body.
    ///
    /// `forward_multi_seq_kv` did not apply `attention_scale` either, so
    /// a served request answered differently the moment it was batched
    /// with another request -- the same weights, the same position, a
    /// different temperature, decided by how busy the server was.
    #[test]
    fn the_multi_seq_path_applies_attention_scale_like_the_contiguous_one() {
        let vocab = 8;
        let histories: [&[usize]; 3] = [&[1, 3, 5], &[2, 7], &[4, 4, 4, 6]];
        let next = [6usize, 1, 2];
        let n_layers = 2;
        let scaled = || {
            let mut cfg = tiny_test_config();
            // See the batched twin: a scalar near 1 does not move this
            // model's logits far enough for a tolerance to see it.
            cfg.attention_scale = Some(8.0);
            cfg
        };

        // Builds every sequence's history with `forward_token`, then
        // takes the next step either per sequence or as one batch.
        let run = |cfg: ModelConfig, batched: bool| -> Vec<Vec<f32>> {
            let decoder = Decoder::new_random_small(cfg, n_layers, vocab);
            let mut per_seq: Vec<Vec<KvCache>> = histories
                .iter()
                .map(|h| {
                    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
                    for (pos, &tok) in h.iter().enumerate() {
                        decoder.forward_token(tok, pos, &mut caches);
                    }
                    caches
                })
                .collect();
            let positions: Vec<usize> = histories.iter().map(|h| h.len()).collect();
            if batched {
                decoder.forward_multi_seq(&next, &positions, &mut per_seq)
            } else {
                next.iter()
                    .zip(positions.iter())
                    .zip(per_seq.iter_mut())
                    .map(|((&tok, &pos), caches)| decoder.forward_token(tok, pos, caches))
                    .collect()
            }
        };

        let want = run(scaled(), false);
        let got = run(scaled(), true);
        let unscaled = run(tiny_test_config(), true);

        assert!(
            got.iter()
                .zip(unscaled.iter())
                .any(|(g, u)| g.iter().zip(u.iter()).any(|(a, b)| (a - b).abs() > 1e-3)),
            "attention_scale must change the multi-seq answer, or this test cannot fail"
        );

        assert_eq!(want.len(), got.len());
        for (s, (a, b)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "sequence {s}: logit count");
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-5,
                    "sequence {s}, logit {i}: independent={x} batched={y}"
                );
            }
        }
    }

    /// The predicate that decides whether a MoE layer may be routed by
    /// the GPU must admit ONLY the routing the GPU actually computes.
    ///
    /// Every Metal MoE path -- `launch_moe_decode_stack`,
    /// `launch_moe_decode_layer_fused`, `launch_moe_prefill_q4_0` and
    /// the fused prefill stack -- routes with a plain top-k softmax over
    /// the raw router logits. `Decoder::route_for_layer` has three more
    /// arms: grouped routing, a per-expert router bias, and
    /// `expert_weights_scale`. The audit found those four call sites
    /// disagreeing about which of the three to refuse -- prefill checked
    /// all three, the fused decode layer checked two, the whole-stack
    /// decode checked none -- so a Softmax-gated MoE checkpoint carrying
    /// a router bias would have routed to different experts on Metal
    /// than on CPU, with no error.
    ///
    /// This asserts the invariant directly rather than the predicate's
    /// spelling: whenever it says yes, plain `route_top_k` and
    /// `route_for_layer` must return the same decision; and each of the
    /// three features on its own must make it say no.
    #[test]
    fn the_gpu_router_predicate_admits_only_routing_it_reproduces() {
        // `tiny_test_config` is GLM-shaped and so gates with sigmoid;
        // the GPU router implements softmax, so start from the case the
        // predicate is supposed to ADMIT.
        let mut base = tiny_test_config();
        base.moe.gating = frink_moe::GatingFunction::Softmax;
        let decoder = Decoder::new_random_small(base.clone(), 2, 8);
        let plain_layer = &decoder.layers[0];
        let n_experts = base.moe.n_experts;
        // Chosen so each feature really bites: the top two experts sit
        // in DIFFERENT groups of two (so grouped routing must reorder
        // them), and the runners-up are close enough behind that a
        // per-expert bias flips the order.
        assert_eq!(n_experts, 6, "the logits below are written for six experts");
        let logits: Vec<f32> = vec![0.90, 0.10, 0.20, 0.85, 0.30, 0.05];

        let agrees = |layer: &LayerWeights, cfg: &ModelConfig| -> bool {
            let host = Decoder::route_for_layer(layer, &logits, cfg);
            let gpu = route_top_k(
                &logits,
                cfg.moe.n_experts_active,
                cfg.moe.gating,
                cfg.moe.norm_topk_prob,
            );
            host.expert_ids == gpu.expert_ids
                && host.weights.len() == gpu.weights.len()
                && host
                    .weights
                    .iter()
                    .zip(gpu.weights.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };

        // The admitted case: the predicate says yes, and the two
        // routers really do agree.
        assert!(
            Decoder::gpu_router_matches_host_routing(plain_layer, &base),
            "a plain softmax MoE layer must stay eligible, or this test proves nothing"
        );
        assert!(agrees(plain_layer, &base));

        // A per-expert router bias.
        let mut biased_decoder = Decoder::new_random_small(base.clone(), 2, 8);
        biased_decoder.layers[0].moe.exp_probs_bias =
            Some((0..n_experts).map(|e| 0.9 - 0.4 * e as f32).collect());
        let biased_layer = &biased_decoder.layers[0];
        assert!(
            !Decoder::gpu_router_matches_host_routing(biased_layer, &base),
            "exp_probs_bias must make the layer ineligible for the GPU router"
        );
        assert!(
            !agrees(biased_layer, &base),
            "the bias must actually change the routing, or the check above is vacuous"
        );

        // `expert_weights_scale`.
        let mut scaled = base.clone();
        scaled.moe.expert_weights_scale = 2.5;
        assert!(
            !Decoder::gpu_router_matches_host_routing(plain_layer, &scaled),
            "expert_weights_scale must make the layer ineligible for the GPU router"
        );
        assert!(
            !agrees(plain_layer, &scaled),
            "the scale must actually change the routing, or the check above is vacuous"
        );

        // Grouped routing.
        let mut grouped = base.clone();
        grouped.moe.expert_group_count = Some(3);
        grouped.moe.expert_group_used_count = Some(1);
        assert!(
            !Decoder::gpu_router_matches_host_routing(plain_layer, &grouped),
            "grouped routing must make the layer ineligible for the GPU router"
        );
        assert!(
            !agrees(plain_layer, &grouped),
            "the grouping must actually change the routing, or the check above is vacuous"
        );

        // A non-softmax gate: the GPU kernel implements softmax only.
        let mut sigmoid = base.clone();
        sigmoid.moe.gating = frink_moe::GatingFunction::Sigmoid;
        assert!(
            !Decoder::gpu_router_matches_host_routing(plain_layer, &sigmoid),
            "a non-softmax gate must make the layer ineligible for the GPU router"
        );

        // A router that reads the raw layer input (`smallthinker`):
        // every GPU router reads `normed2`, so the SAME logits routed
        // the same way are still the wrong experts. `agrees` cannot see
        // this one -- it compares two routers over one logit vector --
        // which is exactly why the predicate has to.
        let mut raw = base.clone();
        raw.router_input = crate::router_input::RouterInput::RawLayerInput;
        assert!(
            !Decoder::gpu_router_matches_host_routing(plain_layer, &raw),
            "a raw-layer-input router must make the layer ineligible for the GPU router"
        );

        // A router (and experts) reading the normed layer input
        // (`arctic`): the same predicate, for the same reason, and the
        // experts would read the wrong tensor too.
        let mut branch = base;
        branch.router_input = crate::router_input::RouterInput::NormedLayerInput;
        assert!(
            !Decoder::gpu_router_matches_host_routing(plain_layer, &branch),
            "a normed-layer-input branch must make the layer ineligible for the GPU router"
        );
    }

    /// OLMoE-style QK-norm (`attn_q_norm`/`attn_k_norm`, see `AttnWeights`'
    /// doc comment): with both set, `forward_batch` must still match
    /// sequential `forward_token` calls exactly -- the same consistency
    /// property `forward_batch_matches_sequential_forward_token_exactly`
    /// checks for the no-QK-norm path, now exercising the norm-applied
    /// per-row slicing (`q_batch.chunks_mut(q_width)`,
    /// `k_batch.chunks_mut(kv_width)`) instead of trusting it by
    /// inspection.
    #[test]
    fn forward_batch_matches_forward_token_with_qk_norm_present() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;

        let mut decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        for layer in &mut decoder_a.layers {
            layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
            layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
        }
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let mut decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        for layer in &mut decoder_b.layers {
            layer.attn.q_norm = Some((0..q_width).map(|i| 1.0 + i as f32 * 0.1).collect());
            layer.attn.k_norm = Some((0..kv_width).map(|i| 0.5 + i as f32 * 0.05).collect());
        }
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    /// QK-norm being present must actually change the output -- otherwise
    /// the `Some(...)` branches in `forward_token`/`forward_batch` could
    /// silently be dead code and this feature would ship unverified. Must
    /// decode at least 2 positions: at position 0 with a fresh cache,
    /// causal softmax has exactly one candidate (the token attending to
    /// itself) and always evaluates to weight 1.0 regardless of the Q*K
    /// dot product -- so the attention output there is Q/K-invariant by
    /// construction, and a single-position version of this test would
    /// pass even with `q_norm`/`k_norm` silently never applied.
    #[test]
    fn qk_norm_present_changes_output_versus_absent() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        let tokens = [3usize, 5];

        let without_norm = Decoder::new_random_small(cfg.clone(), 1, vocab);
        let mut with_norm = Decoder::new_random_small(cfg, 1, vocab);
        for layer in &mut with_norm.layers {
            layer.attn.q_norm = Some(vec![2.0; q_width]);
            layer.attn.k_norm = Some(vec![2.0; kv_width]);
        }

        let mut caches_a: Vec<KvCache> = without_norm.config.new_kv_caches();
        let mut caches_b: Vec<KvCache> = with_norm.config.new_kv_caches();

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            out_a = without_norm.forward_token(t, pos, &mut caches_a);
            out_b = with_norm.forward_token(t, pos, &mut caches_b);
        }

        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "QK-norm weights changed nothing -- forward_token likely isn't applying q_norm/k_norm"
        );
    }

    /// Qwen2/Qwen2-MoE-family QKV attention bias (`AttnWeights::q_bias`/
    /// `k_bias`/`v_bias`): a real, previously-unhandled gap found by
    /// running frink's generic GGUF loader against a real downloaded
    /// Qwen1.5-MoE-A2.7B-Chat checkpoint, which produced fluent-but-wrong
    /// output because these real `attn_{q,k,v}.bias` tensors were
    /// silently never added anywhere. Same two real properties checked
    /// as the QK-norm tests above: (1) `forward_batch` must match
    /// sequential `forward_token` exactly with bias present (batched
    /// per-row broadcast must be correct, not just the single-token
    /// path), and (2) bias must actually change the output at position
    /// 0 or later (not silently dead code) -- checked at position 1
    /// specifically, since position 0's causal softmax has exactly one
    /// candidate and is Q/K-invariant regardless of any additive bias
    /// shifting Q/K, for the same reason the QK-norm test above needs
    /// >=2 positions.
    #[test]
    fn forward_batch_matches_forward_token_with_qkv_bias_present() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;

        let mut decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        for layer in &mut decoder_a.layers {
            layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
            layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
            layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
        }
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let mut decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        for layer in &mut decoder_b.layers {
            layer.attn.q_bias = Some((0..q_width).map(|i| 0.3 + i as f32 * 0.02).collect());
            layer.attn.k_bias = Some((0..kv_width).map(|i| -0.2 + i as f32 * 0.03).collect());
            layer.attn.v_bias = Some((0..kv_width).map(|i| 0.1 - i as f32 * 0.01).collect());
        }
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    #[test]
    fn qkv_bias_present_changes_output_versus_absent() {
        let cfg = tiny_test_config();
        let vocab = 8;
        let q_width = cfg.n_heads * cfg.head_dim;
        let kv_width = cfg.n_kv_heads * cfg.head_dim;
        let tokens = [3usize, 5];

        let without_bias = Decoder::new_random_small(cfg.clone(), 1, vocab);
        let mut with_bias = Decoder::new_random_small(cfg, 1, vocab);
        for layer in &mut with_bias.layers {
            layer.attn.q_bias = Some(vec![0.5; q_width]);
            layer.attn.k_bias = Some(vec![0.5; kv_width]);
            layer.attn.v_bias = Some(vec![0.5; kv_width]);
        }

        let mut caches_a: Vec<KvCache> = without_bias.config.new_kv_caches();
        let mut caches_b: Vec<KvCache> = with_bias.config.new_kv_caches();

        let mut out_a = Vec::new();
        let mut out_b = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            out_a = without_bias.forward_token(t, pos, &mut caches_a);
            out_b = with_bias.forward_token(t, pos, &mut caches_b);
        }

        let differs = out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "QKV bias changed nothing -- forward_token likely isn't applying q_bias/k_bias/v_bias"
        );
    }

    #[test]
    fn forward_batch_and_forward_token_leave_kv_caches_in_the_same_state() {
        let cfg = tiny_test_config();
        let tokens = [2usize, 4, 6];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, 8);
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        for (pos, &t) in tokens.iter().enumerate() {
            decoder_a.forward_token(t, pos, &mut caches_a);
        }

        let decoder_b = Decoder::new_random_small(cfg, 2, 8);
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        for (ca, cb) in caches_a.iter().zip(caches_b.iter()) {
            assert_eq!(ca.positions(), cb.positions());
            assert_eq!(ca.k.len(), cb.k.len());
            for (a, b) in ca.k.iter().zip(cb.k.iter()) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }

    /// Same architecture shape as `tiny_test_config` but genuinely
    /// dense (one expert, no shared experts) -- the shape every non-MoE
    /// model, and every DeepSeek-style leading dense layer, loads as.
    /// Exercises `Decoder::is_dense_layer`'s fast path.
    fn tiny_dense_test_config() -> ModelConfig {
        let mut cfg = tiny_test_config();
        cfg.moe.n_experts = 1;
        cfg.moe.n_experts_active = 1;
        cfg.moe.n_shared_experts = 0;
        cfg
    }

    #[test]
    fn dense_layer_forward_pass_produces_finite_logits_of_correct_shape() {
        let vocab = 10;
        let decoder = Decoder::new_random_small(tiny_dense_test_config(), 2, vocab);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        let logits = decoder.forward_token(3, 0, &mut caches);
        assert_eq!(logits.len(), vocab);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must not contain NaN/Inf"
        );
    }

    #[test]
    fn dense_layer_forward_batch_matches_sequential_forward_token_to_one_ulp() {
        let cfg = tiny_dense_test_config();
        let vocab = 8;
        let tokens = [1usize, 3, 5, 2, 7];

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        let sequential: Vec<Vec<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| decoder_a.forward_token(t, pos, &mut caches_a))
            .collect();

        let decoder_b = Decoder::new_random_small(cfg, 2, vocab);
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        let batched = decoder_b.forward_batch(&tokens, 0, &mut caches_b);

        assert_eq!(batched.len(), sequential.len());
        for (pos, (seq_logits, batch_logits)) in sequential.iter().zip(batched.iter()).enumerate() {
            for (i, (s, b)) in seq_logits.iter().zip(batch_logits.iter()).enumerate() {
                assert!(
                    (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                    "position {pos}, logit {i}: sequential={s} batched={b}"
                );
            }
        }
    }

    #[test]
    fn dense_layer_fast_path_still_records_expert_zero_activations() {
        // The dense fast path bypasses `route_top_k` entirely, but
        // must still record an activation for expert 0 every step --
        // `MoeWeights::placement_plan` and hotness-based GPU placement
        // depend on this being real for every model shape, not just
        // genuinely-MoE ones.
        let decoder = Decoder::new_random_small(tiny_dense_test_config(), 1, 8);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        decoder.forward_token(0, 0, &mut caches);
        decoder.forward_token(1, 1, &mut caches);
        decoder.forward_token(2, 2, &mut caches);

        let count =
            decoder.layers[0].moe.activation_counts[0].load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(count, 3);
    }

    #[test]
    fn forward_batch_with_empty_tokens_returns_empty() {
        let decoder = Decoder::new_random_small(tiny_test_config(), 2, 8);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        let out = decoder.forward_batch(&[], 0, &mut caches);
        assert!(out.is_empty());
    }

    #[test]
    fn forward_batch_continues_correctly_after_prior_forward_token_calls() {
        // Realistic usage pattern: some tokens processed one at a time
        // (e.g. the first generated token), then a batch verifying
        // several draft tokens at once, continuing from the same
        // cache. The batch's positions must be numbered starting from
        // wherever the cache left off, not from zero.
        let cfg = tiny_test_config();

        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, 8);
        let mut caches_a: Vec<KvCache> = decoder_a.config.new_kv_caches();
        decoder_a.forward_token(1, 0, &mut caches_a);
        decoder_a.forward_token(3, 1, &mut caches_a);
        let seq_next = decoder_a.forward_token(5, 2, &mut caches_a);

        let decoder_b = Decoder::new_random_small(cfg, 2, 8);
        let mut caches_b: Vec<KvCache> = decoder_b.config.new_kv_caches();
        decoder_b.forward_token(1, 0, &mut caches_b);
        let batch_next = decoder_b.forward_batch(&[3, 5], 1, &mut caches_b);

        for (s, b) in seq_next.iter().zip(batch_next[1].iter()) {
            assert!(
                (s - b).abs() < BATCH_VS_SEQUENTIAL_ULP,
                "sequential={s} batched={b}"
            );
        }
    }

    /// `PlacementPlan::from_budget` is
    /// real and tested in isolation, but only meaningful once it's fed
    /// genuinely observed per-expert activation counts rather than
    /// zeros. This proves the full loop: run real forward passes,
    /// confirm `MoeWeights::activation_counts` actually reflects what
    /// `route_top_k` selected, and confirm `placement_plan` prioritizes
    /// the expert that was genuinely hottest -- not just that the
    /// budget/size arithmetic works on synthetic inputs.
    #[test]
    fn placement_plan_reflects_real_observed_expert_activations() {
        let cfg = tiny_test_config(); // 6 experts, top-2 active/token
        let decoder = Decoder::new_random_small(cfg, 2, 16);
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

        let n_calls = 20;
        for pos in 0..n_calls {
            decoder.forward_token(pos % 16, pos, &mut caches);
        }

        let layer0 = &decoder.layers[0].moe;
        let counts: Vec<u64> = layer0
            .activation_counts
            .iter()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .collect();
        let total: u64 = counts.iter().sum();
        assert_eq!(
            total,
            (n_calls as u64) * (decoder.config.moe.n_experts_active as u64),
            "total recorded activations must equal calls * experts_active_per_call"
        );

        // Ties are realistic at this small a sample size; break them the
        // same way `PlacementPlan::from_budget` does (lowest index
        // wins), so this assertion can't spuriously fail on a tie that
        // `from_budget` resolves differently than a naive `max_by_key`
        // (which returns the *last* max element) would.
        let hottest_count = *counts.iter().max().unwrap();
        let hottest_idx = counts.iter().position(|&c| c == hottest_count).unwrap();
        assert!(hottest_count > 0);

        // A per-expert resident size big enough for exactly one expert.
        let per_expert_bytes = layer0.expert_bytes(0);
        let plan = layer0.placement_plan(per_expert_bytes as u64);

        assert_eq!(
            plan.placement_for(hottest_idx),
            frink_moe::ExpertPlacement::GpuDevice(0),
            "the genuinely hottest expert (index {hottest_idx}, {hottest_count} activations) \
             must be the one the plan places on GPU when only one expert fits the budget"
        );
    }
}

/// The Metal side of Phi-3/Phi-4's RoPE: partial rotary and LongRoPE's
/// `attn_factor` used to be a refusal in `layer_supports_metal_attn`
/// and are now two uniforms on [`frink_metal::attn::MetalRope`].
#[cfg(all(test, feature = "metal"))]
mod metal_rope_tests {
    use super::*;

    fn phi_like_config() -> ModelConfig {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 128;
        cfg.rope_layout = crate::config::RopeLayout::Neox;
        cfg.rope_dim = Some(96);
        cfg.rope_attn_factor = 1.1902381;
        cfg
    }

    /// Both values must reach the kernels, and they must be the same two
    /// the CPU path reads — otherwise the backends compute different
    /// attention for the same weights, which is the whole reason the
    /// model was refused Metal in the first place.
    #[test]
    fn metal_rope_carries_partial_rotary_and_mscale() {
        let decoder = Decoder::new_random_small(phi_like_config(), 1, 32);
        let rope = decoder.metal_rope();
        assert_eq!(rope.layout, frink_metal::attn::MetalRopeLayout::Neox);
        assert_eq!(rope.rot_dim, Some(96));
        assert_eq!(rope.attn_factor, 1.1902381);
    }

    /// `rope.dimension_count == head_dim` is "the whole head rotates",
    /// which must reach the kernel as `None` rather than as a width —
    /// same graph, one code path.
    #[test]
    fn rot_dim_equal_to_head_dim_becomes_none() {
        let mut cfg = phi_like_config();
        cfg.rope_dim = Some(cfg.head_dim);
        let decoder = Decoder::new_random_small(cfg, 1, 32);
        assert_eq!(decoder.metal_rope().rot_dim, None);
    }

    /// A non-unit `attn_factor` is no longer a reason to refuse Metal;
    /// an odd `n_rot` still is, because ggml's `ggml_rope_impl` asserts
    /// an even width and the split-half pairing is otherwise undefined
    /// for the last channel.
    #[test]
    fn odd_rot_dim_is_still_refused_but_mscale_is_not() {
        let supported = |cfg: ModelConfig| {
            let d = Decoder::new_random_small(cfg, 1, 32);
            d.layer_supports_metal_attn(&d.layers[0])
        };

        // The control: with no rope oddity the fixture is admitted, so
        // the two assertions below are about the rope config and not
        // about the fixture failing some other check.
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        assert!(supported(plain), "fixture must be Metal-eligible to start");

        assert!(
            supported(phi_like_config()),
            "partial rotary + a non-unit attn_factor must no longer refuse Metal"
        );

        let mut odd = phi_like_config();
        odd.rope_dim = Some(95);
        assert!(!supported(odd), "odd n_rot must keep the model off Metal");
    }

    /// A `residual_scale` keeps every fused Metal path off the model,
    /// through ONE predicate the four eligibility checks share.
    ///
    /// Granite multiplies both branch outputs before every residual add.
    /// The fused launches -- dense decode, resident MoE decode, and both
    /// prefill stacks -- fold the residual add in on device with no
    /// uniform for a multiplier, so a Granite layer served by any of
    /// them would be scaled by the host bodies and not by the GPU: the
    /// same weights answering differently depending on which backend
    /// took the token. That is exactly what `attention_scale` is fenced
    /// off for next door, and this repo has already watched the GPU
    /// router's eligibility check drift four ways when it was four
    /// spellings instead of one.
    ///
    /// Only reachable in a `--features metal` build, which is where the
    /// fence exists at all.
    #[test]
    fn a_residual_scale_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );

        let mut scaled = plain;
        scaled.residual_scale = Some(0.22);
        let d = Decoder::new_random_small(scaled.clone(), 1, 32);
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a residual multiplier no Metal kernel applies must refuse the fused attention"
        );
        assert!(
            !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &scaled, false),
            "...and the prefill dense stack"
        );
        assert!(
            !Decoder::metal_can_serve_model(&scaled, false),
            "the shared predicate is what all four read"
        );
    }

    /// A `clamp_kqv` keeps every fused Metal path off the model too,
    /// through the same predicate.
    ///
    /// The fused launches apply the QKV bias inside their kernels via
    /// `AttnExtras` and clamp nothing, while the host bodies clamp
    /// after the bias (`decoder/qkv_bias.rs`). A DBRX layer served by a
    /// fused launch would therefore run unclamped projections -- the
    /// disagreement the residual-scale fence exists for, on a different
    /// scalar. Only reachable in a `--features metal` build.
    #[test]
    fn a_qkv_clamp_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );

        let mut clamped = plain;
        clamped.clamp_kqv = Some(8.0);
        let d = Decoder::new_random_small(clamped.clone(), 1, 32);
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a clamp no Metal kernel applies must refuse the fused attention"
        );
        assert!(
            !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &clamped, false),
            "...and the prefill dense stack"
        );
        assert!(!Decoder::metal_can_serve_model(&clamped, false));
    }

    /// Per-layer shapes keep every fused Metal path off the model,
    /// through the same predicate.
    ///
    /// The fused launches take ONE `n_heads` argument and one
    /// `MetalKvBuffers` geometry per run of layers, and the Metal KV
    /// plane is sized once from the scalars. A deci or openelm layer
    /// served by any of them would be projected and cached at another
    /// layer's width -- the disagreement the two fences above exist
    /// for, on a shape rather than a scalar. Only reachable in a
    /// `--features metal` build.
    #[test]
    fn per_layer_shapes_keep_the_model_off_every_fused_metal_path() {
        use crate::layer_shapes::{AttnShape, LayerShape, LayerShapes};
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );

        let mut shaped = plain;
        shaped.layer_shapes = LayerShapes::PerLayer(vec![LayerShape {
            attention: AttnShape::Gqa {
                n_heads: shaped.n_heads,
                n_kv_heads: shaped.n_kv_heads,
            },
            ffn_dim: shaped.moe.expert_ffn_dim,
        }]);
        let d = Decoder::new_random_small(shaped.clone(), 1, 32);
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a per-layer table, even one whose single entry agrees with the scalars, must \
             refuse the fused attention: the stacks hold one geometry"
        );
        assert!(
            !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &shaped, false),
            "...and the prefill dense stack"
        );
        assert!(!Decoder::metal_can_serve_model(&shaped, false));
    }

    /// A per-position attention temperature keeps every fused Metal
    /// path off the model, through the same predicate.
    ///
    /// No fused launch takes a per-token Q scale: the host bodies
    /// multiply Q by `log(floor(pos / floor) + 1) * scale + 1` after
    /// RoPE (`crate::attn_temperature`), and a Ministral-3 layer served
    /// by a fused launch would attend at temperature 1 while its
    /// neighbours on the host stepped with position -- the same
    /// disagreement the three fences above exist for. Only reachable
    /// in a `--features metal` build.
    #[test]
    fn a_per_position_temperature_keeps_the_model_off_every_fused_metal_path() {
        use crate::attn_temperature::AttnTemperature;
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );

        let mut tempered = plain;
        tempered.attn_temperature = Some(AttnTemperature {
            scale: 0.5,
            floor_scale: std::num::NonZeroU32::new(2).unwrap(),
            offset: 0.0,
            unrotated_layers_only: false,
        });
        let d = Decoder::new_random_small(tempered.clone(), 1, 32);
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a per-position Q scale no Metal kernel applies must refuse the fused attention"
        );
        assert!(
            !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &tempered, false),
            "...and the prefill dense stack"
        );
        assert!(!Decoder::metal_can_serve_model(&tempered, false));
    }

    /// BitNet's two inner norms keep the model off every fused Metal
    /// path, through the same predicate, and a layer that carries the
    /// attention one off the per-layer fused attention through the
    /// exhaustive destructure in `metal_attn_view`.
    ///
    /// No fused kernel norms between the V sum and `wo` or between the
    /// activation and `down`; a BitNet layer served by one would skip
    /// both norms at full speed (`crate::sub_norms`). Only reachable in
    /// a `--features metal` build.
    /// A parallel-residual model (`crate::parallel_residual`) stays off
    /// every fused Metal path: each of them bakes the pre-FFN norm over
    /// the POST-ATTENTION residual into its kernel, and a parallel layer
    /// norms the layer INPUT. The model-level flag is what the
    /// predicate reads, because the per-layer fact lives on
    /// `MoeWeights` and the two config-only callers cannot see it. Only
    /// reachable in a `--features metal` build.
    /// A learned-position model (`crate::position_embd`) stays off every
    /// fused Metal path: the GPU embedding gather has no add, and no
    /// stack sees `pos` for it. Only reachable in a `--features metal`
    /// build.
    /// An ALiBi model (`crate::alibi`) stays off every fused Metal path:
    /// no kernel adds a per-key bias to its scores. Only reachable in a
    /// `--features metal` build.
    #[test]
    fn an_alibi_bias_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );
        let mut biased = plain;
        biased.alibi_max_bias = Some(8.0);
        let d = Decoder::new_random_small(biased.clone(), 1, 32);
        assert!(d.alibi_slopes.is_some(), "derived from the config");
        assert!(!d.layer_supports_metal_attn(&d.layers[0]));
        assert!(!Decoder::metal_can_serve_model(&biased, false));
    }

    #[test]
    fn a_learned_position_table_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );
        let mut positioned = plain;
        positioned.learned_positions = true;
        let d = Decoder::new_random_small(positioned.clone(), 1, 32);
        assert!(!d.layer_supports_metal_attn(&d.layers[0]));
        assert!(!Decoder::metal_can_serve_model(&positioned, false));
    }

    #[test]
    fn a_parallel_residual_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );
        let mut parallel = plain;
        parallel.parallel_residual = true;
        let d = Decoder::new_random_small(parallel.clone(), 1, 32);
        assert!(!d.layer_supports_metal_attn(&d.layers[0]));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &parallel,
            false
        ));
        assert!(!Decoder::metal_can_serve_model(&parallel, false));
    }

    /// Llama 4's two facts (`crate::chunked_swa`,
    /// `crate::weightless_qk_norm`) each keep the model off every fused
    /// launch: one window per layer, no weightless post-RoPE QK norm.
    #[test]
    fn a_chunked_window_and_the_weightless_qk_norm_each_keep_the_model_on_the_host() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        assert!(Decoder::metal_can_serve_model(&plain, false));
        let mut chunked = plain.clone();
        chunked.sliding_window = Some(8192);
        chunked.swa_chunked = true;
        assert!(!Decoder::metal_can_serve_model(&chunked, false));
        let mut normed = plain;
        normed.weightless_qk_norm = true;
        assert!(!Decoder::metal_can_serve_model(&normed, false));
    }

    #[test]
    fn the_inner_norms_keep_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );

        let mut sub_normed = plain;
        sub_normed.block_sub_norms = true;
        let d = Decoder::new_random_small(sub_normed.clone(), 1, 32);
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a norm between attention and `wo` that no Metal kernel applies must refuse the \
             fused attention"
        );
        assert!(
            !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &sub_normed, false),
            "...and the prefill dense stack"
        );
        assert!(!Decoder::metal_can_serve_model(&sub_normed, false));

        // Independently of the config: the tensor on the layer is enough
        // to lose the per-layer fused attention's view of it.
        let mut d = Decoder::new_random_small(plain_config_with_metal_view(), 1, 32);
        assert!(d.metal_attn_view(&d.layers[0]).is_some());
        d.layers[0].attn.attn_sub_norm = Some(vec![1.0; d.config.hidden_dim]);
        assert!(d.metal_attn_view(&d.layers[0]).is_none());
    }

    /// A V head width that differs from K's, and a scale after `wo`,
    /// each keep the model off every fused Metal path through the same
    /// predicate: every fused launch takes ONE head width for its KV
    /// buffers, its attention tile and its `wo` fold, and none scales
    /// after the fold (`crate::kv_head_dims`, `crate::attn_value_scale`).
    /// Only reachable in a `--features metal` build.
    #[test]
    fn a_split_kv_head_width_or_a_value_scale_keeps_the_model_off_every_fused_metal_path() {
        let plain = plain_config_with_metal_view();
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");

        let mut split = plain.clone();
        split.v_head_dim = Some(plain.head_dim / 2);
        assert!(!Decoder::metal_can_serve_model(&split, false));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &split,
            false
        ));

        let mut scaled = plain;
        scaled.attn_value_scale = Some(0.707);
        assert!(!Decoder::metal_can_serve_model(&scaled, false));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &scaled,
            false
        ));
    }

    /// A looped model keeps off every fused Metal path: each launch
    /// indexes weights and KV buffers with one `l`, and a looped model's
    /// logical layers outnumber its weights (`crate::layer_loops`). Only
    /// reachable in a `--features metal` build.
    #[test]
    fn a_layer_loop_keeps_the_model_off_every_fused_metal_path() {
        let plain = plain_config_with_metal_view();
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");
        let mut looped = plain;
        looped.layer_loops = Some(crate::layer_loops::LayerLoops::Repeat {
            n_phys: 1,
            n_loops: 2,
            skip_loop_final_norm: false,
        });
        assert!(!Decoder::metal_can_serve_model(&looped, false));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &looped,
            false
        ));
    }

    /// Talkie's facts keep the model off every fused Metal path: the
    /// skip stream and the per-head scalar QK gain through the model
    /// predicate, and a projection gain on a layer through the
    /// exhaustive destructure and the dense-FFN predicate
    /// (`crate::skip_stream`, `crate::weight_scales`). Only reachable in
    /// a `--features metal` build.
    #[test]
    fn talkie_s_facts_keep_the_model_off_every_fused_metal_path() {
        let plain = plain_config_with_metal_view();
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");

        let mut skip = plain.clone();
        skip.skip_stream = true;
        assert!(!Decoder::metal_can_serve_model(&skip, false));

        let mut scalar = plain.clone();
        scalar.qk_norm_style = crate::capability::QkNormStyle::PerHeadScalar;
        assert!(!Decoder::metal_can_serve_model(&scalar, false));

        let mut d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(d.metal_attn_view(&d.layers[0]).is_some());
        d.layers[0].attn.o_scale = Some(1.5);
        assert!(d.metal_attn_view(&d.layers[0]).is_none());

        let mut d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &plain,
            false
        ));
        d.layers[0].moe.down_scale = Some(0.5);
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &plain,
            false
        ));
        assert!(!Decoder::layer_supports_metal_dense_ffn(&d.layers[0]));
    }

    fn plain_config_with_metal_view() -> ModelConfig {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        plain
    }

    /// An FFN activation no fused kernel spells keeps the model off
    /// every fused Metal path, through the same predicate.
    ///
    /// Six launch sites used to derive the kernels' `gelu: bool` as
    /// `!is_swiglu()`, which would have run the ungated ReLU-squared FFN
    /// (`arcee`) as GELU; `GluAct::fused_kernel_gelu_flag` is `None`
    /// there now, and this is the fence that keeps a layer from being
    /// half-served. Only reachable in a `--features metal` build.
    #[test]
    fn an_activation_no_kernel_spells_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        plain.ffn_activation = crate::config::FfnActivation::Swiglu;
        let d = Decoder::new_random_small(plain.clone(), 1, 32);
        assert!(d.layer_supports_metal_attn(&d.layers[0]));

        let mut ungated = plain;
        ungated.ffn_activation = crate::config::FfnActivation::ReluSqr;
        let d = Decoder::new_random_small(ungated.clone(), 1, 32);
        assert!(!d.layer_supports_metal_attn(&d.layers[0]));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &ungated,
            false
        ));
        assert!(!Decoder::metal_can_serve_model(&ungated, false));
        assert_eq!(
            ungated
                .model_ffn_act()
                .and_then(GluAct::fused_kernel_gelu_flag),
            None
        );

        // The two PARAMETERISED activations answer no whole-model
        // activation at all, and the two-width rotary is a fourth thing
        // the fused stacks cannot take; each keeps the model off alone.
        let mut xielu = ungated.clone();
        xielu.ffn_activation =
            crate::config::FfnActivation::Xielu(crate::act_layers::XieluLayers::new(vec![
                frink_moe::XieluParams::from_gguf(
                    0.8, 0.8, 0.5, -1e-6
                );
                xielu.n_layers
            ]));
        assert_eq!(xielu.model_ffn_act(), None);
        assert!(!Decoder::metal_can_serve_model(&xielu, false));
        let mut clamped = ungated.clone();
        clamped.ffn_activation =
            crate::config::FfnActivation::SwigluClamped(crate::act_layers::SwigluClamps::new(
                vec![0.0; clamped.n_layers],
                vec![7.0; clamped.n_layers],
                frink_moe::ClampForm::AfterSilu,
            ));
        assert_eq!(clamped.model_ffn_act(), None);
        assert!(!Decoder::metal_can_serve_model(&clamped, false));
        let mut two_widths = ungated;
        two_widths.ffn_activation = crate::config::FfnActivation::Swiglu;
        assert!(Decoder::metal_can_serve_model(&two_widths, false));
        two_widths.sliding_window = Some(4);
        two_widths.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        two_widths.n_layers = 2;
        two_widths.rope_dim_swa = Some(two_widths.head_dim / 2);
        assert!(two_widths.rope_dim_varies_by_layer());
        assert!(!Decoder::metal_can_serve_model(&two_widths, false));
    }

    /// An attention output gate or a set of attention sinks keeps THAT
    /// LAYER off every fused Metal attention launch, through the one
    /// exhaustive destructure in `metal_attn_view`.
    ///
    /// Both sit between the softmax and `wo`, which the fused kernels
    /// run with no host round-trip; a launch that ignored either would
    /// answer differently from the host bodies for the same weights.
    /// The gate is `afmoe` / `laguna` (`crate::attn_gate`); the sinks
    /// used to be refused by the gpt-oss NAME, and this pins that they
    /// are refused by the TENSOR now, on a model that is not gpt-oss.
    /// Only reachable in a `--features metal` build.
    #[test]
    fn an_output_gate_or_attention_sinks_keep_the_layer_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let mut d = Decoder::new_random_small(plain.clone(), 2, 32);
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );
        assert!(d.metal_attn_view(&d.layers[1]).is_some());

        // A gate on layer 0 only.
        let (n_heads, hidden) = (plain.n_heads, plain.hidden_dim);
        d.layers[0].attn.output_gate = Some(crate::attn_gate::AttnGate {
            proj: WeightMatrix::F32(Tensor::new(
                vec![0.1; n_heads * hidden],
                vec![n_heads, hidden],
            )),
            act: crate::attn_gate::GateAct::Sigmoid,
            width: crate::attn_gate::GateWidth::PerHead,
        });
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a gate no Metal kernel applies must refuse the fused attention"
        );
        assert!(d.metal_attn_view(&d.layers[0]).is_none());
        assert!(
            d.layer_supports_metal_attn(&d.layers[1]),
            "the fence is per layer: the ungated layer is still served"
        );

        // Sinks on layer 1, with `gpt_oss` still `None`.
        d.layers[1].attn.sinks = Some(vec![0.5; n_heads]);
        assert!(d.gpt_oss.is_none());
        assert!(
            !d.layer_supports_metal_attn(&d.layers[1]),
            "sinks are refused by the tensor, not by the gpt-oss name"
        );
        assert!(d.metal_attn_view(&d.layers[1]).is_none());
    }

    /// A LoRA adapter keeps the WHOLE model off every fused Metal
    /// launch, through the shared predicate, and an adapted matrix has
    /// no raw-bytes Metal descriptor at all -- so a stack that somehow
    /// asked for one anyway could not build it. The delta lives inside
    /// `WeightMatrix::Adapted` and is served by that type's methods;
    /// the stacks read weight bytes past those methods.
    /// Only reachable in a `--features metal` build.
    #[test]
    fn a_lora_adapter_keeps_the_model_off_every_fused_metal_path() {
        let mut plain = phi_like_config();
        plain.rope_dim = None;
        plain.rope_attn_factor = 1.0;
        let mut d = Decoder::new_random_small(plain.clone(), 2, 32);
        assert!(Decoder::metal_can_serve_model(&plain, false));
        assert!(
            d.layer_supports_metal_attn(&d.layers[0]),
            "the fixture must be Metal-eligible to start, or this proves nothing"
        );
        assert!(Decoder::metal_matvec_launch(&d.layers[0].attn.q_proj).is_some());

        // The fact the predicate reads.
        assert!(!Decoder::metal_can_serve_model(&plain, true));
        assert!(!Decoder::metal_prefill_dense_layer_eligible(
            &d.layers[0],
            &plain,
            true
        ));

        // The variant, on one matrix: no descriptor, so no launch.
        let (rows, cols) = (
            d.layers[0].attn.q_proj.rows(),
            d.layers[0].attn.q_proj.cols(),
        );
        let scale = frink_core::weight_matrix::LoraScale::new(1.0);
        d.layers[0].attn.q_proj.attach_lora(
            frink_core::weight_matrix::LoraDelta::new(
                vec![0.01; 2 * cols],
                vec![0.01; rows * 2],
                2,
                rows,
                cols,
                0.0,
                scale.clone(),
            )
            .unwrap(),
        );
        assert!(Decoder::metal_matvec_launch(&d.layers[0].attn.q_proj).is_none());
        assert!(d.layers[0].attn.q_proj.mul_mm_sg_launch().is_none());
        assert!(
            !d.layer_supports_metal_attn(&d.layers[0]),
            "a layer whose projection has no Metal descriptor is not served"
        );

        // The list, on the decoder: what every eligibility check asks.
        d.lora_adapters.push(crate::lora_attach::LoraAttached {
            path: "x.gguf".into(),
            alpha: 0.0,
            task_name: String::new(),
            prompt_prefix: String::new(),
            scale,
            n_tensors: 1,
        });
        assert!(d.lora_attached());
        assert!(
            !d.layer_supports_metal_attn(&d.layers[1]),
            "the fence is the whole model, not the adapted layer alone"
        );
    }

    /// A Gemma-3-4B-shaped config: `rope_scaling {linear, factor 8}`
    /// folded into the full-attention layers' divisors, nothing on the
    /// sliding ones, `sliding_window_pattern = 6` last-dense.
    fn gemma3_4b_shaped_config() -> ModelConfig {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.head_dim = 8;
        cfg.rope_layout = crate::config::RopeLayout::Norm;
        cfg.rope_theta = 1_000_000.0;
        cfg.rope_theta_swa = Some(10_000.0);
        cfg.sliding_window = Some(4);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
        cfg.rope_freqs = Some(crate::config::RopeFreqs {
            full: vec![8.0; 4],
            swa: Some(vec![1.0; 4]),
        });
        // One full period, so the run holds five sliding layers and one
        // full-attention layer -- Gemma-3's ratio, and the smallest one
        // that makes `rope_freqs_vary_by_layer` true.
        cfg.n_layers = 6;
        cfg
    }

    /// What the fused Metal stacks are handed per layer must be BOTH
    /// halves of `ModelConfig::layer_rope`, layer by layer.
    ///
    /// `Decoder::metal_stack_needs_per_layer_rope_freqs` used to refuse
    /// exactly this config off the fused prefill/decode stacks, because
    /// those took one `freq_factors` slice for a whole run beside a
    /// per-layer theta -- half the answer varying and half not, which is
    /// this repo's dominant bug shape. `LayerRope` carries the pair, and
    /// this pins that the decoder fills it from the pair rather than
    /// re-deriving either half on its own.
    #[test]
    fn the_metal_stacks_are_handed_each_layer_s_own_rope_pair() {
        let cfg = gemma3_4b_shaped_config();
        assert!(
            cfg.rope_freqs_vary_by_layer(),
            "fixture must be the shape that used to be refused"
        );
        let decoder = Decoder::new_random_small(cfg, 6, 32);

        for il in 0..decoder.layers.len() {
            let rope = decoder
                .config
                .layer_rope(il)
                .expect("every layer rotates here");
            let sent = decoder
                .metal_layer_rope(il)
                .expect("so every layer is handed a rope");
            assert_eq!(sent.theta, rope.theta, "layer {il} base");
            assert_eq!(sent.freq_factors, rope.freq_factors, "layer {il} divisors");
        }

        // Not vacuous: with `swa_pattern = 6` last-dense, layers 0..=4
        // slide and layer 5 does not, so the run really does hold two
        // different answers.
        let sliding = decoder.metal_layer_rope(0).unwrap();
        let full = decoder.metal_layer_rope(5).unwrap();
        assert_eq!(sliding.freq_factors, Some(&[1.0f32; 4][..]));
        assert_eq!(full.freq_factors, Some(&[8.0f32; 4][..]));
        assert_ne!(
            sliding, full,
            "a run of layers that all rope alike proves nothing here"
        );
    }

    /// The THIRD half of the answer: a layer llama.cpp does not rotate
    /// reaches the Metal stacks as `None`, from the same accessor the
    /// CPU bodies read, and `layer_needs_metal_stack` sees it through
    /// the same `Option` -- so the per-layer launches, which always
    /// rope, can never be handed such a layer.
    #[test]
    fn an_unrotated_layer_is_handed_no_rope_and_routed_to_the_stack() {
        let mut cfg = gemma3_4b_shaped_config();
        // Gemma-3's own graph rotates everything; borrow its shape and
        // give it EXAONE-4 32B's rule so the full-attention layer 5 is
        // the one that does not rotate.
        cfg.rope_layers = crate::rope_layers::RopeLayers::SlidingOnly;
        let decoder = Decoder::new_random_small(cfg, 6, 32);

        for il in 0..5 {
            assert!(
                decoder.metal_layer_rope(il).is_some(),
                "sliding layer {il} rotates"
            );
        }
        assert_eq!(
            decoder.metal_layer_rope(5),
            None,
            "the full-attention layer does not"
        );
        assert!(
            decoder.layer_needs_metal_stack(&decoder.layers[5], 5),
            "an unrotated layer must go to the fused stack, which implements `None`"
        );
    }
}
