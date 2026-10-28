//! The weights one decoder layer holds: attention, the FFN / MoE block,
//! the per-layer bundle, and gpt-oss's side table.

use std::sync::atomic::{AtomicU64, Ordering};

use frink_core::weight_matrix::WeightMatrix;
use frink_moe::{ExpertWeights, PlacementPlan};

use crate::norm::NormOp;

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
    pub(super) gate: frink_core::weight_matrix::WeightBytes,
    pub(super) up: frink_core::weight_matrix::WeightBytes,
    pub(super) down: frink_core::weight_matrix::WeightBytes,
    pub(super) gate_stride: usize,
    pub(super) up_stride: usize,
    pub(super) down_stride: usize,
    pub(super) n_experts: usize,
    pub(super) ffn_rows: usize,
    pub(super) hidden_rows: usize,
    pub(super) gate_row_bytes: usize,
    pub(super) down_row_bytes: usize,
    pub(super) gate_kind: &'static str,
    pub(super) up_kind: &'static str,
    pub(super) down_kind: &'static str,
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
        self.with_experts(&[e], |ex| f(ex[0]))
    }

    /// Runs `f` with ALL of `eids` lent at once, `experts[i]` being
    /// expert `eids[i]`, whichever backing holds them.
    ///
    /// This is what lets a host MoE body be ONE body for both backings.
    /// Before it, the top-k kernels took `&[ExpertWeights]`, which only
    /// a resident layer has, so a streamed layer was sent to a different
    /// body with a different gate/up kernel (`apply_cpu_q8`'s repacked
    /// GEMV where resident ran `dot_pair_cpu_q8`'s 2-row SDOT) and
    /// answered differently in the last bits. The backing should decide
    /// where the bytes come from and nothing else.
    ///
    /// A stored layer asks for the whole batch before touching any of
    /// it, so a miss on the last expert cannot evict the first; each
    /// materialised view clones its lease's `Arc` (`WeightBytes::Shared`)
    /// and so pins its entry until `f` returns.
    pub fn with_experts<R>(&self, eids: &[usize], f: impl FnOnce(&[&ExpertWeights]) -> R) -> R {
        match &self.experts {
            ExpertBacking::Resident(v) => {
                let refs: Vec<&ExpertWeights> = eids.iter().map(|&e| &v[e]).collect();
                f(&refs)
            }
            ExpertBacking::Stored {
                store,
                layouts,
                layer,
            } => {
                let key = |e: usize| frink_core::expert_store::ExpertKey {
                    layer: *layer,
                    expert: e as u32,
                };
                if eids.len() > 1 {
                    let keys: Vec<_> = eids.iter().map(|&e| key(e)).collect();
                    store.prefetch(&keys);
                }
                let held: Vec<ExpertWeights> = eids
                    .iter()
                    .map(|&e| {
                        let lease = store.acquire(key(e)).unwrap_or_else(|err| {
                            panic!(
                                "expert store read failed for layer {layer} expert {e}: {err} \
                                 (checkpoint file unreadable mid-decode)"
                            )
                        });
                        layouts[e].materialize(&lease)
                    })
                    .collect();
                let refs: Vec<&ExpertWeights> = held.iter().collect();
                f(&refs)
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

    pub(super) fn record_activations(&self, expert_ids: &[usize]) {
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
