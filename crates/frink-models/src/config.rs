//! Architecture configs. Prefer GGUF / config.json over preset defaults.
//! Unconfirmed preset fields must be listed in `best_effort_fields`.
//! What actually runs: `docs/MODELS.md`.

use frink_moe::{GatingFunction, MoeLayerConfig};

/// Model-level (not per-layer) tensors `ModelConfig::from_gguf` reads.
///
/// The config is parsed from its own file handle, so these lookups are
/// invisible to the handle the weight loader tracks consumption on.
/// `loader::assert_every_tensor_consumed` replays them; anything added
/// here must actually be *used*, not merely read, or the gate stops
/// meaning what it says.
pub const MODEL_LEVEL_TENSORS_READ_BY_CONFIG: &[&str] = &[
    "rope_freqs.weight",
    "rope_factors_long.weight",
    "rope_factors_short.weight",
];

/// Which attention mechanism a model uses. `Gqa` (grouped-query
/// attention + RoPE, uniform across every layer) is the only variant
/// `frink-core`/`frink-models::decoder` actually implement today --
/// it's what every preset runs through, including the two whose real
/// published attention differs (DeepSeek V4 Pro's CSA/HCA, Kimi K3's
/// hybrid KDA/Gated-MLA). `KimiHybrid` exists so Kimi K3's real,
/// cited attention hyperparameters are captured accurately rather than
/// silently discarded, even though `Decoder` itself still runs the
/// GQA path for every layer (the dedicated Kimi decoder is the one
/// consumer of the hybrid variant today).
#[derive(Debug, Clone)]
pub enum AttentionKind {
    Gqa,
    KimiHybrid(KimiHybridAttention),
}

/// Which concrete attention mechanism a single 0-indexed layer uses --
/// the resolved answer `Decoder` needs per layer once it dispatches on
/// `AttentionKind` instead of always running GQA (see
/// `ModelConfig::layer_attention_kind`; only the dedicated Kimi
/// decoder actually dispatches on it today).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerAttentionKind {
    Gqa,
    /// KDA (Kimi Delta Attention) -- see `frink_models::kda`.
    KimiKda,
    /// Gated MLA -- see `frink_models::mla`.
    KimiMla,
}

/// Kimi K3's real attention topology, transcribed from the published
/// `huggingface.co/moonshotai/Kimi-K3/config.json`'s `linear_attn_config`
/// block. `kda_layers`/`full_attn_layers` are kept exactly as published
/// -- **1-indexed** (layer 1 is the model's first transformer layer),
/// not `frink`'s usual 0-indexed `layers` slice -- so a caller wiring
/// this into `Decoder` must subtract 1 before indexing.
#[derive(Debug, Clone)]
pub struct KimiHybridAttention {
    /// 1-indexed layers using KDA (Kimi Delta Attention: gated
    /// linear/recurrent attention with a short causal conv). 69 of 93
    /// layers.
    pub kda_layers: Vec<usize>,
    /// 1-indexed layers using Gated MLA (DeepSeek-style multi-head
    /// latent attention with an output gate). 24 of 93 layers.
    pub full_attn_layers: Vec<usize>,
    pub mla: MlaConfig,
    pub kda: KdaConfig,
}

/// Gated MLA (multi-head latent attention) hyperparameters, verified
/// against Kimi K3's real `config.json` `text_config` block and the
/// real `KimiMLAAttention` reference implementation
/// (`modeling_kimi_linear.py`).
#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Kimi K3's addition on top of standard DeepSeek-style MLA:
    /// `attn_output *= sigmoid(g_proj(hidden_states))` before `o_proj`.
    pub use_output_gate: bool,
    /// `None` reproduces Kimi K3's real, confirmed behavior: no rotary
    /// embedding at all (`mla.rs`'s module doc comment; the real
    /// `KimiMLAAttention.forward` asserts `use_nope` and never calls a
    /// rotary function). `Some` is for architectures whose decoupled
    /// `q_rot`/`k_rot` slices genuinely are position-rotated -- e.g.
    /// GLM-5.2, whose real `config.json` (`zai-org/GLM-5.2`) sets
    /// `rope_interleave: true` for its main attention (confirmed
    /// against llama.cpp PR #25407's `LLAMA_ROPE_TYPE_NORM` rope call
    /// on `q_pe`/`k_pe` in `src/models/glm-dsa.cpp`).
    pub rope: Option<MlaRopeConfig>,
}

/// RoPE parameters for the decoupled `q_rot`/`k_rot` slices of an MLA
/// attention layer that does apply rotation (unlike Kimi K3 -- see
/// `MlaConfig::rope`'s doc comment). Always the interleaved convention
/// (`frink_core::attention::apply_rope_interleaved`) for every real
/// architecture confirmed so far to use this (GLM-5.2's
/// `rope_interleave: true`); a separate split-half variant isn't wired
/// in here since no confirmed real user of it exists yet.
#[derive(Debug, Clone, Copy)]
pub struct MlaRopeConfig {
    pub theta: f32,
}

/// KDA (Kimi Delta Attention) hyperparameters, verified against Kimi
/// K3's real `config.json` `linear_attn_config` block and the real
/// gated delta-rule reference implementation in
/// `fla-org/flash-linear-attention`'s `fla/ops/kda/naive.py` (the
/// exact recurrence: decay state by `exp(g)`, then add a rank-1
/// `beta * k ⊗ (v - kᵀS)` correction, then read `o = qᵀS`) and
/// `fla/ops/kda/gate.py` (the lower-bounded gate:
/// `g = gate_lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`,
/// and `beta = sigmoid(raw_beta)`).
#[derive(Debug, Clone)]
pub struct KdaConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub short_conv_kernel_size: usize,
    pub gate_lower_bound: f32,
    pub use_full_rank_gate: bool,
}

/// Which RoPE pairing convention a model uses. Confirmed against
/// llama.cpp's `llama_model_rope_type` (`src/llama-model.cpp`):
/// `Norm` is adjacent-pair / GPT-J (`LLAMA_ROPE_TYPE_NORM`); `Neox` is
/// split-half / GPT-NeoX (`LLAMA_ROPE_TYPE_NEOX`). Getting this wrong
/// silently produces fluent-but-wrong logits (the real Llama-3.1-8B
/// early-stop bug: frink applied NeoX to a Norm architecture).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Adjacent pairs `(2*i, 2*i+1)` -- llama.cpp `LLAMA_ROPE_TYPE_NORM`.
    /// Used by `llama` (including Llama 3/3.1/3.2), `deepseek2`,
    /// `mistral3`, and related families. (`llama4` is DedicatedOnly — MoE
    /// graph — but its RoPE type in the inventory is still Norm.)
    Norm,
    /// Split-half pairs `(i, i+half)` -- llama.cpp `LLAMA_ROPE_TYPE_NEOX`.
    /// Used by `olmoe`, `qwen2`/`qwen2moe`/`qwen3`, `phi3`, `gemma*`, and
    /// related families. Frink's historical default before architecture-
    /// aware dispatch existed.
    Neox,
}

impl RopeLayout {
    /// Maps a GGUF `general.architecture` string onto the RoPE pairing
    /// llama.cpp selects for that family. Prefer
    /// [`crate::capability::resolve_architecture`] for load-time
    /// decisions — unknown architectures must fail closed there rather
    /// than guessing. This helper remains for tests and call sites that
    /// already know the arch is registered; unknowns still return `Neox`
    /// only as a last-resort historical default.
    pub fn for_gguf_architecture(arch: &str) -> Self {
        match crate::capability::resolve_profile(arch) {
            Some(p) => p.rope,
            // Unknown: do not invent Norm for a Qwen/Phi/Gemma-shaped
            // string that happened to miss the registry.
            None => RopeLayout::Neox,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: &'static str,
    /// Decoder layers: llama.cpp's `n_layer()`, which is the file's
    /// `block_count` MINUS [`Self::n_mtp_blocks`].
    pub n_layers: usize,
    /// NextN / MTP blocks the file appends after the trunk, inside its
    /// `block_count`, which llama.cpp creates `TENSOR_SKIP` and never
    /// runs (`crate::mtp_blocks`). Their tensors are `blk.N.*` for
    /// `n_layers <= N < n_layers + n_mtp_blocks`; the loader marks them
    /// deliberately unread. Zero for every architecture whose graph
    /// does not read `nextn_predict_layers`.
    pub n_mtp_blocks: usize,
    pub hidden_dim: usize,
    /// Query heads of the WIDEST layer. Every layer's for a uniform
    /// model, which is every model but the per-layer-shape ones
    /// (`crate::layer_shapes`); a layer body must read its own count
    /// through [`Self::layer_shape`], never this field.
    pub n_heads: usize,
    /// KV heads of the WIDEST layer, so that a budget priced from it
    /// over-counts rather than under-counts a heterogeneous model.
    /// Same rule as `n_heads`: per-layer computation reads
    /// [`Self::layer_shape`]; caches come from [`Self::new_kv_caches`].
    pub n_kv_heads: usize,
    /// The K head width (`attention.key_length`, llama.cpp
    /// `n_embd_head_k`): the width of every Q and K head, the width RoPE
    /// rotates within, and the `1/sqrt` of the attention scale.
    pub head_dim: usize,
    /// The V head width (`attention.value_length`, `n_embd_head_v`)
    /// WHEN IT DIFFERS from [`Self::head_dim`]; `None` means V heads are
    /// K's width, which is every architecture but MiMo-V2 (`head_dim:
    /// 192, v_head_dim: 128`). Read it through [`Self::v_head_dim`],
    /// never here: an `Option` rather than a second `usize` so that a
    /// config whose `head_dim` is set or changed cannot leave a stale V
    /// width beside it -- the two-fields-that-must-agree shape. See
    /// [`crate::kv_head_dims`] for which architectures may declare them
    /// apart and which fused paths refuse when they are.
    pub v_head_dim: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// The epsilon the POST-attention and POST-FFN norms run at.
    ///
    /// Equal to [`Self::rms_norm_eps`] for every architecture but the
    /// one whose graph writes a literal (`crate::norm::
    /// POST_NORM_EPS_LITERAL`, `muse-glimmer.cpp:63`). Set by the
    /// loader from that table so the two cannot be given different
    /// answers by two callers, and read through
    /// [`Self::post_norm_eps`].
    pub post_norm_eps: f32,
    /// The norm FUNCTION every weighted site applies
    /// (`crate::norm::norm_function`): the architecture's, or, for
    /// `crate::norm::NORM_BY_RMS_EPS_KEY`, the file's.
    pub norm_function: crate::norm::NormFunction,
    pub moe: MoeLayerConfig,
    /// `Gqa` for every preset except Kimi K3. `Decoder`'s forward pass
    /// does not yet branch on this -- see `AttentionKind`'s doc
    /// comment.
    pub attention: AttentionKind,
    /// Mistral/Mixtral/Qwen2-family sliding-window attention: when
    /// set, every layer attends only to the most recent `N` cached
    /// positions instead of the full causal history (see
    /// `frink_core::attention::causal_gqa_attention_windowed`'s doc
    /// comment for the real source citations). `None` for every
    /// architecture that doesn't use this (most models, including
    /// Qwen1.5/Qwen2-MoE's real published config, which sets
    /// `use_sliding_window: false` despite carrying a `sliding_window`
    /// value -- so this field being `None`/`Some` must come from that
    /// enable flag, not just the window-size field's presence).
    pub sliding_window: Option<usize>,
    /// How many of the model's *first* layers use an ordinary dense
    /// FFN (no expert routing at all) rather than the model's MoE
    /// topology. Found by reading ik_llama.cpp's real GGUF
    /// hparams-loading source (`LLM_KV_LEADING_DENSE_BLOCK_COUNT`):
    /// DeepSeek-2/3-family models don't apply MoE uniformly to every
    /// layer -- the first few layers are always dense. Zero means
    /// "every layer uses this model's MoE topology," the default for
    /// architectures that don't do this.
    pub n_dense_leading_layers: usize,
    /// `{arch}.interleave_moe_layer_step` where the loader honours it
    /// (`crate::moe_interleave::INTERLEAVE_STEP_HONOURED_BY_LOADER`):
    /// layer `il` is MoE when `(il + 1) % step == 0`. `None` everywhere
    /// else, including the ERNIE files whose step is 1.
    pub moe_interleave_step: Option<usize>,
    /// Llama 3/3.1/3.2's real per-band RoPE frequency correction (the
    /// `rope_freqs.weight` GGUF tensor, `head_dim/2` elements,
    /// `TENSOR_NOT_REQUIRED` so most architectures leave this `None`).
    /// See `frink_core::attention::apply_rope_with_freq_factors`'s doc
    /// comment for the real source and the real bug this closes: without
    /// it, every RoPE angle for a Llama-3-family checkpoint is computed
    /// slightly wrong, an error that compounds with position and
    /// eventually produces wrong logits (a spurious early EOS was the
    /// observed real symptom).
    ///
    /// Not only a tensor: this is the *resolved* per-band divisor array,
    /// so a checkpoint declaring `rope.scaling.type = "yarn"` gets its
    /// YaRN frequency rewrite folded in here too (see
    /// `frink_core::attention::yarn_freq_factors`, and
    /// `loader::yarn_scaling_from_gguf` for what the file has to declare
    /// before that happens). A file carrying both a tensor and a YaRN
    /// declaration composes them by multiplication, as llama.cpp does
    /// (`ggml_rope_cache_init` divides by `freq_factors` and *then*
    /// runs `rope_yarn`). Consumers must therefore treat this as "the
    /// correction to apply", not as "the tensor this file shipped".
    ///
    /// Per-LAYER, because llama.cpp's is: see [`RopeFreqs`]. Read it
    /// through [`Self::layer_rope`], never field-by-field.
    pub rope_freqs: Option<RopeFreqs>,
    /// LongRoPE's two candidate factor sets, kept so the choice between
    /// them can be made when the *run's* context size is known rather
    /// than at parse time. llama.cpp picks per request
    /// (`llama_model::get_rope_factors` reads `cparams.n_ctx_seq`), and
    /// the two sets are not interchangeable: Phi-4-mini's short set is
    /// all ones (no correction at all) while its long set reaches 47.
    /// Choosing from the checkpoint's advertised 131072 when the user
    /// runs at 4096 is a different model.
    pub rope_freqs_long: Option<Vec<f32>>,
    pub rope_freqs_short: Option<Vec<f32>>,
    /// `<arch>.rope.scaling.original_context_length` — the threshold the
    /// choice above is made against.
    pub rope_orig_ctx: Option<usize>,
    /// Rotary width when it is narrower than `head_dim`
    /// (`<arch>.rope.dimension_count`, llama.cpp `hparams.n_rot`).
    /// `None` means the whole head rotates, which is the common case.
    /// Phi-3/Phi-4 rotate 96 of 128.
    ///
    /// This is the FULL-attention layers' width, llama.cpp's
    /// `n_rot_full`; [`Self::rope_dim_swa`] is the sliding layers'.
    pub rope_dim: Option<usize>,
    /// The SLIDING layers' rotary width when it differs from
    /// [`Self::rope_dim`] -- llama.cpp's `n_rot_swa`, read from
    /// `rope.dimension_count_swa` or halved-from-full for `step35`
    /// (`crate::swa_geometry`), consumed through `n_rot(il)`
    /// (`llama-hparams.cpp:85-91`). `None` means the sliding layers
    /// rotate the same width as the full ones, which is every
    /// architecture but the ones the table names. `Some(head_dim)` is
    /// the whole head, and [`Self::layer_rope`] normalises it.
    pub rope_dim_swa: Option<usize>,
    /// LongRoPE/YaRN magnitude scaling (`<arch>.rope.scaling.attn_factor`,
    /// llama.cpp `hparams.rope_attn_factor` folded into
    /// `cparams.yarn_attn_factor` at `llama-context.cpp:231`, then applied
    /// as ggml `rope_yarn`'s `mscale`, which multiplies *both* `cos` and
    /// `sin` — so it scales the RoPE'd vector, at every position, whether
    /// or not any frequency correction is active.
    ///
    /// Phi-4-mini ships `1.1902381`. Ignoring it does not merely change
    /// long-context behaviour: q and k are both scaled, so every attention
    /// logit is off by `attn_factor²` and the softmax is sharper than the
    /// model's. Measured symptom: frink and llama.cpp diverge from the
    /// eighth token of a greedy completion on the same GGUF.
    ///
    /// `1.0` for every architecture that does not set the key.
    pub rope_attn_factor: f32,
    /// RoPE pairing convention for this architecture -- see
    /// `RopeLayout`. Independently of `rope_freqs`: a Llama checkpoint
    /// needs both `Norm` pairing *and* the per-band frequency factors.
    pub rope_layout: RopeLayout,
    /// How Q/K RMSNorm weights are applied when present (see
    /// [`crate::capability::QkNormStyle`]).
    pub qk_norm_style: crate::capability::QkNormStyle,
    /// WHICH LAYERS SLIDE -- llama.cpp's `is_swa_impl[il]`, as a
    /// period with a phase, the file's own per-layer array, or every
    /// layer. See [`crate::swa_layers`]. Meaningless without
    /// [`Self::sliding_window`]; [`Self::layer_sliding_window`] is the
    /// one accessor that combines the two.
    ///
    /// Getting the phase wrong is not a near miss: on a 32-layer
    /// period-4 model the two phases disagree about SIXTEEN layers,
    /// each of which then attends over the wrong span at full speed.
    /// `capability::default_swa_layout` carries the per-arch value,
    /// transcribed from llama.cpp.
    pub swa_layers: crate::swa_layers::SwaLayers,
    /// WHICH LAYERS ROTATE -- llama.cpp's per-layer `use_rope`.
    ///
    /// [`crate::rope_layers::RopeLayers::All`] for every architecture
    /// that writes no gate, which is 134 of llama.cpp's 140. The rule
    /// and the table that assigns it live in [`crate::rope_layers`];
    /// nothing else in this crate may branch on an architecture name to
    /// decide it, and [`Self::layer_rope`] returning `None` is the only
    /// way a call site learns of it.
    pub rope_layers: crate::rope_layers::RopeLayers,
    /// WHICH LAYERS HAVE WHICH SHAPE -- llama.cpp's `n_head(il)`,
    /// `n_head_kv(il)` and `n_ff(il)`.
    ///
    /// `Uniform` for every architecture whose graph reads layer 0, which
    /// is all but the rows in `layer_shapes::PER_LAYER_SHAPE_ARCHS`.
    /// [`Self::layer_shape`] is the one accessor; the fused Metal
    /// launches and the CUDA resident KV are fenced off any model that
    /// is not `Uniform`, because each holds one geometry.
    pub layer_shapes: crate::layer_shapes::LayerShapes,
    /// Attention logit soft-capping (Gemma 2+). Applied as
    /// `softcap * tanh(score / softcap)` before softmax.
    pub attn_logit_softcap: Option<f32>,
    /// Final logit soft-capping (Gemma 2+). Applied to lm_head output.
    pub final_logit_softcap: Option<f32>,
    /// Input embedding scale (Gemma: `sqrt(hidden_dim)`; Granite:
    /// `{arch}.embedding_scale`).
    pub embedding_scale: Option<f32>,
    /// Multiplier applied to EVERY branch output -- attention and FFN
    /// alike -- immediately before it rejoins the residual stream
    /// (Granite `residual_multiplier`, `src/models/granite.cpp:235-238`
    /// and `:288-292`).
    ///
    /// `None` means the plain `hidden += branch` every other
    /// architecture computes. The decoder never applies this field
    /// itself: [`crate::scalar_multipliers::residual_add`] is the one
    /// residual add, and it takes this value as a parameter, because
    /// `decoder.rs` spells the add out eighteen times and eighteen
    /// hand-written copies that must agree about one scalar is the
    /// defect shape this repo keeps paying for.
    pub residual_scale: Option<f32>,
    /// `Some(s)`: each sublayer's PRE-NORM OUTPUT, times `s`, REPLACES
    /// the residual stream its branch joins, and the layer input is
    /// discarded (`crate::normed_residual`; `minimax-01.cpp:249,428`).
    ///
    /// Never `Some` together with [`Self::residual_scale`] -- one
    /// column of `MultiplierSupport` resolves both -- and `Some(1.0)`
    /// is a real value here, because the field carries the topology as
    /// well as the multiplier.
    pub normed_residual_scale: Option<f32>,
    /// Multiplier applied to the lm_head's output, after the projection
    /// and before [`Self::final_logit_softcap`].
    ///
    /// Already resolved into a MULTIPLIER at load time, whichever
    /// direction the architecture's graph states it in: Granite divides
    /// by `{arch}.logit_scale` (`granite.cpp:180`), so this field holds
    /// `1.0 / logit_scale`. Keeping the direction in
    /// [`crate::scalar_multipliers`] rather than here is what lets the
    /// decoder have exactly one multiply, and stops a second
    /// architecture with the opposite convention from needing a second
    /// field.
    ///
    /// Guaranteed positive when `Some`, and that is load-bearing rather
    /// than incidental: a Metal decode stack may fold the lm_head and
    /// return an argmax token id, which is only sound while every
    /// post-head transform is monotone increasing.
    pub logit_multiplier: Option<f32>,
    /// Optional override for the attention score scale baked into Q
    /// *instead of* the kernel's default `1/sqrt(head_dim)`. When set,
    /// callers must pass `score_scale = 1.0` into the attention kernel
    /// (llama.cpp Gemma: scale Q then `build_attn(..., 1.0f)`). Prefer
    /// leaving this `None` when the override equals `1/sqrt(head_dim)`.
    pub attention_scale: Option<f32>,
    /// Symmetric clamp on the Q, K and V projections
    /// (`{arch}.attention.clamp_kqv`), applied after the QKV bias and
    /// before the QK-norm and RoPE -- llama.cpp's `build_qkv`
    /// (`llama-graph.cpp:1611-1652`).
    ///
    /// `Some(c)` only when the architecture's graph clamps AND the file
    /// declares a positive value; llama.cpp's own test is `> 0.0f`, so
    /// zero and a negative value are "no clamp" and resolve to `None`
    /// here rather than to a clamp that zeroes every projection. The
    /// resolution lives in [`crate::clamp_kqv`]; the decoder applies it
    /// through ONE helper shared by every host body, and the fused
    /// Metal launches are fenced off by `Decoder::metal_can_serve_model`
    /// because no kernel implements it.
    pub clamp_kqv: Option<f32>,
    /// Per-position attention temperature -- llama.cpp's
    /// `llm_graph_input_attn_temp`, the `[n_tokens]` vector
    /// `log(floor((pos + offset) / floor_scale) + 1) * scale + 1` that
    /// `mistral3.cpp:153-156` multiplies into Q after RoPE, before
    /// `build_attn`, with `kq_scale` untouched. See
    /// [`crate::attn_temperature`] for the census (three graphs of 155)
    /// and the resolution.
    ///
    /// `Some` only for an architecture whose graph builds the input AND
    /// a file declaring a nonzero `attention.temperature_scale`; the
    /// key on any other architecture is dead metadata upstream and is
    /// ignored here the same way. Applied through ONE helper,
    /// `Decoder::apply_attn_temperature`, on every host body, and
    /// fenced off the fused Metal launches by
    /// `Decoder::metal_can_serve_model`, because none has a per-token Q
    /// scale uniform.
    pub attn_temperature: Option<crate::attn_temperature::AttnTemperature>,
    /// WHICH TENSOR THE MoE ROUTER READS -- the normed FFN input for
    /// every graph but one, the raw layer input for `smallthinker`
    /// (`smallthinker.cpp:111`). See [`crate::router_input`] for the
    /// census (four graphs of 155 pass a precomputed `probs_in`, one on
    /// the generic path) and the seam. `Decoder::router_operand` is the
    /// ONE place the operand is captured, and the GPU router paths
    /// refuse a model whose operand they cannot read
    /// (`Decoder::gpu_router_matches_host_routing`).
    pub router_input: crate::router_input::RouterInput,
    /// Whether this model's blocks norm INSIDE the two sublayers:
    /// BitNet's `attn_sub_norm` (on the attention output, BEFORE `wo`)
    /// and `ffn_sub_norm` (on `silu(gate) * up`, BEFORE `down`),
    /// `bitnet.cpp:24,36,101-106,135-140`. See [`crate::sub_norms`] for
    /// the census (one graph of 155) and the two readers: the loader,
    /// which REQUIRES the pair when this is set, and
    /// `Decoder::metal_can_serve_model`, which refuses every fused
    /// launch, since none has a norm at either site.
    pub block_sub_norms: bool,
    /// Whether any layer of this model is a PARALLEL residual,
    /// `x + attn(norm(x)) + ffn(norm(x))` (`crate::parallel_residual`;
    /// `gptneox` under its key, `plamo` always, `stablelm` per layer by
    /// tensor presence). The per-layer fact is `MoeWeights::parallel`;
    /// this is the model-level one `Decoder::metal_can_serve_model`
    /// reads, because every fused Metal launch bakes the pre-FFN norm
    /// over the post-attention residual into its kernel.
    pub parallel_residual: bool,
    /// Whether this model adds a learned position table to its token
    /// embeddings (`crate::position_embd`; `gpt2`, `starcoder`). The
    /// table itself is `Decoder::position_embd`; this is the model-level
    /// fact `Decoder::metal_can_serve_model` reads, because the GPU
    /// embedding gather has no add and the fused stacks never see `pos`.
    pub learned_positions: bool,
    /// `{arch}.attention.value_scale`: MiMo-V2 multiplies the attention
    /// branch by it AFTER `wo` (`mimo2.cpp:180-183`; every real export
    /// carries `0.707`). `None` for no scale; see
    /// [`crate::attn_value_scale`] for the one reader and the values
    /// that mean none. Applied in `Decoder::attn_out_to_residual_rows`;
    /// the fused Metal launches refuse a model that has one.
    pub attn_value_scale: Option<f32>,
    /// llama.cpp's `f_max_alibi_bias` when it is positive: the model
    /// positions by ALiBi and rotates nothing (`crate::alibi` for which
    /// graphs and where each gets the number; `frink_core::alibi` for
    /// the per-head slopes, which `Decoder::alibi_slopes` holds). `None`
    /// for every other model. The fused Metal launches and the CUDA
    /// resident attention refuse a model that has one: their kernels
    /// add no per-key bias.
    pub alibi_max_bias: Option<f32>,
    /// Nanbeige's `num_loops`: `Some` when the model's logical layers
    /// are several passes over its physical ones (`nanbeige.cpp:19-31`).
    /// [`Self::n_layers`] is then the LOGICAL count, `Decoder::layers`
    /// stays physical, and `Decoder::layer_for` maps one to the other.
    /// See [`crate::layer_loops`]; the fused Metal launches refuse a
    /// looped model.
    pub layer_loops: Option<crate::layer_loops::LayerLoops>,
    /// Talkie's embedding skip stream (`talkie.cpp:50-52,123-126`): the
    /// embeddings are RMS-normed without a weight before layer 0 and
    /// every layer adds that vector, times its own
    /// `layer_output_scale`, after its FFN residual. See
    /// [`crate::skip_stream`]; the fused Metal launches refuse a model
    /// that has one.
    pub skip_stream: bool,
    /// Every attention layer ALSO runs a Mamba-2 block on the same
    /// normed input, the two outputs summed (`falcon-h1.cpp:137-161`;
    /// `crate::mamba2::PARALLEL_WITH_ATTENTION`). The layer's cache
    /// holds the attention rows AND the block's `RecurrentState`, so
    /// [`Self::has_recurrent_layers`] is true and the fused Metal
    /// launches refuse the model.
    pub parallel_ssm: bool,
    /// The sliding layers' window is a CHUNK (`crate::chunked_swa`): a
    /// query sees its own `sliding_window`-sized chunk and nothing
    /// before it. [`Self::layer_window_for_query`] is the per-query
    /// window the single-query kernels take for it; the fused Metal
    /// launches refuse the model.
    pub swa_chunked: bool,
    /// A per-head RMSNorm with no weight on Q and K after RoPE, on the
    /// layers that rotate (`crate::weightless_qk_norm`, Llama 4's
    /// `Llama4TextL2Norm`). Applied at the post-RoPE QK-norm hook;
    /// the fused Metal launches refuse the model.
    pub weightless_qk_norm: bool,
    /// RoPE base used on SWA layers (Gemma 3: defaults to `10000` when
    /// the GGUF omits `rope.freq_base_swa`; full-attn layers keep
    /// [`Self::rope_theta`]).
    pub rope_theta_swa: Option<f32>,
    /// Dense/MoE FFN activation pairing.
    pub ffn_activation: FfnActivation,
    /// Every field on this config that is a best-effort estimate rather
    /// than a confirmed value from an official config.json / GGUF file.
    pub best_effort_fields: &'static [&'static str],
}

/// One layer's RoPE, as [`ModelConfig::layer_rope`] hands it out: the
/// three things llama.cpp's `ggml_rope_ext` call takes per layer that
/// vary by layer.
///
/// A struct rather than a tuple so that a consumer names every field
/// it takes; the Metal side destructures it exhaustively and refuses a
/// `rot_dim` its one-uniform kernels cannot honour per layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerRopeParams<'a> {
    /// This layer's frequency base (`rope_theta`, or `rope_theta_swa`
    /// on a sliding layer).
    pub theta: f32,
    /// This layer's per-band divisors, `rot_dim/2` long, or `None` to
    /// divide by nothing.
    pub freq_factors: Option<&'a [f32]>,
    /// This layer's rotary width when narrower than `head_dim`; `None`
    /// rotates the whole head. llama.cpp's `n_rot(il)`.
    pub rot_dim: Option<usize>,
}

/// The resolved per-band RoPE divisors, for BOTH kinds of layer.
///
/// llama.cpp splits RoPE per layer in two places, not one:
///
/// ```cpp
/// // src/llama-model.cpp:2029-2035
/// float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
///     return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa  : cparams.rope_freq_base;
/// }
/// float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
///     return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
/// }
/// ```
///
/// and every alternating-SWA graph calls both, per layer
/// (`gemma3.cpp:112-121`, `gemma2.cpp:79-80`, `laguna.cpp:182-183`).
/// frink folds llama.cpp's `freq_scale` into these divisors -- linear
/// scaling by `s` is exactly "divide every band by `s`" -- so the SCALE
/// half of that split has to live here, beside the BASE half in
/// [`ModelConfig::rope_theta_swa`].
///
/// It did not, and Gemma-3 4B/12B/27B paid for it: their headers declare
/// `rope.scaling.type = linear, factor = 8`, `gemma3.cpp` never assigns
/// `rope_freq_scale_train_swa` so it keeps its `1.0f` default
/// (`src/llama-hparams.h:129`), and five layers in every six are sliding
/// (`sliding_window_pattern = 6`, last-dense). frink rotated all of
/// them at `p/8` where llama.cpp rotates at `p` -- fluent, and worse the
/// longer the prompt. Invisible to the audit because the fixture is
/// Gemma-3-1B, the one size with no `rope_scaling` at all.
///
/// The two fields are one struct so that answering the base question
/// without answering the scale question does not compile.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFreqs {
    /// What the FULL-ATTENTION layers divide each band's theta by.
    pub full: Vec<f32>,
    /// What the SLIDING layers divide by, when the architecture does not
    /// let them inherit the model's trained RoPE scale
    /// (`capability::swa_rope_scale_follows_model`). `None` means they
    /// inherit [`Self::full`], which is llama.cpp's behaviour for every
    /// architecture that assigns `rope_freq_scale_train_swa` from
    /// `rope_freq_scale_train`.
    ///
    /// "No divisors at all" is spelled as an all-ones vector rather than
    /// a third state: dividing by one is exactly not dividing, and one
    /// fewer state is one fewer thing two call sites can disagree about.
    pub swa: Option<Vec<f32>>,
}

impl RopeFreqs {
    /// The divisors layer `il` uses, given whether it slides.
    pub fn for_layer(&self, sliding: bool) -> &[f32] {
        match (sliding, &self.swa) {
            (true, Some(swa)) => swa,
            _ => &self.full,
        }
    }

    /// True when the sliding layers use a different set from the full
    /// ones, i.e. when one `freq_factors` buffer cannot serve a whole
    /// stack of layers.
    pub fn varies_by_layer(&self) -> bool {
        self.swa.as_ref().is_some_and(|swa| *swa != self.full)
    }
}

/// Dense / expert FFN non-linearity used by the generic decoder.
///
/// Not `Copy` and not `Eq` since [`FfnActivation::Xielu`]: that
/// variant CARRIES its per-layer parameters, so the kind and the
/// parameters cannot disagree, and a `ModelConfig` clone shares them
/// through an `Arc`. [`ModelConfig::layer_ffn_act`] is how a layer
/// body turns this into the `GluAct` it runs.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum FfnActivation {
    /// `silu(gate) * up` with separate gate/up matrices (Llama / Qwen).
    #[default]
    Swiglu,
    /// Phi-3 fused gate+up in one `ffn_up` matrix (`2 * n_ff` rows).
    SwigluFused,
    /// Gemma GeGLU: `gelu(gate) * up`.
    Gelu,
    /// UNGATED ReLU-squared: `down(relu(up(x))^2)`, two matrices in
    /// sequence and no `ffn_gate` at all -- llama.cpp's
    /// `LLM_FFN_RELU_SQR` under `LLM_FFN_SEQ` with a null gate
    /// (`arcee.cpp:39-40,123-128`; also `plm`, `nemotron`, `jais2`,
    /// `nemotron-h`, each of which needs more than this).
    ///
    /// The loader ALIASES the expert's `gate` to its `up` matrix (a
    /// zero-copy view of the same bytes) and this maps to
    /// `frink_moe::GluAct::ReluSqr`, which reads `up` alone. That is
    /// what lets every gated path serve it unchanged; the dense hot
    /// paths skip the aliased matmul through `GluAct::ungated`, and no
    /// fused device kernel spells it, so `fused_kernel_gelu_flag` is
    /// `None`.
    ReluSqr,
    /// UNGATED GELU: `down(gelu(up(x)))`, two matrices in sequence and
    /// no `ffn_gate` -- llama.cpp's `LLM_FFN_GELU` under `LLM_FFN_SEQ`
    /// with a null gate (`starcoder2.cpp:125-131`, `codeshell.cpp:
    /// 120-126`; eleven graphs of 155 pass the pair, measured,
    /// `capability::uses_gelu_ungated`). Aliased and served exactly as
    /// [`Self::ReluSqr`], mapping to `frink_moe::GluAct::GeluUngated`;
    /// no fused device kernel spells it. `ggml_gelu` is the tanh form
    /// with an f16 table on the CPU, so its goldens hold at the GeGLU
    /// tolerance.
    GeluUngated,
    /// GATED ReLU, `down(relu(gate(x)) * up(x))` with a REAL gate
    /// matrix -- llama.cpp's `LLM_FFN_RELU` under `build_moe_ffn` with
    /// `gate_exps` present, which takes `ggml_reglu_split(gate, up)`
    /// (`llama-graph.cpp:2195-2197`; `smallthinker.cpp:158`, the only
    /// graph of 155 that passes it there -- `capability::uses_reglu`).
    ///
    /// `frink_moe::GluAct::Reglu`, on a pair the loader did NOT alias.
    /// Two variants rather than [`Self::ReluSqr`] with a flag, because
    /// `ffn_is_ungated` (the loader's aliasing decision) and
    /// `layer_ffn_acts` (the body) must agree about which of the two a
    /// file is, and a variant is the one spelling both read. The
    /// `GluAct` side is two variants for the same reason: it used to
    /// be one, `ungated()` answered `relu(up)^2` for it, and the dense
    /// hot path skipped a gate that was real (the SmallThinker fixture
    /// found it; `frink_moe::GluAct` says how). No fused device kernel
    /// spells it, so `fused_kernel_gelu_flag` is `None` and every Metal
    /// launch refuses.
    Reglu,
    /// UNGATED xIELU with PER-LAYER parameters: `down(xielu_il(up(x)))`
    /// -- llama.cpp's `ggml_xielu(up, alpha_n[il], alpha_p[il],
    /// beta[il], eps[il])` (`apertus.cpp:132-138`), the four read as
    /// `n_layer`-long arrays or broadcast scalars (`:6-9`).
    ///
    /// The table IS the variant, so there is no second field for it to
    /// disagree with. `crate::act_layers` reads it and hands layer
    /// `il`'s set out through [`ModelConfig::layer_ffn_act`]; the
    /// loader aliases gate to up exactly as for [`Self::ReluSqr`], and
    /// `frink_moe::GluAct::Xielu` reads the `up` operand alone. No
    /// fused device kernel spells it, so every Metal launch refuses it.
    Xielu(crate::act_layers::XieluLayers),
    /// SwiGLU with a PER-LAYER, PER-SITE clamp: llama.cpp's
    /// `swiglu_clamp_exp[il]` on the routed experts and
    /// `swiglu_clamp_shexp[il]` on the dense layers and shared experts
    /// (`step35.cpp:28-29`; applied at `llama-graph.cpp:2146-2164` and
    /// `:1751-1768` as `min(silu(gate), l) * clamp(up, -l, l)`). A zero
    /// entry is plain SwiGLU on that site; `frink_moe::GluAct::
    /// SwigluClamped` is the body. No fused device kernel spells it.
    SwigluClamped(crate::act_layers::SwigluClamps),
}

/// [`ModelConfig::batch_window`]'s answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchWindow {
    /// Every query in the batch takes this window (`None`: full causal).
    Uniform(Option<usize>),
    /// Each query takes [`ModelConfig::layer_window_for_query`] at its
    /// own position.
    PerQuery,
}

impl ModelConfig {
    /// The V head width: the width of every V head, of each head's
    /// attention output, and so of `o_proj`'s input (`n_heads *
    /// v_head_dim()`). [`Self::head_dim`] unless the file declared
    /// `attention.value_length` apart from `attention.key_length` on an
    /// architecture that sizes them apart (`crate::kv_head_dims`).
    #[inline]
    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim.unwrap_or(self.head_dim)
    }

    /// Whether V heads are a different width from K heads. The fact
    /// every fused path refuses on.
    #[inline]
    pub fn kv_head_dims_split(&self) -> bool {
        self.v_head_dim() != self.head_dim
    }

    /// Re-picks the LongRoPE factor set now that the run's context size
    /// is known, matching llama.cpp `llama_model::get_rope_factors`:
    /// `rope_freqs.weight` (Llama 3) always wins; otherwise the long set
    /// applies only when the context exceeds
    /// `rope.scaling.original_context_length`, and the short set
    /// otherwise.
    ///
    /// A no-op for every checkpoint that ships neither set, which is all
    /// of them except the Phi-3/Phi-4 family today.
    ///
    /// Because it re-picks `rope_freqs` wholesale it would also discard
    /// a YaRN rewrite folded into that field at parse time (see
    /// [`Self::rope_freqs`]). No real checkpoint hits that: LongRoPE
    /// files declare `rope.scaling.type = "longrope"`, which the loader's
    /// YaRN arm deliberately does not claim, so the two never populate
    /// the field on the same file. The same caveat now covers
    /// [`RopeFreqs::swa`], and for the same reason: no LongRoPE
    /// checkpoint has alternating SWA layers.
    pub fn apply_runtime_context(&mut self, ctx: usize) {
        let (Some(orig), true) = (
            self.rope_orig_ctx,
            self.rope_freqs_long.is_some() || self.rope_freqs_short.is_some(),
        ) else {
            return;
        };
        let picked = if ctx > orig {
            self.rope_freqs_long.as_ref()
        } else {
            self.rope_freqs_short.as_ref()
        };
        if let Some(f) = picked
            .or(self.rope_freqs_long.as_ref())
            .or(self.rope_freqs_short.as_ref())
        {
            self.rope_freqs = Some(RopeFreqs {
                full: f.clone(),
                swa: None,
            });
        }
    }

    /// True if layer `layer_idx` (0-indexed) should be built as an
    /// ordinary dense FFN rather than this model's MoE topology: the
    /// leading-dense prefix, and, where the loader honours the
    /// interleave step (`crate::moe_interleave`, `llama4.cpp:64`), a
    /// layer with `(il + 1) % step != 0`.
    pub fn layer_is_dense(&self, layer_idx: usize) -> bool {
        layer_idx < self.n_dense_leading_layers
            || self
                .moe_interleave_step
                .is_some_and(|step| !(layer_idx + 1).is_multiple_of(step))
    }

    /// Sliding-window size for layer `il`, honouring Gemma-style
    /// alternating SWA patterns. `None` means full causal attention.
    pub fn layer_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        // llama.cpp's `is_swa(il)`, which `set_swa_pattern`
        // (`src/llama-hparams.cpp:8-22`) or the file's own array fills
        // in; `crate::swa_layers` is the one implementation of both.
        let window = self.sliding_window?;
        self.swa_layers.slides(layer_idx).then_some(window)
    }

    /// The window the single-query kernels take for a query at `pos`
    /// on layer `il`: the layer's sliding window, or, when the window
    /// is chunked (`crate::chunked_swa`), the `pos % chunk + 1`
    /// positions of the query's own chunk. `None` for a full layer.
    pub fn layer_window_for_query(&self, il: usize, pos: usize) -> Option<usize> {
        let w = self.layer_sliding_window(il)?;
        Some(if self.swa_chunked { pos % w + 1 } else { w })
    }

    /// The window a batch of `batch_size` queries starting at
    /// `start_pos` takes on layer `il`, for the batched prefill body:
    /// one window for the blocked kernel, or one per query where a
    /// chunked layer's queries do not share a chunk start.
    pub fn batch_window(&self, il: usize, start_pos: usize, batch_size: usize) -> BatchWindow {
        match self.layer_sliding_window(il) {
            None => BatchWindow::Uniform(None),
            Some(w) if !self.swa_chunked => BatchWindow::Uniform(Some(w)),
            // Every query in the first chunk sees its whole causal
            // prefix: `pos / chunk == 0` for all of them
            // (`llama-hparams.h:419-425`), which is the full mask.
            Some(chunk) if start_pos + batch_size <= chunk => BatchWindow::Uniform(None),
            Some(_) => BatchWindow::PerQuery,
        }
    }

    /// The narrowest sliding window any layer of this model uses, or
    /// `None` if every layer is full-causal.
    ///
    /// For an alternating-SWA model (gpt-oss, Gemma-3) the
    /// full-attention layers impose no constraint on the KV block
    /// layout and the sliding ones impose the window -- so the model's
    /// constraint is simply the window, present as soon as *any* layer
    /// slides. A model that is 5/6 full-attention is not 5/6 exempt:
    /// one mis-aligned sliding layer corrupts the answer.
    pub fn kv_block_window(&self) -> Option<usize> {
        (0..self.n_layers).find_map(|il| self.layer_sliding_window(il))
    }

    /// The window EVERY layer slides by, or `None` if any layer attends
    /// over the whole history.
    ///
    /// This is the opposite question to [`Self::kv_block_window`], and
    /// the difference is the whole reason both exist. That one asks
    /// "does any layer constrain the block layout", so one sliding layer
    /// is enough. This one asks "may a page that has fallen behind the
    /// window be taken away", and there one *full-attention* layer is
    /// enough to say no.
    ///
    /// A page group holds one block in every layer and is freed as a
    /// unit, so on an alternating-SWA model (gpt-oss, Gemma-3) freeing
    /// the group behind the window would take the full-attention layers'
    /// block with it -- and those layers still read position 0 at every
    /// step. The result is not a crash: the block is reused by another
    /// request and the full layers attend over its bytes. So this
    /// returns `None` for the alternating case, and a mixed-window model
    /// (were one to appear) gets `None` too rather than the narrowest
    /// window, because the widest is the one that must still be readable.
    pub fn uniform_sliding_window(&self) -> Option<usize> {
        let first = self.layer_sliding_window(0)?;
        (1..self.n_layers)
            .all(|il| self.layer_sliding_window(il) == Some(first))
            .then_some(first)
    }

    /// The KV cache block layout to use for this model, given the block
    /// size an operator asked for.
    ///
    /// The requested size is rounded *down* to something that divides
    /// the window (see [`frink_core::kv_swa`]), so a config that would
    /// straddle the window boundary becomes a smaller block rather than
    /// a startup failure or -- much worse -- a silently wrong mask.
    pub fn kv_block_layout(&self, desired_block_size: usize) -> frink_core::BlockLayout {
        let window = self.kv_block_window();
        let block_size = frink_core::aligned_block_size(desired_block_size, window);
        frink_core::BlockLayout::new(block_size, window)
            .expect("aligned_block_size returns a size BlockLayout accepts")
    }

    /// ALL THREE halves of layer `il`'s RoPE: the frequency base, the
    /// per-band divisors, which llama.cpp varies per layer together
    /// (`llama-model.cpp:2029-2035`, and see [`RopeFreqs`]), and the
    /// rotary WIDTH, which it varies by the same sliding-or-full fact
    /// (`n_rot(il)`, `llama-hparams.cpp:85-91`; [`Self::rope_dim_swa`]).
    ///
    /// Every RoPE call site takes the pair from here. Splitting them was
    /// the defect: `layer_rope_theta` varied the base per layer while
    /// `rope_freqs` was one global vector, so Gemma-3 4B/12B/27B roped
    /// their sliding layers at scaled positions llama.cpp leaves
    /// unscaled.
    ///
    /// **`None` means this layer does not rotate at all**, which is
    /// llama.cpp's per-layer `use_rope` gate --
    /// [`crate::rope_layers`] holds the rule and the six architectures
    /// that have one. It is an `Option` rather than a separate
    /// predicate beside the pair precisely so that a call site cannot
    /// take the base and the divisors without also answering "does this
    /// layer rotate": that is the third thing the three had to agree
    /// about, and two of them were already one value for this reason.
    /// The epsilon the post-attention / post-FFN norms run at. One
    /// accessor so a site cannot read the model's epsilon by habit.
    pub fn post_norm_eps(&self) -> f32 {
        self.post_norm_eps
    }

    pub fn layer_rope(&self, layer_idx: usize) -> Option<LayerRopeParams<'_>> {
        let sliding = self.layer_sliding_window(layer_idx).is_some();
        if !self.rope_layers.rotates(layer_idx, sliding) {
            return None;
        }
        let theta = match (sliding, self.rope_theta_swa) {
            (true, Some(theta)) => theta,
            _ => self.rope_theta,
        };
        let rot_dim = match (sliding, self.rope_dim_swa) {
            (true, Some(w)) => Some(w),
            _ => self.rope_dim,
        }
        // The whole head is spelled `None`, whichever key said so, so
        // nothing downstream special-cases "narrower by zero".
        .filter(|w| *w < self.head_dim);
        Some(LayerRopeParams {
            theta,
            freq_factors: self.rope_freqs.as_ref().map(|f| f.for_layer(sliding)),
            rot_dim,
        })
    }

    /// True when the sliding layers rotate a different width from the
    /// full ones -- the whole-model fact the fused Metal launches refuse
    /// on, since each takes ONE `rot_dim` uniform for every layer.
    ///
    /// Derived from [`Self::layer_rope`] rather than from the field, so
    /// a `rope_dim_swa` that merely restates `rope_dim` (or the whole
    /// head) is not a difference.
    pub fn rope_dim_varies_by_layer(&self) -> bool {
        let widths: Vec<Option<usize>> = (0..self.n_layers)
            .filter_map(|il| self.layer_rope(il).map(|r| r.rot_dim))
            .collect();
        widths.windows(2).any(|w| w[0] != w[1])
    }

    /// Does layer `il` rotate at all? Derived from [`Self::layer_rope`]
    /// rather than restated beside it, so the two can never disagree.
    pub fn layer_rotates(&self, layer_idx: usize) -> bool {
        self.layer_rope(layer_idx).is_some()
    }

    /// True when at least one layer of this model gets no rotation --
    /// the whole-model question, for the eligibility checks and the
    /// receipts that want it once rather than per layer.
    pub fn any_layer_unrotated(&self) -> bool {
        (0..self.n_layers).any(|il| !self.layer_rotates(il))
    }

    /// RoPE frequency base for layer `il` (SWA layers may differ), or
    /// `None` where the layer does not rotate.
    ///
    /// Prefer [`Self::layer_rope`] anywhere the divisors are needed too,
    /// which is every site that actually rotates something. This one is
    /// for the callers that only report or compare the base.
    pub fn layer_rope_theta(&self, layer_idx: usize) -> Option<f32> {
        self.layer_rope(layer_idx).map(|r| r.theta)
    }

    /// True when the sliding layers need different per-band divisors
    /// from the full-attention ones, i.e. when one `freq_factors` slice
    /// cannot describe every layer of this model. Gemma-3 4B/12B/27B
    /// are the shape that answers yes.
    ///
    /// It is NOT an eligibility check any more. It was one: the fused
    /// Metal stacks took a single slice for a whole run of layers and
    /// refused a model that answered yes here. They now take a
    /// `frink_metal::attn::LayerRope` per layer, so this is a statement
    /// about the checkpoint and nothing else -- which is all the loader
    /// tests ever wanted from it.
    pub fn rope_freqs_vary_by_layer(&self) -> bool {
        self.rope_freqs
            .as_ref()
            .is_some_and(RopeFreqs::varies_by_layer)
            // A model whose every layer slides, or none, uses one set
            // whatever the two vectors hold.
            && (0..self.n_layers).any(|il| self.layer_sliding_window(il).is_some())
            && (0..self.n_layers).any(|il| self.layer_sliding_window(il).is_none())
    }

    /// Which attention mechanism layer `layer_idx` (0-indexed, frink's
    /// usual convention) uses. For `AttentionKind::Gqa` every layer is
    /// `LayerAttentionKind::Gqa`; for `AttentionKind::KimiHybrid`, looks
    /// up `layer_idx + 1` (the real `kda_layers`/`full_attn_layers`
    /// lists are 1-indexed -- see `KimiHybridAttention`'s doc comment)
    /// in those real per-layer lists.
    ///
    /// # Panics
    /// If `layer_idx` isn't covered by either list of a `KimiHybrid`
    /// config -- can't happen for `kimi_k3()`, whose lists are tested
    /// (`kimi_k3_hybrid_attention_layers_partition_every_layer_exactly_once`)
    /// to partition every layer with no gaps, but a caller building a
    /// custom `KimiHybridAttention` must uphold the same invariant.
    pub fn layer_attention_kind(&self, layer_idx: usize) -> LayerAttentionKind {
        match &self.attention {
            AttentionKind::Gqa => LayerAttentionKind::Gqa,
            AttentionKind::KimiHybrid(hybrid) => {
                let one_indexed = layer_idx + 1;
                if hybrid.kda_layers.contains(&one_indexed) {
                    LayerAttentionKind::KimiKda
                } else if hybrid.full_attn_layers.contains(&one_indexed) {
                    LayerAttentionKind::KimiMla
                } else {
                    panic!(
                        "layer {layer_idx} (1-indexed {one_indexed}) is in neither \
                         kda_layers nor full_attn_layers"
                    )
                }
            }
        }
    }

    /// Total parameter count implied by the MoE config, as a sanity
    /// check against the publicly reported total (this is an order of
    /// magnitude check, not an exact parameter-count reproduction).
    pub fn approx_active_params_per_token(&self) -> usize {
        let attn_params_per_layer = 4 * self.hidden_dim * self.hidden_dim; // q,k,v,o (rough)
        let active_experts = self.moe.n_experts_active + self.moe.n_shared_experts;
        let expert_params = active_experts * 3 * self.moe.hidden_dim * self.moe.expert_ffn_dim; // gate,up,down
        self.n_layers * (attn_params_per_layer + expert_params)
    }
}

/// GLM-5.2 (Z.ai) **structural sketch only** — not a supported real
/// inference path. Real DSA lives in `glm_dsa` / `glm52_decoder` and is
/// not wired into `Decoder` / `frink-server`. This preset drives
/// smoke/bench with synthetic GQA weights only (~744B / ~40B active
/// hparams as published placeholders).
pub fn glm_5_2() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "glm-5.2",
        attention: AttentionKind::Gqa,
        n_layers: 92,
        n_mtp_blocks: 0,
        hidden_dim: 6144,
        n_heads: 48,
        n_kv_heads: 8,
        head_dim: 128,
        v_head_dim: None,
        vocab_size: 151552,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        post_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 256,
            n_experts_active: 8,
            n_shared_experts: 1,
            hidden_dim: 6144,
            expert_ffn_dim: 2048,
            // Sigmoid, not softmax: reading ik_llama.cpp's real GGUF
            // hparams-loading source (llama-hparams.cpp,
            // LLM_ARCH_GLM4_MOE case) directly showed GLM4-MoE-family
            // models default to sigmoid gating with post-selection
            // score renormalization. GLM-5.2 is presumed to continue
            // this lineage; not confirmed against GLM-5.2's own
            // config.json (unavailable in this environment).
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // No evidence found (via ik_llama.cpp source or public
        // reporting) that GLM-5.2 skips MoE on any leading layers;
        // defaulting to 0 (every layer uses this model's MoE
        // topology) rather than assuming DeepSeek's convention
        // applies here too.
        n_dense_leading_layers: 0,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        // Placeholder GQA path; real GLM-5.2 DSA uses interleaved RoPE
        // via `glm_dsa`/`mla`, not this preset's Decoder path.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_layers",
            "hidden_dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "rope_theta",
            "moe.expert_ffn_dim",
            "moe.n_shared_experts",
            "moe.gating (sigmoid assumed from GLM4-MoE-family convention found in ik_llama.cpp source, not confirmed for GLM-5.2 specifically)",
        ],
    }
}

/// DeepSeek V4 Pro **structural sketch only** — CSA/HCA is not on this
/// GQA `Decoder` path. Real primitives live under
/// `deepseek_v4_attention` / `hyper_connections` and are not assembled
/// into a served decoder yet. Hparams (~1.6T / ~49B active) are
/// placeholders for smoke/bench.
pub fn deepseek_v4_pro() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "deepseek-v4-pro",
        attention: AttentionKind::Gqa,
        n_layers: 96,
        n_mtp_blocks: 0,
        hidden_dim: 7168,
        n_heads: 56,
        n_kv_heads: 8,
        head_dim: 128,
        v_head_dim: None,
        vocab_size: 129280,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        post_norm_eps: 1e-6,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 385,
            n_experts_active: 6,
            n_shared_experts: 1,
            hidden_dim: 7168,
            expert_ffn_dim: 2048,
            // Sigmoid, not softmax: this is the stronger-confidence of
            // the two sigmoid-gating corrections in this file.
            // DeepSeek-V3's own published technical report explicitly
            // documents computing per-expert affinity via sigmoid and
            // renormalizing only the selected experts' scores to sum
            // to one; reading ik_llama.cpp's real GGUF hparams-loading
            // source (llama-hparams.cpp, LLM_ARCH_DEEPSEEK2 case)
            // confirmed this is exactly what that code path defaults
            // to for the DeepSeek-2/3 lineage. DeepSeek V4 Pro is
            // presumed to continue using sigmoid gating for the same
            // reason; not confirmed against V4 Pro's own config.json.
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // DeepSeek-V3's own published technical report documents the
        // first 3 transformer layers as dense (ordinary FFN, no
        // expert routing), with MoE starting from layer 4 onward;
        // ik_llama.cpp's real hparams-loading source
        // (LLM_KV_LEADING_DENSE_BLOCK_COUNT) confirms this is a real,
        // loaded GGUF metadata field for the DeepSeek-2/3 lineage.
        // DeepSeek V4 Pro is presumed to continue this convention;
        // not confirmed against V4 Pro's own config.json.
        n_dense_leading_layers: 3,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        // llama.cpp maps LLM_ARCH_DEEPSEEK4 -> LLAMA_ROPE_TYPE_NORM.
        rope_layout: RopeLayout::Norm,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_layers",
            "hidden_dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "moe.expert_ffn_dim",
            "attention_variant (CSA/HCA hybrid NOT implemented, GQA fallback in use)",
            "moe.gating (sqrtsoftplus: confirmed for real V4 in llama.cpp PR #24162; this preset still uses Sigmoid on the wrong GQA sketch path)",
            "n_dense_leading_layers (3: same confidence basis as gating above, DeepSeek-V3 technical report + ik_llama.cpp source, not confirmed for V4 Pro)",
        ],
    }
}

/// Kimi K3 **structural sketch only** for the generic GQA `Decoder`.
/// Real checkpoint work uses the dedicated Kimi stack (`kimi_loader` /
/// `KimiEngine`); slice-verified, not a full end-to-end run. Do not
/// treat this preset as a runnable Kimi substitute.
pub fn kimi_k3() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "kimi-k3",
        n_layers: 93,
        n_mtp_blocks: 0,
        hidden_dim: 7168,
        // n_heads/n_kv_heads/head_dim describe the Gqa fallback
        // Decoder actually runs today, not Kimi K3's real attention
        // (see `attention` below) -- kept at reasonable stand-in
        // values (matching MLA's num_heads=96 and combined
        // qk_nope+qk_rope head dim) rather than deleted, so the
        // placeholder path stays runnable.
        n_heads: 96,
        n_kv_heads: 96,
        head_dim: 192,
        v_head_dim: None,
        vocab_size: 163840,
        // Not present in the published text_config; RoPE only ever
        // applies to Gated MLA's 64-dim qk_rope_head_dim slice in the
        // real architecture, and Decoder doesn't implement that slicing
        // yet, so this remains an unconfirmed placeholder.
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-5,
        post_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 896,
            n_experts_active: 16,
            n_shared_experts: 2,
            hidden_dim: 7168,
            expert_ffn_dim: 3072,
            // Confirmed directly from the real config.json:
            // "moe_router_activation_func": "sigmoid".
            gating: GatingFunction::Sigmoid,
            norm_topk_prob: true,
         expert_group_count: None, expert_group_used_count: None,},
        // Confirmed directly from the real config.json:
        // "first_k_dense_replace": 1.
        n_dense_leading_layers: 1,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        // Kimi K3's real, published attention topology (verified
        // against huggingface.co/moonshotai/Kimi-K3/config.json's
        // linear_attn_config block and the real KimiDeltaAttention /
        // KimiMLAAttention reference implementations in
        // modeling_kimi_linear.py) -- not yet wired into Decoder's
        // forward pass, which still runs the Gqa placeholder above for
        // every layer regardless of this field.
        attention: AttentionKind::KimiHybrid(KimiHybridAttention {
            kda_layers: vec![
                1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29,
                30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55,
                57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79, 81, 82,
                83, 85, 86, 87, 89, 90, 91,
            ],
            full_attn_layers: vec![
                4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84,
                88, 92, 93,
            ],
            mla: MlaConfig {
                num_heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                use_output_gate: true,
                // Real, confirmed: Kimi K3's `KimiMLAAttention.forward`
                // never rotates -- see `MlaConfig::rope`'s doc comment.
                rope: None,
            },
            kda: KdaConfig {
                num_heads: 96,
                head_dim: 128,
                short_conv_kernel_size: 4,
                gate_lower_bound: -5.0,
                use_full_rank_gate: true,
            },
        }),
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        // GQA placeholder path only; real Kimi attention is rope-less MLA
        // or KDA and never reaches Decoder::apply_rope_head.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &[
            "n_heads/n_kv_heads/head_dim (describe the unimplemented Gqa placeholder, not Kimi K3's real MLA/KDA attention -- see `attention` field)",
            "rope_theta (not present in the published config; real architecture only applies RoPE to Gated MLA's qk_rope_head_dim slice, which Decoder doesn't implement)",
            "entire preset beyond hyperparameters (the real 2.8T-parameter checkpoint has not been run end to end; only real slices have, via the dedicated kimi_decoder/kimi_loader stack -- see docs/MODELS.md)",
        ],
    }
}

/// Matches the generated on-disk fixture exactly (hidden_dim, head
/// counts, ffn_dim, vocab, rope_theta, eps).
/// Used by `frink inspect-run` and the cross-validation test in
/// `crates/frink-models/tests/gguf_roundtrip.rs` to prove the real
/// GGUF loader + forward pass produce the same numbers as an
/// independent NumPy reference implementation reading the same file.
pub fn test_dense_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "frink-test-dense",
        attention: AttentionKind::Gqa,
        n_layers: 2,
        n_mtp_blocks: 0,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        v_head_dim: None,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        post_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 1,
            n_experts_active: 1,
            n_shared_experts: 0,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 0,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        // Matches the independent reference's split-half apply_rope.
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic test fixture, not a real model"],
    }
}

/// Matches the generated on-disk multi-expert MoE fixture: 4 experts,
/// top-2 routing, 1 shared
/// expert, packed 3D expert tensors. Used to verify the previously-
/// untested multi-expert loading path (`split_expert_tensor` in
/// `frink-models::loader`) against a real file, the same way
/// `test_dense_fixture` verifies the single-expert path.
pub fn test_moe_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "frink-test-moe",
        attention: AttentionKind::Gqa,
        n_layers: 2,
        n_mtp_blocks: 0,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        v_head_dim: None,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        post_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 4,
            n_experts_active: 2,
            n_shared_experts: 1,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 0,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic multi-expert test fixture, not a real model"],
    }
}

/// Matches the generated on-disk mixed-topology fixture: 3 layers, the
/// first of which is
/// an ordinary dense FFN and the remaining two are genuine MoE (3
/// experts, top-1 routing, 1 shared expert each). Used to verify the
/// "leading dense layers" loading path
/// (`ModelConfig::layer_is_dense`) against a real file -- the pattern
/// found in DeepSeek-2/3-family models via ik_llama.cpp's source
/// (`LLM_KV_LEADING_DENSE_BLOCK_COUNT`), which was previously only
/// documented, not implemented or tested.
pub fn test_mixed_fixture() -> ModelConfig {
    ModelConfig {
        sliding_window: None,
        name: "frink-test-mixed",
        attention: AttentionKind::Gqa,
        n_layers: 3,
        n_mtp_blocks: 0,
        hidden_dim: 32,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        v_head_dim: None,
        vocab_size: 32,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        post_norm_eps: 1e-5,
        moe: MoeLayerConfig {
            expert_weights_scale: 1.0,
            routed_weight_before_ffn: false,
            n_experts: 3,
            n_experts_active: 1,
            n_shared_experts: 1,
            hidden_dim: 32,
            expert_ffn_dim: 32,
            gating: GatingFunction::Softmax,
            norm_topk_prob: true,
            expert_group_count: None,
            expert_group_used_count: None,
        },
        n_dense_leading_layers: 1,
        moe_interleave_step: None,
        norm_function: crate::norm::NormFunction::Rms,
        rope_freqs: None,
        rope_attn_factor: 1.0,
        rope_dim: None,
        rope_dim_swa: None,
        rope_freqs_long: None,
        rope_freqs_short: None,
        rope_orig_ctx: None,
        rope_layout: RopeLayout::Neox,
        qk_norm_style: crate::capability::QkNormStyle::WholeVector,
        swa_layers: crate::swa_layers::SwaLayers::All,
        rope_layers: crate::rope_layers::RopeLayers::All,
        layer_shapes: crate::layer_shapes::LayerShapes::Uniform,
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
        ffn_activation: FfnActivation::Swiglu,
        best_effort_fields: &["this is a synthetic mixed dense/MoE test fixture, not a real model"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_layout_for_gguf_architecture_matches_llama_cpp() {
        // Confirmed against llama.cpp's llama_model_rope_type
        // (src/llama-model.cpp): llama -> NORM, olmoe/qwen2/phi3/gemma -> NEOX.
        assert_eq!(RopeLayout::for_gguf_architecture("llama"), RopeLayout::Norm);
        assert_eq!(
            RopeLayout::for_gguf_architecture("llama4"),
            RopeLayout::Norm
        );
        assert_eq!(
            RopeLayout::for_gguf_architecture("deepseek2"),
            RopeLayout::Norm
        );
        assert_eq!(RopeLayout::for_gguf_architecture("olmoe"), RopeLayout::Neox);
        assert_eq!(RopeLayout::for_gguf_architecture("qwen2"), RopeLayout::Neox);
        assert_eq!(
            RopeLayout::for_gguf_architecture("qwen2moe"),
            RopeLayout::Neox
        );
        assert_eq!(RopeLayout::for_gguf_architecture("qwen3"), RopeLayout::Neox);
        assert_eq!(RopeLayout::for_gguf_architecture("phi3"), RopeLayout::Neox);
        assert_eq!(
            RopeLayout::for_gguf_architecture("gemma3"),
            RopeLayout::Neox
        );
        // Unknown architectures keep the historical Neox default at this
        // helper only; load-time uses capability::resolve_architecture and
        // fails closed instead of guessing.
        assert_eq!(
            RopeLayout::for_gguf_architecture("totally-unknown-arch"),
            RopeLayout::Neox
        );
    }

    /// gpt-oss's real shape: a 128-token window on every other layer.
    /// A KV block size of 128 or any divisor of it is fine; 48 or 256
    /// are not, and the config layer must round down rather than hand
    /// the cache something it will refuse (or, worse, accept).
    #[test]
    fn an_alternating_swa_model_constrains_the_block_layout() {
        let mut cfg = test_dense_fixture();
        cfg.n_layers = 24;
        cfg.sliding_window = Some(128);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(2, false);

        // Half the layers are full-attention, but the model is still
        // constrained: one mis-aligned sliding layer is enough.
        assert!(cfg.layer_sliding_window(1).is_none() || cfg.layer_sliding_window(0).is_none());
        assert_eq!(cfg.kv_block_window(), Some(128));

        let layout = cfg.kv_block_layout(256);
        assert_eq!(layout.block_size(), 128, "256 must round down, not up");
        assert_eq!(layout.sliding_window(), Some(128));
        assert_eq!(layout.blocks_per_window(), Some(1));

        assert_eq!(cfg.kv_block_layout(48).block_size(), 32);
        assert_eq!(cfg.kv_block_layout(32).block_size(), 32);
    }

    /// Gemma-3: window 512, every 6th layer full-attention.
    #[test]
    fn a_gemma3_shaped_model_takes_its_window_from_the_sliding_layers() {
        let mut cfg = test_dense_fixture();
        cfg.n_layers = 30;
        cfg.sliding_window = Some(512);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
        assert!(
            cfg.layer_sliding_window(5).is_none(),
            "every 6th layer is full-attention"
        );
        assert_eq!(cfg.kv_block_window(), Some(512));
        assert_eq!(cfg.kv_block_layout(100).block_size(), 64);
        assert_eq!(cfg.kv_block_layout(64).blocks_per_window(), Some(8));
    }

    /// The two window questions give OPPOSITE answers on an alternating
    /// model, and that is the point of having both.
    ///
    /// "Does any layer constrain the block layout" is yes, so the block
    /// size rounds down to the window. "May a page behind the window be
    /// taken away" is no, because the group holds the full-attention
    /// layers' blocks too and those layers still read position 0. A
    /// serving path that read `kv_block_window` for the second question
    /// would free pages half the layers are still attending over -- not
    /// a crash, just another request's bytes in this one's answer.
    #[test]
    fn only_a_uniformly_windowed_model_may_give_a_page_back() {
        let mut alternating = test_dense_fixture();
        alternating.n_layers = 24;
        alternating.sliding_window = Some(128);
        alternating.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        assert_eq!(alternating.kv_block_window(), Some(128));
        assert_eq!(
            alternating.uniform_sliding_window(),
            None,
            "a full-attention layer forbids the slide"
        );

        let mut uniform = test_dense_fixture();
        uniform.n_layers = 24;
        uniform.sliding_window = Some(128);
        uniform.swa_layers = crate::swa_layers::SwaLayers::All;
        assert_eq!(uniform.uniform_sliding_window(), Some(128));

        // `Some(0)` is llama.cpp's spelling of "every layer slides"
        // (`set_swa_pattern(0)`), and it is the one that may give a page
        // back. `Some(1)` is the OPPOSITE -- no layer slides -- and this
        // used to assert the two were the same, which is how the
        // inversion stayed invisible.
        let mut period_zero = uniform.clone();
        period_zero.swa_layers = crate::swa_layers::SwaLayers::period(0, false);
        assert_eq!(period_zero.uniform_sliding_window(), Some(128));

        let mut period_one = uniform.clone();
        period_one.swa_layers = crate::swa_layers::SwaLayers::period(1, false);
        assert_eq!(
            period_one.uniform_sliding_window(),
            None,
            "period 1 windows no layer, so there is no window to slide"
        );
        assert_eq!(period_one.kv_block_window(), None);

        let mut full = test_dense_fixture();
        full.sliding_window = None;
        assert_eq!(full.uniform_sliding_window(), None);
    }

    #[test]
    fn a_full_causal_model_keeps_the_block_size_it_was_given() {
        let mut cfg = test_dense_fixture();
        cfg.sliding_window = None;
        cfg.swa_layers = crate::swa_layers::SwaLayers::All;
        assert_eq!(cfg.kv_block_window(), None);
        let layout = cfg.kv_block_layout(48);
        assert_eq!(layout.block_size(), 48);
        assert_eq!(layout.sliding_window(), None);
    }

    #[test]
    fn all_presets_have_consistent_moe_hidden_dim() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert_eq!(
                cfg.hidden_dim, cfg.moe.hidden_dim,
                "{}: attention hidden_dim and MoE hidden_dim must match",
                cfg.name
            );
        }
    }

    #[test]
    fn all_presets_route_fewer_experts_than_total() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert!(
                cfg.moe.n_experts_active < cfg.moe.n_experts,
                "{}: active experts must be a sparse subset of total experts",
                cfg.name
            );
        }
    }

    #[test]
    fn all_presets_have_divisible_heads() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert_eq!(
                cfg.n_heads % cfg.n_kv_heads,
                0,
                "{}: n_heads must be a multiple of n_kv_heads for GQA grouping",
                cfg.name
            );
        }
    }

    #[test]
    fn every_preset_declares_its_uncertain_fields() {
        // This is a documentation-honesty test: any preset with zero
        // best_effort_fields would be silently overclaiming precision
        // we don't have. Fail loudly if that ever happens.
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            assert!(
                !cfg.best_effort_fields.is_empty(),
                "{}: must disclose which fields are unconfirmed estimates",
                cfg.name
            );
        }
    }

    /// Kimi K3's `kda_layers`/`full_attn_layers` were transcribed by
    /// hand from the real published config.json; this test guards
    /// against a transcription slip (duplicate, out-of-range, or
    /// missing layer index) rather than trusting the transcription.
    #[test]
    fn kimi_k3_hybrid_attention_layers_partition_every_layer_exactly_once() {
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };

        let mut seen = std::collections::HashSet::new();
        for &l in hybrid
            .kda_layers
            .iter()
            .chain(hybrid.full_attn_layers.iter())
        {
            assert!(
                (1..=cfg.n_layers).contains(&l),
                "layer {l} is out of the published 1..={} range",
                cfg.n_layers
            );
            assert!(
                seen.insert(l),
                "layer {l} appears in both/either list twice"
            );
        }
        // Dense-vs-MoE (n_dense_leading_layers) and attention-type
        // (KDA vs Gated MLA) are independent per-layer properties in
        // the real config -- e.g. layer 1 is both the sole dense
        // leading layer *and* a KDA layer -- so every one of the 93
        // layers, dense or not, is covered by exactly one of these two
        // lists (confirmed: 69 + 24 == 93, not 93 - 1).
        assert_eq!(
            hybrid.kda_layers.len() + hybrid.full_attn_layers.len(),
            cfg.n_layers,
            "every layer must be assigned exactly one of KDA or Gated MLA"
        );
        assert_eq!(
            hybrid.kda_layers.len(),
            69,
            "expected 69 KDA layers per the published config"
        );
        assert_eq!(
            hybrid.full_attn_layers.len(),
            24,
            "expected 24 Gated MLA layers per the published config"
        );
    }

    #[test]
    fn layer_attention_kind_is_gqa_for_every_layer_of_a_gqa_model() {
        let cfg = glm_5_2();
        for l in 0..cfg.n_layers {
            assert_eq!(cfg.layer_attention_kind(l), LayerAttentionKind::Gqa);
        }
    }

    #[test]
    fn layer_attention_kind_classifies_every_kimi_k3_layer_without_panicking() {
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };
        for l in 0..cfg.n_layers {
            let kind = cfg.layer_attention_kind(l);
            let one_indexed = l + 1;
            if hybrid.kda_layers.contains(&one_indexed) {
                assert_eq!(kind, LayerAttentionKind::KimiKda);
            } else {
                assert_eq!(kind, LayerAttentionKind::KimiMla);
            }
        }
    }

    #[test]
    fn layer_attention_kind_matches_the_real_published_layer_1_and_4() {
        // Layer 1 (1-indexed, so index 0 here) is published as KDA;
        // layer 4 (index 3) is published as the first Gated MLA layer.
        let cfg = kimi_k3();
        assert_eq!(cfg.layer_attention_kind(0), LayerAttentionKind::KimiKda);
        assert_eq!(cfg.layer_attention_kind(3), LayerAttentionKind::KimiMla);
    }

    #[test]
    fn kimi_k3_mla_q_head_dim_matches_gqa_placeholder_head_dim() {
        // The Gqa-placeholder head_dim above is deliberately set to
        // Gated MLA's combined q_head_dim (qk_nope + qk_rope) so the
        // placeholder path at least reflects a real dimension from the
        // published config rather than an arbitrary guess.
        let cfg = kimi_k3();
        let AttentionKind::KimiHybrid(hybrid) = &cfg.attention else {
            panic!("kimi_k3() must use AttentionKind::KimiHybrid");
        };
        assert_eq!(
            cfg.head_dim,
            hybrid.mla.qk_nope_head_dim + hybrid.mla.qk_rope_head_dim
        );
    }

    #[test]
    fn approx_active_params_is_nonzero_and_finite_order_of_magnitude() {
        for cfg in [glm_5_2(), deepseek_v4_pro(), kimi_k3()] {
            let approx = cfg.approx_active_params_per_token();
            // Sanity band: active params/token for these models is
            // reported in the tens of billions; this is a loose
            // order-of-magnitude check (1e9 to 1e12), not a precise
            // parameter-count reproduction.
            assert!(
                approx > 1_000_000_000 && approx < 1_000_000_000_000,
                "{}: approx_active_params_per_token={approx} is outside a plausible range",
                cfg.name
            );
        }
    }
}

#[cfg(test)]
mod longrope_tests {
    use super::*;

    fn cfg_with_factors() -> ModelConfig {
        let mut c = test_dense_fixture();
        c.rope_orig_ctx = Some(4096);
        c.rope_freqs_short = Some(vec![1.0; 48]);
        c.rope_freqs_long = Some((0..48).map(|i| 1.0 + i as f32).collect());
        c.rope_freqs = None;
        c
    }

    /// llama.cpp `llama_model::get_rope_factors`: long only when the
    /// run's context exceeds `original_context_length`. Phi-4-mini's
    /// short set is all ones, so picking long at 4096 would apply a
    /// correction the model never asked for at that length.
    #[test]
    fn long_set_only_above_the_original_context() {
        let mut c = cfg_with_factors();
        c.apply_runtime_context(4096);
        assert_eq!(
            c.rope_freqs.as_ref().unwrap().full[1],
            1.0,
            "at the threshold, short"
        );

        let mut c = cfg_with_factors();
        c.apply_runtime_context(4097);
        assert_eq!(
            c.rope_freqs.as_ref().unwrap().full[1],
            2.0,
            "above it, long"
        );

        let mut c = cfg_with_factors();
        c.apply_runtime_context(1024);
        assert_eq!(
            c.rope_freqs.as_ref().unwrap().full[1],
            1.0,
            "below it, short"
        );
    }

    /// `rope_freqs.weight` (Llama 3) is not a LongRoPE set and outranks
    /// one, the same precedence llama.cpp gives it. The loader encodes
    /// that by leaving the long/short pair empty whenever the explicit
    /// tensor is present, so the runtime re-pick has nothing to apply.
    #[test]
    fn an_explicit_rope_freqs_tensor_is_never_overridden() {
        let mut c = test_dense_fixture();
        c.rope_freqs = Some(RopeFreqs {
            full: vec![7.0; 48],
            swa: None,
        });
        c.rope_orig_ctx = Some(4096);
        c.rope_freqs_long = None;
        c.rope_freqs_short = None;
        c.apply_runtime_context(131072);
        assert_eq!(c.rope_freqs.as_ref().unwrap().full[0], 7.0);
    }

    /// A checkpoint with neither set must come back untouched, so the
    /// call is free to sit on every load path.
    #[test]
    fn models_without_longrope_are_untouched() {
        let mut c = test_dense_fixture();
        c.rope_freqs = None;
        c.apply_runtime_context(8192);
        assert!(c.rope_freqs.is_none());
        assert!(c.rope_orig_ctx.is_none());
    }
}
