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
mod batch;
mod construct;
#[cfg(feature = "cuda")]
mod cuda_prefill;
#[cfg(feature = "metal")]
mod device_attention;
mod embed;
mod entry;
mod ffn_batch;
mod ffn_block;
mod forward_token;
#[cfg(feature = "metal")]
mod fused_attention;
#[cfg(feature = "metal")]
mod fused_recurrent;
#[cfg(any(feature = "metal", feature = "cuda"))]
mod fused_view;
mod host_attention;
mod hrm_streams;
pub mod kv_window;
mod lm_head;
mod metal_eligibility;
#[cfg(feature = "metal")]
mod metal_kv;
#[cfg(feature = "metal")]
mod metal_moe;
#[cfg(feature = "metal")]
mod metal_prefill;
mod moe_host;
mod multi_seq;
mod paged;
mod pre_norm;
mod qk_norm;
mod qkv_bias;
mod recurrent_block;
mod rope;
mod weights;

use frink_core::weight_matrix::WeightMatrix;

use crate::config::ModelConfig;
use crate::norm::NormOp;

pub(crate) use attn_block::KvStep;
pub use kv_window::{KvWindowPolicy, KV_WINDOW_ENV};
pub use multi_seq::MultiSeqKv;
#[cfg(feature = "metal")]
pub use weights::MoePackedQ4Planes;
pub use weights::{
    AttnWeights, ExpertBacking, GptOssLayer, GptOssWeights, LayerWeights, MoeWeights,
};

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

/// Every source file of this module -- `decoder.rs` and everything
/// under `decoder/` -- as `(path relative to src/, contents)`, read at
/// test time. The source-reading tests in `entry.rs` and `pre_norm.rs`
/// walk this rather than a hand-written list, so a body moved into a
/// new file stays inside what they check.
#[cfg(test)]
pub(crate) fn module_sources() -> Vec<(String, String)> {
    use std::path::{Path, PathBuf};
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("src/decoder is readable") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut paths = vec![src.join("decoder.rs")];
    walk(&src.join("decoder"), &mut paths);
    paths.sort();
    let out: Vec<(String, String)> = paths
        .iter()
        .map(|p| {
            let rel = p.strip_prefix(&src).expect("under src/");
            let rel = rel.to_string_lossy().replace('\\', "/");
            (rel, std::fs::read_to_string(p).expect("source reads"))
        })
        .collect();
    assert!(
        out.len() > 2 && out.iter().any(|(n, _)| n == "decoder/entry.rs"),
        "walked src/decoder and did not find the module's files"
    );
    out
}

#[cfg(test)]
mod tests;

/// The Metal side of Phi-3/Phi-4's RoPE: partial rotary and LongRoPE's
/// `attn_factor` used to be a refusal in `layer_supports_metal_attn`
/// and are now two uniforms on [`frink_metal::attn::MetalRope`].
#[cfg(all(test, feature = "metal"))]
mod metal_rope_tests;
