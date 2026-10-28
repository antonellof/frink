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
mod tests {
    use super::*;
    use crate::config::glm_5_2;
    use frink_core::cache::PagedKvStore;
    use frink_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
    use frink_moe::{route_top_k, run_expert, ExpertWeights};

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
            for (a, b) in ca.k.to_f32_vec().iter().zip(cb.k.to_f32_vec().iter()) {
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
    use frink_core::tensor::Tensor;
    use frink_moe::GluAct;

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
