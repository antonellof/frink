//! The decoder's unit tests, by the path they exercise. This file holds
//! the shared configs and decoder transforms the submodules build on.

use super::*;
use crate::config::glm_5_2;
use frink_core::cache::PagedKvStore;
use frink_core::cache::{KvCache, PagedKvCache, SharedPagedKv};
use frink_moe::{route_top_k, run_expert, ExpertWeights};

mod batch;
mod multi_seq;
mod paged;
mod predicates;
mod token;

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
