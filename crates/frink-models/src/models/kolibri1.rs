//! `kolibri1` (Aleph-Alpha Kolibri-1, 78B-A3.5B):
//! `src/models/kolibri1.cpp` at the pinned commit.
//!
//! A sandwich-normed GQA + MoE layer on every layer. What it shares
//! with graphs frink already serves, and what it does not:
//!
//! - **Selection** (`:133-141`, the one new op). `selection = logits +
//!   exp_probs_b`, ranked by `ggml_argsort_top_k` and handed to
//!   `build_moe_ffn` as `selected_experts_in` with `exp_probs_b =
//!   nullptr` (`:149,155`). Every other biased router adds the bias
//!   AFTER the gating (`llama-graph.cpp:2279`), so this is
//!   [`SelectionRule::BiasBeforeGating`]. The weights are
//!   `gating(logits)` at those ids, unbiased (`llama-graph.cpp:2336`);
//!   the expert-group block is skipped with the selection, and `w_scale`
//!   is the literal 1.0 (`:150`).
//! - **Gating and `norm_w`** come from the file, both REQUIRED keys
//!   (`:8-9`). The converter writes SIGMOID (`conversion/kolibri1.py:24`)
//!   and `norm_topk_prob` as `expert_weights_norm`.
//! - **Shared expert** (`:59-61,158-163`): one ungated SiLU FFN, added to
//!   the routed output BEFORE `ffn_post_norm` (`:165-167`), which is the
//!   generic decoder's order. `:12` throws unless the count is exactly
//!   one and its width nonzero.
//! - **Norms**: RMS everywhere; Gemma-2's sandwich, both post norms
//!   REQUIRED (`:41,51`), the post-FFN norm over routed + shared. QK norm
//!   is per head at `{head_dim}` (`:47-48,85-86`), before RoPE.
//! - **Attention**: `build_attn_inp_kv_iswa` (`:69`), the window REQUIRED
//!   and nonzero (`:10-12`), the per-layer bool ARRAY by `get_arr`
//!   (`:17`; `crate::swa_layers::PER_LAYER_ARRAY_READERS`, which the
//!   registry does not carry yet). RoPE on the sliding layers only
//!   (`:88`), NEOX (`llama-model.cpp:3667`), `kq_scale = 1/sqrt(head_dim)`
//!   (`:74`).
//! - **RoPE base and scale on the sliding layers** are the model's
//!   (`:19-20`), and the graph reads no `rope.freq_base_swa`: the key is
//!   dead, though `conversion/base.py:1682` writes it from a
//!   `sliding_attention` rope block. The graph's `n_rot` is layer 0's
//!   (`llama-graph.cpp:1557`) on every rotated layer, not `n_rot(il)`.
//! - **Dead keys the converter can write**: `leading_dense_block_count`
//!   and `expert_weights_scale` (`conversion/afmoe.py:30,36`, the base
//!   class); `load_arch_tensors` builds every layer MoE (`:52-61`) and the
//!   graph passes `w_scale = 1.0`.
//!
//! Evidence: `tests/kolibri1_graphs.rs`, against libllama.

use frink_gguf::TensorSource;
use frink_moe::SelectionRule;

use super::facts::{ArchFacts, HparamRules, MoeFacts, SwaFacts, TensorFacts};
use super::ModelDef;
use crate::norm_sites::{NormSites, StoredNorm};
use crate::rope_layers::RopeLayers;

pub const FACTS: ArchFacts = ArchFacts {
    moe: MoeFacts {
        reads_expert_weights_norm: true,
        leading_dense_key_is_inert: true,
        selection: SelectionRule::BiasBeforeGating,
        ..ArchFacts::DEFAULT.moe
    },
    swa: SwaFacts {
        window_required: Some("kolibri1.cpp:10"),
        rope_base_follows_model: true,
        rope_scale_follows_model: true,
        rope_base_key_unread: Some("kolibri1.cpp:19"),
        ..ArchFacts::DEFAULT.swa
    },
    tensors: TensorFacts {
        norm_sites: NormSites {
            post_attn: Some(StoredNorm::required(&["post_attention_norm"])),
            post_ffn: Some(StoredNorm::required(&["post_ffw_norm"])),
            ..NormSites::GENERIC
        },
        ..ArchFacts::DEFAULT.tensors
    },
    rules: HparamRules {
        rope_layers,
        load_checks,
        ..HparamRules::LEGACY
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["kolibri1"],
    source: "src/models/kolibri1.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/kolibri1_graphs.rs: a fixture whose bias ranks the experts differently before \
         and after the sigmoid, against libllama's logits and a 32-step greedy decode",
    ),
};

/// `:88`: `if (hparams.is_swa(il))` around both `ggml_rope_ext` calls.
/// The window is REQUIRED (`:10`), so the rule does not depend on it.
fn rope_layers(_: &str, _: usize, _: bool, _: usize) -> RopeLayers {
    RopeLayers::SlidingOnly
}

/// The throws in `load_arch_hparams` / `load_arch_tensors` that no table
/// states.
fn load_checks(file: &dyn TensorSource, arch: &str, n_layers: usize) -> Option<String> {
    let key = |s: &str| format!("{arch}.{s}");
    let required = |k: &str, line: &str| {
        file.metadata(&key(k)).is_none().then(|| {
            format!(
                "`{}` is absent; kolibri1.cpp:{line} reads it as a REQUIRED key",
                key(k)
            )
        })
    };
    if let Some(r) = required("attention.layer_norm_rms_epsilon", "4")
        .or_else(|| required("expert_shared_count", "6"))
        .or_else(|| required("expert_shared_feed_forward_length", "7"))
        .or_else(|| required("expert_weights_norm", "8"))
        .or_else(|| required("expert_gating_func", "9"))
    {
        return Some(r);
    }
    let shared = file.metadata_u64(&key("expert_shared_count"));
    let shexp = file.metadata_u64(&key("expert_shared_feed_forward_length"));
    if shared != Some(1) || shexp == Some(0) {
        return Some(format!(
            "expert_shared_count {shared:?} and expert_shared_feed_forward_length {shexp:?}: \
             kolibri1.cpp:12 throws unless there is exactly one shared expert of nonzero width"
        ));
    }
    // 1 softmax, 2 sigmoid. 0 aborts in `build_moe_ffn`
    // (`llama-graph.cpp:2271`); 3 and 4 are gatings the router here
    // does not apply to this graph.
    let gating = file.metadata_u64(&key("expert_gating_func"));
    if !matches!(gating, Some(1 | 2)) {
        return Some(format!(
            "expert_gating_func {gating:?}: only softmax (1) and sigmoid (2) are served for \
             kolibri1"
        ));
    }
    // Every rotated layer takes the graph's `n_rot`, layer 0's
    // (`llama-graph.cpp:1557`), where `ModelConfig::layer_rope` would
    // give a sliding layer `rope.dimension_count_swa`.
    let full = file.metadata_u64(&key("rope.dimension_count"));
    let swa = file.metadata_u64(&key("rope.dimension_count_swa"));
    if swa.is_some() && swa != full {
        return Some(format!(
            "rope.dimension_count_swa {swa:?} differs from rope.dimension_count {full:?}: \
             kolibri1.cpp:93-102 rotates every sliding layer at layer 0's width"
        ));
    }
    // Created with flag 0 (`:43-48,54`), where the generic loader
    // treats each as optional.
    for l in 0..n_layers {
        for t in [
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "exp_probs_b.bias",
        ] {
            let name = format!("blk.{l}.{t}");
            if file.find_tensor(&name).is_none() {
                return Some(format!(
                    "`{name}` is absent; kolibri1.cpp:43-54 creates it as REQUIRED"
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_source::StubSource;
    use frink_gguf::GgufValue;

    /// Keys to replace, add, or (`None`) drop.
    type Set<'a> = &'a [(&'a str, Option<GgufValue>)];

    /// A file that passes, with `set` replacing or adding keys (`None`
    /// drops one): `StubSource` answers the FIRST entry for a key.
    fn file(set: Set) -> StubSource {
        let mut keys = vec![
            (
                "attention.layer_norm_rms_epsilon",
                Some(GgufValue::F32(1e-5)),
            ),
            ("expert_shared_count", Some(GgufValue::U32(1))),
            ("expert_shared_feed_forward_length", Some(GgufValue::U32(8))),
            ("expert_weights_norm", Some(GgufValue::Bool(true))),
            ("expert_gating_func", Some(GgufValue::U32(2))),
        ];
        for (k, v) in set {
            match keys.iter_mut().find(|(have, _)| have == k) {
                Some(slot) => slot.1 = v.clone(),
                None => keys.push((k, v.clone())),
            }
        }
        let mut src = StubSource::with_tensors(&[
            "blk.0.attn_q_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.exp_probs_b.bias",
        ]);
        for (k, v) in keys {
            if let Some(v) = v {
                src = src.with_key(&format!("kolibri1.{k}"), v);
            }
        }
        src
    }

    #[test]
    fn a_complete_file_passes() {
        assert_eq!(load_checks(&file(&[]), "kolibri1", 1), None);
    }

    #[test]
    fn each_throw_upstream_is_a_refusal_here() {
        let cases: [(Set, usize, &str); 6] = [
            (
                &[("expert_shared_count", Some(GgufValue::U32(2)))],
                1,
                "exactly one shared expert",
            ),
            (
                &[("expert_shared_feed_forward_length", Some(GgufValue::U32(0)))],
                1,
                "exactly one shared expert",
            ),
            (
                &[("expert_gating_func", Some(GgufValue::U32(4)))],
                1,
                "expert_gating_func Some(4)",
            ),
            (
                &[
                    ("rope.dimension_count", Some(GgufValue::U32(8))),
                    ("rope.dimension_count_swa", Some(GgufValue::U32(4))),
                ],
                1,
                "layer 0's width",
            ),
            (
                &[("expert_weights_norm", None)],
                1,
                "expert_weights_norm` is absent",
            ),
            (&[], 2, "blk.1.attn_q_norm.weight"),
        ];
        for (i, (set, layers, want)) in cases.into_iter().enumerate() {
            let got = load_checks(&file(set), "kolibri1", layers);
            assert!(
                got.as_deref().is_some_and(|r| r.contains(want)),
                "case {i}: {got:?} should name {want:?}"
            );
        }
    }
}
