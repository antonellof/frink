//! **NextN / MTP blocks are inside `block_count`, and are not layers.**
//!
//! A multi-token-prediction head is exported as one or more EXTRA
//! decoder blocks appended after the trunk -- each with its own
//! `attn_norm`, QKV, `wo`, `ffn_norm`, experts, PLUS the `nextn.*`
//! tensors that make it a speculative head (`eh_proj`, `enorm`, `hnorm`,
//! `shared_head_*`, `embed_tokens`). The converters count them in
//! `block_count` (`conversion/mimo.py:22`, `step3.py:117-119`,
//! `exaone.py:132,224`, `glm.py:99,116`, `deepseek.py:457,525`) and
//! write `{arch}.nextn_predict_layers` beside it.
//!
//! llama.cpp never runs them as layers:
//!
//! - `llama-model.cpp:1092` reads `block_count` into `n_layer_all`;
//! - `llama-hparams.cpp:280-282` defines `n_layer()` as
//!   `n_layer_all - n_layer_nextn`;
//! - `llama-graph.cpp:1433` builds every graph with `n_layer =
//!   hparams.n_layer()`, so `mimo2.cpp:108`, `step35.cpp:205`,
//!   `glm4-moe.cpp:161`, `deepseek2.cpp:470` loop over the trunk only;
//! - each tensor loader loops `0..n_layer_all` and creates the blocks
//!   at `i >= n_layer` with `TENSOR_SKIP` (`exaone4.cpp:43-49`,
//!   `mimo2.cpp:35-37,51-52`) unless the context was opened as an MTP
//!   draft (`load_mtp`), so the bytes are neither read nor missed.
//!
//! **UPSTREAM CHANGED THIS, and the change is recorded before it is
//! followed.** When the pin moved to `5b59b83` on 2026-09-19,
//! `llama-model.cpp:1261` reads `LLM_KV_NEXTN_PREDICT_LAYERS` in the
//! COMMON loader, for every architecture, right after `block_count`
//! and with a `GGML_ASSERT(n_layer_nextn <= n_layer_all)` beside it
//! (upstream commit 9d81721, "load hparams.n_layer_nextn before
//! n_layer() calls"). So upstream now subtracts the trailing blocks
//! whatever the architecture, where it used to subtract only in the
//! per-architecture loaders, and `grep -l
//! LLM_KV_NEXTN_PREDICT_LAYERS src/models/*.cpp` is down to six files
//! that read it a second time for their own reasons.
//!
//! frink still refuses a nonzero key on an architecture outside
//! [`NEXTN_READERS`], which is now an OVER-refusal rather than a
//! divergence in the dangerous direction: a file upstream would run
//! by skipping its MTP blocks stops here instead of running them as
//! decoder layers. Lifting it is a row of its own -- it needs a
//! libllama golden built from the moved pin, on an architecture that
//! is NOT one of the seventeen -- and until that exists the table
//! below is what has evidence.
//!
//! **Only the graphs that READ the key subtract** (the measurement as
//! it stood at the 2026-08-04 pin, and the one the table is built
//! from): `grep -l
//! LLM_KV_NEXTN_PREDICT_LAYERS src/models/*.cpp` was the seventeen in
//! [`NEXTN_READERS`]. For any other architecture `n_layer_nextn` stays
//! 0, every block runs, and a file carrying `nextn.*` tensors fails
//! llama.cpp's own "not all tensors loaded" check -- so for those a
//! nonzero key is refused here rather than honoured. Until 2026-09-11
//! frink refused it for EVERY architecture (`capability::
//! unsupported_feature_keys`), which was safe and over-broad: every
//! real MiMo-V2 (`mimo.py:167`, three blocks), Step-3.5 (`step3.py:223`,
//! three), K-EXAONE (`exaone.py:146`, one), GLM-4.5/4.6 (`glm.py:106`,
//! one) and DeepSeek-V3 (`deepseek.py:498`, one) export carries the key
//! with a nonzero value.
//!
//! **The order of the two reads matters, and is copied.** `exaone4.cpp:4`
//! tests `n_layer() == 64` BEFORE `:18` reads the key, and
//! `mimo2.cpp:12` / `step35.cpp:26` size the sliding-window array by
//! `n_layer()` before `:19` / `:32` read it, so both see `n_layer_all`.
//! The loader therefore feeds `block_count` -- [`TrunkLayers::
//! block_count`], not [`TrunkLayers::n_layers`] -- to
//! `capability::swa_disabled_by_arch` and to every per-layer array
//! length check, and only the trunk to the layer loop. A hypothetical
//! EXAONE-4.5 with 64 trunk layers and one MTP block gets NO window in
//! llama.cpp, and gets none here.
//!
//! This module decides the trunk once ([`trunk_layers`]) and marks the
//! skipped blocks' tensors as deliberately unread
//! ([`note_mtp_blocks_skipped`]) so that `loader::
//! assert_every_tensor_consumed` -- which exists precisely to catch a
//! tensor nobody read -- can tell "skipped on purpose, as llama.cpp
//! does" from "missing from the graph". The four dedicated engines
//! (`glm52_gguf_loader`, `mla_gguf_loader`, `hybrid_gguf_loader`,
//! `gemma4_gguf_loader`) take their layer count from the same function:
//! three of the four own architectures in [`NEXTN_READERS`] and read
//! `block_count` verbatim before this, so a real GLM-4.5 or DeepSeek-V3
//! file would have run its MTP block as one more decoder layer with the
//! `nextn.*` tensors silently unread.

use frink_gguf::{ShardedGguf, TensorSource};

use crate::loader::LoadError;

/// Every graph whose `load_arch_hparams` reads
/// `LLM_KV_NEXTN_PREDICT_LAYERS`, with the file. Measured over all 155
/// `src/models/*.cpp`; `llama-arch.cpp` and `llama-model-saver.cpp` are
/// the only other hits and neither is a graph.
pub const NEXTN_READERS: &[(&str, &str)] = &[
    ("bailingmoe2", "src/models/bailingmoe2.cpp"),
    ("cohere2moe", "src/models/cohere2moe.cpp"),
    ("deepseek2", "src/models/deepseek2.cpp"),
    ("deepseek32", "src/models/deepseek32.cpp"),
    ("deepseek4", "src/models/deepseek4.cpp"),
    ("exaone-moe", "src/models/exaone-moe.cpp:23"),
    ("gemma4-assistant", "src/models/gemma4-assistant.cpp"),
    ("glm-dsa", "src/models/glm-dsa.cpp"),
    ("glm4moe", "src/models/glm4-moe.cpp"),
    ("glm4", "src/models/glm4.cpp"),
    ("hy-v3", "src/models/hy-v3.cpp"),
    ("mimo2", "src/models/mimo2.cpp:19"),
    ("qwen35", "src/models/qwen35.cpp"),
    ("qwen35moe", "src/models/qwen35moe.cpp"),
    ("qwen3next", "src/models/qwen3next.cpp"),
    ("step35", "src/models/step35.cpp:32"),
];

/// Does `arch`'s graph subtract `nextn_predict_layers` from its layer
/// count?
pub fn reads_nextn(arch: &str) -> bool {
    crate::models::ArchFacts::of(arch).topology.reads_nextn
}

/// [`reads_nextn`] from [`NEXTN_READERS`] alone (`exaone4` is in
/// `crate::models::exaone4`).
pub(crate) fn reads_nextn_from_table(arch: &str) -> bool {
    NEXTN_READERS.iter().any(|(a, _)| *a == arch)
}

/// How many of a file's blocks are decoder layers.
///
/// `block_count` is llama.cpp's `n_layer_all`; `n_layers` its
/// `n_layer()`; `n_mtp_blocks` its `n_layer_nextn`. The three are
/// carried together because two of them are what every array-length
/// check and every layer loop must NOT be handed interchangeably.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrunkLayers {
    /// `{arch}.block_count`, every block in the file.
    pub block_count: usize,
    /// The blocks that are decoder layers: `block_count - n_mtp_blocks`.
    pub n_layers: usize,
    /// `{arch}.nextn_predict_layers` where the architecture reads it,
    /// else 0.
    pub n_mtp_blocks: usize,
}

impl TrunkLayers {
    /// The whole file is trunk.
    pub fn all(block_count: usize) -> Self {
        Self {
            block_count,
            n_layers: block_count,
            n_mtp_blocks: 0,
        }
    }
}

/// Decides the trunk for `arch` from `{arch}.nextn_predict_layers`.
///
/// Refuses a nonzero count on an architecture whose graph does not
/// read the key (module doc), and a count that is not below
/// `block_count` (`mimo2.cpp:20`, `step35.cpp:33`, `exaone4.cpp:20`:
/// `GGML_ASSERT(n_layer_nextn < n_layer_all)`).
pub fn trunk_layers(
    file: &impl TensorSource,
    arch: &str,
    block_count: usize,
) -> Result<TrunkLayers, LoadError> {
    let key = format!("{arch}.nextn_predict_layers");
    let n_mtp_blocks = match file.metadata(&key) {
        None => 0,
        Some(v) => v.as_u64().ok_or_else(|| {
            LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("{key} is not an unsigned integer: {v:?}"),
            )
        })? as usize,
    };
    if n_mtp_blocks == 0 {
        return Ok(TrunkLayers::all(block_count));
    }
    if !reads_nextn(arch) {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "NextN/MTP prediction layers are counted in block_count and llama.cpp skips \
                 them (n_layer = n_layer_all - n_layer_nextn) only for the graphs that read \
                 the key; `{arch}` is not one of them (metadata {key}={n_mtp_blocks}), so \
                 upstream would run every block and fail on the unread `nextn.*` tensors, \
                 and the generic decoder would run the speculative head as ordinary \
                 decoder layers"
            ),
        ));
    }
    if n_mtp_blocks >= block_count {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "{key}={n_mtp_blocks} is not below block_count={block_count}; llama.cpp \
                 asserts `n_layer_nextn < n_layer_all` and aborts on this file"
            ),
        ));
    }
    Ok(TrunkLayers {
        block_count,
        n_layers: block_count - n_mtp_blocks,
        n_mtp_blocks,
    })
}

/// Is `name` a tensor of one of the skipped blocks -- `blk.N.*` with
/// `n_layers <= N < block_count`?
///
/// Exactly that range: a `blk.N` at or past `block_count` is not a
/// block llama.cpp would have created either, and stays unread for the
/// consumption gate to report.
pub fn is_mtp_block_tensor(name: &str, trunk: &TrunkLayers) -> bool {
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((idx, _)) = rest.split_once('.') else {
        return false;
    };
    idx.parse::<usize>()
        .is_ok_and(|n| n >= trunk.n_layers && n < trunk.block_count)
}

/// Marks every tensor of the skipped blocks as deliberately unread, as
/// llama.cpp's `TENSOR_SKIP` does, and returns how many it marked.
pub fn note_mtp_blocks_skipped(file: &ShardedGguf, trunk: &TrunkLayers) -> usize {
    if trunk.n_mtp_blocks == 0 {
        return 0;
    }
    let mut marked = 0;
    for (_, info) in file.tensors() {
        if is_mtp_block_tensor(&info.name, trunk) {
            file.note_consumed(&info.name);
            marked += 1;
        }
    }
    marked
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_gguf::GgufValue;
    use std::sync::Arc;

    struct Meta(Vec<(String, GgufValue)>);
    impl TensorSource for Meta {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        fn find_tensor(&self, _: &str) -> Option<&frink_gguf::TensorInfo> {
            None
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], frink_gguf::GgufError> {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<(Arc<frink_gguf::MmapHandle>, std::ops::Range<usize>), frink_gguf::GgufError>
        {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
    }

    fn nextn(arch: &str, n: Option<u32>) -> Meta {
        Meta(
            n.into_iter()
                .map(|n| (format!("{arch}.nextn_predict_layers"), GgufValue::U32(n)))
                .collect(),
        )
    }

    /// A reader subtracts; the key absent or zero is the whole file.
    #[test]
    fn a_reader_subtracts_the_blocks_from_its_layer_count() {
        assert_eq!(
            trunk_layers(&nextn("exaone-moe", Some(1)), "exaone-moe", 5).unwrap(),
            TrunkLayers {
                block_count: 5,
                n_layers: 4,
                n_mtp_blocks: 1
            }
        );
        assert_eq!(
            trunk_layers(&nextn("mimo2", Some(3)), "mimo2", 51).unwrap(),
            TrunkLayers {
                block_count: 51,
                n_layers: 48,
                n_mtp_blocks: 3
            }
        );
        for file in [nextn("exaone-moe", None), nextn("exaone-moe", Some(0))] {
            assert_eq!(
                trunk_layers(&file, "exaone-moe", 4).unwrap(),
                TrunkLayers::all(4)
            );
        }
    }

    /// The key on a graph that never reads it: refused, with the
    /// reason, and the value in it. `grok` is the row the old
    /// `unsupported_feature_keys` test used, kept as the example.
    #[test]
    fn a_non_reader_with_a_nonzero_count_is_refused_and_zero_is_not() {
        match trunk_layers(&nextn("grok", Some(1)), "grok", 4) {
            Err(LoadError::UnsupportedFeature(arch, msg)) => {
                assert_eq!(arch, "grok");
                assert!(msg.contains("NextN/MTP"), "{msg}");
                assert!(msg.contains("grok.nextn_predict_layers=1"), "{msg}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(
            trunk_layers(&nextn("grok", Some(0)), "grok", 4).unwrap(),
            TrunkLayers::all(4)
        );
    }

    /// llama.cpp's own assert, as a refusal rather than an abort.
    #[test]
    fn a_count_not_below_block_count_is_refused() {
        for n in [4, 5] {
            assert!(matches!(
                trunk_layers(&nextn("mimo2", Some(n)), "mimo2", 4),
                Err(LoadError::UnsupportedFeature(a, m)) if a == "mimo2" && m.contains("n_layer_nextn < n_layer_all")
            ));
        }
    }

    /// The census is what the predicate answers, and the three
    /// architectures this task was aimed at are in it.
    #[test]
    fn the_reader_census_is_the_predicate() {
        for (arch, _) in NEXTN_READERS {
            assert!(reads_nextn(arch), "{arch}");
        }
        for arch in [
            "mimo2",
            "step35",
            "exaone4",
            "exaone-moe",
            "glm4moe",
            "deepseek2",
        ] {
            assert!(reads_nextn(arch), "{arch}");
        }
        for arch in ["llama", "grok", "gemma4", "qwen3moe", "mistral4"] {
            assert!(!reads_nextn(arch), "{arch}");
        }
        // The census spans the table and the registry.
        let moved = crate::models::REGISTRY
            .iter()
            .filter(|d| d.facts.topology.reads_nextn)
            .map(|d| d.names.len())
            .sum::<usize>();
        assert_eq!(NEXTN_READERS.len() + moved, 17, "measured on 2026-09-11");
    }

    /// Exactly the skipped range, and only `blk.` names.
    #[test]
    fn only_tensors_of_the_skipped_blocks_are_mtp_tensors() {
        let trunk = TrunkLayers {
            block_count: 6,
            n_layers: 4,
            n_mtp_blocks: 2,
        };
        assert!(!is_mtp_block_tensor("blk.3.attn_norm.weight", &trunk));
        assert!(is_mtp_block_tensor("blk.4.attn_norm.weight", &trunk));
        assert!(is_mtp_block_tensor("blk.4.nextn.eh_proj.weight", &trunk));
        assert!(is_mtp_block_tensor("blk.5.ffn_down_exps.weight", &trunk));
        // Past block_count is not a block llama.cpp would create either.
        assert!(!is_mtp_block_tensor("blk.6.attn_norm.weight", &trunk));
        assert!(!is_mtp_block_tensor("output.weight", &trunk));
        assert!(!is_mtp_block_tensor("blk.x.attn_norm.weight", &trunk));
        assert!(!is_mtp_block_tensor("blk.4", &trunk));
    }
}
