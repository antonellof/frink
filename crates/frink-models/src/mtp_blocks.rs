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
//! llama.cpp at the pin (`604cc430`) never runs them as layers:
//!
//! - the COMMON loader reads `block_count` into `n_layer_all` and
//!   `nextn_predict_layers` into `n_layer_nextn` for EVERY architecture
//!   (`llama-model.cpp:1409-1412`, with `GGML_ASSERT(n_layer_nextn <=
//!   n_layer_all)`), before any `load_arch_hparams` (`:1571`);
//! - `llama-hparams.cpp:348-350` defines `n_layer()` as `n_layer_all -
//!   n_layer_nextn`, so every per-graph `n_layer()` test already sees
//!   the trunk (`exaone4.cpp:4`'s `== 64` among them);
//! - `llama-graph.cpp:1555` builds every graph with `n_layer =
//!   hparams.n_layer()`, so `mimo2.cpp:100`, `step35.cpp:192`,
//!   `glm4-moe.cpp:312`, `deepseek2.cpp:459` loop over the trunk only.
//!
//! **Which architectures load such a file is decided by the TENSOR
//! loader.** `LLAMA_LOAD_LOCALS` binds `n_layer` to the trunk
//! (`llama-model.h:1001`), so a loader that loops `0..n_layer` never
//! creates the trailing blocks and libllama fails on them (`done_getting_
//! tensors: wrong number of tensors`, `llama-model-loader.cpp:1405-1408`;
//! measured on `llama_nextn_tiny.gguf`). The graphs in
//! [`NEXTN_READERS`] create them, either looping `0..n_layer_all` or
//! `n_layer..n_layer_all`, with `TENSOR_SKIP` unless the context was
//! opened as an MTP draft (`nextn_flags`, `llama-model.cpp:3931-3949`),
//! so the bytes are neither read nor missed. That is the census: frink
//! subtracts for those, and refuses a nonzero key on every other
//! architecture, where upstream does not load the file either.
//!
//! Until 2026-10-29 the census was the graphs whose `load_arch_hparams`
//! read the key themselves (measured 2026-09-11 at the `5b59b83` pin).
//! Upstream moved the read into the common loader (commit 9d81721,
//! "load hparams.n_layer_nextn before n_layer() calls"); re-measured
//! at `604cc430` (#441), the loaders that create the blocks are the
//! old sixteen plus `exaone4` (in `crate::models::exaone4`) and eight
//! more: `bailingmoe3`, `dots3note`, `glm5-next` and `qwen4exp`
//! (refused by name for other things), `clef` (deferred; `qwen35`'s
//! loader), `mistral4` (`deepseek2`'s loader, `models.h:1624-1631`),
//! and the two that frink serves, `mellum` and `nemotron_h_moe`, whose
//! MTP files frink refused before. `nemotron_h` shares
//! `nemotron-h.cpp`'s loader but stays out: its MTP loop throws on a
//! block with no expert width (`:157-160`), which a dense file has.
//! `deepseek4` (`deepseek4.cpp:12-18`) resets the count to zero when
//! its first `nextn.eh_proj` is absent; it is refused by name.
//!
//! **Array lengths.** `get_key_or_arr` arrays that `load_arch_hparams`
//! sizes with `hparams.n_layer_all` (`mimo2.cpp:8`, `step35.cpp:12,28`,
//! `cohere2moe.cpp:16`) are one entry per BLOCK, so the loader feeds
//! [`TrunkLayers::block_count`] to every per-layer array length check
//! and only the trunk to the layer loop and to the layer-count gates.
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

/// Every graph whose tensor loader creates the blocks past `n_layer()`,
/// with the loop. Measured over all `src/models/*.cpp` at `604cc430`
/// (module doc); `exaone4` is in `crate::models::exaone4`.
pub const NEXTN_READERS: &[(&str, &str)] = &[
    ("bailingmoe2", "src/models/bailingmoe2.cpp:35"),
    ("bailingmoe3", "src/models/bailingmoe3.cpp:122"),
    ("clef", "src/models/clef.cpp:27 (qwen35.cpp:125)"),
    ("deepseek2", "src/models/deepseek2.cpp:88"),
    ("deepseek32", "src/models/deepseek32.cpp:81"),
    ("deepseek4", "src/models/deepseek4.cpp:101"),
    ("dots3note", "src/models/dots3note.cpp:70"),
    ("exaone-moe", "src/models/exaone-moe.cpp:47"),
    ("gemma4-assistant", "src/models/gemma4-assistant.cpp:47"),
    ("glm-dsa", "src/models/glm-dsa.cpp:109"),
    ("glm4moe", "src/models/glm4-moe.cpp:49"),
    ("glm4", "src/models/glm4.cpp:28"),
    ("glm5-next", "src/models/glm5-next.cpp:76"),
    ("hy-v3", "src/models/hy-v3.cpp:74"),
    ("mimo2", "src/models/mimo2.cpp:38"),
    ("mistral4", "src/models/models.h:1624 (deepseek2.cpp:88)"),
    (
        "nemotron_h_moe",
        "src/models/models.h:1770 (nemotron-h.cpp:149)",
    ),
    ("qwen35", "src/models/qwen35.cpp:125"),
    ("qwen35moe", "src/models/qwen35moe.cpp:144"),
    ("qwen3next", "src/models/qwen3next.cpp:126"),
    ("qwen4exp", "src/models/qwen4exp.cpp:310"),
    ("step35", "src/models/step35.cpp:170"),
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
    /// `{arch}.nextn_predict_layers` where the architecture's loader
    /// creates the blocks, else 0.
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
/// Refuses a nonzero count on an architecture whose tensor loader does
/// not create the trailing blocks (module doc), and a count that is not
/// below `block_count`: `llama-model.cpp:1412` asserts only `<=`, and a
/// file that is all MTP blocks has no trunk for this decoder to run.
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
                "NextN/MTP prediction layers are counted in block_count; llama.cpp \
                 subtracts them for every architecture (n_layer = n_layer_all - n_layer_nextn) \
                 but `{arch}`'s tensor loader creates only the trunk (metadata \
                 {key}={n_mtp_blocks}), so upstream fails on the trailing blocks' tensors, and \
                 the generic decoder would run the speculative head as ordinary decoder layers"
            ),
        ));
    }
    if n_mtp_blocks >= block_count {
        return Err(LoadError::UnsupportedFeature(
            arch.to_string(),
            format!(
                "{key}={n_mtp_blocks} is not below block_count={block_count}: the file has no \
                 trunk layer (n_layer_nextn < n_layer_all)"
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

    /// A file with no trunk layer is refused.
    #[test]
    fn a_count_not_below_block_count_is_refused() {
        for n in [4, 5] {
            assert!(matches!(
                trunk_layers(&nextn("mimo2", Some(n)), "mimo2", 4),
                Err(LoadError::UnsupportedFeature(a, m)) if a == "mimo2" && m.contains("no trunk layer")
            ));
        }
    }

    /// The census is what the predicate answers, the loaders this task
    /// was aimed at are in it, and the ones that create only the trunk
    /// are not.
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
            "mistral4",
            "mellum",
            "nemotron_h_moe",
        ] {
            assert!(reads_nextn(arch), "{arch}");
        }
        // `nemotron_h`: the shared loader throws on a dense MTP block.
        // `deepseek41`: its own tensor loader loops the trunk only.
        for arch in [
            "llama",
            "grok",
            "gemma4",
            "qwen3moe",
            "nemotron_h",
            "deepseek41",
        ] {
            assert!(!reads_nextn(arch), "{arch}");
        }
        // The census spans the table and the registry.
        let moved = crate::models::REGISTRY
            .iter()
            .filter(|d| d.facts.topology.reads_nextn)
            .map(|d| d.names.len())
            .sum::<usize>();
        assert_eq!(NEXTN_READERS.len() + moved, 25, "measured at 604cc430");
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
