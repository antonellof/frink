//! **WHICH GRAPHS USE ALiBi**, and where each one gets its
//! `f_max_alibi_bias` from. The arithmetic is `frink_core::alibi`.
//!
//! # What it is
//!
//! `llama-model.cpp:1240-1242` sets `hparams.use_alibi` whenever
//! `f_max_alibi_bias > 0` after `load_arch_hparams`, and from then on
//! the KV mask carries `-|p_key - p_query|` (`llama-kv-cache.cpp:
//! 1673-1676`) and every `build_attn` hands the bias to
//! `ggml_soft_max_ext` (`llama-graph.cpp:2599`; the flash path takes it
//! at `:2539`). None of these graphs calls `ggml_rope`: ALiBi IS their
//! position encoding, and `crate::rope_layers::RopeLayers::Never` is the
//! other half of the same fact.
//!
//! # Reach -- MEASURED
//!
//! `grep -n f_max_alibi_bias src/models/*.cpp` over all 155 graphs
//! (2026-09-14) is seven files, in three spellings:
//!
//! | arch | where the bias comes from | line |
//! |---|---|---|
//! | `bloom` | the literal `8.0f`, no key ("TODO: become GGUF KV parameter") | `bloom.cpp:18` |
//! | `refact` | the literal `8.0f`, no key | `refact.cpp:12` |
//! | `baichuan` | the literal `8.0f`, ONLY at 40 layers (Baichuan-13B); the 7B rotates | `baichuan.cpp:11-14` |
//! | `mpt` | `attention.max_alibi_bias`, optional, default 0 | `mpt.cpp:6` |
//! | `jais` | `attention.max_alibi_bias`, optional, default 0 | `jais.cpp:5` |
//! | `jina-bert-v2` | the literal | encoder engine |
//! | `minimax-m3` | its own hparams | its own engine |
//!
//! So the table has three rules and five generic-path rows. An `mpt`
//! or `jais` file whose key is absent or zero has NO position encoding
//! at all upstream (the mask is then `0 / -inf`, `slope = 1`), which is
//! what this table answers too; every real export of either writes the
//! key (`conversion/mpt.py`, `conversion/jais.py`).

/// How a graph arrives at its `f_max_alibi_bias`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxAlibiBias {
    /// A literal in `load_arch_hparams`, no key read.
    Literal,
    /// The literal, but only when the trunk has this many layers.
    LiteralAtLayers(usize),
    /// `{arch}.attention.max_alibi_bias`, `required = false`, default 0.
    Key,
}

/// The literal every no-key graph uses.
pub const LLAMA_CPP_ALIBI_LITERAL: f32 = 8.0;

/// The `f_max_alibi_bias` llama.cpp would hold for `arch` with
/// `n_layers` trunk layers and `key` the file's
/// `attention.max_alibi_bias`, or `None` for no ALiBi -- which is every
/// architecture not in the table, `baichuan` at any other depth, and an
/// `mpt` / `jais` file whose key is absent or non-positive.
pub fn max_alibi_bias(arch: &str, n_layers: usize, key: Option<f32>) -> Option<f32> {
    max_alibi_bias_for(
        crate::models::ArchFacts::of(arch).topology.alibi,
        n_layers,
        key,
    )
}

/// [`max_alibi_bias`] given the row (`ArchFacts`). The five ALiBi graphs
/// (`bloom`, `refact`, `baichuan`, `mpt`, `jais`) state their rule in
/// `crate::models`.
pub fn max_alibi_bias_for(
    rule: Option<MaxAlibiBias>,
    n_layers: usize,
    key: Option<f32>,
) -> Option<f32> {
    let rule = rule?;
    let bias = match rule {
        MaxAlibiBias::Literal => LLAMA_CPP_ALIBI_LITERAL,
        MaxAlibiBias::LiteralAtLayers(n) => {
            if n_layers == n {
                LLAMA_CPP_ALIBI_LITERAL
            } else {
                return None;
            }
        }
        MaxAlibiBias::Key => key.unwrap_or(0.0),
    };
    (bias > 0.0).then_some(bias)
}

/// Whether `arch`'s graph is one that positions by ALiBi (and so never
/// rotates), at `n_layers`: the table's answer with the literal, for
/// the rope-layers rule that has no key to consult.
pub fn positions_by_alibi(arch: &str, n_layers: usize) -> bool {
    positions_by_alibi_for(crate::models::ArchFacts::of(arch).topology.alibi, n_layers)
}

/// [`positions_by_alibi`] given the row (`ArchFacts`).
pub fn positions_by_alibi_for(rule: Option<MaxAlibiBias>, n_layers: usize) -> bool {
    rule.is_some_and(|rule| !matches!(rule, MaxAlibiBias::LiteralAtLayers(l) if l != n_layers))
}

/// The per-layer RoPE rule an ALiBi graph's definition sets: no layer
/// rotates where the bias positions the tokens, every layer where it
/// does not (Baichuan-7B). One row decides both, so the bias and the
/// absence of rotation cannot disagree about the layer count.
pub fn alibi_rope_layers(rule: MaxAlibiBias, n_layers: usize) -> crate::rope_layers::RopeLayers {
    if positions_by_alibi_for(Some(rule), n_layers) {
        crate::rope_layers::RopeLayers::Never
    } else {
        crate::rope_layers::RopeLayers::All
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row is a registered architecture on the generic path.
    #[test]
    fn every_row_is_a_registered_generic_architecture() {
        // Every row, over the catalog: they live in `crate::models`.
        let mut rows: Vec<&str> = crate::capability::architecture_catalog()
            .iter()
            .map(|p| p.gguf_name)
            .filter(|n| crate::models::ArchFacts::of(n).topology.alibi.is_some())
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, ["baichuan", "bloom", "jais", "mpt", "refact"]);
        for arch in rows {
            let lines = crate::models::find(arch).map_or("", |d| d.source);
            assert!(positions_by_alibi(arch, 40), "`{arch}` ({lines})");
            assert!(
                matches!(
                    crate::capability::resolve_architecture(arch),
                    Some(crate::capability::ArchPath::GenericGqa { .. })
                ),
                "`{arch}` ({lines}) is not on the generic path"
            );
        }
    }

    /// The three rules, on the rows that carry them.
    #[test]
    fn the_three_rules() {
        assert_eq!(max_alibi_bias("bloom", 30, None), Some(8.0));
        assert_eq!(
            max_alibi_bias("bloom", 30, Some(2.0)),
            Some(8.0),
            "no key is read"
        );
        assert_eq!(max_alibi_bias("refact", 32, None), Some(8.0));
        assert_eq!(max_alibi_bias("baichuan", 40, None), Some(8.0));
        assert_eq!(
            max_alibi_bias("baichuan", 32, None),
            None,
            "Baichuan-7B rotates"
        );
        assert_eq!(max_alibi_bias("mpt", 32, Some(8.0)), Some(8.0));
        assert_eq!(
            max_alibi_bias("mpt", 32, None),
            None,
            "no key: no position at all"
        );
        assert_eq!(max_alibi_bias("jais", 24, Some(0.0)), None);
        assert_eq!(
            max_alibi_bias("llama", 32, Some(8.0)),
            None,
            "the key is dead elsewhere"
        );
        assert!(positions_by_alibi("baichuan", 40));
        assert!(!positions_by_alibi("baichuan", 32));
        assert!(positions_by_alibi("mpt", 32));
        assert!(!positions_by_alibi("llama", 32));
    }
}
