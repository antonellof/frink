//! A key llama.cpp's `load_arch_hparams` reads REQUIRED (`get_key` with
//! no `false`) is refused by name when the file lacks it, never
//! defaulted (ADR 0002, #620).
//!
//! Each fixture is its architecture's synthetic file with that one key
//! left out; libllama refuses each of them too (`tools/oracle/
//! fixtures.tsv` holds them `both-fail`).

mod common;
use common::graph_fixture_path;
use frink_models::{LoadError, ModelConfig};

#[test]
fn a_file_without_a_required_key_is_refused_by_name() {
    for (fixture, key, line) in [
        ("afmoe_noshared", "afmoe.expert_shared_count", "afmoe.cpp:7"),
        (
            "exaone_moe_nogating",
            "exaone-moe.expert_gating_func",
            "exaone-moe.cpp:16",
        ),
        (
            "laguna_nodense",
            "laguna.leading_dense_block_count",
            "laguna.cpp:14",
        ),
        (
            "muse_glimmer_nowindow",
            "muse-glimmer.attention.sliding_window",
            "muse-glimmer.cpp:5",
        ),
    ] {
        let file = frink_gguf::GgufFile::open(graph_fixture_path(fixture)).expect("opens");
        match ModelConfig::from_gguf(&file) {
            Err(LoadError::UnsupportedFeature(_, msg)) => {
                assert!(msg.contains(key), "{fixture}: {msg}");
                assert!(msg.contains(line), "{fixture}: {msg}");
            }
            Err(other) => panic!("{fixture}: expected `{key}` refused by name, got {other:?}"),
            Ok(_) => panic!("{fixture}: loaded without `{key}`"),
        }
    }
}
