//! Integration test: loads `tests/fixtures/frink_real_test.gguf` (a
//! real, genuinely Q8_0-quantized, generated on-disk GGUF file) through
//! the real `Decoder::from_gguf`
//! loader and runs a real forward pass, then checks the output against
//! a golden logits array computed by an entirely independent NumPy
//! reference implementation that reads
//! the same file's bytes directly and reimplements RMSNorm, RoPE, GQA
//! attention, and the SwiGLU FFN from scratch.
//!
//! This is the strongest correctness test in this repository: it is not
//! internally self-consistent (Rust checking Rust), it is cross-checked
//! against a second, independent implementation reading the same bytes.

mod common;

use frink_core::cache::KvCache;
use frink_models::{config::test_dense_fixture, Decoder, ModelConfig};

const GOLDEN_LOGITS: [f32; 32] = [
    0.843769, -0.557965, -0.344701, 0.639505, 0.682568, -0.790475, 0.019445, -0.732593, 0.279821,
    0.159313, -0.273534, 0.342074, 0.221641, -0.076465, 0.167651, 0.216972, -0.244357, -0.994642,
    -0.134574, 0.074153, -0.628791, -0.351837, -0.218014, -0.405603, 0.393343, -0.397728, 0.183012,
    -0.687949, 0.195757, -0.055774, 0.066989, 0.371296,
];

#[test]
fn real_gguf_forward_pass_matches_independent_python_reference() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let cfg = test_dense_fixture();
    let decoder = Decoder::from_gguf(fixture_path, cfg).expect("real GGUF file must load cleanly");

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let logits = decoder.forward_token(3, 0, &mut caches);

    assert_eq!(logits.len(), GOLDEN_LOGITS.len());
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "logit[{i}]: got {got}, expected {expected} (independent Python reference)"
        );
    }
}

#[test]
fn real_gguf_loader_uses_zero_copy_mmap_for_quantized_weights() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let cfg = test_dense_fixture();
    let decoder = Decoder::from_gguf(fixture_path, cfg).expect("real GGUF file must load cleanly");

    match &decoder.layers[0].attn.q_proj {
        frink_core::weight_matrix::WeightMatrix::Quantized { data, .. } => {
            assert!(
                data.is_mapped(),
                "expected the loader to take the zero-copy mmap path for a Q8_0 tensor, not copy it into an owned Vec<u8>"
            );
        }
        _ => panic!("blk.0.attn_q.weight is Q8_0 in the test fixture; got F32 instead"),
    }
}

#[test]
fn real_gguf_file_has_expected_tensor_count_and_dtypes() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();
    // 1 embedding + 2 layers * (1 attn_norm + 4 attn + 1 ffn_norm + 3 ffn) + output_norm + output
    assert_eq!(file.tensors.len(), 1 + 2 * 9 + 2);
    let q8_0_count = file
        .tensors
        .iter()
        .filter(|t| t.dtype == frink_gguf::GgmlType::Q8_0)
        .count();
    // per layer: q, k, v, o, gate, up, down = 7 quantized tensors
    assert_eq!(q8_0_count, 2 * 7);
}

/// Golden logits for the multi-expert MoE fixture, computed by an
/// independent NumPy reference implementation that reads
/// `frink_real_moe_test.gguf`'s bytes directly and reimplements
/// top-2-of-4 MoE routing, the shared expert, and everything else from
/// scratch. This is the cross-validation that closes the "multi-expert
/// loading is implemented but never verified against a real file" gap.
const GOLDEN_MOE_LOGITS: [f32; 32] = [
    -0.187686, -0.540774, -0.148961, 0.181181, -0.558467, -0.187736, 0.011305, -0.110416,
    -0.059392, -0.413899, 0.176243, 0.577102, -0.150234, -0.316515, 0.320917, -0.102361, 0.638307,
    0.120588, 0.167106, 0.377888, -0.233538, 0.049238, -0.154894, -0.708512, 0.089974, -0.632005,
    0.236564, -0.267068, 0.302669, -0.121790, -0.070083, -0.086464,
];

#[test]
fn real_multi_expert_moe_gguf_forward_pass_matches_independent_python_reference() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_moe_test.gguf"
    );
    let cfg = frink_models::config::test_moe_fixture();
    let decoder = Decoder::from_gguf(fixture_path, cfg)
        .expect("real multi-expert MoE GGUF file must load cleanly");

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let logits = decoder.forward_token(3, 0, &mut caches);

    assert_eq!(logits.len(), GOLDEN_MOE_LOGITS.len());
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_MOE_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "logit[{i}]: got {got}, expected {expected} (independent Python MoE reference)"
        );
    }
}

#[test]
fn real_multi_expert_moe_gguf_has_expected_packed_expert_tensors() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_moe_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();

    let gate_exps = file.find_tensor("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(
        gate_exps.shape,
        vec![32, 32, 4],
        "packed 3D expert tensor: real raw GGUF ne[] order is fastest-varying-first, \
         i.e. [hidden, ffn_dim, n_experts] -- the reverse of the semantic \
         [n_experts, ffn_dim, hidden]"
    );
    assert_eq!(gate_exps.dtype, frink_gguf::GgmlType::Q8_0);

    let router = file.find_tensor("blk.0.ffn_gate_inp.weight").unwrap();
    assert_eq!(
        router.shape,
        vec![32, 4],
        "router: real raw order is [hidden, n_experts]"
    );

    assert!(
        file.find_tensor("blk.0.ffn_gate_shexp.weight").is_some(),
        "shared expert tensor must be present"
    );
}

/// Golden logits for the mixed dense+MoE fixture (layer 0 dense, layers
/// 1-2 genuine top-1-of-3 MoE with a shared expert), computed by an
/// independent NumPy reference implementation that reads
/// `frink_real_mixed_test.gguf`'s bytes directly and reimplements the
/// per-layer dense/MoE branching from scratch. This is the
/// cross-validation for `ModelConfig::layer_is_dense`, previously
/// documented (from reading ik_llama.cpp's "leading dense layers"
/// convention) but not implemented or tested until this fixture.
const GOLDEN_MIXED_LOGITS: [f32; 32] = [
    0.050355, -0.067514, 0.659481, 0.278982, -0.744414, 0.190165, -0.705883, 0.387951, -0.511973,
    0.089168, -0.515594, -1.009972, 0.191165, -1.081907, 1.047037, -0.651277, 0.433748, -0.844842,
    -0.262953, 0.191121, -0.497102, -0.095453, -0.107873, -0.578634, 0.772580, 0.135300, -0.687753,
    0.297360, 0.499715, -0.968871, -0.515122, 0.553081,
];

#[test]
fn real_mixed_dense_and_moe_gguf_matches_independent_python_reference() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_mixed_test.gguf"
    );
    let cfg = frink_models::config::test_mixed_fixture();
    let decoder = Decoder::from_gguf(fixture_path, cfg)
        .expect("real mixed dense/MoE GGUF file must load cleanly");

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let logits = decoder.forward_token(3, 0, &mut caches);

    assert_eq!(logits.len(), GOLDEN_MIXED_LOGITS.len());
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_MIXED_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "logit[{i}]: got {got}, expected {expected} (independent Python mixed-topology reference)"
        );
    }
}

#[test]
fn real_mixed_gguf_first_layer_is_dense_and_later_layers_are_moe() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_mixed_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();

    // Layer 0: plain dense tensor names, no router, no packed experts.
    assert!(file.find_tensor("blk.0.ffn_gate.weight").is_some());
    assert!(file.find_tensor("blk.0.ffn_gate_exps.weight").is_none());
    assert!(file.find_tensor("blk.0.ffn_gate_inp.weight").is_none());

    // Layer 1: real packed 3D MoE tensors, no plain dense names.
    let gate_exps = file.find_tensor("blk.1.ffn_gate_exps.weight").unwrap();
    // Real raw GGUF ne[] order is fastest-varying-first: [hidden,
    // ffn_dim, n_experts], the reverse of the semantic [n_experts,
    // ffn_dim, hidden].
    assert_eq!(gate_exps.shape, vec![32, 32, 3]);
    assert!(file.find_tensor("blk.1.ffn_gate.weight").is_none());

    let decoder = Decoder::from_gguf(fixture_path, frink_models::config::test_mixed_fixture())
        .expect("must load cleanly");
    assert_eq!(
        decoder.layers[0].moe.n_experts(),
        1,
        "layer 0 must load as a single dense expert"
    );
    assert_eq!(
        decoder.layers[0].moe.shared_experts.len(),
        0,
        "dense layer must have no shared expert"
    );
    assert_eq!(
        decoder.layers[1].moe.n_experts(),
        3,
        "layer 1 must load all 3 real MoE experts"
    );
    assert_eq!(
        decoder.layers[1].moe.shared_experts.len(),
        1,
        "MoE layer must load its shared expert"
    );
}

/// `ModelConfig::from_gguf` is what lets `frink-server`
/// load an arbitrary checkpoint's architecture shape from
/// its own metadata rather than requiring a hand-tuned preset that
/// already happens to match. These three tests prove it derives the
/// *exact same* config the hand-written test fixtures use, for all
/// three fixture shapes (dense, multi-expert MoE, mixed dense+MoE), and
/// that a `Decoder` built from the derived config produces the same
/// golden logits as one built from the hand-written config -- so this
/// isn't just "the struct fields happen to match," it's "the derived
/// config is actually usable to run a real forward pass correctly."
#[test]
fn model_config_from_gguf_matches_hand_written_dense_fixture_and_produces_identical_logits() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();
    let derived = ModelConfig::from_gguf(&file).expect("hparams must be derivable from this file");
    let hand_written = test_dense_fixture();

    assert_eq!(derived.n_layers, hand_written.n_layers);
    assert_eq!(derived.hidden_dim, hand_written.hidden_dim);
    assert_eq!(derived.n_heads, hand_written.n_heads);
    assert_eq!(derived.n_kv_heads, hand_written.n_kv_heads);
    assert_eq!(derived.head_dim, hand_written.head_dim);
    assert_eq!(derived.vocab_size, hand_written.vocab_size);
    assert_eq!(derived.rope_theta, hand_written.rope_theta);
    assert_eq!(derived.moe.n_experts, hand_written.moe.n_experts);
    assert_eq!(
        derived.n_dense_leading_layers,
        hand_written.n_dense_leading_layers
    );

    let decoder =
        Decoder::from_gguf(fixture_path, derived).expect("derived config must load cleanly");
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let logits = decoder.forward_token(3, 0, &mut caches);
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "logit[{i}] from a from_gguf-derived config: got {got}, expected {expected}"
        );
    }
}

#[test]
fn model_config_from_gguf_matches_hand_written_moe_fixture_and_produces_identical_logits() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_moe_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();
    let derived = ModelConfig::from_gguf(&file).expect("hparams must be derivable from this file");
    let hand_written = frink_models::config::test_moe_fixture();

    assert_eq!(derived.n_layers, hand_written.n_layers);
    assert_eq!(derived.moe.n_experts, hand_written.moe.n_experts);
    assert_eq!(
        derived.moe.n_experts_active,
        hand_written.moe.n_experts_active
    );
    assert_eq!(
        derived.moe.n_shared_experts,
        hand_written.moe.n_shared_experts
    );
    assert_eq!(derived.moe.expert_ffn_dim, hand_written.moe.expert_ffn_dim);

    let decoder =
        Decoder::from_gguf(fixture_path, derived).expect("derived config must load cleanly");
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let logits = decoder.forward_token(3, 0, &mut caches);
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_MOE_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "MoE logit[{i}] from a from_gguf-derived config: got {got}, expected {expected}"
        );
    }
}

#[test]
fn model_config_from_gguf_matches_hand_written_mixed_fixture_and_produces_identical_logits() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_mixed_test.gguf"
    );
    let file = frink_gguf::GgufFile::open(fixture_path).unwrap();
    let derived = ModelConfig::from_gguf(&file).expect("hparams must be derivable from this file");
    let hand_written = frink_models::config::test_mixed_fixture();

    assert_eq!(derived.n_layers, hand_written.n_layers);
    assert_eq!(
        derived.n_dense_leading_layers,
        hand_written.n_dense_leading_layers
    );
    assert_eq!(derived.moe.n_experts, hand_written.moe.n_experts);

    let decoder =
        Decoder::from_gguf(fixture_path, derived).expect("derived config must load cleanly");
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let logits = decoder.forward_token(3, 0, &mut caches);
    for (i, (got, expected)) in logits.iter().zip(GOLDEN_MIXED_LOGITS.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "mixed-topology logit[{i}] from a from_gguf-derived config: got {got}, expected {expected}"
        );
    }
}

/// The split-GGUF twin of the dense fixture (generated from the exact
/// same tensor bytes, with a metadata-only first shard) must load
/// through the same
/// `Decoder::from_gguf` entry point -- pointed at the *first shard*,
/// siblings discovered from the canonical filename -- and produce
/// bit-identical logits to the single-file fixture, and therefore also
/// match the independent Python reference.
#[test]
fn split_gguf_fixture_loads_and_matches_single_file_logits_exactly() {
    let single_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let split_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test_split-00001-of-00003.gguf"
    );

    let single = Decoder::from_gguf(single_path, test_dense_fixture())
        .expect("single-file fixture must load");
    let split = Decoder::from_gguf(split_path, test_dense_fixture())
        .expect("split fixture must load via shard discovery from shard 1");

    let mut caches_a: Vec<KvCache> = single.config.new_kv_caches();
    let mut caches_b: Vec<KvCache> = split.config.new_kv_caches();

    for (pos, tok) in [3usize, 7, 11].iter().enumerate() {
        let a = single.forward_token(*tok, pos, &mut caches_a);
        let b = split.forward_token(*tok, pos, &mut caches_b);
        assert_eq!(
            a, b,
            "split-set logits must be bit-identical to the single file at pos {pos}"
        );
    }
}

/// `ModelConfig::from_gguf` must derive the identical architecture shape
/// from the split set's metadata-only first shard as from the single
/// file (the whole point of a metadata-only shard is that the model's
/// full metadata lives there).
#[test]
fn split_gguf_metadata_only_first_shard_drives_model_config_derivation() {
    let single_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );
    let split_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test_split-00001-of-00003.gguf"
    );
    let single = frink_gguf::GgufFile::open(single_path).unwrap();
    let split = frink_gguf::ShardedGguf::open(split_path).unwrap();

    let a = ModelConfig::from_gguf(&single).expect("single-file hparams must derive");
    let b = ModelConfig::from_gguf(&split).expect("split-set hparams must derive");

    assert_eq!(a.n_layers, b.n_layers);
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.n_heads, b.n_heads);
    assert_eq!(a.n_kv_heads, b.n_kv_heads);
    assert_eq!(a.head_dim, b.head_dim);
    assert_eq!(a.vocab_size, b.vocab_size);
}

/// The unchanged-output gate for store-backed experts: the MoE fixture
/// loaded with routed experts streaming through the bounded
/// The FUSED METAL MoE path must accept streamed experts and agree with
/// resident experts, on hardware.
///
/// This is the configuration that matters most and was silently the
/// slowest: `ExpertBacking::Stored` used to fall back to the CPU inside
/// `metal_moe_topk`, so switching expert streaming on, which is how a
/// model larger than memory runs at all, turned the Metal MoE kernels
/// OFF. The comment there said all of top-k could not be held at once.
/// That was true of `with_expert`, which lends one expert at a time,
/// and not of the store: `materialize` hands back an owned view whose
/// `WeightBytes::Shared` clones the lease's `Arc`, so each view pins
/// its own entry.
///
/// A 1-byte budget is included deliberately: it forces every acquire to
/// miss, so the launches are built over buffers the store is free to
/// drop the instant a pin is released. If the pins were wrong, this is
/// the budget that would expose it.
///
/// `#[ignore]`d because it needs a real Metal device. Run with
/// `cargo test -p frink-models --features metal -- --ignored`.
#[test]
#[ignore]
#[cfg(feature = "metal")]
fn metal_fused_moe_agrees_between_streamed_and_resident_experts() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_moe_test.gguf"
    );
    let cfg = frink_models::config::test_moe_fixture;

    // SAFETY: single-threaded test process, set before any decode.
    unsafe {
        std::env::set_var("FRINK_METAL", "1");
        std::env::set_var("FRINK_METAL_ATTN", "1");
    }

    let resident = Decoder::from_gguf(fixture_path, cfg()).expect("resident load");
    for budget in [64 * 1024 * 1024u64, 1u64] {
        let stored = Decoder::from_gguf_with_expert_cache(fixture_path, cfg(), Some(budget))
            .expect("store-backed load");

        let mut caches_a: Vec<KvCache> = resident.config.new_kv_caches();
        let mut caches_b: Vec<KvCache> = stored.config.new_kv_caches();

        for (pos, tok) in [2usize, 5, 9, 1].iter().enumerate() {
            let a = resident.forward_token(*tok, pos, &mut caches_a);
            let b = stored.forward_token(*tok, pos, &mut caches_b);
            assert_eq!(
                a.len(),
                b.len(),
                "budget={budget}: logit vectors differ in length at pos {pos}"
            );
            // Both sides run the same Metal kernels over the same
            // bytes, so this is exact rather than approximate.
            assert_eq!(
                a, b,
                "budget={budget}: streamed experts must match resident on the fused \
                 Metal path at pos {pos}"
            );
        }
    }
}

/// `ExpertStore` must produce logits bit-identical to the resident
/// (mmap) path, on the LIBRARY's default kernels (int-dot off, the f32
/// dequant-dot). That arm never repacks a weight, so it cannot see a
/// repack-cache defect: it passed at a 1-byte budget for the whole life
/// of the OLMoE streaming bug for exactly that reason.
/// `tests/streaming_int_dot.rs` runs the same body on the shipped
/// default, which can.
#[test]
fn store_backed_experts_produce_bit_identical_logits_to_resident() {
    common::assert_store_backed_matches_resident(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/frink_real_moe_test.gguf"
        ),
        frink_models::config::test_moe_fixture,
    );
}

/// `frink gguf-split --split` has to produce shards the loader people
/// actually use can read, not only ones `ShardedGguf::open` accepts.
///
/// The checked-in `frink_real_test_split-*` fixtures were written by a
/// Python script; this splits the SAME single-file fixture with the
/// Rust tool, at a boundary that lands mid-layer, and runs a real
/// forward pass over the result. Then it merges the shards back and
/// runs the pass a third time. All three logit sequences must be
/// bit-identical: `assert_eq!` on `Vec<f32>`, not a tolerance, because
/// a split moves bytes and must not change one.
///
/// `--no-tensor-first-split` is used because the metadata-only first
/// shard is the layout published checkpoints use, and it is the case
/// where the loader has to find the hparams in a shard holding no
/// tensors at all.
#[test]
fn tool_split_shards_and_their_merge_load_and_decode_identically_to_the_source() {
    use frink_gguf::split::{
        plan_merge, plan_split, write_merge, write_split, SplitMode, SplitOptions,
    };

    let single_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/frink_real_test.gguf"
    );

    let dir = std::env::temp_dir().join(format!("frink_tool_split_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();

    let source = frink_gguf::GgufFile::open(single_path).unwrap();
    let opts = SplitOptions {
        mode: SplitMode::MaxTensors(4),
        no_tensor_first_split: true,
    };
    let plan = plan_split(&source, &opts).expect("the real fixture must plan a split");
    assert!(
        plan.shards.len() > 2,
        "4 tensors per shard over {} tensors has to make several shards",
        plan.n_tensors
    );
    assert!(
        plan.shards[0].tensors.is_empty(),
        "--no-tensor-first-split must leave shard 1 metadata-only"
    );
    let paths = write_split(&source, &plan, &dir.join("m"), |_| {}).expect("shards must write");

    let merged_path = dir.join("merged.gguf");
    let merge = plan_merge(&paths[0]).expect("the tool's own shards must merge");
    write_merge(&merge, &merged_path).expect("the merge must write");

    let logits = |path: &str| {
        let decoder = Decoder::from_gguf(path, test_dense_fixture())
            .unwrap_or_else(|e| panic!("{path} must load: {e}"));
        let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
        [3usize, 7, 11]
            .iter()
            .enumerate()
            .map(|(pos, tok)| decoder.forward_token(*tok, pos, &mut caches))
            .collect::<Vec<_>>()
    };

    let want = logits(single_path);
    assert_eq!(
        logits(paths[0].to_str().unwrap()),
        want,
        "the tool's shards must decode bit-identically to the single file"
    );
    assert_eq!(
        logits(merged_path.to_str().unwrap()),
        want,
        "merging the tool's shards must decode bit-identically to the single file"
    );

    std::fs::remove_dir_all(&dir).ok();
}
