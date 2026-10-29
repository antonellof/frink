//! LoRA adapters, checked against llama.cpp itself.
//!
//! The adapter files were NOT written by frink's fixture script: it
//! writes a PEFT adapter directory (`adapter_config.json` +
//! `adapter_model.safetensors`, what `peft` saves) and runs llama.cpp's
//! own `convert_lora_to_gguf.py` over it, so `lora_a_tiny.gguf` and
//! `lora_b_tiny.gguf` carry upstream's format -- `general.type =
//! "adapter"`, `adapter.type = "lora"`, `adapter.lora.alpha`,
//! `<base>.lora_a` / `.lora_b`, the converter's transposes, the llama
//! Q/K permutation on `lora_b` -- and not a spelling frink's loader
//! and frink's fixture happen to share.
//!
//! **What the adapters cover.** `lora_a` (rank 4, alpha 8) adapts
//! EVERY projection `build_lora_mm` can see on this graph: Q, K, V, O,
//! gate, up, down on both layers, plus `token_embd` (the FLIPPED pair,
//! `llama-graph.cpp:2296-2304`) and `output`. `lora_b` (rank 2, alpha
//! 2) adapts Q and V only, the PEFT default. Loading both at once is
//! the `for (const auto & lora : *loras)` sum, and the goldens for
//! `--lora a:1.0 --lora b:0.75` pin it.
//!
//! **Where the numbers come from.** Every golden below was produced by
//! `scripts/gptoss_reference_logits.cpp --lora` against a real
//! `libllama` built from `.scratch/llama.cpp`, F32 KV, no flash
//! attention. libllama's own logits for `--lora a:0` are byte-identical
//! to the base's, which is what makes the scale-zero test below a
//! statement about llama.cpp and not only about frink.
//!
//! Regenerating:
//!
//! ```text
//! python3 scripts/make_lora_fixture.py crates/frink-models/tests/fixtures $LLAMA
//! F=crates/frink-models/tests/fixtures
//! /tmp/ref_logits $F/lora_base_tiny.gguf 3 7 11 19 23 5
//! /tmp/ref_logits --lora $F/lora_a_tiny.gguf $F/lora_base_tiny.gguf 3 7 11 19 23 5
//! /tmp/ref_logits --lora $F/lora_a_tiny.gguf:0.5 $F/lora_base_tiny.gguf 3 7 11 19 23 5
//! /tmp/ref_logits --lora $F/lora_a_tiny.gguf:1 --lora $F/lora_b_tiny.gguf:0.75 \
//!     $F/lora_base_tiny.gguf 3 7 11 19 23 5
//! /tmp/ref_logits --lora $F/lora_b_tiny.gguf $F/lora_base_tiny.gguf 3 7 11 19 23 5
//! ```

// The goldens are libllama's `%.9g` output verbatim; a ninth digit an
// f32 cannot hold is rounded by the compiler exactly as the C runtime
// would round it, and retyping the constants to appease the lint is
// the one way to get a wrong digit into them.
#![allow(clippy::excessive_precision)]

mod common;
use common::{
    assert_all_three_paths_match, assert_decoder_matches_on_all_three_paths, graph_caches,
    graph_fixture_path, kl_vs_golden, load_graph_fixture, worst_vs, GRAPH_PROMPT, GRAPH_TOL,
};
use frink_gguf::GgufFile;
use frink_models::lora::{LoraAdapter, LoraError};
use frink_models::Decoder;

const BASE: &str = "lora_base";

fn base_file() -> GgufFile {
    GgufFile::open(graph_fixture_path(BASE)).expect("base fixture opens")
}

fn adapter(name: &str) -> LoraAdapter {
    LoraAdapter::open(graph_fixture_path(name)).expect("adapter fixture parses")
}

/// The base with `adapters` attached at their scales, in order.
fn adapted(adapters: &[(&str, f32)]) -> Decoder {
    let mut d = load_graph_fixture(BASE);
    let base = base_file();
    for (i, (name, scale)) in adapters.iter().enumerate() {
        let id = d
            .attach_lora(&base, adapter(name), *scale)
            .unwrap_or_else(|e| panic!("{name} attaches: {e}"));
        assert_eq!(id, i);
    }
    d
}

fn prefill(d: &Decoder) -> Vec<f32> {
    let mut kv = graph_caches(d);
    d.forward_batch_last(&GRAPH_PROMPT, 0, &mut kv)
}

const BASE_GOLDEN: [f32; 48] = [
    -0.370904267,
    -1.12565887,
    -1.74006295,
    -1.08758676,
    0.476774454,
    0.600389838,
    -0.589864194,
    1.80681932,
    2.86660552,
    -1.96880364,
    1.63735247,
    1.32705796,
    -1.13267052,
    0.198578566,
    0.247464299,
    0.335818946,
    -0.311274469,
    1.67584085,
    -0.893444657,
    0.864701152,
    -0.48749727,
    -0.480290055,
    -0.626489639,
    0.805029571,
    -0.804177046,
    -2.09429097,
    1.3663398,
    0.76220727,
    0.343970269,
    0.346324265,
    -2.55922723,
    0.282697946,
    1.32494926,
    0.266927719,
    0.101378471,
    0.038739264,
    -1.10674012,
    -0.653178751,
    0.23567237,
    -1.41555858,
    -0.80852747,
    1.30809903,
    1.44235611,
    -0.389747143,
    0.473889053,
    0.992623746,
    0.52838558,
    0.232206672,
];

const A_GOLDEN: [f32; 48] = [
    -0.536873817,
    -0.865892172,
    -1.06427574,
    -0.360642791,
    -1.36866879,
    1.12833381,
    0.17011711,
    0.503298402,
    -0.845693052,
    2.09086847,
    -0.953029752,
    -0.791327715,
    -0.358710736,
    0.315385282,
    0.289216399,
    0.27945292,
    -2.62314343,
    0.0533467531,
    1.24105656,
    0.890393138,
    -1.35349703,
    -0.287334263,
    -0.469020128,
    -0.771149993,
    -2.01075006,
    1.36664784,
    -1.09304082,
    0.679412365,
    1.54256582,
    2.4239862,
    4.14771175,
    0.736795902,
    -2.54307985,
    -1.21588445,
    -0.00577360392,
    -1.08792377,
    1.53952384,
    -0.979934931,
    -0.481251597,
    -0.00689542294,
    0.65814662,
    -0.798549235,
    2.45526719,
    -1.45747113,
    1.3203969,
    -0.118594617,
    -0.341794401,
    1.13295007,
];

const A_HALF_GOLDEN: [f32; 48] = [
    -1.953094,
    -0.10603371,
    -1.1272352,
    -1.1571698,
    1.9561098,
    0.017521799,
    -1.855238,
    3.5839274,
    1.322493,
    -2.0307927,
    1.5708414,
    -0.67856145,
    -1.0446041,
    -0.5786852,
    -0.9234129,
    -0.2487503,
    0.2809117,
    1.2748075,
    -0.6462314,
    0.94025356,
    -0.6067921,
    0.96796745,
    0.068926156,
    1.1276128,
    -1.2498178,
    -1.2279766,
    1.6192799,
    -0.91402996,
    -1.1700323,
    1.5784234,
    -3.0203469,
    -0.51455605,
    0.74888504,
    -1.1642596,
    -0.67561704,
    -0.45768827,
    -1.9477267,
    0.9254422,
    -1.2825227,
    -0.8443275,
    -1.513934,
    2.832958,
    1.1648672,
    -0.30300266,
    1.0195422,
    1.1422024,
    1.0946232,
    1.0820498,
];

const AB_GOLDEN: [f32; 48] = [
    0.218488634,
    -1.14548278,
    -0.832244158,
    0.547200024,
    -1.21331131,
    1.20635152,
    -0.0414975584,
    -0.383256972,
    0.303418756,
    1.25630832,
    -0.656019807,
    0.641156316,
    -1.11815083,
    0.023001194,
    0.149316788,
    0.73910141,
    -2.91507649,
    -0.539863348,
    -0.0998753458,
    -0.655889273,
    -1.65785527,
    -0.370646477,
    -0.215931475,
    -0.292460054,
    -1.99574506,
    1.58345556,
    -1.40718293,
    0.145563021,
    1.00802922,
    2.88666725,
    3.89027452,
    0.748188913,
    -1.55383968,
    -0.871917307,
    0.683666408,
    -0.218239486,
    0.712480366,
    -0.322091937,
    0.772425592,
    -0.186063498,
    -0.685799718,
    -0.147196501,
    2.36823893,
    -0.336556911,
    1.07483685,
    0.118239105,
    -0.546918273,
    0.52361685,
];

const B_GOLDEN: [f32; 48] = [
    -0.103938207,
    -0.457222104,
    -0.941307843,
    -0.645068645,
    -0.382739455,
    -0.627984405,
    -1.25409985,
    0.0547681004,
    2.35386777,
    -0.943867266,
    2.02643752,
    1.08659089,
    -0.869225919,
    -0.964745045,
    -0.239215657,
    -0.0702250451,
    0.344786733,
    0.671282232,
    0.016603034,
    0.9461115,
    1.2614044,
    0.358712792,
    0.344278932,
    0.763074517,
    0.119232848,
    -1.70412064,
    2.14636588,
    -0.861565888,
    0.413673908,
    0.517497182,
    -2.41648054,
    0.81807375,
    2.21736312,
    0.0818412825,
    -0.126288608,
    1.29192376,
    -1.29586041,
    0.0498075821,
    -1.01258016,
    -0.805461645,
    0.191048965,
    1.71336377,
    0.0372688174,
    -1.38677239,
    1.31535792,
    0.72845459,
    0.109508976,
    0.806996822,
];

/// The base alone, on all three paths: the seam costs nothing when no
/// adapter is attached (the same golden pins the pre-attach output in
/// `scale_zero_is_byte_identical_to_no_adapter`).
#[test]
fn the_base_matches_llama_cpp_on_all_three_paths() {
    assert_all_three_paths_match(BASE, &BASE_GOLDEN);
}

/// One adapter over every projection, embedding and head included.
#[test]
fn adapter_a_matches_llama_cpp_on_all_three_paths() {
    let d = adapted(&[("lora_a", 1.0)]);
    assert_decoder_matches_on_all_three_paths(&d, &A_GOLDEN, GRAPH_TOL, "lora_a");
}

/// `--lora-scaled a:0.5`: the adapter scale multiplies the whole
/// low-rank term, `alpha / rank` included.
#[test]
fn adapter_a_at_half_scale_matches_llama_cpp() {
    let d = adapted(&[("lora_a", 0.5)]);
    assert_decoder_matches_on_all_three_paths(&d, &A_HALF_GOLDEN, GRAPH_TOL, "lora_a:0.5");
}

/// The Q/V-only adapter alone, with its own alpha and rank.
#[test]
fn adapter_b_matches_llama_cpp_on_all_three_paths() {
    let d = adapted(&[("lora_b", 1.0)]);
    assert_decoder_matches_on_all_three_paths(&d, &B_GOLDEN, GRAPH_TOL, "lora_b");
}

/// Two adapters on one model are two terms of one sum, each at its
/// own scale, as `build_lora_mm`'s loop over `*loras` makes them.
#[test]
fn two_adapters_at_two_scales_match_llama_cpp() {
    let d = adapted(&[("lora_a", 1.0), ("lora_b", 0.75)]);
    assert_decoder_matches_on_all_three_paths(&d, &AB_GOLDEN, GRAPH_TOL, "lora_a:1+lora_b:0.75");
}

/// The number in the report, so it can be regenerated rather than
/// trusted. Run with `--nocapture` to see it.
#[test]
fn report_kl_against_llama_cpp() {
    for (label, adapters, golden) in [
        ("base", &[][..], &BASE_GOLDEN),
        ("lora_a", &[("lora_a", 1.0)][..], &A_GOLDEN),
        ("lora_a:0.5", &[("lora_a", 0.5)][..], &A_HALF_GOLDEN),
        ("lora_b", &[("lora_b", 1.0)][..], &B_GOLDEN),
        (
            "lora_a:1 + lora_b:0.75",
            &[("lora_a", 1.0), ("lora_b", 0.75)][..],
            &AB_GOLDEN,
        ),
    ] {
        let got = prefill(&adapted(adapters));
        let kl = kl_vs_golden(&got, golden);
        let worst = worst_vs(&got, golden);
        println!("| `{label}` | {kl:.2e} | {worst:.2e} |");
        assert!(kl < 1e-8, "{label}: KL {kl}");
    }
}

/// An adapter at scale 0 is no adapter, bit for bit -- on frink as it
/// is on libllama (whose `--lora a:0` logits are byte-identical to its
/// base logits, measured while making the goldens). The delta is
/// skipped before any arithmetic, so the adapted matrix's output is
/// the base's `Vec<f32>` unchanged, not a sum with a zero.
#[test]
fn scale_zero_is_byte_identical_to_no_adapter() {
    let plain = load_graph_fixture(BASE);
    let before = prefill(&plain);
    assert_eq!(before, prefill(&adapted(&[("lora_a", 0.0)])));
    assert_eq!(
        before,
        prefill(&adapted(&[("lora_a", 0.0), ("lora_b", 0.0)]))
    );
}

/// Attaching an adapter and then turning it off returns EXACTLY the
/// pre-attach logits: the base's own bytes are untouched.
#[test]
fn attaching_then_zeroing_returns_the_pre_attach_logits() {
    let mut d = load_graph_fixture(BASE);
    let before = prefill(&d);
    d.attach_lora(&base_file(), adapter("lora_a"), 1.0).unwrap();
    assert!(
        worst_vs(&prefill(&d), &before) > 1e-2,
        "the adapter must be visible at 1.0"
    );
    d.set_lora_scales(&[(0, 0.0)]).unwrap();
    assert_eq!(prefill(&d), before);
}

/// `POST /lora-adapters` and a per-request `lora` list are ONE atomic
/// store per adapter: the same decoder answers with a different scale
/// on the next call, no reload, and an adapter left out of the list
/// goes to 0 as `construct_lora_list` sets it.
#[test]
fn scales_change_at_apply_time_without_a_reload() {
    let d = adapted(&[("lora_a", 1.0), ("lora_b", 1.0)]);
    assert_eq!(d.lora_scales(), vec![1.0, 1.0]);

    d.set_lora_scales(&[(0, 0.5)]).unwrap();
    assert_eq!(
        d.lora_scales(),
        vec![0.5, 0.0],
        "an unlisted adapter goes to 0"
    );
    assert_decoder_matches_on_all_three_paths(&d, &A_HALF_GOLDEN, GRAPH_TOL, "a:0.5 via set");

    d.set_lora_scales(&[(0, 1.0), (1, 0.75)]).unwrap();
    assert_decoder_matches_on_all_three_paths(&d, &AB_GOLDEN, GRAPH_TOL, "a:1,b:0.75 via set");

    let err = d.set_lora_scales(&[(2, 1.0)]).unwrap_err();
    assert!(err.contains("out of range"), "{err}");
    assert_eq!(
        d.lora_scales(),
        vec![1.0, 0.75],
        "a refused set changes nothing"
    );
}

/// What the attach recorded: ids in attach order, the file's alpha,
/// the tensor count, and the fact the Metal fence reads.
#[test]
fn the_attached_list_is_what_the_server_will_report() {
    let d = adapted(&[("lora_a", 1.0), ("lora_b", 0.25)]);
    assert!(d.lora_attached());
    assert_eq!(d.lora_adapters.len(), 2);
    let a = &d.lora_adapters[0];
    assert!(a.path.ends_with("lora_a_tiny.gguf"));
    assert_eq!(a.alpha, 8.0);
    assert_eq!(
        a.n_tensors, 16,
        "7 projections x 2 layers + embedding + head"
    );
    assert_eq!(a.scale(), 1.0);
    let b = &d.lora_adapters[1];
    assert_eq!(b.alpha, 2.0);
    assert_eq!(b.n_tensors, 4);
    assert_eq!(b.scale(), 0.25);
    assert!(!load_graph_fixture(BASE).lora_attached());
}

/// The delta lives INSIDE the projection: every adapted matrix is the
/// `Adapted` variant, and the ones the adapter did not name are not.
#[test]
fn the_adapter_decorates_exactly_the_matrices_it_names() {
    let d = adapted(&[("lora_b", 1.0)]);
    for (il, layer) in d.layers.iter().enumerate() {
        assert_eq!(
            layer.attn.q_proj.lora().map(|l| l.len()),
            Some(1),
            "blk.{il} q"
        );
        assert_eq!(
            layer.attn.v_proj.lora().map(|l| l.len()),
            Some(1),
            "blk.{il} v"
        );
        assert!(
            layer.attn.k_proj.lora().is_none(),
            "blk.{il} k is not in lora_b"
        );
        assert!(
            layer.attn.o_proj.lora().is_none(),
            "blk.{il} o is not in lora_b"
        );
    }
    assert!(d.embedding.lora().is_none());
    assert!(d.output_head.lora().is_none());

    let d = adapted(&[("lora_a", 1.0), ("lora_b", 1.0)]);
    assert_eq!(
        d.layers[0].attn.q_proj.lora().map(|l| l.len()),
        Some(2),
        "both on Q"
    );
    assert_eq!(
        d.layers[0].attn.k_proj.lora().map(|l| l.len()),
        Some(1),
        "only A on K"
    );
    assert_eq!(d.embedding.lora().map(|l| l.len()), Some(1));
    assert_eq!(d.output_head.lora().map(|l| l.len()), Some(1));
    assert_eq!(
        d.embedding.rows(),
        48,
        "the adapted matrix keeps the base's shape"
    );
    assert_eq!(d.embedding.cols(), 24);
}

/// The embedding pair really is flipped: reading it in the projection
/// layout (rows from `lora_b`, rank from `lora_a`) diverges. This is
/// the sabotage that a loader which "just used the same code for every
/// pair" would commit, and `token_embd`'s shapes happen to make it
/// well-formed on this fixture only by transposition, so the test
/// forges the un-flipped pair by hand and checks the row delta.
#[test]
fn the_embedding_pair_is_the_flipped_one() {
    let d = adapted(&[("lora_a", 1.0)]);
    let a = adapter("lora_a");
    let pair = &a.pairs["token_embd.weight"];
    // Row t of the delta is `B a_t`: B is [n_embd, rank] and a_t is
    // row t of lora_a ([n_vocab, rank]). `alpha / rank` = 8 / 4 = 2.
    let base = load_graph_fixture(BASE);
    let rank = pair.a.cols();
    for t in [3usize, 7, 11] {
        let want: Vec<f32> = (0..24)
            .map(|e| {
                base.embedding.dequant_row(t)[e]
                    + 2.0
                        * (0..rank)
                            .map(|k| pair.b.data[e * rank + k] * pair.a.data[t * rank + k])
                            .sum::<f32>()
            })
            .collect();
        let got = d.embedding.dequant_row(t);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "token {t}: {g} vs {w}");
        }
    }
}

/// Dropping `alpha / rank` -- treating the file's alpha as absent --
/// diverges: the scale is `adapter_scale * alpha / rank`, and on this
/// adapter that is a factor of 2.
#[test]
fn dropping_alpha_over_rank_diverges_from_llama_cpp() {
    let mut a = adapter("lora_a");
    a.alpha = 0.0;
    let mut d = load_graph_fixture(BASE);
    d.attach_lora(&base_file(), a, 1.0).unwrap();
    let worst = worst_vs(&prefill(&d), &A_GOLDEN);
    assert!(worst > 1e-2, "alpha/rank moved the output by only {worst}");
    // ... and the same file at half scale IS the alpha-less file at
    // scale 1 / (alpha / rank) = 0.5, which pins the formula's shape.
    let mut a = adapter("lora_a");
    a.alpha = 0.0;
    let mut d = load_graph_fixture(BASE);
    d.attach_lora(&base_file(), a, 2.0).unwrap();
    assert_decoder_matches_on_all_three_paths(&d, &A_GOLDEN, GRAPH_TOL, "alpha folded into scale");
}

// --- refusals, each reachable from a file libllama refuses the same way

fn attach_err(base: &str, adapter_name: &str) -> LoraError {
    let mut d = load_graph_fixture(base);
    let file = GgufFile::open(graph_fixture_path(base)).unwrap();
    match d.attach_lora(&file, adapter(adapter_name), 1.0) {
        Ok(_) => panic!("{adapter_name} on {base} must be refused"),
        Err(e) => e,
    }
}

/// An adapter converted for another architecture: `llama-adapter.cpp:
/// 207-211`, "model arch and LoRA arch mismatch".
#[test]
fn an_adapter_for_another_architecture_is_refused() {
    let err = attach_err("arcee", "lora_b");
    assert!(matches!(err, LoraError::ArchMismatch { .. }), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("\"llama\""), "{msg}");
    assert!(msg.contains("\"arcee\""), "{msg}");
}

/// An adapter naming a tensor the base does not carry:
/// `llama-adapter.cpp:349-351`. libllama's message for `lora_a` on the
/// tied base is exactly this one, for `output.weight` (measured).
#[test]
fn an_adapter_naming_a_tensor_the_base_lacks_is_refused() {
    let err = attach_err("lora_base_tied", "lora_a");
    assert!(matches!(err, LoraError::NotInBase { .. }), "{err}");
    assert!(
        err.to_string()
            .contains("'output.weight' does not exist in base model"),
        "{err}"
    );
}

/// The embedding pair on a base whose head is the embedding. libllama
/// loads the adapter and then ABORTS in the graph build --
/// `ggml.c:3282: GGML_ASSERT(ggml_can_mul_mat(a, b))` from
/// `build_lora_mm(model.output, ..)` over the flipped pair (measured
/// with `lora_e_tiny.gguf` on `lora_base_tied_tiny.gguf`). frink
/// refuses at attach, naming that.
#[test]
fn the_embedding_pair_on_a_tied_head_is_refused_like_llama_cpp_aborts() {
    // The same adapter attaches to the untied base, so the refusal is
    // about the head, not the file.
    let mut d = load_graph_fixture(BASE);
    d.attach_lora(&base_file(), adapter("lora_e"), 1.0).unwrap();

    let err = attach_err("lora_base_tied", "lora_e");
    assert!(matches!(err, LoraError::TiedHead { .. }), "{err}");
    assert!(err.to_string().contains("ggml_can_mul_mat"), "{err}");
}

/// A pair whose shapes do not fit the base: `llama-adapter.cpp:361-363`.
/// Forged by hand from the real adapter, since a converter cannot
/// produce it for a base it was given.
#[test]
fn a_pair_of_the_wrong_shape_is_refused_naming_every_dimension() {
    let mut a = adapter("lora_b");
    let pair = a.pairs.get_mut("blk.0.attn_q.weight").unwrap();
    pair.b.data.truncate(12 * 2);
    pair.b.shape = [12, 2];
    let mut d = load_graph_fixture(BASE);
    let err = d.attach_lora(&base_file(), a, 1.0).unwrap_err();
    assert!(matches!(err, LoraError::Shape { .. }), "{err}");
    let msg = err.to_string();
    assert!(msg.contains("incorrect shape"), "{msg}");
    assert!(msg.contains("[24 x 24]"), "{msg}");
    assert!(msg.contains("[12, 2]"), "{msg}");
}

/// A pair whose ranks disagree -- the un-transposed `lora_a` of the
/// old `finetune` example: `llama-adapter.cpp:364-366`.
#[test]
fn an_untransposed_lora_a_is_refused() {
    let mut a = adapter("lora_b");
    let pair = a.pairs.get_mut("blk.1.attn_v.weight").unwrap();
    // [rank, n_in] = [2, 24] -> pretend [3, 16]: same 48 values, a rank
    // B (which is [12, 2]) does not have.
    pair.a.shape = [3, 16];
    let mut d = load_graph_fixture(BASE);
    let err = d.attach_lora(&base_file(), a, 1.0).unwrap_err();
    // cols 16 != 24 trips the shape check first, as it does upstream
    // (`:361` before `:364`); make the cols agree to reach the rank check.
    assert!(matches!(err, LoraError::Shape { .. }), "{err}");

    let mut a = adapter("lora_b");
    let pair = a.pairs.get_mut("blk.1.attn_v.weight").unwrap();
    pair.a.data.extend(std::iter::repeat_n(0.0, 24));
    pair.a.shape = [3, 24];
    let mut d = load_graph_fixture(BASE);
    let err = d.attach_lora(&base_file(), a, 1.0).unwrap_err();
    assert!(matches!(err, LoraError::NotTransposed { .. }), "{err}");
    assert!(err.to_string().contains("not transposed"), "{err}");
}

/// A routed-expert tensor: upstream serves it through
/// `build_lora_mm_id`, which frink has no delta for.
#[test]
fn an_adapter_on_routed_experts_is_refused_by_name() {
    let mut a = adapter("lora_b");
    a.arch = "qwen3moe".to_string();
    let pair = a.pairs.remove("blk.0.attn_q.weight").unwrap();
    a.pairs.clear();
    a.pairs
        .insert("blk.0.ffn_gate_exps.weight".to_string(), pair);
    let base = graph_fixture_path("qwen3moe");
    let file = GgufFile::open(&base).unwrap();
    let mut d = load_graph_fixture("qwen3moe");
    let err = d.attach_lora(&file, a, 1.0).unwrap_err();
    assert!(matches!(err, LoraError::RoutedExperts { .. }), "{err}");
    assert!(err.to_string().contains("build_lora_mm_id"), "{err}");
}

/// An adapter left at its file's scale after a failed second attach
/// is still the first adapter: ids are stable.
#[test]
fn a_refused_attach_leaves_the_earlier_adapter_in_place() {
    let mut d = adapted(&[("lora_a", 1.0)]);
    let tied = GgufFile::open(graph_fixture_path("lora_base_tied")).unwrap();
    assert!(d.attach_lora(&tied, adapter("lora_e"), 1.0).is_err());
    assert_eq!(d.lora_adapters.len(), 1);
    assert_decoder_matches_on_all_three_paths(&d, &A_GOLDEN, GRAPH_TOL, "lora_a after a refusal");
}
