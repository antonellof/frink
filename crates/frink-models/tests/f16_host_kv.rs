//! The f16 host K/V store (`frink_core::kv_store`), end to end.
//!
//! The graph goldens in this directory are libllama's with
//! `type_k = GGML_TYPE_F32` (`scripts/gptoss_reference_logits.cpp`), so
//! they pin the f32 store and stay the oracle for the GRAPH. What they
//! cannot say is whether every path reads an f16 store the same way. A
//! body that read the widened rows while another read the narrow ones,
//! or one that wrote f32 into an f16 cache by a side door, would agree
//! with the golden on one path and not the others. So this holds the
//! f16 store to three things on fixtures that reach every kind of layer
//! the cache serves -- dense attention, a sliding window, a short
//! convolution (whose history stays f32), a Mamba state:
//!
//! 1. every path (prefill, decode, multi-seq, paged) agrees with every
//!    other at f16, to the f32 accumulation-order line;
//! 2. the answer is near the f32 one (f16 rounding, not a wrong graph);
//! 3. and it is NOT the f32 one, so the store really is two bytes wide.

mod common;

use common::{graph_caches, load_graph_fixture, worst_vs, GRAPH_PROMPT};
use frink_core::kv_store::KvDtype;
use frink_models::Decoder;

/// Same rounding at every read, so the paths may differ only by the
/// order their f32 sums run in.
const PATHS_AGREE: f32 = 1e-4;
/// f16 holds 11 significant bits; on these fixtures that moves logits by
/// a few 1e-3 (measured: 2.5e-3 to 4.4e-3 on the ALiBi rows).
const NEAR_F32: f32 = 3e-2;

fn every_path(d: &Decoder) -> [(&'static str, Vec<f32>); 4] {
    let prefill = d.forward_batch_last(&GRAPH_PROMPT, 0, &mut graph_caches(d));

    let mut kv = graph_caches(d);
    let mut decode = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        decode = d.forward_token(tok, pos, &mut kv);
    }

    let mut kv = vec![graph_caches(d)];
    let mut multi = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        multi = d.forward_multi_seq(&[tok], &[pos], &mut kv).remove(0);
    }

    let store = std::sync::Arc::new(d.config.new_paged_kv(4, 8));
    let mut paged_kv: Vec<frink_core::cache::PagedKvCache> = (0..d.config.n_layers)
        .map(|_| frink_core::cache::PagedKvCache::new())
        .collect();
    let mut paged = Vec::new();
    for (pos, &tok) in GRAPH_PROMPT.iter().enumerate() {
        paged = d
            .forward_token_paged(tok, pos, &mut paged_kv, &store)
            .expect("8 blocks of 4 hold the prompt");
    }

    [
        ("prefill", prefill),
        ("decode", decode),
        ("multi-seq", multi),
        ("paged", paged),
    ]
}

fn check(name: &str) {
    let mut d = load_graph_fixture(name);
    assert_eq!(d.config.kv_dtype, KvDtype::F32, "the library default");
    let f32_decode = every_path(&d)[1].1.clone();

    d.config.kv_dtype = KvDtype::F16;
    let caches = graph_caches(&d);
    assert!(
        (0..d.config.n_layers).any(|il| caches[il].dtype() == KvDtype::F16),
        "{name}: no layer's cache came out f16"
    );

    let paths = every_path(&d);
    let (_, reference) = &paths[1];
    for (path, got) in &paths {
        let worst = worst_vs(got, reference);
        assert!(
            worst < PATHS_AGREE,
            "{name}: {path} disagrees with decode at f16 by {worst}"
        );
    }
    let off = worst_vs(reference, &f32_decode);
    assert!(off < NEAR_F32, "{name}: f16 is {off} from f32");
    assert!(
        off > 0.0,
        "{name}: f16 matched f32 exactly, so nothing was narrowed"
    );
}

#[test]
fn dense_attention_reads_one_f16_store_on_every_path() {
    check("refact");
}

#[test]
fn a_sliding_window_reads_one_f16_store_on_every_path() {
    check("cohere2");
}

#[test]
fn a_short_convolution_keeps_its_history_at_f32_beside_f16_attention() {
    let mut d = load_graph_fixture("lfm2");
    d.config.kv_dtype = KvDtype::F16;
    let caches = graph_caches(&d);
    let dtypes: Vec<KvDtype> = caches.iter().map(|c| c.dtype()).collect();
    assert!(dtypes.contains(&KvDtype::F16), "{dtypes:?}");
    assert!(
        dtypes.contains(&KvDtype::F32),
        "the conv layers' history must stay f32: {dtypes:?}"
    );
    check("lfm2");
}

#[test]
fn a_mamba_hybrid_reads_one_f16_store_on_every_path() {
    check("granitehybrid_rope");
}
