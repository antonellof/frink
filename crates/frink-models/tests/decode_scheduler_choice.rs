//! One single-token CPU decode step runs on ONE scheduler (#27).
//!
//! # What this catches
//!
//! The per-operation rule (`frink_core::par::policy::backend`) sent each
//! matvec over `SPIN_MIN_OP_MACS` to the persistent pool and every other
//! region to rayon. Counted on one layer at Llama-3.2-3B's widths
//! (3072 / 8192, 24 / 8 heads, Q8_0), one decode step opened eleven
//! regions, four to six on the pool and the rest on rayon, with six
//! scheduler changes; at 1B's widths the same. The pool's share moved
//! between runs, because Q/K/V (a `join3`) and gate/up (a `join2`) forked
//! and their branches raced to submit to the pool, which admits one and
//! sends the rest to rayon. Under `FRINK_CPU_POOL=spin` the same step
//! is nine regions, all on the pool, none changing.
//!
//! `frink_core::par::decode_step` now commits the step to one scheduler,
//! and this is the guard. It is an operation COUNT through
//! `frink_core::par::region_tally`, so it needs no quiet host and no
//! stopwatch.
//!
//! # Why this file holds one test
//!
//! `region_tally` is process-wide, because the regions that gave the
//! interleave away were opened on other threads by forked join branches.
//! A second test running concurrently in this binary would add its own
//! regions to the count.

mod common;

use common::quant_encode;
use common::synthetic_llama::{self, Shape};
use frink_core::par::{self, Backend, RegionTally};
use frink_gguf::{GgmlType, ShardedGguf};
use frink_models::config::ModelConfig;
use frink_models::decoder::Decoder;

/// One layer at SmolLM2-135M's widths: every layer matvec is under the
/// crossover, and the pool measured -37% on that model.
const BELOW: Shape = Shape {
    n_embd: 576,
    n_ff: 1536,
    n_head: 9,
    n_head_kv: 3,
    n_layer: 1,
    n_vocab: 512,
};

/// One layer whose FFN (1024 x 2816) is over the crossover and whose
/// attention projections (1024 x 1024, 256 x 1024) are under it: the
/// shape that made the per-operation rule alternate, at a size a debug
/// build writes in a moment.
const ABOVE: Shape = Shape {
    n_embd: 1024,
    n_ff: 2816,
    n_head: 8,
    n_head_kv: 2,
    n_layer: 1,
    n_vocab: 512,
};

/// The tally of the second of two back-to-back decode steps after a
/// prefill, so the step is measured warm and its first region does not
/// count as a change from whatever the prefill ran on.
fn one_warm_decode_step(decoder: &Decoder) -> RegionTally {
    let mut kv = decoder.config.new_kv_caches();
    let _ = decoder.forward_batch(&[3, 141, 59, 265], 0, &mut kv);
    let _ = decoder.forward_token(358, 4, &mut kv);
    let before = par::region_tally();
    let _ = decoder.forward_token(97, 5, &mut kv);
    par::region_tally().since(before)
}

fn load(shape: Shape, tag: &str) -> (Decoder, std::path::PathBuf) {
    let path = synthetic_llama::write_shaped(tag, shape, GgmlType::Q8_0, quant_encode::encode);
    let file = ShardedGguf::open(&path).expect("open the fixture");
    let config = ModelConfig::from_gguf(&file).expect("fixture config");
    let decoder = Decoder::from_gguf(&path, config).expect("load the fixture");
    (decoder, path)
}

/// Sabotage, run: make `par::decode_step` return `on_workers(f)` for
/// every width (the per-operation rule) and the ABOVE step reads
/// `rayon: 9, spin: 2, switches: 2` instead of all on the pool.
#[test]
fn a_cpu_decode_step_never_changes_scheduler() {
    if frink_core::weight_matrix::active_backend() != frink_core::kernel_registry::Backend::Cpu {
        eprintln!("skip: a GPU backend is active; decode_step keeps the per-operation rule there");
        return;
    }
    // Probed, not restated: an unpinned process answers differently at
    // the two ends, a pinned one does not.
    let unpinned = par::decode_backend(0) != par::decode_backend(usize::MAX);

    for (shape, tag) in [(BELOW, "below"), (ABOVE, "above")] {
        let (decoder, path) = load(shape, tag);
        let chosen = par::decode_backend(decoder.widest_layer_macs());
        if unpinned {
            let want = if tag == "above" {
                Backend::Spin
            } else {
                Backend::Rayon
            };
            assert_eq!(
                chosen,
                want,
                "{tag}: widest layer {}",
                decoder.widest_layer_macs()
            );
        }
        let step = one_warm_decode_step(&decoder);
        std::fs::remove_file(&path).ok();

        assert_eq!(
            step.switches, 0,
            "{tag}: the step changed scheduler: {step:?}"
        );
        let (on_chosen, on_other) = match chosen {
            Backend::Spin => (step.spin, step.rayon),
            Backend::Rayon => (step.rayon, step.spin),
        };
        assert!(on_chosen > 0, "{tag}: nothing ran on {chosen:?}: {step:?}");
        assert_eq!(on_other, 0, "{tag}: regions escaped {chosen:?}: {step:?}");
    }
}
