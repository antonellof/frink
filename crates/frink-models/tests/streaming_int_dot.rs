//! Streamed experts against resident ones, on the kernels that SHIP.
//!
//! Expert streaming answered " amongst amongst, and of" where resident
//! answered " Paris." on OLMoE-1B-7B Q4_0, deterministically, while
//! `gguf_roundtrip::store_backed_experts_produce_bit_identical_logits_to_resident`
//! passed even at a 1-byte budget. The cause was the repack cache:
//! it keyed an interleaved copy on the weight's ADDRESS, and an expert
//! store's buffers are freed and reallocated, so a second expert landed
//! at the first one's address and was served the first one's packing.
//! `WeightBytes::map_id` answers `None` for those (`Shared`) bytes now.
//!
//! The fixture test could not see it because the library leaves
//! `FRINK_CPU_INT_DOT` OFF (golden cross-validation stays
//! reference-exact) while `frink` and `frink-server` turn it ON
//! (`default_cpu_int_dot_on`), and only the int-dot kernels repack. So
//! the fixture had been testing a path with no cache in it. This binary
//! runs the SAME body (`common::assert_store_backed_matches_resident`)
//! with int-dot on.
//!
//! A test binary of its own because the switch is read once per process
//! into a `OnceLock`: this file must hold exactly ONE test, which sets
//! it before anything can observe it. A second test here could observe
//! it first and silently run the other arm.
//!
//! Sabotage it by giving `WeightBytes::Shared` a `map_id` built from its
//! buffer address: this goes red at both budgets, prefill and decode,
//! while the `gguf_roundtrip` arm stays green.

mod common;

use frink_core::weight_matrix::{cpu_int_dot_enabled, cpu_int_dot_for, IntDotShape};

#[test]
fn streamed_experts_match_resident_with_int_dot_on() {
    // Edition 2021: safe. Nothing in this process has read the switch
    // yet, because this is the binary's only test.
    std::env::set_var("FRINK_CPU_INT_DOT", "1");
    assert!(
        cpu_int_dot_enabled(),
        "the switch was observed before this test set it, so this run is \
         on the library default and checks nothing the gguf_roundtrip arm \
         does not"
    );
    if !cpu_int_dot_for(IntDotShape::Matvec) && !cpu_int_dot_for(IntDotShape::BatchGemm) {
        // No repacked kernel exists on this host for either workload,
        // so there is no repack path to compare: say so rather than
        // pass as though one had been checked.
        eprintln!("no int-dot tier on this host; nothing repacks, skipping");
        return;
    }
    common::assert_store_backed_matches_resident(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/frink_real_moe_test.gguf"
        ),
        frink_models::config::test_moe_fixture,
    );
}
