//! `FRINK_METAL=0` means no Metal work at all, on a build that has the
//! Metal backend compiled in.
//!
//! # What shipped broken
//!
//! `frink_core::gdn_chunk::delta_chunk_rows` launched the chunked delta
//! rule on the GPU whenever a prefill had 32 rows or more, reading the
//! shape and never the switch. So a `--features metal` build under
//! `FRINK_METAL=0` (`--device cpu`) ran every long recurrent prefill on
//! the device, and the host's `qwen3next` decode parted from its own
//! prefill by 4.9e-5 against a 3e-5 line, a drift that vanished without
//! the feature. Six other dispatches in `frink-metal` bypassed the
//! counter this test reads, so it could not have seen them either; the
//! counter's bypass test now walks the crate's whole `src/`.
//!
//! # What this checks
//!
//! Every loadable graph fixture, on the host pin, through every path a
//! golden drives plus one that none of them does: a prefill past the
//! 32-row device floor of the chunked rule. After each fixture the
//! process-wide dispatch count must still be zero. Without `--features
//! metal` there is no counter and this only proves the paths run.

mod common;

#[test]
fn a_host_pinned_process_encodes_no_metal_dispatch_on_any_fixture() {
    // The switch a user sets, decided from the environment BEFORE the
    // test helpers' own pin can stand in for it: this binary holds one
    // test, so nothing else has read it yet.
    std::env::set_var("FRINK_METAL", "0");
    #[cfg(feature = "metal")]
    assert!(!frink_core::weight_matrix::metal_dense_enabled());
    common::pin_backend(common::Backend::Cpu);
    let names = common::graph_fixture_names();
    assert!(names.len() > 100, "found {} fixtures", names.len());
    let mut ran = 0;
    for name in &names {
        let Ok(d) = common::try_load_graph_fixture(name) else {
            continue; // a refusal fixture: it never computes
        };
        let vocab = d.config.vocab_size;
        // Past `gdn_chunk::DEVICE_ROWS` (32), inside every fixture's
        // context of 64.
        let long: Vec<usize> = (0..40).map(|i| (i * 7 + 3) % vocab).collect();
        let mut kv = d.config.new_kv_caches();
        let _ = d.forward_batch(&long, 0, &mut kv);
        let _ = d.forward_token(long[0], long.len(), &mut kv);
        let mut kv = vec![d.config.new_kv_caches()];
        let _ = d.forward_multi_seq(&[long[1]], &[0], &mut kv);
        // Fails naming the first fixture that dispatched.
        #[cfg(feature = "metal")]
        assert_eq!(
            frink_metal::metal_encode_stats().dispatches,
            0,
            "{name}: Metal dispatches in a host-pinned process"
        );
        ran += 1;
    }
    assert!(ran > 100, "only {ran} fixtures loaded");
}
