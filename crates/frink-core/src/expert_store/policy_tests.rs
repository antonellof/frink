//! Slot mode and the hotness policy.

use super::*;
use crate::expert_slab::HostSlotSlab;

/// Same per-key pattern as `tests.rs`, so a wrong-slot read fails.
struct PatternSource {
    len: usize,
}

fn expected_bytes(key: ExpertKey, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (key.layer as usize * 31 + key.expert as usize * 7 + i * 13) as u8)
        .collect()
}

impl ExpertSource for PatternSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        (key.layer < 8 && key.expert < 8).then_some(self.len)
    }
    fn read_expert(&self, key: ExpertKey) -> io::Result<Vec<u8>> {
        if self.expert_len(key).is_none() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "no such expert"));
        }
        Ok(expected_bytes(key, self.len))
    }
}

fn key(layer: u32, expert: u32) -> ExpertKey {
    ExpertKey { layer, expert }
}

/// A slot-mode store: `slots` slots of `slot_bytes`, over 8 layers x 8
/// experts of `len` bytes each.
fn slot_store(len: usize, slots: usize, slot_bytes: usize) -> ExpertStore<PatternSource> {
    ExpertStore::with_slots(
        PatternSource { len },
        Box::new(HostSlotSlab::host(8, slots, slot_bytes).unwrap()),
    )
}

/// Selection counts a test sets by hand, standing in for the decoder's
/// activation counters.
struct Counts(Vec<AtomicU64>);

impl Counts {
    fn new() -> Arc<Self> {
        Arc::new(Counts((0..64).map(|_| AtomicU64::new(0)).collect()))
    }
    fn select(&self, k: ExpertKey) {
        self.0[(k.layer * 8 + k.expert) as usize].fetch_add(1, Ordering::Relaxed);
    }
}

impl ExpertHotness for Counts {
    fn selections(&self, k: ExpertKey) -> u64 {
        self.0[(k.layer * 8 + k.expert) as usize].load(Ordering::Relaxed)
    }
}

/// Slot mode serves the same bytes heap mode does, from slots it
/// reuses, and never holds more slots than it has. The eviction count
/// is asserted too: a run where everything fit would pass the byte
/// check while exercising no reuse at all.
#[test]
fn slot_mode_serves_exact_bytes_from_reused_slots_within_its_budget() {
    // 100-byte experts in 128-byte slots: the view must stop at 100.
    let s = slot_store(100, 3, 128);
    assert_eq!(s.slot_count(), Some(3));
    assert_eq!(s.budget_bytes(), 3 * 128);
    for round in 0..3u32 {
        for e in 0..8 {
            let k = key(round % 2, e);
            let lease = s.acquire(k).unwrap();
            assert_eq!(lease.bytes(), expected_bytes(k, 100), "{k:?}");
            assert!(lease.slot().is_some(), "an admitted expert lives in a slot");
        }
    }
    let st = s.stats();
    assert!(st.evictions > 0, "a 3-slot pool over 16 keys must evict");
    assert!(st.peak_resident_bytes <= 3 * 128, "peak {st:?}");
    assert_eq!(st.resident_bytes, 3 * 128);
}

/// The slot-reuse guard, in slot mode: a leased slot is never rewritten
/// however hard the pool churns, so a view built over it keeps reading
/// its own expert.
#[test]
fn a_leased_slot_is_never_rewritten_under_churn() {
    let s = slot_store(64, 2, 64);
    let pinned = s.acquire(key(3, 3)).unwrap();
    let slot = pinned.slot().expect("admitted into a free slot");
    for e in 0..8 {
        for l in 0..3 {
            let lease = s.acquire(key(l, e)).unwrap();
            assert_ne!(
                lease.slot(),
                Some(slot),
                "{:?} was written into a slot a live lease reads",
                key(l, e)
            );
        }
    }
    assert_eq!(pinned.bytes(), expected_bytes(key(3, 3), 64));
    assert!(s.stats().evictions > 0, "the other slot must have churned");
}

/// What keeps hot experts resident: once an expert has been routed to
/// more often than a newcomer, the newcomer is served without being
/// admitted rather than evicting it. Without the hotness check the
/// pool is LRU and a stream of one-off experts flushes the hot one
/// every time.
///
/// Sabotage: make `victim` ignore `selections(candidate)` and the
/// residency assertion goes red.
#[test]
fn a_cold_miss_does_not_evict_a_hotter_resident_expert() {
    let counts = Counts::new();
    let s = slot_store(32, 1, 32).with_hotness(counts.clone());
    let hot = key(0, 0);
    for _ in 0..5 {
        counts.select(hot);
        s.acquire(hot).unwrap();
    }
    for e in 1..8 {
        let cold = key(1, e);
        counts.select(cold);
        let lease = s.acquire(cold).unwrap();
        assert_eq!(lease.bytes(), expected_bytes(cold, 32), "served correctly");
        assert!(lease.slot().is_none(), "{cold:?} must pass through");
    }
    assert!(
        s.is_resident(hot),
        "the hot expert was evicted by cold ones"
    );
    let st = s.stats();
    assert_eq!(st.refused_admissions, 7);
    assert_eq!(st.evictions, 0);
}

/// The other half of the same rule: an expert that has become hotter
/// than the coldest resident one IS admitted, so the pool tracks what
/// the router actually does rather than freezing on its first fill.
#[test]
fn an_expert_routed_more_than_the_coldest_resident_one_displaces_it() {
    let counts = Counts::new();
    let s = slot_store(32, 1, 32).with_hotness(counts.clone());
    let (a, b) = (key(0, 0), key(0, 1));
    counts.select(a);
    s.acquire(a).unwrap();
    counts.select(b);
    counts.select(b);
    s.acquire(b).unwrap();
    assert!(s.is_resident(b) && !s.is_resident(a));
    assert_eq!(s.stats().evictions, 1);
}

/// A whole top-k is pinned before any of it is read: with fewer slots
/// than the batch, the overflow passes through rather than evicting a
/// batch member admitted a moment earlier, and every lease still holds
/// its own expert's bytes.
#[test]
fn acquire_many_pins_the_batch_and_shares_a_repeated_key() {
    let s = slot_store(48, 2, 48);
    let keys = [key(0, 0), key(0, 1), key(0, 2), key(0, 1), key(0, 3)];
    let leases = s.acquire_many(&keys).unwrap();
    for (k, l) in keys.iter().zip(&leases) {
        assert_eq!(l.bytes(), expected_bytes(*k, 48), "{k:?}");
    }
    assert_eq!(
        leases[1].slot(),
        leases[3].slot(),
        "a repeat shares a lease"
    );
    let st = s.stats();
    assert_eq!(st.misses, 4, "the repeated key is read once");
    assert_eq!(st.evictions, 0, "no batch member evicts another");
    assert_eq!(st.pass_throughs, 2);
}

/// Concurrency, in slot mode: many threads churn a small pool and every
/// lease's bytes still match its key.
#[test]
fn concurrent_slot_acquires_under_eviction_pressure_never_yield_wrong_bytes() {
    let s = Arc::new(slot_store(64, 3, 64));
    let mut handles = Vec::new();
    for t in 0..8u32 {
        let s = Arc::clone(&s);
        handles.push(std::thread::spawn(move || {
            let mut state = t.wrapping_mul(2654435761).wrapping_add(12345);
            for _ in 0..500 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let a = key((state >> 8) % 4, (state >> 16) % 4);
                let b = key((state >> 12) % 4, (state >> 20) % 4);
                for (k, l) in [a, b].iter().zip(s.acquire_many(&[a, b]).unwrap()) {
                    assert_eq!(l.bytes(), expected_bytes(*k, 64), "slot reuse corruption");
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let st = s.stats();
    assert!(st.evictions > 0);
    assert!(st.peak_resident_bytes <= 3 * 64);
}
