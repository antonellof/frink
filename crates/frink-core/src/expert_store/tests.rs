use super::*;

/// Deterministic per-key content so any slot-reuse/wrong-bytes bug
/// fails loudly: byte `i` of expert `(l, e)` is a function of all
/// three.
struct PatternSource {
    len: usize,
    n_layers: u32,
    n_experts: u32,
}

fn expected_bytes(key: ExpertKey, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (key.layer as usize * 31 + key.expert as usize * 7 + i * 13) as u8)
        .collect()
}

impl ExpertSource for PatternSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        (key.layer < self.n_layers && key.expert < self.n_experts).then_some(self.len)
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

fn store(len: usize, budget: usize) -> ExpertStore<PatternSource> {
    ExpertStore::new(
        PatternSource {
            len,
            n_layers: 8,
            n_experts: 8,
        },
        budget,
    )
}

/// The process-wide gauge tracks every live store's budget and
/// releases it on drop.
///
/// It exists so `weight_matrix::repack_cache` can SUBTRACT the
/// expert budget from the same pool rather than declare a second
/// one: on unified memory an expert byte and a repacked byte are
/// the same RAM, and this module is the single holder of the expert
/// half. A gauge that did not fall on drop would starve the repack
/// cache for the rest of the process.
///
/// The gauge is process-wide and the harness runs these tests in
/// parallel, so this asserts against a SENTINEL budget larger than
/// every other budget in this module put together rather than
/// against an exact total another test could move under it.
///
/// Sabotage: delete the `fetch_add` in `new` or the `Drop` impl and
/// this goes red.
#[test]
fn the_committed_gauge_rises_with_a_store_and_falls_when_it_drops() {
    const SENTINEL: usize = 1 << 40;
    {
        let _held = store(64, SENTINEL);
        assert!(
            committed_expert_bytes() >= SENTINEL as u64,
            "a live store must have committed its budget"
        );
    }
    assert!(
        committed_expert_bytes() < SENTINEL as u64,
        "a dropped store must return its budget to the pool, or the \
         repack cache is starved of it for the rest of the process"
    );
}

#[test]
fn hits_misses_and_lru_eviction_order() {
    let s = store(100, 250); // fits 2 experts
    assert_eq!(
        s.acquire(key(0, 0)).unwrap().bytes(),
        expected_bytes(key(0, 0), 100)
    );
    assert_eq!(
        s.acquire(key(0, 1)).unwrap().bytes(),
        expected_bytes(key(0, 1), 100)
    );
    // Touch (0,0) so (0,1) is now least recently used.
    s.acquire(key(0, 0)).unwrap();
    // Third expert evicts (0,1), not (0,0).
    s.acquire(key(0, 2)).unwrap();
    let before = s.stats();
    s.acquire(key(0, 0)).unwrap(); // must still be a hit
    let after = s.stats();
    assert_eq!(after.hits, before.hits + 1);
    assert_eq!(after.misses, before.misses);
    assert_eq!(after.evictions, 1);
    assert!(after.resident_bytes <= 250);
    // (0,1) was the eviction victim: acquiring it again is a miss.
    s.acquire(key(0, 1)).unwrap();
    assert_eq!(s.stats().misses, after.misses + 1);
}

#[test]
fn a_live_lease_pins_its_entry_against_eviction() {
    let s = store(100, 250); // fits 2 experts
    let pinned = s.acquire(key(1, 0)).unwrap();
    // Fill and churn the cache well past the budget: the pinned
    // entry must never be evicted or corrupted while the lease
    // lives...
    for e in 1..6 {
        s.acquire(key(1, e)).unwrap();
    }
    assert_eq!(pinned.bytes(), expected_bytes(key(1, 0), 100));
    // ...and it is still resident (a re-acquire is a hit).
    let hits_before = s.stats().hits;
    let again = s.acquire(key(1, 0)).unwrap();
    assert_eq!(s.stats().hits, hits_before + 1);
    assert_eq!(again.bytes(), expected_bytes(key(1, 0), 100));

    // Once every lease is dropped, pressure may evict it like any
    // other entry.
    drop(pinned);
    drop(again);
    for e in 1..6 {
        s.acquire(key(2, e)).unwrap();
    }
    let misses_before = s.stats().misses;
    s.acquire(key(1, 0)).unwrap();
    assert_eq!(
        s.stats().misses,
        misses_before + 1,
        "unpinned entry was evictable"
    );
}

/// The plan's "cache smaller than one token's expert union" case:
/// a budget that can't hold even one expert still serves correct
/// bytes on every acquire, as uncached pass-throughs.
#[test]
fn budget_smaller_than_one_expert_degrades_to_pass_through_not_failure() {
    let s = store(100, 50);
    for e in 0..4 {
        let lease = s.acquire(key(0, e)).unwrap();
        assert_eq!(lease.bytes(), expected_bytes(key(0, e), 100));
    }
    let st = s.stats();
    assert_eq!(st.pass_throughs, 4);
    assert_eq!(st.resident_bytes, 0);
    assert_eq!(st.evictions, 0);
}

/// If every resident entry is pinned, a new acquire must not
/// deadlock or evict a pinned entry -- it passes through.
#[test]
fn fully_pinned_cache_serves_new_experts_uncached() {
    let s = store(100, 200);
    let _a = s.acquire(key(0, 0)).unwrap();
    let _b = s.acquire(key(0, 1)).unwrap();
    let c = s.acquire(key(0, 2)).unwrap();
    assert_eq!(c.bytes(), expected_bytes(key(0, 2), 100));
    assert_eq!(s.stats().pass_throughs, 1);
    assert_eq!(s.stats().evictions, 0);
    // The two pinned entries are still hits.
    let hits_before = s.stats().hits;
    s.acquire(key(0, 0)).unwrap();
    s.acquire(key(0, 1)).unwrap();
    assert_eq!(s.stats().hits, hits_before + 2);
}

#[test]
fn missing_expert_is_a_clean_error() {
    let s = store(100, 1000);
    assert!(s.acquire(key(99, 0)).is_err());
}

/// Concurrent stress under a deliberately tiny budget: many
/// threads acquiring random keys while evictions churn constantly.
/// Every lease's bytes must match that key's expected pattern --
/// the direct test for slot-reuse corruption.
#[test]
fn concurrent_acquires_under_eviction_pressure_never_yield_wrong_bytes() {
    let s = Arc::new(store(64, 200)); // fits 3 of 16 hot keys
    let mut handles = Vec::new();
    for t in 0..8u32 {
        let s = Arc::clone(&s);
        handles.push(std::thread::spawn(move || {
            let mut state = t.wrapping_mul(2654435761).wrapping_add(12345);
            for _ in 0..500 {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let k = key((state >> 8) % 4, (state >> 16) % 4);
                let lease = s.acquire(k).expect("in-range key must read");
                assert_eq!(
                    lease.bytes(),
                    expected_bytes(k, 64),
                    "wrong bytes for {k:?} -- slot reuse corruption"
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let st = s.stats();
    assert_eq!(
        st.hits + st.misses,
        8 * 500,
        "every acquire is a hit or a miss"
    );
    assert!(st.resident_bytes <= 200, "budget held under concurrency");
}

/// The bridge to the weight types: a quantized `WeightMatrix` built
/// over a lease's shared buffer must (a) compute exactly what the
/// same bytes compute as an owned buffer, and (b) keep the cache
/// entry pinned for as long as the matrix lives, even after the
/// original lease is dropped.
#[test]
fn weight_matrix_over_a_lease_computes_identically_and_extends_the_pin() {
    use crate::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};

    // A source whose "expert" payload is real Q8_0 rows.
    struct QuantSource {
        rows: Vec<u8>,
    }
    impl ExpertSource for QuantSource {
        fn expert_len(&self, _k: ExpertKey) -> Option<usize> {
            Some(self.rows.len())
        }
        fn read_expert(&self, _k: ExpertKey) -> io::Result<Vec<u8>> {
            Ok(self.rows.clone())
        }
    }

    let cols = 64;
    let rows = 2;
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.11).sin())
        .collect();
    let mut packed = Vec::new();
    for r in 0..rows {
        packed.extend(frink_quant::quantize_q8_0(
            &values[r * cols..(r + 1) * cols],
        ));
    }
    let store = ExpertStore::new(
        QuantSource {
            rows: packed.clone(),
        },
        packed.len(),
    );

    let lease = store.acquire(key(0, 0)).unwrap();
    let matrix = WeightMatrix::Quantized {
        data: WeightBytes::Shared {
            buf: lease.shared_buf(),
            range: 0..packed.len(),
        },
        rows,
        cols,
        kind: QuantKind::Q8_0,
    };
    drop(lease); // the matrix alone must keep the entry pinned

    let owned = WeightMatrix::Quantized {
        data: WeightBytes::Owned(packed),
        rows,
        cols,
        kind: QuantKind::Q8_0,
    };
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.07).cos()).collect();
    assert_eq!(
        matrix.apply(&x),
        owned.apply(&x),
        "lease-backed == owned, bit for bit"
    );

    // Entry is still resident (pinned by the matrix): re-acquire is
    // a hit even though the budget is exactly one expert.
    let hits_before = store.stats().hits;
    store.acquire(key(0, 0)).unwrap();
    assert_eq!(store.stats().hits, hits_before + 1);
}

/// End-to-end through a real on-disk file: positional-range reads
/// through the bounded store return exactly the bytes written at
/// each expert's offset, including under concurrent access.
#[test]
fn file_range_source_reads_correct_ranges_through_the_store() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "frink_expert_store_test_{}.bin",
        std::process::id()
    ));
    let mut file_bytes = Vec::new();
    let mut ranges = HashMap::new();
    for l in 0..3u32 {
        for e in 0..4u32 {
            let content = expected_bytes(key(l, e), 96);
            ranges.insert(key(l, e), (file_bytes.len() as u64, content.len()));
            file_bytes.extend_from_slice(&content);
        }
    }
    std::fs::write(&path, &file_bytes).unwrap();
    let source = FileRangeSource::new(std::fs::File::open(&path).unwrap(), ranges);
    let store = Arc::new(ExpertStore::new(source, 300)); // fits 3 of 12

    let mut handles = Vec::new();
    for t in 0..4u32 {
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            for i in 0..200u32 {
                let k = key((t + i) % 3, i % 4);
                let lease = store.acquire(k).unwrap();
                assert_eq!(lease.bytes(), expected_bytes(k, 96));
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    std::fs::remove_file(&path).ok();
}
