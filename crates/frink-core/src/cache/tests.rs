use super::*;
use crate::kv_store::KvDtype;

/// The f16 store holds half the bytes of the f32 one for the same
/// rows, and every edit a cache makes (push, truncate, evict, clone)
/// keeps its dtype and its rounded values.
#[test]
fn an_f16_cache_is_half_the_bytes_and_keeps_its_dtype_through_every_edit() {
    let rows: Vec<Vec<f32>> = (0..6)
        .map(|p| (0..8).map(|i| 0.1 * (p * 8 + i) as f32).collect())
        .collect();
    let mut a = KvCache::with_capacity(2, 4, 6);
    let mut b = KvCache::with_capacity(2, 4, 6).with_dtype(KvDtype::F16);
    for r in &rows {
        a.push(r, r).unwrap();
        b.push(r, r).unwrap();
    }
    assert_eq!(b.dtype(), KvDtype::F16);
    assert_eq!(b.allocated_bytes() * 2, a.allocated_bytes());
    let want: Vec<f32> = rows
        .iter()
        .flatten()
        .map(|&x| half::f16::from_f32(x).to_f32())
        .collect();
    assert_eq!(b.k.to_f32_vec(), want);
    assert_ne!(
        b.k.to_f32_vec(),
        a.k.to_f32_vec(),
        "0.1 * n is not exact in f16"
    );
    b.truncate(4);
    assert_eq!(b.rows(), 4);
    assert_eq!(b.v.to_f32_vec(), want[..32].to_vec());
    let c = b.clone();
    assert_eq!(c.dtype(), KvDtype::F16);
    assert_eq!(c.k_rows(), b.k_rows());
}

#[test]
fn push_grows_seq_len_and_stores_values() {
    let mut cache = KvCache::new(2, 2);
    cache
        .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
        .unwrap();
    assert_eq!(cache.positions(), 1);
    cache
        .push(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0])
        .unwrap();
    assert_eq!(cache.positions(), 2);
    assert_eq!(cache.k.len(), 2 * 2 * 2);
    assert_eq!(cache.k.as_f32().unwrap()[4], 9.0);
}

#[test]
#[should_panic]
fn push_wrong_size_panics() {
    let mut cache = KvCache::new(2, 2);
    let _ = cache.push(&[1.0, 2.0], &[1.0, 2.0]); // too short
}

#[test]
fn clear_resets_state() {
    let mut cache = KvCache::new(1, 1);
    cache.push(&[1.0], &[2.0]).unwrap();
    cache.clear();
    assert_eq!(cache.positions(), 0);
    assert!(cache.k.is_empty());
}

#[test]
fn truncate_rolls_back_to_exact_length_preserving_earlier_data() {
    let mut cache = KvCache::new(2, 2);
    cache
        .push(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0])
        .unwrap();
    cache
        .push(&[5.0, 6.0, 7.0, 8.0], &[50.0, 60.0, 70.0, 80.0])
        .unwrap();
    cache
        .push(&[9.0, 9.0, 9.0, 9.0], &[90.0, 90.0, 90.0, 90.0])
        .unwrap();
    assert_eq!(cache.positions(), 3);

    cache.truncate(1);
    assert_eq!(cache.positions(), 1);
    assert_eq!(cache.k.to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(cache.v.to_f32_vec(), vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn truncate_to_current_length_is_a_no_op() {
    let mut cache = KvCache::new(1, 2);
    cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.truncate(1);
    assert_eq!(cache.positions(), 1);
    assert_eq!(cache.k.to_f32_vec(), vec![1.0, 2.0]);
}

#[test]
fn truncate_to_zero_empties_the_cache() {
    let mut cache = KvCache::new(1, 2);
    cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.truncate(0);
    assert_eq!(cache.positions(), 0);
    assert!(cache.k.is_empty());
    assert!(cache.v.is_empty());
}

#[test]
#[should_panic]
fn truncate_beyond_current_length_panics() {
    let mut cache = KvCache::new(1, 2);
    cache.push(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    cache.truncate(5);
}

#[test]
fn push_after_truncate_continues_correctly() {
    let mut cache = KvCache::new(1, 1);
    cache.push(&[1.0], &[10.0]).unwrap();
    cache.push(&[2.0], &[20.0]).unwrap();
    cache.push(&[3.0], &[30.0]).unwrap(); // this one will be "rejected"
    cache.truncate(2);
    cache.push(&[99.0], &[990.0]).unwrap(); // real continuation after rejection
    assert_eq!(cache.positions(), 3);
    assert_eq!(cache.k.to_f32_vec(), vec![1.0, 2.0, 99.0]);
    assert_eq!(cache.v.to_f32_vec(), vec![10.0, 20.0, 990.0]);
}

#[test]
fn with_capacity_preallocates_and_never_reallocates_within_plan() {
    let n_kv_heads = 4;
    let head_dim = 8;
    let max_seq_len = 16;
    let mut cache = KvCache::with_capacity(n_kv_heads, head_dim, max_seq_len);

    let expected_elems = max_seq_len * n_kv_heads * head_dim;
    assert!(cache.k.capacity() >= expected_elems);
    assert!(cache.v.capacity() >= expected_elems);

    let step = vec![0.5f32; n_kv_heads * head_dim];
    let k_ptr_before = cache.k.as_f32().unwrap().as_ptr();
    for _ in 0..max_seq_len {
        cache.push(&step, &step).unwrap();
    }
    let k_ptr_after = cache.k.as_f32().unwrap().as_ptr();
    assert_eq!(
        k_ptr_before, k_ptr_after,
        "pushing exactly up to the planned capacity must not reallocate"
    );
    assert!(cache.is_within_planned_capacity());
}

#[test]
fn allocated_bytes_reflects_preallocated_capacity_not_just_used_length() {
    let cache = KvCache::with_capacity(4, 8, 100);
    // 100 positions * 4 kv_heads * 8 head_dim * 2 (k+v) * 4 bytes/f32
    let expected_min = 100 * 4 * 8 * 2 * 4;
    assert!(
        cache.allocated_bytes() >= expected_min,
        "allocated_bytes={} expected_min={expected_min}",
        cache.allocated_bytes()
    );
    // Nothing has been pushed yet, but the memory is already reserved.
    assert_eq!(cache.positions(), 0);
}

#[test]
fn grow_as_you_go_cache_reports_not_within_planned_capacity() {
    let mut cache = KvCache::new(2, 2);
    cache
        .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
        .unwrap();
    assert!(
        !cache.is_within_planned_capacity(),
        "a cache built with `new` has no plan to be within"
    );
}

#[test]
fn with_pool_acquires_one_block_and_reports_it_in_free_blocks() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 10)));
    let cache = KvCache::with_pool(2, 2, pool.clone(), 0).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 9);
    assert_eq!(cache.positions(), 0);
}

#[test]
fn with_pool_fails_without_mutating_the_pool_when_exhausted() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 0)));
    let result = KvCache::with_pool(2, 2, pool.clone(), 0);
    assert!(result.is_err());
    assert_eq!(pool.lock().unwrap().free_blocks(), 0);
}

#[test]
fn push_acquires_additional_blocks_as_the_cache_crosses_block_boundaries() {
    let block_size = 2;
    let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 10)));
    let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 9);

    // First block holds `block_size` = 2 positions; pushing them
    // must not need a second block.
    cache.push(&[1.0], &[1.0]).unwrap();
    cache.push(&[2.0], &[2.0]).unwrap();
    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        9,
        "filling exactly the first block must not acquire a second one"
    );

    // The third position crosses into a second block.
    cache.push(&[3.0], &[3.0]).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 8);
    assert_eq!(cache.positions(), 3);
    assert_eq!(cache.k.to_f32_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn push_returns_pool_exhausted_and_leaves_state_unchanged_when_no_blocks_remain() {
    let block_size = 1;
    let pool = Arc::new(Mutex::new(KvBlockPool::new(block_size, 1)));
    let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 0);

    cache.push(&[1.0], &[1.0]).unwrap(); // fills the one held block

    let before_k = cache.k.clone();
    let result = cache.push(&[2.0], &[2.0]);
    assert_eq!(result, Err(KvPoolExhausted));
    assert_eq!(
        cache.positions(),
        1,
        "a failed push must not change seq_len"
    );
    assert_eq!(
        cache.k.to_f32_vec(),
        before_k.to_f32_vec(),
        "a failed push must not append data"
    );
}

#[test]
fn dropping_a_pooled_cache_returns_its_blocks_to_the_pool() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 2)));
    {
        let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
        cache.push(&[1.0], &[1.0]).unwrap(); // fills the first (only held) block
        cache.push(&[2.0], &[2.0]).unwrap(); // crosses into a second block
        assert_eq!(pool.lock().unwrap().free_blocks(), 0);
    }
    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        2,
        "both blocks held by the dropped cache must return to the pool"
    );
}

#[test]
fn release_to_pool_is_explicit_and_idempotent() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 5)));
    let mut cache = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 4);

    cache.release_to_pool();
    assert_eq!(pool.lock().unwrap().free_blocks(), 5);

    cache.release_to_pool(); // no-op, must not over-release
    assert_eq!(pool.lock().unwrap().free_blocks(), 5);

    drop(cache); // must not release again either
    assert_eq!(pool.lock().unwrap().free_blocks(), 5);
}

#[test]
fn two_pooled_caches_share_one_bounded_budget() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(1, 1)));
    let cache_a = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
    let cache_b = KvCache::with_pool(1, 1, pool.clone(), 0);
    assert!(
        cache_b.is_err(),
        "a second concurrent request must not be admitted when the shared budget is full"
    );

    drop(cache_a);
    let cache_c = KvCache::with_pool(1, 1, pool, 0);
    assert!(
        cache_c.is_ok(),
        "once the first request's cache is dropped, its budget must become available again"
    );
}

#[test]
fn cloning_a_pooled_cache_detaches_the_clone_from_pool_accounting() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 3)));
    let original = KvCache::with_pool(1, 1, pool.clone(), 0).unwrap();
    assert_eq!(pool.lock().unwrap().free_blocks(), 2);

    let clone = original.clone();
    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        2,
        "cloning must not acquire additional blocks"
    );
    assert_eq!(clone.k.to_f32_vec(), original.k.to_f32_vec());

    drop(clone);
    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        2,
        "dropping a detached clone must not release the original's blocks"
    );

    drop(original);
    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        3,
        "dropping the original must release its blocks exactly once"
    );
}

/// A resize is arithmetic, and the one rule that is not: shrinking
/// past what is held is refused, and the pool is left exactly as it
/// was.
///
/// Clamping to zero instead would silently over-promise -- the
/// caches holding those blocks do not give them back, so every
/// later acquire would decide against a budget that does not
/// describe the memory in use. This test fails under that clamp.
#[test]
fn a_pool_refuses_to_shrink_below_what_is_already_held() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 10)));
    let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 24).expect("blocks");
    let in_use = {
        let p = pool.lock().unwrap();
        p.total_blocks() - p.free_blocks()
    };
    assert!(in_use > 0, "the fixture must actually hold blocks");

    let mut p = pool.lock().unwrap();
    assert_eq!(p.resize(in_use - 1), Err(in_use));
    assert_eq!(p.total_blocks(), 10, "a refused resize changes nothing");
    assert_eq!(p.free_blocks(), 10 - in_use);

    // Down to exactly what is held is legal, and leaves nothing free.
    assert_eq!(p.resize(in_use), Ok(()));
    assert_eq!(p.free_blocks(), 0);
    drop(p);
    drop(held);
}

/// Growing hands the new blocks to the free list without disturbing
/// what is held, which is the whole point of a live re-split.
#[test]
fn growing_a_pool_adds_to_what_is_free_and_not_to_what_is_held() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 8)));
    let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 16).expect("blocks");
    let mut p = pool.lock().unwrap();
    let in_use = p.total_blocks() - p.free_blocks();

    assert_eq!(p.resize(32), Ok(()));
    assert_eq!(p.total_blocks(), 32);
    assert_eq!(p.free_blocks(), 32 - in_use);
    drop(p);
    drop(held);
}

/// Positions and rows agree for every store that does not evict,
/// and this pins that they are DERIVED separately rather than one
/// being an alias for the other.
///
/// The point of the split is #61: a windowed layer will drop rows
/// behind its window while its position count keeps climbing. Until
/// then the two are equal, so this test cannot prove much about
/// eviction. What it CAN prove, and what matters, is that `rows`
/// reads the buffer rather than the counter: set the counter to a
/// lie and `rows` still reports the truth.
#[test]
fn rows_is_read_from_the_buffer_and_positions_from_the_counter() {
    let mut cache = KvCache::new(2, 4);
    for i in 0..5 {
        let step = vec![i as f32; 8];
        cache.push(&step, &step).expect("unbounded growth");
    }
    assert_eq!(cache.positions(), 5);
    assert_eq!(cache.rows(), 5, "nothing evicts, so they agree");

    // The counter lies; the buffer does not.
    cache.force_positions_for_test(99);
    assert_eq!(cache.positions(), 99);
    assert_eq!(
        cache.rows(),
        5,
        "rows must come from k/v, or it is just a second name for the counter"
    );
}

/// `truncate` is measured in POSITIONS, and it moves both, because
/// nothing evicts yet. Written down because it is the first thing
/// eviction changes: a truncate target will stop being a row index.
#[test]
fn truncate_moves_positions_and_rows_together_while_nothing_evicts() {
    let mut cache = KvCache::new(1, 2);
    for i in 0..4 {
        let step = vec![i as f32; 2];
        cache.push(&step, &step).expect("unbounded growth");
    }
    cache.truncate(2);
    assert_eq!(cache.positions(), 2);
    assert_eq!(cache.rows(), 2);
    assert_eq!(cache.k.len(), 4, "two positions of two elements each");
}

/// The store's rule must be *the* rule, not a second copy of it.
///
/// `KvWindow::rows_after` is what `frink_models::kv_budget` prices
/// against. If the store drained to anything else the budget would
/// be describing a cache that does not exist, which is #33 again.
#[test]
fn a_windowed_cache_holds_exactly_what_the_rule_says_it_holds() {
    let window = KvWindow::new(8, 3).expect("positive window");
    let mut cache = KvCache::new(2, 4);
    cache.arm_window(window);
    let step = vec![1.0f32; 8];
    for p in 1..=200usize {
        cache.push(&step, &step).expect("unbounded growth");
        cache.evict_behind_window();
        assert_eq!(cache.positions(), p);
        assert_eq!(
            cache.rows(),
            window.rows_after(p),
            "at {p} positions the store and the rule disagree"
        );
    }
}

/// The point of the issue: positions keep counting, rows do not.
#[test]
fn a_windowed_layers_resident_rows_stop_growing() {
    let window = KvWindow::with_default_slack(16).expect("positive window");
    let mut cache = KvCache::new(2, 4);
    cache.arm_window(window);
    let step = vec![0.5f32; 8];
    for _ in 0..2000 {
        cache.push(&step, &step).expect("unbounded growth");
        cache.evict_behind_window();
    }
    assert_eq!(cache.positions(), 2000);
    assert!(
        cache.rows() <= window.max_rows(),
        "{} rows resident after 2000 positions",
        cache.rows()
    );
    // And the bytes really went back, not just the length: `drain`
    // frees nothing, and memory is the whole point.
    assert!(
        cache.allocated_bytes() <= window.max_rows() * 8 * 2 * 4 * 2,
        "capacity was never handed back: {} bytes",
        cache.allocated_bytes()
    );
}

/// **The correctness argument, measured.**
///
/// Eviction is only allowed to be token-identical because the rows a
/// windowed kernel reads -- the last `window` of them -- are the
/// same bytes whether or not anything behind them was dropped. This
/// pushes distinguishable rows into an evicting cache and a plain
/// one and compares exactly that slice.
#[test]
fn the_rows_a_windowed_kernel_reads_are_identical_with_and_without_eviction() {
    let window = KvWindow::new(6, 2).expect("positive window");
    let (n_kv_heads, head_dim) = (2usize, 4usize);
    let per = n_kv_heads * head_dim;
    let mut evicting = KvCache::new(n_kv_heads, head_dim);
    evicting.arm_window(window);
    let mut plain = KvCache::new(n_kv_heads, head_dim);

    for p in 0..120usize {
        // A row nothing else could produce, so a misplaced row is
        // not merely a different number but an identifiable one.
        let k: Vec<f32> = (0..per).map(|i| (p * 100 + i) as f32).collect();
        let v: Vec<f32> = k.iter().map(|x| -x).collect();
        evicting.push(&k, &v).expect("unbounded growth");
        evicting.evict_behind_window();
        plain.push(&k, &v).expect("unbounded growth");

        let read = window.window().min(p + 1);
        let e_start = (evicting.rows() - read) * per;
        let p_start = (plain.rows() - read) * per;
        assert_eq!(
            &evicting.k.as_f32().unwrap()[e_start..],
            &plain.k.as_f32().unwrap()[p_start..],
            "K read set diverged at position {p}"
        );
        assert_eq!(
            &evicting.v.as_f32().unwrap()[e_start..],
            &plain.v.as_f32().unwrap()[p_start..],
            "V read set diverged at position {p}"
        );
        assert_eq!(evicting.positions(), plain.positions());
    }
}

/// An unarmed cache is the cache this engine has always had.
#[test]
fn an_unarmed_cache_never_drops_a_row() {
    let mut cache = KvCache::new(2, 4);
    let step = vec![1.0f32; 8];
    for _ in 0..64 {
        cache.push(&step, &step).expect("unbounded growth");
        assert_eq!(cache.evict_behind_window(), 0);
    }
    assert_eq!(cache.rows(), 64);
    assert_eq!(cache.positions(), 64);
    assert!(cache.window().is_none());
}

/// Speculative decoding rolls the cache back by the number of
/// rejected draft tokens. That is representable while the target is
/// still resident.
#[test]
fn truncate_still_works_inside_the_resident_window() {
    let window = KvWindow::new(8, 3).expect("positive window");
    let mut cache = KvCache::new(1, 2);
    cache.arm_window(window);
    let step = vec![1.0f32; 2];
    for _ in 0..50 {
        cache.push(&step, &step).expect("unbounded growth");
        cache.evict_behind_window();
    }
    let rows_before = cache.rows();
    cache.truncate(46);
    assert_eq!(cache.positions(), 46);
    assert_eq!(cache.rows(), rows_before - 4);
}

/// ...and stops, loudly, when it is not. A cache that rolled back to
/// the oldest row it happened to have would answer the next token
/// out of a history with a hole in it.
#[test]
#[should_panic(expected = "rows are resident")]
fn truncate_refuses_to_roll_back_past_what_the_window_kept() {
    let window = KvWindow::new(4, 1).expect("positive window");
    let mut cache = KvCache::new(1, 2);
    cache.arm_window(window);
    let step = vec![1.0f32; 2];
    for _ in 0..50 {
        cache.push(&step, &step).expect("unbounded growth");
        cache.evict_behind_window();
    }
    cache.truncate(3);
}

/// A clone of an evicting cache is still an evicting cache. Reset
/// the window on clone and `positions` becomes a row count again in
/// whatever reads the clone.
#[test]
fn a_clone_carries_the_window() {
    let window = KvWindow::new(4, 1).expect("positive window");
    let mut cache = KvCache::new(1, 2);
    cache.arm_window(window);
    let step = vec![1.0f32; 2];
    for _ in 0..20 {
        cache.push(&step, &step).expect("unbounded growth");
        cache.evict_behind_window();
    }
    let copy = cache.clone();
    assert_eq!(copy.window(), Some(window));
    assert_eq!(copy.rows(), cache.rows());
    assert_eq!(copy.positions(), cache.positions());
}

/// A pool-backed cache grows when its BUFFER is full, not when its
/// position counter reaches the buffer's size. Those are the same
/// number until something evicts, and an evicting pooled cache keyed
/// on positions would acquire a block per token forever and exhaust
/// the pool.
#[test]
fn a_pooled_windowed_cache_stops_acquiring_blocks() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(4, 8)));
    let mut cache =
        KvCache::with_pool(1, 2, Arc::clone(&pool), 8).expect("the pool was sized for this");
    cache.arm_window(KvWindow::new(4, 1).expect("positive window"));
    let step = vec![1.0f32; 2];
    for _ in 0..64 {
        cache
            .push(&step, &step)
            .expect("the buffer never fills again");
        cache.evict_behind_window();
    }
    assert_eq!(cache.positions(), 64);
    assert!(cache.rows() <= 5);
}

/// **The sibling above passes for the wrong reason on its own
/// sizing, so this is the same property where the trap is armed.**
///
/// Eviction hands surplus CAPACITY back with `shrink_to`, which is
/// where its memory saving actually comes from. For a pool-backed
/// cache that is a bug rather than a saving: the pool has already
/// promised these blocks and does not take them back, while `push`
/// decides it needs another block by comparing rows against
/// `k.capacity()`. Shrink that capacity to a window and the
/// condition is true again every `slack + 1` tokens, forever -- so
/// the cache draws a fresh block from the shared pool on a cadence,
/// never releases one, and runs the pool dry underneath every OTHER
/// request in the process. It surfaces where `push` is documented
/// infallible, so the caller panics mid-answer.
///
/// The sibling's cache is 8 positions against a 5-row ceiling,
/// which is under the 2x threshold `shrink_to` is gated on, so it
/// never reaches the shrink at all. This one is 64 positions
/// against the same ceiling, and the pool is given spare blocks so
/// the leak shows up as blocks quietly gone rather than only as the
/// error at the end of them.
#[test]
fn an_evicting_pooled_cache_never_hands_its_capacity_back_to_reacquire_it() {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(8, 24)));
    let mut cache = KvCache::with_pool(1, 2, Arc::clone(&pool), 64).expect("8 of the 24 blocks");
    let free_after_construction = pool.lock().unwrap().free_blocks();
    assert_eq!(free_after_construction, 16, "8 blocks cover 64 positions");

    cache.arm_window(KvWindow::new(4, 1).expect("positive window"));
    let step = vec![1.0f32; 2];
    for _ in 0..512 {
        cache
            .push(&step, &step)
            .expect("a reservation made up front must not be re-made per token");
        cache.evict_behind_window();
    }

    assert_eq!(
        pool.lock().unwrap().free_blocks(),
        free_after_construction,
        "the cache drew more blocks from the shared pool while evicting"
    );
    assert_eq!(cache.positions(), 512);
    assert!(cache.rows() <= 5);
}
