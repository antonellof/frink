use super::*;

/// `blocks_needed_for` is the reservation the whole no-partial-write
/// guarantee rests on, and it is wrong in two opposite directions
/// that fail very differently.
///
/// UNDER-counting is the dangerous one: `append_contiguous` reserves
/// on this answer and then pushes with an `expect`, so too small a
/// number panics part-way through a layer -- exactly the corrupted
/// state the reservation exists to prevent. Over-counting merely
/// refuses a request that would have fitted.
///
/// Both mistakes are one edit away. Flooring instead of ceiling
/// under-counts whenever the append does not land on a block
/// boundary; ignoring the part-full tail over-counts whenever a
/// sequence is mid-block, which after the first token is almost
/// always. Neither shows up when the numbers happen to divide
/// evenly, so the cases here are chosen so that they do not.
#[test]
fn blocks_needed_for_accounts_for_the_part_full_tail_block() {
    let mut store = PagedKvStore::new(/* block_size = */ 4, 64, 1, 1);
    let mut cache = PagedKvCache::new();
    let row = [1.0f32];
    // Real pushes rather than poking `seq_len`: the count is
    // against blocks this sequence HOLDS, so a length with no
    // blocks behind it is a state that cannot occur and would only
    // let the test agree with an arithmetic nothing produces.
    let advance = |cache: &mut PagedKvCache, store: &mut PagedKvStore, n: usize| {
        for _ in 0..n {
            cache.push(store, &row, &row).unwrap();
        }
    };

    // Empty: a whole-block boundary, and a remainder that a floor
    // would round away.
    assert_eq!(cache.blocks_needed_for(&store, 0), 0);
    assert_eq!(cache.blocks_needed_for(&store, 1), 1);
    assert_eq!(cache.blocks_needed_for(&store, 4), 1);
    assert_eq!(cache.blocks_needed_for(&store, 5), 2, "5 into 4s needs 2");

    // One position in: three slots free in the tail, so appending up
    // to three costs NOTHING. Ignoring the tail would say 1.
    advance(&mut cache, &mut store, 1);
    assert_eq!(cache.blocks_needed_for(&store, 3), 0, "fits in the tail");
    assert_eq!(cache.blocks_needed_for(&store, 4), 1);
    assert_eq!(cache.blocks_needed_for(&store, 8), 2);

    // The awkward case: 1 free in the tail, 6 to append. 5 spill
    // over 4-wide blocks, so 2. A floor gives 1 and a tail-blind
    // ceil gives 2 for the wrong reason, so this pins the shape.
    advance(&mut cache, &mut store, 2); // seq_len = 3
    assert_eq!(cache.blocks_needed_for(&store, 6), 2);
    assert_eq!(cache.blocks_needed_for(&store, 5), 1);

    // Tail exactly full: no free slots, so this behaves like empty.
    advance(&mut cache, &mut store, 1); // seq_len = 4
    assert_eq!(cache.blocks_needed_for(&store, 1), 1);
    assert_eq!(cache.blocks_needed_for(&store, 4), 1);

    // A RESERVED block is capacity this sequence already holds, so
    // it must not be asked for twice. Counting from `seq_len` alone
    // would say 1 here and take a second block for positions the
    // reservation already covers.
    cache.reserve(&mut store, 4).unwrap();
    assert_eq!(
        cache.blocks_needed_for(&store, 4),
        0,
        "a reserved block is already held"
    );
    assert_eq!(cache.blocks_needed_for(&store, 5), 1);
}

/// A group takes one block from every layer, and gives them all
/// back together.
///
/// All-or-nothing is the point: a group holding blocks in some
/// layers and not others cannot answer "which block holds position
/// p in layer l", which is the only question it exists for.
#[test]
fn a_group_takes_one_block_from_every_layer_and_returns_them_together() {
    let kv = SharedPagedKv::new(3, 2, 4, 1, 1);
    assert_eq!(kv.free_groups(), 4);

    let g = kv.acquire_group().expect("4 groups available");
    let blocks = kv.group_blocks(g);
    assert_eq!(blocks.len(), 3, "one block per layer");
    for l in 0..3 {
        assert_eq!(kv.free_blocks(l), 3, "layer {l} gave up exactly one");
    }
    assert_eq!(kv.free_groups(), 3);

    assert!(kv.release_group(g), "sole holder, so this frees it");
    for l in 0..3 {
        assert_eq!(kv.free_blocks(l), 4, "layer {l} got its block back");
    }
    assert_eq!(kv.free_groups(), 4);
}

/// A group survives until its LAST holder releases it.
///
/// This is what makes prefix sharing safe. Two sequences off one
/// system prompt hold the same pages; if the first to finish freed
/// them, the second would keep attending over blocks the store had
/// already handed to somebody else -- surfacing as another
/// conversation's tokens, not as a crash.
#[test]
fn a_group_shared_by_two_holders_survives_the_first_release() {
    let kv = SharedPagedKv::new(2, 2, 2, 1, 1);
    let g = kv.acquire_group().unwrap();
    let blocks = kv.group_blocks(g);
    kv.retain_group(g);
    assert_eq!(kv.group_refs(g), 2);

    assert!(
        !kv.release_group(g),
        "one holder remains, so nothing is freed"
    );
    assert_eq!(kv.group_refs(g), 1);
    assert_eq!(kv.free_blocks(0), 1, "the blocks are still held");
    assert_eq!(kv.group_blocks(g), blocks, "and still name the same blocks");

    assert!(kv.release_group(g), "last holder frees it");
    assert_eq!(kv.group_refs(g), 0);
    assert_eq!(kv.free_blocks(0), 2);
}

/// Exhaustion is per group, bounded by the tightest layer.
///
/// A layer with one block left caps the whole pool at one more
/// group however much room the others have, because a group needs
/// one block from each.
#[test]
fn group_capacity_is_bounded_by_the_layer_with_the_fewest_blocks() {
    let kv = SharedPagedKv::from_stores(vec![
        PagedKvStore::new(2, 5, 1, 1),
        PagedKvStore::new(2, 1, 1, 1),
    ]);
    assert_eq!(kv.free_groups(), 1, "layer 1 has only one block");

    let g = kv.acquire_group().expect("one group fits");
    assert_eq!(kv.free_groups(), 0);
    assert!(
        kv.acquire_group().is_none(),
        "layer 1 is empty, so no group can be formed"
    );
    // The refused attempt must not have taken layer 0's block.
    assert_eq!(kv.free_blocks(0), 4, "a refused group leaks nothing");
    kv.release_group(g);
    assert_eq!(kv.free_blocks(0), 5);
}

/// A released id is reused, with a refcount that starts over.
#[test]
fn a_released_group_id_is_reused_with_a_fresh_refcount() {
    let kv = SharedPagedKv::new(1, 2, 2, 1, 1);
    let first = kv.acquire_group().unwrap();
    kv.retain_group(first);
    assert_eq!(kv.group_refs(first), 2);
    kv.release_group(first);
    kv.release_group(first);
    assert_eq!(kv.group_refs(first), 0, "gone, not merely decremented");

    let second = kv.acquire_group().unwrap();
    assert_eq!(second, first, "the id is reused");
    assert_eq!(
        kv.group_refs(second),
        1,
        "a reused id must not inherit the old count"
    );
    assert_eq!(kv.group_blocks(second).len(), 1);
    assert_eq!(kv.free_blocks(0), 1);
}

/// Reading a group after its last holder released it PANICS rather
/// than answering with stale blocks.
///
/// This is the observable half of clearing the entry on release,
/// and the reason it is `take` rather than `clone`: a caller still
/// holding a `PageGroup` after releasing it is exactly the bug
/// refcounting exists to prevent, and blocks that now belong to
/// somebody else are the worst possible answer -- the caller reads
/// another sequence's KV and nothing says so.
///
/// Written after sabotage showed the previous test here passed with
/// `clone` in place of `take`: `insert` overwrites the entry on
/// reuse, so a stale entry was never reachable through the path
/// that test took. This one reaches it.
#[test]
#[should_panic(expected = "already released")]
fn reading_a_released_group_panics_rather_than_returning_stale_blocks() {
    let kv = SharedPagedKv::new(2, 2, 2, 1, 1);
    let g = kv.acquire_group().unwrap();
    assert!(kv.release_group(g));
    let _ = kv.group_blocks(g);
}

/// Releasing a group nobody holds is a bug, not a no-op.
///
/// Silently ignoring it would let a double release return the same
/// blocks to the store twice, after which two sequences are handed
/// the same page and both write it.
#[test]
#[should_panic(expected = "double free of group")]
fn releasing_a_group_twice_panics_rather_than_freeing_it_twice() {
    let kv = SharedPagedKv::new(1, 2, 2, 1, 1);
    let g = kv.acquire_group().unwrap();
    assert!(kv.release_group(g));
    kv.release_group(g);
}

/// A recycled block backs a later position without the store ever
/// being asked for another one, and the later position's writes are
/// what a read at that position returns.
///
/// This is the whole sliding-window mechanism in miniature. Blocks
/// of two, four positions, and only two blocks in the store: without
/// recycling, position 2 has nowhere to go.
#[test]
fn a_recycled_block_backs_a_later_position_without_touching_the_store() {
    let mut store = PagedKvStore::new(2, 2, 1, 2);
    let mut cache = PagedKvCache::new();
    cache.push(&mut store, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
    cache.push(&mut store, &[2.0, 2.0], &[2.0, 2.0]).unwrap();
    assert_eq!(store.free_block_count(), 1, "one block per position pair");

    // Positions 0..2 have fallen behind a window of two. Their block
    // backs positions 2..4 instead, and the store is untouched.
    let recycled = cache.block_table()[0];
    cache.append_block(recycled);
    assert_eq!(
        store.free_block_count(),
        1,
        "recycling must not take a block from the store"
    );
    cache.push(&mut store, &[3.0, 3.0], &[3.0, 3.0]).unwrap();
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(
        cache.block_table(),
        &[recycled, recycled],
        "the same block at the stale index and the live one"
    );

    // Reading position 2 sees the new row. Position 0's row is gone,
    // which is exactly what "behind the window" means -- the kernel
    // never indexes it.
    let flat = cache.to_contiguous(&store);
    assert_eq!(
        &flat.k.as_f32().unwrap()[4..6],
        &[3.0, 3.0],
        "position 2 reads its own row"
    );
    assert_eq!(
        &flat.k.as_f32().unwrap()[0..2],
        &[3.0, 3.0],
        "position 0 now reads the recycled row, and nothing may read it"
    );
}

/// Releasing an aliased table hands each block back ONCE.
///
/// Per index instead of per distinct block would put the recycled id
/// on the free list twice, and the next two acquisitions would hand
/// two sequences the same memory -- which does not fail, it
/// interleaves two conversations' KV.
#[test]
fn releasing_a_recycled_table_gives_each_block_back_once() {
    // Exactly one block in the store, so "handed back twice" is
    // observable as a second acquisition succeeding.
    let mut store = PagedKvStore::new(2, 1, 1, 2);
    let mut cache = PagedKvCache::new();
    cache.push(&mut store, &[1.0, 1.0], &[1.0, 1.0]).unwrap();
    let held = cache.block_table()[0];
    cache.append_block(held);
    cache.append_block(held);

    let free_before = store.free_block_count();
    cache.release(&mut store);
    assert_eq!(
        store.free_block_count(),
        free_before + 1,
        "three table entries naming one block are one block back"
    );
    // And the store agrees: it can hand out that block once.
    assert!(store.acquire_block().is_some());
    assert!(store.acquire_block().is_none());
}

#[test]
fn a_gathered_sequence_round_trips_through_the_store() {
    let mut store = PagedKvStore::new(2, 8, 2, 2);
    let mut cache = PagedKvCache::new();
    // Five positions over blocks of two: the tail block is half
    // full, which is where an off-by-one in the gather shows up.
    let rows: Vec<[f32; 4]> = (0..5)
        .map(|i| {
            let b = i as f32 * 10.0;
            [b + 1.0, b + 2.0, b + 3.0, b + 4.0]
        })
        .collect();
    for r in &rows {
        cache.push(&mut store, r, r).unwrap();
    }

    let flat = cache.to_contiguous(&store);
    assert_eq!(flat.positions(), 5);
    assert_eq!(flat.k.len(), 5 * 4);
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(
            &flat.k.as_f32().unwrap()[i * 4..(i + 1) * 4],
            r,
            "position {i} k"
        );
        assert_eq!(
            &flat.v.as_f32().unwrap()[i * 4..(i + 1) * 4],
            r,
            "position {i} v"
        );
    }

    // And appending those same rows back onto a fresh sequence
    // reproduces the store's view of them exactly.
    let mut rebuilt = PagedKvCache::new();
    let mut store2 = PagedKvStore::new(2, 8, 2, 2);
    rebuilt
        .append_contiguous(&mut store2, &flat.k, &flat.v, 5)
        .unwrap();
    let again = rebuilt.to_contiguous(&store2);
    assert_eq!(again.k.to_f32_vec(), flat.k.to_f32_vec());
    assert_eq!(again.v.to_f32_vec(), flat.v.to_f32_vec());
    assert_eq!(again.positions(), flat.positions());
}
