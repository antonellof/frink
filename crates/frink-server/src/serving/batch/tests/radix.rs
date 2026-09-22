//! What the batched path owes the radix prefix cache.
//!
//! Its own group, because the thing under test is neither admission nor
//! the tick: it is the CONTRACT between a finished row and the tree --
//! publish what you computed, adopt what someone else did, and give the
//! pages back when the pool runs dry. All three have to hold together
//! or the cache is worse than not having one.
//!
//! The bug these were written against: the batched path ADOPTED from
//! the tree and never contributed to it, because the prompt ids were
//! dropped at the prefill-to-decode handover and publishing needs the
//! whole sequence. Prefix sharing under `FRINK_CONTINUOUS_BATCHING=1`
//! therefore ran against a tree nothing filled.

use std::sync::mpsc;
use std::sync::Mutex;

use frink_core::cache::{PageGroup, SharedPagedKv};

use crate::generate::{acquire_paged_caches, PrefixIntent};
use crate::policy::radix::SaltedRadix;

use super::super::row::{Job, RowKv, Rows};
use super::super::worker::accept;
use super::*;

const BLOCK: usize = 4;

/// A paged config over a store of `groups` page groups, with a fresh
/// radix tree the test can inspect.
fn paged_with_radix(
    decoder: &Arc<Decoder>,
    groups: usize,
) -> (PagedKvConfig, Arc<SharedPagedKv>, Arc<Mutex<SaltedRadix>>) {
    let radix = Arc::new(Mutex::new(SaltedRadix::new(BLOCK)));
    let store = Arc::new(SharedPagedKv::new(
        decoder.layers.len(),
        BLOCK,
        /* blocks_per_layer = */ groups,
        decoder.config.n_kv_heads,
        decoder.config.head_dim,
    ));
    let config = PagedKvConfig {
        store: Arc::clone(&store),
        queue_wait: std::time::Duration::ZERO,
        radix: Some(Arc::clone(&radix)),
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    (config, store, radix)
}

fn paged_job(prompt: Vec<usize>, max_tokens: usize) -> (Job, mpsc::Receiver<BatcherEvent>) {
    let (tx, rx) = mpsc::channel();
    (
        Job {
            prompt_tokens: prompt,
            params: greedy_params(max_tokens, 7),
            stop_tokens: StopTokens::default(),
            reply: tx,
            abort: AbortId(0),
            blocks: 1,
        },
        rx,
    )
}

/// Runs one request all the way to an admitted, prefilled row.
///
/// Deliberately the REAL path -- `accept` acquires the lease (and so
/// consults the tree), `step_chunk` runs every prompt token, and
/// `into_slot` performs the prefill-to-decode handover that used to
/// drop the prompt ids. A test that built a `Slot` by hand would prove
/// nothing about the handover, which is where the bug lived.
fn admit_prefilled(
    decoder: &Arc<Decoder>,
    config: &PagedKvConfig,
    prompt: Vec<usize>,
    max_tokens: usize,
) -> Option<(Slot, mpsc::Receiver<BatcherEvent>)> {
    let (job, rx) = paged_job(prompt, max_tokens);
    let mut prefill = accept(decoder, job, /* chunk_size = */ 1, Some(config))?;
    while !prefill.state.step_chunk() {}
    Some((prefill.into_slot(), rx))
}

/// Finishes a row through the one path every finished row takes, which
/// is also the only place the batched publish happens.
fn finish_row(slot: Slot) {
    let mut rows = Rows::default();
    let uid = rows.insert(slot);
    rows.get_mut(uid).expect("just inserted").finish = Some(FinishReason::Length);
    rows.flush_finished(&no_budget());
}

/// The page groups the tree holds for `prompt`, one per block.
fn published_groups(radix: &Arc<Mutex<SaltedRadix>>, prompt: &[usize]) -> (usize, Vec<u32>) {
    let ids: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let mut tree = radix.lock().unwrap();
    let m = tree.match_prefix(None, &ids);
    if m.cached_len == 0 {
        return (0, Vec::new());
    }
    let per_token = tree.matched_indices(m.handle);
    (
        m.cached_len,
        per_token[..m.cached_len]
            .iter()
            .copied()
            .step_by(BLOCK)
            .collect(),
    )
}

/// A finished batched row must PUBLISH its prefix, not just adopt one.
///
/// Asserts the TREE grew, which is the thing that was missing. A test
/// that only checked the second request was fast would pass on a warm
/// page cache.
#[test]
fn a_finished_batched_row_publishes_its_prefix() {
    let decoder = tiny_decoder();
    let (config, _store, radix) = paged_with_radix(&decoder, 64);
    assert_eq!(
        radix.lock().unwrap().total_size(),
        0,
        "the tree starts empty"
    );

    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
    let (slot, _rx) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(slot);

    let (cached, groups) = published_groups(&radix, &prompt);
    assert_eq!(
        cached,
        prompt.len(),
        "a finished paged row must leave its whole prefix in the tree"
    );
    assert_eq!(groups.len(), prompt.len() / BLOCK);
}

/// The point of publishing: the NEXT request adopts those very pages.
///
/// Page IDENTITY, not merely a hit count. The assertion that carries
/// the test is the refcount: while the second row is alive, each group
/// the first published has two holders -- the tree and the second row
/// -- which is only true if the second row is attending over the first
/// row's actual pages rather than fresh copies of them.
#[test]
fn a_second_batched_request_adopts_the_pages_the_first_published() {
    let decoder = tiny_decoder();
    let (config, store, radix) = paged_with_radix(&decoder, 64);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    let (first, _rx1) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(first);

    let (cached, published) = published_groups(&radix, &prompt);
    assert_eq!(cached, prompt.len(), "the first row must have published");
    for &g in &published {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            1,
            "with no row alive, only the tree holds a published page"
        );
    }

    let (second, _rx2) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    let RowKv::Paged(lease) = &second.kv else {
        panic!("a paged config must produce a paged row");
    };
    // Never the whole prompt: prefill has to run over at least one
    // token to produce the logits that predict the next one.
    let adopted = lease.adopted_positions(BLOCK);
    assert!(
        adopted > 0,
        "the second request must adopt the prefix the first published, \
         got {adopted} adopted positions"
    );

    for &g in published.iter().take(adopted / BLOCK) {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            2,
            "an adopted page is held by the tree AND by the row reading it"
        );
    }
    drop(second);
    for &g in &published {
        assert_eq!(
            store.group_refs(PageGroup(g)),
            1,
            "the row's hold goes back on Drop; the tree's survives"
        );
    }
}

/// A cache HIT must not re-run the prompt on top of the prefix it hit.
///
/// The failure this pins down (issue #37): the batched prefill started
/// at token 0 whatever its lease had already adopted, so the second of
/// two identical requests wrote its eight prompt tokens into rows
/// `4..11` of a cache that already held `0..3`, while handing the model
/// RoPE positions `0..7`. The KV ended `prompt + cached` long instead of
/// `prompt` long, decode then pushed past the block table the lease
/// reserved (a per-request page leak, and `PagedStoreExhausted` into an
/// `.expect` on a tight pool), and the answer was simply different.
///
/// `seq_len` is the assertion because it is the one number that cannot
/// be right by accident: it is where `push` writes, so a cache that is
/// `cached_len` too long has put every prompt token in the wrong row.
/// The existing tests here assert page REFCOUNTS, which the bug left
/// perfectly correct.
#[test]
fn a_batched_prefill_over_an_adopted_prefix_ends_exactly_at_the_prompt_length() {
    let decoder = tiny_decoder();
    let (config, _store, _radix) = paged_with_radix(&decoder, 64);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    let (first, _rx1) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(first);

    let (job, _rx2) = paged_job(prompt.clone(), 4);
    let mut prefill = accept(&decoder, job, /* chunk_size = */ 3, Some(&config)).expect("admitted");

    // The starting cursor is the adopted prefix, not zero.
    let adopted = prefill.state.tokens_processed();
    assert_eq!(
        adopted, BLOCK,
        "the second request must start where the prefix it adopted ends"
    );
    assert_eq!(prefill.state.tokens_remaining(), prompt.len() - BLOCK);

    while !prefill.state.step_chunk() {}
    let mut slot = prefill.into_slot();
    assert_eq!(
        slot.pos,
        prompt.len(),
        "the first generated token sits at the end of the prompt"
    );
    let RowKv::Paged(lease) = &mut slot.kv else {
        panic!("a paged config must produce a paged row");
    };
    for (layer, cache) in lease.caches_mut().iter().enumerate() {
        assert_eq!(
            cache.seq_len(),
            prompt.len(),
            "layer {layer}: a warm prefill left {} KV rows for a {}-token \
             prompt, so it re-ran the prompt on top of the adopted prefix",
            cache.seq_len(),
            prompt.len()
        );
    }
}

/// A warm request must produce the SAME logits as a cold one.
///
/// The user-visible half of issue #37: same model, same body, a
/// different answer, 200 OK, no log line. Bit-identical is the right
/// bar here -- both runs go through the same paged kernel over the same
/// page contents, so the only thing that can differ is which row a
/// token was written into and which position it was told it had.
#[test]
fn a_warm_batched_request_produces_the_same_logits_as_a_cold_one() {
    let decoder = tiny_decoder();
    let (config, _store, _radix) = paged_with_radix(&decoder, 64);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    let cold = {
        let (job, _rx) = paged_job(prompt.clone(), 4);
        let mut prefill = accept(&decoder, job, 3, Some(&config)).expect("admitted");
        assert_eq!(
            prefill.state.tokens_processed(),
            0,
            "the first request has nothing to adopt"
        );
        while !prefill.state.step_chunk() {}
        let (_kv, logits, _pos, _ids) = prefill.state.into_decode_start();
        logits
    };
    // Publish, so the second request has something to adopt.
    let (first, _rx1) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(first);

    let (job, _rx2) = paged_job(prompt.clone(), 4);
    let mut prefill = accept(&decoder, job, 3, Some(&config)).expect("admitted");
    assert!(
        prefill.state.tokens_processed() > 0,
        "this test proves nothing unless the second request adopted a prefix"
    );
    while !prefill.state.step_chunk() {}
    let (_kv, warm, _pos, _ids) = prefill.state.into_decode_start();

    assert_eq!(
        warm, cold,
        "a prefix-cache hit changed the answer: the warm request's logits \
         differ from the cold request's for the same prompt"
    );
}

/// The lease's own count of adopted positions and the KV rows it was
/// seeded with must be the same number.
///
/// Two structures that have to agree, walked against each other. The
/// batched prefill reads the cursor off the KV and the private generate
/// loop derives it from `PagedLease::adopted_positions`; if
/// `acquire_paged_caches` ever seeds one without the other, this goes
/// red here rather than as a wrong answer in production.
#[test]
fn the_leases_adopted_count_and_its_seeded_kv_rows_agree() {
    let decoder = tiny_decoder();
    let (config, _store, _radix) = paged_with_radix(&decoder, 64);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    let (first, _rx1) = admit_prefilled(&decoder, &config, prompt.clone(), 4).expect("admitted");
    finish_row(first);

    let mut lease = acquire_paged_caches(
        &decoder,
        &config,
        &prompt,
        prompt.len() + 4,
        PrefixIntent::default(),
    )
    .expect("the store has pages to spare");
    let claimed = lease.adopted_positions(BLOCK);
    assert!(claimed > 0, "the second request must adopt something");
    for (layer, cache) in lease.caches_mut().iter().enumerate() {
        assert_eq!(
            cache.seq_len(),
            claimed,
            "layer {layer}: the lease says {claimed} positions were adopted \
             but its KV was seeded with {} rows",
            cache.seq_len()
        );
    }
}

/// Publishing must not make the pool shrink monotonically.
///
/// `publish_to_radix` retains a group for every page it hands the tree.
/// Nothing released them, so a long-running server ended up refusing
/// requests that fit while the tree sat on pages no request was
/// reading. The batched path acquires through the same
/// `acquire_paged_caches` as the private one, so the eviction wired
/// into its retry loop has to cover batched rows too.
///
/// Sized so the arithmetic is the assertion: the pool holds 8 groups,
/// each request needs 3 and leaves 2 in the tree, so a tree that is
/// never evicted from starves the fourth request. Twelve run here.
#[test]
fn batched_publishing_does_not_starve_the_page_pool() {
    let decoder = tiny_decoder();
    let (config, store, radix) = paged_with_radix(&decoder, 8);
    let free_at_start = store.free_groups();
    assert_eq!(free_at_start, 8);

    for cycle in 0..12u32 {
        // Distinct at the very first token, so no cycle adopts another
        // one's prefix: every pass must pay for its own pages, which is
        // what puts the pool under pressure.
        let prompt: Vec<usize> = std::iter::once(10 + cycle as usize % 20)
            .chain([1, 2, 3, 4, 5, 6, 7])
            .collect();
        let (slot, _rx) = admit_prefilled(&decoder, &config, prompt, 4).unwrap_or_else(|| {
            panic!(
                "cycle {cycle} was refused: the tree is holding pages it will \
                 not give back, so the pool shrank monotonically \
                 (free = {}, tree = {})",
                store.free_groups(),
                radix.lock().unwrap().total_size()
            )
        });
        finish_row(slot);
    }

    assert!(
        store.free_groups() > 0,
        "every page ended up parked in the tree"
    );
    // The pages the tree keeps are the whole point, so the pool does
    // NOT return to its starting size -- it just must not reach zero.
    assert!(store.free_groups() <= free_at_start);
}

/// **`cache_salt` on the PAGED store: two callers, one prompt, no
/// sharing.**
///
/// The refusal this replaces said the tree had no namespace to scope a
/// lookup to, so serving the field would tell a caller they had
/// isolation they did not. Asserted at the level the isolation is
/// real: a second request under the SAME salt adopts pages, and one
/// under a different salt adopts nothing.
#[test]
fn a_salted_paged_request_adopts_only_its_own_callers_pages() {
    let decoder = tiny_decoder();
    let (config, _store, _radix) = paged_with_radix(&decoder, 128);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
    let long = prompt.len() + 4;

    // Caller A publishes its prefix.
    {
        let mut lease = acquire_paged_caches(
            &decoder,
            &config,
            &prompt,
            long,
            PrefixIntent::sharing(Some(1)),
        )
        .expect("the store has pages");
        for cache in lease.caches_mut().iter_mut() {
            cache.adopt_blocks(cache.block_table().to_vec(), prompt.len(), BLOCK);
        }
        crate::generate::publish_to_radix(&mut lease, &prompt, BLOCK);
    }

    // The same caller adopts them.
    let same = acquire_paged_caches(
        &decoder,
        &config,
        &prompt,
        long,
        PrefixIntent::sharing(Some(1)),
    )
    .expect("the store has pages");
    assert!(
        same.adopted_positions(BLOCK) > 0,
        "a caller was not served its own published prefix"
    );
    drop(same);

    // A different caller does not.
    let other = acquire_paged_caches(
        &decoder,
        &config,
        &prompt,
        long,
        PrefixIntent::sharing(Some(2)),
    )
    .expect("the store has pages");
    assert_eq!(
        other.adopted_positions(BLOCK),
        0,
        "a different `cache_salt` was served the first caller's pages"
    );
    drop(other);

    // Nor does an unsalted one, which is the shared namespace.
    let shared = acquire_paged_caches(&decoder, &config, &prompt, long, PrefixIntent::default())
        .expect("the store has pages");
    assert_eq!(
        shared.adopted_positions(BLOCK),
        0,
        "an unsalted request was served a salted caller's pages"
    );
}

/// **`prompt_logprobs` on the PAGED store: a row for every position.**
///
/// The refusal this replaces said a paged prefill skips whatever the
/// tree already holds, so it has no rows for those positions. It is
/// true, and the answer is not to report holes: such a request
/// declines the tree at admission and runs its whole prompt.
///
/// Asserted against the CONTIGUOUS scoring of the same prompt, which
/// is the only comparison that can catch rows computed from the wrong
/// positions: a shape check passes for a prompt scored off by one.
#[test]
fn a_scored_paged_prompt_matches_the_contiguous_scoring() {
    let decoder = tiny_decoder();
    let (config, _store, radix) = paged_with_radix(&decoder, 256);
    let prompt = vec![1usize, 2, 3, 4, 5, 6, 7, 8];

    // Publish a prefix, so there IS something a sharing request would
    // adopt. Without this the test proves nothing about declining.
    {
        let mut lease = acquire_paged_caches(
            &decoder,
            &config,
            &prompt,
            prompt.len() + 4,
            PrefixIntent::default(),
        )
        .expect("the store has pages");
        for cache in lease.caches_mut().iter_mut() {
            cache.adopt_blocks(cache.block_table().to_vec(), prompt.len(), BLOCK);
        }
        crate::generate::publish_to_radix(&mut lease, &prompt, BLOCK);
    }
    assert!(
        radix.lock().unwrap().total_size() > 0,
        "nothing was published, so declining the tree proves nothing"
    );

    // A scoring request declines it and runs every position.
    let mut lease = acquire_paged_caches(
        &decoder,
        &config,
        &prompt,
        prompt.len() + 4,
        PrefixIntent::own_prompt_only(None),
    )
    .expect("the store has pages");
    assert_eq!(
        lease.adopted_positions(BLOCK),
        0,
        "a scoring request adopted a prefix it has no rows for"
    );
    let paged_rows = decoder
        .forward_batch_paged(&prompt, 0, lease.caches_mut(), config.store.as_ref())
        .expect("the store has pages");

    let mut caches = decoder.config.new_kv_caches();
    let want = decoder.forward_batch(&prompt, 0, &mut caches);

    assert_eq!(
        paged_rows.len(),
        prompt.len(),
        "one row per prompt position"
    );
    assert_eq!(paged_rows.len(), want.len());
    for (i, (got, expect)) in paged_rows.iter().zip(&want).enumerate() {
        assert_eq!(got.len(), expect.len(), "position {i}: wrong vocabulary");
        for (j, (a, b)) in got.iter().zip(expect).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "position {i} logit {j}: paged {a} against contiguous {b}"
            );
        }
    }
    // And the rows really differ by position, or comparing them to a
    // constant would pass.
    assert!(
        paged_rows[0] != paged_rows[prompt.len() - 1],
        "every position scored identically, so this proved nothing"
    );
}
