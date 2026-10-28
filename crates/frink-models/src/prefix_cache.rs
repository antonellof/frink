//! KV-prefix caching: when a new request's tokens share a leading
//! subsequence with a previously processed request, skip recomputing
//! the KV state for that shared prefix entirely, restoring it from a
//! stored snapshot instead of running `forward_batch` over tokens
//! that were already processed.
//!
//! This is the harder sibling of `frink-server::cache::ResponseCache`
//! (which only helps *exact*-repeat requests): prefix caching helps
//! any request that *starts with* something seen before, which is the
//! common case for multi-turn chat (each turn's full prompt is the
//! previous turn's prompt plus a little more) even when no single
//! request repeats exactly.
//!
//! Deliberately scoped: this does a linear scan over a small,
//! LRU-bounded set of stored prefixes to find the longest common
//! prefix, not a trie/radix-tree structure (a radix cache does this
//! properly at production scale). For the
//! small number of concurrent conversations a demo server actually
//! handles, a linear scan is simpler and correctness is easier to
//! verify.
//!
//! "LRU-bounded" is now true. It was not: eviction dropped the oldest
//! ARRIVAL while nothing on the hit path recorded that an entry had
//! been used, so the policy was first-in-first-out under an LRU name.
//! That inverted the cache's purpose, because the entry a prefix cache
//! exists for -- the system prompt every request shares -- is the
//! oldest one precisely because it is the most reused.
//!
//! What is still true of the scope: each entry CLONES its
//! `Vec<KvCache>`, so N conversations off one system prompt hold N
//! copies of its KV rather than sharing the pages. Fixing that is the
//! radix cache's job (`crate::policy::radix`), which shares nodes and
//! reference-counts pages, and which needs a serving path that reads
//! paged KV before it can be wired in.

use frink_core::cache::KvCache;

/// A stored snapshot: the tokens processed so far, the resulting
/// per-layer KV cache state, and the logits that predict the token
/// immediately after `tokens` (needed so a request that matches this
/// prefix *exactly* -- no new tokens at all -- doesn't need any
/// computation to know what to generate next).
#[derive(Clone)]
struct StoredPrefix {
    /// Which caller's namespace this entry belongs to, or `None` for
    /// the shared one.
    ///
    /// A prefix cache is shared state keyed by token ids, so without
    /// this one caller's prompt can be answered from another caller's
    /// cached prefix. That is not a performance question, it is an
    /// isolation one: `cache_salt` names a namespace, and an entry
    /// stored under one is invisible to every other.
    salt: Option<u64>,
    tokens: Vec<usize>,
    kv_caches: Vec<KvCache>,
    pending_logits: Vec<f32>,
    /// Monotonic recency stamp; smallest = least recently used.
    ///
    /// A stamp per touch rather than reshuffling a dedicated LRU list,
    /// the same shape `frink_core::expert_store` uses and for the same
    /// reason: eviction pays an O(n) scan, which costs nothing here
    /// because the lookup that precedes it is already O(n) over the
    /// same vector.
    last_used: u64,
}

/// LRU-bounded store of `StoredPrefix` snapshots, searched for the
/// longest common prefix with an incoming token sequence.
pub struct PrefixCache {
    entries: Vec<StoredPrefix>,
    max_entries: usize,
    hits_positions_reused: u64,
    hits_count: u64,
    misses_count: u64,
    /// Candidate matches passed over because a windowed entry could not
    /// be truncated back to them (#61). Counted per candidate, so an
    /// operator can tell "the prefix cache misses" from "the prefix
    /// cache matched and refused".
    refused_windowed: u64,
    /// Ticks on every hit and every store, so `last_used` orders
    /// entries by when they were last USEFUL rather than by when they
    /// arrived.
    clock: u64,
}

/// What was found (or not) for an incoming token sequence.
pub struct PrefixMatch {
    /// How many leading tokens matched a stored prefix (0 if none).
    pub matched_len: usize,
    /// Restored KV cache state covering exactly `matched_len`
    /// positions, ready to continue from. `None` if `matched_len == 0`.
    pub kv_caches: Option<Vec<KvCache>>,
    /// Logits predicting the token at position `matched_len`, valid
    /// only when `matched_len > 0`.
    pub pending_logits: Option<Vec<f32>>,
}

impl PrefixCache {
    pub fn new(max_entries: usize) -> Self {
        PrefixCache {
            entries: Vec::new(),
            max_entries,
            hits_positions_reused: 0,
            hits_count: 0,
            misses_count: 0,
            refused_windowed: 0,
            clock: 0,
        }
    }

    /// Drops every stored prefix, keeping the capacity and the
    /// lifetime hit/miss counters.
    ///
    /// For a KV-side cache rebuild. A stored prefix names positions in
    /// an allocation that is about to stop existing, so handing one back
    /// afterwards would restore another request's state into this one --
    /// silently, since a KV cache carries no identity of its own. The
    /// counters survive because they describe what this process has
    /// served, which a re-split does not undo.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Finds the stored prefix with the longest common leading
    /// subsequence with `tokens`, and returns a ready-to-use clone of
    /// its KV state truncated to exactly that common length (a stored
    /// prefix may itself be longer than the common part, if a later,
    /// different continuation was stored under it -- the KV cache is
    /// truncated to the matching length before being handed back, so
    /// the caller never sees state from a divergent continuation).
    pub fn find_longest_prefix(&mut self, tokens: &[usize]) -> PrefixMatch {
        self.find_longest_prefix_salted(tokens, None)
    }

    /// The same, scoped to a caller's namespace.
    ///
    /// An entry stored under a different salt is not a shorter match,
    /// it is NO match: the whole point is that the two callers cannot
    /// see each other's prefixes, so a partial overlap must not be
    /// reused either.
    pub fn find_longest_prefix_salted(
        &mut self,
        tokens: &[usize],
        salt: Option<u64>,
    ) -> PrefixMatch {
        let mut best: Option<(usize, usize)> = None; // (matched_len, index)
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.salt != salt {
                continue;
            }
            let common = common_prefix_len(&entry.tokens, tokens);
            if common == 0 || best.is_some_and(|(len, _)| common <= len) {
                continue;
            }
            // A windowed entry (#61) is reusable only as far back as its
            // caches can still truncate. See `reusable_at`.
            if !reusable_at(entry, common, tokens.len()) {
                self.refused_windowed += 1;
                continue;
            }
            best = Some((common, i));
        }

        match best {
            Some((matched_len, index)) => {
                self.hits_count += 1;
                self.hits_positions_reused += matched_len as u64;
                // A HIT is what makes an entry worth keeping, so this
                // is where recency has to be recorded. Without it the
                // policy degenerates to first-in-first-out, and the one
                // entry a prefix cache exists for -- a shared system
                // prompt every request starts with -- is evicted as
                // soon as `max_entries` newer prompts arrive, however
                // often it is being reused.
                self.clock += 1;
                self.entries[index].last_used = self.clock;
                let entry = &self.entries[index];

                let mut kv_caches = entry.kv_caches.clone();
                for cache in kv_caches.iter_mut() {
                    cache.truncate(matched_len);
                }

                // The stored pending_logits predict the token
                // immediately after entry.tokens' FULL length. They're
                // only valid to hand back if the match covers that
                // entire stored sequence (matched_len ==
                // entry.tokens.len()); a partial match into the middle
                // of a longer stored sequence means the caller is
                // asking about position `matched_len`, not
                // `entry.tokens.len()`, and reusing the stored logits
                // there would silently answer the wrong question.
                let pending_logits = if matched_len == entry.tokens.len() {
                    Some(entry.pending_logits.clone())
                } else {
                    None
                };

                PrefixMatch {
                    matched_len,
                    kv_caches: Some(kv_caches),
                    pending_logits,
                }
            }
            None => {
                self.misses_count += 1;
                PrefixMatch {
                    matched_len: 0,
                    kv_caches: None,
                    pending_logits: None,
                }
            }
        }
    }

    /// Stores a snapshot for `tokens` (all tokens processed so far,
    /// prompt plus any generated continuation) with the given KV cache
    /// state and next-token logits, evicting the least recently USED
    /// entry if already at capacity.
    ///
    /// Used, not stored. This used to drop `entries[0]` -- the oldest
    /// arrival -- while the type documented itself as LRU-bounded. The
    /// difference is the whole value of the cache: a system prompt that
    /// every request shares is the oldest entry precisely BECAUSE it is
    /// the most reused, so FIFO evicted the one entry worth keeping as
    /// soon as `max_entries` newer prompts arrived, and the next
    /// request off that system prompt recomputed all of it.
    /// A cache that evicts rows behind a sliding window (#61) IS stored,
    /// and costs O(window) for its windowed layers rather than the whole
    /// prefix. What it can no longer do is be truncated to an arbitrary
    /// common length, so the lookup reuses it only for a match it can
    /// still truncate to (`reusable_at`) and treats a deeper one as a
    /// miss. Multi-turn chat, where the next prompt extends the last
    /// one, matches at or near the end and keeps hitting; a request that
    /// diverges a window or more back recomputes, which is the honest
    /// answer for rows this cache no longer holds.
    ///
    /// A cache holding a RECURRENT state (`frink_core::recurrent_state`,
    /// a Mamba layer's) is refused for the same reason from the other
    /// side: the state is a reduction over the whole prefix, and
    /// `truncate` to the common length of a partial match has no
    /// answer for it (`KvCache::can_truncate_to`). llama.cpp's server
    /// re-prefills such models too.
    pub fn store(&mut self, tokens: Vec<usize>, kv_caches: Vec<KvCache>, pending_logits: Vec<f32>) {
        self.store_salted(tokens, kv_caches, pending_logits, None)
    }

    /// The same, into a caller's namespace.
    pub fn store_salted(
        &mut self,
        tokens: Vec<usize>,
        kv_caches: Vec<KvCache>,
        pending_logits: Vec<f32>,
        salt: Option<u64>,
    ) {
        if kv_caches.iter().any(|c| c.recurrent.is_some()) {
            return;
        }
        if self.entries.len() >= self.max_entries {
            // An O(n) scan, over the same vector the lookup above
            // already scans linearly -- so this costs nothing the
            // design was not already paying.
            if let Some(coldest) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(i, e)| (e.last_used, *i))
                .map(|(i, _)| i)
            {
                self.entries.remove(coldest);
            }
        }
        self.clock += 1;
        self.entries.push(StoredPrefix {
            salt,
            last_used: self.clock,
            tokens,
            kv_caches,
            pending_logits,
        });
    }

    pub fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            hits: self.hits_count,
            misses: self.misses_count,
            entries: self.entries.len(),
            total_positions_reused: self.hits_positions_reused,
            refused_windowed: self.refused_windowed,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub total_positions_reused: u64,
    /// Matches a windowed entry could not serve; see
    /// `PrefixCache::refused_windowed`.
    pub refused_windowed: u64,
}

/// Whether `entry` can be handed back for a query of `query_len` tokens
/// that shares its first `matched_len`.
///
/// Every cache must truncate to `matched_len` (`KvCache::can_truncate_to`,
/// which is where a windowed cache says it no longer holds what the next
/// query reads). And when the match covers the whole query but not the
/// whole entry, the caller has no stored logits for that position and
/// backs off one more to recompute them
/// (`frink-server::generate`'s restore), so that position must be
/// reachable too. Asking here keeps the refusal in one place: a lookup
/// that answered yes and a restore that then could not truncate would be
/// two structures disagreeing about one cache.
fn reusable_at(entry: &StoredPrefix, matched_len: usize, query_len: usize) -> bool {
    let backs_off = matched_len == query_len && matched_len < entry.tokens.len();
    let target = if backs_off {
        matched_len - 1
    } else {
        matched_len
    };
    entry.kv_caches.iter().all(|c| c.can_truncate_to(target))
}

fn common_prefix_len(a: &[usize], b: &[usize]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cache(seq_len: usize) -> KvCache {
        let mut cache = KvCache::new(1, 1);
        for i in 0..seq_len {
            cache.push(&[i as f32], &[i as f32 * 10.0]).unwrap();
        }
        cache
    }

    /// The bug this type was named after and did not have.
    ///
    /// A shared system prompt is the entry a prefix cache exists for,
    /// and under FIFO it was the FIRST thing evicted -- it is the
    /// oldest arrival precisely because it is the most reused. Here it
    /// is kept hot by hits while `max_entries` newer prompts arrive,
    /// and it must survive.
    #[test]
    fn a_prefix_that_keeps_being_hit_survives_newer_arrivals() {
        let system: Vec<usize> = (0..8).collect();
        let mut cache = PrefixCache::new(3);
        cache.store(system.clone(), vec![dummy_cache(system.len())], vec![0.5]);

        // Three unrelated prompts arrive, which is capacity twice over.
        // Between each, the system prompt is used again.
        for n in 0..3usize {
            let hit = cache.find_longest_prefix(&system);
            assert_eq!(
                hit.matched_len,
                system.len(),
                "the system prompt must still be here before arrival {n}"
            );
            let other: Vec<usize> = (100 + n * 10..100 + n * 10 + 4).collect();
            cache.store(other.clone(), vec![dummy_cache(other.len())], vec![0.5]);
        }

        let hit = cache.find_longest_prefix(&system);
        assert_eq!(
            hit.matched_len,
            system.len(),
            "a hot prefix was evicted while cold newer ones were kept"
        );
    }

    /// And the converse: the entry nobody has touched is the one that
    /// goes. Without this the first test could pass by never evicting
    /// anything at all.
    #[test]
    fn the_least_recently_used_prefix_is_the_one_evicted() {
        let mut cache = PrefixCache::new(2);
        let cold: Vec<usize> = vec![1, 2, 3, 4];
        let warm: Vec<usize> = vec![5, 6, 7, 8];
        cache.store(cold.clone(), vec![dummy_cache(cold.len())], vec![0.5]);
        cache.store(warm.clone(), vec![dummy_cache(warm.len())], vec![0.5]);

        // Touch `warm` only, then push past capacity.
        assert_eq!(cache.find_longest_prefix(&warm).matched_len, warm.len());
        let fresh: Vec<usize> = vec![9, 10, 11, 12];
        cache.store(fresh.clone(), vec![dummy_cache(fresh.len())], vec![0.5]);

        assert_eq!(
            cache.find_longest_prefix(&cold).matched_len,
            0,
            "the untouched entry should have been evicted"
        );
        assert_eq!(cache.find_longest_prefix(&warm).matched_len, warm.len());
        assert_eq!(cache.find_longest_prefix(&fresh).matched_len, fresh.len());
    }

    /// With nothing ever hit, eviction still has to make progress and
    /// has to be deterministic: equal stamps break toward the lower
    /// index, so the oldest arrival goes, which is the FIFO behaviour
    /// as a degenerate case rather than as the policy.
    #[test]
    fn untouched_entries_evict_oldest_first_and_capacity_is_never_exceeded() {
        let mut cache = PrefixCache::new(2);
        for n in 0..5usize {
            let p: Vec<usize> = (n * 10..n * 10 + 4).collect();
            cache.store(p.clone(), vec![dummy_cache(p.len())], vec![0.5]);
        }
        assert_eq!(cache.entries.len(), 2, "capacity must hold");
        // The two most recent survive.
        for n in [3usize, 4] {
            let p: Vec<usize> = (n * 10..n * 10 + 4).collect();
            assert_eq!(cache.find_longest_prefix(&p).matched_len, 4, "prompt {n}");
        }
        for n in [0usize, 1, 2] {
            let p: Vec<usize> = (n * 10..n * 10 + 4).collect();
            assert_eq!(cache.find_longest_prefix(&p).matched_len, 0, "prompt {n}");
        }
    }

    /// A cache over `len` positions whose one layer evicts behind
    /// `window`, as a windowed decoder leaves it.
    fn windowed_cache(len: usize, window: usize) -> KvCache {
        let w = frink_core::kv_swa::KvWindow::with_default_slack(window).expect("positive window");
        let mut cache = KvCache::new(1, 1);
        cache.arm_window(w);
        for i in 0..len {
            cache.push(&[i as f32], &[i as f32 * 10.0]).unwrap();
            cache.evict_behind_window();
        }
        assert!(cache.rows() < len, "the fixture must have evicted");
        cache
    }

    /// **#61: a windowed entry is stored, and reused only where it can
    /// be.** Before, a windowed cache was refused at `store`, so turning
    /// eviction on turned the prefix cache off. Now it is stored and
    /// the lookup asks the cache itself how far back it can go.
    ///
    /// The extension case (the next chat turn) hits. A query that shares
    /// only the first few tokens names a position the windowed layer
    /// dropped, and it must be a MISS: handing back a cache truncated
    /// across a hole would answer the next token out of a history the
    /// model never had. Sabotaged by making `reusable_at` return `true`,
    /// the restore below panics in `truncate` rather than passing.
    #[test]
    fn a_windowed_entry_is_reused_at_its_end_and_refused_deep_inside_it() {
        let stored: Vec<usize> = (0..64).collect();
        let mut cache = PrefixCache::new(4);
        cache.store(stored.clone(), vec![windowed_cache(64, 8)], vec![0.5]);
        assert_eq!(cache.stats().entries, 1, "a windowed cache is stored");

        // The next turn: the whole stored sequence and then some.
        let mut next = stored.clone();
        next.extend([100, 101]);
        let m = cache.find_longest_prefix(&next);
        assert_eq!(m.matched_len, 64);
        assert_eq!(m.pending_logits, Some(vec![0.5]));
        assert_eq!(m.kv_caches.expect("a hit")[0].positions(), 64);

        // Diverges three tokens before the end: inside the reserve.
        let mut near = stored[..61].to_vec();
        near.push(999);
        let m = cache.find_longest_prefix(&near);
        assert_eq!(
            m.matched_len, 61,
            "a match inside the rollback reserve is served"
        );

        // Diverges at token 5 of 64: that row is gone.
        let mut deep = stored[..5].to_vec();
        deep.push(999);
        let m = cache.find_longest_prefix(&deep);
        assert_eq!(m.matched_len, 0, "a match behind the window must miss");
        assert!(m.kv_caches.is_none());
        assert!(cache.stats().refused_windowed >= 1);
    }

    /// The back-off case: the query is a strict prefix of the entry, so
    /// the caller recomputes the last matched position and must be able
    /// to truncate one further than the match. Refused here when that
    /// one further is out of reach, not discovered by the restore.
    #[test]
    fn a_windowed_entry_is_refused_when_the_restore_would_back_off_past_it() {
        let stored: Vec<usize> = (0..64).collect();
        let entry = StoredPrefix {
            salt: None,
            tokens: stored.clone(),
            kv_caches: vec![windowed_cache(64, 8)],
            pending_logits: vec![0.5],
            last_used: 0,
        };
        let limit = entry.kv_caches[0].rollback_limit();
        let deepest = (0..=64usize)
            .find(|&n| entry.kv_caches[0].can_truncate_to(n))
            .expect("the cache can truncate to its own length");
        assert!(64 - deepest >= limit.min(64));
        // A query ending exactly at the deepest reachable match backs
        // off one more, which is out of reach.
        assert!(!reusable_at(&entry, deepest, deepest));
        // The same match inside a longer query does not back off.
        assert!(reusable_at(&entry, deepest, deepest + 3));
    }

    #[test]
    fn empty_cache_always_misses() {
        let mut cache = PrefixCache::new(4);
        let m = cache.find_longest_prefix(&[1, 2, 3]);
        assert_eq!(m.matched_len, 0);
        assert!(m.kv_caches.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn exact_prefix_match_returns_full_length_and_pending_logits() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![0.1, 0.2]);

        let m = cache.find_longest_prefix(&[1, 2, 3]);
        assert_eq!(m.matched_len, 3);
        assert!(m.kv_caches.is_some());
        assert_eq!(m.pending_logits, Some(vec![0.1, 0.2]));
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn extended_request_matches_the_shared_prefix_length() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3, 4, 5], vec![dummy_cache(5)], vec![9.9]);

        // New request extends the stored one with two more tokens.
        let m = cache.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            m.matched_len, 5,
            "must match the full stored prefix, not just a partial one"
        );
        assert_eq!(m.pending_logits, Some(vec![9.9]));
    }

    #[test]
    fn partial_divergent_match_returns_only_the_common_length_and_no_stale_logits() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3, 4, 5], vec![dummy_cache(5)], vec![9.9]);

        // Diverges after the first 3 tokens.
        let m = cache.find_longest_prefix(&[1, 2, 3, 9, 9]);
        assert_eq!(m.matched_len, 3);
        assert!(
            m.kv_caches.is_some(),
            "a real KV-state saving still exists for the matched prefix"
        );
        assert!(
            m.pending_logits.is_none(),
            "stored pending_logits predicted the token after the FULL stored sequence, not after the partial match point -- must not be reused here"
        );
    }

    /// **Two callers cannot see each other's prefixes.**
    ///
    /// Not "a shorter match": an entry under a different salt is NO
    /// match at all, because a partial overlap is exactly what would
    /// leak -- the shared leading tokens are usually the system
    /// prompt, which is the part a caller most expects to be theirs.
    #[test]
    fn a_salted_entry_is_invisible_to_every_other_salt() {
        let mut cache = PrefixCache::new(8);
        let tokens = vec![1usize, 2, 3, 4];
        cache.store_salted(tokens.clone(), vec![dummy_cache(4)], vec![0.0; 4], Some(7));

        // The same prompt under another salt: no reuse.
        let other = cache.find_longest_prefix_salted(&tokens, Some(9));
        assert_eq!(other.matched_len, 0, "a different caller reused the prefix");
        // And under none.
        let shared = cache.find_longest_prefix_salted(&tokens, None);
        assert_eq!(shared.matched_len, 0, "the shared namespace reused it");
        // The owner still gets it.
        let mine = cache.find_longest_prefix_salted(&tokens, Some(7));
        assert_eq!(mine.matched_len, 4, "the owner lost its own prefix");
    }

    /// A PARTIAL overlap across salts is also no match, which is the
    /// case a length comparison would have let through.
    #[test]
    fn a_partial_overlap_across_salts_is_still_no_match() {
        let mut cache = PrefixCache::new(8);
        let stored = vec![1usize, 2, 3, 4];
        cache.store_salted(stored.clone(), vec![dummy_cache(4)], vec![0.0; 4], Some(1));
        let overlapping = vec![1usize, 2, 9, 9];
        assert_eq!(
            cache
                .find_longest_prefix_salted(&overlapping, Some(2))
                .matched_len,
            0,
            "two leading tokens leaked across a salt boundary"
        );
        assert_eq!(
            cache
                .find_longest_prefix_salted(&overlapping, Some(1))
                .matched_len,
            2,
            "the owner lost its own partial match"
        );
    }

    #[test]
    fn no_common_prefix_at_all_is_a_clean_miss() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![1.0]);
        let m = cache.find_longest_prefix(&[9, 8, 7]);
        assert_eq!(m.matched_len, 0);
    }

    #[test]
    fn picks_the_longest_match_among_several_stored_entries() {
        let mut cache = PrefixCache::new(4);
        cache.store(vec![1, 2], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![1, 2, 3, 4], vec![dummy_cache(4)], vec![0.0]);
        cache.store(vec![1, 2, 3], vec![dummy_cache(3)], vec![0.0]);

        let m = cache.find_longest_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!(
            m.matched_len, 4,
            "the longest stored prefix that's actually a prefix of the query must win"
        );
    }

    #[test]
    fn evicts_oldest_entry_when_at_capacity() {
        let mut cache = PrefixCache::new(2);
        cache.store(vec![1, 1], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![2, 2], vec![dummy_cache(2)], vec![0.0]);
        cache.store(vec![3, 3], vec![dummy_cache(2)], vec![0.0]); // evicts [1,1]

        assert_eq!(
            cache.find_longest_prefix(&[1, 1]).matched_len,
            0,
            "oldest entry must have been evicted"
        );
        assert_eq!(cache.find_longest_prefix(&[2, 2]).matched_len, 2);
        assert_eq!(cache.find_longest_prefix(&[3, 3]).matched_len, 2);
    }

    #[test]
    fn stats_track_positions_reused_not_just_hit_count() {
        let mut cache = PrefixCache::new(4);
        cache.store(
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![dummy_cache(8)],
            vec![0.0],
        );
        cache.find_longest_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(
            cache.stats().total_positions_reused,
            8,
            "should report exactly how many positions were reused, not just that a hit occurred"
        );
    }

    /// The end-to-end property that matters most: using a prefix
    /// cache's restored KV state to continue a real decoder must
    /// produce EXACTLY the same output as processing the full token
    /// sequence from scratch. If this fails, prefix caching is not a
    /// safe optimization -- it would silently change model output
    /// depending on cache state, which is far worse than no caching at
    /// all.
    #[test]
    fn prefix_cached_continuation_matches_from_scratch_decode_exactly() {
        use crate::config::glm_5_2;
        use crate::decoder::Decoder;
        use frink_core::cache::KvCache as RealKvCache;

        let mut cfg = glm_5_2();
        cfg.hidden_dim = 16;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 4;
        cfg.moe.hidden_dim = 16;
        cfg.moe.n_experts = 6;
        cfg.moe.n_experts_active = 2;
        cfg.moe.n_shared_experts = 1;
        cfg.moe.expert_ffn_dim = 8;
        let vocab = 16;

        let shared_prefix = vec![1usize, 2, 3, 4, 5];
        let full_sequence = vec![1usize, 2, 3, 4, 5, 6, 7];

        // "Conversation A": process the shared prefix once, store it.
        let decoder_a = Decoder::new_random_small(cfg.clone(), 2, vocab);
        let mut caches_a: Vec<RealKvCache> = decoder_a.config.new_kv_caches();
        let prefix_logits = decoder_a.forward_batch(&shared_prefix, 0, &mut caches_a);
        let mut prefix_cache = PrefixCache::new(4);
        prefix_cache.store(
            shared_prefix.clone(),
            caches_a,
            prefix_logits.last().unwrap().clone(),
        );

        // "Conversation B": extends the shared prefix. Using the
        // prefix cache, only the new suffix tokens should need
        // computing.
        let decoder_b = Decoder::new_random_small(cfg.clone(), 2, vocab); // same seed => identical weights
        let m = prefix_cache.find_longest_prefix(&full_sequence);
        assert_eq!(m.matched_len, 5);
        let mut restored_caches = m.kv_caches.unwrap();
        let suffix = &full_sequence[m.matched_len..];
        let via_prefix_cache_logits =
            decoder_b.forward_batch(suffix, m.matched_len, &mut restored_caches);

        // Ground truth: process the ENTIRE sequence from scratch on an
        // identically-seeded decoder with a fresh empty cache.
        let decoder_c = Decoder::new_random_small(cfg, 2, vocab);
        let mut fresh_caches: Vec<RealKvCache> = decoder_c.config.new_kv_caches();
        let from_scratch_logits = decoder_c.forward_batch(&full_sequence, 0, &mut fresh_caches);

        // The prefix-cache path's logits for the suffix positions must
        // match the from-scratch path's logits for those same
        // positions exactly.
        let from_scratch_suffix = &from_scratch_logits[m.matched_len..];
        assert_eq!(via_prefix_cache_logits.len(), from_scratch_suffix.len());
        for (pos, (a, b)) in via_prefix_cache_logits
            .iter()
            .zip(from_scratch_suffix.iter())
            .enumerate()
        {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-3,
                    "suffix position {pos}, logit {i}: via_prefix_cache={x} from_scratch={y}"
                );
            }
        }

        // And the KV cache state itself must match too, not just the
        // final logits (in case a later request extends even further).
        for (restored, fresh) in restored_caches.iter().zip(fresh_caches.iter()) {
            assert_eq!(restored.positions(), fresh.positions());
            for (a, b) in restored
                .k
                .to_f32_vec()
                .iter()
                .zip(fresh.k.to_f32_vec().iter())
            {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }
}
