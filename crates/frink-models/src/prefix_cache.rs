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
        let mut best: Option<(usize, usize)> = None; // (matched_len, index)
        for (i, entry) in self.entries.iter().enumerate() {
            let common = common_prefix_len(&entry.tokens, tokens);
            if common > 0 && best.map(|(len, _)| common > len).unwrap_or(true) {
                best = Some((common, i));
            }
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
    /// A cache that has evicted rows behind a sliding window (#61,
    /// `FRINK_KV_WINDOW`) is REFUSED rather than stored. A stored
    /// prefix is handed back truncated to an arbitrary common length,
    /// and a windowed cache no longer holds the rows an arbitrary
    /// truncation names -- so storing one would trade a cheap recompute
    /// for a request that stops. Refusing loses the prefix cache for
    /// the run, which is what the switch's documentation says it costs.
    ///
    /// A cache holding a RECURRENT state (`frink_core::recurrent_state`,
    /// a Mamba layer's) is refused for the same reason from the other
    /// side: the state is a reduction over the whole prefix, and
    /// `truncate` to the common length of a partial match has no
    /// answer for it (`KvCache::can_truncate_to`). llama.cpp's server
    /// re-prefills such models too.
    pub fn store(&mut self, tokens: Vec<usize>, kv_caches: Vec<KvCache>, pending_logits: Vec<f32>) {
        if kv_caches
            .iter()
            .any(|c| c.window().is_some() || c.recurrent.is_some())
        {
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
        }
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PrefixCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    pub total_positions_reused: u64,
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
            for (a, b) in restored.k.iter().zip(fresh.k.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }
}
