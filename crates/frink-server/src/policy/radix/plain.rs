//! The plain radix prefix cache: one currency (full KV pages), LRU
//! eviction over unlocked leaves.
//!
//! This is the cache a dense or ordinary MoE model uses. The two
//! variants in the sibling modules add a *second* currency -- a sliding
//! window ([`super::swa`]) or a recurrent-state snapshot
//! ([`super::hybrid`]) -- and are separate types rather than a
//! parameterization, because what may be evicted, and what an eviction
//! frees, differs at every step.
//!
//! # The accounting rule
//!
//! Every token in the tree is in exactly one of two buckets:
//! `evictable_size` (no live request is reading it) and
//! `protected_size` (some request is). The transfer happens **only on
//! the 0↔1 edge** of a node's lock depth, so a second reader of the
//! same prefix moves nothing. Getting that wrong does not corrupt
//! anything visibly -- it just makes the scheduler's admission
//! arithmetic wrong, and the pool runs out while reporting free space.
//!
//! Ported 1:1 from FreeToken's `python/freetoken/kvcache/radix_cache.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::tree::{align_down, match_len, NodeId, RadixTree, ROOT};

/// What a prefix lookup found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// Tokens of the query that are already computed and now locked
    /// against eviction by the caller's subsequent `lock`.
    pub cached_len: usize,
    /// The node the match ends at -- the handle to lock, and the
    /// insertion point for what follows.
    pub node: NodeId,
}

/// What an insert did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertResult {
    /// How much of the inserted span was **already** in the tree.
    ///
    /// The tree keeps its own pages for that span, so the caller's
    /// pages for `[0, cached_len)` are now duplicates and must be
    /// returned to the pool. This is the single most common source of
    /// a KV leak in a port: the number is not "how much I stored", it
    /// is "how much you must free".
    pub cached_len: usize,
    /// The node covering the whole inserted span.
    pub node: NodeId,
    /// The full inserted length -- what `node` now spells from the
    /// root, after the trailing partial page was dropped.
    pub inserted_len: usize,
}

/// A radix prefix cache over KV pages.
#[derive(Debug)]
pub struct RadixCache {
    tree: RadixTree,
    evictable_size: usize,
    protected_size: usize,
    /// Logical LRU clock, one tick per walk.
    ///
    /// Upstream reads a monotonic nanosecond clock once per walk, so
    /// every node touched by one lookup ties exactly. A counter has the
    /// same property without a clock read, and makes eviction order
    /// reproducible in a test rather than dependent on timer
    /// resolution.
    clock: i64,
}

impl RadixCache {
    pub fn new(page_size: usize) -> Self {
        RadixCache {
            tree: RadixTree::new(page_size),
            evictable_size: 0,
            protected_size: 0,
            clock: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.tree.page_size()
    }

    pub fn tree(&self) -> &RadixTree {
        &self.tree
    }

    /// Tokens held by nodes no request is reading.
    pub fn evictable_size(&self) -> usize {
        self.evictable_size
    }

    /// Tokens held by nodes some request is reading.
    pub fn protected_size(&self) -> usize {
        self.protected_size
    }

    pub fn total_size(&self) -> usize {
        self.evictable_size + self.protected_size
    }

    /// The page indices from the root down to `node`.
    ///
    /// Empty for the root, which is what a cold lookup returns.
    /// (Upstream raises here instead; every caller has to guard on
    /// `cached_len == 0` either way, and an empty answer is the honest
    /// one.)
    pub fn matched_indices(&self, node: NodeId) -> Vec<u32> {
        self.tree.path_value(node)
    }

    fn tick(&mut self) -> i64 {
        self.clock += 1;
        self.clock
    }

    /// Walk `input_ids` down the tree, splitting a node the query
    /// diverges inside, and stamp every node the walk touched.
    ///
    /// Returns the node the walk ended at and how many tokens matched.
    /// The stamp is what makes the *whole matched path* recent, not
    /// just its deepest node -- otherwise a long shared prefix ages out
    /// under the short suffixes that keep being written past it.
    fn tree_walk(&mut self, input_ids: &[u32]) -> (NodeId, usize) {
        let page_size = self.tree.page_size();
        let tic = self.tick();
        let mut prefix_len = 0usize;
        let mut node = ROOT;

        while prefix_len < input_ids.len() {
            let child = match self.tree.child(node, &input_ids[prefix_len..]) {
                Some(child) => child,
                None => return (node, prefix_len),
            };
            node = child;
            // At least one whole page matched -- the child was found by
            // its page key -- so this is never a zero-length step.
            let matched = align_down(
                match_len(&self.tree.node(node).key, &input_ids[prefix_len..]),
                page_size,
            );
            prefix_len += matched;

            if matched != self.tree.node(node).length() {
                // The query diverges inside this node: cut it, and the
                // *prefix* is what matched.
                node = self.tree.split_at(node, matched);
                self.tree.node_mut(node).timestamp = tic;
                return (node, prefix_len);
            }
            self.tree.node_mut(node).timestamp = tic;
        }
        (node, prefix_len)
    }

    /// How much of `input_ids` is already computed, WITHOUT touching
    /// the tree.
    ///
    /// [`Self::match_prefix`] is the real lookup and it mutates: it
    /// splits a node the query diverges inside and stamps every node
    /// the walk touched, because a match is about to be used. An
    /// admission policy asks about jobs it may not admit, and doing
    /// that through the mutating walk would split nodes for queries
    /// that never run and make the LRU clock reflect what was
    /// CONSIDERED rather than what was served.
    ///
    /// So this is the same walk with both side effects removed, and it
    /// stops at the last whole page rather than splitting: the answer
    /// is a lower bound on what `match_prefix` would return, never an
    /// over-estimate, which is the safe direction for a policy that
    /// uses it to rank.
    pub fn peek_cached_len(&self, input_ids: &[u32]) -> usize {
        let page_size = self.tree.page_size();
        let mut prefix_len = 0usize;
        let mut node = ROOT;
        while prefix_len < input_ids.len() {
            let Some(child) = self.tree.child(node, &input_ids[prefix_len..]) else {
                return prefix_len;
            };
            node = child;
            let matched = align_down(
                match_len(&self.tree.node(node).key, &input_ids[prefix_len..]),
                page_size,
            );
            prefix_len += matched;
            if matched != self.tree.node(node).length() {
                return prefix_len;
            }
        }
        prefix_len
    }

    /// The longest already-computed prefix of `input_ids`.
    ///
    /// A ragged query is fine: the answer is truncated to whole pages,
    /// never to the raw match.
    pub fn match_prefix(&mut self, input_ids: &[u32]) -> MatchResult {
        let (node, cached_len) = self.tree_walk(input_ids);
        MatchResult { cached_len, node }
    }

    /// Publish `indices` as the KV pages for `input_ids`.
    ///
    /// A trailing partial page is dropped: its state is still growing,
    /// so storing it would publish a page whose contents depend on
    /// tokens that have not been decoded.
    pub fn insert_prefix(&mut self, input_ids: &[u32], indices: &[u32]) -> InsertResult {
        assert_eq!(
            input_ids.len(),
            indices.len(),
            "one page index per inserted token"
        );
        let insert_len = align_down(input_ids.len(), self.tree.page_size());
        let input_ids = &input_ids[..insert_len];
        let indices = &indices[..insert_len];

        let (node, prefix_len) = self.tree_walk(input_ids);
        let mut node = node;
        if prefix_len != insert_len {
            let tic = self.clock;
            let new_node = self.tree.alloc(tic);
            self.tree.set_key_value(
                new_node,
                input_ids[prefix_len..].to_vec(),
                // Owned, so the caller may reuse its buffer the instant
                // this returns.
                indices[prefix_len..].to_vec(),
            );
            self.tree.set_parent(new_node, node);
            self.evictable_size += self.tree.node(new_node).length();
            node = new_node;
        }
        InsertResult {
            cached_len: prefix_len,
            node,
            inserted_len: insert_len,
        }
    }

    /// Protect `node` and every node between it and the root.
    pub fn lock(&mut self, node: NodeId) {
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            if self.tree.node(cur).ref_count == 0 {
                let length = self.tree.node(cur).length();
                self.evictable_size -= length;
                self.protected_size += length;
            }
            self.tree.node_mut(cur).ref_count += 1;
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    /// Release one lock taken by [`lock`](Self::lock).
    pub fn unlock(&mut self, node: NodeId) {
        let mut cur = node;
        while !self.tree.node(cur).is_root() {
            let count = self.tree.node(cur).ref_count;
            assert!(count > 0, "unlock without a matching lock");
            self.tree.node_mut(cur).ref_count = count - 1;
            if count - 1 == 0 {
                let length = self.tree.node(cur).length();
                self.evictable_size += length;
                self.protected_size -= length;
            }
            cur = self.tree.parent(cur).expect("non-root has a parent");
        }
    }

    /// Evict least-recently-matched unlocked leaves until at least
    /// `size` tokens are freed; returns their page indices.
    ///
    /// Whole nodes only, so the result may exceed `size`. A leaf whose
    /// removal exposes an unlocked parent evicts that parent too,
    /// within the same call -- otherwise a chain of interior nodes,
    /// none of which is ever a leaf until the one below it goes, could
    /// never be reclaimed.
    ///
    /// Asking for more than [`evictable_size`](Self::evictable_size) is
    /// a caller bug: it means the admission arithmetic promised memory
    /// the cache never had. It panics *before* anything is unlinked, so
    /// the tree is untouched.
    pub fn evict(&mut self, size: usize) -> Vec<u32> {
        if size == 0 {
            return Vec::new();
        }
        assert!(
            size <= self.evictable_size,
            "cannot evict {size} tokens, only {} are evictable",
            self.evictable_size
        );

        let mut heap: BinaryHeap<Reverse<(i64, u32)>> = self
            .tree
            .leaves()
            .into_iter()
            .filter(|id| self.tree.node(*id).ref_count == 0)
            .map(|id| Reverse((self.tree.node(id).timestamp, id.0)))
            .collect();

        let mut evicted: Vec<u32> = Vec::new();
        let mut evicted_size = 0usize;
        while evicted_size < size {
            let Reverse((_, raw)) = heap
                .pop()
                .expect("evictable_size promised more than the tree holds");
            let id = NodeId(raw);
            let node = self.tree.node(id);
            debug_assert!(node.ref_count == 0 && node.is_leaf() && !node.is_root());
            evicted_size += node.length();
            evicted.extend_from_slice(&node.value);
            self.evictable_size -= node.length();
            let parent = self.tree.unlink(id);
            if self.tree.node(parent).is_leaf()
                && self.tree.node(parent).ref_count == 0
                && !self.tree.node(parent).is_root()
            {
                heap.push(Reverse((self.tree.node(parent).timestamp, parent.0)));
            }
        }
        evicted
    }

    /// The accounting invariant: the two buckets, recomputed from raw
    /// node state, must equal what the cache has been maintaining.
    pub fn check_integrity(&self) {
        self.tree.check_structure();
        let mut evictable = 0usize;
        let mut protected = 0usize;
        for id in self.tree.walk() {
            let node = self.tree.node(id);
            if node.ref_count == 0 {
                evictable += node.length();
            } else {
                protected += node.length();
            }
        }
        assert_eq!(evictable, self.evictable_size, "evictable_size drifted");
        assert_eq!(protected, self.protected_size, "protected_size drifted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: usize = 4;

    fn ids(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    fn pages(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    /// The basic promise: what one request computed, the next reuses.
    #[test]
    fn a_second_request_reuses_the_first_ones_pages() {
        let mut cache = RadixCache::new(P);
        let prompt = ids(100..108);
        let slots = pages(0, 8);
        let res = cache.insert_prefix(&prompt, &slots);
        assert_eq!(res.cached_len, 0, "nothing was cached before");
        assert_eq!(cache.evictable_size(), 8);

        let m = cache.match_prefix(&ids(100..108));
        assert_eq!(m.cached_len, 8);
        assert_eq!(cache.matched_indices(m.node), slots);
    }

    /// Reuse stops at a page boundary, not at the raw match: a page the
    /// new prompt only partly agrees with holds state it must not read.
    #[test]
    fn reuse_is_truncated_to_whole_pages() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(100..108), &pages(0, 8));

        let mut diverging = ids(100..106);
        diverging.extend([999, 999]);
        let m = cache.match_prefix(&diverging);
        assert_eq!(m.cached_len, 4, "six matched tokens are one whole page");
        assert_eq!(cache.matched_indices(m.node), pages(0, 4));
    }

    /// A still-growing tail has no stable KV state, so it is never
    /// published.
    #[test]
    fn a_trailing_partial_page_is_not_inserted() {
        let mut cache = RadixCache::new(P);
        let res = cache.insert_prefix(&ids(0..6), &pages(0, 6));
        assert_eq!(res.inserted_len, 4);
        assert_eq!(cache.total_size(), 4);
        assert_eq!(cache.match_prefix(&ids(0..6)).cached_len, 4);
    }

    /// A sub-page prompt matches no child at all and resolves to the
    /// root -- it must not half-match an edge.
    #[test]
    fn a_sub_page_prompt_resolves_to_the_root() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(0..8), &pages(0, 8));
        let m = cache.match_prefix(&ids(0..3));
        assert_eq!(m.cached_len, 0);
        assert!(cache.matched_indices(m.node).is_empty());
    }

    /// The number an insert returns is what the caller must FREE, not
    /// what was stored -- and the tree keeps its own pages, never the
    /// duplicates it was handed.
    #[test]
    fn reinserting_a_cached_prefix_reports_the_duplicate_span() {
        let mut cache = RadixCache::new(P);
        let first = pages(0, 8);
        cache.insert_prefix(&ids(100..108), &first);

        let second = pages(50, 8);
        let res = cache.insert_prefix(&ids(100..108), &second);
        assert_eq!(res.cached_len, 8, "the caller must free all eight");
        assert_eq!(
            cache.matched_indices(res.node),
            first,
            "the tree keeps its own pages"
        );
        assert_eq!(cache.total_size(), 8, "and stores nothing new");
    }

    /// A node the caller inserted owns its indices: reusing the buffer
    /// afterwards must not rewrite the tree.
    #[test]
    fn an_inserted_node_owns_its_indices() {
        let mut cache = RadixCache::new(P);
        let mut slots = pages(0, 4);
        let res = cache.insert_prefix(&ids(0..4), &slots);
        slots.iter_mut().for_each(|s| *s = 999);
        assert_eq!(cache.matched_indices(res.node), pages(0, 4));
    }

    #[test]
    fn locking_walks_to_the_root_and_only_the_zero_to_one_edge_moves_size() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(0..8), &pages(0, 8));
        let head = cache.match_prefix(&ids(0..4)); // splits into 4 + 4
        let tail = cache.match_prefix(&ids(0..8));

        cache.lock(tail.node);
        assert_eq!(cache.protected_size(), 8);
        assert_eq!(cache.evictable_size(), 0);

        // A second reader of the same path moves nothing.
        cache.lock(tail.node);
        assert_eq!(cache.protected_size(), 8);
        cache.unlock(tail.node);
        assert_eq!(cache.protected_size(), 8);
        cache.unlock(tail.node);
        assert_eq!(cache.protected_size(), 0);
        assert_eq!(cache.evictable_size(), 8);

        // Locking the head protects only the root path, not what hangs
        // below it.
        cache.lock(head.node);
        assert_eq!(cache.protected_size(), 4);
        assert_eq!(cache.evictable_size(), 4);
        cache.check_integrity();
    }

    #[test]
    fn eviction_takes_the_least_recently_matched_leaf() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(0..4), &pages(0, 4));
        cache.insert_prefix(&ids(100..104), &pages(10, 4));
        // Touch the first branch, making the second the older one.
        cache.match_prefix(&ids(0..4));

        let freed = cache.evict(1);
        assert_eq!(freed, pages(10, 4), "whole nodes only, oldest first");
        assert_eq!(cache.evictable_size(), 4);
        cache.check_integrity();
    }

    #[test]
    fn a_locked_leaf_is_never_evicted() {
        let mut cache = RadixCache::new(P);
        let a = cache.insert_prefix(&ids(0..4), &pages(0, 4));
        cache.insert_prefix(&ids(100..104), &pages(10, 4));
        cache.lock(a.node);

        let freed = cache.evict(4);
        assert_eq!(freed, pages(10, 4));
        cache.check_integrity();
    }

    /// An interior node is only reclaimable once its last child goes.
    /// If exposing it did not re-enter the same eviction pass, a chain
    /// would need one call per level to drain.
    #[test]
    fn eviction_cascades_into_a_newly_childless_parent() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(0..8), &pages(0, 8));
        cache.match_prefix(&ids(0..4)); // split: [0..4) -> [4..8)

        let mut freed = cache.evict(8);
        freed.sort_unstable();
        assert_eq!(freed, pages(0, 8), "both levels, one call");
        assert_eq!(cache.total_size(), 0);
        cache.check_integrity();
    }

    #[test]
    #[should_panic(expected = "cannot evict")]
    fn evicting_more_than_the_tree_holds_is_a_caller_bug() {
        let mut cache = RadixCache::new(P);
        cache.insert_prefix(&ids(0..4), &pages(0, 4));
        cache.evict(8);
    }

    #[test]
    fn evicting_nothing_is_always_safe() {
        let mut cache = RadixCache::new(P);
        assert!(cache.evict(0).is_empty());
        cache.check_integrity();
    }

    /// Page size 1 is a real configuration (window models require it),
    /// and must behave like an ordinary token trie.
    #[test]
    fn page_size_one_degenerates_to_a_token_trie() {
        let mut cache = RadixCache::new(1);
        cache.insert_prefix(&ids(0..5), &pages(0, 5));
        assert_eq!(cache.match_prefix(&ids(0..3)).cached_len, 3);
        let m = cache.match_prefix(&ids(0..5));
        assert_eq!(m.cached_len, 5);
        assert_eq!(cache.matched_indices(m.node), pages(0, 5));
        cache.check_integrity();
    }

    /// The whole cache, driven the way a scheduler drives it, must
    /// conserve pages: every page handed out is either in the tree or
    /// handed back, never both and never neither.
    #[test]
    fn pages_are_conserved_across_a_mixed_workload() {
        let mut cache = RadixCache::new(P);
        let mut next_page = 0u32;
        let mut returned: Vec<u32> = Vec::new();

        for round in 0..6u32 {
            let prompt: Vec<u32> = (0..8).map(|i| if i < 4 { i } else { i + round }).collect();
            let slots = pages(next_page, 8);
            next_page += 8;

            let m = cache.match_prefix(&prompt);
            cache.lock(m.node);
            let res = cache.insert_prefix(&prompt, &slots);
            returned.extend_from_slice(&slots[..res.cached_len]);
            cache.unlock(m.node);
            cache.check_integrity();
        }

        let mut live: Vec<u32> = cache
            .tree()
            .walk()
            .into_iter()
            .flat_map(|id| cache.tree().node(id).value.clone())
            .collect();
        live.extend(cache.evict(cache.evictable_size()));
        live.extend(returned);
        live.sort_unstable();
        live.dedup();
        assert_eq!(
            live.len(),
            next_page as usize,
            "every page is accounted for exactly once"
        );
    }
}
