//! One radix prefix cache per CALLER namespace.
//!
//! [`RadixCache`] walks one tree keyed by token ids, so every request
//! that sends the same leading tokens is served the same pages. That
//! is what a prefix cache is for, and it is also why `cache_salt` was
//! refused on the paged store: the field promises a caller that their
//! pages are theirs, and the tree had no namespace to scope a lookup
//! to.
//!
//! A namespace is a whole tree here, not a synthetic prefix inside
//! one. A prefix would have been cheaper and wrong: the tree matches
//! whole PAGES, so prepending anything shifts every page boundary and
//! the indices it stores stop lining up with the positions the caller
//! adopts.
//!
//! # What is shared and what is not
//!
//! The PAGES are one pool. A namespace does not get its own memory,
//! and eviction reaches across all of them, or one idle tenant could
//! hold pages a busy one needs.
//!
//! What is not shared is the lookup: a salted request can only match
//! nodes in its own tree, and the unsalted namespace (`None`) is its
//! own tree too -- which is what every request got before the field
//! existed.
//!
//! # Eviction is round-robin across namespaces, LRU inside one
//!
//! Each tree keeps its own logical clock, ticked once per walk of
//! THAT tree, so two namespaces' timestamps are two counters that
//! never tick together and comparing them would order by how busy a
//! tenant is rather than by age. So a namespace is asked for its
//! least-recently-matched node, one at a time, in turn.
//!
//! That is also the better policy for the case the field exists for:
//! under a global LRU one busy caller evicts every quiet caller's
//! prefix, which is the noisy-neighbour problem `cache_salt` is meant
//! to bound.

use std::collections::HashMap;

use super::{NodeId, RadixCache};

/// A node, and which namespace's tree it lives in.
///
/// One value rather than two arguments, because a `NodeId` from one
/// tree used against another names a different node and the error is
/// silent: the lock would land on a stranger's prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle {
    pub salt: Option<u64>,
    pub node: NodeId,
}

/// What a salted match found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaltedMatch {
    pub cached_len: usize,
    pub handle: Handle,
}

/// What a salted insert did. Mirrors `InsertResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaltedInsert {
    pub cached_len: usize,
    pub inserted_len: usize,
    pub handle: Handle,
}

/// Per-caller radix prefix caches over one page pool.
#[derive(Debug)]
pub struct SaltedRadix {
    page_size: usize,
    /// Created on first use. A namespace nobody has sent a request
    /// under costs nothing, so a deployment with a salt per tenant
    /// pays for the tenants it has.
    namespaces: HashMap<Option<u64>, RadixCache>,
    /// The order namespaces were first seen, so the round-robin below
    /// is reproducible rather than dependent on hash iteration order.
    order: Vec<Option<u64>>,
    /// Where the next eviction round starts, so two calls in a row do
    /// not both take from the same namespace.
    evict_cursor: usize,
}

impl SaltedRadix {
    pub fn new(page_size: usize) -> Self {
        SaltedRadix {
            page_size,
            namespaces: HashMap::new(),
            order: Vec::new(),
            evict_cursor: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    fn tree(&mut self, salt: Option<u64>) -> &mut RadixCache {
        let page_size = self.page_size;
        self.namespaces.entry(salt).or_insert_with(|| {
            self.order.push(salt);
            RadixCache::new(page_size)
        })
    }

    /// The longest already-computed prefix of `input_ids` IN THIS
    /// CALLER'S namespace.
    pub fn match_prefix(&mut self, salt: Option<u64>, input_ids: &[u32]) -> SaltedMatch {
        let m = self.tree(salt).match_prefix(input_ids);
        SaltedMatch {
            cached_len: m.cached_len,
            handle: Handle { salt, node: m.node },
        }
    }

    pub fn insert_prefix(
        &mut self,
        salt: Option<u64>,
        input_ids: &[u32],
        indices: &[u32],
    ) -> SaltedInsert {
        let r = self.tree(salt).insert_prefix(input_ids, indices);
        SaltedInsert {
            cached_len: r.cached_len,
            inserted_len: r.inserted_len,
            handle: Handle { salt, node: r.node },
        }
    }

    pub fn matched_indices(&mut self, handle: Handle) -> Vec<u32> {
        self.tree(handle.salt).matched_indices(handle.node)
    }

    pub fn lock(&mut self, handle: Handle) {
        self.tree(handle.salt).lock(handle.node);
    }

    pub fn unlock(&mut self, handle: Handle) {
        self.tree(handle.salt).unlock(handle.node);
    }

    /// Tokens held by nodes no request is reading, across every
    /// namespace. The pages are one pool, so this is the number the
    /// admission arithmetic may spend.
    pub fn evictable_size(&self) -> usize {
        self.namespaces.values().map(|c| c.evictable_size()).sum()
    }

    pub fn protected_size(&self) -> usize {
        self.namespaces.values().map(|c| c.protected_size()).sum()
    }

    pub fn total_size(&self) -> usize {
        self.namespaces.values().map(|c| c.total_size()).sum()
    }

    /// Frees at least `size` tokens, taking from each namespace in
    /// turn.
    ///
    /// Whole nodes only, so the result may exceed `size` -- as
    /// `RadixCache::evict` does, and for the same reason. Asking for
    /// more than [`Self::evictable_size`] returns everything rather
    /// than panicking, because with several trees the caller cannot
    /// check a per-namespace figure it never sees.
    pub fn evict(&mut self, size: usize) -> Vec<u32> {
        let mut freed = Vec::new();
        if size == 0 || self.order.is_empty() {
            return freed;
        }
        let mut taken = 0usize;
        // Bounded by a whole pass that frees nothing: every namespace
        // is empty of evictable nodes, so there is nothing left to do.
        loop {
            let mut progress = false;
            for _ in 0..self.order.len() {
                let salt = self.order[self.evict_cursor % self.order.len()];
                self.evict_cursor = self.evict_cursor.wrapping_add(1);
                let cache = self
                    .namespaces
                    .get_mut(&salt)
                    .expect("every ordered namespace exists");
                if cache.evictable_size() == 0 {
                    continue;
                }
                // One token is enough to take a whole node, which is
                // the unit eviction really works in.
                let got = cache.evict(1);
                taken += got.len();
                freed.extend(got);
                progress = true;
                if taken >= size {
                    return freed;
                }
            }
            if !progress {
                return freed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4;

    fn ids(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    /// **The whole point: one caller's prefix is not another's.**
    ///
    /// The same tokens under two salts must MISS each other. A tree
    /// with no namespace answers this with a hit, which is the
    /// disclosure `cache_salt` exists to prevent, and it is invisible
    /// from the answer -- the text is right, it was just computed from
    /// somebody else's pages.
    #[test]
    fn two_salts_cannot_see_each_others_prefixes() {
        let mut cache = SaltedRadix::new(PAGE);
        let tokens = ids(8);
        let pages: Vec<u32> = vec![7; 8];
        cache.insert_prefix(Some(1), &tokens, &pages);

        assert_eq!(
            cache.match_prefix(Some(1), &tokens).cached_len,
            8,
            "a caller must match its own prefix"
        );
        assert_eq!(
            cache.match_prefix(Some(2), &tokens).cached_len,
            0,
            "a different salt was served the first caller's pages"
        );
        assert_eq!(
            cache.match_prefix(None, &tokens).cached_len,
            0,
            "an unsalted request was served a salted caller's pages"
        );
    }

    /// The unsalted namespace is a namespace, not a fallback: a salted
    /// request must not match what an unsalted one published either.
    #[test]
    fn the_shared_namespace_does_not_leak_into_a_salted_one() {
        let mut cache = SaltedRadix::new(PAGE);
        let tokens = ids(8);
        cache.insert_prefix(None, &tokens, &[3; 8]);
        assert_eq!(cache.match_prefix(None, &tokens).cached_len, 8);
        assert_eq!(
            cache.match_prefix(Some(9), &tokens).cached_len,
            0,
            "a salted caller matched the shared namespace"
        );
    }

    /// A handle names its namespace, so a lock lands where the match
    /// did and the accounting moves with it.
    #[test]
    fn a_lock_protects_the_namespace_it_matched_in() {
        let mut cache = SaltedRadix::new(PAGE);
        let tokens = ids(8);
        cache.insert_prefix(Some(1), &tokens, &[1; 8]);
        cache.insert_prefix(Some(2), &tokens, &[2; 8]);
        assert_eq!(cache.evictable_size(), 16, "two namespaces, eight each");

        let m = cache.match_prefix(Some(1), &tokens);
        cache.lock(m.handle);
        assert_eq!(cache.protected_size(), 8, "one namespace's worth is held");
        assert_eq!(cache.evictable_size(), 8, "the other is still evictable");

        cache.unlock(m.handle);
        assert_eq!(cache.protected_size(), 0);
        assert_eq!(cache.evictable_size(), 16);
    }

    /// **The pages are one pool, so eviction reaches every namespace.**
    ///
    /// The failure this catches is a deployment that deadlocks itself:
    /// a busy caller asks for pages, the store is full, and every free
    /// page is held by a tree the evictor never looks in.
    #[test]
    fn eviction_reaches_across_namespaces() {
        let mut cache = SaltedRadix::new(PAGE);
        for salt in 0..4u64 {
            cache.insert_prefix(Some(salt), &ids(8), &[salt as u32; 8]);
        }
        assert_eq!(cache.evictable_size(), 32);

        let freed = cache.evict(24);
        assert!(
            freed.len() >= 24,
            "eviction stopped inside one namespace: freed {}",
            freed.len()
        );
        // And it really took from more than one, or it freed 24 out of
        // a namespace that only held 8.
        let touched: std::collections::BTreeSet<u32> = freed.into_iter().collect();
        assert!(
            touched.len() > 1,
            "every freed page came from one namespace: {touched:?}"
        );
    }

    /// Asking for more than exists returns what there is rather than
    /// panicking: with several trees the caller checks a total it
    /// cannot attribute to any one of them.
    #[test]
    fn asking_for_more_than_is_held_returns_what_there_is() {
        let mut cache = SaltedRadix::new(PAGE);
        cache.insert_prefix(Some(1), &ids(8), &[1; 8]);
        let freed = cache.evict(1_000);
        assert_eq!(freed.len(), 8);
        assert_eq!(cache.evictable_size(), 0);
        assert!(cache.evict(1_000).is_empty(), "a second call must not spin");
    }

    /// A namespace nobody has used costs nothing, and using one is
    /// what creates it.
    #[test]
    fn a_namespace_is_created_on_first_use() {
        let mut cache = SaltedRadix::new(PAGE);
        assert_eq!(cache.total_size(), 0);
        assert!(cache.evict(4).is_empty(), "nothing to evict yet");
        cache.insert_prefix(Some(5), &ids(4), &[0; 4]);
        assert_eq!(cache.total_size(), 4);
    }
}
