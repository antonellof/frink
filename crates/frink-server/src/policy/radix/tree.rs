//! The shared radix tree: nodes, page keys, and the split that every
//! prefix cache in this module is built on.
//!
//! A radix prefix cache is a trie over *pages* of tokens, not over
//! tokens. An edge carries a whole run of token ids; a node's identity
//! is the path from the root to it. Two prompts that share a prefix
//! share the nodes that spell it, and therefore share the KV pages
//! those nodes point at -- which is the entire point: the second
//! request does not recompute what the first already did.
//!
//! # Why pages, not tokens
//!
//! KV memory is allocated in pages. A prefix that is reusable only for
//! part of a page is not reusable at all, because the page it lands in
//! holds state for tokens the new prompt does not have. So every match
//! length is rounded *down* to a whole page ([`align_down`]), and every
//! child is registered under the key of its **whole first page**
//! ([`page_key`]) rather than its first token. Two prompts that agree on
//! three tokens of a four-token page get different keys, no shared
//! edge, and no reuse -- exactly right, and cheaper to check than a
//! per-token walk that would then have to be truncated anyway.
//!
//! # The split contract
//!
//! When a new prompt diverges inside an existing node, that node is cut
//! in two. [`RadixTree::split_at`] allocates the **prefix** as a new
//! node and leaves the **suffix** in the original slot, so every handle
//! anyone is already holding still names the same end boundary. Getting
//! this backwards silently re-points live locks at a shorter prefix.
//! Which fields follow the prefix and which stay with the suffix is a
//! per-field decision, documented on `split_at` and pinned by its
//! tests.
//!
//! Ported 1:1 from FreeToken's `python/freetoken/kvcache/radix_cache.py`
//! (Apache-2.0), which follows the standard radix-cache design; see
//! `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::HashMap;

/// A node's slot in the arena. Stable for the life of the node: a
/// split keeps the original id on the suffix precisely so ids stay
/// meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// The key a child is registered under: its whole first page of tokens.
///
/// A query shorter than a page yields a shorter key, which by
/// construction matches no child -- that is how a sub-page prompt
/// correctly resolves to the root instead of half-matching an edge.
pub type PageKey = Vec<u32>;

/// The key `input_ids` presents to the children of a node, given the
/// page size.
pub fn page_key(input_ids: &[u32], page_size: usize) -> PageKey {
    input_ids[..page_size.min(input_ids.len())].to_vec()
}

/// Round `a` down to a multiple of `b`.
pub fn align_down(a: usize, b: usize) -> usize {
    (a / b) * b
}

/// Round `a` up to a multiple of `b`.
pub fn align_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// The length of the longest common prefix of `a` and `b`.
///
/// Equal on the overlap means the shorter length: a query shorter than
/// the node's key matches all of itself, and the caller decides what
/// that means.
pub fn match_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// One node of the tree.
///
/// The two currencies beyond plain KV are carried here rather than in a
/// subclass, because [`RadixTree::split_at`] has to know how each one
/// splits and a split is a tree operation. A plain cache simply never
/// sets them.
#[derive(Debug, Clone)]
pub struct Node {
    /// The token ids this edge spells. Always a whole number of pages.
    pub key: Vec<u32>,
    /// The KV page indices holding those tokens' state. Owned: a caller
    /// may reuse the buffer it inserted from the instant insert
    /// returns.
    pub value: Vec<u32>,
    pub parent: Option<NodeId>,
    pub children: HashMap<PageKey, NodeId>,
    /// Full-KV lock depth. Non-zero means some live request is reading
    /// these pages, so they may not be evicted.
    pub ref_count: u32,
    /// LRU age. Larger is newer; the evictors pop the minimum.
    pub timestamp: i64,

    /// Recurrent-state (GDN) snapshot slot attached at this node's
    /// **end** boundary, for hybrid models. A point, not a span.
    pub mamba_value: Option<u32>,
    pub mamba_ref_count: u32,

    /// Sliding-window models: the window KV for these tokens has been
    /// freed while the full KV survives. Such a node still serves a
    /// full-KV match, but a window-attention layer cannot read it.
    pub swa_tombstone: bool,
    /// Window lock depth. Bounded by `ref_count`: the window pins a
    /// suffix of what the full lock pins.
    pub swa_ref_count: u32,
    /// Handle identifying *which* reader's window ends here, so one
    /// reader's unlock releases only its own window.
    pub swa_uuid: Option<u64>,
}

impl Node {
    fn new(timestamp: i64) -> Self {
        Node {
            key: Vec::new(),
            value: Vec::new(),
            parent: None,
            children: HashMap::new(),
            ref_count: 0,
            timestamp,
            mamba_value: None,
            mamba_ref_count: 0,
            swa_tombstone: false,
            swa_ref_count: 0,
            swa_uuid: None,
        }
    }

    pub fn length(&self) -> usize {
        self.key.len()
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }
}

/// The arena the three caches share.
///
/// Node 0 is always the root: it spells nothing, is never evicted, and
/// is the answer to "no prefix matched".
#[derive(Debug)]
pub struct RadixTree {
    nodes: Vec<Node>,
    page_size: usize,
    /// Monotonic source for `swa_uuid`.
    uuid: u64,
}

pub const ROOT: NodeId = NodeId(0);

impl RadixTree {
    pub fn new(page_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be positive");
        let mut root = Node::new(0);
        // The root is permanently protected: an evictor that could take
        // it would be evicting "the empty prefix".
        root.ref_count = 1;
        RadixTree {
            nodes: vec![root],
            page_size,
            uuid: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0 as usize]
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).parent
    }

    /// A fresh, unlinked node.
    pub fn alloc(&mut self, timestamp: i64) -> NodeId {
        self.nodes.push(Node::new(timestamp));
        NodeId((self.nodes.len() - 1) as u32)
    }

    /// The next window-boundary handle. Handed out lazily, only to a
    /// node that actually becomes some reader's boundary.
    pub fn next_uuid(&mut self) -> u64 {
        self.uuid += 1;
        self.uuid
    }

    pub fn set_key_value(&mut self, id: NodeId, key: Vec<u32>, value: Vec<u32>) {
        assert_eq!(
            key.len(),
            value.len(),
            "a node must hold one page index per token"
        );
        let node = self.node_mut(id);
        node.key = key;
        node.value = value;
    }

    /// Link `id` under `parent`, registered by its first page.
    pub fn set_parent(&mut self, id: NodeId, parent: NodeId) {
        let key = page_key(&self.node(id).key, self.page_size);
        self.node_mut(id).parent = Some(parent);
        self.node_mut(parent).children.insert(key, id);
    }

    /// The child of `id` that `input_ids` would follow, if any.
    pub fn child(&self, id: NodeId, input_ids: &[u32]) -> Option<NodeId> {
        self.node(id)
            .children
            .get(&page_key(input_ids, self.page_size))
            .copied()
    }

    /// Detach `id` from its parent and return that parent.
    ///
    /// The node itself stays in the arena (its slot is never reused);
    /// what matters is that nothing can reach it any more.
    pub fn unlink(&mut self, id: NodeId) -> NodeId {
        let parent = self.node(id).parent.expect("the root is never unlinked");
        let key = page_key(&self.node(id).key, self.page_size);
        self.node_mut(parent).children.remove(&key);
        self.node_mut(id).parent = None;
        parent
    }

    /// Cut `id` in two at `pos`, returning the **prefix**.
    ///
    /// The original id keeps the *suffix*, so a handle taken before the
    /// split still names the same end boundary afterwards. What each
    /// field does:
    ///
    /// | field | prefix (new node) | suffix (original id) |
    /// |---|---|---|
    /// | `key` / `value` | `[..pos]` | `[pos..]` |
    /// | `ref_count`, `swa_ref_count`, `swa_tombstone` | copied -- both halves are covered by the same locks and the same window state | unchanged |
    /// | `swa_uuid` | **migrates** root-side | cleared |
    /// | `mamba_value`, `mamba_ref_count` | none -- a recurrent snapshot is taken *at* a boundary and cannot be split | kept |
    /// | `timestamp` | inherited | unchanged |
    ///
    /// `swa_uuid` migrates because it marks where one reader's window
    /// *starts*, which is the root-side end of the run -- leaving it on
    /// the suffix would release a different span than the one that was
    /// locked.
    pub fn split_at(&mut self, id: NodeId, pos: usize) -> NodeId {
        let length = self.node(id).length();
        assert!(
            pos > 0 && pos < length,
            "split_at({pos}) is not interior to a node of length {length}"
        );
        let parent = self.node(id).parent.expect("the root is never split");
        let timestamp = self.node(id).timestamp;
        let prefix = self.alloc(timestamp);

        let (head_key, tail_key) = {
            let key = &self.node(id).key;
            (key[..pos].to_vec(), key[pos..].to_vec())
        };
        let (head_value, tail_value) = {
            let value = &self.node(id).value;
            (value[..pos].to_vec(), value[pos..].to_vec())
        };

        self.set_key_value(prefix, head_key, head_value);
        self.set_parent(prefix, parent);
        {
            let original = self.node(id).clone();
            let head = self.node_mut(prefix);
            head.ref_count = original.ref_count;
            head.swa_ref_count = original.swa_ref_count;
            head.swa_tombstone = original.swa_tombstone;
            head.swa_uuid = original.swa_uuid;
        }
        self.node_mut(id).swa_uuid = None;

        self.set_key_value(id, tail_key, tail_value);
        self.set_parent(id, prefix);
        prefix
    }

    /// The concatenated page indices from the root down to `id`.
    pub fn path_value(&self, id: NodeId) -> Vec<u32> {
        let mut chunks: Vec<&[u32]> = Vec::new();
        let mut cur = id;
        while !self.node(cur).is_root() {
            chunks.push(&self.node(cur).value);
            cur = self.node(cur).parent.expect("non-root has a parent");
        }
        chunks.reverse();
        chunks.concat()
    }

    /// The number of tokens from the root down to `id`.
    pub fn path_len(&self, id: NodeId) -> usize {
        let mut total = 0;
        let mut cur = id;
        while !self.node(cur).is_root() {
            total += self.node(cur).length();
            cur = self.node(cur).parent.expect("non-root has a parent");
        }
        total
    }

    /// Every reachable node except the root, deepest-first order
    /// unspecified.
    pub fn walk(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut stack = vec![ROOT];
        while let Some(id) = stack.pop() {
            for child in self.node(id).children.values() {
                stack.push(*child);
                out.push(*child);
            }
        }
        out
    }

    /// Every reachable leaf except the root.
    pub fn leaves(&self) -> Vec<NodeId> {
        self.walk()
            .into_iter()
            .filter(|id| self.node(*id).is_leaf())
            .collect()
    }

    /// Structural invariants, for tests and `check_integrity`.
    pub fn check_structure(&self) {
        for id in self.walk() {
            let node = self.node(id);
            let parent = node.parent.expect("a walked node is not the root");
            assert_eq!(
                self.node(parent)
                    .children
                    .get(&page_key(&node.key, self.page_size)),
                Some(&id),
                "a node must be registered under its own first page key"
            );
            assert_eq!(node.key.len(), node.value.len(), "one page index per token");
            assert_eq!(
                node.length() % self.page_size,
                0,
                "a node must hold whole pages"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_with_chain(page_size: usize, key: &[u32], value: &[u32]) -> (RadixTree, NodeId) {
        let mut tree = RadixTree::new(page_size);
        let id = tree.alloc(1);
        tree.set_key_value(id, key.to_vec(), value.to_vec());
        tree.set_parent(id, ROOT);
        (tree, id)
    }

    /// A page key buckets a whole page, so two prompts that agree on
    /// all but the last token of a page share no edge at all.
    #[test]
    fn a_page_key_covers_a_whole_page_not_a_token() {
        assert_eq!(page_key(&[1, 7, 7, 2, 3, 3], 4), vec![1, 7, 7, 2]);
        assert_eq!(page_key(&[1, 7, 7, 2], 4), vec![1, 7, 7, 2]);
        assert_ne!(page_key(&[1, 7, 7, 2], 4), page_key(&[1, 7, 7, 9], 4));
        // A query shorter than a page yields a shorter key, which
        // matches no child -- it resolves to the root instead.
        assert_eq!(page_key(&[1, 7], 4), vec![1, 7]);
        assert_ne!(page_key(&[1, 7], 4), page_key(&[1, 7, 7, 2], 4));
    }

    #[test]
    fn reuse_is_bounded_to_whole_pages() {
        let key: Vec<u32> = (100..108).collect();
        assert_eq!(match_len(&key, &key), 8);
        assert_eq!(align_down(match_len(&key, &key), 4), 8);

        let mut q: Vec<u32> = (100..106).collect();
        q.extend([0, 0]);
        assert_eq!(match_len(&key, &q), 6);
        assert_eq!(align_down(6, 4), 4, "six matched tokens reuse one page");

        assert_eq!(match_len(&key, &[100, 101, 102]), 3);
        assert_eq!(match_len(&key, &[0, 101, 102]), 0);
    }

    /// The identity contract: handles taken before a split still name
    /// the same end boundary.
    #[test]
    fn split_keeps_the_original_id_as_the_suffix() {
        let (mut tree, id) = tree_with_chain(1, &[1, 2, 3, 4], &[10, 11, 12, 13]);
        let prefix = tree.split_at(id, 2);

        assert_ne!(prefix, id);
        assert_eq!(tree.node(prefix).key, vec![1, 2]);
        assert_eq!(tree.node(prefix).value, vec![10, 11]);
        assert_eq!(tree.node(id).key, vec![3, 4]);
        assert_eq!(tree.node(id).value, vec![12, 13]);
        assert_eq!(tree.node(id).parent, Some(prefix));
        assert_eq!(tree.node(prefix).parent, Some(ROOT));
        assert_eq!(tree.path_value(id), vec![10, 11, 12, 13]);
        tree.check_structure();
    }

    #[test]
    fn split_copies_locks_and_window_state_to_both_halves() {
        let (mut tree, id) = tree_with_chain(1, &[1, 2, 3, 4], &[10, 11, 12, 13]);
        {
            let node = tree.node_mut(id);
            node.ref_count = 2;
            node.swa_ref_count = 1;
            node.swa_tombstone = true;
            node.swa_uuid = Some(77);
            node.mamba_value = Some(99);
            node.mamba_ref_count = 1;
            node.timestamp = 42;
        }
        let prefix = tree.split_at(id, 2);

        assert_eq!(tree.node(prefix).ref_count, 2);
        assert_eq!(tree.node(prefix).swa_ref_count, 1);
        assert!(tree.node(prefix).swa_tombstone);
        assert_eq!(tree.node(prefix).timestamp, 42);
        assert_eq!(tree.node(id).ref_count, 2);

        // The window handle marks where a reader's window starts, so it
        // moves root-side and is cleared from the suffix.
        assert_eq!(tree.node(prefix).swa_uuid, Some(77));
        assert_eq!(tree.node(id).swa_uuid, None);

        // A recurrent snapshot is attached at an end boundary and
        // cannot be halved: it stays with the suffix, which is the half
        // that still ends where it was taken.
        assert_eq!(tree.node(prefix).mamba_value, None);
        assert_eq!(tree.node(prefix).mamba_ref_count, 0);
        assert_eq!(tree.node(id).mamba_value, Some(99));
        assert_eq!(tree.node(id).mamba_ref_count, 1);
    }

    #[test]
    #[should_panic(expected = "not interior")]
    fn splitting_at_an_edge_is_a_bug_not_a_no_op() {
        let (mut tree, id) = tree_with_chain(1, &[1, 2], &[10, 11]);
        tree.split_at(id, 0);
    }

    #[test]
    fn unlinking_makes_a_node_unreachable_and_returns_its_parent() {
        let (mut tree, id) = tree_with_chain(1, &[1, 2], &[10, 11]);
        assert_eq!(tree.leaves(), vec![id]);
        assert_eq!(tree.unlink(id), ROOT);
        assert!(tree.walk().is_empty());
        assert!(tree.node(ROOT).is_leaf());
    }
}
