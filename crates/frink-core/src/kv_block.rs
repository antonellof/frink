//! Content-addressed identity for KV-cache blocks: what makes two
//! stored prefixes *the same* prefix, across processes and across
//! restarts.
//!
//! A block is a fixed-size run of token positions in one sequence. Its
//! identity is a **parent-chained SHA-256**:
//!
//! ```text
//! root       = H(domain, "root",  model, extra_keys)
//! block[i]   = H(domain, "block", model, extra_keys, parent, token_ids)
//! parent     = root for block[0], block[i-1] otherwise
//! ```
//!
//! Chaining is what makes a hash mean *this token run, at this offset,
//! after exactly this history* rather than merely "these tokens".
//! Without it, two different prompts that happen to share an interior
//! token run would collide on a block whose KV state depends on
//! everything before it, and the cache would hand back state computed
//! under a different history -- silent wrong answers, not a miss.
//!
//! `extra_keys` is the salt slot for anything that changes what the KV
//! state *means* without changing the token ids: a LoRA adapter's
//! identity, an image/audio embedding's identity for a multimodal
//! prompt. Sampling parameters are deliberately **not** part of the
//! key: KV state is sampling-independent, and folding temperature or a
//! seed into the key would only shatter the cache.
//!
//! Every field is length-prefixed before hashing, so no two different
//! `(model, extra_keys, parent, tokens)` tuples can serialize to the
//! same byte string. Concatenating raw fields would let
//! `extra_keys = ["ab", "c"]` and `["a", "bc"]` hash identically.
//!
//! This module is identity only -- no storage, no eviction, no I/O.
//! The disk tier that will consume it is `kv-ssd-tier` in
//! `4dc195e5:docs/plans/serving-and-tiered-kv.md`.

use sha2::{Digest, Sha256};

/// Domain separator, versioned. Bumping it invalidates every hash ever
/// computed, which is the intended effect if the encoding below ever
/// has to change: old blocks become unreachable rather than
/// misinterpreted.
const HASH_DOMAIN: &[u8] = b"frink-kv-block-v1";

const TAG_ROOT: &[u8] = b"root";
const TAG_BLOCK: &[u8] = b"block";

/// A block's content address: the 32-byte SHA-256 of its chained
/// identity.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash([u8; 32]);

impl BlockHash {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        BlockHash(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase hex, 64 characters. This is the on-disk file name the
    /// SSD tier will use.
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for byte in self.0 {
            out.push(char::from_digit((byte >> 4) as u32, 16).unwrap());
            out.push(char::from_digit((byte & 0xf) as u32, 16).unwrap());
        }
        out
    }

    /// The first `n` hex characters, for sharding blocks into
    /// subdirectories so one directory never holds every block on the
    /// machine. `n` is clamped to the digest length.
    pub fn shard_prefix(&self, n: usize) -> String {
        self.to_hex().chars().take(n.min(64)).collect()
    }
}

impl std::fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Short form: a full 64-char digest in a log line is noise, and
        // 12 hex characters are plenty to follow one block through a
        // trace.
        write!(f, "BlockHash({}…)", self.shard_prefix(12))
    }
}

impl std::fmt::Display for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Writes one length-prefixed field, so the concatenation of fields is
/// injective (see the module note on `["ab","c"]` vs `["a","bc"]`).
fn absorb(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Hashes blocks for one (model, extra_keys) identity. Cheap to build;
/// the root is computed once and reused for every chain.
#[derive(Clone, Debug)]
pub struct BlockHasher {
    model: String,
    extra_keys: Vec<String>,
    root: BlockHash,
}

impl BlockHasher {
    /// `model` should identify the *weights*, not the file path -- two
    /// servers on different machines must agree on it for a shared or
    /// restored cache to be reusable at all.
    ///
    /// `extra_keys` are ordered: they are hashed in the order given, so
    /// callers must be consistent (sort them if the source is a set).
    pub fn new<S: AsRef<str>>(model: impl Into<String>, extra_keys: &[S]) -> Self {
        let model = model.into();
        let extra_keys: Vec<String> = extra_keys.iter().map(|k| k.as_ref().to_string()).collect();
        let mut hasher = Sha256::new();
        absorb(&mut hasher, HASH_DOMAIN);
        absorb(&mut hasher, TAG_ROOT);
        absorb(&mut hasher, model.as_bytes());
        absorb_extra_keys(&mut hasher, &extra_keys);
        let root = BlockHash(hasher.finalize().into());
        BlockHasher {
            model,
            extra_keys,
            root,
        }
    }

    /// The seed every chain starts from: the identity of "no tokens
    /// yet, under this model and these extra keys". A chain rooted here
    /// can never be confused with one rooted under a different model or
    /// a different LoRA.
    pub fn root(&self) -> BlockHash {
        self.root
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn extra_keys(&self) -> &[String] {
        &self.extra_keys
    }

    /// Hashes one block: this token run, following `parent`.
    ///
    /// `model` and `extra_keys` are folded in again even though the
    /// parent chain already carries them, so a single block hash is
    /// verifiable from `(parent, tokens)` alone without walking back to
    /// the root.
    pub fn block(&self, parent: &BlockHash, token_ids: &[usize]) -> BlockHash {
        let mut hasher = Sha256::new();
        absorb(&mut hasher, HASH_DOMAIN);
        absorb(&mut hasher, TAG_BLOCK);
        absorb(&mut hasher, self.model.as_bytes());
        absorb_extra_keys(&mut hasher, &self.extra_keys);
        absorb(&mut hasher, parent.as_bytes());
        hasher.update((token_ids.len() as u64).to_le_bytes());
        for &token in token_ids {
            hasher.update((token as u64).to_le_bytes());
        }
        BlockHash(hasher.finalize().into())
    }

    /// Hashes `tokens` as a chain of `block_size`-token blocks, rooted
    /// at [`root`](Self::root).
    ///
    /// **Only whole blocks are hashed.** A trailing partial block gets
    /// no hash: its content is still growing, so any identity assigned
    /// to it now would name a different token run a moment later.
    /// [`full_blocks`] reports how many hashes a length yields.
    ///
    /// Because each block's hash covers its parent, the chain of a
    /// prompt is a strict prefix of the chain of anything that extends
    /// it -- which is exactly the lookup a prefix cache needs.
    pub fn chain(&self, tokens: &[usize], block_size: usize) -> Vec<BlockHash> {
        assert!(block_size > 0, "block_size must be positive");
        let mut parent = self.root;
        let mut out = Vec::with_capacity(full_blocks(tokens.len(), block_size));
        for block in tokens.chunks_exact(block_size) {
            parent = self.block(&parent, block);
            out.push(parent);
        }
        out
    }
}

fn absorb_extra_keys(hasher: &mut Sha256, extra_keys: &[String]) {
    hasher.update((extra_keys.len() as u64).to_le_bytes());
    for key in extra_keys {
        absorb(hasher, key.as_bytes());
    }
}

/// How many whole blocks `token_count` tokens make at `block_size`.
pub fn full_blocks(token_count: usize, block_size: usize) -> usize {
    assert!(block_size > 0, "block_size must be positive");
    token_count / block_size
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hasher() -> BlockHasher {
        BlockHasher::new("model-a", &[] as &[&str])
    }

    #[test]
    fn hashing_is_deterministic() {
        let h = hasher();
        let a = h.chain(&[1, 2, 3, 4], 2);
        let b = BlockHasher::new("model-a", &[] as &[&str]).chain(&[1, 2, 3, 4], 2);
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }

    /// The encoding is a persistence format: a block written by one
    /// build must still be findable by the next one. If this fails, the
    /// hash inputs changed and `HASH_DOMAIN` must be bumped so old
    /// blocks become unreachable rather than misread.
    ///
    /// These digests were cross-validated against an independent Python
    /// `hashlib` implementation of the same length-prefixed encoding --
    /// they pin the *encoding*, not merely whatever this code happens
    /// to produce.
    ///
    /// They changed once, on 2026-09-19, when the project was renamed
    /// from Ferrox to Frink and `HASH_DOMAIN` went with it. That is
    /// the bump this comment asks for rather than a break: every
    /// previously written block becomes unreachable instead of being
    /// read under a name that no longer describes it. The three values
    /// below were re-derived in Python, not copied from a failing
    /// assertion.
    #[test]
    fn hash_encoding_is_stable() {
        let h = BlockHasher::new("model-a", &["lora:alpha"]);
        assert_eq!(
            h.root().to_hex(),
            "08d2e26f303283373f6e74dd73fd2d8c00e9d22edeb6d35df6f3557fd78172f1"
        );
        let chain = h.chain(&[1, 2, 3, 4], 2);
        assert_eq!(
            chain[0].to_hex(),
            "fd891b2d4f3154d4877be7163bbba82332178dbf7c52d23c04cdde8b366ef31f"
        );
        assert_eq!(
            chain[1].to_hex(),
            "24a65c69d897153b0a798429b379a9fff184a89a0193e4dc4a6520f537509795"
        );
    }

    #[test]
    fn different_models_never_share_a_chain() {
        let a = BlockHasher::new("model-a", &[] as &[&str]);
        let b = BlockHasher::new("model-b", &[] as &[&str]);
        assert_ne!(a.root(), b.root());
        assert_ne!(a.chain(&[1, 2], 2), b.chain(&[1, 2], 2));
    }

    /// The salt slot: same model, same tokens, different LoRA identity
    /// must be a different block. A cache that ignored this would serve
    /// base-model KV state to an adapter request.
    #[test]
    fn extra_keys_change_identity() {
        let plain = BlockHasher::new("model-a", &[] as &[&str]);
        let lora = BlockHasher::new("model-a", &["lora:alpha"]);
        let other = BlockHasher::new("model-a", &["lora:beta"]);
        assert_ne!(plain.chain(&[1, 2], 2), lora.chain(&[1, 2], 2));
        assert_ne!(lora.chain(&[1, 2], 2), other.chain(&[1, 2], 2));
    }

    /// Length prefixing, stated as the property it protects: two
    /// different key lists whose concatenations are byte-identical must
    /// still hash differently.
    #[test]
    fn extra_keys_are_not_ambiguous_under_concatenation() {
        let a = BlockHasher::new("m", &["ab", "c"]);
        let b = BlockHasher::new("m", &["a", "bc"]);
        assert_ne!(a.root(), b.root());
        let c = BlockHasher::new("mab", &["c"]);
        assert_ne!(a.root(), c.root());
    }

    /// The point of chaining: the same token run after a different
    /// history is a different block, because its KV state is.
    #[test]
    fn same_tokens_under_different_parents_differ() {
        let h = hasher();
        let left = h.chain(&[9, 9, 5, 6], 2);
        let right = h.chain(&[7, 7, 5, 6], 2);
        assert_ne!(left[0], right[0]);
        assert_ne!(
            left[1], right[1],
            "block [5,6] must differ under different parents"
        );
    }

    /// The prefix-cache lookup property: extending a prompt extends its
    /// chain, it does not rewrite it.
    #[test]
    fn chain_of_a_prefix_is_a_prefix_of_the_chain() {
        let h = hasher();
        let short = h.chain(&[1, 2, 3, 4], 2);
        let long = h.chain(&[1, 2, 3, 4, 5, 6], 2);
        assert_eq!(long.len(), 3);
        assert_eq!(&long[..2], &short[..]);
    }

    #[test]
    fn block_boundaries_are_part_of_identity() {
        let h = hasher();
        let by_two = h.chain(&[1, 2, 3, 4], 2);
        let by_four = h.chain(&[1, 2, 3, 4], 4);
        assert_eq!(by_two.len(), 2);
        assert_eq!(by_four.len(), 1);
        assert_ne!(by_two[1], by_four[0]);
    }

    /// A still-growing tail has no stable identity, so it gets no hash.
    #[test]
    fn trailing_partial_block_is_not_hashed() {
        let h = hasher();
        assert_eq!(h.chain(&[1, 2, 3], 2).len(), 1);
        assert_eq!(h.chain(&[1], 2).len(), 0);
        assert_eq!(h.chain(&[], 2).len(), 0);
        assert_eq!(full_blocks(3, 2), 1);
        assert_eq!(full_blocks(0, 2), 0);
        assert_eq!(
            h.chain(&[1, 2, 3], 2)[0],
            h.chain(&[1, 2], 2)[0],
            "a partial tail must not change the blocks before it"
        );
    }

    #[test]
    fn token_values_and_order_matter() {
        let h = hasher();
        assert_ne!(h.chain(&[1, 2], 2), h.chain(&[2, 1], 2));
        assert_ne!(h.chain(&[1, 2], 2), h.chain(&[1, 3], 2));
    }

    #[test]
    fn hex_and_shard_prefix_are_well_formed() {
        let h = hasher();
        let hash = h.chain(&[1, 2], 2)[0];
        let hex = hash.to_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        assert_eq!(hash.shard_prefix(2), hex[..2]);
        assert_eq!(hash.shard_prefix(999).len(), 64);
        assert_eq!(BlockHash::from_bytes(*hash.as_bytes()), hash);
    }
}
