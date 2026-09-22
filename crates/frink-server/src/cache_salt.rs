//! `cache_salt`: which caller's namespace a request's cached prefixes
//! belong to.
//!
//! A prefix cache is shared state keyed by token ids. Without a salt,
//! one caller's prompt can be answered from another caller's cached
//! prefix, and the shared leading tokens are usually the system prompt
//! -- the part a caller most expects to be theirs. That is an
//! ISOLATION property, not a performance knob, which is why frink
//! refused the field by name rather than ignoring it.
//!
//! # Hashed, not stored
//!
//! The caller's string never reaches a cache. It is hashed to a `u64`
//! and only the hash is kept, so a heap dump of the cache does not
//! carry whatever the caller chose to namespace by -- which may be a
//! tenant id, an account, or something they considered secret.
//!
//! A hash collision would merge two namespaces, and `u64` from
//! `DefaultHasher` makes that vanishingly unlikely for the number of
//! distinct salts one server sees. It is written down rather than
//! ignored: the failure mode is two callers sharing a namespace, which
//! is the behaviour they had before the field existed.
//!
//! # Where it is NOT honoured
//!
//! The paged store's radix tree has no per-namespace scoping: it walks
//! one tree keyed by token ids, and its nodes hold block indices the
//! whole deployment shares. Adding a namespace there is a change to
//! the tree and its eviction, not a parameter, so a paged request that
//! names a salt is REFUSED BY NAME. Serving it would be the worst of
//! the three options: a caller who asked for isolation, told they got
//! it, and sharing anyway.

use std::hash::{Hash, Hasher};

/// The namespace id for a caller's salt, or `None` for the shared one.
///
/// An EMPTY string is `None` rather than its own namespace: a caller
/// who sends `""` has named nothing, and giving them a private
/// namespace keyed on emptiness would silently stop them sharing with
/// themselves across requests.
pub(crate) fn namespace(salt: Option<&str>) -> Option<u64> {
    let salt = salt?.trim();
    if salt.is_empty() {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    salt.hash(&mut hasher);
    Some(hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_salt_is_the_same_namespace() {
        assert_eq!(namespace(Some("tenant-a")), namespace(Some("tenant-a")));
    }

    #[test]
    fn different_salts_are_different_namespaces() {
        assert_ne!(namespace(Some("tenant-a")), namespace(Some("tenant-b")));
    }

    /// Absent, empty and whitespace all mean "the caller named
    /// nothing", which is the shared namespace they had before the
    /// field existed -- NOT a private one keyed on emptiness, which
    /// would stop such a caller sharing with themselves.
    #[test]
    fn naming_nothing_is_the_shared_namespace() {
        for nothing in [None, Some(""), Some("   ")] {
            assert_eq!(namespace(nothing), None, "{nothing:?}");
        }
    }

    /// The caller's string is not recoverable from what is stored.
    #[test]
    fn the_salt_itself_is_not_kept() {
        let secret = "acct_9f2c-super-secret";
        let ns = namespace(Some(secret)).expect("a namespace");
        assert!(
            !format!("{ns}").contains("acct"),
            "the salt survived into the id"
        );
    }
}
