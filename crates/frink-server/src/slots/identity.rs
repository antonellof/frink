//! What a saved slot is *of*, and how a restore refuses a slot that is
//! of something else.
//!
//! A slot file holds raw attention state: per-layer K and V floats
//! computed by one set of weights at one quantisation for one token
//! sequence. Restoring those numbers under a different checkpoint does
//! not fail, and does not look like a failure -- the shapes can match,
//! the decode runs, and the answer is confidently wrong. So identity is
//! not a nicety here; it is the difference between a cache and a
//! corruption.
//!
//! # Why the identity is not the path, the size, or the config
//!
//! This repo has been bitten twice in one week by a cache keyed on
//! something that could not tell two things apart: a Metal buffer cache
//! keyed on host address and length, and a resident activation matched
//! on length alone. The candidates that look sufficient and are not:
//!
//! - **The file path.** Two servers, two machines, one path; or the same
//!   path re-quantised in place.
//! - **The file size.** Two fine-tunes of one base at one quantisation
//!   are byte-for-byte the same length.
//! - **The decoder config.** Same reason. Two fine-tunes of Llama-3.2-1B
//!   at Q4_K_M agree on every hyper-parameter there is.
//!
//! Every one of those admits a pair of genuinely different checkpoints,
//! and the pair it admits is the *likely* pair -- fine-tunes of one base
//! are exactly what an operator has several of.
//!
//! # What is actually hashed
//!
//! [`fingerprint_gguf`] takes SHA-256 over a canonical
//! encoding of:
//!
//! 1. the GGUF version, and every metadata key/value in sorted key
//!    order -- architecture, all hyper-parameters, and the entire
//!    tokenizer, typed, so `U32(4)` and `String("4")` cannot collide;
//! 2. the full tensor directory: every tensor's name, shape, dtype tag
//!    and data offset, which is what changes between two quantisations
//!    of one model;
//! 3. up to [`SAMPLE_BYTES`] from the head **and** the tail of every
//!    tensor's data.
//!
//! Step 3 is the one that separates two fine-tunes, and it is why this
//! reads weights at all: steps 1 and 2 are identical for them. Head and
//! tail rather than head alone because a checkpoint edited only in its
//! last rows -- a re-trained output head, a patched embedding -- is a
//! real shape, and a head-only sample would call it the same file.
//!
//! It is a sample, not the whole file, and that is a deliberate trade
//! stated here rather than buried: hashing 70 GB on every slot operation
//! would make the operation unusable, and two checkpoints that agree on
//! every metadata value, every tensor's name/shape/dtype/offset, and the
//! first and last 4 KiB of every single tensor are not a pair that
//! exists. What the sample cannot do is detect deliberate collision, and
//! it is not asked to: a slot file is not a trust boundary, it is an
//! operator's own cache.
//!
//! # Three checks, in order
//!
//! Mirroring [`frink_core::kv_signature`], whose rule this module
//! follows rather than reinvents: a stored slot is (1) rejected if it
//! carries no identity at all -- absence is never agreement; (2) checked
//! against its own payload, so a header may not vouch for a width the
//! floats do not have; and only then (3) compared to what this server
//! is serving. Step 2 is the easy one to skip: without it, "the header
//! says 8 KV heads" and "there are 8 KV heads" are two claims a reader
//! would be treating as one.

use std::path::Path;

use frink_gguf::{GgufFile, GgufValue};
use sha2::{Digest, Sha256};

/// Bytes sampled from each end of each tensor's data. 4 KiB is one page
/// and covers several rows of any real tensor.
pub(crate) const SAMPLE_BYTES: usize = 4096;

/// The identity a slot file carries and a restore checks.
///
/// The shape fields are here as well as the fingerprint because they are
/// what makes a refusal *legible*: "n_kv_heads 8 != 4" sends an operator
/// somewhere, and a hex digest mismatch alone does not. They are also
/// the half that is checkable against the payload, which the fingerprint
/// can never be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotIdentity {
    /// The checkpoint's display name (`ModelConfig::name`, from
    /// `general.name`), for the refusal message. Not part of the
    /// decision on its own -- two names can be one file and one name two
    /// files -- which is why it is compared first only for legibility
    /// and the fingerprint is compared regardless.
    pub(crate) model_name: String,
    pub(crate) n_layers: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// The KV element type. Only `f32` exists today; carrying it means
    /// a future f16 KV tier invalidates these files rather than
    /// reinterpreting their bytes.
    pub(crate) dtype: frink_core::kv_signature::KvDtype,
    /// Lowercase hex SHA-256 over the checkpoint, as described in the
    /// module note.
    pub(crate) fingerprint: String,
}

/// The first field on which two identities differ, and both values.
///
/// One variant carrying the field name rather than one variant per
/// field: a new field must be added to [`SlotIdentity::compare`]'s list,
/// which is a single place, instead of to an enum and a message and a
/// match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IdentityMismatch {
    pub(crate) field: &'static str,
    pub(crate) saved: String,
    pub(crate) serving: String,
}

impl std::fmt::Display for IdentityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: the slot was saved under {} and this server is serving {}",
            self.field, self.saved, self.serving
        )
    }
}

impl SlotIdentity {
    /// Field-by-field comparison, naming the FIRST field that differs.
    ///
    /// Shape before fingerprint on purpose: when both differ, the shape
    /// is the answer an operator can act on.
    pub(crate) fn compare(&self, serving: &SlotIdentity) -> Result<(), IdentityMismatch> {
        // Exhaustive destructure with no `..`: a field added to
        // `SlotIdentity` and not added below stops compiling here,
        // rather than becoming a field nothing compares.
        let SlotIdentity {
            model_name,
            n_layers,
            n_kv_heads,
            head_dim,
            dtype,
            fingerprint,
        } = self;
        let pairs: [(&'static str, String, String); 6] = [
            ("model", model_name.clone(), serving.model_name.clone()),
            (
                "n_layers",
                n_layers.to_string(),
                serving.n_layers.to_string(),
            ),
            (
                "n_kv_heads",
                n_kv_heads.to_string(),
                serving.n_kv_heads.to_string(),
            ),
            (
                "head_dim",
                head_dim.to_string(),
                serving.head_dim.to_string(),
            ),
            (
                "kv_dtype",
                dtype.name().to_string(),
                serving.dtype.name().to_string(),
            ),
            (
                "checkpoint",
                fingerprint.clone(),
                serving.fingerprint.clone(),
            ),
        ];
        for (field, saved, serving) in pairs {
            if saved != serving {
                return Err(IdentityMismatch {
                    field,
                    saved,
                    serving,
                });
            }
        }
        Ok(())
    }
}

/// Why a checkpoint could not be fingerprinted. Never a reason to serve
/// a slot anyway: a restore that cannot establish identity is refused.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FingerprintError {
    #[error(
        "this server is not serving a GGUF checkpoint on disk, so a slot cannot be identified \
         with one. Slots need -m/--model pointing at a .gguf file"
    )]
    NoCheckpoint,
    #[error("reading the checkpoint {path} to identify it: {source}")]
    Unreadable {
        path: String,
        #[source]
        source: frink_gguf::GgufError,
    },
}

/// SHA-256 over the canonical encoding described in the module note.
pub(crate) fn fingerprint_gguf(path: &Path) -> Result<String, FingerprintError> {
    let file = GgufFile::open(path).map_err(|source| FingerprintError::Unreadable {
        path: path.display().to_string(),
        source,
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"frink.slot.checkpoint.v1");
    hasher.update(file.version.to_le_bytes());

    // Sorted, because `HashMap` iteration order is not stable across
    // runs and a fingerprint that changed per process would refuse every
    // slot it had itself written.
    let mut keys: Vec<&String> = file.metadata.keys().collect();
    keys.sort();
    hasher.update((keys.len() as u64).to_le_bytes());
    for key in keys {
        hash_len_prefixed(&mut hasher, key.as_bytes());
        hash_value(&mut hasher, &file.metadata[key]);
    }

    hasher.update((file.tensors.len() as u64).to_le_bytes());
    for tensor in &file.tensors {
        hash_len_prefixed(&mut hasher, tensor.name.as_bytes());
        hasher.update((tensor.shape.len() as u64).to_le_bytes());
        for dim in &tensor.shape {
            hasher.update(dim.to_le_bytes());
        }
        hasher.update(tensor.dtype.to_tag().to_le_bytes());
        hasher.update(tensor.offset.to_le_bytes());
        // A tensor this build cannot size, or one the mmap refuses,
        // contributes its absence rather than being skipped silently:
        // "no sample" is itself part of the identity.
        match file.tensor_bytes(&tensor.name) {
            Ok(bytes) => {
                let head = &bytes[..bytes.len().min(SAMPLE_BYTES)];
                let tail = &bytes[bytes.len().saturating_sub(SAMPLE_BYTES)..];
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(head);
                hasher.update(tail);
            }
            Err(_) => hasher.update(b"\xffunsampled"),
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn hash_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Typed encoding, so two values that print the same but are not the
/// same do not hash the same.
fn hash_value(hasher: &mut Sha256, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => {
            hasher.update([0u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::I8(v) => {
            hasher.update([1u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::U16(v) => {
            hasher.update([2u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            hasher.update([3u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            hasher.update([4u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            hasher.update([5u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            hasher.update([6u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::Bool(v) => {
            hasher.update([7u8]);
            hasher.update([*v as u8]);
        }
        GgufValue::String(v) => {
            hasher.update([8u8]);
            hash_len_prefixed(hasher, v.as_bytes());
        }
        GgufValue::U64(v) => {
            hasher.update([9u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            hasher.update([10u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            hasher.update([11u8]);
            hasher.update(v.to_le_bytes());
        }
        GgufValue::Array(items) => {
            hasher.update([12u8]);
            hasher.update((items.len() as u64).to_le_bytes());
            for item in items {
                hash_value(hasher, item);
            }
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::kv_signature::KvDtype;

    fn identity(name: &str, layers: usize, fingerprint: &str) -> SlotIdentity {
        SlotIdentity {
            model_name: name.to_string(),
            n_layers: layers,
            n_kv_heads: 4,
            head_dim: 8,
            dtype: KvDtype::F32,
            fingerprint: fingerprint.to_string(),
        }
    }

    #[test]
    fn an_identity_matches_itself() {
        let a = identity("llama", 16, "abc");
        assert_eq!(a.compare(&a), Ok(()));
    }

    /// The case this whole module exists for: two checkpoints of the
    /// same architecture, same geometry, same quantisation -- two
    /// fine-tunes of one base -- differ ONLY in their weights. A cache
    /// keyed on shape, path or size calls them the same model and
    /// restores one's attention state under the other's weights.
    #[test]
    fn two_checkpoints_of_identical_shape_are_told_apart_by_their_weights() {
        let saved = identity("llama", 16, "aaaa");
        let serving = identity("llama", 16, "bbbb");
        let err = saved.compare(&serving).unwrap_err();
        assert_eq!(err.field, "checkpoint");
        assert_eq!(err.saved, "aaaa");
        assert_eq!(err.serving, "bbbb");
    }

    /// A refusal has to name what changed. "Cache miss" or "invalid
    /// file" sends an operator looking in the wrong place; `n_layers`
    /// sends them to the model they swapped in.
    #[test]
    fn a_shape_difference_is_named_before_the_digest() {
        let saved = identity("llama", 16, "aaaa");
        let serving = identity("llama", 32, "bbbb");
        let err = saved.compare(&serving).unwrap_err();
        assert_eq!(err.field, "n_layers", "the actionable field wins");
        assert!(err.to_string().contains("16"), "{err}");
        assert!(err.to_string().contains("32"), "{err}");
    }

    #[test]
    fn a_different_model_name_is_named_first_of_all() {
        let saved = identity("llama", 16, "aaaa");
        let serving = identity("qwen2", 16, "aaaa");
        assert_eq!(saved.compare(&serving).unwrap_err().field, "model");
    }

    /// Typed hashing: a metadata value of `U32(4)` and one of
    /// `String("4")` are different checkpoints, and an encoding that
    /// stringified both would hash them the same.
    #[test]
    fn metadata_values_of_different_types_do_not_hash_alike() {
        let mut a = Sha256::new();
        hash_value(&mut a, &GgufValue::U32(4));
        let mut b = Sha256::new();
        hash_value(&mut b, &GgufValue::String("4".to_string()));
        assert_ne!(hex(&a.finalize()), hex(&b.finalize()));
    }

    /// Length-prefixing: without it, the two-key metadata `{"ab": ...,
    /// "c": ...}` and `{"a": ..., "bc": ...}` would feed the hasher the
    /// same byte stream.
    #[test]
    fn adjacent_strings_cannot_be_reassociated_across_their_boundary() {
        let mut a = Sha256::new();
        hash_len_prefixed(&mut a, b"ab");
        hash_len_prefixed(&mut a, b"c");
        let mut b = Sha256::new();
        hash_len_prefixed(&mut b, b"a");
        hash_len_prefixed(&mut b, b"bc");
        assert_ne!(hex(&a.finalize()), hex(&b.finalize()));
    }
}
