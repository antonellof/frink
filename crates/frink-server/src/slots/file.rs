//! The `.fslot` file: one saved KV prefix, on disk.
//!
//! # Layout
//!
//! ```text
//! magic           8   b"FRXSLOT1"
//! format_version  4   u32 LE, checked against READABLE_FORMAT_VERSIONS
//! header_len      4   u32 LE
//! body_len        8   u64 LE
//! digest         32   SHA-256 over header || body
//! header  header_len  shape, counts, and the checkpoint identity
//! body      body_len  prompt tokens, pending logits, then per layer
//!                     all K floats followed by all V floats
//! ```
//!
//! Deliberately the same shape as
//! [`frink_core::kv_disk`](frink_core::kv_disk)'s block file, because
//! it defends against the same things and there is no reason for two
//! answers: a length-and-digest prefix that is checked against the real
//! file size *before* anything is parsed or allocated, an explicit
//! readable-version set rather than `<= CURRENT`, and identity handled
//! separately from framing. It is a different file because a slot is a
//! different thing: `kv_disk` stores exactly one whole block of a fixed
//! `block_size`, keyed by a content hash, and a slot is an
//! arbitrary-length prompt prefix named by an operator.
//!
//! llama.cpp's equivalent (`src/llama-context.cpp:3081-3141`) is magic +
//! version + token count + tokens + sequence state, and carries no model
//! identity at all: restoring a slot saved under a different checkpoint
//! is not detected there. See [`super::identity`] for what this one
//! carries instead.
//!
//! # Every length is bounded by the file
//!
//! Nothing here allocates from a number the file declared. `body_len`
//! and `header_len` are checked against the bytes actually present
//! first, and then the body's own contents must add up to `body_len`
//! EXACTLY -- `n_prompt * 4 + n_logits * 4 + n_layers * 2 * positions *
//! n_kv_heads * head_dim * 4`, in checked arithmetic. A header claiming
//! four billion layers fails that multiplication or that equality, on a
//! file the attacker had to supply, before a single `Vec` is reserved.

use frink_core::cache::KvCache;
use frink_core::kv_signature::KvDtype;
use sha2::{Digest, Sha256};

use super::identity::SlotIdentity;

const MAGIC: &[u8; 8] = b"FRXSLOT1";

/// Layout version written by this build.
pub(crate) const SLOT_FORMAT_VERSION: u32 = 1;

/// Versions this build can read. Explicit rather than `<=
/// SLOT_FORMAT_VERSION` so dropping an old layout is a deliberate edit
/// and not an accident of arithmetic.
pub(crate) const READABLE_FORMAT_VERSIONS: &[u32] = &[1];

/// magic + version + header_len + body_len + digest.
const PREFIX_LEN: usize = 8 + 4 + 4 + 8 + 32;

/// Fixed u32 fields in the header, before the two variable-length
/// strings: n_layers, n_kv_heads, head_dim, dtype, positions, n_prompt,
/// n_logits, model_name_len, fingerprint_len.
const HEADER_FIELDS: usize = 9;

/// Tag for the STORE's KV element type, which the identity check holds
/// against the serving model's (`ModelConfig::kv_dtype`). The body is
/// f32 on the wire for both: an f16 element widens to f32 and narrows
/// back exactly, so the tag says what the cache was, not how wide each
/// body float is, and the body arithmetic is one rule.
const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;

fn dtype_tag(dtype: KvDtype) -> u32 {
    match dtype {
        KvDtype::F32 => DTYPE_F32,
        KvDtype::F16 => DTYPE_F16,
    }
}

/// A slot as it exists in memory: the prompt it covers, the logits that
/// predict the token after it, and one KV cache per layer.
pub(crate) struct SlotPayload {
    pub(crate) tokens: Vec<usize>,
    pub(crate) pending_logits: Vec<f32>,
    pub(crate) layers: Vec<KvCache>,
}

/// A decoded file that has NOT yet been checked against the model this
/// server is serving.
///
/// Named after [`frink_core::kv_signature::UnverifiedBlock`] and for
/// the same reason: the type makes it impossible to reach a usable
/// payload without passing it the reader's own expectation.
pub(crate) struct UnverifiedSlot {
    identity: SlotIdentity,
    payload: SlotPayload,
}

impl UnverifiedSlot {
    pub(crate) fn identity(&self) -> &SlotIdentity {
        &self.identity
    }

    /// Check 3 of the three in [`super::identity`]: is this a slot of
    /// the model we are serving? Checks 1 and 2 already happened in
    /// [`decode`], which cannot produce an `UnverifiedSlot` whose header
    /// disagrees with its own floats.
    pub(crate) fn verify(
        self,
        serving: &SlotIdentity,
    ) -> Result<SlotPayload, super::identity::IdentityMismatch> {
        self.identity.compare(serving)?;
        Ok(self.payload)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum SlotFormatError {
    #[error("not a frink slot file: it is {len} bytes, shorter than the {PREFIX_LEN}-byte header")]
    TooShort { len: usize },
    #[error("not a frink slot file: bad magic")]
    BadMagic,
    #[error("slot file format version {found}, and this build reads {readable:?}")]
    UnsupportedVersion { found: u32, readable: Vec<u32> },
    #[error(
        "slot file declares {declared} bytes of header+body but only {available} follow the \
         prefix: the file is truncated or was written by a different format"
    )]
    LengthBeyondFile { declared: u128, available: usize },
    #[error("slot file digest does not match its contents: the file is corrupt or truncated")]
    DigestMismatch,
    #[error("slot file header is {len} bytes, too short for its own fixed fields")]
    HeaderTooShort { len: usize },
    #[error("slot file declares kv dtype tag {0}, which this build does not know")]
    UnknownDtype(u32),
    #[error("slot file declares a degenerate layer shape: {n_kv_heads} kv heads x {head_dim}")]
    DegenerateShape { n_kv_heads: usize, head_dim: usize },
    #[error("slot file declares no layers")]
    NoLayers,
    #[error(
        "slot file body declares {declared} bytes but its own counts add up to {computed}: the \
         header does not describe this payload"
    )]
    BodyLengthDisagrees { declared: u64, computed: String },
    #[error("slot file header strings do not fit the header: {0}")]
    HeaderStringsOverrun(&'static str),
    #[error("slot file {field} is not valid UTF-8")]
    NotUtf8 { field: &'static str },
}

/// Serializes a slot. The inverse of [`decode`], and the only writer.
pub(crate) fn encode(identity: &SlotIdentity, payload: &SlotPayload) -> Vec<u8> {
    let first = payload.layers.first().expect("a slot has at least a layer");
    let positions = first.positions();
    let n_kv_heads = first.n_kv_heads;
    let head_dim = first.head_dim;
    // One geometry in the header, so every layer must have it. The
    // route refuses a per-layer-shape model before reaching here
    // (`serving_identity`); this is the writer's own check, so the
    // format can never describe layer 0 and store layer 3.
    assert!(
        payload.layers.iter().all(|l| l.n_kv_heads == n_kv_heads
            && l.head_dim == head_dim
            && l.v_head_dim == head_dim),
        "slot file format holds one KV geometry; the layers differ, or V is not K's width"
    );
    let dtype = identity.dtype;
    assert!(
        payload.layers.iter().all(|l| l.dtype() == dtype),
        "slot file format holds one KV dtype, and it is the identity's"
    );

    let mut header = Vec::new();
    for value in [
        payload.layers.len() as u32,
        n_kv_heads as u32,
        head_dim as u32,
        dtype_tag(dtype),
        positions as u32,
        payload.tokens.len() as u32,
        payload.pending_logits.len() as u32,
        identity.model_name.len() as u32,
        identity.fingerprint.len() as u32,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    header.extend_from_slice(identity.model_name.as_bytes());
    header.extend_from_slice(identity.fingerprint.as_bytes());

    let mut body = Vec::new();
    for &token in &payload.tokens {
        body.extend_from_slice(&(token as u32).to_le_bytes());
    }
    for &logit in &payload.pending_logits {
        body.extend_from_slice(&logit.to_le_bytes());
    }
    for layer in &payload.layers {
        for value in layer
            .k
            .to_f32_vec()
            .iter()
            .chain(layer.v.to_f32_vec().iter())
        {
            body.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut digest = Sha256::new();
    digest.update(&header);
    digest.update(&body);

    let mut out = Vec::with_capacity(PREFIX_LEN + header.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SLOT_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&digest.finalize());
    out.extend_from_slice(&header);
    out.extend_from_slice(&body);
    out
}

/// Parses a slot file. Performs checks 1 and 2 of the three in
/// [`super::identity`]: the file carries an identity, and the header
/// describes the floats that follow it.
pub(crate) fn decode(bytes: &[u8]) -> Result<UnverifiedSlot, SlotFormatError> {
    if bytes.len() < PREFIX_LEN {
        return Err(SlotFormatError::TooShort { len: bytes.len() });
    }
    if &bytes[..8] != MAGIC {
        return Err(SlotFormatError::BadMagic);
    }
    let version = read_u32(bytes, 8);
    if !READABLE_FORMAT_VERSIONS.contains(&version) {
        return Err(SlotFormatError::UnsupportedVersion {
            found: version,
            readable: READABLE_FORMAT_VERSIONS.to_vec(),
        });
    }
    let header_len = read_u32(bytes, 12) as usize;
    let body_len = read_u64(bytes, 16);
    let stored_digest = &bytes[24..PREFIX_LEN];

    // The bound is the file, not a constant: header and body together
    // cannot exceed the bytes that actually follow the prefix. Done in
    // u128 so a `header_len + body_len` that would wrap on a 32-bit
    // usize is still compared honestly.
    let available = bytes.len() - PREFIX_LEN;
    let declared = header_len as u128 + body_len as u128;
    if declared != available as u128 {
        return Err(SlotFormatError::LengthBeyondFile {
            declared,
            available,
        });
    }

    let header = &bytes[PREFIX_LEN..PREFIX_LEN + header_len];
    let body = &bytes[PREFIX_LEN + header_len..];

    let mut digest = Sha256::new();
    digest.update(header);
    digest.update(body);
    if digest.finalize().as_slice() != stored_digest {
        return Err(SlotFormatError::DigestMismatch);
    }

    if header.len() < HEADER_FIELDS * 4 {
        return Err(SlotFormatError::HeaderTooShort { len: header.len() });
    }
    let field = |i: usize| read_u32(header, i * 4) as usize;
    let n_layers = field(0);
    let n_kv_heads = field(1);
    let head_dim = field(2);
    let dtype_tag = field(3) as u32;
    let positions = field(4);
    let n_prompt = field(5);
    let n_logits = field(6);
    let name_len = field(7);
    let fingerprint_len = field(8);

    let dtype = match dtype_tag {
        DTYPE_F32 => KvDtype::F32,
        DTYPE_F16 => KvDtype::F16,
        other => return Err(SlotFormatError::UnknownDtype(other)),
    };
    if n_layers == 0 {
        return Err(SlotFormatError::NoLayers);
    }
    if n_kv_heads == 0 || head_dim == 0 {
        return Err(SlotFormatError::DegenerateShape {
            n_kv_heads,
            head_dim,
        });
    }

    let strings = &header[HEADER_FIELDS * 4..];
    let split = name_len
        .checked_add(fingerprint_len)
        .ok_or(SlotFormatError::HeaderStringsOverrun("lengths overflow"))?;
    if split != strings.len() {
        return Err(SlotFormatError::HeaderStringsOverrun(
            "model name + fingerprint do not fill the header",
        ));
    }
    let model_name = std::str::from_utf8(&strings[..name_len])
        .map_err(|_| SlotFormatError::NotUtf8 {
            field: "model name",
        })?
        .to_string();
    let fingerprint = std::str::from_utf8(&strings[name_len..])
        .map_err(|_| SlotFormatError::NotUtf8 {
            field: "fingerprint",
        })?
        .to_string();

    // Check 2: the header must describe exactly the floats that follow.
    // Checked arithmetic throughout, and an EXACT equality rather than
    // "at least": a body longer than its counts is as much a
    // disagreement as one shorter.
    let per_position = n_kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| body_disagrees(body_len, "layer shape overflows"))?;
    let per_layer = per_position
        .checked_mul(positions)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| body_disagrees(body_len, "layer size overflows"))?;
    let computed = n_layers
        .checked_mul(per_layer)
        .and_then(|n| n.checked_add(n_prompt))
        .and_then(|n| n.checked_add(n_logits))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| body_disagrees(body_len, "body size overflows"))?;
    if computed as u64 != body_len {
        return Err(body_disagrees(body_len, "counts do not match the body"));
    }

    let mut cursor = 0usize;
    let mut tokens = Vec::with_capacity(n_prompt);
    for _ in 0..n_prompt {
        tokens.push(read_u32(body, cursor) as usize);
        cursor += 4;
    }
    let mut pending_logits = Vec::with_capacity(n_logits);
    for _ in 0..n_logits {
        pending_logits.push(read_f32(body, cursor));
        cursor += 4;
    }
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let mut cache = KvCache::new(n_kv_heads, head_dim).with_dtype(dtype);
        let k_start = cursor;
        let v_start = cursor + per_position * positions * 4;
        // Rebuilt one position at a time through `push`, the only
        // public constructor that keeps a cache's position counter and
        // its buffers in agreement. Writing `k`/`v` directly would leave
        // `positions()` at zero, which reads as an empty cache
        // everywhere downstream.
        let mut k_step = vec![0.0f32; per_position];
        let mut v_step = vec![0.0f32; per_position];
        for position in 0..positions {
            for element in 0..per_position {
                k_step[element] = read_f32(body, k_start + (position * per_position + element) * 4);
                v_step[element] = read_f32(body, v_start + (position * per_position + element) * 4);
            }
            cache
                .push(&k_step, &v_step)
                .expect("an unpooled cache grows without a pool");
        }
        cursor = v_start + per_position * positions * 4;
        layers.push(cache);
    }

    Ok(UnverifiedSlot {
        identity: SlotIdentity {
            model_name,
            n_layers,
            n_kv_heads,
            head_dim,
            dtype,
            fingerprint,
        },
        payload: SlotPayload {
            tokens,
            pending_logits,
            layers,
        },
    })
}

fn body_disagrees(body_len: u64, why: &str) -> SlotFormatError {
    SlotFormatError::BodyLengthDisagrees {
        declared: body_len,
        computed: why.to_string(),
    }
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("bounds checked"))
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("bounds checked"))
}

fn read_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().expect("bounds checked"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> SlotIdentity {
        SlotIdentity {
            model_name: "llama".to_string(),
            n_layers: 2,
            n_kv_heads: 2,
            head_dim: 3,
            dtype: KvDtype::F32,
            fingerprint: "deadbeef".to_string(),
        }
    }

    fn payload() -> SlotPayload {
        let mut layers = Vec::new();
        for layer in 0..2 {
            let mut cache = KvCache::new(2, 3);
            for position in 0..5 {
                let base = (layer * 100 + position * 10) as f32;
                let k: Vec<f32> = (0..6).map(|i| base + i as f32).collect();
                let v: Vec<f32> = (0..6).map(|i| base - i as f32).collect();
                cache.push(&k, &v).unwrap();
            }
            layers.push(cache);
        }
        SlotPayload {
            tokens: vec![1, 2, 3, 4, 5],
            pending_logits: vec![0.5, -1.5, 2.25],
            layers,
        }
    }

    /// The header holds ONE KV geometry, so a layer set whose geometries
    /// differ (a per-layer-shape model, `frink_models::layer_shapes`)
    /// must never be encoded as if layer 0 spoke for all of them. The
    /// route refuses such a model first; this is the writer's own guard.
    #[test]
    #[should_panic(expected = "one KV geometry")]
    fn a_layer_set_with_differing_geometries_is_refused_by_the_writer() {
        let mut p = payload();
        p.layers[1] = KvCache::new(1, 3);
        let _ = encode(&identity(), &p);
    }

    #[test]
    fn a_slot_round_trips_through_the_file_bit_for_bit() {
        let bytes = encode(&identity(), &payload());
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.identity(), &identity());
        let restored = decoded.verify(&identity()).unwrap();
        let original = payload();
        assert_eq!(restored.tokens, original.tokens);
        assert_eq!(restored.pending_logits, original.pending_logits);
        assert_eq!(restored.layers.len(), original.layers.len());
        for (got, want) in restored.layers.iter().zip(original.layers.iter()) {
            assert_eq!(got.k, want.k);
            assert_eq!(got.v, want.v);
            // The counter, not just the buffer: a cache rebuilt by
            // writing `k`/`v` directly would pass the two assertions
            // above and still report zero positions.
            assert_eq!(got.positions(), want.positions());
        }
    }

    /// The restore this file format exists to refuse.
    #[test]
    fn a_slot_saved_under_another_checkpoint_is_refused_by_name() {
        let bytes = encode(&identity(), &payload());
        let mut serving = identity();
        serving.fingerprint = "0badf00d".to_string();
        let Err(err) = decode(&bytes).unwrap().verify(&serving) else {
            panic!("a slot of another checkpoint must not verify");
        };
        assert_eq!(err.field, "checkpoint");
    }

    /// A truncated file is refused by arithmetic on the length it
    /// declares, before anything is hashed or allocated.
    #[test]
    fn a_truncated_file_is_refused_before_it_is_parsed() {
        let bytes = encode(&identity(), &payload());
        let cut = &bytes[..bytes.len() - 8];
        assert!(matches!(
            decode(cut),
            Err(SlotFormatError::LengthBeyondFile { .. })
        ));
    }

    /// A body edited in place keeps its declared lengths and fails on
    /// the digest, which is the only check that can see it.
    #[test]
    fn a_flipped_float_is_caught_by_the_digest() {
        let mut bytes = encode(&identity(), &payload());
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        assert!(matches!(
            decode(&bytes),
            Err(SlotFormatError::DigestMismatch)
        ));
    }

    /// A header claiming a shape the floats do not have is refused even
    /// with a digest that vouches for it -- because the digest is
    /// recomputed over the edited header too. This is check 2: a stamp
    /// may not vouch for a width the payload does not have.
    #[test]
    fn a_header_that_contradicts_its_own_body_is_refused() {
        let mut bytes = encode(&identity(), &payload());
        // n_layers is the first header field.
        let header_at = PREFIX_LEN;
        bytes[header_at..header_at + 4].copy_from_slice(&7u32.to_le_bytes());
        // Re-stamp the digest so the ONLY thing wrong is the claim.
        let header_len = read_u32(&bytes, 12) as usize;
        let mut digest = Sha256::new();
        digest.update(&bytes[PREFIX_LEN..PREFIX_LEN + header_len]);
        digest.update(&bytes[PREFIX_LEN + header_len..]);
        let stamped = digest.finalize();
        bytes[24..PREFIX_LEN].copy_from_slice(&stamped);
        assert!(matches!(
            decode(&bytes),
            Err(SlotFormatError::BodyLengthDisagrees { .. })
        ));
    }

    /// An unreadable version is refused rather than guessed at. A v2
    /// file might mean anything by these bytes.
    #[test]
    fn an_unknown_format_version_is_refused_rather_than_guessed_at() {
        let mut bytes = encode(&identity(), &payload());
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            decode(&bytes),
            Err(SlotFormatError::UnsupportedVersion { found: 99, .. })
        ));
    }

    /// A hostile header cannot make the parser reserve memory: the
    /// declared shape has to add up to a body the file actually
    /// contains, and `u64::MAX / 64` positions fails that arithmetic
    /// rather than a `Vec::with_capacity`.
    #[test]
    fn an_implausible_position_count_is_refused_by_arithmetic_not_by_allocating() {
        let mut bytes = encode(&identity(), &payload());
        let positions_at = PREFIX_LEN + 4 * 4;
        bytes[positions_at..positions_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let header_len = read_u32(&bytes, 12) as usize;
        let mut digest = Sha256::new();
        digest.update(&bytes[PREFIX_LEN..PREFIX_LEN + header_len]);
        digest.update(&bytes[PREFIX_LEN + header_len..]);
        let stamped = digest.finalize();
        bytes[24..PREFIX_LEN].copy_from_slice(&stamped);
        assert!(matches!(
            decode(&bytes),
            Err(SlotFormatError::BodyLengthDisagrees { .. })
        ));
    }
}
