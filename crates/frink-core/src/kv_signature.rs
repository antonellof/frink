//! Compatibility marking for stored KV blocks: whether a block found
//! in a cache may be *used*, as opposed to merely found.
//!
//! A [`BlockHash`](crate::kv_block::BlockHash) says which tokens a block
//! covers. It says nothing about the shape of the tensors stored under
//! it -- how many layers, how many KV heads, what head dimension, what
//! dtype, how many token positions are really there. A cache that
//! survives a restart will eventually be read by a process configured
//! differently from the one that wrote it, and reading a block whose
//! layout does not match is not a miss: it is silently wrong attention
//! state, which produces confident wrong tokens.
//!
//! The one rule this module exists to enforce:
//!
//! > **A signature is derived from the block's own payload, never from
//! > the manager's expectation, and an unmarked block is rejected
//! > rather than trusted.**
//!
//! So [`CacheSignature::from_payload`] measures the tensors; it takes no
//! "expected" shape to fill gaps with. [`UnverifiedBlock::verify`] then
//! makes three separate checks, in order:
//!
//! 1. the block carries a signature at all ([`SignatureError::Unmarked`]
//!    otherwise -- absence is never treated as agreement);
//! 2. the recorded signature matches what the payload actually is
//!    ([`SignatureError::PayloadMismatch`]) -- a stamp may not vouch for
//!    a width the payload does not have;
//! 3. only then, that this shape is the shape the reader wants
//!    ([`SignatureError::Incompatible`]).
//!
//! Step 2 is the one that is easy to skip and expensive to skip: without
//! it, "the stamp says 32 heads" and "there are 32 heads" are different
//! claims that a cache would be treating as one.
//!
//! # The block layout is part of the signature
//!
//! A signature also carries the [`BlockLayout`] the block was cut
//! under: its block size, and the sliding window that block size had to
//! divide (see [`kv_swa`](crate::kv_swa) for why it must). Both are
//! here rather than checked once at startup because a *durable* cache
//! outlives the configuration that filled it. A block written by a
//! build running gpt-oss at window 128 must not be handed to a build
//! running it at window 256, and the failure mode if it were is not a
//! crash -- it is an attention mask silently wider or narrower than the
//! model's.
//!
//! `block_size` is payload-checkable and is checked: a stored block is
//! exactly one whole block (`kv_block::chain` refuses to name a partial
//! tail), so a stamp claiming `block_size = 64` over a 48-token payload
//! is refused the same way a stamp claiming 32 heads over 16 is. The
//! window is not provable from tensors -- like `model` -- so it is
//! carried and compared against the reader's expectation.

use crate::cache::KvCache;
use crate::kv_swa::{BlockLayout, BlockLayoutError};

/// Layout version of a stored block payload. A reader accepts only the
/// versions in [`READABLE_FORMAT_VERSIONS`]; anything else is rejected
/// rather than guessed at.
pub const BLOCK_FORMAT_VERSION: u32 = 2;

/// Versions this build can read. Kept explicit (rather than `<=
/// BLOCK_FORMAT_VERSION`) so dropping support for an old layout is a
/// deliberate edit and not an accident of arithmetic.
///
/// Version 1 is deliberately **not** readable. Its header had no block
/// size and no sliding window, so a v1 block cannot say what layout it
/// was cut under -- and this module's rule is that absence is never
/// agreement. Reading one would mean assuming it happened to be
/// aligned, which is the exact assumption `kv_swa` exists to refuse.
pub const READABLE_FORMAT_VERSIONS: &[u32] = &[2];

/// Element type of the stored K/V tensors: the host store's own dtype
/// (`crate::kv_store`), one type, so a block file cannot name a width the
/// cache does not have. A block written at one dtype is a MISS for a
/// reader at the other, never a reinterpretation of its bytes.
pub use crate::kv_store::KvDtype;

/// What a block's payload *is*: the shape any reader must match to use
/// it. Construct it from a payload with [`Self::from_payload`], or as a
/// reader's requirement with [`Self::expected`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSignature {
    pub format_version: u32,
    /// Identifies the weights the KV state was computed under. The one
    /// field no payload can prove about itself -- which is exactly why
    /// it is checked against the reader's expectation in step 3.
    pub model: String,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: KvDtype,
    /// Token positions actually stored, per layer. Equal to
    /// `layout.block_size()` for any block this module will stamp or
    /// verify.
    pub tokens: usize,
    /// How the sequence was cut into blocks, and the sliding window
    /// that cut had to line up with. See the module note.
    pub layout: BlockLayout,
}

impl CacheSignature {
    /// Derives a signature by *measuring* `layers`. There is
    /// deliberately no parameter to fill a gap from: every shape field
    /// except `model` and the sliding window comes from the tensors
    /// themselves, and even the declared `layout`'s block size is
    /// checked against the depth the tensors actually have.
    ///
    /// Fails if the payload cannot describe itself coherently: no
    /// layers at all, layers that disagree with each other, a layer
    /// whose buffers do not match its own declared shape, or a token
    /// depth that is not the block size the layout claims.
    pub fn from_payload(
        model: &str,
        layout: BlockLayout,
        layers: &[KvCache],
    ) -> Result<Self, SignatureError> {
        let first = layers.first().ok_or(SignatureError::EmptyPayload)?;
        let n_kv_heads = first.n_kv_heads;
        let head_dim = first.head_dim;
        if n_kv_heads == 0 || head_dim == 0 {
            return Err(SignatureError::DegenerateLayer {
                layer: 0,
                n_kv_heads,
                head_dim,
            });
        }
        let per_token = n_kv_heads * head_dim;
        let dtype = first.dtype();
        let tokens = measure_layer(0, first, per_token)?;

        for (index, layer) in layers.iter().enumerate().skip(1) {
            if layer.n_kv_heads != n_kv_heads || layer.head_dim != head_dim {
                return Err(SignatureError::RaggedPayload {
                    layer: index,
                    field: "layer shape",
                    expected: format!("{n_kv_heads}x{head_dim}"),
                    found: format!("{}x{}", layer.n_kv_heads, layer.head_dim),
                });
            }
            if layer.dtype() != dtype {
                return Err(SignatureError::RaggedPayload {
                    layer: index,
                    field: "dtype",
                    expected: dtype.name().to_string(),
                    found: layer.dtype().name().to_string(),
                });
            }
            let layer_tokens = measure_layer(index, layer, per_token)?;
            if layer_tokens != tokens {
                return Err(SignatureError::RaggedPayload {
                    layer: index,
                    field: "token count",
                    expected: tokens.to_string(),
                    found: layer_tokens.to_string(),
                });
            }
        }

        // The block size is a claim like any other, and this one the
        // payload can settle: a stored block is one whole block.
        if tokens != layout.block_size() {
            return Err(SignatureError::BlockSizeMismatch {
                block_size: layout.block_size(),
                tokens,
            });
        }

        Ok(CacheSignature {
            format_version: BLOCK_FORMAT_VERSION,
            model: model.to_string(),
            n_layers: layers.len(),
            n_kv_heads,
            head_dim,
            dtype,
            tokens,
            layout,
        })
    }

    /// A reader's requirement: the shape this process would compute
    /// itself. Never stamped onto a block -- only compared against one.
    pub fn expected(
        model: &str,
        layout: BlockLayout,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: KvDtype,
        tokens: usize,
    ) -> Self {
        CacheSignature {
            format_version: BLOCK_FORMAT_VERSION,
            model: model.to_string(),
            n_layers,
            n_kv_heads,
            head_dim,
            dtype,
            tokens,
            layout,
        }
    }

    /// Field-by-field comparison, naming the first field that differs
    /// so an operator learns *what* changed rather than "cache miss".
    fn compare(
        &self,
        other: &CacheSignature,
        mismatch: fn(&'static str, String, String) -> SignatureError,
    ) -> Result<(), SignatureError> {
        if self.format_version != other.format_version {
            return Err(mismatch(
                "format_version",
                self.format_version.to_string(),
                other.format_version.to_string(),
            ));
        }
        if self.model != other.model {
            return Err(mismatch("model", self.model.clone(), other.model.clone()));
        }
        if self.n_layers != other.n_layers {
            return Err(mismatch(
                "n_layers",
                self.n_layers.to_string(),
                other.n_layers.to_string(),
            ));
        }
        if self.n_kv_heads != other.n_kv_heads {
            return Err(mismatch(
                "n_kv_heads",
                self.n_kv_heads.to_string(),
                other.n_kv_heads.to_string(),
            ));
        }
        if self.head_dim != other.head_dim {
            return Err(mismatch(
                "head_dim",
                self.head_dim.to_string(),
                other.head_dim.to_string(),
            ));
        }
        if self.dtype != other.dtype {
            return Err(mismatch(
                "dtype",
                self.dtype.name().to_string(),
                other.dtype.name().to_string(),
            ));
        }
        if self.tokens != other.tokens {
            return Err(mismatch(
                "tokens",
                self.tokens.to_string(),
                other.tokens.to_string(),
            ));
        }
        if self.layout.block_size() != other.layout.block_size() {
            return Err(mismatch(
                "block_size",
                self.layout.block_size().to_string(),
                other.layout.block_size().to_string(),
            ));
        }
        if self.layout.sliding_window() != other.layout.sliding_window() {
            return Err(mismatch(
                "sliding_window",
                describe_window(self.layout.sliding_window()),
                describe_window(other.layout.sliding_window()),
            ));
        }
        Ok(())
    }
}

/// Renders a window for an error message. `None` is spelled out rather
/// than printed as an empty string: "this build uses no sliding window"
/// and "this build did not say" must not look the same in a log.
fn describe_window(window: Option<usize>) -> String {
    match window {
        Some(w) => w.to_string(),
        None => "none (full causal)".to_string(),
    }
}

/// Measures one layer, rejecting a layer whose buffers disagree with
/// its own declared `seq_len` -- `seq_len` is a claim, `k.len()` is the
/// evidence.
fn measure_layer(index: usize, layer: &KvCache, per_token: usize) -> Result<usize, SignatureError> {
    if layer.v_head_dim != layer.head_dim {
        return Err(SignatureError::SplitKvHeadWidth {
            layer: index,
            head_dim: layer.head_dim,
            v_head_dim: layer.v_head_dim,
        });
    }
    if !layer.k.len().is_multiple_of(per_token) {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "k length",
            expected: format!("a multiple of {per_token}"),
            found: layer.k.len().to_string(),
        });
    }
    if layer.v.len() != layer.k.len() {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "v length",
            expected: layer.k.len().to_string(),
            found: layer.v.len().to_string(),
        });
    }
    let tokens = layer.k.len() / per_token;
    // Positions against rows. They are equal for anything this codec
    // can currently produce, and a payload where they disagree is
    // ragged rather than windowed. When a store learns to evict (#61)
    // a restored windowed layer will legitimately carry more positions
    // than rows, and THIS CHECK IS WHERE THAT HAS TO BE TAUGHT: the
    // codec will need to serialize the position count separately
    // instead of deriving it from the byte length.
    if layer.positions() != tokens {
        return Err(SignatureError::RaggedPayload {
            layer: index,
            field: "positions",
            expected: tokens.to_string(),
            found: layer.positions().to_string(),
        });
    }
    Ok(tokens)
}

/// Why a stored block was refused. Every variant is a refusal to guess.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignatureError {
    /// The block carries no signature. Not treated as "probably fine":
    /// an unmarked block was written by something whose layout is
    /// unknown, which is precisely the case that must not be trusted.
    Unmarked,
    /// A block with no layers describes nothing and can vouch for
    /// nothing.
    EmptyPayload,
    /// A layer with no heads or zero head dimension.
    DegenerateLayer {
        layer: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    /// A layer whose V head width differs from its K head width
    /// (MiMo-V2). The block format carries ONE `head_dim`, so such a
    /// payload has no honest encoding; refused rather than written with
    /// the K width and read back into V rows of the wrong shape.
    SplitKvHeadWidth {
        layer: usize,
        head_dim: usize,
        v_head_dim: usize,
    },
    /// The payload does not agree with itself: layers of different
    /// shapes or lengths, or a layer whose buffers contradict its own
    /// `seq_len`.
    RaggedPayload {
        layer: usize,
        field: &'static str,
        expected: String,
        found: String,
    },
    /// The recorded signature claims something the payload is not. The
    /// stamp is wrong (or was written by a build with a different
    /// layout); the payload is the truth.
    PayloadMismatch {
        field: &'static str,
        recorded: String,
        actual: String,
    },
    /// The payload is coherent and honestly stamped, but it is not what
    /// this reader needs -- a different model, a config change, a
    /// different block size.
    Incompatible {
        field: &'static str,
        expected: String,
        found: String,
    },
    /// The stamp claims a block size the payload's token depth is not.
    /// A stored block is exactly one whole block, so these are the same
    /// number or the stamp is wrong.
    BlockSizeMismatch { block_size: usize, tokens: usize },
    /// The block layout itself is not usable -- most importantly, a
    /// block size that does not divide the sliding window. See
    /// [`kv_swa`](crate::kv_swa).
    BadLayout(BlockLayoutError),
    /// Written by a build whose payload layout this one cannot read.
    UnsupportedFormat {
        found: u32,
        readable: &'static [u32],
    },
}

impl From<BlockLayoutError> for SignatureError {
    fn from(err: BlockLayoutError) -> Self {
        SignatureError::BadLayout(err)
    }
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignatureError::Unmarked => write!(
                f,
                "KV block carries no cache signature; refusing to trust an unmarked block"
            ),
            SignatureError::EmptyPayload => {
                write!(f, "KV block has no layers; nothing to verify")
            }
            SignatureError::DegenerateLayer {
                layer,
                n_kv_heads,
                head_dim,
            } => write!(
                f,
                "KV block layer {layer} is degenerate: {n_kv_heads} kv heads x {head_dim} head dim"
            ),
            SignatureError::SplitKvHeadWidth {
                layer,
                head_dim,
                v_head_dim,
            } => write!(
                f,
                "KV block layer {layer} has a V head width ({v_head_dim}) that differs from its K \
                 head width ({head_dim}); the block format carries one head_dim and cannot \
                 encode it"
            ),
            SignatureError::RaggedPayload {
                layer,
                field,
                expected,
                found,
            } => write!(
                f,
                "KV block payload is inconsistent at layer {layer}: {field} is {found}, expected {expected}"
            ),
            SignatureError::PayloadMismatch {
                field,
                recorded,
                actual,
            } => write!(
                f,
                "KV block signature vouches for {field}={recorded} but its payload has {field}={actual}"
            ),
            SignatureError::Incompatible {
                field,
                expected,
                found,
            } => write!(
                f,
                "KV block is incompatible: {field} is {found}, this server needs {expected}"
            ),
            SignatureError::BlockSizeMismatch { block_size, tokens } => write!(
                f,
                "KV block signature declares a block size of {block_size} but its payload holds \
                 {tokens} token positions; a stored block is exactly one whole block"
            ),
            SignatureError::BadLayout(err) => write!(f, "KV block layout is unusable: {err}"),
            SignatureError::UnsupportedFormat { found, readable } => write!(
                f,
                "KV block format version {found} is not readable by this build (readable: {readable:?})"
            ),
        }
    }
}

impl std::error::Error for SignatureError {}

/// A block whose signature has been verified against its own payload
/// and against the reader's expectation. Only way to get one is
/// [`KvBlock::stamp`] (writing) or [`UnverifiedBlock::verify`]
/// (reading), so holding one is itself the proof.
pub struct KvBlock {
    signature: CacheSignature,
    layers: Vec<KvCache>,
}

/// Summarizes rather than dumping tensors: a block's `Debug` is for a
/// log line or a failing assertion, and printing every f32 in a KV
/// block helps nobody.
impl std::fmt::Debug for KvBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvBlock")
            .field("signature", &self.signature)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl std::fmt::Debug for UnverifiedBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnverifiedBlock")
            .field("signature", &self.signature)
            .field("layers", &self.layers.len())
            .finish()
    }
}

impl KvBlock {
    /// Stamps a block from its own payload. The caller supplies the
    /// model identity and the layout the block was cut under; every
    /// shape field is measured, and the layout's block size is checked
    /// against the depth measured.
    pub fn stamp(
        model: &str,
        layout: BlockLayout,
        layers: Vec<KvCache>,
    ) -> Result<Self, SignatureError> {
        let signature = CacheSignature::from_payload(model, layout, &layers)?;
        Ok(KvBlock { signature, layers })
    }

    /// The layout this block was cut under.
    pub fn layout(&self) -> BlockLayout {
        self.signature.layout
    }

    pub fn signature(&self) -> &CacheSignature {
        &self.signature
    }

    pub fn tokens(&self) -> usize {
        self.signature.tokens
    }

    pub fn layers(&self) -> &[KvCache] {
        &self.layers
    }

    pub fn into_layers(self) -> Vec<KvCache> {
        self.layers
    }
}

/// A block as it comes back from an untrusted source: a disk tier, a
/// shared cache, or a build that is not this one. `signature` is an
/// `Option` on purpose -- "no signature recorded" is a state a real
/// stored block can be in, and it must be representable so it can be
/// refused.
pub struct UnverifiedBlock {
    pub signature: Option<CacheSignature>,
    pub layers: Vec<KvCache>,
}

impl UnverifiedBlock {
    pub fn new(signature: Option<CacheSignature>, layers: Vec<KvCache>) -> Self {
        UnverifiedBlock { signature, layers }
    }

    /// Verifies in the order the module doc describes: marked, honest,
    /// then compatible. `expected` is used only in the last step -- it
    /// never contributes a field to the signature being checked.
    pub fn verify(self, expected: &CacheSignature) -> Result<KvBlock, SignatureError> {
        let recorded = self.signature.ok_or(SignatureError::Unmarked)?;
        if !READABLE_FORMAT_VERSIONS.contains(&recorded.format_version) {
            return Err(SignatureError::UnsupportedFormat {
                found: recorded.format_version,
                readable: READABLE_FORMAT_VERSIONS,
            });
        }
        // Measured from the payload. `recorded.model` is carried over
        // because no payload can prove which weights produced it; the
        // model check happens against `expected` below, where a wrong
        // model is caught as an incompatibility.
        // `recorded.layout` is carried over for the same reason
        // `recorded.model` is: the reader's expectation must not get a
        // vote before the payload has been checked against the stamp.
        // Carrying it is not trusting it -- `from_payload` rejects a
        // block size the token depth contradicts, and the window is
        // settled against `expected` below.
        let actual = CacheSignature::from_payload(&recorded.model, recorded.layout, &self.layers)?;
        recorded.compare(&actual, |field, recorded, actual| {
            SignatureError::PayloadMismatch {
                field,
                recorded,
                actual,
            }
        })?;
        expected.compare(&actual, |field, expected, found| {
            SignatureError::Incompatible {
                field,
                expected,
                found,
            }
        })?;
        Ok(KvBlock {
            signature: actual,
            layers: self.layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(n_kv_heads: usize, head_dim: usize, tokens: usize) -> KvCache {
        let mut cache = KvCache::new(n_kv_heads, head_dim);
        let step = vec![0.5f32; n_kv_heads * head_dim];
        for _ in 0..tokens {
            cache.push(&step, &step).expect("unpooled push cannot fail");
        }
        cache
    }

    fn payload(n_layers: usize, n_kv_heads: usize, head_dim: usize, tokens: usize) -> Vec<KvCache> {
        (0..n_layers)
            .map(|_| layer(n_kv_heads, head_dim, tokens))
            .collect()
    }

    /// A full-causal layout whose block size is the payload depth --
    /// what every test that is not about SWA wants.
    fn flat(block_size: usize) -> BlockLayout {
        BlockLayout::full_attention(block_size).expect("positive block size")
    }

    /// The core rule, in its positive form: every shape field comes
    /// from the tensors. `stamp` is given a model name and nothing else.
    #[test]
    fn signature_is_measured_from_the_payload() {
        let block = KvBlock::stamp("model-a", flat(4), payload(3, 2, 8, 4)).expect("stamp");
        let sig = block.signature();
        assert_eq!(sig.n_layers, 3);
        assert_eq!(sig.n_kv_heads, 2);
        assert_eq!(sig.head_dim, 8);
        assert_eq!(sig.tokens, 4);
        assert_eq!(sig.dtype, KvDtype::F32);
        assert_eq!(sig.format_version, BLOCK_FORMAT_VERSION);
        assert_eq!(block.tokens(), 4);
        assert_eq!(block.layers().len(), 3);
    }

    #[test]
    fn a_stamped_block_round_trips_through_verification() {
        let layers = payload(3, 2, 8, 4);
        let signature =
            CacheSignature::from_payload("model-a", flat(4), &layers).expect("signature");
        let expected = CacheSignature::expected("model-a", flat(4), 3, 2, 8, KvDtype::F32, 4);
        let block = UnverifiedBlock::new(Some(signature), layers)
            .verify(&expected)
            .expect("a block that is what it says it is must verify");
        assert_eq!(block.layers().len(), 3);
        assert_eq!(block.into_layers().len(), 3);
    }

    /// Absence of a signature is not agreement. This is the difference
    /// between a persistent cache and silent corruption after a config
    /// change: an unmarked block came from something whose layout is
    /// unknown by definition.
    #[test]
    fn an_unmarked_block_is_rejected_not_trusted() {
        let expected = CacheSignature::expected("model-a", flat(4), 3, 2, 8, KvDtype::F32, 4);
        let err = UnverifiedBlock::new(None, payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("an unmarked block must be refused");
        assert_eq!(err, SignatureError::Unmarked);
    }

    /// The rule the plan states as "a signature must never vouch for a
    /// width the payload does not have". Here the stamp is exactly what
    /// the reader expects -- and the payload is not. A cache that
    /// trusted the stamp (or, equivalently, stamped from the manager's
    /// expectation) would hand back tensors of the wrong width and
    /// produce confident wrong tokens.
    #[test]
    fn a_signature_that_overstates_its_payload_is_rejected() {
        let expected = CacheSignature::expected("model-a", flat(4), 3, 2, 16, KvDtype::F32, 4);
        let mut lying = expected.clone();
        assert_eq!(lying.head_dim, 16);
        let err = UnverifiedBlock::new(Some(lying.clone()), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims head_dim 16 over an 8-wide payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "head_dim",
                recorded: "16".into(),
                actual: "8".into(),
            }
        );

        // Same shape, overstated depth: 8 token positions claimed over
        // a 4-position payload.
        lying.head_dim = 8;
        lying.tokens = 8;
        let expected = CacheSignature::expected("model-a", flat(8), 3, 2, 8, KvDtype::F32, 8);
        let err = UnverifiedBlock::new(Some(lying.clone()), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims 8 tokens over a 4-token payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "tokens",
                recorded: "8".into(),
                actual: "4".into(),
            }
        );

        // And overstated layer count.
        lying.tokens = 4;
        lying.n_layers = 4;
        let expected = CacheSignature::expected("model-a", flat(4), 4, 2, 8, KvDtype::F32, 4);
        let err = UnverifiedBlock::new(Some(lying), payload(3, 2, 8, 4))
            .verify(&expected)
            .expect_err("stamp claims 4 layers over a 3-layer payload");
        assert_eq!(
            err,
            SignatureError::PayloadMismatch {
                field: "n_layers",
                recorded: "4".into(),
                actual: "3".into(),
            }
        );
    }

    /// A payload that lies to itself: `seq_len` says 4, the buffers
    /// hold 3. `seq_len` is a claim; `k.len()` is the evidence.
    #[test]
    fn a_layer_whose_seq_len_contradicts_its_buffers_is_rejected() {
        let mut layers = payload(2, 2, 8, 4);
        // The contradiction is the thing under test.
        layers[1].force_positions_for_test(7);
        let err = CacheSignature::from_payload("model-a", flat(4), &layers)
            .expect_err("seq_len must be verified, not believed");
        assert_eq!(
            err,
            SignatureError::RaggedPayload {
                layer: 1,
                field: "positions",
                expected: "4".into(),
                found: "7".into(),
            }
        );
    }

    #[test]
    fn a_ragged_payload_is_rejected() {
        let mut layers = payload(3, 2, 8, 4);
        layers[2] = layer(2, 4, 4);
        let err = CacheSignature::from_payload("model-a", flat(4), &layers)
            .expect_err("shape disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 2,
                field: "layer shape",
                ..
            }
        ));

        let mut layers = payload(3, 2, 8, 4);
        layers[1] = layer(2, 8, 3);
        let err = CacheSignature::from_payload("model-a", flat(4), &layers)
            .expect_err("depth disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 1,
                field: "token count",
                ..
            }
        ));

        let mut layers = payload(2, 2, 8, 4);
        layers[0].v.truncate(8);
        let err = CacheSignature::from_payload("model-a", flat(4), &layers)
            .expect_err("k/v disagreement");
        assert!(matches!(
            err,
            SignatureError::RaggedPayload {
                layer: 0,
                field: "v length",
                ..
            }
        ));
    }

    #[test]
    fn an_empty_payload_is_rejected() {
        assert_eq!(
            CacheSignature::from_payload("model-a", flat(4), &[])
                .expect_err("nothing to vouch for"),
            SignatureError::EmptyPayload
        );
    }

    /// An honest block of the wrong shape is a *miss*, reported as an
    /// incompatibility naming the field that changed -- not a
    /// corruption, and not a silent fallback.
    #[test]
    fn an_honest_block_from_a_different_config_is_incompatible() {
        let layers = payload(3, 2, 8, 4);
        let signature =
            CacheSignature::from_payload("model-a", flat(4), &layers).expect("signature");
        let err = UnverifiedBlock::new(Some(signature.clone()), layers)
            .verify(&CacheSignature::expected(
                "model-b",
                flat(4),
                3,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("a different model must not share KV state");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "model",
                expected: "model-b".into(),
                found: "model-a".into(),
            }
        );

        let layers = payload(3, 2, 8, 4);
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                flat(4),
                3,
                4,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("a different KV head count must not be reused");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "n_kv_heads",
                expected: "4".into(),
                found: "2".into(),
            }
        );
    }

    #[test]
    fn an_unreadable_format_version_is_rejected() {
        let layers = payload(2, 2, 8, 4);
        let mut signature =
            CacheSignature::from_payload("model-a", flat(4), &layers).expect("signature");
        signature.format_version = 99;
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                flat(4),
                2,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("an unknown layout must not be guessed at");
        assert_eq!(
            err,
            SignatureError::UnsupportedFormat {
                found: 99,
                readable: READABLE_FORMAT_VERSIONS,
            }
        );
    }

    /// The `kv-swa-block-alignment` invariant at the signature layer.
    ///
    /// Two builds of the same model, same tensors, same block size --
    /// one configured with a 128-token sliding window and one with 256.
    /// The payload cannot tell them apart, which is precisely why the
    /// window is stamped: without this check the second build reads the
    /// first build's blocks back and runs a mask the model never had.
    #[test]
    fn a_block_written_under_a_different_window_is_refused_not_reused() {
        let layout_128 = BlockLayout::new(4, Some(128)).expect("4 divides 128");
        let layout_256 = BlockLayout::new(4, Some(256)).expect("4 divides 256");
        let layers = payload(3, 2, 8, 4);
        let signature =
            CacheSignature::from_payload("model-a", layout_128, &layers).expect("signature");

        let err = UnverifiedBlock::new(Some(signature.clone()), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                layout_256,
                3,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("a window change must invalidate the block, not be ignored");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "sliding_window",
                expected: "256".into(),
                found: "128".into(),
            }
        );

        // And the same block under the same window still verifies --
        // the check must invalidate on change, not on principle.
        let layers = payload(3, 2, 8, 4);
        UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                layout_128,
                3,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect("unchanged config must still hit");
    }

    /// Turning SWA off (or on) is a config change of exactly the same
    /// kind, and `None` must not read as "matches anything".
    #[test]
    fn a_full_causal_reader_will_not_take_a_sliding_window_block() {
        let sliding = BlockLayout::new(4, Some(128)).expect("aligned");
        let layers = payload(2, 2, 8, 4);
        let signature =
            CacheSignature::from_payload("model-a", sliding, &layers).expect("signature");
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                flat(4),
                2,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("no window and a 128 window are different configurations");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "sliding_window",
                expected: "none (full causal)".into(),
                found: "128".into(),
            }
        );
    }

    /// A block cut at a different block size cannot be spliced into a
    /// sequence cut at this one: the hash chain would not line up and
    /// the eviction unit would not either.
    #[test]
    fn a_block_cut_at_a_different_block_size_is_incompatible() {
        let layers = payload(2, 2, 8, 4);
        let signature =
            CacheSignature::from_payload("model-a", flat(4), &layers).expect("signature");
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                flat(2),
                2,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("a 4-token block is not a 2-token block");
        assert_eq!(
            err,
            SignatureError::Incompatible {
                field: "block_size",
                expected: "2".into(),
                found: "4".into(),
            }
        );
    }

    /// `block_size` is the one new field the payload can settle, so it
    /// is settled: a stamp claiming 8-token blocks over a 4-token
    /// payload is a lying stamp, exactly like an overstated head_dim.
    #[test]
    fn a_stamp_may_not_claim_a_block_size_the_payload_lacks() {
        let err = KvBlock::stamp("model-a", flat(8), payload(2, 2, 8, 4))
            .expect_err("8-token blocks over a 4-token payload");
        assert_eq!(
            err,
            SignatureError::BlockSizeMismatch {
                block_size: 8,
                tokens: 4,
            }
        );

        // Same on the read path, where the stamp comes from a file
        // rather than from this process.
        let honest =
            CacheSignature::from_payload("model-a", flat(4), &payload(2, 2, 8, 4)).expect("sig");
        let mut lying = honest.clone();
        lying.layout = flat(8);
        lying.tokens = 8;
        let err = UnverifiedBlock::new(Some(lying), payload(2, 2, 8, 4))
            .verify(&CacheSignature::expected(
                "model-a",
                flat(8),
                2,
                2,
                8,
                KvDtype::F32,
                8,
            ))
            .expect_err("the payload settles the block size, not the stamp");
        assert_eq!(
            err,
            SignatureError::BlockSizeMismatch {
                block_size: 8,
                tokens: 4,
            }
        );
    }

    /// v1 blocks recorded no window at all. Reading one would mean
    /// assuming it was aligned -- the assumption this whole item
    /// exists to refuse -- so the readable-set drops it.
    #[test]
    fn blocks_from_the_pre_layout_format_are_not_readable() {
        assert!(!READABLE_FORMAT_VERSIONS.contains(&1));
        let layers = payload(2, 2, 8, 4);
        let mut signature =
            CacheSignature::from_payload("model-a", flat(4), &layers).expect("signature");
        signature.format_version = 1;
        let err = UnverifiedBlock::new(Some(signature), layers)
            .verify(&CacheSignature::expected(
                "model-a",
                flat(4),
                2,
                2,
                8,
                KvDtype::F32,
                4,
            ))
            .expect_err("a v1 block cannot say what layout it was cut under");
        assert_eq!(
            err,
            SignatureError::UnsupportedFormat {
                found: 1,
                readable: READABLE_FORMAT_VERSIONS,
            }
        );
    }

    #[test]
    fn errors_name_the_field_that_changed() {
        let text = SignatureError::Incompatible {
            field: "head_dim",
            expected: "128".into(),
            found: "64".into(),
        }
        .to_string();
        assert!(text.contains("head_dim"), "{text}");
        assert!(text.contains("64"), "{text}");
        assert!(text.contains("128"), "{text}");
        assert!(SignatureError::Unmarked.to_string().contains("unmarked"));
    }
}
