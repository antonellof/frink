//! The disk tier for the KV prefix cache: where a block goes so that a
//! prefix survives eviction from RAM, and a process restart.
//!
//! One file per block, named by its
//! [`BlockHash`](crate::kv_block::BlockHash) and sharded into
//! subdirectories by a hex prefix of that hash, so a machine that has
//! cached a million prefixes never puts a million entries in one
//! directory. The payload is the block's per-layer K and V tensors,
//! flattened, in the cache's own dtype -- **no re-encoding, no
//! compression**: a KV block is already dense float data, and a
//! compressor would only spend CPU on the request path to lose.
//!
//! # What a reader is protected from
//!
//! A cache file outlives the process that wrote it, so every failure
//! mode here is "someone else's bytes":
//!
//! - **A torn write.** The publish is temp-file + `fsync` + `rename`,
//!   which is atomic within a directory on every filesystem frink
//!   targets -- a reader sees the whole file or no file. But a crash
//!   mid-`write` to the *temp* file, a truncated copy, or a partial
//!   restore from a backup can still leave a short file lying around,
//!   so the format records its own total length and a SHA-256 of its
//!   body. A file that does not match is refused
//!   ([`BlockFormatError`]), never partially deserialized.
//! - **A different build.** The format is versioned with an explicit
//!   readable-set; an unknown version is refused rather than guessed
//!   at.
//! - **A different model or config.** That is
//!   [`kv_signature`](crate::kv_signature)'s job, and this module does
//!   not duplicate it: a decoded file becomes an
//!   [`UnverifiedBlock`], and only
//!   [`UnverifiedBlock::verify`] against the reader's own expectation
//!   produces a usable [`KvBlock`].
//!
//! # The write-ordering invariant
//!
//! A write is accepted on one thread and finished on another, so for a
//! while a block is "in the store" without being on disk. The rule that
//! makes that safe, and the one every step below is ordered around:
//!
//! > **buffer -> index -> queue.** A concurrent reader must never see
//! > an index hit for a block that has neither a file nor a buffered
//! > payload.
//!
//! So the payload is reachable *before* anything claims the block
//! exists, and on the way out the file is published *before* the
//! buffered copy is released. A reader holds the index lock while it
//! consults the buffer, because "this block is not on disk yet" and
//! "here is its payload" have to be one decision -- as two, the writer
//! can publish and release in between and the reader finds nothing.
//! When that invariant does break, the reader gets
//! [`StoreError::MissingPayload`] rather than a quiet miss: a
//! correctness bug that degrades into a cache miss is a bug nobody ever
//! finds.
//!
//! The queue is bounded and **never drops**: a full queue makes the
//! caller write the block itself ([`DiskStats::inline_writes`] counts
//! it). Dropping writes silently would be indistinguishable from a cold
//! cache later.
//!
//! # Layout
//!
//! ```text
//! <root>/.tmp/<hash>.<pid>.<n>.tmp     in-progress writes
//! <root>/<hh>/<full-hex>.kvb           published blocks (hh = shard prefix)
//! ```
//!
//! # File format (version 2)
//!
//! ```text
//! magic           8   b"FRXKVBLK"
//! format_version  4   u32 LE, checked against READABLE_FORMAT_VERSIONS
//! header_len      4   u32 LE
//! body_len        8   u64 LE
//! digest         32   SHA-256 over header || body
//! header  header_len  block hash, dims, dtype, block layout, model identity
//! body      body_len  per layer: all K elements, then all V elements
//! ```
//!
//! Version 2 added the two block-layout fields -- block size and
//! sliding window -- and version 1 was dropped from the readable set
//! rather than being read with the window assumed absent. A v1 file
//! cannot say what window it was cut under, and "it did not say" is not
//! "there was none": see [`kv_swa`](crate::kv_swa) for what a
//! mis-aligned block does to an answer. A restart onto this build
//! therefore starts from a cold cache once, and the old files are
//! evicted as unreadable rather than reinterpreted.
//!
//! The digest covers header and body but not the fixed prefix, so the
//! lengths are checked against the real file size *before* anything is
//! hashed or parsed -- a 4 GB `body_len` on a 200-byte file is rejected
//! by arithmetic, not by allocating.

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use sha2::{Digest, Sha256};

use crate::cache::KvCache;
use crate::kv_block::BlockHash;
use crate::kv_signature::{
    CacheSignature, KvBlock, KvDtype, UnverifiedBlock, BLOCK_FORMAT_VERSION,
    READABLE_FORMAT_VERSIONS,
};
use crate::kv_store::KvBuf;
use crate::kv_swa::{BlockLayout, BlockLayoutError};

const MAGIC: &[u8; 8] = b"FRXKVBLK";
/// Fixed u32 fields in a block header, after the 32-byte hash:
/// n_layers, n_kv_heads, head_dim, tokens, dtype, block_size,
/// sliding_window, model_len.
const HEADER_FIELDS: usize = 8;
/// magic + version + header_len + body_len + digest.
const PREFIX_LEN: usize = 8 + 4 + 4 + 8 + 32;
/// Extension of a published block file.
pub const BLOCK_FILE_EXT: &str = "kvb";
/// Subdirectory holding in-progress writes. Not a valid shard name --
/// shard directories are lowercase hex, and `.` is not a hex digit --
/// so it can never collide with one.
const TMP_DIR: &str = ".tmp";

const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;

fn dtype_code(dtype: KvDtype) -> u32 {
    match dtype {
        KvDtype::F32 => DTYPE_F32,
        KvDtype::F16 => DTYPE_F16,
    }
}

fn dtype_from_code(code: u32) -> Option<KvDtype> {
    match code {
        DTYPE_F32 => Some(KvDtype::F32),
        DTYPE_F16 => Some(KvDtype::F16),
        _ => None,
    }
}

/// Encodes a sliding window as a u32, with 0 meaning "no window".
/// `BlockLayout` refuses a zero window, so the two cases cannot
/// collide.
fn window_code(window: Option<usize>) -> u32 {
    window.unwrap_or(0) as u32
}

fn window_from_code(code: u32) -> Option<usize> {
    if code == 0 {
        None
    } else {
        Some(code as usize)
    }
}

fn dtype_width(dtype: KvDtype) -> usize {
    dtype.elem_bytes()
}

/// Why a block file was refused. Every variant means "these bytes are
/// not a block this build can read", and none of them is recoverable by
/// reading harder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockFormatError {
    /// Shorter than the fixed prefix: there is not even a header to
    /// check.
    TooShort { len: usize },
    /// Not a frink block file at all.
    BadMagic,
    /// Written by a build whose layout this one does not know.
    UnsupportedFormat {
        found: u32,
        readable: &'static [u32],
    },
    /// The file's own declared length does not match the bytes present
    /// -- a half-written or truncated file.
    Truncated { expected: u64, actual: u64 },
    /// Right length, wrong bytes: bit rot, an interrupted overwrite, or
    /// a file that was edited.
    ChecksumMismatch,
    /// Structurally impossible content: a dimension of zero, a body
    /// that cannot hold the tensors the header describes.
    Malformed(&'static str),
    /// A dtype code this build has no reader for.
    UnknownDtype(u32),
    /// The recorded block layout is not one any correct writer could
    /// have produced -- a block size that does not divide the sliding
    /// window it was cut against.
    BadLayout(BlockLayoutError),
}

impl std::fmt::Display for BlockFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFormatError::TooShort { len } => write!(
                f,
                "KV block file is {len} bytes, shorter than the {PREFIX_LEN}-byte header prefix"
            ),
            BlockFormatError::BadMagic => write!(f, "KV block file has the wrong magic"),
            BlockFormatError::UnsupportedFormat { found, readable } => write!(
                f,
                "KV block file format version {found} is not readable by this build (readable: {readable:?})"
            ),
            BlockFormatError::Truncated { expected, actual } => write!(
                f,
                "KV block file declares {expected} bytes but is {actual}; refusing a torn file"
            ),
            BlockFormatError::ChecksumMismatch => {
                write!(f, "KV block file failed its SHA-256 checksum")
            }
            BlockFormatError::Malformed(what) => {
                write!(f, "KV block file is malformed: {what}")
            }
            BlockFormatError::UnknownDtype(code) => {
                write!(f, "KV block file has unknown dtype code {code}")
            }
            BlockFormatError::BadLayout(err) => {
                write!(f, "KV block file records an impossible block layout: {err}")
            }
        }
    }
}

impl std::error::Error for BlockFormatError {}

/// Serializes a block. The `hash` is stored inside the file as well as
/// in its name, so a block found under a wrong or renamed path can
/// still be checked against the identity it claims.
pub fn encode_block(hash: &BlockHash, block: &KvBlock) -> Vec<u8> {
    let sig = block.signature();
    let mut header = Vec::with_capacity(64 + sig.model.len());
    header.extend_from_slice(hash.as_bytes());
    header.extend_from_slice(&(sig.n_layers as u32).to_le_bytes());
    header.extend_from_slice(&(sig.n_kv_heads as u32).to_le_bytes());
    header.extend_from_slice(&(sig.head_dim as u32).to_le_bytes());
    header.extend_from_slice(&(sig.tokens as u32).to_le_bytes());
    header.extend_from_slice(&dtype_code(sig.dtype).to_le_bytes());
    header.extend_from_slice(&(sig.layout.block_size() as u32).to_le_bytes());
    // 0 encodes "no sliding window". A real window is never 0 --
    // `BlockLayout` refuses `Some(0)` precisely so this encoding is
    // unambiguous.
    header.extend_from_slice(&(window_code(sig.layout.sliding_window())).to_le_bytes());
    header.extend_from_slice(&(sig.model.len() as u32).to_le_bytes());
    header.extend_from_slice(sig.model.as_bytes());

    let mut body = Vec::with_capacity(body_len(sig) as usize);
    for layer in block.layers() {
        // At the store's own width, bit for bit: the signature names the
        // dtype, so a reader at the other one misses rather than widens.
        push_le(&mut body, &layer.k);
        push_le(&mut body, &layer.v);
    }

    let mut digest = Sha256::new();
    digest.update(&header);
    digest.update(&body);
    let digest: [u8; 32] = digest.finalize().into();

    let mut out = Vec::with_capacity(PREFIX_LEN + header.len() + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&BLOCK_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(&digest);
    out.extend_from_slice(&header);
    out.extend_from_slice(&body);
    out
}

/// Bytes the body of a block with this signature occupies. Lets the
/// store charge a block against its budget without serializing it
/// first.
fn body_len(sig: &CacheSignature) -> u64 {
    let per_layer = sig.tokens as u64
        * sig.n_kv_heads as u64
        * sig.head_dim as u64
        * dtype_width(sig.dtype) as u64;
    // K and V.
    per_layer * 2 * sig.n_layers as u64
}

/// Total on-disk size of a block with this signature, header included.
pub fn encoded_len(sig: &CacheSignature) -> u64 {
    let header = 32 + 4 * HEADER_FIELDS as u64 + sig.model.len() as u64;
    PREFIX_LEN as u64 + header + body_len(sig)
}

/// The identity and payload recovered from a block file. The signature
/// is deliberately *unverified*: use
/// [`UnverifiedBlock::verify`](crate::kv_signature::UnverifiedBlock::verify)
/// to turn it into a block this process may use.
#[derive(Debug)]
pub struct DecodedBlock {
    /// The hash the file claims to be stored under.
    pub hash: BlockHash,
    pub block: UnverifiedBlock,
}

/// Parses a block file. Checks, in order: length, magic, format
/// version, declared-vs-actual size, checksum, then structure. Nothing
/// is allocated from a length field until that length has been checked
/// against the bytes actually present.
pub fn decode_block(bytes: &[u8]) -> Result<DecodedBlock, BlockFormatError> {
    if bytes.len() < PREFIX_LEN {
        return Err(BlockFormatError::TooShort { len: bytes.len() });
    }
    if &bytes[..8] != MAGIC {
        return Err(BlockFormatError::BadMagic);
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if !READABLE_FORMAT_VERSIONS.contains(&version) {
        return Err(BlockFormatError::UnsupportedFormat {
            found: version,
            readable: READABLE_FORMAT_VERSIONS,
        });
    }
    let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as u64;
    let body_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let declared = PREFIX_LEN as u64 + header_len + body_len;
    if declared != bytes.len() as u64 {
        return Err(BlockFormatError::Truncated {
            expected: declared,
            actual: bytes.len() as u64,
        });
    }
    let digest_recorded = &bytes[24..PREFIX_LEN];
    let mut digest = Sha256::new();
    digest.update(&bytes[PREFIX_LEN..]);
    let digest: [u8; 32] = digest.finalize().into();
    if digest != digest_recorded {
        return Err(BlockFormatError::ChecksumMismatch);
    }

    let header = &bytes[PREFIX_LEN..PREFIX_LEN + header_len as usize];
    let body = &bytes[PREFIX_LEN + header_len as usize..];
    if header.len() < 32 + 4 * HEADER_FIELDS {
        return Err(BlockFormatError::Malformed(
            "header shorter than its fields",
        ));
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&header[..32]);
    let hash = BlockHash::from_bytes(hash);
    let field = |i: usize| u32::from_le_bytes(header[32 + i * 4..36 + i * 4].try_into().unwrap());
    let n_layers = field(0) as usize;
    let n_kv_heads = field(1) as usize;
    let head_dim = field(2) as usize;
    let tokens = field(3) as usize;
    let dtype_code = field(4);
    let block_size = field(5) as usize;
    let window_code = field(6);
    let model_len = field(7) as usize;
    let dtype = dtype_from_code(dtype_code).ok_or(BlockFormatError::UnknownDtype(dtype_code))?;
    if header.len() != 32 + 4 * HEADER_FIELDS + model_len {
        return Err(BlockFormatError::Malformed(
            "model name length disagrees with header",
        ));
    }
    let model = std::str::from_utf8(&header[32 + 4 * HEADER_FIELDS..])
        .map_err(|_| BlockFormatError::Malformed("model name is not UTF-8"))?
        .to_string();
    if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Err(BlockFormatError::Malformed(
            "zero layers, heads, or head dim",
        ));
    }
    // A file whose recorded layout is not a layout at all -- a block
    // size that does not divide its window -- is refused here rather
    // than reconstructed into a `BlockLayout` that could not have been
    // built by any correct writer.
    let layout = BlockLayout::new(block_size, window_from_code(window_code))
        .map_err(BlockFormatError::BadLayout)?;

    let per_layer_elems = tokens
        .checked_mul(n_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or(BlockFormatError::Malformed("layer size overflows"))?;
    let expected_body = (per_layer_elems as u64)
        .checked_mul(2 * n_layers as u64)
        .and_then(|n| n.checked_mul(dtype_width(dtype) as u64))
        .ok_or(BlockFormatError::Malformed("body size overflows"))?;
    if expected_body != body.len() as u64 {
        return Err(BlockFormatError::Malformed(
            "body does not match declared dims",
        ));
    }

    let mut layers = Vec::with_capacity(n_layers);
    let mut offset = 0usize;
    for _ in 0..n_layers {
        let width = dtype_width(dtype);
        let k = read_le(dtype, &body[offset..offset + per_layer_elems * width]);
        offset += per_layer_elems * width;
        let v = read_le(dtype, &body[offset..offset + per_layer_elems * width]);
        offset += per_layer_elems * width;
        let mut cache = KvCache::new(n_kv_heads, head_dim).with_dtype(dtype);
        cache.k = k;
        cache.v = v;
        // The rows were just written by hand, so this is the one place
        // a position count is set rather than counted by `push`.
        cache.set_positions(tokens);
        layers.push(cache);
    }

    let signature = CacheSignature {
        format_version: version,
        model,
        n_layers,
        n_kv_heads,
        head_dim,
        dtype,
        tokens,
        layout,
    };
    Ok(DecodedBlock {
        hash,
        block: UnverifiedBlock::new(Some(signature), layers),
    })
}

fn push_le(body: &mut Vec<u8>, buf: &KvBuf) {
    match buf {
        KvBuf::F32(v) => v
            .iter()
            .for_each(|x| body.extend_from_slice(&x.to_le_bytes())),
        KvBuf::F16(v) => v
            .iter()
            .for_each(|x| body.extend_from_slice(&x.to_le_bytes())),
    }
}

fn read_le(dtype: KvDtype, bytes: &[u8]) -> KvBuf {
    match dtype {
        KvDtype::F32 => KvBuf::F32(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        ),
        KvDtype::F16 => KvBuf::F16(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c))
                .collect(),
        ),
    }
}

/// Something went wrong reaching the disk tier. Corruption and
/// incompatibility are **not** here: those are misses, reported through
/// [`DiskStats`], because a caller's only sane response to either is to
/// recompute the prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StoreError {
    Io {
        op: &'static str,
        path: PathBuf,
        message: String,
    },
    /// The index named a block whose payload is nowhere: not on disk,
    /// not in the write buffer. This is not a cache miss -- it is the
    /// write-ordering invariant (buffer -> index -> queue) having been
    /// violated, i.e. a bug in this module, and it is surfaced rather
    /// than smoothed into a miss so a test can fail on it.
    MissingPayload { hash: BlockHash },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io { op, path, message } => {
                write!(
                    f,
                    "KV block store failed to {op} {}: {message}",
                    path.display()
                )
            }
            StoreError::MissingPayload { hash } => write!(
                f,
                "KV block store index names {hash:?} but it has neither a file nor a buffered \
                 payload; the write-ordering invariant was violated"
            ),
        }
    }
}

impl std::error::Error for StoreError {}

fn io_err(op: &'static str, path: &Path, err: io::Error) -> StoreError {
    StoreError::Io {
        op,
        path: path.to_path_buf(),
        message: err.to_string(),
    }
}

/// Asks how many bytes are still free on the filesystem holding a
/// path. `None` means "cannot tell", and the store then trusts only its
/// configured byte budget.
///
/// Injectable so the budget's behaviour under a nearly-full disk is
/// testable without actually filling one.
pub type FreeSpaceProbe = Arc<dyn Fn(&Path) -> Option<u64> + Send + Sync>;

/// Free space via `statvfs`. `f_bavail` (blocks available to an
/// unprivileged process), not `f_bfree`, because the reserved blocks a
/// filesystem keeps back are not ours to spend.
#[cfg(unix)]
#[allow(clippy::unnecessary_cast)] // field widths differ across unixes
fn platform_free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `c_path` is a NUL-terminated path that outlives the call,
    // and `stat` is a correctly sized, writable `statvfs`.
    let stat = unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c_path.as_ptr(), &mut stat) != 0 {
            return None;
        }
        stat
    };
    let block = if stat.f_frsize > 0 {
        stat.f_frsize as u64
    } else {
        stat.f_bsize as u64
    };
    Some((stat.f_bavail as u64).saturating_mul(block))
}

#[cfg(not(unix))]
fn platform_free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// A free-space reading and when it was taken. `statvfs` is a syscall
/// per call and the answer changes slowly, so it is cached -- but only
/// for a TTL, and any `ENOSPC` throws it away immediately, because at
/// that moment the cached number is known to be a lie.
struct FreeSpace {
    checked_at: Option<Instant>,
    bytes: Option<u64>,
}

/// How the store is sized, laid out, and how much writing it will do
/// off the calling thread.
#[derive(Clone)]
pub struct DiskConfig {
    /// Directory the store owns. Created if absent.
    pub root: PathBuf,
    /// Byte budget for blocks the store is accounting for. Eviction
    /// keeps the store at or under this.
    pub max_bytes: u64,
    /// Hex characters of the hash used as the shard subdirectory name.
    /// 2 gives 256 shards, which keeps directory sizes sane well past a
    /// million blocks.
    pub shard_chars: usize,
    /// Writes that may be waiting for a writer thread at once. When it
    /// is full, [`DiskKvStore::put`] writes on the calling thread
    /// instead of dropping the block -- backpressure, not loss.
    pub queue_capacity: usize,
    /// Background writer threads. `0` is legitimate and means every
    /// write happens on the thread that asked for it.
    pub writer_threads: usize,
    /// Background reader threads, serving prefetches and any demand
    /// read that does not want to block its own thread. `0` means every
    /// read happens on the thread that asked for it, and a prefetch is
    /// a no-op.
    pub reader_threads: usize,
    /// Blocks that may sit in the prefetch staging area at once. A
    /// prefetch is a hint, so this is a hard refusal rather than
    /// backpressure: reading ahead must never be the thing that runs
    /// the process out of memory.
    pub prefetch_capacity: usize,
    /// Bytes to leave free on the filesystem. The store evicts to stay
    /// this far away from a full disk, rather than letting `ENOSPC` be
    /// the mechanism that tells it to stop.
    pub reserve_bytes: u64,
    /// How long a free-space reading is trusted before it is taken
    /// again.
    pub free_space_ttl: std::time::Duration,
    /// How free space is measured. Defaults to `statvfs` on unix, and
    /// to "cannot tell" elsewhere.
    pub free_space_probe: FreeSpaceProbe,
}

impl std::fmt::Debug for DiskConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskConfig")
            .field("root", &self.root)
            .field("max_bytes", &self.max_bytes)
            .field("shard_chars", &self.shard_chars)
            .field("queue_capacity", &self.queue_capacity)
            .field("writer_threads", &self.writer_threads)
            .field("reader_threads", &self.reader_threads)
            .field("prefetch_capacity", &self.prefetch_capacity)
            .field("reserve_bytes", &self.reserve_bytes)
            .field("free_space_ttl", &self.free_space_ttl)
            .finish_non_exhaustive()
    }
}

impl DiskConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        DiskConfig {
            root: root.into(),
            max_bytes: 1 << 30,
            shard_chars: 2,
            queue_capacity: 64,
            writer_threads: 1,
            reader_threads: 2,
            prefetch_capacity: 64,
            reserve_bytes: 1 << 30,
            free_space_ttl: std::time::Duration::from_secs(2),
            free_space_probe: Arc::new(platform_free_bytes),
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn with_shard_chars(mut self, shard_chars: usize) -> Self {
        self.shard_chars = shard_chars.clamp(1, 8);
        self
    }

    pub fn with_queue_capacity(mut self, queue_capacity: usize) -> Self {
        self.queue_capacity = queue_capacity;
        self
    }

    pub fn with_writer_threads(mut self, writer_threads: usize) -> Self {
        self.writer_threads = writer_threads;
        self
    }

    pub fn with_reader_threads(mut self, reader_threads: usize) -> Self {
        self.reader_threads = reader_threads;
        self
    }

    pub fn with_prefetch_capacity(mut self, prefetch_capacity: usize) -> Self {
        self.prefetch_capacity = prefetch_capacity;
        self
    }

    pub fn with_reserve_bytes(mut self, reserve_bytes: u64) -> Self {
        self.reserve_bytes = reserve_bytes;
        self
    }

    pub fn with_free_space_ttl(mut self, free_space_ttl: std::time::Duration) -> Self {
        self.free_space_ttl = free_space_ttl;
        self
    }

    pub fn with_free_space_probe(mut self, probe: FreeSpaceProbe) -> Self {
        self.free_space_probe = probe;
        self
    }
}

#[derive(Default)]
struct Stats {
    writes: AtomicU64,
    queued_writes: AtomicU64,
    /// Writes that ran on the calling thread because the queue was
    /// full. The plan's "count the fallbacks": a store that is
    /// permanently inline is a store whose queue is too small or whose
    /// disk is too slow, and that is invisible without this.
    inline_writes: AtomicU64,
    write_failures: AtomicU64,
    /// Queued writes whose block was evicted (or superseded) before a
    /// writer thread reached it.
    write_skipped: AtomicU64,
    write_nanos: AtomicU64,
    /// Writes whose block was evicted while it was being written, so
    /// the published file was withdrawn again.
    write_raced_eviction: AtomicU64,
    hits: AtomicU64,
    /// Reads served from the write buffer, before the block reached
    /// disk. These are what make the write path asynchronous *and*
    /// immediately visible.
    buffer_hits: AtomicU64,
    misses: AtomicU64,
    /// Files that failed [`decode_block`] and were quarantined.
    corrupt: AtomicU64,
    /// Blocks that decoded cleanly but do not match this reader's
    /// signature expectation.
    incompatible: AtomicU64,
    read_nanos: AtomicU64,
    evictions: AtomicU64,
    evicted_bytes: AtomicU64,
    /// Reads handed to a reader thread by [`DiskKvStore::prefetch`].
    prefetch_issued: AtomicU64,
    /// Prefetches refused: staging full, no reader threads, or the
    /// block was already staged or in flight.
    prefetch_dropped: AtomicU64,
    /// Demand reads that found a prefetch already *finished*. This is
    /// the one that says the read-ahead paid for itself: the request
    /// did no I/O at all.
    prefetch_hits: AtomicU64,
    /// Demand reads that found a prefetch still running and waited for
    /// it instead of issuing a second read of the same file.
    prefetch_waits: AtomicU64,
    /// Reads that ran on a reader thread rather than the caller's.
    async_reads: AtomicU64,
    /// Writes that failed because the filesystem was full. Non-zero
    /// means the budget lost the race it exists to win.
    enospc: AtomicU64,
    /// Eviction passes whose ceiling came from free disk rather than
    /// from `max_bytes`.
    space_clamped: AtomicU64,
}

/// A snapshot of the tier's behaviour. Note the two time-valued fields:
/// hit *rate* alone cannot tell an operator whether a disk hit was
/// cheaper than recomputing the prefix, which is the only question that
/// decides whether the tier is worth having.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiskStats {
    pub blocks: usize,
    /// Everything the store has accepted, published or still buffered.
    pub bytes: u64,
    /// The subset that has actually reached the filesystem.
    pub disk_bytes: u64,
    pub queue_depth: usize,
    pub writes: u64,
    pub queued_writes: u64,
    pub inline_writes: u64,
    pub write_failures: u64,
    pub write_skipped: u64,
    pub write_raced_eviction: u64,
    pub write_nanos: u64,
    pub hits: u64,
    pub buffer_hits: u64,
    pub misses: u64,
    pub corrupt: u64,
    pub incompatible: u64,
    pub read_nanos: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
    pub prefetch_issued: u64,
    pub prefetch_dropped: u64,
    pub prefetch_hits: u64,
    pub prefetch_waits: u64,
    pub async_reads: u64,
    pub staged_blocks: usize,
    pub enospc: u64,
    pub space_clamped: u64,
    /// The ceiling eviction is currently working to: `max_bytes`, or
    /// less when free disk says so.
    pub effective_capacity: u64,
}

struct Entry {
    bytes: u64,
    last_used: u64,
    /// False between admitting the block and its file landing under its
    /// final name. While it is false the payload lives in the write
    /// buffer, and a reader is served from there.
    published: bool,
    /// Bumped every time this hash is admitted, so a write that
    /// finishes after its entry was evicted and re-created cannot mark
    /// the *new* entry published, and a queued job whose block has been
    /// superseded can tell.
    generation: u64,
}

struct Index {
    entries: HashMap<BlockHash, Entry>,
    /// Everything the store has accepted, published or still buffered.
    bytes: u64,
    /// The subset that is actually occupying filesystem space. The
    /// free-space budget reasons about this one: a block that has been
    /// accepted but not written has not taken any disk yet, and
    /// charging it twice (once here, once as space the device has
    /// already lost) makes the ceiling oscillate.
    disk_bytes: u64,
    clock: u64,
}

impl Index {
    fn touch(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn insert_entry(&mut self, hash: BlockHash, entry: Entry) {
        let bytes = entry.bytes;
        if let Some(previous) = self.entries.insert(hash, entry) {
            self.uncharge(&previous);
        }
        self.bytes += bytes;
    }

    fn remove_entry(&mut self, hash: &BlockHash) -> Option<Entry> {
        let entry = self.entries.remove(hash)?;
        self.uncharge(&entry);
        Some(entry)
    }

    fn uncharge(&mut self, entry: &Entry) {
        self.bytes -= entry.bytes;
        if entry.published {
            self.disk_bytes -= entry.bytes;
        }
    }
}

/// Where a block's payload is, once the index has been consulted.
enum Source {
    Disk(PathBuf),
    Buffer(Arc<KvBlock>),
}

/// A block that has been accepted but whose file is not on disk yet.
struct Buffered {
    generation: u64,
    block: Arc<KvBlock>,
}

#[derive(Clone, Copy)]
struct WriteJob {
    hash: BlockHash,
    generation: u64,
}

struct QueueState {
    jobs: VecDeque<WriteJob>,
    running: usize,
    shutdown: bool,
}

/// A bounded queue of pending block writes.
///
/// Bounded, and **never lossy**: `try_push` refusing is the caller's
/// signal to write the block itself, not to drop it. A dropped write is
/// indistinguishable from a cache miss later, which is exactly the kind
/// of silent degradation that makes a cache tier impossible to trust.
struct WriteQueue {
    state: Mutex<QueueState>,
    ready: Condvar,
    idle: Condvar,
    capacity: usize,
}

impl WriteQueue {
    fn new(capacity: usize) -> Self {
        WriteQueue {
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                running: 0,
                shutdown: false,
            }),
            ready: Condvar::new(),
            idle: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state.lock().expect("kv disk write queue poisoned")
    }

    /// `false` means "full, or shutting down" -- write it yourself.
    fn try_push(&self, job: WriteJob) -> bool {
        let mut state = self.lock();
        if state.shutdown || state.jobs.len() >= self.capacity {
            return false;
        }
        state.jobs.push_back(job);
        self.ready.notify_one();
        true
    }

    fn pop_blocking(&self) -> Option<WriteJob> {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                state.running += 1;
                return Some(job);
            }
            if state.shutdown {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .expect("kv disk write queue poisoned");
        }
    }

    fn pop_now(&self) -> Option<WriteJob> {
        let mut state = self.lock();
        let job = state.jobs.pop_front()?;
        state.running += 1;
        Some(job)
    }

    fn finish(&self) {
        let mut state = self.lock();
        state.running -= 1;
        self.idle.notify_all();
    }

    fn shutdown(&self) {
        let mut state = self.lock();
        state.shutdown = true;
        self.ready.notify_all();
    }

    fn depth(&self) -> usize {
        self.lock().jobs.len()
    }
}

/// What a read produced. `Ok(None)` is a miss (absent, corrupt, or
/// incompatible); `Err` is I/O, or the write-ordering invariant
/// breaking.
pub type ReadOutcome = Result<Option<Arc<KvBlock>>, StoreError>;

/// A read in progress, or one that has finished and is waiting to be
/// claimed. Shared between the thread that asked for it, the thread
/// that runs it, and any later request for the same block.
struct ReadSlot {
    /// The signature this read was issued for. A staged result is only
    /// reusable by a reader that wants the *same* shape -- otherwise a
    /// prefetch issued under one config would hand its answer to a
    /// request under another.
    expected: CacheSignature,
    outcome: Mutex<Option<ReadOutcome>>,
    done: Condvar,
}

impl ReadSlot {
    fn pending(expected: CacheSignature) -> Arc<Self> {
        Arc::new(ReadSlot {
            expected,
            outcome: Mutex::new(None),
            done: Condvar::new(),
        })
    }

    fn ready(expected: CacheSignature, outcome: ReadOutcome) -> Arc<Self> {
        Arc::new(ReadSlot {
            expected,
            outcome: Mutex::new(Some(outcome)),
            done: Condvar::new(),
        })
    }

    fn is_ready(&self) -> bool {
        self.outcome
            .lock()
            .expect("kv disk read slot poisoned")
            .is_some()
    }

    fn fulfil(&self, outcome: ReadOutcome) {
        let mut slot = self.outcome.lock().expect("kv disk read slot poisoned");
        *slot = Some(outcome);
        self.done.notify_all();
    }

    fn wait(&self) -> ReadOutcome {
        let mut slot = self.outcome.lock().expect("kv disk read slot poisoned");
        loop {
            if let Some(outcome) = slot.as_ref() {
                return outcome.clone();
            }
            slot = self.done.wait(slot).expect("kv disk read slot poisoned");
        }
    }
}

struct ReadJob {
    hash: BlockHash,
    path: PathBuf,
    slot: Arc<ReadSlot>,
}

struct ReadQueueState {
    jobs: VecDeque<ReadJob>,
    shutdown: bool,
}

/// Pending disk reads. Unlike the write queue this one *is* allowed to
/// refuse work -- a prefetch is a hint, and a refused hint costs a
/// later request one read. A refused *demand* read is not dropped
/// either: it runs on the calling thread.
struct ReadQueue {
    state: Mutex<ReadQueueState>,
    ready: Condvar,
    capacity: usize,
}

impl ReadQueue {
    fn new(capacity: usize) -> Self {
        ReadQueue {
            state: Mutex::new(ReadQueueState {
                jobs: VecDeque::new(),
                shutdown: false,
            }),
            ready: Condvar::new(),
            capacity: capacity.max(1),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ReadQueueState> {
        self.state.lock().expect("kv disk read queue poisoned")
    }

    /// `demand` jumps the queue: a request that is waiting must not sit
    /// behind speculative read-ahead.
    fn try_push(&self, job: ReadJob, demand: bool) -> bool {
        let mut state = self.lock();
        if state.shutdown || state.jobs.len() >= self.capacity {
            return false;
        }
        if demand {
            state.jobs.push_front(job);
        } else {
            state.jobs.push_back(job);
        }
        self.ready.notify_one();
        true
    }

    fn pop_blocking(&self) -> Option<ReadJob> {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                return Some(job);
            }
            if state.shutdown {
                return None;
            }
            state = self.ready.wait(state).expect("kv disk read queue poisoned");
        }
    }

    fn shutdown(&self) {
        let mut state = self.lock();
        state.shutdown = true;
        self.ready.notify_all();
    }
}

/// A read that may not have finished yet.
///
/// The disk tier is asynchronous **by construction**, not as a later
/// retrofit: [`DiskKvStore::get`] is this handle plus a `wait`, so
/// there is no synchronous read path that a prefetch has to work
/// around.
pub struct ReadHandle {
    shared: Arc<Shared>,
    hash: BlockHash,
    slot: Arc<ReadSlot>,
    /// Whether this handle's slot is registered in the staging map and
    /// should be removed once claimed.
    staged: bool,
}

impl ReadHandle {
    /// True if the block is already in hand -- a memory hit, a miss, or
    /// a prefetch that has landed.
    pub fn is_ready(&self) -> bool {
        self.slot.is_ready()
    }

    /// The result, without blocking. Returns `None` if the read is
    /// still running.
    pub fn try_claim(&self) -> Option<ReadOutcome> {
        if !self.slot.is_ready() {
            return None;
        }
        Some(self.claim())
    }

    /// Blocks until the read finishes.
    pub fn wait(self) -> ReadOutcome {
        self.claim()
    }

    fn claim(&self) -> ReadOutcome {
        let outcome = self.slot.wait();
        if self.staged {
            self.shared.unstage(&self.hash, &self.slot);
        }
        outcome
    }
}

impl std::fmt::Debug for ReadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadHandle")
            .field("hash", &self.hash)
            .field("ready", &self.is_ready())
            .finish()
    }
}

#[cfg(test)]
type Hook = Arc<dyn Fn(&BlockHash) + Send + Sync>;

/// Which order the write path uses. Production is
/// `BufferThenIndex`; the other two exist so a test can prove the
/// invariant test is not vacuous -- a concurrency test that passes on
/// broken code proves nothing.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum WriteOrder {
    #[default]
    BufferThenIndex,
    /// Index the block before buffering it: a reader can then hit the
    /// index for a block with no file and no payload.
    IndexBeforeBuffer,
    /// Release the buffered payload before marking the file published:
    /// same hole, at the other end of the write.
    DropBufferBeforeMarking,
}

#[cfg(test)]
#[derive(Default)]
struct Hooks {
    order: Mutex<WriteOrder>,
    /// Called after the rename and before the post-publish eviction
    /// re-check, so a test can evict the block in exactly the window
    /// the re-check exists to cover.
    after_rename: Mutex<Option<Hook>>,
    /// Called inside the window between the two steps of admission.
    in_put_window: Mutex<Option<Hook>>,
    /// Called inside the window between the two steps of publication.
    in_publish_window: Mutex<Option<Hook>>,
    /// Makes the next write fail as though the filesystem were full,
    /// which is the one failure that cannot be arranged for real
    /// without filling a real disk.
    fail_with_enospc: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl Hooks {
    fn fire(slot: &Mutex<Option<Hook>>, hash: &BlockHash) {
        // Cloned out before calling, so a hook that re-enters the store
        // cannot deadlock on the hook slot itself.
        let hook = slot.lock().expect("kv disk hook poisoned").clone();
        if let Some(hook) = hook {
            hook(hash);
        }
    }
}

/// Everything the store's threads share. Deliberately holds no join
/// handles: the writer threads hold an `Arc<Shared>`, so a `Drop` here
/// that joined them could run *on* a writer thread and deadlock. The
/// handles live in [`DiskKvStore`], which is not `Clone`.
struct Shared {
    root: PathBuf,
    shard_chars: usize,
    max_bytes: u64,
    index: Mutex<Index>,
    /// Blocks accepted but not yet on disk.
    ///
    /// **Lock order: `index`, then `buffer`, never the reverse.** A
    /// reader decides "the index says this block exists" and "here is
    /// its payload" as one atomic act; otherwise the writer could mark
    /// a block published and drop its buffered copy in between, and the
    /// reader would find nothing.
    buffer: Mutex<HashMap<BlockHash, Buffered>>,
    queue: WriteQueue,
    reads: ReadQueue,
    /// Reads in flight or finished and unclaimed, keyed by block. One
    /// slot per block, so a demand read that arrives while a prefetch
    /// is running joins it instead of reading the same file twice.
    staging: Mutex<HashMap<BlockHash, Arc<ReadSlot>>>,
    prefetch_capacity: usize,
    has_readers: bool,
    reserve_bytes: u64,
    free_space_ttl: std::time::Duration,
    free_space_probe: FreeSpaceProbe,
    /// **Lock order: `index`, then `free_space`.** Eviction reads the
    /// budget while holding the index.
    free_space: Mutex<FreeSpace>,
    stats: Stats,
    seq: AtomicU64,
    generation: AtomicU64,
    #[cfg(test)]
    hooks: Hooks,
}

/// A content-addressed block store on disk, with its own writer
/// threads.
///
/// Not `Clone` on purpose -- it owns the writer threads and joins them
/// when it drops. Share it as an `Arc<DiskKvStore>`; every method takes
/// `&self`.
pub struct DiskKvStore {
    shared: Arc<Shared>,
    writers: Vec<std::thread::JoinHandle<()>>,
    readers: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for DiskKvStore {
    /// Stops accepting queued work and joins the worker threads. Blocks
    /// already queued but not started are **not** written: they were
    /// never durable, and a shutdown that waits for an arbitrarily deep
    /// queue is worse than a cold cache. Call [`Self::flush`] first if
    /// they matter.
    fn drop(&mut self) {
        self.shared.queue.shutdown();
        self.shared.reads.shutdown();
        for writer in self.writers.drain(..) {
            let _ = writer.join();
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

impl DiskKvStore {
    /// Creates the store's directories and starts its writer threads.
    /// Does **not** scan `root` for pre-existing blocks: reattaching to
    /// a store left by a previous process is [`Self::reindex`], an
    /// explicit step, because it costs a directory walk and a caller
    /// may prefer to start cold.
    pub fn open(config: DiskConfig) -> Result<Self, StoreError> {
        let root = config.root.clone();
        fs::create_dir_all(&root).map_err(|e| io_err("create", &root, e))?;
        let tmp = root.join(TMP_DIR);
        fs::create_dir_all(&tmp).map_err(|e| io_err("create", &tmp, e))?;
        let shared = Arc::new(Shared {
            root,
            shard_chars: config.shard_chars.clamp(1, 8),
            max_bytes: config.max_bytes,
            index: Mutex::new(Index {
                entries: HashMap::new(),
                bytes: 0,
                disk_bytes: 0,
                clock: 0,
            }),
            buffer: Mutex::new(HashMap::new()),
            queue: WriteQueue::new(config.queue_capacity),
            reads: ReadQueue::new(config.queue_capacity),
            staging: Mutex::new(HashMap::new()),
            prefetch_capacity: config.prefetch_capacity,
            has_readers: config.reader_threads > 0,
            reserve_bytes: config.reserve_bytes,
            free_space_ttl: config.free_space_ttl,
            free_space_probe: Arc::clone(&config.free_space_probe),
            free_space: Mutex::new(FreeSpace {
                checked_at: None,
                bytes: None,
            }),
            stats: Stats::default(),
            seq: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            #[cfg(test)]
            hooks: Hooks::default(),
        });
        let mut writers = Vec::with_capacity(config.writer_threads);
        for n in 0..config.writer_threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("frink-kv-write-{n}"))
                .spawn(move || {
                    while let Some(job) = shared.queue.pop_blocking() {
                        shared.run_job(job);
                        shared.queue.finish();
                    }
                })
                .map_err(|e| io_err("spawn writer for", &config.root, e))?;
            writers.push(handle);
        }
        let mut readers = Vec::with_capacity(config.reader_threads);
        for n in 0..config.reader_threads {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("frink-kv-read-{n}"))
                .spawn(move || {
                    while let Some(job) = shared.reads.pop_blocking() {
                        shared.stats.async_reads.fetch_add(1, Ordering::Relaxed);
                        let outcome = shared.read_timed(&job.path, &job.hash, &job.slot.expected);
                        job.slot.fulfil(outcome);
                    }
                })
                .map_err(|e| io_err("spawn reader for", &config.root, e))?;
            readers.push(handle);
        }
        Ok(DiskKvStore {
            shared,
            writers,
            readers,
        })
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    /// Accepts a block: buffered, indexed, then queued. Returns as soon
    /// as the block is *visible* -- a reader can have it immediately,
    /// whether or not it has reached disk.
    ///
    /// If the write queue is full the block is written on this thread
    /// rather than dropped, and the fallback is counted.
    pub fn put(&self, hash: BlockHash, block: KvBlock) -> Result<(), StoreError> {
        self.shared.put(hash, block, false)
    }

    /// Like [`Self::put`], but always writes on the calling thread.
    pub fn put_blocking(&self, hash: BlockHash, block: KvBlock) -> Result<(), StoreError> {
        self.shared.put(hash, block, true)
    }

    /// Runs every pending write to completion, on this thread if no
    /// writer thread gets there first. Returns once the queue is empty
    /// and nothing is in flight.
    pub fn flush(&self) {
        loop {
            if let Some(job) = self.shared.queue.pop_now() {
                self.shared.run_job(job);
                self.shared.queue.finish();
                continue;
            }
            let state = self.shared.queue.lock();
            if state.jobs.is_empty() && state.running == 0 {
                return;
            }
            // Timed, so a store with no writer threads and a job pushed
            // by another thread cannot park here forever.
            let _ = self
                .shared
                .queue
                .idle
                .wait_timeout(state, std::time::Duration::from_millis(1));
        }
    }

    /// Looks a block up, verifying it against `expected` before
    /// returning it. Blocks until the answer is in hand.
    ///
    /// This is exactly [`Self::read_async`] plus a wait -- the disk
    /// tier has no separate synchronous read path for a prefetch to
    /// have to work around later.
    pub fn get(&self, hash: &BlockHash, expected: &CacheSignature) -> ReadOutcome {
        self.shared.read_async(hash, expected, true).wait()
    }

    /// Starts a read and returns immediately. The handle can be polled
    /// with [`ReadHandle::try_claim`] or waited on.
    pub fn read_async(&self, hash: &BlockHash, expected: &CacheSignature) -> ReadHandle {
        self.shared.read_async(hash, expected, true)
    }

    /// Reads `hashes` ahead of anyone asking for them, on the reader
    /// threads, and returns without waiting.
    ///
    /// Intended for a prefix chain the moment its hashes are known:
    /// every block the request is about to want, read while the tokens
    /// before it are still being processed. Blocks already staged, in
    /// flight, or in memory are skipped; so is everything past
    /// `prefetch_capacity`, because reading ahead must never be the
    /// thing that exhausts memory.
    pub fn prefetch(&self, hashes: &[BlockHash], expected: &CacheSignature) {
        self.shared.prefetch(hashes, expected);
    }

    /// Drops any staged read-ahead results. A caller that abandons a
    /// request it prefetched for should say so rather than leave the
    /// blocks occupying staging until something else needs the room.
    pub fn clear_prefetch(&self) {
        self.shared
            .staging
            .lock()
            .expect("kv disk staging poisoned")
            .clear();
    }

    /// Adopts the blocks already under `root`, as a restart would.
    pub fn reindex(&self) -> Result<usize, StoreError> {
        self.shared.reindex()
    }

    /// Removes a block from the index, the write buffer, and the disk.
    pub fn remove(&self, hash: &BlockHash) {
        self.shared.quarantine(hash);
    }

    /// True if the index holds an entry for `hash` -- on disk or still
    /// buffered. Says nothing about whether its contents will verify.
    pub fn contains(&self, hash: &BlockHash) -> bool {
        let index = self.shared.index.lock().expect("kv disk index poisoned");
        index.entries.contains_key(hash)
    }

    /// Byte ceiling an operator configured.
    pub fn capacity(&self) -> u64 {
        self.shared.max_bytes
    }

    /// The ceiling eviction is actually working to right now: the
    /// configured one, or less when free disk says so.
    pub fn effective_capacity(&self) -> u64 {
        let used = {
            let index = self.shared.index.lock().expect("kv disk index poisoned");
            index.bytes
        };
        self.shared.effective_capacity(used)
    }

    /// Where a block's file lives (or would). Useful to an operator
    /// tracing one block; the file may not exist yet, or at all.
    pub fn block_path(&self, hash: &BlockHash) -> PathBuf {
        self.shared.block_path(hash)
    }

    pub fn stats(&self) -> DiskStats {
        self.shared.stats()
    }
}

impl Shared {
    fn next_generation(&self) -> u64 {
        self.generation.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// The write path, in the order the plan requires:
    ///
    /// 1. **buffer** -- the payload is reachable before anything claims
    ///    it exists;
    /// 2. **index** -- now it is claimed to exist, and is subject to
    ///    eviction and budget accounting;
    /// 3. **queue** -- only now does the write get scheduled.
    ///
    /// Reversing 1 and 2 lets a reader hit the index for a block with
    /// no file and no payload. Reversing 2 and 3 would be harmless but
    /// pointless: a queued write whose block is not indexed cannot be
    /// evicted or accounted for.
    fn put(&self, hash: BlockHash, block: KvBlock, inline: bool) -> Result<(), StoreError> {
        let bytes = encoded_len(block.signature());
        let block = Arc::new(block);
        let generation = self.next_generation();

        #[cfg(test)]
        let index_first = *self.hooks.order.lock().expect("kv disk hook poisoned")
            == WriteOrder::IndexBeforeBuffer;
        #[cfg(not(test))]
        let index_first = false;

        if index_first {
            self.reserve(hash, bytes, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_put_window, &hash);
            self.buffer_block(hash, generation, Arc::clone(&block));
        } else {
            self.buffer_block(hash, generation, Arc::clone(&block));
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_put_window, &hash);
            self.reserve(hash, bytes, generation);
        }

        let job = WriteJob { hash, generation };
        if !inline && self.queue.try_push(job) {
            self.stats.queued_writes.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        if !inline {
            self.stats.inline_writes.fetch_add(1, Ordering::Relaxed);
        }
        self.run_write(job, block)
    }

    fn buffer_block(&self, hash: BlockHash, generation: u64, block: Arc<KvBlock>) {
        self.buffer
            .lock()
            .expect("kv disk buffer poisoned")
            .insert(hash, Buffered { generation, block });
    }

    /// Drops a buffered payload, but only if it is still the one this
    /// generation put there -- a newer `put` for the same hash owns the
    /// slot now.
    fn release_buffer(&self, hash: &BlockHash, generation: u64) {
        let mut buffer = self.buffer.lock().expect("kv disk buffer poisoned");
        if buffer.get(hash).is_some_and(|b| b.generation == generation) {
            buffer.remove(hash);
        }
    }

    fn buffered(&self, hash: &BlockHash, generation: u64) -> Option<Arc<KvBlock>> {
        let buffer = self.buffer.lock().expect("kv disk buffer poisoned");
        buffer
            .get(hash)
            .filter(|b| b.generation == generation)
            .map(|b| Arc::clone(&b.block))
    }

    /// Runs a queued write. A block whose buffered payload is gone was
    /// evicted or superseded while it waited, and is skipped rather
    /// than resurrected.
    fn run_job(&self, job: WriteJob) {
        match self.buffered(&job.hash, job.generation) {
            Some(block) => {
                let _ = self.run_write(job, block);
            }
            None => {
                self.stats.write_skipped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn run_write(&self, job: WriteJob, block: Arc<KvBlock>) -> Result<(), StoreError> {
        let started = Instant::now();
        let result = self.write_and_publish(&job.hash, &block, job.generation);
        self.stats
            .write_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match result {
            Ok(()) => {
                self.stats.writes.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => {
                self.stats.write_failures.fetch_add(1, Ordering::Relaxed);
                self.abandon(&job.hash, job.generation);
                Err(err)
            }
        }
    }

    /// Reserves (or re-reserves) an index entry, charges its bytes, and
    /// evicts whatever that pushes over budget.
    fn reserve(&self, hash: BlockHash, bytes: u64, generation: u64) {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        let last_used = index.touch();
        index.insert_entry(
            hash,
            Entry {
                bytes,
                last_used,
                published: false,
                generation,
            },
        );
        let victims = self.collect_victims(&mut index, Some(&hash));
        drop(index);
        self.discard(victims);
    }

    /// Drops an entry that will never be published (a failed write).
    /// Index first, then the buffer: an entry that is gone from the
    /// index is unreachable, so no reader can be looking for the
    /// payload we are about to free.
    fn abandon(&self, hash: &BlockHash, generation: u64) {
        {
            let mut index = self.index.lock().expect("kv disk index poisoned");
            if index
                .entries
                .get(hash)
                .is_some_and(|e| e.generation == generation)
            {
                index.remove_entry(hash);
            }
        }
        self.release_buffer(hash, generation);
    }

    fn write_and_publish(
        &self,
        hash: &BlockHash,
        block: &KvBlock,
        generation: u64,
    ) -> Result<(), StoreError> {
        let bytes = encode_block(hash, block);
        let final_path = self.block_path(hash);
        let shard = final_path.parent().expect("block path has a parent");
        fs::create_dir_all(shard).map_err(|e| io_err("create", shard, e))?;
        let tmp_path = self.tmp_path(hash);
        {
            let mut file =
                fs::File::create(&tmp_path).map_err(|e| io_err("create", &tmp_path, e))?;
            #[cfg(test)]
            let written = if self.hooks.fail_with_enospc.load(Ordering::Relaxed) {
                Err(io::Error::from(io::ErrorKind::StorageFull))
            } else {
                file.write_all(&bytes)
            };
            #[cfg(not(test))]
            let written = file.write_all(&bytes);
            if let Err(e) = written {
                let _ = fs::remove_file(&tmp_path);
                self.note_if_enospc(&e);
                return Err(io_err("write", &tmp_path, e));
            }
            // Without this the rename can be durable while the contents
            // are not, which is exactly how a zero-length "published"
            // block file appears after a power loss.
            if let Err(e) = file.sync_all() {
                let _ = fs::remove_file(&tmp_path);
                self.note_if_enospc(&e);
                return Err(io_err("sync", &tmp_path, e));
            }
        }
        fs::rename(&tmp_path, &final_path).map_err(|e| {
            let _ = fs::remove_file(&tmp_path);
            io_err("publish", &final_path, e)
        })?;

        #[cfg(test)]
        Hooks::fire(&self.hooks.after_rename, hash);

        #[cfg(test)]
        let drop_buffer_first = *self.hooks.order.lock().expect("kv disk hook poisoned")
            == WriteOrder::DropBufferBeforeMarking;
        #[cfg(not(test))]
        let drop_buffer_first = false;

        // Mark published, *then* release the buffered payload. In
        // between, a reader either sees "published" and reads the file
        // (which exists) or sees "buffered" and reads the buffer (which
        // still holds it). Releasing first opens a window where neither
        // is true.
        let survived = if drop_buffer_first {
            self.release_buffer(hash, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_publish_window, hash);
            self.mark_published(hash, generation)
        } else {
            let survived = self.mark_published(hash, generation);
            #[cfg(test)]
            Hooks::fire(&self.hooks.in_publish_window, hash);
            self.release_buffer(hash, generation);
            survived
        };

        if !survived {
            // Evicted (or superseded) mid-write. The eviction already
            // released this entry's bytes and could not delete a file
            // that did not exist yet, so the file is ours to withdraw.
            let _ = fs::remove_file(&final_path);
            self.stats
                .write_raced_eviction
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn mark_published(&self, hash: &BlockHash, generation: u64) -> bool {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        match index.entries.get_mut(hash) {
            Some(entry) if entry.generation == generation => {
                if !entry.published {
                    entry.published = true;
                    let bytes = entry.bytes;
                    index.disk_bytes += bytes;
                }
                true
            }
            _ => false,
        }
    }

    /// Where a block's payload currently is, decided under the index
    /// lock. `Ok(None)` is an outright miss.
    fn source(&self, hash: &BlockHash) -> Result<Option<Source>, StoreError> {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        let clock = index.clock + 1;
        let Some(entry) = index.entries.get_mut(hash) else {
            return Ok(None);
        };
        entry.last_used = clock;
        let published = entry.published;
        index.clock = clock;
        if published {
            return Ok(Some(Source::Disk(self.block_path(hash))));
        }
        // The index lock is deliberately still held: "not published"
        // and "here is the buffered payload" must be one decision. Two
        // decisions leave a gap for the writer to publish and release
        // in between.
        let buffered = self
            .buffer
            .lock()
            .expect("kv disk buffer poisoned")
            .get(hash)
            .map(|b| Arc::clone(&b.block));
        match buffered {
            Some(block) => Ok(Some(Source::Buffer(block))),
            None => Err(StoreError::MissingPayload { hash: *hash }),
        }
    }

    /// Begins a read.
    ///
    /// `Ok(None)` in the eventual outcome covers every "you will have
    /// to recompute this" case -- absent, corrupt on disk, or built for
    /// a different config -- because they are the same answer to the
    /// caller. The counters in [`DiskStats`] tell them apart. `Err` is
    /// I/O that failed, or the write-ordering invariant breaking.
    ///
    /// Anything answerable from memory (a miss, or a block still in the
    /// write buffer) is answered here and comes back already ready; no
    /// thread is involved. Only a real file read is dispatched, and a
    /// `demand` read jumps ahead of queued prefetches -- or, if the
    /// queue is full, runs on this thread rather than queueing behind
    /// speculative work.
    fn read_async(
        self: &Arc<Self>,
        hash: &BlockHash,
        expected: &CacheSignature,
        demand: bool,
    ) -> ReadHandle {
        // An in-flight or finished read for the same block and the same
        // expectation is the answer -- never a second read of the same
        // file.
        let staged = {
            let staging = self.staging.lock().expect("kv disk staging poisoned");
            staging
                .get(hash)
                .filter(|slot| &slot.expected == expected)
                .map(Arc::clone)
        };
        if let Some(slot) = staged {
            if demand {
                let counter = if slot.is_ready() {
                    &self.stats.prefetch_hits
                } else {
                    &self.stats.prefetch_waits
                };
                counter.fetch_add(1, Ordering::Relaxed);
            }
            return self.handle(*hash, slot, true);
        }

        let ready = |outcome: ReadOutcome| ReadHandle {
            shared: Arc::clone(self),
            hash: *hash,
            slot: ReadSlot::ready(expected.clone(), outcome),
            staged: false,
        };

        let path = match self.source(hash) {
            Err(err) => return ready(Err(err)),
            Ok(None) => {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                return ready(Ok(None));
            }
            Ok(Some(Source::Buffer(block))) => {
                if block.signature() != expected {
                    self.stats.incompatible.fetch_add(1, Ordering::Relaxed);
                    return ready(Ok(None));
                }
                self.stats.buffer_hits.fetch_add(1, Ordering::Relaxed);
                return ready(Ok(Some(block)));
            }
            Ok(Some(Source::Disk(path))) => path,
        };

        let slot = ReadSlot::pending(expected.clone());
        let dispatched = self.has_readers && {
            self.staging
                .lock()
                .expect("kv disk staging poisoned")
                .insert(*hash, Arc::clone(&slot));
            let job = ReadJob {
                hash: *hash,
                path: path.clone(),
                slot: Arc::clone(&slot),
            };
            let pushed = self.reads.try_push(job, demand);
            if !pushed {
                self.unstage(hash, &slot);
            }
            pushed
        };
        if dispatched {
            return self.handle(*hash, slot, true);
        }
        slot.fulfil(self.read_timed(&path, hash, expected));
        self.handle(*hash, slot, false)
    }

    fn handle(self: &Arc<Self>, hash: BlockHash, slot: Arc<ReadSlot>, staged: bool) -> ReadHandle {
        ReadHandle {
            shared: Arc::clone(self),
            hash,
            slot,
            staged,
        }
    }

    /// Removes a staging entry, but only if it is still the slot the
    /// claimant was holding -- a newer read for the same block owns the
    /// entry now.
    fn unstage(&self, hash: &BlockHash, slot: &Arc<ReadSlot>) {
        let mut staging = self.staging.lock().expect("kv disk staging poisoned");
        if staging.get(hash).is_some_and(|s| Arc::ptr_eq(s, slot)) {
            staging.remove(hash);
        }
    }

    fn prefetch(self: &Arc<Self>, hashes: &[BlockHash], expected: &CacheSignature) {
        if !self.has_readers {
            self.stats
                .prefetch_dropped
                .fetch_add(hashes.len() as u64, Ordering::Relaxed);
            return;
        }
        for hash in hashes {
            let room = {
                let staging = self.staging.lock().expect("kv disk staging poisoned");
                !staging.contains_key(hash) && staging.len() < self.prefetch_capacity
            };
            if !room {
                self.stats.prefetch_dropped.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let handle = self.read_async(hash, expected, false);
            if handle.staged {
                self.stats.prefetch_issued.fetch_add(1, Ordering::Relaxed);
            } else {
                // Answered from memory, or the read queue was full and
                // it ran here. Either way there is nothing staged to
                // claim later, so drop the answer: a prefetch is a
                // hint, and re-reading is cheaper than holding blocks
                // nobody asked for.
                self.stats.prefetch_dropped.fetch_add(1, Ordering::Relaxed);
            }
            // Dropped unclaimed on purpose: a staged slot stays in
            // staging for whoever asks for the block next, which is the
            // entire point of reading it early.
            drop(handle);
        }
    }

    /// The ceiling eviction works to, given how many bytes the store
    /// currently holds.
    ///
    /// `max_bytes` is what an operator asked for; free disk is what the
    /// machine can actually give. The store may occupy what it already
    /// occupies plus whatever the filesystem still has, less a reserve:
    ///
    /// ```text
    /// effective = min(max_bytes, used + free - reserve)
    /// ```
    ///
    /// As free space falls below the reserve the ceiling drops below
    /// `used`, so the next write evicts instead of pushing the
    /// filesystem to `ENOSPC`. That is the whole point: the cache
    /// should give the disk back before the disk takes it back.
    fn effective_capacity(&self, used: u64) -> u64 {
        let Some(free) = self.free_bytes() else {
            return self.max_bytes;
        };
        // `free - reserve` is deliberately signed. Clamping it at zero
        // would make the ceiling equal `used` exactly when the disk is
        // fullest -- the store would sit still and let the filesystem
        // do the refusing, which is the failure this whole budget
        // exists to avoid.
        let headroom = free as i128 - self.reserve_bytes as i128;
        let allowed = (used as i128 + headroom).max(0) as u64;
        self.max_bytes.min(allowed)
    }

    /// Free space, re-measured at most once per TTL. A `statvfs` per
    /// block write would be a syscall on the write path for a number
    /// that moves slowly.
    fn free_bytes(&self) -> Option<u64> {
        let mut cache = self.free_space.lock().expect("kv disk free space poisoned");
        if let Some(checked_at) = cache.checked_at {
            if checked_at.elapsed() < self.free_space_ttl {
                return cache.bytes;
            }
        }
        let bytes = (self.free_space_probe)(&self.root);
        cache.checked_at = Some(Instant::now());
        cache.bytes = bytes;
        bytes
    }

    /// The filesystem said "full". Whatever the cached free-space
    /// reading says, it is wrong *now*, so it is thrown away rather
    /// than left to expire -- and an eviction pass runs immediately, so
    /// the next write has somewhere to go.
    fn note_enospc(&self) {
        self.stats.enospc.fetch_add(1, Ordering::Relaxed);
        {
            let mut cache = self.free_space.lock().expect("kv disk free space poisoned");
            cache.checked_at = None;
            cache.bytes = None;
        }
        let victims = {
            let mut index = self.index.lock().expect("kv disk index poisoned");
            self.collect_victims(&mut index, None)
        };
        self.discard(victims);
    }

    /// Recognises the one I/O error the budget exists to prevent.
    fn note_if_enospc(&self, err: &io::Error) {
        if err.kind() == io::ErrorKind::StorageFull {
            self.note_enospc();
        }
    }

    fn read_timed(&self, path: &Path, hash: &BlockHash, expected: &CacheSignature) -> ReadOutcome {
        let started = Instant::now();
        let outcome = self.read_verified(path, hash, expected);
        self.stats
            .read_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        outcome
    }

    fn read_verified(
        &self,
        path: &Path,
        hash: &BlockHash,
        expected: &CacheSignature,
    ) -> Result<Option<Arc<KvBlock>>, StoreError> {
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // The file vanished under us -- an eviction between the
                // index lookup and the open, or an external cleaner.
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                self.drop_entry(hash);
                return Ok(None);
            }
            Err(e) => return Err(io_err("read", path, e)),
        };
        let decoded = match decode_block(&bytes) {
            Ok(decoded) => decoded,
            Err(_) => {
                self.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                self.quarantine(hash);
                return Ok(None);
            }
        };
        if &decoded.hash != hash {
            // The file under this name is some other block. Treat it
            // exactly like corruption: the name is the identity.
            self.stats.corrupt.fetch_add(1, Ordering::Relaxed);
            self.quarantine(hash);
            return Ok(None);
        }
        match decoded.block.verify(expected) {
            Ok(block) => {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                Ok(Some(Arc::new(block)))
            }
            Err(_) => {
                self.stats.incompatible.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn quarantine(&self, hash: &BlockHash) {
        self.drop_entry(hash);
        self.buffer
            .lock()
            .expect("kv disk buffer poisoned")
            .remove(hash);
        let _ = fs::remove_file(self.block_path(hash));
    }

    fn drop_entry(&self, hash: &BlockHash) {
        let mut index = self.index.lock().expect("kv disk index poisoned");
        index.remove_entry(hash);
    }

    fn reindex(&self) -> Result<usize, StoreError> {
        let tmp = self.root.join(TMP_DIR);
        if let Ok(entries) = fs::read_dir(&tmp) {
            for entry in entries.flatten() {
                let _ = fs::remove_file(entry.path());
            }
        }
        let mut found = Vec::new();
        let shards = fs::read_dir(&self.root).map_err(|e| io_err("read", &self.root, e))?;
        for shard in shards.flatten() {
            if !shard.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if shard.file_name() == TMP_DIR {
                continue;
            }
            let Ok(files) = fs::read_dir(shard.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some(BLOCK_FILE_EXT) {
                    continue;
                }
                let Some(hash) = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(parse_hex_hash)
                else {
                    continue;
                };
                let Ok(meta) = file.metadata() else { continue };
                found.push((hash, meta.len()));
            }
        }
        let mut adopted = 0;
        let mut index = self.index.lock().expect("kv disk index poisoned");
        for (hash, bytes) in found {
            if index.entries.contains_key(&hash) {
                continue;
            }
            let last_used = index.touch();
            index.insert_entry(
                hash,
                Entry {
                    bytes,
                    last_used,
                    published: true,
                    generation: self.next_generation(),
                },
            );
            index.disk_bytes += bytes;
            adopted += 1;
        }
        let victims = self.collect_victims(&mut index, None);
        drop(index);
        self.discard(victims);
        Ok(adopted)
    }

    /// Picks least-recently-used entries until the store fits its
    /// budget, removing them from the index and returning them for
    /// disposal. Index first, payload second -- an entry that is gone
    /// from the index is unreachable, whereas a file deleted while the
    /// index still points at it would be a hit that fails to open.
    fn collect_victims(&self, index: &mut Index, protect: Option<&BlockHash>) -> Vec<Victim> {
        // Computed once per pass, against the pre-eviction size. It has
        // to be: the ceiling is a function of how much the store
        // already occupies, so re-evaluating it as blocks leave would
        // make eviction chase its own tail down to empty.
        let budget = self.effective_capacity(index.disk_bytes);
        if budget < self.max_bytes {
            self.stats.space_clamped.fetch_add(1, Ordering::Relaxed);
        }
        if index.bytes <= budget {
            return Vec::new();
        }
        let mut candidates: Vec<(u64, BlockHash)> = index
            .entries
            .iter()
            .filter(|(hash, _)| Some(*hash) != protect)
            .map(|(hash, entry)| (entry.last_used, *hash))
            .collect();
        candidates.sort_unstable();
        let mut victims = Vec::new();
        for (_, hash) in candidates {
            if index.bytes <= budget {
                break;
            }
            if let Some(entry) = index.remove_entry(&hash) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .evicted_bytes
                    .fetch_add(entry.bytes, Ordering::Relaxed);
                victims.push(Victim {
                    hash,
                    published: entry.published,
                });
            }
        }
        victims
    }

    /// Frees what eviction removed from the index: the buffered payload
    /// and, if it got that far, the file.
    fn discard(&self, victims: Vec<Victim>) {
        if victims.is_empty() {
            return;
        }
        {
            let mut buffer = self.buffer.lock().expect("kv disk buffer poisoned");
            for victim in &victims {
                buffer.remove(&victim.hash);
            }
        }
        for victim in victims {
            if victim.published {
                let _ = fs::remove_file(self.block_path(&victim.hash));
            }
        }
    }

    fn stats(&self) -> DiskStats {
        let index = self.index.lock().expect("kv disk index poisoned");
        let stats = &self.stats;
        DiskStats {
            blocks: index.entries.len(),
            bytes: index.bytes,
            queue_depth: self.queue.depth(),
            writes: stats.writes.load(Ordering::Relaxed),
            queued_writes: stats.queued_writes.load(Ordering::Relaxed),
            inline_writes: stats.inline_writes.load(Ordering::Relaxed),
            write_failures: stats.write_failures.load(Ordering::Relaxed),
            write_skipped: stats.write_skipped.load(Ordering::Relaxed),
            write_raced_eviction: stats.write_raced_eviction.load(Ordering::Relaxed),
            write_nanos: stats.write_nanos.load(Ordering::Relaxed),
            hits: stats.hits.load(Ordering::Relaxed),
            buffer_hits: stats.buffer_hits.load(Ordering::Relaxed),
            misses: stats.misses.load(Ordering::Relaxed),
            corrupt: stats.corrupt.load(Ordering::Relaxed),
            incompatible: stats.incompatible.load(Ordering::Relaxed),
            read_nanos: stats.read_nanos.load(Ordering::Relaxed),
            evictions: stats.evictions.load(Ordering::Relaxed),
            evicted_bytes: stats.evicted_bytes.load(Ordering::Relaxed),
            prefetch_issued: stats.prefetch_issued.load(Ordering::Relaxed),
            prefetch_dropped: stats.prefetch_dropped.load(Ordering::Relaxed),
            prefetch_hits: stats.prefetch_hits.load(Ordering::Relaxed),
            prefetch_waits: stats.prefetch_waits.load(Ordering::Relaxed),
            async_reads: stats.async_reads.load(Ordering::Relaxed),
            staged_blocks: self.staging.lock().expect("kv disk staging poisoned").len(),
            enospc: stats.enospc.load(Ordering::Relaxed),
            space_clamped: stats.space_clamped.load(Ordering::Relaxed),
            effective_capacity: self.effective_capacity(index.disk_bytes),
            disk_bytes: index.disk_bytes,
        }
    }

    fn block_path(&self, hash: &BlockHash) -> PathBuf {
        self.root
            .join(hash.shard_prefix(self.shard_chars))
            .join(format!("{}.{BLOCK_FILE_EXT}", hash.to_hex()))
    }

    fn tmp_path(&self, hash: &BlockHash) -> PathBuf {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        self.root.join(TMP_DIR).join(format!(
            "{}.{}.{n}.tmp",
            hash.shard_prefix(16),
            std::process::id()
        ))
    }
}

/// An entry eviction has already removed from the index, awaiting
/// disposal of its payload.
struct Victim {
    hash: BlockHash,
    published: bool,
}

fn parse_hex_hash(text: &str) -> Option<BlockHash> {
    if text.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        let hi = text.as_bytes()[i * 2] as char;
        let lo = text.as_bytes()[i * 2 + 1] as char;
        *byte = ((hi.to_digit(16)? << 4) | lo.to_digit(16)?) as u8;
    }
    Some(BlockHash::from_bytes(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_block::BlockHasher;
    use std::sync::atomic::AtomicUsize;

    /// A throwaway directory that removes itself, so a failing test
    /// does not leave block files behind. `std::env::temp_dir` plus the
    /// pid and a counter: the workspace has no `tempfile` dependency
    /// and this needs eight lines.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "frink-kvdisk-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("temp dir");
            TempDir(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn layer(n_kv_heads: usize, head_dim: usize, tokens: usize, fill: f32) -> KvCache {
        let mut cache = KvCache::new(n_kv_heads, head_dim);
        for t in 0..tokens {
            let k = vec![fill + t as f32; n_kv_heads * head_dim];
            let v = vec![fill - t as f32; n_kv_heads * head_dim];
            cache.push(&k, &v).expect("unpooled push cannot fail");
        }
        cache
    }

    /// A full-causal layout whose block size is the block's own token
    /// depth -- the default for every test that is not about SWA.
    fn flat(tokens: usize) -> BlockLayout {
        BlockLayout::full_attention(tokens).expect("positive block size")
    }

    fn block(model: &str, n_layers: usize, tokens: usize, fill: f32) -> KvBlock {
        block_with_layout(model, n_layers, tokens, fill, flat(tokens))
    }

    fn block_with_layout(
        model: &str,
        n_layers: usize,
        tokens: usize,
        fill: f32,
        layout: BlockLayout,
    ) -> KvBlock {
        let layers = (0..n_layers)
            .map(|l| layer(2, 4, tokens, fill + l as f32 * 100.0))
            .collect();
        KvBlock::stamp(model, layout, layers).expect("stamp")
    }

    fn expected(model: &str, n_layers: usize, tokens: usize) -> CacheSignature {
        CacheSignature::expected(model, flat(tokens), n_layers, 2, 4, KvDtype::F32, tokens)
    }

    fn hash(n: usize) -> BlockHash {
        BlockHasher::new("model-a", &[] as &[&str]).chain(&[n, n + 1], 2)[0]
    }

    /// A free-space probe that reports a terabyte. Tests must not
    /// depend on how full the machine's real disk happens to be -- that
    /// is exactly the variable the budget tests below control on
    /// purpose.
    fn plenty() -> FreeSpaceProbe {
        Arc::new(|_: &Path| Some(1 << 40))
    }

    /// Default test store: everything on the calling thread, so a test
    /// that does not care about the writer pool never races it.
    fn store(dir: &TempDir, max_bytes: u64) -> DiskKvStore {
        DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_max_bytes(max_bytes)
                .with_writer_threads(0)
                .with_free_space_probe(plenty()),
        )
        .expect("open")
    }

    fn put_now(store: &DiskKvStore, hash: BlockHash, block: KvBlock) {
        store.put_blocking(hash, block).expect("put");
    }

    #[test]
    fn a_block_round_trips_through_a_file() {
        let dir = TempDir::new("roundtrip");
        let store = store(&dir, 1 << 20);
        let h = hash(1);
        let written = block("model-a", 3, 4, 1.0);
        let copy = block("model-a", 3, 4, 1.0);
        put_now(&store, h, written);

        let read = store
            .get(&h, &expected("model-a", 3, 4))
            .expect("get")
            .expect("the block just written must be found");
        assert_eq!(read.layers().len(), 3);
        for (a, b) in read.layers().iter().zip(copy.layers()) {
            assert_eq!(a.k.to_f32_vec(), b.k.to_f32_vec());
            assert_eq!(a.v.to_f32_vec(), b.v.to_f32_vec());
            assert_eq!(a.positions(), b.positions());
        }
        let stats = store.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.blocks, 1);
        assert!(stats.read_nanos > 0, "a read must be timed");
        assert!(stats.write_nanos > 0, "a write must be timed");
    }

    #[test]
    fn the_accounted_size_is_the_real_file_size() {
        let dir = TempDir::new("size");
        let store = store(&dir, 1 << 20);
        let h = hash(2);
        let written = block("model-a", 2, 8, 0.25);
        let predicted = encoded_len(written.signature());
        put_now(&store, h, written);
        let on_disk = fs::metadata(store.block_path(&h)).expect("stat").len();
        assert_eq!(
            predicted, on_disk,
            "the budget charges what the file really costs"
        );
        assert_eq!(store.stats().bytes, on_disk);
    }

    #[test]
    fn blocks_are_sharded_by_hash_prefix() {
        let dir = TempDir::new("shard");
        let store = DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_shard_chars(2)
                .with_writer_threads(0)
                .with_free_space_probe(plenty()),
        )
        .expect("open");
        let h = hash(3);
        put_now(&store, h, block("model-a", 1, 2, 1.0));
        let path = store.block_path(&h);
        assert_eq!(
            path.parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            &h.to_hex()[..2]
        );
        assert!(path.exists());
    }

    /// The crash-safety case, stated as the failure it prevents: a file
    /// cut short must be *refused*, not parsed into whatever the
    /// remaining bytes happen to say.
    #[test]
    fn a_truncated_file_is_refused_at_every_cut_point() {
        let h = hash(4);
        let bytes = encode_block(&h, &block("model-a", 2, 4, 3.0));
        assert!(bytes.len() > PREFIX_LEN + 16);

        // Cut inside the fixed prefix: not even a header to read.
        let err = decode_block(&bytes[..PREFIX_LEN - 1]).expect_err("short file");
        assert_eq!(
            err,
            BlockFormatError::TooShort {
                len: PREFIX_LEN - 1
            }
        );

        // Cut inside the body: the declared length no longer matches.
        for cut in [PREFIX_LEN, PREFIX_LEN + 8, bytes.len() - 4, bytes.len() - 1] {
            let err = decode_block(&bytes[..cut]).expect_err("truncated file");
            assert_eq!(
                err,
                BlockFormatError::Truncated {
                    expected: bytes.len() as u64,
                    actual: cut as u64,
                },
                "a file cut at {cut} must be refused"
            );
        }

        // A file that is the right length but has been altered.
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0xff;
        assert_eq!(
            decode_block(&flipped).expect_err("altered file"),
            BlockFormatError::ChecksumMismatch
        );

        // And a file that is not one of ours at all.
        let mut alien = bytes;
        alien[0] = b'X';
        assert_eq!(
            decode_block(&alien).expect_err("foreign file"),
            BlockFormatError::BadMagic
        );
    }

    /// Truncation through the whole store, not just the decoder: a torn
    /// file is a miss and is removed, so the next request recomputes
    /// instead of tripping over it forever.
    #[test]
    fn a_torn_file_on_disk_is_a_miss_and_is_quarantined() {
        let dir = TempDir::new("torn");
        let store = store(&dir, 1 << 20);
        let h = hash(5);
        put_now(&store, h, block("model-a", 2, 4, 1.0));
        let path = store.block_path(&h);

        // Simulate a write that died half way.
        let full = fs::read(&path).expect("read back");
        fs::write(&path, &full[..full.len() / 2]).expect("truncate");

        let got = store.get(&h, &expected("model-a", 2, 4)).expect("get");
        assert!(got.is_none(), "a torn block must not be returned");
        assert_eq!(store.stats().corrupt, 1);
        assert!(!path.exists(), "a torn block must not be left to trip over");
        assert!(!store.contains(&h));
    }

    #[test]
    fn an_unreadable_format_version_is_refused() {
        let h = hash(6);
        let mut bytes = encode_block(&h, &block("model-a", 1, 2, 1.0));
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        // Re-checksum so the version is the only thing wrong.
        let mut digest = Sha256::new();
        digest.update(&bytes[PREFIX_LEN..]);
        let digest: [u8; 32] = digest.finalize().into();
        bytes[24..PREFIX_LEN].copy_from_slice(&digest);
        assert_eq!(
            decode_block(&bytes).expect_err("unknown version"),
            BlockFormatError::UnsupportedFormat {
                found: 99,
                readable: READABLE_FORMAT_VERSIONS,
            }
        );
    }

    /// The signature discipline is not re-implemented here, but it is
    /// enforced here: a block written under one config must not be
    /// handed to a reader expecting another.
    #[test]
    fn a_block_from_a_different_config_is_a_miss_not_a_hit() {
        let dir = TempDir::new("config");
        let store = store(&dir, 1 << 20);
        let h = hash(7);
        put_now(&store, h, block("model-a", 2, 4, 1.0));

        assert!(store
            .get(&h, &expected("model-b", 2, 4))
            .expect("get")
            .is_none());
        assert!(store
            .get(
                &h,
                &CacheSignature::expected("model-a", flat(4), 2, 8, 4, KvDtype::F32, 4)
            )
            .expect("get")
            .is_none());
        assert_eq!(store.stats().incompatible, 2);
        assert_eq!(store.stats().hits, 0);
        // Still readable by a reader that does match: an incompatible
        // read is not destructive.
        assert!(store
            .get(&h, &expected("model-a", 2, 4))
            .expect("get")
            .is_some());
    }

    /// A file whose name says one block and whose contents say another
    /// is treated as corruption. The name is the identity; a store that
    /// trusted the contents instead would serve a prefix under the
    /// wrong hash, which is the silent-wrong-answer case the hashing
    /// exists to prevent.
    #[test]
    fn a_file_stored_under_the_wrong_name_is_rejected() {
        let dir = TempDir::new("misfiled");
        let store = store(&dir, 1 << 20);
        let (a, b) = (hash(8), hash(9));
        put_now(&store, a, block("model-a", 1, 2, 1.0));
        put_now(&store, b, block("model-a", 1, 2, 2.0));
        // Put b's bytes under a's name.
        let bytes = fs::read(store.block_path(&b)).expect("read b");
        fs::write(store.block_path(&a), bytes).expect("misfile");

        assert!(store
            .get(&a, &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert_eq!(store.stats().corrupt, 1);
    }

    #[test]
    fn eviction_keeps_the_store_inside_its_budget() {
        let dir = TempDir::new("evict");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        // Room for two blocks and change, never three.
        let store = store(&dir, one * 2 + 8);
        let hashes: Vec<BlockHash> = (0..4).map(|i| hash(20 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            put_now(&store, *h, block("model-a", 1, 4, i as f32));
            assert!(
                store.stats().bytes <= store.capacity(),
                "the store must never sit over budget"
            );
        }
        let stats = store.stats();
        assert_eq!(stats.blocks, 2);
        assert_eq!(stats.evictions, 2);
        assert!(stats.evicted_bytes >= one * 2);
        // The two oldest are gone, from the index and from the disk.
        for h in &hashes[..2] {
            assert!(!store.contains(h));
            assert!(
                !store.block_path(h).exists(),
                "an evicted file must be deleted"
            );
        }
        for h in &hashes[2..] {
            assert!(store.contains(h));
        }
    }

    #[test]
    fn a_read_makes_a_block_the_least_likely_eviction_victim() {
        let dir = TempDir::new("lru");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        let store = store(&dir, one * 2 + 8);
        let (a, b, c) = (hash(30), hash(31), hash(32));
        put_now(&store, a, block("model-a", 1, 4, 1.0));
        put_now(&store, b, block("model-a", 1, 4, 2.0));
        // Touch `a`, so `b` is now the oldest.
        assert!(store.get(&a, &expected("model-a", 1, 4)).unwrap().is_some());
        put_now(&store, c, block("model-a", 1, 4, 3.0));

        assert!(store.contains(&a), "a recently read block must survive");
        assert!(!store.contains(&b));
        assert!(store.contains(&c));
    }

    /// The post-rename re-check. The block is evicted in the window
    /// between the rename and the index update -- exactly the window
    /// the re-check exists for. Without it the file stays on disk
    /// forever with nothing accounting for its bytes, and the store
    /// drifts over budget one raced write at a time.
    #[test]
    fn a_block_evicted_mid_write_does_not_leave_its_file_behind() {
        let dir = TempDir::new("raced");
        let store = store(&dir, 1 << 20);
        let h = hash(40);
        {
            let evicting = Arc::clone(&store.shared);
            let mut hook = store
                .shared
                .hooks
                .after_rename
                .lock()
                .expect("hook lock poisoned");
            *hook = Some(Arc::new(move |hash: &BlockHash| {
                // Whoever evicts cannot delete a file that does not
                // exist yet; the writer must notice and withdraw it.
                evicting.drop_entry(hash);
            }));
        }
        put_now(&store, h, block("model-a", 1, 4, 1.0));

        assert!(
            !store.block_path(&h).exists(),
            "a file published for an entry that no longer exists must be withdrawn"
        );
        assert!(!store.contains(&h));
        let stats = store.stats();
        assert_eq!(stats.write_raced_eviction, 1);
        assert_eq!(stats.bytes, 0, "no bytes may be left unaccounted");
    }

    #[test]
    fn no_temp_files_survive_a_successful_write() {
        let dir = TempDir::new("tmp");
        let store = store(&dir, 1 << 20);
        for i in 0..4 {
            put_now(&store, hash(50 + i), block("model-a", 1, 2, i as f32));
        }
        let leftovers: Vec<_> = fs::read_dir(dir.path().join(TMP_DIR))
            .expect("tmp dir")
            .flatten()
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files must not accumulate: {leftovers:?}"
        );
    }

    /// Survival across a restart is the whole point of the tier: a
    /// second store opened on the same directory finds what the first
    /// one wrote, and sweeps away temp files that never published.
    #[test]
    fn a_new_store_reattaches_to_what_the_previous_one_published() {
        let dir = TempDir::new("restart");
        let h = hash(60);
        {
            let store = store(&dir, 1 << 20);
            put_now(&store, h, block("model-a", 2, 4, 7.0));
        }
        // A write that died before publishing.
        let orphan = dir.path().join(TMP_DIR).join("dead.tmp");
        fs::write(&orphan, b"half a block").expect("orphan");

        let reopened = store(&dir, 1 << 20);
        assert!(
            !reopened.contains(&h),
            "reattaching must be an explicit step, not a side effect of open()"
        );
        assert_eq!(reopened.reindex().expect("reindex"), 1);
        assert!(reopened.contains(&h));
        assert!(!orphan.exists(), "an unpublished temp file must be swept");

        let read = reopened
            .get(&h, &expected("model-a", 2, 4))
            .expect("get")
            .expect("a block written before the restart must still be readable");
        assert_eq!(read.tokens(), 4);
    }

    /// `kv-swa-block-alignment`, at the layer that makes it dangerous.
    ///
    /// The disk tier is the thing that carries a block past the death
    /// of the process that computed it, so a window change between two
    /// runs is not a hypothetical: run 1 fills the cache at window 128,
    /// somebody edits the config, run 2 reindexes the same directory
    /// and asks for the same prefix. The tokens match, the hash matches,
    /// the tensors are the right shape -- everything except the mask the
    /// state was computed under. The block must be refused, and counted
    /// as incompatible so an operator can see why their hit rate went
    /// to zero.
    #[test]
    fn a_block_written_under_one_window_is_not_served_to_a_reader_expecting_another() {
        let dir = TempDir::new("swa-window-restart");
        let h = hash(90);
        let window_128 = BlockLayout::new(4, Some(128)).expect("4 divides 128");
        let window_256 = BlockLayout::new(4, Some(256)).expect("4 divides 256");
        {
            let store = store(&dir, 1 << 20);
            put_now(
                &store,
                h,
                block_with_layout("model-a", 2, 4, 7.0, window_128),
            );
        }

        let reopened = store(&dir, 1 << 20);
        assert_eq!(reopened.reindex().expect("reindex"), 1);
        assert!(reopened.contains(&h), "the file is there");

        let after_window_change = reopened
            .get(
                &h,
                &CacheSignature::expected("model-a", window_256, 2, 2, 4, KvDtype::F32, 4),
            )
            .expect("a config change is a miss, not an I/O error");
        assert!(
            after_window_change.is_none(),
            "a block cut against a 128 window must not be handed to a 256-window reader"
        );
        assert_eq!(reopened.stats().incompatible, 1);
        assert_eq!(reopened.stats().hits, 0);

        // Unchanged config still hits: the guard invalidates on change,
        // not on principle.
        let same = reopened
            .get(
                &h,
                &CacheSignature::expected("model-a", window_128, 2, 2, 4, KvDtype::F32, 4),
            )
            .expect("get")
            .expect("the same window must still hit");
        assert_eq!(same.tokens(), 4);
        assert_eq!(same.layout(), window_128);
    }

    /// The window survives the round trip through the file at all --
    /// without this, the test above would pass for the wrong reason
    /// (every decoded block reporting "no window" and every reader
    /// expecting one missing).
    #[test]
    fn the_block_layout_round_trips_through_the_file_format() {
        let sliding = BlockLayout::new(4, Some(512)).expect("4 divides 512");
        let h = hash(91);
        let bytes = encode_block(&h, &block_with_layout("model-a", 2, 4, 1.0, sliding));
        let decoded = decode_block(&bytes).expect("decode");
        let sig = decoded.block.signature.as_ref().expect("signature");
        assert_eq!(sig.layout, sliding);
        assert_eq!(sig.layout.sliding_window(), Some(512));
        assert_eq!(sig.layout.block_size(), 4);

        // And a full-causal block round-trips as full-causal, not as
        // "window 0".
        let bytes = encode_block(&h, &block("model-a", 2, 4, 1.0));
        let decoded = decode_block(&bytes).expect("decode");
        let sig = decoded.block.signature.as_ref().expect("signature");
        assert_eq!(sig.layout.sliding_window(), None);
    }

    /// A file whose header records a block size that does not divide
    /// its window could not have been written by any correct build, so
    /// it is refused at parse time rather than reconstructed into a
    /// layout and checked later.
    #[test]
    fn a_file_recording_a_mis_aligned_layout_is_refused_at_parse_time() {
        let h = hash(92);
        let mut bytes = encode_block(&h, &block("model-a", 2, 4, 1.0));
        // Header sits after the fixed prefix; the window is the 7th u32
        // after the 32-byte hash. Block size is 4, so 6 is not a
        // multiple of it.
        let window_at = PREFIX_LEN + 32 + 4 * 6;
        bytes[window_at..window_at + 4].copy_from_slice(&6u32.to_le_bytes());
        // Re-checksum, so the file fails on its layout rather than on
        // its digest -- the point is that a *valid* file with an
        // impossible layout is still refused.
        let mut digest = Sha256::new();
        digest.update(&bytes[PREFIX_LEN..]);
        let digest: [u8; 32] = digest.finalize().into();
        bytes[24..PREFIX_LEN].copy_from_slice(&digest);

        let err = decode_block(&bytes).expect_err("6 is not a multiple of 4");
        assert!(
            matches!(err, BlockFormatError::BadLayout(_)),
            "expected a layout refusal, got {err}"
        );
    }

    #[test]
    fn reindex_evicts_down_to_the_budget() {
        let dir = TempDir::new("reindex-evict");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        {
            let store = store(&dir, 1 << 20);
            for i in 0..4 {
                put_now(&store, hash(70 + i), block("model-a", 1, 4, i as f32));
            }
        }
        let small = store(&dir, one * 2 + 8);
        small.reindex().expect("reindex");
        let stats = small.stats();
        assert_eq!(stats.blocks, 2, "a shrunken budget must bind on restart");
        assert!(stats.bytes <= small.capacity());
    }

    #[test]
    fn an_absent_block_is_a_plain_miss() {
        let dir = TempDir::new("miss");
        let store = store(&dir, 1 << 20);
        assert!(store
            .get(&hash(80), &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert_eq!(store.stats().misses, 1);
        assert_eq!(store.stats().corrupt, 0);
    }

    /// A hit whose file has been deleted behind the store's back (an
    /// external cleaner, a `rm -rf` on the shard) is a miss, and the
    /// stale entry is dropped rather than left to fail forever.
    #[test]
    fn a_file_deleted_behind_the_stores_back_is_a_miss() {
        let dir = TempDir::new("vanished");
        let store = store(&dir, 1 << 20);
        let h = hash(90);
        put_now(&store, h, block("model-a", 1, 2, 1.0));
        fs::remove_file(store.block_path(&h)).expect("remove");
        assert!(store
            .get(&h, &expected("model-a", 1, 2))
            .expect("get")
            .is_none());
        assert!(!store.contains(&h));
        assert_eq!(store.stats().bytes, 0);
    }

    #[test]
    fn rewriting_a_block_does_not_double_charge_it() {
        let dir = TempDir::new("rewrite");
        let store = store(&dir, 1 << 20);
        let h = hash(100);
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        let once = store.stats().bytes;
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        assert_eq!(store.stats().bytes, once);
        assert_eq!(store.stats().blocks, 1);
    }

    #[test]
    fn hex_names_round_trip() {
        let h = hash(110);
        assert_eq!(parse_hex_hash(&h.to_hex()), Some(h));
        assert_eq!(parse_hex_hash("nothex"), None);
        assert_eq!(parse_hex_hash(&"z".repeat(64)), None);
    }

    // ---------------------------------------------------------------
    // Write ordering: buffer -> index -> queue
    // ---------------------------------------------------------------

    /// Runs one `put` with a reader firing inside the window between
    /// the two ordered steps of the write path, and reports what the
    /// reader saw. Deterministic on purpose: a sleep-and-hope
    /// concurrency test that passes tells you nothing about a window it
    /// may simply have missed.
    ///
    /// Returns `(missing_payload_errors, served)`.
    fn probe_window(order: WriteOrder, publish_window: bool) -> (usize, usize) {
        let dir = TempDir::new("ordering");
        let store = store(&dir, 1 << 20);
        *store.shared.hooks.order.lock().unwrap() = order;

        let violations = Arc::new(AtomicUsize::new(0));
        let served = Arc::new(AtomicUsize::new(0));
        let reader = Arc::clone(&store.shared);
        let v = Arc::clone(&violations);
        let s = Arc::clone(&served);
        let hook: Hook = Arc::new(move |hash: &BlockHash| {
            match reader
                .read_async(hash, &expected("model-a", 1, 4), true)
                .wait()
            {
                Ok(Some(_)) => {
                    s.fetch_add(1, Ordering::Relaxed);
                }
                // Not indexed yet: an honest miss, the reader simply
                // recomputes.
                Ok(None) => {}
                Err(StoreError::MissingPayload { .. }) => {
                    v.fetch_add(1, Ordering::Relaxed);
                }
                Err(other) => panic!("unexpected store error: {other}"),
            }
        });
        let slot = if publish_window {
            &store.shared.hooks.in_publish_window
        } else {
            &store.shared.hooks.in_put_window
        };
        *slot.lock().unwrap() = Some(hook);

        put_now(&store, hash(200), block("model-a", 1, 4, 1.0));
        (
            violations.load(Ordering::Relaxed),
            served.load(Ordering::Relaxed),
        )
    }

    /// The invariant. A reader that looks inside either window -- after
    /// the block is buffered but before it is indexed, and after it is
    /// published but before the buffer is released -- either misses
    /// cleanly or gets the block. It never gets an index hit with
    /// nothing behind it.
    #[test]
    fn a_reader_never_sees_an_index_hit_with_no_payload() {
        let (violations, _) = probe_window(WriteOrder::BufferThenIndex, false);
        assert_eq!(violations, 0, "admission window must be safe");

        let (violations, served) = probe_window(WriteOrder::BufferThenIndex, true);
        assert_eq!(violations, 0, "publication window must be safe");
        assert_eq!(
            served, 1,
            "the reader must actually have reached the block, or this test proves nothing"
        );
    }

    /// The proof that the test above is not vacuous: with the two steps
    /// of admission reversed -- index first, buffer second, which is
    /// the natural way to write it -- the very same reader hits an
    /// index entry for a block with no file and no payload.
    #[test]
    fn indexing_before_buffering_is_caught() {
        let (violations, _) = probe_window(WriteOrder::IndexBeforeBuffer, false);
        assert_eq!(
            violations, 1,
            "index-then-buffer must be detected as an invariant violation"
        );
    }

    /// The other end of the write, and the subtler half: releasing the
    /// buffered payload before marking the file published leaves the
    /// same gap.
    #[test]
    fn releasing_the_buffer_before_publishing_is_caught() {
        let (violations, _) = probe_window(WriteOrder::DropBufferBeforeMarking, true);
        assert_eq!(
            violations, 1,
            "release-then-mark must be detected as an invariant violation"
        );
    }

    /// The same invariant under real concurrency rather than a hook:
    /// writers queueing blocks while readers hammer them. This one can
    /// only ever *sample* the windows, which is why the deterministic
    /// probes above exist -- but it also exercises the writer threads,
    /// the queue, and eviction all at once.
    #[test]
    fn concurrent_readers_never_see_an_index_hit_with_no_payload() {
        let dir = TempDir::new("concurrent");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        let store = Arc::new(
            DiskKvStore::open(
                DiskConfig::new(dir.path())
                    // Tight enough that eviction runs constantly.
                    .with_max_bytes(one * 8)
                    .with_queue_capacity(4)
                    .with_writer_threads(2)
                    .with_free_space_probe(plenty()),
            )
            .expect("open"),
        );
        let hashes: Vec<BlockHash> = (0..16).map(|i| hash(300 + i)).collect();
        let violations = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let readers: Vec<_> = (0..4)
            .map(|_| {
                let store = Arc::clone(&store);
                let hashes = hashes.clone();
                let violations = Arc::clone(&violations);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let want = expected("model-a", 1, 4);
                    while !stop.load(Ordering::Relaxed) {
                        for h in &hashes {
                            match store.get(h, &want) {
                                Ok(_) => {}
                                Err(StoreError::MissingPayload { .. }) => {
                                    violations.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(other) => panic!("unexpected store error: {other}"),
                            }
                        }
                    }
                })
            })
            .collect();

        for round in 0..4 {
            for (i, h) in hashes.iter().enumerate() {
                store
                    .put(*h, block("model-a", 1, 4, (round * 16 + i) as f32))
                    .expect("put");
            }
        }
        store.flush();
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().expect("reader thread");
        }

        assert_eq!(
            violations.load(Ordering::Relaxed),
            0,
            "no reader may ever see an index hit with no payload"
        );
        let stats = store.stats();
        assert!(
            stats.buffer_hits > 0,
            "readers must have caught blocks still in the write buffer, \
             or this test never entered the window"
        );
        assert!(stats.evictions > 0, "the budget must have bound");
        assert!(stats.bytes <= store.capacity());
    }

    /// A block is readable the instant `put` returns, before any writer
    /// thread has touched it. This is what makes the queue safe to use
    /// on the request path.
    #[test]
    fn a_queued_block_is_readable_before_it_reaches_disk() {
        let dir = TempDir::new("buffered");
        // No writer threads: nothing can publish until we flush.
        let store = store(&dir, 1 << 20);
        let h = hash(400);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");

        assert!(
            !store.block_path(&h).exists(),
            "nothing has been written yet"
        );
        let got = store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .expect("a queued block must be readable immediately");
        assert_eq!(got.tokens(), 4);
        assert_eq!(store.stats().buffer_hits, 1);

        store.flush();
        assert!(store.block_path(&h).exists(), "flush must publish it");
        assert!(store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .is_some());
        assert_eq!(store.stats().hits, 1, "and now it comes off the disk");
    }

    /// Backpressure, not loss: when the queue is full the block is
    /// written on the calling thread. Nothing is dropped, and the
    /// fallback is counted so an operator can see a queue that is too
    /// small.
    #[test]
    fn a_full_queue_writes_inline_rather_than_dropping_the_block() {
        let dir = TempDir::new("backpressure");
        let store = DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_queue_capacity(2)
                // Nothing drains the queue, so it stays full.
                .with_writer_threads(0)
                .with_free_space_probe(plenty()),
        )
        .expect("open");

        let hashes: Vec<BlockHash> = (0..5).map(|i| hash(500 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            store
                .put(*h, block("model-a", 1, 4, i as f32))
                .expect("put");
        }
        let stats = store.stats();
        assert_eq!(stats.queued_writes, 2, "the queue holds exactly its cap");
        assert_eq!(stats.inline_writes, 3, "the rest fall back to this thread");
        assert_eq!(stats.writes, 3, "and the fallbacks really wrote");

        // Every block is readable regardless of which path it took --
        // the point of "fall back" instead of "drop".
        let want = expected("model-a", 1, 4);
        for h in &hashes {
            assert!(
                store.get(h, &want).expect("get").is_some(),
                "no block may be lost to a full queue"
            );
        }
        store.flush();
        for h in &hashes {
            assert!(store.block_path(h).exists(), "flush publishes the rest");
        }
    }

    /// A queued write whose block was evicted before a writer reached
    /// it is skipped, not resurrected: eviction has already released
    /// its bytes, so writing it would put the store over budget with a
    /// file nothing accounts for.
    #[test]
    fn a_queued_write_evicted_before_it_runs_is_skipped() {
        let dir = TempDir::new("skipped");
        let store = store(&dir, 1 << 20);
        let h = hash(600);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");
        store.remove(&h);
        store.flush();

        let stats = store.stats();
        assert_eq!(stats.write_skipped, 1);
        assert_eq!(stats.writes, 0);
        assert!(!store.block_path(&h).exists());
        assert_eq!(stats.bytes, 0);
    }

    /// A second `put` for the same hash supersedes the first: the
    /// queued job for the older generation finds a payload that is no
    /// longer its own and skips, rather than overwriting the newer
    /// block with the older one.
    #[test]
    fn a_superseded_queued_write_does_not_overwrite_the_newer_block() {
        let dir = TempDir::new("superseded");
        let store = store(&dir, 1 << 20);
        let h = hash(700);
        store.put(h, block("model-a", 1, 4, 1.0)).expect("put");
        store.put(h, block("model-a", 1, 4, 9.0)).expect("put");
        store.flush();

        let got = store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .expect("hit");
        assert_eq!(
            got.layers()[0].k.as_f32().unwrap()[0],
            9.0,
            "the newer block must win, not whichever write ran last"
        );
        assert_eq!(store.stats().write_skipped, 1);
        assert_eq!(store.stats().blocks, 1);
    }

    // ---------------------------------------------------------------
    // Asynchronous and prefetched reads
    // ---------------------------------------------------------------

    /// A store that reads on its own threads, writes on the caller's
    /// (so a test's writes are done when `put_blocking` returns).
    fn reading_store(dir: &TempDir, readers: usize) -> DiskKvStore {
        DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_writer_threads(0)
                .with_reader_threads(readers)
                .with_free_space_probe(plenty()),
        )
        .expect("open")
    }

    /// Blocks until a prefetch for `hash` has landed in staging.
    fn wait_staged(store: &DiskKvStore, hash: &BlockHash) {
        for _ in 0..2000 {
            let ready = store
                .shared
                .staging
                .lock()
                .unwrap()
                .get(hash)
                .map(|slot| slot.is_ready())
                .unwrap_or(false);
            if ready {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("prefetch never completed");
    }

    /// The point of prefetching, proved the only way it can be: the
    /// block file is **deleted** after the prefetch and before the
    /// request. The request still gets its block, so the read
    /// demonstrably happened before it was asked for -- not on its
    /// thread, not on its clock.
    #[test]
    fn a_prefetched_block_is_already_read_when_the_request_arrives() {
        let dir = TempDir::new("prefetch");
        let store = reading_store(&dir, 1);
        let h = hash(800);
        put_now(&store, h, block("model-a", 2, 4, 5.0));
        let want = expected("model-a", 2, 4);

        store.prefetch(&[h], &want);
        wait_staged(&store, &h);
        fs::remove_file(store.block_path(&h)).expect("remove");

        let got = store
            .get(&h, &want)
            .expect("get")
            .expect("the prefetch already had it");
        assert_eq!(got.tokens(), 4);
        assert_eq!(got.layers()[0].k.as_f32().unwrap()[0], 5.0);

        let stats = store.stats();
        assert_eq!(
            stats.prefetch_hits, 1,
            "the request must have found it ready"
        );
        assert_eq!(
            stats.hits, 1,
            "and the file must have been read exactly once"
        );
        assert_eq!(stats.async_reads, 1, "on a reader thread, not the caller's");
        assert_eq!(stats.staged_blocks, 0, "a claimed read leaves staging");
    }

    /// A whole prefix chain read ahead in one call, which is how a
    /// prefix cache would use this: the request's blocks are known
    /// before the request needs them.
    #[test]
    fn a_whole_chain_can_be_read_ahead_in_one_call() {
        let dir = TempDir::new("chain");
        let store = reading_store(&dir, 2);
        let hashes: Vec<BlockHash> = (0..4).map(|i| hash(810 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            put_now(&store, *h, block("model-a", 1, 4, i as f32));
        }
        let want = expected("model-a", 1, 4);

        store.prefetch(&hashes, &want);
        for h in &hashes {
            wait_staged(&store, h);
            fs::remove_file(store.block_path(h)).expect("remove");
        }
        for (i, h) in hashes.iter().enumerate() {
            let got = store.get(h, &want).expect("get").expect("read ahead");
            assert_eq!(got.layers()[0].k.as_f32().unwrap()[0], i as f32);
        }
        let stats = store.stats();
        assert_eq!(stats.prefetch_issued, 4);
        assert_eq!(stats.prefetch_hits, 4);
        assert_eq!(stats.hits, 4, "four blocks, four reads, none repeated");
    }

    /// Whether or not the prefetch has landed, the request never reads
    /// the same file twice: it joins the read in flight.
    #[test]
    fn a_request_joins_a_read_already_running_rather_than_repeating_it() {
        let dir = TempDir::new("join");
        let store = reading_store(&dir, 1);
        let h = hash(820);
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        let want = expected("model-a", 1, 4);

        store.prefetch(&[h], &want);
        // Deliberately no wait: this races the reader thread, and the
        // assertion holds either way.
        let got = store.get(&h, &want).expect("get").expect("hit");
        assert_eq!(got.tokens(), 4);
        let stats = store.stats();
        assert_eq!(
            stats.prefetch_hits + stats.prefetch_waits,
            1,
            "the request either found the read done or waited for it"
        );
        assert_eq!(stats.hits, 1, "one physical read, whichever way it went");
    }

    /// A staged read belongs to the shape it was issued for. Handing a
    /// prefetch's answer to a request that wants a different layout
    /// would defeat the whole signature discipline, so the staged slot
    /// is only reused on an exact match.
    #[test]
    fn a_staged_read_is_not_reused_by_a_reader_that_wants_another_shape() {
        let dir = TempDir::new("staged-shape");
        let store = reading_store(&dir, 1);
        let h = hash(830);
        put_now(&store, h, block("model-a", 1, 4, 1.0));

        store.prefetch(&[h], &expected("model-a", 1, 4));
        wait_staged(&store, &h);

        let got = store.get(&h, &expected("model-b", 1, 4)).expect("get");
        assert!(got.is_none(), "a different model must not be served");
        assert_eq!(store.stats().incompatible, 1);
        assert_eq!(
            store.stats().prefetch_hits,
            0,
            "the staged answer was for another expectation and must not be claimed"
        );
    }

    /// Reading ahead is a hint and must never be the thing that
    /// exhausts memory: past the staging cap, prefetches are refused
    /// and counted rather than queued.
    #[test]
    fn prefetching_is_bounded() {
        let dir = TempDir::new("prefetch-bound");
        let store = DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_writer_threads(0)
                .with_reader_threads(1)
                .with_prefetch_capacity(2)
                .with_free_space_probe(plenty()),
        )
        .expect("open");
        let hashes: Vec<BlockHash> = (0..6).map(|i| hash(840 + i)).collect();
        for (i, h) in hashes.iter().enumerate() {
            put_now(&store, *h, block("model-a", 1, 4, i as f32));
        }

        store.prefetch(&hashes, &expected("model-a", 1, 4));
        let stats = store.stats();
        assert!(
            stats.staged_blocks <= 2,
            "staging must respect its cap, got {}",
            stats.staged_blocks
        );
        assert!(
            stats.prefetch_dropped >= 4,
            "the refusals must be visible, got {}",
            stats.prefetch_dropped
        );

        // Refusing a hint costs a read, never an answer.
        let want = expected("model-a", 1, 4);
        for (i, h) in hashes.iter().enumerate() {
            let got = store.get(h, &want).expect("get").expect("hit");
            assert_eq!(got.layers()[0].k.as_f32().unwrap()[0], i as f32);
        }
    }

    /// With no reader threads every read happens on the calling thread
    /// and a prefetch is an honest no-op -- counted, not pretended.
    #[test]
    fn without_reader_threads_reads_run_on_the_caller() {
        let dir = TempDir::new("no-readers");
        let store = reading_store(&dir, 0);
        let h = hash(850);
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        let want = expected("model-a", 1, 4);

        store.prefetch(&[h], &want);
        assert_eq!(store.stats().prefetch_dropped, 1);
        assert_eq!(store.stats().staged_blocks, 0);

        assert!(store.get(&h, &want).expect("get").is_some());
        let stats = store.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.async_reads, 0);
    }

    /// The handle is pollable: a caller that has other work to do can
    /// start the read, do the work, and collect it.
    #[test]
    fn a_read_handle_can_be_polled_to_completion() {
        let dir = TempDir::new("handle");
        let store = reading_store(&dir, 1);
        let h = hash(860);
        put_now(&store, h, block("model-a", 1, 4, 2.0));
        let want = expected("model-a", 1, 4);

        let handle = store.read_async(&h, &want);
        for _ in 0..2000 {
            if let Some(outcome) = handle.try_claim() {
                let got = outcome.expect("read").expect("hit");
                assert_eq!(got.layers()[0].k.as_f32().unwrap()[0], 2.0);
                assert_eq!(store.stats().staged_blocks, 0);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("read never completed");
    }

    /// A miss needs no thread at all: it is decided under the index
    /// lock and comes back already answered.
    #[test]
    fn a_miss_is_answered_without_dispatching_a_read() {
        let dir = TempDir::new("ready-miss");
        let store = reading_store(&dir, 1);
        let handle = store.read_async(&hash(870), &expected("model-a", 1, 4));
        assert!(handle.is_ready(), "a miss must not cost a thread hop");
        assert!(handle.wait().expect("read").is_none());
        assert_eq!(store.stats().async_reads, 0);
    }

    /// Abandoning a request drops what was read for it, rather than
    /// leaving the blocks parked in staging.
    #[test]
    fn clearing_the_prefetch_releases_staged_blocks() {
        let dir = TempDir::new("clear");
        let store = reading_store(&dir, 1);
        let h = hash(880);
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        store.prefetch(&[h], &expected("model-a", 1, 4));
        wait_staged(&store, &h);
        assert_eq!(store.stats().staged_blocks, 1);
        store.clear_prefetch();
        assert_eq!(store.stats().staged_blocks, 0);
    }

    // ---------------------------------------------------------------
    // The disk budget
    // ---------------------------------------------------------------

    /// A store whose free-space reading the test controls.
    fn budgeted_store(
        dir: &TempDir,
        max_bytes: u64,
        reserve: u64,
        ttl: std::time::Duration,
        probe: FreeSpaceProbe,
    ) -> DiskKvStore {
        DiskKvStore::open(
            DiskConfig::new(dir.path())
                .with_max_bytes(max_bytes)
                .with_reserve_bytes(reserve)
                .with_free_space_ttl(ttl)
                .with_free_space_probe(probe)
                .with_writer_threads(0)
                .with_reader_threads(0),
        )
        .expect("open")
    }

    /// Bytes really occupying the store's directory. The device model
    /// below is driven by this rather than by the store's own
    /// accounting, so the test cannot pass by agreeing with itself.
    fn dir_bytes(root: &Path) -> u64 {
        let mut total = 0;
        let Ok(entries) = fs::read_dir(root) else {
            return 0;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += dir_bytes(&path);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
        total
    }

    /// The budget the plan asks for: the configured ceiling is an upper
    /// bound, and real free disk can lower it. When the filesystem
    /// fills up under a store that is nowhere near its byte budget,
    /// eviction is what fires -- not `ENOSPC`.
    ///
    /// The probe models a real device: free space is what the modelled
    /// device has left after the store's actual files, so evicting
    /// really does give space back and the ceiling has a fixed point
    /// instead of chasing itself down to empty.
    #[test]
    fn the_ceiling_falls_when_the_filesystem_fills_up() {
        let dir = TempDir::new("budget");
        let one = encoded_len(block("model-a", 1, 4, 1.0).signature());
        let device = Arc::new(AtomicU64::new(1 << 40));
        let probe: FreeSpaceProbe = {
            let device = Arc::clone(&device);
            let root = dir.path().to_path_buf();
            Arc::new(move |_: &Path| {
                Some(
                    device
                        .load(Ordering::Relaxed)
                        .saturating_sub(dir_bytes(&root)),
                )
            })
        };
        // Room for a hundred blocks by byte budget; two blocks' worth
        // of headroom demanded on the device.
        let reserve = one * 2;
        let store = budgeted_store(&dir, one * 100, reserve, std::time::Duration::ZERO, probe);

        for i in 0..4 {
            put_now(&store, hash(900 + i), block("model-a", 1, 4, i as f32));
        }
        assert_eq!(store.stats().blocks, 4);
        assert_eq!(store.stats().evictions, 0, "nothing binds yet");
        assert_eq!(
            store.effective_capacity(),
            store.capacity(),
            "with a terabyte free the configured budget is the ceiling"
        );

        // The device turns out to be small -- or something else on it
        // grew. Six blocks total, of which two must stay free.
        device.store(one * 6, Ordering::Relaxed);
        assert_eq!(
            store.effective_capacity(),
            one * 4,
            "the ceiling must follow the device down to total - reserve"
        );

        for i in 0..6 {
            put_now(&store, hash(910 + i), block("model-a", 1, 4, i as f32));
            let on_disk = dir_bytes(dir.path());
            assert!(
                on_disk + reserve <= one * 6,
                "the store must hand the device its reserve back before the \
                 filesystem has to: {on_disk} bytes used of {}, {reserve} reserved",
                one * 6
            );
        }

        let stats = store.stats();
        assert_eq!(stats.blocks, 4, "settled at total - reserve");
        assert!(stats.evictions >= 6, "got {}", stats.evictions);
        assert!(stats.space_clamped > 0, "the clamp must be visible");
        assert!(
            stats.bytes < one * 100,
            "far under the configured budget it never reached"
        );
        assert_eq!(
            stats.disk_bytes,
            dir_bytes(dir.path()),
            "the store's idea of its disk footprint must be the real one"
        );
    }

    /// `statvfs` is a syscall, and free space moves slowly. It is read
    /// at most once per TTL rather than once per block written.
    #[test]
    fn the_free_space_reading_is_cached_for_its_ttl() {
        let dir = TempDir::new("ttl");
        let calls = Arc::new(AtomicUsize::new(0));
        let probe: FreeSpaceProbe = {
            let calls = Arc::clone(&calls);
            Arc::new(move |_: &Path| {
                calls.fetch_add(1, Ordering::Relaxed);
                Some(1 << 40)
            })
        };
        let store = budgeted_store(&dir, 1 << 20, 0, std::time::Duration::from_secs(60), probe);

        for _ in 0..5 {
            store.effective_capacity();
        }
        for i in 0..3 {
            put_now(&store, hash(920 + i), block("model-a", 1, 4, i as f32));
        }
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "a TTL'd reading must not be re-taken per operation"
        );
    }

    /// The invalidation that matters: at the moment the filesystem says
    /// "full", the cached reading is known to be wrong, so it is thrown
    /// away rather than left to expire. And the store does not keep an
    /// index entry for a block it could not write.
    #[test]
    fn enospc_throws_away_the_cached_free_space() {
        let dir = TempDir::new("enospc");
        let calls = Arc::new(AtomicUsize::new(0));
        let probe: FreeSpaceProbe = {
            let calls = Arc::clone(&calls);
            Arc::new(move |_: &Path| {
                calls.fetch_add(1, Ordering::Relaxed);
                Some(1 << 40)
            })
        };
        let store = budgeted_store(
            &dir,
            1 << 20,
            0,
            // Long enough that nothing but the ENOSPC can re-take it.
            std::time::Duration::from_secs(3600),
            probe,
        );
        let h = hash(930);
        put_now(&store, h, block("model-a", 1, 4, 1.0));
        store.effective_capacity();
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        store
            .shared
            .hooks
            .fail_with_enospc
            .store(true, Ordering::Relaxed);
        let full = hash(931);
        let err = store
            .put_blocking(full, block("model-a", 1, 4, 2.0))
            .expect_err("a full filesystem must be reported, not swallowed");
        assert!(matches!(err, StoreError::Io { .. }), "{err}");

        let stats = store.stats();
        assert_eq!(stats.enospc, 1);
        assert!(
            calls.load(Ordering::Relaxed) > 1,
            "ENOSPC must invalidate the cached reading immediately"
        );
        assert!(
            !store.contains(&full),
            "a block that could not be written must not be indexed"
        );
        assert_eq!(stats.write_failures, 1);
        let leftovers: Vec<_> = fs::read_dir(dir.path().join(TMP_DIR))
            .expect("tmp dir")
            .flatten()
            .collect();
        assert!(
            leftovers.is_empty(),
            "a failed write must clean up after itself: {leftovers:?}"
        );

        // The block written before the failure is untouched.
        assert!(store
            .get(&h, &expected("model-a", 1, 4))
            .expect("get")
            .is_some());

        // And once there is room again, writing resumes.
        store
            .shared
            .hooks
            .fail_with_enospc
            .store(false, Ordering::Relaxed);
        put_now(&store, full, block("model-a", 1, 4, 2.0));
        assert!(store.contains(&full));
    }

    /// A filesystem that cannot be measured is not treated as full, and
    /// not treated as infinite either: the configured budget is simply
    /// the only ceiling.
    #[test]
    fn an_unmeasurable_filesystem_falls_back_to_the_configured_budget() {
        let dir = TempDir::new("unknowable");
        let store = budgeted_store(
            &dir,
            1 << 20,
            1 << 30,
            std::time::Duration::ZERO,
            Arc::new(|_: &Path| None),
        );
        assert_eq!(store.effective_capacity(), 1 << 20);
        for i in 0..3 {
            put_now(&store, hash(940 + i), block("model-a", 1, 4, i as f32));
        }
        assert_eq!(store.stats().blocks, 3);
        assert_eq!(store.stats().evictions, 0);
    }

    /// The real probe, exercised: it is `unsafe` FFI, and a binding
    /// that silently returned nonsense would disable the whole budget
    /// without failing anything else.
    #[test]
    #[cfg(unix)]
    fn the_platform_probe_measures_a_real_filesystem() {
        let dir = TempDir::new("statvfs");
        let free = platform_free_bytes(dir.path()).expect("statvfs on a directory that exists");
        assert!(free > 0, "a writable temp dir with zero bytes free?");
        assert!(
            platform_free_bytes(&dir.path().join("no-such-dir")).is_none(),
            "a path that does not exist cannot report free space"
        );
    }
}
