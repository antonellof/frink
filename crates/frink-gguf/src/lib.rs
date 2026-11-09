//! frink-gguf: a reader for the GGUF file format.
//!
//! GGUF is a public, documented binary format originated by the ggml/llama.cpp
//! project (spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
//! See docs/THIRD_PARTY_NOTICES.md for design-credit details.

pub mod ftype;
pub mod sharded;
pub mod split;
pub mod writer;
pub use sharded::{ShardError, ShardName, ShardedGguf};
pub use writer::{GgufWriteError, GgufWriter, TensorPlan};

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

/// Re-exported so downstream crates can name the type in a
/// [`TensorSource`] implementation without taking their own `memmap2`
/// dependency (which would then have to be kept version-compatible with
/// this one for the trait to line up at all).
pub use memmap2::Mmap as MmapHandle;
use thiserror::Error;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// Upper bound on capacity pre-reserved from an untrusted count in the header,
// so a malformed file cannot trigger a huge allocation before any data is read.
const PREALLOC_CAP: usize = 16 * 1024;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("not a GGUF file (bad magic {0:#010x})")]
    BadMagic(u32),
    #[error("unsupported GGUF version {0}")]
    UnsupportedVersion(u32),
    #[error("unknown metadata value type tag {0}")]
    UnknownValueType(u32),
    #[error("unknown tensor dtype tag {0}")]
    UnknownTensorType(u32),
    #[error("tensor '{0}' not found")]
    TensorNotFound(String),
    #[error("malformed tensor data for '{0}': expected {1} bytes, file has {2}")]
    TruncatedTensor(String, usize, usize),
    #[error(
        "tensor '{0}' has dtype {1:?}, whose block layout this build does not know, so its \
         size cannot be computed. Reading it would hand back an empty slice and look like \
         success"
    )]
    UnsizedTensor(String, GgmlType),
    #[error(
        "tensor '{0}' declares shape {1:?}, whose element count is larger than a {2}-byte \
         file can hold: every element occupies at least one bit on the wire"
    )]
    ImplausibleShape(String, Vec<u64>, usize),
    #[error(
        "general.alignment is {0}: the GGUF spec requires a power of two, and it must be one \
         a file can actually be aligned to"
    )]
    BadAlignment(u64),
}

/// A single scalar/array metadata value from the GGUF key-value header.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            // Some exporters store bool hparams as 0/1 integers.
            GgufValue::U8(v) => Some(*v != 0),
            GgufValue::U32(v) => Some(*v != 0),
            GgufValue::U64(v) => Some(*v != 0),
            GgufValue::I32(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// GGML tensor element type tags (subset actually used by frink today).
/// Numeric tag values verified directly against `ggml/include/ggml.h`'s
/// `enum ggml_type` (the public GGUF/ggml tensor-type tag space), not
/// assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    BF16,
    IQ4NL,
    IQ4XS,
    /// The codebook-grid low-bit formats used throughout the published
    /// "Dynamic" low-bit GGUFs of the large MoE targets (see
    /// docs/MODELS.md). Executable on CPU via `frink-quant`
    /// (scalar + AVX2 kernels, goldens cross-validated against the
    /// real compiled ggml implementation); no NEON or GPU kernels yet.
    IQ1S,
    IQ2XXS,
    IQ3XXS,
    /// The second tier of codebook-grid formats: the ones the published
    /// `UD-*` recipes reach for when `*_XXS` is too lossy. Same grid
    /// machinery, but each stores its sign bits *literally* (a byte per
    /// 8 elements) or in wider grid codes rather than as a 7-bit index
    /// into `KSIGNS_IQ2XS`, which is why they need their own kernels
    /// and their own grids rather than a parameter on the `_XXS` ones.
    /// IQ3_S in particular is not optional coverage: it is what an
    /// `IQ3_M` mix is largely made of.
    IQ2XS,
    IQ2S,
    IQ3S,
    IQ1M,
    /// GGUF block-MXFP4 (ggml tag 39): 17-byte blocks of 32 elements
    /// (1 E8M0 scale byte + 16 nibble bytes). Distinct from the
    /// safetensors two-buffer MXFP4 layout Kimi K3 uses -- same math,
    /// different byte layout. Executable on CPU via `frink-quant`.
    MXFP4,
    /// Plain 32-bit integer tensor (not a quantization format) -- used
    /// by e.g. DeepSeek-V4's token-hash routing tables. Recognized and
    /// sized, no execution path.
    I32,
    /// ggml's ternary quants (tags 34/35), used by the BitNet-style
    /// checkpoints. **Recognized and sized, no execution path**, the
    /// same deal `I32` gets.
    ///
    /// Sizing them is the whole point of naming them. Left as
    /// `Other(34)` they had no block layout, so [`TensorInfo::byte_len`]
    /// returned `None` and every consumer -- `inspect`, the footprint
    /// estimate, `tensor_bytes` -- could say no more than "tag 34". Now
    /// a TQ2_0 file inspects correctly, reports its real size, and stops
    /// at the point of *execution* with `UnsupportedDtype("...", TQ2_0)`.
    ///
    /// A named refusal, not an implementation: writing a ternary
    /// dequantizer without a golden vector to check it against is how
    /// you ship a format that loads and is wrong.
    TQ1_0,
    TQ2_0,
    /// NVFP4 (tag 40): 64-element blocks, four UE4M3 sub-block scales
    /// over 32 nibble bytes of E2M1 values. Recognized and sized, no
    /// execution path -- see [`GgmlType::TQ1_0`].
    NVFP4,
    /// ggml's newest legacy-shaped quants (tags 41/42). Recognized and
    /// sized, no execution path -- see [`GgmlType::TQ1_0`].
    Q1_0,
    Q2_0,
    /// PrismML's private types (tags 142/143, `PrismML-Eng/llama.cpp`
    /// branch `prism`, `ggml-common.h`): `PQ2_0` is Q2_0's 2-bit codec
    /// at group 128, `PTQ1_0` is TQ1_0's trit packing at group 128 with
    /// one f16 scale per 128 (`block_ptq1_0 { qs[24]; qh[2]; half d }`).
    /// Bonsai 2 27B ships in PTQ1_0. Executable on CPU via
    /// `frink-quant::ternary`; PQ2_0 is sized only.
    PQ2_0,
    PTQ1_0,
    Other(u32),
}

/// The tag <-> variant table, written ONCE.
///
/// This used to be a `match tag { .. }` with no inverse. The moment a
/// writer needs `variant -> tag` (see [`writer`]) a second spelling of
/// the same 30 numbers appears, with nothing enforcing that the two
/// agree -- this repo's dominant bug shape. One table scanned in both
/// directions, plus `every_named_type_round_trips_through_its_tag`,
/// makes disagreement impossible rather than unlikely.
///
/// Numeric tag values verified directly against `ggml/include/ggml.h`'s
/// `enum ggml_type` (the public GGUF/ggml tensor-type tag space).
///
/// Tags 31/32/33 (`Q4_0_4_4` and friends) are deliberately absent: ggml
/// REMOVED them from gguf files, so a tag in that range is a malformed
/// or ancient file, not a gap. Tags 36/37/38 (`IQ4_NL_4_4` and friends)
/// likewise.
const GGML_TYPE_TAGS: &[(u32, GgmlType)] = &[
    (0, GgmlType::F32),
    (1, GgmlType::F16),
    (2, GgmlType::Q4_0),
    (3, GgmlType::Q4_1),
    (6, GgmlType::Q5_0),
    (7, GgmlType::Q5_1),
    (8, GgmlType::Q8_0),
    (9, GgmlType::Q8_1),
    (10, GgmlType::Q2K),
    (11, GgmlType::Q3K),
    (12, GgmlType::Q4K),
    (13, GgmlType::Q5K),
    (14, GgmlType::Q6K),
    (16, GgmlType::IQ2XXS),
    (17, GgmlType::IQ2XS),
    (18, GgmlType::IQ3XXS),
    (19, GgmlType::IQ1S),
    (20, GgmlType::IQ4NL),
    (21, GgmlType::IQ3S),
    (22, GgmlType::IQ2S),
    (23, GgmlType::IQ4XS),
    (26, GgmlType::I32),
    (29, GgmlType::IQ1M),
    (30, GgmlType::BF16),
    (34, GgmlType::TQ1_0),
    (35, GgmlType::TQ2_0),
    (39, GgmlType::MXFP4),
    (40, GgmlType::NVFP4),
    (41, GgmlType::Q1_0),
    (42, GgmlType::Q2_0),
    (142, GgmlType::PQ2_0),
    (143, GgmlType::PTQ1_0),
];

impl GgmlType {
    fn from_tag(tag: u32) -> Self {
        match GGML_TYPE_TAGS.iter().find(|(t, _)| *t == tag) {
            Some((_, ty)) => *ty,
            None => GgmlType::Other(tag),
        }
    }

    /// The wire tag for this type, for a GGUF *writer* (see [`writer`]).
    ///
    /// `Other(tag)` hands back its own tag, so a tensor of a type this
    /// build does not name survives a read-then-write as itself rather
    /// than silently becoming something else. Whether such a tensor can
    /// be copied at all is [`TensorInfo::byte_len`]'s decision, not
    /// this one.
    pub fn to_tag(&self) -> u32 {
        match self {
            GgmlType::Other(tag) => *tag,
            named => GGML_TYPE_TAGS
                .iter()
                .find(|(_, ty)| ty == named)
                .map(|(t, _)| *t)
                // NOT a gate -- it cannot fire today, and
                // `every_named_type_round_trips_through_its_tag` is
                // what keeps it that way, by walking the enum against
                // the table. It is the total-function fallback Rust
                // requires, and `u32::MAX` is chosen so that if a
                // future variant is ever added without a table row the
                // file is refused by a reader rather than read as F32
                // (tag 0), which is what a `0` here would produce.
                .unwrap_or(u32::MAX),
        }
    }

    /// Bytes per contiguous quantization block, and elements per block.
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (4, 1),
            GgmlType::F16 => (2, 1),
            GgmlType::BF16 => (2, 1),
            GgmlType::Q4_0 => (18, 32), // 1x f16 scale + 16 bytes of nibbles
            GgmlType::Q4_1 => (20, 32), // 1x f16 scale + 1x f16 min + 16 bytes of nibbles
            GgmlType::Q5_0 => (22, 32), // 1x f16 scale + 4 bytes of 5th-bit + 16 bytes of nibbles
            GgmlType::Q5_1 => (24, 32), // 1x f16 scale + 1x f16 min + 4 bytes of 5th-bit + 16 bytes of nibbles
            GgmlType::Q8_0 => (34, 32), // 1x f16 scale + 32x i8
            GgmlType::Q8_1 => (36, 32), // 1x f16 scale + 1x f16 sum + 32x i8
            GgmlType::Q2K => (84, 256), // super-block layout (approximate; see frink-quant)
            GgmlType::Q3K => (110, 256),
            GgmlType::Q4K => (144, 256),
            GgmlType::Q5K => (176, 256),
            GgmlType::Q6K => (210, 256),
            GgmlType::IQ4NL => (18, 32), // 1x f16 scale + 16 bytes of 4-bit codebook indices
            GgmlType::IQ4XS => (136, 256), // 1x f16 scale + split 6-bit sub-scales + 128 bytes of indices
            // Block sizes for the low-bit IQ formats are cross-checked
            // against each format's published bits-per-weight figure,
            // which the layout must reproduce exactly:
            //   IQ1_S    50 bytes/256 elems * 8 = 1.5625 bpw
            //   IQ1_M    56 bytes/256 elems * 8 = 1.75   bpw
            //   IQ2_XXS  66 bytes/256 elems * 8 = 2.0625 bpw
            //   IQ2_XS   74 bytes/256 elems * 8 = 2.3125 bpw
            //   IQ2_S    82 bytes/256 elems * 8 = 2.5625 bpw
            //   IQ3_XXS  98 bytes/256 elems * 8 = 3.0625 bpw
            //   IQ3_S   110 bytes/256 elems * 8 = 3.4375 bpw
            GgmlType::IQ1S => (50, 256), // 1x f16 scale + 32 bytes grid indices + 16 bytes qh
            GgmlType::IQ2XXS => (66, 256), // 1x f16 scale + 64 bytes of u16 grid/sign codes
            GgmlType::IQ3XXS => (98, 256), // 1x f16 scale + 96 bytes of grid/sign codes
            GgmlType::IQ2XS => (74, 256), // 1x f16 scale + 64 bytes of u16 grid/sign codes + 8 scale bytes
            GgmlType::IQ2S => (82, 256),  // 1x f16 scale + 32 grid + 32 sign + 8 qh + 8 scale bytes
            GgmlType::IQ3S => (110, 256), // 1x f16 scale + 64 grid + 8 qh + 32 sign + 4 scale bytes
            // IQ1_M is the one IQ format with *no* f16 scale field: the
            // block scale is reassembled from the four scale words' top
            // nibbles (see `frink-quant`), so all 56 bytes are payload.
            GgmlType::IQ1M => (56, 256), // 32 grid bytes + 16 qh + 8 scale bytes
            GgmlType::MXFP4 => (17, 32), // 1x E8M0 scale byte + 16 nibble bytes
            GgmlType::I32 => (4, 1),
            // Recognized-and-sized, no execution path. Layouts read off
            // `ggml-common.h`'s block structs, not the docs:
            //   block_tq1_0 { qs[(256-4*256/64)/5=48]; qh[4]; half d }
            //   block_tq2_0 { qs[256/4=64]; half d }
            //   block_nvfp4 { d[64/16=4]; qs[64/2=32] }      (QK_NVFP4 = 64)
            //   block_q1_0  { half d; qs[128/8=16] }         (QK1_0   = 128)
            //   block_q2_0  { half d; qs[64/4=16] }          (QK2_0   = 64)
            GgmlType::TQ1_0 => (54, 256),
            //   block_pq2_0  { half d; qs[128/4=32] }        (QK_PQ2_0  = 128)
            //   block_ptq1_0 { qs[24]; qh[2]; half d }       (QK_PTQ1_0 = 128)
            GgmlType::PQ2_0 => (34, 128),
            GgmlType::PTQ1_0 => (28, 128),
            GgmlType::TQ2_0 => (66, 256),
            GgmlType::NVFP4 => (36, 64),
            GgmlType::Q1_0 => (18, 128),
            GgmlType::Q2_0 => (18, 64),
            GgmlType::Other(_) => (0, 1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64, // relative to start of tensor-data section
}

impl TensorInfo {
    /// The declared element count, or `None` when folding the
    /// file-supplied dims wraps or does not fit an address on this
    /// machine.
    ///
    /// This is the ONLY place the shape is folded. It used to be a bare
    /// `product()`, which made every downstream shape-vs-bytes check
    /// agree with a lie: dims `[2^32, 2^32]` wrap to exactly 0, so the
    /// tensor sized as 0 bytes and `tensor_bytes` returned `Ok(&[])` --
    /// the very outcome [`GgufError::UnsizedTensor`] exists to prevent,
    /// reached through a door it does not cover. `[2^63 + 2, 2]` wraps
    /// to 4 and walked straight through `Tensor::new`'s own product
    /// assert.
    pub fn element_count(&self) -> Option<usize> {
        let mut n: u64 = 1;
        for &dim in &self.shape {
            n = n.checked_mul(dim)?;
        }
        usize::try_from(n).ok()
    }

    /// The element count for REPORTING (a parameter-count sum, a log
    /// line) -- never for sizing a read. [`GgufFile::parse`] refuses any
    /// shape `element_count` cannot represent, so for a `TensorInfo` that
    /// came out of a parsed file this is exact; the saturating arm exists
    /// only for a hand-built one, and saturates UP, so it can still only
    /// cause a refusal, never an over-read.
    pub fn n_elements(&self) -> usize {
        self.element_count().unwrap_or(usize::MAX)
    }

    /// Size on disk, or `None` for a dtype whose block layout this
    /// build does not know.
    ///
    /// `None` rather than `0`: a zero here used to flow silently into
    /// every size estimate and into `tensor_bytes`, which then returned
    /// an EMPTY SLICE with no error. A TQ2_0 file did not fail to parse,
    /// it under-counted its own footprint and then failed later
    /// somewhere far less informative.
    pub fn byte_len(&self) -> Option<usize> {
        let (block_bytes, block_elems) = self.dtype.block_layout();
        if block_bytes == 0 {
            return None;
        }
        let n = self.element_count()?;
        (n / block_elems).checked_mul(block_bytes)
    }
}

/// A parsed GGUF file: metadata header plus tensor descriptors, backed by an
/// mmap so multi-hundred-gigabyte weight files are never fully copied into
/// process memory (same read-only mmap strategy llama.cpp uses).
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    mmap: Arc<Mmap>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let file_len = mmap.len();
        let mut cursor = io::Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let kv_count = cursor.read_u64::<LittleEndian>()?;

        // Counts come off the wire; cap the reserved capacity so a tiny header
        // can't request a multi-gigabyte allocation. The collections still grow
        // to fit a file that genuinely contains this many items.
        let mut metadata = HashMap::with_capacity((kv_count as usize).min(PREALLOC_CAP));
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity((tensor_count as usize).min(PREALLOC_CAP));
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity((n_dims as usize).min(PREALLOC_CAP));
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            let dtype_tag = cursor.read_u32::<LittleEndian>()?;
            let offset = cursor.read_u64::<LittleEndian>()?;
            let info = TensorInfo {
                name,
                shape,
                dtype: GgmlType::from_tag(dtype_tag),
                offset,
            };
            // The bound is derived from the input, not chosen: every
            // element occupies at least one bit on the wire (the densest
            // format this build knows is Q1_0, at 1.125 bpw), so a
            // tensor cannot declare more elements than eight times the
            // bytes of the file that has to contain it. Refusing HERE,
            // at the only place a `GgufFile`'s `TensorInfo`s are built,
            // is what lets `n_elements` and `byte_len` be trusted by the
            // consumers downstream that each had their own copy of the
            // arithmetic.
            let max_elements = file_len as u128 * 8;
            if info
                .element_count()
                .is_none_or(|n| n as u128 > max_elements)
            {
                return Err(GgufError::ImplausibleShape(info.name, info.shape, file_len));
            }
            tensors.push(info);
        }

        // Tensor data begins at the next `general.alignment` boundary
        // (default 32) after the header. This matches the GGUF spec.
        // `general.alignment` comes off the wire and is used as a
        // DIVISOR. Zero divided by zero, and `is_power_of_two()` is
        // false for zero, so one predicate refuses both 0 and the
        // spec-violating non-powers like 3. The panic this replaces
        // defeated `admin.rs`'s deliberate
        // `let Ok(file) = GgufFile::open(..) else { .. }` degradation
        // arm, 500-ing the whole model listing over one malformed file.
        let declared_alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let alignment = usize::try_from(declared_alignment)
            .ok()
            .filter(|a| a.is_power_of_two())
            .ok_or(GgufError::BadAlignment(declared_alignment))?;
        let pos = cursor.position() as usize;
        let data_start = pos
            .div_ceil(alignment)
            .checked_mul(alignment)
            .ok_or(GgufError::BadAlignment(declared_alignment))?;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            mmap: Arc::new(mmap),
            data_start,
        })
    }

    /// The single computation of a tensor's byte range in this file.
    ///
    /// It used to be three lines written out verbatim TWICE, once in
    /// each accessor below, with wrapping `usize` arithmetic: a tensor
    /// `offset` near `u64::MAX` wrapped `start` past the end of the
    /// mmap, then wrapped `start + len` back down to something small, so
    /// the `TruncatedTensor` refusal did not fire and the mmap was
    /// sliced with `start > end`. Two copies is two places to forget,
    /// and the twin forgot differently: it did not slice, so it returned
    /// a bogus `Range` and the panic landed in whichever consumer sliced
    /// it, far from the cause.
    fn tensor_range(&self, name: &str) -> Result<std::ops::Range<usize>, GgufError> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let len = info
            .byte_len()
            .ok_or_else(|| GgufError::UnsizedTensor(name.to_string(), info.dtype))?;
        let start = usize::try_from(info.offset)
            .ok()
            .and_then(|off| self.data_start.checked_add(off));
        // `available` is for the message, not for the decision, so it
        // may saturate. The decision below is `checked_add` only.
        let available = start.map_or(0, |s| self.mmap.len().saturating_sub(s));
        start
            .and_then(|s| s.checked_add(len).map(|end| s..end))
            .filter(|r| r.end <= self.mmap.len())
            .ok_or_else(|| GgufError::TruncatedTensor(name.to_string(), len, available))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        Ok(&self.mmap[self.tensor_range(name)?])
    }

    /// Zero-copy accessor: returns a cheaply-cloneable handle to the
    /// underlying mmap plus the byte range for `name`, instead of
    /// copying the tensor's bytes into a new heap allocation the way
    /// `tensor_bytes` + `.to_vec()` would. Reading blocks directly
    /// against the mmap rather than materializing a full in-memory
    /// copy means a loaded checkpoint's resident memory is the mmap
    /// itself, not a second copy of it.
    pub fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        Ok((Arc::clone(&self.mmap), self.tensor_range(name)?))
    }

    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }
}

/// One logical GGUF namespace, single-file or split: everything a model
/// loader or tokenizer needs to read metadata and tensor bytes without
/// caring how many physical files back them. Implemented by both
/// [`GgufFile`] (one file) and [`sharded::ShardedGguf`] (a validated
/// shard set); consumers written against this trait accept either.
pub trait TensorSource {
    fn metadata(&self, key: &str) -> Option<&GgufValue>;
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo>;
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError>;
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError>;

    fn metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata(key).and_then(|v| v.as_u64())
    }
    fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata(key).and_then(|v| v.as_str())
    }
    fn metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata(key).and_then(|v| v.as_f32())
    }
    fn metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata(key).and_then(|v| v.as_bool())
    }
}

impl TensorSource for GgufFile {
    fn metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }
    fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        GgufFile::find_tensor(self, name)
    }
    fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        GgufFile::tensor_bytes(self, name)
    }
    fn tensor_mapped_range(
        &self,
        name: &str,
    ) -> Result<(Arc<Mmap>, std::ops::Range<usize>), GgufError> {
        GgufFile::tensor_mapped_range(self, name)
    }
}

fn read_gguf_string(cursor: &mut io::Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    // A string cannot be longer than the bytes remaining in the file; reject an
    // oversized length before allocating so a bogus header can't request gigabytes.
    let remaining = cursor
        .get_ref()
        .len()
        .saturating_sub(cursor.position() as usize);
    if len > remaining {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
    }
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_gguf_value(cursor: &mut io::Cursor<&[u8]>) -> Result<GgufValue, GgufError> {
    let type_tag = cursor.read_u32::<LittleEndian>()?;
    read_gguf_value_typed(cursor, type_tag, 0)
}

/// Maximum nesting depth for GGUF metadata arrays. Real files nest one level
/// (an array of strings); this bounds the native recursion in
/// `read_gguf_value_typed` so a crafted file of nested arrays cannot overflow
/// the stack.
const MAX_VALUE_DEPTH: u32 = 8;

fn read_gguf_value_typed(
    cursor: &mut io::Cursor<&[u8]>,
    type_tag: u32,
    depth: u32,
) -> Result<GgufValue, GgufError> {
    Ok(match type_tag {
        0 => GgufValue::U8(cursor.read_u8()?),
        1 => GgufValue::I8(cursor.read_i8()?),
        2 => GgufValue::U16(cursor.read_u16::<LittleEndian>()?),
        3 => GgufValue::I16(cursor.read_i16::<LittleEndian>()?),
        4 => GgufValue::U32(cursor.read_u32::<LittleEndian>()?),
        5 => GgufValue::I32(cursor.read_i32::<LittleEndian>()?),
        6 => GgufValue::F32(cursor.read_f32::<LittleEndian>()?),
        7 => GgufValue::Bool(cursor.read_u8()? != 0),
        8 => GgufValue::String(read_gguf_string(cursor)?),
        9 => {
            if depth >= MAX_VALUE_DEPTH {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF array nesting too deep",
                )
                .into());
            }
            let elem_tag = cursor.read_u32::<LittleEndian>()?;
            let len = cursor.read_u64::<LittleEndian>()? as usize;
            // Every array element occupies at least one byte on the wire, so a
            // length larger than the bytes remaining is impossible; reject it
            // before it can size an allocation (the same invariant as
            // read_gguf_string).
            let remaining = cursor
                .get_ref()
                .len()
                .saturating_sub(cursor.position() as usize);
            if len > remaining {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
            }
            let mut items = Vec::with_capacity(len.min(PREALLOC_CAP));
            for _ in 0..len {
                items.push(read_gguf_value_typed(cursor, elem_tag, depth + 1)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(cursor.read_u64::<LittleEndian>()?),
        11 => GgufValue::I64(cursor.read_i64::<LittleEndian>()?),
        12 => GgufValue::F64(cursor.read_f64::<LittleEndian>()?),
        other => return Err(GgufError::UnknownValueType(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;
    use std::io::Write;

    /// Walks the enum, one variant per arm, and hands back the next.
    ///
    /// Its only job is to make the compiler enumerate `GgmlType`: the
    /// match has no `_` arm, so adding a variant stops this file
    /// compiling until the variant is threaded into the chain, and
    /// `every_named_type_round_trips_through_its_tag` then fails until
    /// it is also in `GGML_TYPE_TAGS`. A plain list of variants in the
    /// test would be a second table nothing forces anyone to update --
    /// which is the exact defect the one-table refactor removed.
    fn next_variant(t: GgmlType) -> Option<GgmlType> {
        Some(match t {
            GgmlType::F32 => GgmlType::F16,
            GgmlType::F16 => GgmlType::BF16,
            GgmlType::BF16 => GgmlType::Q4_0,
            GgmlType::Q4_0 => GgmlType::Q4_1,
            GgmlType::Q4_1 => GgmlType::Q5_0,
            GgmlType::Q5_0 => GgmlType::Q5_1,
            GgmlType::Q5_1 => GgmlType::Q8_0,
            GgmlType::Q8_0 => GgmlType::Q8_1,
            GgmlType::Q8_1 => GgmlType::Q2K,
            GgmlType::Q2K => GgmlType::Q3K,
            GgmlType::Q3K => GgmlType::Q4K,
            GgmlType::Q4K => GgmlType::Q5K,
            GgmlType::Q5K => GgmlType::Q6K,
            GgmlType::Q6K => GgmlType::IQ4NL,
            GgmlType::IQ4NL => GgmlType::IQ4XS,
            GgmlType::IQ4XS => GgmlType::IQ1S,
            GgmlType::IQ1S => GgmlType::IQ1M,
            GgmlType::IQ1M => GgmlType::IQ2XXS,
            GgmlType::IQ2XXS => GgmlType::IQ2XS,
            GgmlType::IQ2XS => GgmlType::IQ2S,
            GgmlType::IQ2S => GgmlType::IQ3XXS,
            GgmlType::IQ3XXS => GgmlType::IQ3S,
            GgmlType::IQ3S => GgmlType::MXFP4,
            GgmlType::MXFP4 => GgmlType::I32,
            GgmlType::I32 => GgmlType::TQ1_0,
            GgmlType::TQ1_0 => GgmlType::TQ2_0,
            GgmlType::TQ2_0 => GgmlType::NVFP4,
            GgmlType::NVFP4 => GgmlType::Q1_0,
            GgmlType::Q1_0 => GgmlType::Q2_0,
            GgmlType::Q2_0 => GgmlType::PQ2_0,
            GgmlType::PQ2_0 => GgmlType::PTQ1_0,
            GgmlType::PTQ1_0 => return None,
            GgmlType::Other(_) => return None,
        })
    }

    /// `from_tag` and `to_tag` read one table, and every named variant
    /// is in it. The writer added in `writer.rs` needs the inverse of a
    /// mapping that used to exist only one way; two hand-written
    /// tables agreeing about thirty numbers is this repo's dominant bug
    /// shape, so this walks the enum and the table against each other.
    #[test]
    fn every_named_type_round_trips_through_its_tag() {
        let mut ty = Some(GgmlType::F32);
        let mut seen = 0usize;
        while let Some(t) = ty {
            let tag = t.to_tag();
            assert_ne!(tag, u32::MAX, "{t:?} has no row in GGML_TYPE_TAGS");
            assert_eq!(GgmlType::from_tag(tag), t, "tag {tag} does not map back");
            seen += 1;
            ty = next_variant(t);
        }
        assert_eq!(
            seen,
            GGML_TYPE_TAGS.len(),
            "the enum and GGML_TYPE_TAGS disagree about how many named types there are"
        );
    }

    /// A tag this build does not name survives a read-then-write as
    /// itself. Writing it as F32 (tag 0) would turn an unknown tensor
    /// into a wrong one silently; ggml has already added tags this
    /// build predates, so the arm is reachable, not theoretical.
    #[test]
    fn an_unknown_tag_round_trips_as_itself() {
        let unknown = GgmlType::from_tag(9999);
        assert_eq!(unknown, GgmlType::Other(9999));
        assert_eq!(unknown.to_tag(), 9999);
    }

    /// Builds a tiny synthetic GGUF byte buffer in memory (no real model
    /// weights involved) so the parser can be exercised without any
    /// external file or network access.
    fn build_synthetic_gguf() -> Vec<u8> {
        build_synthetic_gguf_with_dtype(0)
    }

    /// The synthetic file, parameterised by every field a hostile
    /// header varies rather than copied per case. One byte layout, so a
    /// malformed-header case cannot drift from the well-formed file the
    /// rest of these tests parse -- which is the same rule the parser
    /// itself now follows for the tensor range.
    struct Synthetic {
        dtype_tag: u32,
        alignment: u32,
        shape: Vec<u64>,
        offset: u64,
    }

    impl Default for Synthetic {
        fn default() -> Self {
            Self {
                dtype_tag: 0, // F32
                alignment: 32,
                shape: vec![4, 8],
                offset: 0,
            }
        }
    }

    impl Synthetic {
        fn bytes(&self) -> Vec<u8> {
            let mut buf = Vec::new();
            buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
            buf.write_u32::<LittleEndian>(3).unwrap(); // version
            buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
            buf.write_u64::<LittleEndian>(2).unwrap(); // kv_count

            // kv 1: general.alignment (u32)
            write_string(&mut buf, "general.alignment");
            buf.write_u32::<LittleEndian>(4).unwrap(); // type = u32
            buf.write_u32::<LittleEndian>(self.alignment).unwrap();

            // kv 2: general.name = "synthetic-test"
            write_string(&mut buf, "general.name");
            buf.write_u32::<LittleEndian>(8).unwrap(); // type = string
            write_string(&mut buf, "synthetic-test");

            // tensor 0: "tok_embd.weight"
            write_string(&mut buf, "tok_embd.weight");
            buf.write_u32::<LittleEndian>(self.shape.len() as u32)
                .unwrap();
            for &dim in &self.shape {
                buf.write_u64::<LittleEndian>(dim).unwrap();
            }
            buf.write_u32::<LittleEndian>(self.dtype_tag).unwrap();
            buf.write_u64::<LittleEndian>(self.offset).unwrap();

            // Pad to the declared alignment where that is a boundary a
            // file can actually have; a hostile alignment gets the
            // default padding, because what those cases assert is the
            // refusal, not where the data landed.
            let pad = if self.alignment.is_power_of_two() && self.alignment <= 4096 {
                self.alignment as usize
            } else {
                32
            };
            while buf.len() % pad != 0 {
                buf.push(0);
            }
            // tensor data: 32 f32 values
            for i in 0..32u32 {
                buf.write_f32::<LittleEndian>(i as f32 * 0.5).unwrap();
            }
            buf
        }
    }

    /// The synthetic file, parameterised by the tensor's dtype tag
    /// rather than copied per dtype: the byte layout is identical and a
    /// second copy would drift from this one.
    fn build_synthetic_gguf_with_dtype(dtype_tag: u32) -> Vec<u8> {
        Synthetic {
            dtype_tag,
            ..Default::default()
        }
        .bytes()
    }

    /// Writes `bytes` to a uniquely named temp file, opens it, and
    /// removes it. The returned `GgufFile` keeps its own mmap, so the
    /// unlink is safe.
    fn open_temp(bytes: &[u8], tag: &str) -> Result<GgufFile, GgufError> {
        let tmp =
            std::env::temp_dir().join(format!("frink_test_{tag}_{}.gguf", std::process::id()));
        std::fs::write(&tmp, bytes).unwrap();
        let res = GgufFile::open(&tmp);
        std::fs::remove_file(&tmp).ok();
        res
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    #[test]
    fn parses_header_and_metadata() {
        let bytes = build_synthetic_gguf();
        let tmp =
            std::env::temp_dir().join(format!("frink_test_synthetic_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();
        let f = GgufFile::open(&tmp).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.metadata_str("general.name"), Some("synthetic-test"));
        assert_eq!(f.metadata_u64("general.alignment"), Some(32));
        assert_eq!(f.tensors.len(), 1);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn reads_tensor_bytes_correctly() {
        let bytes = build_synthetic_gguf();
        let tmp =
            std::env::temp_dir().join(format!("frink_test_synthetic2_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();
        let f = GgufFile::open(&tmp).unwrap();
        let info = f.find_tensor("tok_embd.weight").unwrap();
        assert_eq!(info.shape, vec![4, 8]);
        assert_eq!(info.n_elements(), 32);
        let raw = f.tensor_bytes("tok_embd.weight").unwrap();
        assert_eq!(raw.len(), 32 * 4);
        let first = f32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(first, 0.0);
        let second = f32::from_le_bytes(raw[4..8].try_into().unwrap());
        assert_eq!(second, 0.5);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn rejects_oversized_counts_without_huge_allocation() {
        // A tiny header must not be able to request a multi-gigabyte allocation
        // via an untrusted kv_count / tensor_count / string length. Each of
        // these headers is 24-40 bytes but declares billions of entries; the
        // parser must return an error rather than pre-allocating (or panicking).
        let cases: [(u64, u64, Option<u64>); 3] = [
            // (tensor_count, kv_count, oversized_string_len)
            (0, 500_000_000, None),
            (500_000_000, 0, None),
            (0, 1, Some(u64::MAX / 2)),
        ];
        for (i, (tensor_count, kv_count, str_len)) in cases.into_iter().enumerate() {
            let mut buf = Vec::new();
            buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
            buf.write_u32::<LittleEndian>(3).unwrap();
            buf.write_u64::<LittleEndian>(tensor_count).unwrap();
            buf.write_u64::<LittleEndian>(kv_count).unwrap();
            if let Some(len) = str_len {
                // start of a KV entry: an oversized key string length
                buf.write_u64::<LittleEndian>(len).unwrap();
            }
            let tmp = std::env::temp_dir().join(format!(
                "frink_test_dos_{}_{}.gguf",
                std::process::id(),
                i
            ));
            std::fs::write(&tmp, &buf).unwrap();
            let res = GgufFile::open(&tmp);
            std::fs::remove_file(&tmp).ok();
            assert!(res.is_err(), "case {i} should error, not parse/allocate");
        }
    }

    #[test]
    fn recognized_low_bit_types_match_their_published_bits_per_weight() {
        // Each low-bit IQ block size must reproduce that format's
        // published bits-per-weight exactly -- an arithmetic
        // cross-check independent of this file's own table, and the
        // cheapest way to catch a transposed digit in a block size
        // (which would otherwise silently mis-stride a whole tensor).
        for (ty, tag, bpw_x16) in [
            (GgmlType::IQ1S, 19u32, 25u32), // 1.5625 bpw * 16
            (GgmlType::IQ1M, 29, 28),       // 1.75   bpw * 16
            (GgmlType::IQ2XXS, 16, 33),     // 2.0625 bpw * 16
            (GgmlType::IQ2XS, 17, 37),      // 2.3125 bpw * 16
            (GgmlType::IQ2S, 22, 41),       // 2.5625 bpw * 16
            (GgmlType::IQ3XXS, 18, 49),     // 3.0625 bpw * 16
            (GgmlType::IQ3S, 21, 55),       // 3.4375 bpw * 16
        ] {
            assert_eq!(GgmlType::from_tag(tag), ty);
            let (bytes, elems) = ty.block_layout();
            assert_eq!(
                bytes * 8 * 16,
                bpw_x16 as usize * elems,
                "{ty:?} block layout does not reproduce its published bits-per-weight"
            );
        }
        assert_eq!(GgmlType::from_tag(26), GgmlType::I32);
        assert_eq!(GgmlType::I32.block_layout(), (4, 1));
        // An unknown tag still degrades to Other, never a wrong name.
        assert_eq!(GgmlType::from_tag(999), GgmlType::Other(999));
    }

    /// The live ggml tags this build cannot *execute* are still
    /// recognized and sized, so a checkpoint using one inspects
    /// correctly and refuses by name at execution instead of being
    /// mis-measured at parse time.
    ///
    /// Block sizes are cross-checked against each format's published
    /// bits-per-weight, the same arithmetic the IQ tiers get: a
    /// transposed digit in a block size would otherwise mis-stride a
    /// whole tensor silently.
    #[test]
    fn live_ggml_tags_without_a_kernel_are_still_named_and_sized() {
        // (type, tag, bytes-per-block * 16 expressed as bpw * 16)
        for (ty, tag, bpw_x16) in [
            (GgmlType::TQ1_0, 34u32, 27u32), // 1.6875 bpw * 16
            (GgmlType::TQ2_0, 35, 33),       // 2.0625 bpw * 16
            (GgmlType::NVFP4, 40, 72),       // 4.5    bpw * 16
            (GgmlType::Q1_0, 41, 18),        // 1.125  bpw * 16
            (GgmlType::Q2_0, 42, 36),        // 2.25   bpw * 16
            (GgmlType::PQ2_0, 142, 34),      // 2.125  bpw * 16 (Prism, group 128)
            (GgmlType::PTQ1_0, 143, 28),     // 1.75   bpw * 16 (Prism, group 128)
        ] {
            assert_eq!(GgmlType::from_tag(tag), ty);
            let (bytes, elems) = ty.block_layout();
            assert_ne!(bytes, 0, "{ty:?} must be sized, not zero-sized");
            assert_eq!(
                bytes * 8 * 16,
                bpw_x16 as usize * elems,
                "{ty:?} block layout does not reproduce its published bits-per-weight"
            );
            // Sized means `byte_len` answers, which is what makes the
            // eventual refusal an execution-time one that names a
            // format rather than a parse-time zero.
            let info = TensorInfo {
                name: "blk.0.ffn_down.weight".to_string(),
                shape: vec![elems as u64, 4],
                dtype: ty,
                offset: 0,
            };
            assert_eq!(info.byte_len(), Some(bytes * 4));
        }

        // Tags 31/32/33 and 36/37/38 were REMOVED from gguf files by
        // ggml. They are not a gap, and adding them would be inventing
        // support for a format no current file can contain.
        for removed in [31u32, 32, 33, 36, 37, 38] {
            assert_eq!(GgmlType::from_tag(removed), GgmlType::Other(removed));
        }
    }

    /// A dtype with no known block layout must REFUSE, naming the
    /// tensor and the tag -- never hand back a zero-length slice that
    /// reads as a successful load of an empty tensor.
    ///
    /// This is the whole reason `byte_len` returns `Option`: with
    /// `Other(_) => (0, 1)` flowing through unguarded, `tensor_bytes`
    /// returned `Ok(&[])` and the failure surfaced hundreds of lines
    /// later as a shape mismatch that named nothing.
    #[test]
    fn an_unsized_dtype_refuses_by_name_rather_than_returning_an_empty_slice() {
        // 250: past GGML_TYPE_COUNT, so it is unknown to any build.
        let bytes = build_synthetic_gguf_with_dtype(250);
        let tmp = std::env::temp_dir().join(format!(
            "frink_test_unsized_{}_{}.gguf",
            std::process::id(),
            250
        ));
        std::fs::write(&tmp, &bytes).unwrap();
        let f = GgufFile::open(&tmp).unwrap();
        let info = f.find_tensor("tok_embd.weight").unwrap();
        assert_eq!(info.dtype, GgmlType::Other(250));
        assert_eq!(info.byte_len(), None);

        match f.tensor_bytes("tok_embd.weight") {
            Err(GgufError::UnsizedTensor(name, GgmlType::Other(250))) => {
                assert_eq!(name, "tok_embd.weight");
            }
            Err(other) => panic!("expected UnsizedTensor, got {other}"),
            Ok(slice) => panic!(
                "tensor_bytes returned Ok({} bytes) for a dtype with no block layout",
                slice.len()
            ),
        }
        // The zero-copy accessor is the same hole through a different
        // door; it must refuse identically.
        match f.tensor_mapped_range("tok_embd.weight") {
            Err(GgufError::UnsizedTensor(_, GgmlType::Other(250))) => {}
            Err(other) => panic!("expected UnsizedTensor, got {other}"),
            Ok((_, range)) => panic!("tensor_mapped_range returned Ok({range:?})"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// A file-supplied shape must be REFUSED when its element count
    /// cannot be what it claims, never folded with a wrapping
    /// `product()` that makes every later shape-vs-bytes check agree
    /// with the lie.
    ///
    /// What shipped: `[2^32, 2^32]` wrapped to exactly 0, so the tensor
    /// sized as 0 bytes and `tensor_bytes` handed back `Ok(&[])` -- a
    /// load that reads as success and yields an empty tensor, which is
    /// the outcome `UnsizedTensor` exists to prevent, reached through a
    /// door it does not cover. `[2^63 + 2, 2]` wrapped to 4, walked
    /// through `Tensor::new`'s own product assert, and the first
    /// `row(0)` then indexed `data[0..2^63 + 2]` on a 4-element `Vec`.
    #[test]
    fn a_shape_larger_than_the_file_could_hold_is_refused_rather_than_wrapped() {
        // Chosen to fail rather than to be round. The first two wrap the
        // u64 fold (to 0 and to 4); the third and fourth wrap nothing
        // and are caught only by the bound derived from the input --
        // one bit per element, so this ~288-byte file cannot hold more
        // than 2304 of them.
        for (i, shape) in [
            vec![1u64 << 32, 1 << 32],
            vec![(1u64 << 63) + 2, 2],
            vec![u64::MAX / 64, 65],
            vec![1u64 << 40],
        ]
        .into_iter()
        .enumerate()
        {
            let bytes = Synthetic {
                shape: shape.clone(),
                ..Default::default()
            }
            .bytes();
            match open_temp(&bytes, &format!("wrapshape{i}")) {
                Err(GgufError::ImplausibleShape(name, got, _)) => {
                    assert_eq!(name, "tok_embd.weight");
                    assert_eq!(got, shape);
                }
                Err(other) => panic!("expected ImplausibleShape for {shape:?}, got {other}"),
                Ok(f) => panic!(
                    "shape {shape:?} parsed: n_elements={} byte_len={:?}",
                    f.tensors[0].n_elements(),
                    f.tensors[0].byte_len()
                ),
            }
        }
    }

    /// A tensor `offset` near `u64::MAX` must be refused, by BOTH
    /// accessors, rather than wrapping past the `TruncatedTensor` check
    /// and slicing the mmap with `start > end`.
    #[test]
    fn a_tensor_offset_that_wraps_the_mmap_is_refused_by_both_accessors() {
        // The offsets below are chosen against this exact file: 288
        // bytes, `data_start` 160, a 128-byte F32 tensor. If the builder
        // changes, they stop probing what they were chosen to probe, so
        // the drift has to be the thing that fails.
        let base = Synthetic::default().bytes();
        assert_eq!(
            base.len(),
            288,
            "the offsets below were chosen against a 288-byte file"
        );

        // With `start = data_start + offset` wrapping:
        //   MAX - 232: `start` wrapped to 2^64 - 73 and `start + len`
        //              wrapped back down to 55, under the file length,
        //              so the refusal did not fire and the mmap was
        //              sliced with start > end. That is the panic.
        //   MAX -  98: `start` wrapped to 61 and `start + len` to 189,
        //              both inside the file: no panic at all, just 128
        //              bytes of the HEADER served as tensor data.
        //   MAX:       `start` 159, `end` 287 -- same silent misread.
        for (i, offset) in [u64::MAX - 232, u64::MAX - 98, u64::MAX]
            .into_iter()
            .enumerate()
        {
            let bytes = Synthetic {
                offset,
                ..Default::default()
            }
            .bytes();
            let f =
                open_temp(&bytes, &format!("wrapoffset{i}")).expect("the header itself is fine");
            match f.tensor_bytes("tok_embd.weight") {
                Err(GgufError::TruncatedTensor(name, len, _)) => {
                    assert_eq!(name, "tok_embd.weight");
                    assert_eq!(len, 128);
                }
                Err(other) => panic!("offset {offset}: expected TruncatedTensor, got {other}"),
                Ok(b) => panic!("offset {offset}: tensor_bytes returned {} bytes", b.len()),
            }
            // The zero-copy accessor was a verbatim copy of the same
            // three lines and forgot differently: it returned a bogus
            // `Range` and let the panic land in whichever consumer
            // sliced it. It cannot differ now -- there is one copy.
            match f.tensor_mapped_range("tok_embd.weight") {
                Err(GgufError::TruncatedTensor(..)) => {}
                Err(other) => panic!("offset {offset}: expected TruncatedTensor, got {other}"),
                Ok((_, r)) => panic!("offset {offset}: tensor_mapped_range returned {r:?}"),
            }
        }
    }

    /// `general.alignment` is a divisor read straight off the wire. Zero
    /// divided by zero inside `parse`, and a panic there defeats
    /// `admin.rs`'s deliberate `let Ok(file) = GgufFile::open(..) else`
    /// degradation arm: one malformed file 500'd the whole model
    /// listing instead of degrading its own row.
    #[test]
    fn a_zero_or_non_power_of_two_alignment_is_refused_rather_than_panicking() {
        for (i, alignment) in [0u32, 3, 33, u32::MAX].into_iter().enumerate() {
            let bytes = Synthetic {
                alignment,
                ..Default::default()
            }
            .bytes();
            match open_temp(&bytes, &format!("align{i}")) {
                Err(GgufError::BadAlignment(got)) => assert_eq!(got, u64::from(alignment)),
                Err(other) => panic!("alignment {alignment}: expected BadAlignment, got {other}"),
                Ok(_) => panic!("alignment {alignment} parsed"),
            }
        }
        // A power of two other than the default still parses, so the
        // refusal is not "anything unusual is refused".
        let bytes = Synthetic {
            alignment: 64,
            ..Default::default()
        }
        .bytes();
        let f = open_temp(&bytes, "align_ok").expect("alignment 64 is a valid alignment");
        assert_eq!(f.tensor_bytes("tok_embd.weight").unwrap().len(), 128);
    }

    #[test]
    fn rejects_bad_magic() {
        let tmp = std::env::temp_dir().join(format!("frink_test_bad_{}.gguf", std::process::id()));
        std::fs::write(&tmp, b"NOPE0000").unwrap();
        match GgufFile::open(&tmp) {
            Err(GgufError::BadMagic(_)) => {}
            Err(other) => panic!("expected BadMagic error, got a different error: {other}"),
            Ok(_) => panic!("expected BadMagic error, got Ok"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// A metadata array whose declared length exceeds the bytes that could
    /// possibly follow must be rejected before it sizes an allocation, rather
    /// than aborting the process with a multi-exabyte allocation request.
    fn header_with_single_kv(key: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(0).unwrap(); // tensor_count
        buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
        write_string(&mut buf, key);
        buf
    }

    #[test]
    fn rejects_array_length_larger_than_the_file() {
        let mut buf = header_with_single_kv("a");
        buf.write_u32::<LittleEndian>(9).unwrap(); // value type = array
        buf.write_u32::<LittleEndian>(4).unwrap(); // element type = u32
        buf.write_u64::<LittleEndian>(u64::MAX / 64).unwrap(); // impossible length

        let tmp = std::env::temp_dir().join(format!("frink_test_arr_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &buf).unwrap();
        match GgufFile::open(&tmp) {
            Err(_) => {}
            Ok(_) => panic!("expected an error for an oversized array length"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn rejects_deeply_nested_arrays() {
        // Each level is an array whose only element is another array. Without a
        // depth limit this recurses one stack frame per level and overflows.
        let mut buf = header_with_single_kv("a");
        buf.write_u32::<LittleEndian>(9).unwrap(); // outer value type = array
        for _ in 0..100_000 {
            buf.write_u32::<LittleEndian>(9).unwrap(); // element type = array
            buf.write_u64::<LittleEndian>(1).unwrap(); // length 1
        }
        buf.write_u32::<LittleEndian>(0).unwrap(); // innermost element type = u8
        buf.write_u64::<LittleEndian>(1).unwrap();
        buf.push(0);

        let tmp =
            std::env::temp_dir().join(format!("frink_test_nested_{}.gguf", std::process::id()));
        std::fs::write(&tmp, &buf).unwrap();
        match GgufFile::open(&tmp) {
            Err(_) => {}
            Ok(_) => panic!("expected an error for deeply nested arrays"),
        }
        std::fs::remove_file(&tmp).ok();
    }

    /// Not a correctness test: a one-off inspection tool for dumping
    /// every real tensor name/shape/dtype from a real downloaded GGUF
    /// shard, gated behind an env var so it never runs in CI (needs a
    /// real multi-GB file on disk). Used to verify
    /// `kimi_gguf_loader`'s tensor-name assumptions against a real
    /// downloaded `unsloth/Kimi-K3-GGUF` payload shard.
    #[test]
    #[ignore = "prints real tensor names from a real GGUF file at FRINK_TEST_INSPECT_PATH; not a correctness assertion"]
    fn dump_real_gguf_tensor_table() {
        let path = std::env::var("FRINK_TEST_INSPECT_PATH")
            .expect("set FRINK_TEST_INSPECT_PATH to a real .gguf file path");
        let file = GgufFile::open(&path).expect("real file must parse");
        println!(
            "architecture: {:?}",
            file.metadata_str("general.architecture")
        );
        println!("tensor_count: {}", file.tensors.len());
        for t in &file.tensors {
            println!("{:<45} shape={:?} dtype={:?}", t.name, t.shape, t.dtype);
        }
    }
}
