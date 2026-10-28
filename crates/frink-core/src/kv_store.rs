//! The element type of the host KV store, and the one storage type every
//! host cache keeps its K and V rows in.
//!
//! llama.cpp keeps K and V at **f16** unless told otherwise:
//! `llama_context_default_params` sets `type_k` and `type_v` to
//! `GGML_TYPE_F16` (`llama-context.cpp:3672-3673`), and `-ctk f32` is the
//! opt-out. This store was `Vec<f32>` for every host cache, so a CPU
//! deployment held twice the reference's bytes per token and reached half
//! its context on the same budget (`docs/plans/parity-audit-2026-09-19.md`
//! §2.4). [`KvBuf`] is the storage and [`KvRows`] the borrowed view the
//! attention kernels read; both carry their dtype, so a reader cannot
//! take f16 bits for f32 ones.
//!
//! The arithmetic stays f32. A row is converted as it is read
//! ([`KvRows::read`]) and as it is written ([`KvBuf::extend_from_f32`]),
//! which is what llama.cpp's own CPU path does with an f16 cache: the
//! rounding is in the STORE, at write time, and every reader sees the
//! same rounded value. The libllama goldens this repo matches were made
//! under that default (`tools/llama_logits.c` takes
//! `llama_context_default_params`), so an f16 store moves frink toward
//! them rather than away.

use half::f16;
use half::slice::HalfFloatSliceExt;
use std::ops::Range;

/// What one element of a host K/V row is stored as.
///
/// No `Default`: the library builds f32 caches (every libllama golden in
/// this repo was made with `type_k = GGML_TYPE_F32`,
/// `scripts/gptoss_reference_logits.cpp`), the binaries choose f16 as
/// llama.cpp's do, and `frink_models::kv_budget::host_kv_dtype` is the
/// one place that decides which. A `Default` would be a second answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvDtype {
    /// Four bytes. What every host cache was before f16 existed, and
    /// llama.cpp's `-ctk f32`.
    F32,
    /// Two bytes, rounded at write. llama.cpp's default.
    F16,
}

impl KvDtype {
    /// Bytes per stored element.
    pub const fn elem_bytes(self) -> usize {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
        }
    }

    /// The `--ctk` spelling this dtype answers to.
    pub const fn name(self) -> &'static str {
        match self {
            KvDtype::F32 => "f32",
            KvDtype::F16 => "f16",
        }
    }
}

/// One growable buffer of K (or V) elements, f32 or f16.
#[derive(Debug, Clone, PartialEq)]
pub enum KvBuf {
    F32(Vec<f32>),
    F16(Vec<f16>),
}

impl KvBuf {
    pub fn new(dtype: KvDtype) -> Self {
        Self::with_capacity(dtype, 0)
    }

    pub fn with_capacity(dtype: KvDtype, elems: usize) -> Self {
        match dtype {
            KvDtype::F32 => KvBuf::F32(Vec::with_capacity(elems)),
            KvDtype::F16 => KvBuf::F16(Vec::with_capacity(elems)),
        }
    }

    /// `elems` zeros.
    pub fn zeroed(dtype: KvDtype, elems: usize) -> Self {
        match dtype {
            KvDtype::F32 => KvBuf::F32(vec![0.0; elems]),
            KvDtype::F16 => KvBuf::F16(vec![f16::ZERO; elems]),
        }
    }

    /// A buffer holding `src`, rounded to `dtype`.
    pub fn from_f32(dtype: KvDtype, src: &[f32]) -> Self {
        let mut buf = Self::with_capacity(dtype, src.len());
        buf.extend_from_f32(src);
        buf
    }

    pub fn dtype(&self) -> KvDtype {
        match self {
            KvBuf::F32(_) => KvDtype::F32,
            KvBuf::F16(_) => KvDtype::F16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            KvBuf::F32(v) => v.len(),
            KvBuf::F16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Capacity in ELEMENTS; [`Self::allocated_bytes`] is the bytes.
    pub fn capacity(&self) -> usize {
        match self {
            KvBuf::F32(v) => v.capacity(),
            KvBuf::F16(v) => v.capacity(),
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.capacity() * self.dtype().elem_bytes()
    }

    /// Appends `src`, rounding to this buffer's dtype.
    pub fn extend_from_f32(&mut self, src: &[f32]) {
        match self {
            KvBuf::F32(v) => v.extend_from_slice(src),
            KvBuf::F16(v) => {
                let start = v.len();
                v.resize(start + src.len(), f16::ZERO);
                v[start..].convert_from_f32_slice(src);
            }
        }
    }

    /// Appends `rows`, bit for bit when the dtypes agree and converting
    /// when they do not.
    pub fn extend_from_rows(&mut self, rows: KvRows<'_>) {
        match (self, rows) {
            (KvBuf::F32(v), KvRows::F32(r)) => v.extend_from_slice(r),
            (KvBuf::F16(v), KvRows::F16(r)) => v.extend_from_slice(r),
            (buf, rows) => buf.extend_from_f32(&rows.to_f32_vec()),
        }
    }

    /// Overwrites `rows.len()` elements starting at `start`, bit for bit
    /// when the dtypes agree.
    pub fn write_rows(&mut self, start: usize, rows: KvRows<'_>) {
        let end = start + rows.len();
        match (self, rows) {
            (KvBuf::F32(v), KvRows::F32(r)) => v[start..end].copy_from_slice(r),
            (KvBuf::F16(v), KvRows::F16(r)) => v[start..end].copy_from_slice(r),
            (buf, rows) => buf.write_f32(start, &rows.to_f32_vec()),
        }
    }

    /// Overwrites `src.len()` elements starting at `start`.
    pub fn write_f32(&mut self, start: usize, src: &[f32]) {
        let end = start + src.len();
        match self {
            KvBuf::F32(v) => v[start..end].copy_from_slice(src),
            KvBuf::F16(v) => v[start..end].convert_from_f32_slice(src),
        }
    }

    pub fn truncate(&mut self, elems: usize) {
        match self {
            KvBuf::F32(v) => v.truncate(elems),
            KvBuf::F16(v) => v.truncate(elems),
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Drops the first `elems` elements, moving the rest down.
    pub fn drain_front(&mut self, elems: usize) {
        match self {
            KvBuf::F32(v) => {
                v.drain(..elems);
            }
            KvBuf::F16(v) => {
                v.drain(..elems);
            }
        }
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        match self {
            KvBuf::F32(v) => v.reserve_exact(additional),
            KvBuf::F16(v) => v.reserve_exact(additional),
        }
    }

    pub fn shrink_to(&mut self, elems: usize) {
        match self {
            KvBuf::F32(v) => v.shrink_to(elems),
            KvBuf::F16(v) => v.shrink_to(elems),
        }
    }

    pub fn copy_within(&mut self, src: Range<usize>, dst: usize) {
        match self {
            KvBuf::F32(v) => v.copy_within(src, dst),
            KvBuf::F16(v) => v.copy_within(src, dst),
        }
    }

    /// The whole buffer as a view.
    pub fn view(&self) -> KvRows<'_> {
        match self {
            KvBuf::F32(v) => KvRows::F32(v),
            KvBuf::F16(v) => KvRows::F16(v),
        }
    }

    /// The whole buffer widened to f32. For cold consumers (a slot file,
    /// a device upload, a test); the hot readers go through
    /// [`KvRows::read`] a row at a time.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.view().to_f32_vec()
    }

    /// The buffer as f32, when that is what it holds.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            KvBuf::F32(v) => Some(v),
            KvBuf::F16(_) => None,
        }
    }
}

/// A borrowed run of K (or V) elements, f32 or f16.
///
/// What every host attention kernel takes for its cache operands.
/// `From<&[f32]>` keeps a caller holding plain f32 rows (a batched
/// prefill's in-flight K, a test) exactly as it was.
#[derive(Debug, Clone, Copy)]
pub enum KvRows<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

/// Equal when the dtypes agree and every element does, bit-pattern
/// comparison aside (f32 `==`, so NaN is unequal as everywhere else).
/// Two runs at different dtypes are never equal: an f16 store and an f32
/// one holding "the same" rows are two different stores.
impl PartialEq for KvRows<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (KvRows::F32(a), KvRows::F32(b)) => a == b,
            (KvRows::F16(a), KvRows::F16(b)) => a == b,
            _ => false,
        }
    }
}

impl<'a> From<&'a [f32]> for KvRows<'a> {
    fn from(s: &'a [f32]) -> Self {
        KvRows::F32(s)
    }
}

impl<'a> From<&'a Vec<f32>> for KvRows<'a> {
    fn from(s: &'a Vec<f32>) -> Self {
        KvRows::F32(s)
    }
}

impl<'a, const N: usize> From<&'a [f32; N]> for KvRows<'a> {
    fn from(s: &'a [f32; N]) -> Self {
        KvRows::F32(s)
    }
}

impl<'a> From<&'a [f16]> for KvRows<'a> {
    fn from(s: &'a [f16]) -> Self {
        KvRows::F16(s)
    }
}

impl<'a> From<&'a KvBuf> for KvRows<'a> {
    fn from(b: &'a KvBuf) -> Self {
        b.view()
    }
}

impl<'a> KvRows<'a> {
    pub fn len(&self) -> usize {
        match self {
            KvRows::F32(s) => s.len(),
            KvRows::F16(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> KvDtype {
        match self {
            KvRows::F32(_) => KvDtype::F32,
            KvRows::F16(_) => KvDtype::F16,
        }
    }

    /// The run as f32, when that is what it holds.
    pub fn as_f32(self) -> Option<&'a [f32]> {
        match self {
            KvRows::F32(s) => Some(s),
            KvRows::F16(_) => None,
        }
    }

    /// A sub-range, still borrowed from the same store.
    pub fn slice(self, range: Range<usize>) -> KvRows<'a> {
        match self {
            KvRows::F32(s) => KvRows::F32(&s[range]),
            KvRows::F16(s) => KvRows::F16(&s[range]),
        }
    }

    /// `range` as f32: borrowed when the store is f32, converted into
    /// `scratch` when it is f16. The hot readers call this once per key
    /// row with a scratch they own for the whole loop, so an f32 store
    /// pays one match and an f16 store one SIMD widen per row.
    #[inline]
    pub fn read<'s>(self, range: Range<usize>, scratch: &'s mut Vec<f32>) -> &'s [f32]
    where
        'a: 's,
    {
        match self {
            KvRows::F32(s) => &s[range],
            KvRows::F16(s) => {
                let src = &s[range];
                if scratch.len() < src.len() {
                    scratch.resize(src.len(), 0.0);
                }
                let dst = &mut scratch[..src.len()];
                src.convert_to_f32_slice(dst);
                dst
            }
        }
    }

    /// Widens into `dst`, which must be exactly as long.
    pub fn copy_to_f32(self, dst: &mut [f32]) {
        match self {
            KvRows::F32(s) => dst.copy_from_slice(s),
            KvRows::F16(s) => s.convert_to_f32_slice(dst),
        }
    }

    /// The whole run as f32: borrowed when it already is, widened once
    /// when it is f16. For a batched prefill, whose blocked kernel reads
    /// every row many times over; one widen per layer per batch costs
    /// less than one per read.
    pub fn to_f32_cow(self) -> std::borrow::Cow<'a, [f32]> {
        match self {
            KvRows::F32(s) => std::borrow::Cow::Borrowed(s),
            KvRows::F16(_) => std::borrow::Cow::Owned(self.to_f32_vec()),
        }
    }

    pub fn to_f32_vec(self) -> Vec<f32> {
        let mut out = vec![0.0; self.len()];
        self.copy_to_f32(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_rounds_at_write_and_every_read_sees_the_rounded_value() {
        let src = [1.0f32, 0.1, -3.333_333, 65504.0, 1e-8];
        let buf = KvBuf::from_f32(KvDtype::F16, &src);
        let want: Vec<f32> = src.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        assert_eq!(buf.to_f32_vec(), want);
        let mut scratch = Vec::new();
        assert_eq!(buf.view().read(1..4, &mut scratch), &want[1..4]);
        // 0.1 is not representable in f16: the store really is two bytes.
        assert_ne!(want[1], 0.1);
        assert_eq!(buf.allocated_bytes(), buf.capacity() * 2);
    }

    #[test]
    fn f32_is_exact_and_borrowed() {
        let src = [1.0f32, 0.1, -3.333_333];
        let buf = KvBuf::from_f32(KvDtype::F32, &src);
        let mut scratch = Vec::new();
        let row = buf.view().read(0..3, &mut scratch);
        assert_eq!(row, &src);
        assert!(scratch.is_empty(), "an f32 read must not copy");
    }

    #[test]
    fn edits_agree_across_dtypes() {
        for dtype in [KvDtype::F32, KvDtype::F16] {
            let mut buf = KvBuf::from_f32(dtype, &[1.0, 2.0, 3.0, 4.0]);
            buf.write_f32(1, &[5.0, 6.0]);
            buf.drain_front(1);
            buf.extend_from_f32(&[7.0]);
            buf.copy_within(0..1, 2);
            buf.truncate(3);
            assert_eq!(buf.to_f32_vec(), vec![5.0, 6.0, 5.0], "{dtype:?}");
        }
    }
}
