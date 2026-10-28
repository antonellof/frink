//! One sequence's K/V for one layer, on the device, so a decode token
//! attends without uploading the history.
//!
//! The resident stack used to take a layer's prefix from the host
//! cache on every call: right for a prefill step, which reads the
//! prefix once, and ruinous for decode, which would upload the whole
//! history per layer per token (at position 512 on Llama-3.2-3B, 117 MB
//! a token). The earlier attempt at a resident cache
//! (`crate::attn::CudaKvBuffers`) re-uploaded its full capacity on
//! every append, which is part of why `FRINK_CUDA_GQA=1` measured 42%
//! slower.
//!
//! # Identity
//!
//! It lives in the sequence's `frink_core::cache::KvCache`, beside the
//! Metal mirror, and for the reason that field gives: a mirror hung off
//! a shared decoder would hand one request another's keys. It is a
//! MIRROR, never the authority: the host cache stays complete, and the
//! stack pushes every row it computes there too.
//!
//! A mirror is used only when [`mirror_matches`] says it holds what the
//! host cache holds: the same row count, the same width, and a last K
//! row bitwise equal to the host's last K row. The row count alone is
//! the check Metal makes, and it misses a cache that was truncated and
//! then grown back to the same length by a host path. The last row
//! catches that, because a different token at that position has a
//! different K row. The stack validates every layer of a run together
//! and re-uploads all of them if any one is stale, because layer 0's K
//! row depends only on (token, position) and could agree between two
//! histories that every later layer tells apart.

use crate::gpu::CudaError;
use crate::xfer;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// Smallest capacity a mirror is allocated at, in rows.
const MIN_ROWS: usize = 256;

/// One layer's device K and V, `[cap_rows, width]` each, of which the
/// first `rows` are this sequence's history.
pub struct CudaKvMirror {
    pub(crate) k: CudaSlice<f32>,
    pub(crate) v: CudaSlice<f32>,
    width: usize,
    cap_rows: usize,
    rows: usize,
    /// The host bytes of the newest K row the mirror holds, empty when
    /// it holds none: the identity check.
    last_k: Vec<f32>,
}

impl std::fmt::Debug for CudaKvMirror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKvMirror")
            .field("width", &self.width)
            .field("cap_rows", &self.cap_rows)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

/// Whether a mirror holding `rows` rows of `width`, whose newest K row
/// is `last_k`, holds exactly the host cache `host_k` / `host_v` of
/// rows of `want_width`. Pure, so the rule is tested without a device.
pub(crate) fn mirror_matches(
    rows: usize,
    width: usize,
    last_k: &[f32],
    host_k: &[f32],
    host_v: &[f32],
    want_width: usize,
) -> bool {
    if width != want_width || want_width == 0 {
        return false;
    }
    let Some(len) = rows.checked_mul(width) else {
        return false;
    };
    if host_k.len() != len || host_v.len() != len {
        return false;
    }
    if rows == 0 {
        return last_k.is_empty();
    }
    let host_last = &host_k[len - width..];
    // Bitwise: the mirror's row came from the same download the host
    // row did, so anything short of identical bits is a different row
    // (and a NaN row, which `==` would call unequal, re-uploads, which
    // is harmless).
    last_k.len() == width
        && last_k
            .iter()
            .zip(host_last)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// The capacity, in rows, a mirror needs to hold `need` rows when it
/// holds `cap` now: unchanged if it fits, else doubled until it does,
/// never below [`MIN_ROWS`]. `None` if that overflows.
pub(crate) fn grown_capacity(cap: usize, need: usize) -> Option<usize> {
    if need <= cap {
        return Some(cap);
    }
    let mut c = cap.max(MIN_ROWS);
    while c < need {
        c = c.checked_mul(2)?;
    }
    Some(c)
}

fn alloc(dev: &Arc<CudaDevice>, elems: usize, what: &str) -> Result<CudaSlice<f32>, CudaError> {
    // Zeroed so the rows past `rows` are never uninitialised memory,
    // although no kernel reads them.
    dev.alloc_zeros::<f32>(elems)
        .map_err(|e| CudaError::Launch(format!("{what} alloc: {e:?}")))
}

impl CudaKvMirror {
    /// Rows of history the mirror holds.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Whether this mirror holds exactly `host_k` / `host_v`.
    pub(crate) fn holds(&self, host_k: &[f32], host_v: &[f32], width: usize) -> bool {
        mirror_matches(self.rows, self.width, &self.last_k, host_k, host_v, width)
    }

    /// A mirror of the host cache's rows, room for `need_rows`. Waits
    /// for the upload (the host rows are borrowed), which is the one
    /// sync a stale mirror costs.
    pub(crate) fn from_host(
        dev: &Arc<CudaDevice>,
        host_k: &[f32],
        host_v: &[f32],
        width: usize,
        need_rows: usize,
    ) -> Result<Self, CudaError> {
        if width == 0 || host_k.len() != host_v.len() || !host_k.len().is_multiple_of(width) {
            return Err(CudaError::Unsupported(format!(
                "KV mirror: {} K and {} V values in rows of {width}",
                host_k.len(),
                host_v.len()
            )));
        }
        let rows = host_k.len() / width;
        let cap_rows = grown_capacity(0, need_rows.max(rows))
            .ok_or_else(|| CudaError::Unsupported("KV mirror capacity overflows".into()))?;
        let elems = cap_rows
            .checked_mul(width)
            .ok_or_else(|| CudaError::Unsupported("KV mirror size overflows".into()))?;
        let mut k = alloc(dev, elems, "KV mirror K")?;
        let mut v = alloc(dev, elems, "KV mirror V")?;
        let n = rows * width;
        if n > 0 {
            xfer::htod_into_sync(dev, "KV mirror K", host_k, &mut k.slice_mut(..n))?;
            xfer::htod_into_sync(dev, "KV mirror V", host_v, &mut v.slice_mut(..n))?;
        }
        let last_k = if rows == 0 {
            Vec::new()
        } else {
            host_k[n - width..].to_vec()
        };
        Ok(Self {
            k,
            v,
            width,
            cap_rows,
            rows,
            last_k,
        })
    }

    /// Room for `need_rows`, growing on the device (the rows already
    /// held are copied, not re-uploaded).
    pub(crate) fn reserve(
        &mut self,
        dev: &Arc<CudaDevice>,
        need_rows: usize,
    ) -> Result<(), CudaError> {
        let cap = grown_capacity(self.cap_rows, need_rows)
            .ok_or_else(|| CudaError::Unsupported("KV mirror capacity overflows".into()))?;
        if cap == self.cap_rows {
            return Ok(());
        }
        let elems = cap
            .checked_mul(self.width)
            .ok_or_else(|| CudaError::Unsupported("KV mirror size overflows".into()))?;
        let mut k = alloc(dev, elems, "KV mirror K")?;
        let mut v = alloc(dev, elems, "KV mirror V")?;
        let n = self.rows * self.width;
        if n > 0 {
            xfer::dtod(
                dev,
                "KV mirror K grow",
                &self.k.slice(..n),
                &mut k.slice_mut(..n),
            )?;
            xfer::dtod(
                dev,
                "KV mirror V grow",
                &self.v.slice(..n),
                &mut v.slice_mut(..n),
            )?;
        }
        self.k = k;
        self.v = v;
        self.cap_rows = cap;
        Ok(())
    }

    /// Records that the stack appended rows up to `rows` and that the
    /// newest K row, as the host now holds it, is `last_k`.
    pub(crate) fn commit(&mut self, rows: usize, last_k: &[f32]) {
        debug_assert!(rows <= self.cap_rows);
        debug_assert_eq!(last_k.len(), self.width);
        self.rows = rows;
        self.last_k.clear();
        self.last_k.extend_from_slice(last_k);
    }

    /// Room for this many rows without growing.
    pub(crate) fn cap_rows(&self) -> usize {
        self.cap_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, w: usize, seed: f32) -> Vec<f32> {
        (0..n * w).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
    }

    /// The identity check is the whole safety argument for the mirror,
    /// so each way a mirror can be stale is a case that must say no.
    #[test]
    fn a_mirror_matches_only_the_history_it_was_built_from() {
        let w = 8;
        let k = rows(5, w, 0.1);
        let v = rows(5, w, 0.2);
        let last = k[4 * w..].to_vec();
        assert!(mirror_matches(5, w, &last, &k, &v, w));
        // One row short or long: the row count disagrees.
        assert!(!mirror_matches(4, w, &last, &k, &v, w));
        assert!(!mirror_matches(6, w, &last, &k, &v, w));
        // A different width.
        assert!(!mirror_matches(5, w, &last, &k, &v, w / 2));
        // Truncated by one and grown back by a host path with another
        // token: same length, different last row.
        let mut k2 = k.clone();
        k2[4 * w + 3] += 1.0;
        assert!(!mirror_matches(5, w, &last, &k2, &v, w));
        // V of a different length than K.
        assert!(!mirror_matches(5, w, &last, &k, &v[..4 * w], w));
        // An empty mirror matches an empty cache and nothing else.
        assert!(mirror_matches(0, w, &[], &[], &[], w));
        assert!(!mirror_matches(0, w, &last, &[], &[], w));
        // -0.0 and 0.0 are different rows to the bitwise check.
        let mut z = vec![0.0f32; w];
        let host = z.clone();
        z[0] = -0.0;
        assert!(!mirror_matches(1, w, &z, &host, &host, w));
    }

    #[test]
    fn capacity_doubles_from_a_floor_and_never_shrinks() {
        assert_eq!(grown_capacity(0, 1), Some(MIN_ROWS));
        assert_eq!(grown_capacity(0, MIN_ROWS + 1), Some(2 * MIN_ROWS));
        assert_eq!(grown_capacity(1024, 10), Some(1024));
        assert_eq!(grown_capacity(1024, 1025), Some(2048));
        assert_eq!(grown_capacity(256, 5000), Some(8192));
        assert_eq!(grown_capacity(usize::MAX / 2 + 1, usize::MAX), None);
    }
}
