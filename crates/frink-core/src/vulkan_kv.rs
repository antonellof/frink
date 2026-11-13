//! A sequence's KV on the Vulkan device: the mirror the decode stack
//! attends over (`frink_models::decoder::vulkan_stack`).
//!
//! The same contract as `metal_attn` and `cuda_attn` on [`KvCache`]: per
//! sequence, a MIRROR and never the authority, used only while it holds
//! exactly the host cache's rows. That is checked by row count AND by the
//! newest K row, so a truncate that a host path grew back is caught, and
//! anything else rebuilds it from the host rows. One rebuild, in exchange
//! for not having to find every place a cache moves backwards.
//!
//! Layout: one buffer of f16 pairs, K for `capacity` positions then V for
//! `capacity` positions, each position `kv_width` values. A rebuild
//! writes the host rows as f16 through the mapping, rounded to nearest
//! even as the device's own `kv_store` and llama.cpp's f16 KV round, so
//! a row is the same bits whichever wrote it.
//!
//! [`KvCache`]: crate::cache::KvCache

use crate::kv_store::KvBuf;
use frink_vulkan::device::VulkanError;
use frink_vulkan::runtime::ops::{free_later, DeviceBuf};
use frink_vulkan::runtime::Runtime;
use half::f16;

/// One sequence-layer's KV on the device.
pub struct VulkanKvMirror {
    buf: DeviceBuf,
    /// The runtime `buf` belongs to (`Runtime::id`).
    owner: u64,
    capacity: usize,
    rows: usize,
    kv_width: usize,
    /// The newest K row as stored, in f16 pairs.
    last_k: Vec<u32>,
}

impl Drop for VulkanKvMirror {
    fn drop(&mut self) {
        // Queued, not freed: a cache can drop on a thread that holds the
        // runtime's lock.
        free_later(self.owner, self.buf);
    }
}

/// `values` as f16 pairs, low half first.
fn pairs(values: impl Iterator<Item = f16>) -> Vec<u32> {
    let halves: Vec<u16> = values.map(f16::to_bits).collect();
    halves
        .chunks(2)
        .map(|p| p[0] as u32 | (p.get(1).copied().unwrap_or(0) as u32) << 16)
        .collect()
}

/// Rows `[from, to)` of a host KV buffer, `width` values each, as f16
/// pairs.
fn host_rows(buf: &KvBuf, width: usize, from: usize, to: usize) -> Vec<u32> {
    let range = from * width..to * width;
    match buf {
        KvBuf::F32(v) => pairs(v[range].iter().map(|x| f16::from_f32(*x))),
        KvBuf::F16(v) => pairs(v[range].iter().copied()),
    }
}

impl VulkanKvMirror {
    /// The buffer, for the attention op.
    pub fn buf(&self) -> DeviceBuf {
        self.buf
    }

    /// Positions held.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Words in the buffer: K then V, each `capacity * kv_width / 2`.
    pub fn words(&self) -> usize {
        self.capacity * self.kv_width
    }

    /// The word where V begins.
    pub fn v_off(&self) -> usize {
        self.capacity * self.kv_width / 2
    }

    /// The word where position `pos`'s K begins.
    pub fn k_word(&self, pos: usize) -> usize {
        pos * self.kv_width / 2
    }

    /// The word where position `pos`'s V begins.
    pub fn v_word(&self, pos: usize) -> usize {
        self.v_off() + pos * self.kv_width / 2
    }

    /// Make `slot` a mirror of `positions` positions with room for
    /// `need`, rebuilding it from `k` and `v` when it is absent, too
    /// small, or no longer the host's history. The host holds the last
    /// `host_rows_held` of those positions, all of them unless its
    /// window evicts.
    #[allow(clippy::too_many_arguments)]
    pub fn sync<'s>(
        slot: &'s mut Option<Self>,
        rt: &mut Runtime,
        k: &KvBuf,
        v: &KvBuf,
        host_rows_held: usize,
        positions: usize,
        kv_width: usize,
        need: usize,
    ) -> Result<&'s mut Self, VulkanError> {
        assert_eq!(kv_width % 2, 0, "f16 pairs need an even KV width");
        assert!(host_rows_held <= positions, "more rows than positions");
        // The host's rows are the LAST `host_rows_held` positions: a
        // windowed cache has evicted the ones before them.
        let (rows, first) = (host_rows_held, positions - host_rows_held);
        let valid = slot.as_ref().is_some_and(|m| {
            m.owner == rt.id()
                && m.kv_width == kv_width
                && m.capacity >= need
                && m.rows == positions
                && (rows == 0 || m.last_k == host_rows(k, kv_width, rows - 1, rows))
        });
        if !valid {
            let capacity = need.max(256).next_power_of_two();
            let buf = rt.alloc_f32(capacity * kv_width)?;
            let mut m = Self {
                buf,
                owner: rt.id(),
                capacity,
                rows: positions,
                kv_width,
                last_k: Vec::new(),
            };
            if rows > 0 {
                // Each row at its position's slot. The slots before
                // `first` stay zero and are never read: a layer whose
                // cache evicts attends only inside its window, which
                // the host keeps.
                rt.write_u32(buf, m.k_word(first), &host_rows(k, kv_width, 0, rows))?;
                rt.write_u32(buf, m.v_word(first), &host_rows(v, kv_width, 0, rows))?;
                m.last_k = host_rows(k, kv_width, rows - 1, rows);
            }
            // The old mirror, if any, drops here and is queued for free.
            *slot = Some(m);
        }
        Ok(slot.as_mut().expect("set above"))
    }

    /// The device stored `k_row` (and its V) at position [`Self::rows`],
    /// and the host has pushed the same row: advance.
    pub fn record_push(&mut self, k_row: &[f32]) {
        debug_assert_eq!(k_row.len(), self.kv_width);
        self.rows += 1;
        self.last_k = pairs(k_row.iter().map(|x| f16::from_f32(*x)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_rows_are_f16_pairs_low_half_first() {
        let buf = KvBuf::F32(vec![1.0, -2.0, 0.5, 65504.0]);
        let words = host_rows(&buf, 2, 0, 2);
        assert_eq!(
            words,
            vec![
                0x3c00 | (0xc000 << 16), // 1.0, -2.0
                0x3800 | (0x7bff << 16), // 0.5, 65504
            ]
        );
        // An f16 store's bits are taken as they are.
        let f16s = KvBuf::F16(vec![f16::from_f32(1.0), f16::from_f32(-2.0)]);
        assert_eq!(host_rows(&f16s, 2, 0, 1), vec![words[0]]);
    }

    /// A mirror is kept only while it is the host's history: the same
    /// row count and the same newest K row. Anything else rebuilds it
    /// from the host rows.
    #[test]
    fn vulkan_mirror_is_rebuilt_when_the_host_rows_moved() {
        let mut rt = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => return frink_vulkan::skipped(&format!("no Vulkan device: {e}")),
        };
        let w = 4;
        let row = |i: usize| -> Vec<f32> { (0..w).map(|j| (i * w + j) as f32 * 0.25).collect() };
        let flat =
            |rows: &[usize]| -> KvBuf { KvBuf::F32(rows.iter().flat_map(|&i| row(i)).collect()) };
        let (k, v) = (flat(&[0, 1, 2]), flat(&[10, 11, 12]));
        let mut slot = None;

        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k, &v, 3, 3, w, 4).unwrap();
        let first = m.buf();
        assert_eq!(rt.read_u32(first, 0, 6).unwrap(), host_rows(&k, w, 0, 3));
        let v_off = slot.as_ref().unwrap().v_off();
        assert_eq!(
            rt.read_u32(first, v_off, 6).unwrap(),
            host_rows(&v, w, 0, 3)
        );

        // Unchanged host rows: the same buffer, nothing rewritten.
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k, &v, 3, 3, w, 4).unwrap();
        assert_eq!(m.buf(), first);

        // The device and the host both took row 3: still the same buffer.
        m.record_push(&row(3));
        let (k4, v4) = (flat(&[0, 1, 2, 3]), flat(&[10, 11, 12, 13]));
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k4, &v4, 4, 4, w, 5).unwrap();
        assert_eq!(m.buf(), first);

        // The host truncated to two rows and grew back to four with
        // another history: the count matches and the newest K row does
        // not.
        let (k_other, v_other) = (flat(&[0, 1, 7, 8]), flat(&[10, 11, 17, 18]));
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k_other, &v_other, 4, 4, w, 5).unwrap();
        let rebuilt = m.buf();
        assert_ne!(rebuilt, first, "a mirror of another history was kept");
        assert_eq!(
            rt.read_u32(rebuilt, 0, 8).unwrap(),
            host_rows(&k_other, w, 0, 4)
        );

        // Fewer host rows, ending in the row the mirror ends in: only
        // the count tells them apart, and the mirror's row 1 is not the
        // host's.
        let (k_short, v_short) = (flat(&[0, 8]), flat(&[10, 18]));
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k_short, &v_short, 2, 2, w, 3).unwrap();
        assert_eq!(m.rows(), 2);
        let short = m.buf();
        assert_ne!(
            short, rebuilt,
            "a mirror with rows the host dropped was kept"
        );
        assert_eq!(
            rt.read_u32(short, 0, 4).unwrap(),
            host_rows(&k_short, w, 0, 2)
        );
        // A windowed host cache: five positions, of which it kept the
        // last two. They sit at their positions' slots, K and V alike,
        // and the mirror counts positions.
        let (k_tail, v_tail) = (flat(&[3, 4]), flat(&[13, 14]));
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k_tail, &v_tail, 2, 5, w, 6).unwrap();
        assert_eq!(m.rows(), 5);
        let (buf, at_k, at_v) = (m.buf(), m.k_word(3), m.v_word(3));
        assert_eq!(
            rt.read_u32(buf, at_k, 4).unwrap(),
            host_rows(&k_tail, w, 0, 2)
        );
        assert_eq!(
            rt.read_u32(buf, at_v, 4).unwrap(),
            host_rows(&v_tail, w, 0, 2)
        );
        // Kept while the host's newest row and position count hold...
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k_tail, &v_tail, 2, 5, w, 6).unwrap();
        assert_eq!(m.buf(), buf);
        // ...and not when the same rows are another position count.
        let m = VulkanKvMirror::sync(&mut slot, &mut rt, &k_tail, &v_tail, 2, 4, w, 6).unwrap();
        assert_ne!(m.buf(), buf, "a mirror of other positions was kept");
    }
}
