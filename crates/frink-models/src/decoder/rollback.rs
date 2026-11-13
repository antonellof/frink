//! Rolling a sequence's caches back, device mirrors included.

use frink_core::cache::{KvCache, KvCheckpoint};

use super::Decoder;

impl Decoder {
    /// Rolls every layer's cache back to `len` positions, the device
    /// mirrors with it: what a rejected speculative block needs.
    ///
    /// `KvCache::truncate` alone is not enough on Metal. The prefill
    /// and verify bodies keep K/V in the decoder's Metal arena and
    /// leave the host rows as placeholders
    /// (`advance_host_kv_after_metal_prefill`), so a host cache shorter
    /// than its mirror made the next batch rebuild the mirror FROM the
    /// placeholders, and the target answered from zeroed keys. A Metal
    /// append writes at `seq_len`, so rewinding `seq_len` is the device
    /// truncation. The CUDA mirror needs nothing: it holds a copy of
    /// complete host rows and re-validates against them
    /// (`frink_cuda::prefill::kv_mirror`).
    pub fn truncate_kv(&self, kv_caches: &mut [KvCache], len: usize) {
        for cache in kv_caches.iter_mut() {
            cache.truncate(len);
        }
        self.rewind_device_kv(kv_caches);
    }

    /// [`Self::truncate_kv`] for a target with recurrent layers: every
    /// cache back to its [`KvCheckpoint`], the recurrent states with
    /// them, and the device mirrors as `truncate_kv` rewinds them. The
    /// recurrent states are restored in place, so a Metal kernel's
    /// no-copy wrap of them stays valid.
    pub fn restore_kv(&self, kv_caches: &mut [KvCache], checkpoints: &[KvCheckpoint]) {
        assert_eq!(
            kv_caches.len(),
            checkpoints.len(),
            "one checkpoint per layer's cache"
        );
        for (cache, cp) in kv_caches.iter_mut().zip(checkpoints) {
            cache.restore(cp);
        }
        self.rewind_device_kv(kv_caches);
    }

    /// Shortens the device mirrors to the host caches' rows.
    #[cfg_attr(not(feature = "metal"), allow(unused_variables))]
    fn rewind_device_kv(&self, kv_caches: &[KvCache]) {
        #[cfg(feature = "metal")]
        {
            let mut guard = Self::lock_metal_attn_kv(&self.metal_attn_kv);
            if let Some(metal_kvs) = guard.as_mut() {
                for (mkv, cache) in metal_kvs.iter_mut().zip(kv_caches.iter()) {
                    // ROWS: the arena holds rows, and a rollback only
                    // ever shortens them.
                    mkv.seq_len = mkv.seq_len.min(cache.rows());
                }
            }
        }
    }

    /// Forgets the Metal arena, so the next forward rebuilds it from
    /// `kv_caches` the caller is about to hand in.
    ///
    /// For caches that did not come from the arena's last sequence: a
    /// prefix-cache restore. Each sync check compares only a length,
    /// so an arena left by another conversation that happens to hold
    /// as many rows as the restored caches passes for them, and the
    /// model attends to that conversation's keys. A no-op without
    /// Metal.
    pub fn forget_device_kv(&self) {
        #[cfg(feature = "metal")]
        {
            *Self::lock_metal_attn_kv(&self.metal_attn_kv) = None;
        }
    }
}
