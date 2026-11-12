//! Rolling a sequence's caches back, device mirrors included.

use frink_core::cache::KvCache;

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
}
