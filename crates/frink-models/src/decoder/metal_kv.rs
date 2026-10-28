//! The device KV mirror the Metal attention reads, and keeping the host
//! KV cache in step with it.

use frink_core::cache::KvCache;

use super::Decoder;

impl Decoder {
    /// Advance the host [`KvCache`] over the positions a Metal prefill
    /// kernel just wrote to the device.
    ///
    /// Two ways to do that, and which one is right depends on whether
    /// anyone will READ the host rows.
    ///
    /// The contiguous path never does: Metal stays authoritative from
    /// prefill through decode, so [`KvCache::advance_len`]'s zero fill
    /// is a placeholder that only has to keep `seq_len` in step for the
    /// sync checks, and skipping the download is the whole point.
    ///
    /// The PAGED path does. `forward_batch_last_paged` scatters these
    /// rows into the page store, and a caller that reads placeholders
    /// gets a prompt the model never saw -- which is exactly how paged
    /// KV on Metal came to answer fluent nonsense while paged-on-CPU
    /// and contiguous-on-Metal were each correct. So it asks for the
    /// real rows and pays one download per layer, against a gather and
    /// a scatter it was already paying.
    ///
    /// Done here, per layer, immediately after the launch, rather than
    /// once at the end: the Metal KV buffers are dropped outright when
    /// a later layer's launch fails, and rows nobody downloaded before
    /// that are simply gone.
    #[cfg(feature = "metal")]
    pub(super) fn advance_host_kv_after_metal_prefill(
        mkv: &frink_metal::attn::MetalKvBuffers,
        cache: &mut KvCache,
        batch_size: usize,
        host_kv_authoritative: bool,
    ) {
        if host_kv_authoritative {
            Self::catch_up_host_kv_from_metal(mkv, cache);
            debug_assert_eq!(cache.positions(), mkv.seq_len);
        } else {
            cache
                .advance_len(batch_size)
                .expect("unbounded/planned KvCache growth is infallible");
        }
    }

    /// Append host [`KvCache`] positions that Metal already holds but host
    /// skipped (dense-stack fast path). No-op when `cache.seq_len` is caught up.
    #[cfg(feature = "metal")]
    pub(super) fn catch_up_host_kv_from_metal(
        mkv: &frink_metal::attn::MetalKvBuffers,
        cache: &mut KvCache,
    ) {
        // ROWS on both sides: this fills the host buffer with rows
        // Metal already holds, and `push` below advances positions with
        // them. Neither store evicts, so the two agree; when one learns
        // to (#61) this is a place that has to say which it meant.
        if cache.rows() >= mkv.seq_len {
            return;
        }
        let start = cache.rows();
        let n = mkv.seq_len - start;
        let (k, v) = mkv.tokens_host(start, n);
        let per = cache.n_kv_heads * cache.head_dim;
        for i in 0..n {
            let off = i * per;
            cache
                .push(&k[off..off + per], &v[off..off + per])
                .expect("unbounded/planned KvCache growth is infallible");
        }
    }

    /// Pull every layer's Metal-ahead suffix into `kv_caches` (prefix-cache
    /// Poison-tolerant lock for the shared Metal KV arena. A panicked
    /// holder must not permanently brick every later decode.
    #[cfg(feature = "metal")]
    pub(super) fn lock_metal_attn_kv(
        mutex: &std::sync::Mutex<Option<Vec<frink_metal::attn::MetalKvBuffers>>>,
    ) -> std::sync::MutexGuard<'_, Option<Vec<frink_metal::attn::MetalKvBuffers>>> {
        mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// True once a fused Metal attention launch has allocated this
    /// decoder's per-layer Metal KV, i.e. once a token has actually been
    /// served by a fused launch rather than by the host bodies.
    ///
    /// For tests that switch Metal ON and need to know whether the
    /// switch reached a kernel: a model the eligibility predicates keep
    /// on the host answers `false` forever, and a test asserting logits
    /// alone could not tell that from a launch that happened to agree.
    #[cfg(feature = "metal")]
    pub fn metal_attn_kv_allocated(&self) -> bool {
        Self::lock_metal_attn_kv(&self.metal_attn_kv).is_some()
    }

    /// store, continuous-batch / CPU readers). Safe no-op without Metal KV.
    #[cfg(feature = "metal")]
    pub fn sync_metal_attn_kv_to_host(&self, kv_caches: &mut [KvCache]) {
        assert_eq!(kv_caches.len(), self.config.n_layers);
        let guard = Self::lock_metal_attn_kv(&self.metal_attn_kv);
        let Some(metal_kvs) = guard.as_ref() else {
            return;
        };
        if metal_kvs.len() != kv_caches.len() {
            return;
        }
        for (mkv, cache) in metal_kvs.iter().zip(kv_caches.iter_mut()) {
            Self::catch_up_host_kv_from_metal(mkv, cache);
        }
    }
}
