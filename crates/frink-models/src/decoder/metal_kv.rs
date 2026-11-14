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
        mkv: &mut frink_metal::attn::MetalKvBuffers,
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

    /// Make the host [`KvCache`] hold, as real values, every row Metal
    /// holds: replace the placeholder rows a Metal prefill advanced it
    /// by ([`Self::advance_host_kv_after_metal_prefill`]) and append the
    /// rows a device-only decode never pushed. A no-op when the host is
    /// caught up, or ahead.
    ///
    /// Placeholders are why this reads `host_rows_real` and not the row
    /// count alone: after a non-authoritative prefill the host has
    /// exactly as many rows as Metal, all zero, and a check on the count
    /// called that caught up. A layer the decode then served on the host
    /// attended over zeros (#630: OLMo, StableLM, Orion, whose LayerNorm
    /// no fused launch bakes, so every decode layer takes the host body).
    #[cfg(feature = "metal")]
    pub(super) fn catch_up_host_kv_from_metal(
        mkv: &mut frink_metal::attn::MetalKvBuffers,
        cache: &mut KvCache,
    ) {
        let rows = cache.rows();
        // The host pushed rows of its own past Metal's: it is the
        // authority for this layer, and the next entry re-uploads it
        // (`resync_stale_metal_layers`).
        if rows > mkv.seq_len {
            return;
        }
        let real = mkv.host_rows_real.min(rows);
        if real == mkv.seq_len {
            return;
        }
        // ROWS, then the POSITION the first placeholder row is:
        // `truncate` takes a position, and a cache that evicted holds
        // fewer rows than it has positions.
        if rows > real {
            cache.truncate(cache.positions() - (rows - real));
        }
        let n = mkv.seq_len - real;
        let (k, v) = mkv.tokens_host(real, n);
        // The Metal KV holds ONE head width for K and V
        // (`metal_can_serve_model` refuses split widths).
        let per = cache.n_kv_heads * cache.head_dim;
        for i in 0..n {
            let off = i * per;
            cache
                .push(&k[off..off + per], &v[off..off + per])
                .expect("unbounded/planned KvCache growth is infallible");
        }
        mkv.host_rows_real = mkv.seq_len;
    }

    /// Re-upload from the host every layer of the device KV that `stale`
    /// picks, and keep the rest.
    ///
    /// Per layer, because a pass that served some layers on the device
    /// and some on the host leaves them in opposite states: a device
    /// layer's host rows are placeholders or missing, a host layer's
    /// device rows are missing. Rebuilding EVERY layer from the host, as
    /// both entry points did, threw the device layers' real rows away
    /// for the host's placeholders (#630: SmolLM3 and EXAONE-4, whose
    /// unrotated layers take the host body in prefill).
    #[cfg(feature = "metal")]
    pub(super) fn resync_stale_metal_layers(
        metal_kvs: &mut [frink_metal::attn::MetalKvBuffers],
        kv_caches: &[KvCache],
        stale: impl Fn(&frink_metal::attn::MetalKvBuffers, &KvCache) -> bool,
    ) -> Result<(), frink_metal::gpu::MetalError> {
        for (m, c) in metal_kvs.iter_mut().zip(kv_caches) {
            if stale(m, c) {
                // ROWS: this uploads `c.k` / `c.v` themselves, so the
                // count must describe those buffers.
                m.upload_from_host(&c.k.to_f32_vec(), &c.v.to_f32_vec(), c.rows())?;
            }
        }
        Ok(())
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
        let mut guard = Self::lock_metal_attn_kv(&self.metal_attn_kv);
        let Some(metal_kvs) = guard.as_mut() else {
            return;
        };
        if metal_kvs.len() != kv_caches.len() {
            return;
        }
        for (mkv, cache) in metal_kvs.iter_mut().zip(kv_caches.iter_mut()) {
            Self::catch_up_host_kv_from_metal(mkv, cache);
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use frink_metal::attn::MetalKvBuffers;

    /// A host cache a Metal prefill advanced with placeholders, then a
    /// device decode left behind, comes back holding EVERY row the
    /// device holds, as real values: the placeholders replaced and the
    /// missing row appended (#630).
    ///
    /// The device here is filled by `upload_from_host` and then told
    /// the host holds none of it, which is the state a prefill kernel
    /// leaves: it writes the device and nothing on the host.
    #[test]
    #[ignore = "needs a real Metal GPU"]
    fn catching_up_replaces_the_placeholders_a_metal_prefill_left() {
        let (n_kv_heads, head_dim, rows) = (1usize, 32usize, 4usize);
        let width = n_kv_heads * head_dim;
        // f16-exact values, so the device's store returns them unchanged.
        let k: Vec<f32> = (0..rows * width)
            .map(|i| (i % 64) as f32 * 0.25 + 1.0)
            .collect();
        let v: Vec<f32> = k.iter().map(|x| -x).collect();
        let mut mkv = MetalKvBuffers::with_capacity(n_kv_heads, head_dim, 8).expect("kv");
        mkv.upload_from_host(&k, &v, rows).expect("upload");
        mkv.host_rows_real = 0;

        let mut cache = KvCache::new(n_kv_heads, head_dim);
        // The prefill's placeholders, one row short of the device: the
        // last row is the decode step the host never saw.
        cache.advance_len(rows - 1).expect("unbounded");
        assert_eq!(cache.rows(), rows - 1);

        Decoder::catch_up_host_kv_from_metal(&mut mkv, &mut cache);

        assert_eq!((cache.rows(), cache.positions()), (rows, rows));
        assert_eq!(cache.k.to_f32_vec(), k, "K rows are the device's");
        assert_eq!(cache.v.to_f32_vec(), v, "V rows are the device's");
        assert_eq!(mkv.host_rows_real, rows);

        // Caught up, it is a no-op: a host row pushed past the device's
        // is the host's own and stays.
        let own = vec![7.0f32; width];
        cache.push(&own, &own).expect("unbounded");
        Decoder::catch_up_host_kv_from_metal(&mut mkv, &mut cache);
        assert_eq!(cache.rows(), rows + 1);
        assert_eq!(cache.k.to_f32_vec()[rows * width..], own[..]);
    }
}
