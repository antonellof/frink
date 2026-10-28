//! An attention layer's attention ON THE DEVICE, against the
//! sequence's own KV mirror.
//!
//! # Why this is the last one
//!
//! With the projections riding in the previous recurrent run and the
//! tail in the next, the only thing an attention layer still returned
//! to the host for was the attention itself, because the KV lived
//! there. That cost two things: a submission boundary, and a term that
//! GROWS with the context. Measured on one box, the reference is flat
//! from 32 to 300 tokens (11.46 to 11.50) and this engine was not
//! (11.1 to 10.95), and that difference is precisely a host attention
//! whose work is linear in `seq_len`.
//!
//! # The mirror, and why it cannot go stale
//!
//! `KvCache::metal_attn` is the sequence's own device copy, appended
//! one row a token. The host `k`/`v` stay complete and authoritative,
//! so nothing else in the engine changes: truncation, the prefix cache,
//! slot files and every host reader work as before.
//!
//! The mirror is trusted only while its `seq_len` is exactly the
//! cache's `rows()`. Anything that moves a cache backwards leaves the
//! two disagreeing, and the next use re-uploads the whole history
//! instead of guessing which rows are still good. That is one upload
//! in exchange for not having to enumerate every path that can rewind
//! a cache -- and the enumeration is what would rot.

use super::{Decoder, LayerWeights};
use frink_core::KvCache;

impl Decoder {
    /// Layer `l`'s attention, gate, `wo`, residual, FFN norm, FFN and
    /// residual in ONE submission, against the cache's device mirror.
    /// `None` when anything refuses, and then the host path runs.
    ///
    /// `q`/`k`/`v` are post-RoPE and post-QK-norm, exactly what the
    /// host attention would have read; `gate` is the half split off a
    /// double-width `wq`.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_attention_layer(
        &self,
        l: usize,
        layer: &LayerWeights,
        cache: &mut KvCache,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: Option<&[f32]>,
        residual: &[f32],
    ) -> Option<Vec<f32>> {
        // The tail's refusals are this path's too: it ends with the
        // same `wo` and the same FFN.
        let (out_proj, fold_branch, _ffn, launches) = self.attn_tail_launch(l, layer, q)?;
        let shape = self.config.layer_shape(l);
        let crate::layer_shapes::AttnShape::Gqa {
            n_heads,
            n_kv_heads,
        } = shape.attention
        else {
            return None;
        };
        if !self.device_attention_shape_ok(l, layer) {
            return None;
        }
        let head_dim = self.config.head_dim;
        let rows = cache.rows();
        let width = rows * n_kv_heads * head_dim;
        Self::ensure_attn_mirror(cache, n_kv_heads, head_dim, rows, width)?;
        let mirror = cache.metal_attn.as_mut()?;
        let out = frink_metal::gdn_branch::launch_attn_layer(
            mirror,
            q,
            k,
            v,
            gate,
            n_heads,
            self.config.attn_logit_softcap,
            &out_proj,
            fold_branch.as_ref(),
            &launches.as_metal(),
            residual,
        )
        .ok()?;
        layer.moe.record_activations_dense();
        let _ = n_kv_heads;
        Some(out)
    }

    /// The same layer at the head of a RUN that then carries the
    /// recurrent layers after it: one wait for the whole group, and
    /// nothing in it returns to the host.
    ///
    /// Returns the index to resume at. `None` when anything refuses or
    /// there are no recurrent layers behind it, and then the caller
    /// falls back to the standalone launch.
    #[cfg(feature = "metal")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn device_attention_layer_then_run(
        &self,
        l: usize,
        layer: &LayerWeights,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: Option<&[f32]>,
        hidden: &mut Vec<f32>,
        kv_caches: &mut [frink_core::KvCache],
        pending: &mut crate::decoder::fused_recurrent::PendingQkv,
    ) -> Option<usize> {
        let mut end = l + 1;
        while end < kv_caches.len() && self.fused_layer_parts(end).is_some() {
            end += 1;
        }
        if end == l + 1 {
            return None;
        }
        let (out_proj, fold_branch, _ffn, launches) = self.attn_tail_launch(l, layer, q)?;
        let shape = self.config.layer_shape(l);
        let crate::layer_shapes::AttnShape::Gqa {
            n_heads,
            n_kv_heads,
        } = shape.attention
        else {
            return None;
        };
        if !self.device_attention_shape_ok(l, layer) {
            return None;
        }
        let head_dim = self.config.head_dim;
        let mut run = frink_metal::gdn_branch::GdnRun::start(hidden).ok()?;
        {
            let cache = &mut kv_caches[l];
            let rows = cache.rows();
            Self::ensure_attn_mirror(
                cache,
                n_kv_heads,
                head_dim,
                rows,
                rows * n_kv_heads * head_dim,
            )?;
            let mirror = cache.metal_attn.as_mut()?;
            // SAFETY: the mirror lives in this cache, which nothing
            // else touches until the run finishes below.
            unsafe {
                run.attn_layer(
                    mirror,
                    q,
                    k,
                    v,
                    gate,
                    n_heads,
                    self.config.attn_logit_softcap,
                    &out_proj,
                    fold_branch.as_ref(),
                    &launches.as_metal(),
                )
            }
            .ok()?;
            cache
                .push(k, v)
                .expect("unbounded/planned KvCache growth is infallible");
        }
        layer.moe.record_activations_dense();
        self.run_layers(&mut run, l + 1, end, kv_caches)?;
        let head = self.encode_next_attn_head(&mut run, end, kv_caches.len());
        let (out, qkv) = run.finish_with_head().ok()?;
        *hidden = out;
        *pending = head.then_some(qkv).flatten().map(|q| (end, q));
        Some(end)
    }

    /// The facts this path needs of a layer beyond the tail's: one
    /// window, no sinks, no ALiBi, one head width.
    #[cfg(feature = "metal")]
    pub(crate) fn device_attention_shape_ok(&self, l: usize, layer: &LayerWeights) -> bool {
        self.config.layer_sliding_window(l).is_none()
            && layer.attn.sinks.is_none()
            && self.alibi_slopes.is_none()
            && self.config.v_head_dim() == self.config.head_dim
    }

    /// The cache's device mirror, made on first use at the capacity the
    /// cache was planned for.
    ///
    /// `None` when there is no planned capacity to size it by: a cache
    /// that can grow without bound cannot have a fixed device buffer
    /// behind it, and guessing a size here would mean a refusal much
    /// later, in the middle of a generation.
    #[cfg(feature = "metal")]
    fn ensure_attn_mirror(
        cache: &mut KvCache,
        n_kv_heads: usize,
        head_dim: usize,
        rows: usize,
        width: usize,
    ) -> Option<()> {
        // Sized to the cache's plan when it has one, and GROWN when it
        // does not: a mirror that refused an unbounded cache would
        // never fire, because the CLI's caches are unbounded. Doubling
        // from the current length costs one re-upload per doubling,
        // which the staleness check below performs anyway.
        let want = rows + 1;
        let too_small = cache
            .metal_attn
            .as_ref()
            .is_some_and(|m| m.capacity() < want);
        if cache.metal_attn.is_none() || too_small {
            let capacity = match cache.capacity_positions() {
                Some(planned) if planned >= want => planned,
                _ => want.next_power_of_two().max(512),
            };
            cache.metal_attn =
                frink_metal::attn::MetalKvBuffers::with_capacity(n_kv_heads, head_dim, capacity)
                    .ok();
        }
        let stale = cache.metal_attn.as_ref()?.seq_len != rows;
        if stale {
            // Split the borrows: the upload reads the authority and
            // writes the mirror, and they are two fields of one cache.
            // An f16 store is widened into a local first (the device
            // mirror is f32 whatever the host keeps); an f32 one is
            // uploaded in place.
            let widened = match (cache.k.as_f32(), cache.v.as_f32()) {
                (Some(_), Some(_)) => None,
                _ => Some((
                    cache.k_rows().slice(0..width).to_f32_vec(),
                    cache.v_rows().slice(0..width).to_f32_vec(),
                )),
            };
            let (k, v): (&[f32], &[f32]) = match &widened {
                Some((k, v)) => (k, v),
                None => {
                    let (k, v) = (&cache.k.as_f32()?[..width], &cache.v.as_f32()?[..width]);
                    // SAFETY-free: `k`/`v` are immutable borrows of other
                    // fields, which the compiler cannot see through a single
                    // `&mut cache`, so they are taken first.
                    unsafe {
                        (
                            std::slice::from_raw_parts(k.as_ptr(), k.len()),
                            std::slice::from_raw_parts(v.as_ptr(), v.len()),
                        )
                    }
                }
            };
            let mirror = cache.metal_attn.as_mut()?;
            mirror.upload_from_host(k, v, rows).ok()?;
            mirror.seq_len = rows;
        }
        Some(())
    }
}
