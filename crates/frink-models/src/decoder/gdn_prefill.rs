//! A run of PREFILL layers in one Metal submission, or nothing
//! (issue #540): recurrent layers, and the attention layers between
//! them whole, so a hybrid's prompt batch is one command buffer.
//!
//! The decode path's [`Decoder::fused_layer_parts`] says which
//! recurrent layers qualify, and [`Decoder::fused_attention_tail_eligible`]
//! which attention tails may lead a run, so prefill and decode cannot
//! disagree about what a fusable layer is; an attention layer rides
//! whole when [`Decoder::prefill_attn_layer`] admits it as well.
//! `frink_metal::gdn_prefill` runs them with the hidden batch resident
//! across the run.

use super::fused_view::FusedAttnCaps;
use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;
use frink_core::KvCache;
use frink_metal::gdn_prefill::PrefillRunLayer;
use frink_metal::gdn_prefill_attn::PrefillAttnLayer;

/// One layer of a run as planned, before any cache is touched.
enum Planned<'a> {
    Gdn(&'a crate::gdn::Gdn, &'a [f32], LayerFfnParts<'a>),
    Attn(Box<PrefillAttnLayer<'a>>),
}

impl Decoder {
    /// Layers `start..` of a `rows`-token batch of one sequence at
    /// positions `start_pos..`, as far as consecutive layers qualify and
    /// no further than `run_end`. Returns where the caller resumes, with
    /// `hidden` replaced by the batch as it leaves the run, or `None`
    /// when nothing ran and the host bodies take layer `start`.
    pub(crate) fn metal_prefill_run_from(
        &self,
        start: usize,
        run_end: usize,
        start_pos: usize,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        self.metal_prefill_run(start, run_end, start_pos, None, hidden, rows, kv_caches)
    }

    /// Attention layer `l`'s tail -- `wo` over `branch` (the attention
    /// output as [`Self::attn_branch_rows`] leaves it), the residual and
    /// the FFN -- and then the recurrent layers after it, in the same
    /// submission. `hidden` is the residual stream as it entered layer
    /// `l`. `None` when the tail is not that shape, and then nothing
    /// has run.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn metal_attn_tail_prefill_run(
        &self,
        l: usize,
        layer: &LayerWeights,
        branch: &[f32],
        run_end: usize,
        start_pos: usize,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        if !self.prefill_rows_take_device(rows) || !self.fused_attention_tail_eligible(l, layer) {
            return None;
        }
        let ffn = LayerFfnParts::for_layer(layer, self.config.rms_norm_eps, true)?;
        let (base, fold) = layer.attn.o_proj.launch_parts();
        let o_proj = base.mul_mm_sg_launch()?;
        let width = branch.len() / rows;
        let fold = match fold {
            None => None,
            Some(f) => Some(f.metal_plan(width)?),
        };
        let tail = frink_metal::gdn_prefill::PrefillAttnTail {
            branch,
            o_proj,
            fold,
            ffn: ffn.prefill_launches()?,
            eps: self.config.rms_norm_eps,
        };
        let end = self.metal_prefill_run(
            l + 1,
            run_end,
            start_pos,
            Some(&tail),
            hidden,
            rows,
            kv_caches,
        )?;
        layer.moe.record_activations_dense();
        Some(end)
    }

    /// Below the chunked recurrence's own device crossover the host
    /// body is the faster one, and it is where `delta_chunk_rows` would
    /// have sent these rows anyway.
    fn prefill_rows_take_device(&self, rows: usize) -> bool {
        rows >= frink_core::gdn_chunk::DEVICE_ROWS
            && frink_core::weight_matrix::metal_dense_enabled()
    }

    /// Attention layer `l` whole on the device, with its KV head count,
    /// or `None` and it is the host body's. Asked BEFORE any cache is
    /// touched; the mirror's own fit is
    /// [`PrefillAttnLayer::is_supported`]'s, once it exists.
    ///
    /// Everything the host body applies between the pre-norm and `wo`
    /// is either here or a refusal: the attention facts every fused
    /// launch shares ([`Self::fused_attn_layer_ok`], under caps that
    /// admit the interleaved Q gate), the tail's
    /// ([`Self::fused_attention_tail_eligible`]) and the device
    /// attention's ([`Self::device_attention_shape_ok`]). The host KV
    /// must be f16 with no row evicted, so the mirror the run appends to
    /// and the host rows written back from it are the same rows.
    fn prefill_attn_layer(
        &self,
        l: usize,
        start_pos: usize,
        cache: &KvCache,
    ) -> Option<(PrefillAttnLayer<'_>, usize)> {
        let layer = self.layer_for(l);
        let AttnShape::Gqa {
            n_heads,
            n_kv_heads,
        } = self.config.layer_shape(l).attention
        else {
            return None;
        };
        let caps = FusedAttnCaps::METAL_PREFILL_RUN;
        if !Self::metal_can_serve_layers(&self.config, self.lora_attached())
            || !self.fused_attn_layer_ok(layer, caps)
            || !self.fused_attention_tail_eligible(l, layer)
            || !self.device_attention_shape_ok(l, layer)
            || cache.dtype() != frink_core::kv_store::KvDtype::F16
            || cache.rows() != start_pos
            || cache.positions() != start_pos
        {
            return None;
        }
        let eps = self.config.rms_norm_eps;
        let attn_norm = layer.attn.norm_weight.rms_weights()?;
        let (q, q_fold) = layer.attn.q_proj.launch_parts();
        let (k, k_fold) = layer.attn.k_proj.launch_parts();
        let (v, v_fold) = layer.attn.v_proj.launch_parts();
        // One rotation for all three: they read the same vector, and
        // the run applies it once.
        let same = |a: Option<&std::sync::Arc<_>>, b: Option<&std::sync::Arc<_>>| match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => std::sync::Arc::ptr_eq(x, y),
            _ => false,
        };
        if !same(q_fold, k_fold) || !same(q_fold, v_fold) {
            return None;
        }
        let fold_x = match q_fold {
            None => None,
            Some(f) => Some(f.metal_plan(attn_norm.len())?),
        };
        let (o, o_fold) = layer.attn.o_proj.launch_parts();
        let fold_branch = match o_fold {
            None => None,
            Some(f) => Some(f.metal_plan(n_heads * self.config.head_dim)?),
        };
        let ffn = LayerFfnParts::for_layer(layer, eps, true)?;
        let attn = PrefillAttnLayer {
            attn_norm,
            eps,
            fold_x,
            q: q.mul_mm_sg_launch()?,
            k: k.mul_mm_sg_launch()?,
            v: v.mul_mm_sg_launch()?,
            n_heads,
            q_gate: layer.attn.q_gate_interleaved,
            extras: self.metal_attn_view_under(layer, caps)?,
            rope: self.metal_rope(),
            layer_rope: self.metal_layer_rope(l),
            o_proj: o.mul_mm_sg_launch()?,
            fold_branch,
            ffn: ffn.prefill_launches()?,
        };
        Some((attn, n_kv_heads))
    }

    /// The ONE body both entries share: `head`, if any, then the
    /// qualifying layers from `start`. Only on a model with recurrent
    /// layers: a dense model's attention layers are the dense stacks'.
    #[allow(clippy::too_many_arguments)]
    fn metal_prefill_run(
        &self,
        start: usize,
        run_end: usize,
        start_pos: usize,
        head: Option<&frink_metal::gdn_prefill::PrefillAttnTail<'_>>,
        hidden: &mut Vec<f32>,
        rows: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<usize> {
        if !self.prefill_rows_take_device(rows) || !self.has_gdn_layers() {
            return None;
        }
        let eps = self.config.rms_norm_eps;
        let head_dim = self.config.head_dim;
        let mut plan = Vec::new();
        let mut end = start;
        while end < run_end.min(kv_caches.len()) {
            if let Some((gdn, attn_norm, ffn)) = self.fused_layer_parts(end) {
                plan.push(Planned::Gdn(gdn, attn_norm, ffn));
            } else if let Some((a, n_kv)) = self.prefill_attn_layer(end, start_pos, &kv_caches[end])
            {
                // The mirror, made or caught up to the host rows with
                // room for this batch; a layer it does not fit ends the
                // run before it.
                let cache = &mut kv_caches[end];
                let prior = cache.rows();
                let fits = Self::ensure_attn_mirror(
                    cache,
                    n_kv,
                    head_dim,
                    prior,
                    prior * n_kv * head_dim,
                    rows,
                )
                .and(cache.metal_attn.as_ref())
                .is_some_and(|m| a.is_supported(m, rows, start_pos));
                if !fits {
                    break;
                }
                plan.push(Planned::Attn(Box::new(a)));
            } else {
                break;
            }
            end += 1;
        }
        if plan.is_empty() && head.is_none() {
            return None;
        }
        let mut entries = Vec::with_capacity(plan.len());
        for (p, cache) in plan.into_iter().zip(kv_caches[start..end].iter_mut()) {
            entries.push(match p {
                Planned::Gdn(gdn, attn_norm, ffn) => {
                    let layer = gdn.prefill_layer(attn_norm, eps, &ffn)?;
                    let state = cache.recurrent.get_or_insert_with(|| gdn.zero_state());
                    PrefillRunLayer::Gdn(
                        layer,
                        frink_metal::gdn_prefill::PrefillGdnState {
                            ssm_ptr: state.ssm.as_ptr(),
                            ssm_bytes: state.ssm.alloc_bytes(),
                            conv_ptr: state.conv.as_ptr(),
                            conv_bytes: state.conv.alloc_bytes(),
                            conv_len: state.conv.len(),
                        },
                    )
                }
                Planned::Attn(a) => PrefillRunLayer::Attn(*a, cache.metal_attn.as_mut()?),
            });
        }
        // SAFETY: each state is page-aligned (`AlignedF32`) and lives in
        // its own cache element, borrowed mutably through `kv_caches`
        // for this whole call and touched by nothing else; the launch
        // waits for the GPU before returning.
        let out = unsafe {
            frink_metal::gdn_prefill::launch_gdn_prefill_run(
                hidden,
                rows,
                start_pos,
                head,
                &mut entries,
            )
        }
        .ok()?;
        drop(entries);
        for (l, cache) in (start..end).zip(kv_caches[start..end].iter_mut()) {
            if matches!(self.config.layer_shape(l).attention, AttnShape::Gdn) {
                cache
                    .advance_len(rows)
                    .expect("unbounded/planned KvCache growth is infallible");
            } else {
                // The host rows, from the mirror the run appended to:
                // the host cache stays the authority, and the mirror
                // stays exactly its rows for the decode after this.
                let prior = cache.rows();
                let mirror = cache
                    .metal_attn
                    .as_ref()
                    .expect("the run appended to this layer's mirror");
                let (k, v) = mirror.tokens_host(prior, rows);
                let width = k.len() / rows;
                for (k, v) in k.chunks_exact(width).zip(v.chunks_exact(width)) {
                    cache
                        .push(k, v)
                        .expect("unbounded/planned KvCache growth is infallible");
                }
            }
            self.layer_for(l).moe.record_activations_dense();
        }
        *hidden = out;
        Some(end)
    }

    /// Whether any layer is a gated delta-net: the only models a
    /// prefill run serves.
    fn has_gdn_layers(&self) -> bool {
        (0..self.config.n_layers)
            .any(|l| matches!(self.config.layer_shape(l).attention, AttnShape::Gdn))
    }
}

#[cfg(test)]
mod tests {
    use crate::fused_layer::LayerFfnParts;
    use frink_core::attention::{apply_rope, causal_gqa_attention_row};
    use frink_core::matmul::{rms_norm, rms_norm_per_head};
    use frink_core::weight_matrix::hadamard::{FoldSite, HadamardFold};
    use frink_core::weight_matrix::{QuantKind, WeightBytes};
    use frink_core::WeightMatrix;
    use frink_metal::attn::{LayerRope, MetalKvBuffers, MetalKvDtype, MetalRope, MetalRopeLayout};
    use frink_metal::gdn_prefill::PrefillRunLayer;
    use frink_metal::gdn_prefill_attn::PrefillAttnLayer;
    use std::sync::Arc;

    /// A Qwen3.5 attention layer whole in the prefill run against the
    /// host body the decoder runs otherwise: `attn_norm`, the folded
    /// Q/K/V, the interleaved gate split, the per-head QK norms, a
    /// PARTIAL NeoX RoPE, causal attention over earlier rows already in
    /// the mirror and this batch, the gate, the folded `wo` and the
    /// FFN. The batch starts at position 8, so a RoPE or a KV offset
    /// read from zero is wrong here, and the mirror's new rows are
    /// checked against the host's roped K and V.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn an_attention_layer_in_the_run_matches_the_host_body() {
        let (n_embd, ffn_dim) = (256usize, 384usize);
        let (n_heads, n_kv, hd, rot) = (4usize, 2usize, 128usize, 32usize);
        let (prior, rows) = (8usize, 40usize);
        let (eps, theta) = (1e-6f32, 10_000.0f32);
        let (q_width, kv_width) = (n_heads * hd, n_kv * hd);
        let mut seed = 77_031u32;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
                })
                .collect()
        };
        let q8 = |rows: usize, cols: usize, v: Vec<f32>| WeightMatrix::Quantized {
            data: WeightBytes::Owned(frink_quant::quantize_q8_0(&v)),
            rows,
            cols,
            kind: QuantKind::Q8_0,
        };
        let mut fold = |block: usize, n: usize| {
            let signs: Arc<[f32]> = rnd(n, 2.0)
                .iter()
                .map(|x| if *x < 0.0 { -1.0f32 } else { 1.0 })
                .collect::<Vec<f32>>()
                .into();
            Arc::new(HadamardFold {
                block,
                signs: Some(signs),
                perm: None,
                site: FoldSite::Input,
            })
        };
        let (fold_x, fold_o, fold_ffn, fold_act) = (
            fold(64, n_embd),
            fold(128, q_width),
            fold(64, n_embd),
            fold(128, ffn_dim),
        );
        let mut wq = q8(2 * q_width, n_embd, rnd(2 * q_width * n_embd, 0.2));
        let mut wk = q8(kv_width, n_embd, rnd(kv_width * n_embd, 0.2));
        let mut wv = q8(kv_width, n_embd, rnd(kv_width * n_embd, 0.2));
        let mut wo = q8(n_embd, q_width, rnd(n_embd * q_width, 0.2));
        let mut gate = q8(ffn_dim, n_embd, rnd(ffn_dim * n_embd, 0.2));
        let mut up = q8(ffn_dim, n_embd, rnd(ffn_dim * n_embd, 0.2));
        let mut down = q8(n_embd, ffn_dim, rnd(n_embd * ffn_dim, 0.2));
        for m in [&mut wq, &mut wk, &mut wv] {
            m.fold_hadamard(fold_x.clone());
        }
        wo.fold_hadamard(fold_o.clone());
        gate.fold_hadamard(fold_ffn.clone());
        up.fold_hadamard(fold_ffn);
        down.fold_hadamard(fold_act);
        let attn_norm: Vec<f32> = rnd(n_embd, 0.5).iter().map(|x| 1.0 + x).collect();
        let ffn_norm: Vec<f32> = rnd(n_embd, 0.5).iter().map(|x| 1.0 + x).collect();
        let q_norm: Vec<f32> = rnd(hd, 0.5).iter().map(|x| 1.0 + x).collect();
        let k_norm: Vec<f32> = rnd(hd, 0.5).iter().map(|x| 1.0 + x).collect();
        let hidden = rnd(rows * n_embd, 1.0);
        // Earlier rows as the mirror holds them: f16.
        let f16 = |v: Vec<f32>| -> Vec<f32> {
            v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect()
        };
        let (k0, v0) = (
            f16(rnd(prior * kv_width, 2.0)),
            f16(rnd(prior * kv_width, 2.0)),
        );

        // The host body, in the decoder's order.
        let normed: Vec<f32> = hidden
            .chunks(n_embd)
            .flat_map(|r| rms_norm(r, &attn_norm, eps))
            .collect();
        let (mut q, g) = crate::attn_gate::split_interleaved_q_gate(
            &wq.apply_batch(&normed, rows),
            rows,
            n_heads,
            hd,
        );
        let mut k = wk.apply_batch(&normed, rows);
        let v = f16(wv.apply_batch(&normed, rows));
        q = rms_norm_per_head(&q, &q_norm, hd, eps);
        k = rms_norm_per_head(&k, &k_norm, hd, eps);
        for (b, (qr, kr)) in q
            .chunks_mut(q_width)
            .zip(k.chunks_mut(kv_width))
            .enumerate()
        {
            for head in qr.chunks_mut(hd).chain(kr.chunks_mut(hd)) {
                apply_rope(&mut head[..rot], prior + b, theta);
            }
        }
        let k = f16(k);
        let (kc, vc) = ([&k0[..], &k[..]].concat(), [&v0[..], &v[..]].concat());
        let mut branch: Vec<f32> = (0..rows)
            .flat_map(|b| {
                let n = prior + b + 1;
                causal_gqa_attention_row(
                    &q[b * q_width..(b + 1) * q_width],
                    &kc[..n * kv_width],
                    &vc[..n * kv_width],
                    n_heads,
                    n_kv,
                    hd,
                    hd,
                    n,
                    None,
                    None,
                    None,
                    None,
                )
            })
            .collect();
        crate::attn_gate::apply_interleaved_gate(&mut branch, &g);
        let mut want = hidden.clone();
        for (r, p) in want.iter_mut().zip(wo.apply_batch(&branch, rows)) {
            *r += p;
        }
        let n2: Vec<f32> = want
            .chunks(n_embd)
            .flat_map(|r| rms_norm(r, &ffn_norm, eps))
            .collect();
        let (gg, uu) = (gate.apply_batch(&n2, rows), up.apply_batch(&n2, rows));
        let act: Vec<f32> = gg
            .iter()
            .zip(&uu)
            .map(|(a, b)| a / (1.0 + (-a).exp()) * b)
            .collect();
        for (r, d) in want.iter_mut().zip(down.apply_batch(&act, rows)) {
            *r += d;
        }

        // The run.
        let ffn = LayerFfnParts::from_parts(&ffn_norm, eps, &gate, &up, &down);
        fn launch(m: &WeightMatrix) -> frink_metal::gpu::MulMmSgLaunch<'_> {
            m.launch_parts()
                .0
                .mul_mm_sg_launch()
                .expect("Q8_0 has a GEMM")
        }
        let layer = PrefillAttnLayer {
            attn_norm: &attn_norm,
            eps,
            fold_x: fold_x.metal_plan(n_embd),
            q: launch(&wq),
            k: launch(&wk),
            v: launch(&wv),
            n_heads,
            q_gate: true,
            extras: frink_metal::attn::AttnExtras {
                q_bias: None,
                k_bias: None,
                v_bias: None,
                q_norm: Some(&q_norm),
                k_norm: Some(&k_norm),
                v_norm: None,
                attn_logit_softcap: None,
                sinks: None,
                o_bias: None,
            },
            rope: MetalRope {
                layout: MetalRopeLayout::Neox,
                rot_dim: Some(rot),
                attn_factor: 1.0,
            },
            layer_rope: Some(LayerRope {
                theta,
                freq_factors: None,
            }),
            o_proj: launch(&wo),
            fold_branch: fold_o.metal_plan(q_width),
            ffn: ffn.prefill_launches().expect("Q8_0 has a GEMM"),
        };
        let mut kv =
            MetalKvBuffers::with_capacity_dtype(n_kv, hd, 64, MetalKvDtype::F16).expect("a mirror");
        kv.upload_from_host(&k0, &v0, prior)
            .expect("the prior rows");
        assert!(layer.is_supported(&kv, rows, prior));
        // SAFETY: no recurrent layer, so no state pointer is handed over.
        let got = unsafe {
            frink_metal::gdn_prefill::launch_gdn_prefill_run(
                &hidden,
                rows,
                prior,
                None,
                &mut [PrefillRunLayer::Attn(layer, &mut kv)],
            )
        }
        .expect("the run launches");
        assert_eq!(kv.seq_len, prior + rows, "the run advanced the mirror");

        let close = |what: &str, d: &[f32], w: &[f32], tol: f32| {
            assert_eq!(d.len(), w.len(), "{what}: length");
            let scale = w.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
            for (i, (x, y)) in d.iter().zip(w).enumerate() {
                assert!(
                    (x - y).abs() <= tol * scale,
                    "{what}[{i}]: device {x} host {y}"
                );
            }
        };
        let (dk, dv) = kv.tokens_host(prior, rows);
        close("k", &dk, &k, 2e-3);
        close("v", &dv, &v, 2e-3);
        close("out", &got, &want, 3e-3);
    }
}
