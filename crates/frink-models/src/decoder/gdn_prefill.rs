//! A run of PREFILL layers in one Metal submission, or nothing
//! (issue #540): recurrent layers of every kind the device serves (the
//! gated delta-net, Mamba-1, Mamba-2, LFM2's short convolution), and
//! the attention layers between them whole, so a hybrid's prompt batch
//! is one command buffer.
//!
//! The decode path's [`Decoder::fused_layer_parts`] says which gated
//! delta-net layers qualify, and [`Decoder::fused_attention_tail_eligible`]
//! which attention tails may lead a run, so prefill and decode cannot
//! disagree about what a fusable layer is; an attention layer rides
//! whole when [`Decoder::prefill_attn_layer`] admits it as well, and a
//! Mamba or conv layer when `decoder::ssm_prefill` does.
//! `frink_metal::prefill_run` runs them with the hidden batch resident
//! across the run.

use super::fused_view::FusedAttnCaps;
use super::{Decoder, LayerWeights};
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;
use frink_core::kv_store::KvRows;
use frink_core::KvCache;
use frink_metal::gdn_prefill_attn::PrefillAttnLayer;
use frink_metal::prefill_run::PrefillRunLayer;

/// An attention layer a device run serves whole, before its kernels
/// are chosen ([`Decoder::resident_attn_parts`]).
pub(super) struct ResidentAttnParts<'a> {
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) attn_norm: &'a [f32],
    pub(super) eps: f32,
    pub(super) fold_x: Option<frink_metal::hadamard::FoldPlan<'a>>,
    pub(super) q: &'a frink_core::WeightMatrix,
    pub(super) k: &'a frink_core::WeightMatrix,
    pub(super) v: &'a frink_core::WeightMatrix,
    pub(super) q_gate: bool,
    pub(super) extras: frink_metal::attn::AttnExtras<'a>,
    pub(super) rope: frink_metal::attn::MetalRope,
    pub(super) layer_rope: Option<frink_metal::attn::LayerRope<'a>>,
    pub(super) q_scale: Option<f32>,
    pub(super) o: &'a frink_core::WeightMatrix,
    pub(super) fold_branch: Option<frink_metal::hadamard::FoldPlan<'a>>,
    pub(super) ffn: LayerFfnParts<'a>,
}

/// Which run asks [`Decoder::resident_attn_parts`]: the decode run
/// applies neither Granite's scales nor a parallel block; the prefill
/// run applies all three.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum RunKind {
    Decode,
    Prefill,
}

/// One layer of a run as planned, before any cache is touched.
pub(super) enum Planned<'a> {
    Gdn(&'a crate::gdn::Gdn, &'a [f32], Box<LayerFfnParts<'a>>),
    Attn(Box<PrefillAttnLayer<'a>>),
    /// An attention layer and its parallel Mamba-2 block.
    Parallel(Box<PrefillAttnLayer<'a>>, &'a crate::mamba2::Mamba2),
    /// A Mamba or short-conv layer (`decoder::ssm_prefill`).
    Ssm(Box<super::ssm_prefill::SsmPlan<'a>>),
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
        let ffn = self.run_ffn_parts(l, layer)?;
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
        Self::record_fused_ffn(layer);
        Some(end)
    }

    /// Below its device crossover the host body is the faster one: the
    /// chunked recurrence's for a gated delta-net model, where
    /// `delta_chunk_rows` would have sent these rows anyway, and the
    /// device scan's for the Mamba and conv models.
    fn prefill_rows_take_device(&self, rows: usize) -> bool {
        let min = if self.has_gdn_layers() {
            frink_core::gdn_chunk::DEVICE_ROWS
        } else {
            frink_core::mamba2::DEVICE_SCAN_MIN_ROWS
        };
        rows >= min && frink_core::weight_matrix::metal_dense_enabled()
    }

    /// Attention layer `l` whole on the device, with its KV head count,
    /// or `None` and it is the host body's. Asked BEFORE any cache is
    /// touched; the mirror's own fit is
    /// [`PrefillAttnLayer::is_supported`]'s, once it exists.
    fn prefill_attn_layer(
        &self,
        l: usize,
        start_pos: usize,
        cache: &KvCache,
    ) -> Option<(PrefillAttnLayer<'_>, usize)> {
        let p = self.resident_attn_parts(l, start_pos, cache, RunKind::Prefill)?;
        let attn = PrefillAttnLayer {
            attn_norm: p.attn_norm,
            eps: p.eps,
            fold_x: p.fold_x,
            q: p.q.mul_mm_sg_launch()?,
            k: p.k.mul_mm_sg_launch()?,
            v: p.v.mul_mm_sg_launch()?,
            n_heads: p.n_heads,
            q_gate: p.q_gate,
            extras: p.extras,
            rope: p.rope,
            layer_rope: p.layer_rope,
            q_scale: p.q_scale,
            o_proj: p.o.mul_mm_sg_launch()?,
            fold_branch: p.fold_branch,
            ffn: p.ffn.prefill_launches()?,
        };
        Some((attn, p.n_kv_heads))
    }

    /// Attention layer `l` as a device RUN takes it whole -- the prefill
    /// run ([`Self::prefill_attn_layer`]) and the decode run
    /// (`Decoder::decode_attn_layer`) -- before either picks its
    /// kernels: GEMMs for a batch, matvecs for a token. ONE set of
    /// refusals, so the two cannot disagree about which layers ride.
    ///
    /// Everything the host body applies between the pre-norm and `wo`
    /// is either here or a refusal: the attention facts every fused
    /// launch shares ([`Self::fused_attn_layer_ok`], under caps that
    /// admit the interleaved Q gate), the tail's
    /// ([`Self::fused_attention_tail_eligible`], or for the prefill run
    /// [`Self::prefill_run_tail_eligible`], which applies Granite's
    /// residual multiplier and runs Falcon-H1's parallel block) and the
    /// device attention's ([`Self::device_attention_shape_ok`]). The
    /// host KV must be f16 with no row evicted, so the mirror the run
    /// appends to and the host rows written back from it are the same
    /// rows.
    pub(super) fn resident_attn_parts(
        &self,
        l: usize,
        start_pos: usize,
        cache: &KvCache,
        kind: RunKind,
    ) -> Option<ResidentAttnParts<'_>> {
        let layer = self.layer_for(l);
        let AttnShape::Gqa {
            n_heads,
            n_kv_heads,
        } = self.config.layer_shape(l).attention
        else {
            return None;
        };
        let prefill = kind == RunKind::Prefill;
        let caps = FusedAttnCaps {
            attention_scale: prefill,
            ..FusedAttnCaps::METAL_PREFILL_RUN
        };
        let beside = layer.attn.ssm.is_some();
        let model_ok = if prefill {
            self.has_run_layers()
                && Self::metal_can_serve_scaled_layers(&self.config, self.lora_attached())
        } else {
            self.has_gdn_layers()
                && Self::metal_can_serve_layers(&self.config, self.lora_attached())
        };
        let tail_ok = if prefill {
            self.prefill_run_tail_eligible(l, layer)
        } else {
            self.fused_attention_tail_eligible(l, layer)
        };
        let layer_ok = if beside {
            self.fused_attn_layer_ok_beside_ssm(layer, caps)
        } else {
            self.fused_attn_layer_ok(layer, caps)
        };
        if !model_ok
            || !layer_ok
            || !tail_ok
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
        let extras = if beside {
            self.metal_attn_view_from(Self::fused_attn_extras_beside_ssm(layer, caps)?)
        } else {
            self.metal_attn_view_under(layer, caps)?
        };
        Some(ResidentAttnParts {
            n_heads,
            n_kv_heads,
            attn_norm,
            eps,
            fold_x,
            q,
            k,
            v,
            q_gate: layer.attn.q_gate_interleaved,
            extras,
            rope: self.metal_rope(),
            layer_rope: self.metal_layer_rope(l),
            q_scale: if prefill { self.metal_q_scale() } else { None },
            o,
            fold_branch,
            ffn: self.run_ffn_parts(l, layer)?,
        })
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
        if !self.prefill_rows_take_device(rows) || !self.has_run_layers() {
            return None;
        }
        let eps = self.config.rms_norm_eps;
        let head_dim = self.config.head_dim;
        let mut plan = Vec::new();
        let mut end = start;
        while end < run_end.min(kv_caches.len()) {
            if let Some((gdn, attn_norm, ffn)) = self.fused_layer_parts(end) {
                plan.push(Planned::Gdn(gdn, attn_norm, Box::new(ffn)));
            } else if let Some(p) = self.ssm_prefill_plan(end) {
                plan.push(Planned::Ssm(Box::new(p)));
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
                let layer = self.layer_for(end);
                plan.push(match layer.attn.ssm.as_ref() {
                    None => Planned::Attn(Box::new(a)),
                    Some(crate::ssm_block::SsmBlock::Mamba2(m)) => {
                        Planned::Parallel(Box::new(a), m)
                    }
                    Some(_) => break,
                });
            } else {
                break;
            }
            end += 1;
        }
        if plan.is_empty() && head.is_none() {
            return None;
        }
        // Each conv layer's window in, and its new rows out, owned here
        // so the run can borrow them.
        let mut conv_io = self.ssm_prefill_conv_io(&plan, &kv_caches[start..end], rows);
        let mut entries = Vec::with_capacity(plan.len());
        let mut io = conv_io.iter_mut();
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
                Planned::Parallel(a, m) => {
                    let state = cache.recurrent.get_or_insert_with(|| m.zero_state());
                    // SAFETY: as `ssm_prefill_layer`'s: this cache's own
                    // page-aligned state, borrowed `&mut` for the call.
                    let block = unsafe { self.prefill_mamba2_block(m, state) }?;
                    PrefillRunLayer::Parallel(*a, Box::new(block), cache.metal_attn.as_mut()?)
                }
                Planned::Ssm(s) => self.ssm_prefill_layer(*s, cache, &mut io)?,
            });
        }
        // SAFETY: each state is page-aligned (`AlignedF32`) and lives in
        // its own cache element, borrowed mutably through `kv_caches`
        // for this whole call and touched by nothing else; the launch
        // waits for the GPU before returning.
        let out = unsafe {
            frink_metal::prefill_run::launch_prefill_run(
                hidden,
                rows,
                start_pos,
                head,
                &mut entries,
                frink_metal::gdn_prefill::RunOptions {
                    residual_scale: self.config.residual_scale,
                },
            )
        }
        .ok()?;
        drop(entries);
        let mut io = conv_io.into_iter();
        for (l, cache) in (start..end).zip(kv_caches[start..end].iter_mut()) {
            match self.config.layer_shape(l).attention {
                AttnShape::Gdn | AttnShape::Mamba1 | AttnShape::Mamba2 => cache
                    .advance_len(rows)
                    .expect("unbounded/planned KvCache growth is infallible"),
                AttnShape::ShortConv => {
                    // The history the conv appended, one `b * x` row per
                    // token, as `contiguous_step` pushes it on the host.
                    let (_, bx) = io.next().expect("one io per conv layer");
                    cache
                        .push_rows(KvRows::F32(&bx), KvRows::F32(&[]), rows)
                        .expect("unbounded/planned KvCache growth is infallible");
                }
                _ => {
                    // The host rows, from the mirror the run appended to:
                    // the host cache stays the authority, and the mirror
                    // stays exactly its rows for the decode after this.
                    let prior = cache.rows();
                    // Out of the cache for the copy, which borrows it.
                    let mirror = cache
                        .metal_attn
                        .take()
                        .expect("the run appended to this layer's mirror");
                    Self::push_rows_from_metal(&mirror, cache, prior, rows);
                    cache.metal_attn = Some(mirror);
                }
            }
            // A routed layer chose its experts on the device; like the
            // dense prefill stack's MoE, nothing reads them back.
            Self::record_fused_ffn(self.layer_for(l));
        }
        *hidden = out;
        Some(end)
    }

    /// Whether any layer is a gated delta-net.
    fn has_gdn_layers(&self) -> bool {
        (0..self.config.n_layers)
            .any(|l| matches!(self.config.layer_shape(l).attention, AttnShape::Gdn))
    }

    /// Whether any layer is recurrent in a way a prefill run serves (or
    /// carries a parallel Mamba-2 block): the only models it serves. A
    /// dense model's attention layers are the dense stacks'.
    fn has_run_layers(&self) -> bool {
        (0..self.config.n_layers).any(|l| {
            matches!(
                self.config.layer_shape(l).attention,
                AttnShape::Gdn | AttnShape::Mamba1 | AttnShape::Mamba2 | AttnShape::ShortConv
            ) || self.layer_for(l).attn.ssm.is_some()
        })
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
    use frink_metal::gdn_prefill_attn::PrefillAttnLayer;
    use frink_metal::prefill_run::PrefillRunLayer;
    use std::sync::Arc;

    /// One Qwen3.5 attention layer and what the host body makes of
    /// `rows` tokens from position 8 through it: the case both device
    /// runs are held to, so the prefill and the decode test cannot test
    /// different layers.
    struct Case {
        n_embd: usize,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        rot: usize,
        prior: usize,
        rows: usize,
        eps: f32,
        theta: f32,
        q_width: usize,
        wq: WeightMatrix,
        wk: WeightMatrix,
        wv: WeightMatrix,
        wo: WeightMatrix,
        gate: WeightMatrix,
        up: WeightMatrix,
        down: WeightMatrix,
        fold_x: Arc<HadamardFold>,
        fold_o: Arc<HadamardFold>,
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        hidden: Vec<f32>,
        /// The mirror's earlier rows, f16 as it holds them.
        k0: Vec<f32>,
        v0: Vec<f32>,
        /// The host's roped K and V for the new rows, and its output.
        k: Vec<f32>,
        v: Vec<f32>,
        want: Vec<f32>,
    }

    fn case(rows: usize) -> Case {
        let (n_embd, ffn_dim) = (256usize, 384usize);
        let (n_heads, n_kv, hd, rot) = (4usize, 2usize, 128usize, 32usize);
        let prior = 8usize;
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

        Case {
            n_embd,
            n_heads,
            n_kv,
            hd,
            rot,
            prior,
            rows,
            eps,
            theta,
            q_width,
            wq,
            wk,
            wv,
            wo,
            gate,
            up,
            down,
            fold_x,
            fold_o,
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            hidden,
            k0,
            v0,
            k,
            v,
            want,
        }
    }

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
        let Case {
            n_embd,
            n_heads,
            n_kv,
            hd,
            rot,
            prior,
            rows,
            eps,
            theta,
            q_width,
            wq,
            wk,
            wv,
            wo,
            gate,
            up,
            down,
            fold_x,
            fold_o,
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            hidden,
            k0,
            v0,
            k,
            v,
            want,
        } = case(40);
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
            q_scale: None,
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
            frink_metal::prefill_run::launch_prefill_run(
                &hidden,
                rows,
                prior,
                None,
                &mut [PrefillRunLayer::Attn(layer, &mut kv)],
                frink_metal::gdn_prefill::RunOptions::default(),
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

    /// The same layer as a DECODE run takes it (`GdnRun::attn_layer_resident`),
    /// one token at position 8: matvecs instead of GEMMs, and nothing
    /// returned to the host between the projections and the FFN. The
    /// mirror's new row is checked against the host's roped K and V, so
    /// a RoPE at position zero, a gate left unsplit or an append at the
    /// wrong offset is red here.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn an_attention_layer_in_a_decode_run_matches_the_host_body() {
        let Case {
            n_embd,
            n_heads,
            n_kv,
            hd,
            rot,
            prior,
            eps,
            theta,
            q_width,
            wq,
            wk,
            wv,
            wo,
            gate,
            up,
            down,
            fold_x,
            fold_o,
            attn_norm,
            ffn_norm,
            q_norm,
            k_norm,
            hidden,
            k0,
            v0,
            k,
            v,
            want,
            ..
        } = case(1);
        let ffn = LayerFfnParts::from_parts(&ffn_norm, eps, &gate, &up, &down);
        let ffn = ffn.launches().expect("Q8_0 has a matvec");
        let ffn = ffn.as_metal();
        fn mv(m: &WeightMatrix) -> frink_metal::gpu::MatvecLaunch<'_> {
            crate::metal_launch::matvec(m.launch_parts().0).expect("Q8_0 has a matvec")
        }
        let (q, kp, vp, o) = (mv(&wq), mv(&wk), mv(&wv), mv(&wo));
        let extras = frink_metal::attn::AttnExtras {
            q_norm: Some(&q_norm),
            k_norm: Some(&k_norm),
            ..Default::default()
        };
        let (fx, fo) = (
            fold_x.metal_plan(n_embd).expect("a plan"),
            fold_o.metal_plan(q_width).expect("a plan"),
        );
        let layer = frink_metal::gdn_decode_attn::DecodeAttnLayer {
            attn_norm: &attn_norm,
            eps,
            fold_x: Some(&fx),
            q: &q,
            k: &kp,
            v: &vp,
            n_heads,
            q_gate: true,
            extras: &extras,
            rope: MetalRope {
                layout: MetalRopeLayout::Neox,
                rot_dim: Some(rot),
                attn_factor: 1.0,
            },
            layer_rope: Some(LayerRope {
                theta,
                freq_factors: None,
            }),
            o_proj: &o,
            fold_branch: Some(&fo),
            ffn: &ffn,
        };
        let mut kv =
            MetalKvBuffers::with_capacity_dtype(n_kv, hd, 64, MetalKvDtype::F16).expect("a mirror");
        kv.upload_from_host(&k0, &v0, prior)
            .expect("the prior rows");
        assert!(layer.is_supported(&kv, prior));
        assert!(
            !layer.is_supported(&kv, prior + 1),
            "a position past the mirror"
        );
        let mut run = frink_metal::gdn_branch::GdnRun::start(&hidden).expect("a run");
        // SAFETY: the mirror is this test's and outlives `finish`.
        unsafe { run.attn_layer_resident(&layer, &mut kv, prior) }.expect("the layer encodes");
        let got = run.finish().expect("the run completes");
        assert_eq!(kv.seq_len, prior + 1, "the run advanced the mirror");
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
        let (dk, dv) = kv.tokens_host(prior, 1);
        close("k", &dk, &k, 2e-3);
        close("v", &dv, &v, 2e-3);
        close("out", &got, &want, 3e-3);
    }

    /// The output head at the end of a decode run IS the host head:
    /// `final_norm`, the rotation a folded `output` reads its input in,
    /// and the projection. A fold skipped or applied to the unnormed
    /// stream is visible here, since the input is not unit scale.
    #[test]
    #[ignore = "needs a real Metal-capable GPU; run manually with --ignored on Apple Silicon"]
    fn the_output_head_in_a_decode_run_matches_the_host_head() {
        let (n_embd, vocab, eps) = (256usize, 1000usize, 1e-6f32);
        let mut seed = 4_711u32;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((seed >> 9) as f32 / (1u32 << 23) as f32) - 0.5) * scale
                })
                .collect()
        };
        let signs: Arc<[f32]> = rnd(n_embd, 2.0)
            .iter()
            .map(|x| if *x < 0.0 { -1.0f32 } else { 1.0 })
            .collect::<Vec<f32>>()
            .into();
        let fold = Arc::new(HadamardFold {
            block: 64,
            signs: Some(signs),
            perm: None,
            site: FoldSite::Input,
        });
        let mut output = WeightMatrix::Quantized {
            data: WeightBytes::Owned(frink_quant::quantize_q8_0(&rnd(vocab * n_embd, 0.2))),
            rows: vocab,
            cols: n_embd,
            kind: QuantKind::Q8_0,
        };
        output.fold_hadamard(fold);
        let norm: Vec<f32> = rnd(n_embd, 0.5).iter().map(|x| 1.0 + x).collect();
        let hidden = rnd(n_embd, 8.0);
        // The host head, as `Decoder::logits_from_normed` projects.
        let want = output.apply(&rms_norm(&hidden, &norm, eps));

        let (base, f) = output.launch_parts();
        let launch = crate::metal_launch::matvec(base).expect("Q8_0 has a matvec");
        let plan = f.expect("folded above").metal_plan(n_embd).expect("a plan");
        let mut run = frink_metal::gdn_branch::GdnRun::start(&hidden).expect("a run");
        run.output_head(&norm, eps, &launch, Some(&plan))
            .expect("the head encodes");
        let got = run
            .finish_with_head()
            .expect("the run completes")
            .logits
            .expect("the head's logits come back");
        assert_eq!(got.len(), vocab);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0);
        for (i, (d, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (d - w).abs() <= 2e-3 * scale,
                "logit[{i}]: device {d} host {w}"
            );
        }
    }
}
