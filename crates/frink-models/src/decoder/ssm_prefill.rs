//! Which Mamba and short-convolution layers a Metal prefill run serves
//! (`decoder::gdn_prefill`, `frink_metal::ssm_prefill`), and the
//! hand-over of their state.
//!
//! The arithmetic is the decode stack's blocks over a batch
//! (`frink_metal::ssm_decode`), and the refusals are the hybrid decode
//! stack's (`decoder::metal_hybrid`) where the two ask the same
//! question, so a layer the decode stack runs on the device is one the
//! prefill run takes too: a Mamba block's GEMMs instead of its matvecs,
//! a model with nothing either launch leaves out.

use frink_core::cache::KvCache;
use frink_core::recurrent_state::RecurrentState;
use frink_core::WeightMatrix;
use frink_metal::gpu::MulMmSgLaunch;
use frink_metal::prefill_run::PrefillRunLayer;
use frink_metal::ssm_decode::{Mamba1Mixer, Mamba1Weights, Mamba2Mixer, Mamba2Weights};
use frink_metal::ssm_prefill::{PrefillConvLayer, PrefillConvState, PrefillMambaLayer};

use super::gdn_prefill::Planned;
use super::Decoder;
use crate::fused_layer::LayerFfnParts;
use crate::layer_shapes::AttnShape;
use crate::ssm_block::SsmBlock;

/// One Mamba or conv layer as planned.
pub(super) struct SsmPlan<'a> {
    attn_norm: &'a std::sync::Arc<[f32]>,
    ffn: Option<LayerFfnParts<'a>>,
    block: SsmPlanBlock<'a>,
}

#[derive(Clone, Copy)]
enum SsmPlanBlock<'a> {
    Conv(&'a crate::shortconv::ShortConv),
    Mamba1(&'a crate::mamba1::Mamba1),
    Mamba2(&'a crate::mamba2::Mamba2),
}

/// A conv layer's window in, in layer order.
pub(super) type ConvIo = Vec<f32>;

/// The GEMM a projection runs as, or `None` for a folded matrix (no
/// Mamba or conv projection carries a rotation, and the blocks apply
/// none) or a kind with no GEMM.
fn gemm(m: &WeightMatrix) -> Option<MulMmSgLaunch<'_>> {
    let (base, fold) = m.launch_parts();
    if fold.is_some() {
        return None;
    }
    base.mul_mm_sg_launch()
}

impl Decoder {
    /// Layer `l` when it is a Mamba or short-conv layer the prefill run
    /// serves, or `None` and the run ends before it.
    pub(super) fn ssm_prefill_plan(&self, l: usize) -> Option<SsmPlan<'_>> {
        let layer = self.layer_for(l);
        let shape = self.config.layer_shape(l);
        if !matches!(
            shape.attention,
            AttnShape::ShortConv | AttnShape::Mamba1 | AttnShape::Mamba2
        ) || !Self::metal_can_serve_scaled_layers(&self.config, self.lora_attached())
            || self.gpt_oss.is_some()
            // `lfm3.cpp:178-181`'s norm on the conv output, and
            // `plamo2.cpp:150`'s on a block's: no run kernel norms the
            // branch.
            || layer.attn.post_attn_norm.is_some()
        {
            return None;
        }
        let attn_norm = layer.attn.norm_weight.rms_weights()?;
        let ffn = if shape.ffn_dim == 0 {
            None
        } else {
            if !self.config.layer_ffn_acts(l).all_swiglu() {
                return None;
            }
            let parts = self.run_ffn_parts(l, layer)?;
            parts.prefill_launches()?;
            Some(parts)
        };
        let block = match shape.attention {
            AttnShape::ShortConv => {
                let c = layer.attn.shortconv.as_ref()?;
                gemm(&c.in_proj)?;
                gemm(&c.out_proj)?;
                SsmPlanBlock::Conv(c)
            }
            AttnShape::Mamba1 => {
                let Some(SsmBlock::Mamba1(m)) = layer.attn.ssm.as_ref() else {
                    return None;
                };
                for w in [&m.in_proj, &m.x_proj, &m.dt_proj, &m.out_proj] {
                    gemm(w)?;
                }
                SsmPlanBlock::Mamba1(m)
            }
            _ => {
                let Some(SsmBlock::Mamba2(m)) = layer.attn.ssm.as_ref() else {
                    return None;
                };
                gemm(&m.in_proj)?;
                gemm(&m.out_proj)?;
                SsmPlanBlock::Mamba2(m)
            }
        };
        Some(SsmPlan {
            attn_norm,
            ffn,
            block,
        })
    }

    /// Each planned conv layer's window (the last `l_cache - 1` rows of
    /// its history, zero rows before the sequence, as `contiguous_step`
    /// reads it), in plan order.
    pub(super) fn ssm_prefill_conv_io(
        &self,
        plan: &[Planned<'_>],
        caches: &[KvCache],
    ) -> Vec<ConvIo> {
        plan.iter()
            .zip(caches)
            .filter_map(|(p, cache)| match p {
                Planned::Ssm(s) => {
                    let SsmPlanBlock::Conv(c) = s.block else {
                        return None;
                    };
                    let n = c.hidden_dim();
                    let history = cache
                        .k
                        .as_f32()
                        .expect(super::recurrent_block::SHORT_CONV_HISTORY_IS_F32);
                    let have = cache.rows();
                    let window =
                        crate::shortconv::window_from_history(c.l_cache - 1, n, have, |r| {
                            &history[r * n..(r + 1) * n]
                        });
                    Some(window)
                }
                _ => None,
            })
            .collect()
    }

    /// The planned layer as the run takes it, on `cache`'s state; a conv
    /// layer takes the next of `io`.
    pub(super) fn ssm_prefill_layer<'a, 'k>(
        &'a self,
        s: SsmPlan<'a>,
        cache: &'k mut KvCache,
        io: &mut std::slice::IterMut<'k, ConvIo>,
    ) -> Option<PrefillRunLayer<'a, 'k>> {
        let eps = self.config.rms_norm_eps;
        let ffn = match &s.ffn {
            None => None,
            Some(f) => Some(f.prefill_launches()?),
        };
        Some(match s.block {
            SsmPlanBlock::Conv(c) => {
                let window = io.next()?;
                // The new history rows go straight from the run's buffer
                // into this layer's cache, as `contiguous_step` pushes
                // them on the host, one `b * x` row per token.
                let n = c.hidden_dim();
                let bx_out = Box::new(move |bx: &[f32]| {
                    let rows = bx.len() / n;
                    cache
                        .push_rows(
                            frink_core::kv_store::KvRows::F32(bx),
                            frink_core::kv_store::KvRows::F32(&[]),
                            rows,
                        )
                        .expect("unbounded/planned KvCache growth is infallible");
                });
                PrefillRunLayer::Conv(
                    PrefillConvLayer {
                        attn_norm: s.attn_norm,
                        eps,
                        in_proj: gemm(&c.in_proj)?,
                        taps: &c.conv,
                        l_cache: c.l_cache,
                        out_proj: gemm(&c.out_proj)?,
                        ffn,
                    },
                    PrefillConvState { window, bx_out },
                )
            }
            SsmPlanBlock::Mamba1(m) => {
                let state = cache.recurrent.get_or_insert_with(|| m.zero_state());
                let w = Mamba1Weights {
                    in_proj: gemm(&m.in_proj)?,
                    conv1d: &m.conv1d,
                    conv_bias: &m.conv1d_bias,
                    d_conv: m.h.d_conv,
                    x_proj: gemm(&m.x_proj)?,
                    dt_bc_norm: self.metal_dt_bc_norm(m),
                    dt_proj: gemm(&m.dt_proj)?,
                    dt_bias: &m.dt_bias,
                    a: &m.a,
                    d: &m.d,
                    out_proj: gemm(&m.out_proj)?,
                    d_inner: m.h.d_inner,
                    d_state: m.h.d_state,
                    dt_rank: m.h.dt_rank,
                };
                // SAFETY: this layer's `RecurrentState` buffers,
                // page-aligned `AlignedF32`s of whole pages sized by these
                // weights (`Mamba1::zero_state`), in a cache borrowed
                // `&mut` for the run, which waits for the GPU.
                let block = unsafe { Mamba1Mixer::new(w, state_conv(state), state_ssm(state)) }?;
                PrefillRunLayer::Mamba1(PrefillMambaLayer {
                    attn_norm: s.attn_norm,
                    eps,
                    block,
                    ffn,
                })
            }
            SsmPlanBlock::Mamba2(m) => {
                let state = cache.recurrent.get_or_insert_with(|| m.zero_state());
                // SAFETY: as the Mamba-1 arm's (`Mamba2::zero_state`).
                let block = unsafe { self.prefill_mamba2_block(m, state) }?;
                PrefillRunLayer::Mamba2(PrefillMambaLayer {
                    attn_norm: s.attn_norm,
                    eps,
                    block,
                    ffn,
                })
            }
        })
    }

    /// A Mamba-2 block with GEMMs, on `state`.
    ///
    /// # Safety
    ///
    /// `state` must be this block's (`Mamba2::zero_state`'s size) and
    /// stay untouched by anything else until the run it is handed to
    /// returns.
    pub(super) unsafe fn prefill_mamba2_block<'a>(
        &self,
        m: &'a crate::mamba2::Mamba2,
        state: &mut RecurrentState,
    ) -> Option<Mamba2Mixer<'a, MulMmSgLaunch<'a>>> {
        let w = Mamba2Weights {
            in_proj: gemm(&m.in_proj)?,
            conv1d: &m.conv1d,
            conv_bias: &m.conv1d_bias,
            d_conv: m.h.d_conv,
            dt_bias: &m.dt_bias,
            a: &m.a,
            d: &m.d,
            norm: m.norm.as_ref(),
            norm_eps: self.config.rms_norm_eps,
            out_proj: gemm(&m.out_proj)?,
            d_inner: m.h.d_inner,
            d_state: m.h.d_state,
            n_head: m.h.n_head,
            n_group: m.h.n_group,
        };
        // SAFETY: the caller's contract.
        unsafe { Mamba2Mixer::new(w, state_conv(state), state_ssm(state)) }
    }
}

fn state_conv(s: &mut RecurrentState) -> (*mut f32, usize) {
    (s.conv.as_ptr(), s.conv.alloc_bytes())
}

fn state_ssm(s: &mut RecurrentState) -> (*mut f32, usize) {
    (s.ssm.as_ptr(), s.ssm.alloc_bytes())
}

#[cfg(test)]
mod tests;
