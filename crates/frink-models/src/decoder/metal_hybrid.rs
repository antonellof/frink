//! A short-convolution hybrid's decode token as ONE Metal submission.
//!
//! LFM2 alternates conv layers and GQA layers around one residual
//! topology (`crate::shortconv`, `lfm2.cpp:192-208`). Until this, every
//! fused decode path refused the whole model because a conv layer is not
//! an attention layer, so a token went layer by layer through the host
//! bodies: two device round trips per conv layer, attention on the host,
//! one more round trip per FFN, about fifty waits a token. The dense
//! decode stack now takes the conv as a second token mixer
//! (`frink_metal::attn::TokenMixer`), so the token is one wait.
//!
//! # Who holds the state
//!
//! The HOST does, before and after every token, for both kinds of layer.
//! The conv history is the layer's `KvCache` (`crate::shortconv`): its
//! last `l_cache - 1` rows are copied into the device windows before the
//! launch and the row the stack appends is pushed back after it. The
//! attention layers' K/V rows are downloaded after the launch
//! (`catch_up_host_kv_from_metal`), so the device KV is a mirror the next
//! token can trust when its length agrees with the host's and rebuilds
//! from the host when it does not. Nothing a prefix restore, a rollback
//! or a host fallback does to the caches can leave the device holding
//! the authoritative copy of anything.

use frink_core::cache::KvCache;
use frink_metal::attn::{
    AttnMixer, DenseLayerMetal, LayerKv, MetalKvBuffers, ShortConvMixer, TokenMixer,
};
use frink_metal::shortconv::ConvWindows;
use frink_moe::GluAct;

use super::lm_head::FoldedLmHead;
use super::{Decoder, ExpertBacking};
use crate::layer_shapes::AttnShape;
use crate::shortconv::window_from_history;

/// The device half: every attention layer's KV mirror, in layer order,
/// and one window per conv layer.
pub(crate) struct HybridDeviceState {
    kvs: Vec<MetalKvBuffers>,
    /// The layer each of `kvs` mirrors.
    kv_layers: Vec<usize>,
    windows: ConvWindows,
}

/// Which layer is which, for a model the stack serves whole.
struct HybridPlan {
    /// Per layer: `Some(slot)` on an attention layer, `None` on a conv
    /// layer.
    slots: Vec<Option<usize>>,
    n_conv: usize,
    n_kv_heads: usize,
    l_cache: usize,
    gelu: bool,
}

/// What the stack handed back.
pub(super) enum HybridOut {
    /// The token's logits (or its argmax id), lm_head having run in the
    /// stack.
    Logits(Vec<f32>),
    /// The last hidden state, normed when the stack ran the final norm.
    Hidden { hidden: Vec<f32>, normed: bool },
}

impl Decoder {
    /// True once a decode token has been served by the hybrid stack: for
    /// tests that switch Metal on and must know the switch reached it,
    /// which logits alone cannot say (the host bodies agree with it).
    pub fn metal_hybrid_allocated(&self) -> bool {
        self.metal_hybrid
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Shortens the attention mirrors to the host caches' rows, as
    /// `rewind_device_kv` shortens the plain arena. A mirror longer than
    /// its host cache would be rebuilt whole on the next token anyway
    /// (its length disagrees); this keeps the rows a rollback did not
    /// touch.
    pub(super) fn rewind_metal_hybrid(&self, kv_caches: &[KvCache]) {
        let mut guard = self
            .metal_hybrid
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(state) = guard.as_mut() {
            for (mkv, &l) in state.kvs.iter_mut().zip(&state.kv_layers) {
                if let Some(cache) = kv_caches.get(l) {
                    // ROWS, as `rewind_device_kv`.
                    mkv.seq_len = mkv.seq_len.min(cache.rows());
                    mkv.host_rows_real = mkv.host_rows_real.min(mkv.seq_len);
                }
            }
        }
    }

    /// Drops the device half, for caches from another sequence
    /// (`forget_device_kv`).
    pub(super) fn forget_metal_hybrid(&self) {
        *self
            .metal_hybrid
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }

    /// The plan, or `None` when any layer is something the stack does
    /// not serve. A model with no conv layer is the plain dense stack's.
    fn metal_hybrid_plan(&self) -> Option<HybridPlan> {
        if !super::metal_eligibility::metal_attn_switched_on()
            || !Self::metal_can_serve_layers(&self.config, self.lora_attached())
            || self.gpt_oss.is_some()
        {
            return None;
        }
        let gelu = self
            .config
            .model_ffn_act()
            .and_then(GluAct::fused_kernel_gelu_flag)?;
        let mut slots = Vec::with_capacity(self.layers.len());
        let (mut n_conv, mut n_attn) = (0usize, 0usize);
        let mut n_kv_heads = None;
        let mut l_cache = None;
        for (l, layer) in self.layers.iter().enumerate() {
            if !Self::layer_supports_metal_dense_ffn(layer)
                || !matches!(layer.moe.experts, ExpertBacking::Resident(_))
                || layer.attn.post_ffn_norm.is_some()
            {
                return None;
            }
            let shape = self.config.layer_shape(l);
            match shape.attention {
                AttnShape::ShortConv => {
                    let conv = layer.attn.shortconv.as_ref()?;
                    if *l_cache.get_or_insert(conv.l_cache) != conv.l_cache
                        || Self::metal_matvec_launch(&conv.in_proj).is_none()
                        || Self::metal_matvec_launch(&conv.out_proj).is_none()
                    {
                        return None;
                    }
                    slots.push(None);
                    n_conv += 1;
                }
                AttnShape::Gqa {
                    n_heads,
                    n_kv_heads: kv_heads,
                } => {
                    // Every attention layer the stack sees has ONE head
                    // geometry: `n_heads` is a launch argument.
                    if *n_kv_heads.get_or_insert(kv_heads) != kv_heads
                        || n_heads != self.config.n_heads
                        || !self.fused_attn_layer_ok(
                            layer,
                            super::fused_view::FusedAttnCaps::METAL_STACK,
                        )
                        || Self::metal_matvec_launch(&layer.attn.q_proj).is_none()
                        || Self::metal_matvec_launch(&layer.attn.k_proj).is_none()
                        || Self::metal_matvec_launch(&layer.attn.v_proj).is_none()
                        || Self::metal_matvec_launch(&layer.attn.o_proj).is_none()
                        || layer.attn.norm_weight.rms_weights().is_none()
                    {
                        return None;
                    }
                    slots.push(Some(n_attn));
                    n_attn += 1;
                }
                _ => return None,
            }
            if layer.attn.norm_weight.rms_weights().is_none()
                || layer.moe.norm_weight.rms_weights().is_none()
            {
                return None;
            }
        }
        if n_conv == 0 || n_attn == 0 {
            return None;
        }
        Some(HybridPlan {
            slots,
            n_conv,
            n_kv_heads: n_kv_heads?,
            l_cache: l_cache?,
            gelu,
        })
    }

    /// The token through every layer in one submission. `hidden` is its
    /// embedding row as the host made it (scale and norm included, so
    /// the stack's input is exactly the host body's). `None` leaves the
    /// token to the host bodies with every cache untouched.
    pub(super) fn try_metal_hybrid_token(
        &self,
        hidden: &[f32],
        pos: usize,
        kv_caches: &mut [KvCache],
    ) -> Option<HybridOut> {
        let plan = self.metal_hybrid_plan()?;
        // Every cache at this position with every row it ever pushed:
        // an evicting window is the host body's to serve.
        if kv_caches
            .iter()
            .any(|c| c.positions() != pos || c.rows() != pos)
        {
            return None;
        }
        let n = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let mut guard = self
            .metal_hybrid
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fits = guard.as_ref().is_some_and(|s| {
            s.kvs.len() == plan.slots.iter().flatten().count()
                && s.kvs.iter().all(|m| m.capacity() > pos)
                && s.windows.fits(plan.n_conv, n, plan.l_cache)
        });
        if !fits {
            let cap = (pos + 1).saturating_add(256).max(512);
            let mut kvs = Vec::new();
            for _ in plan.slots.iter().flatten() {
                kvs.push(MetalKvBuffers::with_capacity(plan.n_kv_heads, head_dim, cap).ok()?);
            }
            let windows = ConvWindows::new(plan.n_conv, n, plan.l_cache).ok()?;
            let kv_layers = (0..plan.slots.len())
                .filter(|&l| plan.slots[l].is_some())
                .collect();
            *guard = Some(HybridDeviceState {
                kvs,
                kv_layers,
                windows,
            });
        }
        let state = guard.as_mut().expect("just ensured");

        // Rebuild any attention mirror whose length is not the host's,
        // and copy every conv layer's history in.
        let mut conv_i = 0usize;
        for (l, slot) in plan.slots.iter().enumerate() {
            let cache = &kv_caches[l];
            match slot {
                Some(s) => {
                    let m = &mut state.kvs[*s];
                    if m.seq_len != cache.rows() {
                        m.upload_from_host(
                            &cache.k.to_f32_vec(),
                            &cache.v.to_f32_vec(),
                            cache.rows(),
                        )
                        .ok()?;
                    }
                }
                None => {
                    let history = cache
                        .k
                        .as_f32()
                        .expect(super::recurrent_block::SHORT_CONV_HISTORY_IS_F32);
                    let rows = cache.rows();
                    let window = window_from_history(plan.l_cache - 1, n, rows, |i| {
                        &history[i * n..(i + 1) * n]
                    });
                    state.windows.history_mut(conv_i).copy_from_slice(&window);
                    conv_i += 1;
                }
            }
        }

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut conv_i = 0usize;
        for (l, slot) in plan.slots.iter().enumerate() {
            let layer = &self.layers[l];
            let ExpertBacking::Resident(experts) = &layer.moe.experts else {
                return None;
            };
            let ex = &experts[0];
            let mixer = match slot {
                Some(s) => TokenMixer::Attention(AttnMixer {
                    q: Self::metal_matvec_launch(&layer.attn.q_proj)?,
                    kv: LayerKv::Own {
                        k: Self::metal_matvec_launch(&layer.attn.k_proj)?,
                        v: Self::metal_matvec_launch(&layer.attn.v_proj)?,
                        slot: *s,
                    },
                    o: Self::metal_matvec_launch(&layer.attn.o_proj)?,
                    extras: self.metal_attn_extras(layer),
                    rope: self.metal_layer_rope(l),
                    window: self.config.layer_sliding_window(l),
                }),
                None => {
                    let conv = layer.attn.shortconv.as_ref()?;
                    let m = TokenMixer::ShortConv(ShortConvMixer {
                        in_proj: Self::metal_matvec_launch(&conv.in_proj)?,
                        taps: &conv.conv,
                        out_proj: Self::metal_matvec_launch(&conv.out_proj)?,
                        windows: &state.windows,
                        window: conv_i,
                    });
                    conv_i += 1;
                    m
                }
            };
            layers.push(DenseLayerMetal {
                attn_norm_w: layer.attn.norm_weight.rms_weights()?,
                ffn_norm_w: layer.moe.norm_weight.rms_weights()?,
                mixer,
                gate: Self::metal_matvec_launch(&ex.gate)?,
                up: Self::metal_matvec_launch(&ex.up)?,
                down: Self::metal_matvec_launch(&ex.down)?,
                // `lfm3.cpp:178-181` norms the branch on BOTH kinds of
                // layer; `lfm2.cpp` creates no such tensor.
                post_attn_norm: layer.attn.post_attn_norm.as_deref(),
                post_ffn_norm: None,
                per_layer: None,
                out_scale: None,
            });
        }

        // lm_head in the stack exactly when the dense stack would put it
        // there (`decoder::lm_head`).
        let greedy_gpu = frink_metal::attn::metal_greedy_argmax_active();
        let lm_head_launch = Self::metal_matvec_launch(&self.output_head);
        let folded = FoldedLmHead::permit(
            greedy_gpu,
            &self.final_norm,
            self.output_bias.as_deref(),
            lm_head_launch,
        );
        let final_norm_w = if folded.is_some() || lm_head_launch.is_some() {
            self.final_norm.rms_weights()
        } else {
            None
        };
        let out = frink_metal::attn::launch_decode_dense_stack(
            hidden,
            &layers,
            &mut state.kvs,
            self.config.n_heads,
            self.metal_rope(),
            pos,
            self.config.rms_norm_eps,
            final_norm_w,
            folded.as_ref().map(FoldedLmHead::launch),
            folded.as_ref().is_some_and(FoldedLmHead::argmax_only),
            None,
            plan.gelu,
            None,
        );
        drop(layers);
        let out = match out {
            Ok(out) => out,
            Err(e) => {
                eprintln!("frink: Metal hybrid stack failed, host fallback: {e}");
                return None;
            }
        };
        // The host takes back every row this token wrote.
        let mut conv_i = 0usize;
        for (l, slot) in plan.slots.iter().enumerate() {
            match slot {
                Some(s) => Self::catch_up_host_kv_from_metal(&mut state.kvs[*s], &mut kv_caches[l]),
                None => {
                    kv_caches[l]
                        .push(state.windows.newest(conv_i), &[])
                        .expect("unbounded/planned KvCache growth is infallible");
                    conv_i += 1;
                }
            }
        }
        Some(match folded {
            Some(f) => HybridOut::Logits(f.interpret(
                out,
                self.output_head.rows(),
                self.config.final_logit_softcap,
                self.config.logit_multiplier,
            )),
            None => HybridOut::Hidden {
                hidden: out,
                normed: final_norm_w.is_some(),
            },
        })
    }
}
