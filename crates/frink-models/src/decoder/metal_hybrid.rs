//! A recurrent model's decode token as ONE Metal submission: LFM2's
//! short-convolution hybrids and the pure Mamba-1 models.
//!
//! LFM2 alternates conv layers and GQA layers around one residual
//! topology (`crate::shortconv`, `lfm2.cpp:192-208`); Mamba and
//! FalconMamba are a norm, a Mamba-1 block and a residual add per layer
//! (`mamba.cpp:96-120`). Every fused decode path refused both, because
//! a conv or Mamba layer is not an attention layer, so a token went
//! layer by layer through the host bodies: two to four device round
//! trips per layer and the attention or the scan on the host, fifty to
//! two hundred waits a token. The dense decode stack now takes the conv
//! and the Mamba-1 block as token mixers (`frink_metal::attn::
//! TokenMixer`) and a layer without an FFN, so the token is one wait.
//!
//! # Who holds the state
//!
//! The HOST does, before and after every token, for every kind of layer.
//! The conv history is the layer's `KvCache` (`crate::shortconv`): its
//! last `l_cache - 1` rows are copied into the device windows before the
//! launch and the row the stack appends is pushed back after it. A
//! Mamba layer's `RecurrentState` is page-aligned host memory the
//! kernels advance in place (`frink_metal::ssm_decode`). The attention
//! layers' K/V rows are downloaded after the launch
//! (`catch_up_host_kv_from_metal`), so the device KV is a mirror the next
//! token can trust when its length agrees with the host's and rebuilds
//! from the host when it does not. Nothing a prefix restore, a rollback
//! or a host fallback does to the caches can leave the device holding
//! the authoritative copy of anything.

use frink_core::cache::KvCache;
use frink_metal::attn::{
    AttnMixer, DenseFfnMetal, DenseLayerMetal, LayerKv, MetalKvBuffers, ShortConvMixer, TokenMixer,
};
use frink_metal::shortconv::ConvWindows;
use frink_metal::ssm_decode::{Mamba1Mixer, Mamba1Weights};
use frink_moe::GluAct;

use super::lm_head::FoldedLmHead;
use super::{Decoder, ExpertBacking, LayerWeights};
use crate::layer_shapes::AttnShape;
use crate::mamba1::DtBcNorm;
use crate::shortconv::window_from_history;
use crate::ssm_block::SsmBlock;

/// The device half: every attention layer's KV mirror, in layer order,
/// and one window per conv layer.
pub(crate) struct HybridDeviceState {
    kvs: Vec<MetalKvBuffers>,
    /// The layer each of `kvs` mirrors.
    kv_layers: Vec<usize>,
    windows: Option<ConvWindows>,
}

/// What one layer is, for the stack.
#[derive(Clone, Copy)]
enum LayerKind {
    /// GQA, with this KV slot.
    Attn(usize),
    /// LFM2's short convolution, with this window.
    Conv(usize),
    /// A Mamba-1 block.
    Mamba1,
}

/// Which layer is which, for a model the stack serves whole.
struct HybridPlan {
    kinds: Vec<LayerKind>,
    n_attn: usize,
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

    /// The layer's FFN as the stack takes it: `Some(None)` for a layer
    /// with none, `None` when it has one the stack cannot run.
    fn hybrid_ffn<'a>(
        &self,
        l: usize,
        layer: &'a LayerWeights,
    ) -> Option<Option<DenseFfnMetal<'a>>> {
        if self.config.layer_shape(l).ffn_dim == 0 {
            return Some(None);
        }
        if !Self::layer_supports_metal_dense_ffn(layer) || layer.attn.post_ffn_norm.is_some() {
            return None;
        }
        let ExpertBacking::Resident(experts) = &layer.moe.experts else {
            return None;
        };
        let ex = experts.first()?;
        Some(Some(DenseFfnMetal {
            norm_w: layer.moe.norm_weight.rms_weights()?,
            gate: Self::metal_matvec_launch(&ex.gate)?,
            up: Self::metal_matvec_launch(&ex.up)?,
            down: Self::metal_matvec_launch(&ex.down)?,
        }))
    }

    /// The plan, or `None` when any layer is something the stack does
    /// not serve. A model whose every layer attends is the plain dense
    /// stack's.
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
        let mut kinds = Vec::with_capacity(self.layers.len());
        let (mut n_conv, mut n_attn) = (0usize, 0usize);
        let mut n_kv_heads = None;
        let mut l_cache = None;
        for (l, layer) in self.layers.iter().enumerate() {
            self.hybrid_ffn(l, layer)?;
            layer.attn.norm_weight.rms_weights()?;
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
                    kinds.push(LayerKind::Conv(n_conv));
                    n_conv += 1;
                }
                AttnShape::Mamba1 => {
                    let Some(SsmBlock::Mamba1(m)) = layer.attn.ssm.as_ref() else {
                        return None;
                    };
                    // `mamba-base.cpp:84-88`'s norms on dt, B and C have
                    // no kernel in the stack.
                    if !matches!(m.dt_bc_norm, DtBcNorm::None)
                        || layer.attn.post_attn_norm.is_some()
                        || [&m.in_proj, &m.x_proj, &m.dt_proj, &m.out_proj]
                            .iter()
                            .any(|w| Self::metal_matvec_launch(w).is_none())
                    {
                        return None;
                    }
                    kinds.push(LayerKind::Mamba1);
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
                    {
                        return None;
                    }
                    kinds.push(LayerKind::Attn(n_attn));
                    n_attn += 1;
                }
                _ => return None,
            }
        }
        if n_attn == self.layers.len() {
            return None;
        }
        Some(HybridPlan {
            kinds,
            n_attn,
            n_conv,
            n_kv_heads: n_kv_heads.unwrap_or(0),
            l_cache: l_cache.unwrap_or(0),
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
        if kv_caches.iter().zip(&plan.kinds).any(|(c, k)| {
            c.positions() != pos || (!matches!(k, LayerKind::Mamba1) && c.rows() != pos)
        }) {
            return None;
        }
        let n = self.config.hidden_dim;
        let head_dim = self.config.head_dim;
        let mut guard = self
            .metal_hybrid
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let fits = guard.as_ref().is_some_and(|s| {
            s.kvs.len() == plan.n_attn
                && s.kvs.iter().all(|m| m.capacity() > pos)
                && match &s.windows {
                    None => plan.n_conv == 0,
                    Some(w) => w.fits(plan.n_conv, n, plan.l_cache),
                }
        });
        if !fits {
            let cap = (pos + 1).saturating_add(256).max(512);
            let mut kvs = Vec::with_capacity(plan.n_attn);
            for _ in 0..plan.n_attn {
                kvs.push(MetalKvBuffers::with_capacity(plan.n_kv_heads, head_dim, cap).ok()?);
            }
            let windows = match plan.n_conv {
                0 => None,
                c => Some(ConvWindows::new(c, n, plan.l_cache).ok()?),
            };
            let kv_layers = plan
                .kinds
                .iter()
                .enumerate()
                .filter_map(|(l, k)| matches!(k, LayerKind::Attn(_)).then_some(l))
                .collect();
            *guard = Some(HybridDeviceState {
                kvs,
                kv_layers,
                windows,
            });
        }
        let state = guard.as_mut().expect("just ensured");

        // Rebuild any attention mirror whose length is not the host's,
        // copy every conv layer's history in, and give every Mamba layer
        // its state (zeros on a sequence's first token, as `build_rs`).
        let mut mamba_states = Vec::new();
        for (l, kind) in plan.kinds.iter().enumerate() {
            let cache = &mut kv_caches[l];
            match *kind {
                LayerKind::Attn(s) => {
                    let m = &mut state.kvs[s];
                    if m.seq_len != cache.rows() {
                        m.upload_from_host(
                            &cache.k.to_f32_vec(),
                            &cache.v.to_f32_vec(),
                            cache.rows(),
                        )
                        .ok()?;
                    }
                }
                LayerKind::Conv(i) => {
                    let history = cache
                        .k
                        .as_f32()
                        .expect(super::recurrent_block::SHORT_CONV_HISTORY_IS_F32);
                    let rows = cache.rows();
                    let window = window_from_history(plan.l_cache - 1, n, rows, |r| {
                        &history[r * n..(r + 1) * n]
                    });
                    let windows = state.windows.as_mut().expect("a conv layer has windows");
                    windows.history_mut(i).copy_from_slice(&window);
                }
                LayerKind::Mamba1 => {
                    let block = self.layers[l].attn.ssm.as_ref()?;
                    let rs = cache.recurrent.get_or_insert_with(|| block.zero_state());
                    mamba_states.push((
                        (rs.conv.as_ptr(), rs.conv.alloc_bytes()),
                        (rs.ssm.as_ptr(), rs.ssm.alloc_bytes()),
                    ));
                }
            }
        }

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut mamba_i = 0usize;
        for (l, kind) in plan.kinds.iter().enumerate() {
            let layer = &self.layers[l];
            let mixer = match *kind {
                LayerKind::Attn(s) => TokenMixer::Attention(AttnMixer {
                    q: Self::metal_matvec_launch(&layer.attn.q_proj)?,
                    kv: LayerKv::Own {
                        k: Self::metal_matvec_launch(&layer.attn.k_proj)?,
                        v: Self::metal_matvec_launch(&layer.attn.v_proj)?,
                        slot: s,
                    },
                    o: Self::metal_matvec_launch(&layer.attn.o_proj)?,
                    extras: self.metal_attn_extras(layer),
                    rope: self.metal_layer_rope(l),
                    window: self.config.layer_sliding_window(l),
                }),
                LayerKind::Conv(i) => {
                    let conv = layer.attn.shortconv.as_ref()?;
                    TokenMixer::ShortConv(ShortConvMixer {
                        in_proj: Self::metal_matvec_launch(&conv.in_proj)?,
                        taps: &conv.conv,
                        out_proj: Self::metal_matvec_launch(&conv.out_proj)?,
                        windows: state.windows.as_ref().expect("a conv layer has windows"),
                        window: i,
                    })
                }
                LayerKind::Mamba1 => {
                    let Some(SsmBlock::Mamba1(m)) = layer.attn.ssm.as_ref() else {
                        return None;
                    };
                    let (conv_state, ssm_state) = mamba_states[mamba_i];
                    mamba_i += 1;
                    let w = Mamba1Weights {
                        in_proj: Self::metal_matvec_launch(&m.in_proj)?,
                        conv1d: &m.conv1d,
                        conv_bias: &m.conv1d_bias,
                        d_conv: m.h.d_conv,
                        x_proj: Self::metal_matvec_launch(&m.x_proj)?,
                        dt_proj: Self::metal_matvec_launch(&m.dt_proj)?,
                        dt_bias: &m.dt_bias,
                        a: &m.a,
                        d: &m.d,
                        out_proj: Self::metal_matvec_launch(&m.out_proj)?,
                        d_inner: m.h.d_inner,
                        d_state: m.h.d_state,
                        dt_rank: m.h.dt_rank,
                    };
                    // SAFETY: both are this layer's `RecurrentState`
                    // buffers, page-aligned `AlignedF32`s of whole pages
                    // sized by these weights (`Mamba1::zero_state`). The
                    // caches they live in are borrowed `&mut` by this call
                    // and nothing else reads or writes them until the
                    // launch below, which waits for the GPU, returns.
                    TokenMixer::Mamba1(unsafe { Mamba1Mixer::new(w, conv_state, ssm_state) }?)
                }
            };
            layers.push(DenseLayerMetal {
                attn_norm_w: layer.attn.norm_weight.rms_weights()?,
                mixer,
                ffn: self.hybrid_ffn(l, layer)?,
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
        // The host takes back every row this token wrote; a Mamba
        // layer's state was advanced in place, and its cache counts the
        // position as `ssm_block` counts it.
        for (l, kind) in plan.kinds.iter().enumerate() {
            match *kind {
                LayerKind::Attn(s) => {
                    Self::catch_up_host_kv_from_metal(&mut state.kvs[s], &mut kv_caches[l])
                }
                LayerKind::Conv(i) => {
                    let windows = state.windows.as_ref().expect("a conv layer has windows");
                    kv_caches[l]
                        .push(windows.newest(i), &[])
                        .expect("unbounded/planned KvCache growth is infallible");
                }
                LayerKind::Mamba1 => kv_caches[l]
                    .advance_len(1)
                    .expect("unbounded/planned KvCache growth is infallible"),
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
