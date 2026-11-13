//! Gemma-4 dedicated text engine (E2B / E4B-style GGUF).
//!
//! Distinct from generic Gemma-2/3 [`crate::decoder::Decoder`]:
//! - per-layer token embeddings (`per_layer_token_embd` + proj)
//! - shared KV layers (`attention.shared_kv_layers`)
//! - SWA vs full layers with different head dims (`key_length` /
//!   `key_length_swa`) and a bool SWA pattern array
//!
//! Graph mirrors `.scratch/llama.cpp/src/models/gemma4.cpp`.

use frink_core::attention::{
    apply_rope, apply_rope_with_freq_factors, causal_gqa_attention, causal_gqa_attention_windowed,
};
use frink_core::cache::KvCache;
use frink_core::matmul::{geglu, gelu, rms_norm, rms_norm_per_head, softcap_inplace};
use frink_core::weight_matrix::WeightMatrix;

use crate::engine::Engine;

/// Architectures served by this engine.
pub const GEMMA4_ARCHES: &[&str] = &["gemma4", "gemma4-assistant"];

/// Hyperparameters from `{arch}.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Gemma4Hparams {
    pub arch: String,
    pub n_layer: usize,
    pub hidden_dim: usize,
    /// Per-layer FFN intermediate sizes (`feed_forward_length` array or scalar).
    pub ffn_dims: Vec<usize>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim_full: usize,
    pub head_dim_swa: usize,
    pub sliding_window: usize,
    /// `true` = SWA layer; length == `n_layer`.
    pub is_swa: Vec<bool>,
    /// First this many layers own a KV cache; later layers reuse.
    pub n_layer_kv_from_start: usize,
    pub embd_per_layer: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub final_logit_softcap: Option<f32>,
    /// Attention score scale passed to the kernel (Gemma-4: 1.0).
    pub attention_scale: f32,
}

impl Gemma4Hparams {
    pub fn is_swa_layer(&self, il: usize) -> bool {
        self.is_swa.get(il).copied().unwrap_or(false)
    }

    pub fn head_dim(&self, il: usize) -> usize {
        if self.is_swa_layer(il) {
            self.head_dim_swa
        } else {
            self.head_dim_full
        }
    }

    pub fn has_kv(&self, il: usize) -> bool {
        il < self.n_layer_kv_from_start
    }

    /// Layer whose KV cache is reused when `!has_kv(il)` (llama.cpp GEMMA4 reuse).
    pub fn kv_reuse_layer(&self, il: usize) -> usize {
        debug_assert!(!self.has_kv(il));
        self.n_layer_kv_from_start - if self.is_swa_layer(il) { 2 } else { 1 }
    }

    pub fn rope_theta_for(&self, il: usize) -> f32 {
        if self.is_swa_layer(il) {
            self.rope_theta_swa
        } else {
            self.rope_theta
        }
    }

    pub fn ffn_dim(&self, il: usize) -> usize {
        self.ffn_dims
            .get(il)
            .copied()
            .unwrap_or_else(|| *self.ffn_dims.last().unwrap_or(&self.hidden_dim))
    }
}

pub struct Gemma4AttnWeights {
    pub q_proj: WeightMatrix,
    pub k_proj: Option<WeightMatrix>,
    pub v_proj: Option<WeightMatrix>,
    pub o_proj: WeightMatrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Option<Vec<f32>>,
    pub post_attn_norm: Vec<f32>,
}

pub struct Gemma4LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn: Gemma4AttnWeights,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: WeightMatrix,
    pub ffn_up: WeightMatrix,
    pub ffn_down: WeightMatrix,
    pub ffn_post_norm: Vec<f32>,
    pub per_layer_inp_gate: Option<WeightMatrix>,
    pub per_layer_proj: Option<WeightMatrix>,
    pub per_layer_post_norm: Option<Vec<f32>>,
    pub out_scale: Option<f32>,
}

pub struct Gemma4Weights {
    pub token_embd: WeightMatrix,
    pub per_layer_token_embd: Option<WeightMatrix>,
    pub per_layer_model_proj: Option<WeightMatrix>,
    pub per_layer_proj_norm: Option<Vec<f32>>,
    pub layers: Vec<Gemma4LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output_head: WeightMatrix,
    /// Full-attn RoPE frequency factors (`rope_freqs.weight`), length `head_dim_full/2`.
    pub rope_freqs: Option<Vec<f32>>,
}

pub struct Gemma4Engine {
    pub weights: Gemma4Weights,
    pub hp: Gemma4Hparams,
    /// What the host body's K/V caches store their rows as: a RUN
    /// choice, set by the caller from `--ctk` and the backend through
    /// `kv_budget::host_kv_dtype`, exactly as `ModelConfig::kv_dtype` is
    /// for the generic decoder. The loader starts it at llama.cpp's f16.
    ///
    /// The body used to build f32 caches whatever `--ctk` said, which
    /// ADR 0009 names as a deviation from llama.cpp, while the Metal
    /// stacks it hands rows to and from stored f16 (#634).
    pub kv_dtype: frink_core::kv_store::KvDtype,
    /// What the Metal stacks read that the file does not hold as such:
    /// the Q norm with the attention scale folded in, and V's weightless
    /// norm spelled as ones. Built on first use. See `crate::gemma4_metal`.
    #[cfg(feature = "metal")]
    pub(crate) device_aux: std::sync::OnceLock<crate::gemma4_metal::DeviceAux>,
}

pub struct Gemma4DecodeState {
    /// One cache per layer that `has_kv`; indexed by layer id (holes for shared layers).
    ///
    /// While the Metal stacks serve this state the device holds the KV
    /// and these advance by placeholder rows, so their position counts
    /// stay true; `crate::gemma4_metal` downloads the real rows before
    /// the host body reads them again.
    pub kv: Vec<Option<KvCache>>,
    #[cfg(feature = "metal")]
    pub(crate) metal: crate::gemma4_metal::Gemma4MetalState,
}

/// The three constants of the per-layer-embedding prologue, from one
/// place for the host body and the Metal one (`gemma4.cpp`
/// `project_per_layer_inputs`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PleScales {
    /// On the token's `per_layer_token_embd` row: `sqrt(n_embd_per_layer)`.
    pub(crate) tok: f32,
    /// On `per_layer_model_proj · h`: `1 / sqrt(n_embd)`.
    pub(crate) proj: f32,
    /// On the sum of the two: `1 / sqrt(2)`.
    pub(crate) input: f32,
}

impl Gemma4Engine {
    /// See [`crate::decoder::Decoder::probe_kernels`]. Called at the end
    /// of loading, before the registry is sealed.
    ///
    /// On Metal the whole model runs on the fused dense stacks
    /// (`crate::gemma4_metal`): a prompt is one command buffer per chunk
    /// through the batched GEMMs and a token is one command buffer. On
    /// every other backend there is no batched path, so a 512-token
    /// prefill is 512 sequential single-token passes; the registry
    /// records that as a missing engine capability rather than leaving
    /// it to be rediscovered from a benchmark.
    pub fn probe_kernels(&self) {
        use frink_core::kernel_registry as reg;

        if !reg::enabled() {
            return;
        }
        let w = &self.weights;
        w.token_embd.probe_kernels("token_embd");
        w.output_head.probe_kernels("output_head");
        if let Some(m) = &w.per_layer_token_embd {
            m.probe_kernels("per_layer_token_embd");
        }
        if let Some(m) = &w.per_layer_model_proj {
            m.probe_kernels("per_layer_model_proj");
        }
        for layer in &w.layers {
            layer.attn.q_proj.probe_kernels("attn_q");
            if let Some(m) = &layer.attn.k_proj {
                m.probe_kernels("attn_k");
            }
            if let Some(m) = &layer.attn.v_proj {
                m.probe_kernels("attn_v");
            }
            layer.attn.o_proj.probe_kernels("attn_o");
            layer.ffn_gate.probe_kernels("ffn_gate");
            layer.ffn_up.probe_kernels("ffn_up");
            layer.ffn_down.probe_kernels("ffn_down");
            if let Some(m) = &layer.per_layer_inp_gate {
                m.probe_kernels("per_layer_inp_gate");
            }
            if let Some(m) = &layer.per_layer_proj {
                m.probe_kernels("per_layer_proj");
            }
        }
        reg::record_build(
            reg::Lookup::new(
                frink_core::weight_matrix::active_backend(),
                reg::op::ENGINE_PREFILL_BATCH,
                None,
            )
            .with_role("gemma4_engine"),
            if self.batched_prefill_here() {
                reg::Outcome::Hit
            } else {
                reg::Outcome::slow_path("sequential forward_token (batch=1 matvec per projection)")
            },
        );
    }

    /// Does [`Engine::prefill`] batch on the backend this process runs?
    fn batched_prefill_here(&self) -> bool {
        #[cfg(feature = "metal")]
        if frink_core::weight_matrix::active_backend()
            == frink_core::kernel_registry::Backend::Metal
        {
            return self.metal_stacks_cover();
        }
        false
    }

    pub fn new_state(&self) -> Gemma4DecodeState {
        let kv = (0..self.hp.n_layer)
            .map(|il| {
                if self.hp.has_kv(il) {
                    let hd = self.hp.head_dim(il);
                    Some(KvCache::new(self.hp.n_kv_heads, hd).with_dtype(self.kv_dtype))
                } else {
                    None
                }
            })
            .collect();
        Gemma4DecodeState {
            kv,
            #[cfg(feature = "metal")]
            metal: Default::default(),
        }
    }

    /// The token's embedding row, scaled by `sqrt(n_embd)`
    /// (`build_inp_embd(tok_embd, sqrtf(n_embd))`).
    pub(crate) fn embed_scaled(&self, token_id: usize) -> Vec<f32> {
        let mut hidden = self.weights.token_embd.dequant_row(token_id);
        let emb_scale = (self.hp.hidden_dim as f32).sqrt();
        for x in hidden.iter_mut() {
            *x *= emb_scale;
        }
        hidden
    }

    pub(crate) fn ple_scales(&self) -> PleScales {
        PleScales {
            tok: (self.hp.embd_per_layer as f32).sqrt(),
            proj: 1.0 / (self.hp.hidden_dim as f32).sqrt(),
            input: 1.0 / 2f32.sqrt(),
        }
    }

    /// The token's `per_layer_token_embd` row times its scale:
    /// `[n_layer * n]`, layer-major chunks of `n`. `None` without PLE.
    pub(crate) fn per_layer_token_row(&self, token_id: usize) -> Option<Vec<f32>> {
        let pl_embd = self.weights.per_layer_token_embd.as_ref()?;
        let scale_tok = self.ple_scales().tok;
        let mut per_tok = pl_embd.dequant_row(token_id);
        for x in per_tok.iter_mut() {
            *x *= scale_tok;
        }
        Some(per_tok)
    }

    fn project_per_layer_inputs(&self, token_id: usize, hidden: &[f32]) -> Option<Vec<Vec<f32>>> {
        let pl_proj = self.weights.per_layer_model_proj.as_ref()?;
        let pl_norm = self.weights.per_layer_proj_norm.as_ref()?;
        let n = self.hp.embd_per_layer;
        let n_layer = self.hp.n_layer;
        let scales = self.ple_scales();
        let per_tok = self.per_layer_token_row(token_id)?;
        // per_tok: [n_layer * n] contiguous as layer-major chunks of n
        let proj_scale = scales.proj;
        let mut from_model = pl_proj.apply(hidden);
        for x in from_model.iter_mut() {
            *x *= proj_scale;
        }
        // from_model: [n_layer * n]
        let mut out = Vec::with_capacity(n_layer);
        let input_scale = scales.input;
        for il in 0..n_layer {
            let start = il * n;
            let mut chunk: Vec<f32> = from_model[start..start + n].to_vec();
            chunk = rms_norm(&chunk, pl_norm, self.hp.rms_norm_eps);
            for (c, t) in chunk.iter_mut().zip(per_tok[start..start + n].iter()) {
                *c = (*c + *t) * input_scale;
            }
            out.push(chunk);
        }
        Some(out)
    }

    /// The host body: the reference every device path is checked
    /// against, and what serves a token no device path takes.
    pub(crate) fn forward_token_host(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Gemma4DecodeState,
    ) -> Vec<f32> {
        let hp = &self.hp;
        let mut hidden = self.embed_scaled(token_id);

        let per_layer_in = self.project_per_layer_inputs(token_id, &hidden);

        for (il, layer) in self.weights.layers.iter().enumerate() {
            let head_dim = hp.head_dim(il);
            let n_heads = hp.n_heads;
            let n_kv = hp.n_kv_heads;
            let attn_in = rms_norm(&hidden, &layer.attn_norm, hp.rms_norm_eps);

            let mut q = layer.attn.q_proj.apply(&attn_in);
            q = rms_norm_per_head(&q, &layer.attn.q_norm, head_dim, hp.rms_norm_eps);

            let theta = hp.rope_theta_for(il);
            let freq = if hp.is_swa_layer(il) {
                None
            } else {
                self.weights.rope_freqs.as_deref()
            };
            for h in 0..n_heads {
                let slice = &mut q[h * head_dim..(h + 1) * head_dim];
                match freq {
                    Some(f) => apply_rope_with_freq_factors(slice, pos, theta, f),
                    None => apply_rope(slice, pos, theta),
                }
            }
            // Gemma-4 attention_scale = 1.0: compensate kernel's 1/sqrt(d).
            let compensate = hp.attention_scale * (head_dim as f32).sqrt();
            for v in q.iter_mut() {
                *v *= compensate;
            }

            let attn_out = if hp.has_kv(il) {
                let k_proj = layer
                    .attn
                    .k_proj
                    .as_ref()
                    .expect("has_kv layer missing attn_k");
                let k_raw = k_proj.apply(&attn_in);
                let k_norm = layer
                    .attn
                    .k_norm
                    .as_ref()
                    .expect("has_kv layer missing attn_k_norm");
                let mut k = rms_norm_per_head(&k_raw, k_norm, head_dim, hp.rms_norm_eps);
                for h in 0..n_kv {
                    let slice = &mut k[h * head_dim..(h + 1) * head_dim];
                    match freq {
                        Some(f) => apply_rope_with_freq_factors(slice, pos, theta, f),
                        None => apply_rope(slice, pos, theta),
                    }
                }

                // Without `attn_v`, V is K's projection before K's norm
                // and RoPE (`gemma4.cpp`: `Vcur = Kcur` ahead of both).
                let mut v = match layer.attn.v_proj.as_ref() {
                    Some(vp) => vp.apply(&attn_in),
                    None => k_raw,
                };
                // ggml_rms_norm without weight on V
                v = rms_norm_per_head(&v, &vec![1.0; head_dim], head_dim, hp.rms_norm_eps);

                let cache = state.kv[il].as_mut().expect("kv slot");
                cache
                    .push(&k, &v)
                    .expect("unbounded KvCache growth is infallible");

                if hp.is_swa_layer(il) {
                    causal_gqa_attention_windowed(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv,
                        head_dim,
                        cache.rows(),
                        hp.sliding_window,
                    )
                } else {
                    causal_gqa_attention(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv,
                        head_dim,
                        cache.rows(),
                    )
                }
            } else {
                let reuse = hp.kv_reuse_layer(il);
                let cache = state.kv[reuse]
                    .as_ref()
                    .expect("reuse kv layer missing cache");
                // Shared-KV layers may have a different head_dim than the
                // reused cache (SWA vs full). Attention only works when dims match.
                assert_eq!(
                    cache.head_dim, head_dim,
                    "gemma4 shared-KV reuse head_dim mismatch layer {il} -> {reuse}"
                );
                if hp.is_swa_layer(il) {
                    causal_gqa_attention_windowed(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv,
                        head_dim,
                        cache.rows(),
                        hp.sliding_window,
                    )
                } else {
                    causal_gqa_attention(
                        &q,
                        &cache.k,
                        &cache.v,
                        n_heads,
                        n_kv,
                        head_dim,
                        cache.rows(),
                    )
                }
            };

            let mut attn_proj = layer.attn.o_proj.apply(&attn_out);
            attn_proj = rms_norm(&attn_proj, &layer.attn.post_attn_norm, hp.rms_norm_eps);
            let mut attn_out_res = hidden;
            for (a, p) in attn_out_res.iter_mut().zip(attn_proj.iter()) {
                *a += p;
            }

            let ffn_in = rms_norm(&attn_out_res, &layer.ffn_norm, hp.rms_norm_eps);
            let gate = layer.ffn_gate.apply(&ffn_in);
            let up = layer.ffn_up.apply(&ffn_in);
            let mut ffn_out = layer.ffn_down.apply(&geglu(&gate, &up));
            ffn_out = rms_norm(&ffn_out, &layer.ffn_post_norm, hp.rms_norm_eps);

            let mut cur = attn_out_res;
            for (c, f) in cur.iter_mut().zip(ffn_out.iter()) {
                *c += f;
            }

            if let (Some(gate_w), Some(proj_w), Some(post_n), Some(pl_in)) = (
                layer.per_layer_inp_gate.as_ref(),
                layer.per_layer_proj.as_ref(),
                layer.per_layer_post_norm.as_ref(),
                per_layer_in.as_ref(),
            ) {
                let pe_in = cur.clone();
                let mut g = gate_w.apply(&cur);
                for x in g.iter_mut() {
                    *x = gelu(*x);
                }
                for (gx, p) in g.iter_mut().zip(pl_in[il].iter()) {
                    *gx *= *p;
                }
                let mut pe = proj_w.apply(&g);
                pe = rms_norm(&pe, post_n, hp.rms_norm_eps);
                cur = pe_in;
                for (c, p) in cur.iter_mut().zip(pe.iter()) {
                    *c += p;
                }
            }

            if let Some(s) = layer.out_scale {
                for x in cur.iter_mut() {
                    *x *= s;
                }
            }
            hidden = cur;
        }

        let mut logits = self.weights.output_head.apply(&rms_norm(
            &hidden,
            &self.weights.output_norm,
            hp.rms_norm_eps,
        ));
        if let Some(sc) = hp.final_logit_softcap {
            softcap_inplace(&mut logits, sc);
        }
        logits
    }
}

impl Engine for Gemma4Engine {
    type State = Gemma4DecodeState;

    fn new_state(&self) -> Gemma4DecodeState {
        Gemma4Engine::new_state(self)
    }

    fn vocab_size(&self) -> usize {
        self.weights.output_head.rows()
    }

    fn forward_token_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32> {
        #[cfg(feature = "metal")]
        if let Some(logits) = self.forward_token_metal(token_id, pos, state) {
            return logits;
        }
        self.forward_token_host(token_id, pos, state)
    }

    /// Batched on Metal (`crate::gemma4_metal`), one token at a time
    /// everywhere else.
    fn prefill(&self, tokens: &[usize], start_pos: usize, state: &mut Self::State) -> Vec<f32> {
        #[cfg(feature = "metal")]
        let (done, mut last) = self.prefill_metal(tokens, start_pos, state);
        #[cfg(not(feature = "metal"))]
        let (done, mut last) = (0usize, Vec::new());
        for (i, &tok) in tokens.iter().enumerate().skip(done) {
            last = self.forward_token(tok, start_pos + i, state);
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hp() -> Gemma4Hparams {
        // E2B-like: 10 layers, shared_kv=4 → n_kv_from_start=6; SWA pattern 4+1
        let n_layer = 10;
        let mut is_swa = Vec::new();
        for i in 0..n_layer {
            is_swa.push((i + 1) % 5 != 0);
        }
        Gemma4Hparams {
            arch: "gemma4".into(),
            n_layer,
            hidden_dim: 64,
            ffn_dims: vec![128; n_layer],
            n_heads: 4,
            n_kv_heads: 1,
            head_dim_full: 32,
            head_dim_swa: 16,
            sliding_window: 8,
            is_swa,
            n_layer_kv_from_start: 6,
            embd_per_layer: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_theta_swa: 10_000.0,
            final_logit_softcap: Some(30.0),
            attention_scale: 1.0,
        }
    }

    #[test]
    fn layer_routing_swa_and_shared_kv() {
        let hp = sample_hp();
        assert!(hp.is_swa_layer(0));
        assert!(!hp.is_swa_layer(4));
        assert_eq!(hp.head_dim(0), 16);
        assert_eq!(hp.head_dim(4), 32);
        assert!(hp.has_kv(5));
        assert!(!hp.has_kv(6));
        // layer 6 SWA → reuse 6-2=4; layer 9 full → reuse 6-1=5
        assert_eq!(hp.kv_reuse_layer(6), 4);
        assert_eq!(hp.kv_reuse_layer(9), 5);
        assert_eq!(hp.rope_theta_for(0), 10_000.0);
        assert_eq!(hp.rope_theta_for(4), 1_000_000.0);
    }
}
