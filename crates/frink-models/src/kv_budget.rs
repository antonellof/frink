//! Pre-load KV budget arithmetic: answer "will this fit" *before*
//! allocating anything, from terms that are all exact in the GGUF
//! header.
//!
//! ```text
//! weights + n_ctx * per_token_kv + activation_headroom  <=  device_budget
//! per_token_kv = n_layers * n_kv_heads * head_dim * bytes_per_elem * 2
//! ```
//!
//! Everything here is a pure function of a shape plus a byte budget --
//! no I/O, no device handles, no allocation -- so the arithmetic can be
//! unit-tested against hand-computed numbers. The device side (how many
//! bytes a backend actually offers) lives in
//! [`crate::device_budget`]; the whole-checkpoint report that consumes
//! both is [`crate::residency_report`].
//!
//! # Where this is approximate, stated up front
//!
//! - **Weights.** frink mmaps quantized tensors and reads them in
//!   place, so "weights resident" is not a number frink controls: the
//!   kernel can evict those pages under pressure and fault them back in
//!   later. `weights_bytes` is therefore the *checkpoint's* byte count,
//!   an upper bound on resident cost and a lower bound on the I/O the
//!   run will do -- not a measurement of RSS. A model can exceed this
//!   budget and still run (slowly, page-faulting), and it can fit this
//!   budget and still be killed by something else on the machine.
//! - **Activations.** `activation_headroom_bytes` is a caller-supplied
//!   reserve, not a derived quantity. Nothing here models scratch
//!   buffers, the logits vector, tokenizer state or allocator slack.
//! - **KV element width.** [`KvElem`] is the width of the store that
//!   the *selected backend* keeps. With Metal attention on, the device
//!   holds an f16 KV while the host may still hold an f32 mirror
//!   (`FRINK_CPU_KV_OFFLOAD`); budget the tier you are checking
//!   against, and do not assume the two add up to one number.
//!
//! A conservative, explainable number beats a clever one: none of this
//! tries to track real resident bytes over time.
//!
//! # Why a sliding window is not a saving here
//!
//! This module used to cap sliding-window layers at `window + chunk - 1`
//! positions and subtract them out of the divisor, which made a
//! Gemma-3-4B context look 5.8x cheaper than it is and gpt-oss 2x. **No
//! KV store frink allocates ever gave that cap back** (#33):
//!
//! - `frink_core::cache::KvCache` had no window concept at all. `push`
//!   extended `k`/`v` for every position, so a plain or pool-backed
//!   cache held the whole sequence in every layer. This is what the CLI
//!   allocates and what the server allocates on its non-paged paths, and
//!   it is still what they do unless `FRINK_KV_WINDOW` is on -- see the
//!   section after next.
//! - The paged store *can* recycle pages behind a window, but only for a
//!   model whose every layer shares one window
//!   (`ModelConfig::uniform_sliding_window`, `None` by design for the
//!   alternating models -- gpt-oss, Gemma-2/3 -- because a page group
//!   holds one block per layer and the full-attention layers still read
//!   position 0). Even there it recycles only the GENERATION tail: its
//!   own admission arithmetic (`frink_server::generate::
//!   paged_hold_positions`) holds `prompt + bound + a page`, and a
//!   budget priced in *context length* has to survive a prompt that
//!   fills that context.
//!
//! So the budget prices every layer at every position, for every model.
//! That is exactly what the two `KvCache` stores allocate and an upper
//! bound on what the paged store reserves, which is the direction that
//! matters: an over-estimate costs context, an under-estimate is
//! admitted and then arrives as an OOM instead of the refusal this
//! engine exists to give.
//!
//! # ...unless the store evicts, which it now can
//!
//! #61 step 2 taught the contiguous `KvCache` to drop rows behind a
//! layer's sliding window, behind `FRINK_KV_WINDOW`. So the paragraph
//! above is still the default and no longer the only case, and the
//! difference is expressed the way #33 said it had to be: **the number
//! the store keeps belongs to the store.** [`KvResidency`] carries the
//! per-layer windows, [`KvShape::resident_kv_bytes_for_tokens`] prices
//! them through `frink_core::kv_swa::KvWindow::rows_after`, and
//! `KvCache::evict_behind_window` calls the same function to decide what
//! to drop. There is no second statement of the rule here to drift.
//!
//! Two numbers, not one, and admission wants the larger:
//! [`KvShape::peak_kv_bytes_for_tokens`] adds the one layer that is
//! still mid-prefill and holding the whole prompt, because
//! `Decoder::forward_batch` evicts per layer rather than after the
//! stack. `resident_` is what a measurement of the caches finds at rest;
//! `peak_` is what the machine has to survive.
//!
//! [`KvBudget`] CARRIES the residency rather than taking it as an
//! argument, and [`KvBudget::kv_bytes_at`] is the one expression the
//! estimate, the refusal text and `--ctx auto` all read. Until #61 step
//! 2 was wired here, the store took the saving and the admission check
//! did not know: `-c auto` still divided a Gemma-3 budget by every
//! layer's full per-token cost, so a context that would have fit was
//! refused. That is #33 read backwards, and it is the same defect
//! shape -- two statements of one rule, with nothing making them agree.
//!
//! What is NOT priced here, because no store does it yet: eviction
//! inside the paged store (#61 step 4) and eviction of the prompt region
//! while the prompt is still being written (#61 step 5). Both stay at
//! the full every-layer-every-position number.

use frink_core::kv_swa::KvWindow;

use crate::config::ModelConfig;
use crate::decoder::KvWindowPolicy;

/// Element width of one cached K/V scalar, per backend store.
///
/// The block-quantized variants are the wire formats
/// `frink-metal` writes for `FRINK_CTK` (see
/// `frink_metal::attn::MetalKvDtype`), so their cost is per 32-element
/// block, not per scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvElem {
    /// Host `frink_core::cache::KvCache`, which stores `Vec<f32>`.
    F32,
    /// Metal device KV default (`FRINK_CTK=f16`, llama.cpp `-ctk f16`).
    F16,
    /// ggml Q8_0 wire: 32 elems -> 2-byte scale + 32 int8 = 34 bytes.
    /// `FRINK_CTK=q8_0|fp8` both land on this width.
    Q8_0,
    /// 4-bit wire: 32 elems -> 2-byte scale + 16 nibble bytes.
    /// llama.cpp's `-ctk q4_0` spelling.
    Q4_0,
}

impl KvElem {
    /// Bytes needed to store `elems` cached scalars, rounding up to a
    /// whole block for the block-quantized wires (a partial block still
    /// costs a full one).
    ///
    /// Saturating rather than wrapping or panicking. This is a
    /// REPORTING number: it exists to put bytes in a refusal message,
    /// and it is reached with position counts that came off an HTTP
    /// body. `max_tokens: u64::MAX / 64` does not overflow the position
    /// sum, so it reaches here and multiplied past `u64::MAX`, panicking
    /// the request thread while computing the text of the very refusal
    /// that was about to reject it (#36).
    ///
    /// Saturating is right HERE and wrong for a bound. A saturated byte
    /// count still reports "astronomically large", which is the only
    /// thing the message needs to convey. A saturated position bound
    /// would silently turn a nonsense request into a plausible one and
    /// serve it.
    pub fn bytes_for(self, elems: u64) -> u64 {
        match self {
            KvElem::F32 => elems.saturating_mul(4),
            KvElem::F16 => elems.saturating_mul(2),
            KvElem::Q8_0 => {
                let blocks = elems.div_ceil(frink_quant::Q8_0_BLOCK_ELEMS as u64);
                blocks.saturating_mul(frink_quant::Q8_0_BLOCK_BYTES as u64)
            }
            KvElem::Q4_0 => {
                let blocks = elems.div_ceil(frink_quant::Q4_KV_GROUP as u64);
                blocks.saturating_mul(frink_quant::Q4_KV_BLOCK_BYTES as u64)
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            KvElem::F32 => "f32",
            KvElem::F16 => "f16",
            KvElem::Q8_0 => "q8_0",
            KvElem::Q4_0 => "q4_0",
        }
    }

    /// Maps a `FRINK_CTK` / `--ctk` value onto the width the Metal KV
    /// store really keeps. Mirrors
    /// `frink_metal::attn::effective_metal_kv_dtype`: `fp8` shares
    /// Q8_0's 34-byte wire, and anything unrecognised falls back to
    /// f16 rather than being budgeted at a width no kernel writes.
    ///
    /// Note this does *not* check the block alignment that function
    /// also checks (`n_kv_heads * head_dim` divisible by 32), so a
    /// misaligned shape is budgeted at the requested width while the
    /// runtime silently uses f16 -- an under-estimate, called out here
    /// rather than papered over.
    pub fn from_ctk(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            // llama.cpp's `-ctk f32`, and the width of frink's own
            // host `KvCache`.
            "f32" => KvElem::F32,
            "q8_0" | "fp8" => KvElem::Q8_0,
            "q4_0" => KvElem::Q4_0,
            _ => KvElem::F16,
        }
    }
}

/// How one layer's KV cache is shaped. Which variant applies is a
/// property of the *decoder that will run*, not of the architecture
/// name -- see [`KvLayout::MlaLatent`]'s doc comment for the one place
/// that distinction bites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayout {
    /// Multi-head / grouped-query attention: one K vector and one V
    /// vector of `n_kv_heads * head_dim` per token, per layer. MHA is
    /// just the `n_kv_heads == n_heads` case -- there is no separate
    /// variant for it, and the halving GQA buys shows up entirely in
    /// `n_kv_heads`. `head_dim` is the K head width and `v_head_dim`
    /// the V's; equal for every architecture but MiMo-V2
    /// (`crate::kv_head_dims`).
    Gqa {
        n_kv_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
    },
    /// MLA in its *absorbed* form: the cache holds only the compressed
    /// latent plus the decoupled RoPE slice, `kv_lora_rank + rope_dim`
    /// scalars per token per layer, and K/V are reconstructed from it
    /// on the fly. One vector, not two -- there is no `* 2` here.
    ///
    /// **frink does not run this form today.** `mla::mla_forward_token`
    /// (and therefore `kimi_decoder`, `glm_dsa`, `glm52_decoder`)
    /// caches the *expanded* per-head K and V, so a real frink MLA run
    /// costs [`KvLayout::MlaExpanded`]. This variant is what the
    /// absorbed form would cost, and is the right number to plan
    /// against only once a decoder actually caches the latent.
    MlaLatent {
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    },
    /// MLA as frink actually caches it: per-head K of
    /// `qk_nope_head_dim + qk_rope_head_dim` and per-head V of
    /// `v_head_dim`, both materialised (`mla::mla_forward_token`'s
    /// `k_cache`/`v_cache`). K and V head dims differ, which is exactly
    /// why this cannot reuse the `Gqa` arm.
    MlaExpanded {
        n_heads: usize,
        k_head_dim: usize,
        v_head_dim: usize,
    },
}

impl KvLayout {
    /// Cached scalars one token contributes to one layer.
    pub fn elems_per_token_per_layer(self) -> u64 {
        match self {
            KvLayout::Gqa {
                n_kv_heads,
                head_dim,
                v_head_dim,
            } => n_kv_heads as u64 * (head_dim as u64 + v_head_dim as u64),
            KvLayout::MlaLatent {
                kv_lora_rank,
                qk_rope_head_dim,
            } => kv_lora_rank as u64 + qk_rope_head_dim as u64,
            KvLayout::MlaExpanded {
                n_heads,
                k_head_dim,
                v_head_dim,
            } => n_heads as u64 * (k_head_dim as u64 + v_head_dim as u64),
        }
    }

    /// One-line description of the arithmetic, for the report a user
    /// reads when they want to know why they got the context they got.
    pub fn describe(self) -> String {
        match self {
            KvLayout::Gqa {
                n_kv_heads,
                head_dim,
                v_head_dim,
            } if head_dim == v_head_dim => {
                format!("2 (K+V) x {n_kv_heads} kv-heads x {head_dim} head-dim")
            }
            KvLayout::Gqa {
                n_kv_heads,
                head_dim,
                v_head_dim,
            } => {
                format!("{n_kv_heads} kv-heads x ({head_dim} K head-dim + {v_head_dim} V head-dim)")
            }
            KvLayout::MlaLatent {
                kv_lora_rank,
                qk_rope_head_dim,
            } => format!(
                "MLA latent: {kv_lora_rank} kv_lora_rank + {qk_rope_head_dim} rope-dim \
                 (one vector, no K/V doubling)"
            ),
            KvLayout::MlaExpanded {
                n_heads,
                k_head_dim,
                v_head_dim,
            } => format!(
                "MLA expanded: {n_heads} heads x ({k_head_dim} K head-dim + \
                 {v_head_dim} V head-dim)"
            ),
        }
    }
}

/// The KV shape of a whole model: enough to price any context length.
///
/// How big one position is, times how many layers. How many positions
/// each of those layers still HOLDS is [`KvResidency`], and it is a
/// separate value because it is a property of the run rather than of
/// the model: by default every layer keeps every position, and behind
/// `FRINK_KV_WINDOW` a windowed layer does not (#61).
///
/// That is a statement about the STORES this engine allocates, not
/// about the architectures it runs -- see the module doc, and the two
/// tests that measure real `frink_core::cache::KvCache`s rather than
/// restating this multiplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub n_layers: usize,
    pub layout: KvLayout,
    pub elem: KvElem,
}

impl KvShape {
    /// Reads the shape off a config.
    ///
    /// `config.sliding_window` / `config.swa_pattern` are deliberately
    /// NOT read HERE: they describe what attention *reads*, and this
    /// module prices what the store *keeps*. The default store keeps
    /// everything (#33), so a windowed layer costs exactly what a
    /// full-attention one does. When a run evicts, that is what
    /// [`KvResidency::from_config`] is for -- and it reaches the window
    /// through the same `KvWindowPolicy` the decoder evicts with, not by
    /// reading those two fields a second time.
    ///
    /// Always produces a [`KvLayout::Gqa`] layout, because
    /// `ModelConfig` describes the generic GQA decoder -- the MLA
    /// stacks carry their own hyperparameters (`Deepseek2Hparams`,
    /// `MlaConfig`) and should build their shape with
    /// [`KvShape::mla_expanded`].
    pub fn from_config(config: &ModelConfig, elem: KvElem) -> Self {
        KvShape {
            n_layers: config.n_layers,
            layout: KvLayout::Gqa {
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                v_head_dim: config.v_head_dim(),
            },
            elem,
        }
    }

    /// The shape a frink MLA decoder really allocates -- see
    /// [`KvLayout::MlaExpanded`].
    pub fn mla_expanded(
        n_layers: usize,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        elem: KvElem,
    ) -> Self {
        KvShape {
            n_layers,
            layout: KvLayout::MlaExpanded {
                n_heads,
                k_head_dim: qk_nope_head_dim + qk_rope_head_dim,
                v_head_dim,
            },
            elem,
        }
    }

    /// The plan's headline number, and the only per-token number there
    /// is: bytes one token costs across every layer. Exact for f32/f16;
    /// for the block-quantized wires it is exact whenever a layer's
    /// per-token element count is a multiple of the 32-element block
    /// (true for every real head-dim/kv-head combination), and rounds
    /// up otherwise.
    ///
    /// This is also the divisor [`KvBudget::max_context`] uses. There is
    /// no separate "marginal" number any more: a marginal cost below the
    /// per-token cost would mean some layer stops growing, and none
    /// does.
    pub fn per_token_kv_bytes(&self) -> u64 {
        (self.n_layers as u64)
            .saturating_mul(self.elem.bytes_for(self.layout.elems_per_token_per_layer()))
    }

    /// Bytes one request's KV costs at `tokens` of context.
    pub fn kv_bytes_for_tokens(&self, tokens: usize) -> u64 {
        // Every multiplication here saturates, for the reason on
        // `KvElem::bytes_for`: `tokens` can arrive from an HTTP body.
        let per_layer = self.layout.elems_per_token_per_layer();
        (self.n_layers as u64)
            .saturating_mul(self.elem.bytes_for(per_layer.saturating_mul(tokens as u64)))
    }

    /// Bytes one request's KV costs at `tokens` of context when the
    /// stores EVICT behind a window (#61 step 2), once every layer has
    /// been through -- the number a measurement of the caches finds.
    ///
    /// The row counts come from [`KvWindow::rows_after`], which is the
    /// store's own rule and not a restatement of it: `KvCache` calls the
    /// same function to decide what to drop. Equal to
    /// [`Self::kv_bytes_for_tokens`] when `residency` keeps everything,
    /// which is what the default policy produces and what a test below
    /// asserts rather than assumes.
    ///
    /// [`Self::peak_kv_bytes_for_tokens`] is the number to ADMIT on;
    /// this one is smaller, and the difference is prefill.
    pub fn resident_kv_bytes_for_tokens(&self, tokens: usize, residency: &KvResidency) -> u64 {
        let per_layer = self.layout.elems_per_token_per_layer();
        residency
            .rows_per_layer(self.n_layers, tokens)
            .map(|rows| self.elem.bytes_for(per_layer.saturating_mul(rows as u64)))
            .fold(0u64, |acc, b| acc.saturating_add(b))
    }

    /// The number an admission decision must use: the resting ceiling,
    /// plus the one layer that is still mid-prefill.
    ///
    /// `Decoder::forward_batch` writes a whole prompt into layer `l`'s
    /// cache, attends over it, and only then hands the rows behind the
    /// window back -- before layer `l + 1` allocates any. So a long
    /// prompt costs ONE windowed layer's full history at a time rather
    /// than every windowed layer's at once, and that transient is real
    /// memory that has to be budgeted for. Charging only the resting
    /// number would be #33 in the other direction: an admitted request
    /// whose peak exceeds the estimate arrives as an OOM.
    ///
    /// The extra term is the largest single windowed layer's shortfall,
    /// because layers are prefilled one at a time.
    ///
    /// # Why the resting term is the CEILING and not `rows_after`
    ///
    /// [`KvWindow::rows_after`] is exact and it OSCILLATES: a cache
    /// that runs `slack` rows past its window and then drains holds
    /// anywhere in `[window, window + slack]`, cycling with period
    /// `slack + 1`. So bytes priced from it are not monotone in
    /// `tokens`, and [`KvBudget::max_context`] searches for the largest
    /// context that fits -- a search over a function that goes back
    /// down cannot be trusted to find the largest one.
    ///
    /// [`KvWindow::max_rows`] is the same type's own statement of the
    /// top of that cycle, so pricing against it is still the store's
    /// rule rather than a second opinion about it, it is monotone, and
    /// it is wrong only in the direction that refuses a context instead
    /// of OOMing on one. The gap is at most `slack` rows per windowed
    /// layer, against a term that already carries a whole layer's
    /// prompt.
    pub fn peak_kv_bytes_for_tokens(&self, tokens: usize, residency: &KvResidency) -> u64 {
        let per_layer = self.layout.elems_per_token_per_layer();
        let full = self.elem.bytes_for(per_layer.saturating_mul(tokens as u64));
        let resting = residency
            .ceiling_rows_per_layer(self.n_layers, tokens)
            .map(|rows| self.elem.bytes_for(per_layer.saturating_mul(rows as u64)))
            .fold(0u64, |acc, b| acc.saturating_add(b));
        let transient = residency
            .ceiling_rows_per_layer(self.n_layers, tokens)
            .map(|rows| {
                full.saturating_sub(self.elem.bytes_for(per_layer.saturating_mul(rows as u64)))
            })
            .max()
            .unwrap_or(0);
        resting.saturating_add(transient)
    }

    /// The sentence a user should be able to read and reproduce with a
    /// calculator.
    pub fn describe(&self) -> String {
        format!(
            "{} layers x [{}] x {} = {} bytes/token",
            self.n_layers,
            self.layout.describe(),
            self.elem.as_str(),
            self.per_token_kv_bytes()
        )
    }
}

/// What the stores really keep, per layer.
///
/// [`KvShape`] answers "how big is one position, times how many layers".
/// This answers "how many positions does each of those layers still
/// hold", which used to be "all of them" for every layer of every model
/// and now depends on whether `FRINK_KV_WINDOW` is on (#61).
///
/// **Deliberately not a field on `KvShape`.** `KvShape` is `Copy`, is
/// built by struct literal in more than one crate, and is the thing
/// every existing caller already has; a new field there would make the
/// no-eviction default a thing every caller restates. A residency is
/// asked for by the callers that price an evicting run, and the ones
/// that do not keep the number they always had.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvResidency {
    /// One entry per layer, in layer order. `None` means that layer
    /// keeps every position it was ever given.
    per_layer: Vec<Option<KvWindow>>,
}

impl KvResidency {
    /// Every layer keeps every position: the engine before #61, and the
    /// engine today unless the switch is on.
    pub fn keeps_everything(n_layers: usize) -> Self {
        KvResidency {
            per_layer: vec![None; n_layers],
        }
    }

    /// What `policy` will make the stores of `config` keep.
    ///
    /// Goes through [`KvWindowPolicy::layer_window`], which is the same
    /// call `Decoder::kv_window_for_layer` makes to decide what to
    /// evict. One expression, so there is nothing for the budget and the
    /// store to disagree about -- the disagreement being #33, where the
    /// budget capped a sliding layer no store ever capped and `-c auto`
    /// approved a context that did not fit.
    pub fn from_config(config: &ModelConfig, policy: KvWindowPolicy) -> Self {
        KvResidency {
            per_layer: (0..config.n_layers)
                .map(|l| policy.layer_window(config, l))
                .collect(),
        }
    }

    /// True when no layer evicts, i.e. this prices exactly what
    /// [`KvShape::kv_bytes_for_tokens`] prices.
    pub fn keeps_every_position(&self) -> bool {
        self.per_layer.iter().all(Option::is_none)
    }

    /// The window layer `layer_idx` evicts behind, if any.
    pub fn layer_window(&self, layer_idx: usize) -> Option<KvWindow> {
        self.per_layer.get(layer_idx).copied().flatten()
    }

    /// Rows each of `n_layers` layers holds at `tokens` of context.
    ///
    /// `n_layers` comes from the [`KvShape`] being priced rather than
    /// from `self`, and a layer this residency says nothing about keeps
    /// everything. A shape and a residency built from different configs
    /// is a caller error; charging the full cost is the safe way to be
    /// wrong about it.
    fn rows_per_layer(&self, n_layers: usize, tokens: usize) -> impl Iterator<Item = usize> + '_ {
        (0..n_layers).map(move |l| match self.layer_window(l) {
            Some(w) => w.rows_after(tokens),
            None => tokens,
        })
    }

    /// The most rows each layer can hold at `tokens` of context.
    ///
    /// [`Self::rows_per_layer`] is the instantaneous count and cycles
    /// through `[window, window + slack]`; this is the top of that
    /// cycle, taken from [`KvWindow::max_rows`] so the ceiling is the
    /// window type's own and not a second arithmetic beside it. Never
    /// below `rows_per_layer`, never above `tokens`, and non-decreasing
    /// in `tokens` -- which is what [`KvBudget::max_context`]'s search
    /// needs and the oscillating count cannot give it.
    fn ceiling_rows_per_layer(
        &self,
        n_layers: usize,
        tokens: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        (0..n_layers).map(move |l| match self.layer_window(l) {
            Some(w) => w.max_rows().min(tokens),
            None => tokens,
        })
    }

    /// How many layers evict, for a report that has to explain why the
    /// context is not simply the budget divided by a per-token cost.
    pub fn evicting_layers(&self) -> usize {
        self.per_layer.iter().filter(|w| w.is_some()).count()
    }
}

/// Which ceiling a rejection hit. The point of naming it is that the
/// two send an operator to different knobs: `ContextLength` is the
/// request's fault and shrinking the prompt fixes it, `DeviceMemory`
/// is the machine's and only a smaller model / smaller `n_ctx` /
/// bigger box does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ceiling {
    /// The request asked for more context than this deployment admitted.
    ContextLength,
    /// weights + KV + headroom does not fit the backend's budget.
    DeviceMemory,
}

impl Ceiling {
    /// Stable machine-readable code, safe to match on in a client.
    pub fn code(self) -> &'static str {
        match self {
            Ceiling::ContextLength => "context_length_exceeded",
            Ceiling::DeviceMemory => "device_memory_budget_exceeded",
        }
    }
}

/// A structured refusal: what it would have cost, what the ceiling was,
/// and which ceiling. Deliberately *not* an allocation failure -- the
/// whole point of computing this before the load is that nobody has to
/// read an OOM to find out.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {detail} (estimated {estimated_bytes} bytes vs limit {limit_bytes} bytes)",
        code = self.binding.code())]
pub struct KvBudgetError {
    pub binding: Ceiling,
    pub estimated_bytes: u64,
    pub limit_bytes: u64,
    pub detail: String,
}

impl KvBudgetError {
    pub fn code(&self) -> &'static str {
        self.binding.code()
    }

    /// Bytes over the ceiling (saturating, so a fit reads as `0`).
    pub fn overage_bytes(&self) -> u64 {
        self.estimated_bytes.saturating_sub(self.limit_bytes)
    }
}

/// A priced plan: every term of the inequality, kept separately so the
/// report can show the arithmetic rather than just the verdict.
///
/// Not `Copy`, because [`Self::residency`] is a per-layer vector. That
/// is deliberate: the residency belongs IN the budget rather than
/// beside it as an argument every caller has to remember to pass. A
/// method taking it as a parameter is exactly the shape that let the
/// store and the budget disagree in #33 -- one caller passes it, the
/// next one does not, and nothing says which run the number describes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBudget {
    /// Checkpoint bytes. See the module doc on why this is an
    /// approximation for mmap'd weights.
    pub weights_bytes: u64,
    /// Caller-supplied reserve for activations/scratch/allocator slack.
    pub activation_headroom_bytes: u64,
    /// What the backend says it can give us (see
    /// [`crate::device_budget::DeviceBudget::usable_bytes`]).
    pub device_budget_bytes: u64,
    pub shape: KvShape,
    /// What the stores this run allocates will really keep, per layer.
    ///
    /// [`KvResidency::keeps_everything`] is the engine's default and
    /// reproduces every number this type produced before #61.
    /// [`KvResidency::from_config`] with a live [`KvWindowPolicy`] is
    /// what a run with `FRINK_KV_WINDOW` on must be priced against --
    /// otherwise the store takes a saving the admission check refuses
    /// to spend, which is #33 read backwards: a context that would have
    /// fit, turned away.
    pub residency: KvResidency,
    /// KV caches are per request; concurrency multiplies them.
    pub concurrent_requests: usize,
}

impl KvBudget {
    /// Bytes left for KV after weights and headroom, or `0` when those
    /// two alone already overflow the budget.
    pub fn kv_bytes_available(&self) -> u64 {
        self.device_budget_bytes
            .saturating_sub(self.weights_bytes)
            .saturating_sub(self.activation_headroom_bytes)
    }

    /// KV bytes at `tokens` of context, across every concurrent
    /// request, at the moment the run costs most.
    ///
    /// The ONE expression the estimate, the refusal message and the
    /// context search all read, so a change to what the stores keep
    /// cannot reach two of the three and miss the other.
    pub fn kv_bytes_at(&self, tokens: usize) -> u64 {
        self.shape
            .peak_kv_bytes_for_tokens(tokens, &self.residency)
            .saturating_mul(self.concurrent_requests.max(1) as u64)
    }

    /// Total estimated resident bytes at `tokens` of context.
    pub fn estimated_bytes(&self, tokens: usize) -> u64 {
        self.weights_bytes + self.activation_headroom_bytes + self.kv_bytes_at(tokens)
    }

    /// The one-line check the plan is named for. `Ok` carries the
    /// estimate so a caller can log it on the happy path too.
    pub fn check(&self, tokens: usize) -> Result<u64, KvBudgetError> {
        let estimated = self.estimated_bytes(tokens);
        if estimated <= self.device_budget_bytes {
            return Ok(estimated);
        }
        Err(KvBudgetError {
            binding: Ceiling::DeviceMemory,
            estimated_bytes: estimated,
            limit_bytes: self.device_budget_bytes,
            detail: format!(
                "{} weight bytes + {} KV bytes at {tokens} tokens x{} concurrent + {} \
                 activation headroom exceeds the {} byte device budget",
                self.weights_bytes,
                self.kv_bytes_at(tokens),
                self.concurrent_requests.max(1),
                self.activation_headroom_bytes,
                self.device_budget_bytes,
            ),
        })
    }

    /// Largest context that fits: the biggest `tokens` for which
    /// [`Self::kv_bytes_at`] still sits inside
    /// `budget - weights - headroom`, floored to `granularity` and
    /// clamped to `cap` (the model's own trained context length).
    ///
    /// Every layer is in the cost. A sliding-window model used to have
    /// its windowed layers subtracted out of a divisor and added back
    /// as a saturated constant, which is the #33 under-estimate:
    /// nothing evicted, so nothing saturated. What is different now is
    /// that a run CAN evict (#61), and this asks the residency instead
    /// of assuming either answer.
    ///
    /// # Why a search and not a division
    ///
    /// With no eviction the cost is linear and
    /// `available / per_token_kv` is exact; a test below asserts this
    /// search returns that same number for a residency that keeps
    /// everything, so the closed form is not lost, it is checked
    /// against. With eviction the cost is piecewise: a windowed layer
    /// stops charging past `window + slack` while the dense ones keep
    /// going, so there is no single divisor to divide by and a division
    /// would price a 32k Gemma-3 context at 5.4x what it costs. The
    /// searched function is non-decreasing in `tokens` -- that is what
    /// `ceiling_rows_per_layer` is for -- so bisection finds the
    /// largest fitting context rather than any fitting context.
    pub fn max_context(&self, cap: usize, granularity: usize) -> ContextFit {
        let granularity = granularity.max(1);
        let concurrency = self.concurrent_requests.max(1) as u64;
        let available = self.kv_bytes_available();

        let (tokens, capped_by) = if available == 0 {
            (0, ContextCap::DeviceBudget)
        } else {
            // Bisection over `[0, cap]` only, never past it: the answer
            // above `cap` is always `cap`, so a search that stops there
            // needs no upper bound invented for it and cannot overflow
            // on a model with no KV at all (where every probe fits and
            // the answer is `cap`).
            let raw = self.largest_fitting_context(cap, available);
            if raw >= cap {
                (cap, ContextCap::ModelContextLength)
            } else {
                // Flooring must never turn a real answer into
                // "nothing fits": under one granularity step, report
                // the exact number of tokens rather than rounding it
                // away.
                let floored = if raw >= granularity {
                    (raw / granularity) * granularity
                } else {
                    raw
                };
                if floored >= cap {
                    (cap, ContextCap::ModelContextLength)
                } else {
                    (floored, ContextCap::DeviceBudget)
                }
            }
        };

        ContextFit {
            tokens,
            cap,
            granularity,
            capped_by,
            kv_available_bytes: available,
            per_token_kv_bytes: self.shape.per_token_kv_bytes(),
            concurrent_requests: concurrency as usize,
            kv_bytes: self.kv_bytes_at(tokens),
            evicting_layers: self.residency.evicting_layers(),
            weights_bytes: self.weights_bytes,
            activation_headroom_bytes: self.activation_headroom_bytes,
            device_budget_bytes: self.device_budget_bytes,
        }
    }

    /// Largest `tokens` in `0..=cap` whose KV still fits `available`.
    ///
    /// `kv_bytes_at` is non-decreasing in `tokens`, so the predicate
    /// "fits" is a prefix of the range and bisection is exact.
    fn largest_fitting_context(&self, cap: usize, available: u64) -> usize {
        if self.kv_bytes_at(cap) <= available {
            return cap;
        }
        // Invariant: `lo` fits and `hi` does not. `0` fits because a
        // zero-token context costs no KV bytes at all.
        let (mut lo, mut hi) = (0usize, cap);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.kv_bytes_at(mid) <= available {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Why `--ctx auto` chose the number it chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextCap {
    /// The model's own trained context length was the smaller ceiling.
    ModelContextLength,
    /// Memory ran out first.
    DeviceBudget,
}

/// The answer `--ctx auto` produces, with every term that went into it
/// so the user can check the division by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextFit {
    pub tokens: usize,
    pub cap: usize,
    pub granularity: usize,
    pub capped_by: ContextCap,
    pub kv_available_bytes: u64,
    /// Bytes one token of context costs across every layer, with
    /// nothing evicting. The divisor when `evicting_layers` is 0, and
    /// an upper bound on the marginal cost otherwise.
    pub per_token_kv_bytes: u64,
    pub concurrent_requests: usize,
    /// KV bytes at `tokens`: [`KvBudget::kv_bytes_at`], which is what
    /// the fit was actually decided on.
    pub kv_bytes: u64,
    /// How many layers stop growing at their window. Zero for every
    /// model unless `FRINK_KV_WINDOW` is on, and the reason the
    /// division in [`Display`] stops being the whole story when it is
    /// not.
    ///
    /// [`Display`]: std::fmt::Display
    pub evicting_layers: usize,
    pub weights_bytes: u64,
    pub activation_headroom_bytes: u64,
    pub device_budget_bytes: u64,
}

impl std::fmt::Display for ContextFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ctx auto = {} tokens ({}): ({} device budget - {} weights - {} activation headroom) \
             = {} for KV; / {} bytes/token/request / {} request(s) -> rounded down to a multiple \
             of {} (reported exactly below one step), capped at the model's {} trained context. \
             KV at the chosen context: {} bytes.{}",
            self.tokens,
            match self.capped_by {
                ContextCap::ModelContextLength => "limited by the model's context length",
                ContextCap::DeviceBudget => "limited by the device memory budget",
            },
            self.device_budget_bytes,
            self.weights_bytes,
            self.activation_headroom_bytes,
            self.kv_available_bytes,
            self.per_token_kv_bytes,
            self.concurrent_requests,
            self.granularity,
            self.cap,
            self.kv_bytes,
            // Said explicitly rather than left for the reader to
            // notice the division does not reproduce the answer: with
            // eviction on, the per-token figure above is the cost only
            // until each windowed layer saturates, and the chosen
            // context came from the search that knows that.
            match self.evicting_layers {
                0 => String::new(),
                n => format!(
                    " {n} of those layers stop growing at their sliding window \
                     (FRINK_KV_WINDOW), so the per-token figure is the cost before \
                     they saturate, not a divisor that reproduces this answer."
                ),
            },
        )
    }
}

/// Granularity `--ctx auto` floors to. Small enough that the rounding
/// never costs a meaningful amount of context, round enough that the
/// reported number looks chosen rather than computed.
pub const CTX_AUTO_GRANULARITY: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3.1-8B's real shape: 32 layers, 8 kv-heads (GQA 4:1),
    /// head_dim 128. llama.cpp reports 1 MiB/token at f32 for exactly
    /// this model, which is the number reproduced here by hand:
    /// 32 * 2 * 8 * 128 * 4 = 262144 bytes.
    fn llama31_8b() -> KvShape {
        KvShape {
            n_layers: 32,
            layout: KvLayout::Gqa {
                n_kv_heads: 8,
                head_dim: 128,
                v_head_dim: 128,
            },
            elem: KvElem::F32,
        }
    }

    #[test]
    fn gqa_per_token_kv_matches_the_hand_computed_byte_count() {
        let shape = llama31_8b();
        assert_eq!(shape.layout.elems_per_token_per_layer(), 2 * 8 * 128);
        assert_eq!(shape.per_token_kv_bytes(), 32 * 2 * 8 * 128 * 4);
        assert_eq!(shape.per_token_kv_bytes(), 262_144);
        // f16 is exactly half; a block-quantized store is 34/32 of the
        // element count, not 1 byte flat.
        assert_eq!(
            KvShape {
                elem: KvElem::F16,
                ..shape
            }
            .per_token_kv_bytes(),
            131_072
        );
        assert_eq!(
            KvShape {
                elem: KvElem::Q8_0,
                ..shape
            }
            .per_token_kv_bytes(),
            32 * (2 * 8 * 128 / 32) * 34
        );
        assert_eq!(
            KvShape {
                elem: KvElem::Q4_0,
                ..shape
            }
            .per_token_kv_bytes(),
            32 * (2 * 8 * 128 / 32) * 18
        );
    }

    #[test]
    fn ctk_names_map_onto_the_widths_metal_really_writes() {
        assert_eq!(KvElem::from_ctk("f16"), KvElem::F16);
        assert_eq!(KvElem::from_ctk("f32"), KvElem::F32);
        assert_eq!(KvElem::from_ctk("Q8_0"), KvElem::Q8_0);
        // fp8 shares Q8_0's wire, per MetalKvDtype.
        assert_eq!(KvElem::from_ctk("fp8"), KvElem::Q8_0);
        assert_eq!(KvElem::from_ctk("fp8"), KvElem::Q8_0);
        assert_eq!(KvElem::from_ctk("q4_0"), KvElem::Q4_0);
        // An unrecognised value falls back to f16, as does junk.
        assert_eq!(KvElem::from_ctk("nonsense"), KvElem::F16);
        assert_eq!(KvElem::from_ctk("  nonsense "), KvElem::F16);
    }

    #[test]
    fn mha_costs_exactly_the_gqa_ratio_more_than_gqa() {
        // Same model with n_kv_heads == n_heads (32) instead of 8: MHA
        // is 4x the KV of 4:1 GQA, and nothing else changes.
        let gqa = llama31_8b();
        let mha = KvShape {
            layout: KvLayout::Gqa {
                n_kv_heads: 32,
                head_dim: 128,
                v_head_dim: 128,
            },
            ..gqa
        };
        assert_eq!(mha.per_token_kv_bytes(), 4 * gqa.per_token_kv_bytes());
        assert_eq!(mha.per_token_kv_bytes(), 32 * 2 * 32 * 128 * 4);
    }

    /// A small alternating-SWA config: 6 layers, every 3rd of them full
    /// attention, a 4-position window. Small enough that a test can
    /// allocate the real stores; alternating, which is the case
    /// `ModelConfig::uniform_sliding_window` refuses to let any store
    /// recycle.
    fn alternating_swa_config() -> ModelConfig {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 6;
        cfg.n_kv_heads = 1;
        cfg.head_dim = 8;
        cfg.sliding_window = Some(4);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(3, false);
        cfg
    }

    /// **The property this module got wrong, measured rather than
    /// restated.**
    ///
    /// The old budget capped a sliding layer at `window + chunk - 1`
    /// positions, but `frink_core::cache::KvCache` -- the store the CLI
    /// allocates and the store the server allocates on every non-paged
    /// path -- has no window concept: `push` extends `k`/`v` for every
    /// position, in every layer. So the budget under-priced gpt-oss by
    /// 2x and Gemma-3-4B by 5.8x, `-c auto` approved a context that did
    /// not fit, and the failure arrived as an OOM instead of a refusal
    /// (#33).
    ///
    /// This pushes real positions into the real caches and compares the
    /// bytes they hold against the budget's number. Recomputing the
    /// budget's own multiplication here would assert nothing: the code
    /// was not wrong about arithmetic, it was wrong about the world.
    #[test]
    fn the_budget_prices_exactly_what_the_kv_store_allocates_for_an_alternating_swa_model() {
        let cfg = alternating_swa_config();
        // Well past the 4-position window, which is the whole point:
        // under the old cap the sliding layers stopped being charged
        // here.
        let tokens = 64;
        assert!(
            cfg.sliding_window.is_some() && cfg.uniform_sliding_window().is_none(),
            "the fixture must be an alternating-SWA model, or this proves nothing"
        );

        let mut caches: Vec<frink_core::cache::KvCache> = cfg.new_kv_caches();
        let step = vec![0f32; cfg.n_kv_heads * cfg.head_dim];
        for _ in 0..tokens {
            for cache in caches.iter_mut() {
                cache
                    .push(&step, &step)
                    .expect("a cache built with `new` always accepts a push");
            }
        }
        let allocated: u64 = caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) as u64 * std::mem::size_of::<f32>() as u64)
            .sum();

        let shape = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(
            shape.kv_bytes_for_tokens(tokens),
            allocated,
            "the budget must price what the store holds"
        );
        // The store kept every position in every layer, window or not.
        assert_eq!(allocated, shape.per_token_kv_bytes() * tokens as u64);
        // And the two entry points agree when nothing evicts, rather
        // than being two independent multiplications that happen to
        // match today.
        assert_eq!(
            shape
                .resident_kv_bytes_for_tokens(tokens, &KvResidency::keeps_everything(cfg.n_layers)),
            allocated
        );
    }

    /// **The same property, measured again, now that a store evicts.**
    ///
    /// The sibling above is the default and stays the default. This is
    /// the `FRINK_KV_WINDOW` case, and it is asserted the same way for
    /// the same reason: by pushing real positions into real
    /// `frink_core::cache::KvCache`s, evicting them the way
    /// `Decoder::evict_layer_kv` does, and comparing the bytes they hold
    /// against the budget's number. If the budget restated the window
    /// rule instead of taking it from `KvWindow::rows_after`, this test
    /// would pass while the two drifted -- which is exactly how #33
    /// survived long enough to approve a context that did not fit.
    #[test]
    fn the_budget_prices_exactly_what_an_evicting_kv_store_holds() {
        let cfg = alternating_swa_config();
        let tokens = 64;
        let residency = KvResidency::from_config(&cfg, KvWindowPolicy::on());
        assert!(
            !residency.keeps_every_position(),
            "the fixture must have windowed layers, or this proves nothing"
        );
        assert!(
            (0..cfg.n_layers).any(|l| residency.layer_window(l).is_none()),
            "the fixture must ALSO have dense layers: they are the half that keeps costing"
        );

        let mut caches: Vec<frink_core::cache::KvCache> = cfg.new_kv_caches();
        for (l, cache) in caches.iter_mut().enumerate() {
            if let Some(w) = residency.layer_window(l) {
                cache.arm_window(w);
            }
        }
        let step = vec![0f32; cfg.n_kv_heads * cfg.head_dim];
        for _ in 0..tokens {
            for cache in caches.iter_mut() {
                cache
                    .push(&step, &step)
                    .expect("a cache built with `new` always accepts a push");
                cache.evict_behind_window();
            }
        }
        let held: u64 = caches
            .iter()
            .map(|c| (c.k.len() + c.v.len()) as u64 * std::mem::size_of::<f32>() as u64)
            .sum();

        let shape = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(
            shape.resident_kv_bytes_for_tokens(tokens, &residency),
            held,
            "the budget must price what the evicting store holds"
        );
        // The saving is real: strictly less than pricing every position.
        assert!(
            held < shape.kv_bytes_for_tokens(tokens),
            "eviction saved nothing: {held} vs {}",
            shape.kv_bytes_for_tokens(tokens)
        );
        // And the number to admit on is above the number at rest, because
        // one layer holds the whole prompt while it is being prefilled.
        assert!(shape.peak_kv_bytes_for_tokens(tokens, &residency) > held);
        // ...but never above pricing every layer at every position,
        // which is what the engine costs today.
        assert!(
            shape.peak_kv_bytes_for_tokens(tokens, &residency) <= shape.kv_bytes_for_tokens(tokens)
        );
    }

    /// The default policy prices exactly what it always did. A switch
    /// that is off must be invisible to the arithmetic.
    #[test]
    fn the_default_policy_prices_every_layer_at_every_position() {
        let cfg = alternating_swa_config();
        let residency = KvResidency::from_config(&cfg, KvWindowPolicy::off());
        assert!(residency.keeps_every_position());
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        for tokens in [0usize, 1, 63, 64, 4096] {
            assert_eq!(
                shape.resident_kv_bytes_for_tokens(tokens, &residency),
                shape.kv_bytes_for_tokens(tokens)
            );
            assert_eq!(
                shape.peak_kv_bytes_for_tokens(tokens, &residency),
                shape.kv_bytes_for_tokens(tokens)
            );
        }
    }

    /// The headline number from #61, priced through the residency rather
    /// than asserted: Gemma-3-4B at a 32k context.
    ///
    /// 34 layers, 4 kv-heads, head_dim 256, host f32, a 1024-position
    /// window on five layers out of every six. The full price is the
    /// 9.13 GB the issue measured; the windowed one is what the store
    /// now holds.
    #[test]
    fn gemma3_4b_at_32k_costs_a_fraction_of_what_it_did() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 34;
        cfg.n_kv_heads = 4;
        cfg.head_dim = 256;
        cfg.sliding_window = Some(1024);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        let tokens = 32_768;

        let full = shape.kv_bytes_for_tokens(tokens);
        assert_eq!(full, 9_126_805_504, "the number #61 measured");

        let residency = KvResidency::from_config(&cfg, KvWindowPolicy::on());
        let resting = shape.resident_kv_bytes_for_tokens(tokens, &residency);
        let peak = shape.peak_kv_bytes_for_tokens(tokens, &residency);
        // Pinned rather than bounded, so a change to the default slack
        // shows up as a memory number moving rather than as nothing.
        // 5 of the 34 layers are full attention (`swa_pattern` 6) and
        // still hold every position; at 1.34 GB they are most of what is
        // left. The 29 windowed ones hold 1475 rows each instead of
        // 32768.
        assert_eq!(resting, 1_692_590_080, "5.4x less than the 9.13 GB above");
        // The admission number prices each windowed layer at the top of
        // its cycle (`KvWindow::max_rows`, 1536 here) rather than at the
        // instantaneous `rows_after`, because `max_context` searches
        // this function and a search needs it not to fall. That costs
        // 13,991,936 bytes -- 0.7% -- against a term that already
        // carries a whole layer's prompt.
        assert_eq!(
            peak, 1_962_934_272,
            "resting ceiling plus the one windowed layer still mid-prefill"
        );
        assert!(peak < full && peak > resting);
        // The saving is what makes the difference worth having: the
        // whole point of #61 is that this is the number `-c auto`
        // divides a machine by, and 4.65x is a 32k context fitting on a
        // 16 GB box or not.
        assert!(full / peak >= 4, "{full} / {peak}");
    }

    /// The pool-backed store is the other thing a server allocates, and
    /// it reserves `max_seq_len` positions for EVERY layer up front
    /// (`KvCache::with_pool`), rounded up to whole blocks. The budget
    /// must never be under that either -- an admitted request whose
    /// reservation exceeds the estimate is exactly the OOM #33 is about.
    #[test]
    fn the_pool_backed_store_never_reserves_more_positions_than_the_budget_priced() {
        use frink_core::cache::{KvBlockPool, KvCache};
        use std::sync::{Arc, Mutex};

        let cfg = alternating_swa_config();
        let tokens = 64usize;
        let block_size = 16usize;
        let pool = Arc::new(Mutex::new(KvBlockPool::new(
            block_size,
            tokens.div_ceil(block_size) * cfg.n_layers,
        )));
        let caches: Vec<KvCache> = (0..cfg.n_layers)
            .map(|_| {
                KvCache::with_pool(cfg.n_kv_heads, cfg.head_dim, Arc::clone(&pool), tokens)
                    .expect("the pool was sized for exactly this")
            })
            .collect();
        let reserved: u64 = caches
            .iter()
            .map(|c| c.k.capacity() as u64 + c.v.capacity() as u64)
            .sum::<u64>()
            * std::mem::size_of::<f32>() as u64;

        let priced = KvShape::from_config(&cfg, KvElem::F32).kv_bytes_for_tokens(tokens);
        // Equal here because `tokens` is a whole number of blocks; the
        // assertion that matters is the direction, which holds for any
        // block size.
        assert!(
            priced >= reserved,
            "budget priced {priced} bytes, the pool reserved {reserved}"
        );
        assert_eq!(priced, reserved);
    }

    /// The two checkpoints #33 measured, at their own byte counts.
    ///
    /// These constants are what the stores allocate, taken from the
    /// issue, not from this module's formula. The numbers the old code
    /// produced were 6,448,742,400 for gpt-oss (half) and 1,585,446,912
    /// for Gemma-3-4B (a sixth).
    #[test]
    fn gpt_oss_and_gemma3_cost_what_the_issue_measured() {
        // gpt-oss-20b: 24 layers, 8 kv-heads, head_dim 64, host f32,
        // 131072 context. Alternating 128-position window, priced at 0.
        let mut gpt_oss = crate::config::test_dense_fixture();
        gpt_oss.n_layers = 24;
        gpt_oss.n_kv_heads = 8;
        gpt_oss.head_dim = 64;
        gpt_oss.sliding_window = Some(128);
        gpt_oss.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
        assert_eq!(
            KvShape::from_config(&gpt_oss, KvElem::F32).kv_bytes_for_tokens(131_072),
            12_884_901_888
        );

        // Gemma-3-4B: 34 layers, 4 kv-heads, head_dim 256, 32768 tokens.
        let mut gemma3 = crate::config::test_dense_fixture();
        gemma3.n_layers = 34;
        gemma3.n_kv_heads = 4;
        gemma3.head_dim = 256;
        gemma3.sliding_window = Some(1024);
        gemma3.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
        assert_eq!(
            KvShape::from_config(&gemma3, KvElem::F32).kv_bytes_for_tokens(32_768),
            9_126_805_504
        );
    }

    /// A window changes what attention READS, not what the store KEEPS,
    /// so it may not change the price. Stated as an equality between two
    /// configs rather than as a comment, so re-introducing a cap fails
    /// here.
    #[test]
    fn a_windowed_config_is_priced_identically_to_the_same_config_without_a_window() {
        let windowed = alternating_swa_config();
        let mut full = windowed.clone();
        full.sliding_window = None;
        full.swa_layers = crate::swa_layers::SwaLayers::All;
        for tokens in [1, 3, 4, 5, 64, 100_000] {
            assert_eq!(
                KvShape::from_config(&windowed, KvElem::F32).kv_bytes_for_tokens(tokens),
                KvShape::from_config(&full, KvElem::F32).kv_bytes_for_tokens(tokens),
                "tokens={tokens}"
            );
        }
    }

    #[test]
    fn mla_latent_is_one_vector_and_far_cheaper_than_the_expanded_form() {
        // DeepSeek-V2's real MLA numbers: kv_lora_rank 512,
        // qk_rope_head_dim 64, qk_nope_head_dim 128, v_head_dim 128,
        // 128 heads, 60 layers.
        let latent = KvShape {
            n_layers: 60,
            layout: KvLayout::MlaLatent {
                kv_lora_rank: 512,
                qk_rope_head_dim: 64,
            },
            elem: KvElem::F32,
        };
        // 512 + 64 = 576 scalars per token per layer -- one vector, no
        // K/V doubling.
        assert_eq!(latent.layout.elems_per_token_per_layer(), 576);
        assert_eq!(latent.per_token_kv_bytes(), 60 * 576 * 4);

        let expanded = KvShape::mla_expanded(60, 128, 128, 64, 128, KvElem::F32);
        // 128 heads x (192 K + 128 V) = 40960 scalars per token/layer.
        assert_eq!(
            expanded.layout.elems_per_token_per_layer(),
            128 * (192 + 128)
        );
        assert_eq!(expanded.per_token_kv_bytes(), 60 * 40_960 * 4);
        // The absorbed form is ~71x cheaper; this is exactly why the
        // distinction is worth carrying rather than assuming.
        assert!(expanded.per_token_kv_bytes() / latent.per_token_kv_bytes() > 70);

        // A same-sized GQA model for scale: 128 kv-heads x 128 head_dim.
        let gqa = KvShape {
            layout: KvLayout::Gqa {
                n_kv_heads: 128,
                head_dim: 128,
                v_head_dim: 128,
            },
            ..latent
        };
        assert_eq!(gqa.per_token_kv_bytes(), 60 * 2 * 128 * 128 * 4);
    }

    #[test]
    fn from_config_reads_layers_heads_and_head_dim() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 12;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 64;
        cfg.sliding_window = None;
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        assert_eq!(shape.n_layers, 12);
        assert_eq!(shape.per_token_kv_bytes(), 12 * 2 * 2 * 64 * 4);

        // A uniform window changes nothing either: the paged store that
        // could recycle for one still holds the whole prompt, and it is
        // a context length this prices.
        cfg.sliding_window = Some(256);
        cfg.swa_layers = crate::swa_layers::SwaLayers::All;
        assert_eq!(KvShape::from_config(&cfg, KvElem::F32), shape);
    }

    fn budget(weights: u64, device: u64, shape: KvShape) -> KvBudget {
        KvBudget {
            weights_bytes: weights,
            activation_headroom_bytes: 0,
            device_budget_bytes: device,
            shape,
            residency: KvResidency::keeps_everything(shape.n_layers),
            concurrent_requests: 1,
        }
    }

    #[test]
    fn check_accepts_a_fitting_context_and_names_the_binding_ceiling_otherwise() {
        let shape = llama31_8b(); // 262144 bytes/token
        let b = budget(1_000_000, 1_000_000 + 262_144 * 10, shape);
        assert_eq!(b.check(10).unwrap(), 1_000_000 + 262_144 * 10);
        let err = b.check(11).expect_err("one token past the budget");
        assert_eq!(err.binding, Ceiling::DeviceMemory);
        assert_eq!(err.code(), "device_memory_budget_exceeded");
        assert_eq!(err.estimated_bytes, 1_000_000 + 262_144 * 11);
        assert_eq!(err.limit_bytes, 1_000_000 + 262_144 * 10);
        assert_eq!(err.overage_bytes(), 262_144);
    }

    #[test]
    fn concurrency_multiplies_kv_but_not_weights() {
        let shape = llama31_8b();
        let one = budget(1_000, 1 << 40, shape);
        let four = KvBudget {
            concurrent_requests: 4,
            ..one.clone()
        };
        assert_eq!(
            four.estimated_bytes(100) - 1_000,
            4 * (one.estimated_bytes(100) - 1_000)
        );
    }

    #[test]
    fn max_context_is_the_closed_form_division_floored_to_granularity() {
        let shape = llama31_8b(); // 262144 bytes/token
                                  // Room for exactly 1000 tokens of KV after weights.
        let b = budget(5_000_000, 5_000_000 + 262_144 * 1000, shape);
        let fit = b.max_context(131_072, 256);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        // 1000 floored to a 256-token step is 768.
        assert_eq!(fit.tokens, 768);
        assert_eq!(fit.kv_available_bytes, 262_144 * 1000);
        assert_eq!(fit.per_token_kv_bytes, 262_144);
        // The chosen context really does fit.
        assert!(b.check(fit.tokens).is_ok());
        // One granularity step further does not.
        assert!(b.check(fit.tokens + 256).is_err());
    }

    /// **The closed form is not gone, it is checked against.**
    ///
    /// `max_context` used to be one division and is now a search,
    /// because with eviction there is no single divisor. A search is
    /// free to be subtly wrong in a way a division cannot be, so the
    /// division stays here as the oracle: for a run where nothing
    /// evicts -- which is every run unless `FRINK_KV_WINDOW` is on --
    /// the searched answer must be exactly
    /// `available / per_token_kv`, at every budget, not just at the
    /// round ones.
    #[test]
    fn the_context_search_reproduces_the_closed_form_when_nothing_evicts() {
        let shape = llama31_8b(); // 262144 bytes/token
        let per_token = shape.per_token_kv_bytes();
        for tokens_of_room in [0u64, 1, 7, 999, 1000, 1001, 65_536] {
            for slack in [0u64, 1, per_token - 1] {
                let device = 5_000_000 + per_token * tokens_of_room + slack;
                let b = budget(5_000_000, device, shape);
                // Granularity 1 so the comparison is against the raw
                // division rather than against the rounding.
                let fit = b.max_context(131_072, 1);
                assert_eq!(
                    fit.tokens as u64,
                    (per_token * tokens_of_room + slack) / per_token,
                    "budget {device} disagreed with the division it replaced"
                );
            }
        }
    }

    /// **The property the search rests on.**
    ///
    /// Bisection finds the largest fitting context only if "fits" is a
    /// prefix of the range, i.e. if the cost never falls as the context
    /// grows. `KvWindow::rows_after` OSCILLATES between `window` and
    /// `window + slack`, so pricing admission from it directly would
    /// break exactly that, and a search over it could stop one cycle
    /// early and report a context smaller than the one that fits.
    #[test]
    fn the_admission_ceiling_never_falls_as_the_context_grows() {
        let cfg = alternating_swa_config();
        let residency = KvResidency::from_config(&cfg, KvWindowPolicy::on());
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        let mut previous = 0u64;
        // Well past the window (4) and its default slack (2), so the
        // whole oscillation is covered rather than only the ramp.
        for tokens in 0..64 {
            let bytes = shape.peak_kv_bytes_for_tokens(tokens, &residency);
            assert!(
                bytes >= previous,
                "cost fell from {previous} to {bytes} between {} and {tokens} tokens",
                tokens.saturating_sub(1)
            );
            previous = bytes;
        }
    }

    /// The ceiling admission is decided on must never sit below what a
    /// measurement of the caches would find, or the run is admitted
    /// against a number smaller than the memory it takes. `resident_`
    /// is that measurement (asserted against real `KvCache`s above);
    /// this is the ordering between the two, at every context, not just
    /// at the one the sibling test measures.
    #[test]
    fn the_admission_ceiling_is_never_below_what_the_store_will_hold() {
        let cfg = alternating_swa_config();
        let residency = KvResidency::from_config(&cfg, KvWindowPolicy::on());
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        for tokens in 0..64 {
            assert!(
                shape.peak_kv_bytes_for_tokens(tokens, &residency)
                    >= shape.resident_kv_bytes_for_tokens(tokens, &residency),
                "admission under-priced the resting store at {tokens} tokens"
            );
        }
    }

    /// **The gap this wiring closed.**
    ///
    /// #61 step 2 taught the store to evict and left the budget
    /// pricing every layer at every position, so a Gemma-3-shaped model
    /// with `FRINK_KV_WINDOW` on kept 5x less KV than `-c auto` was
    /// dividing by, and the context it could really carry was refused.
    /// The two arms here differ ONLY in the policy the residency was
    /// built from.
    #[test]
    fn an_evicting_run_is_offered_more_context_than_a_non_evicting_one() {
        let cfg = alternating_swa_config();
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        let base = budget(1_000, 1_000 + shape.per_token_kv_bytes() * 64, shape);
        let evicting = KvBudget {
            residency: KvResidency::from_config(&cfg, KvWindowPolicy::on()),
            ..base.clone()
        };
        assert!(
            base.residency.keeps_every_position(),
            "the control arm must be the engine's default"
        );

        let plain = base.max_context(131_072, 1);
        let windowed = evicting.max_context(131_072, 1);
        assert!(
            windowed.tokens > plain.tokens,
            "eviction bought no context: {} vs {}",
            windowed.tokens,
            plain.tokens
        );
        assert_eq!(windowed.evicting_layers, 4, "4 of 6 layers slide");
        assert_eq!(plain.evicting_layers, 0);
        // The context the evicting run was offered is one the plain
        // budget refuses, and the evicting budget accepts. That is the
        // whole difference, stated as the decision rather than as a
        // number.
        assert!(evicting.check(windowed.tokens).is_ok());
        assert!(base.check(windowed.tokens).is_err());
        // And the report says why the division above it no longer
        // reproduces the answer, rather than leaving a reader to
        // subtract two numbers that do not match.
        assert!(
            windowed
                .to_string()
                .contains("stop growing at their sliding window"),
            "{windowed}"
        );
    }

    #[test]
    fn max_context_clamps_to_the_models_trained_context_when_memory_is_plentiful() {
        let b = budget(1_000, 1 << 40, llama31_8b());
        let fit = b.max_context(8192, 256);
        assert_eq!(fit.tokens, 8192);
        assert_eq!(fit.capped_by, ContextCap::ModelContextLength);
    }

    /// Flooring must not round a small-but-real answer down to "nothing
    /// fits" -- found by running `--ctx-size auto` under a tight
    /// `FRINK_DEVICE_BUDGET_BYTES`, where 227 tokens genuinely fitted
    /// and the 256-token granularity reported 0.
    #[test]
    fn a_context_under_one_granularity_step_is_reported_exactly_not_floored_away() {
        let shape = llama31_8b(); // 262144 bytes/token
        let b = budget(1_000, 1_000 + 262_144 * 100, shape);
        let fit = b.max_context(131_072, 256);
        assert_eq!(fit.tokens, 100);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert!(b.check(fit.tokens).is_ok());
        assert!(b.check(fit.tokens + 1).is_err());
    }

    #[test]
    fn max_context_is_zero_when_the_weights_alone_do_not_fit() {
        let b = budget(10_000_000, 1_000_000, llama31_8b());
        let fit = b.max_context(8192, 256);
        assert_eq!(fit.tokens, 0);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert_eq!(fit.kv_available_bytes, 0);
        assert!(b.check(0).is_err(), "weights alone already overflow");
    }

    /// `--ctx auto` on a windowed model used to answer "the model's own
    /// context length" however small the budget was, because the
    /// divisor had every sliding layer taken out of it and a model whose
    /// every layer slid divided by zero bytes per token. It is now
    /// bounded by memory like any other model, and the context it picks
    /// has to survive `check` -- which is the assertion that would have
    /// caught the OOM.
    #[test]
    fn a_windowed_model_is_bounded_by_memory_like_any_other() {
        let mut cfg = alternating_swa_config();
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(1, false); // every layer slides: the old zero divisor
        let shape = KvShape::from_config(&cfg, KvElem::F32);
        // Room for 1024 tokens, against a model that would like 1e6.
        let b = budget(1_000, 1_000 + shape.per_token_kv_bytes() * 1024, shape);
        let fit = b.max_context(1_000_000, 256);
        assert_eq!(fit.capped_by, ContextCap::DeviceBudget);
        assert_eq!(fit.tokens, 1024);
        assert!(b.check(fit.tokens).is_ok());
        assert!(
            b.check(fit.tokens + 1).is_err(),
            "the chosen context must be the largest that fits"
        );
    }

    #[test]
    fn ctx_auto_explanation_names_every_term_it_divided() {
        let b = budget(5_000_000, 5_000_000 + 262_144 * 1000, llama31_8b());
        let text = b.max_context(131_072, CTX_AUTO_GRANULARITY).to_string();
        assert!(text.contains("ctx auto = 768 tokens"), "{text}");
        assert!(text.contains("262144"), "per-token divisor missing: {text}");
        assert!(text.contains("5000000"), "weights term missing: {text}");
        assert!(text.contains("131072"), "model cap missing: {text}");
    }
}
