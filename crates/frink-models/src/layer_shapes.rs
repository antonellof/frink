//! Per-layer attention and FFN shapes: llama.cpp's `n_head(il)`,
//! `n_head_kv(il)` and `n_ff(il)`.
//!
//! llama.cpp reads `{arch}.attention.head_count`, `.head_count_kv` and
//! `{arch}.feed_forward_length` as a scalar OR an `n_layer`-long array
//! for EVERY architecture (`get_key_or_arr`, `llama-model.cpp:1149-1158`)
//! and keeps three `std::array<uint32_t, LLAMA_MAX_LAYERS>`
//! (`llama-hparams.h:83-85`). `LLAMA_LOAD_LOCALS` then hands most
//! graphs layer 0's value (`n_head()` defaults `il = 0`,
//! `llama-hparams.h:320-324`), so an array only matters to the graphs
//! that index it. **Measured, not assumed**: every one of the 140
//! `src/models/*.cpp` was scanned for `n_head(i)`, `n_head_kv(i)`,
//! `n_ff(i)`, `n_embd_k_gqa(i)`, `n_embd_v_gqa(i)` and the `_arr`
//! fields, in both the tensor loader and the graph. The architectures
//! that honour a per-layer shape in BOTH are [`PER_LAYER_SHAPE_ARCHS`];
//! `granite.cpp:204` reads `n_head(il)` in its graph but sizes its
//! tensors from layer 0 (`:68`), so a heterogeneous Granite file fails
//! in llama.cpp's own loader and is not in the list.
//!
//! frink's [`ModelConfig`] carries `n_heads`, `n_kv_heads` and
//! `moe.expert_ffn_dim` as scalars, and every host body read them once
//! above its layer loop. This module is the seam that makes them
//! per-layer, with the uniform case being the one where every layer
//! agrees: [`ModelConfig::layer_shape`] is the ONLY way a layer body
//! learns its head counts, and the scalars are documented as the
//! WIDEST layer's, which is what a memory budget needs and what no
//! per-layer computation may read.
//!
//! Where disagreement is made to fail rather than drift:
//!
//! - [`ModelConfig::new_kv_caches`] sizes each layer's cache from its
//!   own shape, and `KvCache::push` asserts the row width, so a cache
//!   built from the scalar for a narrower layer panics on the first
//!   token rather than storing a misaligned history.
//! - `Decoder::metal_can_serve_model` refuses every fused Metal launch
//!   for a non-uniform model: those launches take ONE `n_heads`
//!   argument and one `MetalKvBuffers` geometry.
//! - [`AttnShape`] is an enum, so a layer with no attention or with
//!   deci's `wo`-only "linear attention" cannot be spelled as
//!   `n_heads = 0` and fall through a `0..n_heads` loop doing nothing.
//!
//! What it does NOT close, and says so: `nanbeige` rewrites the arrays
//! to walk its physical layers more than once (`nanbeige.cpp:13-31`);
//! `mimo2` and `step35` read per-layer heads AND something else (both closed since, on
//! the seams their entries name)
//! (a V head width differing from K's; per-layer clamp arrays and a
//! half-width rotary -- their window arrays and NextN blocks are
//! `crate::swa_layers` and `crate::mtp_blocks` now); `laguna` closed the day after, when its other thing (the
//! gated attention, `crate::attn_gate`) landed; the hybrid recurrent rows (`jamba`, `lfm2`, `nemotron-h`,
//! `plamo2`, `granite-hybrid`, `kimi-linear`) use `n_head_kv(i) == 0`
//! to mean "this layer is recurrent", a different graph entirely.

use crate::config::ModelConfig;
use crate::decoder::AttnWeights;
use crate::loader::{load_weight_matrix, LoadError};
use crate::norm::NormOp;
use crate::norm_sites::NormSites;
use frink_core::cache::{KvCache, PagedKvStore, SharedPagedKv};
use frink_core::{Tensor, WeightMatrix};
use frink_gguf::{GgufValue, TensorSource};
use frink_moe::ExpertWeights;

/// Architectures whose llama.cpp tensor loader AND graph both index the
/// per-layer arrays, with the lines. Anything else gets layer 0 from
/// `LLAMA_LOAD_LOCALS` upstream, so a file whose layers disagree cannot
/// load there either, and frink refuses it by name rather than picking
/// a layer to believe.
///
/// Rows marked `generic` run on frink's generic GQA path and are what
/// this seam serves; the rest are listed so the reach of the seam is
/// recorded where the next person will look, and each names what else
/// it needs.
pub const PER_LAYER_SHAPE_ARCHS: &[(&str, &str)] = &[
    (
        "deci",
        "generic. deci.cpp:30-34 (loader) and :103-105 (graph): all three per layer, with \
         n_head == 0 an attention-free layer, n_head_kv == 0 a wo-only layer and n_ff == 0 \
         an FFN-free layer",
    ),
    (
        "openelm",
        "generic. openelm.cpp:26-28 (loader) and :67-69 (graph): all three per layer, sizing \
         one fused wqkv per layer",
    ),
    (
        "plamo3",
        "generic. plamo3.cpp:39-44 (loader) and :110-111 (graph); no published PLaMo-3 \
         export writes an array (conversion/plamo.py:27-30 writes scalars), so the seam is \
         latent there",
    ),
    (
        "laguna",
        "generic. laguna.cpp:87-88 (loader) and :176-177 (graph) read n_head(i) per layer; \
         KV heads uniform (:86). Closed with the gated attention (`crate::attn_gate`); the \
         second rotary width at :50 is `ModelConfig::rope_dim_swa` (`crate::swa_geometry`)",
    ),
    (
        "mimo2",
        "mimo2.cpp:47-49,111-112 read heads per layer (`swa_num_key_value_heads` on the \
         sliding layers, the converter's array). Closed with the split K/V head width \
         (`crate::kv_head_dims`, the V width :47-48 sizes apart from K's) and the value \
         scale (`crate::attn_value_scale`, :16,181); the sinks at :58 are \
         `AttnWeights::sinks`, the is_swa array at :12 is `crate::swa_layers` and the NEXTN \
         blocks at :19 are `crate::mtp_blocks`",
    ),
    (
        "step35",
        "generic. step35.cpp:76-78,208-209 (loader and graph) read heads and KV widths per \
         layer. Closed with the per-layer activation seam (`crate::act_layers`, the clamp \
         arrays at :28-29) and the two-valued rotary width (`crate::swa_geometry`, :9); the \
         gate at :96 is `crate::attn_gate`, the is_swa array at :26 `crate::swa_layers`, \
         the NEXTN blocks at :32 `crate::mtp_blocks`",
    ),
    (
        "spark2_5",
        "generic. spark2-5.cpp:33-37 (loader) and :76-77 (graph) read n_head(i) and \
         n_head_kv(i) per layer, sizing the per-head attention gate (:41) by each layer's \
         own count. Landed upstream after the 2026-08-04 pin and closed on 2026-09-19 with \
         one `crate::attn_gate` row",
    ),
    (
        "maple",
        "generic. maple.cpp:6 reads `expert_feed_forward_length` as an ARRAY at \
         n_layer_all length; the tensors are sized from n_ff_exp() (layer 0) at :27, so \
         the array must LOAD even where every entry agrees. Landed upstream after the \
         2026-08-04 pin and closed on 2026-09-19 with one `crate::rope_layers` row",
    ),
    (
        "nanbeige",
        "nanbeige.cpp:24-26 copies each physical layer's arrays to every logical slot; \
         `LayerShapes::replicated` does the same and `crate::layer_loops` is the seam the row \
         closed on",
    ),
    (
        "gemma4",
        "dedicated engine. gemma4.cpp:64-67,91 (loader) and :179-184 (graph)",
    ),
    (
        "gemma4-assistant",
        "dedicated engine. gemma4-assistant.cpp:53-55 (loader) and :134-138 (graph)",
    ),
    (
        "jamba",
        "generic. jamba.cpp:8-10 (hparams), :37-58 (loader) and :90-92 (graph): n_head_kv(i) \
         == 0 marks a Mamba-1 layer (`crate::mamba1`), served since 2026-09-14",
    ),
    (
        "lfm2",
        "hybrid: n_head_kv(il) == 0 marks a recurrent layer (lfm2.cpp:10,72,130-132)",
    ),
    (
        "lfm2moe",
        "hybrid: n_head_kv(il) == 0 marks a recurrent layer (lfm2moe.cpp:13,63)",
    ),
    (
        "nemotron_h",
        "generic. nemotron-h.cpp:9-11 (hparams), :53-98 (loader) and :146-153 (graph): \
         n_head_kv(i) == 0 && n_ff(i) == 0 marks a Mamba-2 layer, n_ff(i) == 0 alone an \
         attention layer, the rest an FFN-only layer; one block per layer \
         (`BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`), served since 2026-09-14",
    ),
    (
        "nemotron_h_moe",
        "nemotron-h.cpp:9-11, the same rule; its latent ungated ReLU-squared MoE (:79-90) is \
         not served yet",
    ),
    (
        "plamo2",
        "hybrid: n_head_kv(i) == 0 marks a recurrent layer (plamo2.cpp:19,82-84,218-219)",
    ),
    (
        "granitehybrid",
        "generic. granite-hybrid.cpp:17-19 (hparams), :58-77 (loader) and :137-140 (graph): \
         n_head_kv(i) == 0 marks a Mamba-2 layer (`crate::mamba2`), served since 2026-09-14",
    ),
    (
        "granite-hybrid",
        "generic. the frink alias of `granitehybrid` (granite-hybrid.cpp:17-19), the same rule",
    ),
    (
        "kimi-linear",
        "hybrid: n_head_kv(i) == 0 marks a KDA layer (kimi-linear.cpp:18)",
    ),
];

/// True when llama.cpp itself honours a per-layer shape for `arch`.
pub fn per_layer_shapes_read_by_llama_cpp(arch: &str) -> bool {
    PER_LAYER_SHAPE_ARCHS.iter().any(|(a, _)| *a == arch)
}

/// What one layer's attention block is.
///
/// An enum rather than two counts, because two of deci's three layer
/// kinds are spelled with a zero count upstream and a zero count is
/// exactly what a `for h in 0..n_heads` loop silently accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnShape {
    /// Grouped-query attention with this many query and KV heads.
    Gqa { n_heads: usize, n_kv_heads: usize },
    /// deci's "linear attention" (`deci.cpp:36-40`, `:115-118`):
    /// `n_head > 0 && n_head_kv == 0`. The block is `attn_norm` then
    /// `wo` alone -- `{n_embd, n_embd}`, no Q/K/V, no RoPE, nothing
    /// cached -- and its output joins the residual like any attention.
    Linear,
    /// deci's attention-free layer (`deci.cpp:107-109`, `:150-153`):
    /// `n_head == 0`. No norm, no projection; the residual passes
    /// straight into the FFN's input.
    Absent,
    /// LFM2's short convolution (`lfm2.cpp:9-11`, `:139-189`): the same
    /// `n_head > 0 && n_head_kv == 0` counts as [`AttnShape::Linear`],
    /// meaning a different block, decided by architecture
    /// ([`ZeroKvLayer`]). `attn_norm`, then `crate::shortconv`, then
    /// the residual add; the layer's cache holds its conv inputs
    /// ([`Self::cache_geometry`]).
    ShortConv,
    /// A Mamba-2 block (`build_mamba2_layer`, `mamba-base.cpp:149-288`;
    /// `granite-hybrid.cpp:163`): the same two counts again, decided
    /// by architecture ([`ZeroKvLayer`]). `attn_norm`, then
    /// `crate::mamba2`, then the residual add; the layer's cache holds
    /// no rows and carries a `RecurrentState` instead.
    Mamba2,
    /// A Mamba-1 block (`build_mamba_layer`, `mamba-base.cpp:4-148`;
    /// `jamba.cpp:128`, `mamba.cpp:106`): as [`AttnShape::Mamba2`] with
    /// `crate::mamba1`'s block.
    Mamba1,
    /// PLaMo-2's block (`plamo2.cpp:218-343`; `crate::plamo2_ssm`): as
    /// [`AttnShape::Mamba2`] with that block.
    Plamo2Ssm,
    /// The gated delta net (`qwen35.cpp:236-317`; `crate::gdn`): as
    /// [`AttnShape::Mamba2`] with that block. Decided by
    /// `crate::gdn::recurrent_layers`, not by the head counts, which
    /// are uniform on such a file.
    Gdn,
    /// MiniMax-01's lightning attention (`minimax-01.cpp:293-420`;
    /// `crate::lightning`): as [`AttnShape::Gdn`] with that block,
    /// decided by the same mask read from the same two keys.
    Lightning,
}

/// Architectures whose graph ADDS a block's output to the residual on a
/// layer with `feed_forward_length 0`, so "attention with no FFN" and
/// "Mamba-2 with no FFN" are layers rather than a defect.
///
/// `nemotron-h.cpp:157` adds `cur` for every kind of layer. deci is the
/// other reading: `deci.cpp:147-149` `continue`s BEFORE the add, so its
/// attention output is DISCARDED, and `LayerShapes::resolve` refuses
/// that combination for every architecture not listed here rather than
/// pin a dropped branch as the reference.
pub const BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT: &[&str] =
    &["nemotron_h", "nemotron_h_moe", "mamba", "mamba2"];

/// Architectures with NO attention anywhere: every layer is the named
/// block and nothing else (`mamba.cpp:73-88`, `mamba2.cpp`; the
/// converter writes `head_count 0` and `feed_forward_length 0`,
/// `conversion/mamba.py:155-156`). `LayerShapes::resolve` builds every
/// layer from this table, because the counts alone -- `(0, 0)` -- are
/// deci's attention-free layer on every other architecture.
pub const PURE_RECURRENT: &[(&str, ZeroKvLayer)] = &[
    ("mamba", ZeroKvLayer::Mamba1),
    ("mamba2", ZeroKvLayer::Mamba2),
];

/// The block every layer of a pure recurrent model is, or `None`.
pub fn pure_recurrent_block(arch: &str) -> Option<ZeroKvLayer> {
    PURE_RECURRENT
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, k)| *k)
}

/// What `head_count_kv == 0` with `head_count > 0` MEANS for an
/// architecture, because two graphs spell two different blocks with the
/// same two counts and the counts alone cannot tell them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroKvLayer {
    /// `deci.cpp:115-118`: `attn_norm` then `wo`.
    Linear,
    /// `lfm2.cpp:197`: the short convolution (`crate::shortconv`).
    ShortConv,
    /// `granite-hybrid.cpp:163`: the Mamba-2 block (`crate::mamba2`).
    Mamba2,
    /// `jamba.cpp:128`: the Mamba-1 block (`crate::mamba1`).
    Mamba1,
    /// `plamo2.cpp:218`: PLaMo-2's own block (`crate::plamo2_ssm`).
    Plamo2,
    /// `nemotron-h.cpp:9-11`: the Mamba-2 block when the layer's FFN
    /// width is ALSO zero, and an FFN-only layer (no attention block at
    /// all, [`AttnShape::Absent`]) when it is not -- every Nemotron-H
    /// layer is one block, and `feed_forward_length` is the second
    /// array that says which.
    Mamba2UnlessFfn,
    /// A recurrent block frink has no body for; the reason names it.
    Unserved(&'static str),
}

impl ZeroKvLayer {
    /// The table. Every architecture in [`PER_LAYER_SHAPE_ARCHS`] whose
    /// graph reads `n_head_kv(il) == 0` as a layer kind has a row here;
    /// anything else that reaches a zero is `Linear`, the reading the
    /// generic path had before the table existed, which only deci's
    /// converter writes.
    pub fn for_arch(arch: &str) -> Self {
        if crate::shortconv::is_shortconv_architecture(arch) {
            return ZeroKvLayer::ShortConv;
        }
        match arch {
            // `granite-hybrid.cpp:17-19,163`: Mamba-2 where the KV count
            // is zero, attention elsewhere, an FFN on every layer.
            "granitehybrid" | "granite-hybrid" => ZeroKvLayer::Mamba2,
            // `jamba.cpp:8-10,128`: `build_mamba_layer`, Mamba-1.
            "jamba" => ZeroKvLayer::Mamba1,
            // `falcon-h1.cpp:161` runs the Mamba-2 block IN PARALLEL with
            // attention on every layer (`crate::mamba2::
            // PARALLEL_WITH_ATTENTION`), so its KV count is never zero;
            // a zero here is not that graph.
            "falcon-h1" => ZeroKvLayer::Unserved(
                "no falcon-h1 layer has a zero KV count: falcon-h1.cpp:137-161 runs attention \
                 AND the Mamba-2 block on every layer (`ModelConfig::parallel_ssm`)",
            ),
            // `nemotron-h.cpp:9-11,143-152`: a layer is ONE of Mamba-2,
            // attention, or FFN, with one residual add. On the generic
            // layer that is a block with `ffn_dim 0`
            // ([`BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT`]) or an FFN with no
            // block.
            "nemotron_h" | "nemotron_h_moe" => ZeroKvLayer::Mamba2UnlessFfn,
            // `plamo2.cpp:19,142-146`: PLaMo-2's block where the KV
            // count is zero, attention elsewhere; served since 2026-09-18
            // by its own spelling (`crate::plamo2_ssm`).
            "plamo2" => ZeroKvLayer::Plamo2,
            "kimi-linear" => ZeroKvLayer::Unserved(
                "a KDA block (kimi-linear.cpp:18), served by `crate::kimi_decoder` and not \
                 the generic path",
            ),
            _ => ZeroKvLayer::Linear,
        }
    }
}

impl AttnShape {
    /// llama.cpp's three-way branch on the two counts (`deci.cpp:107-137`).
    ///
    /// `n_head == 0` with `n_head_kv > 0` is refused: the graph would
    /// take the attention-free branch while the loader (`:36-45`)
    /// would create a zero-wide Q, which is not a shape any converter
    /// writes.
    ///
    /// `ffn_dim` is this layer's FFN width, which one rule
    /// ([`ZeroKvLayer::Mamba2UnlessFfn`]) reads.
    pub fn from_counts(
        n_heads: usize,
        n_kv_heads: usize,
        ffn_dim: usize,
        zero_kv: ZeroKvLayer,
    ) -> Result<Self, String> {
        match (n_heads, n_kv_heads) {
            // `plamo2.cpp:19` reads ONLY `n_head_kv(il)`, and
            // `conversion/plamo.py:87-88` writes BOTH arrays as 0 on an
            // SSM layer, so on that architecture the pair means the
            // block and not deci's attention-free layer.
            (0, 0) if zero_kv == ZeroKvLayer::Plamo2 => Ok(AttnShape::Plamo2Ssm),
            (0, 0) => Ok(AttnShape::Absent),
            (0, kv) => Err(format!(
                "head_count 0 with head_count_kv {kv}: deci.cpp:107 would skip attention while \
                 :44 sizes a zero-wide Q projection"
            )),
            (_, 0) => match zero_kv {
                ZeroKvLayer::Linear => Ok(AttnShape::Linear),
                ZeroKvLayer::ShortConv => Ok(AttnShape::ShortConv),
                ZeroKvLayer::Mamba2 => Ok(AttnShape::Mamba2),
                ZeroKvLayer::Mamba1 => Ok(AttnShape::Mamba1),
                ZeroKvLayer::Plamo2 => Ok(AttnShape::Plamo2Ssm),
                // nemotron-h.cpp:9-11: `n_head_kv == 0 && n_ff == 0`.
                ZeroKvLayer::Mamba2UnlessFfn if ffn_dim == 0 => Ok(AttnShape::Mamba2),
                // :152-153: the FFN alone, under `attn_norm` (:145).
                ZeroKvLayer::Mamba2UnlessFfn => Ok(AttnShape::Absent),
                ZeroKvLayer::Unserved(what) => Err(format!(
                    "head_count_kv 0 marks {what}; `layer_shapes::ZeroKvLayer` is the table"
                )),
            },
            (q, kv) if q % kv != 0 => Err(format!(
                "head_count {q} is not a multiple of head_count_kv {kv}"
            )),
            (n_heads, n_kv_heads) => Ok(AttnShape::Gqa {
                n_heads,
                n_kv_heads,
            }),
        }
    }

    /// KV heads this layer caches: zero for the two shapes that write
    /// no history.
    pub fn n_kv_heads(self) -> usize {
        match self {
            AttnShape::Gqa { n_kv_heads, .. } => n_kv_heads,
            AttnShape::Linear
            | AttnShape::Absent
            | AttnShape::ShortConv
            | AttnShape::Mamba2
            | AttnShape::Mamba1
            | AttnShape::Plamo2Ssm
            | AttnShape::Gdn
            | AttnShape::Lightning => 0,
        }
    }

    /// Query heads, zero where there are none.
    pub fn n_heads(self) -> usize {
        match self {
            AttnShape::Gqa { n_heads, .. } => n_heads,
            AttnShape::Linear
            | AttnShape::Absent
            | AttnShape::ShortConv
            | AttnShape::Mamba2
            | AttnShape::Mamba1
            | AttnShape::Plamo2Ssm
            | AttnShape::Gdn
            | AttnShape::Lightning => 0,
        }
    }

    /// True for a block whose state between tokens is a
    /// `RecurrentState` rather than rows (`crate::mamba2`).
    pub fn is_recurrent(self) -> bool {
        matches!(
            self,
            AttnShape::Mamba2
                | AttnShape::Mamba1
                | AttnShape::Plamo2Ssm
                | AttnShape::Gdn
                | AttnShape::Lightning
        )
    }

    /// The layer's cache as `(n_kv_heads, k_head_dim, v_head_dim)`, the
    /// three numbers every `KvCache` / `PagedKvStore` constructor takes.
    ///
    /// A GQA layer's is its head geometry; the two attention-less
    /// shapes and the Mamba-2 block write no history and get an empty
    /// cache (the Mamba-2 block keeps its state beside it and pushes
    /// EMPTY rows so the cache still counts positions); a short-conv
    /// layer keeps its conv inputs as ONE row of `hidden_dim` per token
    /// with no V (`crate::shortconv`). ONE function, because
    /// `ModelConfig::new_kv_caches` and its three siblings each build
    /// the caches and a fourth reading of the shape would be a fourth
    /// place to disagree.
    pub fn cache_geometry(
        self,
        head_dim: usize,
        v_head_dim: usize,
        hidden_dim: usize,
    ) -> (usize, usize, usize) {
        match self {
            AttnShape::Gqa { n_kv_heads, .. } => (n_kv_heads, head_dim, v_head_dim),
            AttnShape::Linear
            | AttnShape::Absent
            | AttnShape::Mamba2
            | AttnShape::Mamba1
            | AttnShape::Plamo2Ssm
            | AttnShape::Gdn
            | AttnShape::Lightning => (0, head_dim, v_head_dim),
            AttnShape::ShortConv => (1, hidden_dim, 0),
        }
    }
}

/// One layer's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShape {
    pub attention: AttnShape,
    /// The dense FFN width, `n_ff(il)`. Zero is deci's FFN-free layer
    /// (`deci.cpp:147-149`): no `ffn_norm`, no gate/up/down.
    pub ffn_dim: usize,
}

/// Every layer's shape, or the statement that they all agree.
///
/// `Uniform` is not `PerLayer(vec![same; n])`: the fused Metal stacks,
/// the CUDA resident KV and the slot-file format each hold ONE
/// geometry, and "is this model uniform" is a question they ask
/// through [`Self::is_uniform`] rather than by comparing entries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum LayerShapes {
    /// Every layer is `ModelConfig::{n_heads, n_kv_heads,
    /// moe.expert_ffn_dim}`.
    #[default]
    Uniform,
    /// One entry per layer, at least two of which differ.
    PerLayer(Vec<LayerShape>),
}

impl LayerShapes {
    pub fn is_uniform(&self) -> bool {
        matches!(self, LayerShapes::Uniform)
    }

    /// The same shapes for `n_loops` passes over the layers, as
    /// `nanbeige.cpp:24-26` copies each physical layer's arrays to every
    /// logical slot (`crate::layer_loops`). Uniform stays uniform.
    pub fn replicated(self, n_loops: usize) -> Self {
        match self {
            LayerShapes::PerLayer(v) if n_loops > 1 => {
                LayerShapes::PerLayer(v.iter().copied().cycle().take(v.len() * n_loops).collect())
            }
            other => other,
        }
    }

    /// Builds the table from the three per-layer arrays, collapsing to
    /// `Uniform` when nothing varies and refusing a varying file for an
    /// architecture llama.cpp itself reads at layer 0.
    ///
    /// `ffn` may be absent (a MoE file that declares only
    /// `expert_feed_forward_length`); then every layer takes
    /// `expert_ffn_dim`.
    ///
    /// `recurrent` is `crate::gdn::recurrent_layers`' answer: the layers
    /// that run a recurrent block AND which block, on an architecture
    /// whose head counts are uniform and say nothing about either.
    pub fn resolve(
        arch: &str,
        heads: &[u64],
        kv_heads: &[u64],
        ffn: Option<&[u64]>,
        expert_ffn_dim: usize,
        recurrent: Option<&crate::gdn::RecurrentMask>,
    ) -> Result<Self, LoadError> {
        let n = heads.len();
        assert_eq!(kv_heads.len(), n);
        if let Some(recurrent) = recurrent {
            assert_eq!(recurrent.layers.len(), n);
            let zero_kv = ZeroKvLayer::for_arch(arch);
            let mut shapes = Vec::with_capacity(n);
            for il in 0..n {
                let ffn_dim = ffn.map_or(expert_ffn_dim, |f| f[il] as usize);
                let attention = if recurrent.layers[il] {
                    recurrent.block
                } else {
                    AttnShape::from_counts(
                        heads[il] as usize,
                        kv_heads[il] as usize,
                        ffn_dim,
                        zero_kv,
                    )
                    .map_err(|why| {
                        LoadError::UnsupportedFeature(arch.to_string(), format!("blk.{il}: {why}"))
                    })?
                };
                shapes.push(LayerShape { attention, ffn_dim });
            }
            return Ok(LayerShapes::PerLayer(shapes));
        }
        // A pure recurrent model: every layer the one block, no FFN
        // (`PURE_RECURRENT`). Its arrays are uniform zeros, which would
        // otherwise read as a zero-head GQA model.
        if let Some(kind) = pure_recurrent_block(arch) {
            let shape = AttnShape::from_counts(1, 0, 0, kind)
                .map_err(|why| LoadError::UnsupportedFeature(arch.to_string(), why))?;
            for il in 0..n {
                let ffn_dim = ffn.map_or(0, |f| f[il] as usize);
                if heads[il] != 0 || kv_heads[il] != 0 || ffn_dim != 0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.to_string(),
                        format!(
                            "blk.{il}: head_count {} / head_count_kv {} / feed_forward_length \
                             {ffn_dim} on a pure recurrent architecture, whose converter writes \
                             0 for all three (conversion/mamba.py:155-156) and whose graph has no \
                             attention and no FFN (mamba.cpp:73-88)",
                            heads[il], kv_heads[il]
                        ),
                    ));
                }
            }
            return Ok(LayerShapes::PerLayer(vec![
                LayerShape {
                    attention: shape,
                    ffn_dim: 0
                };
                n
            ]));
        }
        let uniform = heads.windows(2).all(|w| w[0] == w[1])
            && kv_heads.windows(2).all(|w| w[0] == w[1])
            && ffn.is_none_or(|f| f.windows(2).all(|w| w[0] == w[1]));
        if uniform {
            return Ok(LayerShapes::Uniform);
        }
        if !per_layer_shapes_read_by_llama_cpp(arch) {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "per-layer head_count / head_count_kv / feed_forward_length arrays whose \
                     entries differ (heads {heads:?}, kv {kv_heads:?}, ff {ffn:?}). llama.cpp \
                     reads these arrays for every architecture (llama-model.cpp:1149-1158) but \
                     this one's graph takes layer 0 through LLAMA_LOAD_LOCALS \
                     (llama-model.h:760-767), so such a file cannot load there either; \
                     `layer_shapes::PER_LAYER_SHAPE_ARCHS` lists the ones that index per layer"
                ),
            ));
        }
        let mut shapes = Vec::with_capacity(n);
        let zero_kv = ZeroKvLayer::for_arch(arch);
        let keeps_output = BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT.contains(&arch);
        for il in 0..n {
            let ffn_dim = ffn.map_or(expert_ffn_dim, |f| f[il] as usize);
            let attention =
                AttnShape::from_counts(heads[il] as usize, kv_heads[il] as usize, ffn_dim, zero_kv)
                    .map_err(|why| {
                        LoadError::UnsupportedFeature(arch.to_string(), format!("blk.{il}: {why}"))
                    })?;
            if ffn_dim == 0 && attention != AttnShape::Absent && !keeps_output {
                // deci.cpp:147-149 `continue`s BEFORE the residual add
                // at :150-153, so the attention output computed at
                // :115-137 is discarded and `inpL` is left untouched.
                // That is llama.cpp's graph and it is almost certainly
                // not the model's (HF's DeciLM adds the attention
                // residual before asking whether the FFN is a no-op).
                // frink will not pin a dropped branch as the golden
                // answer, so the combination is refused by name; the
                // attention-free FFN-free layer, where both agree the
                // layer is the identity, is admitted.
                return Err(LoadError::UnsupportedFeature(
                    arch.to_string(),
                    format!(
                        "blk.{il}: feed_forward_length 0 on a layer WITH attention \
                         (head_count {}). deci.cpp:147-149 `continue`s before the residual \
                         add at :150-153, discarding the attention output that :115-137 \
                         computed, and frink will not reproduce a dropped branch as the \
                         reference. An FFN-free layer with head_count 0 is supported",
                        heads[il]
                    ),
                ));
            }
            shapes.push(LayerShape { attention, ffn_dim });
        }
        Ok(LayerShapes::PerLayer(shapes))
    }
}

/// llama.cpp's `get_key_or_arr` (`llama-model-loader.cpp:446-470`): a
/// scalar is broadcast to every layer, an array must be exactly
/// `n_layers` long, and an absent key is `None`.
///
/// `GgufValue::as_u64` returns `None` for an array, which is how
/// `openelm` used to die on a missing-hparam error for a key its file
/// carries; this is the read that sees both spellings.
/// [`read_u64_per_layer`] for a file that may carry NextN/MTP blocks:
/// the array is length-checked against `block_count`, which is what
/// llama.cpp passes (`llama-model.cpp:1148-1156` read the three shape
/// arrays with `hparams.n_layer()` BEFORE `load_arch_hparams` at `:1233`
/// has read `nextn_predict_layers`, so `n_layer()` is still
/// `n_layer_all`, and `conversion/mimo.py:146-150` writes the arrays at
/// that length with the MTP entries appended), and only the trunk's
/// entries are returned.
///
/// The loader reads every per-layer shape through this and never
/// through the raw function, so a call site cannot hand the trunk count
/// to the length check by mistake.
pub fn read_u64_trunk_layers(
    file: &impl TensorSource,
    key: &str,
    trunk: &crate::mtp_blocks::TrunkLayers,
) -> Result<Option<Vec<u64>>, LoadError> {
    Ok(
        read_u64_per_layer(file, key, trunk.block_count)?.map(|mut v| {
            v.truncate(trunk.n_layers);
            v
        }),
    )
}

pub fn read_u64_per_layer(
    file: &impl TensorSource,
    key: &str,
    n_layers: usize,
) -> Result<Option<Vec<u64>>, LoadError> {
    let Some(value) = file.metadata(key) else {
        return Ok(None);
    };
    match value {
        GgufValue::Array(items) => {
            if items.len() != n_layers {
                return Err(LoadError::UnsupportedFeature(
                    key.to_string(),
                    format!(
                        "array of {} entries for {n_layers} layers; llama.cpp refuses this too \
                         (`key has wrong array length`, llama-model-loader.cpp:464-465)",
                        items.len()
                    ),
                ));
            }
            let mut out = Vec::with_capacity(n_layers);
            for (il, item) in items.iter().enumerate() {
                out.push(item.as_u64().ok_or_else(|| {
                    LoadError::UnsupportedFeature(
                        key.to_string(),
                        format!("entry {il} is not an unsigned integer: {item:?}"),
                    )
                })?);
            }
            Ok(Some(out))
        }
        scalar => scalar
            .as_u64()
            .map(|v| Some(vec![v; n_layers]))
            .ok_or_else(|| LoadError::MissingHparam(key.to_string())),
    }
}

/// A projection with no rows: the placeholder held by a layer that has
/// no such projection, so that `AttnWeights` / `ExpertWeights` -- which
/// have thirty construction sites and no `Option` in them -- can carry
/// deci's two attention-less shapes without a new struct.
///
/// Never applied: every host body branches on [`AttnShape`] /
/// `LayerShape::ffn_dim` before it reaches a projection. If one did
/// not, `apply` of a zero-row matrix is an empty vector, and every
/// kernel downstream of an empty Q panics on its own arithmetic rather
/// than answering.
fn no_rows(cols: usize) -> WeightMatrix {
    WeightMatrix::F32(Tensor::new(Vec::new(), vec![0, cols]))
}

/// The attention weights of a [`AttnShape::Linear`], [`AttnShape::
/// Absent`] or [`AttnShape::ShortConv`] layer.
///
/// Linear (`deci.cpp:36-40`): `attn_norm` and a `{n_embd, n_embd}`
/// `attn_output`, nothing else. Absent (`:107-109`): nothing at all --
/// no norm tensor exists for the layer, and `NormOp::None` is what the
/// body applies before a block it then skips. ShortConv (`lfm2.cpp:
/// 70,80-82`): `attn_norm` and the three `shortconv.*` tensors
/// (`crate::shortconv`).
pub(crate) fn load_non_gqa_attention(
    shape: AttnShape,
    file: &impl TensorSource,
    arch: &str,
    layer: usize,
    norm_sites: &NormSites,
    config: &ModelConfig,
) -> Result<AttnWeights, LoadError> {
    let hidden_dim = config.hidden_dim;
    let mut shortconv = None;
    let mut ssm = None;
    let (norm_weight, o_proj) = match shape {
        AttnShape::Linear => (
            norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
            load_weight_matrix(file, &format!("blk.{layer}.attn_output.weight"))?,
        ),
        AttnShape::Absent => (NormOp::None, no_rows(0)),
        AttnShape::ShortConv => {
            shortconv = Some(crate::shortconv::ShortConv::load(
                file, arch, layer, hidden_dim,
            )?);
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Mamba2 => {
            ssm = Some(crate::ssm_block::SsmBlock::Mamba2(
                crate::mamba2::Mamba2::load(file, arch, layer, hidden_dim)?,
            ));
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Mamba1 => {
            ssm = Some(crate::ssm_block::SsmBlock::Mamba1(
                crate::mamba1::Mamba1::load(file, arch, layer, hidden_dim)?,
            ));
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Plamo2Ssm => {
            ssm = Some(crate::ssm_block::SsmBlock::Plamo2(
                crate::plamo2_ssm::Plamo2Ssm::load(file, arch, layer, hidden_dim)?,
            ));
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Gdn => {
            ssm = Some(crate::ssm_block::SsmBlock::Gdn(crate::gdn::Gdn::load(
                file, arch, layer, hidden_dim,
            )?));
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Lightning => {
            ssm = Some(crate::ssm_block::SsmBlock::Lightning(
                crate::lightning::Lightning::load(
                    file,
                    layer,
                    config.n_layers,
                    config.n_heads,
                    config.head_dim,
                    hidden_dim,
                )?,
            ));
            (
                norm_sites.load_pre_norm(norm_sites.attn, file, Some(layer))?,
                no_rows(0),
            )
        }
        AttnShape::Gqa { .. } => unreachable!("a GQA layer loads its projections"),
    };
    if let AttnShape::Linear = shape {
        if o_proj.rows() != hidden_dim || o_proj.cols() != hidden_dim {
            return Err(LoadError::UnsupportedFeature(
                format!("blk.{layer}.attn_output.weight"),
                format!(
                    "a wo-only layer's projection is {{n_embd, n_embd}} (deci.cpp:39); this one \
                     is {}x{} for hidden_dim {hidden_dim}",
                    o_proj.rows(),
                    o_proj.cols()
                ),
            ));
        }
    }
    Ok(AttnWeights {
        q_proj: no_rows(hidden_dim),
        k_proj: no_rows(hidden_dim),
        v_proj: no_rows(hidden_dim),
        o_proj,
        norm_weight,
        q_norm: None,
        k_norm: None,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        // `plamo2.cpp:150` norms the BLOCK's output with `attn_post_norm`
        // on every layer, SSM or attention; the table's row is read for
        // that shape (and `Decoder::recurrent_block` applies it). No
        // other recurrent graph creates one on a recurrent layer, and
        // the two attention-less shapes have no output to norm.
        post_attn_norm: match shape {
            AttnShape::Plamo2Ssm => NormSites::load_post_norm(norm_sites.post_attn, file, layer)?,
            _ => None,
        },
        // The FFN's post-norm lives on this struct; a layer with no
        // attention may still have one, so the table decides.
        post_ffn_norm: NormSites::load_post_norm(norm_sites.post_ffn, file, layer)?,
        output_gate: None,
        sinks: None,
        // No attention, no attention output to norm; the table row
        // that has these (`bitnet`) has a uniform GQA shape.
        attn_sub_norm: None,
        o_scale: None,
        o_bias: None,
        shortconv,
        ssm,
        q_gate_interleaved: false,
    })
}

/// The expert of an FFN-free layer (`deci.cpp:63-67` creates no
/// gate/up/down when `n_ff == 0`): three placeholders, never applied.
///
/// `down` has ZERO rows rather than `hidden_dim` rows of nothing, on
/// purpose: a `{hidden_dim, 0}` placeholder would project an empty
/// activation to a vector of zeros, and a host body that forgot to
/// skip the FFN would add zeros to the residual and be right by
/// accident. With no rows the forgotten branch is an empty vector, and
/// `residual_add`'s length check turns the omission into a panic.
pub(crate) fn absent_ffn(hidden_dim: usize) -> ExpertWeights {
    ExpertWeights {
        gate: no_rows(hidden_dim),
        up: no_rows(hidden_dim),
        down: no_rows(0),
    }
}

/// llama.cpp's `check_tensor_dims` for the three projections, against
/// THIS layer's shape: `create_tensor_qkv` sizes Q `{n_embd,
/// n_embd_head_k * n_head}` and K/V `{n_embd, n_embd_head_k *
/// n_head_kv}` (`llama-model.cpp:2886-2900`), and `wo` is
/// `{n_embd_head_k * n_head, n_embd}`. A file whose tensors disagree
/// with its own header is refused there, and was silently accepted here.
pub(crate) fn check_gqa_projection_widths(
    layer: usize,
    shape: AttnShape,
    head_dim: usize,
    v_head_dim: usize,
    hidden_dim: usize,
    attn: &AttnWeights,
) -> Result<(), LoadError> {
    let AttnShape::Gqa {
        n_heads,
        n_kv_heads,
    } = shape
    else {
        unreachable!("only GQA layers have Q/K/V to check")
    };
    // `qwen35.cpp:59`: the query and its gate share one projection
    // (`crate::attn_gate::Q_INTERLEAVED_GATE_ARCHS`).
    let q_rows = if attn.q_gate_interleaved { 2 } else { 1 } * n_heads * head_dim;
    let want = [
        ("attn_q", attn.q_proj.rows(), q_rows),
        ("attn_k", attn.k_proj.rows(), n_kv_heads * head_dim),
        // V and the output projection at the V width: `mimo2.cpp:52`
        // creates `wo` as `{n_embd_head_v * n_head, n_embd}`
        // (`crate::kv_head_dims`); one width everywhere else.
        ("attn_v", attn.v_proj.rows(), n_kv_heads * v_head_dim),
        ("attn_output (rows)", attn.o_proj.rows(), hidden_dim),
        (
            "attn_output (cols)",
            attn.o_proj.cols(),
            n_heads * v_head_dim,
        ),
    ];
    for (name, got, expected) in want {
        if got != expected {
            return Err(LoadError::UnsupportedFeature(
                format!("blk.{layer}.{name}.weight"),
                format!(
                    "{got} does not match this layer's head_count {n_heads} / head_count_kv \
                     {n_kv_heads} x head_dim {head_dim} / v_head_dim {v_head_dim} (expected \
                     {expected}); llama.cpp's check_tensor_dims refuses the same file"
                ),
            ));
        }
    }
    Ok(())
}

impl ModelConfig {
    /// True when any layer carries a `RecurrentState` between tokens
    /// (`AttnShape::is_recurrent`): the fact every caller that rolls a
    /// cache back to a middle position -- speculative verification, the
    /// draft model, the prefix cache -- is fenced on
    /// (`frink_core::recurrent_state`).
    pub fn has_recurrent_layers(&self) -> bool {
        self.parallel_ssm
            || (0..self.n_layers).any(|il| self.layer_shape(il).attention.is_recurrent())
    }

    /// Layer `il`'s cache geometry (`AttnShape::cache_geometry` at this
    /// model's widths).
    pub fn layer_cache_geometry(&self, il: usize) -> (usize, usize, usize) {
        self.layer_shape(il).attention.cache_geometry(
            self.head_dim,
            self.v_head_dim(),
            self.hidden_dim,
        )
    }

    /// Layer `il`'s shape. THE accessor: every layer body reads its head
    /// counts here and nowhere else.
    pub fn layer_shape(&self, il: usize) -> LayerShape {
        match &self.layer_shapes {
            LayerShapes::Uniform => LayerShape {
                attention: AttnShape::Gqa {
                    n_heads: self.n_heads,
                    n_kv_heads: self.n_kv_heads,
                },
                ffn_dim: self.moe.expert_ffn_dim,
            },
            LayerShapes::PerLayer(v) => v[il],
        }
    }

    /// One contiguous cache per layer, each sized for that layer.
    ///
    /// The twenty-odd call sites that used to spell
    /// `KvCache::new(config.n_kv_heads, config.head_dim)` per layer were
    /// twenty copies of one geometry decision, and every one of them was
    /// wrong for a model whose layers differ.
    pub fn new_kv_caches(&self) -> Vec<KvCache> {
        (0..self.n_layers)
            .map(|il| {
                let (n_kv_heads, head_dim, v_head_dim) = self.layer_cache_geometry(il);
                KvCache::new_split(n_kv_heads, head_dim, v_head_dim)
            })
            .collect()
    }

    /// The same, pre-allocated for `max_seq_len` positions.
    pub fn new_kv_caches_with_capacity(&self, max_seq_len: usize) -> Vec<KvCache> {
        (0..self.n_layers)
            .map(|il| {
                let (n_kv_heads, head_dim, v_head_dim) = self.layer_cache_geometry(il);
                KvCache::with_capacity_split(n_kv_heads, head_dim, v_head_dim, max_seq_len)
            })
            .collect()
    }

    /// The same, each layer's storage leased from `pool`. The first
    /// layer that cannot be leased fails the whole set, as before.
    pub fn new_kv_caches_with_pool(
        &self,
        pool: &std::sync::Arc<std::sync::Mutex<frink_core::cache::KvBlockPool>>,
        max_seq_len: usize,
    ) -> Result<Vec<KvCache>, frink_core::cache::KvPoolExhausted> {
        (0..self.n_layers)
            .map(|il| {
                let (n_kv_heads, head_dim, v_head_dim) = self.layer_cache_geometry(il);
                KvCache::with_pool_split(
                    n_kv_heads,
                    head_dim,
                    v_head_dim,
                    std::sync::Arc::clone(pool),
                    max_seq_len,
                )
            })
            .collect()
    }

    /// One paged store per layer, each sized for that layer.
    pub fn new_paged_kv(&self, block_size: usize, blocks_per_layer: usize) -> SharedPagedKv {
        SharedPagedKv::from_stores(
            (0..self.n_layers)
                .map(|il| {
                    let (n_kv_heads, head_dim, v_head_dim) = self.layer_cache_geometry(il);
                    PagedKvStore::new_split(
                        block_size,
                        blocks_per_layer,
                        n_kv_heads,
                        head_dim,
                        v_head_dim,
                    )
                })
                .collect(),
        )
    }

    /// KV heads summed over every layer: what a per-token memory budget
    /// multiplies by `head_dim * elem_size`. `n_layers * n_kv_heads`
    /// for a uniform model, and an over-count for a heterogeneous one
    /// wherever it is still spelled that way.
    pub fn kv_heads_all_layers(&self) -> usize {
        (0..self.n_layers)
            .map(|il| self.layer_shape(il).attention.n_kv_heads())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deci_like() -> Vec<LayerShape> {
        vec![
            LayerShape {
                attention: AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2,
                },
                ffn_dim: 16,
            },
            LayerShape {
                attention: AttnShape::Linear,
                ffn_dim: 8,
            },
            LayerShape {
                attention: AttnShape::Absent,
                ffn_dim: 16,
            },
            LayerShape {
                attention: AttnShape::Absent,
                ffn_dim: 0,
            },
        ]
    }

    /// The three-way branch, pinned to deci.cpp's conditions.
    #[test]
    fn the_two_zero_counts_are_two_different_layer_kinds() {
        let deci = ZeroKvLayer::for_arch("deci");
        assert_eq!(
            AttnShape::from_counts(0, 0, 16, deci),
            Ok(AttnShape::Absent)
        );
        assert_eq!(
            AttnShape::from_counts(4, 0, 16, deci),
            Ok(AttnShape::Linear)
        );
        assert_eq!(
            AttnShape::from_counts(4, 2, 16, deci),
            Ok(AttnShape::Gqa {
                n_heads: 4,
                n_kv_heads: 2
            })
        );
        assert!(AttnShape::from_counts(0, 2, 16, deci).is_err());
        assert!(AttnShape::from_counts(3, 2, 16, deci).is_err());
        assert_eq!(AttnShape::Linear.n_kv_heads(), 0);
        assert_eq!(AttnShape::Absent.n_heads(), 0);
    }

    /// The same two counts are a different block on LFM2 (lfm2.cpp:197)
    /// and an unserved one on the Mamba hybrids: the architecture
    /// decides, and the counts alone cannot.
    #[test]
    fn a_zero_kv_layer_means_what_the_architecture_says() {
        let lfm2 = ZeroKvLayer::for_arch("lfm2");
        assert_eq!(
            AttnShape::from_counts(4, 0, 16, lfm2),
            Ok(AttnShape::ShortConv)
        );
        // GQA layers are GQA on every architecture.
        assert!(matches!(
            AttnShape::from_counts(4, 2, 16, lfm2),
            Ok(AttnShape::Gqa { .. })
        ));
        assert_eq!(
            AttnShape::from_counts(4, 0, 16, ZeroKvLayer::for_arch("jamba")),
            Ok(AttnShape::Mamba1)
        );
        // plamo2: the block under either head-count spelling -- the
        // converter's `(0, 0)` (`conversion/plamo.py:87-88`) and the
        // scalar-heads `(4, 0)` -- because `plamo2.cpp:19` reads only
        // the KV count; `(0, 0)` is deci's attention-free layer
        // everywhere else.
        assert_eq!(
            AttnShape::from_counts(4, 0, 16, ZeroKvLayer::for_arch("plamo2")),
            Ok(AttnShape::Plamo2Ssm)
        );
        assert_eq!(
            AttnShape::from_counts(0, 0, 16, ZeroKvLayer::for_arch("plamo2")),
            Ok(AttnShape::Plamo2Ssm)
        );
        assert_eq!(
            AttnShape::from_counts(0, 0, 16, ZeroKvLayer::for_arch("jamba")),
            Ok(AttnShape::Absent)
        );
        // A pure recurrent model: every layer the block, from uniform
        // zeros that would otherwise read as a zero-head GQA model.
        let s = LayerShapes::resolve("mamba", &[0, 0], &[0, 0], Some(&[0, 0]), 0, None).unwrap();
        let LayerShapes::PerLayer(v) = s else {
            panic!("per layer");
        };
        assert!(v
            .iter()
            .all(|l| l.attention == AttnShape::Mamba1 && l.ffn_dim == 0));
        assert!(
            LayerShapes::resolve("mamba2", &[0, 0], &[0, 0], None, 0, None)
                .is_ok_and(|s| matches!(s, LayerShapes::PerLayer(_)))
        );
        assert!(LayerShapes::resolve("mamba", &[4, 4], &[0, 0], None, 0, None).is_err());
        // The cache: one row of n_embd per token, no V.
        assert_eq!(AttnShape::ShortConv.cache_geometry(6, 6, 24), (1, 24, 0));
        // Mamba-2: no rows at all, the state rides beside the cache.
        assert_eq!(
            AttnShape::from_counts(4, 0, 16, ZeroKvLayer::for_arch("granitehybrid")),
            Ok(AttnShape::Mamba2)
        );
        // nemotron-h.cpp:9-11: the FFN width is the second array.
        let nh = ZeroKvLayer::for_arch("nemotron_h");
        assert_eq!(AttnShape::from_counts(4, 0, 0, nh), Ok(AttnShape::Mamba2));
        assert_eq!(AttnShape::from_counts(4, 0, 40, nh), Ok(AttnShape::Absent));
        // Attention with no FFN: refused as deci's discarded branch,
        // served as Nemotron-H's one-block layer.
        assert!(LayerShapes::resolve("deci", &[4, 4], &[2, 2], Some(&[16, 0]), 16, None).is_err());
        let s = LayerShapes::resolve(
            "nemotron_h",
            &[4, 4, 4],
            &[0, 2, 0],
            Some(&[0, 0, 40]),
            16,
            None,
        )
        .unwrap();
        let LayerShapes::PerLayer(v) = s else {
            panic!("per layer");
        };
        assert_eq!(
            v.iter().map(|l| l.attention).collect::<Vec<_>>(),
            [
                AttnShape::Mamba2,
                AttnShape::Gqa {
                    n_heads: 4,
                    n_kv_heads: 2
                },
                AttnShape::Absent
            ]
        );
        assert_eq!(v.iter().map(|l| l.ffn_dim).collect::<Vec<_>>(), [0, 0, 40]);
        assert_eq!(AttnShape::Mamba2.cache_geometry(6, 6, 24), (0, 6, 6));
        assert!(AttnShape::Mamba2.is_recurrent() && !AttnShape::ShortConv.is_recurrent());
        assert_eq!(AttnShape::Linear.cache_geometry(6, 6, 24), (0, 6, 6));
        assert_eq!(AttnShape::ShortConv.n_kv_heads(), 0);
        let s = LayerShapes::resolve("plamo2", &[4, 4], &[2, 0], None, 16, None).unwrap();
        let LayerShapes::PerLayer(v) = s else {
            panic!("per layer");
        };
        assert_eq!(v[1].attention, AttnShape::Plamo2Ssm);
        assert!(AttnShape::Plamo2Ssm.is_recurrent());
        assert_eq!(AttnShape::Plamo2Ssm.cache_geometry(8, 8, 32), (0, 8, 8));
        let s = LayerShapes::resolve("lfm2", &[4, 4], &[0, 2], None, 16, None).unwrap();
        let LayerShapes::PerLayer(v) = s else {
            panic!("per layer");
        };
        assert_eq!(v[0].attention, AttnShape::ShortConv);
    }

    /// Equal arrays are the uniform model, for ANY architecture: the
    /// converter is free to spell a scalar as an array.
    #[test]
    fn equal_arrays_collapse_to_uniform_even_for_a_layer_zero_architecture() {
        let s = LayerShapes::resolve("llama", &[4, 4], &[2, 2], Some(&[16, 16]), 16, None).unwrap();
        assert!(s.is_uniform());
    }

    /// A varying array on an architecture whose graph reads layer 0 is
    /// refused, naming the table; on one that indexes per layer it is
    /// the per-layer table.
    #[test]
    fn a_varying_array_is_refused_unless_llama_cpp_indexes_it_per_layer() {
        let err = LayerShapes::resolve("llama", &[4, 4], &[2, 1], None, 16, None).unwrap_err();
        assert!(format!("{err}").contains("PER_LAYER_SHAPE_ARCHS"), "{err}");
        let s = LayerShapes::resolve(
            "deci",
            &[4, 4, 0, 0],
            &[2, 0, 0, 0],
            Some(&[16, 8, 16, 0]),
            16,
            None,
        )
        .unwrap();
        assert_eq!(s, LayerShapes::PerLayer(deci_like()));
    }

    /// The dropped-branch combination is refused by name, and the
    /// combination both graphs agree on is not.
    #[test]
    fn an_ffn_free_layer_with_attention_is_refused_and_one_without_is_not() {
        let err =
            LayerShapes::resolve("deci", &[4, 4], &[2, 2], Some(&[16, 0]), 16, None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("deci.cpp:147-149"), "{msg}");
        assert!(msg.contains("blk.1"), "{msg}");
        assert!(LayerShapes::resolve("deci", &[4, 0], &[2, 0], Some(&[16, 0]), 16, None).is_ok());
    }

    /// The accessor and the two cache constructors read the same table.
    #[test]
    fn caches_are_sized_per_layer_and_the_scalar_is_never_consulted() {
        let mut cfg = crate::config::glm_5_2();
        cfg.n_layers = 4;
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 8;
        cfg.layer_shapes = LayerShapes::PerLayer(deci_like());
        let caches = cfg.new_kv_caches();
        assert_eq!(
            caches.iter().map(|c| c.n_kv_heads).collect::<Vec<_>>(),
            vec![2, 0, 0, 0]
        );
        assert_eq!(cfg.kv_heads_all_layers(), 2);
        assert_eq!(cfg.layer_shape(1).attention, AttnShape::Linear);
        assert_eq!(cfg.layer_shape(3).ffn_dim, 0);
        // A cache built from the scalar for the wo-only layer refuses
        // the first row: this is what turns a missed call site into a
        // panic instead of a misaligned history.
        let mut wrong = KvCache::new(cfg.n_kv_heads, cfg.head_dim);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            wrong.push(&[], &[]).unwrap();
        }));
        assert!(res.is_err(), "push must assert the row width");
        cfg.layer_shapes = LayerShapes::Uniform;
        assert!(cfg
            .new_kv_caches()
            .iter()
            .all(|c| c.n_kv_heads == 2 && c.head_dim == 8));
        assert_eq!(cfg.kv_heads_all_layers(), 8);
    }

    /// The projection-width check refuses a Q/K/V/wo whose rows disagree
    /// with the layer's own counts, naming the tensor, and passes one
    /// that agrees. Built by hand rather than from a fixture: a GGUF
    /// whose tensors disagree with its header is exactly what no
    /// converter writes, so this is the only way the refusal can be
    /// shown to fire.
    #[test]
    fn a_projection_sized_for_another_layer_s_counts_is_refused_naming_the_tensor() {
        let m = |rows: usize, cols: usize| {
            WeightMatrix::F32(Tensor::new(vec![0.0; rows * cols], vec![rows, cols]))
        };
        let shape = AttnShape::Gqa {
            n_heads: 4,
            n_kv_heads: 2,
        };
        let (head_dim, hidden) = (6, 24);
        let build = |q_rows: usize, k_rows: usize| AttnWeights {
            q_proj: m(q_rows, hidden),
            k_proj: m(k_rows, hidden),
            v_proj: m(k_rows, hidden),
            o_proj: m(hidden, q_rows),
            norm_weight: NormOp::None,
            q_norm: None,
            k_norm: None,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            post_attn_norm: None,
            post_ffn_norm: None,
            output_gate: None,
            sinks: None,
            attn_sub_norm: None,
            o_scale: None,
            o_bias: None,
            shortconv: None,
            ssm: None,
            q_gate_interleaved: false,
        };
        assert!(
            check_gqa_projection_widths(0, shape, head_dim, head_dim, hidden, &build(24, 12))
                .is_ok()
        );
        // K sized for 3 KV heads on a 2-KV-head layer.
        let err = check_gqa_projection_widths(1, shape, head_dim, head_dim, hidden, &build(24, 18))
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("blk.1.attn_k.weight"), "{msg}");
        assert!(msg.contains("head_count_kv 2"), "{msg}");
        // Q sized for 3 heads on a 4-head layer.
        let err = check_gqa_projection_widths(2, shape, head_dim, head_dim, hidden, &build(18, 12))
            .unwrap_err();
        assert!(format!("{err}").contains("blk.2.attn_q.weight"), "{err}");
    }

    /// Every row of the reach table cites a llama.cpp line, and the
    /// five that this seam serves are the ones on the generic path.
    #[test]
    fn the_reach_table_cites_its_lines_and_names_what_each_row_still_needs() {
        for (arch, note) in PER_LAYER_SHAPE_ARCHS {
            assert!(note.contains(".cpp:"), "`{arch}` cites no line: {note}");
        }
        let generic: Vec<&str> = PER_LAYER_SHAPE_ARCHS
            .iter()
            .filter(|(_, n)| n.starts_with("generic"))
            .map(|(a, _)| *a)
            .collect();
        assert_eq!(
            generic,
            [
                "deci",
                "openelm",
                "plamo3",
                "laguna",
                "step35",
                "spark2_5",
                "maple",
                "jamba",
                "nemotron_h",
                "granitehybrid",
                "granite-hybrid"
            ]
        );
        assert!(per_layer_shapes_read_by_llama_cpp("deci"));
        assert!(!per_layer_shapes_read_by_llama_cpp("granite"));
    }
}
