//! Which tensor each of a decoder layer's norm sites is stored under,
//! and with which function, per architecture -- resolved ONCE.
//!
//! A generic layer has up to four norm sites, plus the final one before
//! the lm_head:
//!
//! ```text
//! ffn_inp = x       + post_attn( attn( pre_attn(x) ) )
//! out     = ffn_inp + post_ffn ( ffn ( pre_ffn(ffn_inp) ) )
//! logits  = lm_head( output( out ) )
//! ```
//!
//! llama.cpp names each site's tensor per architecture, in that
//! architecture's `load_arch_tensors`, and reuses ONE tensor name for
//! DIFFERENT sites: `blk.N.attn_output_norm` (`LLM_TENSOR_ATTN_OUT_NORM`,
//! `llama-arch.cpp:423`) is `dbrx`'s pre-FFN norm (`dbrx.cpp:34,110-113`)
//! and `grok`'s post-attention norm (`grok.cpp:62,143-146`), and
//! `blk.N.post_attention_norm` (`LLM_TENSOR_ATTN_POST_NORM`) is Gemma-2's
//! post-attention norm and `gpt-oss`'s pre-FFN one. So which SITE a
//! tensor feeds is decided by the architecture string, never by which
//! tensors happen to be present: reading one file's tensor with
//! another's meaning moves a whole norm to the other side of a residual
//! add, silently.
//!
//! **Why one table.** `loader.rs` used to decide all of this with a
//! chain of `if` branches restated at the pre-attention, pre-FFN and
//! final sites -- three copies of one decision, and each new
//! architecture added a branch to each. Two of those branches also
//! answered "which FUNCTION" (RMS, LayerNorm, none) alongside "which
//! TENSOR", so the norm function was decided three times too. Here the
//! function is [`NormFunction`], read once, and each site is a
//! [`StoredNorm`] naming the tensor spellings it accepts and whether the
//! file must have one. The loader reads the table; it does not restate
//! it.
//!
//! The lists below are the whole content of the table, and each one
//! cites the graph it was read from. `loader.rs`'s
//! `the_norm_slot_and_function_lists_cannot_contradict` pins that no
//! architecture is on two lists that would read one tensor two ways,
//! and `every_architecture_keyed_behaviour_table_names_a_real_generic_row`
//! that every name is a row this loader can reach.

use crate::loader::load_f32_vec;
use crate::norm::{NormFunction, NormOp, NormParam};
use crate::LoadError;
use frink_gguf::{GgufError, TensorSource};

/// Architectures that store their **pre-FFN** norm under the tensor name
/// `blk.N.post_attention_norm.weight` and carry no `blk.N.ffn_norm`.
///
/// Gemma writes the same tensor name for a genuinely different norm: it
/// is applied to the attention output *inside* the attention residual,
/// and Gemma also carries `ffn_norm`. Reading one file's tensor with the
/// other's meaning silently moves a whole RMSNorm to the wrong side of a
/// residual add, so the meaning is decided by architecture, not by which
/// tensors happen to be present.
///
/// - `gpt-oss`: `openai-moe.cpp` norms `ffn_inp` with `attn_post_norm`.
/// - `glm4moe`: `src/models/glm4-moe.cpp:75` creates `attn_post_norm`
///   and no `ffn_norm`, and `:215` norms `ffn_inp` with it -- the slot
///   its refusal had named for a year, and the one line that admits
///   GLM-4.5 / 4.5-Air / 4.6.
/// - `seed_oss`: `src/models/seed-oss.cpp:36-37` creates `attn_norm` and
///   `attn_post_norm` and **no** `ffn_norm`, and `:113-115` norms
///   `ffn_inp` -- the post-attention residual -- with `attn_post_norm`.
///
/// This is deliberately NOT `arch == "gpt-oss"`, which is what it used
/// to be. That one flag also gated gpt-oss's five extra per-layer
/// tensors (sinks, biases, the SwiGLU clamp), and widening it would have
/// handed `seed_oss` attention sinks it does not have. Two facts, two
/// predicates.
pub const PRE_FFN_NORM_IS_POST_ATTENTION_NORM: &[&str] = &[
    "gpt-oss",
    "seed_oss",
    "glm4moe",
    "qwen35",
    "qwen35moe",
    "qwen3next",
];

/// Architectures that store their **pre-FFN** norm under
/// `blk.N.attn_output_norm.weight` (`LLM_TENSOR_ATTN_OUT_NORM`) and carry
/// no `blk.N.ffn_norm`.
///
/// `dbrx`: `dbrx.cpp:34` creates `attn_out_norm` and no `ffn_norm`, and
/// `:110-113` norms `ffn_inp` -- the post-attention residual -- with it,
/// which is the same slot `gpt-oss` keeps under the other name. Its
/// norm FUNCTION is a weighted LayerNorm
/// (`capability::WEIGHTED_LAYER_NORM`); that is [`NormFunction`]'s
/// business, not this list's.
pub const PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM: &[&str] = &["dbrx"];

/// Architectures whose **post-attention** norm (Gemma-2's
/// `post_attention_norm` slot) is stored as `blk.N.attn_output_norm` and
/// whose **post-FFN** norm is `blk.N.layer_output_norm`
/// (`LLM_TENSOR_LAYER_OUT_NORM`, `llama-arch.cpp:421`) with
/// `post_ffw_norm` as the fallback spelling -- both REQUIRED.
///
/// `grok`: `grok.cpp:62` creates `attn_out_norm` as required and
/// `:143-146` applies it to the attention output BEFORE the residual
/// add (`ffn_inp = add(cur, inpSA)` at :148), which is the post-norm
/// slot and not `dbrx`'s pre-FFN one -- `grok` has its own `ffn_norm`
/// at :64. `:75-78` create `ffn_post_norm` from `LAYER_OUT_NORM` when
/// present (`conversion/grok.py`'s Grok-1 mapping, `rms_norm_3`) and
/// from `FFN_POST_NORM` otherwise (Grok-2's `post_moe_norm`), required
/// either way, and `:185-188` apply it before the FFN residual add.
pub const POST_NORMS_UNDER_GROK_NAMES: &[&str] = &["grok"];

/// Architectures whose OPTIONAL second pre-norm `blk.N.attn_norm_2`
/// feeds ATTENTION when present, with `attn_norm` moving to the FFN.
///
/// `falcon`: `falcon.cpp:35-36` create `attn_norm_2` / its bias
/// `TENSOR_NOT_REQUIRED` (Falcon-40B and 180B have them, Falcon-7B does
/// not), `:79-85` norm the layer input with it for attention when it is
/// there and take `attn_norm`'s output otherwise, and `:124` hands the
/// FFN `attn_norm(x)` either way. So a 40B layer is the two-norm
/// parallel residual (`crate::parallel_residual`, `TwoNorms`) with the
/// tensor NAMES crossed relative to `gptneox`: the attention slot is
/// `attn_norm_2` and the pre-FFN slot is `attn_norm`. Decided PER LAYER
/// by tensor presence, which is why it is [`NormSites::for_layer`] and
/// not a row of [`NormSites::for_arch`]. Measured: `grep -l attn_norm_2
/// src/models/*.cpp` over all 155 is `falcon`, `bert`, `jina-bert-v2`
/// (encoders, a post-norm there), `rwkv6` / `rwkv7` / `eagle3` (other
/// engines); one generic-path graph.
pub const ATTN_NORM_2_FEEDS_ATTENTION: &[&str] = &["falcon"];

/// Architectures that norm the TOKEN EMBEDDINGS before layer 0, with
/// the model-level `token_embd_norm` pair (`LLM_TENSOR_TOKEN_EMBD_NORM`,
/// spelled without a `blk.` prefix).
///
/// `bloom`: `bloom.cpp:25-26` create `tok_norm` / `tok_norm_b`
/// REQUIRED and `:77-80` norm `inpL` with them, `LLM_NORM`, right after
/// `build_inp_embd`. Measured over all 155 graphs: `bloom` is the one
/// decoder on the generic path; `bert` / `nomic-bert` / `nomic-bert-moe`
/// / `jina-bert-v2` / `jina-bert-v3` / `modern-bert` are the encoder
/// engine, `rwkv6` / `rwkv7` / `wavtokenizer-dec` their own. The site
/// takes the architecture's norm FUNCTION like every other
/// ([`NormSites::function`]), so `bloom`'s is the biased LayerNorm.
pub const EMBEDDING_NORM_ARCHITECTURES: &[&str] = &["bloom"];

/// Architectures that norm the token embeddings with a WEIGHTLESS RMS:
/// `build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)` before layer
/// 0, with no `token_embd_norm` tensor to load.
///
/// `muse-glimmer.cpp:69` is the one (measured over the 155 graphs on
/// 2026-09-19, the same scan that found the weightless RMS in
/// `hrm-text.cpp`, which norms INSIDE its layers and not here). It is
/// a separate table from [`EMBEDDING_NORM_ARCHITECTURES`] because the
/// two answer different questions -- "is there a tensor" and "is there
/// a norm" -- and `bloom`'s site has a weight where this one has none.
pub const WEIGHTLESS_EMBEDDING_NORM: &[&str] = &["muse-glimmer"];

/// Architectures with NO final norm tensor at all.
///
/// `hrm-text.cpp:37-45` creates `token_embd`, `output` and
/// `hrm_z_l_init` and no `output_norm`: every stack ends with its own
/// weightless RMS (`:162`, `crate::layer_loops::LoopNorm::Weightless`)
/// and the last stack's IS the final norm, so the lm_head reads what
/// that norm produced. Asking for the tensor would refuse every real
/// HRM-Text file.
pub const NO_OUTPUT_NORM: &[&str] = &["hrm_text"];

/// Does this architecture norm its embeddings without a weight? See
/// [`WEIGHTLESS_EMBEDDING_NORM`].
pub fn weightless_embedding_norm(arch: &str) -> bool {
    WEIGHTLESS_EMBEDDING_NORM.contains(&arch)
}

/// Architectures whose OUTPUT norm is stored under the embedding-norm
/// NAME: `token_embd_norm.weight` is `LLM_TENSOR_OUTPUT_NORM_LFM2`
/// (`llama-arch.cpp:384`, "fix for wrong tensor name"), created at
/// `lfm2.cpp:36` as `output_norm` and applied at `:212` after the last
/// layer; nothing norms the embeddings (`:191`). The same name is
/// `bloom`'s embedding norm ([`EMBEDDING_NORM_ARCHITECTURES`]), one
/// tensor name feeding two different sites, decided by architecture --
/// the `attn_output_norm` case again. `lfm2moe` shares the loader line
/// (`lfm2moe.cpp:31`).
pub const OUTPUT_NORM_UNDER_EMBEDDING_NAME: &[&str] = &["lfm2", "lfm2moe"];

/// Architectures whose two OPTIONAL post norms must be present together
/// or not at all, with the line that refuses a file carrying one. The
/// default row loads each independently, so without this a file with
/// one would be served where llama.cpp throws.
pub const POST_NORMS_BOTH_OR_NEITHER: &[(&str, &str)] = &[("lfm3", "lfm3.cpp:53-55")];

/// `arch`'s [`POST_NORMS_BOTH_OR_NEITHER`] row: the llama.cpp line.
pub fn post_norms_both_or_neither(arch: &str) -> Option<&'static str> {
    POST_NORMS_BOTH_OR_NEITHER
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, line)| *line)
}

/// Why layer `layer`'s post norms cannot be served, or `None`.
pub fn post_norm_pairing_refusal(
    arch: &str,
    layer: usize,
    post_attn: bool,
    post_ffn: bool,
) -> Option<String> {
    post_norm_pairing_refusal_for(
        arch,
        post_norms_both_or_neither(arch),
        layer,
        post_attn,
        post_ffn,
    )
}

/// [`post_norm_pairing_refusal`] given the row (`ArchFacts`).
pub fn post_norm_pairing_refusal_for(
    arch: &str,
    row: Option<&str>,
    layer: usize,
    post_attn: bool,
    post_ffn: bool,
) -> Option<String> {
    let line = row?;
    let (has, lacks) = if post_attn {
        ("post_attention_norm", "post_ffw_norm")
    } else {
        ("post_ffw_norm", "post_attention_norm")
    };
    (post_attn != post_ffn).then(|| {
        format!(
            "blk.{layer} carries {has} without {lacks}: llama.cpp requires both post norms or \
             neither for `{arch}` (src/models/{line})"
        )
    })
}

/// Architectures with ONE norm per layer, `attn_norm`, whatever the
/// layer's block is: `nemotron-h.cpp:52` ("all blocks use the attn
/// norm"), `:145`. Every Nemotron-H layer is a single block with a
/// single residual add (`layer_shapes::BLOCK_WITHOUT_FFN_KEEPS_ITS_
/// OUTPUT`), so a layer reads EITHER the attention slot (a Mamba-2 or
/// attention layer, `ffn_dim 0`) OR the FFN slot (an FFN-only layer,
/// `AttnShape::Absent`), never both, and pointing the FFN slot at
/// `attn_norm` is exact.
pub const ONE_NORM_PER_LAYER: &[&str] = &["nemotron_h", "nemotron_h_moe"];

/// A norm site whose weight the file stores.
///
/// `names` are base names tried in order; each is looked up as
/// `blk.N.{name}.weight` first and `blk.N.{name}` second (or
/// `{name}.weight` / `{name}` for the final norm). The first spelling
/// present wins.
///
/// **Both spellings, because llama.cpp's own trees disagree.** `LLM_TN`
/// appends `.weight` only when it is given a suffix
/// (`src/llama-arch.cpp:898-910`). Every architecture that creates the
/// post-norms passes one -- `tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i)`
/// in gemma2, gemma3, glm4, exaone4, afmoe and the rest -- EXCEPT
/// `plamo3`, which uses the two-argument overload
/// (`src/models/plamo3.cpp:52,55`) and therefore asks for
/// `blk.N.post_attention_norm` with no suffix at all. The converter
/// agrees with it by a second accident: `gguf-py/gguf/tensor_mapping.py`
/// gives the PLaMo entries as `...post_mixer_norm.weight` -- keys that
/// already END in `.weight` -- and `TensorNameMap.get_type_and_name`
/// tries an exact match first, so the mapped name is emitted with
/// nothing appended. frink read only `.weight`, which is why `plamo3`
/// could not have loaded a real checkpoint: fail-closed rather than
/// wrong, but not "a fixture away", which is what its triage verdict
/// said. Both spellings are accepted rather than one chosen per
/// architecture, because the choice is a property of the file and
/// nothing in the metadata declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredNorm {
    pub names: &'static [&'static str],
    /// `create_tensor(..., 0)` in llama.cpp: the file must have one.
    /// `false` is `TENSOR_NOT_REQUIRED`.
    pub required: bool,
}

impl StoredNorm {
    pub(crate) const fn required(names: &'static [&'static str]) -> Self {
        Self {
            names,
            required: true,
        }
    }

    const fn optional(names: &'static [&'static str]) -> Self {
        Self {
            names,
            required: false,
        }
    }

    /// Every full tensor name this site accepts for its weight, in
    /// lookup order.
    fn candidates(&self, layer: Option<usize>) -> Vec<String> {
        self.candidates_for(layer, NormParam::Weight)
    }

    /// The names for one part. The bare name (no suffix) is a weight
    /// spelling some exporters use and is never a bias.
    fn candidates_for(&self, layer: Option<usize>, part: NormParam) -> Vec<String> {
        self.names
            .iter()
            .flat_map(|base| {
                let full = match layer {
                    Some(l) => format!("blk.{l}.{base}"),
                    None => (*base).to_string(),
                };
                match part {
                    NormParam::Weight => vec![format!("{full}.weight"), full],
                    NormParam::Bias => vec![format!("{full}.bias")],
                }
            })
            .collect()
    }

    /// The weight, or `None` for an optional site the file omits.
    pub fn load(
        &self,
        file: &impl TensorSource,
        layer: Option<usize>,
    ) -> Result<Option<Vec<f32>>, LoadError> {
        let candidates = self.candidates(layer);
        if let Some(name) = candidates.iter().find(|n| file.find_tensor(n).is_some()) {
            return load_f32_vec(file, name).map(Some);
        }
        if self.required {
            return Err(LoadError::Gguf(GgufError::TensorNotFound(
                candidates.join(" | "),
            )));
        }
        Ok(None)
    }

    /// One REQUIRED part of a REQUIRED site: the weight, or the bias of
    /// a `LayerNormBias` architecture, which `orion.cpp:18,25,31` and
    /// `nemotron.cpp:19,26,35` create with `create_tensor(..., 0)`.
    fn load_required_part(
        &self,
        file: &impl TensorSource,
        layer: Option<usize>,
        part: NormParam,
    ) -> Result<Vec<f32>, LoadError> {
        debug_assert!(self.required, "load_required_part on an optional site");
        let candidates = self.candidates_for(layer, part);
        match candidates.iter().find(|n| file.find_tensor(n).is_some()) {
            Some(name) => load_f32_vec(file, name),
            None => Err(LoadError::Gguf(GgufError::TensorNotFound(
                candidates.join(" | "),
            ))),
        }
    }
}

/// Where every norm of one architecture's layer lives.
///
/// The three pre-norm sites are `Option<StoredNorm>`: `None` is the
/// post-norm-only topology (`capability::POST_NORM_ONLY_ARCHITECTURES`),
/// where the site has no tensor AND no norm and
/// [`NormSites::load_pre_norm`] answers [`NormOp::None`]. A
/// non-parametric architecture (`olmo`) keeps its `Some`: the
/// [`NormFunction`] decides the tensor is never read, and that decision
/// belongs in one place rather than in a second `None` here.
///
/// The two post-norm sites are RMSNorm-only slots, as
/// `AttnWeights::post_attn_norm` / `post_ffn_norm` are: no admitted
/// architecture post-norms with anything else. `None` means the
/// architecture has no such site, which for `gpt-oss` / `seed_oss` is
/// how the tensor that WOULD be read there is kept for the pre-FFN
/// slot instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormSites {
    pub function: NormFunction,
    pub attn: Option<StoredNorm>,
    pub ffn: Option<StoredNorm>,
    pub post_attn: Option<StoredNorm>,
    pub post_ffn: Option<StoredNorm>,
    /// The final norm before the LM head, or `None` for
    /// [`NO_OUTPUT_NORM`] -- the one architecture whose last stack
    /// norm IS the final one.
    pub output: Option<StoredNorm>,
    /// The norm on the token embeddings before layer 0, or `None` for
    /// every architecture but [`EMBEDDING_NORM_ARCHITECTURES`].
    pub embedding: Option<StoredNorm>,
}

impl NormSites {
    /// The row for `arch`, with the architecture's own norm function:
    /// its `crate::models` definition, else the legacy tables.
    pub fn for_arch(arch: &str) -> Self {
        crate::models::ArchFacts::of(arch).tensors.norm_sites
    }

    /// The row of an architecture no norm table names: RMSNorm, the
    /// four layer sites under their usual names, `output_norm`, and no
    /// embedding norm.
    pub const GENERIC: Self = Self {
        function: NormFunction::Rms,
        attn: Some(StoredNorm::required(&["attn_norm"])),
        ffn: Some(StoredNorm::required(&["ffn_norm"])),
        post_attn: Some(StoredNorm::optional(&["post_attention_norm"])),
        post_ffn: Some(StoredNorm::optional(&["post_ffw_norm"])),
        output: Some(StoredNorm::required(&["output_norm"])),
        embedding: None,
    };

    /// The post-norm-only topology (`olmo2.cpp`, `exaone4.cpp`): no
    /// pre-attention and no pre-FFN norm, each branch's output normed
    /// before its residual add. One constant for the rows that share it.
    pub const POST_NORM_ONLY: Self = Self {
        attn: None,
        ffn: None,
        ..Self::GENERIC
    };

    /// The legacy tables' row for `arch` under `function`, for
    /// `ArchFacts::from_tables`; a `crate::models` architecture states
    /// its row there instead.
    pub fn with_function(arch: &str, function: NormFunction) -> Self {
        let mut sites = Self {
            function,
            ..Self::GENERIC
        };
        if NO_OUTPUT_NORM.contains(&arch) {
            sites.output = None;
        } else if OUTPUT_NORM_UNDER_EMBEDDING_NAME.contains(&arch) {
            sites.output = Some(StoredNorm::required(&["token_embd_norm"]));
        }
        if EMBEDDING_NORM_ARCHITECTURES.contains(&arch) {
            sites.embedding = Some(StoredNorm::required(&["token_embd_norm"]));
        }
        if PRE_FFN_NORM_IS_POST_ATTENTION_NORM.contains(&arch) {
            sites.ffn = Some(StoredNorm::required(&["post_attention_norm"]));
            sites.post_attn = None;
        }
        if PRE_FFN_NORM_IS_ATTN_OUTPUT_NORM.contains(&arch) {
            sites.ffn = Some(StoredNorm::required(&["attn_output_norm"]));
        }
        if ONE_NORM_PER_LAYER.contains(&arch) {
            sites.ffn = Some(StoredNorm::required(&["attn_norm"]));
        }
        if POST_NORMS_UNDER_GROK_NAMES.contains(&arch) {
            sites.post_attn = Some(StoredNorm::required(&["attn_output_norm"]));
            sites.post_ffn = Some(StoredNorm::required(&[
                "layer_output_norm",
                "post_ffw_norm",
            ]));
        }
        sites
    }

    /// This layer's row: the per-architecture row, with the two
    /// pre-norm slots crossed for a layer that carries `attn_norm_2` on
    /// an architecture in [`ATTN_NORM_2_FEEDS_ATTENTION`]. Every other
    /// layer of every other architecture gets the row unchanged.
    pub fn for_layer(&self, arch: &str, file: &impl TensorSource, layer: usize) -> Self {
        if !ATTN_NORM_2_FEEDS_ATTENTION.contains(&arch)
            || file
                .find_tensor(&format!("blk.{layer}.attn_norm_2.weight"))
                .is_none()
        {
            return *self;
        }
        Self {
            attn: Some(StoredNorm::required(&["attn_norm_2"])),
            ffn: Some(StoredNorm::required(&["attn_norm"])),
            ..*self
        }
    }

    /// A pre-norm site (attention, FFN, or with `layer == None` the
    /// final norm), as the [`NormOp`] the decoder applies there.
    pub fn load_pre_norm(
        &self,
        site: Option<StoredNorm>,
        file: &impl TensorSource,
        layer: Option<usize>,
    ) -> Result<NormOp, LoadError> {
        match site {
            None => Ok(NormOp::None),
            Some(stored) => self
                .function
                .resolve(|part| stored.load_required_part(file, layer, part)),
        }
    }

    /// A post-norm site: an RMSNorm weight, or `None` when this
    /// architecture has no such site or the file omits an optional one.
    pub fn load_post_norm(
        site: Option<StoredNorm>,
        file: &impl TensorSource,
        layer: usize,
    ) -> Result<Option<Vec<f32>>, LoadError> {
        match site {
            None => Ok(None),
            Some(stored) => stored.load(file, Some(layer)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falcon's second pre-norm crosses the two slots on exactly the
    /// layers that carry it; a 7B layer, and every layer of every other
    /// architecture, keeps the row.
    #[test]
    fn attn_norm_2_crosses_the_two_pre_norm_slots_per_layer() {
        use crate::test_source::StubSource;
        let row = NormSites::for_arch("falcon");
        assert_eq!(row.function, NormFunction::LayerNormBias);
        let seven_b = StubSource::with_tensors(&["blk.0.attn_norm.weight"]);
        assert_eq!(row.for_layer("falcon", &seven_b, 0), row);
        let forty_b = StubSource::with_tensors(&["blk.1.attn_norm_2.weight"]);
        assert_eq!(
            row.for_layer("falcon", &forty_b, 0),
            row,
            "layer 0 has none"
        );
        let crossed = row.for_layer("falcon", &forty_b, 1);
        assert_eq!(crossed.attn, Some(StoredNorm::required(&["attn_norm_2"])));
        assert_eq!(crossed.ffn, Some(StoredNorm::required(&["attn_norm"])));
        assert_eq!(crossed.output, row.output);
        // The name alone does nothing on another architecture.
        let llama = NormSites::for_arch("llama");
        assert_eq!(llama.for_layer("llama", &forty_b, 1), llama);
    }

    /// `lfm3.cpp:53-55` throws on a layer with one post norm; the
    /// default row would serve it, so the pairing table refuses it, and
    /// only for the architectures it names.
    #[test]
    fn one_post_norm_without_the_other_is_refused_where_upstream_throws() {
        assert_eq!(post_norm_pairing_refusal("lfm3", 0, true, true), None);
        assert_eq!(post_norm_pairing_refusal("lfm3", 0, false, false), None);
        let msg = post_norm_pairing_refusal("lfm3", 2, true, false).expect("refused");
        assert!(
            msg.contains("blk.2 carries post_attention_norm without post_ffw_norm")
                && msg.contains("lfm3.cpp:53-55"),
            "{msg}"
        );
        assert!(post_norm_pairing_refusal("lfm3", 1, false, true)
            .is_some_and(|m| m.contains("post_ffw_norm without post_attention_norm")));
        assert_eq!(post_norm_pairing_refusal("gemma2", 0, true, false), None);
    }

    /// The default row is the plain pre-norm layer with both post-norms
    /// optional, which is what every architecture not named in a list
    /// gets.
    #[test]
    fn the_default_row_is_the_plain_pre_norm_layer() {
        let s = NormSites::for_arch("llama");
        assert_eq!(s.function, NormFunction::Rms);
        assert_eq!(s.attn, Some(StoredNorm::required(&["attn_norm"])));
        assert_eq!(s.ffn, Some(StoredNorm::required(&["ffn_norm"])));
        assert_eq!(
            s.post_attn,
            Some(StoredNorm::optional(&["post_attention_norm"]))
        );
        assert_eq!(s.post_ffn, Some(StoredNorm::optional(&["post_ffw_norm"])));
        assert_eq!(s.output, Some(StoredNorm::required(&["output_norm"])));
        assert_eq!(s.embedding, None);
        assert_eq!(
            NormSites::for_arch("bloom").embedding,
            Some(StoredNorm::required(&["token_embd_norm"]))
        );
    }

    /// Nemotron-H's FFN-only layer norms with `attn_norm`
    /// (nemotron-h.cpp:52,145).
    #[test]
    fn nemotron_h_has_one_norm_per_layer() {
        let s = NormSites::for_arch("nemotron_h");
        assert_eq!(s.attn, Some(StoredNorm::required(&["attn_norm"])));
        assert_eq!(s.ffn, Some(StoredNorm::required(&["attn_norm"])));
    }

    /// `token_embd_norm` is bloom's EMBEDDING norm and LFM2's OUTPUT
    /// norm (llama-arch.cpp:384); each row reads it at its own site and
    /// not the other's.
    #[test]
    fn token_embd_norm_is_the_output_norm_on_lfm2_alone() {
        let s = NormSites::for_arch("lfm2");
        assert_eq!(s.output, Some(StoredNorm::required(&["token_embd_norm"])));
        assert_eq!(s.embedding, None);
        let b = NormSites::for_arch("bloom");
        assert_eq!(b.output, Some(StoredNorm::required(&["output_norm"])));
        for arch in OUTPUT_NORM_UNDER_EMBEDDING_NAME {
            assert!(
                !EMBEDDING_NORM_ARCHITECTURES.contains(arch),
                "{arch}: one tensor cannot feed both sites"
            );
        }
    }

    /// `attn_output_norm` feeds a DIFFERENT site on the two rows that
    /// store it, and neither row inherits the other's reading.
    ///
    /// This is the test that separates `grok` from `dbrx`: the same
    /// tensor name, one on each side of the attention residual add.
    #[test]
    fn attn_output_norm_is_dbrxs_pre_ffn_norm_and_groks_post_attention_norm() {
        let dbrx = NormSites::for_arch("dbrx");
        assert_eq!(dbrx.function, NormFunction::LayerNorm);
        assert_eq!(
            dbrx.ffn,
            Some(StoredNorm::required(&["attn_output_norm"])),
            "dbrx.cpp:110-113 norms ffn_inp with attn_out_norm"
        );
        assert_eq!(
            dbrx.post_attn,
            Some(StoredNorm::optional(&["post_attention_norm"])),
            "no post-attention norm of its own"
        );

        let grok = NormSites::for_arch("grok");
        assert_eq!(grok.function, NormFunction::Rms);
        assert_eq!(grok.ffn, Some(StoredNorm::required(&["ffn_norm"])));
        assert_eq!(
            grok.post_attn,
            Some(StoredNorm::required(&["attn_output_norm"])),
            "grok.cpp:143-146 norms the attention output before the residual add"
        );
        assert_eq!(
            grok.post_ffn,
            Some(StoredNorm::required(&[
                "layer_output_norm",
                "post_ffw_norm"
            ])),
            "grok.cpp:75-78: LAYER_OUT_NORM first, FFN_POST_NORM as the fallback"
        );
    }

    /// The post-norm-only rows have no pre-norm tensors and no pre-norm
    /// at all; the parameterless row keeps its sites and never reads
    /// them.
    #[test]
    fn the_two_no_tensor_shapes_are_kept_apart() {
        let olmo2 = NormSites::for_arch("olmo2");
        assert_eq!(olmo2.attn, None);
        assert_eq!(olmo2.ffn, None);
        assert_eq!(olmo2.function, NormFunction::Rms);

        let olmo = NormSites::for_arch("olmo");
        assert_eq!(olmo.function, NormFunction::LayerNormNoParams);
        assert!(
            olmo.attn.is_some(),
            "the site exists; the function skips the read"
        );
    }

    /// `gpt-oss` keeps `post_attention_norm` for the pre-FFN slot and
    /// therefore has NO post-attention site: the tensor cannot be read
    /// twice.
    #[test]
    fn the_gpt_oss_slot_reads_the_tensor_once() {
        let s = NormSites::for_arch("gpt-oss");
        assert_eq!(s.ffn, Some(StoredNorm::required(&["post_attention_norm"])));
        assert_eq!(s.post_attn, None);
    }

    /// Both spellings of every candidate, suffixed first, in order.
    #[test]
    fn candidates_try_the_suffixed_spelling_first_for_each_name() {
        let s = StoredNorm::required(&["layer_output_norm", "post_ffw_norm"]);
        assert_eq!(
            s.candidates(Some(3)),
            vec![
                "blk.3.layer_output_norm.weight",
                "blk.3.layer_output_norm",
                "blk.3.post_ffw_norm.weight",
                "blk.3.post_ffw_norm",
            ]
        );
        assert_eq!(
            StoredNorm::required(&["output_norm"]).candidates(None),
            vec!["output_norm.weight", "output_norm"]
        );
    }
}
