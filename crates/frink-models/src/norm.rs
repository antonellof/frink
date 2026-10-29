//! The normalisation at ONE site in a decoder: before the attention
//! branch, before the FFN branch, or before the LM head.
//!
//! Three variants, because llama.cpp's `build_norm`
//! (`llama-graph.cpp`) has three answers at those sites and frink used
//! to have one. It takes a norm TYPE (`LLM_NORM` = LayerNorm,
//! `LLM_NORM_RMS` = RMSNorm) and a weight that may be null, and applies
//! the multiply only `if (mw)`. Everything below is a reading of that
//! function and of the three graphs that reach its corners.
//!
//! ```text
//! // the ordinary case, and nearly every architecture on the generic path
//! ffn_inp = x       + attn(rms(x, attn_norm))
//! out     = ffn_inp + ffn(rms(ffn_inp, ffn_norm))
//! ```
//!
//! Gemma-2 added a *sandwich*: the same two pre-norms, plus a norm on
//! each branch's OUTPUT before its residual add. frink has carried
//! those two as [`crate::decoder::AttnWeights::post_attn_norm`] and
//! [`crate::decoder::AttnWeights::post_ffn_norm`] for a long time, and
//! they are not this type -- they are `Option<Vec<f32>>` RMSNorms, and
//! no architecture has ever wanted anything else there.
//!
//! # [`NormOp::None`]: `olmo2` and `exaone4`
//!
//! **The sandwich with the bread taken off.** They have the two
//! post-norms and NO pre-norms at all, and both sublayers read the raw
//! residual:
//!
//! ```text
//! ffn_inp = x       + post_attn_norm(attn(x))
//! out     = ffn_inp + post_ffn_norm(ffn(ffn_inp))
//! ```
//!
//! That claim is a reading of both files, not a family resemblance:
//!
//! | | `src/models/olmo2.cpp` | `src/models/exaone4.cpp` |
//! |---|---|---|
//! | per-layer norms created | `attn_q_norm`, `attn_k_norm`, `attn_post_norm`, `ffn_post_norm` (:45-52) | `attn_post_norm`, `attn_q_norm`, `attn_k_norm`, `ffn_post_norm` (:60-67) |
//! | `attn_norm` | absent | absent |
//! | `ffn_norm` | absent | absent |
//! | Q/K/V read | `cur = inpL` (:92) | `cur = inpL` (:118) |
//! | attention output | `build_norm(cur, attn_post_norm)` (:160-163) | `build_norm(cur, attn_post_norm)` (:152) |
//! | `ffn_inp` | `add(cur, inpSA)` (:165) | `add(cur, inpSA)` (:155) |
//! | FFN input | `build_ffn(ffn_inp, ...)` (:169) | `build_ffn(ffn_inp, ...)` (:159) |
//! | FFN output | `build_norm(cur, ffn_post_norm)` (:177-179) | `build_norm(cur, ffn_post_norm)` (:166) |
//! | residual | `add(cur, ffn_inp)` (:182) | `add(cur, ffn_inp)` (:169) |
//!
//! Line for line the same graph. So the two rows share ONE
//! implementation -- this module -- rather than getting one arm each.
//! What they do NOT share is their QK-norm style (`olmo2` norms the 2-D
//! projection over its whole width, `exaone4` per head after
//! `build_qkv` has reshaped), which is why each still needs its own
//! fixture: `tests/post_norm_only_graphs.rs`.
//!
//! # [`NormOp::LayerNormNoParams`]: `olmo`, and only `olmo`
//!
//! OLMo-1 is a THIRD shape, and it is not about the residual at all.
//! `src/models/olmo.cpp:27-35` creates Q/K/V, `attn_output` and
//! gate/up/down and **not one norm tensor** -- no `attn_norm`, no
//! `ffn_norm`, no `output_norm` -- and its graph normalises at all
//! three sites with a null weight AND a null bias:
//!
//! ```text
//! // olmo.cpp:65-67, :104-106, :128-130
//! cur = build_norm(inpL, NULL, NULL, LLM_NORM, il);
//! ```
//!
//! `LLM_NORM` is `ggml_norm` (`llama-graph.cpp`'s `build_norm`), which
//! subtracts the mean and divides by the standard deviation
//! (`ggml/src/ggml-cpu/ops.cpp:3716-3745`); with both weight and bias
//! null, `build_norm` does nothing further. So it is a pre-norm layer
//! like `llama`, with a different norm FUNCTION and no parameters at
//! all. `NormOp::None` is no help here and neither is `NormOp::Rms`.
//!
//! **This is the only architecture in llama.cpp that does it.** Scanned
//! over every `build_norm` call in all 155 `src/models/*.cpp` graphs,
//! extracting the weight argument: three calls pass a null weight to
//! `LLM_NORM`, and all three are `olmo.cpp`. (`talkie.cpp` passes a null
//! weight to `LLM_NORM_RMS` at five sites, which is a different
//! function and a different row, and since the 2026-09-19 pin move so
//! do `hrm-text.cpp` at three sites and `muse-glimmer.cpp` at one --
//! the RMS row grew, the LayerNorm row did not.) So this variant
//! closes exactly one refusal and the hoped-for shared cause is not
//! there -- see
//! `capability::NON_PARAMETRIC_LAYER_NORM`, which says so where the next
//! person will look.
//!
//! # [`NormOp::LayerNorm`]: `dbrx`, the row that gave the variant a caller
//!
//! The LayerNorm FUNCTION is shared more widely than the parameterless
//! corner: `dbrx` and the `nemotron` / `orion` / `stablelm` /
//! `codeshell` / `jais2` / `starcoder` / `starcoder2` / `phimoe` group
//! all normalise with `LLM_NORM` and a learned weight. This variant was
//! deliberately NOT written beside `LayerNormNoParams`, because at that
//! point every one of those rows refused for more than the norm, and a
//! variant with no caller silently rots. `dbrx` is the caller:
//! `src/models/dbrx.cpp:69-71`, `:110-112` and `:140-142` are
//! `build_norm(x, w, NULL, LLM_NORM, il)` -- weight, no bias -- and its
//! other two blockers, `{arch}.attention.clamp_kqv` and a pre-FFN norm
//! stored as `blk.N.attn_output_norm`, took one implementation each
//! (`crate::clamp_kqv`, `crate::norm_sites`).
//!
//! **Weight but no bias, on purpose.** Every other row in that group
//! creates `*_norm.bias` as REQUIRED and `build_norm` adds it after the
//! multiply. That is a fourth variant, `LayerNorm(w, b)`, and it is
//! absent for the reason this one was absent before `dbrx`: no admitted
//! row calls it. [`NormFunction`] is where the choice is made per
//! architecture; a bias means a variant there and at every match on
//! this enum, which is the compile error that is wanted.
//!
//! # Why this is a type and not a `bool` on `ModelConfig`
//!
//! The RMSNorm weights are handed to fused Metal kernels that apply the
//! norm INSIDE the kernel (`PrefillDenseLayerMetal::attn_norm_w`,
//! `MoeLayerMetal::ffn_norm_w`, `launch_decode_dense_layer`). A flag on
//! the config would leave every one of those launches free to keep
//! reading a `&[f32]` that no longer means anything, and nothing would
//! fail: that is precisely this repo's dominant bug shape, two
//! structures that must agree with nothing enforcing it, and it is how
//! `post_attn_norm` was lost from a decode path once already.
//!
//! Making the slot an enum instead means a fused launch cannot compile
//! until it has said what it does when there is no weight to hand over.
//! [`NormOp::rms_weights`] returns `None` for BOTH non-RMS variants, and
//! every GPU call site turns that into a fall-back to the host body,
//! which computes the right thing. The disagreement is a type error
//! rather than a silent wrong answer. `Decoder::final_norm` is this type
//! for the same reason: `olmo` is the first architecture whose FINAL
//! norm is not an RMSNorm either, and the fused stacks that fold
//! `final_norm + lm_head + argmax` had `Some(&self.final_norm)` written
//! into them unconditionally.

use frink_core::matmul::rms_norm;

/// The normalisation applied at one norm site.
#[derive(Debug, Clone, PartialEq)]
pub enum NormOp {
    /// RMSNorm with these learned weights. Every architecture on the
    /// generic path except the two families below.
    Rms(Vec<f32>),
    /// Non-parametric LayerNorm: subtract the mean, divide by the
    /// standard deviation, no learned weight and no bias.
    ///
    /// `olmo` (OLMo-1) and nothing else in llama.cpp. The weighted form
    /// is [`NormOp::LayerNorm`], which arrived only when `dbrx` gave it
    /// a caller; the biased form still has none.
    LayerNormNoParams,
    /// `build_norm(x, nullptr, nullptr, LLM_NORM_RMS, il)`: `x /
    /// sqrt(mean(x^2) + eps)` with no weight. `talkie`'s every norm
    /// site (`capability::NON_PARAMETRIC_RMS_NORM`); the RMS twin of
    /// [`Self::LayerNormNoParams`].
    RmsNoParams,
    /// LayerNorm with a learned weight and no bias:
    /// `(x - mean) / sqrt(var + eps) * w`.
    ///
    /// `dbrx` (`dbrx.cpp:69-71`, `:110-112`, `:140-142`).
    LayerNorm(Vec<f32>),
    /// LayerNorm with a learned weight AND bias:
    /// `(x - mean) / sqrt(var + eps) * w + b` -- `build_norm(x, w, b,
    /// LLM_NORM, il)`, which multiplies `if (mw)` and then adds `if (mb)`
    /// (`llama-graph.cpp`).
    ///
    /// `orion` (`orion.cpp:63-66,104-107,127-130`) and `nemotron`
    /// (`nemotron.cpp:71-74,111-114,136-139`), both with all six per-layer
    /// tensors and both output-norm tensors REQUIRED
    /// (`capability::BIASED_LAYER_NORM`). The variant every row in the
    /// old "LayerNorm-with-bias group" shares; it arrived when two rows
    /// needed nothing else, and the six that need more say what.
    LayerNormBias { weight: Vec<f32>, bias: Vec<f32> },
    /// RMSNorm with a learned weight AND bias: `x / sqrt(mean(x^2) +
    /// eps) * w + b` -- `build_norm(x, w, b, LLM_NORM_RMS, il)`, the
    /// same multiply-then-add as [`Self::LayerNormBias`] over the RMS
    /// rather than the centred vector.
    ///
    /// `phimoe` (Phi-3.5-MoE): `phimoe.cpp:20-21,28-29,35-36` create
    /// the six per-layer and two output tensors REQUIRED, and its graph
    /// is `phi3`'s (`models.h:632`), whose `phi3.cpp:99-102,137-139,
    /// 174-177` pass each pair to `LLM_NORM_RMS`; `phi3` itself never
    /// creates the biases, so its files take [`Self::Rms`]. One graph of
    /// 140 on the generic path (`capability::BIASED_RMS_NORM`;
    /// `deepseek32` and `glm-dsa` pass a bias to `LLM_NORM_RMS` on other
    /// engines, `chameleon` passes NULL).
    RmsBias { weight: Vec<f32>, bias: Vec<f32> },
    /// No norm at all: the branch reads the raw residual.
    ///
    /// `olmo2` and `exaone4`. NOT "an RMSNorm whose weights are all
    /// ones" -- that would still divide by the RMS of the residual, and
    /// the whole point of this variant is that nothing is divided.
    None,
}

impl NormOp {
    /// The site's output: `rms_norm(x, w, eps)`, the non-parametric
    /// LayerNorm, or `x` itself.
    ///
    /// Returns an owned vector in every arm, which is what every caller
    /// already had: `rms_norm` allocates too.
    ///
    /// `eps` is `ModelConfig::rms_norm_eps`, which for a LayerNorm
    /// architecture is read from `{arch}.attention.layer_norm_epsilon`
    /// -- llama.cpp's `f_norm_eps` rather than `f_norm_rms_eps`. The two
    /// are one field here because no architecture reads both keys, and
    /// `loader.rs` already accepted either spelling into that field
    /// before this variant existed.
    pub fn apply(&self, x: &[f32], eps: f32) -> Vec<f32> {
        match self {
            Self::Rms(w) => rms_norm(x, w, eps),
            Self::RmsNoParams => rms_norm_no_params(x, eps),
            Self::LayerNormNoParams => layer_norm_no_params(x, eps),
            Self::LayerNorm(w) => {
                // `build_norm` (llama-graph.cpp): `ggml_norm`, then
                // `ggml_mul(cur, mw)` -- the same centred vector as the
                // parameterless variant, scaled per element.
                let mut out = layer_norm_no_params(x, eps);
                debug_assert_eq!(out.len(), w.len());
                for (o, w) in out.iter_mut().zip(w.iter()) {
                    *o *= w;
                }
                out
            }
            Self::LayerNormBias { weight, bias } => {
                // The same, then `ggml_add(cur, mb)`: the bias lands
                // AFTER the multiply, so it is not scaled by `w`.
                let mut out = layer_norm_no_params(x, eps);
                debug_assert_eq!(out.len(), weight.len());
                debug_assert_eq!(out.len(), bias.len());
                for ((o, w), b) in out.iter_mut().zip(weight.iter()).zip(bias.iter()) {
                    *o = *o * w + b;
                }
                out
            }
            Self::RmsBias { weight, bias } => {
                // `rms_norm` is the multiply; then the same `ggml_add`.
                let mut out = rms_norm(x, weight, eps);
                debug_assert_eq!(out.len(), bias.len());
                for (o, b) in out.iter_mut().zip(bias.iter()) {
                    *o += b;
                }
                out
            }
            Self::None => x.to_vec(),
        }
    }

    /// The weights a fused GPU kernel needs, or `None` when this site
    /// has no RMSNorm weights and the kernel therefore must not run.
    ///
    /// Every fused Metal launch that bakes an RMSNorm into its kernel
    /// goes through here and falls back to the host body on `None`.
    /// See the module docs for why that is a `?` and not a comment.
    pub fn rms_weights(&self) -> Option<&[f32]> {
        match self {
            Self::Rms(w) => Some(w),
            // A biased RMSNorm has RMS weights, and no fused kernel adds
            // the bias after them: the host body, not the launch.
            Self::RmsNoParams
            | Self::LayerNormNoParams
            | Self::LayerNorm(_)
            | Self::LayerNormBias { .. }
            | Self::RmsBias { .. }
            | Self::None => None,
        }
    }
}

/// One learned tensor of a norm site: the `.weight` or the `.bias`
/// suffix of its GGUF name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormParam {
    Weight,
    Bias,
}

impl NormParam {
    /// The GGUF suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            NormParam::Weight => "weight",
            NormParam::Bias => "bias",
        }
    }
}

/// The norm FUNCTION an architecture applies at its parametric sites --
/// the `LLM_NORM` / `LLM_NORM_RMS` argument of `build_norm`, plus
/// whether there is a weight to hand it.
///
/// One answer per architecture, read at every site. The loader used to
/// decide this with a chain of `if` branches restated at the
/// pre-attention, pre-FFN and final sites: three places that had to
/// agree about one thing. `crate::norm_sites` resolves it ONCE through
/// [`norm_function`] and the three sites read the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormFunction {
    /// `LLM_NORM_RMS` with a weight. The generic path's default.
    Rms,
    /// `LLM_NORM` with a weight and no bias: `dbrx`
    /// (`capability::WEIGHTED_LAYER_NORM`).
    LayerNorm,
    /// `LLM_NORM` with a weight AND a bias: `orion`, `nemotron`
    /// (`capability::BIASED_LAYER_NORM`), [`NormOp::LayerNormBias`].
    LayerNormBias,
    /// `LLM_NORM` with neither: `olmo`
    /// (`capability::NON_PARAMETRIC_LAYER_NORM`).
    LayerNormNoParams,
    /// `LLM_NORM_RMS` with a null weight: `talkie`
    /// (`capability::NON_PARAMETRIC_RMS_NORM`), [`NormOp::RmsNoParams`].
    RmsNoParams,
    /// `LLM_NORM_RMS` with a weight AND a bias: `phimoe`
    /// (`capability::BIASED_RMS_NORM`), [`NormOp::RmsBias`].
    RmsBias,
}

impl NormFunction {
    /// The [`NormOp`] for one site, reading its weight through `load`
    /// only when this function has a weight to read.
    ///
    /// The closure rather than a `Vec<f32>` argument is the point:
    /// `LayerNormNoParams` never calls it, so a file that carries no
    /// norm tensor (OLMo-1 ships none) is never asked for one, and a
    /// site cannot be handed a weight its function would drop.
    ///
    /// `load` is asked for each PART the function has -- `Weight`, and
    /// for the biased form `Bias` too -- so a function that has no bias
    /// never asks the file for one and the biased form cannot be built
    /// with the bias forgotten.
    pub fn resolve<E>(
        self,
        mut load: impl FnMut(NormParam) -> Result<Vec<f32>, E>,
    ) -> Result<NormOp, E> {
        Ok(match self {
            Self::Rms => NormOp::Rms(load(NormParam::Weight)?),
            Self::LayerNorm => NormOp::LayerNorm(load(NormParam::Weight)?),
            Self::LayerNormBias => NormOp::LayerNormBias {
                weight: load(NormParam::Weight)?,
                bias: load(NormParam::Bias)?,
            },
            Self::LayerNormNoParams => NormOp::LayerNormNoParams,
            Self::RmsNoParams => NormOp::RmsNoParams,
            Self::RmsBias => NormOp::RmsBias {
                weight: load(NormParam::Weight)?,
                bias: load(NormParam::Bias)?,
            },
        })
    }
}

/// Which norm function `arch` applies, from the two capability lists
/// that name the exceptions.
///
/// Both lists are consulted here and nowhere else, so an architecture
/// on both would be answered ONE way rather than by whichever branch
/// came first -- and `crate::loader`'s
/// `the_norm_slot_and_function_lists_cannot_contradict` pins that the
/// situation never arises.
pub fn norm_function(arch: &str) -> NormFunction {
    norm_function_for_file(arch, None)
}

/// Architectures whose POST-attention and POST-FFN norms run at an eps
/// the graph writes as a LITERAL, not at the model's
/// `attention.layer_norm_rms_epsilon`.
///
/// `muse-glimmer.cpp:63` is `const float post_norm_eps = 1e-8f;` with
/// the comment "Different to f_norm_rms_eps for post-attn / post-FFN
/// norms", and `:140-141,166-167` call `ggml_rms_norm` with it while
/// `:90,153` norm with the model's. `grep -rn 'post_norm_eps'
/// src/models/*.cpp` over the 155 graphs is that one file (measured
/// 2026-09-19), so for everybody else the two epsilons are the same
/// number and [`post_norm_eps`] answers the model's.
///
/// A table rather than a second epsilon field on every config: the
/// fact belongs to the architecture, and a `ModelConfig` that carried
/// two independent epsilons would let a loader set one and forget the
/// other.
pub const POST_NORM_EPS_LITERAL: &[(&str, f32)] = &[("muse-glimmer", 1e-8)];

/// The epsilon this architecture's post-norm sites run at, given the
/// model's own. See [`POST_NORM_EPS_LITERAL`].
pub fn post_norm_eps(arch: &str, model_eps: f32) -> f32 {
    POST_NORM_EPS_LITERAL
        .iter()
        .find(|(a, _)| *a == arch)
        .map_or(model_eps, |(_, eps)| *eps)
}

/// The ONE graph of 155 whose norm function is decided by the FILE:
/// `cohere2moe.cpp:4-11` read both epsilon keys as optional, zero the
/// RMS one when it is absent, and `:166,314` pick `LLM_NORM` when
/// `f_norm_rms_eps == 0.0f` and `LLM_NORM_RMS` otherwise. Every real
/// export writes `attention.layer_norm_epsilon` alone
/// (`conversion/base.py:1354-1355` from `layer_norm_eps`), so the
/// architecture's default is the weighted LayerNorm
/// (`capability::WEIGHTED_LAYER_NORM`) and a file carrying a nonzero
/// `attention.layer_norm_rms_epsilon` switches to RMS; measured
/// (`grep -n 'f_norm_rms_eps == 0' src/models/*.cpp`).
pub const NORM_BY_RMS_EPS_KEY: &[(&str, &str)] =
    &[("cohere2moe", "src/models/cohere2moe.cpp:4-11,166")];

/// [`norm_function`] with what the file declares for
/// `attention.layer_norm_rms_epsilon`, for [`NORM_BY_RMS_EPS_KEY`];
/// every other architecture ignores the argument.
pub fn norm_function_for_file(arch: &str, declared_rms_eps: Option<f32>) -> NormFunction {
    let facts = crate::models::ArchFacts::of(arch);
    file_norm_function(
        facts.norm.norm_by_rms_eps_key,
        facts.tensors.norm_sites.function,
        declared_rms_eps,
    )
}

/// [`norm_function_for_file`] given the rows (`ArchFacts`): RMSNorm
/// where the file decides it, else the architecture's own function.
pub fn file_norm_function(
    by_rms_eps_key: bool,
    arch_function: NormFunction,
    declared_rms_eps: Option<f32>,
) -> NormFunction {
    if by_rms_eps_key && declared_rms_eps.is_some_and(|eps| eps != 0.0) {
        return NormFunction::Rms;
    }
    arch_function
}

/// The architecture's own function from the legacy lists alone, for
/// `ArchFacts::from_tables` (`olmo`'s non-parametric LayerNorm is in
/// `crate::models::olmo`).
pub(crate) fn norm_function_from_table(arch: &str) -> NormFunction {
    if crate::capability::uses_non_parametric_rms_norm(arch) {
        NormFunction::RmsNoParams
    } else if crate::capability::WEIGHTED_LAYER_NORM.contains(&arch) {
        NormFunction::LayerNorm
    } else if crate::capability::uses_biased_layer_norm(arch) {
        NormFunction::LayerNormBias
    } else {
        NormFunction::Rms
    }
}

/// `(x - mean) / sqrt(var + eps)`, with the BIASED variance.
///
/// `ggml_compute_forward_norm_f32` (`ggml/src/ggml-cpu/ops.cpp:3716-3745`)
/// divides the sum of squared deviations by `ne00`, not by `ne00 - 1`.
/// Bessel's correction on a 4096-wide hidden state is a factor of
/// 1.00012, which is far too small to fail a smoke test and far too
/// large to be right.
/// `ggml_rms_norm` with no multiply after it: `x * rsqrt(mean(x^2) + eps)`.
pub fn rms_norm_no_params(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    debug_assert!(n > 0.0, "a norm site with no elements");
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().map(|v| v * scale).collect()
}

fn layer_norm_no_params(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    debug_assert!(n > 0.0, "a norm site with no elements");
    let mean = x.iter().sum::<f32>() / n;
    // The deviations are computed once and reused, which is also what
    // ggml does: it writes `x - mean` into the destination and takes
    // the variance from there.
    let mut out: Vec<f32> = x.iter().map(|v| v - mean).collect();
    let var = out.iter().map(|d| d * d).sum::<f32>() / n;
    let scale = 1.0 / (var + eps).sqrt();
    for v in out.iter_mut() {
        *v *= scale;
    }
    out
}

// There is deliberately no `is_present()` beside `rms_weights()`.
// "Does this site norm?" and "what weights does the kernel get?" are
// the same fact for a fused kernel, and two spellings of one fact is
// the shape this repo keeps shipping bugs in: `rms_weights().is_some()`
// is the only way to ask.

#[cfg(test)]
mod tests {
    use super::*;

    /// `NormOp::None` is the identity, and an all-ones RMSNorm is not.
    ///
    /// The tempting shortcut for `olmo2` / `exaone4` was to load a
    /// vector of ones into the existing slot and change nothing else.
    /// It is wrong by exactly the RMS scale factor, which for a residual
    /// with any magnitude at all is not close to 1. This test is what
    /// stops somebody re-discovering that as a "simplification".
    #[test]
    fn no_norm_is_the_identity_and_an_all_ones_rmsnorm_is_not() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let eps = 1e-5;

        assert_eq!(NormOp::None.apply(&x, eps), x);

        let ones = NormOp::Rms(vec![1.0; x.len()]);
        let normed = ones.apply(&x, eps);
        let worst = x
            .iter()
            .zip(normed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst > 1.0,
            "an all-ones RMSNorm moved the vector by only {worst}; if it were the \
             identity the post-norm-only topology would not need a variant at all"
        );
    }

    /// The non-parametric LayerNorm is not the all-ones RMSNorm either,
    /// and the difference is the MEAN.
    ///
    /// This is the shortcut somebody will reach for next: "OLMo-1 just
    /// has no norm weights, so load ones". On a vector with a non-zero
    /// mean the two differ in every element, and the assertion below
    /// measures that rather than trusting it -- a centred input would
    /// make them agree and would prove nothing.
    #[test]
    fn the_layer_norm_subtracts_the_mean_and_an_all_ones_rmsnorm_does_not() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let eps = 1e-5;
        let ln = NormOp::LayerNormNoParams.apply(&x, eps);
        let rms = NormOp::Rms(vec![1.0; x.len()]).apply(&x, eps);

        let mean: f32 = x.iter().sum::<f32>() / x.len() as f32;
        assert!(mean.abs() > 1.0, "the input must not be centred: {mean}");

        let out_mean: f32 = ln.iter().sum::<f32>() / ln.len() as f32;
        assert!(
            out_mean.abs() < 1e-5,
            "a LayerNorm's output is centred; got mean {out_mean}"
        );

        let worst = ln
            .iter()
            .zip(rms.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            worst > 0.1,
            "the two norms differ by only {worst}; an all-ones RMSNorm would then be a \
             legitimate stand-in for OLMo-1's norm and this variant would be decoration"
        );
    }

    /// The variance is BIASED (divide by n), which is ggml's.
    ///
    /// Checked against arithmetic written out here rather than against
    /// the implementation: the two spellings differ by sqrt(n/(n-1)),
    /// which on this 4-element vector is 1.155 and on a real hidden
    /// state is 1.0001 -- big enough to be wrong, small enough that no
    /// end-to-end smoke test would notice.
    #[test]
    fn the_variance_is_the_biased_one_ggml_uses() {
        let x = [1.0f32, 2.0, 3.0, 10.0];
        let got = NormOp::LayerNormNoParams.apply(&x, 0.0);

        let n = x.len() as f64;
        let mean = x.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let want: Vec<f32> = x
            .iter()
            .map(|v| ((*v as f64 - mean) / var.sqrt()) as f32)
            .collect();

        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }

        // ... and the SAMPLE variance would be visibly different here,
        // so this test can tell them apart.
        let sample = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let worst = got
            .iter()
            .zip(x.iter())
            .map(|(g, v)| (g - ((*v as f64 - mean) / sample.sqrt()) as f32).abs())
            .fold(0f32, f32::max);
        assert!(worst > 0.1, "the two variances differ by only {worst}");
    }

    /// A fused GPU launch cannot be handed weights that do not exist,
    /// from EITHER non-RMS variant.
    #[test]
    fn only_the_rms_variant_offers_weights_to_a_fused_kernel() {
        assert_eq!(
            NormOp::Rms(vec![2.0, 3.0]).rms_weights(),
            Some(&[2.0f32, 3.0][..])
        );
        assert_eq!(NormOp::None.rms_weights(), None);
        assert_eq!(NormOp::LayerNormNoParams.rms_weights(), None);
        assert_eq!(NormOp::LayerNorm(vec![2.0, 3.0]).rms_weights(), None);
    }

    /// The weighted LayerNorm is the parameterless one times its
    /// weight, element by element -- `build_norm`'s `ggml_norm` then
    /// `ggml_mul(cur, mw)`.
    ///
    /// Checked against the composition rather than against a rewrite of
    /// the arithmetic, because the composition IS the claim: if the
    /// variant ever centred differently from `LayerNormNoParams`, `olmo`
    /// and `dbrx` would disagree about what `LLM_NORM` means.
    #[test]
    fn the_weighted_layer_norm_is_the_parameterless_one_times_its_weight() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let w = vec![0.5f32, -2.0, 1.5, 4.0];
        let eps = 1e-5;
        let got = NormOp::LayerNorm(w.clone()).apply(&x, eps);
        let base = NormOp::LayerNormNoParams.apply(&x, eps);
        for ((g, b), w) in got.iter().zip(base.iter()).zip(w.iter()) {
            assert!((g - b * w).abs() < 1e-6, "got {got:?}, base {base:?}");
        }
    }

    /// ... and it is NOT an RMSNorm with the same weight, which is the
    /// substitution a loader makes by reading `dbrx`'s `attn_norm.weight`
    /// into the slot every other architecture uses.
    #[test]
    fn the_weighted_layer_norm_is_not_an_rmsnorm_with_the_same_weight() {
        let x = vec![3.0f32, -4.0, 12.0, 0.5];
        let w = vec![0.5f32, -2.0, 1.5, 4.0];
        let eps = 1e-5;
        let ln = NormOp::LayerNorm(w.clone()).apply(&x, eps);
        let rms = NormOp::Rms(w).apply(&x, eps);
        let worst = ln
            .iter()
            .zip(rms.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst > 0.1, "the two norms differ by only {worst}");
    }

    /// `NormFunction` maps each capability list to exactly one variant,
    /// and the default is RMS.
    #[test]
    fn the_norm_function_is_read_off_the_capability_lists() {
        assert_eq!(norm_function("olmo"), NormFunction::LayerNormNoParams);
        assert_eq!(norm_function("dbrx"), NormFunction::LayerNorm);
        for arch in ["llama", "qwen3", "olmo2", "gemma3", "grok"] {
            assert_eq!(norm_function(arch), NormFunction::Rms, "{arch}");
        }
        assert_eq!(norm_function("orion"), NormFunction::LayerNormBias);
        assert_eq!(norm_function("nemotron"), NormFunction::LayerNormBias);
        let w = |p: NormParam| -> Result<Vec<f32>, ()> {
            Ok(match p {
                NormParam::Weight => vec![1.0, 2.0],
                NormParam::Bias => vec![0.5, -0.5],
            })
        };
        assert_eq!(
            NormFunction::LayerNorm.resolve(w),
            Ok(NormOp::LayerNorm(vec![1.0, 2.0]))
        );
        assert_eq!(
            NormFunction::Rms.resolve(w),
            Ok(NormOp::Rms(vec![1.0, 2.0]))
        );
        assert_eq!(
            NormFunction::LayerNormBias.resolve(w),
            Ok(NormOp::LayerNormBias {
                weight: vec![1.0, 2.0],
                bias: vec![0.5, -0.5],
            })
        );
    }

    /// The parameterless function never asks the file for a weight.
    ///
    /// OLMo-1 files carry no norm tensor at all, so a loader that read
    /// one "just in case" would fail on every real checkpoint; and a
    /// loader that read one and dropped it would hide a file that is
    /// not what the architecture string says.
    #[test]
    fn the_parameterless_function_never_reads_a_weight() {
        let mut asked = false;
        let got = NormFunction::LayerNormNoParams.resolve(|_| -> Result<Vec<f32>, ()> {
            asked = true;
            Err(())
        });
        assert_eq!(got, Ok(NormOp::LayerNormNoParams));
        assert!(!asked, "the loader closure must not run");
    }
}
