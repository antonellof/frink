//! The fences that keep a model or a layer off the fused Metal paths:
//! the RoPE descriptions those paths are handed, and every per-model and
//! per-layer fact no fused kernel implements.

use super::*;
use frink_core::tensor::Tensor;
use frink_moe::GluAct;

fn phi_like_config() -> ModelConfig {
    let mut cfg = crate::config::test_dense_fixture();
    cfg.head_dim = 128;
    cfg.rope_layout = crate::config::RopeLayout::Neox;
    cfg.rope_dim = Some(96);
    cfg.rope_attn_factor = 1.1902381;
    cfg
}

/// Both values must reach the kernels, and they must be the same two
/// the CPU path reads — otherwise the backends compute different
/// attention for the same weights, which is the whole reason the
/// model was refused Metal in the first place.
#[test]
fn metal_rope_carries_partial_rotary_and_mscale() {
    let decoder = Decoder::new_random_small(phi_like_config(), 1, 32);
    let rope = decoder.metal_rope();
    assert_eq!(rope.layout, frink_metal::attn::MetalRopeLayout::Neox);
    assert_eq!(rope.rot_dim, Some(96));
    assert_eq!(rope.attn_factor, 1.1902381);
}

/// `rope.dimension_count == head_dim` is "the whole head rotates",
/// which must reach the kernel as `None` rather than as a width —
/// same graph, one code path.
#[test]
fn rot_dim_equal_to_head_dim_becomes_none() {
    let mut cfg = phi_like_config();
    cfg.rope_dim = Some(cfg.head_dim);
    let decoder = Decoder::new_random_small(cfg, 1, 32);
    assert_eq!(decoder.metal_rope().rot_dim, None);
}

/// A non-unit `attn_factor` is no longer a reason to refuse Metal;
/// an odd `n_rot` still is, because ggml's `ggml_rope_impl` asserts
/// an even width and the split-half pairing is otherwise undefined
/// for the last channel.
#[test]
fn odd_rot_dim_is_still_refused_but_mscale_is_not() {
    let supported = |cfg: ModelConfig| {
        let d = Decoder::new_random_small(cfg, 1, 32);
        d.layer_supports_metal_attn(&d.layers[0])
    };

    // The control: with no rope oddity the fixture is admitted, so
    // the two assertions below are about the rope config and not
    // about the fixture failing some other check.
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    assert!(supported(plain), "fixture must be Metal-eligible to start");

    assert!(
        supported(phi_like_config()),
        "partial rotary + a non-unit attn_factor must no longer refuse Metal"
    );

    let mut odd = phi_like_config();
    odd.rope_dim = Some(95);
    assert!(!supported(odd), "odd n_rot must keep the model off Metal");
}

/// A `residual_scale` keeps every fused Metal path off the model,
/// through ONE predicate the four eligibility checks share.
///
/// Granite multiplies both branch outputs before every residual add.
/// The fused launches -- dense decode, resident MoE decode, and both
/// prefill stacks -- fold the residual add in on device with no
/// uniform for a multiplier, so a Granite layer served by any of
/// them would be scaled by the host bodies and not by the GPU: the
/// same weights answering differently depending on which backend
/// took the token. That is exactly what `attention_scale` is fenced
/// off for next door, and this repo has already watched the GPU
/// router's eligibility check drift four ways when it was four
/// spellings instead of one.
///
/// Only reachable in a `--features metal` build, which is where the
/// fence exists at all.
#[test]
fn a_residual_scale_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );

    let mut scaled = plain;
    scaled.residual_scale = Some(0.22);
    let d = Decoder::new_random_small(scaled.clone(), 1, 32);
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a residual multiplier no Metal kernel applies must refuse the fused attention"
    );
    assert!(
        !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &scaled, false),
        "...and the prefill dense stack"
    );
    assert!(
        !Decoder::metal_can_serve_model(&scaled, false),
        "the shared predicate is what all four read"
    );
}

/// A `clamp_kqv` keeps every fused Metal path off the model too,
/// through the same predicate.
///
/// The fused launches apply the QKV bias inside their kernels via
/// `AttnExtras` and clamp nothing, while the host bodies clamp
/// after the bias (`decoder/qkv_bias.rs`). A DBRX layer served by a
/// fused launch would therefore run unclamped projections -- the
/// disagreement the residual-scale fence exists for, on a different
/// scalar. Only reachable in a `--features metal` build.
#[test]
fn a_qkv_clamp_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );

    let mut clamped = plain;
    clamped.clamp_kqv = Some(8.0);
    let d = Decoder::new_random_small(clamped.clone(), 1, 32);
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a clamp no Metal kernel applies must refuse the fused attention"
    );
    assert!(
        !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &clamped, false),
        "...and the prefill dense stack"
    );
    assert!(!Decoder::metal_can_serve_model(&clamped, false));
}

/// Per-layer shapes keep every fused Metal path off the model,
/// through the same predicate.
///
/// The fused launches take ONE `n_heads` argument and one
/// `MetalKvBuffers` geometry per run of layers, and the Metal KV
/// plane is sized once from the scalars. A deci or openelm layer
/// served by any of them would be projected and cached at another
/// layer's width -- the disagreement the two fences above exist
/// for, on a shape rather than a scalar. Only reachable in a
/// `--features metal` build.
#[test]
fn per_layer_shapes_keep_the_model_off_every_fused_metal_path() {
    use crate::layer_shapes::{AttnShape, LayerShape, LayerShapes};
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );

    let mut shaped = plain;
    shaped.layer_shapes = LayerShapes::PerLayer(vec![LayerShape {
        attention: AttnShape::Gqa {
            n_heads: shaped.n_heads,
            n_kv_heads: shaped.n_kv_heads,
        },
        ffn_dim: shaped.moe.expert_ffn_dim,
    }]);
    let d = Decoder::new_random_small(shaped.clone(), 1, 32);
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a per-layer table, even one whose single entry agrees with the scalars, must \
         refuse the fused attention: the stacks hold one geometry"
    );
    assert!(
        !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &shaped, false),
        "...and the prefill dense stack"
    );
    assert!(!Decoder::metal_can_serve_model(&shaped, false));
}

/// A per-position attention temperature keeps every fused Metal
/// path off the model, through the same predicate.
///
/// No fused launch takes a per-token Q scale: the host bodies
/// multiply Q by `log(floor(pos / floor) + 1) * scale + 1` after
/// RoPE (`crate::attn_temperature`), and a Ministral-3 layer served
/// by a fused launch would attend at temperature 1 while its
/// neighbours on the host stepped with position -- the same
/// disagreement the three fences above exist for. Only reachable
/// in a `--features metal` build.
#[test]
fn a_per_position_temperature_keeps_the_model_off_every_fused_metal_path() {
    use crate::attn_temperature::AttnTemperature;
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );

    let mut tempered = plain;
    tempered.attn_temperature = Some(AttnTemperature {
        scale: 0.5,
        floor_scale: std::num::NonZeroU32::new(2).unwrap(),
        offset: 0.0,
        unrotated_layers_only: false,
    });
    let d = Decoder::new_random_small(tempered.clone(), 1, 32);
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a per-position Q scale no Metal kernel applies must refuse the fused attention"
    );
    assert!(
        !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &tempered, false),
        "...and the prefill dense stack"
    );
    assert!(!Decoder::metal_can_serve_model(&tempered, false));
}

/// BitNet's two inner norms keep the model off every fused Metal
/// path, through the same predicate, and a layer that carries the
/// attention one off the per-layer fused attention through the
/// exhaustive destructure in `metal_attn_view`.
///
/// No fused kernel norms between the V sum and `wo` or between the
/// activation and `down`; a BitNet layer served by one would skip
/// both norms at full speed (`crate::sub_norms`). Only reachable in
/// a `--features metal` build.
/// A parallel-residual model (`crate::parallel_residual`) stays off
/// every fused Metal path: each of them bakes the pre-FFN norm over
/// the POST-ATTENTION residual into its kernel, and a parallel layer
/// norms the layer INPUT. The model-level flag is what the
/// predicate reads, because the per-layer fact lives on
/// `MoeWeights` and the two config-only callers cannot see it. Only
/// reachable in a `--features metal` build.
/// A learned-position model (`crate::position_embd`) stays off every
/// fused Metal path: the GPU embedding gather has no add, and no
/// stack sees `pos` for it. Only reachable in a `--features metal`
/// build.
/// An ALiBi model (`crate::alibi`) stays off every fused Metal path:
/// no kernel adds a per-key bias to its scores. Only reachable in a
/// `--features metal` build.
#[test]
fn an_alibi_bias_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );
    let mut biased = plain;
    biased.alibi_max_bias = Some(8.0);
    let d = Decoder::new_random_small(biased.clone(), 1, 32);
    assert!(d.alibi_slopes.is_some(), "derived from the config");
    assert!(!d.layer_supports_metal_attn(&d.layers[0]));
    assert!(!Decoder::metal_can_serve_model(&biased, false));
}

#[test]
fn a_learned_position_table_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );
    let mut positioned = plain;
    positioned.learned_positions = true;
    let d = Decoder::new_random_small(positioned.clone(), 1, 32);
    assert!(!d.layer_supports_metal_attn(&d.layers[0]));
    assert!(!Decoder::metal_can_serve_model(&positioned, false));
}

#[test]
fn a_parallel_residual_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );
    let mut parallel = plain;
    parallel.parallel_residual = true;
    let d = Decoder::new_random_small(parallel.clone(), 1, 32);
    assert!(!d.layer_supports_metal_attn(&d.layers[0]));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &parallel,
        false
    ));
    assert!(!Decoder::metal_can_serve_model(&parallel, false));
}

/// Llama 4's two facts (`crate::chunked_swa`,
/// `crate::weightless_qk_norm`) each keep the model off every fused
/// launch: one window per layer, no weightless post-RoPE QK norm.
#[test]
fn a_chunked_window_and_the_weightless_qk_norm_each_keep_the_model_on_the_host() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    assert!(Decoder::metal_can_serve_model(&plain, false));
    let mut chunked = plain.clone();
    chunked.sliding_window = Some(8192);
    chunked.swa_chunked = true;
    assert!(!Decoder::metal_can_serve_model(&chunked, false));
    let mut normed = plain;
    normed.weightless_qk_norm = true;
    assert!(!Decoder::metal_can_serve_model(&normed, false));
}

#[test]
fn the_inner_norms_keep_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );

    let mut sub_normed = plain;
    sub_normed.block_sub_norms = true;
    let d = Decoder::new_random_small(sub_normed.clone(), 1, 32);
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a norm between attention and `wo` that no Metal kernel applies must refuse the \
         fused attention"
    );
    assert!(
        !Decoder::metal_prefill_dense_layer_eligible(&d.layers[0], &sub_normed, false),
        "...and the prefill dense stack"
    );
    assert!(!Decoder::metal_can_serve_model(&sub_normed, false));

    // Independently of the config: the tensor on the layer is enough
    // to lose the per-layer fused attention's view of it.
    let mut d = Decoder::new_random_small(plain_config_with_metal_view(), 1, 32);
    assert!(d.metal_attn_view(&d.layers[0]).is_some());
    d.layers[0].attn.attn_sub_norm = Some(vec![1.0; d.config.hidden_dim]);
    assert!(d.metal_attn_view(&d.layers[0]).is_none());
}

/// A V head width that differs from K's, and a scale after `wo`,
/// each keep the model off every fused Metal path through the same
/// predicate: every fused launch takes ONE head width for its KV
/// buffers, its attention tile and its `wo` fold, and none scales
/// after the fold (`crate::kv_head_dims`, `crate::attn_value_scale`).
/// Only reachable in a `--features metal` build.
#[test]
fn a_split_kv_head_width_or_a_value_scale_keeps_the_model_off_every_fused_metal_path() {
    let plain = plain_config_with_metal_view();
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");

    let mut split = plain.clone();
    split.v_head_dim = Some(plain.head_dim / 2);
    assert!(!Decoder::metal_can_serve_model(&split, false));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &split,
        false
    ));

    let mut scaled = plain;
    scaled.attn_value_scale = Some(0.707);
    assert!(!Decoder::metal_can_serve_model(&scaled, false));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &scaled,
        false
    ));
}

/// A looped model keeps off every fused Metal path: each launch
/// indexes weights and KV buffers with one `l`, and a looped model's
/// logical layers outnumber its weights (`crate::layer_loops`). Only
/// reachable in a `--features metal` build.
#[test]
fn a_layer_loop_keeps_the_model_off_every_fused_metal_path() {
    let plain = plain_config_with_metal_view();
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");
    let mut looped = plain;
    looped.layer_loops = Some(crate::layer_loops::LayerLoops::Repeat {
        n_phys: 1,
        n_loops: 2,
        skip_loop_final_norm: false,
    });
    assert!(!Decoder::metal_can_serve_model(&looped, false));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &looped,
        false
    ));
}

/// Talkie's facts keep the model off every fused Metal path: the
/// skip stream and the per-head scalar QK gain through the model
/// predicate, and a projection gain on a layer through the
/// exhaustive destructure and the dense-FFN predicate
/// (`crate::skip_stream`, `crate::weight_scales`). Only reachable in
/// a `--features metal` build.
#[test]
fn talkie_s_facts_keep_the_model_off_every_fused_metal_path() {
    let plain = plain_config_with_metal_view();
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(d.layer_supports_metal_attn(&d.layers[0]), "the premise");

    let mut skip = plain.clone();
    skip.skip_stream = true;
    assert!(!Decoder::metal_can_serve_model(&skip, false));

    let mut scalar = plain.clone();
    scalar.qk_norm_style = crate::capability::QkNormStyle::PerHeadScalar;
    assert!(!Decoder::metal_can_serve_model(&scalar, false));

    let mut d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(d.metal_attn_view(&d.layers[0]).is_some());
    d.layers[0].attn.o_scale = Some(1.5);
    assert!(d.metal_attn_view(&d.layers[0]).is_none());

    let mut d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &plain,
        false
    ));
    d.layers[0].moe.down_scale = Some(0.5);
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &plain,
        false
    ));
    assert!(!Decoder::layer_supports_metal_dense_ffn(&d.layers[0]));
}

fn plain_config_with_metal_view() -> ModelConfig {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    plain
}

/// An FFN activation no fused kernel spells keeps the model off
/// every fused Metal path, through the same predicate.
///
/// Six launch sites used to derive the kernels' `gelu: bool` as
/// `!is_swiglu()`, which would have run the ungated ReLU-squared FFN
/// (`arcee`) as GELU; `GluAct::fused_kernel_gelu_flag` is `None`
/// there now, and this is the fence that keeps a layer from being
/// half-served. Only reachable in a `--features metal` build.
#[test]
fn an_activation_no_kernel_spells_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    plain.ffn_activation = crate::config::FfnActivation::Swiglu;
    let d = Decoder::new_random_small(plain.clone(), 1, 32);
    assert!(d.layer_supports_metal_attn(&d.layers[0]));

    let mut ungated = plain;
    ungated.ffn_activation = crate::config::FfnActivation::ReluSqr;
    let d = Decoder::new_random_small(ungated.clone(), 1, 32);
    assert!(!d.layer_supports_metal_attn(&d.layers[0]));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &ungated,
        false
    ));
    assert!(!Decoder::metal_can_serve_model(&ungated, false));
    assert_eq!(
        ungated
            .model_ffn_act()
            .and_then(GluAct::fused_kernel_gelu_flag),
        None
    );

    // The two PARAMETERISED activations answer no whole-model
    // activation at all, and the two-width rotary is a fourth thing
    // the fused stacks cannot take; each keeps the model off alone.
    let mut xielu = ungated.clone();
    xielu.ffn_activation =
        crate::config::FfnActivation::Xielu(crate::act_layers::XieluLayers::new(vec![
            frink_moe::XieluParams::from_gguf(
                0.8, 0.8, 0.5, -1e-6
            );
            xielu.n_layers
        ]));
    assert_eq!(xielu.model_ffn_act(), None);
    assert!(!Decoder::metal_can_serve_model(&xielu, false));
    let mut clamped = ungated.clone();
    clamped.ffn_activation =
        crate::config::FfnActivation::SwigluClamped(crate::act_layers::SwigluClamps::new(
            vec![0.0; clamped.n_layers],
            vec![7.0; clamped.n_layers],
            frink_moe::ClampForm::AfterSilu,
        ));
    assert_eq!(clamped.model_ffn_act(), None);
    assert!(!Decoder::metal_can_serve_model(&clamped, false));
    let mut two_widths = ungated;
    two_widths.ffn_activation = crate::config::FfnActivation::Swiglu;
    assert!(Decoder::metal_can_serve_model(&two_widths, false));
    two_widths.sliding_window = Some(4);
    two_widths.swa_layers = crate::swa_layers::SwaLayers::period(2, false);
    two_widths.n_layers = 2;
    two_widths.rope_dim_swa = Some(two_widths.head_dim / 2);
    assert!(two_widths.rope_dim_varies_by_layer());
    assert!(!Decoder::metal_can_serve_model(&two_widths, false));
}

/// An attention output gate or a set of attention sinks keeps THAT
/// LAYER off every fused Metal attention launch, through the one
/// exhaustive destructure in `metal_attn_view`.
///
/// Both sit between the softmax and `wo`, which the fused kernels
/// run with no host round-trip; a launch that ignored either would
/// answer differently from the host bodies for the same weights.
/// The gate is `afmoe` / `laguna` (`crate::attn_gate`); the sinks
/// used to be refused by the gpt-oss NAME, and this pins that they
/// are refused by the TENSOR now, on a model that is not gpt-oss.
/// Only reachable in a `--features metal` build.
#[test]
fn an_output_gate_or_attention_sinks_keep_the_layer_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let mut d = Decoder::new_random_small(plain.clone(), 2, 32);
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );
    assert!(d.metal_attn_view(&d.layers[1]).is_some());

    // A gate on layer 0 only.
    let (n_heads, hidden) = (plain.n_heads, plain.hidden_dim);
    d.layers[0].attn.output_gate = Some(crate::attn_gate::AttnGate {
        proj: WeightMatrix::F32(Tensor::new(
            vec![0.1; n_heads * hidden],
            vec![n_heads, hidden],
        )),
        act: crate::attn_gate::GateAct::Sigmoid,
        width: crate::attn_gate::GateWidth::PerHead,
    });
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a gate no Metal kernel applies must refuse the fused attention"
    );
    assert!(d.metal_attn_view(&d.layers[0]).is_none());
    assert!(
        d.layer_supports_metal_attn(&d.layers[1]),
        "the fence is per layer: the ungated layer is still served"
    );

    // Sinks on layer 1, with `gpt_oss` still `None`.
    d.layers[1].attn.sinks = Some(vec![0.5; n_heads]);
    assert!(d.gpt_oss.is_none());
    assert!(
        !d.layer_supports_metal_attn(&d.layers[1]),
        "sinks are refused by the tensor, not by the gpt-oss name"
    );
    assert!(d.metal_attn_view(&d.layers[1]).is_none());
}

/// A LoRA adapter keeps the WHOLE model off every fused Metal
/// launch, through the shared predicate, and an adapted matrix has
/// no raw-bytes Metal descriptor at all -- so a stack that somehow
/// asked for one anyway could not build it. The delta lives inside
/// `WeightMatrix::Adapted` and is served by that type's methods;
/// the stacks read weight bytes past those methods.
/// Only reachable in a `--features metal` build.
#[test]
fn a_lora_adapter_keeps_the_model_off_every_fused_metal_path() {
    let mut plain = phi_like_config();
    plain.rope_dim = None;
    plain.rope_attn_factor = 1.0;
    let mut d = Decoder::new_random_small(plain.clone(), 2, 32);
    assert!(Decoder::metal_can_serve_model(&plain, false));
    assert!(
        d.layer_supports_metal_attn(&d.layers[0]),
        "the fixture must be Metal-eligible to start, or this proves nothing"
    );
    assert!(Decoder::metal_matvec_launch(&d.layers[0].attn.q_proj).is_some());

    // The fact the predicate reads.
    assert!(!Decoder::metal_can_serve_model(&plain, true));
    assert!(!Decoder::metal_prefill_dense_layer_eligible(
        &d.layers[0],
        &plain,
        true
    ));

    // The variant, on one matrix: no descriptor, so no launch.
    let (rows, cols) = (
        d.layers[0].attn.q_proj.rows(),
        d.layers[0].attn.q_proj.cols(),
    );
    let scale = frink_core::weight_matrix::LoraScale::new(1.0);
    d.layers[0].attn.q_proj.attach_lora(
        frink_core::weight_matrix::LoraDelta::new(
            vec![0.01; 2 * cols],
            vec![0.01; rows * 2],
            2,
            rows,
            cols,
            0.0,
            scale.clone(),
        )
        .unwrap(),
    );
    assert!(Decoder::metal_matvec_launch(&d.layers[0].attn.q_proj).is_none());
    assert!(d.layers[0].attn.q_proj.mul_mm_sg_launch().is_none());
    assert!(
        !d.layer_supports_metal_attn(&d.layers[0]),
        "a layer whose projection has no Metal descriptor is not served"
    );

    // The list, on the decoder: what every eligibility check asks.
    d.lora_adapters.push(crate::lora_attach::LoraAttached {
        path: "x.gguf".into(),
        alpha: 0.0,
        task_name: String::new(),
        prompt_prefix: String::new(),
        scale,
        n_tensors: 1,
    });
    assert!(d.lora_attached());
    assert!(
        !d.layer_supports_metal_attn(&d.layers[1]),
        "the fence is the whole model, not the adapted layer alone"
    );
}

/// A Gemma-3-4B-shaped config: `rope_scaling {linear, factor 8}`
/// folded into the full-attention layers' divisors, nothing on the
/// sliding ones, `sliding_window_pattern = 6` last-dense.
fn gemma3_4b_shaped_config() -> ModelConfig {
    let mut cfg = crate::config::test_dense_fixture();
    cfg.head_dim = 8;
    cfg.rope_layout = crate::config::RopeLayout::Norm;
    cfg.rope_theta = 1_000_000.0;
    cfg.rope_theta_swa = Some(10_000.0);
    cfg.sliding_window = Some(4);
    cfg.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
    cfg.rope_freqs = Some(crate::config::RopeFreqs {
        full: vec![8.0; 4],
        swa: Some(vec![1.0; 4]),
    });
    // One full period, so the run holds five sliding layers and one
    // full-attention layer -- Gemma-3's ratio, and the smallest one
    // that makes `rope_freqs_vary_by_layer` true.
    cfg.n_layers = 6;
    cfg
}

/// What the fused Metal stacks are handed per layer must be BOTH
/// halves of `ModelConfig::layer_rope`, layer by layer.
///
/// `Decoder::metal_stack_needs_per_layer_rope_freqs` used to refuse
/// exactly this config off the fused prefill/decode stacks, because
/// those took one `freq_factors` slice for a whole run beside a
/// per-layer theta -- half the answer varying and half not, which is
/// this repo's dominant bug shape. `LayerRope` carries the pair, and
/// this pins that the decoder fills it from the pair rather than
/// re-deriving either half on its own.
#[test]
fn the_metal_stacks_are_handed_each_layer_s_own_rope_pair() {
    let cfg = gemma3_4b_shaped_config();
    assert!(
        cfg.rope_freqs_vary_by_layer(),
        "fixture must be the shape that used to be refused"
    );
    let decoder = Decoder::new_random_small(cfg, 6, 32);

    for il in 0..decoder.layers.len() {
        let rope = decoder
            .config
            .layer_rope(il)
            .expect("every layer rotates here");
        let sent = decoder
            .metal_layer_rope(il)
            .expect("so every layer is handed a rope");
        assert_eq!(sent.theta, rope.theta, "layer {il} base");
        assert_eq!(sent.freq_factors, rope.freq_factors, "layer {il} divisors");
    }

    // Not vacuous: with `swa_pattern = 6` last-dense, layers 0..=4
    // slide and layer 5 does not, so the run really does hold two
    // different answers.
    let sliding = decoder.metal_layer_rope(0).unwrap();
    let full = decoder.metal_layer_rope(5).unwrap();
    assert_eq!(sliding.freq_factors, Some(&[1.0f32; 4][..]));
    assert_eq!(full.freq_factors, Some(&[8.0f32; 4][..]));
    assert_ne!(
        sliding, full,
        "a run of layers that all rope alike proves nothing here"
    );
}

/// The THIRD half of the answer: a layer llama.cpp does not rotate
/// reaches the Metal stacks as `None`, from the same accessor the
/// CPU bodies read, and `layer_needs_metal_stack` sees it through
/// the same `Option` -- so the per-layer launches, which always
/// rope, can never be handed such a layer.
#[test]
fn an_unrotated_layer_is_handed_no_rope_and_routed_to_the_stack() {
    let mut cfg = gemma3_4b_shaped_config();
    // Gemma-3's own graph rotates everything; borrow its shape and
    // give it EXAONE-4 32B's rule so the full-attention layer 5 is
    // the one that does not rotate.
    cfg.rope_layers = crate::rope_layers::RopeLayers::SlidingOnly;
    let decoder = Decoder::new_random_small(cfg, 6, 32);

    for il in 0..5 {
        assert!(
            decoder.metal_layer_rope(il).is_some(),
            "sliding layer {il} rotates"
        );
    }
    assert_eq!(
        decoder.metal_layer_rope(5),
        None,
        "the full-attention layer does not"
    );
    assert!(
        decoder.layer_needs_metal_stack(&decoder.layers[5], 5),
        "an unrotated layer must go to the fused stack, which implements `None`"
    );
}
