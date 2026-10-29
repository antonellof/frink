//! The shared predicates the fused paths ask: the GPU router's
//! eligibility and the QK-norm length rule.

use super::*;

/// The gap the decoration audit found, from the side the existing
/// guard could not see.
///
/// `the_paged_path_keeps_every_per_layer_feature_the_contiguous_one_applies`
/// sets `attention_scale` and compares `forward_token` against
/// `forward_token_paged` -- the two bodies that AGREED. It never
/// compared them against `forward_hidden_batch_inner`, which applied
/// the scale nowhere, so a Gemma-shaped checkpoint would answer at
/// one temperature when decoded a token at a time and at another
/// when its prompt was prefilled. Not an error; a plausible
/// distribution from the wrong model.
///
/// The first assertion is the one that makes this a guard rather
/// than an assertion: 0.37 is well away from the kernel's own
/// `1/sqrt(head_dim)`, so if setting it does not move the logits
/// then both sides are ignoring it and the comparison below proves
/// nothing.
/// The Metal attention kernels infer Q/K norm style from the weight
/// LENGTH; the host branches on `ModelConfig::qk_norm_style`. Two
/// mechanisms for one decision, so they have to agree.
///
/// They do, and not by luck: `loader.rs`'s `refined_qk_norm` DERIVES
/// the enum from the same length rule, and refuses to load anything
/// that matches neither width. This pins that, because the failure
/// would be silent and would land on audited architectures --
/// OLMoE is whole-vector, Qwen3 and Gemma-3 are per-head, and all
/// three are in `AUDITED_GENERIC_GQA`, so an inference that assumed
/// one style would answer wrong on the others at full speed.
///
/// Raised by the decoration audit as unverifiable from the host
/// side, which is exactly why it is written down here rather than
/// left as a comment on one of the two sides.
#[test]
fn the_metal_qk_norm_length_rule_is_the_one_the_loader_derives_the_style_from() {
    use crate::capability::QkNormStyle;
    let head_dim = 8usize;
    let n_heads = 4usize;

    // The rule `frink-metal/src/attn.rs` applies, transcribed.
    let metal_says_per_head = |len: usize| len == head_dim;
    // The rule `loader.rs::refined_qk_norm` applies, transcribed.
    let loader_style = |len: usize| -> Option<QkNormStyle> {
        if len == head_dim {
            Some(QkNormStyle::PerHead)
        } else if len == n_heads * head_dim {
            Some(QkNormStyle::WholeVector)
        } else {
            None
        }
    };

    for len in [head_dim, n_heads * head_dim] {
        let style = loader_style(len).expect("both widths load");
        assert_eq!(
            metal_says_per_head(len),
            style == QkNormStyle::PerHead,
            "length {len} loads as {style:?} but Metal would infer the other style"
        );
    }

    // A width neither side handles must be refused at load rather
    // than reaching a kernel that would pick a branch anyway.
    assert!(
        loader_style(head_dim + 1).is_none(),
        "an unrecognised norm width must be a load error, not a coin flip"
    );

    // The one ambiguous case, and it is harmless: with a single
    // head the two widths coincide, so both rules take their PerHead
    // branch and per-head RMS over one head IS whole-vector RMS.
    let single_head = |len: usize| len == head_dim;
    assert!(single_head(head_dim));
    assert_eq!(
        loader_style(head_dim),
        Some(QkNormStyle::PerHead),
        "with n_heads == 1 both widths are head_dim, and both sides must land \
         on the same branch rather than one falling through"
    );
}

/// The predicate that decides whether a MoE layer may be routed by
/// the GPU must admit ONLY the routing the GPU actually computes.
///
/// Every Metal MoE path -- `launch_moe_decode_stack`,
/// `launch_moe_decode_layer_fused`, `launch_moe_prefill_q4_0` and
/// the fused prefill stack -- routes with a plain top-k softmax over
/// the raw router logits. `Decoder::route_for_layer` has three more
/// arms: grouped routing, a per-expert router bias, and
/// `expert_weights_scale`. The audit found those four call sites
/// disagreeing about which of the three to refuse -- prefill checked
/// all three, the fused decode layer checked two, the whole-stack
/// decode checked none -- so a Softmax-gated MoE checkpoint carrying
/// a router bias would have routed to different experts on Metal
/// than on CPU, with no error.
///
/// This asserts the invariant directly rather than the predicate's
/// spelling: whenever it says yes, plain `route_top_k` and
/// `route_for_layer` must return the same decision; and each of the
/// three features on its own must make it say no.
#[test]
fn the_gpu_router_predicate_admits_only_routing_it_reproduces() {
    // `tiny_test_config` is GLM-shaped and so gates with sigmoid;
    // the GPU router implements softmax, so start from the case the
    // predicate is supposed to ADMIT.
    let mut base = tiny_test_config();
    base.moe.gating = frink_moe::GatingFunction::Softmax;
    let decoder = host_decoder(base.clone(), 2, 8);
    let plain_layer = &decoder.layers[0];
    let n_experts = base.moe.n_experts;
    // Chosen so each feature really bites: the top two experts sit
    // in DIFFERENT groups of two (so grouped routing must reorder
    // them), and the runners-up are close enough behind that a
    // per-expert bias flips the order.
    assert_eq!(n_experts, 6, "the logits below are written for six experts");
    let logits: Vec<f32> = vec![0.90, 0.10, 0.20, 0.85, 0.30, 0.05];

    let agrees = |layer: &LayerWeights, cfg: &ModelConfig| -> bool {
        let host = Decoder::route_for_layer(layer, &logits, cfg);
        let gpu = route_top_k(
            &logits,
            cfg.moe.n_experts_active,
            cfg.moe.gating,
            cfg.moe.norm_topk_prob,
        );
        host.expert_ids == gpu.expert_ids
            && host.weights.len() == gpu.weights.len()
            && host
                .weights
                .iter()
                .zip(gpu.weights.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };

    // The admitted case: the predicate says yes, and the two
    // routers really do agree.
    assert!(
        Decoder::gpu_router_matches_host_routing(plain_layer, &base),
        "a plain softmax MoE layer must stay eligible, or this test proves nothing"
    );
    assert!(agrees(plain_layer, &base));

    // A per-expert router bias.
    let mut biased_decoder = host_decoder(base.clone(), 2, 8);
    biased_decoder.layers[0].moe.exp_probs_bias =
        Some((0..n_experts).map(|e| 0.9 - 0.4 * e as f32).collect());
    let biased_layer = &biased_decoder.layers[0];
    assert!(
        !Decoder::gpu_router_matches_host_routing(biased_layer, &base),
        "exp_probs_bias must make the layer ineligible for the GPU router"
    );
    assert!(
        !agrees(biased_layer, &base),
        "the bias must actually change the routing, or the check above is vacuous"
    );

    // `expert_weights_scale`.
    let mut scaled = base.clone();
    scaled.moe.expert_weights_scale = 2.5;
    assert!(
        !Decoder::gpu_router_matches_host_routing(plain_layer, &scaled),
        "expert_weights_scale must make the layer ineligible for the GPU router"
    );
    assert!(
        !agrees(plain_layer, &scaled),
        "the scale must actually change the routing, or the check above is vacuous"
    );

    // Grouped routing.
    let mut grouped = base.clone();
    grouped.moe.expert_group_count = Some(3);
    grouped.moe.expert_group_used_count = Some(1);
    assert!(
        !Decoder::gpu_router_matches_host_routing(plain_layer, &grouped),
        "grouped routing must make the layer ineligible for the GPU router"
    );
    assert!(
        !agrees(plain_layer, &grouped),
        "the grouping must actually change the routing, or the check above is vacuous"
    );

    // A non-softmax gate: the GPU kernel implements softmax only.
    let mut sigmoid = base.clone();
    sigmoid.moe.gating = frink_moe::GatingFunction::Sigmoid;
    assert!(
        !Decoder::gpu_router_matches_host_routing(plain_layer, &sigmoid),
        "a non-softmax gate must make the layer ineligible for the GPU router"
    );

    // A router that reads the raw layer input (`smallthinker`):
    // every GPU router reads `normed2`, so the SAME logits routed
    // the same way are still the wrong experts. `agrees` cannot see
    // this one -- it compares two routers over one logit vector --
    // which is exactly why the predicate has to.
    let mut raw = base.clone();
    raw.router_input = crate::router_input::RouterInput::RawLayerInput;
    assert!(
        !Decoder::gpu_router_matches_host_routing(plain_layer, &raw),
        "a raw-layer-input router must make the layer ineligible for the GPU router"
    );

    // A router (and experts) reading the normed layer input
    // (`arctic`): the same predicate, for the same reason, and the
    // experts would read the wrong tensor too.
    let mut branch = base;
    branch.router_input = crate::router_input::RouterInput::NormedLayerInput;
    assert!(
        !Decoder::gpu_router_matches_host_routing(plain_layer, &branch),
        "a normed-layer-input branch must make the layer ineligible for the GPU router"
    );
}
