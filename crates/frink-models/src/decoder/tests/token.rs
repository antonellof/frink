//! The single-token contiguous path: shapes, determinism, the KV
//! cache's growth, the expert bodies a layer runs, and expert placement.

use super::*;

/// A GeGLU model's ROUTED experts must run GeGLU.
///
/// `run_ffn_block` used to consult `ffn_activation` only in its dense
/// arm; `combine_ffn_outputs_for_position` and everything under it
/// was unconditionally SwiGLU, so a GeGLU MoE would have produced
/// fluent, wrong logits with nothing in the tree to notice. That is
/// not hypothetical: llama.cpp's `grok` passes `LLM_FFN_GELU` to
/// `build_moe_ffn` (`.scratch/llama.cpp/src/models/grok.cpp`), and
/// `grok` sits on `ArchPath::GenericGqa` in `capability.rs`.
///
/// The reference is written out here in plain loops -- its own GELU
/// and SiLU, not `frink_core`'s -- so it cannot agree with the code
/// under test by sharing its bug. The second assertion is the one
/// that makes this a test rather than a smoke check: the SwiGLU
/// answer must be visibly different, so an implementation that
/// ignores the activation cannot pass.
#[test]
fn a_geglu_moe_layer_runs_geglu_in_its_routed_experts_not_swiglu() {
    let mut cfg = tiny_test_config();
    cfg.ffn_activation = crate::config::FfnActivation::Gelu;
    let decoder = host_decoder(cfg, 2, 8);
    let hidden_dim = decoder.config.hidden_dim;
    let layer = &decoder.layers[1];
    assert!(
        !Decoder::is_dense_layer(layer),
        "this test is about the ROUTED path; layer 1 must be a real MoE layer"
    );

    // Larger than the usual unit inputs on purpose: GELU and SiLU
    // are close near zero, and a reference that cannot tell them
    // apart cannot catch the bug this test exists for.
    let normed2: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 * 0.37).sin() * 12.0)
        .collect();

    let gelu = |x: f32| {
        let t = (0.797_884_6f32 * (x + 0.044_715 * x * x * x)).tanh();
        0.5 * x * (1.0 + t)
    };
    let silu = |x: f32| x / (1.0 + (-x).exp());
    let expert_ref = |ex: &ExpertWeights, f: &dyn Fn(f32) -> f32| -> Vec<f32> {
        let g = ex.gate.apply(&normed2);
        let u = ex.up.apply(&normed2);
        let a: Vec<f32> = g.iter().zip(u.iter()).map(|(&g, &u)| f(g) * u).collect();
        ex.down.apply(&a)
    };

    let ExpertBacking::Resident(experts) = &layer.moe.experts else {
        panic!("new_random_small builds resident experts");
    };
    let router_logits = layer.moe.router.apply(&normed2);
    let decision = Decoder::route_for_layer(layer, &router_logits, &decoder.config);
    let block_ref = |f: &dyn Fn(f32) -> f32| -> Vec<f32> {
        let mut out = vec![0f32; hidden_dim];
        for (&eid, &w) in decision.expert_ids.iter().zip(decision.weights.iter()) {
            for (o, e) in out.iter_mut().zip(expert_ref(&experts[eid], f).iter()) {
                *o += w * e;
            }
        }
        assert!(
            layer.moe.shared_expert_gate.is_none(),
            "tiny_test_config's shared experts are ungated; reference assumes it"
        );
        for shex in &layer.moe.shared_experts {
            for (o, e) in out.iter_mut().zip(expert_ref(shex, f).iter()) {
                *o += e;
            }
        }
        out
    };
    let expected_geglu = block_ref(&gelu);
    let expected_swiglu = block_ref(&silu);

    let got = Decoder::run_ffn_block(
        1,
        layer,
        &normed2,
        &decoder.config,
        hidden_dim,
        None,
        ffn_block::RouterOperand::FfnInput,
    );
    assert_eq!(got.len(), hidden_dim);
    for (i, (a, b)) in got.iter().zip(expected_geglu.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4 * b.abs().max(1.0),
            "routed GeGLU FFN element {i}: got {a}, expected {b}"
        );
    }
    assert!(
        expected_geglu
            .iter()
            .zip(expected_swiglu.iter())
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "GeGLU and SwiGLU must differ measurably on this input, or this test \
         could not detect a routed expert that silently ran SwiGLU"
    );
}

#[test]
fn forward_pass_produces_finite_logits_of_correct_shape() {
    let vocab = 10;
    let decoder = host_decoder(tiny_test_config(), 2, vocab);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let logits = decoder.forward_token(3, 0, &mut caches);
    assert_eq!(logits.len(), vocab);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must not contain NaN/Inf"
    );
}

/// `gpu_vram_budget_bytes` must be a real zero-behavior-change
/// default at `None`, and a *real placement plan that places
/// nothing* (a zero VRAM budget, so `PlacementPlan::from_budget`
/// fits no expert at all) must produce byte-identical output to
/// `None` too -- proving the new plumbing (building a plan,
/// looking up each routed expert's placement, dispatching through
/// `run_expert_placed`) doesn't change results when nothing is
/// actually GPU-placed, without needing real CUDA hardware to
/// check (that hardware-dependent half is
/// `frink-moe`'s/`frink-core`'s own `#[ignore]`d tests).
#[test]
fn gpu_vram_budget_bytes_with_nothing_placed_matches_the_default() {
    let mut decoder = host_decoder(tiny_test_config(), 2, 10);
    let mut caches_default: Vec<KvCache> = decoder.config.new_kv_caches();
    let default_logits = decoder.forward_token(3, 0, &mut caches_default);

    decoder.gpu_vram_budget_bytes = Some(0);
    let mut caches_zero_budget: Vec<KvCache> = decoder.config.new_kv_caches();
    let zero_budget_logits = decoder.forward_token(3, 0, &mut caches_zero_budget);

    assert_eq!(
        default_logits, zero_budget_logits,
        "a placement plan that places nothing on GPU must match the None default exactly"
    );
}

/// Qwen2-MoE's real shared-expert sigmoid gate
/// (`MoeWeights::shared_expert_gate`): exact math check by mutating
/// `layer.moe.shared_expert_gate` in place on an already-built
/// decoder (no need to reconstruct a `LayerWeights`/`MoeWeights`
/// from scratch) and comparing against a hand-derived expectation:
/// the *only* thing the gate changes is the shared experts' own
/// contribution, scaled by `sigmoid(gate . x)` -- so
/// `gated_shared_output == ungated_shared_output * sigmoid_value`
/// exactly, computed independently here via `run_expert` on the
/// same layer's shared expert.
#[test]
fn shared_expert_gate_scales_shared_output_by_sigmoid_of_the_gate_logit() {
    let cfg = tiny_test_config();
    let mut decoder = host_decoder(cfg, 2, 8);
    let hidden_dim = decoder.config.hidden_dim;
    assert_eq!(
        decoder.layers[1].moe.shared_experts.len(),
        1,
        "test assumes tiny_test_config's real MoE layer has exactly one shared expert"
    );

    let normed2: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.37).sin()).collect();
    let gate_vec: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.13 - 0.5).collect();

    // Independently compute what the shared expert alone produces,
    // and what sigmoid(gate . x) should scale it by -- this is the
    // ground truth the gated code path must reproduce exactly.
    let shared_out_raw = run_expert(
        &normed2,
        &decoder.layers[1].moe.shared_experts[0],
        decoder.config.layer_ffn_acts(1).dense,
    );
    let gate_logit: f32 = gate_vec
        .iter()
        .zip(normed2.iter())
        .map(|(g, x)| g * x)
        .sum();
    let gate_value = 1.0 / (1.0 + (-gate_logit).exp());
    let expected_gated_shared: Vec<f32> = shared_out_raw.iter().map(|x| x * gate_value).collect();

    // Run the real FFN combine path twice (gate absent, then
    // present) and recover each run's shared-only contribution by
    // subtracting the routed contribution, which the gate never
    // touches and is identical between the two runs (same router,
    // same experts, same input).
    let router_logits = decoder.layers[1].moe.router.apply(&normed2);
    let ungated_total = Decoder::combine_ffn_outputs_for_position(
        1,
        &decoder.layers[1],
        &normed2,
        &normed2,
        &router_logits,
        &decoder.config,
        hidden_dim,
        None,
    );
    decoder.layers[1].moe.shared_expert_gate = Some(gate_vec);
    let gated_total = Decoder::combine_ffn_outputs_for_position(
        1,
        &decoder.layers[1],
        &normed2,
        &normed2,
        &router_logits,
        &decoder.config,
        hidden_dim,
        None,
    );

    for (i, ((u, g), expected_shared)) in ungated_total
        .iter()
        .zip(gated_total.iter())
        .zip(expected_gated_shared.iter())
        .enumerate()
    {
        let routed_contribution = u - shared_out_raw[i];
        let gated_shared_recovered = g - routed_contribution;
        assert!(
            (gated_shared_recovered - expected_shared).abs() < 1e-4,
            "index {i}: recovered gated shared output {gated_shared_recovered} != expected {expected_shared} (sigmoid({gate_logit})={gate_value})"
        );
    }
}

#[test]
fn kv_cache_grows_by_one_position_per_layer_per_step() {
    let decoder = host_decoder(tiny_test_config(), 3, 5);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    decoder.forward_token(0, 0, &mut caches);
    decoder.forward_token(1, 1, &mut caches);
    decoder.forward_token(2, 2, &mut caches);

    for cache in &caches {
        assert_eq!(cache.positions(), 3);
    }
}

#[test]
fn same_token_same_position_is_deterministic() {
    let decoder = host_decoder(tiny_test_config(), 2, 8);
    let mut caches_a: Vec<KvCache> = decoder.config.new_kv_caches();
    let mut caches_b: Vec<KvCache> = decoder.config.new_kv_caches();

    let out_a = decoder.forward_token(4, 0, &mut caches_a);
    let out_b = decoder.forward_token(4, 0, &mut caches_b);
    assert_eq!(out_a, out_b, "identical input state must yield identical output (no hidden randomness in the forward pass)");
}

#[test]
fn multi_step_decode_stays_finite_across_positions() {
    let decoder = host_decoder(tiny_test_config(), 2, 8);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    for pos in 0..16 {
        let logits = decoder.forward_token(pos % 8, pos, &mut caches);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "position {pos}: logits must stay finite across an extended decode run"
        );
    }
}

#[test]
fn dense_layer_forward_pass_produces_finite_logits_of_correct_shape() {
    let vocab = 10;
    let decoder = host_decoder(tiny_dense_test_config(), 2, vocab);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let logits = decoder.forward_token(3, 0, &mut caches);
    assert_eq!(logits.len(), vocab);
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must not contain NaN/Inf"
    );
}

#[test]
fn dense_layer_fast_path_still_records_expert_zero_activations() {
    // The dense fast path bypasses `route_top_k` entirely, but
    // must still record an activation for expert 0 every step --
    // `MoeWeights::placement_plan` and hotness-based GPU placement
    // depend on this being real for every model shape, not just
    // genuinely-MoE ones.
    let decoder = host_decoder(tiny_dense_test_config(), 1, 8);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    decoder.forward_token(0, 0, &mut caches);
    decoder.forward_token(1, 1, &mut caches);
    decoder.forward_token(2, 2, &mut caches);

    let count =
        decoder.layers[0].moe.activation_counts[0].load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(count, 3);
}

/// `PlacementPlan::from_budget` is
/// real and tested in isolation, but only meaningful once it's fed
/// genuinely observed per-expert activation counts rather than
/// zeros. This proves the full loop: run real forward passes,
/// confirm `MoeWeights::activation_counts` actually reflects what
/// `route_top_k` selected, and confirm `placement_plan` prioritizes
/// the expert that was genuinely hottest -- not just that the
/// budget/size arithmetic works on synthetic inputs.
#[test]
fn placement_plan_reflects_real_observed_expert_activations() {
    let cfg = tiny_test_config(); // 6 experts, top-2 active/token
    let decoder = host_decoder(cfg, 2, 16);
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let n_calls = 20;
    for pos in 0..n_calls {
        decoder.forward_token(pos % 16, pos, &mut caches);
    }

    let layer0 = &decoder.layers[0].moe;
    let counts: Vec<u64> = layer0
        .activation_counts
        .iter()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .collect();
    let total: u64 = counts.iter().sum();
    assert_eq!(
        total,
        (n_calls as u64) * (decoder.config.moe.n_experts_active as u64),
        "total recorded activations must equal calls * experts_active_per_call"
    );

    // Ties are realistic at this small a sample size; break them the
    // same way `PlacementPlan::from_budget` does (lowest index
    // wins), so this assertion can't spuriously fail on a tie that
    // `from_budget` resolves differently than a naive `max_by_key`
    // (which returns the *last* max element) would.
    let hottest_count = *counts.iter().max().unwrap();
    let hottest_idx = counts.iter().position(|&c| c == hottest_count).unwrap();
    assert!(hottest_count > 0);

    // A per-expert resident size big enough for exactly one expert.
    let per_expert_bytes = layer0.expert_bytes(0);
    let plan = layer0.placement_plan(per_expert_bytes as u64);

    assert_eq!(
        plan.placement_for(hottest_idx),
        frink_moe::ExpertPlacement::GpuDevice(0),
        "the genuinely hottest expert (index {hottest_idx}, {hottest_count} activations) \
         must be the one the plan places on GPU when only one expert fits the budget"
    );
}
