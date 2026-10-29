//! frink-moe: sparse Mixture-of-Experts routing, a shared-expert path,
//! and a CPU/GPU expert-placement scheduler.
//!
//! The placement design mirrors the pattern popularized by ik_llama.cpp
//! (tensor-name-regex overrides deciding which experts live on GPU vs
//! CPU RAM, e.g. `--cpu-moe` / `-ncmoe`) and by llama.cpp's
//! layer-split conventions, adapted here to a config-driven Rust
//! scheduler rather than copied CLI-flag parsing code. See
//! docs/THIRD_PARTY_NOTICES.md.

pub mod dense_bias;
pub mod glu_act;
pub mod selection;

pub use selection::SelectionRule;

use frink_core::weight_matrix::WeightMatrix;

/// Where a given expert's weights currently live. `GpuDevice`-placed
/// experts only actually execute on a GPU under `--features cuda` and/or
/// `--features metal` (see `run_expert_placed`); without a GPU feature
/// the CPU path executes regardless. Device id is meaningful for CUDA;
/// Metal currently uses the system default device and ignores the id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPlacement {
    Cpu,
    GpuDevice(u32),
}

/// Static per-layer MoE configuration. One of these is built per layer
/// from a ModelConfig preset (frink-models).
#[derive(Debug, Clone)]
pub struct MoeLayerConfig {
    pub n_experts: usize,
    pub n_experts_active: usize,
    pub n_shared_experts: usize,
    pub hidden_dim: usize,
    pub expert_ffn_dim: usize,
    /// Which function converts router logits into selection scores.
    /// See `GatingFunction`'s doc comment: this is not a stylistic
    /// choice, it's an evidence-backed architectural detail that
    /// differs by model family.
    pub gating: GatingFunction,
    /// Only meaningful for `GatingFunction::Softmax` (the `Sigmoid` path
    /// has its own separate, always-renormalized convention -- see
    /// `route_top_k_sigmoid`'s doc comment). Whether the top-k selected
    /// experts' softmax weights get renormalized to sum to one after
    /// selection. Mixtral's real routing does this
    /// (`routing_weights /= routing_weights.sum(...)` in its reference
    /// implementation) and it's the right default for any architecture
    /// that doesn't document otherwise -- but it is a real, per-model
    /// choice, not a law of nature: OLMoE's real `config.json` sets
    /// `norm_topk_prob: false`, confirmed against
    /// `OlmoeTopKRouter.forward` in
    /// `transformers/models/olmoe/modeling_olmoe.py` (`router_top_value
    /// /= router_top_value.sum(...)` only runs `if self.norm_topk_prob`)
    /// and against llama.cpp's real hardcoded `build_moe_ffn(..., false,
    /// ..., LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, ...)` call for
    /// `LLM_ARCH_OLMOE` in `src/models/olmoe.cpp` (GGUF carries no
    /// metadata key for this -- it's an architecture-hardcoded fact in
    /// the reference implementation, not something read from the file).
    /// Getting this wrong silently produces a real, wrong generation:
    /// caught by comparing frink's real OLMoE output directly against
    /// llama.cpp loading the identical GGUF file (llama.cpp answered
    /// "Paris" for "the capital of France is"; frink, with this bug,
    /// answered something else entirely).
    pub norm_topk_prob: bool,
    /// Optional DeepSeek-V3 / GLM-family expert grouping
    /// (`expert_group_count` / `expert_group_used_count` in GGUF, i.e.
    /// `n_group` / `topk_group` in the HF configs). `None` means flat
    /// top-k over all experts (Llama / OLMoE / Qwen2-MoE). Grouped
    /// routing is selected at load time; the hot path reads these fields
    /// as data.
    ///
    /// `expert_group_used_count` is the number of *groups* that survive
    /// the group filter -- it is **not** a per-group expert quota. All
    /// `n_experts_active` experts are then chosen by one global top-k
    /// over the surviving groups, so a token may (and normally does) put
    /// several experts in the same group. See
    /// [`route_top_k_grouped_biased`] for the exact rule and for what
    /// reading this field as "experts per group" silently does instead.
    pub expert_group_count: Option<usize>,
    pub expert_group_used_count: Option<usize>,
    /// Where a layer's `exp_probs_b` enters expert SELECTION
    /// ([`SelectionRule`]): after the gating function (DeepSeek-V3, what
    /// `build_moe_ffn` does when it is handed the bias) or on the raw
    /// logits (`kolibri1`). Inert for a layer without the bias.
    pub selection: SelectionRule,
    /// llama.cpp's `expert_weights_scale` hparam (GGUF
    /// `{arch}.expert_weights_scale`, `LLM_KV_EXPERT_WEIGHTS_SCALE`): a
    /// constant every routed expert's combine weight is multiplied by
    /// *after* the optional top-k renormalisation
    /// (`build_moe_ffn`: `weights = ggml_scale(ctx0, weights, w_scale)`).
    /// `1.0` when the checkpoint does not carry the key, which is a real
    /// no-op rather than a guess -- llama.cpp skips the scale for both
    /// `0.0` and `1.0`. DeepSeek-V3-lineage MoE recipes (dots1,
    /// bailingmoe2, hunyuan-moe, …) set it to values like 2.5, and
    /// ignoring it scales every routed contribution wrong.
    pub expert_weights_scale: f32,
    /// The routing weight multiplies the expert's INPUT rather than its
    /// output (`llama-graph.cpp:1947`, `weight_before_ffn`: Llama 4
    /// alone; `frink_models::routed_weight_site`). The host sites
    /// that gather a routed expert's input scale the row and carry a
    /// weight of 1 through the output sum.
    pub routed_weight_before_ffn: bool,
}

/// Router output for one token: which experts fire, and their
/// (already-normalized) combination weights.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Top-k softmax router over per-expert logits, as used by DeepSeek /
/// GLM / Kimi-style MoE layers (a linear gate scores every expert, top-k
/// experts are kept, and their scores are renormalized to sum to one).
/// Which function converts a router's raw per-expert logits into
/// selection scores, before top-k selection and normalization.
///
/// This distinction is not cosmetic: reading ik_llama.cpp's actual
/// GGUF-loading source (`llama-hparams.cpp`) directly showed that
/// DeepSeek-2/3-family models (`LLM_ARCH_DEEPSEEK2`) and newer
/// GLM-MoE-family models (`LLM_ARCH_GLM4_MOE`) both default to
/// **sigmoid** gating with post-selection score normalization, not
/// softmax -- matching DeepSeek-V3's own published technical report,
/// which documents computing per-expert affinity via sigmoid and then
/// normalizing the *selected* experts' scores to sum to one. Only
/// older DeepSeek-2.0/2.5-era models default to softmax. Using softmax
/// unconditionally, which frink did before this was found, would
/// silently produce wrong routing decisions for any model in the
/// DeepSeek-3/GLM4-MoE lineage -- which very plausibly includes
/// DeepSeek V4 Pro and GLM-5.2, both presumed continuations of these
/// architecture families (see docs/MODELS.md for the exact
/// confidence level on this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingFunction {
    /// exp(logit) / sum(exp(selected logits)) -- the older convention.
    Softmax,
    /// sigmoid(logit) for scoring and top-k selection, then the
    /// selected sigmoid scores are renormalized to sum to one -- the
    /// DeepSeek-V3 / GLM4-MoE convention.
    Sigmoid,
    /// `sqrt(softplus(logit))`, i.e. `sqrt(ln(1 + exp(logit)))` --
    /// DeepSeek V4's real MoE scoring function. Confirmed two ways from
    /// llama.cpp PR #24162 (`src/models/deepseek4.cpp`): (1)
    /// `load_arch_hparams` hard-throws
    /// (`"DeepSeek-V4 loader currently expects sqrtsoftplus MoE
    /// scoring"`) unless the GGUF's `expert_gating_func` metadata is
    /// exactly `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`; (2)
    /// `llm_graph_context::build_moe_ffn`'s real scoring switch computes
    /// `probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits))` for that
    /// enum case. This supersedes the earlier sigmoid guess this crate
    /// carried (inherited from the DeepSeek-2/3 lineage), which the real
    /// loader source shows is wrong for V4 specifically.
    SqrtSoftplus,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `sqrt(softplus(x))` = `sqrt(ln(1 + exp(x)))`, computed the numerically
/// stable way (`softplus(x) = max(x, 0) + ln(1 + exp(-|x|))`, avoiding
/// overflow in `exp(x)` for large positive `x`) -- DeepSeek V4's real
/// per-expert MoE scoring function, see [`GatingFunction::SqrtSoftplus`].
fn sqrt_softplus(x: f32) -> f32 {
    let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
    softplus.sqrt()
}

/// Top-k router over per-expert logits, dispatching to softmax, sigmoid,
/// or sqrt-softplus scoring per `gating`. See `GatingFunction`'s doc
/// comment for why this distinction is real and evidence-backed, not a
/// stylistic choice.
pub fn route_top_k(
    logits: &[f32],
    k: usize,
    gating: GatingFunction,
    norm_topk_prob: bool,
) -> RoutingDecision {
    match gating {
        GatingFunction::Softmax => route_top_k_softmax(logits, k, norm_topk_prob),
        GatingFunction::Sigmoid => route_top_k_sigmoid(logits, k, norm_topk_prob),
        GatingFunction::SqrtSoftplus => route_top_k_sqrtsoftplus(logits, k, norm_topk_prob),
    }
}

/// llama.cpp's `build_moe_ffn` selection, with the DeepSeek-V3
/// aux-loss-free bias applied to the **selection score only**.
///
/// Port of `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`),
/// which is where every architecture carrying `exp_probs_b` routes:
///
/// 1. `probs = gating(logits)`;
/// 2. `selection_probs = probs + exp_probs_b` -- the comment in the
///    reference reads "leave probs unbiased as it's later used to get
///    expert weights", which is the whole point: the bias steers *which*
///    experts fire, never *how much* each one counts. That is
///    [`SelectionRule::BiasAfterGating`]; `rule` can instead say
///    [`SelectionRule::BiasBeforeGating`], `logits + exp_probs_b`;
/// 3. top-k over `selection_probs`, weights gathered from `probs`;
/// 4. optional renormalisation of the selected weights (`norm_w`,
///    i.e. `{arch}.expert_weights_norm`);
/// 5. optional constant scale (`w_scale`,
///    i.e. `{arch}.expert_weights_scale`).
///
/// With `bias = None` this is the unbiased routing plus the scale, so a
/// caller does not need a second code path for layers that happen not to
/// carry the tensor.
///
/// This is exactly [`route_top_k_grouped_biased`] with a single expert
/// group, i.e. no group filter at all -- the two share one body so that
/// a checkpoint carrying `exp_probs_b` *and* expert groups cannot end up
/// routed by a second, differently-behaving copy of the same rule.
pub fn route_top_k_biased(
    logits: &[f32],
    bias: Option<&[f32]>,
    rule: SelectionRule,
    k: usize,
    gating: GatingFunction,
    norm_w: bool,
    w_scale: f32,
) -> RoutingDecision {
    route_top_k_grouped_biased(logits, bias, rule, 1, 1, k, gating, norm_w, w_scale)
}

/// Scores each expert group as the **sum of its top two** member
/// scores, keeps the `topk_group` highest-scoring groups, and masks
/// every expert of every other group to `-inf` in place.
///
/// Transcribed from the reference's `_group_limited` (FreeToken
/// `models/glm4_moe/moe.py`, identical copy in `models/glm_moe_dsa/`),
/// which is itself HF's `Glm4MoeMoE` / DeepSeek-V3 group filter:
///
/// ```text
/// group_scores = scores_for_choice.view(m, g, e // g).topk(2, -1)[0].sum(-1)
/// group_idx    = topk(group_scores, topk_group)[1]
/// scores_for_choice.masked_fill(~group_mask, -inf)
/// ```
///
/// Two details are load-bearing. The group score is the top-**two**
/// sum, not the group max: a group holding two good experts must be
/// able to beat a group holding one great expert and nothing else,
/// which is the entire reason the filter exists. And the masking runs
/// on the **biased** selection scores (`probs + exp_probs_b`), so the
/// aux-loss-free bias steers which *groups* survive, not only which
/// experts win inside them.
///
/// `topk_group` is clamped to `1..=n_groups`: a stored zero would mask
/// every expert to `-inf` and leave the following top-k picking experts
/// out of an all-`-inf` array, which is a silently arbitrary routing
/// rather than a loud failure.
///
/// Groups smaller than two experts sum whatever the group has (the
/// reference cannot express this case at all -- `topk(2)` on a
/// one-element group raises -- and no real checkpoint ships it).
fn mask_unselected_groups(selection: &mut [f32], n_groups: usize, topk_group: usize) {
    let group_size = selection.len() / n_groups;
    let topk_group = topk_group.clamp(1, n_groups);

    let mut group_scores: Vec<(usize, f32)> = (0..n_groups)
        .map(|g| {
            let mut members: Vec<f32> = selection[g * group_size..(g + 1) * group_size].to_vec();
            members.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            (g, members.iter().take(2).sum::<f32>())
        })
        .collect();
    // Descending by score, ties broken by group index so the choice is
    // reproducible run to run (`torch.topk(..., sorted=False)` makes no
    // ordering promise, but a router must not be nondeterministic).
    group_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));

    let mut keep = vec![false; n_groups];
    for &(g, _) in group_scores.iter().take(topk_group) {
        keep[g] = true;
    }
    for (g, kept) in keep.iter().enumerate() {
        if !kept {
            for s in selection[g * group_size..(g + 1) * group_size].iter_mut() {
                *s = f32::NEG_INFINITY;
            }
        }
    }
}

/// The real DeepSeek-V3 / GLM-family `n_group` / `topk_group` router:
/// group-limited **selection**, followed by ONE GLOBAL top-k.
///
/// Port of the reference's `Glm4MoeSparseBlock._route` (FreeToken
/// `models/glm4_moe/moe.py:63`, identical in `models/glm_moe_dsa/`),
/// which matches HF `Glm4MoeMoE.route_tokens_to_experts` and
/// llama.cpp's `build_moe_ffn` `n_expert_groups > 1` block:
///
/// 1. `probs = gating(logits)`;
/// 2. `selection = probs + exp_probs_b` (bias steers selection only),
///    or `logits + exp_probs_b` under [`SelectionRule::BiasBeforeGating`]
///    (see [`SelectionRule`] for which graph does which);
/// 3. if `n_groups > 1`, [`mask_unselected_groups`] masks every expert
///    outside the `topk_group` best groups to `-inf` (not under
///    `BiasBeforeGating`, whose graph bypasses the group block);
/// 4. **one global** top-`k` over the surviving `selection` scores --
///    the groups constrain *where* experts may come from, they do not
///    hand out per-group quotas;
/// 5. weights gathered from the **unbiased** `probs` at the chosen ids;
/// 6. optional renormalisation (`norm_w`) and constant scale
///    (`w_scale`).
///
/// The invariant that makes this a different algorithm from "take a
/// fixed number of experts from every group": the number of experts a
/// surviving group contributes is decided by the scores, not by the
/// config. A token whose eight active experts all belong to two hot
/// groups must fire exactly those eight. Taking a quota per group
/// instead spreads the same token's experts one-per-group across all
/// eight groups -- eight *different* experts, a different FFN output,
/// and no error anywhere: the wrong routing is only visible as degraded
/// generation quality. That was frink's real behaviour before this
/// function existed (see the `grouped_routing_concentrates_*` test).
///
/// `n_groups <= 1`, or an expert count that is not a multiple of
/// `n_groups`, skips the group filter entirely and leaves plain flat
/// biased routing -- the shape [`route_top_k_biased`] delegates here
/// with.
#[allow(clippy::too_many_arguments)]
pub fn route_top_k_grouped_biased(
    logits: &[f32],
    bias: Option<&[f32]>,
    rule: SelectionRule,
    n_groups: usize,
    topk_group: usize,
    k: usize,
    gating: GatingFunction,
    norm_w: bool,
    w_scale: f32,
) -> RoutingDecision {
    let probs: Vec<f32> = match gating {
        GatingFunction::Softmax => {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        }
        GatingFunction::Sigmoid => logits.iter().map(|&l| sigmoid(l)).collect(),
        GatingFunction::SqrtSoftplus => logits.iter().map(|&l| sqrt_softplus(l)).collect(),
    };

    let mut selection = rule.selection_scores(logits, &probs, bias);

    if rule.masks_groups()
        && n_groups > 1
        && !selection.is_empty()
        && selection.len().is_multiple_of(n_groups)
    {
        mask_unselected_groups(&mut selection, n_groups, topk_group);
    }

    let mut idx: Vec<usize> = (0..selection.len()).collect();
    // Ties break by expert index: without the group filter exact ties
    // essentially never happen, but every masked-out expert is exactly
    // `-inf`, so a `k` larger than the surviving population would
    // otherwise pick arbitrary losers in unspecified order.
    idx.sort_unstable_by(|&a, &b| {
        selection[b]
            .partial_cmp(&selection[a])
            .unwrap()
            .then(a.cmp(&b))
    });
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| probs[i]).collect();
    if norm_w {
        // ggml clamps the divisor to the smallest normal f16 rather than
        // testing for zero (`ggml_clamp(..., 6.103515625e-5, INFINITY)`),
        // so a degenerate all-zero row yields zeros, not a uniform split.
        let sum = weights.iter().sum::<f32>().max(6.103_515_6e-5);
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    if w_scale != 0.0 && w_scale != 1.0 {
        for w in weights.iter_mut() {
            *w *= w_scale;
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// Bias-free grouped routing: [`route_top_k_grouped_biased`] for the
/// checkpoints that declare `expert_group_count` /
/// `expert_group_used_count` but carry no `exp_probs_b` tensor and no
/// expert-weight scale.
///
/// `topk_group` is GGUF's `expert_group_used_count`, i.e. how many
/// **groups** survive the filter -- not how many experts to take from
/// each group. Reading it the second way is the routing bug this
/// function used to have: see [`route_top_k_grouped_biased`]'s doc
/// comment for the exact failure it produces.
///
/// When `n_groups <= 1`, or when the expert count is not a multiple of
/// `n_groups`, this falls back to flat [`route_top_k`] -- including that
/// function's per-gating conventions (notably `Sigmoid`'s
/// always-renormalize rule, see [`route_top_k_sigmoid`]).
pub fn route_top_k_grouped(
    logits: &[f32],
    n_groups: usize,
    topk_group: usize,
    total_k: usize,
    gating: GatingFunction,
    norm_topk_prob: bool,
) -> RoutingDecision {
    if n_groups <= 1 || !logits.len().is_multiple_of(n_groups) {
        return route_top_k(logits, total_k, gating, norm_topk_prob);
    }
    route_top_k_grouped_biased(
        logits,
        None,
        SelectionRule::BiasAfterGating,
        n_groups,
        topk_group,
        total_k,
        gating,
        norm_topk_prob,
        1.0,
    )
}

/// sqrt-softplus top-k routing, DeepSeek V4's real non-hash-routed MoE
/// layers (`ffn_exp_probs_b` present, i.e. every layer at or past
/// `hash_layer_count`): score every expert with `sqrt(softplus(logit))`
/// (independently per expert, like [`route_top_k_sigmoid`]'s sigmoid --
/// not a joint softmax distribution), pick the top-k by that score, then
/// (if `norm_topk_prob`) renormalize the selected scores to sum to one.
/// See [`GatingFunction::SqrtSoftplus`] for the real citation. DeepSeek
/// V4's real non-hash MoE layers additionally add a learned bias
/// (`ffn_exp_probs_b`) to the *selection* score only -- see
/// [`route_top_k_sqrtsoftplus_with_bias`] for that variant; this plain
/// version is the bias-free building block, analogous to
/// [`route_top_k_sigmoid`] vs [`route_top_k_sigmoid_with_bias`].
pub fn route_top_k_sqrtsoftplus(logits: &[f32], k: usize, norm_topk_prob: bool) -> RoutingDecision {
    let scores: Vec<f32> = logits.iter().map(|&l| sqrt_softplus(l)).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if norm_topk_prob {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// sqrt-softplus top-k routing with a selection-only bias term, mirroring
/// [`route_top_k_sigmoid_with_bias`] but for DeepSeek V4's real
/// [`GatingFunction::SqrtSoftplus`] scoring: selection uses
/// `sqrt(softplus(logit)) + bias[expert]`, but each selected expert's
/// combine *weight* uses the raw, unbiased `sqrt(softplus(logit))`.
/// Weights are renormalized to sum to one (if `k>1` and `renormalize`),
/// then multiplied by `scaling_factor` (DeepSeek V4's real
/// `expert_weights_norm`/`expert_weights_scale` hparams, read directly in
/// `load_arch_hparams`).
pub fn route_top_k_sqrtsoftplus_with_bias(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits.iter().map(|&l| sqrt_softplus(l)).collect();
    let scores_for_choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        scores_for_choice[b]
            .partial_cmp(&scores_for_choice[a])
            .unwrap()
    });
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if k > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// DeepSeek V4's real hash-based first-layer MoE routing: for the first
/// `hash_layer_count` layers, which experts fire is *not* learned
/// top-k/sigmoid/sqrt-softplus selection at all -- it's a direct
/// token-id-to-expert-id lookup table (`ffn_gate_tid2eid`, GGUF shape
/// `[n_expert_used, n_vocab]`; real per-layer dispatch in
/// `src/models/deepseek4.cpp`: `selected_experts =
/// ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens)`, with
/// `exp_probs_b` (the selection-bias tensor) set to `nullptr` for these
/// layers specifically because there is no learned selection to bias --
/// the expert ids are fixed by the table, not chosen by a score).
///
/// The selected experts' *combine weights*, however, are **not** fixed by
/// the table -- `build_moe_ffn` still computes `sqrt(softplus(logits))`
/// from the real per-token router logits (`ffn_gate_inp`) and gathers
/// those scores at the table-provided expert ids, exactly like the
/// weight half of [`route_top_k_sqrtsoftplus_with_bias`] (just with a
/// fixed selection instead of a chosen top-k). `hash_expert_ids` must
/// have exactly the model's real `n_expert_used` length (one lookup-table
/// row for this token's id); `logits` is the full `[n_expert]`-wide
/// router output for this token.
pub fn route_hash(
    hash_expert_ids: &[usize],
    logits: &[f32],
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    let mut weights: Vec<f32> = hash_expert_ids
        .iter()
        .map(|&e| sqrt_softplus(logits[e]))
        .collect();
    if hash_expert_ids.len() > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: hash_expert_ids.to_vec(),
        weights,
    }
}

/// softmax-then-top-k routing (the Mixtral/older-DeepSeek convention:
/// softmax over *every* expert first, then select the top-k of those
/// probabilities -- not "top-k logits, then softmax just those"; the two
/// are mathematically different since softmax's denominator would only
/// sum the selected subset in the latter). `norm_topk_prob` controls
/// whether the selected top-k probabilities are then renormalized to sum
/// to one -- true is the right default for any architecture that doesn't
/// document otherwise (Mixtral does this), but it is a real per-model
/// choice: see `MoeLayerConfig::norm_topk_prob`'s doc comment for why
/// OLMoE specifically needs `false`.
pub fn route_top_k_softmax(logits: &[f32], k: usize, norm_topk_prob: bool) -> RoutingDecision {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| probs[i]).collect();
    if norm_topk_prob {
        let top_sum: f32 = weights.iter().sum();
        for w in weights.iter_mut() {
            *w /= top_sum;
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// Sigmoid top-k routing: score every expert with `sigmoid(logit)`
/// (independently per expert, not a joint softmax distribution), pick
/// the top-k by that score, and renormalize just the selected experts'
/// scores to sum to one WHEN `norm_topk_prob` says so -- llama.cpp's
/// `build_moe_ffn` (`llama-graph.cpp`) applies `norm_w` after either
/// gating function, the same way. DeepSeek-V3 / GLM4-MoE renormalise
/// (`expert_weights_norm = true`); `llama4.cpp:228` passes `false`
/// beside its SIGMOID literal, and until it did this body renormalised
/// UNCONDITIONALLY: every sigmoid row that had reached it declared the
/// key true, so the ignored argument had never changed an answer.
pub fn route_top_k_sigmoid(logits: &[f32], k: usize, norm_topk_prob: bool) -> RoutingDecision {
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if norm_topk_prob {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        } else {
            // Degenerate case (all selected scores are exactly zero,
            // essentially never in practice for a real trained router):
            // fall back to a uniform split rather than dividing by zero.
            weights = vec![1.0 / top.len() as f32; top.len()];
        }
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// Sigmoid top-k routing with a real "aux-loss-free" per-expert bias
/// term added *only* for top-k selection (`topk_method: "noaux_tc"` in
/// Kimi K3's real `config.json`, `KimiMoEGate.forward` in
/// `modeling_kimi_linear.py`, adapted from DeepSeek-V3's own MoE gate --
/// the same convention, not Kimi-specific): the selection scores are
/// `sigmoid(logit) + bias[expert]`, but the *weight* each selected
/// expert's output gets multiplied by uses the raw, unbiased
/// `sigmoid(logit)` -- getting this backwards (biasing the weight
/// itself, not just the selection) would silently skew routed-expert
/// contribution away from what the router actually learned. Weights are
/// renormalized to sum to 1 (if `k>1`) then multiplied by
/// `scaling_factor` (Kimi K3's `routed_scaling_factor`, 1.0 in its real
/// config, i.e. a no-op there, but a real multiplier for any other
/// model using this same convention with a different value).
pub fn route_top_k_sigmoid_with_bias(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    renormalize: bool,
    scaling_factor: f32,
) -> RoutingDecision {
    assert_eq!(logits.len(), bias.len());
    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let scores_for_choice: Vec<f32> = scores.iter().zip(bias.iter()).map(|(s, b)| s + b).collect();

    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        scores_for_choice[b]
            .partial_cmp(&scores_for_choice[a])
            .unwrap()
    });
    let top = &idx[..k.min(idx.len())];

    let mut weights: Vec<f32> = top.iter().map(|&i| scores[i]).collect();
    if k > 1 && renormalize {
        let sum: f32 = weights.iter().sum::<f32>() + 1e-20;
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    for w in weights.iter_mut() {
        *w *= scaling_factor;
    }

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// A CPU/GPU placement plan for a layer's experts.
#[derive(Debug, Clone)]
pub struct PlacementPlan {
    pub default_placement: ExpertPlacement,
    pub overrides: std::collections::HashMap<usize, ExpertPlacement>,
}

impl PlacementPlan {
    pub fn all_cpu(n_experts: usize) -> Self {
        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides: (0..n_experts).map(|i| (i, ExpertPlacement::Cpu)).collect(),
        }
    }

    /// Index-based placeholder: puts the first `n_gpu_resident`
    /// experts on GPU regardless of their actual size or how often
    /// they're activated. Kept only as a trivial fallback for callers
    /// with no real budget/hotness data at all (e.g. `frink smoke`'s
    /// synthetic-weight demo); real deployments should use
    /// `from_budget` instead, which places by measured VRAM budget and
    /// observed expert hotness rather than by index.
    pub fn hot_experts_on_gpu(n_experts: usize, n_gpu_resident: usize) -> Self {
        let mut overrides = std::collections::HashMap::new();
        for i in 0..n_experts.min(n_gpu_resident) {
            overrides.insert(i, ExpertPlacement::GpuDevice(0));
        }
        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides,
        }
    }

    /// Builds a placement plan from a real VRAM budget and each
    /// expert's actual resident byte size (e.g. summed
    /// `WeightMatrix::resident_bytes()` across an expert's gate/up/down
    /// matrices), following ik_llama.cpp's `--cpu-moe`/`--override-tensor`
    /// pattern of deciding CPU-vs-GPU per tensor rather than by a fixed
    /// index cutoff.
    ///
    /// `activation_counts`, if given (one count per expert, e.g.
    /// accumulated from `RoutingDecision::expert_ids` over a real or
    /// representative workload), places the *most frequently activated*
    /// experts on GPU first -- the actual point of expert offload,
    /// since keeping a rarely-used expert resident in VRAM wastes the
    /// budget a hot expert could have used instead. Without observed
    /// counts, falls back to a documented, deterministic policy (index
    /// order) rather than guessing at hotness.
    ///
    /// Greedy, not globally optimal (a smaller-but-colder expert can
    /// still be skipped in favor of trying the next candidate once a
    /// larger higher-priority expert doesn't fit) -- optimal knapsack
    /// packing is not worth the complexity here, and greedy-by-priority
    /// is the same approach real offload tooling uses.
    pub fn from_budget(
        expert_bytes: &[usize],
        activation_counts: Option<&[u64]>,
        vram_budget_bytes: u64,
    ) -> Self {
        let n = expert_bytes.len();
        let mut order: Vec<usize> = (0..n).collect();
        if let Some(counts) = activation_counts {
            if counts.len() == n {
                order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
            }
        }

        let mut overrides = std::collections::HashMap::new();
        let mut used: u64 = 0;
        for idx in order {
            let size = expert_bytes[idx] as u64;
            if size == 0 || used + size > vram_budget_bytes {
                continue;
            }
            used += size;
            overrides.insert(idx, ExpertPlacement::GpuDevice(0));
        }

        PlacementPlan {
            default_placement: ExpertPlacement::Cpu,
            overrides,
        }
    }

    /// A device-placement plan for EVERY layer's routed experts against
    /// ONE shared VRAM budget -- the fix for the real accounting bug
    /// where each layer independently called `from_budget` with the
    /// full budget, so a model with N layers would plan N x the
    /// configured bytes of GPU residency. All `(layer, expert)`
    /// candidates compete in one global priority order (hottest first,
    /// ties broken by layer then expert index for determinism), and a
    /// candidate is only placed on the device if the *global* running
    /// total still fits.
    pub fn plan_layers_against_global_budget(
        expert_bytes_per_layer: &[Vec<usize>],
        activation_counts_per_layer: Option<&[Vec<u64>]>,
        vram_budget_bytes: u64,
    ) -> ResidencyPlan {
        let mut candidates: Vec<(u64, usize, usize)> = Vec::new(); // (count, layer, expert)
        for (l, sizes) in expert_bytes_per_layer.iter().enumerate() {
            for e in 0..sizes.len() {
                let count = activation_counts_per_layer
                    .and_then(|cs| cs.get(l))
                    .and_then(|c| c.get(e))
                    .copied()
                    .unwrap_or(0);
                candidates.push((count, l, e));
            }
        }
        candidates.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));

        let mut layer_overrides: Vec<std::collections::HashMap<usize, ExpertPlacement>> =
            expert_bytes_per_layer
                .iter()
                .map(|_| std::collections::HashMap::new())
                .collect();
        let mut used: u64 = 0;
        for (_, l, e) in candidates {
            let size = expert_bytes_per_layer[l][e] as u64;
            if size == 0 || used + size > vram_budget_bytes {
                continue;
            }
            used += size;
            layer_overrides[l].insert(e, ExpertPlacement::GpuDevice(0));
        }

        ResidencyPlan {
            layer_plans: layer_overrides
                .into_iter()
                .map(|overrides| PlacementPlan {
                    default_placement: ExpertPlacement::Cpu,
                    overrides,
                })
                .collect(),
            device_bytes_planned: used,
            vram_budget_bytes,
        }
    }

    pub fn placement_for(&self, expert_id: usize) -> ExpertPlacement {
        self.overrides
            .get(&expert_id)
            .copied()
            .unwrap_or(self.default_placement)
    }
}

/// The output of `PlacementPlan::plan_layers_against_global_budget`:
/// one per-layer `PlacementPlan` view over a single, globally-accounted
/// device budget. `device_bytes_planned <= vram_budget_bytes` holds by
/// construction across ALL layers combined -- the property the old
/// per-layer planning could not provide.
pub struct ResidencyPlan {
    layer_plans: Vec<PlacementPlan>,
    /// Total bytes this plan places on the device, summed across every
    /// layer.
    pub device_bytes_planned: u64,
    /// The single budget every layer's placements were accounted
    /// against.
    pub vram_budget_bytes: u64,
}

impl ResidencyPlan {
    pub fn layer_plan(&self, layer: usize) -> &PlacementPlan {
        &self.layer_plans[layer]
    }

    pub fn n_layers(&self) -> usize {
        self.layer_plans.len()
    }
}

/// One expert's gate/up/down weight matrices. Each may be plain f32
/// (synthetic/test weights) or still-quantized bytes loaded straight
/// from a GGUF file (real checkpoints) -- `WeightMatrix::apply`
/// dispatches to the right kernel either way.
pub struct ExpertWeights {
    pub gate: WeightMatrix,
    pub up: WeightMatrix,
    pub down: WeightMatrix,
}

/// One expert's gate/up/down bias vectors.
///
/// Kept separate from [`ExpertWeights`] because only the gpt-oss family
/// ships them: every other MoE checkpoint frink loads has bias-free
/// experts, and threading three `Option`s through the thirty
/// `ExpertWeights` construction sites would pay for a feature one
/// architecture uses.
///
/// Lengths are `expert_ffn_dim` for `gate`/`up` and `hidden_dim` for
/// `down`, matching llama.cpp's `ffn_{gate,up}_exps_b` `{n_ff_exp,
/// n_expert}` and `ffn_down_exps_b` `{n_embd, n_expert}`.
#[derive(Debug, Clone, Default)]
pub struct ExpertBias {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// gpt-oss's SwiGLU sigmoid steepness (`llama-graph.cpp`,
/// `LLM_FFN_SWIGLU_OAI_MOE`: `constexpr float alpha = 1.702f`).
pub const SWIGLU_OAI_ALPHA: f32 = 1.702;
/// gpt-oss's SwiGLU clamp (`constexpr float limit = 7.0f`, same site).
pub const SWIGLU_OAI_LIMIT: f32 = 7.0;

/// gpt-oss's clamped SwiGLU.
///
/// Transcribed from `ggml/src/ggml-cpu/ops.cpp`
/// `ggml_compute_forward_swiglu_oai_f32`:
///
/// ```text
/// x = min(gate, limit);
/// y = clamp(up, -limit, limit);
/// out_glu = x / (1 + expf(alpha * -x));
/// dst = out_glu * (y + 1);
/// ```
///
/// Three things differ from ordinary SwiGLU and all three matter: the
/// gate is clamped from *above only*, the sigmoid is scaled by `alpha`
/// rather than being plain `silu`, and the up branch carries a `+1`
/// offset so a zero `up` passes the gate through instead of killing it.
pub fn swiglu_oai(gate: &[f32], up: &[f32], alpha: f32, limit: f32) -> Vec<f32> {
    debug_assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let x = g.min(limit);
            let y = u.clamp(-limit, limit);
            let out_glu = x / (1.0 + (alpha * -x).exp());
            out_glu * (y + 1.0)
        })
        .collect()
}

/// gpt-oss routing: pick the top-`k` experts by their **raw** router
/// logits, then softmax over just those `k`.
///
/// This is llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`
/// (`llama-graph.cpp::build_moe_ffn`), and it is *not* the same as
/// [`route_top_k_softmax`]: there the softmax runs over all `n_expert`
/// logits before selection, so the surviving weights are a slice of the
/// full distribution and sum to less than one; here the normalization
/// happens after selection, so the `k` weights sum to exactly one.
/// Feeding a gpt-oss checkpoint through the ordinary softmax gating
/// picks the same experts but weights them wrong.
pub fn route_top_k_softmax_weight(logits: &[f32], k: usize) -> RoutingDecision {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let top = &idx[..k.min(idx.len())];

    let selected: Vec<f32> = top.iter().map(|&i| logits[i]).collect();
    let max = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = selected.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights = if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        exps
    };

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// [`run_expert`] for gpt-oss: per-expert biases on all three matmuls
/// and [`swiglu_oai`] in place of SwiGLU.
///
/// Deliberately plain CPU with no GPU or shared-activation fast path.
/// gpt-oss is admitted to the CPU graph only, so a fast path here would
/// be a second, unvalidated copy of the math.
pub fn run_expert_oai(
    hidden: &[f32],
    expert: &ExpertWeights,
    bias: &ExpertBias,
    alpha: f32,
    limit: f32,
) -> Vec<f32> {
    let mut gate = expert.gate.apply(hidden);
    let mut up = expert.up.apply(hidden);
    for (x, b) in gate.iter_mut().zip(bias.gate.iter()) {
        *x += b;
    }
    for (x, b) in up.iter_mut().zip(bias.up.iter()) {
        *x += b;
    }
    let activated = swiglu_oai(&gate, &up, alpha, limit);
    let mut out = expert.down.apply(&activated);
    for (x, b) in out.iter_mut().zip(bias.down.iter()) {
        *x += b;
    }
    out
}

pub use dense_bias::{run_expert_biased, DenseBias};
pub use glu_act::{relu_sqr, ClampForm, GluAct, Ungated, XieluParams};

/// Runs one token's hidden state through a single expert's gated FFN.
///
/// `act` is not optional on purpose -- see [`GluAct`].
pub fn run_expert(hidden: &[f32], expert: &ExpertWeights, act: GluAct) -> Vec<f32> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    if act.is_swiglu() {
        // Full SwiGLU on-device (1× upload + 1× download) when dense GPU
        // is on. SwiGLU-only kernel: a GeGLU expert must not take it.
        // The ONE path that does not go through `expert_activated`,
        // which is why `run_expert_sub_normed` cannot reach it.
        if let Some(out) = frink_core::WeightMatrix::apply_gpu_dense_ffn_swiglu(
            &expert.gate,
            &expert.up,
            &expert.down,
            hidden,
        ) {
            return out;
        }
    }
    expert.down.apply(&expert_activated(hidden, expert, act))
}

/// [`run_expert`] with BitNet's `ffn_sub_norm` between the activation
/// and the down projection: `down(rms_norm(act(gate(x), up(x)), w))`
/// (`bitnet.cpp:127-141`; `frink_models::sub_norms`).
///
/// Shares [`expert_activated`] with `run_expert` rather than copying
/// its four arms, and CANNOT take the fused on-device SwiGLU, because
/// that kernel runs the down projection itself with no site for a
/// norm in between: the norm is applied to the vector the kernel
/// never hands back.
pub fn run_expert_sub_normed(
    hidden: &[f32],
    expert: &ExpertWeights,
    act: GluAct,
    sub_norm: &[f32],
    eps: f32,
) -> Vec<f32> {
    let activated = expert_activated(hidden, expert, act);
    debug_assert_eq!(activated.len(), sub_norm.len());
    expert
        .down
        .apply(&frink_core::matmul::rms_norm(&activated, sub_norm, eps))
}

/// The gate/up half of one expert: `act(gate(x), up(x))`, the vector
/// the down projection reads, on the fastest path the weights and the
/// backend admit.
///
/// The one body behind [`run_expert`], [`run_expert_sub_normed`] and
/// `dense_bias::run_expert_biased`; a fifth arm added here reaches all.
fn expert_activated(hidden: &[f32], expert: &ExpertWeights, act: GluAct) -> Vec<f32> {
    // Ungated: `gate` is an alias of `up`, so one projection is the
    // whole FFN input. `apply` is the full matvec dispatcher (int-dot,
    // GPU when placed), so nothing is given up by skipping the pair.
    if let Some(f) = act.ungated() {
        let up = expert.up.apply(hidden);
        return f.apply(&up);
    }
    let (gate, up) = gate_up_projections(hidden, expert);
    act.apply(&gate, &up)
}

/// `gate(x)` and `up(x)` for one gated expert, on the fastest path the
/// weights and the backend admit: one GPU multi-matvec, one shared Q8
/// activation quant, or two host matvecs in parallel. The projections
/// BEFORE any activation, which is where a bias lands
/// (`dense_bias::run_expert_biased`).
pub(crate) fn gate_up_projections(hidden: &[f32], expert: &ExpertWeights) -> (Vec<f32>, Vec<f32>) {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        // Gate and up share `hidden` — one GPU upload / multi-matvec.
        // Activation-agnostic: the combine happens on the host below.
        if let Some(mut outs) =
            frink_core::WeightMatrix::apply_gpu_multi(&[&expert.gate, &expert.up], hidden)
        {
            let up = outs.pop().unwrap();
            let gate = outs.pop().unwrap();
            return (gate, up);
        }
    }
    // Share one Q8 activation quant across gate+up when INT_DOT is on
    // (OLMoE: avoids 2× quantize_activations_q8 per expert).
    if frink_core::weight_matrix::cpu_int_dot_for(frink_core::weight_matrix::IntDotShape::Matvec)
        && hidden.len().is_multiple_of(32)
    {
        let act_q8 = frink_quant::quantize_activations_q8(hidden);
        // gate and up are independent over the same activation, so their
        // parallel regions can overlap instead of running back to back.
        // Decode's deficit is scheduling, not kernels -- see
        // `WeightMatrix::apply_three`, which does this for q/k/v.
        let (g, u) = frink_core::par::join2(
            || expert.gate.apply_cpu_q8(&act_q8),
            || expert.up.apply_cpu_q8(&act_q8),
        );
        if let (Some(gate), Some(up)) = (g, u) {
            return (gate, up);
        }
    }
    frink_core::par::join2(|| expert.gate.apply(hidden), || expert.up.apply(hidden))
}

/// `run_expert`, but actually consulting `placement` instead of always
/// running on CPU -- this is the real execution consequence
/// `PlacementPlan` previously computed but nothing acted on: a
/// `GpuDevice`-placed expert's gate/up/down matvecs go through
/// `WeightMatrix::apply_gpu` (real CUDA and/or Metal kernels for
/// Q8_0/Q4_0/Q4_K/Q5_K/Q6_K), falling straight through to the ordinary
/// CPU path for any matrix `apply_gpu` returns `None` for (an
/// unsupported quant kind, or a real launch failure) -- so this is
/// always correct, never a hard failure, regardless of GPU availability.
///
/// Every call re-uploads each weight matrix to the device from scratch
/// (see `WeightMatrix::apply_gpu`'s doc comment) -- correct, but not
/// yet the persistent-GPU-residency throughput win real expert offload
/// needs; a real, disclosed limit of this round, not overclaimed.
///
/// Without a GPU feature (`cuda` / `metal`) compiled in, this has the
/// exact same signature and always calls `run_expert` (ignoring
/// `placement`), so callers (e.g. `frink-models::decoder::Decoder`)
/// can call it unconditionally regardless of how this crate was built,
/// with correct behavior either way.
#[cfg(any(feature = "cuda", feature = "metal"))]
pub fn run_expert_placed(
    hidden: &[f32],
    expert: &ExpertWeights,
    placement: ExpertPlacement,
    act: GluAct,
) -> Vec<f32> {
    if matches!(placement, ExpertPlacement::GpuDevice(_)) {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        if act.is_swiglu() {
            // SwiGLU-only fused kernel; a GeGLU expert falls through to
            // the split gate/up launches below, which are activation
            // agnostic because the combine runs on the host.
            if let Some(out) = frink_core::WeightMatrix::apply_gpu_dense_ffn_swiglu(
                &expert.gate,
                &expert.up,
                &expert.down,
                hidden,
            ) {
                return out;
            }
        }
        #[cfg(any(feature = "cuda", feature = "metal"))]
        {
            if let Some(mut outs) =
                frink_core::WeightMatrix::apply_gpu_multi(&[&expert.gate, &expert.up], hidden)
            {
                let up = outs.pop().unwrap();
                let gate = outs.pop().unwrap();
                let activated = act.apply(&gate, &up);
                if let Some(down) = expert.down.apply_gpu(&activated) {
                    return down;
                }
                return expert.down.apply(&activated);
            }
        }
        if let Some(gate) = expert.gate.apply_gpu(hidden) {
            if let Some(up) = expert.up.apply_gpu(hidden) {
                let activated = act.apply(&gate, &up);
                if let Some(down) = expert.down.apply_gpu(&activated) {
                    return down;
                }
            }
        }
    }
    run_expert(hidden, expert, act)
}

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub fn run_expert_placed(
    hidden: &[f32],
    expert: &ExpertWeights,
    _placement: ExpertPlacement,
    act: GluAct,
) -> Vec<f32> {
    run_expert(hidden, expert, act)
}

/// Combines routed + shared expert outputs for one token.
pub fn combine_expert_outputs(
    routed_outputs: &[(Vec<f32>, f32)],
    shared_outputs: &[Vec<f32>],
    hidden_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; hidden_dim];
    for (expert_out, weight) in routed_outputs {
        for (o, e) in out.iter_mut().zip(expert_out.iter()) {
            *o += e * weight;
        }
    }
    for shared_out in shared_outputs {
        for (o, e) in out.iter_mut().zip(shared_out.iter()) {
            *o += e;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    /// The property the global planner exists for: with N layers of
    /// identical experts and a budget that fits exactly K experts,
    /// exactly K experts are device-placed across ALL layers combined
    /// -- not K per layer, which is what independent per-layer
    /// `from_budget` calls with the same budget would produce (N*K).
    #[test]
    fn global_budget_cannot_be_multiplied_across_layers() {
        let n_layers = 10;
        let sizes: Vec<Vec<usize>> = (0..n_layers).map(|_| vec![100usize; 4]).collect();
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, None, 250);

        let total_placed: usize = (0..n_layers)
            .map(|l| {
                (0..4)
                    .filter(|&e| plan.layer_plan(l).placement_for(e) != ExpertPlacement::Cpu)
                    .count()
            })
            .sum();
        assert_eq!(
            total_placed, 2,
            "250 bytes fits exactly 2 x 100-byte experts, globally"
        );
        assert_eq!(plan.device_bytes_planned, 200);
        assert!(plan.device_bytes_planned <= plan.vram_budget_bytes);

        // The old shape of the bug, for contrast: per-layer planning
        // with the same budget places 2 experts in EVERY layer.
        let per_layer_total: usize = (0..n_layers)
            .map(|_| {
                let p = PlacementPlan::from_budget(&[100; 4], None, 250);
                (0..4)
                    .filter(|&e| p.placement_for(e) != ExpertPlacement::Cpu)
                    .count()
            })
            .sum();
        assert_eq!(per_layer_total, 20, "per-layer planning overcommits 10x");
    }

    /// Hot experts win device slots across layer boundaries: a single
    /// very hot expert in a late layer beats cold experts in earlier
    /// layers.
    #[test]
    fn global_planning_prioritizes_hotness_across_layers() {
        let sizes: Vec<Vec<usize>> = (0..3).map(|_| vec![100usize; 2]).collect();
        let mut counts: Vec<Vec<u64>> = (0..3).map(|_| vec![0u64; 2]).collect();
        counts[2][1] = 50; // the only hot expert lives in the last layer
        counts[0][0] = 10;
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, Some(&counts), 200);

        assert_eq!(
            plan.layer_plan(2).placement_for(1),
            ExpertPlacement::GpuDevice(0),
            "hottest expert (layer 2) must win a slot"
        );
        assert_eq!(
            plan.layer_plan(0).placement_for(0),
            ExpertPlacement::GpuDevice(0),
            "second-hottest expert (layer 0) takes the remaining slot"
        );
        assert_eq!(plan.device_bytes_planned, 200);
    }

    /// Zero budget places nothing anywhere; empty (dense) layers are
    /// legal and contribute no candidates.
    #[test]
    fn global_planning_handles_zero_budget_and_dense_layers() {
        let sizes = vec![Vec::new(), vec![100usize; 3], Vec::new()];
        let plan = PlacementPlan::plan_layers_against_global_budget(&sizes, None, 0);
        assert_eq!(plan.device_bytes_planned, 0);
        assert_eq!(plan.n_layers(), 3);
        for e in 0..3 {
            assert_eq!(plan.layer_plan(1).placement_for(e), ExpertPlacement::Cpu);
        }
    }

    use super::*;

    #[test]
    fn top_k_selects_highest_scoring_experts() {
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        assert_eq!(decision.expert_ids, vec![1, 3]);
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(decision.weights[0] > decision.weights[1]);
    }

    #[test]
    fn top_k_weights_always_sum_to_one_regardless_of_k() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::Softmax, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "k={k} sum={sum}");
        }
    }

    /// `norm_topk_prob: false` -- OLMoE's real convention (see
    /// `MoeLayerConfig::norm_topk_prob`'s doc comment). Golden values
    /// hand-computed independently: full softmax over all 8 logits
    /// (sum of exp(l_i - 8) = 1.5814460129), then the raw (un-renormalized)
    /// probabilities of the top-3 selected experts (indices 7, 6, 5 --
    /// logits 8, 7, 6). This is the exact bug that was silently producing
    /// wrong OLMoE output: the old code could only ever compute a
    /// top-k-local softmax (mathematically identical to
    /// always-renormalize), with no way to recover the un-renormalized
    /// probability relative to *all* experts.
    #[test]
    fn norm_topk_prob_false_uses_raw_full_softmax_probability_not_renormalized() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let decision = route_top_k(&logits, 3, GatingFunction::Softmax, false);

        assert_eq!(decision.expert_ids, vec![7, 6, 5]);

        let expected = [0.6323223_f32, 0.2326232, 0.0855683];
        for (got, want) in decision.weights.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-4, "got={got} want={want}");
        }

        let sum: f32 = decision.weights.iter().sum();
        assert!(
            (sum - 0.9505138).abs() < 1e-4,
            "raw top-3 probability mass should be < 1 (it's a subset of a full 8-way softmax), got sum={sum}"
        );

        // Selecting the same experts with norm_topk_prob=true must
        // renormalize to the exact same values divided by that sum --
        // proving the two modes agree on *which* experts fire and differ
        // only in the final weight scaling.
        let normalized = route_top_k(&logits, 3, GatingFunction::Softmax, true);
        assert_eq!(normalized.expert_ids, decision.expert_ids);
        for (raw, norm) in decision.weights.iter().zip(normalized.weights.iter()) {
            assert!(
                (raw / sum - norm).abs() < 1e-4,
                "raw={raw} sum={sum} normalized={norm}"
            );
        }
    }

    #[test]
    fn sigmoid_gating_selects_same_top_experts_as_softmax_for_monotonic_logits() {
        // Sigmoid is monotonic in its input, so for a given set of
        // logits, top-k-by-sigmoid-score must select the exact same
        // expert ids as top-k-by-raw-logit (sigmoid just changes the
        // *weights*, not which experts are chosen).
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sigmoid_decision = route_top_k(&logits, 2, GatingFunction::Sigmoid, true);
        assert_eq!(softmax_decision.expert_ids, sigmoid_decision.expert_ids);
    }

    #[test]
    fn sigmoid_gating_weights_sum_to_one() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::Sigmoid, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "k={k} sum={sum}");
        }
    }

    /// `norm_w = false` under sigmoid gating (`llama4.cpp:228`): the
    /// selected experts keep their raw sigmoid scores. This body used
    /// to renormalise whatever the flag said.
    #[test]
    fn sigmoid_gating_without_renormalisation_keeps_the_raw_scores() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        let decision = route_top_k(&logits, 2, GatingFunction::Sigmoid, false);
        assert_eq!(decision.expert_ids, vec![5, 2]);
        assert_eq!(decision.weights, vec![sigmoid(4.0), sigmoid(3.0)]);
        let one = route_top_k(&logits, 1, GatingFunction::Sigmoid, false);
        assert_eq!(one.weights, vec![sigmoid(4.0)], "top-1 is NOT 1.0");
    }

    #[test]
    fn bias_only_affects_selection_not_the_final_weight_value() {
        // Expert 0 has the lower raw score but a large positive bias, so
        // biased selection must pick it over expert 1 -- but the WEIGHT
        // it ends up with must be its raw (unbiased) sigmoid score, not
        // score+bias. Getting this backwards would silently make a
        // barely-selected expert dominate the combine.
        let logits = vec![0.1, 2.0];
        let bias = vec![10.0, 0.0];
        let decision = route_top_k_sigmoid_with_bias(&logits, &bias, 1, true, 1.0);
        assert_eq!(decision.expert_ids, vec![0]);
        // k=1 -> renormalization is a no-op (single weight / itself = 1,
        // scaled by 1.0), so the weight is just sigmoid(0.1), not 1.0.
        assert!((decision.weights[0] - sigmoid(0.1)).abs() < 1e-5);
    }

    #[test]
    fn without_bias_selection_falls_back_to_plain_sigmoid_top_k() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        let zero_bias = vec![0.0; logits.len()];
        let biased = route_top_k_sigmoid_with_bias(&logits, &zero_bias, 3, true, 1.0);
        let plain = route_top_k(&logits, 3, GatingFunction::Sigmoid, true);
        assert_eq!(biased.expert_ids, plain.expert_ids);
        for (a, b) in biased.weights.iter().zip(plain.weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn scaling_factor_multiplies_every_weight() {
        let logits = vec![1.0, 2.0, 3.0];
        let bias = vec![0.0; 3];
        let unscaled = route_top_k_sigmoid_with_bias(&logits, &bias, 2, true, 1.0);
        let scaled = route_top_k_sigmoid_with_bias(&logits, &bias, 2, true, 2.5);
        for (u, s) in unscaled.weights.iter().zip(scaled.weights.iter()) {
            assert!((u * 2.5 - s).abs() < 1e-5);
        }
    }

    #[test]
    fn sigmoid_and_softmax_weights_differ_for_the_same_logits() {
        // The whole point of the distinction: sigmoid scores each
        // expert independently (not as a joint distribution), so the
        // relative weighting between two selected experts differs from
        // softmax's, even though both sum to one and pick the same
        // experts. If this test ever fails by finding the two paths
        // identical, something has collapsed the sigmoid path back
        // into softmax.
        let logits = vec![3.0, 1.0, -2.0, 0.5];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sigmoid_decision = route_top_k(&logits, 2, GatingFunction::Sigmoid, true);
        assert!(
            (softmax_decision.weights[0] - sigmoid_decision.weights[0]).abs() > 1e-3,
            "softmax and sigmoid gating should generally produce different weight splits for the same logits"
        );
    }

    #[test]
    fn sqrt_softplus_matches_hand_computed_values_at_zero_and_positive_logit() {
        // softplus(0) = ln(2), sqrt(ln(2)) -- exact closed form, not just a
        // property check, to pin the real DeepSeek V4 formula
        // (sqrt(softplus(x)), not e.g. softplus(sqrt(x)) or sqrt(sigmoid)).
        assert!((sqrt_softplus(0.0) - 2.0_f32.ln().sqrt()).abs() < 1e-6);
        // softplus(x) -> x for large positive x, so sqrt_softplus(x) -> sqrt(x).
        assert!((sqrt_softplus(20.0) - 20.0_f32.sqrt()).abs() < 1e-3);
    }

    /// Group-limited routing CONCENTRATES; it does not spread.
    ///
    /// This test replaces one that asserted the opposite -- that
    /// `total_k = 2` over two groups keeps "both group winners" -- which
    /// was the shape of the bug, not a property of the rule. The real
    /// DeepSeek-V3 / GLM `n_group`/`topk_group` router scores each group
    /// by the sum of its top-2 members, keeps the `topk_group` best
    /// groups, masks every expert in the rest to `-inf`, and then runs
    /// ONE GLOBAL top-k over what survives. With `topk_group = 1` only
    /// group 0 survives, so both selected experts must come from it and
    /// expert 3 must NOT fire.
    ///
    /// It therefore fails against the previous implementation, which
    /// took `k_per_group` from every group and truncated afterwards.
    #[test]
    fn group_limited_routing_concentrates_into_the_surviving_groups() {
        // 4 experts, 2 groups of 2. Group 0 holds the two best scores.
        let logits = vec![5.0, 4.5, 0.2, 0.1];
        let d = route_top_k_grouped(&logits, 2, 1, 2, GatingFunction::Softmax, true);
        assert_eq!(d.expert_ids.len(), 2);
        let mut ids = d.expert_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1], "both experts must come from group 0");
        let sum: f32 = d.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    /// The group score is the sum of a group's top TWO, not its best
    /// single member. A group with one spike and nothing behind it loses
    /// to a group with two strong members.
    #[test]
    fn a_group_is_scored_by_its_top_two_not_by_its_best_member() {
        // Group 0: one spike and a dead expert, carrying 0.426 of the
        // softmax mass between them. Group 1: two solid members
        // carrying 0.574 together, neither of which beats the spike on
        // its own. Scoring by best member picks group 0; scoring by
        // top-2 sum picks group 1, which is the reference rule.
        let logits = vec![1.0, -20.0, 0.7, 0.5];
        let d = route_top_k_grouped(&logits, 2, 1, 2, GatingFunction::Softmax, true);
        let mut ids = d.expert_ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 3], "the two-strong-members group wins");
    }

    #[test]
    fn sqrtsoftplus_gating_selects_same_top_experts_as_softmax_for_monotonic_logits() {
        // sqrt(softplus(x)) is monotonically increasing in x (both sqrt
        // and softplus are), so top-k-by-score must agree with top-k by
        // raw logit on *which* experts fire, same reasoning as the
        // sigmoid monotonicity test above.
        let logits = vec![0.1, 5.0, 0.2, 3.0, -1.0];
        let softmax_decision = route_top_k(&logits, 2, GatingFunction::Softmax, true);
        let sqrtsoftplus_decision = route_top_k(&logits, 2, GatingFunction::SqrtSoftplus, true);
        assert_eq!(
            softmax_decision.expert_ids,
            sqrtsoftplus_decision.expert_ids
        );
    }

    #[test]
    fn sqrtsoftplus_weights_sum_to_one_when_normalized() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        for k in 1..=8 {
            let decision = route_top_k(&logits, k, GatingFunction::SqrtSoftplus, true);
            let sum: f32 = decision.weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "k={k} sum={sum}");
        }
    }

    #[test]
    fn sqrtsoftplus_bias_only_affects_selection_not_the_final_weight_value() {
        // Same structure as `bias_only_affects_selection_not_the_final_weight_value`
        // but for the sqrt-softplus scoring function DeepSeek V4's real
        // non-hash MoE layers use.
        let logits = vec![0.1, 2.0];
        let bias = vec![10.0, 0.0];
        let decision = route_top_k_sqrtsoftplus_with_bias(&logits, &bias, 1, true, 1.0);
        assert_eq!(decision.expert_ids, vec![0]);
        assert!((decision.weights[0] - sqrt_softplus(0.1)).abs() < 1e-5);
    }

    #[test]
    fn sqrtsoftplus_without_bias_selection_falls_back_to_plain_top_k() {
        let logits = vec![-2.0, 0.5, 3.0, 1.2, -0.3, 4.0, 0.0, -1.5];
        let zero_bias = vec![0.0; logits.len()];
        let biased = route_top_k_sqrtsoftplus_with_bias(&logits, &zero_bias, 3, true, 1.0);
        let plain = route_top_k(&logits, 3, GatingFunction::SqrtSoftplus, true);
        assert_eq!(biased.expert_ids, plain.expert_ids);
        for (a, b) in biased.weights.iter().zip(plain.weights.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn hash_routing_uses_the_fixed_table_ids_regardless_of_logit_ranking() {
        // Expert 0 has by far the highest logit, but the real mechanism
        // never looks at the router's ranking to choose experts for a
        // hash-routed layer -- the table says [2, 1], so that's what
        // fires, full stop.
        let logits = vec![100.0, 1.0, 0.5, -3.0];
        let hash_expert_ids = vec![2usize, 1usize];
        let decision = route_hash(&hash_expert_ids, &logits, true, 1.0);
        assert_eq!(decision.expert_ids, vec![2, 1]);
    }

    #[test]
    fn hash_routing_weights_come_from_the_real_router_logits_not_a_fixed_split() {
        // The table fixes *which* experts fire, but their relative
        // combine weight still comes from sqrt(softplus(logit)) gathered
        // at those ids -- not a uniform 1/n split. Expert 2's logit (3.0)
        // is much larger than expert 1's (0.1), so its weight must
        // dominate even though both were unconditionally selected.
        let logits = vec![-5.0, 0.1, 3.0, -5.0];
        let hash_expert_ids = vec![2usize, 1usize];
        let decision = route_hash(&hash_expert_ids, &logits, true, 1.0);
        assert!(decision.weights[0] > decision.weights[1]);
        let sum: f32 = decision.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let expected0 = sqrt_softplus(3.0) / (sqrt_softplus(3.0) + sqrt_softplus(0.1));
        assert!((decision.weights[0] - expected0).abs() < 1e-5);
    }

    #[test]
    fn hash_routing_scaling_factor_multiplies_every_weight() {
        let logits = vec![1.0, 2.0, 3.0];
        let hash_expert_ids = vec![0usize, 2usize];
        let unscaled = route_hash(&hash_expert_ids, &logits, true, 1.0);
        let scaled = route_hash(&hash_expert_ids, &logits, true, 2.5);
        for (u, s) in unscaled.weights.iter().zip(scaled.weights.iter()) {
            assert!((u * 2.5 - s).abs() < 1e-5);
        }
    }

    #[test]
    fn placement_plan_defaults_to_cpu_for_unlisted_experts() {
        let plan = PlacementPlan::hot_experts_on_gpu(256, 8);
        assert_eq!(plan.placement_for(0), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(7), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(8), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(255), ExpertPlacement::Cpu);
    }

    #[test]
    fn all_cpu_plan_never_returns_gpu() {
        let plan = PlacementPlan::all_cpu(64);
        for i in 0..64 {
            assert_eq!(plan.placement_for(i), ExpertPlacement::Cpu);
        }
    }

    #[test]
    fn from_budget_fits_as_many_experts_as_the_vram_budget_allows() {
        // 4 experts, 100 bytes each: a 250-byte budget fits exactly 2.
        let sizes = vec![100usize, 100, 100, 100];
        let plan = PlacementPlan::from_budget(&sizes, None, 250);
        let on_gpu = (0..4)
            .filter(|&i| plan.placement_for(i) == ExpertPlacement::GpuDevice(0))
            .count();
        assert_eq!(on_gpu, 2);
    }

    #[test]
    fn from_budget_prioritizes_the_most_frequently_activated_experts() {
        // Expert 2 is by far the hottest but is neither first nor
        // largest -- a real budget-aware plan must still pick it first.
        let sizes = vec![50usize, 50, 50, 50];
        let counts = vec![1u64, 2, 100, 3];
        // Budget for exactly one expert.
        let plan = PlacementPlan::from_budget(&sizes, Some(&counts), 50);
        assert_eq!(
            plan.placement_for(2),
            ExpertPlacement::GpuDevice(0),
            "the hottest expert (index 2) must be the one placed on GPU"
        );
        assert_eq!(plan.placement_for(0), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(1), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(3), ExpertPlacement::Cpu);
    }

    #[test]
    fn from_budget_skips_an_expert_that_does_not_fit_and_tries_the_next() {
        // Expert 0 is too big for the budget alone; experts 1 and 2
        // together fit and should both be placed.
        let sizes = vec![200usize, 60, 60];
        let plan = PlacementPlan::from_budget(&sizes, None, 120);
        assert_eq!(plan.placement_for(0), ExpertPlacement::Cpu);
        assert_eq!(plan.placement_for(1), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(2), ExpertPlacement::GpuDevice(0));
    }

    #[test]
    fn from_budget_with_zero_vram_places_nothing_on_gpu() {
        let sizes = vec![10usize, 20, 30];
        let plan = PlacementPlan::from_budget(&sizes, None, 0);
        for i in 0..3 {
            assert_eq!(plan.placement_for(i), ExpertPlacement::Cpu);
        }
    }

    #[test]
    fn from_budget_ignores_mismatched_activation_counts_length_rather_than_panicking() {
        let sizes = vec![10usize, 10];
        let counts = vec![1u64]; // wrong length
        let plan = PlacementPlan::from_budget(&sizes, Some(&counts), 100);
        // Falls back to index order; both fit within the budget either way.
        assert_eq!(plan.placement_for(0), ExpertPlacement::GpuDevice(0));
        assert_eq!(plan.placement_for(1), ExpertPlacement::GpuDevice(0));
    }

    #[test]
    fn combine_expert_outputs_weights_routed_and_adds_shared() {
        let routed = vec![(vec![2.0, 2.0], 0.5), (vec![4.0, 4.0], 0.5)];
        let shared = vec![vec![1.0, 1.0]];
        let out = combine_expert_outputs(&routed, &shared, 2);
        assert_eq!(out, vec![4.0, 4.0]);
    }

    #[test]
    fn run_expert_produces_correct_output_dimension() {
        use frink_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 3;
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(
                vec![0.1; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            up: WeightMatrix::F32(Tensor::new(
                vec![0.2; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            down: WeightMatrix::F32(Tensor::new(
                vec![0.3; hidden_dim * ffn_dim],
                vec![hidden_dim, ffn_dim],
            )),
        };
        let hidden = vec![1.0, -1.0, 0.5, 0.5];
        let out = run_expert(&hidden, &expert, GluAct::Swiglu);
        assert_eq!(out.len(), hidden_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// The aliasing invariant `GluAct::ReluSqr` rests on: with `gate`
    /// the same matrix as `up`, the `combine` path (the one every
    /// routed/placed/batched site runs) and the ungated shortcut
    /// (`relu(up)^2`, the one `run_expert` takes) are the same FFN, and
    /// so is `GluAct::Reglu` on that pair -- `relu(up) * up` IS
    /// `relu(up)^2` -- which is why one variant used to serve both and
    /// why the SmallThinker fixture, whose gate is NOT `up`, was the
    /// first thing that could see the difference. Neither is SwiGLU.
    ///
    /// If a later change makes `reglu` compute something other than
    /// `relu(gate) * up`, or `relu_sqr` something other than
    /// `relu(up)^2`, the two disagree here before any model does.
    #[test]
    fn relu_sqr_with_gate_aliased_to_up_agrees_with_reglu() {
        use frink_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 6;
        let u: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| (i as f32 * 0.9).cos() * 2.0)
            .collect();
        let d: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let up = WeightMatrix::F32(Tensor::new(u.clone(), vec![ffn_dim, hidden_dim]));
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(u, vec![ffn_dim, hidden_dim])),
            up,
            down: WeightMatrix::F32(Tensor::new(d, vec![hidden_dim, ffn_dim])),
        };
        let hidden = vec![1.0, -0.5, 0.25, 2.0];
        // Scalar reference, written independently of both paths.
        let up_out = expert.up.apply(&hidden);
        let want = expert.down.apply(
            &up_out
                .iter()
                .map(|&x| if x > 0.0 { x * x } else { 0.0 })
                .collect::<Vec<_>>(),
        );
        assert!(
            up_out.iter().any(|&x| x < 0.0) && up_out.iter().any(|&x| x > 0.0),
            "the projection must cross zero or relu is invisible: {up_out:?}"
        );
        let shortcut = run_expert(&hidden, &expert, GluAct::ReluSqr);
        let gated_out = expert.gate.apply(&hidden);
        let gated = expert.down.apply(&GluAct::Reglu.apply(&gated_out, &up_out));
        let reglu_expert = run_expert(&hidden, &expert, GluAct::Reglu);
        let slotted: Vec<f32> = gated_out
            .iter()
            .zip(up_out.iter())
            .map(|(g, u)| GluAct::ReluSqr.combine(*g, *u))
            .collect();
        let slotted = expert.down.apply(&slotted);
        for (name, got) in [
            ("run_expert", &shortcut),
            ("reglu apply on the aliased pair", &gated),
            ("reglu run_expert on the aliased pair", &reglu_expert),
            ("combine", &slotted),
        ] {
            for (a, b) in got.iter().zip(want.iter()) {
                assert!((a - b).abs() < 1e-6, "{name}: {got:?} vs {want:?}");
            }
        }
        let swiglu = run_expert(&hidden, &expert, GluAct::Swiglu);
        assert!(
            swiglu
                .iter()
                .zip(want.iter())
                .any(|(a, b)| (a - b).abs() > 1e-3),
            "SwiGLU on the aliased pair must differ, or the suite cannot see the activation"
        );
        assert_eq!(GluAct::Reglu.fused_kernel_gelu_flag(), None);
        assert_eq!(GluAct::ReluSqr.fused_kernel_gelu_flag(), None);
        assert_eq!(GluAct::Swiglu.fused_kernel_gelu_flag(), Some(false));
        assert_eq!(GluAct::Geglu.fused_kernel_gelu_flag(), Some(true));
        assert!(GluAct::Swiglu.ungated().is_none() && GluAct::Geglu.ungated().is_none());
    }

    /// A GeGLU expert must compute `gelu(gate) * up`, not SwiGLU.
    ///
    /// Checked against an independently written scalar reference, and
    /// separately asserted to DIFFER from the SwiGLU result: without the
    /// second half, an implementation that ignored `act` entirely and
    /// always ran SwiGLU could still pass a loose tolerance check on
    /// weights where the two activations happen to be close.
    #[test]
    fn geglu_expert_computes_gelu_not_silu() {
        use frink_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 3;
        let g: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| (i as f32 * 0.7).sin() * 3.0)
            .collect();
        let u: Vec<f32> = (0..ffn_dim * hidden_dim)
            .map(|i| (i as f32 * 0.31).cos())
            .collect();
        let d: Vec<f32> = (0..hidden_dim * ffn_dim)
            .map(|i| (i as f32 * 0.19).sin() * 0.5)
            .collect();
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(g.clone(), vec![ffn_dim, hidden_dim])),
            up: WeightMatrix::F32(Tensor::new(u.clone(), vec![ffn_dim, hidden_dim])),
            down: WeightMatrix::F32(Tensor::new(d.clone(), vec![hidden_dim, ffn_dim])),
        };
        let hidden = vec![1.0, -1.0, 0.5, 0.25];

        // Independent reference: three plain loops, tanh-approx GELU
        // spelled out rather than borrowed from frink-core.
        let row = |w: &[f32], cols: usize, r: usize, x: &[f32]| -> f32 {
            (0..cols).map(|c| w[r * cols + c] * x[c]).sum()
        };
        let mut activated = vec![0f32; ffn_dim];
        for (r, a) in activated.iter_mut().enumerate() {
            let gv = row(&g, hidden_dim, r, &hidden);
            let uv = row(&u, hidden_dim, r, &hidden);
            let t = (0.797_884_6f32 * (gv + 0.044_715 * gv * gv * gv)).tanh();
            *a = 0.5 * gv * (1.0 + t) * uv;
        }
        let expected: Vec<f32> = (0..hidden_dim)
            .map(|r| row(&d, ffn_dim, r, &activated))
            .collect();

        let got = run_expert(&hidden, &expert, GluAct::Geglu);
        assert_eq!(got.len(), hidden_dim);
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "GeGLU expert row {i}: got {a}, expected {b}"
            );
        }

        let swiglu_out = run_expert(&hidden, &expert, GluAct::Swiglu);
        assert!(
            got.iter()
                .zip(swiglu_out.iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "GeGLU and SwiGLU must not agree on these weights -- if they do, \
             this test cannot detect a routed expert that silently ran SwiGLU"
        );

        // Placement must not change the activation either.
        assert_eq!(
            run_expert_placed(
                &hidden,
                &expert,
                ExpertPlacement::GpuDevice(0),
                GluAct::Geglu
            ),
            got,
            "a GPU-placed GeGLU expert must not fall into a SwiGLU kernel"
        );
    }

    /// `run_expert_placed` must be a real drop-in for `run_expert` when
    /// no GPU dispatch actually happens -- true unconditionally without
    /// the `cuda` feature, and true even *with* the feature for `Cpu`
    /// placement (which never calls `apply_gpu` at all) or an
    /// unsupported quant kind (F32 here, which `apply_gpu` always
    /// returns `None` for, falling through to `run_expert`).
    #[test]
    fn run_expert_placed_matches_run_expert_when_nothing_is_gpu_dispatched() {
        use frink_core::tensor::Tensor;
        let hidden_dim = 4;
        let ffn_dim = 3;
        let expert = ExpertWeights {
            gate: WeightMatrix::F32(Tensor::new(
                vec![0.1; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            up: WeightMatrix::F32(Tensor::new(
                vec![0.2; ffn_dim * hidden_dim],
                vec![ffn_dim, hidden_dim],
            )),
            down: WeightMatrix::F32(Tensor::new(
                vec![0.3; hidden_dim * ffn_dim],
                vec![hidden_dim, ffn_dim],
            )),
        };
        let hidden = vec![1.0, -1.0, 0.5, 0.5];
        let expected = run_expert(&hidden, &expert, GluAct::Swiglu);

        assert_eq!(
            run_expert_placed(&hidden, &expert, ExpertPlacement::Cpu, GluAct::Swiglu),
            expected
        );
        assert_eq!(
            run_expert_placed(
                &hidden,
                &expert,
                ExpertPlacement::GpuDevice(0),
                GluAct::Swiglu
            ),
            expected,
            "F32 has no GPU kernel, so GpuDevice placement must still fall through to the CPU path"
        );
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    #[test]
    #[ignore = "requires real GPU hardware (CUDA or Metal) -- run with --ignored"]
    fn run_expert_placed_on_gpu_matches_cpu_for_a_real_quantized_expert() {
        let hidden_dim = 32;
        let ffn_dim = 32; // must be a multiple of Q8_0 block elems (32)
        let make_row = |cols: usize, seed: f32| -> Vec<f32> {
            (0..cols)
                .map(|i| ((i as f32) - (cols as f32) / 2.0) * 0.01 * seed)
                .collect()
        };
        let quantize_matrix = |rows: usize, cols: usize, seed: f32| {
            let mut packed = Vec::new();
            for r in 0..rows {
                packed.extend(frink_quant::quantize_q8_0(&make_row(cols, seed + r as f32)));
            }
            WeightMatrix::Quantized {
                data: frink_core::weight_matrix::WeightBytes::Owned(packed),
                rows,
                cols,
                kind: frink_core::weight_matrix::QuantKind::Q8_0,
            }
        };
        let expert = ExpertWeights {
            gate: quantize_matrix(ffn_dim, hidden_dim, 1.0),
            up: quantize_matrix(ffn_dim, hidden_dim, 2.0),
            down: quantize_matrix(hidden_dim, ffn_dim, 3.0),
        };
        let hidden = make_row(hidden_dim, 0.5);

        let cpu = run_expert_placed(&hidden, &expert, ExpertPlacement::Cpu, GluAct::Swiglu);
        let gpu = run_expert_placed(
            &hidden,
            &expert,
            ExpertPlacement::GpuDevice(0),
            GluAct::Swiglu,
        );
        assert_eq!(cpu.len(), gpu.len());
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            assert!((c - g).abs() < 1e-1, "cpu={c} gpu={g}");
        }
    }
}

/// Gemma-4's MoE router: how a hidden state becomes routing weights.
///
/// Four things differ from every other family here, and three of them
/// change the numbers without changing any shape -- so getting one
/// wrong produces a model that is fluent and wrong, with nothing to
/// catch it.
///
/// 1. The router input is normalized by a **weightless** RMSNorm. No
///    learned per-channel scale, unlike every other norm in the stack.
///    Reusing a weighted `rms_norm` here silently applies whatever
///    weight vector happened to be at hand.
/// 2. The normalized state is multiplied by a learned `router_scale`
///    vector, **and** by `hidden^-0.5`. The second factor is a
///    function of the width alone, so it is easy to omit and impossible
///    to notice: it rescales every logit by the same constant, which
///    changes the softmax temperature over the selected experts and
///    therefore the mixing weights, while leaving the top-k selection
///    itself identical.
/// 3. Selection is by **raw logit**, and the softmax runs over just the
///    selected `k`. That is not the same as softmaxing all experts and
///    slicing (see [`route_top_k_softmax`], where the surviving weights
///    sum to less than one); here they sum to exactly one.
/// 4. Each selected weight is then multiplied by
///    `per_expert_scale[expert_id]` -- a per-expert rescale applied to
///    the ROUTING WEIGHT rather than to the expert's output. No other
///    family here has it, and after it the weights no longer sum to
///    one, which is correct and must not be "fixed" by renormalizing.
///
/// `hidden` is the router's input; `router_weight` is the router
/// projection's output for it (one logit per expert, already computed
/// by the caller from the scaled state -- see
/// [`gemma4_router_logits`]).
pub fn route_gemma4_moe(logits: &[f32], k: usize, per_expert_scale: &[f32]) -> RoutingDecision {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    // Top-k by RAW logit, ties toward the lower expert id so a cached
    // prefix cannot disagree with the run that produced it.
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let top = &idx[..k.min(idx.len())];

    let selected: Vec<f32> = top.iter().map(|&i| logits[i]).collect();
    let max = selected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = selected.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let weights: Vec<f32> = if sum > 0.0 {
        top.iter()
            .zip(exps.iter())
            .map(|(&e, &x)| {
                // The per-expert scale lands on the weight, after the
                // softmax. The weights deliberately no longer sum to
                // one afterwards.
                (x / sum) * per_expert_scale.get(e).copied().unwrap_or(1.0)
            })
            .collect()
    } else {
        exps
    };

    RoutingDecision {
        expert_ids: top.to_vec(),
        weights,
    }
}

/// The router logits Gemma-4 feeds to [`route_gemma4_moe`]: a
/// weightless RMSNorm of the hidden state, scaled by `router_scale` and
/// by `hidden^-0.5`, then projected.
///
/// Split from the routing itself so the two unusual scalings are
/// testable without a projection matrix -- see [`route_gemma4_moe`]'s
/// docs on why the `hidden^-0.5` factor is the easy one to lose.
pub fn gemma4_router_logits(
    hidden: &[f32],
    router_scale: &[f32],
    router_proj: &WeightMatrix,
    eps: f32,
) -> Vec<f32> {
    debug_assert_eq!(hidden.len(), router_scale.len());
    let n = hidden.len() as f32;
    // Weightless RMSNorm: no learned per-channel term.
    let mean_sq = hidden.iter().map(|v| v * v).sum::<f32>() / n;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let width_scale = n.powf(-0.5);
    let scaled: Vec<f32> = hidden
        .iter()
        .zip(router_scale.iter())
        .map(|(&v, &s)| v * inv_rms * s * width_scale)
        .collect();
    router_proj.apply(&scaled)
}

#[cfg(test)]
mod gemma4_router_tests {
    use super::*;

    /// The per-expert scale lands on the routing WEIGHT, after the
    /// softmax, and the weights deliberately stop summing to one. A
    /// renormalization "fixing" that would cancel the scale exactly,
    /// which is the whole failure this pins: no shape changes, and the
    /// model stays fluent.
    #[test]
    fn the_per_expert_scale_multiplies_the_weight_and_breaks_the_sum_to_one() {
        let logits = vec![3.0, 1.0, 2.0, 0.0];
        let flat = route_gemma4_moe(&logits, 2, &[1.0; 4]);
        let sum: f32 = flat.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "with unit scales, k weights sum to one"
        );

        let scaled = route_gemma4_moe(&logits, 2, &[2.0, 1.0, 0.5, 1.0]);
        assert_eq!(scaled.expert_ids, flat.expert_ids, "selection is unchanged");
        // Experts 0 and 2 were selected; their scales are 2.0 and 0.5.
        assert!((scaled.weights[0] - flat.weights[0] * 2.0).abs() < 1e-6);
        assert!((scaled.weights[1] - flat.weights[1] * 0.5).abs() < 1e-6);
        let sum: f32 = scaled.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() > 1e-3,
            "the scaled weights must NOT be renormalized back to one, got {sum}"
        );
    }

    /// Softmax over just the selected k, not a slice of the full
    /// distribution. The two pick the same experts and weight them
    /// differently, which is exactly the kind of difference that
    /// produces a fluent wrong model.
    #[test]
    fn the_softmax_runs_over_the_selected_experts_only() {
        let logits = vec![3.0, 1.0, 2.0, 0.0];
        let gemma = route_gemma4_moe(&logits, 2, &[1.0; 4]);
        let sliced = route_top_k_softmax(&logits, 2, false);
        assert_eq!(gemma.expert_ids, sliced.expert_ids);
        let gemma_sum: f32 = gemma.weights.iter().sum();
        let sliced_sum: f32 = sliced.weights.iter().sum();
        assert!((gemma_sum - 1.0).abs() < 1e-6);
        assert!(
            sliced_sum < 0.99,
            "a slice of the full softmax sums to less than one, got {sliced_sum}"
        );
    }

    /// Selection is by raw logit, so the largest logits win regardless
    /// of the per-expert scales -- the scale rescales a weight, it does
    /// not buy an expert its way into the selection.
    #[test]
    fn selection_is_by_raw_logit_and_the_scale_cannot_change_it() {
        let logits = vec![3.0, 1.0, 2.0, 0.0];
        let huge = route_gemma4_moe(&logits, 2, &[1.0, 1000.0, 1.0, 1000.0]);
        assert_eq!(
            huge.expert_ids,
            vec![0, 2],
            "expert 1's scale must not select it"
        );
    }

    /// The width factor is a function of the hidden size alone, so it
    /// leaves the selection identical and changes the softmax
    /// temperature -- which is what makes omitting it invisible.
    #[test]
    fn the_width_scaling_changes_the_weights_but_not_the_selection() {
        let hidden = vec![1.0, -2.0, 0.5, 3.0];
        let router_scale = vec![1.0; 4];
        let proj = WeightMatrix::F32(frink_core::tensor::Tensor::new(
            vec![
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ],
            vec![4, 4],
        ));
        let with_width = gemma4_router_logits(&hidden, &router_scale, &proj, 1e-6);

        // The same thing without the hidden^-0.5 factor: every logit is
        // larger by exactly sqrt(hidden).
        let n = hidden.len() as f32;
        let without: Vec<f32> = with_width.iter().map(|v| v * n.sqrt()).collect();

        let a = route_gemma4_moe(&with_width, 2, &[1.0; 4]);
        let b = route_gemma4_moe(&without, 2, &[1.0; 4]);
        assert_eq!(a.expert_ids, b.expert_ids, "the selection is unaffected");
        assert!(
            a.weights
                .iter()
                .zip(b.weights.iter())
                .any(|(x, y)| (x - y).abs() > 1e-4),
            "but the mixing weights are not: {:?} vs {:?}",
            a.weights,
            b.weights
        );
    }

    /// The router's norm is WEIGHTLESS. Feeding a non-unit scale vector
    /// through must change the logits, which is what proves the norm
    /// itself is not quietly applying one.
    #[test]
    fn the_router_norm_carries_no_learned_weight_of_its_own() {
        let hidden = vec![1.0, -2.0, 0.5, 3.0];
        let proj = WeightMatrix::F32(frink_core::tensor::Tensor::new(
            (0..16)
                .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
                .collect(),
            vec![4, 4],
        ));
        let unit = gemma4_router_logits(&hidden, &[1.0; 4], &proj, 1e-6);
        let scaled = gemma4_router_logits(&hidden, &[2.0; 4], &proj, 1e-6);
        for (u, s) in unit.iter().zip(scaled.iter()) {
            assert!(
                (s - u * 2.0).abs() < 1e-5,
                "router_scale is the ONLY learned scale on this path: {u} -> {s}"
            );
        }
    }

    /// Ties break toward the lower expert id, deterministically.
    #[test]
    fn ties_break_toward_the_lower_expert_id() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        for _ in 0..8 {
            assert_eq!(
                route_gemma4_moe(&logits, 2, &[1.0; 4]).expert_ids,
                vec![0, 1]
            );
        }
    }
}
