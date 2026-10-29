//! `{arch}.interleave_moe_layer_step`, and why frink serves only the
//! value real checkpoints carry.
//!
//! llama.cpp's `LLM_KV_INTERLEAVE_MOE_LAYER_STEP`
//! (`llama-arch.cpp:223`) says how often a MoE layer appears past the
//! leading dense prefix. `src/models/ernie4-5-moe.cpp:64` is the graph's
//! rule:
//!
//! ```text
//! bool is_moe_layer =
//!     static_cast<uint32_t>(il) >= hparams.n_layer_dense_lead && (il + 1) % hparams.n_moe_layer_step == 0;
//! ```
//!
//! frink's `ModelConfig::layer_is_dense` implements only the first
//! half, the leading-dense prefix. This module is the gate that stops a
//! file where the second half would change the answer.
//!
//! # The finding: llama.cpp cannot load such a file either
//!
//! Implementing the modulo was the plan. Building the fixture is what
//! stopped it, because llama.cpp's own tensor loader and its own graph
//! disagree about which layers are MoE, and only the graph has the
//! modulo:
//!
//! * `src/models/ernie4-5.cpp:49` -- the loader --
//!   `if (arch == LLM_ARCH_ERNIE4_5_MOE && (uint32_t) i >= hparams.n_layer_dense_lead)`
//!   creates `ffn_gate_inp`, `ffn_down_exps` and `ffn_up_exps` as
//!   REQUIRED for every layer at or past the prefix, and creates
//!   `ffn_gate` / `ffn_up` / `ffn_down` for NO such layer. There is no
//!   `n_moe_layer_step` in that condition.
//! * `src/models/ernie4-5-moe.cpp:64` -- the graph -- runs the DENSE
//!   branch on a layer past the prefix whenever `(il + 1) % step != 0`,
//!   reading the `ffn_up`/`ffn_gate`/`ffn_down` the loader never
//!   created.
//!
//! The two agree only when every layer at or past the prefix satisfies
//! `(il + 1) % step == 0`, and that is exactly the case where the modulo
//! changes nothing. There is no configuration in which the interleave
//! interleaves AND llama.cpp can load the file: a converter-produced
//! checkpoint puts `blk.N.ffn_gate.weight` on the interleaved dense
//! layers, and llama.cpp dies on the missing `blk.N.ffn_down_exps.weight`.
//! `tests/one_match_arm_graphs.rs` ships the two-step fixture that
//! demonstrates it.
//!
//! Every real ERNIE-4.5 MoE export therefore carries a step of 1:
//! `conversion/ernie.py:88` writes `moe_layer_interval` straight from
//! the HF config, and both published checkpoints
//! (`ERNIE-4.5-21B-A3B`, `ERNIE-4.5-300B-A47B`) set it to 1, at which
//! point `(il + 1) % 1 == 0` for every layer and the rule collapses to
//! the leading-dense prefix frink already implements and has now
//! evidenced against libllama.
//!
//! So the arm is a REFUSAL rather than an implementation: frink runs
//! the files that exist and stops, by name, on the shape that no
//! reference can confirm. Implementing the modulo would have added a
//! decode path that no loadable checkpoint reaches and no golden run
//! could check.
//!
//! # The one loader that HONOURS the step
//!
//! `grep -n n_moe_layer_step src/models/*.cpp` over all 155 graphs is
//! the two ERNIE lines above and `llama4.cpp:6,64`. Llama 4 reads the
//! key as REQUIRED (`:6`) and its TENSOR LOADER branches on it
//! (`:64`, `is_moe_layer = n_moe_layer_step > 0 && (i + 1) % step ==
//! 0`, the routed triple plus a shared expert on those layers and the
//! dense triple on the rest, `:77-94`), so a Llama 4 file whose step
//! interleaves loads upstream, and its graph branches on the router's
//! presence (`:202`). Scout and Maverick export step 1 (every layer
//! MoE); `interleave_moe_layer_step: 2` is Llama-4's config for the
//! shape the fixture takes. [`INTERLEAVE_STEP_HONOURED_BY_LOADER`] is
//! that one row, [`interleave_step`] its answer, and
//! `ModelConfig::layer_is_dense` the ONE place the modulo is spelled.
//! A step of 0 there makes every layer dense with `n_expert` still
//! nonzero, a file no converter writes and no fixture has a golden
//! for; it is refused by name rather than served unmeasured.

/// Architectures for which llama.cpp reads the interleave step as a
/// REQUIRED key, so a file lacking it is one llama.cpp refuses to load.
///
/// `ernie4-5.cpp:11` calls `ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,
/// hparams.n_moe_layer_step)` with no `required = false`, and
/// `ernie4-5-moe.cpp:26` then asserts the value is positive. Both are
/// hard failures upstream.
pub const INTERLEAVE_STEP_IS_REQUIRED: &[&str] = &["ernie4_5-moe"];

/// The step `ModelConfig::layer_is_dense` applies for `arch`, or
/// `None` where the loader does not branch on it. `n_experts` is the
/// file's `expert_count`, refused at zero where llama.cpp throws.
pub fn interleave_step(
    arch: &str,
    step: Option<u64>,
    n_experts: usize,
) -> Result<Option<usize>, String> {
    interleave_step_for(
        arch,
        crate::models::ArchFacts::of(arch)
            .moe
            .interleave_step_honoured,
        step,
        n_experts,
    )
}

/// [`interleave_step`] given the row (`ArchFacts`): the llama.cpp lines
/// where the loader honours the step (`llama4`, `llama4.cpp:6,49-51,64`,
/// is the one graph with one).
pub fn interleave_step_for(
    arch: &str,
    row: Option<&'static str>,
    step: Option<u64>,
    n_experts: usize,
) -> Result<Option<usize>, String> {
    let Some(lines) = row else {
        return Ok(None);
    };
    if n_experts == 0 {
        return Err(format!(
            "`{arch}.expert_count` is 0: llama.cpp throws `{arch} model cannot have zero \
             experts` before creating a tensor ({lines}; measured on \
             scripts/make_llama4_fixture.py --dense), so a dense Llama 4 (MobileLLM) has no \
             reference graph to match"
        ));
    }
    match step {
        None => Err(format!(
            "`{arch}.interleave_moe_layer_step` is missing; llama.cpp reads it as a REQUIRED \
             key ({lines}) and every export writes it (conversion/llama.py:392)"
        )),
        Some(0) => Err(format!(
            "`{arch}.interleave_moe_layer_step` is 0: {lines} then loads EVERY layer dense \
             with `expert_count` nonzero, a file no converter writes and no fixture has a \
             libllama golden for"
        )),
        Some(step) => Ok(Some(step as usize)),
    }
}

/// Architectures whose LOADER decides dense-or-MoE per layer by the
/// presence of `blk.N.ffn_gate_inp.weight`, with no key involved:
/// `jamba.cpp:89-101` creates the router `TENSOR_NOT_REQUIRED` and
/// creates the expert triple when it exists, the dense triple when it
/// does not, and `:152` branches the graph on the same pointer. Jamba
/// interleaves MoE and dense layers (every second layer on Jamba-v0.1)
/// and its converter writes no `leading_dense_block_count` or
/// interleave step.
///
/// Sixteen graphs branch on the pointer in their GRAPH (measured:
/// `grep -l 'ffn_gate_inp == nullptr' src/models/*.cpp`), but every
/// other loader creates it from `n_expert > 0` for every layer or from
/// the leading-dense prefix, which `ModelConfig::layer_is_dense`
/// already spells; only Jamba's loader reads the file.
pub const DENSE_LAYER_BY_ROUTER_ABSENCE: &[&str] = &["jamba"];

/// True when layer `layer` of `arch` is a dense layer BECAUSE the file
/// has no router for it (`DENSE_LAYER_BY_ROUTER_ABSENCE`).
pub fn dense_by_router_absence(
    arch: &str,
    file: &impl frink_gguf::TensorSource,
    layer: usize,
) -> bool {
    DENSE_LAYER_BY_ROUTER_ABSENCE.contains(&arch)
        && file
            .find_tensor(&format!("blk.{layer}.ffn_gate_inp.weight"))
            .is_none()
}

/// Why this file's interleave step cannot be served, or `None` when it
/// can.
///
/// `step` is the raw `{arch}.interleave_moe_layer_step` value, `None`
/// when the file does not carry the key.
///
/// Three outcomes, and each one is reachable:
///
/// * absent, and the architecture does not require it -> served. Every
///   architecture but `ernie4_5-moe` is here, including every dense one.
/// * `1` -> served. The rule at `ernie4-5-moe.cpp:64` reduces to the
///   leading-dense prefix, which frink implements.
/// * absent-but-required, `0`, or `> 1` -> refused, by name.
pub fn interleave_step_refusal(arch: &str, step: Option<u64>) -> Option<String> {
    interleave_step_refusal_for(
        arch,
        crate::models::ArchFacts::of(arch)
            .moe
            .interleave_step_honoured,
        step,
    )
}

/// [`interleave_step_refusal`] given the row (`ArchFacts`).
pub fn interleave_step_refusal_for(
    arch: &str,
    row: Option<&'static str>,
    step: Option<u64>,
) -> Option<String> {
    if row.is_some() {
        return None;
    }
    match step {
        None if INTERLEAVE_STEP_IS_REQUIRED.contains(&arch) => Some(format!(
            "`{arch}.interleave_moe_layer_step` is missing. llama.cpp reads it as a REQUIRED \
             key for this architecture (src/models/ernie4-5.cpp:11) and asserts it is positive \
             (ernie4-5-moe.cpp:26), so a file without it is one llama.cpp will not load \
             either. Every real ERNIE-4.5 MoE export carries it (conversion/ernie.py:88)"
        )),
        None | Some(1) => None,
        Some(0) => Some(format!(
            "`{arch}.interleave_moe_layer_step` is 0. llama.cpp asserts \
             `hparams.n_moe_layer_step > 0` (src/models/ernie4-5-moe.cpp:26) before building \
             the graph, and a step of 0 would divide by zero in its own layer rule at :64"
        )),
        Some(step) => Some(format!(
            "`{arch}.interleave_moe_layer_step` is {step}, and frink serves only 1. \
             src/models/ernie4-5-moe.cpp:64 makes a layer MoE when \
             `il >= n_layer_dense_lead && (il + 1) % n_moe_layer_step == 0`, while \
             ModelConfig::layer_is_dense implements only the leading-dense prefix -- but \
             NEITHER does llama.cpp load such a file. Its tensor loader \
             (src/models/ernie4-5.cpp:49) creates the expert tensors as REQUIRED for EVERY \
             layer at or past `n_layer_dense_lead`, with no step in the condition, and \
             creates the dense `ffn_gate`/`ffn_up`/`ffn_down` for none of them. The loader \
             and the graph therefore agree only when the step changes nothing, so a \
             checkpoint whose interleave really interleaves cannot be loaded by llama.cpp \
             and there is no reference to check frink against. Both published ERNIE-4.5 MoE \
             checkpoints carry a step of 1, which frink runs and \
             tests/one_match_arm_graphs.rs pins against libllama's own logits"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value every real checkpoint carries is served, and so is a
    /// file that has no such key at all.
    #[test]
    fn a_step_of_one_and_an_absent_key_are_both_served() {
        assert!(interleave_step_refusal("ernie4_5-moe", Some(1)).is_none());
        assert!(interleave_step_refusal("llama", None).is_none());
        assert!(interleave_step_refusal("qwen3moe", Some(1)).is_none());
    }

    /// A step that really interleaves is refused, and the refusal says
    /// that llama.cpp cannot load it either.
    ///
    /// The second half is the part worth asserting. A refusal that only
    /// said "frink does not implement this" would send whoever picks it
    /// up to write a decode path against a reference that cannot run.
    #[test]
    fn a_step_above_one_is_refused_and_names_llama_cpps_own_contradiction() {
        let r = interleave_step_refusal("ernie4_5-moe", Some(2)).expect("refused");
        assert!(r.contains("ernie4-5-moe.cpp:64"), "{r}");
        assert!(r.contains("ernie4-5.cpp:49"), "{r}");
        assert!(r.contains("cannot be loaded by llama.cpp"), "{r}");
    }

    /// The required-key half fires for `ernie4_5-moe` and for nobody
    /// else.
    ///
    /// A gate that cannot fire is worse than no gate, and the inverse
    /// holds too: a gate that fires on every architecture would refuse
    /// every dense checkpoint in the repo.
    #[test]
    fn the_key_is_required_for_ernie_and_optional_everywhere_else() {
        assert!(interleave_step_refusal("ernie4_5-moe", None).is_some());
        for arch in ["llama", "qwen3moe", "dots1", "ernie4_5"] {
            assert!(
                interleave_step_refusal(arch, None).is_none(),
                "{arch} must not require the key"
            );
        }
    }

    /// The honoured row: served at any positive step, refused by name
    /// at zero, without the key, and without experts; everyone else
    /// answers `None` and keeps the ERNIE rule.
    #[test]
    fn llama4_s_step_is_served_and_its_three_refusals_name_the_lines() {
        assert_eq!(interleave_step("llama4", Some(2), 16).unwrap(), Some(2));
        assert_eq!(interleave_step("llama4", Some(1), 128).unwrap(), Some(1));
        assert!(interleave_step("llama4", None, 16)
            .unwrap_err()
            .contains("REQUIRED"));
        assert!(interleave_step("llama4", Some(0), 16)
            .unwrap_err()
            .contains("EVERY layer dense"));
        assert!(interleave_step("llama4", Some(2), 0)
            .unwrap_err()
            .contains("cannot have zero"));
        assert_eq!(interleave_step("ernie4_5-moe", Some(2), 16).unwrap(), None);
        assert!(interleave_step_refusal("llama4", Some(2)).is_none());
    }

    /// Zero is refused separately, because it is the one value that
    /// would panic rather than mis-route.
    #[test]
    fn a_step_of_zero_is_refused_rather_than_dividing_by_zero() {
        let r = interleave_step_refusal("ernie4_5-moe", Some(0)).expect("refused");
        assert!(r.contains("divide by zero"), "{r}");
    }
}
