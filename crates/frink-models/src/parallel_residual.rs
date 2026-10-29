//! **THE PARALLEL RESIDUAL** -- `x + attn(norm(x)) + ffn(norm(x))`, the
//! layer shape the generic decoder does not have, and which of llama.cpp's
//! graphs build it.
//!
//! # What it is
//!
//! The generic layer is sequential: `h = x + attn(norm1(x))`, then
//! `h + ffn(norm2(h))`. The parallel shape feeds the FFN the LAYER INPUT
//! -- normed -- rather than the attention output, and sums the three
//! terms once. Two spellings exist upstream:
//!
//! - **One shared norm** (`SharedNorm`): the FFN reads the SAME normed
//!   tensor attention read. `stablelm.cpp:135-137` (`cur = inpSA` when
//!   `ffn_norm` is ABSENT), `phi2.cpp:67,108,116-117`,
//!   `falcon.cpp:124-135` (Falcon-7B, no `attn_norm_2`),
//!   `command-r.cpp:68,106-119`, `cohere2.cpp:120-134`,
//!   `cohere2moe.cpp:222-266`, `plamo.cpp:59-64,97-98,111-112` (`cur =
//!   sa_inp`, over an RMSNorm).
//! - **Two norms** (`TwoNorms`): `x + attn(ln1(x)) + ffn(ln2(x))`,
//!   `gptneox.cpp:143-166` (`use_parallel_residual`, read at `:5`) and
//!   `falcon.cpp:79-85` (Falcon-40B: `attn_norm_2` present, and it is
//!   ATTENTION's norm, `attn_norm` staying the FFN's; every Falcon layer
//!   is parallel, the optional tensor only picks the arm).
//!
//! # Reach -- MEASURED
//!
//! Over all 155 `src/models/*.cpp` (2026-09-12): `grep -l "par_res\|
//! parallel residual"` is `gptneox.cpp` and `stablelm.cpp`; a scan for
//! TWO consecutive `cur = ggml_add(ctx0, cur, ...)` lines -- the
//! three-term sum spelled out -- is `cohere2`, `cohere2moe` (twice, the
//! trunk and its MTP block), `command-r`, `falcon`, `phi2` and `plamo`,
//! with `gptneox` and `stablelm` separating their two adds by a `cb`
//! line. `gemma4.cpp:260` names an `attn_out` that is ALREADY `cur +
//! inpL`, so it is sequential and not in the table. Eight graphs, two
//! spellings, and `stablelm` is the one where BOTH shapes sit behind one
//! architecture string, decided by tensor presence. (`plamo` was missed
//! by a first grep that looked for `attn_out` by name; its attention
//! output is `sa_out`. The two-adds scan is the measurement.)
//!
//! # How it is served
//!
//! The sequential bodies already compute `h = x + attn(attn_norm(x))`
//! and then `h + ffn(normed2)`; `(x + attn) + ffn` IS the three-term
//! sum, so the parallel shape differs from the sequential one in
//! exactly one thing: WHAT `normed2` is. Sequential: `ffn_norm(h)`.
//! Parallel: a norm of `x`, the LAYER INPUT, which attention has
//! already been added on top of by the time the FFN body runs. So the
//! FFN input is captured BEFORE attention, at the same point
//! `crate::router_input` captures the router's operand, and the two
//! travel together as `decoder::ffn_block::BranchInputs` -- one
//! constructor, `Decoder::branch_inputs`, called at the top of every
//! layer of every host body, so a body cannot take one and forget the
//! other. `MoeWeights::parallel` is the per-layer fact (the `stablelm`
//! row is decided per layer, as `stablelm.cpp:129` decides it), and
//! for a `SharedNorm` layer the pre-FFN slot is `NormOp::None`,
//! because there is no tensor and the norm was applied when attention
//! took its input. `ModelConfig::parallel_residual` is the model-level
//! fact `Decoder::metal_can_serve_model` reads: every fused Metal
//! launch bakes `ffn_norm` over the post-attention residual into its
//! kernel, so a model with a parallel layer stays on the host bodies.
//!
//! Evidence (`tests/parallel_residual_graphs.rs`): `gptneox` under the
//! key, both values, and `plamo`, against libllama; the `stablelm`
//! shape that was refused by this module for one PR
//! (`tests/stablelm_graphs.rs`) matches now, and `command-r` followed
//! on the shared-norm arm over a weighted LayerNorm
//! (`tests/command_r_graphs.rs`). `use_parallel_residual`
//! is read by `gptneox.cpp:5` and by NOTHING in `stablelm.cpp`
//! (libllama's logits with and without it are byte-identical there,
//! measured), which is why `stablelm`'s rule is the tensor and
//! `gptneox`'s is the key.

use frink_gguf::TensorSource;

/// Which normed input the FFN reads in a parallel layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelNorm {
    /// The FFN reads the tensor attention read (`attn_norm(x)`).
    SharedNorm,
    /// The FFN reads its own norm of the layer input (`ffn_norm(x)`).
    TwoNorms,
}

/// How a graph decides that a layer is parallel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelWhen {
    /// Every layer, unconditionally.
    Always,
    /// `blk.N.ffn_norm.weight` is absent (`stablelm.cpp:129`).
    FfnNormAbsent,
    /// `{arch}.use_parallel_residual` is true (`gptneox.cpp:5,143`).
    ParallelResidualKey,
}

/// One graph that builds the parallel residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelResidual {
    pub arch: &'static str,
    /// The arm a parallel layer takes when `second_norm` is absent or
    /// `None`.
    pub norm: ParallelNorm,
    pub when: ParallelWhen,
    /// A per-layer OPTIONAL tensor whose presence turns the layer into
    /// the two-norm arm: `falcon.cpp:35-36,79-85`'s `attn_norm_2`, which
    /// norms the layer input FOR ATTENTION while `attn_norm` keeps
    /// feeding the FFN (`crate::norm_sites::ATTN_NORM_2_FEEDS_ATTENTION`
    /// crosses the two slots on such a layer).
    pub second_norm: Option<&'static str>,
    pub lines: &'static str,
}

/// `stablelm`'s row, stated by `crate::models::stablelm` (#403 step 7a).
pub const STABLELM: ParallelResidual = ParallelResidual {
    arch: "stablelm",
    norm: ParallelNorm::SharedNorm,
    when: ParallelWhen::FfnNormAbsent,
    second_norm: None,
    lines: "src/models/stablelm.cpp:38-39,129-138,147",
};

/// `gptneox`'s row, stated by `crate::models::gptneox` (#403 step 7a).
pub const GPTNEOX: ParallelResidual = ParallelResidual {
    arch: "gptneox",
    norm: ParallelNorm::TwoNorms,
    when: ParallelWhen::ParallelResidualKey,
    second_norm: None,
    lines: "src/models/gptneox.cpp:5,143-166",
};

/// `phi2`'s row, stated by `crate::models::phi2` (#403 step 7a).
pub const PHI2: ParallelResidual = ParallelResidual {
    arch: "phi2",
    norm: ParallelNorm::SharedNorm,
    when: ParallelWhen::Always,
    second_norm: None,
    lines: "src/models/phi2.cpp:67,108,116-117",
};

/// `falcon`'s row, stated by `crate::models::falcon` (#403 step 7a).
pub const FALCON: ParallelResidual = ParallelResidual {
    arch: "falcon",
    norm: ParallelNorm::SharedNorm,
    when: ParallelWhen::Always,
    second_norm: Some("attn_norm_2"),
    lines: "src/models/falcon.cpp:35-36,79-85,124-135",
};

/// `plamo`'s row, stated by `crate::models::plamo` (#403 step 7a).
pub const PLAMO: ParallelResidual = ParallelResidual {
    arch: "plamo",
    norm: ParallelNorm::SharedNorm,
    when: ParallelWhen::Always,
    second_norm: None,
    lines: "src/models/plamo.cpp:59-64,97-98,111-112",
};

/// The graphs not yet in `crate::models`, with the lines; the census of
/// all eight is [`parallel_residual_rows`].
pub const PARALLEL_RESIDUAL_GRAPHS: &[ParallelResidual] = &[
    ParallelResidual {
        arch: "command-r",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        second_norm: None,
        lines: "src/models/command-r.cpp:68,106-119",
    },
    ParallelResidual {
        arch: "cohere2",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        second_norm: None,
        lines: "src/models/cohere2.cpp:120-134",
    },
    ParallelResidual {
        arch: "cohere2moe",
        norm: ParallelNorm::SharedNorm,
        when: ParallelWhen::Always,
        second_norm: None,
        lines: "src/models/cohere2moe.cpp:222-266",
    },
];

/// The row for an architecture, or `None` for a sequential graph.
pub fn parallel_residual(arch: &str) -> Option<&'static ParallelResidual> {
    crate::models::ArchFacts::of(arch)
        .topology
        .parallel_residual
}

/// [`parallel_residual`] from [`PARALLEL_RESIDUAL_GRAPHS`] alone, for
/// `ArchFacts::from_tables`.
pub(crate) fn parallel_residual_from_table(arch: &str) -> Option<&'static ParallelResidual> {
    PARALLEL_RESIDUAL_GRAPHS.iter().find(|row| row.arch == arch)
}

/// Every row: the table's and the `crate::models` definitions'.
pub fn parallel_residual_rows() -> impl Iterator<Item = &'static ParallelResidual> {
    PARALLEL_RESIDUAL_GRAPHS.iter().chain(
        crate::models::REGISTRY
            .iter()
            .filter_map(|d| d.facts.topology.parallel_residual),
    )
}

/// Whether layer `l` of `arch` in `file` is a parallel layer, by the
/// row's rule. `false` for a graph not in the table.
pub fn layer_is_parallel(file: &impl TensorSource, arch: &str, l: usize) -> bool {
    layer_is_parallel_for(file, arch, parallel_residual(arch), l)
}

/// [`layer_is_parallel`] given the row (`ArchFacts`).
pub fn layer_is_parallel_for(
    file: &impl TensorSource,
    arch: &str,
    row: Option<&ParallelResidual>,
    l: usize,
) -> bool {
    let Some(row) = row else {
        return false;
    };
    match row.when {
        ParallelWhen::Always => true,
        ParallelWhen::FfnNormAbsent => file
            .find_tensor(&format!("blk.{l}.ffn_norm.weight"))
            .is_none(),
        ParallelWhen::ParallelResidualKey => file
            .metadata_bool(&format!("{arch}.use_parallel_residual"))
            .unwrap_or(false),
    }
}

/// The FFN input rule for layer `l` of `arch` in `file`: `Some(norm)`
/// for a parallel layer, `None` for a sequential one (every graph not in
/// the table, and a table row whose rule does not fire on this layer).
pub fn layer_parallel_norm(file: &impl TensorSource, arch: &str, l: usize) -> Option<ParallelNorm> {
    layer_parallel_norm_for(file, arch, parallel_residual(arch), l)
}

/// [`layer_parallel_norm`] given the row (`ArchFacts`).
pub fn layer_parallel_norm_for(
    file: &impl TensorSource,
    arch: &str,
    row: Option<&ParallelResidual>,
    l: usize,
) -> Option<ParallelNorm> {
    if !layer_is_parallel_for(file, arch, row, l) {
        return None;
    }
    let row = row?;
    let has_second = row.second_norm.is_some_and(|name| {
        file.find_tensor(&format!("blk.{l}.{name}.weight"))
            .is_some()
    });
    Some(if has_second {
        ParallelNorm::TwoNorms
    } else {
        row.norm
    })
}

/// Whether any trunk layer of `arch` in `file` is parallel: the
/// model-level fact the fused Metal launches refuse on.
pub fn model_has_parallel_layer(file: &impl TensorSource, arch: &str, n_layers: usize) -> bool {
    model_has_parallel_layer_for(file, arch, parallel_residual(arch), n_layers)
}

/// [`model_has_parallel_layer`] given the row (`ArchFacts`).
pub fn model_has_parallel_layer_for(
    file: &impl TensorSource,
    arch: &str,
    row: Option<&ParallelResidual>,
    n_layers: usize,
) -> bool {
    (0..n_layers).any(|l| layer_is_parallel_for(file, arch, row, l))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row names a real architecture string llama.cpp has, and the
    /// rows the loader can reach are exactly the audited ones; the rest
    /// are refused or deferred before any tensor is read, which is what
    /// makes their `when` a recorded fact rather than a live rule.
    #[test]
    fn every_row_is_a_registered_architecture_and_the_generic_ones_are_audited() {
        for row in parallel_residual_rows() {
            assert!(
                crate::capability::resolve_profile(row.arch).is_some(),
                "`{}` ({}) is not a registered architecture",
                row.arch,
                row.lines
            );
            let generic = matches!(
                crate::capability::resolve_architecture(row.arch),
                Some(crate::capability::ArchPath::GenericGqa { .. })
            );
            assert_eq!(
                generic,
                matches!(
                    row.arch,
                    "stablelm"
                        | "gptneox"
                        | "plamo"
                        | "command-r"
                        | "falcon"
                        | "phi2"
                        | "cohere2"
                        | "cohere2moe"
                ),
                "`{}`: a generic-path row here must have a golden in \
                 tests/parallel_residual_graphs.rs or its own graph test",
                row.arch
            );
        }
    }

    /// The rows are distinct, so a graph cannot be given two rules.
    #[test]
    fn no_architecture_has_two_rows() {
        let mut names: Vec<&str> = parallel_residual_rows().map(|r| r.arch).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), parallel_residual_rows().count());
        assert_eq!(parallel_residual_rows().count(), 8, "the measured reach");
    }

    /// A graph not in the table is sequential on every layer, whatever
    /// its tensors say.
    #[test]
    fn a_sequential_graph_is_never_parallel() {
        let file = crate::test_source::StubSource::with_tensors(&[]);
        assert!(!layer_is_parallel(&file, "llama", 0));
        assert_eq!(layer_parallel_norm(&file, "llama", 0), None);
        assert!(!model_has_parallel_layer(&file, "llama", 4));
    }

    /// The `stablelm` rule: parallel exactly when the layer's
    /// `ffn_norm.weight` is missing, per layer, with the shared norm.
    #[test]
    fn stablelm_is_decided_by_ffn_norm_presence_per_layer() {
        use crate::test_source::StubSource;
        let sequential =
            StubSource::with_tensors(&["blk.0.ffn_norm.weight", "blk.1.ffn_norm.weight"]);
        assert_eq!(layer_parallel_norm(&sequential, "stablelm", 0), None);
        assert_eq!(layer_parallel_norm(&sequential, "stablelm", 1), None);
        assert!(!model_has_parallel_layer(&sequential, "stablelm", 2));

        let mixed = StubSource::with_tensors(&["blk.0.ffn_norm.weight"]);
        assert_eq!(layer_parallel_norm(&mixed, "stablelm", 0), None);
        assert_eq!(
            layer_parallel_norm(&mixed, "stablelm", 1),
            Some(ParallelNorm::SharedNorm)
        );
        assert!(model_has_parallel_layer(&mixed, "stablelm", 2));
        // The trunk length bounds the scan: a parallel block past it is
        // not this loader's layer.
        assert!(!model_has_parallel_layer(&mixed, "stablelm", 1));
    }

    /// The other three rules, on the rows that carry them, so the
    /// table's `when` column is exercised and not only recorded.
    #[test]
    fn the_key_the_second_norm_and_the_unconditional_rules() {
        use crate::test_source::StubSource;
        use frink_gguf::GgufValue;
        let neox_seq = StubSource::with_tensors(&["blk.0.ffn_norm.weight"])
            .with_key("gptneox.use_parallel_residual", GgufValue::Bool(false));
        assert!(!layer_is_parallel(&neox_seq, "gptneox", 0));
        let neox_par = StubSource::with_tensors(&["blk.0.ffn_norm.weight"])
            .with_key("gptneox.use_parallel_residual", GgufValue::Bool(true));
        assert_eq!(
            layer_parallel_norm(&neox_par, "gptneox", 0),
            Some(ParallelNorm::TwoNorms)
        );

        // Every Falcon layer is parallel; `attn_norm_2` picks the arm.
        let falcon_7b = StubSource::with_tensors(&["blk.0.attn_norm.weight"]);
        assert_eq!(
            layer_parallel_norm(&falcon_7b, "falcon", 0),
            Some(ParallelNorm::SharedNorm)
        );
        let falcon_40b = StubSource::with_tensors(&["blk.0.attn_norm_2.weight"]);

        let phi2 = StubSource::with_tensors(&["blk.0.ffn_norm.weight"]);
        assert_eq!(
            layer_parallel_norm(&phi2, "phi2", 0),
            Some(ParallelNorm::SharedNorm)
        );
        assert_eq!(
            layer_parallel_norm(&falcon_40b, "falcon", 0),
            Some(ParallelNorm::TwoNorms)
        );
    }
}
