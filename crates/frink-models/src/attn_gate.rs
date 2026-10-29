//! The learned attention output gate: `attn_out *= act(W_g · x)` before
//! `wo`, llama.cpp's `LLM_TENSOR_ATTN_GATE` (`blk.N.attn_gate.weight`).
//!
//! `afmoe`, `laguna` and `step35` each refused NEW CODE naming this
//! tensor as what was left. The three were read side by side BEFORE
//! being called one cause, and they are one cause with two free
//! parameters, not one graph:
//!
//! | arch | gate input | activation | width | presence | lines |
//! |---|---|---|---|---|---|
//! | `afmoe` | `attn_norm(x)` | sigmoid | per element (`n_head * head_dim`) | required | `afmoe.cpp:73,154,183-185` |
//! | `laguna` | `attn_norm(x)` | **softplus** | per head OR per element, by tensor shape | required | `laguna.cpp:110-124,211,246-257` |
//! | `step35` | `attn_norm(x)` | sigmoid | per head (`n_head`) | **optional** | `step35.cpp:96,268-284` |
//!
//! What is IDENTICAL: the gate is projected from the same normed input
//! the Q/K/V projections read (`attn_inp` at `afmoe.cpp:148`,
//! `laguna.cpp:203`; `cur` at `step35.cpp:269` is the `attn_norm`
//! output of `:221`), it multiplies the attention output AFTER the
//! softmax-weighted V sum and BEFORE `wo`, and a per-head gate
//! broadcasts one scalar over that head's `head_dim` channels
//! (`ggml_reshape_3d(gate, 1, n_head, n_tokens)` then `ggml_mul`,
//! `laguna.cpp:251-253`, `step35.cpp:276-280`). What DIFFERS is the
//! activation (`ggml_sigmoid` vs `ggml_softplus`) and the width, and
//! `laguna` decides the width from the stored tensor's second dimension
//! (`:112-123`) with an abort for any other value. So the type has two
//! axes, [`GateAct`] and [`GateWidth`], the architecture table pins the
//! activation and the ADMISSIBLE widths, and the loader reads the width
//! off the tensor and refuses a width the table does not admit.
//!
//! **Measured before built.** Six of the then-140 `src/models/*.cpp`
//! created `LLM_TENSOR_ATTN_GATE`; re-measured on 2026-09-19 against
//! the moved pin it is FIFTEEN of 155, and the nine new ones all
//! landed upstream in one six-week window -- `bailingmoe3`,
//! `dots3note`, `hrm-text`, `hy-v4`, `kimi-k3`, `minimax-01`,
//! `muse-glimmer`, `qwen4exp`, `spark2-5`. Every one of the nine is a
//! refusal today (`capability::NORM_ROPE_TRIAGED` /
//! `NEOX_ROPE_TRIAGED` / the `dedicated` rows), and `spark2_5` is the
//! cheapest row in the tree because it needs ONE line of the table
//! below: sigmoid, per head, required, `src/models/spark2-5.cpp:41,
//! 97-105`. A gate that is one table row away should not be a
//! refusal for long. The other three of the original six --
//! `qwen3next.cpp:92,335`,
//! `qwen35.cpp:82,241`, `qwen35moe.cpp:88,265` -- store the gated
//! delta-net's `z` projection under the same name, sized
//! `{n_embd, value_dim}` and consumed by `build_norm_gated` on the
//! recurrent layers; their full-attention layers gate through a
//! double-width `wq` instead. That is `crate::gdn`'s `attn_gate`, a
//! different op on a different engine, and it is why the table below
//! is keyed by architecture and not by tensor presence alone.
//!
//! Every backend: the CPU row body ([`crate::decoder`]'s `attn_block`)
//! and the two batched host bodies apply it through ONE function,
//! [`AttnGate::apply_rows`]; the fused Metal attention launches fold the
//! softmax-V product straight into `wo` with no host round-trip in
//! between, so a layer carrying a gate is refused by the exhaustive
//! destructure in `Decoder::metal_attn_view` rather than served without
//! it -- the fifth thing found written into those stacks
//! unconditionally, after the final norm, the rotation, the residual
//! scale and the activation.

use frink_core::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_weight_matrix, LoadError};

/// The non-linearity the gate logits go through. Two, because the
/// three graphs use two; a third architecture adds a variant here and
/// a row in [`ATTN_GATE_ARCHS`], nowhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateAct {
    /// `1 / (1 + exp(-x))`, ggml `op_sigmoid` (`unary-ops.cpp:31-33`).
    Sigmoid,
    /// `x > 20 ? x : ln(1 + exp(x))`, ggml `op_softplus`
    /// (`unary-ops.cpp:80-82`), including the branch at 20.
    Softplus,
}

impl GateAct {
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            GateAct::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            GateAct::Softplus => {
                if x > 20.0 {
                    x
                } else {
                    (1.0 + x.exp()).ln()
                }
            }
        }
    }
}

/// How many gate values one token gets, and therefore how they
/// broadcast over the attention output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateWidth {
    /// `n_head` values; each scales its head's `head_dim` channels.
    PerHead,
    /// `n_head * head_dim` values, one per channel.
    PerElement,
}

/// Whether the tensor may be absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatePresence {
    /// `create_tensor(..., 0)`: llama.cpp refuses the file without it.
    Required,
    /// `TENSOR_NOT_REQUIRED`, and the graph tests the pointer
    /// (`step35.cpp:268`): absent means ungated.
    Optional,
}

/// One architecture's gate, as its llama.cpp graph spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnGateSpec {
    pub act: GateAct,
    /// Which widths the graph accepts. `laguna.cpp:116-120` aborts on
    /// any other; `afmoe.cpp:73` and `step35.cpp:96` size the tensor
    /// for exactly one, so any other fails their loaders' shape check.
    pub widths: &'static [GateWidth],
    pub presence: GatePresence,
    /// The lines that decide the row.
    pub lines: &'static str,
}

/// The three architectures whose graph gates its softmax attention
/// output through `LLM_TENSOR_ATTN_GATE`. See the module doc for the
/// three that store a different thing under the same name.
pub const ATTN_GATE_ARCHS: &[(&str, AttnGateSpec)] = &[
    (
        "afmoe",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerElement],
            presence: GatePresence::Required,
            lines: "src/models/afmoe.cpp:73,154,183-185",
        },
    ),
    (
        "laguna",
        AttnGateSpec {
            act: GateAct::Softplus,
            widths: &[GateWidth::PerHead, GateWidth::PerElement],
            presence: GatePresence::Required,
            lines: "src/models/laguna.cpp:110-124,211,246-257",
        },
    ),
    (
        "step35",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerHead],
            presence: GatePresence::Optional,
            lines: "src/models/step35.cpp:96,268-284",
        },
    ),
    // Landed upstream after the 2026-08-04 pin and closed on
    // 2026-09-19 as ONE row: `spark2-5.cpp:41` creates the tensor
    // REQUIRED at `{n_embd, n_head}` and `:97-105` sigmoids it and
    // multiplies it in per head, which is `step35`'s corner of the two
    // axes with the presence flipped. Nothing else in that graph was
    // new (`tests/gated_attention_graphs.rs`).
    // `hrm-text.cpp:77-78` creates the tensor REQUIRED at
    // `{n_embd, n_head * head_dim}` and `:113-134` sigmoid it and
    // multiply it into the attention output PER ELEMENT before `wo`;
    // its own comment calls it "the same shape as qwen3next attention
    // layers", which is `afmoe`'s corner here.
    (
        "hrm_text",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerElement],
            presence: GatePresence::Required,
            lines: "src/models/hrm-text.cpp:77-78,113-134",
        },
    ),
    // `muse-glimmer.cpp:46` creates the tensor REQUIRED at
    // `{n_embd, n_head * head_dim}` and `:100-135` sigmoid it and
    // multiply it into the attention output PER ELEMENT before `wo`,
    // which is `afmoe`'s corner; its own comment says "same as afmoe".
    (
        "muse-glimmer",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerElement],
            presence: GatePresence::Required,
            lines: "src/models/muse-glimmer.cpp:46,100-135",
        },
    ),
    (
        "spark2_5",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerHead],
            presence: GatePresence::Required,
            lines: "src/models/spark2-5.cpp:41,97-105",
        },
    ),
    // Landed upstream after the 2026-09-19 pin: `lfm3.cpp:77` creates
    // the tensor REQUIRED at `{n_embd, n_head}` on every attention
    // layer, and `:132-139` project it from the `attn_norm` output
    // (`cur_in`), sigmoid it and multiply it in per head before `wo`.
    // `spark2_5`'s corner.
    (
        "lfm3",
        AttnGateSpec {
            act: GateAct::Sigmoid,
            widths: &[GateWidth::PerHead],
            presence: GatePresence::Required,
            lines: "src/models/lfm3.cpp:77,132-139",
        },
    ),
];

/// The three graphs whose FULL-attention layers gate the softmax output
/// through a double-width `wq` (`qwen35.cpp:59,191-199,229-231`,
/// `qwen35moe.cpp`, `qwen3next.cpp`): `attn_q` is `2 * n_head *
/// head_dim` rows, each head's `[q, gate]` interleaved, and
/// `sigmoid(gate) * attn` runs before `wo`. Loaded as the fused matrix
/// (`AttnWeights::q_gate_interleaved`) and split AFTER the projection
/// ([`split_interleaved_q_gate`]), so a quantized `wq` stays one matrix
/// on its quantized path.
pub const Q_INTERLEAVED_GATE_ARCHS: &[(&str, &str)] = &[
    ("qwen35", "src/models/qwen35.cpp:59,191-199,229-231"),
    ("qwen35moe", "src/models/qwen35moe.cpp"),
    ("qwen3next", "src/models/qwen3next.cpp"),
];

/// True when `arch`'s full-attention layers project their gate inside
/// `attn_q`.
pub fn q_gate_interleaved(arch: &str) -> bool {
    Q_INTERLEAVED_GATE_ARCHS.iter().any(|(a, _)| *a == arch)
}

/// Splits `rows` rows of a `2 * n_heads * head_dim`-wide `wq` output
/// into the query rows (`n_heads * head_dim`) and the gate rows, per
/// head: `qwen35.cpp:191-199` view the query at offset 0 and the gate at
/// offset `head_dim` of each `2 * head_dim` stride.
pub fn split_interleaved_q_gate(
    fused: &[f32],
    rows: usize,
    n_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let width = n_heads * head_dim;
    assert_eq!(fused.len(), rows * 2 * width);
    let mut q = Vec::with_capacity(rows * width);
    let mut gate = Vec::with_capacity(rows * width);
    for r in 0..rows {
        let row = &fused[r * 2 * width..(r + 1) * 2 * width];
        for h in 0..n_heads {
            q.extend_from_slice(&row[h * 2 * head_dim..h * 2 * head_dim + head_dim]);
            gate.extend_from_slice(&row[h * 2 * head_dim + head_dim..(h + 1) * 2 * head_dim]);
        }
    }
    (q, gate)
}

/// `qwen35.cpp:229-230`: `attn_out *= sigmoid(gate)`, element for
/// element, on `rows` rows.
pub fn apply_interleaved_gate(attn_out: &mut [f32], gate: &[f32]) {
    assert_eq!(attn_out.len(), gate.len());
    for (a, g) in attn_out.iter_mut().zip(gate) {
        *a *= 1.0 / (1.0 + (-g).exp());
    }
}

/// The three graphs that create `LLM_TENSOR_ATTN_GATE` for the gated
/// delta-net's `z` projection instead. Recorded so the measurement
/// behind [`ATTN_GATE_ARCHS`] is checkable, and so nobody adds them to
/// it: their `attn_gate` is consumed by `build_norm_gated` on recurrent
/// layers and is `crate::gdn`'s business.
pub const GDN_Z_GATE_ARCHS: &[(&str, &str)] = &[
    ("qwen3next", "src/models/qwen3next.cpp:92,335"),
    ("qwen35", "src/models/qwen35.cpp:82,241"),
    ("qwen35moe", "src/models/qwen35moe.cpp:88,265"),
];

/// The gate `arch`'s graph applies, or `None` for the 137 that apply
/// none -- for which a file carrying `blk.N.attn_gate.weight` is
/// refused by `loader::assert_every_tensor_consumed`, not honoured.
pub fn attn_gate_spec(arch: &str) -> Option<AttnGateSpec> {
    ATTN_GATE_ARCHS
        .iter()
        .find(|(n, _)| *n == arch)
        .map(|(_, s)| *s)
}

/// One layer's loaded gate.
pub struct AttnGate {
    /// `blk.N.attn_gate.weight`, `{n_embd, n_gate_out}` on disk, so
    /// `n_gate_out` rows of `n_embd` here.
    pub proj: WeightMatrix,
    pub act: GateAct,
    pub width: GateWidth,
}

impl std::fmt::Debug for AttnGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttnGate")
            .field(
                "proj",
                &format_args!("{}x{}", self.proj.rows(), self.proj.cols()),
            )
            .field("act", &self.act)
            .field("width", &self.width)
            .finish()
    }
}

impl AttnGate {
    /// Reads layer `l`'s gate for `arch`, deciding the width from the
    /// tensor's row count the way `laguna.cpp:112-123` does and refusing
    /// a width the architecture's graph does not accept.
    ///
    /// `Ok(None)` only for an architecture with no gate at all, or an
    /// [`GatePresence::Optional`] one whose tensor is absent.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        l: usize,
        n_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Result<Option<AttnGate>, LoadError> {
        let Some(spec) = attn_gate_spec(arch) else {
            return Ok(None);
        };
        let name = format!("blk.{l}.attn_gate.weight");
        if file.find_tensor(&name).is_none() {
            return match spec.presence {
                GatePresence::Optional => Ok(None),
                GatePresence::Required => Err(LoadError::UnsupportedFeature(
                    arch.to_string(),
                    format!(
                        "{name} is missing; {arch}'s graph gates its attention output through \
                         it ({}) and llama.cpp refuses the file without it",
                        spec.lines
                    ),
                )),
            };
        }
        let proj = load_weight_matrix(file, &name)?;
        let width = match proj.rows() {
            r if r == n_heads * head_dim && spec.widths.contains(&GateWidth::PerElement) => {
                GateWidth::PerElement
            }
            r if r == n_heads && spec.widths.contains(&GateWidth::PerHead) => GateWidth::PerHead,
            r => {
                let admissible: Vec<String> = spec
                    .widths
                    .iter()
                    .map(|w| match w {
                        GateWidth::PerHead => format!("{n_heads} (per head)"),
                        GateWidth::PerElement => {
                            format!("{} (per element)", n_heads * head_dim)
                        }
                    })
                    .collect();
                return Err(LoadError::UnsupportedFeature(
                    arch.to_string(),
                    format!(
                        "{name} has {r} output rows; {arch}'s graph ({}) accepts {}",
                        spec.lines,
                        admissible.join(" or ")
                    ),
                ));
            }
        };
        if proj.cols() != hidden_dim {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "{name} reads {} inputs but the hidden width is {hidden_dim}",
                    proj.cols()
                ),
            ));
        }
        Ok(Some(AttnGate {
            proj,
            act: spec.act,
            width,
        }))
    }

    /// `attn_out[b] *= act(proj · normed[b])` for every row `b`, with a
    /// per-head gate broadcast over each head's `head_dim` channels.
    ///
    /// The ONE application, called by the row body with `rows == 1`
    /// and by the batched bodies with the whole batch, so the three
    /// host paths cannot disagree about which input the gate reads
    /// or which side of `wo` it sits on.
    pub fn apply_rows(&self, normed: &[f32], attn_out: &mut [f32], rows: usize, head_dim: usize) {
        debug_assert_eq!(normed.len(), rows * self.proj.cols());
        let gate_width = self.proj.rows();
        let gates = if rows == 1 {
            self.proj.apply(normed)
        } else {
            self.proj.apply_batch(normed, rows)
        };
        debug_assert_eq!(gates.len(), rows * gate_width);
        let out_width = attn_out.len() / rows;
        for (row, g) in attn_out.chunks_mut(out_width).zip(gates.chunks(gate_width)) {
            match self.width {
                GateWidth::PerElement => {
                    debug_assert_eq!(row.len(), g.len());
                    for (x, &gv) in row.iter_mut().zip(g.iter()) {
                        *x *= self.act.apply(gv);
                    }
                }
                GateWidth::PerHead => {
                    debug_assert_eq!(row.len(), g.len() * head_dim);
                    for (head, &gv) in row.chunks_mut(head_dim).zip(g.iter()) {
                        let s = self.act.apply(gv);
                        for x in head.iter_mut() {
                            *x *= s;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::Tensor;

    fn gate(rows: usize, cols: usize, act: GateAct, width: GateWidth) -> AttnGate {
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        AttnGate {
            proj: WeightMatrix::F32(Tensor::new(data, vec![rows, cols])),
            act,
            width,
        }
    }

    /// The two activations match ggml's scalar definitions, including
    /// softplus's branch at 20 (`unary-ops.cpp:80-82`), which a naive
    /// `ln(1 + exp(x))` overflows to `inf` at 89 and drifts from well
    /// before that.
    #[test]
    fn the_activations_are_ggml_s() {
        assert!((GateAct::Sigmoid.apply(0.0) - 0.5).abs() < 1e-7);
        assert!((GateAct::Sigmoid.apply(2.0) - 0.880_797).abs() < 1e-5);
        assert!((GateAct::Softplus.apply(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((GateAct::Softplus.apply(-30.0)).abs() < 1e-7);
        assert_eq!(GateAct::Softplus.apply(25.0), 25.0);
        assert_eq!(GateAct::Softplus.apply(100.0), 100.0);
    }

    /// A per-head gate scales every channel of a head by the same
    /// value, and a per-element gate scales each channel by its own --
    /// checked against a hand-written loop so the broadcast cannot be
    /// off by a head.
    #[test]
    fn per_head_broadcasts_over_head_dim_and_per_element_does_not() {
        let (n_heads, head_dim, hidden) = (3, 4, 5);
        let normed: Vec<f32> = (0..hidden).map(|i| 0.1 * i as f32 - 0.2).collect();
        let base: Vec<f32> = (0..n_heads * head_dim).map(|i| 1.0 + i as f32).collect();

        let ph = gate(n_heads, hidden, GateAct::Softplus, GateWidth::PerHead);
        let mut out = base.clone();
        ph.apply_rows(&normed, &mut out, 1, head_dim);
        let g = ph.proj.apply(&normed);
        for (h, &gh) in g.iter().enumerate() {
            for d in 0..head_dim {
                let i = h * head_dim + d;
                assert!((out[i] - base[i] * GateAct::Softplus.apply(gh)).abs() < 1e-6);
            }
        }

        let pe = gate(
            n_heads * head_dim,
            hidden,
            GateAct::Sigmoid,
            GateWidth::PerElement,
        );
        let mut out = base.clone();
        pe.apply_rows(&normed, &mut out, 1, head_dim);
        let g = pe.proj.apply(&normed);
        for i in 0..n_heads * head_dim {
            assert!((out[i] - base[i] * GateAct::Sigmoid.apply(g[i])).abs() < 1e-6);
        }
    }

    /// The batched body and the row body are one function: gating two
    /// rows at once equals gating each alone.
    #[test]
    fn a_batch_gates_each_row_as_the_row_body_would() {
        let (n_heads, head_dim, hidden) = (2, 3, 4);
        let g = gate(n_heads, hidden, GateAct::Sigmoid, GateWidth::PerHead);
        let normed: Vec<f32> = (0..2 * hidden).map(|i| (i as f32).cos()).collect();
        let base: Vec<f32> = (0..2 * n_heads * head_dim)
            .map(|i| i as f32 * 0.5)
            .collect();
        let mut batched = base.clone();
        g.apply_rows(&normed, &mut batched, 2, head_dim);
        for b in 0..2 {
            let mut row = base[b * 6..(b + 1) * 6].to_vec();
            g.apply_rows(&normed[b * hidden..(b + 1) * hidden], &mut row, 1, head_dim);
            assert_eq!(row, &batched[b * 6..(b + 1) * 6], "row {b}");
        }
    }

    /// The table is keyed by architecture, every row cites its lines,
    /// and the three GDN rows that share the tensor NAME are not in it.
    ///
    /// Six rows since 2026-09-19: `spark2_5`, `muse-glimmer` and
    /// `hrm_text` all landed upstream after the 2026-08-04 pin -- the
    /// first is `step35`'s pair with the tensor required and the other
    /// two are `afmoe`'s. `lfm3` made seven on 2026-10-28, `spark2_5`'s
    /// corner again.
    #[test]
    fn the_table_covers_the_softmax_gates_and_excludes_the_gdn_z_gates() {
        let names: Vec<&str> = ATTN_GATE_ARCHS.iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            [
                "afmoe",
                "laguna",
                "step35",
                "hrm_text",
                "muse-glimmer",
                "spark2_5",
                "lfm3"
            ]
        );
        for (arch, spec) in ATTN_GATE_ARCHS {
            assert!(spec.lines.contains(".cpp:"), "`{arch}` cites no line");
            assert!(!spec.widths.is_empty(), "`{arch}` admits no width");
        }
        for (arch, _) in GDN_Z_GATE_ARCHS {
            assert!(attn_gate_spec(arch).is_none(), "`{arch}` is a z gate");
        }
        assert!(attn_gate_spec("llama").is_none());
        assert_eq!(attn_gate_spec("laguna").unwrap().act, GateAct::Softplus);
        assert_eq!(
            attn_gate_spec("step35").unwrap().presence,
            GatePresence::Optional
        );
    }
}
