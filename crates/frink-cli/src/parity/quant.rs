//! Which quantization a parity verdict is about.
//!
//! One value, read by everything that needs to know: the DRIFT message
//! that explains a divergence, and [`super::calibration`] when it has to
//! decide whether a WRONG line exists at all. Deriving it twice was
//! [#109](https://github.com/antonellof/frink/issues/109).

/// The quantization of the per-layer weights, and whether llama.cpp
/// dots it against 8-bit-quantized ACTIVATIONS.
///
/// This exists because a `DRIFT` verdict on a K-quant is expected rather
/// than suspicious, and the old message sent the reader off to do a
/// per-layer divergence run on a difference that has a known cause.
///
/// ggml declares a `vec_dot_type` per quantization
/// (`ggml/src/ggml-cpu/ggml-cpu.c`). For the K-quants it is
/// `GGML_TYPE_Q8_K`: llama.cpp quantizes the activation to 8 bits and
/// accumulates in integers. Since #476 frink quantizes the activation
/// the same way, bit for bit; the reduction order of the integer dots
/// and the repack GEMMs still differs (ADR 0009's known deviations), so
/// these kinds can DRIFT by more than f32 summation order.
///
/// Measured on five quantizations of one checkpoint: the verdict tracked
/// this predicate on the 96 per-layer tensors exactly (before #476).
pub(super) fn llama_dots_this_against_q8k(kind: &str) -> bool {
    // Spelled as `GgmlType`'s own variant names (`Q4K`, not `Q4_K`) --
    // these come from `format!("{:?}", dtype)`, and writing the ggml
    // spelling here would have matched nothing while looking correct.
    matches!(
        kind,
        "Q2K"
            | "Q3K"
            | "Q4K"
            | "Q5K"
            | "Q6K"
            | "IQ2XXS"
            | "IQ2XS"
            | "IQ2S"
            | "IQ3XXS"
            | "IQ3S"
            | "IQ1S"
            | "IQ1M"
            | "IQ4XS"
    )
}

/// The most common quantization among a checkpoint's PER-LAYER tensors.
///
/// The FILENAME is not this, because the name is the quantization
/// RECIPE and a recipe falls back per tensor. Reading the tensor table
/// is the only way to know.
///
/// `Llama-3.2-1B-Instruct-IQ4_XS.gguf`, counted from its own header, is
/// **147 tensors: 96 IQ4_XS, 16 Q5_K (every `attn_v`), 1 Q6_K
/// (`token_embd`, which is also the tied head), 34 F32 norms**. So the
/// name gets the body right and says nothing about the other 17
/// quantized tensors, and this function answers `IQ4_XS`.
///
/// That census replaces a claim this comment used to make -- that the
/// same file "contains no IQ4_XS tensors at all" and is 96x `IQ4_NL` --
/// which was inverted. The count was right and the type was wrong, and
/// it mattered in the one direction that changes a verdict:
/// [`llama_dots_this_against_q8k`] lists `IQ4XS` and does NOT list
/// `IQ4NL`, so the comment described a file whose drift would be
/// unexplained while the real file's drift is the expected §10 one. A
/// reader who checked the claim against the file would have found the
/// opposite and had no reason to trust the rule it was illustrating.
///
/// Re-checkable in one line, which is why the numbers are here rather
/// than an assertion that it "falls back":
/// `frink inspect models/Llama-3.2-1B-Instruct-IQ4_XS.gguf`
/// (it truncates; the full census is a header parse).
///
/// The output head and the embedding table are EXCLUDED, so "body" here
/// means the body and cannot be outvoted into meaning the head on a
/// one-layer model. [`lm_head_quant`] is the other half, and
/// [`DominantQuant`] is the single place that weighs the two.
pub(super) fn body_quant(tensors: &[frink_gguf::TensorInfo]) -> Option<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in tensors {
        if t.name == "output.weight" || t.name == "token_embd.weight" {
            continue;
        }
        let kind = format!("{:?}", t.dtype);
        if kind != "F32" && kind != "F16" && kind != "BF16" {
            *counts.entry(kind).or_default() += 1;
        }
    }
    counts.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k)
}

/// Dtype of the logits projection: untied `output.weight`, else tied embed.
pub(super) fn lm_head_quant(tensors: &[frink_gguf::TensorInfo]) -> Option<String> {
    tensors
        .iter()
        .find(|t| t.name == "output.weight")
        .or_else(|| tensors.iter().find(|t| t.name == "token_embd.weight"))
        .map(|t| format!("{:?}", t.dtype))
}

/// WHICH QUANTIZATION'S ARITHMETIC DOMINATES THIS COMPARISON — the one
/// value the DRIFT message and the WRONG line's fallback both read.
///
/// Until [#109](https://github.com/antonellof/frink/issues/109) they
/// were two rules. The threshold keyed on the OUTPUT HEAD alone and the
/// message keyed on `lm_head.or(body)`, so a `Q8_0` head over a K-quant
/// body was judged against the line for a model containing no K-quant
/// arithmetic at all, and told to go run a per-layer divergence for a
/// difference §10 fully explains. They agreed on every checkpoint in
/// `models/`, which is why it never fired there — so a checkpoint of
/// that shape was BUILT to see what happens (`llama-quantize --pure
/// --output-tensor-type q8_0 --token-embedding-type q8_0 … Q4_K_S`),
/// against Homebrew libllama b7650:
///
/// | checkpoint | head | body | KL(llama‖frink) | verdict then |
/// |---|---|---|---|---|
/// | Qwen3-0.6B q8-head | Q8_0 | Q4_K | **1.297e-2** | **WRONG** |
/// | Qwen3-0.6B `--pure` | Q4_K | Q4_K | 1.975e-2 | DRIFT |
/// | Llama-3.2-1B q8-head | Q8_0 | Q4_K | 1.417e-3 | DRIFT |
/// | Llama-3.2-1B `--pure` | Q4_K | Q4_K | 1.126e-3 | DRIFT |
/// | Llama-3.2-1B q6-head | Q6_K | Q8_0 | **2.313e-4** | MATCH |
///
/// The first two rows are the same body arithmetic on the same
/// architecture, and the row with the MORE precise head read WRONG at a
/// SMALLER KL than the row with the K-quant head read DRIFT. `frink
/// parity` exited non-zero on the more accurate of the two.
///
/// The last row is what settles which tensor to key on: a K-quant head
/// over a Q8_0 body lands at 2.313e-4, three orders of magnitude below
/// its K-quant-bodied siblings and inside the MATCH floor. **The body
/// carries the divergence and the head barely contributes** — the body
/// is every layer, the head is one matvec — so the body is what this
/// answers with when it is Q8_K-dotted.
///
/// The head still gets to TRIGGER the relaxation when the body is not
/// Q8_K-dotted, even though the row above says the effect is tiny. The
/// alternative is a rule that can invent a WRONG for a checkpoint whose
/// only Q8_K arithmetic is in the head, which is the defect being
/// fixed, in the other direction; and the sensitivity given up sits at
/// 2.3e-4, four orders below either line, so nothing that could be
/// measured is being traded away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DominantQuant(Option<String>);

impl DominantQuant {
    /// Reads the tensor table. An empty slice — the file could not be
    /// opened — and a table with no quantized tensor both give `None`.
    pub(super) fn of(tensors: &[frink_gguf::TensorInfo]) -> Self {
        Self::weigh(
            lm_head_quant(tensors).as_deref(),
            body_quant(tensors).as_deref(),
        )
    }

    /// The rule itself, taking the two halves directly so it can be
    /// exercised on shapes no checkpoint on this disk has.
    pub(super) fn weigh(lm_head: Option<&str>, body: Option<&str>) -> Self {
        // A Q8_K-dotted BODY wins outright: it is every layer, and the
        // measurement above says it is what carries the divergence.
        // Otherwise the head, which still relaxes the line when it is
        // itself a K-quant — that fallback is the `or`, not a third
        // branch, because "the head when it is Q8_K-dotted" and "the
        // head as the plain label" are the same expression and writing
        // them separately gives one arm that can never be reached.
        let picked = if body.is_some_and(llama_dots_this_against_q8k) {
            body
        } else {
            lm_head.or(body)
        };
        Self(picked.map(str::to_owned))
    }

    /// What to call it in the report.
    pub(super) fn label(&self) -> Option<&str> {
        self.0.as_deref()
    }

    /// Whether llama.cpp dots this checkpoint's dominant arithmetic
    /// against Q8_K-quantized activations. The DRIFT message and the
    /// uncalibrated WRONG line both go through here, so they cannot
    /// disagree.
    pub(super) fn q8k_dotted(&self) -> bool {
        self.0.as_deref().is_some_and(llama_dots_this_against_q8k)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Every K-quant / IQ spelling that llama.cpp dots against Q8_K
    /// activations. Shared with [`super::super::calibration`]'s tests,
    /// which walk the same list against the WRONG line.
    pub(in crate::parity) const Q8K_DOTTED: &[&str] = &[
        "Q2K", "Q3K", "Q4K", "Q5K", "Q6K", "IQ2XXS", "IQ2XS", "IQ2S", "IQ3XXS", "IQ3S", "IQ1S",
        "IQ1M", "IQ4XS",
    ];

    /// Quantizations llama.cpp dots against Q8_0 activations instead.
    pub(in crate::parity) const Q8_0_DOTTED: &[&str] = &["Q8_0", "Q4_0", "Q5_0", "IQ4NL"];

    pub(super) fn tensor(name: &str, dtype: frink_gguf::GgmlType) -> frink_gguf::TensorInfo {
        frink_gguf::TensorInfo {
            name: name.to_string(),
            shape: vec![1],
            dtype,
            offset: 0,
        }
    }

    /// A `Q8_0` OUTPUT HEAD OVER A K-QUANT BODY IS A K-QUANT
    /// COMPARISON — the shape #109 is about.
    ///
    /// It shipped judged as a Q8_0 one, because the threshold read the
    /// head and never asked what the layers were. No checkpoint in
    /// `models/` has the shape, so one was built:
    /// `llama-quantize --pure --output-tensor-type q8_0
    /// --token-embedding-type q8_0 Qwen3-0.6B-BF16.gguf out.gguf
    /// Q4_K_S`. Against Homebrew libllama b7650 it measured KL
    /// **1.297e-2** and read **WRONG** — while the same model quantized
    /// `--pure` (K-quant head as well) measured a LARGER 1.975e-2 and
    /// read DRIFT.
    #[test]
    fn a_q8_0_head_over_a_kquant_body_is_judged_as_the_kquant_it_is() {
        for body in Q8K_DOTTED {
            let q = DominantQuant::weigh(Some("Q8_0"), Some(body));
            assert!(
                q.q8k_dotted(),
                "a {body} body dots against Q8_K activations whatever the output head is"
            );
            assert_eq!(q.label(), Some(*body), "the body is what the report names");
        }
        // When BOTH halves are Q8_K-dotted only the label distinguishes
        // the rules — and the body is the honest label, because it is
        // the layers that carry the drift (2.313e-4 for a K-quant head
        // alone). Keying the label on the head, as the message did
        // before #109, must be visible here rather than only in the
        // shape that changes the verdict.
        assert_eq!(
            DominantQuant::weigh(Some("Q6K"), Some("Q4K")).label(),
            Some("Q4K"),
            "with a K-quant on both ends the report names the body"
        );
    }

    /// The head can still TRIGGER the allowance, so no row can move
    /// DRIFT → WRONG because of #109's fix.
    ///
    /// Measured cost of keeping it: a Q6_K head over a Q8_0 body
    /// (`--pure Q8_0 --output-tensor-type q6_K`) lands at KL 2.313e-4,
    /// four orders below either line, so the sensitivity being conceded
    /// is not sensitivity anything could use.
    #[test]
    fn a_kquant_head_over_a_q8_0_body_still_relaxes_rather_than_inventing_a_wrong() {
        let q = DominantQuant::weigh(Some("Q6K"), Some("Q8_0"));
        assert!(q.q8k_dotted());
        assert_eq!(q.label(), Some("Q6K"));
    }

    /// The body is read off the LAYERS, and the output head cannot vote
    /// in it.
    ///
    /// `body_quant` counts tensors, so on a checkpoint with few layers
    /// and a head at a different precision the head could otherwise
    /// join the tally it is being weighed against — two roles for one
    /// tensor, which is how the head came to decide the threshold in
    /// the first place.
    #[test]
    fn the_body_is_read_off_the_layers_and_the_output_head_does_not_vote_in_it() {
        use frink_gguf::GgmlType;
        // One layer, so the head would outvote it if it were counted.
        let table = vec![
            tensor("token_embd.weight", GgmlType::Q8_0),
            tensor("output.weight", GgmlType::Q8_0),
            tensor("blk.0.attn_q.weight", GgmlType::Q4K),
            tensor("blk.0.attn_norm.weight", GgmlType::F32),
        ];
        assert_eq!(body_quant(&table).as_deref(), Some("Q4K"));
        assert_eq!(lm_head_quant(&table).as_deref(), Some("Q8_0"));
        let q = DominantQuant::of(&table);
        assert_eq!(
            q.label(),
            Some("Q4K"),
            "a Q8_0 head does not hide a Q4_K body"
        );
        assert!(q.q8k_dotted());

        // A tied-embedding model: no `output.weight`, and the embedding
        // is the head. It still must not count as the body.
        let tied = vec![
            tensor("token_embd.weight", GgmlType::Q8_0),
            tensor("blk.0.ffn_up.weight", GgmlType::Q4K),
        ];
        assert_eq!(body_quant(&tied).as_deref(), Some("Q4K"));
        assert!(DominantQuant::of(&tied).q8k_dotted());

        // An unreadable file relaxes nothing.
        assert_eq!(DominantQuant::of(&[]).label(), None);
        assert!(!DominantQuant::of(&[]).q8k_dotted());
    }

    /// The Q8_K predicate is spelled in `GgmlType`'s variant names, not
    /// ggml's.
    ///
    /// `format!("{:?}", dtype)` yields `Q4K`, and ggml calls the same
    /// thing `Q4_K`. Writing the ggml spelling here matches NOTHING
    /// while looking exactly right, and the symptom is the generic
    /// "go run a per-layer divergence" message on every K-quant — which
    /// is the message this predicate exists to suppress. That mistake
    /// was made once already; this is what catches it.
    #[test]
    fn the_q8k_predicate_uses_the_dtype_debug_spelling() {
        for kind in Q8K_DOTTED {
            assert!(
                llama_dots_this_against_q8k(kind),
                "{kind} declares vec_dot_type = Q8_K in ggml-cpu.c"
            );
        }
        // The ggml spelling must NOT match, or the underscore bug is
        // back and invisible.
        for wrong in ["Q4_K", "Q6_K", "IQ4_XS"] {
            assert!(
                !llama_dots_this_against_q8k(wrong),
                "{wrong} is ggml's spelling, not GgmlType's -- if this now matches, the \
                 predicate is accepting both and the next reader cannot tell which is real"
            );
        }
        // Q8_0-dotted quants must stay out: these are the ones that
        // MATCH, and claiming the divergence is expected for them would
        // excuse a real bug.
        for q8_0_dotted in Q8_0_DOTTED {
            assert!(!llama_dots_this_against_q8k(q8_0_dotted));
        }
    }

    /// `IQ4XS` and `IQ4NL` are one character apart and land on OPPOSITE
    /// sides of this predicate. Named as a pair, because the tests above
    /// cannot catch them being swapped.
    ///
    /// `the_q8k_predicate_uses_the_dtype_debug_spelling` walks whatever
    /// [`Q8K_DOTTED`] and [`Q8_0_DOTTED`] happen to contain, so moving
    /// `IQ4NL` from one list to the other keeps every assertion above
    /// green while inverting what the tool says about a real
    /// checkpoint. This pins the two to their ggml facts directly.
    ///
    /// It is worth its runtime because the swap has already happened in
    /// prose: `body_quant`'s doc claimed
    /// `Llama-3.2-1B-Instruct-IQ4_XS.gguf` was 96x `IQ4_NL` and contained
    /// no `IQ4_XS` at all. Counted from the file's own header it is 96x
    /// `IQ4_XS` and zero `IQ4_NL` -- the count right, the type inverted,
    /// and inverted across exactly this line, so the comment described a
    /// file whose DRIFT would be unexplained while the real one's is the
    /// expected §10 case.
    ///
    /// Sabotage: move `"IQ4XS"` out of the `matches!` in
    /// [`llama_dots_this_against_q8k`], or add `"IQ4NL"` to it.
    #[test]
    fn iq4_xs_is_q8k_dotted_and_iq4_nl_is_not() {
        assert!(
            llama_dots_this_against_q8k("IQ4XS"),
            "ggml declares vec_dot_type = Q8_K for IQ4_XS, so its drift is expected"
        );
        assert!(
            !llama_dots_this_against_q8k("IQ4NL"),
            "ggml declares vec_dot_type = Q8_0 for IQ4_NL, so a drift there is NOT \
             excused by §10 and would be a real finding"
        );
    }
}
