//! The output head's two invariants, held as types rather than as
//! coincidences between call sites.
//!
//! 1. [`Logits`]: a vocabulary of logits is softcapped exactly once, by
//!    the only constructor that can produce one. Gemma-2 caps at 30.0,
//!    so a path that projects and returns raw is not an error, just a
//!    quietly different distribution.
//!
//! 2. [`FoldedLmHead`]: `final_norm` + `lm_head` may be folded into a
//!    fused Metal decode stack ONLY when the stack's result is a greedy
//!    argmax token id. That result bypasses `Decoder::logits_from_normed`
//!    entirely, so `final_logit_softcap` never runs on it, and the ONLY
//!    reason that is sound is that `sc * tanh(x / sc)` is strictly
//!    increasing: capping cannot reorder logits, so it cannot move an
//!    argmax. On a full vocabulary of logits it changes every value.
//!
//! Before this module, invariant 2 held by accident. `decoder.rs`
//! computed `out_launch = if greedy_gpu { launch } else { None }` and
//! then passed `greedy_gpu && out_launch.is_some()` as the *separate*
//! `argmax_only` argument -- two hand-written expressions of one fact,
//! either of which could be edited without the other, and
//! `launch_decode_dense_stack(output: Some(..), argmax_only: false)` is
//! a representable call that downloads full uncapped logits which
//! `forward_token` then returned as-is. Nothing stated the coupling and
//! no test could see it break. Here the launch and the `argmax_only`
//! flag are one value with one constructor, and the result is
//! interpreted by [`FoldedLmHead::interpret`], which softcaps anything
//! vocabulary-shaped on the way out.

use frink_core::matmul::softcap_inplace;
use frink_core::weight_matrix::WeightMatrix;

/// A vocabulary of logits that has already had `final_logit_softcap`
/// applied (or has none to apply).
///
/// A newtype with a single constructor, so "these logits skipped the
/// cap" is a state that cannot be built. Every output-head path in
/// the decoder module goes through it.
pub(crate) struct Logits(Vec<f32>);

impl Logits {
    /// The one place the output head's two post-projection transforms
    /// are applied, in llama.cpp's order.
    ///
    /// `multiplier` is [`crate::ModelConfig::logit_multiplier`], the
    /// already-resolved Granite `1.0 / logit_scale` (`granite.cpp:180`),
    /// and it goes FIRST because it belongs to the head: llama.cpp
    /// scales the `build_lora_mm` result, and a softcap applied before
    /// it would be capping a differently-scaled distribution.
    ///
    /// No architecture declares both today -- Granite scales and does
    /// not cap, Gemma-2 caps and does not scale -- so the order has
    /// never been exercised by a real checkpoint. It is fixed here
    /// anyway rather than left to whichever call site runs first,
    /// because "no model does both" is exactly the kind of fact that
    /// stops being true without anybody editing this file.
    ///
    /// `bias` is [`crate::decoder::Decoder::output_bias`], `output.bias`
    /// (`phi2.cpp:136`, `qwen2.cpp:147-148`: `ggml_add(cur, output_b)`
    /// right after `build_lora_mm`), and it goes before both: it belongs
    /// to the matmul. `raw` may be `[batch, vocab]`; the bias is added to
    /// every row.
    pub(crate) fn from_output_head(
        mut raw: Vec<f32>,
        bias: Option<&[f32]>,
        softcap: Option<f32>,
        multiplier: Option<f32>,
    ) -> Self {
        if let Some(b) = bias {
            debug_assert!(!b.is_empty() && raw.len().is_multiple_of(b.len()));
            for row in raw.chunks_mut(b.len()) {
                for (v, bv) in row.iter_mut().zip(b) {
                    *v += bv;
                }
            }
        }
        if let Some(m) = multiplier {
            for v in raw.iter_mut() {
                *v *= m;
            }
        }
        if let Some(sc) = softcap {
            softcap_inplace(&mut raw, sc);
        }
        Logits(raw)
    }

    /// Project one final-normed hidden state through `head` and apply
    /// the two post-projection transforms. The only way to build a
    /// single-row `Logits` from a head, so that where the cap runs is
    /// decided here and nowhere else: with a cap and no multiplier it
    /// is [`WeightMatrix::apply_softcapped`], which folds the cap into
    /// the matvec's own command buffer on Metal; otherwise it is
    /// [`Self::from_output_head`] on the raw projection. No
    /// architecture declares both today; the order is pinned there.
    pub(crate) fn project(
        head: &WeightMatrix,
        x: &[f32],
        bias: Option<&[f32]>,
        softcap: Option<f32>,
        multiplier: Option<f32>,
    ) -> Self {
        match (bias, softcap, multiplier) {
            // The fused cap runs on the raw projection, so a bias, which
            // goes BEFORE the cap, keeps the head on the plain path.
            (None, Some(cap), None) => Logits(head.apply_softcapped(x, cap)),
            _ => Self::from_output_head(head.apply(x), bias, softcap, multiplier),
        }
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub(crate) fn into_vec(self) -> Vec<f32> {
        self.0
    }
}

/// Permission to fold `final_norm` + `lm_head` + `argmax` into a fused
/// Metal decode stack, carrying the launch it is permission for.
///
/// Generic over the launch type so this compiles and is testable
/// without the `metal` feature and without a device;
/// `decoder/forward_token.rs` instantiates it at `frink_metal::gpu::MatvecLaunch<'_>`.
///
/// The only constructor is [`Self::permit`], which refuses unless greedy
/// argmax is active for this thread. That is what makes
/// [`Self::argmax_only`] able to be a constant instead of a second
/// expression the caller has to keep in step.
/// Only the Metal decode stacks in `decoder/forward_token.rs` fold an
/// lm_head, so outside a
/// `metal` build this type has no caller -- but its tests are the ones
/// that pin the invariant, and they must run in the default build the
/// gates actually exercise.
#[cfg(any(feature = "metal", test))]
pub(crate) struct FoldedLmHead<L> {
    launch: L,
}

#[cfg(any(feature = "metal", test))]
impl<L> FoldedLmHead<L> {
    /// `Some` only when the stack is allowed to run lm_head on device:
    /// greedy argmax is active for this thread, the model's FINAL NORM
    /// is an RMSNorm the stack can bake in, the head has NO bias (no
    /// stack adds `output.bias`, and unlike the cap and the multiplier
    /// a bias is not monotone across the vocabulary, so it moves the
    /// argmax), AND the output head has a Metal launch. Any other
    /// combination keeps lm_head on the host, where
    /// `Decoder::logits_from_normed` applies all three.
    ///
    /// The final norm is a condition here rather than at the two call
    /// sites because what folds is `final_norm + lm_head + argmax`, one
    /// operation: `launch_moe_decode_stack` asserts outright that an
    /// on-device lm_head requires a final norm, and the dense stack's
    /// `final_norm_w` is derived from the same `NormOp`. `olmo` is the
    /// architecture that makes this reachable -- `olmo.cpp:128-130`
    /// norms with a null weight, so there is nothing to hand the kernel
    /// and the whole fold has to stay on the host.
    pub(crate) fn permit(
        greedy_argmax: bool,
        final_norm: &crate::norm::NormOp,
        output_bias: Option<&[f32]>,
        launch: Option<L>,
    ) -> Option<Self> {
        if !greedy_argmax || final_norm.rms_weights().is_none() || output_bias.is_some() {
            return None;
        }
        launch.map(|launch| FoldedLmHead { launch })
    }

    pub(crate) fn launch(&self) -> &L {
        &self.launch
    }

    /// Always `true`, and that is the point: the stack's `argmax_only`
    /// argument is now read off the same value that decided lm_head may
    /// fold at all, so the two cannot disagree. Folding with
    /// `argmax_only = false` returns a full uncapped vocabulary and is
    /// exactly the state this type exists to make unrepresentable.
    pub(crate) fn argmax_only(&self) -> bool {
        true
    }

    /// Turn a fused stack's on-device lm_head result into what
    /// `forward_token` returns.
    ///
    /// Both stacks document two shapes and return no others: a
    /// 1-element `vec![token_id as f32]` under `argmax_only`, or
    /// `vocab_size` logits otherwise. The vocabulary case is checked
    /// FIRST, which both removes the ambiguity at `vocab_size == 1` and
    /// means that the day either stack starts handing back logits from
    /// this path, they arrive capped instead of silently raw.
    ///
    /// The id case is passed through untouched, and must be: softcapping
    /// a token id would corrupt it. It is safe uncapped for the reason
    /// in this module's header -- the cap is monotone, so it cannot move
    /// an argmax -- and this is the only place that reasoning is relied
    /// upon. The same reasoning, and only that reasoning, is what makes
    /// `multiplier` safe to skip on an id: it is guaranteed POSITIVE by
    /// `scalar_multipliers::resolve`, so it cannot reorder a vocabulary
    /// either. A negative one could, which is why that function rejects
    /// it rather than trusting no checkpoint to declare one.
    pub(crate) fn interpret(
        &self,
        out: Vec<f32>,
        vocab_size: usize,
        softcap: Option<f32>,
        multiplier: Option<f32>,
    ) -> Vec<f32> {
        debug_assert!(
            multiplier.is_none_or(|m| m > 0.0),
            "a non-positive logit multiplier would reorder the vocabulary, so a folded \
             argmax id could not be passed through"
        );
        if out.len() == vocab_size {
            // No bias: `permit` refused the fold for a head that has one.
            Logits::from_output_head(out, None, softcap, multiplier).into_vec()
        } else {
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Softcap is `sc * tanh(x / sc)`, computed here rather than taken
    /// from `frink_core` so the assertions cannot agree with the code
    /// under test by sharing its bug.
    fn capped(x: f32, sc: f32) -> f32 {
        sc * (x / sc).tanh()
    }

    #[test]
    fn logits_cannot_be_built_without_the_cap_being_applied() {
        let out =
            Logits::from_output_head(vec![100.0, -100.0, 0.5], None, Some(30.0), None).into_vec();
        for (got, &raw) in out.iter().zip([100.0f32, -100.0, 0.5].iter()) {
            assert!(
                (got - capped(raw, 30.0)).abs() < 1e-5,
                "got {got} for raw {raw}"
            );
        }
        assert_eq!(
            Logits::from_output_head(vec![100.0, -100.0], None, None, None).into_vec(),
            vec![100.0, -100.0],
            "no cap configured must leave the head's output exactly alone"
        );
    }

    /// `project` is the one way to build single-row logits from a head,
    /// and with a cap it takes a different route (`apply_softcapped`,
    /// the Metal epilogue when there is one) from the raw projection
    /// plus `from_output_head`. The two routes must agree, or the cap
    /// would depend on which backend answered. Checked on the host
    /// here; the Metal epilogue is checked against the host kernel by
    /// `frink_metal::elem`'s hardware test.
    #[test]
    fn project_with_a_cap_agrees_with_the_raw_projection_capped_afterwards() {
        use frink_core::tensor::Tensor;
        let (rows, cols) = (5usize, 4usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.7).sin() * 40.0)
            .collect();
        let head = WeightMatrix::F32(Tensor::new(data, vec![rows, cols]));
        let x = [1.0f32, -2.0, 0.5, 3.0];
        let via_project = Logits::project(&head, &x, None, Some(30.0), None).into_vec();
        let via_raw = Logits::from_output_head(head.apply(&x), None, Some(30.0), None).into_vec();
        assert_eq!(via_project, via_raw);
        for v in &via_project {
            assert!(v.abs() < 30.0, "a capped logit is inside (-cap, cap): {v}");
        }
        // Without a cap there is only the raw route.
        assert_eq!(
            Logits::project(&head, &x, None, None, None).into_vec(),
            head.apply(&x)
        );
    }

    /// Granite's logit multiplier goes through the same constructor as
    /// the cap, and multiplies rather than divides.
    ///
    /// The direction is resolved at load time
    /// (`scalar_multipliers::resolve` inverts Granite's `logit_scale`),
    /// so a reader of THIS file cannot tell which way round it should
    /// be. Getting it backwards produces a perfectly ordered
    /// distribution at the wrong temperature -- no error, no NaN, just a
    /// model that samples differently from llama.cpp on the same file.
    #[test]
    fn the_logit_multiplier_multiplies_and_runs_before_the_cap() {
        assert_eq!(
            Logits::from_output_head(vec![8.0, -4.0, 1.0], None, None, Some(0.25)).into_vec(),
            vec![2.0, -1.0, 0.25]
        );

        // Order: cap AFTER the multiply. With a multiplier of 0.25 and a
        // cap of 3.0, `cap(0.25 * 100)` is 3.0 * tanh(25/3) and
        // `0.25 * cap(100)` would be 0.75 -- far enough apart that a
        // swapped order cannot pass.
        let got = Logits::from_output_head(vec![100.0], None, Some(3.0), Some(0.25)).into_vec();
        assert!(
            (got[0] - capped(25.0, 3.0)).abs() < 1e-5,
            "got {got:?}, want cap applied to the SCALED logit"
        );
        assert!(
            (got[0] - 0.25 * capped(100.0, 3.0)).abs() > 1.0,
            "the two orders must be distinguishable here"
        );
    }

    /// An RMSNorm final norm, for the tests that are about the other
    /// two conditions.
    fn rms() -> crate::norm::NormOp {
        crate::norm::NormOp::Rms(vec![1.0; 4])
    }

    /// Invariant 2's constructor. Folding lm_head into the stack when
    /// greedy argmax is NOT active would return a full vocabulary that
    /// bypasses `logits_from_normed` and therefore the cap.
    #[test]
    fn lm_head_folds_into_the_stack_only_under_greedy_argmax() {
        assert!(
            FoldedLmHead::permit(false, &rms(), None, Some(())).is_none(),
            "without greedy argmax the stack would return uncapped logits"
        );
        assert!(FoldedLmHead::permit(true, &rms(), None, None::<u32>).is_none());
        let folded = FoldedLmHead::permit(true, &rms(), None, Some(7u32))
            .expect("greedy + launch permits folding");
        assert_eq!(
            *folded.launch(),
            7,
            "the permission must carry the launch it was granted for"
        );
        assert!(
            folded.argmax_only(),
            "a permitted fold is an argmax fold; anything else returns raw logits"
        );
    }

    /// The hazard itself: if that path ever hands back a vocabulary
    /// instead of an id, Gemma-2's 30.0 cap must still apply.
    #[test]
    fn a_folded_stack_returning_logits_gets_them_softcapped() {
        let folded = FoldedLmHead::permit(true, &rms(), None, Some(())).unwrap();
        let vocab = 4;
        let raw = vec![100.0f32, -100.0, 31.0, 0.25];
        let got = folded.interpret(raw.clone(), vocab, Some(30.0), None);
        for (i, (g, r)) in got.iter().zip(raw.iter()).enumerate() {
            assert!(
                (g - capped(*r, 30.0)).abs() < 1e-4,
                "logit {i}: got {g}, expected {} (raw {r})",
                capped(*r, 30.0)
            );
        }
        assert!(
            got.iter().zip(raw.iter()).any(|(g, r)| (g - r).abs() > 1.0),
            "the cap must actually bite at these magnitudes, or this test \
             cannot tell a capped path from a raw one"
        );
    }

    /// The other half, and the one that stops the lazy fix: a greedy
    /// argmax id is NOT logits. Capping it would corrupt the token id --
    /// with a 30.0 cap, id 100 would come back as 29.98 and the caller
    /// would emit token 29.
    #[test]
    fn a_folded_stack_returning_an_argmax_id_is_passed_through_untouched() {
        let folded = FoldedLmHead::permit(true, &rms(), None, Some(())).unwrap();
        assert_eq!(
            folded.interpret(vec![100.0], 32_000, Some(30.0), None),
            vec![100.0],
            "a 1-element argmax id must not be softcapped"
        );
    }

    /// A model whose FINAL NORM has no RMS weights cannot fold, however
    /// greedy the caller is and whatever launch the head has.
    ///
    /// What folds is `final_norm + lm_head + argmax`, one operation:
    /// `launch_moe_decode_stack` asserts that an on-device lm_head
    /// requires a final norm, and the dense stack derives `final_norm_w`
    /// from the same `NormOp`. Before `olmo` no architecture could reach
    /// either non-RMS variant here, so both call sites wrote
    /// `Some(&self.final_norm)` and the MoE one hardcoded
    /// `final_norm_done_in_stack = true`.
    /// `output.bias` keeps lm_head on the host: no stack adds it, and a
    /// bias is not monotone across the vocabulary, so a folded argmax
    /// would be a different token.
    #[test]
    fn a_head_with_an_output_bias_cannot_fold() {
        let bias = [0.0f32, 3.0];
        assert!(FoldedLmHead::permit(true, &rms(), Some(&bias), Some(())).is_none());
        assert!(FoldedLmHead::permit(true, &rms(), None, Some(())).is_some());
    }

    /// The bias goes first, before the multiplier and the cap, and is
    /// added to every row of a batch.
    #[test]
    fn the_output_bias_is_added_per_row_before_the_other_transforms() {
        let bias = [1.0f32, -1.0];
        let out = Logits::from_output_head(vec![1.0, 1.0, 2.0, 2.0], Some(&bias), None, Some(2.0));
        assert_eq!(out.as_slice(), &[4.0, 0.0, 6.0, 2.0]);
    }

    #[test]
    fn a_non_rms_final_norm_cannot_fold_however_greedy_the_caller_is() {
        for norm in [
            crate::norm::NormOp::LayerNormNoParams,
            crate::norm::NormOp::RmsNoParams,
            crate::norm::NormOp::None,
        ] {
            assert!(
                FoldedLmHead::permit(true, &norm, None, Some(())).is_none(),
                "{norm:?}: the stack has no weights to bake the final norm from"
            );
        }
        assert!(
            FoldedLmHead::permit(true, &rms(), None, Some(())).is_some(),
            "an RMSNorm final norm still folds, or this test proves nothing"
        );
    }
}
