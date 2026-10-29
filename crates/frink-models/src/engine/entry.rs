//! The [`Engine`] seam, and the one place every engine behind it enters
//! the CPU worker pool.
//!
//! # Why this trait lives in its own file
//!
//! `decoder/entry.rs` holds the same rule for [`crate::decoder::Decoder`]:
//! **a forward pass enters the pool once, at its outermost boundary.**
//! #167 applied it to the generic decoder's ten entry points and stopped
//! there, so every dedicated engine -- Gemma-4, Kimi K3, GLM-5.2,
//! DeepSeek-V4 Pro, the MLA stack -- still paid the cold arm once per
//! parallel region.
//!
//! Those five engines have exactly one thing in common, and it is this
//! trait. So the wrapper is written ONCE, as [`Engine::forward_token`]'s
//! provided body, and an engine supplies only
//! [`Engine::forward_token_on_worker`]. There is no per-engine copy of
//! the wrapper to drift, and an engine added later is promoted without
//! its author having to know the rule exists.
//!
//! # What the wrapper buys
//!
//! `rayon::join` and the `par_iter` bridges cost very different things
//! depending on who calls them. From a rayon worker the caller runs one
//! half itself and waits on a spin latch; from any other thread the job
//! is injected and the caller blocks on a pthread condvar, contributing
//! no arithmetic while it sleeps. A decode step opens roughly five
//! parallel regions per layer, so a 30-layer model was paying ~150 of
//! the second kind per token, and the driving thread was measured
//! holding 74% of the token inside `__psynch_cvwait`.
//!
//! [`frink_core::par::on_workers`] turns those ~150 cold entries into
//! one, and [`frink_core::par::cold_regions`] is how a test says so
//! without a stopwatch. Nesting is free -- a call from inside another
//! `on_workers` returns directly -- so an engine whose body already
//! promotes ([`crate::decoder::Decoder`]) costs one extra branch and
//! nothing else.
//!
//! # The invariant this file exists to hold
//!
//! Nothing outside this file may declare `fn forward_token(`, because an
//! `impl Engine for _` that overrode the provided body would bypass the
//! wrapper for that engine alone while every other one still read as
//! fixed. `no_engine_may_override_the_promoted_forward_token` walks the
//! crate's sources and says so.

use frink_core::par;

/// A decoder that can run one incremental forward step given a token id
/// and position, updating its own per-layer state in place.
///
/// # What an implementor writes
///
/// [`Engine::forward_token_on_worker`], never [`Engine::forward_token`].
/// The second is provided, and providing it again is the one way to lose
/// the pool promotion silently; see the module docs.
///
/// # Why `Sync` and `State: Send`
///
/// [`Engine::forward_token`] hands `&self` and `&mut Self::State` to a
/// rayon worker for the duration of the step, which is exactly what
/// those two bounds say. They are stated on the trait rather than as a
/// `where` clause on the method so that an engine which cannot satisfy
/// them fails to compile at its `impl`, where the author can see why,
/// instead of at every call site of a method it silently would not have.
pub trait Engine: Sync {
    type State: Send;

    /// Builds fresh (empty) per-layer state for a new request.
    fn new_state(&self) -> Self::State;

    fn vocab_size(&self) -> usize;

    /// One decode step, already running on a rayon worker.
    ///
    /// This is the body an engine writes. Callers want
    /// [`Engine::forward_token`] instead: same computation, with the
    /// pool entered once for the whole step.
    fn forward_token_on_worker(
        &self,
        token_id: usize,
        pos: usize,
        state: &mut Self::State,
    ) -> Vec<f32>;

    /// Multiply-accumulates of the widest matvec inside one layer, LM
    /// head excluded, when the engine knows it.
    ///
    /// `Some` lets [`Engine::forward_token`] commit the whole step to
    /// one CPU scheduler ([`frink_core::par::decode_step`]); `None`, the
    /// default, keeps the per-operation rule. Only engines whose decode
    /// step has been counted under the per-step rule say `Some`.
    fn widest_layer_macs(&self) -> Option<usize> {
        None
    }

    /// One decode step for `token_id` at position `pos`.
    ///
    /// Enters the CPU worker pool once for the whole call, so every
    /// parallel region the step opens takes rayon's in-worker path. See
    /// the module docs for what that is worth, and
    /// [`frink_core::par::on_workers`] for the three cases it declines
    /// to promote (a backend whose thread-affine state `par::carry` does
    /// not carry, a pinned spin pool, and a caller already on a worker).
    ///
    /// Do not override this. The body is one line, and the only reason
    /// it is a provided method rather than a free function is that a
    /// free function could not be spelled `engine.forward_token(..)` by
    /// the generation loops that already spell it that way.
    fn forward_token(&self, token_id: usize, pos: usize, state: &mut Self::State) -> Vec<f32> {
        par::decode_step(self.widest_layer_macs(), move || {
            self.forward_token_on_worker(token_id, pos, state)
        })
    }
}

/// Whether [`par::on_workers`] promotes in this process at all.
///
/// Probed rather than restated: an empty step through
/// [`par::on_workers`] costs exactly one cold region when it promotes
/// and zero when it declines, so this asks the same predicate the
/// wrapper itself asks instead of keeping a second copy of "not pinned
/// to spin, and the active backend is CPU" beside it. `par::policy::
/// pinned` is `pub(crate)` to `frink-core` and cannot be asked
/// directly; a hand-written copy of its env-var spellings is how this
/// repo has repeatedly ended up with a gate that could not fire.
#[cfg(test)]
pub(crate) fn on_workers_promotes_here() -> bool {
    let before = par::cold_regions();
    par::on_workers(|| {});
    par::cold_regions() - before == 1
}

/// One decode step through `engine` costs exactly ONE entry into the CPU
/// worker pool, where the same step run as its raw body costs many.
///
/// The shared assertion behind every engine's own pool-entry test. It is
/// one function rather than one copy per engine for the reason this repo
/// keeps relearning: a copied check drifts, and a check that drifted
/// into asserting nothing still passes. Each engine supplies only the
/// instance, which is the only part that differs.
///
/// Both halves matter. `promoted == 1` is the claim; `raw > promoted` is
/// what makes the 1 mean anything, because a step that opened no regions
/// at all would also read as one.
#[cfg(test)]
pub(crate) fn assert_one_pool_entry_per_step<E: Engine>(engine: &E, token_id: usize) {
    if !on_workers_promotes_here() {
        return;
    }
    let mut state = engine.new_state();
    // Warm up outside the measurement: a weight matrix builds its repack
    // cache on first use, and that opens regions of its own.
    let _ = engine.forward_token_on_worker(token_id, 0, &mut state);

    let before = par::cold_regions();
    let _ = engine.forward_token_on_worker(token_id, 1, &mut state);
    let raw = par::cold_regions() - before;

    let before = par::cold_regions();
    let _ = Engine::forward_token(engine, token_id, 2, &mut state);
    let promoted = par::cold_regions() - before;

    assert_eq!(
        promoted, 1,
        "a decode step must enter the pool once, not once per matvec"
    );
    assert!(
        raw > promoted,
        "the unpromoted body must open a region per parallel section \
         ({raw} vs {promoted}); equal counts mean this engine's step is \
         too small to be measuring the promotion at all"
    );
}

#[cfg(test)]
mod tests {
    use super::{assert_one_pool_entry_per_step, on_workers_promotes_here, Engine};
    use frink_core::par;
    use std::path::{Path, PathBuf};

    /// Every `.rs` file under this crate's `src/`, found rather than
    /// listed. A restated list is the shape this repo keeps being bitten
    /// by: an engine added in a file nobody added to the list would be
    /// checked by nothing while the test still passed.
    fn crate_sources() -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("src/ is readable") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut out = Vec::new();
        walk(&src, &mut out);
        assert!(!out.is_empty(), "walked src/ and found no Rust at all");
        out
    }

    /// No engine may declare its own `forward_token`.
    ///
    /// The promoted entry point is provided by the trait, so an `impl
    /// Engine for _` that writes `fn forward_token` overrides it and
    /// opens a parallel region per matvec again -- for that engine
    /// alone, while every other one still reads as fixed. Nothing but a
    /// throughput measurement on a quiet host would notice.
    ///
    /// `decoder/entry.rs` is the one exemption: `Decoder::forward_token`
    /// there is an inherent method that does its own promotion, and the
    /// `Decoder` engine impl delegates to it.
    ///
    /// Sabotage: give any `impl Engine for _` a `fn forward_token` and
    /// this goes red naming the file and line.
    #[test]
    fn no_engine_may_override_the_promoted_forward_token() {
        let decoder_entry = Path::new("decoder").join("entry.rs");
        let engine_entry = Path::new("engine").join("entry.rs");
        let mut stray: Vec<String> = Vec::new();
        for path in crate_sources() {
            if path.ends_with(&decoder_entry) || path.ends_with(&engine_entry) {
                continue;
            }
            let body = std::fs::read_to_string(&path).expect("source reads");
            for (i, line) in body.lines().enumerate() {
                let t = line.trim_start();
                if t.starts_with("fn forward_token(") || t.starts_with("pub fn forward_token(") {
                    stray.push(format!("{}:{}", path.display(), i + 1));
                }
            }
        }
        assert!(
            stray.is_empty(),
            "these declarations bypass the pool wrapper on Engine::forward_token: {stray:?}"
        );
    }

    /// Three regions per "layer" over four "layers": twelve parallel
    /// regions if nothing promotes, one if the trait does. The shape
    /// every real engine has, without needing a checkpoint to build one.
    struct TwelveRegions;

    impl Engine for TwelveRegions {
        type State = Vec<f32>;

        fn new_state(&self) -> Vec<f32> {
            vec![0.0; 4096]
        }

        fn vocab_size(&self) -> usize {
            1
        }

        fn forward_token_on_worker(
            &self,
            _token_id: usize,
            _pos: usize,
            state: &mut Vec<f32>,
        ) -> Vec<f32> {
            for _ in 0..4 {
                for _ in 0..3 {
                    par::items_mut(state, 1, |_, x| *x += 1.0);
                }
            }
            vec![0.0]
        }
    }

    /// The provided body promotes, and promotes ONCE.
    ///
    /// Both halves matter. `promoted == 1` is the claim; `raw > promoted`
    /// is what makes the 1 mean anything, because a step that opened no
    /// regions at all would also read as one. This is the same
    /// before/after the dedicated engines could not be measured on
    /// directly: several of them have no checkpoint this machine can
    /// load, and the wrapper they share is this one.
    ///
    /// Sabotage: replace `par::on_workers(..)` in
    /// `Engine::forward_token` with a direct call to
    /// `forward_token_on_worker` and the promoted count becomes twelve.
    #[test]
    fn a_step_through_the_trait_enters_the_pool_once_and_the_raw_body_does_not() {
        assert_one_pool_entry_per_step(&TwelveRegions, 0);
    }

    /// The shared helper the engines' own tests call must be able to
    /// fail, or every one of those tests is decoration.
    ///
    /// An engine whose step opens ONE region is indistinguishable from a
    /// promoted one by the count alone, which is exactly the case
    /// `assert_one_pool_entry_per_step`'s second assertion exists to
    /// reject. Without it, an engine that stopped doing parallel work at
    /// all would still read as fixed.
    #[test]
    fn the_shared_assertion_rejects_a_step_too_small_to_measure() {
        struct OneRegion;

        impl Engine for OneRegion {
            type State = Vec<f32>;

            fn new_state(&self) -> Vec<f32> {
                vec![0.0; 64]
            }

            fn vocab_size(&self) -> usize {
                1
            }

            fn forward_token_on_worker(
                &self,
                _token_id: usize,
                _pos: usize,
                state: &mut Vec<f32>,
            ) -> Vec<f32> {
                par::items_mut(state, 1, |_, x| *x += 1.0);
                vec![0.0]
            }
        }

        if !on_workers_promotes_here() {
            return;
        }
        let caught = std::panic::catch_unwind(|| assert_one_pool_entry_per_step(&OneRegion, 0));
        assert!(
            caught.is_err(),
            "a one-region step must be refused as unmeasurable, not accepted as promoted"
        );
    }
}
