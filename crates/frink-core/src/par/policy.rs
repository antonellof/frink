//! Which CPU scheduler one operation's parallel regions run on, and the
//! work quantity that decides it.
//!
//! `FRINK_CPU_POOL=spin` is **+123% at 3B and +87% at 8B** on aarch64
//! and takes decode past llama.cpp; it is **-37% at 135M** on the same
//! host (#27, and the aarch64 table in
//! `docs/plans/cpu-cuda-parity.md`). A default that is right at 3B is
//! wrong at 135M and vice versa, so the switch cannot flip and cannot
//! stay: it needs a rule, and the rule has to be about the operation
//! rather than the process. That is step 3 of
//! `docs/plans/cpu-cuda-parity.md`.
//!
//! [`backend`] is that rule, and it is the ONLY place a region's
//! scheduler is read; [`decode_backend`] is the same constant applied
//! once per decode step, and [`backend`] defers to what it chose. Every
//! helper in [`crate::par`] calls it, so no call site
//! can grow its own opinion: `weight_matrix` had four spellings of one
//! GPU-router eligibility test and they drifted four ways, which is the
//! failure this module exists to make impossible.
//!
//! # Per operation inside a single-token CPU decode step was the bug
//!
//! Deciding per operation means one token uses both schedulers, and #27
//! measured that default at about the rayon number on Llama-3.2-3B, with
//! none of the pool's win. Counted rather than timed
//! (`frink-models/tests/decode_scheduler_choice.rs`, one layer at the
//! 3B widths, Q8_0, per-op default):
//!
//! - every layer alternated: an unpublished region (norm, attention) on
//!   rayon, the projections on the pool, back to rayon, back to the
//!   pool, about four to six scheduler changes per layer;
//! - Q, K and V are three branches of one `join3`, and the join itself
//!   published nothing, so it FORKED: three threads then submitted to
//!   the pool at once, the pool admitted one (it refuses a second
//!   submitter by design) and the other two projections fell back to
//!   rayon. Gate and up did the same through `join2`. Both schedulers
//!   were running regions at the same moment.
//!
//! What that costs is in how each idles. A pool worker that has just
//! finished a region spins (`spin_loop`, no yield) for
//! `FRINK_CPU_POOL_SPIN_US` (100 µs) before parking; a rayon worker
//! yields 32 times and then sleeps on a condvar. So a rayon region right
//! after a pool region runs while N-1 pool workers still burn the same
//! cores, and the pool region after it wakes workers that have parked:
//! the handoff the pool exists to avoid, paid at every boundary.
//!
//! So a single-token CPU decode step now picks ONE scheduler for the
//! whole step from the model's shape ([`decode_backend`]), and every
//! region inside it, joins included, follows that pick ([`backend`]).
//! That reproduces exactly the two configurations #27 measured, rayon
//! for a model below the crossover and the pool for one above it,
//! rather than a mixture nobody measured. The per-operation rule is
//! still what an operation outside such a step gets (prefill, and every
//! engine that has not said how wide its layers are).

use std::cell::Cell;

use super::Backend;

/// Multiply-accumulates below which one operation's parallel regions are
/// worth less than the persistent pool's wake-up, so they fork with
/// rayon instead.
///
/// **This value is BRACKETED by measurement, not measured.** What is
/// measured (#27, `docs/CONFIG.md`, quiet rented hosts, 2026-09-04) is
/// per MODEL, not per operation:
///
/// | host | 135M | 3B | 8B |
/// |---|---|---|---|
/// | 20-core Cortex-A725 (aarch64) | **-37%** | +123% | +87% |
/// | 10-core Xeon E5-2630 v4 (x86) | +49% | +23% | +15% |
///
/// Turning that into a per-operation number is arithmetic over the
/// shapes, not a sweep, and nobody has run the sweep:
///
/// | model | widest decode matvec | narrowest |
/// |---|---|---|
/// | SmolLM2-135M (576 / 1536) | 576 x 1536 = 0.88M | 576 x 576 = 0.33M |
/// | Llama-3.2-3B (3072 / 8192) | 3072 x 8192 = 25.2M | 3072 x 1024 = 3.1M |
/// | Llama-3.1-8B (4096 / 14336) | 4096 x 14336 = 58.7M | 4096 x 1024 = 4.2M |
///
/// So every operation in the model where the pool LOST is under 0.9M and
/// every operation in the models where it WON is over 3.1M. `1 << 21`
/// (2.1M) is the middle of that bracket in log space. The true crossover
/// is somewhere in `0.9M ..= 3.1M` and this constant is a guess inside
/// it; the exit criterion in `docs/plans/cpu-cuda-parity.md` asks for a
/// sweep on aarch64 AND x86, and that sweep is still owed.
///
/// One constant, both architectures. Above it the pool won on both;
/// below it the pool lost on aarch64 and WON on x86 (+49% at 135M), so
/// forking below the crossover leaves a measured x86 win unclaimed.
/// That is deliberate: an architecture-conditional default is what
/// `FRINK_CPU_INT_DOT` was, and it cost x86 between 4x and 8.8x of
/// decode before anyone noticed
/// (`weight_matrix::int_dot_is_a_win_here`). Claiming the x86 small-
/// model win needs the sweep, not a second `cfg!`.
///
/// `FRINK_CPU_POOL` stays as the A/B override precisely because of
/// all that: `rayon` restores the pre-rule behaviour exactly and `spin`
/// forces the pool at every size, so bracketing the constant is one
/// environment variable rather than two builds.
pub const SPIN_MIN_OP_MACS: usize = 1 << 21;

/// What `FRINK_CPU_POOL` pins, if anything. Read once, cached.
///
/// - `spin` / `persistent` / `1` / `on` / `true` — the pool, at every size
/// - `rayon` / `0` / `off` / `false` — fork-join, at every size
/// - unset / anything else — no pin: [`backend`] decides per operation
pub(crate) fn pinned() -> Option<Backend> {
    use std::sync::OnceLock;
    static PINNED: OnceLock<Option<Backend>> = OnceLock::new();
    *PINNED.get_or_init(|| {
        match std::env::var("FRINK_CPU_POOL")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("spin" | "persistent" | "1" | "on" | "true") => Some(Backend::Spin),
            Some("rayon" | "0" | "off" | "false") => Some(Backend::Rayon),
            _ => None,
        }
    })
}

thread_local! {
    /// `(rows, macs per row)` of the operation whose parallel regions the
    /// calling thread is about to open, published by [`with_op_work`].
    ///
    /// One value, two readers: [`op_macs`] multiplies it for the pool
    /// rule and [`macs_per_row`] takes half of it for rayon's task floor.
    /// They are derived from the same publish rather than restated
    /// beside each other, so a call site cannot tell the rule one size
    /// and the floor another.
    ///
    /// Read only on the thread that published it, before a region opens,
    /// so it never has to propagate into a worker.
    static OP_WORK: Cell<(usize, usize)> = const { Cell::new((0, 0)) };

    /// The scheduler the enclosing decode step committed to, set by
    /// [`with_forward`] on the step's own thread and carried into the
    /// branches of a rayon `join` by [`super::join2`] / [`super::join3`].
    /// `None` outside a step, where the per-operation rule decides.
    static FORWARD: Cell<Option<Backend>> = const { Cell::new(None) };
}

/// The scheduler the enclosing decode step committed to, if any.
pub(crate) fn forward() -> Option<Backend> {
    FORWARD.with(Cell::get)
}

/// Run `f` with the calling thread committed to `which` (or to no step,
/// for `None`), restoring the previous commitment afterwards, including
/// when `f` unwinds.
pub(crate) fn with_forward<R>(which: Option<Backend>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Backend>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORWARD.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FORWARD.with(|c| c.replace(which)));
    f()
}

/// **The per-step rule.** Which scheduler a single-token CPU decode step
/// runs ALL of its regions on, given the multiply-accumulates of the
/// widest matvec inside one of its layers (not the LM head).
///
/// The same [`SPIN_MIN_OP_MACS`], applied to the layer rather than to
/// each operation. The bracket behind that constant was measured per
/// MODEL anyway: every per-layer matvec of the model where the pool lost
/// is under 0.9M, and every one of the models where it won is over
/// 3.1M, so a model's widest layer matvec lands on the same side as all
/// of its others.
///
/// Why the layer and not the head: SmolLM2-135M's head is 576 x 49152 =
/// 28M, far over the crossover, and the pool measured -37% on that
/// model. A step that took the pool because of its head would be the
/// configuration that lost.
///
/// `FRINK_CPU_POOL` still pins it either way.
pub fn decode_backend(widest_layer_macs: usize) -> Backend {
    if let Some(pinned) = pinned() {
        return pinned;
    }
    if widest_layer_macs >= SPIN_MIN_OP_MACS {
        Backend::Spin
    } else {
        Backend::Rayon
    }
}

/// Publish the shape of the operation `f` performs: `rows` output rows,
/// each dotting `macs_per_row` elements.
///
/// Restores the previous value, so nesting is safe.
pub fn with_op_work<R>(rows: usize, macs_per_row: usize, f: impl FnOnce() -> R) -> R {
    let prev = OP_WORK.with(|c| c.replace((rows, macs_per_row)));
    let out = f();
    OP_WORK.with(|c| c.set(prev));
    out
}

/// Total multiply-accumulates of the published operation; `0` when
/// nothing published one.
fn op_macs() -> usize {
    OP_WORK.with(|c| {
        let (rows, per_row) = c.get();
        rows.saturating_mul(per_row)
    })
}

/// Multiply-accumulates per output row of the published operation; `0`
/// when nothing published one. Read by
/// [`crate::weight_matrix::WeightMatrix::min_rows_per_task`].
pub fn macs_per_row() -> usize {
    OP_WORK.with(|c| c.get().1)
}

/// **The one predicate.** Which scheduler the parallel regions of the
/// operation now being set up run on. In order:
///
/// 1. `FRINK_CPU_POOL`, when it pins one.
/// 2. Inside a pool task, the pool. [`crate::cpu_pool::CpuPool::run`]
///    runs a nested region inline, because the outer region already has
///    every core; forking rayon from a pool worker instead would put a
///    second set of threads on cores the pool is using.
/// 3. The enclosing decode step's commitment ([`decode_backend`]).
/// 4. Otherwise per operation: published work over
///    [`SPIN_MIN_OP_MACS`] takes the pool, everything else forks. An
///    unpublished operation (`op_macs() == 0`) reads as small, which is
///    deliberate: attention, sampling and the norm kernels do not
///    publish a matrix shape, and their regions are the ones the -37%
///    at 135M is made of.
pub fn backend() -> Backend {
    if let Some(pinned) = pinned() {
        return pinned;
    }
    if crate::cpu_pool::in_region() {
        return Backend::Spin;
    }
    if let Some(step) = forward() {
        return step;
    }
    if op_macs() >= SPIN_MIN_OP_MACS {
        Backend::Spin
    } else {
        Backend::Rayon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule is a function of the published work and nothing else, so
    /// it can be asserted without a process whose env var is set -- but
    /// only when nothing has pinned it, which is the CI case.
    fn unpinned() -> bool {
        pinned().is_none()
    }

    /// The two ends of the measured bracket land on different
    /// schedulers. This is the whole claim of #27 step 3: one operation
    /// the size of a 135M matvec and one the size of a 3B matvec must
    /// not get the same answer.
    ///
    /// Sabotage: return `Backend::Rayon` unconditionally from `backend`
    /// (or drop the `op_macs()` term) and the 3B row goes red.
    #[test]
    fn a_135m_matvec_and_a_3b_matvec_choose_different_schedulers() {
        if !unpinned() {
            return;
        }
        // SmolLM2-135M's widest decode matvec, 576 x 1536.
        with_op_work(1536, 576, || {
            assert_eq!(
                backend(),
                Backend::Rayon,
                "the pool is -37% at this size and must not be chosen"
            );
        });
        // Llama-3.2-3B's narrowest, 3072 x 1024.
        with_op_work(1024, 3072, || {
            assert_eq!(
                backend(),
                Backend::Spin,
                "the pool is +123% at this size and must be chosen"
            );
        });
    }

    /// An operation that published nothing is treated as small. Attention
    /// and sampling reach the helpers this way, and they are regions the
    /// pool measured WORSE on.
    #[test]
    fn work_that_was_never_published_forks_with_rayon() {
        if !unpinned() {
            return;
        }
        assert_eq!(op_macs(), 0);
        assert_eq!(backend(), Backend::Rayon);
    }

    /// The published shape has one source: the rule and rayon's task
    /// floor read the same cell. If they could be published separately,
    /// a call site could tell one a size the other never sees, which is
    /// this repo's dominant defect shape.
    #[test]
    fn the_rule_and_the_task_floor_read_one_published_shape() {
        with_op_work(64, 4096, || {
            assert_eq!(macs_per_row(), 4096);
            assert_eq!(op_macs(), 64 * 4096);
        });
        // And it is restored, so nesting cannot leak one op's size into
        // the next.
        assert_eq!(op_macs(), 0);
        assert_eq!(macs_per_row(), 0);
    }

    /// Nesting restores the outer shape rather than clearing it.
    #[test]
    fn a_nested_publish_restores_the_shape_it_replaced() {
        with_op_work(8, 8, || {
            with_op_work(4096, 4096, || {
                assert_eq!(op_macs(), 4096 * 4096);
            });
            assert_eq!(op_macs(), 64);
        });
    }

    /// A decode step's commitment overrides the per-operation rule in
    /// both directions, and is gone once the step ends.
    ///
    /// Sabotage: drop the `forward()` arm from `backend` and the first
    /// assertion goes red (an unpublished region inside a pool step would
    /// fork, which is the interleave #27 measured).
    #[test]
    fn a_decode_steps_commitment_overrides_the_per_operation_rule() {
        if !unpinned() {
            return;
        }
        with_forward(Some(Backend::Spin), || {
            assert_eq!(backend(), Backend::Spin, "unpublished, inside a pool step");
        });
        with_forward(Some(Backend::Rayon), || {
            with_op_work(4096, 4096, || {
                assert_eq!(backend(), Backend::Rayon, "wide, inside a rayon step");
            });
        });
        assert_eq!(forward(), None);
        assert_eq!(backend(), Backend::Rayon);
    }

    /// The step rule is the same crossover as the operation rule, read
    /// off the widest LAYER matvec: SmolLM2-135M's FFN (576 x 1536) forks
    /// and Llama-3.2-1B's (2048 x 8192) takes the pool.
    ///
    /// Sabotage: compare against `SPIN_MIN_OP_MACS * 64` and the 1B row
    /// goes red.
    #[test]
    fn the_step_rule_puts_135m_on_rayon_and_1b_on_the_pool() {
        if !unpinned() {
            return;
        }
        assert_eq!(decode_backend(576 * 1536), Backend::Rayon);
        assert_eq!(decode_backend(2048 * 8192), Backend::Spin);
        assert_eq!(decode_backend(3072 * 8192), Backend::Spin);
    }

    /// A region opened from inside a pool task stays on the pool (where
    /// `CpuPool::run` runs it inline) instead of forking rayon from a
    /// pool worker, whatever the per-operation rule would have said.
    ///
    /// Sabotage, run: drop the `in_region()` arm from `backend` and the
    /// unpublished region below reads `Rayon`.
    #[test]
    fn a_region_nested_in_a_pool_task_stays_on_the_pool() {
        if !unpinned() {
            return;
        }
        let pool = crate::cpu_pool::CpuPool::new(2);
        assert!(pool.run(4, &|_| assert_eq!(backend(), Backend::Spin)));
    }

    /// The commitment is restored when the step unwinds, so a panicking
    /// step cannot leave its thread committed to a scheduler for every
    /// operation that follows.
    #[test]
    fn a_panicking_step_does_not_leave_its_commitment_behind() {
        let caught = std::panic::catch_unwind(|| {
            with_forward(Some(Backend::Spin), || panic!("frink test panic"));
        });
        assert!(caught.is_err());
        assert_eq!(forward(), None);
    }

    /// A row count times a row width cannot be allowed to wrap into a
    /// small number and quietly pick the wrong scheduler.
    #[test]
    fn an_absurd_shape_saturates_instead_of_wrapping() {
        with_op_work(usize::MAX, usize::MAX, || {
            assert_eq!(op_macs(), usize::MAX);
        });
    }
}
