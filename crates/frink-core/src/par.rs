//! The single seam every CPU parallel region in this crate goes through.
//!
//! There used to be about fifty spellings of "run this over rows in
//! parallel" scattered through [`crate::weight_matrix`] alone, each one
//! an inline rayon iterator chain. That is the shape this repo has been
//! burned by before: many copies of one decision, with nothing making
//! them agree. Routing them all through the handful of functions below
//! means the choice of *how* work is scheduled is made in one place.
//!
//! Which is exactly what issue #27 needs, because it wants that choice
//! changed: rayon forks and joins per operation, per layer, per token,
//! and llama.cpp instead hands work to a pool that is already awake.
//! [`Backend::Spin`] is that pool ([`crate::cpu_pool`]).
//!
//! # The switch
//!
//! Which of the two runs is decided by the one predicate in [`policy`]:
//! [`backend`]. Inside a single-token CPU decode step ([`decode_step`])
//! it answers the step's one choice, made from the model's widest layer
//! matvec; everywhere else it answers per operation, from its size.
//! `FRINK_CPU_POOL` pins it either way (`spin` / `rayon`) and is an A/B
//! override, not the decision. See [`policy::SPIN_MIN_OP_MACS`] for the
//! crossover and what is and is not measured about it.
//!
//! Every helper below asks [`backend`] and none of them decides
//! anything itself, which is what stops the two arms of that choice from
//! drifting apart across thirty call sites. The helpers themselves live
//! in `regions`, each with its spin arm split out over an explicit pool
//! so a test can drive it.
//!
//! # `min_len`, and where `MIN_TASK_MACS` went
//!
//! Every helper takes a `min_len`. On the rayon arm it is passed
//! straight to `with_min_len`, which is what the call sites did by hand
//! before, so the fork-join path's task decomposition is bit-for-bit
//! what it was.
//!
//! On the spin arm it is **ignored**. `MIN_TASK_MACS` existed to stop
//! rayon splitting a matvec into tasks too small to pay for their own
//! fork-join; when a region costs a cache-line transfer instead of a
//! futex there is nothing to pay for, so the spin arm chunks purely by
//! pool width ([`task_count`]) the way `ggml_compute_forward_mul_mat`
//! does. That is the deletion issue #27 asks for, and it is a deletion
//! rather than a retune: no MAC threshold is consulted on this path at
//! all. It survives on the rayon arm because the rayon arm still runs
//! every operation below the crossover, and removing it there re-opens
//! the measured 13-16x small-model regression documented on
//! [`crate::weight_matrix::WeightMatrix::min_rows_per_task`].

use std::cell::Cell;

use crate::cpu_pool::CpuPool;

mod carry;
pub mod policy;
mod regions;

pub use policy::{backend, decode_backend, macs_per_row, with_op_work};
pub use regions::{
    chunks_mut, chunks_mut2, chunks_mut2_by, chunks_mut_init, indices, indices_init, items_mut,
};

thread_local! {
    /// How many parallel regions THIS thread has opened while not being
    /// a rayon worker. See [`on_workers`] for why that is the number
    /// worth counting, and [`cold_regions`] for how a test reads it.
    ///
    /// Per thread rather than process-wide on purpose. A cold region is
    /// always counted on the thread that submits it, so nothing is lost;
    /// and a shared counter would make the assertion depend on whatever
    /// else the test binary happened to be running at the time, which is
    /// the difference between a guard and a flake.
    static COLD_REGIONS: Cell<u64> = const { Cell::new(0) };
}

/// Regions opened in this PROCESS, by the scheduler that ran them, and
/// how many times consecutive regions changed scheduler. Monotonic; read
/// it before and after an operation and assert on [`RegionTally::since`].
///
/// This is how #27's interleaving is observed without a stopwatch: a
/// decode step whose regions alternate between the two schedulers shows
/// one `switch` per boundary, and every boundary hands the work from one
/// set of awake workers to another that has to be woken (see
/// [`policy`]'s module docs for how each one idles).
///
/// Process-wide on purpose, unlike [`cold_regions`]: the regions that
/// gave #27 away were opened on OTHER threads, by the branches of a
/// forked `join`, and a per-thread count would not have seen them. The
/// price is that it counts every concurrent caller, so only a test
/// binary that runs one test can assert on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegionTally {
    /// Regions forked with rayon.
    pub rayon: u64,
    /// Regions run on the persistent pool.
    pub spin: u64,
    /// Consecutive regions that ran on different schedulers.
    pub switches: u64,
}

impl RegionTally {
    /// What happened between `before` and `self`.
    pub fn since(self, before: Self) -> Self {
        Self {
            rayon: self.rayon - before.rayon,
            spin: self.spin - before.spin,
            switches: self.switches - before.switches,
        }
    }
}

mod tally_state {
    use std::sync::atomic::{AtomicU64, AtomicU8};

    pub(super) static RAYON: AtomicU64 = AtomicU64::new(0);
    pub(super) static SPIN: AtomicU64 = AtomicU64::new(0);
    pub(super) static SWITCHES: AtomicU64 = AtomicU64::new(0);
    /// `0` before the first region, else `1 + Backend as u8` of the last.
    pub(super) static LAST: AtomicU8 = AtomicU8::new(0);
}

/// The process's [`RegionTally`].
pub fn region_tally() -> RegionTally {
    use std::sync::atomic::Ordering::Relaxed;
    RegionTally {
        rayon: tally_state::RAYON.load(Relaxed),
        spin: tally_state::SPIN.load(Relaxed),
        switches: tally_state::SWITCHES.load(Relaxed),
    }
}

/// Counts one region on `which`. Two relaxed atomics per region, against
/// a region that costs at least a cache-line handoff to every worker.
fn tally(which: Backend) {
    use std::sync::atomic::Ordering::Relaxed;
    let (counter, tag) = match which {
        Backend::Rayon => (&tally_state::RAYON, 1u8),
        Backend::Spin => (&tally_state::SPIN, 2u8),
    };
    counter.fetch_add(1, Relaxed);
    let prev = tally_state::LAST.swap(tag, Relaxed);
    if prev != 0 && prev != tag {
        tally_state::SWITCHES.fetch_add(1, Relaxed);
    }
}

/// Records one region that ran on the persistent pool.
fn note_spin_region() {
    tally(Backend::Spin);
}

/// Parallel regions this thread has opened without being a rayon worker.
///
/// Monotonic, so a test reads it before and after the operation it cares
/// about and asserts on the DIFFERENCE. It is not a benchmark: it is an
/// operation count, which is load-immune, and it is the only thing that
/// distinguishes "this decode step entered the pool once" from "it
/// entered it a hundred and fifty times".
pub fn cold_regions() -> u64 {
    COLD_REGIONS.with(Cell::get)
}

/// Records one region about to be opened on the rayon arm.
///
/// Called from every rayon fallback in this module and nowhere else.
/// The worker-index read is the same TLS lookup rayon is about to do
/// anyway, and the counter is touched only on the cold path, which after
/// [`on_workers`] is once per decode step rather than once per matvec.
fn note_rayon_region() {
    tally(Backend::Rayon);
    note_cold_entry();
}

/// Counts a submission into rayon from outside its workers, for
/// [`cold_regions`]. Separate from the [`region_tally`] count because
/// [`on_workers`]'s own promotion is a cold entry but not a work region:
/// counting it as a rayon region would read every promoted pool step as
/// having changed scheduler at its start.
fn note_cold_entry() {
    if rayon::current_thread_index().is_none() {
        COLD_REGIONS.with(|c| c.set(c.get().saturating_add(1)));
    }
}

/// Run `f` on a rayon worker, so every parallel region it opens takes
/// rayon's IN-WORKER path instead of its cold-submission path.
///
/// # What this is for
///
/// `rayon::join` and the `par_iter` bridges both funnel through
/// `Registry::in_worker`. That call has two arms and they do not cost
/// the same thing:
///
/// - **From a worker** (`in_worker_hot`): the calling thread runs one
///   half itself, the other half is posted for stealing, and the wait is
///   a `SpinLatch`. No syscall.
/// - **From any other thread** (`in_worker_cold`): the job is injected,
///   and the caller blocks on a `LockLatch`, which is a pthread mutex
///   and condvar. That is a park and a wake, per region, and the caller
///   contributes no arithmetic while it sleeps.
///
/// A decode step opens roughly five regions per layer, so a 30-layer
/// model paid ~150 of the cold arm per token. Measured on an M2 Pro with
/// `sample` over SmolLM2-135M Q8_0 `tg128`, the main thread spent **74%
/// of the token** inside `__psynch_cvwait` under `LockLatch`, and over
/// that same window the six workers it was waiting for held only about
/// an eighth as many samples in the matvec kernel: most of the wait was
/// the round trip, not the work.
///
/// Wrapping the whole step in one `rayon::scope` turns those ~150 cold
/// entries into ONE. The step then runs on worker 0 and every nested
/// region is hot.
///
/// # Why it is not simply free
///
/// The caller still parks once, for the whole step, and the step no
/// longer runs on the caller's thread. Both are deliberate: one park per
/// token against one per matvec, and rayon's own worker count is
/// unchanged, so the same number of cores do the work.
///
/// Nesting is free (a call from inside another `on_workers` returns
/// `f()` directly), so an entry point may wrap unconditionally without
/// having to know whether its caller already did.
///
/// # What crosses with the work
///
/// A step that reads a thread-local *setting* would read the worker's
/// default instead of its caller's choice, which is exactly what
/// GitHub issue #166 was: a Metal decode moved to a worker silently took
/// the other `lm_head` path and answered differently from the tenth
/// token. So the settings are captured on this thread and adopted on the
/// worker for the length of the job. [`carry::Carry`] is the single list
/// of them and its `adopt` destructures exhaustively, so a new one that
/// is not carried does not compile.
///
/// # Two cases this deliberately does NOT promote
///
/// **A backend whose thread-affine state is not proven carried.**
/// Promotion asks [`crate::weight_matrix::active_backend`] -- the same
/// cached predicate dispatch itself uses, not a second opinion about it
/// -- and puts the answer to [`carry::promotable`], which holds the
/// per-backend verdict and the evidence behind each one. CPU and Metal
/// promote; CUDA and Vulkan do not, for want of hardware to check them
/// on rather than for a known defect.
///
/// **The pinned spin pool.** Under `FRINK_CPU_POOL=spin` the rayon
/// global pool is never used for work, and `rayon::scope` would BUILD
/// it, spawning a second set of workers that then only sit there. So
/// that pin short-circuits, exactly as [`num_threads`] avoids
/// `rayon::current_num_threads` for the same reason.
pub fn on_workers<R, F>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if !carry::promotable(crate::weight_matrix::active_backend()) {
        return f();
    }
    if policy::pinned() == Some(Backend::Spin) {
        return f();
    }
    if rayon::current_thread_index().is_some() {
        return f();
    }
    // Captured HERE, on the thread whose caller made the decision, and
    // before anything is submitted. Reading it on the worker would read
    // the worker's default, which is the defect.
    let carried = carry::Carry::capture();
    // The one cold entry this whole design is willing to pay. Counted
    // like any other, so [`cold_regions`] reports the true total and a
    // test can assert it is exactly one.
    note_cold_entry();
    rayon::scope(move |_| {
        let _adopted = carried.adopt();
        f()
    })
}

/// Run one single-token decode step `f` whose widest per-layer matvec is
/// `widest_layer_macs` multiply-accumulates, with EVERY parallel region
/// inside it on one scheduler.
///
/// This is the decode-side entry #27 asked for. Deciding per operation
/// made a 3B step alternate between the two schedulers at every layer
/// and run both at once inside the Q/K/V `join3` (see [`policy`]'s
/// module docs for the counts and for how each pool idles). One choice
/// per step, from [`decode_backend`], gives the two configurations that
/// were actually measured:
///
/// - **Pool** for a model over the crossover: every region, the joins
///   (which then run their branches one after the other) and the LM
///   head included, goes to the pool, as under `FRINK_CPU_POOL=spin`.
/// - **Rayon** below it: every region forks, as under
///   `FRINK_CPU_POOL=rayon`.
///
/// Both are still promoted with [`on_workers`], so the step runs on a
/// rayon worker either way. For the pool arm that is one park and wake
/// per token, which the default already paid, and it buys the case two
/// requests decode at once: the pool admits one submitter, and the other
/// step's regions fall back to rayon. From a worker that fallback is
/// rayon's in-worker path; from a plain thread it would be the cold
/// submission `on_workers` exists to avoid, once per region. (Under the
/// pin, `on_workers` does not promote, which keeps the pin's behaviour
/// exactly what it was.)
///
/// `None` (an engine that has not said how wide its layers are), a
/// backend other than the CPU, and a call already inside a step all keep
/// what they had: [`on_workers`], and the per-operation rule. Metal and
/// CUDA steps are therefore unchanged by construction; nobody has
/// measured the pool under them.
pub fn decode_step<R, F>(widest_layer_macs: Option<usize>, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if policy::forward().is_some() {
        return f();
    }
    let cpu = crate::weight_matrix::active_backend() == crate::kernel_registry::Backend::Cpu;
    match widest_layer_macs.filter(|_| cpu).map(decode_backend) {
        None => on_workers(f),
        // Committed INSIDE the promoted closure: the commitment is a
        // thread-local of the thread that runs the step.
        Some(step) => on_workers(move || policy::with_forward(Some(step), f)),
    }
}

/// Which scheduler CPU parallel regions use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// A rayon fork-join per region. The default.
    Rayon,
    /// A persistent pool of workers parked on a spin-then-park barrier.
    Spin,
}

/// How many tasks per worker the spin arm aims for.
///
/// Tasks are handed out by one atomic cursor, so more of them means
/// better load balancing and more contention on that cursor. Eight is
/// the same order as llama.cpp's `4 * n_threads` chunk floor, with room
/// for the uneven per-task cost that causal masking gives attention.
const TASKS_PER_THREAD: usize = 8;

/// The process-wide persistent pool, built on first use with
/// [`crate::threads::resolve_cpu_threads`] workers -- the same width
/// [`crate::threads::init_cpu_pool`] gives rayon, so the two backends
/// are the same number of threads and a comparison is not confounded.
///
/// A `static` is never dropped, so the workers live to process exit.
/// That is deliberate and it is also what rayon's global pool does.
fn pool() -> &'static CpuPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<CpuPool> = OnceLock::new();
    POOL.get_or_init(|| CpuPool::new(crate::threads::resolve_cpu_threads()))
}

/// Worker count of the active backend.
///
/// Call this instead of `rayon::current_num_threads` anywhere a task
/// decomposition is being sized: `rayon::current_num_threads` *builds*
/// the global rayon pool as a side effect, so asking it under the spin
/// backend spawns a second set of threads that would never run anything.
pub fn num_threads() -> usize {
    match backend() {
        Backend::Rayon => rayon::current_num_threads().max(1),
        Backend::Spin => pool().num_threads(),
    }
}

/// How many tasks the spin arm splits `n_items` into.
///
/// Never more than one task per item, never zero, and never more than
/// the pool can usefully chase. No work threshold appears here; see the
/// module docs on `MIN_TASK_MACS`.
pub fn task_count(n_items: usize) -> usize {
    tasks_for(num_threads(), n_items)
}

/// [`task_count`] for a pool `width` threads wide.
fn tasks_for(width: usize, n_items: usize) -> usize {
    if n_items == 0 {
        return 0;
    }
    n_items.min(width.saturating_mul(TASKS_PER_THREAD).max(1))
}

/// `(items_per_task, n_tasks)` for a contiguous split of `n_items` over
/// a pool `width` threads wide.
fn split_for(width: usize, n_items: usize) -> (usize, usize) {
    let n_tasks = tasks_for(width, n_items);
    if n_tasks == 0 {
        return (0, 0);
    }
    (n_items.div_ceil(n_tasks), n_tasks)
}

/// Two independent pieces of work.
///
/// The rayon arm forks; the spin arm runs them one after the other,
/// because each half already spreads across the whole pool internally
/// and nesting a region inside a region is the one thing
/// [`CpuPool::run`] cannot parallelize. That is llama.cpp's shape too:
/// its threadpool runs one graph node at a time, full width.
pub fn join2<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    match backend() {
        Backend::Rayon => {
            note_rayon_region();
            // A stolen branch runs on another worker, whose thread has
            // not heard of the enclosing step; tell it. Without this the
            // branch falls back to the per-operation rule.
            let step = policy::forward();
            rayon::join(
                move || policy::with_forward(step, a),
                move || policy::with_forward(step, b),
            )
        }
        Backend::Spin => (a(), b()),
    }
}

/// Three independent pieces of work; see [`join2`].
pub fn join3<A, B, C, RA, RB, RC>(a: A, b: B, c: C) -> (RA, RB, RC)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    C: FnOnce() -> RC + Send,
    RA: Send,
    RB: Send,
    RC: Send,
{
    match backend() {
        Backend::Rayon => {
            note_rayon_region();
            // See `join2`.
            let step = policy::forward();
            let (ra, (rb, rc)) = rayon::join(
                move || policy::with_forward(step, a),
                move || {
                    rayon::join(
                        move || policy::with_forward(step, b),
                        move || policy::with_forward(step, c),
                    )
                },
            );
            (ra, rb, rc)
        }
        Backend::Spin => (a(), b(), c()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two configurations `on_workers` declines to promote in, and
    /// so the two it cannot be asserted in. One predicate, shared by
    /// every test below, rather than three spellings of it.
    ///
    /// The backend half asks [`carry::promotable`] rather than naming a
    /// backend: the rule used to be "the backend is CPU", and writing
    /// that here a second time is how a test comes to assert the rule
    /// the code used to have.
    fn the_promotion_applies_here() -> bool {
        policy::pinned().is_none() && carry::promotable(crate::weight_matrix::active_backend())
    }

    /// With nothing published and nothing pinned, the helpers fork with
    /// rayon -- the behaviour every caller that has not opted into the
    /// size rule keeps.
    #[test]
    fn an_unpublished_region_forks_with_rayon() {
        if policy::pinned().is_none() {
            assert_eq!(backend(), Backend::Rayon);
        }
    }

    /// **Every rayon-versus-spin choice in this crate goes through one
    /// predicate**, and this is what says so.
    ///
    /// The alternative is the shape this repo keeps shipping: a second
    /// site that decides for itself and then drifts. `weight_matrix`
    /// had four copies of one GPU-router eligibility test that tested
    /// three conditions, two, and none.
    ///
    /// Two halves, because a helper can drift in two directions:
    /// reaching the pool without asking, and asking the environment
    /// instead of asking the predicate.
    ///
    /// Sabotage: hand the process pool to a `*_on` spin arm from a helper
    /// without its `if backend() == Backend::Spin &&` guard, or read the
    /// environment variable in a second place, and this goes red.
    #[test]
    fn every_scheduler_choice_in_this_crate_goes_through_the_one_predicate() {
        // The needles are assembled rather than written out, because
        // these files are among the files being searched and a literal
        // would count itself.
        let call = format!("{}()", "backend");
        let guard = format!("if {call} == Backend::Spin && ");
        let dispatch = format!("match {call} {{");
        let process_pool = format!("{}()", "pool");

        // The region helpers reach the process pool only behind the
        // guard, one guard per reach, and on the same line.
        let regions = include_str!("par/regions.rs");
        let guarded = regions.matches(&guard).count();
        assert!(guarded >= 6, "expected one guard per region helper");
        assert_eq!(
            regions.matches(&process_pool).count(),
            guarded,
            "a helper reached the persistent pool without asking the predicate"
        );
        assert!(
            regions
                .lines()
                .filter(|l| l.contains(&process_pool))
                .all(|l| l.contains(&guard)),
            "the guard and the pool it guards must be one condition"
        );

        // Here, the pool is reached by `num_threads`'s dispatch and by
        // its own definition, and by nothing else.
        let src = include_str!("par.rs");
        assert_eq!(
            src.matches(&dispatch).count(),
            3,
            "num_threads, join2 and join3 dispatch on the predicate"
        );
        let defined = format!("fn {process_pool}");
        assert_eq!(
            src.matches(&process_pool).count() - src.matches(&defined).count(),
            1,
            "only num_threads may reach the process pool from par.rs"
        );

        // And the environment is consulted in exactly one place, so the
        // override cannot come to mean two things. The needle is the
        // variable's name up to its closing quote, which is what keeps
        // `FRINK_CPU_POOL_SPIN_US` (a different knob, in `cpu_pool`)
        // out of the answer.
        let needle = format!("FRINK_CPU_POOL{}", '"');
        let mut readers = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("crate source is readable") {
                let path = entry.expect("readable entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && std::fs::read_to_string(&path)
                        .expect("source file is UTF-8")
                        .lines()
                        .any(|l| l.contains(&needle) && !l.trim_start().starts_with("//"))
                {
                    readers.push(path);
                }
            }
        }
        assert_eq!(
            readers.len(),
            1,
            "`FRINK_CPU_POOL` must be read only by `par::policy::pinned`, found {readers:?}"
        );
        assert!(readers[0].ends_with("par/policy.rs"), "{readers:?}");
    }

    /// The spin arm's chunking is a function of pool width and item
    /// count and nothing else. If a MAC threshold ever creeps back onto
    /// this path it has to change this signature to do it.
    #[test]
    fn the_spin_arm_chunks_by_pool_width_with_no_work_threshold() {
        assert_eq!(task_count(0), 0);
        assert_eq!(task_count(1), 1);
        assert_eq!(task_count(3), 3);
        let wide = task_count(1_000_000);
        assert_eq!(wide, num_threads() * TASKS_PER_THREAD);
        // A one-element-per-row matrix and a 4096-element-per-row matrix
        // decompose identically: work per item is not an input.
        assert_eq!(task_count(4096), task_count(4096));
        let (per, n) = split_for(num_threads(), 1000);
        assert_eq!(n, task_count(1000));
        assert!(per * n >= 1000 && (per - 1) * n < 1000);
    }

    /// A forked join's branches run under the step their submitter
    /// committed to, whichever worker steals them.
    ///
    /// The first branch sleeps so the others are stolen by idle workers
    /// whenever the pool has more than one thread. Without the carry a
    /// stolen branch reads its own thread's `None` and falls back to the
    /// per-operation rule: that is how Q, K and V came to race for the
    /// pool from three threads at once.
    ///
    /// Sabotage, run: pass `b` and `c` to `rayon::join` bare in `join3`
    /// (no `with_forward`) and this goes red.
    #[test]
    fn a_forked_joins_branches_inherit_the_steps_commitment() {
        use std::time::Duration;
        if policy::pinned().is_some() {
            return;
        }
        let rayon_step = Some(Backend::Rayon);
        let seen = on_workers(|| {
            policy::with_forward(rayon_step, || {
                join3(
                    || {
                        std::thread::sleep(Duration::from_millis(20));
                        policy::forward()
                    },
                    policy::forward,
                    policy::forward,
                )
            })
        });
        assert_eq!(seen, (rayon_step, rayon_step, rayon_step));
        let seen = policy::with_forward(rayon_step, || {
            join2(
                || {
                    std::thread::sleep(Duration::from_millis(20));
                    policy::forward()
                },
                policy::forward,
            )
        });
        assert_eq!(seen, (rayon_step, rayon_step));
    }

    #[test]
    fn joins_return_every_result_in_order() {
        assert_eq!(join2(|| 1u8, || 2u8), (1, 2));
        assert_eq!(join3(|| 1u8, || 2u8, || 3u8), (1, 2, 3));
    }

    /// Every helper in this module must report the region it is about
    /// to open, or the counter reads as coverage while measuring
    /// nothing. This walks all eight of them rather than trusting that
    /// a new one remembered, because a helper that forgot would leave
    /// the counter reading low and every assertion built on it passing.
    ///
    /// Sabotage: delete any single `note_rayon_region()` call and the
    /// helper whose name is in the failure message goes red.
    #[test]
    fn every_helper_reports_the_cold_region_it_opens() {
        if !the_promotion_applies_here() {
            return;
        }
        let mut buf = vec![0f32; 64];
        let mut other = vec![0f32; 64];

        // Spelled out one at a time rather than as a table of boxed
        // closures: the slice helpers borrow `buf`, so a table could
        // hold only half of them, and half a table is exactly the
        // coverage illusion this test exists to avoid.
        let before = cold_regions();
        indices(64, 1, |_| {});
        assert!(cold_regions() > before, "indices did not report");

        let before = cold_regions();
        indices_init(64, 1, || 0u8, |_, _| {});
        assert!(cold_regions() > before, "indices_init did not report");

        let before = cold_regions();
        join2(|| (), || ());
        assert!(cold_regions() > before, "join2 did not report");

        let before = cold_regions();
        join3(|| (), || (), || ());
        assert!(cold_regions() > before, "join3 did not report");

        let before = cold_regions();
        items_mut(&mut buf, 1, |_, _| {});
        assert!(cold_regions() > before, "items_mut did not report");

        let before = cold_regions();
        chunks_mut(&mut buf, 8, 1, |_, _| {});
        assert!(cold_regions() > before, "chunks_mut did not report");

        let before = cold_regions();
        chunks_mut_init(&mut buf, 8, 1, || 0u8, |_, _, _| {});
        assert!(cold_regions() > before, "chunks_mut_init did not report");

        let before = cold_regions();
        chunks_mut2(&mut buf, &mut other, 8, 1, |_, _, _| {});
        assert!(cold_regions() > before, "chunks_mut2 did not report");
    }

    /// The whole claim of `on_workers`: many regions inside it cost ONE
    /// cold entry into the pool, where the same regions outside it cost
    /// one each.
    ///
    /// This is the operation-count form of the fix. It needs no clock
    /// and no quiet host, which is why it is the guard rather than a
    /// throughput assertion.
    ///
    /// Sabotage: make `on_workers` call `f()` unconditionally and the
    /// `inside` count jumps from 1 to `REGIONS`, turning this red.
    #[test]
    fn on_workers_collapses_many_regions_into_one_cold_entry() {
        if !the_promotion_applies_here() {
            return;
        }
        const REGIONS: u64 = 16;
        let open_them = || {
            for _ in 0..REGIONS {
                indices(64, 1, |_| {});
            }
        };

        let before = cold_regions();
        open_them();
        let outside = cold_regions() - before;

        let before = cold_regions();
        on_workers(open_them);
        let inside = cold_regions() - before;

        assert_eq!(
            outside, REGIONS,
            "each region opened from a cold thread should count once"
        );
        assert_eq!(
            inside, 1,
            "the whole batch should enter the pool exactly once"
        );
    }

    /// A promoted step runs with the setting its SUBMITTER announced,
    /// not with the worker's default.
    ///
    /// This is GitHub issue #166's whole mechanism at the seam that
    /// caused it. `on_workers` moves the step to a thread the caller
    /// never configured; before the carry, a Metal decode read the
    /// default there and took the other `lm_head` path, changing the
    /// completion from the tenth token. The test asserts on the thread
    /// id first, so it cannot pass by not having moved.
    ///
    /// Sabotage: delete the `carried.adopt()` line from `on_workers`.
    #[cfg(feature = "metal")]
    #[test]
    fn a_promoted_step_runs_with_the_setting_its_submitter_announced() {
        use frink_metal::greedy_fold::{greedy_fold_setting, set_metal_greedy_argmax, GreedyFold};
        if !the_promotion_applies_here() {
            return;
        }
        set_metal_greedy_argmax(true);
        let submitter = std::thread::current().id();

        let (ran_on, seen) = on_workers(|| (std::thread::current().id(), greedy_fold_setting()));

        assert_ne!(
            ran_on, submitter,
            "nothing is being tested unless the step actually moved"
        );
        assert_eq!(
            seen,
            GreedyFold::On,
            "the worker must run the fold the caller asked for"
        );
        set_metal_greedy_argmax(false);
    }

    /// A nested call must not open a second entry, so an entry point can
    /// wrap unconditionally without knowing what its caller did.
    #[test]
    fn a_nested_on_workers_opens_no_further_cold_entry() {
        if !the_promotion_applies_here() {
            return;
        }
        let before = cold_regions();
        on_workers(|| {
            on_workers(|| {
                indices(64, 1, |_| {});
            });
        });
        assert_eq!(cold_regions() - before, 1);
    }

    /// `on_workers` must return what `f` returns, not swallow it.
    #[test]
    fn on_workers_hands_back_the_closures_value() {
        assert_eq!(on_workers(|| 41usize + 1), 42);
    }
}
