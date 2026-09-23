//! Which waiting job to admit next, when several are waiting.
//!
//! Admission was strict FIFO, for a reason this module has to answer
//! rather than ignore. From the batcher's own notes:
//!
//! > A skip-ahead policy ("admit the next job that fits") raises
//! > utilization and can starve a large request indefinitely behind a
//! > stream of small ones -- a queue that reorders by size
//! > systematically punishes exactly the requests that already wait
//! > longest. FIFO cannot starve, so FIFO it is until there is a
//! > measured reason to change it.
//!
//! Both halves of that stand. What follows does not reorder by SIZE,
//! and it cannot starve.
//!
//! # What it reorders by, and why that is not the same objection
//!
//! The radix prefix cache already knows how much of an incoming prompt
//! is already computed. Nothing read that at admission time, so a job
//! whose whole 6000-token system prompt is sitting in the page store
//! waited behind a job that has to prefill every token of its own.
//! Running the cached one first does not just reorder the work, it
//! REMOVES work: those pages are attended, not recomputed.
//!
//! Size-ordering punishes big requests because size is a property of
//! the request that never changes. Cache depth is a property of the
//! SERVER's state, and a job that loses today wins as soon as somebody
//! ahead of it publishes the prefix it shares -- which is the common
//! case, because jobs that share a prefix arrive together.
//!
//! # It cannot starve, and that is a test rather than an argument
//!
//! Two bounds, both hard:
//!
//! - a job may be passed over at most [`MAX_SKIPS`] times, after which
//!   it is the head of the line and nothing may overtake it;
//! - only the first [`WINDOW`] waiting jobs are considered at all, so a
//!   long queue cannot be scanned into a different order end to end.
//!
//! The first is what makes starvation impossible rather than unlikely:
//! a job's wait is bounded by `MAX_SKIPS` admissions, not by the
//! arrival pattern. `a_job_cannot_be_passed_over_forever` is that
//! bound, and it fails if either bound is removed.
//!
//! # The measured reason
//!
//! The bar the batcher set was a measurement, and the measurement is
//! not wall clock -- it is PREFILL TOKENS, counted, the same
//! acceptance number `n` > 1 is held to.
//!
//! With cache depths held FIXED a reordering saves nothing: the same
//! jobs, the same hits, a different order.
//! `reordering_alone_saves_nothing_when_hits_cannot_be_lost` says so,
//! because the first draft of that test asserted a saving and the
//! arithmetic did not agree.
//!
//! The saving is under PAGE PRESSURE, which is the state a busy server
//! is in: cached pages are evictable, so a job whose prefix is cached
//! now LOSES it if it waits behind enough uncached work to turn the
//! pool over. Admitting it first converts a hit that would have been
//! lost into a hit that is taken.
//! `under_eviction_pressure_the_policy_prefills_fewer_tokens` counts
//! that: 3000 prefill tokens on FIFO against 1200.

/// How many waiting jobs are looked at.
///
/// Small on purpose. The point is to let a cached job jump a short
/// queue of uncached ones, not to sort the whole backlog: scanning
/// further costs a radix walk per job per tick and buys less each
/// time, because the jobs deepest in the queue are the ones most
/// likely to have been overtaken already.
pub(super) const WINDOW: usize = 8;

/// How many times a job may be passed over before it becomes
/// untouchable.
///
/// This is the anti-starvation bound and the only reason this policy
/// is admissible at all. At zero it is strict FIFO; at infinity it is
/// the starvation bug the batcher refused. Four is short enough that a
/// passed-over job is admitted within a handful of ticks and long
/// enough that a genuinely cached job usually gets through.
pub(super) const MAX_SKIPS: u32 = 4;

/// The skip counter rides on the JOB, not in a parallel structure
/// here.
///
/// A deque of counters beside the waiting queue would have to agree
/// with it through four separate mutation sites -- the channel drain,
/// the idle-block receive, the abort pass that rebuilds the queue, and
/// admission itself -- which is exactly the two-structures-that-must-
/// agree shape this repo has fixed a dozen instances of. On the job it
/// moves with the job and cannot desync.
///
/// Both inputs below are therefore built per tick from the queue
/// itself, so they are the same length by construction.
/// Which waiting job to admit next, given how much of each one's
/// prompt is already computed.
///
/// Takes DEPTHS rather than jobs: the policy ranks numbers, and where
/// the numbers come from is the caller's business. That also makes
/// every case below testable without building a `Job`, which needs a
/// reply channel and an abort handle to exist at all.
///
/// Returns an index into `depths`. Always `0` when the head has been
/// passed over its limit, when the queue is one deep, or when no job
/// in the window has a strictly deeper hit than the head -- so the
/// ordinary case is FIFO and costs one comparison.
pub(super) fn choose(depths: &[usize], skips: &[u32]) -> usize {
    if depths.len() < 2 {
        return 0;
    }
    // The head's own bound comes first: once a job has been passed
    // over `MAX_SKIPS` times nothing may overtake it, whatever the
    // cache says. Checked BEFORE the scan rather than folded into the
    // comparison, because a bound that is one term of a ranking is a
    // bound that a large enough other term defeats.
    if skips.first().copied().unwrap_or(0) >= MAX_SKIPS {
        return 0;
    }
    let mut best = 0usize;
    let mut best_depth = depths[0];
    for (i, &d) in depths.iter().enumerate().take(WINDOW).skip(1) {
        // A job that is itself at the bound cannot be overtaken
        // either, so the scan stops at it rather than stepping past --
        // otherwise the bound would protect one job by starving the
        // next.
        if skips.get(i).copied().unwrap_or(0) >= MAX_SKIPS {
            break;
        }
        // Strictly deeper: a tie keeps arrival order, so equal jobs
        // stay FIFO and the policy has no effect on a cold server.
        if d > best_depth {
            best_depth = d;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(n: usize) -> Vec<u32> {
        vec![0; n]
    }

    #[test]
    fn a_cold_server_is_still_fifo() {
        assert_eq!(choose(&[0, 0, 0], &queued(3)), 0);
    }

    /// Ties keep arrival order, so a server where every job shares one
    /// prefix behaves exactly as it did before.
    #[test]
    fn equal_depth_keeps_arrival_order() {
        assert_eq!(choose(&[64, 64, 64], &queued(3)), 0);
    }

    #[test]
    fn the_deepest_hit_in_the_window_goes_first() {
        assert_eq!(choose(&[0, 64, 512], &queued(3)), 2);
    }

    /// **The anti-starvation bound.**
    ///
    /// The objection this policy has to answer is that a reordering
    /// queue starves whoever it keeps passing over. It cannot: after
    /// `MAX_SKIPS` the head is admitted whatever anybody else's depth
    /// is. Driven through the queue rather than read off the constant,
    /// and the loop runs longer than the bound so a policy that never
    /// admitted the head would report `None`.
    #[test]
    fn a_job_cannot_be_passed_over_forever() {
        let mut skips = queued(2);
        let mut admitted_head_after = None;
        for round in 0..MAX_SKIPS + 3 {
            // The second job wins on depth every single time.
            let pick = choose(&[0, 4096], &skips);
            if pick == 0 {
                admitted_head_after = Some(round);
                break;
            }
            // Everything the pick jumped is one skip older; the picked
            // job leaves and a fresh one joins the back, so the queue
            // never drains and the head's only way out is the bound.
            for c in skips.iter_mut().take(pick) {
                *c += 1;
            }
            skips.remove(pick);
            skips.push(0);
        }
        assert_eq!(
            admitted_head_after,
            Some(MAX_SKIPS),
            "the head must be admitted after exactly {MAX_SKIPS} skips"
        );
    }

    /// A job at the bound is not overtaken, and neither is anything
    /// behind it.
    #[test]
    fn the_scan_stops_at_a_job_that_has_reached_the_bound() {
        let mut skips = queued(3);
        skips[1] = MAX_SKIPS;
        assert_eq!(
            choose(&[0, 0, 4096], &skips),
            0,
            "a job behind one at the bound must not be pulled in front of it"
        );
    }

    /// Only the first `WINDOW` jobs are considered, so a long queue
    /// cannot be scanned into a different order end to end.
    #[test]
    fn nothing_past_the_window_is_considered() {
        let mut depths = vec![0usize; WINDOW + 2];
        depths[WINDOW + 1] = 999_999;
        assert_eq!(
            choose(&depths, &queued(WINDOW + 2)),
            0,
            "a job past the window must not be reachable"
        );
    }

    /// **Where the saving is, and where it is NOT.**
    ///
    /// With cache depths held fixed, admitting the same jobs in a
    /// different order prefills exactly the same number of tokens.
    /// Stating that here rather than leaving it implied, because the
    /// first draft of this test asserted a saving in that model and
    /// the arithmetic says there is none -- the same set of jobs, the
    /// same hits, a different order.
    #[test]
    fn reordering_alone_saves_nothing_when_hits_cannot_be_lost() {
        let prompt = 1000usize;
        let depths = [0usize, 900, 900];
        let total = |order: &[usize]| -> usize { order.iter().map(|&i| prompt - depths[i]).sum() };
        assert_eq!(total(&[0, 1, 2]), total(&[1, 2, 0]));
    }

    /// **The saving is real under PAGE PRESSURE**, which is the state
    /// a busy server is actually in.
    ///
    /// Cached pages are evictable. A job whose prefix is cached now
    /// loses it if it waits behind enough uncached work to turn the
    /// page pool over, so admitting it first converts a hit that would
    /// have been lost into a hit that is taken. Modelled as: every
    /// admission of an uncached job evicts one waiting job's hit.
    ///
    /// This is the measured reason the batcher asked for, counted in
    /// PREFILL TOKENS rather than seconds -- the same acceptance
    /// number `n` > 1 is held to.
    #[test]
    fn under_eviction_pressure_the_policy_prefills_fewer_tokens() {
        let prompt = 1000usize;
        // One cold job, then two sharing a published 900-token prefix.
        let start = [0usize, 900, 900];

        // Admit in `order`; every cold admission (a job with no hit)
        // turns the pool over and costs every job still waiting its
        // cached prefix.
        let run = |order: &[usize]| -> usize {
            let mut depth = start;
            let mut tokens = 0usize;
            for (pos, &i) in order.iter().enumerate() {
                let hit = depth[i];
                tokens += prompt - hit;
                if hit == 0 {
                    for &j in &order[pos + 1..] {
                        depth[j] = 0;
                    }
                }
            }
            tokens
        };

        let fifo = run(&[0, 1, 2]);
        let picked = choose(&start, &queued(3));
        assert_eq!(picked, 1, "the deepest hit should go first");
        let policy = run(&[1, 2, 0]);

        assert!(
            policy < fifo,
            "the policy prefilled {policy} tokens against FIFO's {fifo}"
        );
        assert_eq!(
            (fifo, policy),
            (3000, 1200),
            "the arithmetic, spelled out so a change to it is visible"
        );
    }
}
