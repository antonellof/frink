//! The transaction around a live cache resize.
//!
//! [`crate::policy::pool_budget::validate_rebuild`] is the *pre-check*: it decides
//! whether a split fits, before anything is freed. That is only half of
//! a resize. The other half is what happens when the arithmetic said
//! yes and the allocation still fails -- which is the case that turns a
//! serving engine into a permanent 503 if it is not handled.
//!
//! # Why a teardown flag rather than a `Result`
//!
//! A resize is destructive in the middle. It frees the old pools, then
//! allocates the new ones, and a failure on either side of that line
//! means something completely different:
//!
//! - **Before** the free: nothing has been given up. The engine is
//!   untouched and still serving, the request is simply rejected, and
//!   the caller may retry immediately.
//! - **After** the free: the old pools are gone. Rejecting the request
//!   is not enough -- the engine has no pools at all, and every
//!   subsequent request would 503 forever. The old sizes must be
//!   allocated back.
//!
//! A plain `Result` cannot tell those apart, because both are `Err`.
//! The teardown mark [`RebuildTxn::run`] hands its caller is what
//! distinguishes them, and it is the whole reason this type exists
//! rather than a closure.
//!
//! # Why only a failed ROLLBACK latches
//!
//! A failed rebuild is bad; a failed rollback is unrecoverable. The
//! first leaves an engine serving what it was serving before. The
//! second leaves one with no pools and no way to get them back, and the
//! only honest thing to do is latch
//! [`crate::MaintenanceState::Failed`] and stop pretending requests can
//! be served. Latching on the first would take a serving engine out of
//! service for a resize that changed nothing.
//!
//! # Why the snapshot is only the touched pools
//!
//! The rollback target is the pools the request NAMED, not every pool.
//! Restoring an untouched pool means freeing and re-allocating it, and
//! on the KV side that trips the invalidation gate and wipes a prefix
//! cache that had no reason to be touched -- so a resize of the expert
//! cache would silently cost every cached prefix in the engine.
//!
//! Ported from FreeToken's `scheduler/scheduler.py`; see
//! `docs/THIRD_PARTY_NOTICES.md`.

use crate::policy::pool_budget::{PoolSizes, RebuildRequest};

/// How a resize ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildOutcome {
    /// The new sizes are live.
    Applied(PoolSizes),
    /// The resize failed before anything was freed. The engine is
    /// untouched and still serving exactly what it was.
    RejectedIntact { reason: String },
    /// The resize failed after the free, and the OLD sizes were put
    /// back. The request did not happen; the engine serves what it did
    /// before. Reported apart from `RejectedIntact` because an operator
    /// should know the engine went through a teardown -- a prefix cache
    /// invalidated by it does not come back.
    RolledBack { reason: String, restored: PoolSizes },
    /// The rollback ITSELF failed. The engine has no pools and cannot
    /// get them back. Unrecoverable, and the only state that latches.
    Latched {
        reason: String,
        rollback_error: String,
    },
}

/// Which pools a request actually names.
///
/// The point of tracking this rather than restoring everything: a
/// rollback that touches a pool the request never named still FREES and
/// RE-ALLOCATES it, and on the KV side that trips the invalidation gate
/// and wipes a prefix cache which had no reason to be disturbed. So a
/// resize of the expert cache would silently cost every cached prefix
/// in the engine -- a real loss, to undo a change that pool never saw.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TouchedPools {
    pub moe_cache: bool,
    pub kv: bool,
}

impl TouchedPools {
    fn of(request: &RebuildRequest) -> Self {
        TouchedPools {
            moe_cache: request.moe_cache_slots.is_some(),
            // The three KV-side pools share one allocation gate, so any
            // of them makes the KV side dirty.
            kv: request.kv_pages.is_some()
                || request.mamba_slots.is_some()
                || request.swa_pages.is_some(),
        }
    }
}

/// One live resize, from safe point to outcome.
///
/// Time-free, allocation-free and device-free: the caller supplies the
/// two operations that actually touch memory, so every ordering rule
/// here is asserted with closures rather than against a GPU.
#[derive(Debug)]
pub struct RebuildTxn {
    /// The pools this request names, and their CURRENT sizes -- the
    /// rollback target. Only the touched ones; see the module doc.
    rollback_to: PoolSizes,
    /// Which of them the request named, so a rollback restores those
    /// and leaves the rest alone.
    touched: TouchedPools,
    /// Set the instant the destructive free begins.
    teardown_started: bool,
}

impl RebuildTxn {
    /// Opens a transaction against `current`, keeping as the rollback
    /// target only what `request` actually touches.
    pub fn open(request: &RebuildRequest, current: &PoolSizes) -> Self {
        RebuildTxn {
            rollback_to: *current,
            touched: TouchedPools::of(request),
            // Nothing has been freed yet, by definition: this is the
            // flag that later tells a failure "the engine is intact"
            // from "the pools are gone".
            teardown_started: false,
        }
    }

    /// Runs the resize.
    ///
    /// `free_then_allocate` is the destructive half. It must call
    /// `mark_teardown` the moment it begins freeing and BEFORE it frees
    /// anything, because that flag is the only thing that tells a
    /// failure "nothing was given up" from "the pools are gone". A
    /// caller that marks late turns a recoverable rollback into a
    /// permanent 503.
    ///
    /// `restore` is asked for the old sizes only when the teardown had
    /// begun; when it had not, it is never called, because there is
    /// nothing to restore. It is handed [`TouchedPools`] alongside
    /// them, and must restore ONLY those -- see that type for what
    /// restoring an untouched pool costs.
    pub fn run(
        mut self,
        target: PoolSizes,
        free_then_allocate: impl FnOnce(&mut dyn FnMut(), PoolSizes) -> Result<(), String>,
        restore: impl FnOnce(PoolSizes, TouchedPools) -> Result<(), String>,
    ) -> RebuildOutcome {
        let started = std::cell::Cell::new(false);
        let mut mark = || started.set(true);

        match free_then_allocate(&mut mark, target) {
            Ok(()) => {
                self.teardown_started = started.get();
                RebuildOutcome::Applied(target)
            }
            Err(reason) => {
                self.teardown_started = started.get();
                if !self.teardown_started {
                    // Nothing was freed. The engine is exactly as it
                    // was, so there is nothing to put back and no
                    // reason to disturb it.
                    return RebuildOutcome::RejectedIntact { reason };
                }
                match restore(self.rollback_to, self.touched) {
                    Ok(()) => RebuildOutcome::RolledBack {
                        reason,
                        restored: self.rollback_to,
                    },
                    Err(rollback_error) => RebuildOutcome::Latched {
                        reason,
                        rollback_error,
                    },
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes(kv: u64, moe: u64) -> PoolSizes {
        PoolSizes {
            moe_cache_slots: moe,
            kv_pages: kv,
            prefill_overlap: true,
        }
    }

    fn kv_request(pages: u64) -> RebuildRequest {
        RebuildRequest {
            moe_cache_slots: None,
            kv_pages: Some(pages),
            mamba_slots: None,
            swa_pages: None,
        }
    }

    /// The failure that costs nothing. The arithmetic said yes, the
    /// allocation said no, and nothing had been given up yet -- so the
    /// engine keeps serving exactly what it was and `restore` is never
    /// even called.
    #[test]
    fn a_failure_before_the_free_leaves_the_engine_untouched() {
        let txn = RebuildTxn::open(&kv_request(128), &sizes(64, 8));
        let mut restored = false;

        let outcome = txn.run(
            sizes(128, 8),
            // Fails without ever marking: nothing was freed.
            |_mark, _target| Err("device refused the reservation".to_string()),
            |_, _| {
                restored = true;
                Ok(())
            },
        );

        assert_eq!(
            outcome,
            RebuildOutcome::RejectedIntact {
                reason: "device refused the reservation".to_string()
            }
        );
        assert!(
            !restored,
            "there is nothing to restore, so restore must not run"
        );
    }

    /// The failure that matters. Past the free the old pools are gone,
    /// so rejecting is not enough: without putting them back the engine
    /// has no pools at all and every later request 503s forever.
    #[test]
    fn a_failure_after_the_free_puts_the_old_sizes_back() {
        let current = sizes(64, 8);
        let txn = RebuildTxn::open(&kv_request(4096), &current);
        let mut asked_to_restore = None;

        let outcome = txn.run(
            sizes(4096, 8),
            |mark, _target| {
                mark();
                Err("out of memory allocating the new pool".to_string())
            },
            |old, touched| {
                asked_to_restore = Some((old, touched));
                Ok(())
            },
        );

        let (old, touched) = asked_to_restore.expect("the rollback ran");
        assert_eq!(old, current, "the rollback target is what was there before");
        assert!(touched.kv, "the KV side is what this request named");
        assert!(
            !touched.moe_cache,
            "and the expert cache is left alone, so its prefixes survive"
        );
        match outcome {
            RebuildOutcome::RolledBack { restored, .. } => assert_eq!(restored, current),
            other => panic!("expected RolledBack, got {other:?}"),
        }
    }

    /// A failed rebuild is bad; a failed ROLLBACK is unrecoverable, and
    /// only the second latches. Latching on the first would take a
    /// serving engine out of service over a resize that changed
    /// nothing.
    #[test]
    fn only_a_failed_rollback_latches_the_engine() {
        let txn = RebuildTxn::open(&kv_request(4096), &sizes(64, 8));
        let outcome = txn.run(
            sizes(4096, 8),
            |mark, _| {
                mark();
                Err("oom".to_string())
            },
            |_, _| Err("could not re-allocate the old pool either".to_string()),
        );

        match &outcome {
            RebuildOutcome::Latched { rollback_error, .. } => {
                assert!(rollback_error.contains("old pool"))
            }
            other => panic!("expected Latched, got {other:?}"),
        }
    }

    #[test]
    fn a_resize_that_succeeds_reports_the_sizes_that_are_now_live() {
        let txn = RebuildTxn::open(&kv_request(256), &sizes(64, 8));
        let outcome = txn.run(
            sizes(256, 8),
            |mark, target| {
                mark();
                assert_eq!(target.kv_pages, 256);
                Ok(())
            },
            |_, _| panic!("a successful resize must not restore anything"),
        );
        assert_eq!(outcome, RebuildOutcome::Applied(sizes(256, 8)));
    }

    /// A rollback restores only what the request named. Restoring an
    /// untouched pool still frees and re-allocates it, and on the KV
    /// side that trips the invalidation gate -- so a resize of the
    /// EXPERT cache would silently cost every cached prefix in the
    /// engine, to undo a change the KV pool never saw.
    #[test]
    fn a_rollback_leaves_the_pools_the_request_never_named_alone() {
        let current = sizes(64, 8);

        let kv_only = RebuildTxn::open(&kv_request(256), &current);
        assert_eq!(
            kv_only.touched,
            TouchedPools {
                moe_cache: false,
                kv: true
            }
        );
        assert_eq!(
            kv_only.rollback_to.kv_pages, 64,
            "the touched pool's old size"
        );
        assert!(
            !kv_only.teardown_started,
            "a fresh transaction has freed nothing"
        );

        let moe_only = RebuildTxn::open(
            &RebuildRequest {
                moe_cache_slots: Some(16),
                kv_pages: None,
                mamba_slots: None,
                swa_pages: None,
            },
            &current,
        );
        assert_eq!(
            moe_only.touched,
            TouchedPools {
                moe_cache: true,
                kv: false
            },
            "an expert-cache resize must not drag the KV pool through a \
             teardown"
        );

        // The three KV-side pools share one allocation gate, so any of
        // them makes the KV side dirty.
        for request in [
            RebuildRequest {
                moe_cache_slots: None,
                kv_pages: None,
                mamba_slots: Some(4),
                swa_pages: None,
            },
            RebuildRequest {
                moe_cache_slots: None,
                kv_pages: None,
                mamba_slots: None,
                swa_pages: Some(4),
            },
        ] {
            assert!(
                RebuildTxn::open(&request, &current).touched.kv,
                "{request:?}"
            );
        }

        assert_eq!(
            RebuildTxn::open(&RebuildRequest::default(), &current).touched,
            TouchedPools::default()
        );
    }
}
