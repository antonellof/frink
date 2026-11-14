//! Re-splitting the pools of a server that is already running.
//!
//! The arithmetic that decides the split in the first place is
//! [`frink_core::expert_budget`], beside the expert cache whose slots
//! it counts. This module is the other half: validating a *change* to
//! that split before anything is touched.
//!
//! # Why a rebuild validates before it frees
//!
//! Re-splitting the pools means dropping allocations and taking new
//! ones. If the new size does not fit, the old pool is already gone and
//! the server is down -- for a request that was never going to work.
//! So every size check happens up front, on arithmetic alone
//! ([`validate_rebuild`]), and a rejection leaves the engine serving
//! exactly what it was serving before. `POST /v1/cache/rebuild`
//! (`crate::cache_admin::cache_rebuild`) calls it before it opens the
//! transaction that moves the pool.
//!
//! Ported from FreeToken's `engine/cache_budget.py` (Apache-2.0); see
//! `docs/THIRD_PARTY_NOTICES.md`. Two departures, both because frink's
//! KV pool is not FreeToken's: there is no idle check here, since
//! `KvBlockPool::resize` frees nothing and refuses on its own to shrink
//! below what in-flight requests hold; and the KV floor is what the
//! caller says, not at least two pages, since frink's pool reserves no
//! null page.

pub(crate) use frink_core::expert_budget::{BudgetTooSmall, PoolSizes};

use frink_core::expert_budget::required_bytes;

/// How many tokens a window request keeps live past the window itself,
/// so a decode step that slides the window does not immediately need
/// state it just freed.
pub const SWA_RETAIN_GAP: usize = 16;

/// How often the window is slid, in decode steps. Sliding every step
/// would cost a pool operation per token; sliding rarely means the pool
/// must hold that many extra tokens per request.
pub const DEFAULT_SWA_EVICTION_INTERVAL: usize = 128;

/// A live re-split request. `None` means "leave this pool alone".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildRequest {
    pub moe_cache_slots: Option<u64>,
    pub kv_pages: Option<u64>,
    pub mamba_slots: Option<u64>,
    pub swa_pages: Option<u64>,
}

impl RebuildRequest {
    pub fn is_empty(&self) -> bool {
        self.moe_cache_slots.is_none()
            && self.kv_pages.is_none()
            && self.mamba_slots.is_none()
            && self.swa_pages.is_none()
    }

    /// Whether this request invalidates the prefix cache.
    ///
    /// Any KV-side resize does: a page index, a recurrent slot id and a
    /// window slot id are all positions in an allocation that is about
    /// to stop existing, and a cached prefix pointing into the old one
    /// would hand back another request's state. A MoE-only resize does
    /// not -- the expert cache holds no per-request state.
    pub fn invalidates_prefix_cache(&self) -> bool {
        self.kv_pages.is_some() || self.mamba_slots.is_some() || self.swa_pages.is_some()
    }
}

/// Why a rebuild was refused. The engine is untouched in every case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RebuildRejected {
    /// The deployment has no such pool. Carries the request's own name
    /// for it (`moe`, `kv`, `mamba`, `swa`), so the refusal points at
    /// the field the caller sent.
    NoSuchPool(&'static str),
    /// The target is below what the pool must hold to function.
    BelowFloor {
        pool: &'static str,
        requested: u64,
        floor: u64,
    },
    /// The split does not fit in the budget.
    TooLarge(BudgetTooSmall),
}

impl std::fmt::Display for RebuildRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebuildRejected::NoSuchPool(pool) => {
                write!(f, "this deployment has no {pool} pool to rebuild")
            }
            RebuildRejected::BelowFloor {
                pool,
                requested,
                floor,
            } => write!(
                f,
                "{pool}={requested} is below the working-set floor of {floor}"
            ),
            RebuildRejected::TooLarge(inner) => write!(f, "{inner}"),
        }
    }
}

impl std::error::Error for RebuildRejected {}

/// What each pool must hold to work at all, and `None` for a pool this
/// deployment does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolFloors {
    pub moe_slots: Option<u64>,
    pub kv_pages: Option<u64>,
    pub mamba_slots: Option<u64>,
    pub swa_pages: Option<u64>,
}

/// The bytes the two resizable pools may spend, and what one unit of
/// each costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteBudget {
    pub budget_bytes: i64,
    pub bytes_per_expert: u64,
    pub bytes_per_page: u64,
}

/// Check a rebuild against the floors and the budget, changing nothing.
///
/// `budget` is `None` when the deployment sized its pools in units
/// rather than bytes (`FRINK_KV_POOL_BLOCKS`): there is no byte figure
/// to hold the target to, and inventing one would refuse rebuilds the
/// operator's own configuration allows.
pub fn validate_rebuild(
    request: &RebuildRequest,
    current: &PoolSizes,
    floors: &PoolFloors,
    budget: Option<&ByteBudget>,
) -> Result<PoolSizes, RebuildRejected> {
    for (asked, floor, pool) in [
        (request.moe_cache_slots, floors.moe_slots, "moe"),
        (request.kv_pages, floors.kv_pages, "kv"),
        (request.mamba_slots, floors.mamba_slots, "mamba"),
        (request.swa_pages, floors.swa_pages, "swa"),
    ] {
        let Some(requested) = asked else { continue };
        match floor {
            None => return Err(RebuildRejected::NoSuchPool(pool)),
            Some(floor) if requested < floor => {
                return Err(RebuildRejected::BelowFloor {
                    pool,
                    requested,
                    floor,
                })
            }
            Some(_) => {}
        }
    }

    let target = PoolSizes {
        moe_cache_slots: request.moe_cache_slots.unwrap_or(current.moe_cache_slots),
        kv_pages: request.kv_pages.unwrap_or(current.kv_pages),
        prefill_overlap: current.prefill_overlap,
    };
    if let Some(budget) = budget {
        let needed = required_bytes(
            target.moe_cache_slots,
            target.kv_pages,
            budget.bytes_per_expert,
            budget.bytes_per_page,
        );
        if needed > budget.budget_bytes {
            return Err(RebuildRejected::TooLarge(BudgetTooSmall {
                needed_bytes: needed,
                budget_bytes: budget.budget_bytes,
                sizes: target,
            }));
        }
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn current() -> PoolSizes {
        PoolSizes {
            moe_cache_slots: 64,
            kv_pages: 1024,
            prefill_overlap: true,
        }
    }

    fn floors() -> PoolFloors {
        PoolFloors {
            moe_slots: Some(8),
            kv_pages: Some(16),
            mamba_slots: None,
            swa_pages: Some(32),
        }
    }

    fn budget(bytes: u64) -> ByteBudget {
        ByteBudget {
            budget_bytes: bytes as i64,
            bytes_per_expert: MIB,
            bytes_per_page: MIB,
        }
    }

    #[test]
    fn a_rebuild_below_a_floor_or_over_the_budget_changes_nothing() {
        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    moe_cache_slots: Some(4),
                    ..RebuildRequest::default()
                },
                &current(),
                &floors(),
                None,
            ),
            Err(RebuildRejected::BelowFloor { pool: "moe", .. })
        ));

        // A pool this deployment does not have is named, not silently
        // accepted.
        assert_eq!(
            validate_rebuild(
                &RebuildRequest {
                    mamba_slots: Some(64),
                    ..RebuildRequest::default()
                },
                &current(),
                &floors(),
                None,
            ),
            Err(RebuildRejected::NoSuchPool("mamba"))
        );

        assert!(matches!(
            validate_rebuild(
                &RebuildRequest {
                    kv_pages: Some(1 << 40),
                    ..RebuildRequest::default()
                },
                &current(),
                &floors(),
                Some(&budget(2 * GIB)),
            ),
            Err(RebuildRejected::TooLarge(_))
        ));
    }

    /// The budget is checked only when there is one. A pool sized in
    /// blocks has no byte ceiling, and the same request that a byte
    /// budget refuses is a valid target there.
    #[test]
    fn only_a_byte_budget_can_refuse_a_size() {
        let big = RebuildRequest {
            kv_pages: Some(1 << 20),
            ..RebuildRequest::default()
        };
        assert!(validate_rebuild(&big, &current(), &floors(), Some(&budget(GIB))).is_err());
        assert_eq!(
            validate_rebuild(&big, &current(), &floors(), None)
                .expect("no budget, no ceiling")
                .kv_pages,
            1 << 20
        );
    }

    /// A target so large its price overflows is refused, not wrapped
    /// into a small number that fits.
    #[test]
    fn a_price_that_overflows_is_too_large_rather_than_small() {
        let huge = RebuildRequest {
            kv_pages: Some(u64::MAX),
            ..RebuildRequest::default()
        };
        assert!(matches!(
            validate_rebuild(&huge, &current(), &floors(), Some(&budget(GIB))),
            Err(RebuildRejected::TooLarge(_))
        ));
    }

    #[test]
    fn a_rebuild_leaves_untouched_pools_at_their_current_size() {
        let target = validate_rebuild(
            &RebuildRequest {
                moe_cache_slots: Some(128),
                ..RebuildRequest::default()
            },
            &current(),
            &floors(),
            Some(&budget(4 * GIB)),
        )
        .expect("fits");
        assert_eq!(target.moe_cache_slots, 128);
        assert_eq!(target.kv_pages, 1024);
    }

    /// Only a KV-side resize invalidates cached prefixes; the expert
    /// cache holds no per-request state.
    #[test]
    fn only_a_kv_side_resize_invalidates_the_prefix_cache() {
        assert!(!RebuildRequest {
            moe_cache_slots: Some(128),
            ..RebuildRequest::default()
        }
        .invalidates_prefix_cache());
        for request in [
            RebuildRequest {
                kv_pages: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                mamba_slots: Some(1),
                ..RebuildRequest::default()
            },
            RebuildRequest {
                swa_pages: Some(1),
                ..RebuildRequest::default()
            },
        ] {
            assert!(request.invalidates_prefix_cache());
        }
        assert!(RebuildRequest::default().is_empty());
    }
}
