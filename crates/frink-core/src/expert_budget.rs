//! Splitting a memory budget between the GPU expert cache and the KV
//! pool.
//!
//! # Why the expert cache gets first claim
//!
//! Both pools compete for the same bytes, but they do not degrade the
//! same way. A KV pool one page short means one fewer concurrent
//! request or a shorter context -- a scheduling limit, felt as a queue.
//! An expert cache one slot short means every step that routes to the
//! missing expert pays a PCIe transfer or a CPU detour -- a *per-token*
//! tax on every request, forever. So [`plan_cache_budget`] fills the
//! expert cache first, up to full residency, and gives the remainder to
//! KV -- with a floor of `kv_reserve_pages`, because a server that
//! cannot hold a context serves nothing at all.
//!
//! # A byte budget becomes a slot count
//!
//! A user says a number of bytes; a bounded pool of fixed-size slots is
//! the only thing you can actually cap. [`expert_bytes_per_slot`] prices
//! one slot as the sum of its row across every weight bank, and
//! [`plan_cache_budget`] divides. It lives beside
//! [`expert_store`](crate::expert_store), which holds the budget it
//! sizes, and beside [`expert_cache`](crate::expert_cache), whose slots
//! it counts.
//!
//! Ported 1:1 from FreeToken's `engine/cache_budget.py` (Apache-2.0);
//! see `docs/THIRD_PARTY_NOTICES.md`.

/// The VRAM a rebuild may spend: a fraction of what was free before the
/// weights loaded, minus the weights, minus whatever the pools need
/// unconditionally.
///
/// The fraction is not padding for its own sake -- it is the room the
/// activations, the workspace, and the captured graphs occupy, none of
/// which is in `weights_bytes`. Spending it produces an allocation
/// failure at the first long prompt rather than at startup.
///
/// Signed on purpose: an over-committed deployment gets a negative
/// budget, which every consumer below then refuses, rather than a
/// wrapped-around enormous one.
pub fn net_cache_budget_bytes(
    memory_ratio: f64,
    baseline_free_bytes: u64,
    weights_bytes: u64,
    fixed_cache_bytes: u64,
) -> i64 {
    (memory_ratio * baseline_free_bytes as f64) as i64
        - weights_bytes as i64
        - fixed_cache_bytes as i64
}

/// What a given split would actually cost.
///
/// Saturating, because a live rebuild prices a size a client typed: a
/// product that wrapped would come back small and fit any budget, and
/// one cast past `i64::MAX` would come back negative and fit all of
/// them.
pub fn required_bytes(
    moe_cache_slots: u64,
    kv_pages: u64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
) -> i64 {
    let bytes = moe_cache_slots
        .saturating_mul(bytes_per_expert)
        .saturating_add(kv_pages.saturating_mul(bytes_per_page));
    i64::try_from(bytes).unwrap_or(i64::MAX)
}

/// The KV budget at startup, when the weights have just been loaded.
///
/// `init_free - new_free` is what loading actually consumed, measured
/// rather than predicted -- which is why this is not the same
/// expression as [`net_cache_budget_bytes`], and why it is the one used
/// before any pool exists.
pub fn startup_kv_budget(memory_ratio: f64, init_free_bytes: u64, new_free_bytes: u64) -> i64 {
    (memory_ratio * init_free_bytes as f64) as i64
        - (init_free_bytes as i64 - new_free_bytes as i64)
}

/// A pool split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSizes {
    /// Slots in the GPU expert cache.
    pub moe_cache_slots: u64,
    /// Pages in the KV pool.
    pub kv_pages: u64,
    /// Whether the prefill double buffer survived the split. It needs
    /// two layers' worth of slots, and a tight budget can take that
    /// away.
    pub prefill_overlap: bool,
}

/// A split that does not fit, refused before anything was freed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetTooSmall {
    pub needed_bytes: i64,
    pub budget_bytes: i64,
    /// What the arithmetic wanted to allocate.
    pub sizes: PoolSizes,
}

impl std::fmt::Display for BudgetTooSmall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requested cache (moe={} slots, kv={} pages) needs {} bytes but the budget is {}; \
             the old cache is kept and still serving",
            self.sizes.moe_cache_slots, self.sizes.kv_pages, self.needed_bytes, self.budget_bytes
        )
    }
}

impl std::error::Error for BudgetTooSmall {}

/// What one expert costs in the GPU cache: the sum of its row across
/// every weight bank.
///
/// Fixed per-model tensors that do **not** scale with the number of
/// cached experts (folded quantization scales, for instance) are
/// deliberately excluded -- counting them here would make each slot
/// look more expensive than it is and under-size the cache.
pub fn expert_bytes_per_slot(bank_row_bytes: &[u64]) -> u64 {
    bank_row_bytes.iter().sum()
}

/// Split `budget_bytes` between the expert cache and the KV pool.
///
/// `max_slots` is a backend ceiling (some fused kernels cannot address
/// more than a fixed number of experts); `total_experts` is full
/// residency, past which more slots buy nothing.
#[allow(clippy::too_many_arguments)]
pub fn plan_cache_budget(
    budget_bytes: i64,
    bytes_per_expert: u64,
    bytes_per_page: u64,
    num_experts: u64,
    total_experts: u64,
    prefill_overlap: bool,
    kv_reserve_pages: u64,
    max_slots: u64,
) -> Result<PoolSizes, BudgetTooSmall> {
    assert!(
        bytes_per_expert > 0 && bytes_per_page > 0,
        "an unpriced pool cannot be sized"
    );
    let hi = total_experts.min(max_slots);
    // The double buffer needs two layers of slots; without room for it
    // the split is planned without it rather than failing.
    let mut overlap = prefill_overlap && hi >= 2 * num_experts;
    let lo = if overlap {
        2 * num_experts
    } else {
        num_experts
    };
    assert!(
        hi >= lo,
        "the expert-cache ceiling of {hi} slots cannot hold the {lo} slots a layer needs"
    );

    // Fill the cache first, but never at the cost of the KV floor.
    let spare = budget_bytes - (kv_reserve_pages * bytes_per_page) as i64;
    let raw = if spare <= 0 {
        0
    } else {
        (spare as u64) / bytes_per_expert
    };
    let moe_cache_slots = raw.min(hi).max(lo);
    overlap = overlap && moe_cache_slots >= 2 * num_experts;

    let remaining = budget_bytes - (moe_cache_slots * bytes_per_expert) as i64;
    let kv_pages = if remaining <= 0 {
        kv_reserve_pages
    } else {
        ((remaining as u64) / bytes_per_page).max(kv_reserve_pages)
    };

    let sizes = PoolSizes {
        moe_cache_slots,
        kv_pages,
        prefill_overlap: overlap,
    };
    let needed = required_bytes(moe_cache_slots, kv_pages, bytes_per_expert, bytes_per_page);
    if needed > budget_bytes || kv_pages <= 1 {
        return Err(BudgetTooSmall {
            needed_bytes: needed,
            budget_bytes,
            sizes,
        });
    }
    Ok(sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A price past what the arithmetic can hold is the largest price,
    /// never a wrapped small one or a negative one that fits every
    /// budget.
    #[test]
    fn a_split_too_large_to_price_costs_the_most_not_the_least() {
        assert_eq!(required_bytes(2, 3, 5, 7), 31);
        assert_eq!(required_bytes(0, u64::MAX, 0, 2), i64::MAX);
        assert_eq!(required_bytes(0, 1 << 62, 0, 4), i64::MAX);
        // Fits a u64, not an i64: the cast would make it negative.
        assert_eq!(required_bytes(0, 1 << 61, 0, 4), i64::MAX);
        assert_eq!(required_bytes(u64::MAX, u64::MAX, 1, 1), i64::MAX);
    }

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    #[test]
    fn the_budget_is_free_vram_less_the_weights_and_the_headroom() {
        assert_eq!(
            net_cache_budget_bytes(0.9, 10 * GIB, 4 * GIB, 0),
            (0.9 * (10 * GIB) as f64) as i64 - (4 * GIB) as i64
        );
        // An over-committed deployment gets a negative budget, not a
        // wrapped-around enormous one.
        assert!(net_cache_budget_bytes(0.5, GIB, 4 * GIB, 0) < 0);
    }

    #[test]
    fn the_startup_budget_prices_what_loading_actually_consumed() {
        assert_eq!(startup_kv_budget(0.9, 1000, 400), 900 - 600);
        assert_eq!(startup_kv_budget(1.0, 1000, 1000), 1000);
    }

    /// The expert cache is filled first: it is the pool whose shortfall
    /// is paid per token rather than per request.
    #[test]
    fn the_expert_cache_is_filled_before_kv_gets_the_remainder() {
        let sizes = plan_cache_budget(
            (8 * GIB) as i64,
            16 * MIB, // per expert
            MIB,      // per page
            32,       // experts per layer
            256,      // total experts
            false,
            64,
            u64::MAX,
        )
        .expect("fits");
        assert_eq!(sizes.moe_cache_slots, 256, "full residency");
        let spent = 256 * 16 * MIB;
        assert_eq!(sizes.kv_pages, (8 * GIB - spent) / MIB);
    }

    /// ... but never past full residency: extra slots buy nothing, so
    /// the bytes go to KV.
    #[test]
    fn the_expert_cache_is_capped_at_full_residency() {
        let sizes = plan_cache_budget((64 * GIB) as i64, MIB, MIB, 8, 64, false, 16, u64::MAX)
            .expect("fits");
        assert_eq!(sizes.moe_cache_slots, 64);
        assert!(sizes.kv_pages > 60_000);
    }

    /// A backend ceiling rolls the freed bytes into KV rather than
    /// wasting them.
    #[test]
    fn a_backend_slot_ceiling_gives_its_bytes_to_kv() {
        let capped =
            plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 4096, false, 16, 992).expect("fits");
        assert_eq!(capped.moe_cache_slots, 992);
        let uncapped = plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 4096, false, 16, u64::MAX)
            .expect("fits");
        assert!(uncapped.moe_cache_slots > capped.moe_cache_slots);
        assert!(capped.kv_pages > uncapped.kv_pages);
    }

    /// The KV floor is not negotiable: a server that cannot hold a
    /// context serves nothing, however good its expert residency is.
    #[test]
    fn the_kv_reserve_is_taken_out_before_the_cache_is_sized() {
        let reserve = 1024;
        let sizes = plan_cache_budget(
            (2 * GIB) as i64,
            MIB,
            MIB,
            8,
            4096,
            false,
            reserve,
            u64::MAX,
        )
        .expect("fits");
        assert!(sizes.kv_pages >= reserve);
        assert!(sizes.moe_cache_slots <= 2048 - reserve);
    }

    /// The prefill double buffer costs two layers of slots. A budget
    /// that cannot pay for it loses the buffer rather than the split.
    #[test]
    fn a_tight_budget_drops_the_prefill_double_buffer() {
        let roomy =
            plan_cache_budget((4 * GIB) as i64, MIB, MIB, 8, 64, true, 16, u64::MAX).unwrap();
        assert!(roomy.prefill_overlap);

        // A ceiling below two layers cannot host the buffer at all.
        let cramped = plan_cache_budget(
            (4 * GIB) as i64,
            MIB,
            MIB,
            8,
            64,
            true,
            16,
            12, // fewer than 2 * 8 slots
        )
        .unwrap();
        assert!(!cramped.prefill_overlap);
    }

    #[test]
    fn a_budget_that_cannot_hold_one_layer_is_refused() {
        let err = plan_cache_budget(MIB as i64, MIB, MIB, 8, 64, false, 16, u64::MAX).unwrap_err();
        assert!(err.needed_bytes > err.budget_bytes);
        assert_eq!(err.sizes.moe_cache_slots, 8, "one layer is the minimum");
    }

    #[test]
    fn expert_slot_cost_is_the_sum_over_banks() {
        assert_eq!(expert_bytes_per_slot(&[512, 256]), 768);
        assert_eq!(expert_bytes_per_slot(&[]), 0);
    }
}
