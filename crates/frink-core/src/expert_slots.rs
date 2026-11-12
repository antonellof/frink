//! The executor for [`crate::expert_cache`]'s plans: a bounded slot
//! pool, the copies a plan asks for, and the counter that says a warm
//! step moved nothing.
//!
//! # Why this module exists
//!
//! [`ExpertCache`](crate::expert_cache::ExpertCache) decides which
//! experts are resident and returns a
//! [`CopyPlan`](crate::expert_cache::CopyPlan) -- and that is where it
//! stops, deliberately: "nothing here moves bytes; the plan is the
//! whole output". Until something executes those plans the whole `q*`
//! split is inert, because a policy that decides how much to fetch
//! decides nothing while no fetch happens. This is the other half: it
//! takes a plan, validates it against a pool it owns the geometry of,
//! performs exactly the copies the plan names, and records what
//! crossed the link.
//!
//! # Still no device memory here
//!
//! The bytes live behind [`SlotDevice`], so this crate keeps holding
//! no tensors and no device allocations, and every rule below stays
//! testable on a host with no GPU at all.
//! [`HostSlotMemory`] is a real implementation of that trait rather
//! than a test mock -- it is the pool a CPU-only build uses, where the
//! "link" is a memcpy.
//!
//! # A plan is applied whole or not at all
//!
//! Every check happens in a pre-pass, before a single byte is written.
//! A plan is a unit: the cache has *already* recorded the residency the
//! plan describes by the time the caller gets it, so a half-applied
//! plan leaves the cache claiming an expert lives in a slot that holds
//! someone else's bytes. That does not fail -- it multiplies, and
//! returns a confident wrong answer. Refusing up front keeps the
//! device exactly as it was, which is a state the cache can be told to
//! resync to.
//!
//! A device fault is the one thing that can still land mid-plan, since
//! only the device knows it failed. Whatever it cost is marked unknown
//! here and named in the error, so the caller can hand the same slots
//! to
//! [`ExpertCache::forget_slot`](crate::expert_cache::ExpertCache::forget_slot)
//! and have the next step re-fetch them instead of reading them. How
//! much is suspect depends on when it failed: a refused copy costs its
//! own slot ([`SlotFault::Device`]), while a failed *flush* costs every
//! slot the plan wrote ([`SlotFault::DeviceFlush`]) -- a backend that
//! batches its transfers cannot say which of them landed.
//!
//! # Row size is checked exactly, not as a minimum
//!
//! A host row shorter than the slot would leave the slot's tail
//! holding the *previous* occupant's bytes: a coherent-looking expert
//! spliced from two, which produces plausible tokens rather than an
//! error. So a length mismatch in either direction is refused.

use crate::expert_cache::{CopyPlan, ExpertId, GatherPlan};
use crate::residency::{BankResidency, CopyRoute, ResidencyError};

/// The shape of one pool: how many slots, how many banks, and how wide
/// a row is in each bank.
///
/// `row_bytes` is per bank because the banks are not interchangeable:
/// gate and up are `[ffn_dim, hidden]` while down is `[hidden,
/// ffn_dim]`, and a checkpoint may quantize them differently, so one
/// row width for all three would be wrong for at least one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotGeometry {
    pub num_layers: usize,
    /// Slots in the pool. This is the `cache_size` the
    /// [`ExpertCache`](crate::expert_cache::ExpertCache) was built
    /// with; the two must agree or a slot the planner names will not
    /// exist here.
    pub slots: usize,
    /// Bytes per expert row, one entry per bank, in the bank order the
    /// caller uses everywhere else (conventionally gate, up, down).
    pub row_bytes: Vec<usize>,
}

impl SlotGeometry {
    pub fn banks(&self) -> usize {
        self.row_bytes.len()
    }

    /// Device bytes the whole pool occupies. What a VRAM budget is
    /// checked against before anything is allocated.
    pub fn bytes(&self) -> u64 {
        self.row_bytes
            .iter()
            .map(|b| *b as u64 * self.slots as u64)
            .sum()
    }
}

/// Where a slot's bytes actually live.
///
/// One implementation per backend. Both methods take a bank index and
/// absolute slot numbers; translating those into an address is the
/// implementation's business, which is what keeps this crate free of
/// device memory.
///
/// Implementations may defer the work and do it in [`flush`](Self::flush)
/// -- an async copy engine is the normal case -- but must not report
/// success for a copy they later discover failed without failing the
/// flush.
pub trait SlotDevice {
    /// Announces how the copies about to be issued must be carried
    /// out. Called once per plan, before the first copy.
    ///
    /// The default ignores it, which is right for any backend whose
    /// copies are already synchronous. A backend that captures work
    /// into a graph must not capture a
    /// [`CopyRoute::WholeLayerPageable`] plan: that route exists
    /// precisely because the layer's host rows have no device address,
    /// so its copy is a synchronous pageable one and a captured graph
    /// would replay a transfer whose source is not addressable.
    fn begin_plan(&mut self, route: CopyRoute) -> Result<(), String> {
        let _ = route;
        Ok(())
    }

    /// Writes one expert row from host memory into `dst_slot`. `src`
    /// is exactly `row_bytes[bank]` long; the implementation may rely
    /// on that.
    fn write_slot(&mut self, bank: usize, dst_slot: u32, src: &[u8]) -> Result<(), String>;

    /// Copies one slot onto another *within* the pool, without
    /// touching the host. This is what makes a prefill gather cheaper
    /// than a re-fetch: the bytes are already on the far side of the
    /// link.
    fn copy_slot(&mut self, bank: usize, dst_slot: u32, src_slot: u32) -> Result<(), String>;

    /// Completes any deferred copies. Called once per plan, after the
    /// last copy is issued.
    fn flush(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// The host-side expert rows a [`CopyPlan`] reads from.
///
/// Layer-local by construction, matching
/// [`CopyPlan::src_rows`](crate::expert_cache::CopyPlan::src_rows):
/// row `r` of layer `L`, not a flat index. That is what lets each
/// layer's bank live in its own allocation, which in turn is what lets
/// [`crate::residency`] give different layers different host residency
/// classes.
pub trait ExpertRows {
    /// The bytes of one expert row, or `None` if the bank has no such
    /// row. Returning a slice of the wrong length is refused by the
    /// caller rather than trusted.
    fn row(&self, bank: usize, layer: u32, row: u32) -> Option<&[u8]>;
}

/// What one applied plan moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Applied {
    /// Expert rows written, summed over banks.
    pub rows: u64,
    pub bytes: u64,
    /// True when the plan asked for nothing: every routed expert was
    /// already resident. The property real expert offload is judged
    /// on, per plan rather than per step.
    pub warm: bool,
}

/// Running totals for `/metrics` and for the A/B that says whether
/// residency is paying for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotStats {
    /// Plans applied, warm ones included.
    pub plans: u64,
    /// Plans that copied nothing at all.
    pub warm_plans: u64,
    /// Rows and bytes that crossed the link from host memory.
    pub host_rows: u64,
    pub host_bytes: u64,
    /// Rows and bytes moved slot-to-slot without touching the host.
    pub device_rows: u64,
    pub device_bytes: u64,
}

impl SlotStats {
    /// Fraction of plans that moved nothing. The headline number for
    /// expert residency: at `1.0` the pool holds the whole working set
    /// and decode issues no weight traffic at all.
    pub fn warm_plan_rate(&self) -> f64 {
        if self.plans == 0 {
            return 0.0;
        }
        self.warm_plans as f64 / self.plans as f64
    }

    /// Bytes that crossed the link per plan. What a link budget is
    /// spent against, and the figure a `q*` split is supposed to move.
    pub fn host_bytes_per_plan(&self) -> f64 {
        if self.plans == 0 {
            return 0.0;
        }
        self.host_bytes as f64 / self.plans as f64
    }
}

/// Why a plan was refused, or how applying it failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotFault {
    /// The two halves of a [`CopyPlan`] have different lengths, so
    /// pair `i` is not a pair. Applying it anyway would write row `i`
    /// into some *other* expert's slot -- silently wrong weights, the
    /// worst outcome available here.
    PlanHalvesDisagree { dst_slots: usize, src_rows: usize },
    /// A plan named a slot the pool does not have. The planner and
    /// this pool were built with different `cache_size` values.
    SlotOutOfRange { slot: u32, slots: usize },
    /// One plan writes the same slot twice. The second write wins and
    /// the first expert is simply absent from the slot the plan
    /// promised it in, while the cache records both as resident.
    SlotWrittenTwice { slot: u32 },
    /// A plan named a layer the pool was not built for.
    LayerOutOfRange { layer: u32, num_layers: usize },
    /// The host bank has no such row.
    RowMissing { bank: usize, layer: u32, row: u32 },
    /// A host row is not exactly one slot wide -- see the module docs
    /// on why a short row is worse than a missing one.
    RowSizeMismatch {
        bank: usize,
        layer: u32,
        row: u32,
        expected: usize,
        got: usize,
    },
    /// This layer's host residency does not permit the copy the plan
    /// describes; see [`crate::residency::CopyRoute`].
    Residency(ResidencyError),
    /// The device refused a copy. `slot` has been marked unknown here
    /// and must be forgotten by the planner too.
    Device {
        bank: usize,
        slot: u32,
        detail: String,
    },
    /// The device failed to complete a plan's deferred copies.
    ///
    /// Distinct from [`Device`](Self::Device) because no single slot is
    /// at fault: a backend that batches its copies cannot say which of
    /// them landed, so **every** slot the plan wrote is suspect. They
    /// are all marked unknown here and all named, so the planner can
    /// be told to forget the same set -- forgetting only the last one
    /// would leave the rest reading back as resident.
    DeviceFlush { slots: Vec<u32>, detail: String },
}

impl std::fmt::Display for SlotFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotFault::PlanHalvesDisagree {
                dst_slots,
                src_rows,
            } => write!(
                f,
                "copy plan has {dst_slots} destination slots and {src_rows} source rows: the \
                 pairs do not line up, so applying it would load experts into each other's slots"
            ),
            SlotFault::SlotOutOfRange { slot, slots } => write!(
                f,
                "plan names slot {slot} but the pool has {slots}: the planner and the pool were \
                 built with different cache sizes"
            ),
            SlotFault::SlotWrittenTwice { slot } => write!(
                f,
                "plan writes slot {slot} twice: one of the two experts would be absent from the \
                 slot the plan promised it in"
            ),
            SlotFault::LayerOutOfRange { layer, num_layers } => write!(
                f,
                "plan names layer {layer} but the pool was built for {num_layers}"
            ),
            SlotFault::RowMissing { bank, layer, row } => {
                write!(f, "bank {bank} has no row {row} for layer {layer}")
            }
            SlotFault::RowSizeMismatch {
                bank,
                layer,
                row,
                expected,
                got,
            } => write!(
                f,
                "bank {bank} row {row} of layer {layer} is {got} bytes, not {expected}: a \
                 short row would leave the slot's tail holding the previous occupant"
            ),
            SlotFault::Residency(e) => write!(f, "{e}"),
            SlotFault::Device { bank, slot, detail } => write!(
                f,
                "device refused the copy into bank {bank} slot {slot}: {detail}"
            ),
            SlotFault::DeviceFlush { slots, detail } => write!(
                f,
                "device failed to complete {} deferred cop{}: {detail}; slots {slots:?} are all \
                 suspect, since the backend cannot say which landed",
                slots.len(),
                if slots.len() == 1 { "y" } else { "ies" },
            ),
        }
    }
}

impl std::error::Error for SlotFault {}

impl From<ResidencyError> for SlotFault {
    fn from(e: ResidencyError) -> Self {
        SlotFault::Residency(e)
    }
}

/// A bounded pool of expert slots, and the executor for the plans that
/// fill it.
///
/// Holds no bytes: `occupant` is the pool's own record of what the
/// *device* contains, kept separately from the planner's residency map
/// on purpose. The two agreeing is the invariant; a single map could
/// not disagree, and so could never reveal that a copy failed.
pub struct ExpertSlots {
    geometry: SlotGeometry,
    residency: BankResidency,
    occupant: Vec<Option<ExpertId>>,
    stats: SlotStats,
}

impl ExpertSlots {
    /// Builds a pool, refusing a geometry no plan could be valid
    /// against.
    ///
    /// A zero-width bank is refused rather than tolerated: it would
    /// make every row size check pass trivially and every copy a
    /// no-op, so the pool would report warm plans forever while
    /// holding nothing.
    pub fn new(geometry: SlotGeometry) -> Result<Self, SlotFault> {
        if geometry.slots == 0 {
            return Err(SlotFault::SlotOutOfRange { slot: 0, slots: 0 });
        }
        if geometry.num_layers == 0 {
            return Err(SlotFault::LayerOutOfRange {
                layer: 0,
                num_layers: 0,
            });
        }
        for (bank, bytes) in geometry.row_bytes.iter().enumerate() {
            if *bytes == 0 {
                return Err(SlotFault::RowSizeMismatch {
                    bank,
                    layer: 0,
                    row: 0,
                    expected: 0,
                    got: 0,
                });
            }
        }
        let residency = BankResidency::all_pinned(geometry.num_layers);
        Ok(ExpertSlots {
            occupant: vec![None; geometry.slots],
            geometry,
            residency,
            stats: SlotStats::default(),
        })
    }

    /// Attaches the per-layer host residency this pool copies from.
    ///
    /// Without it every layer is assumed device-addressable, which is
    /// what an all-pinned host actually is. With it, a plan for an
    /// unpinned layer is refused here rather than issued to a device
    /// that has no address for those host rows.
    pub fn with_residency(mut self, residency: BankResidency) -> Self {
        self.residency = residency;
        self
    }

    pub fn geometry(&self) -> &SlotGeometry {
        &self.geometry
    }

    pub fn stats(&self) -> SlotStats {
        self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = SlotStats::default();
    }

    /// Which expert the *device* holds in `slot`, as far as this pool
    /// knows. `None` for an empty slot and for one whose last copy
    /// failed.
    pub fn occupant(&self, slot: u32) -> Option<ExpertId> {
        self.occupant.get(slot as usize).copied().flatten()
    }

    /// Slots currently holding a known expert.
    pub fn occupied(&self) -> usize {
        self.occupant.iter().filter(|o| o.is_some()).count()
    }

    /// Marks a slot as holding nothing known. Idempotent.
    pub fn invalidate_slot(&mut self, slot: u32) {
        if let Some(o) = self.occupant.get_mut(slot as usize) {
            *o = None;
        }
    }

    /// Forgets the whole pool. The counters survive: they describe
    /// traffic that really happened, and a resize does not unmake it.
    pub fn invalidate_all(&mut self) {
        self.occupant.iter_mut().for_each(|o| *o = None);
    }

    /// Resizes the pool, dropping everything resident.
    ///
    /// Slot ids are positions in an allocation that no longer exists,
    /// so keeping the occupancy map would point at other experts'
    /// bytes -- the same reasoning as
    /// [`ExpertCache::rebuild`](crate::expert_cache::ExpertCache::rebuild),
    /// and the two must be resized together or the planner will name
    /// slots this pool does not have.
    pub fn resize(&mut self, slots: usize) -> Result<(), SlotFault> {
        if slots == 0 {
            return Err(SlotFault::SlotOutOfRange { slot: 0, slots: 0 });
        }
        self.geometry.slots = slots;
        self.occupant = vec![None; slots];
        Ok(())
    }

    /// Applies an [`ensure`](crate::expert_cache::ExpertCache::ensure)
    /// plan: the LRU picked the slots, so this is the indexed device
    /// path and an unpinned layer cannot take it.
    pub fn apply_copy_plan(
        &mut self,
        layer: u32,
        plan: &CopyPlan,
        rows: &dyn ExpertRows,
        device: &mut dyn SlotDevice,
    ) -> Result<Applied, SlotFault> {
        self.apply_plan(layer, plan, false, rows, device)
    }

    /// Applies a
    /// [`materialize_layer`](crate::expert_cache::ExpertCache::materialize_layer)
    /// plan: the whole layer, slot `e` for expert `e`. This is the one
    /// shape an unpinned layer accepts, because it needs no device
    /// alias for the host rows.
    pub fn apply_materialize(
        &mut self,
        layer: u32,
        plan: &CopyPlan,
        rows: &dyn ExpertRows,
        device: &mut dyn SlotDevice,
    ) -> Result<Applied, SlotFault> {
        self.apply_plan(layer, plan, true, rows, device)
    }

    fn apply_plan(
        &mut self,
        layer: u32,
        plan: &CopyPlan,
        whole_layer: bool,
        rows: &dyn ExpertRows,
        device: &mut dyn SlotDevice,
    ) -> Result<Applied, SlotFault> {
        if plan.dst_slots.len() != plan.src_rows.len() {
            return Err(SlotFault::PlanHalvesDisagree {
                dst_slots: plan.dst_slots.len(),
                src_rows: plan.src_rows.len(),
            });
        }
        if layer as usize >= self.geometry.num_layers {
            return Err(SlotFault::LayerOutOfRange {
                layer,
                num_layers: self.geometry.num_layers,
            });
        }
        // Resolved even for an empty plan: a layer whose residency
        // forbids this route has a configuration problem, and a step
        // that happened to hit every expert is no evidence against it.
        let route = self.residency.copy_route(layer, whole_layer)?;

        self.validate_slots(&plan.dst_slots)?;
        self.validate_rows(layer, &plan.src_rows, rows)?;

        self.stats.plans += 1;
        if plan.is_empty() {
            self.stats.warm_plans += 1;
            return Ok(Applied {
                rows: 0,
                bytes: 0,
                warm: true,
            });
        }

        device
            .begin_plan(route)
            .map_err(|detail| SlotFault::DeviceFlush {
                slots: plan.dst_slots.clone(),
                detail,
            })?;

        let mut applied = Applied::default();
        for (&dst, &src) in plan.dst_slots.iter().zip(plan.src_rows.iter()) {
            // The occupant is cleared *before* the write, so a device
            // fault anywhere in this expert's banks leaves the slot
            // unknown rather than half-labelled.
            self.occupant[dst as usize] = None;
            for bank in 0..self.geometry.banks() {
                let bytes = rows
                    .row(bank, layer, src)
                    .expect("validated by validate_rows");
                device
                    .write_slot(bank, dst, bytes)
                    .map_err(|detail| SlotFault::Device {
                        bank,
                        slot: dst,
                        detail,
                    })?;
                applied.rows += 1;
                applied.bytes += bytes.len() as u64;
            }
            self.occupant[dst as usize] = Some(ExpertId { layer, expert: src });
        }
        self.flush(device, &plan.dst_slots)?;

        self.stats.host_rows += applied.rows;
        self.stats.host_bytes += applied.bytes;
        Ok(applied)
    }

    /// Completes the plan's copies, forgetting every slot it wrote if
    /// the device cannot confirm they landed.
    fn flush(&mut self, device: &mut dyn SlotDevice, written: &[u32]) -> Result<(), SlotFault> {
        let Err(detail) = device.flush() else {
            return Ok(());
        };
        for &slot in written {
            self.invalidate_slot(slot);
        }
        Err(SlotFault::DeviceFlush {
            slots: written.to_vec(),
            detail,
        })
    }

    /// Applies a [`GatherPlan`]: rows the prefill buffer can take from
    /// slots that already hold them, rather than from the host.
    ///
    /// A source slot whose occupant is unknown is refused. The planner
    /// believes it is resident; this pool knows a copy into it failed,
    /// and gathering from it would propagate garbage into a second
    /// slot while both are recorded as valid.
    pub fn apply_gather_plan(
        &mut self,
        plan: &GatherPlan,
        device: &mut dyn SlotDevice,
    ) -> Result<Applied, SlotFault> {
        if plan.dst_slots.len() != plan.src_slots.len() {
            return Err(SlotFault::PlanHalvesDisagree {
                dst_slots: plan.dst_slots.len(),
                src_rows: plan.src_slots.len(),
            });
        }
        self.validate_slots(&plan.dst_slots)?;
        for &src in &plan.src_slots {
            if src as usize >= self.geometry.slots {
                return Err(SlotFault::SlotOutOfRange {
                    slot: src,
                    slots: self.geometry.slots,
                });
            }
            if self.occupant(src).is_none() {
                return Err(SlotFault::Device {
                    bank: 0,
                    slot: src,
                    detail: "gather source holds no known expert; a failed copy would be \
                             propagated into a second slot"
                        .to_string(),
                });
            }
        }

        self.stats.plans += 1;
        if plan.is_empty() {
            self.stats.warm_plans += 1;
            return Ok(Applied {
                rows: 0,
                bytes: 0,
                warm: true,
            });
        }

        let mut applied = Applied::default();
        for (&dst, &src) in plan.dst_slots.iter().zip(plan.src_slots.iter()) {
            let carried = self.occupant(src);
            self.occupant[dst as usize] = None;
            for (bank, &row_bytes) in self.geometry.row_bytes.iter().enumerate() {
                device
                    .copy_slot(bank, dst, src)
                    .map_err(|detail| SlotFault::Device {
                        bank,
                        slot: dst,
                        detail,
                    })?;
                applied.rows += 1;
                applied.bytes += row_bytes as u64;
            }
            self.occupant[dst as usize] = carried;
        }
        self.flush(device, &plan.dst_slots)?;

        self.stats.device_rows += applied.rows;
        self.stats.device_bytes += applied.bytes;
        Ok(applied)
    }

    fn validate_slots(&self, slots: &[u32]) -> Result<(), SlotFault> {
        for (i, &slot) in slots.iter().enumerate() {
            if slot as usize >= self.geometry.slots {
                return Err(SlotFault::SlotOutOfRange {
                    slot,
                    slots: self.geometry.slots,
                });
            }
            // Quadratic, and deliberately so: a plan is one decode
            // step's misses -- a handful of entries -- so a scan beats
            // allocating a set, and this runs on the hot path.
            if slots[..i].contains(&slot) {
                return Err(SlotFault::SlotWrittenTwice { slot });
            }
        }
        Ok(())
    }

    fn validate_rows(
        &self,
        layer: u32,
        src_rows: &[u32],
        rows: &dyn ExpertRows,
    ) -> Result<(), SlotFault> {
        for &row in src_rows {
            for (bank, &expected) in self.geometry.row_bytes.iter().enumerate() {
                let Some(bytes) = rows.row(bank, layer, row) else {
                    return Err(SlotFault::RowMissing { bank, layer, row });
                };
                if bytes.len() != expected {
                    return Err(SlotFault::RowSizeMismatch {
                        bank,
                        layer,
                        row,
                        expected,
                        got: bytes.len(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// A pool whose "device" is host memory.
///
/// A real implementation, not a stand-in: on a CPU-only build the
/// expert pool *is* host memory and the link *is* a memcpy, so this is
/// the backend that build uses. It is also what makes every rule above
/// testable on a machine with no GPU, which is the whole reason the
/// device lives behind a trait.
pub struct HostSlotMemory {
    banks: Vec<Vec<u8>>,
    row_bytes: Vec<usize>,
}

impl HostSlotMemory {
    pub fn new(geometry: &SlotGeometry) -> Self {
        HostSlotMemory {
            banks: geometry
                .row_bytes
                .iter()
                .map(|b| vec![0u8; b * geometry.slots])
                .collect(),
            row_bytes: geometry.row_bytes.clone(),
        }
    }

    /// The bytes one slot holds in one bank. This is what makes the
    /// pool checkable: a test can assert the slot contains the expert
    /// the plan promised, not merely that a copy was counted.
    pub fn slot(&self, bank: usize, slot: u32) -> &[u8] {
        let w = self.row_bytes[bank];
        let at = w * slot as usize;
        &self.banks[bank][at..at + w]
    }
}

impl SlotDevice for HostSlotMemory {
    fn write_slot(&mut self, bank: usize, dst_slot: u32, src: &[u8]) -> Result<(), String> {
        let w = self.row_bytes[bank];
        let at = w * dst_slot as usize;
        self.banks[bank][at..at + w].copy_from_slice(src);
        Ok(())
    }

    fn copy_slot(&mut self, bank: usize, dst_slot: u32, src_slot: u32) -> Result<(), String> {
        let w = self.row_bytes[bank];
        let (dst, src) = (w * dst_slot as usize, w * src_slot as usize);
        self.banks[bank].copy_within(src..src + w, dst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_cache::ExpertCache;
    use crate::residency::HostResidency;

    const LAYERS: usize = 3;
    const EXPERTS: usize = 8;
    const GATE: usize = 6;
    const UP: usize = 6;
    const DOWN: usize = 4;

    fn geometry(slots: usize) -> SlotGeometry {
        SlotGeometry {
            num_layers: LAYERS,
            slots,
            row_bytes: vec![GATE, UP, DOWN],
        }
    }

    /// Host banks whose bytes name the expert they belong to, so a slot
    /// can be checked for *which* expert it holds rather than only for
    /// having been written.
    struct NamedRows {
        /// `[bank][layer * EXPERTS + row]`, materialized once so the
        /// trait can hand out borrows.
        banks: Vec<Vec<Vec<u8>>>,
    }

    impl NamedRows {
        fn new() -> Self {
            let banks = [GATE, UP, DOWN]
                .iter()
                .enumerate()
                .map(|(bank, &width)| {
                    (0..LAYERS as u32)
                        .flat_map(|layer| {
                            (0..EXPERTS as u32).map(move |row| named_row(bank, width, layer, row))
                        })
                        .collect()
                })
                .collect();
            NamedRows { banks }
        }

        fn expected(&self, bank: usize, layer: u32, row: u32) -> &[u8] {
            &self.banks[bank][layer as usize * EXPERTS + row as usize]
        }
    }

    /// Bytes that name the expert they belong to, so a slot can be
    /// checked for *which* expert it holds rather than only for having
    /// been written.
    fn named_row(bank: usize, width: usize, layer: u32, row: u32) -> Vec<u8> {
        (0..width)
            .map(|i| {
                (bank as u8 + 1)
                    .wrapping_mul(37)
                    .wrapping_add(layer as u8)
                    .wrapping_add((row as u8) << 3)
                    ^ i as u8
            })
            .collect()
    }

    impl ExpertRows for NamedRows {
        fn row(&self, bank: usize, layer: u32, row: u32) -> Option<&[u8]> {
            if bank >= self.banks.len() || layer as usize >= LAYERS || row as usize >= EXPERTS {
                return None;
            }
            Some(self.expected(bank, layer, row))
        }
    }

    /// The property real expert offload is judged on, and the one that
    /// was unverifiable while nothing executed a plan: once the working
    /// set is resident, a decode step copies **zero** bytes, and the
    /// counter says so.
    #[test]
    fn a_step_that_hits_the_cache_copies_nothing() {
        let mut cache = ExpertCache::new(LAYERS, EXPERTS, 32);
        let mut slots = ExpertSlots::new(geometry(32)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());

        let routed = [1u32, 4, 6];
        let cold = cache.ensure(0, &routed);
        let first = slots
            .apply_copy_plan(0, &cold.copy, &rows, &mut device)
            .unwrap();
        assert!(!first.warm);
        assert_eq!(first.rows, 3 * 3, "three experts across three banks");
        assert_eq!(first.bytes as usize, 3 * (GATE + UP + DOWN));

        let before = slots.stats();
        for _ in 0..10 {
            let warm = cache.ensure(0, &routed);
            assert!(warm.copy.is_empty(), "the cache should report every hit");
            let applied = slots
                .apply_copy_plan(0, &warm.copy, &rows, &mut device)
                .unwrap();
            assert!(applied.warm);
            assert_eq!(applied.bytes, 0);
        }
        let after = slots.stats();
        assert_eq!(
            after.host_bytes, before.host_bytes,
            "ten warm steps moved bytes"
        );
        assert_eq!(after.warm_plans, before.warm_plans + 10);
        assert_eq!(after.warm_plan_rate(), 10.0 / 11.0);
    }

    /// The plan is not merely counted: the slot the planner named holds
    /// the expert it named, in every bank. Counting a copy proves a
    /// copy happened, not that it went to the right place.
    #[test]
    fn every_slot_holds_the_expert_the_plan_promised() {
        let mut cache = ExpertCache::new(LAYERS, EXPERTS, 16);
        let mut slots = ExpertSlots::new(geometry(16)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());

        for layer in 0..LAYERS as u32 {
            let routed: Vec<u32> = (0..4).map(|e| (e + layer) % EXPERTS as u32).collect();
            let plan = cache.ensure(layer, &routed);
            slots
                .apply_copy_plan(layer, &plan.copy, &rows, &mut device)
                .unwrap();

            for (&expert, slot) in routed.iter().zip(plan.slots.iter()) {
                let slot = slot.expect("pure offload places every route");
                assert_eq!(
                    slots.occupant(slot),
                    Some(ExpertId { layer, expert }),
                    "layer {layer} expert {expert}"
                );
                for bank in 0..3 {
                    assert_eq!(
                        device.slot(bank, slot),
                        rows.expected(bank, layer, expert),
                        "layer {layer} expert {expert} bank {bank}"
                    );
                }
            }
        }
        assert_eq!(slots.occupied(), 12, "three layers of four experts");
    }

    /// An eviction must overwrite the slot's bytes, not merely its
    /// label. A pool that relabels without copying reports a hit for
    /// the new expert and multiplies the old one's weights.
    #[test]
    fn an_evicted_slot_is_overwritten_and_not_merely_relabelled() {
        // Two slots, two experts per layer, two layers: layer 1's
        // routes can only land on slots layer 0 already holds. A cache
        // must be able to hold one whole layer, so forcing eviction
        // takes a second layer rather than a smaller cache.
        let mut cache = ExpertCache::new(2, 2, 2);
        let mut slots = ExpertSlots::new(SlotGeometry {
            num_layers: 2,
            slots: 2,
            row_bytes: vec![GATE, UP, DOWN],
        })
        .unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());

        let first = cache.ensure(0, &[0, 1]);
        slots
            .apply_copy_plan(0, &first.copy, &rows, &mut device)
            .unwrap();
        let second = cache.ensure(1, &[0, 1]);
        assert_eq!(second.missing, 2, "layer 1 must evict layer 0");
        slots
            .apply_copy_plan(1, &second.copy, &rows, &mut device)
            .unwrap();

        for (&expert, slot) in [0u32, 1].iter().zip(second.slots.iter()) {
            let slot = slot.unwrap();
            assert_eq!(slots.occupant(slot), Some(ExpertId { layer: 1, expert }));
            assert_eq!(
                device.slot(0, slot),
                rows.expected(0, 1, expert),
                "the slot must hold layer 1's bytes, not layer 0's"
            );
        }
        assert_eq!(cache.slot_of(0, 0), None, "layer 0's expert 0 was evicted");
    }

    /// A plan whose halves disagree is refused before anything is
    /// written. Pair `i` would not be a pair, so applying it loads
    /// experts into each other's slots -- which does not fail, it
    /// returns a confident wrong answer.
    #[test]
    fn a_plan_whose_halves_disagree_is_refused_untouched() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        let plan = CopyPlan {
            dst_slots: vec![0, 1, 2],
            src_rows: vec![0, 1],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &rows, &mut device),
            Err(SlotFault::PlanHalvesDisagree {
                dst_slots: 3,
                src_rows: 2,
            })
        );
        assert_eq!(slots.occupied(), 0);
        assert_eq!(
            slots.stats().plans,
            0,
            "a refused plan is not an applied one"
        );
    }

    /// Two writes to one slot in one plan: the second wins and the
    /// first expert is absent from the slot the plan promised it in,
    /// while the planner records both as resident.
    #[test]
    fn a_plan_that_writes_one_slot_twice_is_refused() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        let plan = CopyPlan {
            dst_slots: vec![3, 1, 3],
            src_rows: vec![0, 1, 2],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &rows, &mut device),
            Err(SlotFault::SlotWrittenTwice { slot: 3 })
        );
        assert_eq!(slots.occupied(), 0);
    }

    /// A planner built with a larger cache than the pool names slots
    /// that do not exist. Refused rather than clamped: clamping would
    /// silently place two experts in one slot.
    #[test]
    fn a_slot_the_pool_does_not_have_is_refused() {
        let mut slots = ExpertSlots::new(geometry(4)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        let plan = CopyPlan {
            dst_slots: vec![0, 9],
            src_rows: vec![0, 1],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &rows, &mut device),
            Err(SlotFault::SlotOutOfRange { slot: 9, slots: 4 })
        );
        assert_eq!(slots.occupied(), 0);
    }

    /// A row that is not exactly one slot wide would leave the tail of
    /// the slot holding the previous occupant: an expert spliced from
    /// two, which produces plausible tokens rather than an error.
    #[test]
    fn a_row_that_is_not_exactly_one_slot_wide_is_refused() {
        struct ShortDownBank;
        impl ExpertRows for ShortDownBank {
            fn row(&self, bank: usize, _layer: u32, _row: u32) -> Option<&[u8]> {
                match bank {
                    0 => Some(&[0u8; GATE]),
                    1 => Some(&[0u8; UP]),
                    _ => Some(&[0u8; DOWN - 1]),
                }
            }
        }
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let mut device = HostSlotMemory::new(slots.geometry());
        let plan = CopyPlan {
            dst_slots: vec![0],
            src_rows: vec![5],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &ShortDownBank, &mut device),
            Err(SlotFault::RowSizeMismatch {
                bank: 2,
                layer: 0,
                row: 5,
                expected: DOWN,
                got: DOWN - 1,
            })
        );
        assert_eq!(slots.occupied(), 0, "nothing was written");
    }

    /// A missing row is refused with the bank that lacks it, so the
    /// caller learns which bank is short rather than that "a copy
    /// failed".
    #[test]
    fn a_missing_host_row_names_its_bank() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        let plan = CopyPlan {
            dst_slots: vec![0],
            src_rows: vec![EXPERTS as u32],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &rows, &mut device),
            Err(SlotFault::RowMissing {
                bank: 0,
                layer: 0,
                row: EXPERTS as u32,
            })
        );
    }

    /// The failure the executor makes reachable. `copy_route` and
    /// `SlotRemapOnUnpinnedLayer` were written for exactly this and had
    /// no caller: an LRU slot remap needs a device alias for the host
    /// rows, which an unpinned layer does not have. The whole-layer
    /// materialize is the one shape it does accept.
    #[test]
    fn an_unpinned_layer_refuses_an_lru_remap_but_takes_a_materialize() {
        // Layer 1 is pageable, and so is on the CPU executor -- the
        // pairing `BankResidency::new` insists on, since an unpinned
        // layer the GPU cannot see must be computed somewhere.
        let residency = BankResidency::new(
            &[
                HostResidency::Pinned,
                HostResidency::Pageable,
                HostResidency::Pinned,
            ],
            LAYERS,
            &[1u32].into_iter().collect(),
            false,
        )
        .unwrap();
        let mut slots = ExpertSlots::new(geometry(EXPERTS * LAYERS))
            .unwrap()
            .with_residency(residency);
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());

        let remap = CopyPlan {
            dst_slots: vec![0],
            src_rows: vec![3],
        };
        assert!(matches!(
            slots.apply_copy_plan(1, &remap, &rows, &mut device),
            Err(SlotFault::Residency(
                ResidencyError::SlotRemapOnUnpinnedLayer { layer: 1 }
            ))
        ));

        let whole = CopyPlan {
            dst_slots: (0..EXPERTS as u32).collect(),
            src_rows: (0..EXPERTS as u32).collect(),
        };
        let applied = slots
            .apply_materialize(1, &whole, &rows, &mut device)
            .unwrap();
        assert_eq!(applied.rows, EXPERTS as u64 * 3);

        // A pinned layer takes either route.
        assert!(slots.apply_copy_plan(0, &remap, &rows, &mut device).is_ok());
    }

    /// A device fault leaves the slot marked unknown rather than
    /// labelled with the expert whose copy failed. Reading it back as
    /// residency is the failure this prevents; the error names the slot
    /// so the planner can be told to forget it too.
    #[test]
    fn a_device_fault_leaves_its_slot_unknown_and_names_it() {
        struct FailsOnDownBank;
        impl SlotDevice for FailsOnDownBank {
            fn write_slot(&mut self, bank: usize, _d: u32, _s: &[u8]) -> Result<(), String> {
                if bank == 2 {
                    return Err("out of device memory".to_string());
                }
                Ok(())
            }
            fn copy_slot(&mut self, _b: usize, _d: u32, _s: u32) -> Result<(), String> {
                Ok(())
            }
        }
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let plan = CopyPlan {
            dst_slots: vec![5],
            src_rows: vec![2],
        };
        let err = slots
            .apply_copy_plan(0, &plan, &rows, &mut FailsOnDownBank)
            .unwrap_err();
        assert_eq!(
            err,
            SlotFault::Device {
                bank: 2,
                slot: 5,
                detail: "out of device memory".to_string(),
            }
        );
        assert_eq!(
            slots.occupant(5),
            None,
            "a slot whose copy failed must not read back as resident"
        );
    }

    /// A flush that fails must fail the plan, and must forget **every**
    /// slot the plan wrote -- not just the last one.
    ///
    /// A backend that batches its copies reports each one as issued and
    /// only discovers at flush that they did not land, and it cannot
    /// say which. Blaming one slot would leave the others reading back
    /// as resident while holding whatever the failed transfer left.
    #[test]
    fn a_failing_flush_forgets_every_slot_the_plan_wrote() {
        struct FlushFails;
        impl SlotDevice for FlushFails {
            fn write_slot(&mut self, _b: usize, _d: u32, _s: &[u8]) -> Result<(), String> {
                Ok(())
            }
            fn copy_slot(&mut self, _b: usize, _d: u32, _s: u32) -> Result<(), String> {
                Ok(())
            }
            fn flush(&mut self) -> Result<(), String> {
                Err("copy engine reported an error".to_string())
            }
        }
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let plan = CopyPlan {
            dst_slots: vec![1, 4, 6],
            src_rows: vec![0, 2, 3],
        };
        assert_eq!(
            slots.apply_copy_plan(0, &plan, &rows, &mut FlushFails),
            Err(SlotFault::DeviceFlush {
                slots: vec![1, 4, 6],
                detail: "copy engine reported an error".to_string(),
            })
        );
        assert_eq!(
            slots.occupied(),
            0,
            "no slot may read back as resident after an unconfirmed flush"
        );
    }

    /// A gather moves bytes that are already across the link, and the
    /// counters keep it separate from host traffic: a link budget is
    /// spent by the host side alone, so folding the two together would
    /// report a device-to-device copy as bandwidth consumed.
    #[test]
    fn a_gather_is_counted_separately_from_host_traffic() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        slots
            .apply_copy_plan(
                0,
                &CopyPlan {
                    dst_slots: vec![4],
                    src_rows: vec![6],
                },
                &rows,
                &mut device,
            )
            .unwrap();

        let gather = GatherPlan {
            dst_slots: vec![1],
            src_slots: vec![4],
        };
        let applied = slots.apply_gather_plan(&gather, &mut device).unwrap();
        assert_eq!(applied.rows, 3);
        assert_eq!(applied.bytes as usize, GATE + UP + DOWN);

        let stats = slots.stats();
        assert_eq!(stats.device_bytes as usize, GATE + UP + DOWN);
        assert_eq!(
            stats.host_bytes as usize,
            GATE + UP + DOWN,
            "the gather must not be billed to the link"
        );
        assert_eq!(
            slots.occupant(1),
            Some(ExpertId {
                layer: 0,
                expert: 6
            }),
            "the gathered slot carries the source's identity"
        );
        assert_eq!(device.slot(1, 1), rows.expected(1, 0, 6));
    }

    /// Gathering from a slot this pool knows nothing about is refused.
    /// The planner believes it is resident; the pool knows a copy into
    /// it failed, and a gather would launder that garbage into a second
    /// slot that both then record as valid.
    #[test]
    fn a_gather_from_an_unknown_slot_is_refused() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let mut device = HostSlotMemory::new(slots.geometry());
        let gather = GatherPlan {
            dst_slots: vec![0],
            src_slots: vec![7],
        };
        assert!(matches!(
            slots.apply_gather_plan(&gather, &mut device),
            Err(SlotFault::Device { slot: 7, .. })
        ));
    }

    /// A resize drops residency, because a slot id is a position in an
    /// allocation that no longer exists. The counters survive: they
    /// describe traffic that really happened.
    #[test]
    fn a_resize_drops_residency_but_keeps_the_counters() {
        let mut slots = ExpertSlots::new(geometry(8)).unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());
        slots
            .apply_copy_plan(
                0,
                &CopyPlan {
                    dst_slots: vec![0, 1],
                    src_rows: vec![0, 1],
                },
                &rows,
                &mut device,
            )
            .unwrap();
        let before = slots.stats();
        assert_eq!(slots.occupied(), 2);

        slots.resize(64).unwrap();
        assert_eq!(slots.occupied(), 0);
        assert_eq!(slots.geometry().slots, 64);
        assert_eq!(slots.stats(), before);
        assert_eq!(
            slots.resize(0),
            Err(SlotFault::SlotOutOfRange { slot: 0, slots: 0 })
        );
    }

    /// A zero-width bank would make every size check pass and every
    /// copy a no-op, so the pool would report warm plans forever while
    /// holding nothing. Refused at construction.
    #[test]
    fn a_degenerate_geometry_is_refused_at_construction() {
        assert!(ExpertSlots::new(geometry(0)).is_err());
        assert!(ExpertSlots::new(SlotGeometry {
            num_layers: 0,
            slots: 4,
            row_bytes: vec![GATE],
        })
        .is_err());
        assert!(ExpertSlots::new(SlotGeometry {
            num_layers: 1,
            slots: 4,
            row_bytes: vec![GATE, 0],
        })
        .is_err());
    }

    /// The pool's device footprint, which a VRAM budget is checked
    /// against before anything is allocated.
    #[test]
    fn the_geometry_reports_the_device_bytes_it_needs() {
        let g = geometry(1024);
        assert_eq!(g.bytes(), 1024 * (GATE + UP + DOWN) as u64);
        assert_eq!(g.banks(), 3);
    }

    /// Forgetting a slot in the planner and in the pool must leave the
    /// two agreeing: the next step treats it as a miss and re-fetches,
    /// rather than reading a slot whose copy failed.
    #[test]
    fn a_forgotten_slot_is_refetched_by_the_next_step() {
        let mut cache = ExpertCache::new(1, EXPERTS, 8);
        let mut slots = ExpertSlots::new(SlotGeometry {
            num_layers: 1,
            slots: 8,
            row_bytes: vec![GATE, UP, DOWN],
        })
        .unwrap();
        let rows = NamedRows::new();
        let mut device = HostSlotMemory::new(slots.geometry());

        let plan = cache.ensure(0, &[2]);
        slots
            .apply_copy_plan(0, &plan.copy, &rows, &mut device)
            .unwrap();
        let slot = plan.slots[0].unwrap();

        assert_eq!(
            cache.forget_slot(slot),
            Some(ExpertId {
                layer: 0,
                expert: 2
            })
        );
        slots.invalidate_slot(slot);

        let again = cache.ensure(0, &[2]);
        assert_eq!(again.missing, 1, "a forgotten expert must miss");
        assert!(!again.copy.is_empty(), "and must be re-fetched");
        slots
            .apply_copy_plan(0, &again.copy, &rows, &mut device)
            .unwrap();
        assert_eq!(
            slots.occupant(again.slots[0].unwrap()),
            Some(ExpertId {
                layer: 0,
                expert: 2
            })
        );
        assert_eq!(
            again.slots[0],
            Some(slot),
            "a forgotten slot is the first candidate, so the re-fetch reclaims \
             it rather than spending a slot that really holds an expert"
        );

        // Forgetting a slot that holds nothing is a no-op, so a caller
        // recovering from a fault need not first work out whether the
        // copy got far enough to be recorded.
        let empty = (0..8).find(|&s| cache.resident_in(s).is_none()).unwrap();
        assert_eq!(cache.forget_slot(empty), None);
        assert_eq!(cache.forget_slot(9_999), None, "and an unknown slot too");
    }
}
