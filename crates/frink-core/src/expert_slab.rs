//! Slot memory the CPU can address: the two [`SlotDevice`]s the expert
//! store executes its residency policy on.
//!
//! # Two devices, one shape
//!
//! [`HostSlotSlab`] is an anonymous mapping, for a CPU-only build.
//! `MetalSlotSlab` (feature `metal`) is one Shared `MTLBuffer`
//! (`frink_metal::expert_slab::MetalSlab`), for Apple silicon. Both are
//! [`SlotSlab`] over a different [`SlabRegion`], so the slot arithmetic
//! and the copy rules are written once: the only thing that differs is
//! who allocated the bytes.
//!
//! # Why the Metal one is not a second budget
//!
//! On unified memory a host byte and a device byte are the same RAM.
//! The Metal slab is therefore not a device copy of a host cache; it IS
//! the cache. The CPU kernels read it through [`SlotMemory::slot_ptr`],
//! and a GPU launch over a slice inside it binds the slab at an offset
//! (the weight cache aliases it and charges nothing). The expert store
//! sizes the slab from its budget and counts it once, in
//! [`crate::expert_store::committed_expert_bytes`].
//!
//! # One bank
//!
//! [`SlotGeometry`] allows several banks (gate, up, down rows in
//! separate arrays) because a CUDA gather wants them that way. A slot
//! here holds one expert's whole byte image, in the layout its
//! `ExpertSource` defines, so the geometry always has exactly one bank.

use std::ptr::NonNull;

use crate::expert_slots::{SlotDevice, SlotGeometry};

/// Memory a slab of slots lives in.
///
/// # Safety
///
/// `base()` must be valid and writable for `len()` bytes for as long as
/// the region lives, and never move. `SlotSlab` reads and writes through
/// it -- `write_slot`, `copy_slot` and `slot_ptr` all add a slot offset
/// to it and copy `slot_bytes` bytes without checking either number --
/// so an implementation that reports a longer `len()` than it allocated
/// turns those into reads and writes past the end of its own buffer, in
/// safe code. An out-of-range `slot` is refused; a lying `base()`/`len()`
/// pair cannot be, which is why the trait is `unsafe`.
pub unsafe trait SlabRegion: Send + Sync {
    /// Where the bytes live, for reports and tests: `"host"`, `"metal"`.
    const KIND: &'static str;

    /// First byte. Valid and writable for [`len`](Self::len) bytes for
    /// as long as the region lives, and never moves.
    fn base(&self) -> NonNull<u8>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A slot pool whose slots the CPU can read in place.
///
/// What the expert store holds: [`SlotDevice`] for the writes, and
/// [`slot_ptr`](Self::slot_ptr) for building weight views over a
/// resident slot without copying it out again.
///
/// # Safety
///
/// `slot_ptr(slot)` must be valid for `geometry().row_bytes[0]` bytes
/// for as long as the value lives, and `geometry()` must be the geometry
/// those bytes really hold. `SlotPool::new` caches one pointer per slot
/// and `ExpertBytes::as_slice` builds a `&[u8]` of the source's length
/// from that pointer, so an implementation returning a pointer into a
/// shorter buffer makes the store read past it, in safe code. Nothing
/// can check the promise at runtime -- there is no portable way to ask
/// an allocation how big it is -- so it is stated as an invariant of an
/// `unsafe trait` instead: an implementation cannot be written by safe
/// code, and the `SAFETY:` note in `as_slice` can only be reached from
/// one that spells the promise out.
pub unsafe trait SlotMemory: SlotDevice + Send + Sync {
    fn geometry(&self) -> &SlotGeometry;

    /// [`SlabRegion::KIND`] of the memory behind the slots.
    fn kind(&self) -> &'static str;

    /// First byte of `slot`, valid for `geometry().row_bytes[0]` bytes
    /// for as long as this value lives. Panics on an out-of-range slot.
    fn slot_ptr(&self, slot: u32) -> NonNull<u8>;
}

/// `slots` slots of `slot_bytes` each, back to back in one region.
pub struct SlotSlab<R: SlabRegion> {
    region: R,
    geometry: SlotGeometry,
}

impl<R: SlabRegion> SlotSlab<R> {
    /// Refuses a region too small for the geometry, or a geometry with
    /// other than one bank (see the module docs).
    pub fn new(
        region: R,
        num_layers: usize,
        slots: usize,
        slot_bytes: usize,
    ) -> Result<Self, String> {
        let need = slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| format!("{slots} slots of {slot_bytes} bytes overflows"))?;
        if region.len() < need {
            return Err(format!(
                "slab region is {} bytes, {slots} slots of {slot_bytes} need {need}",
                region.len()
            ));
        }
        Ok(SlotSlab {
            region,
            geometry: SlotGeometry {
                num_layers,
                slots,
                row_bytes: vec![slot_bytes],
            },
        })
    }

    fn slot_bytes(&self) -> usize {
        self.geometry.row_bytes[0]
    }

    fn check(&self, bank: usize, slot: u32) -> Result<(), String> {
        if bank != 0 {
            return Err(format!("an expert slab has one bank, not bank {bank}"));
        }
        if slot as usize >= self.geometry.slots {
            return Err(format!(
                "slot {slot} out of range ({} slots)",
                self.geometry.slots
            ));
        }
        Ok(())
    }
}

impl<R: SlabRegion> SlotDevice for SlotSlab<R> {
    fn write_slot(&mut self, bank: usize, dst_slot: u32, src: &[u8]) -> Result<(), String> {
        self.check(bank, dst_slot)?;
        // Exact, as `expert_slots` requires of every device: a short row
        // would leave the previous occupant's tail in the slot.
        if src.len() != self.slot_bytes() {
            return Err(format!(
                "row is {} bytes, slot is {}",
                src.len(),
                self.slot_bytes()
            ));
        }
        let dst = self.slot_ptr(dst_slot);
        // SAFETY: `dst` is the start of an in-range slot (checked
        // above), valid for `slot_bytes` bytes, and `src` is exactly that
        // long. `src` is a caller's buffer, never this region, so the two
        // do not overlap. The owner writes a slot only while no lease
        // points into it, so no reader observes the write.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), src.len()) };
        Ok(())
    }

    fn copy_slot(&mut self, bank: usize, dst_slot: u32, src_slot: u32) -> Result<(), String> {
        self.check(bank, dst_slot)?;
        self.check(bank, src_slot)?;
        if dst_slot == src_slot {
            return Ok(());
        }
        let (dst, src) = (self.slot_ptr(dst_slot), self.slot_ptr(src_slot));
        // SAFETY: two distinct in-range slots of one region are disjoint
        // `slot_bytes`-long ranges, both valid for as long as `self`.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), self.slot_bytes()) };
        Ok(())
    }
}

unsafe impl<R: SlabRegion> SlotMemory for SlotSlab<R> {
    fn geometry(&self) -> &SlotGeometry {
        &self.geometry
    }

    fn kind(&self) -> &'static str {
        R::KIND
    }

    fn slot_ptr(&self, slot: u32) -> NonNull<u8> {
        assert!(
            (slot as usize) < self.geometry.slots,
            "slot {slot} out of range ({} slots)",
            self.geometry.slots
        );
        let at = slot as usize * self.slot_bytes();
        // SAFETY: `at + slot_bytes <= slots * slot_bytes <= region.len()`
        // (checked in `new`), so the offset stays inside the region.
        unsafe { NonNull::new_unchecked(self.region.base().as_ptr().add(at)) }
    }
}

/// An anonymous private mapping: page-aligned, zero-filled, and freed
/// on drop.
pub struct HostRegion {
    map: memmap2::MmapMut,
    base: NonNull<u8>,
}

// SAFETY: `base` points into `map`, which this value owns and never
// remaps. Access to the bytes is serialised by the owner of the slab
// (see `SlotSlab::write_slot`), not by this handle.
unsafe impl Send for HostRegion {}
unsafe impl Sync for HostRegion {}

impl HostRegion {
    pub fn new(len: usize) -> std::io::Result<Self> {
        let mut map = memmap2::MmapMut::map_anon(len.max(1))?;
        let base = NonNull::new(map.as_mut_ptr()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::OutOfMemory, "null anonymous map")
        })?;
        Ok(HostRegion { map, base })
    }
}

unsafe impl SlabRegion for HostRegion {
    const KIND: &'static str = "host";

    fn base(&self) -> NonNull<u8> {
        self.base
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

/// The CPU-only build's expert slot pool.
pub type HostSlotSlab = SlotSlab<HostRegion>;

impl HostSlotSlab {
    pub fn host(num_layers: usize, slots: usize, slot_bytes: usize) -> Result<Self, String> {
        let len = slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| "slab size overflows".to_string())?;
        let region = HostRegion::new(len).map_err(|e| format!("host slab: {e}"))?;
        SlotSlab::new(region, num_layers, slots, slot_bytes)
    }
}

#[cfg(feature = "metal")]
unsafe impl SlabRegion for frink_metal::expert_slab::MetalSlab {
    const KIND: &'static str = "metal";

    fn base(&self) -> NonNull<u8> {
        frink_metal::expert_slab::MetalSlab::base(self)
    }

    fn len(&self) -> usize {
        frink_metal::expert_slab::MetalSlab::len(self)
    }
}

/// Apple silicon's expert slot pool: one Shared `MTLBuffer` the CPU
/// writes and both the CPU and the GPU read in place.
#[cfg(feature = "metal")]
pub type MetalSlotSlab = SlotSlab<frink_metal::expert_slab::MetalSlab>;

#[cfg(feature = "metal")]
impl MetalSlotSlab {
    pub fn metal(num_layers: usize, slots: usize, slot_bytes: usize) -> Result<Self, String> {
        let len = slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| "slab size overflows".to_string())?;
        let region = frink_metal::expert_slab::MetalSlab::new(len)
            .map_err(|e| format!("Metal slab: {e}"))?;
        SlotSlab::new(region, num_layers, slots, slot_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(slab: &HostSlotSlab, s: u32) -> &[u8] {
        let n = slab.geometry().row_bytes[0];
        // SAFETY: `slot_ptr` is valid for `row_bytes[0]` bytes while
        // `slab` lives, and the borrow ties the slice to it.
        unsafe { std::slice::from_raw_parts(slab.slot_ptr(s).as_ptr(), n) }
    }

    /// The slab writes exactly the slot it is told to and nothing
    /// beside it: a write that strayed by one slot would hand an
    /// expert's neighbour its bytes, which multiplies without error.
    #[test]
    fn a_write_lands_in_its_slot_and_nowhere_else() {
        let mut slab = HostSlotSlab::host(2, 4, 16).unwrap();
        slab.write_slot(0, 2, &[5u8; 16]).unwrap();
        for s in 0..4 {
            let want = if s == 2 { 5 } else { 0 };
            assert!(slot(&slab, s).iter().all(|&b| b == want), "slot {s}");
        }
        slab.copy_slot(0, 0, 2).unwrap();
        assert!(slot(&slab, 0).iter().all(|&b| b == 5));
        assert!(slot(&slab, 1).iter().all(|&b| b == 0));
    }

    /// The `expert_slots` contract, held here too: a row that is not
    /// exactly one slot long is refused, so a short expert cannot
    /// inherit its predecessor's tail; and an out-of-range slot or a
    /// second bank is an error, not a write past the region.
    #[test]
    fn a_wrong_length_row_a_bad_slot_and_a_second_bank_are_refused() {
        let mut slab = HostSlotSlab::host(1, 2, 8).unwrap();
        assert!(slab.write_slot(0, 0, &[1u8; 7]).is_err());
        assert!(slab.write_slot(0, 2, &[1u8; 8]).is_err());
        assert!(slab.write_slot(1, 0, &[1u8; 8]).is_err());
        assert!(slab.copy_slot(0, 0, 5).is_err());
        assert!(
            slot(&slab, 0).iter().all(|&b| b == 0),
            "refusals write nothing"
        );
    }

    /// The contracts on [`SlabRegion`](Self::SlabRegion) and
    /// [`SlotMemory`](Self::SlotMemory) are invariants an `unsafe trait`
    /// enforces, not prose a caller may ignore: an implementation that
    /// has not read them cannot be written in safe code at all. Dropping
    /// the `unsafe` compiles again and quietly reopens both reports these
    /// two traits close -- a safe `SlotMemory::slot_ptr` over a short
    /// buffer has `ExpertBytes::as_slice` read past its allocation, and a
    /// safe `SlabRegion::base` does the same through `write_slot` -- so
    /// the modifier is pinned here, in the idiom of
    /// `frink-models/tests/documented_counts.rs`: prose and code agree,
    /// or a test says so.
    #[test]
    fn the_slot_contracts_are_unsafe_trait_invariants() {
        let src = include_str!("expert_slab.rs");
        for name in ["SlabRegion", "SlotMemory"] {
            // Built at runtime: the text searched is this same file, so a
            // needle spelled out in one piece would match this test's own
            // source and pass for a file that declares a safe trait.
            let declared = format!("pub unsafe trait {name}");
            let safe = format!("pub trait {name}");
            assert!(
                src.contains(&declared),
                "expert_slab.rs must declare `{declared}`"
            );
            assert!(
                !src.contains(&safe),
                "`{safe}` is a safe trait, which is the bug"
            );
        }
    }

    #[test]
    fn a_region_smaller_than_the_geometry_is_refused() {
        let region = HostRegion::new(10).unwrap();
        // An anonymous map rounds up to a page, so ask for more than one.
        let too_many = region.len() / 8 + 1;
        assert!(SlotSlab::new(region, 1, too_many, 8).is_err());
    }
}
