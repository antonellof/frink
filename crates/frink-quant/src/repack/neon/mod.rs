//! aarch64 kernels for the interleaved repack tier, one file per kind
//! family. The dispatchers in `super::*` call these as `neon::<fn>`,
//! which the re-exports below keep true after the split.

use std::arch::aarch64::*;

mod q4_0x4;
mod q4_kx8;
mod q5_kx8;
mod q6_kx8;
mod q8_0x4;

pub(crate) use q4_0x4::*;
pub(crate) use q4_kx8::*;
pub(crate) use q5_kx8::*;
pub(crate) use q6_kx8::*;
pub(crate) use q8_0x4::*;

#[target_feature(enable = "neon,i8mm")]
unsafe fn vmmla_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    std::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    acc
}

#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot_lane(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t, lane: u32) -> int32x4_t {
    // sdot Vd.4S, Vn.16B, Vm.4B[lane]
    match lane {
        0 => std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[0]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        ),
        1 => std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[1]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        ),
        2 => std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[2]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        ),
        3 => std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[3]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        ),
        _ => unreachable!(),
    }
    acc
}

/// `sdot Vd.4S, Vn.16B, Vm.16B` -- the plain (non-lane) signed dot.
/// Stable SDOT through inline asm (`vdotq_s32` is nightly-only); the one
/// copy every int8 kernel in the crate calls.
#[target_feature(enable = "neon,dotprod")]
pub(crate) unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    std::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack),
    );
    acc
}
