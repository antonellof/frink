//! The IQ1/IQ2/IQ3 grid formats: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
//! IQ3_S, IQ1_S and IQ1_M. Each is a [`MulMmKind`] row, the grid tables
//! its dequant reads ([`IqGrid`], emitted from `frink_quant::iq_tables`
//! by [`crate::iq_grids`]), and the Rust twin of its `dequant_src`
//! statement for statement, reading the same `frink_quant` tables.
//!
//! # Provenance
//!
//! The arithmetic is ggml's: `ggml-cuda/dequantize.cuh` at the pin
//! (`dequantize_iq2_xxs` .. `dequantize_iq1_m`, lines 274 to 404) and
//! the CPU's `dequantize_row_iq*`, which `frink_quant`'s `for_each_iq*`
//! walks transcribe. The two agree on every product: a scale
//! `d * (0.5 + s) * 0.25` (IQ2), `d * (0.5 + s) * 0.5` (IQ3_XXS),
//! `d * (1 + 2 s)` (IQ3_S) or `d * (2 s + 1)` (IQ1), times a grid
//! magnitude, times a sign, left to right. ggml CUDA splits a block
//! across 32 threads of eight values; this file's partition is the
//! GEMM's sixteen (`il`), so `il` names 32-element group `il / 2` and
//! its half `il % 2`: grid entries `2 (il % 2)` and `2 (il % 2) + 1` of
//! the group's four. The twins are held to `frink_quant`'s whole-block
//! dequant (`every_dequant_twin_matches_the_cpu_dequant`), and the
//! gather to it value for value.
//!
//! # UNVERIFIED ON HARDWARE
//!
//! No kernel here has executed on a GPU.

use crate::iq_grids::IqGrid;
use crate::mul_mm::{f16_to_f32, MulMmKind, SUB};
use frink_quant::iq_tables as t;

fn f16_at(xb: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from(xb[at]) | (u16::from(xb[at + 1]) << 8))
}

/// `-1` when bit `j` of `signs` is set, else `1`: `frink_quant`'s
/// `iq_sign`.
fn sign(signs: u8, j: usize) -> f32 {
    if signs & t::KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

/// The eight magnitudes of a packed IQ2 grid row, times `db` and the
/// signs, into `out`.
fn iq2_row(out: &mut [f32], db: f32, row: u64, signs: u8) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        let mag = ((row >> (8 * j)) & 0xFF) as f32;
        *o = db * mag * sign(signs, j);
    }
}

/// IQ2_XXS: `half d`, 32 `u16` code words. 66 bytes.
pub const IQ2_XXS: MulMmKind = MulMmKind {
    name: "IQ2_XXS",
    module_name: "frink_mul_mm_iq2_xxs",
    fn_name: "iq2_xxs_mul_mm",
    block_bytes: 66,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const unsigned char* g = xb + 2 + 8 * (il / 2);
    const unsigned int aux32 = (unsigned int)g[4] | ((unsigned int)g[5] << 8)
        | ((unsigned int)g[6] << 16) | ((unsigned int)g[7] << 24);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.25f;
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned long long row = FRINK_IQ2XXS_GRID[g[l]];
        const unsigned char signs = FRINK_KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127u];
        for (int j = 0; j < 8; j++) {
            const float mag = (float)((row >> (8 * j)) & 0xFFull);
            reg[8 * k + j] = db * mag * ((signs & FRINK_KMASK_IQ2XS[j]) ? -1.0f : 1.0f);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq2_xxs,
    codebook: None,
    grids: &[IqGrid::Kmask, IqGrid::Ksigns, IqGrid::Iq2xxs],
};

fn dequant_sub_iq2_xxs(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let g = &xb[2 + 8 * (il / 2)..];
    let aux32 = u32::from_le_bytes([g[4], g[5], g[6], g[7]]);
    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.25;
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let row = t::IQ2XXS_GRID[g[l] as usize];
        let signs = t::KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
        iq2_row(&mut reg[8 * k..], db, row, signs);
    }
}

/// IQ2_XS: `half d`, 32 `u16` codes (9-bit grid index, 7-bit sign
/// index), 8 scale bytes. 74 bytes.
pub const IQ2_XS: MulMmKind = MulMmKind {
    name: "IQ2_XS",
    module_name: "frink_mul_mm_iq2_xs",
    fn_name: "iq2_xs_mul_mm",
    block_bytes: 74,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const int ib32 = il / 2;
    const unsigned char sc = xb[66 + ib32];
    const float db = (il % 2)
        ? d * (0.5f + (float)(sc >> 4)) * 0.25f
        : d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned char* c = xb + 2 + 8 * ib32 + 2 * l;
        const unsigned int code = (unsigned int)c[0] | ((unsigned int)c[1] << 8);
        const unsigned long long row = FRINK_IQ2XS_GRID[code & 511u];
        const unsigned char signs = FRINK_KSIGNS_IQ2XS[code >> 9];
        for (int j = 0; j < 8; j++) {
            const float mag = (float)((row >> (8 * j)) & 0xFFull);
            reg[8 * k + j] = db * mag * ((signs & FRINK_KMASK_IQ2XS[j]) ? -1.0f : 1.0f);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq2_xs,
    codebook: None,
    grids: &[IqGrid::Kmask, IqGrid::Ksigns, IqGrid::Iq2xs],
};

fn dequant_sub_iq2_xs(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let ib32 = il / 2;
    let sc = xb[66 + ib32];
    let db = if !il.is_multiple_of(2) {
        d * (0.5 + (sc >> 4) as f32) * 0.25
    } else {
        d * (0.5 + (sc & 0x0F) as f32) * 0.25
    };
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let at = 2 + 8 * ib32 + 2 * l;
        let code = u16::from_le_bytes([xb[at], xb[at + 1]]);
        let row = t::IQ2XS_GRID[(code & 511) as usize];
        let signs = t::KSIGNS_IQ2XS[(code >> 9) as usize];
        iq2_row(&mut reg[8 * k..], db, row, signs);
    }
}

/// IQ2_S: `half d`, 32 low-index bytes, 32 sign bytes, 8 `qh` bytes, 8
/// scale bytes. 82 bytes.
pub const IQ2_S: MulMmKind = MulMmKind {
    name: "IQ2_S",
    module_name: "frink_mul_mm_iq2_s",
    fn_name: "iq2_s_mul_mm",
    block_bytes: 82,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const int ib32 = il / 2;
    const unsigned char sc = xb[74 + ib32];
    const float db = (il % 2)
        ? d * (0.5f + (float)(sc >> 4)) * 0.25f
        : d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
    const unsigned int qh = xb[66 + ib32];
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned int idx = (unsigned int)xb[2 + 4 * ib32 + l]
            | ((qh << (8 - 2 * l)) & 0x300u);
        const unsigned long long row = FRINK_IQ2S_GRID[idx];
        const unsigned char signs = xb[34 + 4 * ib32 + l];
        for (int j = 0; j < 8; j++) {
            const float mag = (float)((row >> (8 * j)) & 0xFFull);
            reg[8 * k + j] = db * mag * ((signs & FRINK_KMASK_IQ2XS[j]) ? -1.0f : 1.0f);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq2_s,
    codebook: None,
    grids: &[IqGrid::Kmask, IqGrid::Iq2s],
};

fn dequant_sub_iq2_s(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let ib32 = il / 2;
    let sc = xb[74 + ib32];
    let db = if !il.is_multiple_of(2) {
        d * (0.5 + (sc >> 4) as f32) * 0.25
    } else {
        d * (0.5 + (sc & 0x0F) as f32) * 0.25
    };
    let qh = u32::from(xb[66 + ib32]);
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let idx = u32::from(xb[2 + 4 * ib32 + l]) | ((qh << (8 - 2 * l)) & 0x300);
        let row = t::IQ2S_GRID[idx as usize];
        let signs = xb[34 + 4 * ib32 + l];
        iq2_row(&mut reg[8 * k..], db, row, signs);
    }
}

/// IQ3_XXS: `half d`, 64 grid-index bytes, 8 `u32` scale/sign words.
/// 98 bytes.
pub const IQ3_XXS: MulMmKind = MulMmKind {
    name: "IQ3_XXS",
    module_name: "frink_mul_mm_iq3_xxs",
    fn_name: "iq3_xxs_mul_mm",
    block_bytes: 98,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const int ib32 = il / 2;
    const unsigned char* s = xb + 66 + 4 * ib32;
    const unsigned int aux32 = (unsigned int)s[0] | ((unsigned int)s[1] << 8)
        | ((unsigned int)s[2] << 16) | ((unsigned int)s[3] << 24);
    const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned char signs = FRINK_KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127u];
        const unsigned int g1 = FRINK_IQ3XXS_GRID[xb[2 + 8 * ib32 + 2 * l]];
        const unsigned int g2 = FRINK_IQ3XXS_GRID[xb[2 + 8 * ib32 + 2 * l + 1]];
        for (int j = 0; j < 4; j++) {
            reg[8 * k + j] = db * (float)((g1 >> (8 * j)) & 0xFFu)
                * ((signs & FRINK_KMASK_IQ2XS[j]) ? -1.0f : 1.0f);
        }
        for (int j = 0; j < 4; j++) {
            reg[8 * k + 4 + j] = db * (float)((g2 >> (8 * j)) & 0xFFu)
                * ((signs & FRINK_KMASK_IQ2XS[j + 4]) ? -1.0f : 1.0f);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq3_xxs,
    codebook: None,
    grids: &[IqGrid::Kmask, IqGrid::Ksigns, IqGrid::Iq3xxs],
};

/// Two packed IQ3 grid rows of four magnitudes, signed by bits 0..8.
fn iq3_rows(out: &mut [f32], db: f32, g1: u32, g2: u32, signs: u8) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        let g = if j < 4 { g1 } else { g2 };
        *o = db * ((g >> (8 * (j % 4))) & 0xFF) as f32 * sign(signs, j);
    }
}

fn dequant_sub_iq3_xxs(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let ib32 = il / 2;
    let s = &xb[66 + 4 * ib32..];
    let aux32 = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let signs = t::KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
        let g1 = t::IQ3XXS_GRID[xb[2 + 8 * ib32 + 2 * l] as usize];
        let g2 = t::IQ3XXS_GRID[xb[2 + 8 * ib32 + 2 * l + 1] as usize];
        iq3_rows(&mut reg[8 * k..], db, g1, g2, signs);
    }
}

/// IQ3_S: `half d`, 64 low-index bytes, 8 `qh` bytes, 32 sign bytes, 4
/// scale bytes. 110 bytes.
pub const IQ3_S: MulMmKind = MulMmKind {
    name: "IQ3_S",
    module_name: "frink_mul_mm_iq3_s",
    fn_name: "iq3_s_mul_mm",
    block_bytes: 110,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const int ib32 = il / 2;
    const unsigned int nib = (ib32 % 2)
        ? ((unsigned int)xb[106 + ib32 / 2] >> 4)
        : ((unsigned int)xb[106 + ib32 / 2] & 0x0Fu);
    const float db = d * (1.0f + 2.0f * (float)nib);
    const unsigned int h = xb[66 + ib32];
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned int i1 = (unsigned int)xb[2 + 8 * ib32 + 2 * l]
            | ((h << (8 - 2 * l)) & 256u);
        const unsigned int i2 = (unsigned int)xb[2 + 8 * ib32 + 2 * l + 1]
            | ((h << (7 - 2 * l)) & 256u);
        const unsigned int g1 = FRINK_IQ3S_GRID[i1];
        const unsigned int g2 = FRINK_IQ3S_GRID[i2];
        const unsigned char signs = xb[74 + 4 * ib32 + l];
        for (int j = 0; j < 4; j++) {
            reg[8 * k + j] = db * (float)((g1 >> (8 * j)) & 0xFFu)
                * ((signs & FRINK_KMASK_IQ2XS[j]) ? -1.0f : 1.0f);
        }
        for (int j = 0; j < 4; j++) {
            reg[8 * k + 4 + j] = db * (float)((g2 >> (8 * j)) & 0xFFu)
                * ((signs & FRINK_KMASK_IQ2XS[j + 4]) ? -1.0f : 1.0f);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq3_s,
    codebook: None,
    grids: &[IqGrid::Kmask, IqGrid::Iq3s],
};

fn dequant_sub_iq3_s(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let ib32 = il / 2;
    let nib = if !ib32.is_multiple_of(2) {
        xb[106 + ib32 / 2] >> 4
    } else {
        xb[106 + ib32 / 2] & 0x0F
    };
    let db = d * (1.0 + 2.0 * nib as f32);
    let h = u32::from(xb[66 + ib32]);
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let i1 = u32::from(xb[2 + 8 * ib32 + 2 * l]) | ((h << (8 - 2 * l)) & 256);
        let i2 = u32::from(xb[2 + 8 * ib32 + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256);
        let g1 = t::IQ3S_GRID[i1 as usize];
        let g2 = t::IQ3S_GRID[i2 as usize];
        let signs = xb[74 + 4 * ib32 + l];
        iq3_rows(&mut reg[8 * k..], db, g1, g2, signs);
    }
}

/// IQ1_S: `half d`, 32 low-index bytes, 8 `u16` words (3 high index
/// bits per code, a 3-bit scale, the delta's sign). 50 bytes.
pub const IQ1_S: MulMmKind = MulMmKind {
    name: "IQ1_S",
    module_name: "frink_mul_mm_iq1_s",
    fn_name: "iq1_s_mul_mm",
    block_bytes: 50,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    const float d = frink_f16_to_f32(
        (unsigned short)xb[0] | ((unsigned short)xb[1] << 8));
    const int ib = il / 2;
    const unsigned int h = (unsigned int)xb[34 + 2 * ib] | ((unsigned int)xb[35 + 2 * ib] << 8);
    const float dl = d * (2.0f * (float)((h >> 12) & 7u) + 1.0f);
    const float delta = (h & 0x8000u) ? -0.125f : 0.125f;
    for (int k = 0; k < 2; k++) {
        const int l = 2 * (il % 2) + k;
        const unsigned int gi = (unsigned int)xb[2 + 4 * ib + l] | (((h >> (3 * l)) & 7u) << 8);
        const unsigned long long row = FRINK_IQ1S_GRID[gi];
        for (int j = 0; j < 8; j++) {
            const signed char v = (signed char)(unsigned char)((row >> (8 * j)) & 0xFFull);
            reg[8 * k + j] = dl * ((float)v + delta);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq1_s,
    codebook: None,
    grids: &[IqGrid::Iq1s],
};

/// One IQ1 grid row of eight signed bytes, shifted by `delta` and
/// scaled by `dl`.
fn iq1_row(out: &mut [f32], dl: f32, row: u64, delta: f32) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        let v = ((row >> (8 * j)) & 0xFF) as u8 as i8;
        *o = dl * (f32::from(v) + delta);
    }
}

fn dequant_sub_iq1_s(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let d = f16_at(xb, 0);
    let ib = il / 2;
    let h = u32::from(xb[34 + 2 * ib]) | (u32::from(xb[35 + 2 * ib]) << 8);
    let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
    let delta = if h & 0x8000 != 0 { -0.125 } else { 0.125 };
    for k in 0..2 {
        let l = 2 * (il % 2) + k;
        let gi = u32::from(xb[2 + 4 * ib + l]) | (((h >> (3 * l)) & 7) << 8);
        iq1_row(&mut reg[8 * k..], dl, t::IQ1S_GRID[gi as usize], delta);
    }
}

/// IQ1_M: 32 low-index bytes, 16 `qh` bytes, 8 scale bytes whose top
/// nibbles assemble the f16 scale. 56 bytes, no `half d` field.
pub const IQ1_M: MulMmKind = MulMmKind {
    name: "IQ1_M",
    module_name: "frink_mul_mm_iq1_m",
    fn_name: "iq1_m_mul_mm",
    block_bytes: 56,
    block_elems: 256,
    dequant_src: r#"
__device__ __forceinline__ void frink_dequant_sub(
    const unsigned char* xb, int il, float* reg
) {
    unsigned int sc[4];
    for (int k = 0; k < 4; k++) {
        sc[k] = (unsigned int)xb[48 + 2 * k] | ((unsigned int)xb[49 + 2 * k] << 8);
    }
    const float d = frink_f16_to_f32((unsigned short)((sc[0] >> 12)
        | ((sc[1] >> 8) & 0x00F0u) | ((sc[2] >> 4) & 0x0F00u) | (sc[3] & 0xF000u)));
    const int ib = il / 2;
    const int half = il % 2;
    const int shift = 6 * (ib % 2) + 3 * half;
    const float dl = d * (2.0f * (float)((sc[ib / 2] >> shift) & 7u) + 1.0f);
    const unsigned int hb = xb[32 + 2 * ib + half];
    for (int k = 0; k < 2; k++) {
        const int l = 2 * half + k;
        const unsigned int gi = (unsigned int)xb[4 * ib + l]
            | ((hb << (k ? 4 : 8)) & 0x700u);
        const float delta = (hb & (k ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        const unsigned long long row = FRINK_IQ1S_GRID[gi];
        for (int j = 0; j < 8; j++) {
            const signed char v = (signed char)(unsigned char)((row >> (8 * j)) & 0xFFull);
            reg[8 * k + j] = dl * ((float)v + delta);
        }
    }
}
"#,
    dequant_twin: dequant_sub_iq1_m,
    codebook: None,
    grids: &[IqGrid::Iq1s],
};

fn dequant_sub_iq1_m(xb: &[u8], il: usize, reg: &mut [f32; SUB]) {
    let sc: [u32; 4] =
        std::array::from_fn(|k| u32::from(xb[48 + 2 * k]) | (u32::from(xb[49 + 2 * k]) << 8));
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    let d = f16_to_f32(bits as u16);
    let ib = il / 2;
    let half = il % 2;
    let shift = 6 * (ib % 2) + 3 * half;
    let dl = d * (2.0 * ((sc[ib / 2] >> shift) & 7) as f32 + 1.0);
    let hb = u32::from(xb[32 + 2 * ib + half]);
    for k in 0..2 {
        let l = 2 * half + k;
        let gi = u32::from(xb[4 * ib + l]) | ((hb << if k != 0 { 4 } else { 8 }) & 0x700);
        let delta = if hb & (if k != 0 { 0x80 } else { 0x08 }) != 0 {
            -0.125
        } else {
            0.125
        };
        iq1_row(&mut reg[8 * k..], dl, t::IQ1S_GRID[gi as usize], delta);
    }
}
