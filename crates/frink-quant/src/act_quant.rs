//! The activation quantizers in front of every integer `vec_dot`: ggml's
//! `Q8_K` (the K-quants, IQ and TQ formats) and `Q8_0` (Q8_0, Q4_0/1,
//! Q5_0/1, IQ4_NL, MXFP4), bit for bit with what llama.cpp's CPU backend
//! computes at the pin (ADR 0009, #476).
//!
//! Bit for bit, not "within quantization error": a quantizer is a
//! staircase, and an input on a step's edge lands on one side or the
//! other. With the old forms (`d = amax/127`, `inv = 1/d`, half away from
//! zero, `Q8_0`'s `d` kept in f32) frink's matmul outputs, fed llama.cpp's
//! own dumped inputs, differed from libllama's by up to 2e-3 per op on
//! Llama-3.2-1B Q8_0 and 3e-3 on gpt-oss-20b. With these they are
//! identical (`frink parity --dump-tensors`; the per-layer tables are in
//! #476's PR).

use crate::{Q4_K_BLOCK_ELEMS, Q8_0_BLOCK_ELEMS};
use half::f16;

/// An activation vector quantized to signed 8-bit in 32-element blocks,
/// each with its own scale (`d`), so it can feed the integer `vec_dot`
/// paths against Q8_0-family weights: ggml's `block_q8_0`.
#[derive(Clone, Debug)]
pub struct Q8Activations {
    /// Signed 8-bit quantized values, `n_blocks * 32` long.
    pub q: Vec<i8>,
    /// Per-block scale, `n_blocks` long. `x ≈ q * d`. Held as f32 but
    /// always an f16 value: `block_q8_0` stores `d` as f16 and every ggml
    /// dot reads it back from there.
    pub d: Vec<f32>,
}

impl Q8Activations {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }
}

/// ggml `block_q8_K` activations for K-quant int-dot (`Q4_K`/`Q5_K`/`Q6_K`).
/// Super-blocks of 256 elements with 16-wide `bsums` for the min term.
#[derive(Clone, Debug)]
pub struct Q8KActivations {
    pub q: Vec<i8>,
    /// Per-block scale, never negative; see [`quantize_activations_q8_k`].
    pub d: Vec<f32>,
    /// Per 16-wide group sums of `q`, `n_blocks * 16` long.
    pub bsums: Vec<i16>,
}

impl Q8KActivations {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }
}

/// Quantize activations to ggml `Q8_K` (256-element super-blocks):
/// `quantize_row_q8_K_ref` (`ggml-quants.c`), which `quantize_row_q8_K`
/// calls on every ISA, and whose arithmetic the repack quantizers
/// (`ggml_quantize_mat_q8_K_4x4/4x8`, scalar and AVX2) repeat:
///
/// - `iscale = 127/amax`, `d = 1/iscale`. NOT `d = amax/127` and
///   `1/d`: the two differ by an ulp on about one block in five;
/// - `q = nearest_int(iscale * x)`: round half to EVEN.
///
/// ggml takes `iscale = -127/max`, with `max` the signed value of the
/// first largest element, so its `d` carries that element's sign. This
/// keeps `d >= 0` instead. That negates `d`, `q` and `bsums` together,
/// and every product the dots take of them comes out with the same bits:
/// IEEE negation is exact and round-half-even is odd-symmetric. `bsums`
/// enable the K-quant min correction without re-scanning `q`.
pub fn quantize_activations_q8_k(x: &[f32]) -> Q8KActivations {
    debug_assert_eq!(x.len() % Q4_K_BLOCK_ELEMS, 0);
    let n_blocks = x.len() / Q4_K_BLOCK_ELEMS;
    let mut q = vec![0i8; n_blocks * Q4_K_BLOCK_ELEMS];
    let mut d = vec![0f32; n_blocks];
    let mut bsums = vec![0i16; n_blocks * 16];
    // Serial on purpose: every batch caller is already inside a Rayon
    // region (one task per activation), so an inner region here nested
    // ~batch_size fork-joins per matmul; and one row's blocks are far too
    // little work to amortize one. llama quantizes serially per thread
    // chunk too (`ggml_compute_forward_mul_mat`, `ggml-cpu.c`).
    for (b, chunk) in x.as_chunks::<Q4_K_BLOCK_ELEMS>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            // ggml zeroes `d` and `qs` and leaves `bsums` unwritten; zero
            // is what they sum to.
            continue;
        }
        let iscale = 127.0 / amax;
        let q_slot = &mut q[b * Q4_K_BLOCK_ELEMS..(b + 1) * Q4_K_BLOCK_ELEMS];
        for (slot, &v) in q_slot.iter_mut().zip(chunk) {
            // |iscale * v| <= 127 up to one rounding, so the clamp never
            // binds; it keeps -128, which the AVX2 sign trick cannot
            // negate, out of `q` whatever the input.
            *slot = (iscale * v).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
        d[b] = 1.0 / iscale;
        for (slot, group) in bsums[b * 16..(b + 1) * 16]
            .iter_mut()
            .zip(q_slot.as_chunks::<16>().0)
        {
            *slot = group.iter().map(|&q| q as i32).sum::<i32>() as i16;
        }
    }
    Q8KActivations { q, d, bsums }
}

/// How the host's ggml computes `block_q8_0` activations. ggml's
/// `quantize_row_q8_0` is per ISA, and the three bodies are three
/// answers:
///
/// | | inverse scale | rounding |
/// |---|---|---|
/// | NEON (`arch/arm/quants.c`) | `1/d` | `vcvtnq`: half to even |
/// | AVX (`arch/x86/quants.c`) | `127/amax` | `_MM_ROUND_NEAREST`: half to even |
/// | scalar (`quantize_row_q8_0_ref`) | `1/d` | `roundf`: half away from zero |
///
/// with `d = amax/127` stored as f16 in all three. The repack quantizers
/// (`ggml_quantize_mat_q8_0_4x4/4x8`) repeat their ISA's row body.
///
/// Every host builds all three so every host tests all three; a host
/// constructs only its own outside the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
enum Q8_0Form {
    Neon,
    Avx,
    Scalar,
}

impl Q8_0Form {
    fn host() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx") {
                Self::Avx
            } else {
                Self::Scalar
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            Self::Scalar
        }
    }
}

/// Quantize an activation row to [`Q8Activations`]: ggml's
/// `quantize_row_q8_0` as the CPU backend runs it on this host (see
/// [`Q8_0Form`]). `x.len()` must be a multiple of 32.
pub fn quantize_activations_q8(x: &[f32]) -> Q8Activations {
    quantize_q8_0_as(x, Q8_0Form::host())
}

fn quantize_q8_0_as(x: &[f32], form: Q8_0Form) -> Q8Activations {
    debug_assert_eq!(x.len() % Q8_0_BLOCK_ELEMS, 0);
    let n_blocks = x.len() / Q8_0_BLOCK_ELEMS;
    let mut q = vec![0i8; n_blocks * Q8_0_BLOCK_ELEMS];
    let mut d = vec![0f32; n_blocks];
    // Serial on purpose: see `quantize_activations_q8_k`. The parallel
    // split this replaces was also 32-byte `q` chunks (two per cache
    // line) with adjacent `d` writes: false sharing on every store.
    for (b, chunk) in x.as_chunks::<Q8_0_BLOCK_ELEMS>().0.iter().enumerate() {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = match form {
            Q8_0Form::Avx if amax != 0.0 => 127.0 / amax,
            Q8_0Form::Neon | Q8_0Form::Scalar if scale != 0.0 => 1.0 / scale,
            _ => 0.0,
        };
        // The quants use the f32 inverse; only `d` goes through f16.
        d[b] = f16::from_f32(scale).to_f32();
        for (slot, &v) in q[b * Q8_0_BLOCK_ELEMS..(b + 1) * Q8_0_BLOCK_ELEMS]
            .iter_mut()
            .zip(chunk)
        {
            let scaled = v * inv;
            let r = match form {
                Q8_0Form::Neon | Q8_0Form::Avx => scaled.round_ties_even(),
                Q8_0Form::Scalar => scaled.round(),
            };
            // Never binds (see `quantize_activations_q8_k`).
            *slot = r.clamp(-127.0, 127.0) as i8;
        }
    }
    Q8Activations { q, d }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every expected value below is libggml's own output at the pin
    /// (`quantize_row_q8_K_ref` from libggml-base, the NEON
    /// `quantize_row_q8_0` from libggml-cpu, called through ctypes on an
    /// M2), not a re-derivation. The old forms' answers are noted beside
    /// them: each fixture is one the old form gets wrong.
    fn block_k(head: &[f32]) -> Vec<f32> {
        let mut x = vec![0f32; Q4_K_BLOCK_ELEMS];
        x[..head.len()].copy_from_slice(head);
        x
    }

    fn block_0(head: &[f32]) -> Vec<f32> {
        let mut x = vec![0f32; Q8_0_BLOCK_ELEMS];
        x[..head.len()].copy_from_slice(head);
        x
    }

    /// `amax = 127` makes `iscale` exactly 1, so every `.5` is a tie.
    /// ggml: d = -1 (the largest element is positive), q = -127, -2, 4,
    /// 0, 0, -126. Old form: 127, 3, -4, 1, -1, 127 (half away from zero).
    #[test]
    fn q8_k_rounds_ties_to_even() {
        let act = quantize_activations_q8_k(&block_k(&[127.0, 2.5, -3.5, 0.5, -0.5, 126.5]));
        assert_eq!(act.d[0], 1.0);
        assert_eq!(&act.q[..6], &[127, 2, -4, 0, 0, 126]);
        assert_eq!(act.bsums[0], 127 + 2 - 4 + 126);
    }

    /// An `amax` where `1/(127/amax)` and `amax/127` are an ulp apart, and
    /// an element that rounds differently because of it (not a tie).
    /// ggml: d = 0x3c6636d1, q[1] = 8. Old form: d = 0x3c6636d2, q[1] = 7.
    #[test]
    fn q8_k_takes_its_scale_from_the_inverse() {
        let amax = f32::from_bits(0x3fe4_6a64);
        let x = f32::from_bits(0x3dd7_d365);
        let act = quantize_activations_q8_k(&block_k(&[-amax, x]));
        assert_eq!(act.d[0].to_bits(), 0x3c66_36d1);
        assert_eq!(&act.q[..2], &[-127, 8]);
    }

    /// ggml negates `d` and `q` when the largest element is positive;
    /// this keeps `d` positive. The products are the same either way.
    #[test]
    fn q8_k_scale_is_never_negative() {
        let neg = quantize_activations_q8_k(&block_k(&[-127.0, 2.5]));
        let pos = quantize_activations_q8_k(&block_k(&[127.0, -2.5]));
        assert_eq!((neg.d[0], pos.d[0]), (1.0, 1.0));
        assert_eq!((neg.q[1], pos.q[1]), (2, -2));
        for (a, b) in neg.q.iter().zip(&pos.q) {
            assert_eq!(*a, -*b);
        }
    }

    #[test]
    fn q8_k_all_zero_block_is_all_zero() {
        let act = quantize_activations_q8_k(&[0f32; Q4_K_BLOCK_ELEMS]);
        assert_eq!(act.d[0], 0.0);
        assert!(act.q.iter().all(|&q| q == 0));
        assert!(act.bsums.iter().all(|&s| s == 0));
    }

    /// NEON, ties: ggml d = 1, q = 127, 2, -4, 0, 2. Old form: 127, 3, -4,
    /// 1, 2.
    #[test]
    fn q8_0_rounds_ties_to_even() {
        let act = quantize_q8_0_as(&block_0(&[127.0, 2.5, -3.5, 0.5, 1.5]), Q8_0Form::Neon);
        assert_eq!(act.d[0], 1.0);
        assert_eq!(&act.q[..5], &[127, 2, -4, 0, 2]);
        let avx = quantize_q8_0_as(&block_0(&[127.0, 2.5, -3.5, 0.5, 1.5]), Q8_0Form::Avx);
        assert_eq!(&avx.q[..5], &[127, 2, -4, 0, 2]);
        // `quantize_row_q8_0_ref` is the one body that rounds away.
        let scalar = quantize_q8_0_as(&block_0(&[127.0, 2.5, -3.5, 0.5, 1.5]), Q8_0Form::Scalar);
        assert_eq!(&scalar.q[..5], &[127, 3, -4, 1, 2]);
    }

    /// `d` is what `block_q8_0` holds, an f16: amax 1 gives d = f16
    /// 0x2008 (0.00787353515625), not 1/127 = 0.0078740157. The quants
    /// come from the f32 inverse: q = 127, 64.
    #[test]
    fn q8_0_scale_is_stored_as_f16() {
        let act = quantize_q8_0_as(&block_0(&[1.0, 0.5]), Q8_0Form::Neon);
        assert_eq!(act.d[0], f16::from_bits(0x2008).to_f32());
        assert_ne!(act.d[0], 1.0 / 127.0);
        assert_eq!(&act.q[..2], &[127, 64]);
    }

    /// The NEON and AVX bodies part on an `amax` where `1/(amax/127)` and
    /// `127/amax` are an ulp apart. libggml (NEON): q[1] = 2, d = f16
    /// 0x24a9; the AVX body gives 1.
    #[test]
    fn q8_0_inverse_scale_follows_the_isa() {
        let x = block_0(&[f32::from_bits(0x4013_e709), f32::from_bits(0x3cdf_99c0)]);
        let neon = quantize_q8_0_as(&x, Q8_0Form::Neon);
        assert_eq!(neon.d[0], f16::from_bits(0x24a9).to_f32());
        assert_eq!(&neon.q[..2], &[127, 2]);
        assert_eq!(quantize_q8_0_as(&x, Q8_0Form::Avx).q[1], 1);
    }

    #[test]
    fn q8_0_all_zero_block_is_all_zero() {
        for form in [Q8_0Form::Neon, Q8_0Form::Avx, Q8_0Form::Scalar] {
            let act = quantize_q8_0_as(&[0f32; 32], form);
            assert_eq!(act.d[0], 0.0);
            assert!(act.q.iter().all(|&q| q == 0));
        }
    }

    #[test]
    fn q8_0_reconstructs_within_quant_error() {
        let x: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.13 - 4.0).sin() * 3.0)
            .collect();
        let act = quantize_activations_q8(&x);
        assert_eq!(act.n_blocks(), 2);
        assert_eq!(act.q.len(), 64);
        for (b, chunk) in x.as_chunks::<32>().0.iter().enumerate() {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            // Half a step, plus what the f16 scale costs at |q| = 127.
            let tol = amax / 127.0 * (0.5 + 127.0 / 2048.0) + 1e-6;
            for (i, &v) in chunk.iter().enumerate() {
                let recon = act.q[b * 32 + i] as f32 * act.d[b];
                assert!((recon - v).abs() <= tol, "b={b} i={i} v={v} recon={recon}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_host_takes_the_neon_body() {
        assert_eq!(Q8_0Form::host(), Q8_0Form::Neon);
    }
}
