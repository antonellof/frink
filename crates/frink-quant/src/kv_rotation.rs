//! The randomized Hadamard rotation the 4-bit KV wire uses, host side.
//!
//! A 4-bit KV cache is lossy in one specific way: a handful of channels
//! in every K vector carry a magnitude the rest do not, so the absmax
//! that sets the scale is set by those channels and the other twenty-six
//! of every thirty-two land in two or three codes. The answer
//! is to rotate the head vector first, which spreads that magnitude
//! across every channel, and to rotate the query by the same matrix at
//! read time so the dot product is unchanged:
//!
//! ```text
//!   (H S q) . (H S k) = q S H^T H S k = q . k
//! ```
//!
//! because a normalized Walsh-Hadamard matrix is orthogonal
//! (`H^T H = I`) and the sign diagonal is its own inverse (`S S = I`).
//! Note what that argument does NOT say: `H S` is not symmetric, so
//! applying [`rotate_head_inplace`] twice does not return the original
//! vector. The inverse is [`unrotate_head_inplace`], the same two steps
//! in the other order, and it exists because one site really does read
//! a rotated store back as plain K: `MetalKvBuffers::tokens_host`,
//! which fills the HOST cache from the device one so the host attention
//! can continue a sequence the GPU started. A rotated row handed to a
//! host kernel that does not rotate its query is not a small error.
//!
//! This module is the definition the Metal `kv_append_q4` kernel is
//! checked against, the way `weight_matrix::hadamard` is the definition
//! for `frink-metal`'s folded rotation. The two must agree bit for bit
//! about the sign pattern, so the hash lives here and the kernel
//! carries a copy of the same three constants with a comment pointing
//! at this function.
//!
//! What is NOT rotated is V. Measured on K-shaped draws with six
//! channels at 25x (128 dims, 4096 rows), the rotation is worth 39% of
//! the error on K and 12% on V, and V would additionally need the
//! attention output rotated back per head, so K alone carries it.

/// Per-channel sign of the randomized Hadamard rotation.
///
/// Any deterministic pattern that decorrelates the heads will do, and
/// what matters is only that the append kernel and the query rotation
/// compute the SAME one; nothing outside this engine ever reads a
/// rotated block, so the choice is free. This is a 32-bit integer hash
/// (xorshift-multiply, the finalizer shape) over the head and channel
/// mixed together, taking the low bit.
///
/// Two heads must not share a pattern: a shared one would leave a K
/// vector that happens to align with a Hadamard basis row concentrated
/// in every head at once, which is the case the randomization exists to
/// rule out. `heads_do_not_share_a_sign_pattern` holds that.
#[inline]
pub fn rotation_sign(head_idx: usize, channel: usize) -> f32 {
    let mut h = (head_idx as u32)
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add((channel as u32).wrapping_mul(0x85EB_CA6B));
    h ^= h >> 15;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 13;
    if h & 1 == 1 {
        -1.0
    } else {
        1.0
    }
}

/// Head widths this rotation can serve.
///
/// A power of two because the transform is a butterfly, and at least 2
/// because a one-element Hadamard is the identity and would silently
/// buy nothing. A head the rotation cannot serve is a model that keeps
/// the unrotated wire, never a head that is rotated by the wrong
/// matrix.
#[inline]
pub fn rotation_viable(head_dim: usize) -> bool {
    head_dim >= 2 && head_dim.is_power_of_two()
}

/// In-place Fast Walsh-Hadamard Transform. `x.len()` must be a power of
/// two. Output is unnormalized (each butterfly is `a+b`, `a-b`);
/// callers that need the orthonormal transform divide by `sqrt(n)`.
pub fn fwht_inplace(x: &mut [f32]) {
    let n = x.len();
    assert!(
        n.is_power_of_two() && n > 0,
        "fwht length must be 2^k, got {n}"
    );
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        for i in (0..n).step_by(step) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h = step;
    }
}

/// Orthonormal FWHT: [`fwht_inplace`] then scale by `1/sqrt(n)`.
pub fn fwht_orthonormal_inplace(x: &mut [f32]) {
    let n = x.len();
    fwht_inplace(x);
    let inv = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Rotate one head vector: sign flip, then orthonormal FWHT.
///
/// Panics on a width [`rotation_viable`] rejects rather than rotating
/// by a matrix the reader cannot reproduce.
pub fn rotate_head_inplace(v: &mut [f32], head_idx: usize) {
    assert!(
        rotation_viable(v.len()),
        "kv_rotation rotation needs a power-of-two head width, got {}",
        v.len()
    );
    for (c, x) in v.iter_mut().enumerate() {
        *x *= rotation_sign(head_idx, c);
    }
    fwht_orthonormal_inplace(v);
}

/// Rotate every head of one token's row, laid out head-major as the KV
/// store and the Q projection both are.
pub fn rotate_row_inplace(row: &mut [f32], head_dim: usize) {
    assert!(
        head_dim > 0 && row.len().is_multiple_of(head_dim),
        "row of {} is not whole heads of {head_dim}",
        row.len()
    );
    for (h, head) in row.chunks_exact_mut(head_dim).enumerate() {
        rotate_head_inplace(head, h);
    }
}

/// Inverse of [`rotate_head_inplace`]: `S H` undoes `H S`.
pub fn unrotate_head_inplace(v: &mut [f32], head_idx: usize) {
    assert!(
        rotation_viable(v.len()),
        "kv_rotation rotation needs a power-of-two head width, got {}",
        v.len()
    );
    fwht_orthonormal_inplace(v);
    for (c, x) in v.iter_mut().enumerate() {
        *x *= rotation_sign(head_idx, c);
    }
}

/// [`unrotate_head_inplace`] over one token's row, head-major.
pub fn unrotate_row_inplace(row: &mut [f32], head_dim: usize) {
    assert!(
        head_dim > 0 && row.len().is_multiple_of(head_dim),
        "row of {} is not whole heads of {head_dim}",
        row.len()
    );
    for (h, head) in row.chunks_exact_mut(head_dim).enumerate() {
        unrotate_head_inplace(head, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(n: usize, d: usize, seed: u64) -> Vec<f32> {
        // xorshift, so the test carries its own numbers rather than a
        // dev-dependency on an RNG crate.
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        let mut v = vec![0.0f32; n * d];
        for x in v.iter_mut() {
            *x = next();
        }
        // Six outlier channels at 25x, the shape a real K row has.
        for row in v.chunks_exact_mut(d) {
            for c in [3usize, 11, 29, 64, 97, 120] {
                if c < d {
                    row[c] *= 25.0;
                }
            }
        }
        v
    }

    fn quant4_groups(x: &[f32], group: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for chunk in x.chunks_exact(group) {
            let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d = if amax > 0.0 { amax / 7.0 } else { 0.0 };
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            for &v in chunk {
                out.push((v * id).round().clamp(-8.0, 7.0) * d);
            }
        }
        out
    }

    #[test]
    fn fwht_involutory_up_to_scale() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.25, -0.5];
        let orig = x.clone();
        fwht_inplace(&mut x);
        fwht_inplace(&mut x);
        let n = orig.len() as f32;
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a * n - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn rotation_actually_moves_the_vector() {
        // The dot-product test below is only evidence if the rotation
        // is not quietly the identity for some head.
        for head in [0usize, 1, 7, 31] {
            let orig: Vec<f32> = (0..128)
                .map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.3)
                .collect();
            let mut x = orig.clone();
            rotate_head_inplace(&mut x, head);
            let moved = x
                .iter()
                .zip(orig.iter())
                .filter(|(a, b)| (*a - *b).abs() > 1e-3)
                .count();
            assert!(
                moved > 64,
                "head {head}: only {moved} of 128 channels moved"
            );
        }
    }

    #[test]
    fn unrotate_undoes_rotate() {
        for head in [0usize, 1, 7, 31] {
            let orig: Vec<f32> = (0..128)
                .map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.3)
                .collect();
            let mut x = orig.clone();
            rotate_head_inplace(&mut x, head);
            unrotate_head_inplace(&mut x, head);
            for (a, b) in orig.iter().zip(x.iter()) {
                assert!((a - b).abs() < 1e-4, "head {head}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rotate_then_rotate_is_not_the_identity() {
        // The reason `unrotate_head_inplace` exists: `H S` is not
        // symmetric, so the obvious shortcut is wrong.
        let orig: Vec<f32> = (0..64).map(|i| (i as f32 * 0.11).sin()).collect();
        let mut x = orig.clone();
        rotate_head_inplace(&mut x, 3);
        rotate_head_inplace(&mut x, 3);
        let max = orig
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max > 1e-2, "rotating twice came back to the original");
    }

    #[test]
    fn heads_do_not_share_a_sign_pattern() {
        let a: Vec<f32> = (0..128).map(|c| rotation_sign(0, c)).collect();
        let b: Vec<f32> = (0..128).map(|c| rotation_sign(1, c)).collect();
        assert_ne!(a, b);
    }

    #[test]
    fn rotation_preserves_the_dot_product() {
        let d = 128usize;
        let q = draw(1, d, 0x9E37);
        let k = draw(1, d, 0x1234);
        let exact: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
        let (mut qr, mut kr) = (q.clone(), k.clone());
        rotate_head_inplace(&mut qr, 5);
        rotate_head_inplace(&mut kr, 5);
        let rotated: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (exact - rotated).abs() <= 1e-3 * exact.abs().max(1.0),
            "{exact} vs {rotated}"
        );
    }

    /// The measurement the row exists for: on outlier-heavy K the
    /// rotation cuts the 4-bit error by about a third, with the scale
    /// granularity held fixed at the 32-element group frink's wire
    /// already uses.
    #[test]
    fn rotation_cuts_the_four_bit_error() {
        let (n, d) = (256usize, 128usize);
        let q = draw(n, d, 0xBEEF);
        let k = draw(n, d, 0xCAFE);

        let mut plain_err = 0.0f64;
        let mut rot_err = 0.0f64;
        let mut mag = 0.0f64;
        for i in 0..n {
            let (qi, ki) = (&q[i * d..(i + 1) * d], &k[i * d..(i + 1) * d]);
            let exact: f32 = qi.iter().zip(ki.iter()).map(|(a, b)| a * b).sum();
            mag += exact.abs() as f64;

            let kq = quant4_groups(ki, 32);
            let plain: f32 = qi.iter().zip(kq.iter()).map(|(a, b)| a * b).sum();
            plain_err += (exact - plain).abs() as f64;

            let (mut qr, mut kr) = (qi.to_vec(), ki.to_vec());
            rotate_head_inplace(&mut qr, 0);
            rotate_head_inplace(&mut kr, 0);
            let krq = quant4_groups(&kr, 32);
            let rot: f32 = qr.iter().zip(krq.iter()).map(|(a, b)| a * b).sum();
            rot_err += (exact - rot).abs() as f64;
        }
        let (plain_rel, rot_rel) = (plain_err / mag, rot_err / mag);
        assert!(
            rot_rel < plain_rel * 0.8,
            "rotation bought too little: plain {plain_rel:.5} vs rotated {rot_rel:.5}"
        );
    }
}
