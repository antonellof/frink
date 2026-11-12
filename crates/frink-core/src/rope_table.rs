//! RoPE's `(cos, sin)` per rotation band at one position, as a table a
//! device kernel applies (`frink_vulkan`'s `rope.comp`).
//!
//! A GPU's `sin` and `cos` are not the CPU's. Vulkan guarantees an
//! absolute error of 2^-11 only inside [-pi, pi], and a decode position
//! times a band's frequency leaves that range for every position past a
//! few. So the device does not compute the angles. The host computes
//! them here with the same `rope_freqs` cache, the same `pos * freq /
//! factor` and the same `sin_cos` as [`crate::attention`]'s rotations,
//! and the device only multiplies. The test below holds the table to
//! those rotations bit for bit, because the table and the rotations are
//! two structures that must agree.

/// `[cos_0, sin_0, cos_1, sin_1, ...]` for the `dim / 2` bands of a
/// `dim`-wide rotation at `pos`, base `theta`, each band's angle divided
/// by its `freq_factors` entry when given (Llama 3's `rope_freqs`).
pub fn rope_cos_sin(pos: usize, theta: f32, dim: usize, freq_factors: Option<&[f32]>) -> Vec<f32> {
    let half = dim / 2;
    if let Some(ff) = freq_factors {
        assert_eq!(ff.len(), half, "one freq factor per rotation band");
    }
    let freqs = crate::attention::rope_freqs(theta, dim);
    (0..half)
        .flat_map(|i| {
            let angle = match freq_factors {
                Some(ff) => pos as f32 * freqs[i] / ff[i],
                None => pos as f32 * freqs[i],
            };
            let (sin, cos) = angle.sin_cos();
            [cos, sin]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{
        apply_rope, apply_rope_interleaved, apply_rope_interleaved_with_freq_factors,
        apply_rope_with_freq_factors,
    };

    fn rotate(v: &[f32], tab: &[f32], neox: bool) -> Vec<f32> {
        let mut out = v.to_vec();
        let half = v.len() / 2;
        for k in 0..half {
            let (c, s) = (tab[2 * k], tab[2 * k + 1]);
            let (a, b) = if neox {
                (k, k + half)
            } else {
                (2 * k, 2 * k + 1)
            };
            out[a] = v[a] * c - v[b] * s;
            out[b] = v[a] * s + v[b] * c;
        }
        out
    }

    /// The table, applied the way the device applies it, gives every CPU
    /// rotation's bits, for both pairings, with and without freq factors,
    /// at positions far outside [-pi, pi].
    #[test]
    fn the_table_reproduces_every_cpu_rotation_bit_for_bit() {
        let dim = 64;
        let v: Vec<f32> = (0..dim)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
            .collect();
        let ff: Vec<f32> = (0..dim / 2).map(|i| 1.0 + (i % 5) as f32 * 0.75).collect();
        for pos in [0usize, 1, 17, 4093, 131_071] {
            for theta in [10_000.0f32, 500_000.0] {
                let plain = rope_cos_sin(pos, theta, dim, None);
                let scaled = rope_cos_sin(pos, theta, dim, Some(&ff));
                let mut want = v.clone();
                apply_rope_interleaved(&mut want, pos, theta);
                assert_eq!(rotate(&v, &plain, false), want, "norm {pos}");
                let mut want = v.clone();
                apply_rope(&mut want, pos, theta);
                assert_eq!(rotate(&v, &plain, true), want, "neox {pos}");
                let mut want = v.clone();
                apply_rope_interleaved_with_freq_factors(&mut want, pos, theta, &ff);
                assert_eq!(rotate(&v, &scaled, false), want, "norm ff {pos}");
                let mut want = v.clone();
                apply_rope_with_freq_factors(&mut want, pos, theta, &ff);
                assert_eq!(rotate(&v, &scaled, true), want, "neox ff {pos}");
            }
        }
    }
}
