//! A cheap identity sample of a host weight buffer, for the GPU
//! backends' residency caches (`frink-cuda`, `frink-vulkan`).
//!
//! Both cache device copies by host `(address, length)`, and a host
//! buffer can be freed and its address handed to a different buffer of
//! the same length, which `(address, length)` cannot tell apart: the
//! tensor-core `mul_mm` hardware test multiplied one fixture's weights by
//! another fixture's activations for exactly that reason (2026-09-17).
//! Production weights are one mmap for the life of the process, which is
//! why nothing had noticed. One definition, here, so the two caches make
//! the same check (ADR 0004).

/// The first and last 64 bytes of a buffer and 64 from its middle. Not
/// a hash of the whole tensor (that is gigabytes per token); enough that
/// a different buffer at the same address is caught unless it agrees on
/// 192 sampled bytes.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct WeightFingerprint {
    head: [u8; 64],
    mid: [u8; 64],
    tail: [u8; 64],
}

impl WeightFingerprint {
    pub fn of(bytes: &[u8]) -> Self {
        fn take(bytes: &[u8], at: usize) -> [u8; 64] {
            let mut out = [0u8; 64];
            let at = at.min(bytes.len());
            let n = (bytes.len() - at).min(64);
            out[..n].copy_from_slice(&bytes[at..at + n]);
            out
        }
        Self {
            head: take(bytes, 0),
            mid: take(bytes, bytes.len() / 2),
            tail: take(bytes, bytes.len().saturating_sub(64)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_change_in_any_sampled_region_is_seen_and_short_buffers_work() {
        let a: Vec<u8> = (0..1000).map(|i| i as u8).collect();
        for at in [0usize, 500, 999] {
            let mut b = a.clone();
            b[at] ^= 1;
            assert_ne!(
                WeightFingerprint::of(&a),
                WeightFingerprint::of(&b),
                "byte {at}"
            );
        }
        assert_eq!(WeightFingerprint::of(&a), WeightFingerprint::of(&a.clone()));
        assert_eq!(
            WeightFingerprint::of(&[1, 2, 3]),
            WeightFingerprint::of(&[1, 2, 3])
        );
        assert_ne!(
            WeightFingerprint::of(&[1, 2, 3]),
            WeightFingerprint::of(&[1, 2, 4])
        );
    }
}
