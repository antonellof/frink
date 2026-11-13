//! The accept rate at each position inside a draft block, for
//! `usage.draft_accept_rate_per_position`.
//!
//! Verification by agreement accepts a PREFIX of the block: the first
//! draft the sampler disagrees with ends it. So a round that offered `d`
//! drafts and accepted `a` offered positions `0..d` and accepted exactly
//! positions `0..a`, and the two counts are all a round needs to say.

/// Offered and accepted counts per draft position, summed over rounds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PositionTally {
    offered: Vec<usize>,
    accepted: Vec<usize>,
}

impl PositionTally {
    /// One verified round: `drafted` offered, the first `accepted` kept.
    pub(crate) fn record(&mut self, drafted: usize, accepted: usize) {
        let accepted = accepted.min(drafted);
        if self.offered.len() < drafted {
            self.offered.resize(drafted, 0);
            self.accepted.resize(drafted, 0);
        }
        for slot in &mut self.offered[..drafted] {
            *slot += 1;
        }
        for slot in &mut self.accepted[..accepted] {
            *slot += 1;
        }
    }

    /// Adds another request's (or choice's) rounds to this one.
    pub(crate) fn merge(&mut self, other: &Self) {
        let n = self.offered.len().max(other.offered.len());
        self.offered.resize(n, 0);
        self.accepted.resize(n, 0);
        for (i, (o, a)) in other.offered.iter().zip(&other.accepted).enumerate() {
            self.offered[i] += o;
            self.accepted[i] += a;
        }
    }

    /// Accepted over offered at each position. Every position in the
    /// vector was offered at least once, so nothing divides by zero.
    pub(crate) fn rates(&self) -> Vec<f64> {
        self.offered
            .iter()
            .zip(&self.accepted)
            .map(|(&o, &a)| a as f64 / o as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_count_each_position_over_the_rounds_that_offered_it() {
        let mut t = PositionTally::default();
        t.record(3, 3); // every draft kept
        t.record(3, 1); // the second disagreed
        t.record(2, 0); // the first disagreed
        assert_eq!(t.rates(), vec![2.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0]);
        let mut other = PositionTally::default();
        other.record(4, 4);
        t.merge(&other);
        assert_eq!(t.rates(), vec![3.0 / 4.0, 2.0 / 4.0, 2.0 / 3.0, 1.0]);
        assert!(PositionTally::default().rates().is_empty());
    }
}
