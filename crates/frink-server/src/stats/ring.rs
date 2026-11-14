//! A bounded ring of finished requests, addressed by an all-time
//! sequence number.
//!
//! A ring that renumbers on eviction leaves a poller two bad options:
//! re-read every row each time, or silently skip whatever fell out
//! between polls. Rows here keep a monotonic sequence number for the
//! life of the process, so [`RequestRing::since`] can say what is new
//! *and* how much was missed -- a client that came back after a burst
//! learns it lost rows instead of quietly under-reporting.
//!
//! The `limit` rule matters as much as the cursor: a truncated page
//! reports the cursor of its last returned row plus one, and only a page
//! that returned everything it matched may report the all-time count.
//! Return the all-time count from a truncated page and the next poll
//! skips exactly the rows the limit cut off.

use std::collections::VecDeque;

/// A bounded ring of finished requests, addressed by an all-time
/// sequence number.
///
/// The sequence counts every row ever pushed, not every row retained,
/// which is what lets a poller detect its own gap.
#[derive(Debug, Clone)]
pub struct RequestRing<T> {
    rows: VecDeque<(u64, T)>,
    capacity: usize,
    /// The sequence the next pushed row will get, and the count of every
    /// row this process has ever recorded.
    next_seq: u64,
}

/// What one poll of the ring returned, and what it could not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingPage<'a, T> {
    pub rows: Vec<&'a T>,
    /// Pass this back as the next `since`.
    ///
    /// When the page was truncated by `limit`, this is one past the last
    /// row actually returned -- *not* the all-time count, which would
    /// skip everything the limit cut off. Only a page that returned
    /// every row it matched may report the all-time count.
    pub cursor: u64,
    /// Rows that existed and were evicted before this poll could see
    /// them. Non-zero means the caller is polling slower than the server
    /// is finishing requests: a fact worth surfacing, not one worth
    /// hiding by returning fewer rows.
    pub missed: u64,
}

impl<T> RequestRing<T> {
    /// `capacity` is clamped to at least 1: a zero-capacity ring counts
    /// rows it can never show, which is a stats endpoint that reports
    /// nothing while claiming to have seen everything.
    pub fn new(capacity: usize) -> Self {
        RequestRing {
            rows: VecDeque::new(),
            capacity: capacity.max(1),
            next_seq: 0,
        }
    }

    /// Records one row and returns the sequence it was given.
    pub fn push(&mut self, row: T) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        if self.rows.len() == self.capacity {
            self.rows.pop_front();
        }
        self.rows.push_back((seq, row));
        seq
    }

    /// Up to `limit` rows recorded at or after `since`.
    ///
    /// A caller starting fresh passes 0 and gets whatever is retained;
    /// `missed` then says how much history predates the ring, which is a
    /// legitimate answer rather than an error.
    pub fn since(&self, since: u64, limit: usize) -> RingPage<'_, T> {
        let oldest = self
            .rows
            .front()
            .map(|(seq, _)| *seq)
            .unwrap_or(self.next_seq);
        let missed = oldest.saturating_sub(since);
        let matched: Vec<&(u64, T)> = self.rows.iter().filter(|(seq, _)| *seq >= since).collect();
        let returned = &matched[..limit.min(matched.len())];
        let cursor = if returned.len() < matched.len() {
            // Truncated: resume at the row after the last one delivered.
            returned.last().map(|(seq, _)| seq + 1).unwrap_or(since)
        } else {
            self.next_seq
        };
        RingPage {
            rows: returned.iter().map(|(_, row)| row).collect(),
            cursor,
            missed,
        }
    }

    /// Every retained row, oldest first.
    pub fn rows(&self) -> impl Iterator<Item = &T> {
        self.rows.iter().map(|(_, row)| row)
    }

    /// How many rows this process has recorded in total, retained or
    /// not.
    pub fn recorded_total(&self) -> u64 {
        self.next_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_keeps_the_newest_rows_and_numbers_them_for_all_time() {
        let mut ring = RequestRing::new(3);
        for i in 0..5u32 {
            assert_eq!(ring.push(i), i as u64);
        }
        assert_eq!(ring.rows.len(), 3);
        assert_eq!(ring.rows().copied().collect::<Vec<_>>(), vec![2, 3, 4]);
        assert_eq!(ring.recorded_total(), 5);
    }

    /// The reason the cursor is all-time: a poller that keeps up reads
    /// each row exactly once and never re-reads.
    #[test]
    fn a_poller_that_keeps_up_reads_every_row_exactly_once() {
        let mut ring = RequestRing::new(10);
        let mut cursor = 0;
        let mut seen = Vec::new();
        for round in 0..3u32 {
            for i in 0..2 {
                ring.push(round * 2 + i);
            }
            let page = ring.since(cursor, 100);
            assert_eq!(page.missed, 0);
            seen.extend(page.rows.iter().copied().copied());
            cursor = page.cursor;
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }

    /// The rule that makes `limit` safe. A truncated page that reported
    /// the all-time count would skip exactly the rows the limit cut off,
    /// which is a pagination bug that only shows up under load.
    #[test]
    fn a_truncated_page_resumes_at_the_row_after_the_last_one_delivered() {
        let mut ring = RequestRing::new(10);
        for i in 0..6u32 {
            ring.push(i);
        }
        let page = ring.since(0, 2);
        assert_eq!(page.rows, [&0, &1]);
        assert_eq!(page.cursor, 2, "not 6");

        let page = ring.since(page.cursor, 2);
        assert_eq!(page.rows, [&2, &3]);
        assert_eq!(page.cursor, 4);

        // The last page returns everything it matched, so it may report
        // the all-time count.
        let page = ring.since(page.cursor, 100);
        assert_eq!(page.rows, [&4, &5]);
        assert_eq!(page.cursor, 6);
    }

    /// And a poller that falls behind has to be able to tell, or its own
    /// numbers quietly stop adding up.
    #[test]
    fn a_poller_that_falls_behind_is_told_how_much_it_lost() {
        let mut ring = RequestRing::new(3);
        for i in 0..10u32 {
            ring.push(i);
        }
        let page = ring.since(0, 100);
        assert_eq!(page.rows.len(), 3);
        assert_eq!(page.missed, 7);
        assert_eq!(page.cursor, 10);

        let page = ring.since(page.cursor, 100);
        assert!(page.rows.is_empty());
        assert_eq!(page.missed, 0);
        assert_eq!(page.cursor, 10);
    }
}
