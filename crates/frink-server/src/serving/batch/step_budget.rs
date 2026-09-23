//! How many prompt tokens one tick may prefill, given how many rows
//! are decoding in the same tick.
//!
//! # The defect this closes
//!
//! A tick runs one bounded prefill chunk and then one decode step over
//! every in-flight row. The chunk was a CONSTANT (`prefill_chunk`,
//! 128 by default) and the decode step is `rows.len()` tokens wide, so
//! the work in a tick was `128 + rows`, growing with the batch, and
//! nothing bounded the pair.
//!
//! That is a real cost and it lands on the wrong request. Every
//! decoding row waits for the whole prefill chunk before its next
//! token, so admitting one prompt raises inter-token latency for
//! everybody already answering -- and the more rows are decoding, the
//! more waiting the same chunk causes. The knob that fixes it does not
//! exist if prefill and decode are budgeted separately, because the
//! quantity to bound is their SUM.
//!
//! # What the high-throughput serving engines do
//!
//! One budget per step, spent by prompt tokens and generated tokens
//! alike, with no phase distinction: a scheduler walks its requests
//! and decrements a single `token_budget` whether the tokens it is
//! adding are a prompt chunk or one more decode step. The batch is
//! whatever fits.
//!
//! [`prefill_budget`] is that idea at this worker's shape. The decode
//! width is known before the prefill chunk runs -- it is the number of
//! rows -- so the chunk takes what is left of the step budget rather
//! than a fixed number.
//!
//! # Why it cannot starve a prompt
//!
//! A budget alone deadlocks: once enough rows are decoding, the
//! remainder is zero, no prompt advances, and the rows that would free
//! the budget are waiting on tokens from a prompt that never runs.
//! [`MIN_PREFILL_CHUNK`] is the floor that makes progress
//! unconditional, and `a_saturated_batch_still_advances_its_prompts`
//! is that property rather than an argument for it.
//!
//! The floor means the budget is a TARGET, not a cap: a very wide
//! batch still costs `rows + MIN_PREFILL_CHUNK`. Bounding it strictly
//! would mean preempting a decoding row, which this engine does not
//! do (`docs/plans/serving-parity-audit.md`).

/// Default tokens per tick, prompt and generated together.
///
/// Sized to the old behaviour at a typical batch so this is a
/// re-shaping rather than a slowdown: the previous tick ran
/// `prefill_chunk` (128) plus the decode width, so 256 leaves a full
/// 128-token chunk until 128 rows are decoding, which is past
/// `max_seqs` on every default. What changes is the shape under load,
/// not the throughput of a quiet server.
pub const DEFAULT_MAX_BATCH_TOKENS: usize = 256;

/// The smallest prefill chunk a tick may run, however wide the batch.
///
/// Without it a saturated batch spends the whole budget on decode and
/// no prompt ever advances -- and the prompts are what the decoding
/// rows will eventually become, so the queue deadlocks rather than
/// merely slowing.
pub const MIN_PREFILL_CHUNK: usize = 16;

/// Prompt tokens this tick may run.
///
/// `decode_rows` is the number of rows that will take a decode step in
/// the same tick, one token each. `configured_chunk` is the ceiling
/// (`prefill_chunk`), so this can only ever make a chunk SMALLER than
/// the setting -- a server tuned by that knob keeps its meaning.
pub fn prefill_budget(
    max_batch_tokens: usize,
    decode_rows: usize,
    configured_chunk: usize,
) -> usize {
    max_batch_tokens
        .saturating_sub(decode_rows)
        .clamp(MIN_PREFILL_CHUNK, configured_chunk.max(MIN_PREFILL_CHUNK))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: usize = 128;

    /// An idle server prefills at the configured chunk: the budget
    /// only ever takes work away, so a quiet server behaves exactly as
    /// it did.
    #[test]
    fn an_empty_batch_prefills_at_the_configured_chunk() {
        assert_eq!(
            prefill_budget(DEFAULT_MAX_BATCH_TOKENS, 0, CHUNK),
            CHUNK,
            "with nothing decoding the chunk must not shrink"
        );
    }

    /// **The point of the module.** Prompt tokens give way as the
    /// batch widens, so a tick's total work stays near the budget
    /// instead of growing with the number of answers in flight.
    #[test]
    fn a_wider_batch_prefills_less() {
        let narrow = prefill_budget(DEFAULT_MAX_BATCH_TOKENS, 8, CHUNK);
        let wide = prefill_budget(DEFAULT_MAX_BATCH_TOKENS, 160, CHUNK);
        assert!(
            wide < narrow,
            "a batch of 160 rows prefilled {wide} where a batch of 8 \
             prefilled {narrow}; the budget is not shared"
        );

        // The first draft asserted the total was FLAT and it is not:
        // at a narrow batch the configured chunk binds, not the
        // budget, so a tick costs `rows + CHUNK` there and
        // `max_batch_tokens` once the budget starts binding. The test
        // failed and the claim was wrong, which is the same lesson
        // `cache_aware`'s "reordering alone saves nothing" recorded.
        //
        // The true property is a CEILING, and it is the one the old
        // constant did not have.
        let total = |rows: usize| rows + prefill_budget(DEFAULT_MAX_BATCH_TOKENS, rows, CHUNK);
        for rows in [0usize, 1, 8, 64, 128, 200, 1000] {
            let old = rows + CHUNK; // what a fixed chunk cost
            assert!(
                total(rows) <= old,
                "at {rows} rows the budget costs {} against the old {old}; \
                 it may only ever take work away",
                total(rows)
            );
            if rows + MIN_PREFILL_CHUNK <= DEFAULT_MAX_BATCH_TOKENS {
                assert!(
                    total(rows) <= DEFAULT_MAX_BATCH_TOKENS,
                    "at {rows} rows a tick costs {}, over the {DEFAULT_MAX_BATCH_TOKENS} budget",
                    total(rows)
                );
            } else {
                // Past that point the floor is what is left, and the
                // module says so rather than pretending to a bound it
                // cannot hold without preempting a decoding row.
                assert_eq!(
                    total(rows),
                    rows + MIN_PREFILL_CHUNK,
                    "a batch wider than the budget must degrade to the floor"
                );
            }
        }
    }

    /// **It cannot deadlock.** However wide the batch, some prompt
    /// tokens run, so the prompts that become the next rows always
    /// advance.
    ///
    /// Driven to completion rather than asserted on the constant: a
    /// floor that existed but was not reached would pass a test that
    /// only read it.
    #[test]
    fn a_saturated_batch_still_advances_its_prompts() {
        let prompt = 600usize;
        let mut done = 0usize;
        let mut ticks = 0usize;
        // Far more rows than the whole budget, forever.
        while done < prompt {
            let step = prefill_budget(DEFAULT_MAX_BATCH_TOKENS, 100_000, CHUNK);
            assert!(step > 0, "a saturated batch prefilled nothing");
            done += step;
            ticks += 1;
            assert!(ticks < 1000, "prompt is not advancing: {done}/{prompt}");
        }
        assert_eq!(
            ticks,
            prompt.div_ceil(MIN_PREFILL_CHUNK),
            "a saturated batch runs exactly the floor each tick"
        );
    }

    /// The configured chunk is a ceiling, so setting it low keeps
    /// meaning it. A budget that raised it would silently override the
    /// one knob an operator already had.
    #[test]
    fn the_configured_chunk_is_never_exceeded() {
        for rows in [0usize, 1, 7, 64] {
            assert!(
                prefill_budget(DEFAULT_MAX_BATCH_TOKENS, rows, 32) <= 32,
                "a 32-token chunk was exceeded at {rows} rows"
            );
        }
    }

    /// A chunk configured below the floor still runs the floor,
    /// because the floor is what makes progress unconditional. Stated
    /// so the interaction is a decision rather than an accident of
    /// clamp ordering -- `clamp` panics when min exceeds max, and this
    /// is the case that would.
    #[test]
    fn a_chunk_smaller_than_the_floor_does_not_panic() {
        assert_eq!(
            prefill_budget(DEFAULT_MAX_BATCH_TOKENS, 0, 4),
            MIN_PREFILL_CHUNK
        );
    }
}
