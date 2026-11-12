//! The DRY repetition sampler ("Don't Repeat Yourself"), ported from
//! llama.cpp's `llama_sampler_dry` (`src/llama-sampler.cpp:3078-3400`),
//! which is itself a port of Koboldcpp PR 982 by pi6am.
//!
//! # What it does that the repetition penalty cannot
//!
//! `penalties` looks at SINGLE tokens: a token that occurred in the
//! window is made less likely, once. DRY looks at SEQUENCES. It asks,
//! for every token the model could emit next, "how long a repetition of
//! earlier context would emitting this token continue?", and penalises
//! by `multiplier * base ^ (that length - allowed_length)`. A model
//! looping on a four-token phrase is not emitting one over-represented
//! token, so the repetition penalty barely moves it while DRY's
//! exponential grows every time round the loop.
//!
//! # The three parts, and where each one comes from
//!
//! * [`DryBreakers`] is `get_overlapping_token_sequences`
//!   (`src/llama-sampler.cpp:3095`): the sequence breakers a caller
//!   gives as STRINGS, tokenised against the loaded vocabulary. A
//!   breaker is a point the repetition detector refuses to look past,
//!   so `"\n"` stops one paragraph's phrasing from penalising the next.
//!   It has to be done against the model's own vocabulary because a
//!   breaker string is usually not a whole token: `":"` may only ever
//!   appear as the tail of `"foo:"`, so the entry is keyed on the token
//!   that CONTAINS it and carries whatever tokens must follow.
//! * [`DryParams::penalties`] is `llama_sampler_dry_apply`
//!   (`src/llama-sampler.cpp:3151`), including the reverse Z-algorithm
//!   that finds, in one linear pass, how long a suffix ending at each
//!   position also occurs elsewhere in the window.
//! * [`crate::sampler_chain::Candidates::dry`] subtracts the result.
//!
//! # Why the parameters are a struct with private fields
//!
//! Because enabling DRY without tokenising its breakers is a silent
//! wrong answer, not an error: the sampler runs, penalises across
//! newlines it was told to stop at, and nothing says so. So there is no
//! way to build an ENABLED [`DryParams`] without handing it a
//! [`DryBreakers`], and the only way to build a non-empty
//! [`DryBreakers`] is [`DryBreakers::from_vocab`], which needs a
//! vocabulary. A caller that genuinely wants none writes
//! [`DryBreakers::none`] and that is visible in the diff.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::penalty_window::PenaltyWindow;

/// llama.cpp's `MAX_CHAR_LEN` (`src/llama-sampler.cpp:3382`): a
/// sequence breaker longer than this is truncated rather than refused.
const MAX_CHAR_LEN: usize = 40;

/// llama.cpp's `MAX_SEQ_LEN` (`src/llama-sampler.cpp:3383`): the tail
/// of a breaker is clamped to this many tokens, which is what keeps the
/// restart-sequence scan in [`DryParams::penalties`] linear in the
/// window length rather than quadratic.
const MAX_SEQ_LEN: usize = 20;

/// llama.cpp's `FLOAT_MAX_LOG` (`src/llama-sampler.cpp:3312`), the
/// approximate natural log of `FLT_MAX`. The exponent is clamped to
/// `FLOAT_MAX_LOG / ln(base)` so `base ^ exponent` cannot overflow to
/// infinity on a long repetition.
const FLOAT_MAX_LOG: f32 = 88.722_84;

/// llama.cpp's default sequence breakers
/// (`common/common.h:259`, `dry_sequence_breakers = {"\n", ":", "\"", "*"}`).
///
/// One definition, read by the CLI flag's default and by the server's,
/// because two lists that must agree about four strings is this repo's
/// dominant defect shape at its smallest.
pub const DEFAULT_SEQUENCE_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];

/// What DRY needs from a loaded model's vocabulary, and nothing else.
///
/// Three methods rather than a dependency on any particular tokenizer
/// type: `frink-models` has four of those and `frink-server` wraps
/// them in a fifth, so a concrete type here would either pick one or
/// force an enum that grows with every new vocabulary format.
pub trait DryVocab {
    /// The number of token ids in the vocabulary. Every id in
    /// `0..n_tokens()` must be valid for [`Self::detokenize`].
    fn n_tokens(&self) -> usize;

    /// The text of ONE token, special tokens rendered rather than
    /// hidden -- llama.cpp's `vocab.detokenize({token_id}, true)`
    /// (`src/llama-sampler.cpp:3097`).
    fn detokenize(&self, token: usize) -> String;

    /// `text` as token ids, with no BOS and no special-token parsing --
    /// llama.cpp's `vocab.tokenize(str.substr(i), false, false)`
    /// (`src/llama-sampler.cpp:3114`).
    fn tokenize(&self, text: &str) -> Vec<usize>;
}

/// Sequence breakers, tokenised against one model's vocabulary.
///
/// Keyed on the token that HOLDS the start of the breaker string, with
/// the tokens that must follow it as the value. An empty tail means the
/// token by itself is a whole breaker, which is both the common case and
/// the one that suppresses a penalty entirely (see
/// [`DryParams::penalties`] step 4).
///
/// A token may head more than one breaker -- llama.cpp keeps a
/// `std::unordered_multimap` for exactly that -- so the value is a list
/// of tails, deduplicated as upstream deduplicates it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DryBreakers {
    /// The strings this was built from, kept so a cache key can name
    /// the configuration without hashing the whole tokenised map.
    raw: Vec<String>,
    heads: HashMap<usize, Vec<Vec<usize>>>,
}

impl DryBreakers {
    /// No breakers: the repetition scan may look back across anything.
    ///
    /// This is what `--dry-sequence-breaker none` asks for upstream, and
    /// what a caller with no vocabulary to tokenise against must write
    /// explicitly rather than get by accident.
    pub fn none() -> Self {
        DryBreakers::default()
    }

    /// The strings these breakers were built from, in the order given.
    pub fn raw(&self) -> &[String] {
        &self.raw
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Tokenise `breakers` against `vocab`.
    ///
    /// This is `get_overlapping_token_sequences`
    /// (`src/llama-sampler.cpp:3095-3134`), and the "overlapping" is the
    /// whole point: a breaker is rarely a token of its own. The scan
    /// walks every token in the vocabulary and asks either
    ///
    /// * does this token's text CONTAIN the breaker? Then the token
    ///   alone breaks the sequence (an empty tail); or
    /// * does this token's text END with a prefix of the breaker? Then
    ///   the token breaks the sequence only when the REST of the breaker
    ///   follows, and that rest is tokenised and kept as the tail.
    ///
    /// **Cost.** One `detokenize` per vocabulary entry per breaker
    /// string, which is upstream's cost too (`llama_sampler_init_dry` at
    /// `:3399` runs it per sampler, and llama.cpp's server builds a
    /// sampler per request). It is paid only when DRY is switched on --
    /// [`DryParams::new`] is the only caller and it is only reached for
    /// a non-zero multiplier.
    ///
    /// An empty breaker is skipped rather than refused, as upstream
    /// skips it (`:3388`); one longer than [`MAX_CHAR_LEN`] bytes is
    /// truncated.
    pub fn from_vocab(vocab: &dyn DryVocab, breakers: &[String]) -> Self {
        let mut out = DryBreakers {
            raw: breakers.to_vec(),
            heads: HashMap::new(),
        };
        for breaker in breakers {
            if breaker.is_empty() {
                continue;
            }
            let mut bytes = breaker.as_bytes();
            if bytes.len() > MAX_CHAR_LEN {
                bytes = &bytes[..MAX_CHAR_LEN];
            }
            out.add_overlapping(vocab, bytes);
        }
        out
    }

    /// Breakers given directly as token sequences.
    ///
    /// This is `llama_sampler_init_dry_testing`
    /// (`src/llama-sampler.cpp:3420`), the entry point llama.cpp's own
    /// `tests/test-sampling.cpp` uses, and it exists here for the same
    /// reason: the golden values in [`mod tests`](self::tests) were
    /// produced by running that function against `libllama`, and a test
    /// that had to build a vocabulary first would be testing the
    /// tokenizer as well as the sampler.
    ///
    /// The first token of each sequence is the head, the rest the tail,
    /// exactly as upstream splits it.
    pub fn from_token_sequences(sequences: &[Vec<usize>]) -> Self {
        let mut heads: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
        for sequence in sequences {
            let Some((&head, tail)) = sequence.split_first() else {
                continue;
            };
            let mut tail = tail.to_vec();
            tail.truncate(MAX_SEQ_LEN);
            let entry = heads.entry(head).or_default();
            if !entry.contains(&tail) {
                entry.push(tail);
            }
        }
        DryBreakers {
            raw: Vec::new(),
            heads,
        }
    }

    fn tails(&self, token: usize) -> Option<&[Vec<usize>]> {
        self.heads.get(&token).map(|v| v.as_slice())
    }

    /// True when this token is a breaker all by itself, which is what
    /// makes upstream skip its penalty entirely (`:3325-3334`).
    fn is_single_token_breaker(&self, token: usize) -> bool {
        self.heads
            .get(&token)
            .is_some_and(|tails| tails.iter().any(|tail| tail.is_empty()))
    }

    fn push(&mut self, head: usize, tail: Vec<usize>) {
        let entry = self.heads.entry(head).or_default();
        // Upstream's `equal_range` duplicate check (`:3117-3126`).
        if !entry.contains(&tail) {
            entry.push(tail);
        }
    }

    /// One breaker string against the whole vocabulary.
    ///
    /// Byte-wise rather than char-wise because upstream is:
    /// `word.find(str[0], pos + 1)` indexes `std::string` by byte, and a
    /// token's text is not guaranteed to split on a character boundary
    /// where the breaker does. The tail is only tokenised when the
    /// remaining bytes are valid UTF-8, which they are whenever the
    /// breaker itself is.
    fn add_overlapping(&mut self, vocab: &dyn DryVocab, breaker: &[u8]) {
        for token in 0..vocab.n_tokens() {
            let word = vocab.detokenize(token);
            let word = word.as_bytes();
            if contains_subslice(word, breaker) {
                self.push(token, Vec::new());
                continue;
            }
            let mut from = 0usize;
            while from < word.len() {
                let Some(offset) = word[from..].iter().position(|&c| c == breaker[0]) else {
                    break;
                };
                let pos = from + offset;
                from = pos + 1;
                // How many bytes of the breaker this occurrence
                // matched before running off the end of the token.
                let mut i = 1usize;
                let mut matched = true;
                while i < breaker.len() && i + pos < word.len() {
                    if word[pos + i] != breaker[i] {
                        matched = false;
                        break;
                    }
                    i += 1;
                }
                if !matched {
                    continue;
                }
                let Ok(rest) = std::str::from_utf8(&breaker[i..]) else {
                    // A breaker whose tail starts mid-character cannot
                    // be tokenised; upstream would hand the tokenizer
                    // the same broken bytes and get nothing useful.
                    continue;
                };
                let mut tail = vocab.tokenize(rest);
                tail.truncate(MAX_SEQ_LEN);
                self.push(token, tail);
            }
        }
    }
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return needle.is_empty();
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// One request's DRY configuration.
///
/// Fields are private and there is no `Default` that can be enabled:
/// [`DryParams::off`] is the only way to get a disabled one and
/// [`DryParams::new`] the only way to get an enabled one, and it takes
/// the breakers. See the module docs for why that is a type invariant
/// rather than a convention.
#[derive(Debug, Clone)]
pub struct DryParams {
    /// llama.cpp's `dry_multiplier` (`common/common.h:242`), `0.0` off.
    multiplier: f32,
    /// llama.cpp's `dry_base` (`common/common.h:243`), default 1.75.
    /// Below 1.0 disables DRY, exactly as upstream's guard reads
    /// (`src/llama-sampler.cpp:3154`).
    base: f32,
    /// llama.cpp's `dry_allowed_length` (`common/common.h:244`),
    /// default 2: a repetition this long or shorter is free.
    allowed_length: i32,
    /// llama.cpp's `dry_penalty_last_n` (`common/common.h:245`),
    /// default `-1` meaning "the whole context"; `0` disables.
    penalty_last_n: i32,
    /// llama.cpp's `n_ctx_train`, the ceiling every window is clamped
    /// to (`src/llama-sampler.cpp:3158-3159`).
    total_context_size: usize,
    /// Shared rather than cloned: this is a map over the vocabulary and
    /// [`crate::sampling::SamplingParams`] is cloned per request.
    breakers: Arc<DryBreakers>,
}

impl DryParams {
    /// DRY switched off, which is llama.cpp's default
    /// (`dry_multiplier = 0.0f`, `common/common.h:248`).
    pub fn off() -> Self {
        DryParams {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 64,
            total_context_size: 0,
            breakers: Arc::new(DryBreakers::none()),
        }
    }

    /// DRY with the given configuration and breakers.
    ///
    /// `total_context_size` is llama.cpp's `n_ctx_train`: the ceiling a
    /// `penalty_last_n` of `-1` resolves to, and the clamp every window
    /// is passed through.
    pub fn new(
        multiplier: f32,
        base: f32,
        allowed_length: i32,
        penalty_last_n: i32,
        total_context_size: usize,
        breakers: DryBreakers,
    ) -> Self {
        DryParams {
            multiplier,
            base,
            allowed_length,
            penalty_last_n,
            total_context_size,
            breakers: Arc::new(breakers),
        }
    }

    /// The single predicate for "DRY does nothing", transcribed from the
    /// guard both `llama_sampler_dry_accept` (`:3143`) and
    /// `llama_sampler_dry_apply` (`:3154`) open with, and the same three
    /// conditions `llama_sampler_init_dry` reads to return an empty
    /// sampler (`:3387`).
    ///
    /// One function rather than the three copies upstream has, because
    /// this is read by the chain, by the CLI banner and by the server's
    /// refusal, and a disagreement between them is a sampler that is
    /// reported on but not run.
    pub fn is_enabled(&self) -> bool {
        self.multiplier != 0.0 && self.base >= 1.0 && self.penalty_last_n != 0
    }

    pub fn multiplier(&self) -> f32 {
        self.multiplier
    }

    pub fn base(&self) -> f32 {
        self.base
    }

    pub fn allowed_length(&self) -> i32 {
        self.allowed_length
    }

    pub fn penalty_last_n(&self) -> i32 {
        self.penalty_last_n
    }

    pub fn total_context_size(&self) -> usize {
        self.total_context_size
    }

    pub fn breakers(&self) -> &DryBreakers {
        &self.breakers
    }

    /// How many of the most recent tokens the scan looks at:
    /// `dry_penalty_last_n` resolved against the context size, exactly
    /// as `src/llama-sampler.cpp:3158` resolves it.
    fn effective_last_n(&self) -> usize {
        if self.penalty_last_n == -1 {
            self.total_context_size
        } else {
            self.penalty_last_n.max(0) as usize
        }
    }

    /// The penalty to SUBTRACT from each token's logit, keyed by token
    /// id. Empty when DRY is off or has nothing to say.
    ///
    /// `llama_sampler_dry_apply` (`src/llama-sampler.cpp:3151-3345`),
    /// in its four documented steps:
    ///
    /// 1. walk backwards for a sequence breaker, and clamp how far a
    ///    repetition may reach (`rep_limit`);
    /// 2. the reverse Z-algorithm, which fills `repeat_count[i]` with
    ///    the length of the window's suffix that also occurs ending at
    ///    position `i`;
    /// 3. for each non-zero count, the token that WOULD extend that
    ///    repetition is the one to penalise, and the longest repetition
    ///    ending in it wins;
    /// 4. `multiplier * base ^ (length - allowed_length)`, skipped for a
    ///    token that is a whole sequence breaker by itself.
    ///
    /// `history` is the same [`PenaltyWindow`] the repetition penalties
    /// use, for the same reason: llama.cpp's DRY sampler is fed by
    /// `common_sampler_accept`, which both front ends call for prompt
    /// tokens as well as generated ones (see [`crate::penalty_window`]).
    pub fn penalties(&self, history: PenaltyWindow<'_>) -> HashMap<usize, f32> {
        let empty = HashMap::new();
        if !self.is_enabled() {
            return empty;
        }
        // `last_tokens` is a ring buffer of `effective_last_n` capacity
        // upstream, so its size is already clamped; here the window is
        // asked for that many and clamped again by the context size,
        // which is upstream's second `std::min` (`:3159`).
        let effective = self.effective_last_n();
        let recent: Vec<usize> = history
            .recent(effective.min(self.total_context_size))
            .collect();
        let n = recent.len();
        if n as i32 <= self.allowed_length {
            return empty;
        }
        // `rat(i)`: llama.cpp's `ring_buffer::rat`, i tokens back from
        // the most recent. `recent` is oldest-first.
        let rat = |i: usize| recent[n - 1 - i];

        // Step 1: how far back a repetition may reach before it hits a
        // sequence breaker.
        let mut rep_limit = n as i32;
        for i in 0..n {
            let Some(tails) = self.breakers.tails(rat(i)) else {
                continue;
            };
            let mut longest_match: i32 = -1;
            for tail in tails {
                let seq_len = tail.len() as i32;
                // `<= i`: the tail has to fit in the window behind the
                // head, and the head itself is already matched.
                if seq_len <= longest_match || seq_len > i as i32 {
                    continue;
                }
                let matched = (0..tail.len()).all(|offset| tail[offset] == rat(i - offset - 1));
                if matched {
                    longest_match = seq_len;
                }
            }
            if longest_match >= 0 {
                rep_limit = i as i32 - longest_match;
                break;
            }
        }
        if rep_limit < self.allowed_length {
            return empty;
        }

        // Step 2: the reverse Z-algorithm (`:3243-3283`).
        let mut repeat_count = vec![0i32; n];
        let last = n - 1;
        let mut lt = 0usize;
        let mut rt = 0usize;
        for k in 1..n {
            if k > rt {
                // Outside the current Z-box: compare naively.
                let mut matched = 0usize;
                while matched + k < n && rat(matched) == rat(matched + k) {
                    matched += 1;
                }
                repeat_count[last - k] = (matched as i32).min(rep_limit);
                if matched > 0 {
                    lt = k;
                    rt = k + matched - 1;
                }
            } else {
                let p = k - lt;
                let right_part_len = (rt - k + 1) as i32;
                if repeat_count[last - p] < right_part_len {
                    repeat_count[last - k] = repeat_count[last - p].min(rep_limit);
                } else {
                    let mut i = rt + 1;
                    while i < n && rat(i) == rat(i - k) {
                        i += 1;
                    }
                    repeat_count[last - k] = ((i - k) as i32).min(rep_limit);
                    lt = k;
                    rt = i - 1;
                }
            }
        }

        // Step 3: the token that would EXTEND each repetition
        // (`:3296-3308`). `repeat_count[i]` is about the window position
        // `i`, so the token that follows it is `recent[i + 1]`, which is
        // upstream's `rat(last_n_repeat - 2 - i)`.
        let mut max_token_repeat: HashMap<usize, i32> = HashMap::new();
        for i in 0..n - 1 {
            let repeat_len = repeat_count[i];
            if repeat_len < self.allowed_length {
                continue;
            }
            let token = recent[i + 1];
            let slot = max_token_repeat.entry(token).or_insert(repeat_len);
            if *slot < repeat_len {
                *slot = repeat_len;
            }
        }

        // Step 4: the exponential (`:3310-3343`).
        let mut max_exponent = 0i32;
        if self.base > 1.000_001 {
            max_exponent = (FLOAT_MAX_LOG / self.base.ln()) as i32;
        }
        let mut out = HashMap::with_capacity(max_token_repeat.len());
        for (&token, &max_repeat) in &max_token_repeat {
            // A token that is a whole sequence breaker is exempt: it is
            // the thing repetition is allowed to run into.
            if self.breakers.is_single_token_breaker(token) {
                continue;
            }
            let mut repeat_exp = max_repeat - self.allowed_length;
            if max_exponent > 0 && repeat_exp > max_exponent {
                repeat_exp = max_exponent;
            }
            // `std::pow(float, int)` promotes to double upstream, and
            // the product is narrowed back to float on assignment;
            // computing it in f32 throughout drifts on long repeats.
            let penalty = (self.multiplier as f64) * (self.base as f64).powi(repeat_exp);
            out.insert(token, penalty as f32);
        }
        out
    }
}

/// DRY as a caller SPELLS it: four numbers and a list of strings, with
/// the breakers not yet tokenised.
///
/// This is the shape a CLI flag set and an HTTP request body arrive in,
/// and it exists so that both front ends resolve them the same way.
/// [`Self::resolve`] is the only bridge to [`DryParams`], and it is the
/// one place that decides what happens when DRY is asked for and there
/// is no vocabulary to tokenise the breakers against.
#[derive(Debug, Clone, PartialEq)]
pub struct DryRequest {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: i32,
    pub penalty_last_n: i32,
    pub sequence_breakers: Vec<String>,
}

impl Default for DryRequest {
    /// llama.cpp's defaults at the pin (`common/common.h:248-251, 264`),
    /// which are DISABLED: `dry_multiplier` is 0. The scan window is 64
    /// tokens, not the whole context (`-1`, which an older `common.h`
    /// had): a request that sets only `dry_multiplier` scans what
    /// llama-server scans.
    fn default() -> Self {
        DryRequest {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 64,
            sequence_breakers: DEFAULT_SEQUENCE_BREAKERS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

/// DRY was asked for and there is no vocabulary to tokenise its
/// sequence breakers against.
///
/// A refusal rather than a silent fallback to no breakers, because DRY
/// without its breakers is not DRY with fewer features: it scans across
/// the newline the caller told it to stop at, and the output looks like
/// working DRY. The only checkpoints this can happen on are the ones
/// with no real vocabulary at all (`ByteTokenizer`, the synthetic-weight
/// demo model), where no sampler configured by text could mean what it
/// says anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DryVocabMissing;

impl fmt::Display for DryVocabMissing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "the DRY sampler needs the model's vocabulary to tokenise its sequence \
             breakers, and this checkpoint has none loaded. Pass \
             `--dry-sequence-breaker none` to run DRY with no breakers, or \
             `--dry-multiplier 0` to switch DRY off",
        )
    }
}

impl std::error::Error for DryVocabMissing {}

impl DryRequest {
    /// The same three conditions [`DryParams::is_enabled`] reads, asked
    /// before the breakers have been tokenised -- so the expensive
    /// vocabulary walk is skipped for the overwhelming majority of
    /// requests, which leave DRY off.
    pub fn is_enabled(&self) -> bool {
        self.multiplier != 0.0 && self.base >= 1.0 && self.penalty_last_n != 0
    }

    /// Tokenise the breakers and build the sampler's parameters.
    ///
    /// `vocab` is `None` when the loaded checkpoint has no real
    /// vocabulary. That is fine while DRY is off and a refusal when it
    /// is on; see [`DryVocabMissing`].
    ///
    /// An EMPTY breaker list needs no vocabulary either: it is the
    /// caller's `--dry-sequence-breaker none`, and there is nothing to
    /// tokenise.
    pub fn resolve(
        &self,
        vocab: Option<&dyn DryVocab>,
        total_context_size: usize,
    ) -> Result<DryParams, DryVocabMissing> {
        if !self.is_enabled() {
            return Ok(DryParams::off());
        }
        let breakers = match (self.sequence_breakers.is_empty(), vocab) {
            (true, _) => DryBreakers::none(),
            (false, Some(vocab)) => DryBreakers::from_vocab(vocab, &self.sequence_breakers),
            (false, None) => return Err(DryVocabMissing),
        };
        Ok(DryParams::new(
            self.multiplier,
            self.base,
            self.allowed_length,
            self.penalty_last_n,
            total_context_size,
            breakers,
        ))
    }
}

#[cfg(test)]
mod tests {

    /// The request defaults are llama.cpp's at the pin
    /// (`common/common.h:248-251`), the scan window included: 64, not
    /// the whole context.
    #[test]
    fn the_dry_defaults_are_llama_cpp_s_at_the_pin() {
        let d = DryRequest::default();
        assert_eq!(
            (d.multiplier, d.base, d.allowed_length, d.penalty_last_n),
            (0.0, 1.75, 2, 64)
        );
        assert_eq!(d.sequence_breakers, ["\n", ":", "\"", "*"]);
    }

    use super::*;

    /// Every expected value below was produced by calling the real
    /// `llama_sampler_init_dry_testing` in `libllama` (llama.cpp b7650,
    /// `.scratch/llama.cpp`) from a small C++ harness and reading the
    /// logits back, not by reasoning about the algorithm. The harness
    /// applied the sampler to logits `ln(p)` for the probabilities
    /// llama.cpp's own `tests/test-sampling.cpp:357-361` uses, so these
    /// cases are upstream's cases with the arithmetic checked against
    /// upstream's code as it actually runs.
    fn dry(
        multiplier: f32,
        base: f32,
        allowed: i32,
        last_n: i32,
        breakers: &[Vec<usize>],
    ) -> DryParams {
        DryParams::new(
            multiplier,
            base,
            allowed,
            last_n,
            1024,
            DryBreakers::from_token_sequences(breakers),
        )
    }

    fn penalties_of(params: &DryParams, history: &[usize]) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = params
            .penalties(PenaltyWindow::new(&[], history))
            .into_iter()
            .collect();
        out.sort_unstable_by_key(|&(token, _)| token);
        out
    }

    /// The window `a b c a b` repeats its own two-token suffix `a b` at
    /// its head, so emitting `c` would extend that repetition to three.
    /// At `allowed_length = 2` the exponent is `2 - 2 = 0`, so the
    /// penalty is the bare multiplier.
    ///
    /// libllama: logit `ln(0.25) = -1.3862944` became `-2.3862944` for
    /// token 2 and nothing else moved.
    #[test]
    fn a_repeated_suffix_penalises_only_the_token_that_would_extend_it() {
        let params = dry(1.0, 1.1, 2, 5, &[]);
        assert_eq!(penalties_of(&params, &[0, 1, 2, 0, 1]), vec![(2, 1.0)]);

        // The multiplier scales it linearly: libllama gave -3.6094379
        // from ln(0.2) = -1.6094379, i.e. exactly 2.0.
        let doubled = dry(2.0, 1.1, 2, 5, &[]);
        assert_eq!(penalties_of(&doubled, &[0, 1, 2, 0, 1]), vec![(2, 2.0)]);
    }

    /// A window no longer than `allowed_length` cannot contain a
    /// repetition worth penalising, and upstream returns before it looks
    /// (`src/llama-sampler.cpp:3161`).
    ///
    /// libllama left all four logits at `ln(0.25)`.
    #[test]
    fn a_window_within_the_allowed_length_is_never_penalised() {
        let params = dry(1.0, 1.1, 2, 4, &[]);
        assert!(penalties_of(&params, &[0, 1]).is_empty());
    }

    /// `allowed_length` really is a free allowance: the window
    /// `0 1 2 3 4 0 1` repeats a two-token suffix, and at
    /// `allowed_length = 4` that is below the threshold, so nothing is
    /// penalised at all.
    ///
    /// libllama left all five logits at `ln(0.2)`.
    #[test]
    fn a_repetition_shorter_than_the_allowed_length_is_free() {
        let params = dry(1.0, 1.1, 4, 7, &[]);
        assert!(penalties_of(&params, &[0, 1, 2, 3, 4, 0, 1]).is_empty());
    }

    /// The exponent is the repetition length MINUS the allowance, and it
    /// grows the penalty geometrically -- which is the whole reason DRY
    /// exists where a flat repetition penalty does not.
    ///
    /// Window `0 1 2 3 0 1 2 3 0 1 2` (11 tokens): its seven-token
    /// suffix `0 1 2 3 0 1 2` also occurs at the head, so emitting `3`
    /// would extend a repetition of length 7. At `allowed_length = 2`
    /// and llama.cpp's default `base = 1.75` that is
    /// `0.8 * 1.75^5 = 13.130469`.
    ///
    /// libllama: logit `0.0` became `-13.1304693` for token 3, and no
    /// other logit moved.
    #[test]
    fn the_penalty_is_multiplier_times_base_to_the_length_over_the_allowance() {
        let params = dry(0.8, 1.75, 2, -1, &[]);
        let got = penalties_of(&params, &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]);
        assert_eq!(got.len(), 1, "only token 3 extends the repetition: {got:?}");
        assert_eq!(got[0].0, 3);
        assert!(
            (got[0].1 - 13.130_469).abs() < 1e-4,
            "0.8 * 1.75^5 = 13.130469, got {}",
            got[0].1
        );
    }

    /// A sequence breaker stops the scan, and a breaker that is a whole
    /// token by itself is additionally exempt from being penalised.
    ///
    /// Window `0 1 3 4 0 1` with `3` as a single-token breaker: the
    /// suffix `0 1` does repeat, and the token that would extend it is
    /// `3` -- which is the breaker, so upstream skips it
    /// (`src/llama-sampler.cpp:3325-3334`) and NOTHING is penalised.
    ///
    /// libllama left all five logits at `ln(0.2)`. Without the breaker
    /// the same window penalises token 3, which is what the second half
    /// asserts: a test that only checked the breaker case would pass for
    /// an implementation that penalised nothing ever.
    #[test]
    fn a_single_token_sequence_breaker_is_never_itself_penalised() {
        let with_breaker = dry(1.0, 1.1, 2, 6, &[vec![3]]);
        assert!(penalties_of(&with_breaker, &[0, 1, 3, 4, 0, 1]).is_empty());

        let without = dry(1.0, 1.1, 2, 6, &[]);
        assert_eq!(penalties_of(&without, &[0, 1, 3, 4, 0, 1]), vec![(3, 1.0)]);
    }

    /// Each of the three switches upstream reads turns DRY off on its
    /// own (`src/llama-sampler.cpp:3154`), and
    /// [`DryParams::is_enabled`] is the one place that decides.
    #[test]
    fn a_zero_multiplier_a_base_below_one_or_a_zero_window_all_disable_it() {
        let history = [0usize, 1, 2, 0, 1];
        for params in [
            DryParams::off(),
            dry(0.0, 1.75, 2, -1, &[]),
            dry(1.0, 0.5, 2, -1, &[]),
            dry(1.0, 1.75, 2, 0, &[]),
        ] {
            assert!(!params.is_enabled(), "{params:?} must be disabled");
            assert!(
                params
                    .penalties(PenaltyWindow::new(&[], &history))
                    .is_empty(),
                "{params:?} penalised something while disabled"
            );
        }
        assert!(dry(1.0, 1.75, 2, -1, &[]).is_enabled());
    }

    /// `dry_penalty_last_n` bounds how far back the scan looks, so a
    /// repetition that has fallen out of the window stops being
    /// penalised. `-1` means the whole context.
    #[test]
    fn the_scan_only_sees_the_last_n_tokens() {
        let history = [0usize, 1, 2, 0, 1];
        // The whole window: the `0 1` prefix is visible, so token 2 is
        // penalised (the case pinned against libllama above).
        assert_eq!(
            penalties_of(&dry(1.0, 1.1, 2, 5, &[]), &history),
            vec![(2, 1.0)]
        );
        // Only the last three tokens `2 0 1`: no repetition left.
        assert!(penalties_of(&dry(1.0, 1.1, 2, 3, &[]), &history).is_empty());
        // `-1` resolves to the context size, which is wide enough here.
        assert_eq!(
            penalties_of(&dry(1.0, 1.1, 2, -1, &[]), &history),
            vec![(2, 1.0)]
        );
    }

    /// The window is the tail of PROMPT ++ GENERATED, like every other
    /// penalty in this engine: llama.cpp feeds prompt tokens to the DRY
    /// sampler through `common_sampler_accept` exactly as it feeds
    /// generated ones (see [`crate::penalty_window`]).
    ///
    /// The same five tokens split differently across the seam must give
    /// the same answer, or the seam is being treated as a boundary DRY
    /// does not have.
    #[test]
    fn the_scan_reads_across_the_prompt_and_generation_seam() {
        let params = dry(1.0, 1.1, 2, 5, &[]);
        let whole = params.penalties(PenaltyWindow::new(&[], &[0, 1, 2, 0, 1]));
        let split = params.penalties(PenaltyWindow::new(&[0, 1, 2], &[0, 1]));
        let all_prompt = params.penalties(PenaltyWindow::new(&[0, 1, 2, 0, 1], &[]));
        assert_eq!(whole, split);
        assert_eq!(whole, all_prompt);
        assert_eq!(whole.len(), 1);
    }

    /// The exponent is clamped so `base ^ exponent` cannot overflow to
    /// infinity, which would make the penalty NaN once subtracted from
    /// a `-inf` logit (`src/llama-sampler.cpp:3310-3318`).
    ///
    /// A 400-token repetition at base 1.75 would ask for `1.75 ^ 398`,
    /// which is `inf` in an f32.
    #[test]
    fn a_very_long_repetition_clamps_the_exponent_rather_than_overflowing() {
        let params = dry(1.0, 1.75, 2, -1, &[]);
        let mut history: Vec<usize> = vec![0usize; 400];
        history.push(1);
        history.extend(std::iter::repeat_n(0usize, 400));
        let penalties = params.penalties(PenaltyWindow::new(&[], &history));
        for (token, penalty) in penalties {
            assert!(
                penalty.is_finite(),
                "token {token} got a non-finite penalty {penalty}"
            );
        }
    }

    /// A vocabulary whose tokens are whole words, so a breaker string
    /// can be contained in one token, span the end of one, or be absent.
    struct WordVocab {
        words: Vec<&'static str>,
    }

    impl DryVocab for WordVocab {
        fn n_tokens(&self) -> usize {
            self.words.len()
        }
        fn detokenize(&self, token: usize) -> String {
            self.words[token].to_string()
        }
        fn tokenize(&self, text: &str) -> Vec<usize> {
            // Longest-match-first over the same word list, which is
            // enough to tokenise the short tails a breaker leaves.
            let mut out = Vec::new();
            let mut rest = text;
            'outer: while !rest.is_empty() {
                let mut candidates: Vec<usize> = (0..self.words.len()).collect();
                candidates.sort_by_key(|&i| std::cmp::Reverse(self.words[i].len()));
                for i in candidates {
                    let w = self.words[i];
                    if !w.is_empty() && rest.starts_with(w) {
                        out.push(i);
                        rest = &rest[w.len()..];
                        continue 'outer;
                    }
                }
                break;
            }
            out
        }
    }

    /// A breaker that is a whole token gets an EMPTY tail, and one that
    /// only ever appears at the end of a longer token gets the tokens
    /// that must follow it.
    ///
    /// This is the half of `get_overlapping_token_sequences` a naive
    /// "is this token equal to the breaker" implementation misses
    /// entirely: with a vocabulary where `":"` is only ever the tail of
    /// `"foo:"`, such an implementation finds no breakers at all and DRY
    /// silently scans across every boundary it was told to stop at.
    #[test]
    fn a_breaker_is_found_inside_a_token_and_across_a_token_boundary() {
        let vocab = WordVocab {
            words: vec!["foo", "bar", ":", "ab", "c", "abc", "xab"],
        };
        // ":" is token 2 outright.
        let colon = DryBreakers::from_vocab(&vocab, &[":".to_string()]);
        assert_eq!(colon.tails(2), Some([Vec::new()].as_slice()));
        assert_eq!(colon.tails(0), None, "`foo` does not contain a colon");
        assert!(colon.is_single_token_breaker(2));

        // "abc" is token 5 outright, is CONTAINED by nothing else, and
        // OVERLAPS the end of "ab" (token 3, tail "c" = token 4) and of
        // "xab" (token 6, same tail).
        let abc = DryBreakers::from_vocab(&vocab, &["abc".to_string()]);
        assert_eq!(abc.tails(5), Some([Vec::new()].as_slice()));
        assert_eq!(abc.tails(3), Some([vec![4usize]].as_slice()));
        assert_eq!(abc.tails(6), Some([vec![4usize]].as_slice()));
        assert!(!abc.is_single_token_breaker(3), "`ab` needs `c` to follow");
        assert!(abc.is_single_token_breaker(5));
        assert_eq!(abc.tails(0), None);
    }

    /// A multi-token breaker only fires when its tail actually FOLLOWS,
    /// which is the reason the tail is stored at all.
    ///
    /// Two windows differing in one token, and the token differs only in
    /// whether it completes the breaker:
    ///
    /// * `foo bar : ab c foo bar : ab c` -- `ab`(3) is immediately
    ///   followed by `c`(4), so the breaker matches one token from the
    ///   end and `rep_limit` collapses to 0, below `allowed_length`;
    /// * `foo bar : ab xab foo bar : ab xab` -- `ab` is followed by
    ///   `xab`(6) instead, which is not its tail, so no breaker matches,
    ///   the five-token suffix is found repeating, and `foo` is
    ///   penalised for extending it.
    ///
    /// The third case is the control: the FIRST window with no breakers
    /// configured at all penalises too, so the emptiness above is the
    /// breaker's doing and not the window's.
    #[test]
    fn a_breaker_with_a_tail_only_stops_the_scan_when_the_tail_follows() {
        let vocab = WordVocab {
            words: vec!["foo", "bar", ":", "ab", "c", "abc", "xab"],
        };
        let breakers = DryBreakers::from_vocab(&vocab, &["abc".to_string()]);
        assert_eq!(breakers.tails(3), Some([vec![4usize]].as_slice()));
        let params = DryParams::new(1.0, 1.1, 2, -1, 1024, breakers);

        let broken = [0usize, 1, 2, 3, 4, 0, 1, 2, 3, 4];
        let unbroken = [0usize, 1, 2, 3, 6, 0, 1, 2, 3, 6];
        assert!(
            params
                .penalties(PenaltyWindow::new(&[], &broken))
                .is_empty(),
            "`ab` followed by `c` is the breaker, one token from the end"
        );
        assert!(
            !params
                .penalties(PenaltyWindow::new(&[], &unbroken))
                .is_empty(),
            "`ab` followed by `xab` is not the breaker, so the scan runs"
        );

        let no_breakers = DryParams::new(1.0, 1.1, 2, -1, 1024, DryBreakers::none());
        assert!(
            !no_breakers
                .penalties(PenaltyWindow::new(&[], &broken))
                .is_empty(),
            "the same window without breakers repeats, so the first \
             assertion is about the breaker and not about the window"
        );
    }

    /// An empty breaker string is skipped rather than turning every
    /// token into a breaker: `word.find("")` is 0 for every string, so a
    /// missing guard here would exempt the entire vocabulary from DRY
    /// while leaving it switched on.
    #[test]
    fn an_empty_breaker_string_is_skipped() {
        let vocab = WordVocab {
            words: vec!["foo", "bar"],
        };
        let breakers = DryBreakers::from_vocab(&vocab, &[String::new()]);
        assert!(breakers.is_empty());
        assert_eq!(breakers.raw(), &[String::new()]);
    }

    /// llama.cpp's defaults, spelled once and read by both front ends.
    #[test]
    fn the_default_breakers_are_llama_cpps_four() {
        assert_eq!(DEFAULT_SEQUENCE_BREAKERS, ["\n", ":", "\"", "*"]);
    }
}
