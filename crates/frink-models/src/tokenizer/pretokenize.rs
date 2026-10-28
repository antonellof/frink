//! Pre-tokenization: which regex a checkpoint's `tokenizer.ggml.pre`
//! selects, and how the text is actually cut with it.
//!
//! Its own module rather than another section of `tokenizer.rs`, which
//! is already past two thousand lines: everything here is a
//! transcription of a foreign source file and is reviewed *against that
//! file*, which is a different job from reviewing the BPE merge loop
//! next door.
//!
//! Two upstream files are the source of truth, and both matter:
//!
//! * `.scratch/llama.cpp/src/llama-vocab.cpp` — the `tokenizer_pre ==
//!   "..."` chain that maps a GGUF string onto a
//!   `LLAMA_VOCAB_PRE_TYPE_*`, and the `switch` that maps that type onto
//!   a regex. The mapping is many-to-one in both directions, so an arm
//!   here is keyed by the *set* of `pre` strings upstream sends to it.
//! * `.scratch/llama.cpp/src/unicode.cpp` — `unicode_regex_split`, which
//!   decides what happens to the text a pattern does NOT match. That
//!   half is [`split_with_gaps`], and getting it wrong loses input bytes
//!   rather than merely splitting them elsewhere.
//!
//! Patterns are transcribed from the strings llama.cpp actually
//! executes, not from the `// original regex from tokenizer.json`
//! comments beside them. The two are meant to be equivalent, but they
//! are not: upstream writes contractions as `'[sS]` where the original
//! writes `(?i:'s)`, and Rust's `(?i)` folds Unicode, so `(?i:'s)` here
//! would also match `'ſ` (U+017F) and `(?i:'k)` would match `'K`
//! (U+212A). Matching the executed form keeps the two engines equal.

/// The llama3 family (`LLAMA_VOCAB_PRE_TYPE_LLAMA3`, shared with
/// `CHATGLM4`). Digits group in threes; contractions fold case; a
/// punctuation run may swallow the newlines after it.
const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2 and the arms that share its regex (`QWEN2`, `STABLELM2`,
/// `HUNYUAN`, `SOLAR_OPEN`). Identical to [`LLAMA3`] except that digits
/// are emitted ONE at a time: `\p{N}` rather than `\p{N}{1,3}`.
///
/// The contractions fold case here too. Frink used to spell this arm
/// with lowercase-only contractions on the theory that qwen2 differed
/// from llama3 in that respect; upstream says otherwise, and the
/// difference is invisible on the obvious input (`DON'T` splits the same
/// either way, because `[^\r\n\p{L}\p{N}]?\p{L}+` also matches `'T`).
/// It shows on input like `'Sa`, where the contraction alternative wins
/// only if it folds case.
const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// `LLAMA_VOCAB_PRE_TYPE_QWEN35` (`llama-vocab.cpp:382-388`): Qwen2's
/// pattern with combining marks (`\p{M}`) admitted into a letter run
/// and excluded from a punctuation run. Not an alias: on the parity
/// corpus the two split `(x` apart (qwen2 arm, `(` then `x`) and
/// together (this arm) on Bonsai 2 27B, found by `frink parity`.
const QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// `LLAMA_VOCAB_PRE_TYPE_GPT4O` (shared with `MINIMAX_M2`). Its own
/// pattern, not an alias for anything: letter runs are split at the
/// case boundary and may carry a trailing contraction, digits group in
/// threes, and a punctuation run swallows `/` as well as `\r\n`.
///
/// The case split is written as `(?=[\p{L}])([^a-z])` — "a letter that
/// is not ASCII lowercase" — rather than `\p{Lu}`, because that is the
/// form llama.cpp executes. It is not a paraphrase of `\p{Lu}` and does
/// not behave like one: upstream collapses every non-ASCII letter to a
/// single marker byte before matching, so a non-ASCII letter takes the
/// `[^a-z]` branch there. Transcribed literally, Rust agrees with it
/// character for character.
const GPT4O: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Plain GPT-2 (`LLAMA_VOCAB_PRE_TYPE_GPT2`, shared verbatim with
/// `MPT`, `OLMO`, `JAIS`, `TRILLION` and `GRANITE_DOCLING`) and the
/// fallback for a `pre` nobody has an arm for, which is what upstream
/// does after printing its "GENERATION QUALITY WILL BE DEGRADED"
/// warning.
///
/// Note there is no trailing `|\s+`. That is not an omission: upstream
/// ends this pattern at `\s+(?!\S)`, and the whitespace it therefore
/// leaves unmatched comes out of [`split_with_gaps`] instead. Frink
/// used to carry two arms here — one with `|\s+` for "gpt2" and one
/// without for "olmo" — which is the same code path written twice to
/// vary it, and only the second one was ever exercised.
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Gemma-4 / SARVAM-style pre-split (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`):
/// carve on newlines only, so BPE can merge across ordinary word
/// boundaries after `" "` → `▁` escaping.
const NEWLINES: &str = r"[^\n]+|[\n]+";

fn compile(pattern: &str) -> fancy_regex::Regex {
    fancy_regex::Regex::new(pattern).expect("pre-tokenization patterns are fixed and valid")
}

/// The pre-tokenization pattern for a checkpoint's `tokenizer.ggml.pre`.
///
/// # Arms this deliberately does not have
///
/// Several upstream pre-types are a *list* of regexes applied in
/// sequence, each one re-splitting the chunks the previous one produced:
/// `SUPERBPE`, `MINERVA`/`MELLUM2`, `DEEPSEEK_LLM`, `DEEPSEEK_CODER`,
/// `DEEPSEEK3_LLM`, `TINY_AYA`, `PORO`, `VIKING`. One `Regex` cannot
/// express those, so the `pre` strings that select them are not listed
/// below and fall through to [`GPT2`] — except `deepseek-llm`,
/// `deepseek-coder` and `deepseek-v3`, which a previous change put on
/// the [`LLAMA3`] arm. Both are wrong; neither is proven wrong here,
/// because `frink parity`'s corpus has no checkpoint for any of them.
/// Adding them means teaching this function to return a sequence, which
/// is a change with an oracle behind it, not a guess.
pub(super) fn regex_for(pre: &str) -> fancy_regex::Regex {
    let pattern = match pre {
        // `LLAMA_VOCAB_PRE_TYPE_LLAMA3`. `falcon-h1`, `midm-2.0`,
        // (`glm5` joined `glm4` on the `CHATGLM4` arm at the
        // `604cc430` pin, `llama-vocab.cpp:2602-2607`; that arm's pattern,
        // `:414-417`, is this one byte for byte.)
        // `lfm2` and `jina-v5-nano` are in the same upstream arm and
        // were missing here; `deepseek-*` are NOT (see above).
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "deepseek-llm" | "deepseek-coder" | "deepseek-v3"
        | "chatglm-bpe" | "glm4" | "glm5" => LLAMA3,

        // `LLAMA_VOCAB_PRE_TYPE_QWEN2` and the arms sharing its regex.
        // `deepseek-r1-qwen`, `kormo`, `f2llmv2` and `megrez` all land
        // here upstream; `deepseek-r1-qwen` used to fall through to
        // GPT-2, which mis-split 7 of the 19 parity cases on
        // DeepSeek-R1-Distill-Qwen-1.5B.
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" | "stablelm2" | "hunyuan"
        | "solar-open" => QWEN2,

        // `LLAMA_VOCAB_PRE_TYPE_QWEN35` (`llama-vocab.cpp:2221`).
        "qwen35" => QWEN35,

        // `LLAMA_VOCAB_PRE_TYPE_GPT4O`. Previously aliased onto the
        // qwen2 arm, which tokenized every multi-digit number on
        // Phi-4-mini one digit at a time.
        "gpt-4o" | "llama4" | "kanana2" | "talkie" | "minimax-m2" => GPT4O,

        // `LLAMA_VOCAB_PRE_TYPE_GPT2`/`OLMO`/... and the fallback.
        _ => GPT2,
    };
    compile(pattern)
}

/// `tokenizer.ggml.pre` values llama.cpp gained at the `604cc430` pin
/// (2026-10-28) whose arm this module does not have, with the line that
/// says why. Each would otherwise fall through to [`GPT2`] in
/// [`regex_for`] and tokenize fluently and WRONGLY, where llama.cpp
/// either runs its own arm or, for a string it does not know, throws
/// `unknown pre-tokenizer type` (`llama-vocab.cpp:2760`).
///
/// Refused rather than added because none of them is one regex: a
/// sequence, a raw-UTF-8 mode without the GPT-2 byte map, or both. The
/// older multi-regex arms this module also lacks (see [`regex_for`]'s
/// doc) predate the pin and are not re-litigated here.
pub(super) const UNSERVED_PRE: &[(&str, &str)] = &[
    (
        "domyn-small",
        "`LLAMA_VOCAB_PRE_TYPE_DOMYN_SMALL` (llama-vocab.cpp:2653-2656, `:548-556`): SPM-style \
         BPE over raw UTF-8 with `escape_whitespaces`, not the GPT-2 byte map; it is the \
         Nemotron BPE tokenizer of #23343, on an architecture frink serves",
    ),
    (
        "flint",
        "`LLAMA_VOCAB_PRE_TYPE_FLINT` (llama-vocab.cpp:285-290): a SEQUENCE of two regexes",
    ),
    (
        "k2-horizon",
        "`LLAMA_VOCAB_PRE_TYPE_K2_HORIZON` (llama-vocab.cpp:571-574): letters may carry \
         combining marks and ZWJ/ZWNJ, a pattern no arm here has",
    ),
    (
        "mmbert",
        "`LLAMA_VOCAB_PRE_TYPE_MMBERT` (llama-vocab.cpp:556-562,679): raw UTF-8 and words \
         split in `tokenize()`, not by a regex",
    ),
    (
        "nemotron-3.5-embed",
        "`LLAMA_VOCAB_PRE_TYPE_NEMOTRON_3_5_EMBD` (llama-vocab.cpp:425-431,649-654,2752-2755): \
         the tekken pattern after a ZWNJ normalisation, with `ignore_merges` and `add_bos`",
    ),
    (
        "soprano",
        "`LLAMA_VOCAB_PRE_TYPE_SOPRANO` (llama-vocab.cpp:576-578,655,793): a SEQUENCE of two \
         regexes over raw UTF-8",
    ),
];

/// Why `pre` cannot be tokenized here, or `None` when an arm (or the
/// GPT-2 fallback upstream also takes) covers it.
pub(super) fn unserved_pre(pre: &str) -> Option<&'static str> {
    UNSERVED_PRE
        .iter()
        .find(|(name, _)| *name == pre)
        .map(|(_, why)| *why)
}

/// The Gemma-4 newline-only pattern.
pub(super) fn newline_regex() -> fancy_regex::Regex {
    compile(NEWLINES)
}

/// Cuts `text` into the chunks BPE then merges over — matches AND the
/// text between them.
///
/// # Why the gaps are not optional
///
/// A regex `find_iter` loop yields only what matched. llama.cpp's
/// `unicode_regex_split_stl` (`unicode.cpp`) emits every unmatched span
/// as a chunk of its own, and its hand-written fast paths do the same
/// thing by construction — `unicode_regex_split_custom_gpt2` ends its
/// loop with `_add_token(++pos)` for a codepoint nothing claimed.
///
/// Frink used to drop them. On any arm whose pattern does not end in a
/// catch-all — [`GPT2`], which is most of them — that is not a different
/// segmentation, it is **input the model never sees**: tabs, NBSP, form
/// feeds and interior newlines disappeared out of the middle of the
/// prompt. The property this function exists to hold is therefore not
/// "the chunks are right" but the stronger one that
/// `split_with_gaps(re, t).concat() == t` for every `re` and every `t`.
///
/// A `fancy_regex` match can fail at run time (its backtrack limit), and
/// the same property decides what to do about it: stop matching and hand
/// back the rest of the input as one chunk, so a pathological pattern
/// degrades the split and never eats the text.
pub(super) fn split_with_gaps<'t>(re: &fancy_regex::Regex, text: &'t str) -> Vec<&'t str> {
    let mut chunks = Vec::new();
    let mut cursor = 0usize;
    for m in re.find_iter(text) {
        let Ok(m) = m else { break };
        if m.start() > cursor {
            chunks.push(&text[cursor..m.start()]);
        }
        chunks.push(m.as_str());
        cursor = m.end();
    }
    if cursor < text.len() {
        chunks.push(&text[cursor..]);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(pre: &str, text: &str) -> Vec<String> {
        split_with_gaps(&regex_for(pre), text)
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// The three divergences from llama.cpp that one hardcoded regex
    /// caused, pinned so they cannot come back.
    ///
    /// Every BPE checkpoint used to get the GPT-2 pattern regardless of
    /// what `tokenizer.ggml.pre` said. Against the llama3 family that
    /// meant different ids for any number of four or more digits, for
    /// uppercase contractions, and for any interior run of two or more
    /// spaces, which is every indented line of code.
    ///
    /// Expected splits are llama.cpp's, from its `LLAMA_VOCAB_PRE_TYPE`
    /// patterns, not from what this implementation happens to produce.
    #[test]
    fn the_llama3_pretokenizer_groups_digits_and_keeps_uppercase_contractions() {
        // Digits group in threes. The GPT-2 pattern took the whole run.
        assert_eq!(split("llama3", "1234567"), vec!["123", "456", "7"]);

        // Contractions are case-insensitive here, so DON'T keeps its
        // apostrophe with the T rather than splitting three ways.
        assert_eq!(split("llama3", "DON'T"), vec!["DON", "'T"]);

        // An interior run of spaces is one piece, and the word after it
        // keeps its leading space. This is the one that moves indented
        // code and blank lines.
        assert_eq!(
            split("llama3", "hello  world"),
            vec!["hello", " ", " world"]
        );
    }

    /// Qwen2 is a SEPARATE arm, not an alias for llama3, and the
    /// difference is the digit rule: `\p{N}` one at a time versus
    /// llama3's `\p{N}{1,3}`.
    ///
    /// Worth pinning because the two patterns look alike and the
    /// temptation is to merge them. Note what is NOT different here:
    /// `DON'T` splits the same either way, because
    /// `[^\r\n\p{L}\p{N}]?\p{L}+` matches `'T` whether or not the
    /// contraction alternatives fold case.
    #[test]
    fn qwen2_splits_digits_singly_where_llama3_groups_them() {
        assert_eq!(split("qwen2", "1234"), vec!["1", "2", "3", "4"]);
        assert_eq!(split("llama3", "1234"), vec!["123", "4"]);
        assert_eq!(
            split("qwen2", "DON'T"),
            split("llama3", "DON'T"),
            "the contraction case rule does not show on uppercase input"
        );
    }

    /// **Defect 1.** `deepseek-r1-qwen` had no arm and fell through to
    /// GPT-2, which takes a digit run whole and has no `\s*[\r\n]+`
    /// clause. Upstream sends it, `kormo` and `f2llmv2` to
    /// `LLAMA_VOCAB_PRE_TYPE_QWEN2` (`llama-vocab.cpp`, the
    /// `tokenizer_pre == "qwen2"` arm).
    ///
    /// Asserted against the qwen2 arm's own output rather than against a
    /// literal, so the claim is "these are the same pre-tokenizer",
    /// which is what upstream says — and against GPT-2's output too, so
    /// that a day when qwen2 and gpt2 happen to agree cannot make this
    /// pass vacuously.
    #[test]
    fn the_deepseek_r1_qwen_group_is_the_qwen2_arm_not_the_gpt2_fallback() {
        // Digit grouping and blank-line handling both differ between
        // the two candidate arms, so one input shows either mistake.
        let text = "para 1234\n\n  end";
        assert_eq!(
            split("qwen2", text),
            vec!["para", " ", "1", "2", "3", "4", "\n\n", " ", " end"]
        );
        assert_ne!(
            split("qwen2", text),
            split("some-pre-with-no-arm", text),
            "the two arms must differ, or this test proves nothing"
        );
        for pre in ["deepseek-r1-qwen", "kormo", "f2llmv2"] {
            assert_eq!(
                split(pre, text),
                split("qwen2", text),
                "{pre} is the qwen2 pre-tokenizer upstream"
            );
        }
    }

    /// **Defect 2.** `gpt-4o` was aliased onto the qwen2 arm.
    /// `LLAMA_VOCAB_PRE_TYPE_GPT4O` is its own pattern
    /// (`llama-vocab.cpp`), and all three of the ways it differs are
    /// asserted here because each one moves real ids on Phi-4-mini:
    /// three-digit grouping, the case-split letter run that carries its
    /// contraction, and `/` joining `\r\n` in the punctuation tail.
    #[test]
    fn gpt_4o_is_its_own_arm_and_not_the_qwen2_one() {
        // Digits in threes, where qwen2 emits them one at a time.
        assert_eq!(split("gpt-4o", "1234567"), vec!["123", "456", "7"]);
        assert_eq!(
            split("qwen2", "1234567"),
            vec!["1", "2", "3", "4", "5", "6", "7"]
        );

        // A capitalised word keeps its contraction: llama.cpp gives
        // Phi-4-mini a single `It's` token here, qwen2 gives `It` `'s`.
        assert_eq!(split("gpt-4o", "It's"), vec!["It's"]);
        assert_eq!(split("qwen2", "It's"), vec!["It", "'s"]);

        // `[\r\n/]*` rather than `[\r\n]*` after a punctuation run.
        assert_eq!(split("gpt-4o", "a.\n/b"), vec!["a", ".\n/", "b"]);
        assert_eq!(split("qwen2", "a.\n/b"), vec!["a", ".\n", "/b"]);

        // The whole GPT4O group, so an alias cannot silently drop out.
        for pre in ["llama4", "kanana2", "talkie"] {
            assert_eq!(split(pre, "It's 1234567"), split("gpt-4o", "It's 1234567"));
        }
    }

    /// **Defect 3, the one that lost data.** Everything the pattern does
    /// not match still has to come out, in place.
    ///
    /// The GPT-2/OLMo arm is the one that shows it, because its pattern
    /// stops at `\s+(?!\S)` with no trailing `\s+`: a tab between two
    /// words, an NBSP, a form feed and a single interior newline are all
    /// unmatched, and frink used to drop them on the floor. Asserted as
    /// the concatenation identity rather than as a chunk list, because
    /// the property that matters is that no BYTE is lost, whatever the
    /// splits turn out to be.
    #[test]
    fn no_arm_can_drop_the_text_between_its_matches() {
        let hostile = "a\tb\u{a0}c\nd\u{c}e\u{7}f  \u{2009}\r\n\tg\t";
        for pre in [
            "olmo",
            "gpt-2",
            "llama3",
            "qwen2",
            "gpt-4o",
            "something-nobody-has-heard-of",
        ] {
            let chunks = split(pre, hostile);
            assert_eq!(
                chunks.concat(),
                hostile,
                "the {pre} arm lost input: {chunks:?}"
            );
        }

        // And the specific splits llama.cpp produces for the gaps, so
        // that "no bytes lost" cannot be satisfied by emitting the whole
        // string as one chunk.
        assert_eq!(split("olmo", "a\tb"), vec!["a", "\t", "b"]);
        assert_eq!(
            split("olmo", "para\n\nnext"),
            vec!["para", "\n", "\n", "next"]
        );
        assert_eq!(split("olmo", "a\u{a0}b"), vec!["a", "\u{a0}", "b"]);
    }

    /// The GPT-2 arm and the OLMo arm are the same string upstream
    /// (`case LLAMA_VOCAB_PRE_TYPE_GPT2: case ... OLMO:`), and an
    /// unknown `pre` falls back to it, which is what llama.cpp does
    /// after its "quality will be degraded" warning.
    #[test]
    fn gpt2_olmo_and_the_fallback_are_one_arm() {
        let text = "hi   there\tand\n\nmore  ";
        let expected = split("gpt-2", text);
        assert_eq!(split("olmo", text), expected);
        assert_eq!(split("jais", text), expected);
        assert_eq!(split("something-nobody-has-heard-of", text), expected);
        assert_eq!(
            split("gpt-2", "hi   "),
            vec!["hi", "   "],
            "trailing whitespace is its own run, which is what `\\s+(?!\\S)` is for"
        );
    }

    /// The Gemma-4 pattern carves on newlines and nothing else, and a
    /// newline run is its own chunk so a multi-newline vocabulary entry
    /// can be looked up whole.
    #[test]
    fn the_gemma4_pattern_only_carves_on_newlines() {
        let chunks = split_with_gaps(&newline_regex(), "one two\n\nthree");
        assert_eq!(chunks, vec!["one two", "\n\n", "three"]);
    }

    /// `glm5` is `glm4`'s arm upstream (`CHATGLM4`), so it must split
    /// exactly as `glm4` does here and not as the GPT-2 fallback: digits
    /// in threes is the visible difference.
    #[test]
    fn glm5_splits_as_glm4_and_not_as_the_fallback() {
        let text = "pay 12345 now";
        assert_eq!(split("glm5", text), split("glm4", text));
        assert_ne!(split("glm5", text), split("gpt2", text));
    }

    /// Every refused `pre` is one this module has NO arm for: a name on
    /// both lists would be a refusal that hides a served arm, or an arm
    /// that a refusal makes unreachable.
    #[test]
    fn an_unserved_pre_is_not_also_an_arm() {
        let text = "Hello 12345 world's  end\n\nx";
        for (pre, why) in UNSERVED_PRE {
            assert!(why.contains("llama-vocab.cpp:"), "{pre}: {why}");
            assert_eq!(
                split(pre, text),
                split("gpt2", text),
                "{pre} is refused but regex_for has an arm for it"
            );
            assert_eq!(unserved_pre(pre), Some(*why));
        }
        assert_eq!(unserved_pre("llama-bpe"), None);
        assert_eq!(unserved_pre("glm5"), None);
    }
}
