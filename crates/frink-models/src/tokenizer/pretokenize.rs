//! Pre-tokenization: which regex list a checkpoint's `tokenizer.ggml.pre`
//! selects, and whether its vocabulary skips merges for a whole word.
//!
//! Its own module rather than another section of `tokenizer.rs`: this
//! is a transcription of a foreign source file and is reviewed *against
//! that file*, which is a different job from reviewing the BPE merge
//! loop next door.
//!
//! Two upstream files are the source of truth, and both matter:
//!
//! * `src/llama-vocab.cpp` — the `tokenizer_pre == "..."` chain that
//!   maps a GGUF string onto a `LLAMA_VOCAB_PRE_TYPE_*` (and sets
//!   `ignore_merges`), and the `switch` that maps that type onto a list
//!   of regexes ([`super::pre_patterns`]). The mapping is many-to-one in
//!   both directions, so an arm here is keyed by the *set* of `pre`
//!   strings upstream sends to it.
//! * `src/unicode.cpp` — `unicode_regex_split`, which decides what
//!   happens to the text a pattern does NOT match, how a sequence of
//!   patterns composes, and which engine runs each pattern. That half is
//!   [`super::regex_split`].
//!
//! Every `pre` string the chain knows is either an arm here or refused
//! by name in [`UNSERVED_PRE`]; a string it does not know is refused
//! too, because upstream throws `unknown pre-tokenizer type` for it
//! (`llama-vocab.cpp:2760`). There is no fallback arm: serving an
//! unknown `pre` with the GPT-2 pattern would tokenize fluently and
//! wrongly. A file with no `pre` at all gets upstream's `DEFAULT` list,
//! which is not the GPT-2 pattern either.
//!
//! Patterns are transcribed from the strings llama.cpp actually
//! executes, not from the `// original regex from tokenizer.json`
//! comments beside them. The two are meant to be equivalent, but they
//! are not: upstream writes contractions as `'[sS]` where the original
//! writes `(?i:'s)`, and Rust's `(?i)` folds Unicode, so `(?i:'s)` here
//! would also match `'ſ` (U+017F) and `(?i:'k)` would match `'K`
//! (U+212A). Matching the executed form keeps the two engines equal.

use super::pre_patterns::*;
use super::regex_split::{PreTokenizer, Step};

/// One `LLAMA_VOCAB_PRE_TYPE_*` as upstream's load selects it: the
/// `pre` strings that reach it with the same flags, its regex list, and
/// whether `ignore_merges` is set (`llama-vocab.cpp:2452-2761`).
struct Arm {
    pres: &'static [&'static str],
    regexes: &'static [&'static str],
    ignore_merges: bool,
}

const fn arm(
    pres: &'static [&'static str],
    regexes: &'static [&'static str],
    ignore_merges: bool,
) -> Arm {
    Arm {
        pres,
        regexes,
        ignore_merges,
    }
}

/// Every arm frink serves. Two rows may share a list when upstream sets
/// different flags for them (`chatglm-bpe` and `glm4` are both
/// `CHATGLM4`, but only the second sets `ignore_merges`).
#[rustfmt::skip]
const ARMS: &[Arm] = &[
    // `""` is the missing key (`llama-vocab.cpp:2456-2464`).
    arm(&["", "default"], DEFAULT, false),
    arm(&["flint"], FLINT, false),
    arm(&["minicpm5"], MINICPM5, true),
    arm(&["llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "midm-2.0", "jina-v5-nano", "lfm2"], &[LLAMA3], true),
    arm(&["dbrx", "smaug-bpe", "chatglm-bpe"], &[LLAMA3], false),
    arm(&["glm4", "glm5"], &[LLAMA3], true),
    arm(&["jais-2"], JAIS2, false),
    arm(&["deepseek-llm"], DEEPSEEK_LLM, false),
    arm(&["deepseek-coder"], DEEPSEEK_CODER, false),
    arm(&["deepseek-v3", "hunyuan-dense", "joyai-llm", "hy_v4"], DEEPSEEK3_LLM, false),
    arm(&["spark2_5"], SPARK2_5, false),
    arm(&["youtu"], YOUTU, true),
    arm(&["falcon"], FALCON, false),
    arm(&["starcoder", "refact", "command-r", "smollm", "codeshell", "exaone", "minerva-7b", "mellum2"], STARCODER, false),
    arm(&[
        "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0",
        "mellum", "modern-bert", "jina-v1-en", "jina-v2-code", "roberta-bpe", "mpt", "olmo", "jais",
        "exaone4", "trillion", "granite-docling",
    ], &[GPT2], false),
    arm(&["qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "megrez", "stablelm2", "hunyuan", "solar-open", "grok-2"], &[QWEN2], false),
    arm(&["qwen35"], &[QWEN35], false),
    arm(&["poro-chat", "bloom", "gpt3-finnish"], PORO, false),
    arm(&["viking"], VIKING, false),
    arm(&["tekken", "pixtral"], TEKKEN, true),
    arm(&["chameleon"], CHAMELEON, false),
    arm(&["gpt-4o", "llama4", "kanana2", "talkie", "minimax-m2"], GPT4O, false),
    // `GRANITE_EMB_MULTI` is GPT4O's string with `ignore_merges`.
    arm(&["granite-embed-multi-97m"], GPT4O, true),
    arm(&["tiny_aya", "cohere2moe"], TINY_AYA, false),
    arm(&["kimi-k2"], KIMI_K2, false),
    arm(&["superbpe"], SUPERBPE, false),
    arm(&["bailingmoe", "bailingmoe2", "llada-moe"], BAILINGMOE, false),
    arm(&["seed-coder"], SEED_CODER, false),
    arm(&["ufakzeka"], UFAKZEKA, false),
    arm(&["afmoe"], AFMOE, false),
    arm(&["laguna"], LAGUNA, false),
    arm(&["exaone-moe"], EXAONE_MOE, false),
];

/// `unicode_regex_split_custom`'s dispatch: a pattern upstream has a
/// hand-written splitter for is run by frink's stand-in for that
/// splitter, not as a regex. Any other pattern goes to the generic
/// engines in [`super::regex_split`].
fn custom_splitter(pattern: &str) -> Option<Step> {
    match pattern {
        GPT2 | LLAMA3 | QWEN2 | QWEN35 | NEWLINES => Some(Step::native(pattern)),
        TINY_AYA_DIGITS => Some(Step::digit_groups()),
        p if p == AFMOE[0] => Some(Step::digit_groups()),
        p if p == KIMI_K2[0] => Some(Step::kimi_k2()),
        _ => None,
    }
}

fn sequence(patterns: &[&str]) -> PreTokenizer {
    PreTokenizer::sequence(patterns, custom_splitter)
}

/// What a served `pre` selects.
pub(super) struct PreSplit {
    pub(super) pretokenizer: PreTokenizer,
    /// `llama_vocab::get_ignore_merges`: a pre-split word that is itself
    /// a vocabulary entry is emitted as that one token, without BPE.
    pub(super) ignore_merges: bool,
}

/// The pre-tokenizer for a checkpoint's `tokenizer.ggml.pre` under
/// `tokenizer.ggml.model = "gpt2"`, or why it is refused.
pub(super) fn pre_split(pre: &str) -> Result<PreSplit, &'static str> {
    if let Some(why) = unserved_pre(pre) {
        return Err(why);
    }
    let arm = ARMS
        .iter()
        .find(|a| a.pres.contains(&pre))
        .ok_or(UNKNOWN_PRE)?;
    Ok(PreSplit {
        pretokenizer: sequence(arm.regexes),
        ignore_merges: arm.ignore_merges,
    })
}

/// The refusal for a `pre` upstream's chain does not know.
const UNKNOWN_PRE: &str = "llama.cpp does not know this value either and refuses the file \
     (`unknown pre-tokenizer type`, llama-vocab.cpp:2760)";

/// `tokenizer.ggml.pre` values upstream serves that frink refuses, with
/// the line that says why. None is a regex list alone.
pub(super) const UNSERVED_PRE: &[(&str, &str)] = &[
    (
        "domyn-small",
        "`LLAMA_VOCAB_PRE_TYPE_DOMYN_SMALL` (llama-vocab.cpp:2653-2656, `:548-556`): SPM-style \
         BPE over raw UTF-8 with `escape_whitespaces`, not the GPT-2 byte map; it is the \
         Nemotron BPE tokenizer of #23343, on an architecture frink serves",
    ),
    (
        "gemma4",
        "`LLAMA_VOCAB_PRE_TYPE_GEMMA4` under `tokenizer.ggml.model = gpt2` \
         (llama-vocab.cpp:2537-2541, `:537-546`): SPM-style BPE over raw UTF-8 with \
         `escape_whitespaces`; frink runs that only for `tokenizer.ggml.model = gemma4`",
    ),
    (
        "granite-embed-multi-311m",
        "`LLAMA_VOCAB_PRE_TYPE_GEMMA4` (llama-vocab.cpp:2537-2541, `:537-546`): SPM-style \
         BPE over raw UTF-8 with `escape_whitespaces`; frink runs that only for \
         `tokenizer.ggml.model = gemma4`",
    ),
    (
        "k2-horizon",
        "`LLAMA_VOCAB_PRE_TYPE_K2_HORIZON` (llama-vocab.cpp:571-574): run by \
         `unicode_regex_split_custom_k2_horizon`, a hand-written splitter frink has no port of",
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
        "sarvam-moe",
        "`LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE` (llama-vocab.cpp:2547-2550, `:547-555`): SPM-style \
         BPE over raw UTF-8 with `escape_whitespaces`, not the GPT-2 byte map",
    ),
    (
        "soprano",
        "`LLAMA_VOCAB_PRE_TYPE_SOPRANO` (llama-vocab.cpp:576-578,655-669,793-794): the text is \
         lowercased and its whitespace runs collapsed before the split, BPE runs over raw \
         UTF-8 without the GPT-2 byte map, and a piece missing from the vocabulary becomes \
         `<unk>`; frink's BPE has none of the three",
    ),
    (
        "whitespace",
        "`LLAMA_VOCAB_PRE_TYPE_WHITESPACE` (llama-vocab.cpp:2558-2561, `:580-586`): BPE over \
         raw UTF-8 without the GPT-2 byte map, with its own normalizer options",
    ),
];

/// Why `pre` is refused, when it is one upstream serves and frink does
/// not.
pub(super) fn unserved_pre(pre: &str) -> Option<&'static str> {
    UNSERVED_PRE
        .iter()
        .find(|(name, _)| *name == pre)
        .map(|(_, why)| *why)
}

/// The Gemma-4 newline-only pattern, for `tokenizer.ggml.model =
/// "gemma4"`, whose `pre` upstream forces to `gemma4`
/// (`llama-vocab.cpp:2446`).
pub(super) fn newline_regex() -> PreTokenizer {
    sequence(&[NEWLINES])
}

#[cfg(test)]
#[path = "pretokenize_golden.rs"]
mod golden;

#[cfg(test)]
mod tests {
    use super::golden::GOLDEN;
    use super::*;

    fn split(pre: &str, text: &str) -> Vec<String> {
        pre_split(pre)
            .unwrap_or_else(|why| panic!("{pre:?} is refused: {why}"))
            .pretokenizer
            .split(text)
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
            split("gpt-2", text),
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
        for pre in ["olmo", "gpt-2", "llama3", "qwen2", "gpt-4o", ""] {
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
    /// (`case LLAMA_VOCAB_PRE_TYPE_GPT2: case ... OLMO:`).
    #[test]
    fn gpt2_and_olmo_are_one_arm() {
        let text = "hi   there\tand\n\nmore  ";
        let expected = split("gpt-2", text);
        assert_eq!(split("olmo", text), expected);
        assert_eq!(split("jais", text), expected);
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
        let chunks = newline_regex().split("one two\n\nthree");
        assert_eq!(chunks, vec!["one two", "\n\n", "three"]);
    }

    /// `glm5` is `glm4`'s arm upstream (`CHATGLM4`), so it must split
    /// exactly as `glm4` does here and not as GPT-2: digits in threes is
    /// the visible difference.
    #[test]
    fn glm5_splits_as_glm4_and_not_as_gpt2() {
        let text = "pay 12345 now";
        assert_eq!(split("glm5", text), split("glm4", text));
        assert_ne!(split("glm5", text), split("gpt-2", text));
    }

    /// Every `pre` string upstream's chain knows, in the order of
    /// `llama-vocab.cpp:2456-2761` (`""` is the missing key). Each one is
    /// an arm or refused by name, exactly once; anything else is refused
    /// as upstream refuses it. `tools/oracle/regex_split.py` checks the
    /// served ones against libllama.
    #[rustfmt::skip]
    const UPSTREAM_PRES: &[&str] = &[
        "", "flint", "default", "minicpm5", "llama3", "llama-v3", "llama-bpe", "falcon3",
        "falcon-h1", "midm-2.0", "jina-v5-nano", "lfm2", "deepseek-llm", "deepseek-coder",
        "deepseek-v3", "spark2_5", "youtu", "falcon", "mpt", "starcoder", "gpt-2", "phi-2",
        "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0", "mellum",
        "modern-bert", "jais-2", "gemma4", "granite-embed-multi-311m", "mmbert", "sarvam-moe",
        "jina-v1-en", "jina-v2-code", "roberta-bpe", "whitespace", "refact", "command-r",
        "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "qwen35", "stablelm2", "olmo", "dbrx",
        "smaug-bpe", "poro-chat", "chatglm-bpe", "glm4", "glm5", "viking", "jais", "tekken",
        "pixtral", "smollm", "codeshell", "bloom", "gpt3-finnish", "exaone", "exaone4",
        "exaone-moe", "chameleon", "minerva-7b", "domyn-small", "megrez", "gpt-4o", "llama4",
        "kanana2", "talkie", "granite-embed-multi-97m", "tiny_aya", "cohere2moe", "superbpe",
        "trillion", "granite-docling", "bailingmoe", "bailingmoe2", "llada-moe", "seed-coder",
        "hunyuan", "hunyuan-dense", "hy_v4", "joyai-llm", "kimi-k2", "ufakzeka", "grok-2",
        "afmoe", "laguna", "minimax-m2", "solar-open", "mellum2", "k2-horizon",
        "nemotron-3.5-embed", "soprano",
    ];

    #[test]
    fn every_upstream_pre_is_an_arm_or_refused_by_name_exactly_once() {
        for pre in UPSTREAM_PRES {
            let arms = ARMS.iter().filter(|a| a.pres.contains(pre)).count();
            let refused = UNSERVED_PRE.iter().filter(|(n, _)| n == pre).count();
            assert_eq!(
                arms + refused,
                1,
                "{pre:?}: {arms} arms, {refused} refusals"
            );
        }
        let names = ARMS.iter().flat_map(|a| a.pres.iter());
        for name in names.chain(UNSERVED_PRE.iter().map(|(n, _)| n)) {
            assert!(
                UPSTREAM_PRES.contains(name),
                "{name:?} is not a pre upstream knows"
            );
        }
        for (pre, why) in UNSERVED_PRE {
            assert!(why.contains("llama-vocab.cpp:"), "{pre}: {why}");
            assert_eq!(pre_split(pre).err(), Some(*why));
        }
    }

    /// A `pre` upstream does not know is refused, as upstream refuses
    /// it, and not served with the GPT-2 pattern as frink once did.
    #[test]
    fn an_unknown_pre_is_refused_not_defaulted() {
        for pre in ["gpt2", "something-nobody-has-heard-of", "LLAMA3"] {
            assert_eq!(pre_split(pre).err(), Some(UNKNOWN_PRE), "{pre}");
        }
    }

    /// The vocabularies upstream sets `ignore_merges` for
    /// (`llama-vocab.cpp:2452-2761`), and no others.
    #[test]
    fn ignore_merges_is_set_where_upstream_sets_it() {
        #[rustfmt::skip]
        const SET: &[&str] = &[
            "minicpm5", "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "midm-2.0",
            "jina-v5-nano", "lfm2", "youtu", "glm4", "glm5", "tekken", "pixtral",
            "granite-embed-multi-97m",
        ];
        for pre in UPSTREAM_PRES {
            if let Ok(arm) = pre_split(pre) {
                assert_eq!(arm.ignore_merges, SET.contains(pre), "{pre:?}");
            }
        }
    }

    #[test]
    fn every_list_arm_splits_as_libllama_does() {
        for (pre, text, want) in GOLDEN {
            assert_eq!(split(pre, text), *want, "{pre:?} on {text:?}");
        }
    }

    /// Each list arm must cut at least one of its golden texts
    /// differently from the arm it used to land on, or the golden could
    /// not tell them apart. #624: `deepseek-*` were on the llama3 arm and
    /// the others on the GPT-2 fallback. #629: the rest of the fallback's
    /// passengers, `pixtral` (on llama3), and a file with no `pre`.
    #[test]
    fn a_list_arm_is_not_the_arm_it_used_to_fall_on() {
        for (pre, old) in [
            ("deepseek-llm", "llama3"),
            ("deepseek-coder", "llama3"),
            ("deepseek-v3", "llama3"),
            ("starcoder", "gpt-2"),
            ("poro-chat", "gpt-2"),
            ("viking", "gpt-2"),
            ("tiny_aya", "gpt-2"),
            ("superbpe", "gpt-2"),
            ("falcon", "gpt-2"),
            ("jais-2", "llama3"),
            ("chameleon", "gpt-2"),
            ("kimi-k2", "gpt-2"),
            ("spark2_5", "gpt-2"),
            ("youtu", "gpt-2"),
            ("flint", "gpt-2"),
            ("tekken", "llama3"),
            ("", "gpt-2"),
            ("afmoe", "gpt-2"),
            ("laguna", "gpt-2"),
            ("exaone-moe", "gpt-2"),
            ("minicpm5", "gpt-2"),
            ("bailingmoe", "gpt-2"),
            ("seed-coder", "gpt-2"),
            ("ufakzeka", "gpt-2"),
        ] {
            assert!(
                GOLDEN
                    .iter()
                    .filter(|(p, _, _)| *p == pre)
                    .any(|(_, text, want)| split(old, text) != *want),
                "no golden text separates {pre:?} from {old}"
            );
        }
    }

    /// The GPT4O lookahead is `(?=[\p{L}\p{M}])` at the pin; frink had
    /// `(?=[\p{L}])`. The difference shows where a combining mark follows
    /// a lowercase run (`ae\u{301}B`): with marks admitted the mark stays
    /// on `ae`, without them it starts the next piece. The expected cut
    /// is libllama's (`pretokenize_golden.rs`, the `gpt-4o` rows).
    #[test]
    fn the_gpt4o_lookahead_admits_marks() {
        let old = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert!(GPT4O[0].contains(r"(?=[\p{L}\p{M}])"));
        let text = "CamelCase ae\u{301}B";
        let want = [" Camel", "Case", " ae\u{301}", "B"];
        assert_eq!(split("gpt-4o", &format!(" {text}"))[..], want);
        assert_ne!(sequence(&[old]).split(&format!(" {text}"))[..], want);
    }

    /// Every `pre` string upstream sends to a list arm splits as the
    /// arm's golden representative does. An arm is keyed by the set of
    /// strings; a member left out is refused as unknown.
    #[test]
    fn every_member_of_a_list_arm_is_served_by_it() {
        for (arm, members) in [
            ("deepseek-v3", &["hunyuan-dense", "joyai-llm", "hy_v4"][..]),
            (
                "starcoder",
                &[
                    "refact",
                    "command-r",
                    "smollm",
                    "codeshell",
                    "exaone",
                    "minerva-7b",
                    "mellum2",
                ][..],
            ),
            ("poro-chat", &["bloom", "gpt3-finnish"][..]),
            ("tiny_aya", &["cohere2moe"][..]),
            ("tekken", &["pixtral"][..]),
            (
                "gpt-4o",
                &[
                    "llama4",
                    "kanana2",
                    "talkie",
                    "minimax-m2",
                    "granite-embed-multi-97m",
                ][..],
            ),
            ("", &["default"][..]),
            ("bailingmoe", &["bailingmoe2", "llada-moe"][..]),
        ] {
            for member in members {
                for (_, text, want) in GOLDEN.iter().filter(|(p, _, _)| *p == arm) {
                    assert_eq!(split(member, text), *want, "{member} is the {arm:?} arm");
                }
            }
        }
    }

    /// The patterns upstream does NOT run as a regex go to frink's
    /// stand-in for the splitter upstream does run, and nothing else
    /// does. Asserted on the dispatch itself because a splitter and the
    /// collapsed regex of the same string agree on almost every text, so
    /// no golden can be relied on to tell them apart. GPT4O, tekken and
    /// jais-2 look like splitter strings and are not.
    #[test]
    fn a_pattern_upstream_has_a_splitter_for_is_not_run_as_a_regex() {
        for p in [GPT2, LLAMA3, QWEN2, QWEN35, NEWLINES] {
            assert!(matches!(custom_splitter(p), Some(Step::Native(_))), "{p}");
        }
        assert!(matches!(
            custom_splitter(TINY_AYA_DIGITS),
            Some(Step::DigitGroups)
        ));
        assert!(matches!(custom_splitter(AFMOE[0]), Some(Step::DigitGroups)));
        assert!(matches!(custom_splitter(KIMI_K2[0]), Some(Step::KimiK2)));
        for patterns in [
            DEEPSEEK_LLM,
            DEEPSEEK_CODER,
            DEEPSEEK3_LLM,
            PORO,
            VIKING,
            SUPERBPE,
            FLINT,
            JAIS2,
            SPARK2_5,
            YOUTU,
            TEKKEN,
            GPT4O,
            BAILINGMOE,
            SEED_CODER,
            UFAKZEKA,
            EXAONE_MOE,
            &CHAMELEON[..5],
            &MINICPM5[..1],
            &DEFAULT[2..],
            &[
                STARCODER[0],
                TINY_AYA[1],
                FALCON[0],
                FALCON[2],
                DEFAULT[0],
                AFMOE[1],
            ],
        ] {
            for p in patterns {
                assert!(custom_splitter(p).is_none(), "{p} has no splitter upstream");
            }
        }
    }
}

/// frink's half of `tools/oracle/regex_split.py`: cuts the corpus the
/// script wrote with every `pre` it names and writes the pieces in the
/// dumper's format. Ignored, because it needs the script's files; the
/// script runs it.
#[cfg(test)]
mod oracle {
    fn unhex(h: &str) -> String {
        let bytes = (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex"))
            .collect();
        String::from_utf8(bytes).expect("the corpus is UTF-8")
    }

    fn hex(s: &str) -> String {
        s.bytes().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    #[ignore = "run by tools/oracle/regex_split.py"]
    fn dump_splits() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
        let corpus: Vec<String> = std::fs::read_to_string(var("FRINK_SPLIT_CORPUS"))
            .expect("corpus")
            .lines()
            .map(unhex)
            .collect();
        let out = std::path::PathBuf::from(var("FRINK_SPLIT_OUT"));
        for pre in var("FRINK_SPLIT_PRES").split(',') {
            let stem = if pre.is_empty() { "_missing" } else { pre };
            match super::pre_split(pre) {
                Err(why) => std::fs::write(out.join(format!("{stem}.refused")), why),
                Ok(arm) => {
                    let mut lines = String::new();
                    for text in &corpus {
                        let pieces: Vec<String> =
                            arm.pretokenizer.split(text).into_iter().map(hex).collect();
                        lines.push_str(&pieces.join(" "));
                        lines.push('\n');
                    }
                    std::fs::write(out.join(format!("{stem}.frink")), lines).and_then(|()| {
                        std::fs::write(
                            out.join(format!("{stem}.ignore_merges")),
                            arm.ignore_merges.to_string(),
                        )
                    })
                }
            }
            .expect("write the dump");
        }
    }
}
