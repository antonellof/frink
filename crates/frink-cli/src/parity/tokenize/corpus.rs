//! The corpus `frink parity` tokenizes on both engines.
//!
//! Its own module because the corpus IS the coverage claim: what these
//! twenty-two strings contain decides which tokenizer bugs the oracle
//! next door can see, and that is a separate thing to review from the
//! comparison machinery. The test at the bottom asserts the claim, so
//! trimming the corpus to "the cases that pass" goes red instead of
//! quietly narrowing the instrument.

/// One corpus entry. `why` is printed only when the case diverges, so
/// the reader is told which pre-tokenizer clause they are looking at
/// instead of having to reverse-engineer it from the bytes.
pub(super) struct Case {
    pub name: &'static str,
    pub why: &'static str,
    pub text: &'static str,
}

/// The corpus is chosen against the *clauses that differ between
/// llama.cpp's pre-tokenizer regexes*, not against prose. Ordinary
/// English is exactly what the old single-regex implementation got
/// right, which is why running it for years proved nothing.
///
/// Every entry names the clause it exercises. The first six would each
/// have failed on their own before `pretokenize_regex_for` existed.
pub(super) const CORPUS: &[Case] = &[
    Case {
        name: "digit-runs",
        why: "llama3/qwen2 group digits (\\p{N}{1,3} / \\p{N}); gpt2 takes ?\\p{N}+ whole. \
              Any run of 4+ digits splits differently.",
        text: "Build 1234567 of 89 took 100000 ms on port 8080 with seed 42.",
    },
    Case {
        name: "digit-boundaries",
        why: "walks the 1..10 digit-run lengths across the 3-digit grouping boundary.",
        text: "1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890",
    },
    Case {
        name: "double-space",
        why: "\\s+(?!\\S) separates an interior whitespace run from a trailing one; \
              without the lookahead every multi-space run splits differently.",
        text: "one  two   three    four     five",
    },
    Case {
        name: "indented-code",
        why: "4- and 8-space indents are the most common multi-space runs in real input.",
        text: "def f(x):\n    if x > 10:\n        return x * 2\n    return 0\n",
    },
    Case {
        name: "blank-lines",
        why: "\\s*[\\r\\n]+ vs a generic whitespace run: blank lines and the indent that \
              follows them are one chunk or two depending on the arm.",
        text: "para one\n\npara two\n\n\n    indented after two blanks\n",
    },
    Case {
        name: "trailing-space",
        why: "the negative lookahead exists exactly for whitespace at end of input; the \
              gpt2 arm frink used to run for everything had no trailing \\s+ clause at all.",
        text: "wrapped up   ",
    },
    Case {
        name: "tabs",
        why: "tab-indented code, plus a tab/space mixture inside a line.",
        text: "\tif x:\n\t\treturn 1\n\t \treturn 2\n",
    },
    Case {
        name: "contractions-upper",
        why: "the llama3 arm makes contractions case-insensitive with (?i:); the qwen2 and \
              gpt2 arms do not. Uppercase contractions are the only input that shows it.",
        text: "DON'T SAY IT'S HERS. I'LL GO. THEY'VE WON. WE'RE READY. HE'D KNOWN.",
    },
    Case {
        name: "contractions-mixed",
        why: "mixed case and stacked contractions, where the alternation order decides the \
              split.",
        text: "It's Bob's, isn't it? You'd've said so. O'Neill's team we're sure They'Ve gone.",
    },
    Case {
        name: "cjk",
        why: "\\p{L} runs with no spaces: the letter clause has to carry the whole line, and \
              byte-fallback lands differently if it does not.",
        text: "日本語のテキスト、中文字符和한국어が混在。城市名：東京、北京、서울。",
    },
    Case {
        name: "emoji",
        why: "astral-plane symbols, a ZWJ family sequence and a regional-indicator flag: \
              multi-codepoint graphemes that no clause treats as letters.",
        text: "ship it 🚀 done ✅ 👨‍👩‍👧‍👦 🇯🇵 café e\u{0301}clair",
    },
    Case {
        name: "punct-runs",
        why: "?[^\\s\\p{L}\\p{N}]+[\\r\\n]* — a punctuation run may swallow following \
              newlines in some arms and not others.",
        text: "-- --- **bold** `code` <<x>> !!!??? ... ->|<- ;;;\n\n",
    },
    Case {
        name: "alnum-mixtures",
        why: "digits welded to letters and dots: version strings, addresses, hex literals.",
        text: "Qwen2.5-1.5B v0.13.3 IPv4 192.168.0.1:8443 0xDEADBEEF sha256:9f86d0",
    },
    Case {
        name: "crlf",
        why: "\\r\\n is two whitespace characters, and \\s*[\\r\\n]+ vs \\s+ disagree about \
              where the run starts.",
        text: "line one\r\nline two\r\n\r\n    line three\r\n",
    },
    Case {
        name: "leading-whitespace",
        why: "whitespace before any content, which the ?\\p{L}+ clause may or may not absorb.",
        text: "   leading spaces, then text.",
    },
    Case {
        name: "unicode-space",
        why: "\\s in these patterns is Unicode-aware: NBSP, ideographic space and thin space \
              are whitespace to the regex but not to a naive byte split.",
        text: "a\u{a0}b\u{3000}c\u{2009}d\u{200b}e",
    },
    Case {
        name: "control-bytes",
        why: "ANSI escapes and a bare form feed: bytes that the GPT-2 byte remap moves out of \
              their own code range.",
        text: "a\u{1b}[31mred\u{1b}[0m b\u{c}c\u{7}d",
    },
    Case {
        name: "long-words",
        why: "long single-token-space runs, where a merge-order bug shows up as one extra \
              split rather than as different text.",
        text: "supercalifragilisticexpialidocious antidisestablishmentarianism \
               pneumonoultramicroscopicsilicovolcanoconiosis",
    },
    Case {
        name: "single-space",
        why: "a whole input that is one whitespace character: the degenerate case of every \
              whitespace clause.",
        text: " ",
    },
    Case {
        name: "sequence-scripts",
        why: "the regex-SEQUENCE arms (deepseek-*, starcoder, poro, viking, tiny_aya, superbpe) \
              cut by script and category one pass at a time: non-Latin letters, non-ASCII \
              digits, fullwidth and CJK punctuation, digits glued to letters, combining marks.",
        text: "\u{395}\u{3bb}\u{3bb}\u{3b7}\u{3bd}\u{3b9}\u{3ba}\u{3ac} \
               \u{41a}\u{438}\u{440}\u{438}\u{43b}\u{43b}\u{438}\u{446}\u{430} ab1234cd \
               \u{661}\u{662}\u{663}\u{664}\u{665} x\u{b2}\u{b3} \u{ff21}\u{ff22}\u{ff01}\u{ff1f} \
               \u{4e2d}\u{6587}123\u{4e2d}\u{6587} \u{936}\u{92c}\u{94d}\u{926}\u{964} \
               e\u{301}t\u{e9} (x) [y] \"z\" a.b,c\u{2026}d\u{3002}e\u{3001}f",
    },
    Case {
        name: "special-markers-as-prose",
        why: "special-token markers MENTIONED in text. Under parse_special=false llama.cpp \
              tokenizes a CONTROL marker as the characters it is written with; frink used \
              to parse it unconditionally, and promoted any NORMAL entry shaped like `<...>` \
              (Qwen2.5's `<s>`) to special under BOTH settings. Off by one token per mention \
              on any document that discusses tokenizers.",
        text: "Mistral wraps a turn in `<s>[INST] ... [/INST]` and ends it with `</s>`; \
               Llama-3 opens with `<|begin_of_text|>` and ends a turn with `<|eot_id|>`; \
               ChatML puts `<|im_start|>user` before and `<|im_end|>` after; gemma uses \
               `<bos>`, `<start_of_turn>`, `<end_of_turn>` and `<eos>`; Qwen's pad is \
               `<|endoftext|>`, Phi's is `<|end|>`, and BERT brackets with `[CLS]` and \
               `[SEP]`. An `<unk>` or `<pad>` may also appear.",
    },
    Case {
        name: "special-markers-glued",
        why: "markers at the very start of the input, glued to text, and followed by a \
              newline: the shape a rendered chat template has. Exercises the SPM dummy-prefix \
              rule after a special (`add_space_prefix && is_prev_special`) as well as which \
              entries are special at all.",
        text: "<s>hello</s><|im_start|>user\nhi<|im_end|>\n<start_of_turn>model\n\
               <|begin_of_text|>[CLS] a [SEP]<|endoftext|>",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The text of the named case, or a failure that says the case is
    /// gone. Every claim below is pinned to the case that carries it:
    /// asserting only over the concatenated corpus would let a case be
    /// deleted as long as some other case happened to contain the same
    /// character, which is how a coverage claim quietly shrinks.
    fn case(name: &str) -> &'static str {
        CORPUS
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("corpus case '{name}' was removed"))
            .text
    }

    fn longest_digit_run(s: &str) -> usize {
        s.split(|c: char| !c.is_ascii_digit())
            .map(str::len)
            .max()
            .unwrap_or(0)
    }

    /// The corpus is the actual coverage claim, so it is asserted rather
    /// than trusted. Trimming it to "the ones that pass" would restore
    /// the hole with the instrument still reporting MATCH.
    #[test]
    fn the_corpus_covers_the_inputs_that_pre_tokenizers_disagree_on() {
        // A run of 4+ digits is the exact input the llama3/qwen2/gpt-4o
        // grouping splits and the single hardcoded regex did not.
        assert!(
            longest_digit_run(case("digit-runs")) >= 4,
            "digit-runs no longer holds a run of 4+ digits"
        );
        assert!(
            longest_digit_run(case("digit-boundaries")) >= 10,
            "digit-boundaries must walk past the 3-digit grouping boundary"
        );

        assert!(case("double-space").contains("  "), "two spaces");
        assert!(case("indented-code").contains("\n    "), "a 4-space indent");
        assert!(
            case("indented-code").contains("\n        "),
            "an 8-space indent"
        );
        assert!(case("blank-lines").contains("\n\n"), "a blank line");
        assert!(case("crlf").contains("\r\n\r\n"), "a blank CRLF line");
        assert!(case("tabs").contains("\t\t"), "a double tab indent");
        assert!(
            case("trailing-space").ends_with(' '),
            "trailing whitespace, which is what \\s+(?!\\S) is for"
        );
        assert!(
            case("cjk")
                .chars()
                .any(|c| ('\u{4e00}'..'\u{9fff}').contains(&c)),
            "CJK ideographs"
        );
        assert!(
            case("emoji").contains('\u{1f680}') && case("emoji").contains('\u{1f1ef}'),
            "an astral-plane emoji and a regional-indicator flag"
        );
        assert!(
            case("contractions-upper").contains("DON'T"),
            "an uppercase contraction"
        );
        assert!(
            case("contractions-mixed").contains("It's"),
            "a mixed-case contraction"
        );
        assert!(
            case("unicode-space").contains('\u{a0}'),
            "a non-breaking space"
        );
        assert!(
            case("control-bytes").contains('\u{1b}'),
            "an ANSI escape byte"
        );
        assert!(
            case("leading-whitespace").starts_with("   "),
            "whitespace before any content"
        );
        assert!(
            case("single-space") == " ",
            "the degenerate whitespace case"
        );
        assert!(
            case("punct-runs").contains("!!!"),
            "a run of punctuation characters"
        );
        assert!(
            longest_digit_run(case("alnum-mixtures")) >= 3,
            "digits welded to letters and dots"
        );
        // One marker per tokenizer family the sweep covers, so that on
        // every local checkpoint at least one of them is a real special
        // entry and the parse_special=false run has something to keep
        // as prose.
        for marker in [
            "<s>",
            "</s>",
            "<|begin_of_text|>",
            "<|eot_id|>",
            "<|im_start|>",
            "<|im_end|>",
            "<bos>",
            "<eos>",
            "<end_of_turn>",
            "<|endoftext|>",
            "<|end|>",
            "[CLS]",
            "[SEP]",
            "<unk>",
        ] {
            assert!(
                case("special-markers-as-prose").contains(marker),
                "the prose case must mention {marker}"
            );
        }
        // The sequence arms cut by category, so the case needs a
        // non-ASCII letter, a non-ASCII digit and a non-ASCII
        // punctuation mark, and a digit glued to a letter.
        let seq = case("sequence-scripts");
        assert!(seq.contains('\u{3bb}'), "a Greek letter");
        assert!(seq.contains('\u{661}'), "an Arabic-Indic digit");
        assert!(seq.contains('\u{ff01}'), "fullwidth punctuation");
        assert!(seq.contains("ab1234cd"), "digits glued to letters");
        assert!(
            case("special-markers-glued").starts_with("<s>"),
            "a marker at the very start of the input"
        );
        assert!(
            case("special-markers-glued").contains("<|im_end|>\n"),
            "a marker followed by a newline, as a chat template renders it"
        );

        // Names are printed in the report and looked up above, so they
        // have to be unique to be useful.
        let mut names: Vec<&str> = CORPUS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let n = names.len();
        names.dedup();
        assert_eq!(names.len(), n, "corpus case names must be unique");

        // Every case explains itself, because the `why` line is what a
        // reader gets when the case fails on their checkpoint.
        for c in CORPUS {
            assert!(!c.text.is_empty(), "case '{}' has no text", c.name);
            assert!(c.why.len() > 20, "case '{}' does not say why", c.name);
        }
    }
}
