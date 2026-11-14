//! `unicode_regex_split_custom_kimi_k2` (`unicode.cpp` at the pin),
//! transcribed: the splitter upstream runs for `tokenizer.ggml.pre =
//! "kimi-k2"`, whose `regex_exprs` is the single trigger string
//! `\p{Han}+`.
//!
//! A port of the hand-written loop, not of the regex in the comment
//! above it upstream, because the two differ and libllama runs the
//! loop: a letter run here is `is_letter` and not Han, with no case
//! split and without `\p{M}`; a contraction is taken only after
//! letters, compared through `unicode_tolower`; and digits group in
//! threes. Its own module because it is reviewed against that one
//! function.

use super::unicode;

/// `unicode_cpt_is_han`.
fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0x2CEB0..=0x2EBEF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F)
}

/// The fields of upstream's `unicode_cpt_flags` the loop reads, for
/// one position of the fragment. `in_range` stands for `as_uint() != 0`:
/// every codepoint's flags carry at least one category bit (`UNDEFINED`
/// for an unassigned one), so only a position past the fragment's end
/// reads as zero.
#[derive(Clone, Copy, Default)]
struct Flags {
    in_range: bool,
    letter: bool,
    number: bool,
    whitespace: bool,
}

/// Pushes the pieces of `text`, which is one fragment (upstream's
/// `offsets` entry).
pub(super) fn split<'t>(text: &'t str, out: &mut Vec<&'t str>) {
    let cpts: Vec<(usize, char)> = text.char_indices().collect();
    let n = cpts.len();
    let byte_at = |k: usize| cpts.get(k).map_or(text.len(), |&(b, _)| b);
    let cpt = |k: usize| cpts.get(k).map(|&(_, c)| c);
    let flags = |k: usize| match cpt(k) {
        Some(c) => {
            let f = unicode::flags(c);
            Flags {
                in_range: true,
                letter: f.is_letter(),
                number: f.is_number(),
                whitespace: unicode::is_whitespace(c),
            }
        }
        None => Flags::default(),
    };
    let is_han_at = |k: usize| cpt(k).is_some_and(is_han);
    let is_cr_lf = |k: usize| matches!(cpt(k), Some('\r' | '\n'));
    let letter_not_han = |k: usize| flags(k).letter && !is_han_at(k);
    let lower = |k: usize| cpt(k).map(unicode::to_lower);

    let mut prev_end = 0;
    let mut add = |end: usize, out: &mut Vec<&'t str>| {
        if end > prev_end {
            out.push(&text[byte_at(prev_end)..byte_at(end)]);
        }
        prev_end = end;
    };

    let mut pos = 0;
    while pos < n {
        let f = flags(pos);

        // `[\p{Han}]+`
        if is_han_at(pos) {
            while is_han_at(pos) {
                pos += 1;
            }
            add(pos, out);
            continue;
        }

        // The letter run with an optional leading non-letter, non-digit.
        let can_lead = !(is_cr_lf(pos) || f.letter || f.number);
        if letter_not_han(pos) || (can_lead && letter_not_han(pos + 1)) {
            let has_leading = can_lead;
            if has_leading {
                pos += 1;
            }
            let mut has_letters = false;
            while letter_not_han(pos) {
                has_letters = true;
                pos += 1;
            }
            if has_letters || (!has_leading && letter_not_han(pos)) {
                if !has_letters {
                    pos += 1;
                }
                while letter_not_han(pos) {
                    pos += 1;
                }
                if cpt(pos) == Some('\'') && pos + 1 < n {
                    let next = lower(pos + 1);
                    if matches!(next, Some('s' | 't' | 'm' | 'd')) {
                        pos += 2;
                    } else if pos + 2 < n {
                        let next2 = lower(pos + 2);
                        if matches!(
                            (next, next2),
                            (Some('r'), Some('e'))
                                | (Some('v'), Some('e'))
                                | (Some('l'), Some('l'))
                        ) {
                            pos += 3;
                        }
                    }
                }
                add(pos, out);
                continue;
            } else if has_leading {
                pos -= 1;
            }
        }

        // `\p{N}{1,3}`
        if f.number {
            let mut ini = pos;
            while flags(pos).number {
                pos += 1;
                if pos - ini >= 3 {
                    add(pos, out);
                    ini = pos;
                }
            }
            add(pos, out);
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let mut f2 = if cpt(pos) == Some(' ') {
            flags(pos + 1)
        } else {
            f
        };
        let not_word = |g: Flags| !(g.whitespace || g.letter || g.number) && g.in_range;
        if not_word(f2) {
            if cpt(pos) == Some(' ') {
                pos += 1;
            }
            while not_word(f2) {
                pos += 1;
                f2 = flags(pos);
            }
            while is_cr_lf(pos) {
                pos += 1;
            }
            add(pos, out);
            continue;
        }

        let mut num_ws = 0;
        let mut last_end_r_or_n = 0;
        while flags(pos + num_ws).whitespace {
            if is_cr_lf(pos + num_ws) {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // `\s*[\r\n]+`
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            add(pos, out);
            continue;
        }
        // `\s+(?!\S)`
        if num_ws > 1 && cpt(pos + num_ws).is_some() {
            pos += num_ws - 1;
            add(pos, out);
            continue;
        }
        // `\s+`
        if num_ws > 0 {
            pos += num_ws;
            add(pos, out);
            continue;
        }

        pos += 1;
        add(pos, out);
    }
}

#[cfg(test)]
mod tests {
    fn split(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        super::split(text, &mut out);
        out
    }

    /// The places the loop is not the regex upstream quotes beside it:
    /// no case split (`CamelCase` is one piece), a combining mark is not
    /// a letter, Han is carved out of a letter run, and a contraction
    /// folds case through `unicode_tolower`. Each expected cut is
    /// libllama's (`tools/oracle/regex_split.cpp` with `\p{Han}+`).
    #[test]
    fn the_loop_is_not_the_regex_in_its_comment() {
        assert_eq!(split("CamelCase"), ["CamelCase"]);
        assert_eq!(split("e\u{301}te"), ["e", "\u{301}te"]);
        assert_eq!(split("ab中文cd"), ["ab", "中文", "cd"]);
        assert_eq!(split("DON'T we'LL I'd"), ["DON'T", " we'LL", " I'd"]);
        assert_eq!(split("1234567"), ["123", "456", "7"]);
        assert_eq!(split("a  \n\n b  "), ["a", "  \n\n", " b", "  "]);
    }

    #[test]
    fn pieces_concatenate_to_the_text() {
        let text = "日本語 Hello, 世界! 123 x\u{a0}y\t\r\n e\u{301}  ";
        assert_eq!(split(text).concat(), text);
    }
}
