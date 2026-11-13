//! A port of `unicode_regex_split` (`.scratch/llama.cpp/src/unicode.cpp`):
//! text cut by a SEQUENCE of regexes, each one re-splitting the
//! fragments the one before it produced.
//!
//! Its own module because it is a transcription of a foreign function,
//! reviewed against that function, while [`super::pretokenize`] is the
//! table of which `tokenizer.ggml.pre` selects which sequence.
//!
//! # What upstream actually runs
//!
//! For each regex of the sequence, in order, `unicode_regex_split` takes
//! one of three engines, and the choice changes the answer:
//!
//! 1. A hand-written splitter, when the pattern is byte for byte one it
//!    has one for (`unicode_regex_split_custom`). Frink stands in for
//!    those with [`Step::Native`], the same Rust-regex transcription the
//!    single-regex arms have always used.
//! 2. `std::regex` over a COLLAPSED copy of the text, when the pattern
//!    names a Unicode category (`\p{N}`, `\p{L}`, `\p{P}`, `\p{M}`,
//!    `\p{S}`, `\p{Lu}`/`Ll`/`Lt`/`Lm`/`Lo`). Every non-ASCII codepoint
//!    becomes one marker per category, and each `\p{..}` in the pattern
//!    becomes the class "that marker, or the ASCII members of the
//!    category". This is not Unicode matching: `\p{Lu}` and `\p{Ll}`
//!    collapse to the same marker, so case is invisible outside ASCII;
//!    and a codepoint that is in none of the five categories (a
//!    separator that is not whitespace, a control, an unassigned one)
//!    matches no `\p{..}` at all. [`Step::Collapsed`] runs the same
//!    collapse.
//! 3. `std::wregex` over the codepoints, when it names none, with every
//!    non-ASCII whitespace codepoint replaced by U+000B first.
//!    [`Step::Codepoints`].
//!
//! Matching itself is ECMAScript backtracking with leftmost-first
//! alternation, which is also what `fancy_regex` does. The places the
//! two syntaxes mean different things are translated, not hoped away:
//! ECMAScript's `\d` is ASCII (`0-9`) where Rust's is Unicode, so it is
//! rewritten; and the escapes this module does not translate are
//! refused when the pattern is compiled (see [`translate`]).
//!
//! # Fragments
//!
//! Each regex sees one fragment at a time, never the whole text: its
//! lookarounds stop at the fragment's edges, and what it does not match
//! comes out as a fragment of its own (`unicode_regex_split_stl` emits
//! the gaps). A zero-length match yields a zero-length fragment upstream,
//! which tokenizes to nothing; this drops it, which is the same answer.

use super::unicode;
use super::unicode_data as data;

/// One regex of a sequence, compiled for the engine upstream runs it on.
pub(super) enum Step {
    /// A pattern run on the text as is, with Rust's Unicode classes.
    Native(fancy_regex::Regex),
    /// A pattern with `\p{..}` in it, collapsed as upstream collapses it
    /// and run on the collapsed text.
    Collapsed(fancy_regex::Regex),
    /// A pattern with no `\p{..}`, run on the codepoints with non-ASCII
    /// whitespace replaced by U+000B.
    Codepoints(fancy_regex::Regex),
    /// `unicode_regex_split_custom_afmoe`, which upstream also runs for
    /// tiny_aya's `\d{1,3}(?=(?:\d{3})*\b)`. See [`digit_groups`].
    DigitGroups,
}

/// A compiled pre-tokenizer: one or more [`Step`]s, applied in order.
pub(super) struct PreTokenizer {
    steps: Vec<Step>,
}

/// The categories upstream collapses by (`k_ucat_enum`), in the spelling
/// a pattern names them. Every letter sub-category maps to the one
/// `\p{L}` marker.
const CATEGORIES: &[(&str, u8)] = &[
    (r"\p{N}", data::NUMBER),
    (r"\p{L}", data::LETTER),
    (r"\p{P}", data::PUNCTUATION),
    (r"\p{M}", data::ACCENT_MARK),
    (r"\p{S}", data::SYMBOL),
    (r"\p{Lu}", data::LETTER),
    (r"\p{Ll}", data::LETTER),
    (r"\p{Lt}", data::LETTER),
    (r"\p{Lm}", data::LETTER),
    (r"\p{Lo}", data::LETTER),
];

/// The marker a category collapses to (`k_ucat_cpt`, there bytes
/// `0xD1`..`0xD5`; here the codepoints of the same value, which no other
/// collapsed character can be) and the ASCII members of the category
/// (`k_ucat_map`), written as the inside of a Rust character class.
fn collapsed_class(category: u8) -> (char, &'static str) {
    match category {
        data::NUMBER => ('\u{D1}', "0-9"),
        data::LETTER => ('\u{D2}', "A-Za-z"),
        // `!-#%-*,-/:-;?-@\[-\]_\{\}` upstream.
        data::PUNCTUATION => (
            '\u{D3}',
            r"\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\x5B-\x5D\x5F\x7B\x7D",
        ),
        // No codepoint below 128 is a mark.
        data::ACCENT_MARK => ('\u{D4}', ""),
        // `$+<=>^`|~` upstream.
        data::SYMBOL => ('\u{D5}', r"\x24\x2B\x3C-\x3E\x5E\x60\x7C\x7E"),
        _ => unreachable!("only the five collapsed categories have a class"),
    }
}

/// The fallback marker (`0xD0`) for a non-ASCII codepoint that is in
/// none of the five categories and is not whitespace.
const OTHER: char = '\u{D0}';

/// Whether upstream would collapse the text for `pattern`.
fn names_a_category(pattern: &str) -> bool {
    CATEGORIES.iter().any(|(name, _)| pattern.contains(name))
}

/// One codepoint of upstream's `text_collapsed`.
fn collapse_char(c: char) -> char {
    if c.is_ascii() {
        return c;
    }
    if unicode::is_whitespace(c) {
        return '\u{B}';
    }
    match unicode::flags(c).category() {
        cat @ (data::NUMBER
        | data::LETTER
        | data::PUNCTUATION
        | data::ACCENT_MARK
        | data::SYMBOL) => collapsed_class(cat).0,
        _ => OTHER,
    }
}

/// One codepoint of upstream's `wtext`.
fn whitespace_char(c: char) -> char {
    if !c.is_ascii() && unicode::is_whitespace(c) {
        '\u{B}'
    } else {
        c
    }
}

/// Upstream's `regex_expr_collapsed` loop, transcribed: every `\p{..}`
/// it knows becomes its marker plus the category's ASCII members,
/// bracketed unless it is already inside a class. The class tracking is
/// upstream's own (a `[` or `]` counts unless the byte before it is a
/// backslash), kept as is so the two loops cannot disagree.
fn collapse_pattern(pattern: &str) -> String {
    assert!(
        pattern.is_ascii(),
        "upstream throws on a pattern with both a category and non-ASCII: {pattern}"
    );
    let b = pattern.as_bytes();
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut inside = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'[' && (i == 0 || b[i - 1] != b'\\') {
            out.push('[');
            inside = true;
            i += 1;
            continue;
        }
        if inside && c == b']' && b[i - 1] != b'\\' {
            out.push(']');
            inside = false;
            i += 1;
            continue;
        }
        if c == b'\\' && i + 3 < b.len() && b[i + 1] == b'p' && b[i + 2] == b'{' {
            if let Some(close) = pattern[i + 3..].find('}').map(|j| i + 3 + j) {
                if close <= i + 10 {
                    let name = &pattern[i..=close];
                    if let Some(&(_, cat)) = CATEGORIES.iter().find(|(n, _)| *n == name) {
                        let (marker, members) = collapsed_class(cat);
                        if !inside {
                            out.push('[');
                        }
                        out.push(marker);
                        out.push_str(members);
                        if !inside {
                            out.push(']');
                        }
                        i = close + 1;
                        continue;
                    }
                }
            }
        }
        out.push(c as char);
        i += 1;
    }
    out
}

/// ECMAScript to Rust, for the escapes whose meaning differs.
///
/// `\d` is `[0-9]` in ECMAScript under the C locale both `std::regex`
/// and `std::wregex` run in (an Arabic-Indic digit is not one), and
/// Unicode `Nd` in Rust; it is rewritten. `\s` needs nothing: every
/// codepoint Rust's `\s` would add has already become U+000B, which both
/// engines call whitespace. `\D`, `\w`, `\W`, `\b` and `\B` differ too
/// and no sequence uses them, so they are refused rather than left to
/// mean something else here.
fn translate(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len());
    let mut in_class = false;
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                let Some(e) = chars.next() else {
                    out.push(c);
                    break;
                };
                match e {
                    'd' if in_class => out.push_str("0-9"),
                    'd' => out.push_str("[0-9]"),
                    'D' | 'w' | 'W' | 'b' | 'B' => {
                        panic!("`\\{e}` means something else in ECMAScript: {pattern}")
                    }
                    _ => {
                        out.push('\\');
                        out.push(e);
                    }
                }
            }
            '[' if !in_class => {
                in_class = true;
                out.push(c);
            }
            ']' if in_class => {
                in_class = false;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn compile(pattern: &str) -> fancy_regex::Regex {
    fancy_regex::Regex::new(pattern)
        .unwrap_or_else(|e| panic!("pre-tokenization pattern {pattern:?} must compile: {e:?}"))
}

impl PreTokenizer {
    /// One pattern, run on the text as is.
    pub(super) fn native(pattern: &str) -> Self {
        PreTokenizer {
            steps: vec![Step::Native(compile(pattern))],
        }
    }

    /// Upstream's `regex_exprs`, verbatim, each dispatched the way
    /// `unicode_regex_split` dispatches it. `native` maps a pattern
    /// upstream has a hand-written splitter for onto frink's stand-in
    /// for that splitter; it is consulted first, as
    /// `unicode_regex_split_custom` is.
    pub(super) fn sequence(patterns: &[&str], native: impl Fn(&str) -> Option<Step>) -> Self {
        let steps = patterns
            .iter()
            .map(|p| {
                native(p).unwrap_or_else(|| {
                    if names_a_category(p) {
                        Step::Collapsed(compile(&translate(&collapse_pattern(p))))
                    } else {
                        Step::Codepoints(compile(&translate(p)))
                    }
                })
            })
            .collect();
        PreTokenizer { steps }
    }

    /// Cuts `text` into the chunks BPE then merges over. The chunks
    /// concatenate back to `text` exactly; none is empty.
    pub(super) fn split<'t>(&self, text: &'t str) -> Vec<&'t str> {
        let mut fragments = vec![text];
        for step in &self.steps {
            let mut next = Vec::with_capacity(fragments.len());
            for fragment in fragments {
                match step {
                    Step::Native(re) => split_with_gaps(re, fragment, &mut next),
                    Step::Collapsed(re) => split_mapped(re, fragment, collapse_char, &mut next),
                    Step::Codepoints(re) => split_mapped(re, fragment, whitespace_char, &mut next),
                    Step::DigitGroups => digit_groups(fragment, &mut next),
                }
            }
            fragments = next;
        }
        fragments
    }
}

impl Step {
    /// The stand-in for one of upstream's hand-written splitters.
    pub(super) fn native(pattern: &str) -> Self {
        Step::Native(compile(pattern))
    }

    /// `unicode_regex_split_custom_afmoe`.
    pub(super) fn digit_groups() -> Self {
        Step::DigitGroups
    }
}

/// Every match of `re` in `text` and every gap between them, in order,
/// as `unicode_regex_split_stl` emits them; empty pieces are dropped.
///
/// The iteration is `std::regex_iterator`'s for the patterns used here:
/// after a non-empty match the next search starts where it ended, and
/// after an empty one it starts a character later. (After an empty
/// match upstream first retries a NON-empty match at the same place; no
/// pattern here can match both empty and non-empty, so that retry never
/// finds anything and is not carried.) A `fancy_regex` run-time error
/// (its backtrack limit) stops matching and hands back the rest as one
/// piece, so a pathological input degrades the split and never loses
/// text.
fn matches(re: &fancy_regex::Regex, text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos <= text.len() {
        let Ok(Some(m)) = re.find_from_pos(text, pos) else {
            break;
        };
        out.push((m.start(), m.end()));
        pos = if m.end() > m.start() {
            m.end()
        } else {
            match text[m.end()..].chars().next() {
                Some(c) => m.end() + c.len_utf8(),
                None => break,
            }
        };
    }
    out
}

/// Pushes the matches of `re` in `text` and the gaps between them.
pub(super) fn split_with_gaps<'t>(re: &fancy_regex::Regex, text: &'t str, out: &mut Vec<&'t str>) {
    let mut cursor = 0;
    for (start, end) in matches(re, text) {
        if start > cursor {
            out.push(&text[cursor..start]);
        }
        if end > start {
            out.push(&text[start..end]);
        }
        cursor = end;
    }
    if cursor < text.len() {
        out.push(&text[cursor..]);
    }
}

/// [`split_with_gaps`] on a codepoint-for-codepoint image of `text`
/// (`map` sends each codepoint to the one upstream matches against), with
/// the cuts carried back to `text`.
fn split_mapped<'t>(
    re: &fancy_regex::Regex,
    text: &'t str,
    map: fn(char) -> char,
    out: &mut Vec<&'t str>,
) {
    // `starts[k]` is where codepoint k begins in `text`, `image_starts[k]`
    // where its image begins in `image`; both carry the end as a last row.
    let mut image = String::with_capacity(text.len());
    let mut starts = Vec::with_capacity(text.len() + 1);
    let mut image_starts = Vec::with_capacity(text.len() + 1);
    for (at, c) in text.char_indices() {
        starts.push(at);
        image_starts.push(image.len());
        image.push(map(c));
    }
    starts.push(text.len());
    image_starts.push(image.len());

    let mut pieces = Vec::new();
    split_with_gaps(re, &image, &mut pieces);
    for piece in pieces {
        // `piece` is a subslice of `image`, so its offset is exact.
        let from = piece.as_ptr() as usize - image.as_ptr() as usize;
        let to = from + piece.len();
        let k0 = image_starts.partition_point(|&s| s < from);
        let k1 = image_starts.partition_point(|&s| s < to);
        out.push(&text[starts[k0]..starts[k1]]);
    }
}

/// `unicode_regex_split_custom_afmoe`, transcribed, including what it
/// does that the regex it stands in for would not.
///
/// A run of `\p{N}` (upstream's own table, so `²` and `٣` count) is cut
/// into groups of three from the right, the leading 1 or 2 first. The
/// text BEFORE a run is not emitted on its own: the first group is cut
/// from the end of the previous piece, so `ab1234` comes out as `ab1`,
/// `234`. That is upstream's behaviour, not the regex's, and the later
/// regex of the sequence is what separates `ab` from `1` again.
fn digit_groups<'t>(text: &'t str, out: &mut Vec<&'t str>) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let at = |k: usize| chars.get(k).map_or(text.len(), |&(b, _)| b);
    let mut prev_end = 0;
    let mut cut = |end: usize, out: &mut Vec<&'t str>| {
        if end > prev_end {
            out.push(&text[at(prev_end)..at(end)]);
        }
        prev_end = end;
    };
    let mut pos = 0;
    while pos < chars.len() {
        if !unicode::flags(chars[pos].1).is_number() {
            pos += 1;
            continue;
        }
        let run_start = pos;
        while pos < chars.len() && unicode::flags(chars[pos].1).is_number() {
            pos += 1;
        }
        let run = pos - run_start;
        let mut current = run_start;
        if run % 3 > 0 {
            current += run % 3;
            cut(current, out);
        }
        while current < run_start + run {
            current += 3;
            cut(current, out);
        }
    }
    cut(chars.len(), out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(p: &PreTokenizer, text: &str) -> Vec<String> {
        p.split(text).into_iter().map(str::to_string).collect()
    }

    /// The collapse loop's output for the patterns it has to get right:
    /// a category outside a class gets brackets, one inside does not, the
    /// letter sub-categories share `\p{L}`'s marker, and an escaped `[`
    /// or `]` does not open or close a class.
    #[test]
    fn categories_collapse_as_upstream_collapses_them() {
        assert_eq!(collapse_pattern(r"\p{N}{1,3}"), "[\u{D1}0-9]{1,3}");
        assert_eq!(
            collapse_pattern(r"[^\r\n\p{L}\p{N}]?"),
            "[^\\r\\n\u{D2}A-Za-z\u{D1}0-9]?"
        );
        assert_eq!(collapse_pattern(r"[\p{Lu}\p{M}]+"), "[\u{D2}A-Za-z\u{D4}]+");
        assert_eq!(collapse_pattern(r"\[\p{M}\]"), "\\[[\u{D4}]\\]");
        assert!(names_a_category(r"\s?\p{Lo}"));
        assert!(!names_a_category(r"[一-龥]+|\s+$"));
    }

    /// The five categories and the two fallbacks, codepoint by codepoint.
    #[test]
    fn a_codepoint_collapses_to_its_category_marker() {
        assert_eq!(collapse_char('a'), 'a', "ASCII is kept");
        assert_eq!(collapse_char('é'), '\u{D2}');
        assert_eq!(collapse_char('٣'), '\u{D1}');
        assert_eq!(collapse_char('。'), '\u{D3}');
        assert_eq!(collapse_char('\u{301}'), '\u{D4}');
        assert_eq!(collapse_char('😀'), '\u{D5}');
        assert_eq!(collapse_char('\u{a0}'), '\u{B}', "whitespace first");
        assert_eq!(
            collapse_char('\u{200b}'),
            OTHER,
            "a format char is no category"
        );
        assert_eq!(whitespace_char('\u{3000}'), '\u{B}');
        assert_eq!(whitespace_char('é'), 'é');
    }

    #[test]
    fn ecmascript_digits_are_ascii() {
        assert_eq!(translate(r"(?=(\d{3})+(?!\d))"), "(?=([0-9]{3})+(?![0-9]))");
        assert_eq!(translate(r"[\d,]"), "[0-9,]");
        assert_eq!(translate(r"\\d"), r"\\d", "an escaped backslash is not \\d");
    }

    #[test]
    #[should_panic(expected = "means something else in ECMAScript")]
    fn an_untranslated_escape_is_refused() {
        translate(r"\w+");
    }

    /// `ab1234` is `ab1`, `234`: the text before a digit run rides on
    /// its first group, which is what upstream's splitter does.
    #[test]
    fn digit_groups_cut_from_the_right_and_carry_the_prefix() {
        let p = PreTokenizer {
            steps: vec![Step::DigitGroups],
        };
        assert_eq!(split(&p, "ab1234"), ["ab1", "234"]);
        assert_eq!(split(&p, "123456 x"), ["123", "456", " x"]);
        assert_eq!(split(&p, "1234567"), ["1", "234", "567"]);
        assert_eq!(split(&p, "x²³⁴⁵"), ["x²", "³⁴⁵"]);
        assert_eq!(split(&p, "none"), ["none"]);
    }

    /// A zero-length match is a cut, not a piece, and the search moves
    /// on past it rather than matching there forever.
    #[test]
    fn a_lookahead_only_pattern_cuts_without_emitting_empty_pieces() {
        let p = PreTokenizer::sequence(&[r"(?=(\d{3})+(?!\d))"], |_| None);
        assert_eq!(split(&p, "1234567"), ["1", "234", "567"]);
        assert_eq!(split(&p, "123456"), ["123", "456"]);
    }

    /// The cuts made on the image land on the same codepoints in the
    /// text, multi-byte ones included.
    #[test]
    fn cuts_on_the_collapsed_image_map_back_to_the_text() {
        let p = PreTokenizer::sequence(&[r"\p{L}+"], |_| None);
        assert_eq!(
            split(&p, "日本 語x\u{a0}é"),
            ["日本", " ", "語x", "\u{a0}", "é"]
        );
    }
}
