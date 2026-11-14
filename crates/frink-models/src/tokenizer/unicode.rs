//! Codepoint classification, simple lowercasing and NFD base folding,
//! over the tables in [`unicode_data`].
//!
//! A port of the four `unicode.cpp` functions the WordPiece normalizer
//! calls: `unicode_cpt_flags_from_cpt`, the `is_whitespace` bit
//! `unicode_cpt_flags_array` ORs on afterwards, `unicode_tolower` and
//! `unicode_cpts_normalize_nfd`. Its own module rather than a section of
//! [`super::wordpiece`] because it is a transcription of a foreign
//! source file plus that file's data, which is a different thing to
//! review from a tokenizer loop.
//!
//! Everything here takes and returns `char`. Upstream works in `u32`
//! codepoints and therefore has to carry the surrogate range; a Rust
//! `char` cannot be a surrogate, so those rows are simply never hit.

use super::unicode_data as data;

/// The category bits of upstream's `unicode_cpt_flags` for one
/// codepoint: the low byte, `MASK_CATEGORIES`. The whitespace, case and
/// NFD bits above it are not carried; whitespace has its own lookup,
/// [`is_whitespace`], as upstream ORs it on separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CptFlags(u8);

impl CptFlags {
    /// Upstream's `category_flag()`: every category bit at once, so a
    /// caller can ask "exactly `\p{N}`" the way `unicode_regex_split`
    /// does with its `k_ucat_cpt.find(flags.category_flag())`.
    pub(super) fn category(self) -> u8 {
        self.0
    }
    pub(super) fn is_number(self) -> bool {
        self.0 & data::NUMBER != 0
    }
    pub(super) fn is_letter(self) -> bool {
        self.0 & data::LETTER != 0
    }
    pub(super) fn is_separator(self) -> bool {
        self.0 & data::SEPARATOR != 0
    }
    pub(super) fn is_accent_mark(self) -> bool {
        self.0 & data::ACCENT_MARK != 0
    }
    pub(super) fn is_punctuation(self) -> bool {
        self.0 & data::PUNCTUATION != 0
    }
    pub(super) fn is_symbol(self) -> bool {
        self.0 & data::SYMBOL != 0
    }
    pub(super) fn is_control(self) -> bool {
        self.0 & data::CONTROL != 0
    }
}

/// The category flags for `c`.
///
/// `RANGES_FLAGS` is `(start, flags)` with each row running to the next
/// row's `start` minus one, so this is "the last row whose start is
/// `<= c`". `partition_point` is that search written without an
/// off-by-one: it returns the count of rows strictly before the answer.
/// The table's first row starts at 0, so the count is never zero and
/// the subtraction cannot underflow.
pub(super) fn flags(c: char) -> CptFlags {
    let cpt = c as u32;
    let i = data::RANGES_FLAGS.partition_point(|&(start, _)| start <= cpt);
    debug_assert!(i > 0, "RANGES_FLAGS must start at codepoint 0");
    CptFlags(data::RANGES_FLAGS[i - 1].1)
}

/// The Unicode `White_Space` property.
///
/// Kept as its own lookup rather than folded into [`flags`] because
/// upstream keeps it separate too: whitespace is ORed onto the category
/// flags after the fact, so a codepoint can be both `\p{Z}` and
/// whitespace (U+0020) or whitespace without being `\p{Z}` (U+000A, a
/// control).
pub(super) fn is_whitespace(c: char) -> bool {
    data::WHITESPACE.binary_search(&(c as u32)).is_ok()
}

/// UnicodeData's *simple* lowercase mapping, or `c` unchanged.
///
/// Not `char::to_lowercase`. See [`unicode_data`](super::unicode_data)
/// for the two places they disagree and why the reference's answer is
/// the one this needs.
pub(super) fn to_lower(c: char) -> char {
    let cpt = c as u32;
    match data::LOWERCASE.binary_search_by_key(&cpt, |&(from, _)| from) {
        Ok(i) => char::from_u32(data::LOWERCASE[i].1).unwrap_or(c),
        Err(_) => c,
    }
}

/// The base character of `c`'s NFD decomposition, or `c` when it has
/// none.
///
/// This drops the combining marks rather than emitting them, which is
/// what upstream's table encodes and what accent stripping wants. So
/// `é` (U+00E9) comes back as `e`, and the acute never exists as a
/// separate codepoint to be filtered out afterwards. A combining mark
/// that was already standalone in the input is untouched here and is
/// dropped by the normalizer's `is_accent_mark` test instead.
pub(super) fn nfd_base(c: char) -> char {
    let cpt = c as u32;
    let i = data::NFD.partition_point(|&(start, _, _)| start <= cpt);
    if i == 0 {
        return c;
    }
    let (start, last, base) = data::NFD[i - 1];
    if start <= cpt && cpt <= last {
        char::from_u32(base).unwrap_or(c)
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_classification_matches_the_categories() {
        assert!(flags('.').is_punctuation());
        assert!(flags(',').is_punctuation());
        assert!(flags('-').is_punctuation());
        assert!(flags('\'').is_punctuation());
        assert!(flags('$').is_symbol());
        assert!(flags('+').is_symbol());
        assert!(flags('=').is_symbol());
        assert!(flags('|').is_symbol());
        assert!(flags('~').is_symbol());
        assert!(flags('\u{1b}').is_control(), "ESC is Cc");
        assert!(flags('\u{7}').is_control(), "BEL is Cc");
        assert!(flags(' ').is_separator());

        // Letters and digits carry none of the five bits this table
        // keeps, which is what makes them the "append to the word" case.
        for c in ['a', 'Z', '7', 'é', '日'] {
            let f = flags(c);
            assert!(
                !f.is_punctuation() && !f.is_symbol() && !f.is_control() && !f.is_separator(),
                "{c:?} should carry no category bit here"
            );
        }
    }

    /// The format/private-use half of `\p{C}`. `char::is_control` in std
    /// is `Cc` only, so a normalizer built on std would keep U+200B and
    /// U+FEFF in the text and tokenize them.
    #[test]
    fn control_covers_cf_and_co_not_just_cc() {
        assert!(flags('\u{200b}').is_control(), "ZWSP is Cf");
        assert!(flags('\u{feff}').is_control(), "BOM is Cf");
        assert!(flags('\u{00ad}').is_control(), "soft hyphen is Cf");
        assert!(flags('\u{e000}').is_control(), "private use is Co");
        assert!(
            !flags('\u{0378}').is_control(),
            "an unassigned codepoint is Cn, which is not a control"
        );
    }

    #[test]
    fn accent_marks_are_marks_and_letters_are_not() {
        assert!(flags('\u{0301}').is_accent_mark(), "combining acute is Mn");
        assert!(flags('\u{0323}').is_accent_mark(), "combining dot below");
        assert!(!flags('e').is_accent_mark());
    }

    /// The full `White_Space` property, not `c.is_ascii_whitespace()`.
    /// U+00A0 and U+3000 appear in the parity corpus for exactly this
    /// reason, and U+200B looks like a space and is not one.
    #[test]
    fn whitespace_is_the_unicode_property() {
        for c in [' ', '\t', '\n', '\r', '\u{b}', '\u{c}', '\u{85}'] {
            assert!(is_whitespace(c), "{c:?} is White_Space");
        }
        for c in [
            '\u{a0}', '\u{1680}', '\u{2000}', '\u{200a}', '\u{2009}', '\u{3000}',
        ] {
            assert!(is_whitespace(c), "{c:?} is White_Space");
        }
        assert!(!is_whitespace('\u{200b}'), "ZWSP is Cf, not White_Space");
        assert!(!is_whitespace('a'));
    }

    #[test]
    fn simple_lowercase_folds_the_cases_std_would_expand() {
        assert_eq!(to_lower('A'), 'a');
        assert_eq!(to_lower('a'), 'a');
        assert_eq!(to_lower('É'), 'é');
        assert_eq!(to_lower('Д'), 'д');
        assert_eq!(to_lower('日'), '日');
        // The one codepoint where std's full lowercasing produces two
        // characters and the reference's simple mapping produces one.
        assert_eq!(to_lower('\u{130}'), 'i');
        assert_eq!('\u{130}'.to_lowercase().count(), 2, "std disagrees here");
    }

    #[test]
    fn nfd_folds_a_precomposed_char_to_its_base() {
        assert_eq!(nfd_base('é'), 'e');
        assert_eq!(nfd_base('É'), 'E');
        assert_eq!(nfd_base('ñ'), 'n');
        assert_eq!(nfd_base('\u{1e69}'), 's', "double-decomposing ṩ");
        assert_eq!(nfd_base('e'), 'e', "no decomposition, unchanged");
        assert_eq!(nfd_base('日'), '日');
        assert_eq!(
            nfd_base('\u{0301}'),
            '\u{0301}',
            "a standalone mark is not decomposed here"
        );
    }

    /// The tables are searched with binary search and
    /// `partition_point`, both of which return a wrong answer rather
    /// than an error on unsorted input. Assert the order once.
    #[test]
    fn the_generated_tables_are_sorted() {
        assert!(
            data::RANGES_FLAGS.windows(2).all(|w| w[0].0 < w[1].0),
            "RANGES_FLAGS must ascend by start"
        );
        assert_eq!(data::RANGES_FLAGS[0].0, 0, "must cover codepoint 0");
        assert!(
            data::WHITESPACE.windows(2).all(|w| w[0] < w[1]),
            "WHITESPACE must ascend"
        );
        assert!(
            data::LOWERCASE.windows(2).all(|w| w[0].0 < w[1].0),
            "LOWERCASE must ascend by codepoint"
        );
        assert!(
            data::NFD.windows(2).all(|w| w[0].1 < w[1].0),
            "NFD ranges must ascend and not overlap"
        );
        assert!(
            data::NFD.iter().all(|&(start, last, _)| start <= last),
            "every NFD range must be non-empty"
        );
    }
}
