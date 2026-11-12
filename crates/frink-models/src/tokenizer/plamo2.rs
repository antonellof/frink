//! PLaMo-2's tokenizer (`tokenizer.ggml.model == "plamo2"`), a port of
//! llama.cpp's `llm_tokenizer_plamo2` (`src/llama-vocab.cpp:1351-1616`).
//!
//! It is a scored-vocabulary segmenter like Unigram, but the search
//! runs BACKWARDS over the text on a table of every token suffix: the
//! vocabulary's tokens and all their proper suffixes are sorted by
//! their reversed bytes, each gets a suffix id, and a flattened trie
//! (`table`) lists, per suffix, every prefix of it that is a scored
//! piece, longest first, followed by a sentinel row. Encoding walks
//! the code points from the end, follows `(code point, suffix id)` into
//! the next suffix id, and relaxes a dynamic program `scores[i] = min
//! over pieces at i of scores[i + len] - score(piece)`; a code point no
//! piece covers takes the sentinel (`UNKNOWN_SCORE`) and falls back to
//! the `<0xXX>` byte tokens at decode time.
//!
//! Transcribed rather than reinvented, in the same order with the same
//! constants (`INVALID_SCORE`, `UNKNOWN_SCORE`, the `1e4` score
//! quantisation, the `1 << 60` initial cost), because the tie-breaking
//! of a segmenter is the part a plausible re-derivation gets wrong
//! silently. Two details worth naming: the suffix order is the
//! reversed BYTE string's order (`std::string(rbegin, rend)`), not the
//! reversed code points'; and special tokens (the `<|plamo:...|>`
//! controls) are in the suffix table like any other entry, so the
//! table sees them, while chat-template markers are still carved out
//! first by [`SpecialTokenTable`] as every other tokenizer here does.
//!
//! `add_bos` defaults false for this vocabulary type
//! (`llama-vocab.cpp:1813`, no per-type override at `:2372-2395`),
//! which `super::should_add_bos_token` already answers.

use std::collections::HashMap;

use super::scored_vocab::ScoredVocab;
use super::special::{SpecialTokenTable, TextOrSpecial};
use super::{SpecialTokens, TokenizerLoadError};

const TABLE_PIECE_LENGTH: usize = 0;
const TABLE_TOKEN_ID: usize = 1;
const TABLE_SCORE: usize = 2;
const TABLE_PIECE_ID: usize = 3;

const INVALID_SCORE: i32 = -20_000_000;
const UNKNOWN_SCORE: i32 = -10_000_000;

/// The GGUF token type of a byte token (`llama_token_type`, BYTE = 6).
const GGML_TOKEN_TYPE_BYTE: i64 = 6;

pub struct GgufPlamo2Tokenizer {
    vocab: ScoredVocab,
    /// Token id of `<0xXX>` for every byte value.
    bytes: [u32; 256],
    /// Which ids ARE byte tokens (decoded as one raw byte).
    is_byte: Vec<bool>,
    /// `(code point << 32 | suffix id of the rest) -> suffix id`.
    to_suffix_id: HashMap<u64, i32>,
    /// `[piece_length, token_id (or -1), score, piece_id]` rows.
    table: Vec<[i32; 4]>,
    special_tokens: SpecialTokenTable,
}

impl GgufPlamo2Tokenizer {
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let vocab = ScoredVocab::from_gguf(file)?;
        let n = vocab.len();

        // Byte tokens by type, as `vocab.is_byte` reads them (`:1373`).
        let mut is_byte = vec![false; n];
        if let Some(frink_gguf::GgufValue::Array(items)) =
            file.metadata("tokenizer.ggml.token_type")
        {
            for (flag, v) in is_byte.iter_mut().zip(items) {
                let ty = match v {
                    frink_gguf::GgufValue::I32(t) => *t as i64,
                    frink_gguf::GgufValue::U32(t) => *t as i64,
                    _ => continue,
                };
                *flag = ty == GGML_TOKEN_TYPE_BYTE;
            }
        }

        let mut bytes = [u32::MAX; 256];
        // `suffix_to_score`: every non-byte token's text at its score,
        // and every proper suffix of it at NaN unless it is a token
        // itself (`:1383-1397`).
        let mut suffix_to_score: HashMap<String, f32> = HashMap::new();
        for (id, text) in vocab.tokens().iter().enumerate() {
            if is_byte[id] {
                if let Some(b) = byte_token_value(text) {
                    bytes[b as usize] = id as u32;
                }
                continue;
            }
            let (_, score) = vocab
                .lookup(text)
                .expect("token text is in its own vocabulary");
            suffix_to_score.insert(text.clone(), score);
            let cpts: Vec<char> = text.chars().collect();
            for i in 1..cpts.len() {
                let suffix: String = cpts[i..].iter().collect();
                suffix_to_score.entry(suffix).or_insert(f32::NAN);
            }
        }
        if let Some(missing) = bytes.iter().position(|&id| id == u32::MAX) {
            return Err(TokenizerLoadError::Plamo2ByteTokenMissing {
                byte: missing as u8,
            });
        }

        // Suffixes in the order of their REVERSED BYTES (`:1408-1419`),
        // the empty suffix among them.
        let mut suffixes: Vec<&str> = suffix_to_score.keys().map(String::as_str).collect();
        suffixes.push("");
        suffixes.sort_by(|a, b| {
            let ra = a.bytes().rev();
            let rb = b.bytes().rev();
            ra.cmp(rb)
        });

        // Suffix ids, the `(code point, rest) -> suffix` map, and the
        // row count (`:1421-1455`).
        let mut suffix_to_id: HashMap<&str, i32> = HashMap::with_capacity(suffixes.len());
        let mut to_suffix_id: HashMap<u64, i32> = HashMap::new();
        let mut num_pieces: i32 = 0;
        for &suffix in &suffixes {
            suffix_to_id.insert(suffix, num_pieces);
            if suffix.is_empty() {
                num_pieces += 1;
                continue;
            }
            let mut chars = suffix.char_indices();
            let (_, first) = chars.next().expect("non-empty");
            let rest = match chars.next() {
                Some((at, _)) => &suffix[at..],
                None => "",
            };
            let rest_id = *suffix_to_id
                .get(rest)
                .expect("a suffix's rest sorts before it in reversed order");
            to_suffix_id.insert(piece_code(first, rest_id), num_pieces);
            // One row per scored prefix of the suffix, plus the sentinel.
            let mut rows = 1;
            for (end, _) in suffix.char_indices().skip(1) {
                if suffix_to_score.contains_key(&suffix[..end]) {
                    rows += 1;
                }
            }
            rows += 1; // the whole suffix is in `suffix_to_score` by construction
            num_pieces += rows;
        }

        // The flattened table (`:1457-1494`).
        let mut table: Vec<[i32; 4]> = Vec::with_capacity(num_pieces as usize);
        for &suffix in &suffixes {
            // Prefixes in decreasing length: the whole suffix first.
            let mut ends: Vec<usize> = suffix.char_indices().skip(1).map(|(i, _)| i).collect();
            ends.push(suffix.len());
            for &end in ends.iter().rev() {
                if end == 0 {
                    continue;
                }
                let piece = &suffix[..end];
                let Some(&score) = suffix_to_score.get(piece) else {
                    continue;
                };
                let mut row = [0i32; 4];
                row[TABLE_PIECE_LENGTH] = piece.chars().count() as i32;
                row[TABLE_TOKEN_ID] = vocab.id_of(piece).map_or(-1, |id| id as i32);
                row[TABLE_SCORE] = if score.is_finite() {
                    (score * 1e4).round() as i32
                } else {
                    INVALID_SCORE
                };
                row[TABLE_PIECE_ID] = suffix_to_id[piece];
                table.push(row);
            }
            let mut sentinel = [0i32; 4];
            sentinel[TABLE_PIECE_LENGTH] = 1;
            sentinel[TABLE_TOKEN_ID] = -1;
            sentinel[TABLE_SCORE] = UNKNOWN_SCORE;
            table.push(sentinel);
        }
        debug_assert_eq!(table.len(), num_pieces as usize);

        let special_tokens = SpecialTokenTable::from_gguf(file, vocab.tokens());
        Ok(Self {
            vocab,
            bytes,
            is_byte,
            to_suffix_id,
            table,
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Chat-template markers carved out first, then each raw run
    /// through the suffix-table search (`:1509-1616`).
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        self.special_tokens
            .split(text, specials)
            .into_iter()
            .flat_map(|seg| match seg {
                TextOrSpecial::Special(id) => vec![id],
                TextOrSpecial::Text(t) => self.encode_run(t),
            })
            .collect()
    }

    fn encode_run(&self, text: &str) -> Vec<u32> {
        let mut cpts: Vec<char> = text.chars().collect();
        if cpts.first() == Some(&'\u{FEFF}') {
            cpts.remove(0);
        }
        if cpts.is_empty() {
            return Vec::new();
        }
        let n = cpts.len();
        let mut scores = vec![1i64 << 60; n + 1];
        scores[n] = 0;
        // Per position: (token length, token id, token count).
        let mut path = vec![(0i32, -1i32, 0i32); n + 1];
        let mut suffix_id: i32 = 0;

        for i in (0..n).rev() {
            let c = cpts[i];
            // Follow the trie to the suffix that starts at `i`.
            for p in (suffix_id as usize)..self.table.len() {
                let code = piece_code(c, self.table[p][TABLE_PIECE_ID]);
                suffix_id = self.to_suffix_id.get(&code).copied().unwrap_or(0);
                if suffix_id > 0 || self.table[p][TABLE_SCORE] == UNKNOWN_SCORE {
                    break;
                }
            }
            // Relax every scored piece at that suffix, sentinel last.
            for entry in &self.table[suffix_id as usize..] {
                let score = entry[TABLE_SCORE];
                if score > INVALID_SCORE {
                    let len = entry[TABLE_PIECE_LENGTH];
                    let s = scores[i + len as usize] - score as i64;
                    if s < scores[i] {
                        scores[i] = s;
                        let mut count = path[i + len as usize].2 + 1;
                        if score == UNKNOWN_SCORE {
                            count += utf8_extra_bytes(c);
                        }
                        path[i] = (len, entry[TABLE_TOKEN_ID], count);
                    }
                }
                if score == UNKNOWN_SCORE {
                    break;
                }
            }
        }

        let mut out = Vec::with_capacity(path[0].2.max(0) as usize);
        let mut pos = 0;
        while pos < n {
            let (len, id, _) = path[pos];
            if id >= 0 {
                out.push(id as u32);
            } else {
                let mut buf = [0u8; 4];
                for &b in cpts[pos].encode_utf8(&mut buf).as_bytes() {
                    out.push(self.bytes[b as usize]);
                }
            }
            debug_assert!(len > 0, "every position advances");
            pos += len.max(1) as usize;
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Byte tokens are one raw byte each, every other token its text
    /// (`llama-vocab.cpp:3623-3639`).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            let Some(text) = self.vocab.token(id) else {
                continue;
            };
            if self.is_byte.get(id as usize).copied().unwrap_or(false) {
                if let Some(b) = byte_token_value(text) {
                    out.push(b);
                    continue;
                }
            }
            out.extend_from_slice(text.as_bytes());
        }
        out
    }
}

/// `<0xXX>` -> `XX`, else `None`.
fn byte_token_value(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn piece_code(c: char, rest_id: i32) -> u64 {
    ((c as u64) << 32) | (rest_id as u32 as u64)
}

/// The extra UTF-8 bytes of `c` beyond the first (`:1571`).
fn utf8_extra_bytes(c: char) -> i32 {
    let c = c as u32;
    (c >= 0x80) as i32 + (c >= 0x800) as i32 + (c >= 0x10000) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::scored_vocab::MetadataOnlyGguf as MetaOnly;

    /// A vocabulary of the shape the real file has: the four
    /// `<|plamo:...|>` controls, the 256 byte tokens, and a few scored
    /// pieces with overlapping suffixes.
    fn fixture() -> GgufPlamo2Tokenizer {
        let mut tokens: Vec<String> = vec![
            "<|plamo:unk|>".into(),
            "<|plamo:bos|>".into(),
            "<|plamo:eos|>".into(),
            "<|plamo:pad|>".into(),
        ];
        let mut types: Vec<i32> = vec![2, 3, 3, 3];
        for b in 0..256u32 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(6);
        }
        let pieces: &[(&str, f32)] = &[
            ("a", -3.0),
            ("b", -3.0),
            ("ab", -4.0),
            ("abc", -2.0),
            ("bc", -3.5),
            ("c", -3.0),
            (" the", -1.5),
            ("the", -2.5),
            (" ", -4.0),
            ("日本", -2.0),
            ("本", -5.0),
        ];
        let mut scores = vec![0.0f32; tokens.len()];
        for (t, s) in pieces {
            tokens.push((*t).to_string());
            types.push(1);
            scores.push(*s);
        }
        let toks: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let meta = MetaOnly::new()
            .with_tokens(&toks)
            .with_scores(&scores)
            .with(
                "tokenizer.ggml.token_type",
                frink_gguf::GgufValue::Array(
                    types.into_iter().map(frink_gguf::GgufValue::I32).collect(),
                ),
            );
        GgufPlamo2Tokenizer::from_gguf(&meta).unwrap()
    }

    fn id(t: &GgufPlamo2Tokenizer, s: &str) -> u32 {
        t.vocab.id_of(s).unwrap()
    }

    #[test]
    fn picks_the_highest_scoring_segmentation_not_the_greedy_one() {
        let t = fixture();
        // "abc" as one piece (-2.0) beats "ab"+"c" (-7.0) and "a"+"bc".
        assert_eq!(t.encode("abc", SpecialTokens::AsText), vec![id(&t, "abc")]);
        // "the" alone, then " the" with its leading space as one piece.
        assert_eq!(
            t.encode("the the", SpecialTokens::AsText),
            vec![id(&t, "the"), id(&t, " the")]
        );
        assert_eq!(
            t.encode("日本", SpecialTokens::AsText),
            vec![id(&t, "日本")]
        );
    }

    #[test]
    fn a_character_no_piece_covers_falls_back_to_its_utf8_bytes() {
        let t = fixture();
        // 'z' is in no piece: one byte token.
        assert_eq!(
            t.encode("z", SpecialTokens::AsText),
            vec![t.bytes[b'z' as usize]]
        );
        // '語' (3 bytes) is in no piece: three byte tokens, in order.
        let got = t.encode("語", SpecialTokens::AsText);
        let want: Vec<u32> = "語".bytes().map(|b| t.bytes[b as usize]).collect();
        assert_eq!(got, want);
        // ...and round-trips through decode as the character.
        assert_eq!(t.decode(&got), "語");
        assert_eq!(
            t.decode(&t.encode("abc z 日本", SpecialTokens::AsText)),
            "abc z 日本"
        );
    }

    #[test]
    fn a_missing_byte_token_is_refused_at_load() {
        let toks = ["<|plamo:unk|>", "<0x00>", "a"];
        let meta = MetaOnly::new()
            .with_tokens(&toks)
            .with_scores(&[0.0, 0.0, -1.0])
            .with(
                "tokenizer.ggml.token_type",
                frink_gguf::GgufValue::Array(
                    [2, 6, 1]
                        .into_iter()
                        .map(frink_gguf::GgufValue::I32)
                        .collect(),
                ),
            );
        assert!(matches!(
            GgufPlamo2Tokenizer::from_gguf(&meta),
            Err(TokenizerLoadError::Plamo2ByteTokenMissing { byte: 1 })
        ));
    }
}
