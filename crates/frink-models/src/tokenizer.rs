//! A real, reversible byte-level tokenizer: each UTF-8 byte maps to
//! token id `byte as u32` (vocabulary 0..256). This is not a full
//! BPE/tokenizer.json implementation -- GLM-5.2, DeepSeek V4 Pro, and
//! Kimi K3 each ship their own trained BPE vocabulary alongside their
//! weights, and none of those vocab files are guessable or available in
//! this environment (see docs/MODELS.md) -- but unlike the
//! previous placeholder (`byte % vocab_size`, which was lossy and could
//! not decode back to the original text), this tokenizer is exact and
//! round-trips perfectly. It is the honest "smallest real thing that
//! works" rather than a fake stand-in.
//!
//! Loading a real BPE merge table from a GGUF file's
//! `tokenizer.ggml.tokens` / `tokenizer.ggml.merges` metadata arrays
//! (see `frink-gguf`'s `GgufValue::Array` support, already verified
//! against a real downloaded llama.cpp vocab fixture) was the natural
//! next step and now exists below (`GgufBpeTokenizer`,
//! `GgufSpmTokenizer`, `GgufUnigramTokenizer`).
//!
//! The per-checkpoint pre-tokenization rules live next door in
//! [`pretokenize`], which is a transcription of llama.cpp and is
//! reviewed against it.
//!
//! Four of llama.cpp's six `tokenizer.ggml.model` values are covered:
//! `gpt2`/`gemma4` by [`GgufBpeTokenizer`], `llama` by
//! [`GgufSpmTokenizer`], `t5` by [`GgufUnigramTokenizer`], and `bert` by
//! [`GgufWordPieceTokenizer`] in `wordpiece`, which brings its own
//! normalizer and its own Unicode tables (`unicode`, `unicode_data`)
//! because WordPiece does not use the pre-tokenizer regexes at all,
//! and `plamo2` by [`GgufPlamo2Tokenizer`] in `plamo2`, a suffix-table
//! segmenter. Still missing: `rwkv`, which needs a trie tokenizer, and
//! `none`.

mod plamo2;
mod pretokenize;
mod scored_vocab;
mod special;
mod unicode;
mod unicode_data;
mod wordpiece;

pub use plamo2::GgufPlamo2Tokenizer;
use scored_vocab::ScoredVocab;
pub use special::SpecialTokens;
pub(crate) use special::{SpecialKind, SpecialTokenTable, TextOrSpecial};
pub use wordpiece::{GgufWordPieceTokenizer, NormalizerOptions};

/// The `tokenizer.ggml.pre` values whose llama.cpp arm sets
/// `add_bos = true` for a BPE vocabulary.
///
/// Transcribed from `.scratch/llama.cpp/src/llama-vocab.cpp`: the
/// `LLAMA_VOCAB_PRE_TYPE_LLAMA3` arm sets it for the whole llama3 group
/// in one statement, and `tekken` and `chameleon` set it in arms of
/// their own. Llama-3.x GGUFs ship no explicit
/// `tokenizer.ggml.add_bos_token`, so leaving the group out made every
/// raw completion prompt one `<|begin_of_text|>` short of llama.cpp's.
const ADD_BOS_PRE: &[&str] = &[
    // LLAMA_VOCAB_PRE_TYPE_LLAMA3
    "llama3",
    "llama-v3",
    "llama-bpe",
    "falcon3",
    "falcon-h1",
    "pixtral",
    "midm-2.0",
    "lfm2",
    "jina-v5-nano",
    // arms of their own, same flag
    "tekken",
    "chameleon",
];

/// Whether prompt encoding should prepend the GGUF BOS token.
///
/// Port of llama.cpp `llama_vocab` add_bos defaults
/// (`.scratch/llama.cpp/src/llama-vocab.cpp`): explicit
/// `tokenizer.ggml.add_bos_token` wins; else SPM → true, BPE → false
/// unless the checkpoint's `pre` is one of [`ADD_BOS_PRE`]. Qwen2-MoE
/// ships `bos_token_id=<|endoftext|>` but `add_bos=false` — always
/// prepending that token poisons greedy decode.
pub fn should_add_bos_token(file: &impl frink_gguf::TensorSource) -> bool {
    if let Some(v) = file.metadata_bool("tokenizer.ggml.add_bos_token") {
        return v;
    }
    let model = file.metadata_str("tokenizer.ggml.model").unwrap_or("");
    let pre = file.metadata_str("tokenizer.ggml.pre").unwrap_or("");
    // llama.cpp: SPM/WPM default add_bos=true; BPE defaults false unless
    // its pre-tokenizer arm opts in. qwen2 leaves false.
    // `bert` is WPM, whose upstream arm sets add_bos AND add_sep true.
    // It was missing here, so every WordPiece prompt was one `[CLS]`
    // short of llama.cpp's.
    if matches!(model, "llama" | "spm" | "bert") || model.contains("sentencepiece") {
        return true;
    }
    ADD_BOS_PRE.contains(&pre)
}

/// Prepends the checkpoint's BOS id to an already-encoded prompt, unless
/// the prompt already starts with it.
///
/// # The rule, stated once
///
/// **The chat template owns BOS when it prints one; the loader owns it
/// otherwise.** Which of the two happens is a property of the individual
/// checkpoint, not of the family:
///
/// * Many upstream templates open with `{{ bos_token }}` — gemma-2/3
///   (`<bos>`), Mistral-Instruct and TinyLlama (`<s>`), Llama-3
///   (`<|begin_of_text|>`). Rendering one of those already puts BOS in
///   the *text*, and both [`GgufBpeTokenizer::encode`] and
///   [`GgufSpmTokenizer::encode`] split on special-token text first, so
///   it comes back as the BOS *id* in position 0.
/// * Unsloth deliberately **strips** `{{ bos_token }}` out of the
///   templates it bakes into its GGUF exports, precisely so that a
///   runtime which adds BOS itself does not double it. On those
///   checkpoints the render carries no BOS and the loader must add it.
///
/// So neither "always add" nor "never add" is right, and a renderer
/// cannot be sniffed for which case it is. This function implements the
/// only rule that is correct for both: add the id, **idempotently**.
/// `bos` is already the gated value — pass `None` when
/// [`should_add_bos_token`] says this vocabulary does not take one
/// (BPE/qwen2 ship a `bos_token_id` they never prepend).
///
/// Note this is *stricter* than llama.cpp, whose `add_special` path
/// pushes BOS unconditionally and leaves the duplicate to a warning.
/// Frink has no user-visible "you asked for two BOS tokens" surface, so
/// it dedupes instead of warning.
/// Generic over the id width because the CLI and server carry prompts as
/// `Vec<usize>` and the tokenizers emit `Vec<u32>`.
pub fn prepend_bos<T: Copy + PartialEq>(tokens: &mut Vec<T>, bos: Option<T>) {
    let Some(bos) = bos else { return };
    if tokens.first() != Some(&bos) {
        tokens.insert(0, bos);
    }
}

/// Token texts llama.cpp treats as end-of-generation regardless of what
/// the metadata ids say (`llama-vocab.cpp`, the literal list right above
/// its "sanity checks" block). Copied verbatim, including the comments
/// naming which family each entry exists for, because the set is not
/// derivable: it is a hand-maintained list of what real checkpoints ship.
///
/// Note `<|end|>` *is* here. The Unsloth study recorded in
/// `docs/plans/llama-cpp-parity-push.md` claimed gpt-oss's `<|end|>` must
/// not be EOG or every reply truncates; llama.cpp's own source says
/// otherwise, and llama.cpp serves gpt-oss. Following the reference
/// implementation, and flagging the claim as contradicted.
const EOG_TOKEN_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>", // o200k_harmony
    "<|call|>",   // o200k_harmony
    "<|flush|>",  // solar-open
    "<|calls|>",  // solar-open
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>", // paddleocr
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]", // Kimi-K2
    "[EOS]", // Kimi-K2
    "<|end_of_text|>",
    "<end_of_utterance>",    // smoldocling
    "<eos>",                 // gemma4
    "<turn|>",               // gemma4
    "<|tool_response>",      // gemma4
    "<｜end▁of▁sentence｜>", // deepseek-ocr
    "[e~[",                  // minimax-m2/m3
];

/// Every token id that ends generation, not just `eos_token_id`.
///
/// A single EOS id is wrong for most modern chat checkpoints: Llama-3
/// ends turns with `<|eot_id|>` while its `eos_token_id` is
/// `<|end_of_text|>`, and gemma-4 ends with `<turn|>`. Stopping only on
/// the metadata EOS means the model keeps generating past the end of its
/// turn and starts a new one — the "it answers, then interviews itself"
/// failure.
///
/// Mirrors llama.cpp: the literal-name list above, plus the
/// `eos`/`eot`/`eom` metadata ids, which it folds in with a warning when
/// they were not already caught by name.
pub fn eog_token_ids(file: &impl frink_gguf::TensorSource) -> std::collections::HashSet<u32> {
    let mut out = std::collections::HashSet::new();
    for key in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
    ] {
        if let Some(id) = file.metadata_u64(key) {
            out.insert(id as u32);
        }
    }
    if let Some(frink_gguf::GgufValue::Array(items)) = file.metadata("tokenizer.ggml.tokens") {
        for (id, v) in items.iter().enumerate() {
            if let frink_gguf::GgufValue::String(text) = v {
                if EOG_TOKEN_TEXTS.contains(&text.as_str()) {
                    out.insert(id as u32);
                }
            }
        }
    }
    out
}

/// The set of token ids a decode loop must stop on, carried as one value
/// so a caller cannot accidentally carry only half of it.
///
/// This type exists because `Option<usize>` was the shape of a real bug:
/// every `frink-server` decode loop threaded a single `eos_id` from the
/// loader to the sampler, so a Llama-3 or gemma checkpoint served over
/// HTTP ran past `<|eot_id|>` / `<end_of_turn>` to `max_tokens` even
/// after [`eog_token_ids`] landed for the CLI. Passing a `StopTokens`
/// makes "I only have the metadata EOS" an explicit choice
/// ([`StopTokens::from_eos`], for the synthetic-weights and Kimi paths
/// that have no GGUF metadata to read) rather than the default.
#[derive(Clone, Debug, Default)]
pub struct StopTokens {
    ids: std::collections::HashSet<u32>,
}

impl StopTokens {
    /// Everything [`eog_token_ids`] finds in this checkpoint: the
    /// `eos`/`eot`/`eom` metadata ids plus every vocabulary entry whose
    /// text is on llama.cpp's literal EOG list.
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Self {
        Self {
            ids: eog_token_ids(file),
        }
    }

    /// Just the one id. For callers with no GGUF metadata behind them —
    /// the synthetic random-weights demo model, and the Kimi checkpoint
    /// directory whose tokenizer is a separate file format.
    pub fn from_eos(eos: Option<usize>) -> Self {
        Self {
            ids: eos.map(|e| e as u32).into_iter().collect(),
        }
    }

    /// For checkpoints whose vocabulary is not GGUF metadata — Kimi K3
    /// ships a `tokenizer_config.json` with a name→id special-token map.
    /// Folds in every entry whose *text* is on llama.cpp's EOG list, so
    /// `[EOT]` stops a turn there exactly as it does in a GGUF.
    pub fn from_special_tokens<'a>(specials: impl IntoIterator<Item = (&'a str, u32)>) -> Self {
        Self {
            ids: specials
                .into_iter()
                .filter(|(name, _)| EOG_TOKEN_TEXTS.contains(name))
                .map(|(_, id)| id)
                .collect(),
        }
    }

    /// Folds one more id in — used to keep a metadata `eos_token_id` that
    /// a vocabulary spells in a way the literal list does not know.
    pub fn with_id(mut self, id: Option<usize>) -> Self {
        if let Some(id) = id {
            self.ids.insert(id as u32);
        }
        self
    }

    pub fn contains(&self, id: usize) -> bool {
        u32::try_from(id).is_ok_and(|id| self.ids.contains(&id))
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn encode(text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }

    /// Decodes token ids back to a string. Ids outside 0..256 are
    /// dropped rather than silently corrupting output; invalid UTF-8
    /// byte sequences are replaced per Rust's standard lossy conversion.
    pub fn decode(ids: &[u32]) -> String {
        String::from_utf8_lossy(&Self::decode_bytes(ids)).into_owned()
    }

    /// The raw bytes, before any UTF-8 decision is made about them.
    ///
    /// A caller decoding ONE token at a time must have these: a
    /// multi-byte character split across two tokens is two invalid
    /// fragments, and `decode` would turn each into U+FFFD and lose the
    /// bytes for good. See `frink_server::utf8_stream`.
    pub fn decode_bytes(ids: &[u32]) -> Vec<u8> {
        ids.iter().filter_map(|&id| u8::try_from(id).ok()).collect()
    }

    pub const VOCAB_SIZE: usize = 256;
}

/// How a GGUF BPE vocabulary remaps text before merge lookup.
///
/// GPT-2-style vocabs store merges in the OpenAI byte↔unicode remapped
/// space; Gemma-4 (and similar SPM-flavoured BPE) stores merges over
/// raw UTF-8 with spaces already escaped to U+2581 (`▁`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BpeEncodingStyle {
    Gpt2,
    /// llama.cpp `LLAMA_VOCAB_PRE_TYPE_GEMMA4`: escape `" "` → `▁`,
    /// split only on newlines, merge on raw UTF-8 codepoints
    /// (`byte_encode = false`).
    SpmWhitespace,
}

/// Builds the GPT2 byte-to-unicode remap table: bytes in the "already
/// printable, unambiguous" ranges (33..=126, 161..=172, 174..=255) map
/// to themselves as Unicode codepoints; every other byte (control
/// characters, space, and a few others that would be ambiguous or
/// unprintable as raw codepoints) maps to a codepoint starting at 256.
/// This is the exact algorithm from OpenAI's GPT-2 `encoder.py`
/// `bytes_to_unicode()`, reimplemented independently in Rust: real BPE
/// vocabularies (llama.cpp, and anything using the `tokenizers`
/// crate) list
/// merge-table entries in *this* remapped space (e.g. "\u{0120}the",
/// where the leading char is U+0120, the remapped space byte 0x20), not
/// in raw byte or `char` space, so skipping this step -- which frink
/// did before this function existed -- silently fails to match any real
/// vocabulary's merge table on space- and control-byte-adjacent tokens.
fn gpt2_byte_to_unicode() -> ([char; 256], std::collections::HashMap<char, u8>) {
    let is_printable =
        |b: u16| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);

    let mut forward = ['\0'; 256];
    let mut extra_offset = 0u32;
    for b in 0..256u16 {
        if is_printable(b) {
            forward[b as usize] = char::from_u32(b as u32).unwrap();
        } else {
            forward[b as usize] = char::from_u32(256 + extra_offset).unwrap();
            extra_offset += 1;
        }
    }

    let mut reverse = std::collections::HashMap::with_capacity(256);
    for (b, &c) in forward.iter().enumerate() {
        reverse.insert(c, b as u8);
    }
    (forward, reverse)
}

/// The chunking that runs before BPE: raw text is cut into
/// contractions, letter runs, digit runs, symbol runs and whitespace
/// runs, and each chunk is merged separately. Without it `encode_word`
/// would treat a whole sentence as one word and could merge across word
/// boundaries in ways no real tokenizer does.
///
/// Which pattern a checkpoint gets, and what happens to the text
/// between matches, is [`pretokenize`]'s job — it is a transcription of
/// llama.cpp's `llama-vocab.cpp` and `unicode.cpp` and is reviewed
/// against them.
/// U+2581 FIGURE SPACE used by SentencePiece-style BPE merge tables.
const SPM_SPACE: char = '\u{2581}';

/// A real BPE tokenizer built from a GGUF file's own
/// `tokenizer.ggml.tokens` / `tokenizer.ggml.merges` metadata arrays.
/// Supports GPT-2 byte-remap BPE (`tokenizer.ggml.model == "gpt2"`) and
/// Gemma-4 SPM-style BPE (`"gemma4"`: escape spaces to `▁`, merge on
/// raw UTF-8, newline-only pre-split).
///
/// Verified against `tests/fixtures/llama-bpe-vocab.gguf` (GPT-2 path).
/// See `crates/frink-models/tests/gguf_vocab.rs`.
pub struct GgufBpeTokenizer {
    token_to_id: std::collections::HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// merge rank: lower = merges earlier (higher priority), matching
    /// the standard BPE convention of applying the most-frequent
    /// (lowest-rank) merge first.
    merge_rank: std::collections::HashMap<(String, String), usize>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: std::collections::HashMap<char, u8>,
    /// The vocabulary's special entries, carved out of the input before
    /// BPE runs on what is left -- see [`special::SpecialTokenTable`].
    special_tokens: SpecialTokenTable,
    /// Compiled pre-tokenization pattern (GPT-2 word regex, or
    /// newline-only for Gemma-4 SPM-BPE).
    pretokenize_pattern: fancy_regex::Regex,
    style: BpeEncodingStyle,
}

#[derive(Debug, thiserror::Error)]
pub enum TokenizerLoadError {
    #[error("GGUF file has no 'tokenizer.ggml.tokens' metadata array")]
    MissingTokens,
    #[error("'tokenizer.ggml.tokens' is present but is not a string array")]
    TokensNotStringArray,
    #[error("'tokenizer.ggml.tokens' is present but empty: a vocabulary with no entries cannot tokenize anything, and its scores have no minimum")]
    EmptyVocabulary,
    #[error(
        "vocabulary and scores disagree about the vocabulary size: 'tokenizer.ggml.tokens' has \
         {tokens} entries but 'tokenizer.ggml.scores' has {scores}. A score-carrying vocabulary \
         needs one score per token; this checkpoint cannot be tokenized"
    )]
    ScoresVocabLengthMismatch { tokens: usize, scores: usize },
    #[error(
        "PLaMo-2 vocabulary has no byte token <0x{byte:02X}>: llama-vocab.cpp:1400-1404 refuses \
         the file for the same reason, because a character no piece covers is spelled in these"
    )]
    Plamo2ByteTokenMissing { byte: u8 },
}

impl GgufBpeTokenizer {
    /// Loads the vocabulary + merge table from a GGUF file's metadata.
    /// Merges are optional (some tokenizer types, e.g. byte-level
    /// unigram, don't use them); if absent, encoding falls back to
    /// per-byte token lookup. `tokenizer.ggml.model == "gemma4"` selects
    /// SPM-whitespace BPE; everything else with merges uses GPT-2 style.
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let tokens_value = file
            .metadata("tokenizer.ggml.tokens")
            .ok_or(TokenizerLoadError::MissingTokens)?;
        let id_to_token: Vec<String> = match tokens_value {
            frink_gguf::GgufValue::Array(items) => items
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Option<Vec<_>>>()
                .ok_or(TokenizerLoadError::TokensNotStringArray)?,
            _ => return Err(TokenizerLoadError::TokensNotStringArray),
        };

        let token_to_id: std::collections::HashMap<String, u32> = id_to_token
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let style = match file.metadata_str("tokenizer.ggml.model") {
            Some("gemma4") => BpeEncodingStyle::SpmWhitespace,
            _ => BpeEncodingStyle::Gpt2,
        };

        let mut merge_rank = std::collections::HashMap::new();
        if let Some(frink_gguf::GgufValue::Array(items)) = file.metadata("tokenizer.ggml.merges") {
            for (rank, item) in items.iter().enumerate() {
                if let Some(s) = item.as_str() {
                    if let Some((a, b)) = split_bpe_merge_pair(s, style) {
                        merge_rank.insert((a, b), rank);
                    }
                }
            }
        }

        let (byte_to_unicode, unicode_to_byte) = gpt2_byte_to_unicode();
        let pretokenize_pattern = match style {
            // Keyed on the checkpoint's own `tokenizer.ggml.pre`, which
            // was previously read only to decide BOS prepending.
            BpeEncodingStyle::Gpt2 => {
                pretokenize::regex_for(file.metadata_str("tokenizer.ggml.pre").unwrap_or(""))
            }
            BpeEncodingStyle::SpmWhitespace => pretokenize::newline_regex(),
        };
        let special_tokens = SpecialTokenTable::from_gguf(file, &id_to_token);

        Ok(GgufBpeTokenizer {
            token_to_id,
            id_to_token,
            merge_rank,
            byte_to_unicode,
            unicode_to_byte,
            special_tokens,
            pretokenize_pattern,
            style,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn has_merges(&self) -> bool {
        !self.merge_rank.is_empty()
    }

    /// Greedy BPE merge over one pre-split chunk. GPT-2 style remaps
    /// bytes through `byte_to_unicode`; Gemma-4 style merges raw UTF-8
    /// codepoints (after `" "` → `▁` escaping in `encode`).
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut pieces: Vec<String> = match self.style {
            BpeEncodingStyle::Gpt2 => word
                .bytes()
                .map(|b| self.byte_to_unicode[b as usize].to_string())
                .collect(),
            BpeEncodingStyle::SpmWhitespace => word.chars().map(|c| c.to_string()).collect(),
        };
        if pieces.is_empty() {
            return Vec::new();
        }

        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..pieces.len().saturating_sub(1) {
                if let Some(&rank) = self
                    .merge_rank
                    .get(&(pieces[i].clone(), pieces[i + 1].clone()))
                {
                    if best.map(|(r, _)| rank < r).unwrap_or(true) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let merged = format!("{}{}", pieces[i], pieces[i + 1]);
                    pieces.splice(i..=i + 1, [merged]);
                }
                None => break,
            }
        }

        pieces.iter().flat_map(|p| self.piece_to_ids(p)).collect()
    }

    fn piece_to_ids(&self, piece: &str) -> Vec<u32> {
        if let Some(&id) = self.token_to_id.get(piece) {
            return vec![id];
        }
        match self.style {
            BpeEncodingStyle::Gpt2 => {
                // Fall back to first remapped-byte character (GPT-2 base).
                piece
                    .chars()
                    .next()
                    .and_then(|c| self.token_to_id.get(&c.to_string()))
                    .copied()
                    .map(|id| vec![id])
                    .unwrap_or_else(|| vec![0])
            }
            BpeEncodingStyle::SpmWhitespace => {
                // llama.cpp non-byte-encoded BPE: unknown pieces → `<0xXX>`.
                piece
                    .bytes()
                    .filter_map(|b| {
                        let hex = format!("<0x{b:02X}>");
                        self.token_to_id.get(&hex).copied()
                    })
                    .collect()
            }
        }
    }

    /// Encodes text: specials first (those `specials` lets through),
    /// then style-specific pretokenize + `encode_word`. Gemma-4 escapes
    /// spaces to `▁` and splits only on newlines; newline-only chunks
    /// look up the whole string in vocab (multi-newline tokens) before
    /// BPE.
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        self.special_tokens
            .split(text, specials)
            .into_iter()
            .flat_map(|seg| -> Vec<u32> {
                match seg {
                    TextOrSpecial::Special(id) => vec![id],
                    TextOrSpecial::Text(t) => self.encode_text_run(t),
                }
            })
            .collect()
    }

    /// `pretokenize::split_with_gaps` rather than a bare `find_iter`
    /// loop: the text BETWEEN matches is input too, and llama.cpp emits
    /// it as its own chunk. Dropping it lost tabs, NBSPs, form feeds and
    /// interior newlines out of the middle of every OLMo prompt.
    fn encode_text_run(&self, text: &str) -> Vec<u32> {
        match self.style {
            BpeEncodingStyle::Gpt2 => pretokenize::split_with_gaps(&self.pretokenize_pattern, text)
                .into_iter()
                .flat_map(|chunk| self.encode_word(chunk))
                .collect(),
            BpeEncodingStyle::SpmWhitespace => {
                let escaped: String = text
                    .chars()
                    .map(|c| if c == ' ' { SPM_SPACE } else { c })
                    .collect();
                // Manual newline split (O(n)); avoids regex stack issues
                // on long non-newline spans (llama.cpp PR #21587).
                let mut out = Vec::new();
                let bytes = escaped.as_bytes();
                let mut i = 0usize;
                while i < bytes.len() {
                    let is_nl = bytes[i] == b'\n';
                    let mut j = i + 1;
                    while j < bytes.len() && (bytes[j] == b'\n') == is_nl {
                        j += 1;
                    }
                    // Safe: we only split on ASCII `\n`, so `i..j` is UTF-8.
                    let word = std::str::from_utf8(&bytes[i..j]).expect("newline split keeps utf8");
                    if is_nl {
                        if let Some(&id) = self.token_to_id.get(word) {
                            out.push(id);
                        } else {
                            out.extend(self.encode_word(word));
                        }
                    } else {
                        out.extend(self.encode_word(word));
                    }
                    i = j;
                }
                out
            }
        }
    }

    /// GPT-2: remapped unicode → bytes. Gemma-4: unescape `▁` → space and
    /// expand `<0xXX>` byte tokens (same shape as SPM decode).
    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// The raw bytes, before any UTF-8 decision is made about them.
    /// See [`GgufBpeTokenizer::decode`] and `frink_server::utf8_stream`.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        match self.style {
            BpeEncodingStyle::Gpt2 => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .filter_map(|&id| self.id_to_token.get(id as usize))
                    .flat_map(|token| token.chars())
                    .filter_map(|c| self.unicode_to_byte.get(&c).copied())
                    .collect();
                bytes
            }
            BpeEncodingStyle::SpmWhitespace => {
                let mut bytes: Vec<u8> = Vec::new();
                for &id in ids {
                    let Some(token) = self.id_to_token.get(id as usize) else {
                        continue;
                    };
                    if let Some(b) = spm_byte_fallback_value(token) {
                        bytes.push(b);
                    } else {
                        bytes.extend(token.replace(SPM_SPACE, " ").into_bytes());
                    }
                }
                bytes
            }
        }
    }
}

/// Split a GGUF merge line `"left right"` into pair. Gemma-4 / llama.cpp
/// use `find(' ', 1)` on the raw byte string so a leading ASCII space in
/// `left` is not the separator; search from byte 1 (not char 1) to match.
fn split_bpe_merge_pair(s: &str, style: BpeEncodingStyle) -> Option<(String, String)> {
    match style {
        BpeEncodingStyle::Gpt2 => s
            .split_once(' ')
            .map(|(a, b)| (a.to_string(), b.to_string())),
        BpeEncodingStyle::SpmWhitespace => {
            let bytes = s.as_bytes();
            if bytes.len() < 2 {
                return None;
            }
            let pos = bytes[1..].iter().position(|&b| b == b' ')? + 1;
            // ASCII space is always a UTF-8 char boundary.
            Some((s[..pos].to_string(), s[pos + 1..].to_string()))
        }
    }
}

fn spm_byte_fallback_value(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// A real SentencePiece-BPE tokenizer, built from a GGUF file's
/// `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` metadata
/// (`tokenizer.ggml.model == "llama"` in GGUF's convention -- this is
/// SentencePiece's *BPE* model type, not its Unigram model type,
/// despite both living under the umbrella term "SentencePiece"; the
/// distinction matters because the encode algorithms are different).
///
/// # How this differs from `GgufBpeTokenizer`
///
/// `GgufBpeTokenizer` implements GPT2-style BPE: a fixed merge-rank
/// table applied greedily left-to-right after GPT2's own
/// byte-to-unicode remap and regex pre-tokenization. SentencePiece-BPE
/// vocabularies (used by the original LLaMA, and generally any model
/// whose GGUF reports `tokenizer.ggml.model = "llama"`) don't ship a
/// merge-rank table at all -- instead every vocabulary entry carries a
/// score, and encoding works by repeatedly merging whichever *currently
/// adjacent* pair of symbols forms the highest-scoring known vocabulary
/// piece, using a priority queue over merge candidates (this is the
/// `llm_tokenizer_spm` algorithm from llama.cpp, reimplemented here
/// independently against the public GGUF metadata, not from llama.cpp
/// source). Preprocessing replaces spaces with `▁` (U+2581) and adds a
/// leading `▁`, matching SentencePiece's own convention, rather than
/// GPT2's byte-to-unicode remap.
///
/// # A real bug found and fixed while building this
///
/// The first implementation of this algorithm checked merge-candidate
/// validity by adjacency alone (`is this pair still directly next to
/// each other in the linked list?`). That's necessary but not
/// sufficient: a symbol's *content* can change between when a
/// candidate merge is queued and when it's popped, if that symbol was
/// itself the survivor of a *different* merge in the meantime, while
/// staying adjacency-valid at the same list position. The fix is to
/// also store the exact left/right text expected at queue time and
/// re-check it at pop time, discarding (not re-queuing) any candidate
/// whose content has since changed. This was caught immediately by
/// testing against real reference data (see below) rather than by
/// code review -- the bug produced plausible-looking but wrong output
/// ("Hello world" tokenized as 6 pieces instead of the correct 2)
/// which would have been easy to miss without a real ground truth to
/// check against.
///
/// # Verification
///
/// Tested against `tests/fixtures/llama-spm-vocab.gguf` (downloaded
/// directly from `ggml-org/llama.cpp`'s own repository, the real
/// LLaMA-1/2 tokenizer vocabulary) and its accompanying
/// `.gguf.inp`/`.gguf.out` files -- llama.cpp's own CI test corpus of
/// 45 input strings and their exact expected token ID sequences,
/// covering ASCII, whitespace runs, control characters, CJK/Khmer/
/// Vietnamese text, emoji, and byte-fallback. All 45 match exactly.
pub struct GgufSpmTokenizer {
    /// The vocabulary and its per-token scores, checked against each
    /// other at load -- see [`scored_vocab`]. Shared with
    /// [`GgufUnigramTokenizer`] so that the score lookup exists once
    /// rather than once per tokenizer.
    vocab: ScoredVocab,
    /// The vocabulary's special entries, carved out of the input before
    /// SentencePiece-BPE runs on what is left -- see
    /// [`special::SpecialTokenTable`].
    special_tokens: SpecialTokenTable,
    /// `tokenizer.ggml.add_space_prefix` (llama.cpp default `true` for
    /// SPM). When true, each normal-text run after a special (and the
    /// start of the string) is prefixed with SentencePiece `▁`. Gemma
    /// GGUFs set this to `false` so `<start_of_turn>user` encodes as
    /// `[start_of_turn, user]` not `[start_of_turn, ▁user]`.
    add_space_prefix: bool,
}

/// A merge candidate in the priority queue: pairs of currently-adjacent
/// symbol positions, ordered by score (highest first), with ties
/// broken in favor of the LEFTMOST candidate (smallest `left` symbol
/// index) -- confirmed against llama.cpp's own real
/// `llm_bigram_spm::comparator` (`src/llama-vocab.cpp`):
/// `(l.score < r.score) || (l.score == r.score && l.left > r.left)`.
/// This matters in practice: many real GGUF vocabularies carry an
/// exact-zero score for every merge-derived (non-base) piece, so
/// large stretches of a real tokenization are decided by this tie
/// rule alone, not by score magnitude. `insertion_order` is kept only
/// as a last-resort deterministic tiebreak for the (real, possible)
/// case of two candidates tied on both score AND left index.
struct SpmMergeCandidate {
    score: f32,
    left: usize,
    right: usize,
    insertion_order: u64,
    expected_left_text: String,
    expected_right_text: String,
}

impl PartialEq for SpmMergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.insertion_order == other.insertion_order
    }
}
impl Eq for SpmMergeCandidate {}
impl PartialOrd for SpmMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpmMergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: higher score must compare Greater.
        // On an exact score tie, the LEFTMOST candidate (smaller
        // `left`) must compare Greater, so it pops first -- hence the
        // reversed comparison on `left`. A final tie on `left` too
        // (impossible for real distinct bigrams, kept for a total
        // order) falls back to earliest-queued-first.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
    }
}

impl GgufSpmTokenizer {
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let vocab = ScoredVocab::from_gguf(file)?;
        let special_tokens = SpecialTokenTable::from_gguf(file, vocab.tokens());
        // llama.cpp defaults SPM `add_space_prefix` to true, then lets
        // `tokenizer.ggml.add_space_prefix` override (Gemma sets false).
        let add_space_prefix = match file.metadata("tokenizer.ggml.add_space_prefix") {
            Some(frink_gguf::GgufValue::Bool(v)) => *v,
            _ => true,
        };

        Ok(GgufSpmTokenizer {
            vocab,
            special_tokens,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encodes `text` using SentencePiece's space-replacement
    /// convention (`' '` -> `▁`, plus a leading `▁`) and the
    /// score-prioritized pairwise-merge algorithm described in this
    /// struct's doc comment. Characters with no direct vocabulary
    /// entry are expanded to UTF-8 byte-fallback tokens (`<0xXX>`,
    /// which every real SentencePiece-BPE vocabulary includes for
    /// exactly this purpose) before merging begins.
    ///
    /// Special entries that `specials` lets through (chat-template
    /// markers like `<|user|>`) are first carved out as atomic
    /// substrings, matching real llama.cpp's `tokenizer_st_partition`
    /// behavior, so they're never shattered into byte-fallback pieces;
    /// each remaining raw-text run between them is merged
    /// independently. A leading dummy `▁` is applied to a run only when
    /// [`Self::add_space_prefix`] is true (llama.cpp `add_space_prefix
    /// && is_prev_special` for each fragment).
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        self.special_tokens
            .split(text, specials)
            .into_iter()
            .flat_map(|seg| match seg {
                TextOrSpecial::Special(id) => vec![id],
                TextOrSpecial::Text(t) => self.encode_normal_run(t),
            })
            .collect()
    }

    fn encode_normal_run(&self, text: &str) -> Vec<u32> {
        let replaced: String = text
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let normalized = if self.add_space_prefix {
            format!("\u{2581}{replaced}")
        } else {
            replaced
        };

        let mut symbols: Vec<String> = Vec::new();
        for ch in normalized.chars() {
            let s = ch.to_string();
            if self.vocab.id_of(&s).is_some() {
                symbols.push(s);
            } else {
                for byte in s.as_bytes() {
                    symbols.push(format!("<0x{byte:02X}>"));
                }
            }
        }

        let n = symbols.len();
        if n == 0 {
            return Vec::new();
        }
        let mut nexts: Vec<Option<usize>> = (1..=n)
            .map(|i| if i < n { Some(i) } else { None })
            .collect();
        let mut prevs: Vec<Option<usize>> = (0..n)
            .map(|i| if i == 0 { None } else { Some(i - 1) })
            .collect();
        let mut alive = vec![true; n];

        let mut heap: std::collections::BinaryHeap<SpmMergeCandidate> =
            std::collections::BinaryHeap::new();
        let mut insertion_order = 0u64;

        let try_add_merge = |l: Option<usize>,
                             r: Option<usize>,
                             symbols: &[String],
                             heap: &mut std::collections::BinaryHeap<SpmMergeCandidate>,
                             insertion_order: &mut u64| {
            let (Some(l), Some(r)) = (l, r) else { return };
            let merged = format!("{}{}", symbols[l], symbols[r]);
            // `lookup` hands back the score with the id it belongs to,
            // so there is no second, separately-written id-to-score
            // step here for the Unigram twin to spell differently.
            if let Some((_id, score)) = self.vocab.lookup(&merged) {
                *insertion_order += 1;
                heap.push(SpmMergeCandidate {
                    score,
                    left: l,
                    right: r,
                    insertion_order: *insertion_order,
                    expected_left_text: symbols[l].clone(),
                    expected_right_text: symbols[r].clone(),
                });
            }
        };

        for i in 0..n.saturating_sub(1) {
            try_add_merge(
                Some(i),
                Some(i + 1),
                &symbols,
                &mut heap,
                &mut insertion_order,
            );
        }

        while let Some(candidate) = heap.pop() {
            let (l, r) = (candidate.left, candidate.right);
            if !alive[l] || !alive[r] {
                continue;
            }
            if nexts[l] != Some(r) {
                continue;
            }
            if symbols[l] != candidate.expected_left_text
                || symbols[r] != candidate.expected_right_text
            {
                continue; // stale: content changed since this candidate was queued
            }

            symbols[l] = format!("{}{}", symbols[l], symbols[r]);
            alive[r] = false;
            nexts[l] = nexts[r];
            if let Some(next_of_r) = nexts[r] {
                prevs[next_of_r] = Some(l);
            }

            try_add_merge(prevs[l], Some(l), &symbols, &mut heap, &mut insertion_order);
            try_add_merge(Some(l), nexts[l], &symbols, &mut heap, &mut insertion_order);
        }

        let mut result = Vec::new();
        let mut i = Some(0usize);
        while let Some(idx) = i {
            if alive[idx] {
                result.push(self.vocab.id_of(&symbols[idx]).unwrap_or(0));
            }
            i = nexts[idx];
        }
        result
    }

    /// Reverses a real SentencePiece byte-fallback token (`<0xXX>`,
    /// uppercase hex -- the exact format `encode` produces, see its doc
    /// comment) back to the raw byte it represents. `None` for any
    /// other (normal vocabulary) token.
    fn byte_fallback_value(token: &str) -> Option<u8> {
        let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
        if hex.len() != 2 {
            return None;
        }
        u8::from_str_radix(hex, 16).ok()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// The raw bytes, before any UTF-8 decision is made about them.
    ///
    /// The comment below is about several `<0xXX>` tokens inside ONE
    /// call. The same character can just as easily straddle the
    /// boundary BETWEEN two calls, which is why this is public: a
    /// per-token caller has to do its own buffering, and it cannot do
    /// that from a `String` that has already been made lossy.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        // Byte-fallback tokens must be collected as raw bytes (not
        // pushed as their 6-character literal token string) and
        // UTF-8-decoded together with the rest -- a single real
        // multi-byte UTF-8 character can be split across several
        // consecutive `<0xXX>` tokens, each individually invalid UTF-8
        // on its own. Found and fixed via real-world testing (a real
        // downloaded checkpoint's generated text was printing literal
        // "<0x0A>" instead of a newline).
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(token) = self.vocab.token(id) else {
                continue;
            };
            if let Some(b) = Self::byte_fallback_value(token) {
                bytes.push(b);
            } else {
                bytes.extend(token.replace('\u{2581}', " ").into_bytes());
            }
        }
        bytes
    }
}

/// A real SentencePiece Unigram (ULM) tokenizer, built from a GGUF
/// file's `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` metadata
/// (`tokenizer.ggml.model == "t5"` in GGUF's convention -- confirmed
/// directly against llama.cpp's real vocab-type-loading source
/// (`src/llama-vocab.cpp`'s `tokenizer_model == "t5"` case), not
/// guessed; T5-family models are the real-world users of this tag).
///
/// # How this differs from `GgufSpmTokenizer`
///
/// Both are "SentencePiece" vocabularies, but with entirely different
/// encoding algorithms: `GgufSpmTokenizer` implements SentencePiece's
/// *BPE* model type (a merge-rank table, greedy pairwise merging).
/// Unigram has no merge table at all -- every vocabulary entry carries
/// a real log-probability score, and the *optimal* (highest total
/// log-probability) segmentation of the whole input is found by a
/// forward Viterbi dynamic-programming pass: `best[j]` is the highest-
/// scoring way to reach position `j`, computed as
/// `max over every vocabulary piece P that ends at j` of
/// `best[j - len(P)] + score(P)`. This is reimplemented independently
/// against real llama.cpp source read for this purpose
/// (`src/llama-vocab.cpp`'s `llm_tokenizer_ugm_session` class) -- not
/// copied, but the algorithm (including its unknown-token fallback
/// score and tie-breaking) is transcribed deliberately rather than
/// guessed, since a plausible-looking-but-wrong Viterbi variant would
/// silently produce different segmentations than the model was
/// actually trained to expect.
///
/// Preprocessing matches `GgufSpmTokenizer`'s exactly (`' '` -> `▁`
/// U+2581, plus a leading `▁`) -- both are real SentencePiece
/// conventions, this being the default `add_dummy_prefix=true` /
/// `treat_whitespace_as_suffix=false` behavior. Real SentencePiece
/// models can optionally ship a `precompiled_charsmap` (an auxiliary
/// normalization table, e.g. NFKC folding) via GGUF's
/// `tokenizer.ggml.precompiled_charsmap` key; this implementation does
/// not read or apply it (a real, disclosed scope decision, not an
/// oversight -- llama.cpp's own loader treats this key as optional
/// too, falling back to plain UTF-8 handling when absent).
///
/// Unlike `GgufSpmTokenizer`, Unigram has no byte-fallback token
/// convention in the real reference implementation: a character with
/// no matching vocabulary entry is scored via a fixed unknown-token
/// penalty (`min_score - 10.0`, matching the real
/// `unknown_token_score_penalty` constant) and mapped to the
/// vocabulary's real unknown-token id
/// (`tokenizer.ggml.unknown_token_id`, defaulting to `0` if absent)
/// rather than expanded into raw bytes.
///
/// Real user-defined/control tokens (GGUF's `tokenizer.ggml.token_type`
/// metadata) are not yet given longest-match priority over the
/// Viterbi pass the way the real reference implementation does --
/// deferred alongside `GgufSpmTokenizer`'s equivalent gap
/// (chat-template special-token handling), rather
/// than solved once per tokenizer independently.
///
/// # Verification
///
/// Cross-validated against a real Unigram model trained with the real
/// `sentencepiece` Python library (not a hand-built fixture) --
/// exact-match token-id-sequence comparison across ASCII text,
/// mixed-case, punctuation, digit runs, repeated whitespace, and
/// non-ASCII (accented Latin) text, plus text containing no matching
/// vocabulary substrings at all (exercising the unknown-token
/// fallback repeatedly).
pub struct GgufUnigramTokenizer {
    /// The vocabulary and its per-token scores, checked against each
    /// other at load -- see [`scored_vocab`]. This used to be three
    /// fields spelled out again here, with the Viterbi pass below
    /// indexing `scores[id]` raw while the SPM twin guarded the same
    /// lookup: a short `tokenizer.ggml.scores` array loaded and then
    /// panicked once per request (issue #34).
    vocab: ScoredVocab,
    unk_id: u32,
    /// Longest vocabulary piece, in characters -- bounds the Viterbi
    /// pass's inner loop so it only ever tries substrings that could
    /// possibly be a real vocabulary entry, rather than every possible
    /// substring length.
    max_piece_chars: usize,
    /// `min_score - 10.0`, the real fixed penalty score assigned to the
    /// single-character "unknown token" fallback transition, matching
    /// the real `unknown_token_score_penalty` constant.
    unknown_token_score: f64,
    /// The vocabulary's special entries, carved out of the input before
    /// Viterbi runs on what is left -- see [`special::SpecialTokenTable`].
    special_tokens: SpecialTokenTable,
}

impl GgufUnigramTokenizer {
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Result<Self, TokenizerLoadError> {
        let vocab = ScoredVocab::from_gguf(file)?;

        let unk_id = file
            .metadata("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let max_piece_chars = vocab
            .tokens()
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(1)
            .max(1);
        // A real score, not `+INFINITY`: `ScoredVocab` refuses an empty
        // vocabulary, so this fold always sees at least one entry.
        let unknown_token_score = vocab.min_score() as f64 - 10.0;
        let special_tokens = SpecialTokenTable::from_gguf(file, vocab.tokens());

        Ok(GgufUnigramTokenizer {
            vocab,
            unk_id,
            max_piece_chars,
            unknown_token_score,
            special_tokens,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encodes `text` via the real forward-Viterbi Unigram algorithm
    /// described in this struct's doc comment. Score accumulation uses
    /// `f64` (matching the real reference's `double score_sum`), since
    /// summing many `f32` log-probabilities over a long input can
    /// accumulate enough rounding error to flip which of two
    /// near-tied segmentations looks best.
    ///
    /// Special entries that `specials` lets through (chat-template
    /// markers) are first carved out as atomic substrings; each
    /// remaining raw-text run is Viterbi-segmented independently.
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<u32> {
        self.special_tokens
            .split(text, specials)
            .into_iter()
            .flat_map(|seg| match seg {
                TextOrSpecial::Special(id) => vec![id],
                TextOrSpecial::Text(t) => self.encode_normal_run(t),
            })
            .collect()
    }

    fn encode_normal_run(&self, text: &str) -> Vec<u32> {
        // Real SentencePiece's default normalization rule ("nmt_nfkc",
        // used by the overwhelming majority of trained Unigram models
        // unless a model deliberately opts into the plain "identity"
        // rule) collapses any run of whitespace to a single space and
        // trims leading/trailing whitespace, before the dummy-prefix +
        // space->▁ substitution below -- confirmed empirically against
        // a real trained model, not assumed (a naive per-character
        // space->▁ substitution, `GgufSpmTokenizer`'s approach, gives
        // a different, wrong segmentation here: one `▁` per space
        // instead of one per whitespace *run*). A GGUF file does not
        // carry its normalization rule name as its own metadata key,
        // so this implements the common default rather than something
        // read from the file's own specific config.
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let replaced: String = collapsed
            .chars()
            .map(|c| if c == ' ' { '\u{2581}' } else { c })
            .collect();
        let normalized = format!("\u{2581}{replaced}");
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }

        struct Best {
            token_id: u32,
            from: usize,
            score: f64,
        }
        let mut dp: Vec<Best> = (0..=n)
            .map(|_| Best {
                token_id: 0,
                from: 0,
                score: f64::NEG_INFINITY,
            })
            .collect();
        dp[0].score = 0.0;

        for i in 0..n {
            if dp[i].score == f64::NEG_INFINITY {
                continue; // unreachable position; never happens since the
                          // unknown-token fallback below always advances by 1
            }
            let base = dp[i].score;
            let max_len = self.max_piece_chars.min(n - i);
            for len in 1..=max_len {
                let piece: String = chars[i..i + len].iter().collect();
                // One lookup for id and score together: this line used
                // to index `self.scores[id]` on its own, which is the
                // out-of-bounds panic of issue #34.
                if let Some((id, score)) = self.vocab.lookup(&piece) {
                    let candidate = base + score as f64;
                    let j = i + len;
                    if candidate > dp[j].score {
                        dp[j] = Best {
                            token_id: id,
                            from: i,
                            score: candidate,
                        };
                    }
                }
            }
            let j = i + 1;
            let candidate = base + self.unknown_token_score;
            if candidate > dp[j].score {
                dp[j] = Best {
                    token_id: self.unk_id,
                    from: i,
                    score: candidate,
                };
            }
        }

        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            result.push(dp[pos].token_id);
            pos = dp[pos].from;
        }
        result.reverse();
        result
    }

    /// Reverses `encode`'s `' '` <-> `▁` convention. Unigram has no
    /// byte-fallback token convention (see this struct's doc comment),
    /// so every token here is decoded as plain text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.token(id) {
                out.push_str(&token.replace('\u{2581}', " "));
            }
        }
        out
    }

    /// The raw bytes. Unigram has no byte-fallback convention, so every
    /// token is already whole text and this can never split a
    /// character -- it exists so a per-token caller can treat every
    /// tokenizer the same way.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        self.decode(ids).into_bytes()
    }
}

#[cfg(test)]
mod gguf_vocab_tests {
    use super::*;

    fn load_real_fixture() -> GgufBpeTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-bpe-vocab.gguf"
        );
        let file = frink_gguf::GgufFile::open(path).expect("real vocab fixture must open");
        GgufBpeTokenizer::from_gguf(&file).expect("real vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_downloaded_llama_bpe_vocab() {
        let tok = load_real_fixture();
        // llama-bpe's real vocab is on the order of 128k tokens; assert
        // a loose lower bound so this test doesn't depend on an exact
        // upstream count.
        assert!(
            tok.vocab_size() > 100_000,
            "vocab_size={}",
            tok.vocab_size()
        );
        assert!(tok.has_merges(), "llama-bpe vocab ships a real merge table");
    }

    #[test]
    fn decode_of_known_ids_is_stable() {
        let tok = load_real_fixture();
        // token id 0 exists in every llama-bpe vocab; decoding it must
        // not panic and must return the same string every call.
        let a = tok.decode(&[0]);
        let b = tok.decode(&[0]);
        assert_eq!(a, b);
    }

    #[test]
    fn encode_word_never_panics_on_arbitrary_input() {
        let tok = load_real_fixture();
        for word in ["hello", "", "a", "the quick brown fox", "\u{1f980}"] {
            let ids = tok.encode_word(word);
            // round-trip through decode must not panic either
            let _ = tok.decode(&ids);
        }
    }

    #[test]
    fn encode_sentence_round_trips_through_real_vocab() {
        let tok = load_real_fixture();
        for sentence in [
            "the quick brown fox jumps over the lazy dog",
            "Hello, World! 123",
            "frink is a pure-Rust inference engine.",
        ] {
            let ids = tok.encode(sentence, SpecialTokens::AsText);
            assert!(!ids.is_empty());
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, sentence,
                "full sentence encode/decode through the pre-tokenizer must reproduce the input exactly"
            );
        }
    }

    #[test]
    fn pretokenizer_splits_on_word_boundaries_not_mid_word() {
        let tok = load_real_fixture();
        // "cat dog" pre-tokenizes into ["cat", " dog"] (GPT2 convention:
        // leading space attaches to the following word). Encoding the
        // full sentence and encoding those two pieces separately with
        // encode_word must produce the exact same id sequence -- if
        // frink were still doing one giant merge over the whole
        // string (the pre-pretokenizer behavior), a cross-boundary
        // merge could produce a different sequence.
        let combined = tok.encode("cat dog", SpecialTokens::AsText);
        let mut separate = tok.encode_word("cat");
        separate.extend(tok.encode_word(" dog"));
        assert_eq!(
            combined, separate,
            "pre-tokenized sentence encoding must match word-by-word encoding at real word boundaries"
        );
    }

    #[test]
    fn pretokenizer_keeps_contractions_as_gpt2_does() {
        let tok = load_real_fixture();
        // GPT2's pattern treats "'t" as its own pre-token (from the
        // 's|'t|'re|... alternatives), splitting "don't" into "don" +
        // "'t" pieces before BPE, not "do" + "n't" or a single
        // 6-character chunk. Confirm the pre-tokenizer actually
        // produces that split.
        let pieces: Vec<&str> = tok
            .pretokenize_pattern
            .find_iter("don't")
            .map(|m| m.expect("a fixed pattern cannot fail").as_str())
            .collect();
        assert_eq!(pieces, vec!["don", "'t"]);
    }

    /// **Defect 3, at the tokenizer level.** The pre-tokenizer arms
    /// whose pattern has no catch-all leave text unmatched, and the
    /// encoder used to drop it: `find_iter(..).flat_map(..)` sees only
    /// the matches. That is silent data loss on the PROMPT, not a
    /// different segmentation, so the guard is a byte-for-byte
    /// round-trip rather than an id list.
    ///
    /// The fixture ships `pre = llama-bpe`, whose pattern ends in a
    /// catch-all `\s+` and so has no gaps to lose. The OLMo arm is
    /// swapped in to reproduce the checkpoint that actually broke —
    /// only the split rule changes, the vocabulary and merges stay real.
    #[test]
    fn every_byte_survives_encoding_on_an_arm_with_unmatched_gaps() {
        let mut tok = load_real_fixture();
        tok.pretokenize_pattern = super::pretokenize::regex_for("olmo");

        // A tab, an NBSP, an interior newline, a form feed and a
        // trailing tab: every one of these was unmatched by the OLMo
        // pattern and vanished from the prompt.
        for text in [
            "a\tb\u{a0}c\nd\u{c}e",
            "\tif x:\n\t\treturn 1\n\t \treturn 2\n",
            "para one\n\npara two\n",
            "line one\r\nline two\r\n",
        ] {
            let ids = tok.encode(text, SpecialTokens::AsText);
            assert_eq!(
                tok.decode(&ids),
                text,
                "encoding {text:?} on the olmo arm lost input bytes"
            );
        }

        // And the loss was real: the tab between `a` and `b` is its own
        // token, not absorbed into either neighbour.
        let ids = tok.encode("a\tb", SpecialTokens::AsText);
        assert_eq!(
            ids.len(),
            3,
            "a, the tab, b — the tab is a token of its own"
        );
    }

    #[test]
    fn ascii_word_round_trips_through_real_vocab_encode_decode() {
        let tok = load_real_fixture();
        for word in ["hello", "frink", "test", "quick brown fox"] {
            let ids = tok.encode_word(word);
            assert!(!ids.is_empty(), "encoding {word:?} produced no tokens");
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, word,
                "round-trip through the real vocab's encode/decode should reproduce ASCII text exactly"
            );
        }
    }

    #[test]
    fn multibyte_utf8_round_trips_through_real_vocab_encode_decode() {
        let tok = load_real_fixture();
        for word in ["caf\u{e9}", "\u{1f980}", "\u{4e2d}\u{6587}"] {
            let ids = tok.encode_word(word);
            let decoded = tok.decode(&ids);
            assert_eq!(
                decoded, word,
                "byte-level BPE must round-trip arbitrary UTF-8, not just ASCII"
            );
        }
    }

    #[test]
    fn gpt2_remap_matches_known_reference_points() {
        // These are well-known fixed points of the real GPT-2
        // byte-to-unicode table (verifiable against OpenAI's published
        // encoder.py): printable ASCII '!' (0x21) maps to itself, and
        // the space byte (0x20), which is NOT in the "already
        // printable" ranges, maps to U+0120 ("\u{120}", conventionally
        // rendered as "Ġ" in BPE merge tables).
        let (fwd, rev) = super::gpt2_byte_to_unicode();
        assert_eq!(fwd[0x21], '!');
        assert_eq!(fwd[0x20], '\u{120}');
        assert_eq!(rev[&'!'], 0x21);
        assert_eq!(rev[&'\u{120}'], 0x20);
    }

    #[test]
    fn real_vocab_uses_gpt2_space_remap_in_its_own_tokens() {
        // If frink's remap table matches the real llama-bpe vocab's
        // own convention, at least one real vocabulary entry should
        // start with the remapped-space character (a leading-space
        // word piece, extremely common in any GPT2-style BPE vocab).
        let tok = load_real_fixture();
        let has_space_prefixed_token = tok.id_to_token.iter().any(|t| t.starts_with('\u{120}'));
        assert!(
            has_space_prefixed_token,
            "expected at least one real vocab token starting with the GPT2 remapped-space character"
        );
    }
}

#[cfg(test)]
mod gguf_spm_tests {
    use super::*;

    fn load_real_fixture() -> GgufSpmTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf"
        );
        let file = frink_gguf::GgufFile::open(path).expect("real SPM vocab fixture must open");
        GgufSpmTokenizer::from_gguf(&file)
            .expect("real SPM vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_downloaded_llama_spm_vocab() {
        let tok = load_real_fixture();
        assert_eq!(
            tok.vocab_size(),
            32000,
            "the real LLaMA-1/2 tokenizer vocab is exactly 32000 tokens"
        );
    }

    #[test]
    fn matches_known_reference_encodings() {
        let tok = load_real_fixture();
        assert_eq!(
            tok.encode("Hello world", SpecialTokens::AsText),
            vec![15043, 3186]
        );
        assert_eq!(
            tok.encode(" Hello world", SpecialTokens::AsText),
            vec![29871, 15043, 3186]
        );
        assert_eq!(
            tok.encode("Hello World", SpecialTokens::AsText),
            vec![15043, 2787]
        );
    }

    /// Real regression test, found serving a real chat checkpoint:
    /// chat-template control tokens (`<|user|>`, `<|assistant|>`) must
    /// be recognized as atomic vocabulary entries, not shattered into
    /// byte-fallback pieces. Uses a real, hand-built GGUF fixture with
    /// genuine `tokenizer.ggml.token_type` CONTROL entries from the
    /// fixture generator, not the
    /// downloaded real-LLaMA fixture above (which carries no
    /// `token_type` array at all).
    #[test]
    fn chat_template_control_tokens_are_encoded_atomically_not_shattered() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/spm-special-tokens-test-vocab.gguf"
        );
        let file = frink_gguf::GgufFile::open(path).expect("fixture must open");
        let tok = GgufSpmTokenizer::from_gguf(&file).expect("fixture must parse");

        let user_id = 269u32;
        let assistant_id = 270u32;
        let ids = tok.encode("<|user|>hello<|assistant|>", SpecialTokens::Parse);

        assert_eq!(ids.first().copied(), Some(user_id), "ids={ids:?}");
        assert_eq!(ids.last().copied(), Some(assistant_id), "ids={ids:?}");
        // The control tokens' own byte-fallback expansions must NOT
        // appear anywhere in the output -- they'd show up as a long
        // run of ids >= the byte-fallback range if the old shattering
        // bug were still present.
        assert!(
            !ids[1..ids.len() - 1].contains(&user_id)
                && !ids[1..ids.len() - 1].contains(&assistant_id),
            "control tokens must appear exactly once each, at the boundaries: ids={ids:?}"
        );
    }

    #[test]
    fn byte_fallback_handles_control_characters() {
        let tok = load_real_fixture();
        assert_eq!(
            tok.encode("\t", SpecialTokens::AsText),
            vec![29871, 12],
            "tab must byte-fallback to <0x09> = token 12"
        );
        assert_eq!(
            tok.encode("\n", SpecialTokens::AsText),
            vec![29871, 13],
            "newline must byte-fallback to <0x0A> = token 13"
        );
    }

    /// The strongest test in this file: every one of llama.cpp's own
    /// 45 CI test cases for this exact vocabulary
    /// (`tests/fixtures/llama-spm-vocab.gguf.inp`/`.out`, downloaded
    /// directly from `ggml-org/llama.cpp`), covering ASCII, whitespace
    /// runs of every length, control characters, CJK/Khmer/Vietnamese
    /// text, emoji (including a ZWJ sequence), and mixed-script text,
    /// must produce EXACTLY the token IDs llama.cpp's own tokenizer
    /// produces for the same inputs. This is what caught the
    /// stale-merge-candidate bug described in `GgufSpmTokenizer`'s doc
    /// comment during development.
    #[test]
    fn matches_llama_cpp_full_reference_test_suite_exactly() {
        let tok = load_real_fixture();

        let inp_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf.inp"
        );
        let out_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/llama-spm-vocab.gguf.out"
        );
        let inp_raw = std::fs::read_to_string(inp_path).expect("reference .inp file must exist");
        let out_raw = std::fs::read_to_string(out_path).expect("reference .out file must exist");

        let marker = "__ggml_vocab_test__\n";
        let mut inputs: Vec<&str> = inp_raw.split(marker).collect();
        // The split produces a leading/trailing artifact from the
        // marker boundaries; drop empty fragments and any trailing
        // newline each fragment carries from the format.
        inputs.retain(|s| !s.is_empty());
        let inputs: Vec<String> = inputs
            .iter()
            .map(|s| s.strip_suffix('\n').unwrap_or(s).to_string())
            .collect();

        let outputs: Vec<&str> = out_raw.split('\n').collect();

        assert!(
            inputs.len() >= 40,
            "expected the full ~45-case reference suite, got {}",
            inputs.len()
        );

        let mut checked = 0;
        for (i, text) in inputs.iter().enumerate() {
            let Some(expected_line) = outputs.get(i) else {
                break;
            };
            let expected_line = expected_line.trim();
            if expected_line.is_empty() {
                continue;
            }
            let expected: Vec<u32> = expected_line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let got = tok.encode(text, SpecialTokens::AsText);
            assert_eq!(got, expected, "case #{i}: text={text:?}");
            checked += 1;
        }
        assert!(
            checked >= 40,
            "expected to actually check at least 40 real cases, only checked {checked}"
        );
    }

    #[test]
    fn decode_reverses_encode_for_ascii_text() {
        let tok = load_real_fixture();
        // SentencePiece's real convention (confirmed by the reference
        // suite above) always prepends a dummy leading space before
        // tokenizing, so decoding round-trips to " Hello world" (WITH
        // a leading space), not "Hello world" -- this is genuine
        // LLaMA-tokenizer behavior, not a bug in this test or the
        // encoder; downstream text-generation code conventionally
        // strips exactly one leading space from decoded output, but
        // the raw decode legitimately includes it.
        let text = "Hello world";
        let ids = tok.encode(text, SpecialTokens::AsText);
        assert_eq!(tok.decode(&ids), " Hello world");
    }

    #[test]
    fn decode_reverses_byte_fallback_tokens_to_the_real_raw_bytes() {
        // Real bug found via real-world testing:
        // decode() used to emit the literal 6-character token string
        // "<0x0A>" instead of an actual newline byte.
        let tok = load_real_fixture();
        // `encode` always prepends a dummy leading space (SentencePiece
        // convention, see `decode_reverses_encode_for_ascii_text`
        // above), so the decoded round-trip carries it too.
        let newline_id = tok.encode("\n", SpecialTokens::AsText);
        assert_eq!(tok.decode(&newline_id), " \n");

        // A multi-byte UTF-8 character split across several
        // consecutive byte-fallback tokens must still decode correctly
        // once reassembled -- not as mojibake or individually-invalid
        // UTF-8 fragments.
        let emoji = "🦀";
        let ids = tok.encode(emoji, SpecialTokens::AsText);
        assert_eq!(tok.decode(&ids), format!(" {emoji}"));
    }
}

#[cfg(test)]
mod gguf_unigram_tests {
    use super::*;

    /// Real trained SentencePiece Unigram model (100 pieces, trained
    /// with the real `sentencepiece` Python library on a small text
    /// corpus through a fixture generator), not a
    /// hand-guessed vocabulary.
    fn load_real_fixture() -> GgufUnigramTokenizer {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/unigram-test-vocab.gguf"
        );
        let file = frink_gguf::GgufFile::open(path).expect("real Unigram vocab fixture must open");
        GgufUnigramTokenizer::from_gguf(&file)
            .expect("real Unigram vocab fixture must parse as a tokenizer")
    }

    #[test]
    fn loads_real_trained_unigram_vocab() {
        let tok = load_real_fixture();
        assert_eq!(tok.vocab_size(), 100);
    }

    /// Cross-validated against the exact same trained model's own
    /// `sentencepiece.SentencePieceProcessor.Encode` output -- not a
    /// hand-computed expectation. Covers ASCII, mixed case, digit runs,
    /// repeated whitespace, punctuation, and non-ASCII (accented Latin)
    /// text, plus a string with no real vocabulary substrings at all
    /// (exercising the unknown-token fallback repeatedly, including
    /// consecutive unknown tokens).
    #[test]
    fn matches_real_sentencepiece_reference_encodings() {
        let tok = load_real_fixture();
        let cases: &[(&str, &[u32])] = &[
            ("hello world", &[3, 63, 4, 95, 8, 3, 36, 14, 11]),
            (
                "The quick brown fox",
                &[34, 3, 89, 10, 65, 70, 57, 49, 73, 12, 54, 8, 30],
            ),
            (
                "Testing unicode: café",
                &[74, 44, 20, 35, 47, 4, 83, 3, 62, 13, 25, 18],
            ),
            (
                "Numbers 12345",
                &[3, 86, 50, 15, 53, 5, 3, 75, 76, 77, 81, 82],
            ),
            ("a", &[58]),
            (
                "   multiple   spaces   ",
                &[55, 10, 14, 64, 16, 99, 22, 3, 5, 99, 13, 27, 5],
            ),
            (
                "Zurich naive resume",
                &[3, 88, 10, 7, 16, 51, 38, 16, 33, 60, 4, 5, 50, 4],
            ),
            (
                "punctuation! test? yes.",
                &[24, 10, 72, 43, 29, 80, 3, 64, 44, 84, 3, 28, 4, 5, 6],
            ),
            (
                "unknown_gibberish_xyz_qqq_zzz",
                &[
                    3, 10, 12, 70, 12, 8, 73, 12, 0, 17, 16, 15, 15, 53, 56, 63, 0, 30, 28, 90, 0,
                    89, 89, 89, 0, 90, 90, 90,
                ],
            ),
        ];
        for (text, expected) in cases {
            let got = tok.encode(text, SpecialTokens::AsText);
            assert_eq!(&got, expected, "text={text:?}");
        }
    }

    #[test]
    fn decode_reverses_encode_for_ascii_text() {
        let tok = load_real_fixture();
        let ids = tok.encode("hello world", SpecialTokens::AsText);
        // encode's leading dummy `▁` decodes back to a leading space,
        // same SentencePiece convention as GgufSpmTokenizer.
        assert_eq!(tok.decode(&ids), " hello world");
    }

    /// A hundred tokens and three scores.
    fn short_scores_gguf() -> scored_vocab::MetadataOnlyGguf {
        let tokens: Vec<String> = (0..100).map(|i| format!("\u{2581}piece{i}")).collect();
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        scored_vocab::MetadataOnlyGguf::new()
            .with_tokens(&refs)
            .with_scores(&[-1.0, -2.0, -3.0])
    }

    /// Issue #34, and the reason the two tokenizers now share one
    /// vocabulary type. This file used to LOAD CLEANLY -- accepted by
    /// `/admin/models/load`, listed as the loaded model -- and then
    /// panic with an index-out-of-bounds inside the generation task on
    /// the first prompt whose Viterbi pass matched a piece with id >=
    /// 3, once per request, forever. The SPM twin survived the same
    /// file only because its copy of the lookup happened to be the
    /// guarded spelling.
    ///
    /// Both must now refuse it at load, and refuse it the same way:
    /// one lookup, one check, no room for the two to disagree again.
    #[test]
    fn a_scores_array_too_short_for_the_vocabulary_is_refused_at_load_by_both_tokenizers() {
        let file = short_scores_gguf();
        let unigram = GgufUnigramTokenizer::from_gguf(&file)
            .err()
            .expect("unigram must refuse a vocabulary its scores do not cover");
        assert!(
            matches!(
                unigram,
                TokenizerLoadError::ScoresVocabLengthMismatch {
                    tokens: 100,
                    scores: 3
                }
            ),
            "unigram={unigram:?}"
        );
        let spm = GgufSpmTokenizer::from_gguf(&file)
            .err()
            .expect("spm must refuse the same file the same way");
        assert!(
            matches!(
                spm,
                TokenizerLoadError::ScoresVocabLengthMismatch {
                    tokens: 100,
                    scores: 3
                }
            ),
            "spm={spm:?}"
        );
    }

    /// The refusal above must be about the DISAGREEMENT, not about
    /// synthetic vocabularies in general: the same 100 pieces with 100
    /// scores load and encode, reaching ids far past the three the
    /// broken file carried.
    #[test]
    fn the_same_vocabulary_with_one_score_per_token_loads_and_encodes() {
        let tokens: Vec<String> = (0..100).map(|i| format!("\u{2581}piece{i}")).collect();
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let scores: Vec<f32> = (0..100).map(|i| -(i as f32)).collect();
        let file = scored_vocab::MetadataOnlyGguf::new()
            .with_tokens(&refs)
            .with_scores(&scores);
        let tok = GgufUnigramTokenizer::from_gguf(&file).expect("lengths agree");
        assert_eq!(tok.vocab_size(), 100);
        let ids = tok.encode("piece97", SpecialTokens::AsText);
        assert!(
            ids.contains(&97),
            "the piece with the highest id must be reachable: ids={ids:?}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_round_trips_exactly() {
        let text = "hello frink";
        let ids = ByteTokenizer::encode(text);
        assert_eq!(ids.len(), text.len());
        assert_eq!(ByteTokenizer::decode(&ids), text);
    }

    #[test]
    fn utf8_multibyte_round_trips_exactly() {
        let text = "caffe\u{300} \u{1f980}"; // combining accent + emoji, multi-byte UTF-8
        let ids = ByteTokenizer::encode(text);
        assert_eq!(ByteTokenizer::decode(&ids), text);
    }

    #[test]
    fn all_ids_are_within_byte_vocab_range() {
        let ids = ByteTokenizer::encode("mixed ASCII and \u{00e9}\u{00e8} text");
        assert!(ids
            .iter()
            .all(|&id| (id as usize) < ByteTokenizer::VOCAB_SIZE));
    }

    #[test]
    fn empty_string_round_trips() {
        assert_eq!(ByteTokenizer::encode(""), Vec::<u32>::new());
        assert_eq!(ByteTokenizer::decode(&[]), "");
    }

    #[test]
    fn out_of_range_ids_are_dropped_not_corrupting() {
        // 300 is outside the byte vocab; decode should simply skip it
        // rather than panicking or wrapping into a wrong byte.
        let decoded = ByteTokenizer::decode(&[104, 105, 300, 33]); // "hi" + garbage + "!"
        assert_eq!(decoded, "hi!");
    }
}

#[cfg(test)]
mod eog_tests {
    use super::*;
    use frink_gguf::{GgufValue, TensorInfo, TensorSource};
    use std::collections::HashMap;

    struct MetaOnly(HashMap<String, GgufValue>);

    impl TensorSource for MetaOnly {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            self.0.get(key)
        }
        fn find_tensor(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], frink_gguf::GgufError> {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<frink_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            frink_gguf::GgufError,
        > {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
    }

    fn source(tokens: &[&str], kv: &[(&str, u64)]) -> MetaOnly {
        let mut m = HashMap::new();
        m.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::Array(
                tokens
                    .iter()
                    .map(|t| GgufValue::String((*t).to_string()))
                    .collect(),
            ),
        );
        for (k, v) in kv {
            m.insert((*k).to_string(), GgufValue::U32(*v as u32));
        }
        MetaOnly(m)
    }

    /// A vocabulary described only by the three metadata keys
    /// `should_add_bos_token` reads.
    fn vocab_meta(model: &str, pre: Option<&str>, add_bos: Option<bool>) -> MetaOnly {
        let mut m = HashMap::new();
        m.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String(model.to_string()),
        );
        if let Some(pre) = pre {
            m.insert(
                "tokenizer.ggml.pre".to_string(),
                GgufValue::String(pre.to_string()),
            );
        }
        if let Some(v) = add_bos {
            m.insert(
                "tokenizer.ggml.add_bos_token".to_string(),
                GgufValue::Bool(v),
            );
        }
        MetaOnly(m)
    }

    /// **Defect 4.** llama.cpp sets `add_bos = true` for the whole
    /// `LLAMA_VOCAB_PRE_TYPE_LLAMA3` group (`llama-vocab.cpp`, the
    /// `tokenizer_pre == "llama-bpe"` arm), and Llama-3.x GGUFs ship no
    /// explicit `tokenizer.ggml.add_bos_token`, so a missing group
    /// member is a prompt one `<|begin_of_text|>` short of llama.cpp's
    /// on every raw completion.
    #[test]
    fn the_llama_bpe_group_takes_bos_even_with_no_metadata_flag() {
        for pre in [
            "llama3",
            "llama-v3",
            "llama-bpe",
            "falcon3",
            "falcon-h1",
            "pixtral",
            "midm-2.0",
            "lfm2",
            "jina-v5-nano",
            "tekken",
            "chameleon",
        ] {
            assert!(
                should_add_bos_token(&vocab_meta("gpt2", Some(pre), None)),
                "llama.cpp sets add_bos for pre={pre}"
            );
        }
    }

    /// The other half of the same rule, so the fix cannot be "return
    /// true": BPE arms outside that group leave `add_bos` false, and
    /// Qwen2's `bos_token_id` is `<|endoftext|>` — prepending it poisons
    /// greedy decode.
    #[test]
    fn other_bpe_pretokenizers_still_do_not_take_bos() {
        for pre in ["qwen2", "deepseek-r1-qwen", "gpt-4o", "olmo", "gpt-2", ""] {
            assert!(
                !should_add_bos_token(&vocab_meta("gpt2", Some(pre), None)),
                "llama.cpp leaves add_bos false for pre={pre}"
            );
        }
    }

    /// An explicit `tokenizer.ggml.add_bos_token` still wins in both
    /// directions — the group default only applies when the key is
    /// absent, which is what makes this a *default* rather than an
    /// override.
    #[test]
    fn an_explicit_add_bos_flag_beats_the_pretokenizer_default() {
        assert!(!should_add_bos_token(&vocab_meta(
            "gpt2",
            Some("llama-bpe"),
            Some(false)
        )));
        assert!(should_add_bos_token(&vocab_meta(
            "gpt2",
            Some("qwen2"),
            Some(true)
        )));
        // SPM still defaults to true with no flag and no pre.
        assert!(should_add_bos_token(&vocab_meta("llama", None, None)));
    }

    /// The failure this exists to stop: a Llama-3 chat checkpoint whose
    /// `eos_token_id` is `<|end_of_text|>` while turns actually end with
    /// `<|eot_id|>`. Stopping only on the metadata id runs the model past
    /// its own turn and it starts interviewing itself.
    #[test]
    fn turn_enders_count_even_when_they_are_not_the_metadata_eos() {
        let src = source(
            &["hello", "<|end_of_text|>", "<|eot_id|>", "world"],
            &[("tokenizer.ggml.eos_token_id", 1)],
        );
        let eog = eog_token_ids(&src);
        assert!(eog.contains(&1), "metadata eos");
        assert!(eog.contains(&2), "<|eot_id|> ends the turn");
        assert!(
            !eog.contains(&0) && !eog.contains(&3),
            "ordinary tokens are not EOG"
        );
    }

    /// gemma-4 ends on `<turn|>`; both it and `<eos>` are in llama.cpp's
    /// list, so a gemma checkpoint must stop on either.
    #[test]
    fn gemma_style_turn_and_eos_are_both_end_of_generation() {
        let src = source(&["<eos>", "<turn|>", "x"], &[]);
        let eog = eog_token_ids(&src);
        assert!(eog.contains(&0) && eog.contains(&1));
        assert!(!eog.contains(&2));
    }

    /// `eot`/`eom` ids are folded in even when the vocabulary spells them
    /// something llama.cpp's literal list does not know.
    #[test]
    fn eot_and_eom_metadata_ids_are_included() {
        let src = source(
            &["a", "b", "c"],
            &[
                ("tokenizer.ggml.eot_token_id", 1),
                ("tokenizer.ggml.eom_token_id", 2),
            ],
        );
        let eog = eog_token_ids(&src);
        assert!(eog.contains(&1) && eog.contains(&2));
    }

    /// A file with neither the ids nor any known name yields an empty
    /// set, so callers keep their previous `eos_id`-only behaviour rather
    /// than stopping on something arbitrary.
    #[test]
    fn a_file_with_nothing_to_go_on_yields_no_stop_tokens() {
        let src = source(&["a", "b"], &[]);
        assert!(eog_token_ids(&src).is_empty());
    }

    /// The case a template-evaluating loader creates: gemma-3, Mistral,
    /// Phi-3 and DeepSeek-R1-Distill all open their real
    /// `tokenizer.chat_template` with `{{ bos_token }}`, so the encoded
    /// prompt already starts with the BOS id before the loader gets a
    /// look. Measured on the local corpus by
    /// `tests/bos_policy.rs::sweep_local_gguf_bos_policy`: 6 of 26
    /// checkpoints double their BOS if this is an unconditional insert.
    #[test]
    fn a_template_that_already_emitted_bos_is_not_given_a_second_one() {
        let mut ids = vec![2u32, 105, 2364];
        prepend_bos(&mut ids, Some(2));
        assert_eq!(ids, vec![2, 105, 2364]);
    }

    /// The other half of the same rule: Unsloth strips `{{ bos_token }}`
    /// out of the templates it exports (TinyLlama's checked-in template
    /// is the local example), so on those checkpoints nobody adds BOS
    /// unless the loader does.
    #[test]
    fn a_template_that_stripped_bos_gets_one_from_the_loader() {
        let mut ids = vec![529u32, 29989];
        prepend_bos(&mut ids, Some(1));
        assert_eq!(ids, vec![1, 529, 29989]);
    }

    /// `None` is the `should_add_bos_token` gate having said no — BPE
    /// vocabularies ship a `bos_token_id` they never prepend, and
    /// Qwen2-MoE's is `<|endoftext|>`.
    #[test]
    fn a_vocabulary_that_does_not_take_bos_gets_nothing() {
        let mut ids = vec![151644u32, 872];
        prepend_bos(&mut ids, None);
        assert_eq!(ids, vec![151644, 872]);
    }
}
