//! The fill-in-the-middle tokens of one vocabulary: which ids mark the
//! prefix, the suffix and the middle, and the three repo-level markers
//! beside them.
//!
//! Transcribed from llama.cpp's `src/llama-vocab.cpp` at the pin, in its
//! two steps:
//!
//! 1. **The metadata keys** (`special_token_types`, around `:2933`). The
//!    current `tokenizer.ggml.fim_*_token_id` keys are read first and
//!    the deprecated `prefix_token_id` / `suffix_token_id` /
//!    `middle_token_id` after them, so a file carrying both ends up with
//!    the deprecated value, exactly as upstream's loop leaves it. An id
//!    past the end of the vocabulary is ignored ("bad special token").
//! 2. **Detection by text** (`auto-detect special tokens by text`, around
//!    `:3093`), for every kind the keys left unset: `<|fim_prefix|>`
//!    (Qwen), `<fim_prefix>` (Granite), `<｜fim▁begin｜>` (DeepSeek),
//!    `▁<PRE>` (CodeLlama) and the rest of upstream's lists below.
//!
//! One thing upstream leaves unspecified is pinned here. llama.cpp walks
//! `token_to_id`, an `unordered_map`, and keeps the FIRST entry its
//! iteration order reaches when two texts of one kind are both present
//! (`<PRE>` and `▁<PRE>`, say). That order is the hash map's, so this
//! module keeps the LOWEST id instead: the same answer on every run, and
//! upstream's answer whenever a vocabulary has one candidate per kind,
//! which is every checkpoint those lists were written for.
//!
//! What llama.cpp also does with these ids and this module does not:
//! it promotes each detected token to `CONTROL` and adds `fim_pad`,
//! `fim_rep` and `fim_sep` to the end-of-generation set. Both change how
//! every route tokenizes and stops, not only `/infill`, so they are not
//! folded in here without the parity evidence such a change needs; the
//! server's API notes say so.

/// The six FIM ids a vocabulary may carry. `None` is llama.cpp's
/// `LLAMA_TOKEN_NULL`: the vocabulary has no such token.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FimTokens {
    /// `FIM_PRE`, the marker before the code ahead of the hole.
    pub pre: Option<u32>,
    /// `FIM_SUF`, the marker before the code after the hole.
    pub suf: Option<u32>,
    /// `FIM_MID`, after which the model writes the middle.
    pub mid: Option<u32>,
    /// `FIM_PAD`.
    pub pad: Option<u32>,
    /// `FIM_REP`, which opens the repo-level pattern.
    pub rep: Option<u32>,
    /// `FIM_SEP`, the file separator of the repo-level pattern.
    pub sep: Option<u32>,
}

/// One FIM kind: its metadata keys in the order upstream reads them,
/// and the texts it is detected by when no key set it.
struct Kind {
    keys: &'static [&'static str],
    texts: &'static [&'static str],
}

/// llama.cpp's lists, in [`FimTokens`]' field order. The comments name
/// the family upstream wrote each text for.
const KINDS: [Kind; 6] = [
    Kind {
        keys: &[
            "tokenizer.ggml.fim_pre_token_id",
            "tokenizer.ggml.prefix_token_id",
        ],
        texts: &[
            "<|fim_prefix|>", // Qwen
            "<fim-prefix>",
            "<fim_prefix>",    // Granite
            "<｜fim▁begin｜>", // DeepSeek
            "<PRE>",
            "▁<PRE>",               // CodeLlama
            "<|code_prefix|>",      // GLM-4.5
            "<|prefix|>",           // Falcon-H1-Tiny-Coder
            "<|plamo:fim_prefix|>", // PLaMo-3
        ],
    },
    Kind {
        keys: &[
            "tokenizer.ggml.fim_suf_token_id",
            "tokenizer.ggml.suffix_token_id",
        ],
        texts: &[
            "<|fim_suffix|>", // Qwen
            "<fim-suffix>",
            "<fim_suffix>",   // Granite
            "<｜fim▁hole｜>", // DeepSeek
            "<SUF>",
            "▁<SUF>",               // CodeLlama
            "<|code_suffix|>",      // GLM-4.5
            "<|suffix|>",           // Falcon-H1-Tiny-Coder
            "<|plamo:fim_suffix|>", // PLaMo-3
        ],
    },
    Kind {
        keys: &[
            "tokenizer.ggml.fim_mid_token_id",
            "tokenizer.ggml.middle_token_id",
        ],
        texts: &[
            "<|fim_middle|>", // Qwen
            "<fim-middle>",
            "<fim_middle>",  // Granite
            "<｜fim▁end｜>", // DeepSeek
            "<MID>",
            "▁<MID>",               // CodeLlama
            "<|code_middle|>",      // GLM-4.5
            "<|middle|>",           // Falcon-H1-Tiny-Coder
            "<|plamo:fim_middle|>", // PLaMo-3
        ],
    },
    Kind {
        keys: &["tokenizer.ggml.fim_pad_token_id"],
        texts: &[
            "<|fim_pad|>", // Qwen
            "<fim-pad>",
            "<fim_pad>", // Granite
            "<PAD>",
            "[PAD]", // Kimi-K2
        ],
    },
    Kind {
        keys: &["tokenizer.ggml.fim_rep_token_id"],
        texts: &[
            "<|fim_repo|>", // Qwen
            "<|repo_name|>",
            "<fim-repo>",
            "<REPO>",
            "<reponame>", // Granite
        ],
    },
    Kind {
        keys: &["tokenizer.ggml.fim_sep_token_id"],
        texts: &[
            "<|file_sep|>",             // Qwen
            "<|plamo:file_separator|>", // PLaMo-2/3
        ],
    },
];

impl FimTokens {
    /// The FIM ids of a GGUF vocabulary: its metadata keys, then
    /// detection by text over `tokenizer.ggml.tokens`.
    pub fn from_gguf(file: &impl frink_gguf::TensorSource) -> Self {
        let id_to_token: Vec<&str> = match file.metadata("tokenizer.ggml.tokens") {
            Some(frink_gguf::GgufValue::Array(items)) => items
                .iter()
                .map(|v| match v {
                    frink_gguf::GgufValue::String(s) => s.as_str(),
                    _ => "",
                })
                .collect(),
            _ => Vec::new(),
        };
        Self::detect(|key| file.metadata_u64(key), &id_to_token)
    }

    /// Both of upstream's steps over a key lookup and the vocabulary in
    /// id order. Separate from [`Self::from_gguf`] so a test can hand it
    /// a vocabulary without building a file.
    pub fn detect(key: impl Fn(&str) -> Option<u64>, id_to_token: &[&str]) -> Self {
        let n_vocab = id_to_token.len() as u64;
        let ids: Vec<Option<u32>> = KINDS
            .iter()
            .map(|kind| {
                // Every key in order, a later one overwriting an earlier
                // one, an out-of-range one ignored: upstream's loop.
                let from_keys = kind
                    .keys
                    .iter()
                    .filter_map(|k| key(k))
                    .filter(|&id| id < n_vocab)
                    .last()
                    .map(|id| id as u32);
                from_keys.or_else(|| {
                    id_to_token
                        .iter()
                        .position(|text| kind.texts.contains(text))
                        .map(|id| id as u32)
                })
            })
            .collect();
        FimTokens {
            pre: ids[0],
            suf: ids[1],
            mid: ids[2],
            pad: ids[3],
            rep: ids[4],
            sep: ids[5],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_keys(_: &str) -> Option<u64> {
        None
    }

    /// Qwen2.5-Coder carries no FIM keys at all; its six markers are
    /// found by their text, which is the case upstream's detection was
    /// written for.
    #[test]
    fn a_vocabulary_with_qwens_markers_is_detected_by_text() {
        let vocab = [
            "a",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|fim_pad|>",
            "<|repo_name|>",
            "<|file_sep|>",
        ];
        let fim = FimTokens::detect(no_keys, &vocab);
        assert_eq!(
            fim,
            FimTokens {
                pre: Some(1),
                suf: Some(3),
                mid: Some(2),
                pad: Some(4),
                rep: Some(5),
                sep: Some(6),
            }
        );
    }

    /// The deprecated key is read after the current one, so it wins:
    /// that is upstream's loop order, odd as it is.
    #[test]
    fn a_key_beats_the_text_and_the_deprecated_key_is_read_last() {
        let vocab = ["<|fim_prefix|>", "x", "y", "z"];
        let keys = |k: &str| match k {
            "tokenizer.ggml.fim_pre_token_id" => Some(2),
            "tokenizer.ggml.prefix_token_id" => Some(3),
            "tokenizer.ggml.fim_suf_token_id" => Some(1),
            _ => None,
        };
        let fim = FimTokens::detect(keys, &vocab);
        assert_eq!(fim.pre, Some(3), "the deprecated key overwrites");
        assert_eq!(fim.suf, Some(1));
        assert_eq!(fim.mid, None);
    }

    /// An id past the vocabulary is upstream's "bad special token": the
    /// key is ignored and detection by text still runs.
    #[test]
    fn an_out_of_range_key_is_ignored() {
        let vocab = ["<PRE>", "b"];
        let keys = |k: &str| (k == "tokenizer.ggml.fim_pre_token_id").then_some(99);
        assert_eq!(FimTokens::detect(keys, &vocab).pre, Some(0));
    }

    /// A vocabulary with none of the texts and no keys has no FIM
    /// tokens, which is what makes `/infill` refuse it by name.
    #[test]
    fn a_vocabulary_without_markers_has_none() {
        let vocab = ["<s>", "</s>", "hello", "<|im_end|>"];
        assert_eq!(FimTokens::detect(no_keys, &vocab), FimTokens::default());
    }

    /// Two texts of one kind: the lower id, every time.
    #[test]
    fn two_candidates_of_one_kind_resolve_to_the_lower_id() {
        let vocab = ["q", "▁<PRE>", "<PRE>"];
        assert_eq!(FimTokens::detect(no_keys, &vocab).pre, Some(1));
    }
}
