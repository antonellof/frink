//! The tokenizer `frink run` picks from the GGUF's `tokenizer.ggml.model`.
//! Moved out of `run/mod.rs` verbatim.

use super::*;

/// The tokenizer a GGUF's own metadata names, or a refusal.
///
/// Four call sites had a byte-by-byte copy of this match, each falling
/// back to `CliTokenizer::Byte` on anything unrecognised. That fallback
/// produced FLUENT GARBAGE: the model was fed ids from a vocabulary it
/// was never trained on, so it generated confidently and wrongly with
/// nothing in the output saying so. This project refuses everywhere
/// else rather than compute something different; the tokenizer was the
/// one place that did not.
///
/// The `Byte` variant went with it: the CLI has no synthetic-weight
/// path, so with the fallback gone nothing could construct it.
pub(super) fn cli_tokenizer_from_gguf(file: &ShardedGguf) -> anyhow::Result<CliTokenizer> {
    match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => Ok(CliTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(
            file,
        )?))),
        Some("llama") => Ok(CliTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?)),
        Some("t5") => Ok(CliTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(
            file,
        )?)),
        Some("plamo2") => Ok(CliTokenizer::Plamo2(Box::new(
            GgufPlamo2Tokenizer::from_gguf(file)?,
        ))),
        // `bert` is NOT here because the tokenizer is missing -- frink
        // has WordPiece, and it is byte-exact against llama.cpp
        // (`frink parity`). It is here because this is the *generation*
        // path and a `bert` checkpoint is an encoder: no output head,
        // no logits, nothing to sample. The refusal names where it can
        // be used instead rather than repeating a claim that stopped
        // being true.
        Some("bert") => anyhow::bail!(
            "this checkpoint's tokenizer is `bert` (WordPiece), which means it is a BERT-family \
             ENCODER: it has no output head and cannot generate text, so there is nothing for \
             `frink run` to sample. Frink can embed with it: start frink-server with \
             FRINK_EMBEDDING_MODEL_PATH pointing at this file and POST /v1/embeddings."
        ),
        Some(known @ ("rwkv" | "none")) => anyhow::bail!(
            "this checkpoint's tokenizer is `{known}`, which frink cannot read yet. \
             Supported: `llama` (SentencePiece), `gpt2` and `gemma4` (BPE), `t5` (Unigram), \
             `plamo2`."
        ),
        other => anyhow::bail!(
            "this checkpoint declares tokenizer.ggml.model = {other:?}, which frink does \
             not recognise. Supported: `llama`, `gpt2`, `gemma4`, `t5`, `plamo2`. Serving it would \
             mean feeding the model ids from a vocabulary it was not trained on, which \
             produces fluent text that is wrong rather than an error."
        ),
    }
}

pub(super) enum CliTokenizer {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    Plamo2(Box<GgufPlamo2Tokenizer>),
}

impl CliTokenizer {
    /// `specials` is llama.cpp's `parse_special`. The prompt sites pass
    /// `Parse`, as `llama-completion` does for its prompt
    /// (`tools/completion/completion.cpp`: `common_tokenize(ctx, prompt,
    /// true, true)`); the DRY breakers below pass `AsText`.
    pub(super) fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        match self {
            CliTokenizer::Bpe(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            CliTokenizer::Spm(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            CliTokenizer::Unigram(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            CliTokenizer::Plamo2(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
        }
    }

    pub(super) fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            CliTokenizer::Bpe(t) => t.decode(&ids32),
            CliTokenizer::Spm(t) => t.decode(&ids32),
            CliTokenizer::Unigram(t) => t.decode(&ids32),
            CliTokenizer::Plamo2(t) => t.decode(&ids32),
        }
    }

    pub(super) fn kind(&self) -> &'static str {
        match self {
            CliTokenizer::Bpe(_) => "gguf-bpe",
            CliTokenizer::Spm(_) => "gguf-spm",
            CliTokenizer::Unigram(_) => "gguf-unigram",
            CliTokenizer::Plamo2(_) => "gguf-plamo2",
        }
    }

    pub(super) fn vocab_size(&self) -> usize {
        match self {
            CliTokenizer::Bpe(t) => t.vocab_size(),
            CliTokenizer::Spm(t) => t.vocab_size(),
            CliTokenizer::Unigram(t) => t.vocab_size(),
            CliTokenizer::Plamo2(t) => t.vocab_size(),
        }
    }
}

/// What the DRY sampler needs to tokenise its sequence breakers.
///
/// `frink-server` implements the same trait for its own tokenizer enum.
/// Two implementations rather than one shared type because the two
/// enums genuinely differ (the server carries a byte-level fallback the
/// CLI does not), but they are held to ONE trait so the flag and the
/// request field cannot mean different things.
impl frink_models::dry::DryVocab for CliTokenizer {
    fn n_tokens(&self) -> usize {
        self.vocab_size()
    }

    fn detokenize(&self, token: usize) -> String {
        self.decode(&[token])
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        self.encode(text, SpecialTokens::AsText)
    }
}
