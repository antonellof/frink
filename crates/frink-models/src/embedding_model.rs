//! One GGUF path in, one embedding vector out.
//!
//! Binds a tokenizer to a [`TextEncoder`] and owns the two steps
//! between them that neither half should own alone: adding the model's
//! own special tokens (`[CLS] … [SEP]`) around the tokenizer's pieces,
//! and pooling the hidden states the way the checkpoint's
//! `pooling_type` says.
//!
//! This is the type `/v1/embeddings` and the CLI both hold. It exists
//! so neither of them has to know that `bert` is an encoder, that
//! WordPiece does not add its own specials, or that CLS pooling means
//! row zero.

use thiserror::Error;

use crate::bert_gguf_loader::{load_bert_encoder, read_bert_hparams, BERT_ARCH};
use crate::encoder::{EncodeError, PairSequence, TextEncoder};
use crate::loader::LoadError;
use crate::pooling::{l2_normalize, pool, PoolingType};
use crate::rank_head::{load_rank_head, RankHead};
use crate::tokenizer::{GgufWordPieceTokenizer, SpecialTokens, TokenizerLoadError};

/// Encoder architectures upstream builds from `bert.cpp` and the other
/// embedding rows in the capability catalog, with what each one needs
/// that this crate does not have. Used to refuse *by name* instead of
/// with a generic "unsupported".
const NOT_YET: &[(&str, &str)] = &[
    // `neo-bert` and `eurobert` were HERE until 2026-09-19: they are
    // ONE topology (RMSNorm before each block, a bare residual after
    // it, one final norm) with three table columns between them, and
    // `bert_gguf_loader::EncoderSpec` is that table.
    // `jina-bert-v2` was HERE until 2026-09-19: GEGLU in both its
    // spellings (a separate gate, or one fused into a `2 * n_ff`-wide
    // `ffn_up`), the second attention norm, the whole-projection QK
    // LayerNorm and ALiBi at a literal 8.0 are all served now
    // (`tests/bert_family_graphs.rs`).
    // `jina-bert-v3` was HERE until 2026-09-19, refused for "RoPE and
    // per-projection QK norm". The first half is served
    // (`nomic-bert`'s rotation) and the second half was WRONG:
    // `jina-bert-v3.cpp:25-43` creates no `attn_q_norm` at all, so the
    // QK-norm branch of the shared graph (`bert.cpp:109-123`) is dead
    // for it. A verdict read from the graph's branches rather than
    // from the architecture's own loader named a blocker it does not
    // have.
    // `nomic-bert` was HERE until 2026-09-19: its two deltas from
    // `bert` -- NEOX RoPE on Q/K and a gated SiLU FFN -- are
    // `bert_encoder::BertFfn` and `BertHparams::rope_theta`, read from
    // the architecture through `bert_gguf_loader::ENCODER_ARCHS` and
    // checked against llama.cpp's own pooled embedding
    // (`tests/nomic_bert_graphs.rs`).
    (
        "nomic-bert-moe",
        "a second FFN shape on its MoE layers (moe_every_n_layers)",
    ),
    (
        "modern-bert",
        "its own graph (local/global alternating attention)",
    ),
    ("t5encoder", "the T5 encoder stack"),
    (
        "gemma-embedding",
        "a decoder embedding path, not an encoder",
    ),
];

/// True when `general.architecture` names an encoder / embedding model
/// rather than something with an output head.
///
/// This is the question a *server* asks before it decides which loader
/// a checkpoint path goes to: an encoder can never reach the decoder
/// path, so routing it there produces a refusal about a missing tensor
/// instead of "this is an embedding model". The answer comes from the
/// capability registry's own [`crate::capability::ArchScope`] and not
/// from a second list beside [`NOT_YET`], because two lists of the same
/// architectures is the copy this repo has already paid for seven times
/// — a row added to the registry is covered here the moment it lands.
///
/// `true` does not mean frink can serve it. It means
/// [`EmbeddingModel::from_gguf_path`] is the loader that will either
/// build it or refuse it *by name*.
pub fn is_embedding_arch(arch: &str) -> bool {
    crate::capability::resolve_profile(arch).is_some_and(|p| {
        matches!(
            p.scope,
            crate::capability::ArchScope::DeferredEncoderEmbedding
        )
    })
}

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerLoadError),
    #[error(transparent)]
    Encode(#[from] EncodeError),
    #[error(
        "architecture {arch:?} is an embedding model frink cannot serve yet: it needs {needs}. \
         Only {BERT_ARCH:?} is implemented"
    )]
    NotYetImplemented { arch: String, needs: &'static str },
    #[error(
        "architecture {0:?} is not an embedding model this build knows. \
         `frink_models::bert_gguf_loader::ENCODER_ARCHS` is the list it serves"
    )]
    NotAnEmbeddingModel(String),
    #[error(
        "{arch:?} carries tokenizer.ggml.model = {model:?}, but this embedding path only has \
         WordPiece (\"bert\")"
    )]
    UnsupportedTokenizer { arch: String, model: String },
    #[error(
        "the embedding model {name:?} ({arch}) carries no reranker classification head: the \
         checkpoint has no cls / cls.output tensors, so it has no relevance score to \
         report. It can only produce embeddings"
    )]
    NoRankHead { name: String, arch: String },
    #[error(
        "the encoder for {arch:?} has no two-segment (query, document) input form, which a \
         cross-encoder rerank needs. Concatenating the two texts would score fluently and \
         wrongly, so this refuses instead"
    )]
    NoPairInput { arch: String },
    #[error(
        "the reranker checkpoint {name:?} ({arch}) carries a classification head but only \
         {rows} token-type row(s): there is no \"Sentence B\" embedding to put the document \
         half of a pair on. Scoring both halves as Sentence A is what this cross-encoder was \
         NOT trained on, and it reorders the results rather than merely shifting them, so \
         this refuses at load instead of serving a plausible wrong ranking"
    )]
    NoSegmentB {
        name: String,
        arch: String,
        rows: usize,
    },
}

/// The one condition under which a checkpoint that HAS a classification
/// head still cannot serve `/v1/rerank`: its token-type table has no
/// "Sentence B" row, so a pair would put both halves on segment 0.
///
/// A function rather than an `if` inside the loader so the arm is
/// testable without a GGUF carrying that shape. A refusal whose
/// condition cannot be shown to fire reads as coverage and is not: this
/// repo has shipped one keyed on a GGUF spelling nothing writes.
///
/// It is checked at LOAD, and only here, because this is the only place
/// the head and the encoder are both in hand — the BERT loader does not
/// know whether a head was found, and asking once per request would put
/// the answer in two places. A checkpoint in this state cannot answer
/// the one route it exists for, so it does not load.
fn refuse_unpairable_reranker(
    has_rank_head: bool,
    n_segments: usize,
    name: &str,
    arch: &str,
) -> Option<EmbedError> {
    if has_rank_head && n_segments < 2 {
        return Some(EmbedError::NoSegmentB {
            name: name.to_string(),
            arch: arch.to_string(),
            rows: n_segments,
        });
    }
    None
}

/// A loaded embedding model: tokenizer + encoder + the checkpoint's own
/// pooling rule.
pub struct EmbeddingModel {
    encoder: Box<dyn TextEncoder + Send + Sync>,
    tokenizer: GgufWordPieceTokenizer,
    /// The reranker classification head, when the checkpoint carries
    /// one. `None` for a plain embedding model, and that is what makes
    /// `/v1/rerank` refuse rather than substitute a cosine similarity.
    rank_head: Option<RankHead>,
    arch: String,
    name: String,
}

impl EmbeddingModel {
    /// Opens `path` and builds whichever embedding stack its
    /// `general.architecture` names, or refuses naming what is missing.
    pub fn from_gguf_path(path: impl AsRef<std::path::Path>) -> Result<Self, EmbedError> {
        let file = frink_gguf::ShardedGguf::open(path.as_ref()).map_err(LoadError::from)?;
        let arch = frink_gguf::TensorSource::metadata_str(&file, "general.architecture")
            .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
            .to_string();
        if !crate::bert_gguf_loader::ENCODER_ARCHS
            .iter()
            .any(|(a, _)| *a == arch)
        {
            return Err(match NOT_YET.iter().find(|(a, _)| *a == arch) {
                Some((_, needs)) => EmbedError::NotYetImplemented { arch, needs },
                None => EmbedError::NotAnEmbeddingModel(arch),
            });
        }
        let tok_model = frink_gguf::TensorSource::metadata_str(&file, "tokenizer.ggml.model")
            .unwrap_or_default()
            .to_string();
        if tok_model != "bert" {
            return Err(EmbedError::UnsupportedTokenizer {
                arch,
                model: tok_model,
            });
        }
        let name = frink_gguf::TensorSource::metadata_str(&file, "general.name")
            .map(str::to_string)
            .unwrap_or_else(|| arch.clone());
        let tokenizer = GgufWordPieceTokenizer::from_gguf(&file)?;

        // ORDER IS LOAD-BEARING. `load_rank_head` MUST run before
        // `load_bert_encoder`, which ends in
        // `assert_every_tensor_consumed`: `cls.weight`, `cls.output.*`
        // and `cls.norm.weight` are read by nothing else in this crate,
        // so with the two lines swapped every reranker checkpoint dies
        // with an `UnconsumedTensors` refusal listing tensors frink
        // does in fact read. `read_bert_hparams` touches metadata only,
        // so asking for the geometry twice costs nothing.
        let hp = read_bert_hparams(&file)?;
        let rank_head = load_rank_head(&file, &hp.arch, hp.n_embd, hp.layer_norm_eps)?;
        let encoder = load_bert_encoder(&file)?;

        if let Some(refusal) =
            refuse_unpairable_reranker(rank_head.is_some(), encoder.n_segments(), &name, &arch)
        {
            return Err(refusal);
        }

        Ok(Self {
            encoder: Box::new(encoder),
            tokenizer,
            rank_head,
            arch,
            name,
        })
    }

    /// The encoder's hyper-parameters, including the two facts that
    /// differ between the architectures on `bert.cpp`'s graph: the
    /// rotation and the FFN shape (`crate::bert_encoder::BertFfn`).
    pub fn hparams(&self) -> Option<&crate::bert_encoder::BertHparams> {
        self.encoder.bert_hparams()
    }

    pub fn architecture(&self) -> &str {
        &self.arch
    }

    /// The checkpoint's `general.name`, or its architecture when the
    /// file carries none. What `/v1/embeddings` reports as `model`.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn n_embd(&self) -> usize {
        self.encoder.n_embd()
    }

    pub fn n_ctx_train(&self) -> usize {
        self.encoder.n_ctx_train()
    }

    pub fn pooling_type(&self) -> PoolingType {
        self.encoder.pooling_type()
    }

    /// The exact ids the encoder will see for `text`: the tokenizer's
    /// pieces wrapped in the model's own special tokens. Public because
    /// `/v1/embeddings` has to report `usage.prompt_tokens`, and that
    /// number is this length — llama.cpp counts the specials too.
    ///
    /// `SpecialTokens::Parse`, as llama.cpp's `/v1/embeddings` does
    /// (`tools/server/server-context.cpp`, `handle_embeddings_impl`:
    /// `tokenize_input_prompts(..., /* add_special */ true,
    /// /* parse_special */ true)`).
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        self.encoder
            .wrap_special(&self.tokenizer.encode(text, SpecialTokens::Parse))
    }

    /// Text for `ids`, through this checkpoint's own vocabulary.
    ///
    /// The counterpart to [`Self::token_ids`], so `/v1/detokenize`
    /// answers for an encoder rather than refusing. An embedding
    /// model's whole contract is the vector it returns for a string,
    /// and when that vector is surprising the first question is what
    /// tokens it actually saw. Without this the only way to ask was to
    /// load the checkpoint in a second tool.
    ///
    /// Not `wrap_special`'s inverse: it decodes exactly the ids given,
    /// including specials if the caller passes them, because a caller
    /// checking a tokenization wants to see what it sent.
    pub fn decode_tokens(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    /// Pooled embedding for `text`. `normalize` applies L2 normalization,
    /// which is what an OpenAI-compatible `/v1/embeddings` response is
    /// expected to carry and what llama.cpp's server does by default;
    /// the raw pooled vector is what the graph produced.
    pub fn embed(&self, text: &str, normalize: bool) -> Result<Vec<f32>, EmbedError> {
        let ids = self.token_ids(text);
        let mut v = self.encoder.embed_tokens(&ids)?;
        if normalize {
            l2_normalize(&mut v);
        }
        Ok(v)
    }

    /// Un-pooled `n_tokens × n_embd` hidden states, for a caller that
    /// wants to pool differently (or not at all).
    pub fn hidden_states(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.encoder.encode_tokens(&self.token_ids(text))?)
    }

    /// The checkpoint's reranker classification head, or `None` for a
    /// plain embedding model. What `/v1/rerank` checks before it
    /// promises a caller a relevance score.
    pub fn rank_head(&self) -> Option<&RankHead> {
        self.rank_head.as_ref()
    }

    /// The exact input [`Self::rerank_score`] will see for one
    /// `(query, document)` pair: `[CLS] query [SEP] document [SEP]`,
    /// **with** the segment id of every position.
    ///
    /// Separate from the scoring call for the same reason
    /// [`Self::token_ids`] is separate from [`Self::embed`] — a route
    /// has to report `usage.prompt_tokens`, and that number is
    /// `tokens.len()`.
    ///
    /// `SpecialTokens::AsText` for both halves, as llama.cpp's
    /// `format_prompt_rerank` does (`tools/server/server-common.cpp`:
    /// `tokenize_input_subprompt(vocab, mctx, query, false, false)` and
    /// the same for `doc`). A document that mentions `[SEP]` must not be
    /// able to end the query half early.
    pub fn rerank_input(&self, query: &str, document: &str) -> Result<PairSequence, EmbedError> {
        self.encoder
            .wrap_special_pair(
                &self.tokenizer.encode(query, SpecialTokens::AsText),
                &self.tokenizer.encode(document, SpecialTokens::AsText),
            )
            .ok_or_else(|| EmbedError::NoPairInput {
                arch: self.arch.clone(),
            })
    }

    /// The head's relevance score for a pair sequence built by
    /// [`Self::rerank_input`].
    ///
    /// This is upstream's RANK path in full: encode, take the **CLS**
    /// row, run the classification head, report output 0
    /// (`send_rerank`'s `embd[0]`). The CLS row is taken here regardless
    /// of what `{arch}.pooling_type` says, because the head was trained
    /// on that position — `pooling_type = RANK` is the checkpoint
    /// *declaring* this path, not naming a pooling rule, which is why
    /// [`crate::pooling::pool`] still refuses RANK and must keep
    /// refusing it.
    ///
    /// No L2 normalization and no sigmoid: upstream reports the raw
    /// logit, so a score is comparable only against other scores from
    /// the same head, and this must not quietly squash it into `0..1`.
    pub fn rerank_score(&self, pair: &PairSequence) -> Result<f32, EmbedError> {
        let head = self
            .rank_head
            .as_ref()
            .ok_or_else(|| EmbedError::NoRankHead {
                name: self.name.clone(),
                arch: self.arch.clone(),
            })?;
        let hidden = self.pair_hidden_states(pair)?;
        let cls = pool(&hidden, self.encoder.n_embd(), PoolingType::Cls)
            .map_err(|e| EmbedError::Encode(EncodeError::Pooling(e)))?;
        Ok(head.score(&cls))
    }

    /// Un-pooled `n_tokens × n_embd` hidden states for a pair built by
    /// [`Self::rerank_input`] — [`Self::hidden_states`]'s counterpart
    /// for the cross-encoder input, and the one graph call
    /// [`Self::rerank_score`] itself makes.
    ///
    /// Public for the same reason [`Self::hidden_states`] is: when a
    /// relevance score is surprising, the first questions are what
    /// tokens the model saw and what came out before the head, and
    /// without this the only way to ask was to load the checkpoint a
    /// second time — which for a reranker does not even work, because
    /// [`crate::load_bert_encoder_from_path`] alone leaves `cls.*`
    /// unconsumed and refuses.
    ///
    /// The pair's own `segments` are honoured, so passing a
    /// [`PairSequence`] whose segments are all zero reproduces the
    /// segment-blind graph exactly, without a second copy of it to
    /// drift.
    pub fn pair_hidden_states(&self, pair: &PairSequence) -> Result<Vec<f32>, EmbedError> {
        Ok(self.encoder.encode(&pair.tokens, Some(&pair.segments))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every deferred embedding architecture must produce a refusal
    /// that names it and names what it needs — not a generic error.
    #[test]
    fn every_deferred_embedding_arch_is_named_in_its_own_refusal() {
        for (arch, needs) in NOT_YET {
            let err = EmbedError::NotYetImplemented {
                arch: (*arch).to_string(),
                needs,
            };
            let msg = err.to_string();
            assert!(msg.contains(arch), "{msg} does not name {arch}");
            assert!(msg.contains(needs), "{msg} does not say what is missing");
        }
    }

    /// The catalog rows this module claims to cover must actually be
    /// the encoder/embedding rows the capability registry defers, so a
    /// new row added there cannot silently fall through to the generic
    /// "not an embedding model" arm.
    #[test]
    fn the_deferred_list_is_a_subset_of_the_capability_registry() {
        for (arch, _) in NOT_YET {
            assert!(
                crate::capability::resolve_profile(arch).is_some(),
                "{arch} is not in the capability registry"
            );
        }
    }

    /// [`is_embedding_arch`] is what a server routes on, so it has to
    /// name *exactly* the architectures this module can answer for:
    /// `bert`, which loads, plus every row in [`NOT_YET`], which
    /// refuses by name. A registry row scoped
    /// `DeferredEncoderEmbedding` that is in neither would be routed
    /// here and hit the generic `NotAnEmbeddingModel` arm, which says
    /// the opposite of the truth about it.
    #[test]
    fn is_embedding_arch_covers_the_registry_rows_and_nothing_else() {
        let mut registry: Vec<&str> = crate::capability::architecture_catalog()
            .iter()
            .filter(|p| {
                matches!(
                    p.scope,
                    crate::capability::ArchScope::DeferredEncoderEmbedding
                )
            })
            .map(|p| p.gguf_name)
            .collect();
        registry.sort_unstable();
        // The rows this module can be handed: the ones it serves
        // (`ENCODER_ARCHS`) plus the ones it refuses BY NAME
        // (`NOT_YET`). Both halves, because a row in neither would be
        // routed here and then answer with a generic error.
        let mut known: Vec<&str> = NOT_YET
            .iter()
            .map(|(a, _)| *a)
            .chain(
                crate::bert_gguf_loader::ENCODER_ARCHS
                    .iter()
                    .map(|(a, _)| *a),
            )
            .collect();
        known.sort_unstable();
        assert_eq!(
            registry, known,
            "the registry's encoder/embedding rows and this module's own list disagree"
        );
        for arch in &registry {
            assert!(is_embedding_arch(arch), "{arch} is not routed to this path");
        }
        // A decoder must NOT be routed here, or `FRINK_MODEL_PATH`
        // pointing at a llama GGUF would be told it is an embedding
        // model.
        for arch in ["llama", "qwen3", "gemma3", "deepseek2"] {
            assert!(!is_embedding_arch(arch), "{arch} was routed to this path");
        }
    }

    /// A reranker that cannot express "Sentence B" is refused at load,
    /// and a plain embedding model in the same state is NOT — an
    /// embedding pass is all segment 0 and has nothing to say about a
    /// second row.
    ///
    /// The point of the test is that the refusing arm is REACHABLE.
    /// Written as a condition inside the loader it could only be
    /// exercised by a checkpoint nobody publishes, which is how a gate
    /// comes to read as coverage while never firing.
    #[test]
    fn only_a_reranker_needs_a_second_token_type_row_and_it_is_refused_without_one() {
        assert!(refuse_unpairable_reranker(true, 2, "r", "bert").is_none());
        assert!(refuse_unpairable_reranker(false, 1, "e", "bert").is_none());
        assert!(refuse_unpairable_reranker(false, 0, "e", "bert").is_none());

        let err = refuse_unpairable_reranker(true, 1, "some-reranker", "bert")
            .expect("a head with one segment row must refuse");
        let msg = err.to_string();
        for fact in ["some-reranker", "bert", "1 token-type row"] {
            assert!(msg.contains(fact), "{msg} does not carry {fact}");
        }
        assert!(matches!(err, EmbedError::NoSegmentB { rows: 1, .. }));
    }
}
