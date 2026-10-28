//! Loads the model `frink-server` actually serves.
//!
//! If `FRINK_MODEL_PATH` is set, this loads a real checkpoint: a
//! `.gguf` **file** through either the generic `Decoder` path or
//! [`frink_models::load_mla_engine_from_path`] for MLA architectures
//! (deepseek2 / mistral4 dense-lead), or [`frink_models::load_glm52_engine_from_path`]
//! for GLM-5.2 GGUFs (`glm-dsa`) when
//! the file itself names via `tokenizer.ggml.model`
//! (`gpt2` -> `GgufBpeTokenizer`, `llama` -> `GgufSpmTokenizer`, `t5` ->
//! `GgufUnigramTokenizer`); or a Kimi K3 checkpoint **directory** (real
//! `model.safetensors.index.json` + shards, `tiktoken.model`, optional
//! `tokenizer_config.json`) through `kimi_loader::load_kimi_checkpoint`
//! and `KimiTokenizer`, wrapped in the generic `frink_models::Engine`/
//! `TextTokenizer` traits (see `engine` module) so `frink-server` can
//! serve either through the same `/v1/chat/completions` endpoint (see
//! `LoadedModel`/`crate::Model` in `main.rs`). A GGUF checkpoint is a
//! single file; a Kimi K3 checkpoint is always a directory (its shards,
//! index, and tokenizer files can't be one file) -- `FRINK_MODEL_PATH`
//! pointing at a directory vs. a file is what selects between them.
//!
//! If `FRINK_MODEL_PATH` is unset, falls back to the previous
//! synthetic-weight demo behavior (small random weights for one of the
//! three built-in presets, byte tokenizer, always the GGUF-shaped path)
//! so the server still starts and demonstrates the request/response
//! pipeline without requiring a checkpoint on disk -- but logs a loud
//! warning that this is not real model output.

use std::path::Path;
use std::sync::Arc;

use frink_gguf::ShardedGguf;
use frink_models::tokenizer::{SpecialTokens, StopTokens};
use frink_models::{
    deepseek_v4_pro, glm_5_2, kimi_k3, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, ByteTokenizer, Decoder, Gemma4Engine,
    GgufBpeTokenizer, GgufPlamo2Tokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, Glm52Engine,
    KimiEngine, MlaEngine, ModelConfig, SelectedEngineKind, ServedEngine, TextTokenizer,
};

/// Default expert-cache budget when `FRINK_SSD_STREAMING` is set without
/// an explicit `FRINK_EXPERT_CACHE_BYTES`.
const DEFAULT_SSD_STREAMING_CACHE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Headroom left for the KV cache, activations and the OS when
/// deciding whether weights fit. A model that fills memory to the last
/// byte leaves nothing to decode with.
const FIT_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Streams experts only when the weights do not fit, and says why.
///
/// Streaming is OFF by default and stays off: it is strictly slower
/// than resident, so it is a way to run a model that otherwise could
/// not run at all, not a default. But requiring an operator to know in
/// advance is its own failure: without this, meeting a model too large
/// for the machine means a failed load rather than a slow success.
///
/// Explicit settings always win, in both directions.
/// `FRINK_SSD_STREAMING=0` forces resident even when the weights do
/// not fit, because an operator who says so may know something this
/// does not, such as a machine about to free memory.
fn auto_stream_if_needed(weight_bytes: u64) -> Option<u64> {
    // An explicit refusal is authoritative.
    if matches!(
        std::env::var("FRINK_SSD_STREAMING").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    ) {
        return None;
    }
    let available = frink_core::host_memory::available_bytes();
    match frink_core::host_memory::plan_for(
        weight_bytes,
        available,
        FIT_HEADROOM_BYTES,
        DEFAULT_SSD_STREAMING_CACHE_BYTES,
    ) {
        frink_core::host_memory::FitPlan::Resident => None,
        frink_core::host_memory::FitPlan::Stream { cache_bytes } => {
            // REFUSE rather than stream. Expert streaming produces WRONG
            // OUTPUT on real checkpoints: OLMoE-1B-7B Q4_0 answers
            // "Paris." resident and "amongst amongst, and of" streamed,
            // deterministically at temperature 0. The fixture test that
            // pins streaming as bit-identical passes, so whatever
            // differs is not exercised by it.
            //
            // Serving nonsense is worse than refusing to start, and an
            // operator who set no flag did not ask for either.
            let gib = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
            tracing::error!(
                "weights are {:.1} GiB and only {:.1} GiB is available. Expert streaming \
                 would fit them in about {:.1} GiB, but it currently produces WRONG \
                 OUTPUT on real checkpoints, so it is NOT enabled automatically. Use a \
                 smaller quantization, or set FRINK_EXPERT_CACHE_BYTES explicitly to \
                 try it anyway.",
                gib(weight_bytes),
                available.map(gib).unwrap_or(0.0),
                gib(cache_bytes),
            );
            None
        }
    }
}

/// Resolves the opt-in expert-streaming cache budget from
/// `FRINK_EXPERT_CACHE_BYTES` and/or `FRINK_SSD_STREAMING`.
fn resolve_expert_cache_bytes() -> anyhow::Result<Option<u64>> {
    let ssd_streaming = env_truthy("FRINK_SSD_STREAMING");
    let explicit = match std::env::var("FRINK_EXPERT_CACHE_BYTES") {
        Ok(v) => Some(v.parse::<u64>().map_err(|_| {
            anyhow::anyhow!("FRINK_EXPERT_CACHE_BYTES must be a non-negative integer, got {v:?}")
        })?),
        Err(_) => None,
    };
    if ssd_streaming {
        let budget = explicit.unwrap_or(DEFAULT_SSD_STREAMING_CACHE_BYTES);
        tracing::info!(
            "SSD streaming mode: expert cache streaming enabled with bounded cache of {budget} bytes"
        );
        Ok(Some(budget))
    } else if let Some(b) = explicit {
        tracing::info!("expert streaming enabled: bounded cache of {b} bytes");
        Ok(Some(b))
    } else {
        // Nothing explicit. Fall through to the automatic decision at
        // the call site, which needs the file size and so cannot be
        // made here.
        Ok(None)
    }
}

/// The budget to use for a model of `weight_bytes`, explicit if the
/// operator set one and automatic otherwise.
fn expert_cache_bytes_for(weight_bytes: u64) -> anyhow::Result<Option<u64>> {
    match resolve_expert_cache_bytes()? {
        Some(explicit) => Ok(Some(explicit)),
        None => Ok(auto_stream_if_needed(weight_bytes)),
    }
}

/// Whichever real tokenizer a loaded GGUF file's own metadata named, or
/// the byte-level fallback when no real vocabulary is available.
pub enum ServerTokenizer {
    // Boxed: GgufBpeTokenizer's merge-rank table makes it far larger
    // than the other variants (~1.2KB vs Spm's ~96 bytes vs Byte's 0),
    // so leaving it unboxed would size every ServerTokenizer value --
    // even Byte -- to the size of the largest variant.
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    Plamo2(Box<GgufPlamo2Tokenizer>),
    Byte,
}

impl ServerTokenizer {
    /// `specials` is llama.cpp's `parse_special`. Every route picks the
    /// setting its llama.cpp counterpart uses; see the callers of
    /// `Model::encode`.
    pub fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        match self {
            ServerTokenizer::Bpe(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            ServerTokenizer::Spm(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            ServerTokenizer::Unigram(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            ServerTokenizer::Plamo2(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
            // A byte vocabulary has no special entries, so the setting
            // has nothing to select.
            ServerTokenizer::Byte => ByteTokenizer::encode(text)
                .into_iter()
                .map(|id| id as usize)
                .collect(),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            ServerTokenizer::Bpe(t) => t.decode(&ids32),
            ServerTokenizer::Spm(t) => t.decode(&ids32),
            ServerTokenizer::Unigram(t) => t.decode(&ids32),
            ServerTokenizer::Plamo2(t) => t.decode(&ids32),
            ServerTokenizer::Byte => ByteTokenizer::decode(&ids32),
        }
    }

    /// The raw bytes, before any UTF-8 decision is made about them.
    /// See `crate::utf8_stream` for why a per-token caller needs them.
    pub fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            ServerTokenizer::Bpe(t) => t.decode_bytes(&ids32),
            ServerTokenizer::Spm(t) => t.decode_bytes(&ids32),
            ServerTokenizer::Unigram(t) => t.decode_bytes(&ids32),
            ServerTokenizer::Plamo2(t) => t.decode_bytes(&ids32),
            ServerTokenizer::Byte => frink_models::tokenizer::ByteTokenizer::decode_bytes(&ids32),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            ServerTokenizer::Bpe(_) => "gguf-bpe",
            ServerTokenizer::Spm(_) => "gguf-spm",
            ServerTokenizer::Unigram(_) => "gguf-unigram",
            ServerTokenizer::Plamo2(_) => "gguf-plamo2",
            ServerTokenizer::Byte => "byte (no real vocabulary loaded)",
        }
    }
}

impl TextTokenizer for ServerTokenizer {
    fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        ServerTokenizer::encode(self, text, specials)
    }

    fn decode(&self, ids: &[usize]) -> String {
        ServerTokenizer::decode(self, ids)
    }

    fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        ServerTokenizer::decode_bytes(self, ids)
    }
}

fn tokenizer_from_gguf(file: &ShardedGguf) -> anyhow::Result<ServerTokenizer> {
    match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => Ok(ServerTokenizer::Bpe(Box::new(GgufBpeTokenizer::from_gguf(
            file,
        )?))),
        Some("llama") => Ok(ServerTokenizer::Spm(GgufSpmTokenizer::from_gguf(file)?)),
        // "t5" is the real GGUF tag for a SentencePiece Unigram (ULM)
        // vocabulary -- confirmed against llama.cpp's own real
        // vocab-type-loading source, not guessed (see
        // GgufUnigramTokenizer's doc comment).
        Some("t5") => Ok(ServerTokenizer::Unigram(GgufUnigramTokenizer::from_gguf(
            file,
        )?)),
        Some("plamo2") => Ok(ServerTokenizer::Plamo2(Box::new(
            GgufPlamo2Tokenizer::from_gguf(file)?,
        ))),
        // A vocabulary this engine cannot read is not a warning, it is a
        // refusal.
        //
        // Falling back to raw bytes here produced FLUENT GARBAGE: the
        // model was fed ids from a vocabulary it was never trained on,
        // so it generated confidently and wrongly, and nothing in the
        // response said so. That is the exact failure this project
        // refuses everywhere else, and this was the one place that did
        // not. `ByteTokenizer` stays for synthetic-weight test models,
        // which have no real vocabulary to mismatch.
        Some(known @ ("bert" | "rwkv" | "none")) => anyhow::bail!(
            "this checkpoint's tokenizer is `{known}`, which frink cannot read yet. \
             Supported: `llama` (SentencePiece), `gpt2` and `gemma4` (BPE), `t5` \
             (Unigram). {}",
            match known {
                "bert" =>
                    "WordPiece is what BERT-family embedding models use, and a BERT \
                           checkpoint is an ENCODER: it has no output head and cannot \
                           generate text at all, so there is nothing for this path to \
                           serve. Frink can embed with it — set \
                           FRINK_EMBEDDING_MODEL_PATH to this file and \
                           POST /v1/embeddings.",
                "rwkv" => "RWKV uses a trie tokenizer frink does not implement.",
                _ => "`none` means the file carries no vocabulary at all.",
            }
        ),
        other => anyhow::bail!(
            "this checkpoint declares tokenizer.ggml.model = {other:?}, which frink does \
             not recognise. Supported: `llama`, `gpt2`, `gemma4`, `t5`. Serving it would \
             mean feeding the model ids from a vocabulary it was not trained on, which \
             produces fluent text that is wrong rather than an error."
        ),
    }
}

pub struct GgufLoaded {
    pub decoder: Decoder,
    pub tokenizer: ServerTokenizer,
    pub stop_tokens: StopTokens,
    pub bos_id: Option<usize>,
    /// True when this is the synthetic-random-weights fallback rather
    /// than a real checkpoint -- surfaced in `/v1/models` and logged so
    /// nobody mistakes demo output for a real completion.
    pub is_synthetic: bool,
    /// Compiled from the GGUF's own `tokenizer.chat_template` metadata
    /// (see `crate::chat_template`); the role-labeled builtin for the
    /// synthetic fallback, since there's no real checkpoint to carry one.
    pub chat_template: crate::chat_template::PromptTemplate,
}

pub struct KimiLoaded {
    pub engine: KimiEngine,
    pub tokenizer: frink_models::kimi_tokenizer::KimiTokenizer,
    pub stop_tokens: StopTokens,
    pub chat_template: crate::chat_template::PromptTemplate,
}

pub struct MlaLoaded {
    pub engine: MlaEngine,
    pub tokenizer: ServerTokenizer,
    pub stop_tokens: StopTokens,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::PromptTemplate,
}

pub struct Gemma4Loaded {
    pub engine: Gemma4Engine,
    pub tokenizer: ServerTokenizer,
    pub stop_tokens: StopTokens,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::PromptTemplate,
}

pub struct Glm52Loaded {
    pub engine: Glm52Engine,
    pub tokenizer: ServerTokenizer,
    pub stop_tokens: StopTokens,
    pub bos_id: Option<usize>,
    pub name: String,
    pub chat_template: crate::chat_template::PromptTemplate,
}

/// Either real checkpoint shape this server can serve. See this
/// module's doc comment for how `load()` picks between them.
#[allow(clippy::large_enum_variant)]
pub enum LoadedModel {
    Gguf(GgufLoaded),
    Kimi(KimiLoaded),
    Mla(MlaLoaded),
    Gemma4(Gemma4Loaded),
    Glm52(Glm52Loaded),
    /// An encoder-only embedding checkpoint (`bert`/BGE). It has no
    /// output head, so it never reaches any of the decoder loaders --
    /// see `crate::loaded::Loaded` for why it is not a `Model`.
    Encoder(Arc<frink_models::EmbeddingModel>),
}

/// Opens an encoder-only checkpoint, or refuses naming what it needs.
///
/// Reached from [`load_gguf_file`] when the file's own
/// `general.architecture` is one the capability registry scopes as
/// encoder/embedding. That test happens BEFORE any decoder dispatch on
/// purpose: sending a `bert` GGUF down the generic path answered a real
/// download with "tokenizer `bert` ... frink cannot read yet", which is
/// true of the decoder path and beside the point -- the file is an
/// embedding model, and saying so is the difference between a user
/// loading it correctly and a user going looking for a tokenizer bug.
fn load_encoder_checkpoint(path: &str) -> anyhow::Result<Arc<frink_models::EmbeddingModel>> {
    let model = frink_models::EmbeddingModel::from_gguf_path(path)
        .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
    tracing::info!(
        "loaded embedding model '{}' ({}, {} dims, pooling {}, max {} tokens). This is an \
         ENCODER: /v1/embeddings serves it and /v1/chat/completions refuses it.",
        model.name(),
        model.architecture(),
        model.n_embd(),
        model.pooling_type().name(),
        model.n_ctx_train(),
    );
    Ok(Arc::new(model))
}

/// The GLM families that really do need the MLA loader.
///
/// `glm4moe` was here and must not be: GLM-4.5 / 4.5-Air / 4.6 all tag
/// it, and none carries the four MLA hyper-parameters
/// `read_glm52_hparams` demands, because `src/models/glm4-moe.cpp`
/// builds plain Q/K/V and never reads them. Routing it here answered a
/// real download with "missing hparam glm4moe.attention.q_lora_rank" --
/// true, and about a key the architecture is not supposed to have.
/// Refusing on the generic path names the norm slot that is genuinely
/// missing instead, and runs it there since 2026-09-12
/// (`frink-models/tests/glm4moe_graphs.rs`). `glm4` (GLM-4-0414) left
/// the same day for the same reason (`tests/glm4_graphs.rs`).
fn is_glm52_arch(arch: &str) -> bool {
    arch == "glm-dsa"
}

pub fn load() -> anyhow::Result<LoadedModel> {
    match std::env::var("FRINK_MODEL_PATH") {
        Ok(path) => load_from_path(&path),
        Err(_) => {
            let preset = std::env::var("FRINK_PRESET").unwrap_or_else(|_| "glm-5.2".to_string());
            tracing::warn!(
                "FRINK_MODEL_PATH not set -- serving synthetic random weights for preset \
                 '{preset}' with a byte tokenizer, not a real checkpoint. Set \
                 FRINK_MODEL_PATH=/path/to/model.gguf (or a Kimi K3 checkpoint directory) to \
                 serve a real model."
            );
            Ok(LoadedModel::Gguf(GgufLoaded {
                decoder: build_synthetic_decoder(&preset)?,
                tokenizer: ServerTokenizer::Byte,
                stop_tokens: StopTokens::default(),
                bos_id: None,
                is_synthetic: true,
                chat_template: crate::chat_template::PromptTemplate::plain(),
            }))
        }
    }
}

fn load_gguf_file(path: &str) -> anyhow::Result<LoadedModel> {
    let file = ShardedGguf::open(path)?;
    if file.shard_count() > 1 {
        tracing::info!(
            "split GGUF checkpoint: {} shards, {} tensors total",
            file.shard_count(),
            file.tensor_count()
        );
    }
    let arch = file.metadata_str("general.architecture");
    frink_models::mmproj::warn_mmproj_if_present(Path::new(path), arch);
    if let Some(arch) = arch {
        // Before every decoder dispatch: an encoder has no output head,
        // so none of them could ever serve it.
        if frink_models::is_embedding_arch(arch) {
            return load_encoder_checkpoint(path).map(LoadedModel::Encoder);
        }
        if is_glm52_arch(arch) {
            crate::lora::refuse_env_for_engine("GLM-5.2")?;
            return load_glm52_checkpoint(path, &file).map(LoadedModel::Glm52);
        }
        if matches!(select_engine_kind(arch), Ok(SelectedEngineKind::Mla)) {
            crate::lora::refuse_env_for_engine("MLA")?;
            return load_mla_checkpoint(path, &file).map(LoadedModel::Mla);
        }
        if matches!(select_engine_kind(arch), Ok(SelectedEngineKind::Gemma4)) {
            crate::lora::refuse_env_for_engine("Gemma-4")?;
            return load_gemma4_checkpoint(path, &file).map(LoadedModel::Gemma4);
        }
    }
    load_real_gguf_checkpoint(path, &file).map(LoadedModel::Gguf)
}

fn load_glm52_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<Glm52Loaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("glm-dsa")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as GLM-5.2 engine (arch={arch}, name={name})");
    let served =
        load_glm52_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Glm52(engine) = served else {
        anyhow::bail!("expected ServedEngine::Glm52 for architecture {arch}");
    };

    let tokenizer = tokenizer_from_gguf(file)?;
    // The whole EOG set, not `eos_token_id` alone: see
    // `frink_models::tokenizer::StopTokens`.
    let stop_tokens = StopTokens::from_gguf(file);
    // Only surface BOS when llama.cpp would add it (see
    // `frink_models::tokenizer::should_add_bos_token`). Qwen2/BPE leave
    // add_bos=false; prepending `<|endoftext|>` poisons decode.
    let bos_id = if frink_models::tokenizer::should_add_bos_token(file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::PromptTemplate::from_gguf_metadata(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
        frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
        file.token_text("tokenizer.ggml.bos_token_id"),
        file.token_text("tokenizer.ggml.eos_token_id"),
    );
    tracing::info!("chat template: {}", chat_template.describe());

    Ok(Glm52Loaded {
        engine,
        tokenizer,
        stop_tokens,
        bos_id,
        name,
        chat_template,
    })
}

fn load_mla_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<MlaLoaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("mla")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as MLA engine (arch={arch}, name={name})");
    let served = load_mla_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Mla(engine) = served else {
        anyhow::bail!("expected ServedEngine::Mla for architecture {arch}");
    };

    let tokenizer = tokenizer_from_gguf(file)?;
    // The whole EOG set, not `eos_token_id` alone: see
    // `frink_models::tokenizer::StopTokens`.
    let stop_tokens = StopTokens::from_gguf(file);
    // Only surface BOS when llama.cpp would add it (see
    // `frink_models::tokenizer::should_add_bos_token`). Qwen2/BPE leave
    // add_bos=false; prepending `<|endoftext|>` poisons decode.
    let bos_id = if frink_models::tokenizer::should_add_bos_token(file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::PromptTemplate::from_gguf_metadata(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
        frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
        file.token_text("tokenizer.ggml.bos_token_id"),
        file.token_text("tokenizer.ggml.eos_token_id"),
    );
    tracing::info!("chat template: {}", chat_template.describe());

    Ok(MlaLoaded {
        engine,
        tokenizer,
        stop_tokens,
        bos_id,
        name,
        chat_template,
    })
}
fn load_gemma4_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<Gemma4Loaded> {
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("gemma4")
        .to_string();
    let name = file
        .metadata_str("general.name")
        .unwrap_or(arch.as_str())
        .to_string();
    tracing::info!("loading GGUF as Gemma4 engine (arch={arch}, name={name})");
    let served =
        load_gemma4_engine_from_path(Path::new(path)).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Gemma4(engine) = served else {
        anyhow::bail!("expected ServedEngine::Gemma4 for architecture {arch}");
    };
    let engine = *engine;

    let tokenizer = tokenizer_from_gguf(file)?;
    // The whole EOG set, not `eos_token_id` alone: see
    // `frink_models::tokenizer::StopTokens`.
    let stop_tokens = StopTokens::from_gguf(file);
    // Only surface BOS when llama.cpp would add it (see
    // `frink_models::tokenizer::should_add_bos_token`). Qwen2/BPE leave
    // add_bos=false; prepending `<|endoftext|>` poisons decode.
    let bos_id = if frink_models::tokenizer::should_add_bos_token(file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };
    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::PromptTemplate::from_gguf_metadata(
        file.metadata_str("tokenizer.chat_template"),
        Some(arch.as_str()),
        byte_tokenizer,
        frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
        file.token_text("tokenizer.ggml.bos_token_id"),
        file.token_text("tokenizer.ggml.eos_token_id"),
    );
    tracing::info!("chat template: {}", chat_template.describe());

    Ok(Gemma4Loaded {
        engine,
        tokenizer,
        stop_tokens,
        bos_id,
        name,
        chat_template,
    })
}

fn load_real_gguf_checkpoint(path: &str, file: &ShardedGguf) -> anyhow::Result<GgufLoaded> {
    let mut config = ModelConfig::from_gguf(file)?;
    config.kv_dtype = crate::budget::server_host_kv_dtype();
    if let Some(arch) = file.metadata_str("general.architecture") {
        frink_models::ensure_generic_decoder(arch).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    let all_confirmed =
        config.best_effort_fields.len() == 1 && config.best_effort_fields[0].starts_with("none --");
    if !all_confirmed {
        tracing::warn!(
            "some model config fields could not be read from this GGUF file's own metadata \
             and were inferred instead: {:?}",
            config.best_effort_fields
        );
    }

    let tokenizer = tokenizer_from_gguf(file)?;

    // The whole EOG set, not `eos_token_id` alone: see
    // `frink_models::tokenizer::StopTokens`.
    let stop_tokens = StopTokens::from_gguf(file);
    // Only surface BOS when llama.cpp would add it (see
    // `frink_models::tokenizer::should_add_bos_token`). Qwen2/BPE leave
    // add_bos=false; prepending `<|endoftext|>` poisons decode.
    let bos_id = if frink_models::tokenizer::should_add_bos_token(file) {
        file.metadata_u64("tokenizer.ggml.bos_token_id")
            .map(|v| v as usize)
    } else {
        None
    };

    let byte_tokenizer = matches!(tokenizer, ServerTokenizer::Byte);
    let chat_template = crate::chat_template::PromptTemplate::from_gguf_metadata(
        file.metadata_str("tokenizer.chat_template"),
        file.metadata_str("general.architecture"),
        byte_tokenizer,
        frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
        file.token_text("tokenizer.ggml.bos_token_id"),
        file.token_text("tokenizer.ggml.eos_token_id"),
    );
    tracing::info!("chat template: {}", chat_template.describe());

    // Opt-in expert streaming: with FRINK_EXPERT_CACHE_BYTES set (or
    // FRINK_SSD_STREAMING=1, which defaults the cache to 2 GiB),
    // routed experts are not loaded resident -- they stream through a
    // bounded, lease-protected cache of that many bytes (one global
    // budget across all layers), reading from the checkpoint file on
    // miss. Output is bit-identical to resident loading (same bytes,
    // same kernels); the trade is RAM footprint vs. read I/O.
    // The file's own size is the weight footprint, which is what the
    // automatic decision needs and why it cannot be made without a path.
    let weight_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let expert_cache_bytes = expert_cache_bytes_for(weight_bytes)?;
    let mut decoder = Decoder::from_gguf_with_expert_cache(path, config, expert_cache_bytes)?;
    // After the base, before it becomes the active model: an adapter the
    // checkpoint does not fit fails the load by name.
    crate::lora::attach_from_env(&mut decoder, file)?;

    Ok(GgufLoaded {
        decoder,
        tokenizer,
        stop_tokens,
        bos_id,
        is_synthetic: false,
        chat_template,
    })
}

/// Loads a real Kimi K3 checkpoint directory, mirroring
/// `frink-cli`'s `run-kimi` command exactly (same real file layout,
/// same real hparams/config assembly) so this is genuinely new
/// plumbing wired into the server, not a reimplementation with its own
/// untested assumptions.
fn load_real_kimi_checkpoint(dir: &str) -> anyhow::Result<KimiLoaded> {
    load_kimi_checkpoint_with_config(
        dir,
        kimi_k3(),
        frink_models::kimi_loader::KimiRealHparams::real(),
    )
}

/// The real loading logic, parametrized over `model_cfg`/`hp` so it can
/// be exercised end to end against a small synthetic checkpoint in
/// tests (Kimi K3's real shape -- 7168 hidden dim, 896 experts -- makes
/// a real-shaped on-disk fixture impractical for a unit test) --
/// mirrors how `kimi_loader::load_kimi_checkpoint` itself is already
/// generic over these two, rather than hardcoding the real preset the
/// way `frink-cli`'s `run-kimi` command does. `load_real_kimi_checkpoint`
/// (the real entry point `load()` calls) always passes the real
/// `kimi_k3()`/`KimiRealHparams::real()` -- this function's
/// parametrization is purely for testability, not a behavior change.
pub(crate) fn load_kimi_checkpoint_with_config(
    dir: &str,
    model_cfg: ModelConfig,
    hp: frink_models::kimi_loader::KimiRealHparams,
) -> anyhow::Result<KimiLoaded> {
    let dir_path = Path::new(dir);
    let index_path = dir_path.join("model.safetensors.index.json");
    tracing::info!(
        "loading real Kimi K3 checkpoint index: {}",
        index_path.display()
    );
    let shard = frink_safetensors::ShardedSafetensors::open_index(&index_path)?;

    tracing::info!(
        "loading all {} real layers (eagerly touches every routed expert's mmap range, \
         never materializes a dequantized f32 copy)...",
        model_cfg.n_layers
    );
    // Same opt-in as the GGUF path: FRINK_EXPERT_CACHE_BYTES (or
    // FRINK_SSD_STREAMING) streams routed experts through one bounded,
    // lease-protected store instead of holding every expert object
    // resident (bit-identical output).
    // Every shard beside the index, summed. An unreadable directory
    // yields 0, which resolves to Resident: refusing to guess beats
    // putting a model that would have fitted onto the slow path.
    let weight_bytes: u64 = index_path
        .parent()
        .and_then(|dir| std::fs::read_dir(dir).ok())
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .is_some_and(|x| x == "safetensors" || x == "gguf")
                })
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum()
        })
        .unwrap_or(0);
    let expert_cache_bytes = expert_cache_bytes_for(weight_bytes)?;
    let weights = frink_models::kimi_loader::load_kimi_checkpoint_with_expert_cache(
        &shard,
        &model_cfg,
        &hp,
        expert_cache_bytes,
    )?;
    tracing::info!("loaded. vocab size: {}", weights.output_head.rows());

    let vocab_path = dir_path.join("tiktoken.model");
    let vocab_text = std::fs::read_to_string(&vocab_path)?;
    let ranks = frink_models::kimi_tokenizer::parse_tiktoken_vocab(&vocab_text)?;
    let tokenizer_config_path = dir_path.join("tokenizer_config.json");
    let tokenizer_config_text = if tokenizer_config_path.exists() {
        Some(std::fs::read_to_string(&tokenizer_config_path)?)
    } else {
        None
    };
    let special_tokens = match &tokenizer_config_text {
        Some(text) => frink_models::kimi_tokenizer::parse_special_tokens(text)?,
        None => std::collections::HashMap::new(),
    };
    // Kimi K3's vocabulary is `tokenizer_config.json`, not GGUF
    // metadata, so the EOG set is derived from the special-token names:
    // `[EOT]` ends a turn there and `[EOS]` ends the sequence, and
    // stopping only on the latter runs the model past its own turn.
    let stop_tokens = StopTokens::from_special_tokens(
        special_tokens.iter().map(|(name, id)| (name.as_str(), *id)),
    );
    let tokenizer =
        frink_models::kimi_tokenizer::KimiTokenizer::new(ranks, special_tokens.clone())?;
    tracing::info!(
        "loaded real tokenizer: {} base tokens, {} stop tokens",
        tokenizer.vocab_size(),
        stop_tokens.len()
    );

    // Kimi K3's checkpoint carries its chat template as a top-level
    // string field in `tokenizer_config.json` (the real HuggingFace
    // convention), not GGUF metadata -- read it the same way, then
    // compile it with the same evaluator every other checkpoint uses.
    let chat_template_str = tokenizer_config_text.as_deref().and_then(|text| {
        serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .and_then(|v| v.get("chat_template")?.as_str().map(str::to_string))
    });
    let chat_template =
        crate::chat_template::PromptTemplate::from_source(chat_template_str.as_deref(), None, None);
    tracing::info!("chat template: {}", chat_template.describe());

    let frink_models::config::AttentionKind::KimiHybrid(hybrid) = &model_cfg.attention else {
        anyhow::bail!("kimi_k3() preset must use AttentionKind::KimiHybrid");
    };
    let decoder_cfg = frink_models::kimi_decoder::KimiDecoderConfig {
        attn_res_block_size: 12,
        rms_norm_eps: model_cfg.rms_norm_eps,
        situ_beta: 4.0,
        situ_linear_beta: 25.0,
        moe: frink_models::latent_moe::KimiMoeConfig {
            n_experts_active: model_cfg.moe.n_experts_active,
            moe_renormalize: true,
            routed_scaling_factor: 1.0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            rms_norm_eps: model_cfg.rms_norm_eps,
        },
    };
    let engine = KimiEngine {
        weights,
        cfg: decoder_cfg,
        mla_cfg: hybrid.mla.clone(),
        kda_cfg: hybrid.kda.clone(),
    };

    Ok(KimiLoaded {
        engine,
        tokenizer,
        stop_tokens,
        chat_template,
    })
}

fn build_synthetic_decoder(preset: &str) -> anyhow::Result<Decoder> {
    let base_cfg: ModelConfig = match preset {
        "glm-5.2" => glm_5_2(),
        "deepseek-v4-pro" => deepseek_v4_pro(),
        "kimi-k3" => kimi_k3(),
        other => anyhow::bail!("unknown preset '{other}'"),
    };
    // Small synthetic dims: this fallback demonstrates the
    // request/response/decode plumbing when no real checkpoint is
    // configured; it does not produce meaningful completions.
    let mut cfg = base_cfg;
    cfg.hidden_dim = 32;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 8;
    cfg.moe.hidden_dim = 32;
    cfg.moe.n_experts = cfg.moe.n_experts.min(16);
    cfg.moe.expert_ffn_dim = 16;

    Ok(Decoder::new_random_small(cfg, 2, 256))
}

/// Loads one checkpoint by path, with no reference to the environment.
///
/// Split out of [`load`] for `/admin/models/load`: the admin surface
/// has already resolved an id to a path it discovered itself, so
/// re-reading `FRINK_MODEL_PATH` there would load the *startup* model
/// no matter which one the user picked. A directory is a Kimi K3
/// checkpoint and a file is a GGUF, exactly as at startup -- see the
/// module docs.
///
/// Blocking and CPU-bound (it mmaps, and for a Kimi checkpoint touches
/// every expert range). Callers on the Tokio runtime must put it on
/// `spawn_blocking`.
pub fn load_from_path(path: &str) -> anyhow::Result<LoadedModel> {
    let path = frink_models::hf_pull::resolve_model_path(path)?;
    if Path::new(&path).is_dir() {
        crate::lora::refuse_env_for_engine("Kimi")?;
        load_real_kimi_checkpoint(&path).map(LoadedModel::Kimi)
    } else {
        load_gguf_file(&path)
    }
}

#[cfg(test)]
mod glm_dispatch_tests {
    /// A wrong reason is worse than a refusal, because it sends the
    /// reader after a key that does not exist.
    #[test]
    fn glm4moe_does_not_go_to_the_mla_loader() {
        assert!(!super::is_glm52_arch("glm4moe"));
        assert!(!super::is_glm52_arch("glm4"));
        assert!(super::is_glm52_arch("glm-dsa"));
    }
}
