//! llama.cpp-style GGUF completion (`-m` / `-p` / `-n` / …).

use std::fmt;
use std::io::{self, Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, ValueEnum};
use frink_core::cache::KvCache;
use frink_gguf::ShardedGguf;
use frink_models::tokenizer::SpecialTokens;
use frink_models::{
    ensure_generic_decoder, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, Decoder, Engine, GgufBpeTokenizer,
    GgufPlamo2Tokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, ModelConfig, PenaltyWindow,
    Sampler, SamplerOrder, SamplingParams, SelectedEngineKind, ServedEngine,
};

/// llama.cpp-compatible completion flags.
#[derive(Args, Debug, Clone)]
pub struct InferArgs {
    /// Model path (GGUF). Alias of llama.cpp `-m`.
    #[arg(
        short = 'm',
        long = "model",
        value_name = "FILE",
        required_unless_present_any = ["list_devices", "hf_repo"]
    )]
    pub model: Option<String>,

    /// Hugging Face repo to run, `user/repo[:QUANT]`, llama.cpp's
    /// `-hf`.
    ///
    /// Fetched into the frink cache on first use and reused after. The
    /// tag after the colon is a QUANT LABEL, not a git revision, and it
    /// matches without regard to case.
    #[arg(
        long = "hf-repo",
        visible_alias = "hf",
        value_name = "REPO[:QUANT]",
        conflicts_with = "model"
    )]
    pub hf_repo: Option<String>,

    /// Exact filename inside `--hf-repo`, llama.cpp's `-hff`.
    #[arg(long = "hf-file", value_name = "FILE", requires = "hf_repo")]
    pub hf_file: Option<String>,

    /// Penalise a token for having appeared at all, llama.cpp's
    /// `--presence-penalty`. `0.0` = off, which is llama.cpp's default.
    ///
    /// The engine and `/v1/chat/completions` have always supported
    /// this; the CLI hardcoded it to zero, so the two disagreed about
    /// what `frink` could do.
    #[arg(long = "presence-penalty", value_name = "P", default_value_t = 0.0)]
    pub presence_penalty: f32,

    /// Penalise a token in proportion to how often it has appeared,
    /// llama.cpp's `--frequency-penalty`. `0.0` = off.
    #[arg(long = "frequency-penalty", value_name = "P", default_value_t = 0.0)]
    pub frequency_penalty: f32,

    /// Prompt string. Alias of llama.cpp `-p`.
    #[arg(short = 'p', long = "prompt", default_value = "")]
    pub prompt: String,

    /// Prompt from file. Alias of llama.cpp `-f`.
    #[arg(short = 'f', long = "file", value_name = "FILE")]
    pub file: Option<String>,

    /// Number of tokens to predict (`-1` = fill remaining context).
    #[arg(
        short = 'n',
        long = "n-predict",
        visible_alias = "predict",
        default_value_t = 128
    )]
    pub n_predict: i64,

    /// Context size: `auto` = largest that fits the device memory
    /// budget, `0` = the GGUF's own `{arch}.context_length` (else
    /// 4096), or an explicit token count.
    #[arg(short = 'c', long = "ctx-size", default_value_t = ContextSize::FromModel)]
    pub ctx_size: ContextSize,

    /// Refuse to load (exit 1) when the pre-load budget says the
    /// requested context will not fit, instead of warning and trying
    /// anyway. Off by default because frink mmaps its weights: an
    /// over-budget model really can run, page-faulting, so the check
    /// is advisory unless you say otherwise.
    #[arg(long = "strict-budget", default_value_t = false)]
    pub strict_budget: bool,

    /// CPU threads (0 = leave rayon / env defaults). Sets `RAYON_NUM_THREADS`.
    #[arg(short = 't', long = "threads", default_value_t = 0)]
    pub threads: usize,

    /// Sampling temperature (`0` = greedy).
    #[arg(long = "temp", default_value_t = 0.8)]
    pub temperature: f32,

    /// Top-k sampling (`0` = disabled).
    #[arg(long = "top-k", default_value_t = 40)]
    pub top_k: usize,

    /// Top-p nucleus sampling.
    #[arg(long = "top-p", default_value_t = 0.95)]
    pub top_p: f32,

    /// Constrain generation to a GBNF grammar (llama.cpp's `--grammar`).
    #[arg(long = "grammar")]
    pub grammar: Option<String>,

    /// Read the GBNF grammar from a file (llama.cpp's `--grammar-file`).
    #[arg(long = "grammar-file")]
    pub grammar_file: Option<std::path::PathBuf>,

    /// Constrain generation to a JSON Schema, converted to GBNF
    /// (llama.cpp's `-j` / `--json-schema`).
    #[arg(short = 'j', long = "json-schema")]
    pub json_schema: Option<String>,

    /// Min-p sampling: drop every candidate less than this fraction as
    /// likely as the most likely one (`0.0` = disabled).
    ///
    /// llama.cpp's `--min-p`, and its default is **0.05**, not off
    /// (`common/common.h:231`, `common/arg.cpp:1987`). frink had no
    /// min-p at all, so it could not reproduce llama.cpp's own
    /// out-of-the-box output for any prompt.
    #[arg(long = "min-p", default_value_t = 0.05)]
    pub min_p: f32,

    /// How many recent tokens the penalties consider (`0` = off).
    ///
    /// llama.cpp's `--repeat-last-n`, default 64
    /// (`common/common.h:238`). frink had no window and scanned the
    /// whole history.
    #[arg(long = "repeat-last-n", default_value_t = 64)]
    pub repeat_last_n: usize,

    /// Repetition penalty (`1.0` = off).
    #[arg(long = "repeat-penalty", default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// Locally typical sampling, llama.cpp's `--typical` (`1.0` = off).
    ///
    /// Keeps the candidates whose surprisal is closest to the
    /// distribution's entropy, from the middle outward, rather than the
    /// most likely ones -- so it can drop the most likely token.
    #[arg(long = "typical", visible_alias = "typical-p", default_value_t = 1.0)]
    pub typical_p: f32,

    /// Truncate at `n` standard deviations of the logits below the
    /// maximum, llama.cpp's `--top-nsigma` (`-1.0` = off).
    #[arg(
        long = "top-nsigma",
        visible_alias = "top-n-sigma",
        default_value_t = -1.0
    )]
    pub top_n_sigma: f32,

    /// The probability that XTC removes the top candidates on any one
    /// token, llama.cpp's `--xtc-probability` (`0.0` = off).
    #[arg(long = "xtc-probability", default_value_t = 0.0)]
    pub xtc_probability: f32,

    /// The probability a candidate must reach before XTC may remove it,
    /// llama.cpp's `--xtc-threshold`. **Above 0.5 disables XTC**, which
    /// is upstream's guard: above a half at most one candidate can clear
    /// it and XTC never removes the last one.
    #[arg(long = "xtc-threshold", default_value_t = 0.1)]
    pub xtc_threshold: f32,

    /// DRY sequence-repetition penalty multiplier, llama.cpp's
    /// `--dry-multiplier` (`0.0` = off).
    ///
    /// Unlike `--repeat-penalty`, which looks at single tokens, DRY
    /// penalises the token that would EXTEND a repeated sequence, by
    /// `multiplier * base ^ (length - allowed-length)`.
    #[arg(long = "dry-multiplier", default_value_t = 0.0)]
    pub dry_multiplier: f32,

    /// The base of DRY's exponential, llama.cpp's `--dry-base`. Below
    /// 1.0 disables DRY.
    #[arg(long = "dry-base", default_value_t = 1.75)]
    pub dry_base: f32,

    /// Repetitions this long or shorter are free, llama.cpp's
    /// `--dry-allowed-length`.
    #[arg(long = "dry-allowed-length", default_value_t = 2)]
    pub dry_allowed_length: i32,

    /// How many recent tokens DRY scans for repetitions, llama.cpp's
    /// `--dry-penalty-last-n` (`0` = off, `-1` = the context size).
    #[arg(long = "dry-penalty-last-n", default_value_t = -1)]
    pub dry_penalty_last_n: i32,

    /// A string DRY refuses to look past, llama.cpp's
    /// `--dry-sequence-breaker`. Repeatable.
    ///
    /// Giving any breaker CLEARS llama.cpp's defaults (`\n`, `:`, `"`,
    /// `*`), exactly as upstream's flag does (`common/arg.cpp:2119`),
    /// and the literal `none` clears them without adding one. The
    /// strings are tokenised against the loaded model's own vocabulary,
    /// so a checkpoint with no real vocabulary refuses DRY rather than
    /// running it with no breakers.
    #[arg(long = "dry-sequence-breaker", value_name = "STRING")]
    pub dry_sequence_breaker: Vec<String>,

    /// The order the sampler chain runs in, `;`-separated, llama.cpp's
    /// `--samplers`.
    ///
    /// frink implements five of upstream's samplers, so a chain naming
    /// `dry`, `xtc`, `typ_p`, `top_n_sigma`, `mirostat` or `infill` is
    /// REFUSED by that name rather than built without it: a caller who
    /// asked for `xtc` and was quietly served a chain with no XTC in it
    /// got a different sampler and no way to tell.
    ///
    /// Two further rules the refusal explains when it fires:
    /// `penalties` must be first (frink penalises the whole vocabulary
    /// before the candidate list exists) and `temperature` must be
    /// present (the greedy-versus-sampled decision is taken from
    /// `--temp` before the chain runs).
    /// `--sampler-seq` is llama.cpp's other spelling of the same flag
    /// (`common/arg.cpp`), and `docs/CLI.md` documented it in two
    /// places while clap rejected it: a user copying the documented
    /// spelling got `unexpected argument`.
    #[arg(
        long = "samplers",
        alias = "sampler-seq",
        value_name = "LIST",
        default_value_t = SamplerOrder::default()
    )]
    pub samplers: SamplerOrder,

    /// RNG seed (`-1` = time-based).
    #[arg(short = 's', long = "seed", default_value_t = -1)]
    pub seed: i64,

    /// Devices used for offloading (`none` disables GPU use).
    #[arg(
        long = "device",
        visible_alias = "dev",
        value_name = "DEVICE",
        ignore_case = true
    )]
    pub device: Option<OffloadDevice>,

    /// Print available offload devices and exit.
    #[arg(long = "list-devices", default_value_t = false)]
    pub list_devices: bool,

    /// GPU layers: `0`, `auto`, `all`, or a count at or above the
    /// model's layer count.
    ///
    /// Partial placement is not implemented, and a PARTIAL count is now
    /// REFUSED rather than silently rounded up -- see
    /// [`GpuLayers::check_supported`]. This comment used to say "any
    /// value above zero currently enables all supported operations",
    /// which described the behaviour that was the bug: llama.cpp's
    /// `-ngl N` offloads exactly N layers, so accepting the count and
    /// offloading everything turned the flag into an out-of-memory on
    /// the machine it exists to accommodate.
    #[arg(
        long = "n-gpu-layers",
        visible_aliases = ["gpu-layers", "ngl"],
        default_value = "auto",
        value_name = "N"
    )]
    pub n_gpu_layers: GpuLayers,

    /// Draft model for speculative decoding, llama.cpp's `-md`.
    ///
    /// A smaller checkpoint from the SAME family and tokenizer as the
    /// target. Decode reads every weight of the target per token, so
    /// bandwidth divided by model bytes is a hard ceiling; a drafter
    /// proposes several tokens and the target checks them all in one
    /// pass, which changes what is read per token rather than how fast.
    /// The output is exactly what the target would have written alone.
    #[arg(long = "model-draft", short = 'd', value_name = "FILE")]
    pub model_draft: Option<String>,

    /// LoRA adapter GGUF (llama.cpp's `--lora`), applied at scale 1.
    /// Repeatable, and comma-separated values are accepted as upstream
    /// accepts them. The file is what `convert_lora_to_gguf.py` writes.
    ///
    /// Every adapter is applied inside the projections it names
    /// (`W x + scale * alpha / rank * B (A x)`); the fused Metal stacks
    /// cannot see it and are refused for the whole model, so an adapted
    /// model runs on the per-matrix path on every backend.
    #[arg(long = "lora", value_name = "FILE", action = clap::ArgAction::Append)]
    pub lora: Vec<String>,

    /// LoRA adapter with a scale, `FILE:SCALE` (llama.cpp's
    /// `--lora-scaled`). Repeatable; adapters are numbered in the order
    /// given, every `--lora` before every `--lora-scaled`.
    #[arg(long = "lora-scaled", value_name = "FILE:SCALE", action = clap::ArgAction::Append)]
    pub lora_scaled: Vec<String>,

    /// Tokens the drafter proposes per verification step (llama.cpp's
    /// `--draft-max`, also spelled `--draft`).
    #[arg(
        long = "draft-max",
        visible_aliases = ["draft"],
        value_name = "N",
        default_value_t = 5
    )]
    pub draft_max: usize,

    /// Stop drafting when the drafter's own probability for the token
    /// it just sampled is below this (llama.cpp's `--draft-p-min`).
    ///
    /// A guessing drafter is worse than none: the target pays for the
    /// position either way, and a rejection also discards every
    /// position after it.
    #[arg(long = "draft-p-min", value_name = "P", default_value_t = 0.75)]
    pub draft_p_min: f32,

    /// Optional system prompt (chat mode only).
    #[arg(long = "system")]
    pub system: Option<String>,

    /// Raw prompt: skip chat-template wrap (llama.cpp `--no-cnv`).
    ///
    /// llama.cpp REMOVED `--no-cnv` in its 0.4 launcher and spells the
    /// same intent `-st` / `--single-turn`. Both are accepted here:
    /// dropping the old name would break every command line written
    /// against the older tool, and refusing the new one means a
    /// current llama.cpp invocation fails against frink -- which is
    /// the whole thing this flag exists to avoid.
    #[arg(long = "no-cnv", default_value_t = false)]
    pub no_cnv: bool,

    /// Accepted and ignored: `frink run` is already one turn.
    ///
    /// llama.cpp's `-st` / `--single-turn` means "run the CONVERSATION
    /// for one turn, then exit" -- the chat template still applies. It
    /// is NOT `--no-cnv`, which skips the template, and mapping one to
    /// the other changes the answer: measured on the same prompt, the
    /// templated reply is "The capital of France is Paris." and the
    /// raw one continues " Paris\nThe capital city of France is...".
    ///
    /// `frink run` generates once and exits, so the flag already
    /// describes what it does. Accepted so a llama.cpp command line
    /// runs unchanged, and ignored rather than silently redirected.
    #[arg(long = "single-turn", default_value_t = false)]
    pub single_turn: bool,

    /// Process `\\n` / `\\t` / `\\r` / `\\\\` escapes in `-p`. Use
    /// `--no-escape` to pass the prompt through literally.
    ///
    /// Defaults TRUE, matching llama.cpp (`common/common.h:563`), which
    /// also spells the negation `--no-escape` (`common/arg.cpp:1799`).
    /// frink defaulted false, so `-p "line one\\nline two"` reached the
    /// model as a literal backslash-n on frink and as a newline on
    /// llama.cpp -- the same command, a different prompt, and no error
    /// either way.
    #[arg(
        short = 'e',
        long = "escape",
        default_value_t = true,
        overrides_with = "no_escape"
    )]
    pub escape: bool,

    /// Pass the prompt through literally, without expanding escapes.
    #[arg(long = "no-escape", action = clap::ArgAction::SetTrue)]
    pub no_escape: bool,

    /// Ignore EOS and always emit up to `-n` tokens.
    #[arg(long = "ignore-eos", default_value_t = false)]
    pub ignore_eos: bool,

    /// Print the final prompt before generation.
    #[arg(long = "verbose-prompt", default_value_t = false)]
    pub verbose_prompt: bool,

    /// Multi-token prediction (MTP) draft heads — not loaded from GGUF yet.
    #[arg(long = "mtp", default_value_t = false)]
    pub mtp: bool,

    /// KV cache dtype (llama.cpp `-ctk` analogue). Sets `FRINK_CTK`.
    /// Values: `f16` (default), `q8_0`, `fp8` (the Q8_0 wire) and `q4`
    /// (4 bits per element, with a Hadamard rotation on K where the
    /// head width allows it). An unrecognised value falls back to
    /// `f16`, as does every value on a backend whose KV cache is the
    /// host `Vec<f32>`.
    ///
    /// `env` is not decoration: `docs/CONFIG.md` has always documented
    /// `FRINK_CTK` as "same as `--ctk`", and it could not be, because
    /// the resolution below writes this field's value into that
    /// variable unconditionally and the field's default is `f16`. An
    /// environment that said `q4` was overwritten before any Metal
    /// code read it (GitHub issue #297). Letting clap read the variable
    /// as the default keeps one spelling: the flag wins when given, the
    /// environment when it is not, and the write-back below is then
    /// idempotent rather than destructive.
    #[arg(
        long = "ctk",
        visible_alias = "cache-type-k",
        value_name = "TYPE",
        env = "FRINK_CTK",
        default_value = "f16",
        value_parser = frink_models::ctk::parse_value
    )]
    pub ctk: String,
}

/// Build the shared decode step, compiling the grammar against this
/// model's vocabulary if one was asked for.
///
/// Takes the tokenizer rather than a closure so the vocabulary view is
/// built once per run, not once per token.
fn token_step(
    args: &InferArgs,
    sampler: Sampler,
    tokenizer: &CliTokenizer,
    stop_tokens: &frink_models::tokenizer::StopTokens,
    vocab_size: usize,
) -> anyhow::Result<TokenStep> {
    let Some(src) = args.grammar_source()? else {
        return Ok(TokenStep::new(sampler, None));
    };
    // Compiled here, before the decode loop, so a grammar that does not
    // parse fails the command rather than the first token.
    let grammar = frink_models::grammar::Grammar::from_str_with_root(&src, "root")
        .map_err(|e| anyhow::anyhow!("grammar does not parse: {e}"))?;
    let grammar = frink_models::grammar_sampler::GrammarSampler::new(
        grammar,
        vocab_size,
        |id| tokenizer.decode(&[id]).into_bytes(),
        |id| stop_tokens.contains(id),
    );
    Ok(TokenStep::new(sampler, Some(grammar)))
}

/// One decode step, shared by every generation loop in this file.
///
/// There were FOUR byte-identical `sampler.sample(&logits, &sampling,
/// &generated)` call sites here -- the dense path, the engine path and
/// two chat paths. Adding a grammar to three of them and missing the
/// fourth would have produced unconstrained output on one code path with
/// every test still green, which is this repo's most-repeated bug and
/// was the same shape `InferArgs::sampling()` was introduced to kill.
///
/// Holds the grammar because the mask and the accept are two halves of
/// one hook: a caller that could take the mask without the accept would
/// keep asking "what may the FIRST token be" forever.
pub struct TokenStep {
    sampler: frink_models::sampling::Sampler,
    grammar: Option<frink_models::grammar_sampler::GrammarSampler>,
}

impl TokenStep {
    pub fn new(
        sampler: frink_models::sampling::Sampler,
        grammar: Option<frink_models::grammar_sampler::GrammarSampler>,
    ) -> Self {
        Self { sampler, grammar }
    }

    /// Whether this step must see one logit per vocabulary entry.
    ///
    /// A backend may fold `lm_head + argmax` into its decode stack and
    /// return a single token id instead of logits. That is sound only
    /// when nothing needs to look at the vocabulary first, and a grammar
    /// does. Read by the Metal greedy guard, which used to test the
    /// temperature alone.
    ///
    /// `sampling` is taken because the CHAIN can need the vocabulary
    /// too: `xtc` and `typ_p` remove candidates the argmax may be one
    /// of, and `dry` and the repetition / presence / frequency
    /// penalties move logits, so at `temperature <= 0` the answer is not
    /// the argmax of what the device would fold.
    /// `SamplingParams::greedy_equals_raw_argmax` is the one predicate
    /// that decides it, shared with `frink_server::generate`'s copy of
    /// this gate.
    ///
    /// RAW argmax, not `chain_keeps_the_argmax`: the fold argmaxes the
    /// logits before anything on the host touches them, so the
    /// penalties are skipped too. Reading the sampler's own
    /// already-penalised predicate here was GitHub issue #170, and with
    /// `--repeat-penalty` defaulting to 1.1 it was live on every plain
    /// `--ngl 99 --temp 0` run.
    // Read only by the Metal greedy guard, so a CPU-only build has no
    // fold to refuse and this is genuinely dead there. Same shape and
    // same reason as `frink-models`'s `FoldedLmHead`.
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub fn needs_vocab_logits(&self, sampling: &frink_models::sampling::SamplingParams) -> bool {
        self.grammar.is_some() || !sampling.greedy_equals_raw_argmax()
    }

    /// `Ok(None)` means the grammar is SATISFIED and has no legal
    /// continuation -- a finished answer, not a failure. An unsatisfied
    /// dead end is the `Err`.
    ///
    /// `history` is a [`PenaltyWindow`], not the generated tokens: the
    /// penalties look back over the PROMPT as well, which is what
    /// llama.cpp does (see `frink_models::penalty_window`). Passing a
    /// slice here is what let these four loops disagree with
    /// `speculative` about the same flags.
    pub fn next(
        &mut self,
        logits: &[f32],
        sampling: &frink_models::sampling::SamplingParams,
        history: PenaltyWindow<'_>,
    ) -> anyhow::Result<Option<usize>> {
        let Some(grammar) = self.grammar.as_mut() else {
            return Ok(Some(self.sampler.sample(logits, sampling, history)));
        };
        let mut refusal = None;
        let mut outcome = frink_models::grammar_sampler::MaskOutcome::Allowed;
        let next = {
            let g = &*grammar;
            let mut mask = |scores: &mut [f32]| match g.mask_logits(scores) {
                Ok(o) => outcome = o,
                Err(e) => refusal = Some(e),
            };
            self.sampler
                .sample_with_mask(logits, sampling, history, Some(&mut mask))
        };
        if let Some(e) = refusal {
            anyhow::bail!("grammar refused every continuation: {e}");
        }
        if outcome == frink_models::grammar_sampler::MaskOutcome::Complete {
            return Ok(None);
        }
        grammar.accept(next)?;
        Ok(Some(next))
    }
}

impl InferArgs {
    /// The grammar these flags describe, if any.
    ///
    /// The three spellings are llama.cpp's and are MUTUALLY EXCLUSIVE
    /// there. Refused together rather than silently picking one, because
    /// a caller who passed both asked for two different constraints and
    /// honouring either is answering a question they did not ask.
    pub fn grammar_source(&self) -> anyhow::Result<Option<String>> {
        let given = [
            self.grammar.is_some(),
            self.grammar_file.is_some(),
            self.json_schema.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if given > 1 {
            anyhow::bail!(
                "--grammar, --grammar-file and --json-schema are mutually exclusive; \
                 pass exactly one"
            );
        }
        if let Some(g) = &self.grammar {
            return Ok(Some(g.clone()));
        }
        if let Some(path) = &self.grammar_file {
            return Ok(Some(std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("--grammar-file {}: {e}", path.display())
            })?));
        }
        if let Some(schema) = &self.json_schema {
            // Converted here rather than at the sampler, so a schema that
            // cannot be expressed fails BEFORE the model is loaded.
            return Ok(Some(
                frink_models::grammar::json_schema_to_grammar(schema)
                    .map_err(|e| anyhow::anyhow!("--json-schema: {e}"))?,
            ));
        }
        Ok(None)
    }

    /// The DRY configuration these flags spell, before its sequence
    /// breakers are tokenised.
    ///
    /// llama.cpp's `--dry-sequence-breaker` CLEARS the defaults the
    /// first time it is given (`common/arg.cpp:2119-2126`) and reads
    /// the literal `none` as "no breakers at all". Both are reproduced
    /// here, and `none` anywhere in the list clears it, because a caller
    /// who wrote it meant it.
    pub fn dry_request(&self) -> frink_models::dry::DryRequest {
        let sequence_breakers = if self.dry_sequence_breaker.is_empty() {
            frink_models::dry::DEFAULT_SEQUENCE_BREAKERS
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else if self.dry_sequence_breaker.iter().any(|s| s == "none") {
            Vec::new()
        } else {
            self.dry_sequence_breaker.clone()
        };
        frink_models::dry::DryRequest {
            multiplier: self.dry_multiplier,
            base: self.dry_base,
            allowed_length: self.dry_allowed_length,
            penalty_last_n: self.dry_penalty_last_n,
            sequence_breakers,
        }
    }

    /// The sampler these flags describe.
    ///
    /// One function rather than one copy per generation path. There were
    /// four identical literals here (the dense path, the engine path and
    /// two chat paths), which is the shape `CLAUDE.md` names as this
    /// repo's most expensive failure: adding `--min-p` meant editing
    /// four places, and a sampler added to three of them would be
    /// silently absent from the fourth with every test still green.
    ///
    /// Fallible, and taking the vocabulary, because of DRY: its sequence
    /// breakers are STRINGS that only mean something against a
    /// particular tokenizer, so a checkpoint with no real vocabulary
    /// must refuse `--dry-multiplier` rather than run DRY with no
    /// breakers. `vocab` is `None` for exactly those checkpoints, and
    /// `ctx_size` is what `--dry-penalty-last-n -1` resolves to.
    pub fn sampling(
        &self,
        vocab: Option<&dyn frink_models::dry::DryVocab>,
        ctx_size: usize,
    ) -> anyhow::Result<SamplingParams> {
        Ok(SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            typical_p: self.typical_p,
            top_n_sigma: self.top_n_sigma,
            xtc_probability: self.xtc_probability,
            xtc_threshold: self.xtc_threshold,
            dry: self.dry_request().resolve(vocab, ctx_size)?,
            repetition_penalty: self.repeat_penalty,
            penalty_last_n: self.repeat_last_n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            sampler_order: self.samplers,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OffloadDevice {
    Auto,
    None,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayers {
    Auto,
    All,
    Count(u32),
}

impl GpuLayers {
    fn offload_enabled(self) -> bool {
        !matches!(self, Self::Count(0))
    }

    /// Reject a PARTIAL offload rather than silently offloading
    /// everything.
    ///
    /// llama.cpp's `-ngl N` puts exactly `N` layers in VRAM and runs the
    /// rest on the CPU (`common/arg.cpp`), which is how people fit a
    /// model that does not otherwise fit. frink parses the count and
    /// then reads only `offload_enabled()`, a bool -- so `--ngl 10` on a
    /// 32-layer model offloaded all 32.
    ///
    /// That is the worst shape of divergence: same flag, same value, no
    /// error, and the failure lands as an out-of-memory on the machine
    /// the flag existed to accommodate.
    ///
    /// Partial offload is a real feature and not implemented here, so
    /// this REFUSES and names it. `0` (all CPU) and any count at or
    /// above the layer count (all GPU) are exact, and stay accepted.
    fn check_supported(self, n_layers: usize) -> anyhow::Result<()> {
        if let Self::Count(n) = self {
            let n = n as usize;
            if n > 0 && n < n_layers {
                anyhow::bail!(
                    "--ngl {n} asks for a PARTIAL offload ({n} of {n_layers} layers), which \
                     frink does not implement -- it would silently offload all {n_layers}. \
                     Use `--ngl 0` for CPU only, or `--ngl {n_layers}` / `--ngl all` for \
                     every layer."
                );
            }
        }
        Ok(())
    }
}

impl FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            _ => value
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| "expected 0, a positive integer, 'auto', or 'all'".into()),
        }
    }
}

impl fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Count(value) => value.fmt(f),
        }
    }
}

/// What `-c` / `--ctx-size` was asked for, before any model is opened.
/// Same shape as [`GpuLayers`]: a symbolic value alongside the literal
/// one, resolved once the header and the device budget are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSize {
    /// Largest context that fits the device memory budget.
    Auto,
    /// llama.cpp's `-c 0`: whatever the GGUF says it was trained for.
    FromModel,
    Tokens(usize),
}

impl FromStr for ContextSize {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "auto" => Ok(Self::Auto),
            "0" => Ok(Self::FromModel),
            other => other
                .parse::<usize>()
                .map(Self::Tokens)
                .map_err(|_| "expected 'auto', 0, or a positive token count".into()),
        }
    }
}

impl fmt::Display for ContextSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::FromModel => f.write_str("0"),
            Self::Tokens(n) => n.fmt(f),
        }
    }
}

/// Which memory pool the resolved backend draws from, so the budget is
/// probed against the device that will actually hold the KV cache.
/// `Auto` is reported as CPU: without a `--device`/`--ngl` choice the
/// generic decoder keeps its host `KvCache`, and claiming a GPU budget
/// we may never use would be the wrong kind of optimism.
fn budget_backend_for(args: &InferArgs) -> frink_models::BudgetBackend {
    use frink_models::BudgetBackend;
    let offload = args.n_gpu_layers.offload_enabled();
    match args.device {
        Some(OffloadDevice::Metal) => BudgetBackend::Metal,
        Some(OffloadDevice::Cuda) => BudgetBackend::Cuda,
        None | Some(OffloadDevice::Auto) if offload && cfg!(feature = "metal") => {
            BudgetBackend::Metal
        }
        None | Some(OffloadDevice::Auto) if offload && cfg!(feature = "cuda") => {
            BudgetBackend::Cuda
        }
        _ => BudgetBackend::Cpu,
    }
}

/// The startup banner, as a value so a test can hold it to the
/// same `kv_elem_for` the budget prices with.
fn banner_line(args: &InferArgs, device: OffloadDevice) -> String {
    // The banner reports what this run WILL DO, not what was typed.
    // It used to echo `--ctk` verbatim, so a CPU run printed `ctk=f16`
    // while the host `KvCache` is `Vec<f32>` and the budget priced it
    // at f32 -- the same two-structures-must-agree shape as everywhere
    // else in this repo, and it made the memory warning look wrong
    // (double the KV bytes the banner implied) when the warning was
    // the only honest line of the two. `kv_elem_for` is now the single
    // source, so the number in the banner is the number in the budget.
    //
    // The note says "ignored" only when the request really was not
    // honoured. Comparing the resolved name to the typed STRING said
    // otherwise for every alias: `--ctk fp8` resolves to the Q8_0 wire
    // by design and `--ctk q4_0` to the 4-bit one, and both printed
    // "ignored" while doing exactly what was asked. What is genuinely
    // ignored is a selectable dtype on a backend that has no device KV
    // store at all, which is what `KvElem::F32` means here.
    let effective_ctk = kv_elem_for(args);
    let requested_ctk = args.ctk.trim();
    let ctk_note = if !frink_models::ctk::is_served(requested_ctk) {
        // Accepted because llama.cpp accepts it, and there is no store
        // behind it here.
        format!(
            " (--ctk {requested_ctk} has no frink store; using {})",
            effective_ctk.as_str()
        )
    } else if effective_ctk == frink_models::kv_budget::KvElem::from_ctk(requested_ctk) {
        String::new()
    } else {
        format!(" (--ctk {requested_ctk} ignored: only the Metal KV store has a selectable dtype)")
    };

    format!(
        "frink: device={} gpu-layers={} ctk={}{}",
        match device {
            OffloadDevice::Auto => "auto",
            OffloadDevice::None => "none",
            OffloadDevice::Cpu => "cpu",
            OffloadDevice::Metal => "Metal",
            OffloadDevice::Cuda => "CUDA",
        },
        gpu_layers_note(args, device),
        effective_ctk.as_str(),
        ctk_note
    )
}

/// `-ngl` as the run will honour it. `-dev cpu -ngl all` offloads
/// nothing, and printing a bare `gpu-layers=all` there reads as a
/// promise the run does not keep.
fn gpu_layers_note(args: &InferArgs, device: OffloadDevice) -> String {
    let requested = args.n_gpu_layers.to_string();
    match device {
        OffloadDevice::None | OffloadDevice::Cpu if args.n_gpu_layers.offload_enabled() => {
            format!("{requested} (ignored, no GPU offload on this device)")
        }
        _ => requested,
    }
}

/// Width of the KV store the selected backend will really keep. The
/// host `frink_core::cache::KvCache` is `Vec<f32>`; only the Metal
/// path has a device KV whose dtype `--ctk` selects.
fn kv_elem_for(args: &InferArgs) -> frink_models::KvElem {
    use frink_models::{BudgetBackend, KvElem};
    match budget_backend_for(args) {
        BudgetBackend::Metal => KvElem::from_ctk(&args.ctk),
        // CUDA has no device KV store of its own yet, and CPU is the
        // f32 host cache.
        BudgetBackend::Cuda | BudgetBackend::Cpu => KvElem::F32,
    }
}

/// Resolves `-c/--ctx-size` against the pre-load budget, printing the
/// arithmetic behind the answer.
///
/// This is the whole point of Phase 2: the terms are exact in the GGUF
/// header, so the check happens *before* the weights load rather than
/// being discovered as an allocation failure later. `auto` picks the
/// largest fitting context; an explicit context that does not fit is
/// reported as a typed rejection naming the estimate, the limit and
/// which ceiling binds -- fatal under `--strict-budget`, a warning
/// otherwise (see that flag's doc comment for why the default is
/// advisory).
fn resolve_ctx_size(args: &InferArgs, path: &Path, gguf_ctx: usize) -> anyhow::Result<usize> {
    use frink_models::residency_report::{ResidencyAssumptions, ResidencyReport};
    use frink_models::DeviceBudget;

    let backend = budget_backend_for(args);
    let budget = DeviceBudget::detect(backend);
    let assumptions = ResidencyAssumptions {
        context_tokens: gguf_ctx,
        concurrent_requests: 1,
        expert_cache_bytes: expert_cache_bytes_from_env(),
        kv_elem: kv_elem_for(args),
        ..ResidencyAssumptions::default()
    };

    // No probe, no ceiling: fall back to the requested context rather
    // than refusing on the strength of a number we do not have.
    if budget.is_unknown() {
        let requested = match args.ctx_size {
            ContextSize::Tokens(n) => n,
            ContextSize::Auto | ContextSize::FromModel => gguf_ctx,
        };
        eprintln!("frink: {budget}; using ctx={requested} unchecked");
        return Ok(requested);
    }

    let report = match ResidencyReport::from_gguf(path, assumptions, budget.usable_bytes) {
        Ok(r) => r,
        // A header this planner cannot read is not a reason to refuse
        // a run the loader may well handle (MLA/Gemma4/GLM stacks have
        // their own hparams and do not go through `ModelConfig`).
        Err(e) => {
            let requested = match args.ctx_size {
                ContextSize::Tokens(n) => n,
                ContextSize::Auto | ContextSize::FromModel => gguf_ctx,
            };
            eprintln!("frink: KV budget not computed for this checkpoint ({e}); ctx={requested}");
            return Ok(requested);
        }
    };
    let priced = report.kv_budget();

    let tokens = match args.ctx_size {
        ContextSize::Auto => {
            let fit = report.auto_context(gguf_ctx);
            eprintln!("frink: {budget}");
            eprintln!("frink: {fit}");
            eprintln!("frink: {}", budget.caveat());
            if fit.tokens == 0 {
                anyhow::bail!(
                    "{}: no context fits -- {} of weights leave nothing for KV inside the \
                     {} budget. Quantize further, stream experts \
                     (FRINK_EXPERT_CACHE_BYTES), or raise FRINK_DEVICE_BUDGET_BYTES.",
                    frink_models::Ceiling::DeviceMemory.code(),
                    report.weights_bytes,
                    budget.usable_bytes,
                );
            }
            fit.tokens
        }
        ContextSize::FromModel => gguf_ctx,
        ContextSize::Tokens(n) => n,
    };

    if let Err(e) = priced.check(tokens) {
        let fit = report.auto_context(gguf_ctx);
        let message = format!(
            "{}: {} bytes estimated at ctx={tokens} against a {} byte {} budget ({}); \
             {} bytes over. That estimate is {}. `--ctx-size auto` would pick {}. {}",
            e.code(),
            e.estimated_bytes,
            e.limit_bytes,
            backend,
            budget.usable_provenance(),
            e.overage_bytes(),
            e.detail,
            fit.tokens,
            budget.caveat(),
        );
        if args.strict_budget {
            anyhow::bail!("{message}");
        }
        eprintln!("frink: WARNING {message}");
        eprintln!("frink: continuing anyway (pass --strict-budget to refuse instead)");
    }
    Ok(tokens)
}

/// `FRINK_EXPERT_CACHE_BYTES`, so the plan charges streamed routed
/// experts at their cache budget rather than fully resident.
fn expert_cache_bytes_from_env() -> Option<u64> {
    std::env::var("FRINK_EXPERT_CACHE_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| *v > 0)
}

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
fn cli_tokenizer_from_gguf(file: &ShardedGguf) -> anyhow::Result<CliTokenizer> {
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

enum CliTokenizer {
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
    fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
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

    fn decode(&self, ids: &[usize]) -> String {
        let ids32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        match self {
            CliTokenizer::Bpe(t) => t.decode(&ids32),
            CliTokenizer::Spm(t) => t.decode(&ids32),
            CliTokenizer::Unigram(t) => t.decode(&ids32),
            CliTokenizer::Plamo2(t) => t.decode(&ids32),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            CliTokenizer::Bpe(_) => "gguf-bpe",
            CliTokenizer::Spm(_) => "gguf-spm",
            CliTokenizer::Unigram(_) => "gguf-unigram",
            CliTokenizer::Plamo2(_) => "gguf-plamo2",
        }
    }

    fn vocab_size(&self) -> usize {
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

/// The checkpoint's own chat template, evaluated.
///
/// This used to be a near-identical copy of `frink-server`'s marker
/// sniffer -- six hand-written renderers picked by which literal marker
/// a template string happened to contain. Both are gone; both now
/// compile the real Jinja source with
/// [`frink_models::chat_template`], so `frink -m mistral.gguf -p hi`
/// and `POST /v1/chat/completions` frame the same conversation the same
/// way, including for the families the sniffer never recognised.
struct ChatKind {
    template: frink_models::chat_template::ChatTemplate,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatKind {
    fn detect_for_gguf(file: &ShardedGguf, byte_tokenizer: bool) -> Self {
        ChatKind {
            template: frink_models::chat_template::ChatTemplate::from_gguf_metadata(
                file.metadata_str("tokenizer.chat_template"),
                file.metadata_str("general.architecture"),
                byte_tokenizer,
                frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
            ),
            bos_token: file.token_text("tokenizer.ggml.bos_token_id"),
            eos_token: file.token_text("tokenizer.ggml.eos_token_id"),
        }
    }

    /// One conversation turn, framed the way this checkpoint expects.
    ///
    /// A template that will not render is an error, never a fallback to
    /// a guessed framing: a silently mis-framed prompt is exactly the
    /// bug that made this stop sniffing, and it shows up as degenerate
    /// output rather than as a message.
    fn wrap_user(&self, system: Option<&str>, user: &str) -> anyhow::Result<String> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user}));
        let opts = frink_models::chat_template::RenderOptions {
            add_generation_prompt: true,
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            ..Default::default()
        };
        self.template
            .render(&messages, &opts)
            .map_err(|e| anyhow::anyhow!("chat template failed to render: {e}"))
    }
}

fn apply_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn resolve_prompt(args: &InferArgs) -> anyhow::Result<String> {
    let mut prompt = if let Some(path) = &args.file {
        let mut buf = String::new();
        let mut f = std::fs::File::open(path)?;
        f.read_to_string(&mut buf)?;
        buf
    } else {
        args.prompt.clone()
    };
    if args.escape && !args.no_escape {
        prompt = apply_escapes(&prompt);
    }
    Ok(prompt)
}

fn apply_backend_env(args: &InferArgs) -> anyhow::Result<()> {
    if args.threads > 0 {
        // SAFETY: single-threaded init before rayon workers spawn.
        unsafe {
            std::env::set_var("RAYON_NUM_THREADS", args.threads.to_string());
            std::env::set_var("FRINK_CPU_THREADS", args.threads.to_string());
        }
    }
    // SAFETY: still single-threaded here; rayon/Metal workers below.
    unsafe { frink_core::weight_matrix::default_cpu_int_dot_on() };
    // Same pool policy as `frink-server`: explicit width (performance
    // cores by default, as llama.cpp does) and explicit QoS.
    frink_core::threads::init_cpu_pool();

    let device = if args.n_gpu_layers.offload_enabled() {
        args.device.unwrap_or(OffloadDevice::Auto)
    } else {
        OffloadDevice::None
    };

    match device {
        OffloadDevice::None | OffloadDevice::Cpu => unsafe {
            std::env::set_var("FRINK_METAL", "0");
            std::env::set_var("FRINK_METAL_ATTN", "0");
            std::env::set_var("FRINK_CUDA", "0");
        },
        OffloadDevice::Auto => unsafe {
            std::env::set_var("FRINK_METAL", "auto");
            std::env::set_var("FRINK_CUDA", "auto");
            // Honor a pre-set FRINK_METAL_ATTN so ablations like
            // `FRINK_METAL_ATTN=0 … --ngl 99` actually disable attn.
            if std::env::var_os("FRINK_METAL_ATTN").is_none() {
                std::env::set_var("FRINK_METAL_ATTN", "1");
            }
        },
        OffloadDevice::Metal => {
            #[cfg(not(feature = "metal"))]
            {
                anyhow::bail!("Metal requested but this binary was built without --features metal");
            }
            #[cfg(feature = "metal")]
            {
                if !frink_metal::MetalProfile::detect().available {
                    anyhow::bail!("Metal requested but no Metal device is available");
                }
                unsafe {
                    std::env::set_var("FRINK_METAL", "1");
                    if std::env::var_os("FRINK_METAL_ATTN").is_none() {
                        std::env::set_var("FRINK_METAL_ATTN", "1");
                    }
                    std::env::set_var("FRINK_CUDA", "0");
                }
            }
        }
        OffloadDevice::Cuda => {
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA requested but this binary was built without --features cuda");
            }
            #[cfg(feature = "cuda")]
            {
                if !frink_cuda::HardwareProfile::detect().cuda_available {
                    anyhow::bail!("CUDA requested but no CUDA device is available");
                }
                unsafe {
                    std::env::set_var("FRINK_CUDA", "1");
                    std::env::set_var("FRINK_METAL", "0");
                    std::env::set_var("FRINK_METAL_ATTN", "0");
                }
            }
        }
    }

    // SAFETY: single-threaded init before Metal/CUDA workers spawn.
    unsafe {
        std::env::set_var("FRINK_CTK", args.ctk.trim());
    }

    // Held for the banner rather than printed here: this runs before
    // the model is open, and printing it now puts it above the
    // wordmark, where llama.cpp has nothing.
    crate::cli_output::set_device_line(banner_line(args, device));
    Ok(())
}

fn seed_from_args(seed: i64) -> u64 {
    if seed < 0 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
    } else {
        seed as u64
    }
}

/// Run llama.cpp-style GGUF completion.
/// Loads a GGUF decoder, streaming experts when the weights will not
/// fit in memory.
///
/// The CLI could not stream AT ALL before this: `from_gguf_with_expert_cache`
/// was used only by `frink-server`, so `FRINK_SSD_STREAMING` and
/// `FRINK_EXPERT_CACHE_BYTES` were silently ignored by `frink -m`.
/// The CLI even printed advice to set the latter, for a feature it did
/// not implement. That matters because running a model too big for the
/// machine is the project's headline capability and the CLI is how
/// people run models.
///
/// Same decision as the server: explicit settings win in both
/// directions, an unknown amount of memory resolves to resident rather
/// than guessing, and enabling it says so, because streaming is slower
/// than resident and a slow run should never be a silent one.
pub(crate) fn load_decoder_streaming_if_needed(
    path: &std::path::Path,
    config: frink_models::config::ModelConfig,
) -> anyhow::Result<Decoder> {
    let explicit = std::env::var("FRINK_EXPERT_CACHE_BYTES")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let refused = matches!(
        std::env::var("FRINK_SSD_STREAMING").ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    );
    let budget = if let Some(b) = explicit {
        Some(b)
    } else if refused {
        None
    } else {
        let weights = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let available = frink_core::host_memory::available_bytes();
        match frink_core::host_memory::plan_for(
            weights,
            available,
            frink_core::host_memory::FIT_HEADROOM_BYTES,
            /* floor = */ 2 * 1024 * 1024 * 1024,
        ) {
            frink_core::host_memory::FitPlan::Resident => None,
            frink_core::host_memory::FitPlan::Stream { cache_bytes } => {
                // REFUSE rather than stream. Expert streaming produces
                // WRONG OUTPUT on real checkpoints: OLMoE-1B-7B Q4_0
                // answers "Paris." resident and "amongst amongst, and
                // of" streamed, deterministically, at temperature 0.
                //
                // The fixture test
                // `store_backed_experts_produce_bit_identical_logits_to_resident`
                // passes, so whatever differs is not exercised by it.
                // Until that is understood, enabling this automatically
                // would turn "your model does not fit" into "your model
                // answers nonsense", which is far worse.
                let gib = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
                anyhow::bail!(
                    "this checkpoint is {:.1} GiB and only {:.1} GiB is available. Expert \
                     streaming would fit it in about {:.1} GiB, but it currently produces \
                     WRONG OUTPUT on real checkpoints and is not enabled automatically \
                     for that reason. Use a smaller quantization, or set \
                     FRINK_EXPERT_CACHE_BYTES explicitly to try streaming anyway and \
                     compare the output against llama.cpp yourself.",
                    gib(weights),
                    available.map(gib).unwrap_or(0.0),
                    gib(cache_bytes),
                );
            }
        }
    };
    Ok(Decoder::from_gguf_with_expert_cache(path, config, budget)?)
}

pub fn run_infer(args: InferArgs) -> anyhow::Result<()> {
    if args.list_devices {
        frink_models::devices::print_available_devices();
        return Ok(());
    }
    // `-hf` resolves to a local path before anything else looks at
    // `--model`, so the rest of this function sees one kind of input.
    let mut args = args;
    if let Some(spec) = args.hf_repo.clone() {
        args.model = Some(crate::hf::resolve(&spec, args.hf_file.as_deref())?);
    }
    if args.mtp {
        anyhow::bail!(
            "--mtp: MTP draft heads not yet loaded from GGUF (num_nextn_predict_layers); \
             prompt-lookup speculative decoding remains available via `frink speculative`"
        );
    }
    apply_backend_env(&args)?;

    let model = args
        .model
        .clone()
        .ok_or_else(|| anyhow::anyhow!("--model is required"))?;
    let model = crate::pull::resolve_model_path(&model)?;
    let path = Path::new(&model);
    if !path.exists() {
        anyhow::bail!("model not found: {model}");
    }

    let file = ShardedGguf::open(path)?;
    let arch_early = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();
    frink_models::mmproj::eprint_mmproj_if_present(path, Some(arch_early.as_str()));
    let lora_specs = frink_models::lora_attach::LoraSpec::from_flags(&args.lora, &args.lora_scaled)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !lora_specs.is_empty()
        && (matches!(
            select_engine_kind(&arch_early),
            Ok(SelectedEngineKind::Mla | SelectedEngineKind::Gemma4)
        ) || arch_early == "glm-dsa")
    {
        // The dedicated engines do not go through `Decoder`, and a
        // flag that is accepted must reach the thing it names.
        anyhow::bail!(
            "--lora is not implemented for the {arch_early} engine (only the generic decoder \
             attaches adapters); refusing rather than running the base weights"
        );
    }
    if matches!(select_engine_kind(&arch_early), Ok(SelectedEngineKind::Mla)) {
        return run_mla_infer(args, path, &file);
    }
    if matches!(
        select_engine_kind(&arch_early),
        Ok(SelectedEngineKind::Gemma4)
    ) {
        return run_gemma4_infer(args, path, &file);
    }
    // `glm4moe` is NOT here, and its absence is the fix. GLM-4.5,
    // GLM-4.5-Air and GLM-4.6 all tag `glm4moe`, and none of them is an
    // MLA model: `src/models/glm4-moe.cpp` reads no `q_lora_rank` and
    // builds plain Q/K/V. Sending them here made a real GLM-4.5-Air
    // download fail with "missing hparam glm4moe.attention.q_lora_rank",
    // a true statement about a key the architecture is not supposed to
    // have. It runs on the generic path now, audited against libllama
    // (`crates/frink-models/tests/glm4moe_graphs.rs`), and so does
    // `glm4` (GLM-4-0414, `tests/glm4_graphs.rs`), which had been sent
    // here for the same four keys.
    if arch_early == "glm-dsa" {
        return run_glm52_infer(args, path, &file);
    }

    let config = ModelConfig::from_gguf(&file)?;
    // Checked here rather than at parse time: the layer count is what
    // makes a given `--ngl N` exact or partial, and it is in the file.
    args.n_gpu_layers.check_supported(config.n_layers)?;
    if let Some(arch) = file.metadata_str("general.architecture") {
        ensure_generic_decoder(arch).map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    if !(config.best_effort_fields.is_empty()
        || (config.best_effort_fields.len() == 1
            && config.best_effort_fields[0].starts_with("none --")))
    {
        eprintln!(
            "frink: inferred config fields: {:?}",
            config.best_effort_fields
        );
    }

    let tokenizer = cli_tokenizer_from_gguf(&file)?;
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = frink_models::tokenizer::StopTokens::from_gguf(&file);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);

    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(&file, false);
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    if args.verbose_prompt {
        eprintln!("----- prompt -----");
        eprintln!("{prompt}");
        eprintln!("------------------");
    }

    let load_t = Instant::now();
    // LongRoPE picks its factor set from the run's context size, not the
    // checkpoint's advertised maximum (llama.cpp does the same, per
    // request, from `cparams.n_ctx_seq`).
    let mut config = config;
    config.apply_runtime_context(ctx_size);
    let mut decoder = load_decoder_streaming_if_needed(path, config)?;
    decoder
        .attach_lora_specs(&file, &lora_specs)
        .map_err(|e| anyhow::anyhow!("lora: {e}"))?;
    let decoder = decoder;
    // llama.cpp's banner, printed where its own is: once the model is
    // open and before anything is generated. The load duration moved
    // into it rather than onto a line of its own, because llama.cpp
    // has no such line and the point of this block is to look like
    // theirs.
    crate::cli_output::print_logo(&mut io::stderr())?;
    crate::cli_output::Banner {
        model_path: &model,
        ftype: &crate::cli_output::ftype_name(file.metadata_u64("general.file_type")),
        modalities: "text",
    }
    .print(&mut io::stderr())?;
    if std::env::var_os("FRINK_QUIET").is_none() {
        eprintln!(
            "loaded in {:.2}s (tokenizer={}, ctx={ctx_size})",
            load_t.elapsed().as_secs_f64(),
            tokenizer.kind()
        );
    }
    crate::cli_output::print_prompt_echo(&mut io::stderr(), &args.prompt)?;

    let mut tokens = tokenizer.encode(&prompt, SpecialTokens::Parse);
    // Match llama.cpp vocab add_bos (qwen2/BPE default false). Blindly
    // prepending bos_token_id poisons Qwen2-MoE (`<|endoftext|>`).
    frink_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| frink_models::tokenizer::should_add_bos_token(&file)),
    );
    let vocab_size = decoder.config.vocab_size;
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = args.sampling(Some(&tokenizer), ctx_size)?;
    let seed = seed_from_args(args.seed);
    let sampler = Sampler::new(seed);
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        decoder.config.vocab_size,
    )?;

    #[cfg(feature = "metal")]
    let _metal_greedy_guard = {
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                frink_models::set_metal_greedy_argmax(false);
            }
        }
        // NOT `temperature <= 0.0` alone. The fold makes the stack
        // return ONE element holding the chosen id, and a grammar needs
        // one logit per vocabulary entry to mask. Gating on temperature
        // only produced exactly that: `--json-schema` at `--temp 0`
        // failed with "was handed 1 for a vocabulary of 128256".
        //
        // Same defect `frink-server`'s `greedy_gpu_fold_allowed` fixed
        // for `json_object`, and the third instance of it. The rule is
        // the server's: the fold is sound only when NOTHING needs to
        // inspect the vocabulary before a token is chosen.
        if sampling.temperature <= 0.0 && !step.needs_vocab_logits(&sampling) {
            frink_models::set_metal_greedy_argmax(true);
            Some(Guard)
        } else {
            None
        }
    };

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    // Speculative decoding with a real draft model, when `-d` names
    // one. The verification rule lives in `frink_models::speculative`
    // and is lossless at every temperature; this is only the wiring.
    if let Some(draft_path) = args.model_draft.as_deref() {
        return run_infer_speculative(
            &args,
            &decoder,
            draft_path,
            &tokenizer,
            &tokens,
            max_new,
            &sampling,
            seed,
            &stop_tokens,
            &mut caches,
        );
    }

    // Before the timer, never inside it: the first forward pass
    // builds every pipeline and allocates every buffer, and that
    // cost divided by a short prompt is what made this line read
    // 12.64 t/s where llama.cpp reads 197.7 for the same file.
    crate::cli_output::warm_up(&decoder);
    let prefill_t = Instant::now();
    let mut pos;
    let mut logits = if tokens.is_empty() {
        let l = decoder.forward_token(0, 0, &mut caches);
        pos = 1;
        l
    } else {
        let l = decoder.forward_batch_last(&tokens, 0, &mut caches);
        pos = tokens.len();
        l
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let Some(next) = step.next(&logits, &sampling, PenaltyWindow::new(&tokens, &generated))?
        else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
        if !args.ignore_eos && stop_tokens.contains(next) {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = decoder.forward_token(next, pos, &mut caches);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    crate::cli_output::Timings {
        prompt_tokens: prompt_n,
        prompt_secs: prefill_secs,
        predicted_tokens: gen_n,
        predicted_secs: decode_secs,
    }
    .print(&mut io::stderr())?;
    crate::cli_output::print_exiting(&mut io::stderr())?;

    Ok(())
}

/// Dense-lead DeepSeek-2 / Mistral-4 path via [`MlaEngine`].
fn run_mla_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = cli_tokenizer_from_gguf(file)?;
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = frink_models::tokenizer::StopTokens::from_gguf(file);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, false);
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    eprintln!(
        "frink: loading {} as MLA engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_mla_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Mla(engine) = served else {
        anyhow::bail!("expected ServedEngine::Mla");
    };
    eprintln!("frink: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt, SpecialTokens::Parse);
    frink_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| frink_models::tokenizer::should_add_bos_token(file)),
    );
    let vocab_size = Engine::vocab_size(&engine);
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = args.sampling(Some(&tokenizer), ctx_size)?;
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
    let mut state = Engine::new_state(&engine);

    // Before the timer, never inside it: the first forward pass
    // builds every pipeline and allocates every buffer, and that
    // cost divided by a short prompt is what made this line read
    // 12.64 t/s where llama.cpp reads 197.7 for the same file.
    crate::cli_output::warm_up_engine(&engine);
    let prefill_t = Instant::now();
    let mut pos = 0usize;
    let mut logits = if tokens.is_empty() {
        let l = engine.forward_token(0, 0, &mut state);
        pos = 1;
        l
    } else {
        let mut last = Vec::new();
        for &tok in &tokens {
            last = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        last
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let Some(next) = step.next(&logits, &sampling, PenaltyWindow::new(&tokens, &generated))?
        else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
        if !args.ignore_eos && stop_tokens.contains(next) {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = engine.forward_token(next, pos, &mut state);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    crate::cli_output::Timings {
        prompt_tokens: prompt_n,
        prompt_secs: prefill_secs,
        predicted_tokens: gen_n,
        predicted_secs: decode_secs,
    }
    .print(&mut io::stderr())?;
    Ok(())
}

/// GLM-5.2 / GLM4-family path via [`Glm52Engine`]./// Gemma-4 dedicated path via [`frink_models::Gemma4Engine`].
fn run_gemma4_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = cli_tokenizer_from_gguf(file)?;
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = frink_models::tokenizer::StopTokens::from_gguf(file);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, false);
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    eprintln!(
        "frink: loading {} as Gemma4 engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_gemma4_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Gemma4(engine) = served else {
        anyhow::bail!("expected ServedEngine::Gemma4");
    };
    let engine = *engine;
    eprintln!("frink: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt, SpecialTokens::Parse);
    frink_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| frink_models::tokenizer::should_add_bos_token(file)),
    );
    let vocab_size = Engine::vocab_size(&engine);
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = args.sampling(Some(&tokenizer), ctx_size)?;
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
    let mut state = Engine::new_state(&engine);

    // Before the timer, never inside it: the first forward pass
    // builds every pipeline and allocates every buffer, and that
    // cost divided by a short prompt is what made this line read
    // 12.64 t/s where llama.cpp reads 197.7 for the same file.
    crate::cli_output::warm_up_engine(&engine);
    let prefill_t = Instant::now();
    let mut pos = 0usize;
    let mut logits = if tokens.is_empty() {
        let l = engine.forward_token(0, 0, &mut state);
        pos = 1;
        l
    } else {
        let mut last = Vec::new();
        for &tok in &tokens {
            last = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        last
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let Some(next) = step.next(&logits, &sampling, PenaltyWindow::new(&tokens, &generated))?
        else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
        if !args.ignore_eos && stop_tokens.contains(next) {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = engine.forward_token(next, pos, &mut state);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    crate::cli_output::Timings {
        prompt_tokens: prompt_n,
        prompt_secs: prefill_secs,
        predicted_tokens: gen_n,
        predicted_secs: decode_secs,
    }
    .print(&mut io::stderr())?;
    Ok(())
}

/// GLM-5.2 / GLM4-family path via [`Glm52Engine`].
fn run_glm52_infer(args: InferArgs, path: &Path, file: &ShardedGguf) -> anyhow::Result<()> {
    let tokenizer = cli_tokenizer_from_gguf(file)?;
    // Not just `eos_token_id`: Llama-3 ends a turn with `<|eot_id|>` and
    // gemma-4 with `<turn|>`, neither of which is the metadata EOS.
    let stop_tokens = frink_models::tokenizer::StopTokens::from_gguf(file);
    let bos_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or("unknown");
    let gguf_ctx = file
        .metadata_u64(&format!("{arch}.context_length"))
        .map(|v| v as usize)
        .unwrap_or(4096);
    let ctx_size = resolve_ctx_size(&args, path, gguf_ctx)?;

    let chat = ChatKind::detect_for_gguf(file, false);
    let user_prompt = resolve_prompt(&args)?;
    let prompt = if args.no_cnv {
        user_prompt
    } else {
        chat.wrap_user(args.system.as_deref(), &user_prompt)?
    };

    eprintln!(
        "frink: loading {} as GLM-5.2 engine (tokenizer={}, ctx={ctx_size})",
        args.model.as_deref().unwrap_or("?"),
        tokenizer.kind()
    );
    let load_t = Instant::now();
    let served = load_glm52_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Glm52(engine) = served else {
        anyhow::bail!("expected ServedEngine::Glm52");
    };
    eprintln!("frink: loaded in {:.2}s", load_t.elapsed().as_secs_f64());

    let mut tokens = tokenizer.encode(&prompt, SpecialTokens::Parse);
    frink_models::tokenizer::prepend_bos(
        &mut tokens,
        bos_id.filter(|_| frink_models::tokenizer::should_add_bos_token(file)),
    );
    let vocab_size = Engine::vocab_size(&engine);
    if let Some(&bad) = tokens.iter().find(|&&t| t >= vocab_size) {
        anyhow::bail!("prompt token {bad} outside vocab_size {vocab_size}");
    }
    if tokens.len() >= ctx_size {
        anyhow::bail!(
            "prompt length {} >= context size {ctx_size}; raise -c or shorten prompt",
            tokens.len()
        );
    }

    let room = ctx_size - tokens.len();
    let max_new = if args.n_predict < 0 {
        room
    } else {
        (args.n_predict as usize).min(room)
    };

    let sampling = args.sampling(Some(&tokenizer), ctx_size)?;
    let sampler = Sampler::new(seed_from_args(args.seed));
    let mut step = token_step(
        &args,
        sampler,
        &tokenizer,
        &stop_tokens,
        engine.vocab_size(),
    )?;
    let mut state = Engine::new_state(&engine);

    // Before the timer, never inside it: the first forward pass
    // builds every pipeline and allocates every buffer, and that
    // cost divided by a short prompt is what made this line read
    // 12.64 t/s where llama.cpp reads 197.7 for the same file.
    crate::cli_output::warm_up_engine(&engine);
    let prefill_t = Instant::now();
    let mut pos = 0usize;
    let mut logits = if tokens.is_empty() {
        let l = engine.forward_token(0, 0, &mut state);
        pos = 1;
        l
    } else {
        let mut last = Vec::new();
        for &tok in &tokens {
            last = engine.forward_token(tok, pos, &mut state);
            pos += 1;
        }
        last
    };
    let prefill_secs = prefill_t.elapsed().as_secs_f64();

    let mut generated: Vec<usize> = Vec::with_capacity(max_new);
    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    for _ in 0..max_new {
        let Some(next) = step.next(&logits, &sampling, PenaltyWindow::new(&tokens, &generated))?
        else {
            // The grammar is satisfied and permits nothing further: a
            // finished answer, not a failure.
            break;
        };
        if !args.ignore_eos && stop_tokens.contains(next) {
            break;
        }
        generated.push(next);
        let piece = tokenizer.decode(&[next]);
        stdout.write_all(piece.as_bytes())?;
        stdout.flush()?;
        logits = engine.forward_token(next, pos, &mut state);
        pos += 1;
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let prompt_n = tokens.len();
    let gen_n = generated.len();
    crate::cli_output::Timings {
        prompt_tokens: prompt_n,
        prompt_secs: prefill_secs,
        predicted_tokens: gen_n,
        predicted_secs: decode_secs,
    }
    .print(&mut io::stderr())?;
    crate::cli_output::print_exiting(&mut io::stderr())?;

    Ok(())
}

/// `frink run` with a draft model, llama.cpp's `-md`.
///
/// The verification rule lives in `frink_models::speculative` and is
/// lossless at every temperature, not only at `--temp 0`. Nothing here
/// re-implements it: this function loads the second checkpoint, refuses
/// the combinations speculation cannot honour, and streams the tokens
/// the shared loop commits.
#[allow(clippy::too_many_arguments)]
fn run_infer_speculative(
    args: &InferArgs,
    decoder: &frink_models::Decoder,
    draft_path: &str,
    tokenizer: &CliTokenizer,
    tokens: &[usize],
    max_new: usize,
    sampling: &frink_models::sampling::SamplingParams,
    seed: u64,
    stop_tokens: &frink_models::tokenizer::StopTokens,
    caches: &mut [KvCache],
) -> anyhow::Result<()> {
    // A grammar masks the candidate set per token. Speculation compares
    // the drafter's probability for a token against the target's for
    // the same token, and neither of those distributions is the masked
    // one, so running both would either break the constraint or break
    // losslessness. Refused by name rather than silently dropping one
    // of the two, which is the failure this engine exists not to have:
    // a grammar that is accepted and not applied is served with a 200
    // and read as compliance.
    if args.grammar_source()?.is_some() {
        anyhow::bail!(
            "--model-draft cannot be combined with --grammar / --grammar-file / --json-schema \
             yet: constrained decoding masks the candidate set per token, and the speculative \
             rejection rule compares unmasked draft and target probabilities, so the two \
             together would either drop the constraint or stop being lossless. Run with one or \
             the other"
        );
    }
    if tokens.is_empty() {
        anyhow::bail!("--model-draft needs a prompt to continue");
    }
    if decoder.config.has_recurrent_layers() {
        anyhow::bail!(
            "--model-draft cannot be used with a target model that has recurrent (Mamba) \
             layers: a rejected draft rolls the KV caches back to the last accepted position, \
             and a Mamba layer's state is a reduction over the whole prefix that cannot be \
             rolled back (llama.cpp's server re-prefills such models for the same reason)"
        );
    }

    let config = frink_models::ModelConfig::from_gguf(&frink_gguf::ShardedGguf::open(draft_path)?)?;
    if config.has_recurrent_layers() {
        anyhow::bail!(
            "--model-draft cannot be a model with recurrent (Mamba) layers: the draft cache is \
             rolled back after every verification block"
        );
    }
    let draft = frink_models::Decoder::from_gguf(draft_path, config)?;
    eprintln!("frink: draft model {draft_path}");

    // Refused at construction when the vocabularies differ. The two
    // models must number their tokens identically or the rejection rule
    // is comparing probabilities of different tokens, which produces
    // fluent text with a plausible accept rate and no error at all.
    let mut drafter = frink_models::DraftModelSpeculator::new(
        draft,
        &decoder.config,
        sampling.clone(),
        seed,
        args.draft_max,
        args.draft_p_min,
    )?;

    // One warm-up proposal, to find out whether this drafter's KV
    // actually lands in the host caches it owns. A backend that keeps
    // KV on the device leaves them empty, and a drafter that cannot see
    // its own rows cannot roll back the ones the target rejected. Found
    // by running it: on Metal this panicked mid-answer, after the first
    // block had already been printed.
    {
        use frink_models::speculative::Drafter;
        let _ = drafter.propose(tokens, &[], 1);
    }
    if !drafter.keeps_host_kv() {
        anyhow::bail!(
            "--model-draft needs the draft model's KV cache in host memory, and this \
             backend keeps it on the device, so the drafter cannot roll back the \
             positions the target rejects. Re-run with --device cpu, or without \
             --model-draft. Speculative decoding on a device-resident KV cache is \
             not implemented yet"
        );
    }

    let mut stdout = io::stdout().lock();
    let decode_t = Instant::now();
    let mut emitted = 0usize;
    let mut write_err = None;

    let result = frink_models::speculative::speculative_decode_observed(
        decoder,
        tokens,
        caches,
        &mut drafter,
        &mut |token| {
            if !args.ignore_eos && stop_tokens.contains(token) {
                return false;
            }
            let piece = tokenizer.decode(&[token]);
            if let Err(e) = stdout
                .write_all(piece.as_bytes())
                .and_then(|()| stdout.flush())
            {
                write_err = Some(e);
                return false;
            }
            emitted += 1;
            true
        },
        &frink_models::speculative::SpeculativeOptions {
            max_new_tokens: max_new,
            start_pos: 0,
            sampling: sampling.clone(),
            seed,
        },
    );
    if let Some(e) = write_err {
        return Err(e.into());
    }
    let decode_secs = decode_t.elapsed().as_secs_f64();
    writeln!(stdout)?;

    let tps = if decode_secs > 0.0 {
        emitted as f64 / decode_secs
    } else {
        0.0
    };
    eprintln!(
        "frink: predict {emitted} tokens, {tps:.2} t/s over {} verification steps",
        result.verification_steps
    );
    // Reported as a pair with the throughput, and per position rather
    // than folded into the mean: a drafter that is right at position 0
    // and useless by position 7 has the same mean as a uniformly
    // mediocre one, and the two want opposite block sizes. A speedup
    // without an accept rate cannot be reproduced or debugged.
    match result.acceptance_length() {
        Some(len) => eprintln!(
            "frink: acceptance length {len:.2} tokens/step, accepted {} of {} drafted",
            result.accepted_tokens, result.drafted_tokens
        ),
        // `None` and 1.00 are different answers: "the drafter never got
        // to propose" is not "it proposed and never helped".
        None => eprintln!("frink: the drafter proposed nothing, so no acceptance length exists"),
    }
    let per_pos: Vec<String> = result
        .accept_rate_per_position()
        .iter()
        .map(|r| format!("{r:.2}"))
        .collect();
    if !per_pos.is_empty() {
        eprintln!("frink: accept rate per position [{}]", per_pos.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::GpuLayers;
    use std::str::FromStr;

    use super::{banner_line, kv_elem_for, InferArgs, OffloadDevice};
    use clap::Parser;

    /// `InferArgs` is a `clap::Args` group, not a `Parser`, so the test
    /// gives it the top-level command it is normally flattened into.
    #[derive(Parser, Debug)]
    struct Cli {
        #[command(flatten)]
        infer: InferArgs,
    }

    fn args(argv: &[&str]) -> InferArgs {
        let mut full = vec!["frink"];
        full.extend_from_slice(argv);
        Cli::parse_from(full).infer
    }

    /// GitHub issue #170, at the flag rather than at the predicate: the
    /// DEFAULT `frink run -m … --temp 0 --ngl 99` must not let Metal
    /// fold `lm_head + argmax` onto the device.
    ///
    /// `--repeat-penalty` defaults to **1.1** here, deliberately unlike
    /// llama.cpp's 1.0 (`docs/FEATURES.md` records the difference), and
    /// a device argmax over raw logits never applies it. The fold gate
    /// read a predicate that tested XTC, typical-p and DRY and not the
    /// penalties, so a plain greedy Metal run returned a token the host
    /// sampler would not have chosen -- and agreed instead, byte for
    /// byte, with the same run at `--repeat-penalty 1.0`.
    ///
    /// This test is here and not only in `frink-models` because the
    /// DEFAULT is the thing that made it live: the predicate and the
    /// flag are two structures that have to agree, and `frink-models`
    /// cannot see this crate's `default_value_t`.
    ///
    /// Sabotage: set `default_value_t = 1.0` on `--repeat-penalty`; the
    /// first assertion goes red.
    #[test]
    fn the_default_flags_forbid_the_metal_greedy_argmax_fold() {
        let step = super::TokenStep::new(frink_models::sampling::Sampler::new(1), None);
        let sampling = |argv: &[&str]| {
            args(argv)
                .sampling(None, 4096)
                .expect("no --dry-multiplier, so no vocabulary is needed")
        };

        let defaults = sampling(&["-m", "m.gguf", "--temp", "0"]);
        assert_eq!(defaults.repetition_penalty, 1.1, "llama.cpp's is 1.0");
        assert!(
            step.needs_vocab_logits(&defaults),
            "the default repetition penalty is applied on the host, so the \
             device must hand back a vocabulary and not one token id"
        );

        // Both of llama.cpp's off switches restore the fold, which is
        // what makes the assertion above about the penalty and not about
        // `--top-k 40` or `--min-p 0.05`, which default on too.
        for off in [
            ["-m", "m.gguf", "--temp", "0", "--repeat-penalty", "1.0"],
            ["-m", "m.gguf", "--temp", "0", "--repeat-last-n", "0"],
        ] {
            let s = sampling(&off);
            assert!(
                !step.needs_vocab_logits(&s),
                "{off:?} switches the penalties off, so the fold is exact again"
            );
        }
    }

    /// The banner may not promise a KV dtype the run will not use.
    ///
    /// `frink -m m.gguf -dev cpu -ngl all` printed `ctk=f16` because
    /// the banner echoed the flag's default, while the host `KvCache`
    /// is `Vec<f32>` and the budget priced it at f32. That made the
    /// memory warning look like a bug: it charged 229376 bytes/token
    /// where the banner implied 114688, so a 3B model at its 131072
    /// trained context read as 37.5 GB instead of 22.5 GB. The warning
    /// was right and the banner was wrong.
    ///
    /// The flag and the store are two things that must agree, so the
    /// banner now derives from `kv_elem_for`, the same function the
    /// budget prices with.
    #[test]
    fn the_banner_reports_the_kv_dtype_the_run_will_actually_keep() {
        let a = args(&["-m", "m.gguf", "--device", "cpu", "--ctk", "f16"]);
        assert_eq!(kv_elem_for(&a).as_str(), "f32", "the host KV cache is f32");

        let line = banner_line(&a, OffloadDevice::Cpu);
        assert!(line.contains("ctk=f32"), "{line}");
        assert!(
            !line.contains("ctk=f16"),
            "the banner echoed the flag: {line}"
        );
        assert!(
            line.contains("--ctk f16 ignored"),
            "a flag with no effect must say so: {line}"
        );
    }

    /// GitHub issue #297: `FRINK_CTK` is documented as "same as
    /// `--ctk`" and could not be.
    ///
    /// The resolution writes `args.ctk` into the variable
    /// unconditionally, so before clap read it as the default an
    /// environment that said `q4` was overwritten with the flag's
    /// `f16` before `frink_metal::attn::metal_kv_dtype` ever looked.
    /// Nothing checked it, which is why a documented spelling shipped
    /// dead.
    ///
    /// Serialised, because it mutates process-wide state and the other
    /// tests in this file parse the same argument.
    #[test]
    fn the_environment_supplies_the_kv_dtype_when_the_flag_does_not() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let restore = std::env::var("FRINK_CTK").ok();

        // SAFETY: single-threaded test body, holding ENV_LOCK.
        unsafe { std::env::set_var("FRINK_CTK", "q4_0") };
        let a = args(&["-m", "m.gguf"]);
        assert_eq!(a.ctk, "q4_0", "the environment was ignored");

        // An explicit flag still wins over it.
        let a = args(&["-m", "m.gguf", "--ctk", "q8_0"]);
        assert_eq!(a.ctk, "q8_0", "the flag lost to the environment");

        // SAFETY: same.
        unsafe {
            match restore {
                Some(v) => std::env::set_var("FRINK_CTK", v),
                None => std::env::remove_var("FRINK_CTK"),
            }
        }
    }

    /// `-dev cpu` is not `-dev none`, and `-ngl all` under either does
    /// nothing. Both were printed as if honoured.
    #[test]
    fn the_banner_does_not_promise_gpu_layers_on_a_cpu_device() {
        let a = args(&["-m", "m.gguf", "--device", "cpu", "--ngl", "all"]);
        let line = banner_line(&a, OffloadDevice::Cpu);
        assert!(line.contains("device=cpu"), "{line}");
        assert!(line.contains("ignored, no GPU offload"), "{line}");
    }

    /// And a run that really does select the dtype keeps a clean line.
    #[test]
    fn a_metal_run_reports_the_requested_dtype_with_no_caveat() {
        let a = args(&[
            "-m", "m.gguf", "--device", "metal", "--ngl", "all", "--ctk", "f16",
        ]);
        let line = banner_line(&a, OffloadDevice::Metal);
        assert!(line.contains("ctk=f16"), "{line}");
        assert!(!line.contains("ignored"), "{line}");
    }

    /// `-ngl N` must not silently mean "all layers".
    ///
    /// llama.cpp's `-ngl N` puts exactly N layers in VRAM and runs the
    /// rest on the CPU, which is how people fit a model that otherwise
    /// does not fit. frink parsed the count and read only
    /// `offload_enabled()`, a bool, so `--ngl 10` on a 32-layer model
    /// offloaded all 32 -- same flag, same value, no error, and the
    /// failure arrives as an OOM on the machine the flag existed to
    /// accommodate.
    ///
    /// Partial offload is not implemented, so it refuses. The two exact
    /// cases still work.
    #[test]
    fn a_partial_gpu_layer_count_is_refused_rather_than_rounded_up() {
        let err = GpuLayers::Count(10)
            .check_supported(32)
            .expect_err("10 of 32 is partial");
        let msg = err.to_string();
        assert!(msg.contains("PARTIAL"), "{msg}");
        assert!(
            msg.contains("--ngl 0"),
            "the message must say what works: {msg}"
        );
        assert!(msg.contains("--ngl 32"), "{msg}");

        // The exact cases are not partial and must stay accepted.
        GpuLayers::Count(0)
            .check_supported(32)
            .expect("0 = CPU only");
        GpuLayers::Count(32).check_supported(32).expect("32 = all");
        GpuLayers::Count(99)
            .check_supported(32)
            .expect("clamps to all");
        GpuLayers::All.check_supported(32).expect("all");
        GpuLayers::Auto.check_supported(32).expect("auto");
    }

    /// `--escape` defaults TRUE, as llama.cpp does.
    ///
    /// frink defaulted false, so `-p "a\\nb"` reached the model as a
    /// literal backslash-n on frink and as a newline on llama.cpp: the
    /// same command, a different prompt, and no error on either side.
    #[test]
    fn escapes_are_processed_by_default_like_llama_cpp() {
        use clap::Parser;
        #[derive(Parser)]
        struct Probe {
            #[command(flatten)]
            args: super::InferArgs,
        }
        let parsed = Probe::try_parse_from(["frink", "-m", "x.gguf"]).expect("defaults parse");
        assert!(parsed.args.escape, "llama.cpp common/common.h:563 is true");
        assert!(!parsed.args.no_escape);

        let off = Probe::try_parse_from(["frink", "-m", "x.gguf", "--no-escape"])
            .expect("--no-escape parses");
        assert!(off.args.no_escape, "llama.cpp spells the negation this way");
    }

    /// `--samplers` is llama.cpp's, and the DEFAULT is the chain frink
    /// already ran: a command line that does not mention the flag must
    /// sample exactly what it did before the flag existed.
    ///
    /// The distribution-level proof of that is
    /// `frink_models::sampling::tests::the_default_order_is_the_chain_frink_already_ran`;
    /// this is the CLI half -- that the flag's default resolves to the
    /// same chain, so the two cannot drift apart.
    #[test]
    fn samplers_defaults_to_the_chain_frink_already_ran() {
        let default = args(&["-m", "x.gguf"])
            .sampling(None, 4096)
            .expect("no dry, so no vocabulary is needed")
            .sampler_order;
        assert_eq!(default, frink_models::SamplerOrder::default());
        assert_eq!(
            default.to_string(),
            "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
        );
    }

    /// A caller-supplied order reaches the sampler, in the order typed.
    #[test]
    fn a_caller_supplied_order_reaches_the_sampler() {
        let order = args(&["-m", "x.gguf", "--samplers", "penalties;temperature;top_k"])
            .sampling(None, 4096)
            .expect("no dry")
            .sampler_order;
        assert_eq!(order.to_string(), "penalties;temperature;top_k");
        // llama.cpp's own aliases, so an upstream command line works.
        assert_eq!(
            args(&["-m", "x.gguf", "--samplers", "top-k;min-p;temp"])
                .sampling(None, 4096)
                .expect("no dry")
                .sampler_order
                .to_string(),
            "top_k;min_p;temperature"
        );
    }

    /// A sampler frink does not implement is refused BY NAME at the
    /// command line, rather than dropped out of the chain.
    ///
    /// Upstream's own default string names three samplers this engine
    /// lacks, so pasting it must fail loudly. A caller who asked for
    /// `xtc` and was given a chain without it was silently handed a
    /// different sampler.
    #[test]
    fn a_sampler_frink_lacks_is_refused_by_name_on_the_command_line() {
        let err = Cli::try_parse_from([
            "frink",
            "-m",
            "x.gguf",
            "--samplers",
            "penalties;mirostat;temperature",
        ])
        .expect_err("mirostat is not a chain member here")
        .to_string();
        assert!(err.contains("mirostat"), "{err}");
        assert!(err.contains("not implemented"), "{err}");

        // And llama.cpp's OWN default string now parses, which is the
        // point of this change: pasting an upstream command line works.
        Cli::try_parse_from([
            "frink",
            "-m",
            "x.gguf",
            "--samplers",
            "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature",
        ])
        .expect("llama.cpp's default chain is frink's default chain");

        let unknown = Cli::try_parse_from(["frink", "-m", "x.gguf", "--samplers", "top_kk"])
            .expect_err("no such sampler")
            .to_string();
        assert!(unknown.contains("top_kk"), "{unknown}");
        assert!(unknown.contains("unknown sampler"), "{unknown}");
    }

    #[test]
    fn parses_llama_gpu_layer_values() {
        assert_eq!(GpuLayers::from_str("0"), Ok(GpuLayers::Count(0)));
        assert_eq!(GpuLayers::from_str("42"), Ok(GpuLayers::Count(42)));
        assert_eq!(GpuLayers::from_str("auto"), Ok(GpuLayers::Auto));
        assert_eq!(GpuLayers::from_str("all"), Ok(GpuLayers::All));
        assert!(GpuLayers::from_str("-1").is_err());
        assert!(GpuLayers::from_str("some").is_err());
    }
}
