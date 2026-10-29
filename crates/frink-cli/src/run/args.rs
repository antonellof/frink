//! `frink run`'s flags: the clap surface and the value types it parses into.
//!
//! Moved out of `run.rs` verbatim (a pure move) so the flag surface can
//! change without editing a 2500-line file.

use std::fmt;
use std::str::FromStr;

use clap::{Args, ValueEnum};
use frink_models::{SamplerOrder, SamplingParams};

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
        visible_aliases = ["hf", "hfr"],
        value_name = "REPO[:QUANT]",
        conflicts_with = "model"
    )]
    pub hf_repo: Option<String>,

    /// Exact filename inside `--hf-repo`, llama.cpp's `-hff`.
    #[arg(
        long = "hf-file",
        visible_alias = "hff",
        value_name = "FILE",
        requires = "hf_repo"
    )]
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
    #[arg(long = "temp", visible_alias = "temperature", default_value_t = 0.8)]
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

    /// Read the JSON Schema from a file, llama.cpp's `-jf` /
    /// `--json-schema-file`. Exclusive with the other two grammar flags.
    #[arg(long = "json-schema-file", visible_alias = "jf", value_name = "FILE")]
    pub json_schema_file: Option<std::path::PathBuf>,

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
    #[arg(
        long = "samplers",
        value_name = "LIST",
        default_value_t = SamplerOrder::default()
    )]
    pub samplers: SamplerOrder,

    /// The same chain as `--samplers`, one character per sampler,
    /// llama.cpp's `--sampler-seq` (`edskypmxt`).
    ///
    /// It was an alias of `--samplers` once, which read upstream's
    /// character string as one unknown sampler NAME: the spelling was
    /// accepted and its grammar was not. The characters are
    /// `common_sampler_type_to_chr`'s.
    #[arg(
        long = "sampler-seq",
        visible_alias = "sampling-seq",
        value_name = "CHARS",
        conflicts_with = "samplers",
        value_parser = |s: &str| SamplerOrder::from_chars(s).map_err(|e| e.to_string())
    )]
    pub sampler_seq: Option<SamplerOrder>,

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
    #[arg(
        long = "model-draft",
        short = 'd',
        visible_aliases = ["md", "spec-draft-model"],
        value_name = "FILE"
    )]
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
        visible_aliases = ["draft", "draft-n"],
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
    #[arg(
        long = "draft-p-min",
        visible_alias = "spec-draft-p-min",
        value_name = "P",
        default_value_t = 0.75
    )]
    pub draft_p_min: f32,

    /// System prompt, placed by the chat template, llama.cpp's `-sys` /
    /// `--system-prompt`. Refused with `--no-cnv`, which has no template
    /// to place it in.
    #[arg(long = "system", visible_aliases = ["sys", "system-prompt"])]
    pub system: Option<String>,

    /// Read the system prompt from a file, llama.cpp's `-sysf` /
    /// `--system-prompt-file`. One trailing newline is dropped, as
    /// upstream drops it (`common/arg.cpp:1828`); the rest is verbatim.
    #[arg(
        long = "system-prompt-file",
        visible_alias = "sysf",
        value_name = "FNAME",
        conflicts_with = "system"
    )]
    pub system_prompt_file: Option<std::path::PathBuf>,

    /// Accepted and already what happens: frink renders the GGUF's own
    /// `tokenizer.chat_template` with a Jinja engine, which is what
    /// llama.cpp's `--jinja` turns on.
    #[arg(long = "jinja", default_value_t = false)]
    pub jinja: bool,

    /// Refused: frink has no non-Jinja template matcher to fall back to.
    /// `--no-cnv` is the raw completion.
    #[arg(long = "no-jinja", default_value_t = false)]
    pub no_jinja: bool,

    /// Run one throwaway forward pass before the timed prompt, llama.cpp's
    /// `--warmup` (the default, on both).
    #[arg(long = "warmup", overrides_with = "no_warmup", default_value_t = false)]
    pub warmup: bool,

    /// Skip the throwaway forward pass, llama.cpp's `--no-warmup`. The
    /// prompt rate then includes first-dispatch cost.
    #[arg(long = "no-warmup", overrides_with = "warmup", default_value_t = false)]
    pub no_warmup: bool,

    /// Flash attention, llama.cpp's `-fa` (`on`, `off`, `auto`). `on`
    /// and `auto` are what frink does wherever the backend has a fused
    /// kernel; `off` is refused ([`frink_core::argv::check_flash_attn`]).
    #[arg(
        long = "flash-attn",
        visible_alias = "fa",
        value_name = "MODE",
        num_args = 0..=1,
        default_missing_value = "auto"
    )]
    pub flash_attn: Option<String>,

    /// Hugging Face access token for `-hf`, llama.cpp's `-hft` /
    /// `--hf-token`. Sets `HF_TOKEN`, the variable the downloader reads.
    #[arg(long = "hf-token", visible_alias = "hft", value_name = "TOKEN")]
    pub hf_token: Option<String>,

    /// Print the timing lines after the answer, llama.cpp's
    /// `--show-timings` (the default).
    #[arg(
        long = "show-timings",
        overrides_with = "no_show_timings",
        default_value_t = false
    )]
    pub show_timings: bool,

    /// Do not print the timing lines, llama.cpp's `--no-show-timings`.
    #[arg(
        long = "no-show-timings",
        overrides_with = "show_timings",
        default_value_t = false
    )]
    pub no_show_timings: bool,

    /// Colour the prompt echo and the timings on stderr, llama.cpp's
    /// `-co` / `--color` (`on`, `off`, `auto`; `auto` is the default and
    /// means "when stderr is a terminal"). A bare `-co` means `on`.
    #[arg(
        long = "color",
        visible_alias = "co",
        value_name = "on|off|auto",
        num_args = 0..=1,
        default_missing_value = "on"
    )]
    pub color: Option<String>,

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
    #[arg(long = "single-turn", visible_alias = "st", default_value_t = false)]
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

impl InferArgs {
    /// Whether the throwaway warm-up pass runs (`--warmup`, the default,
    /// or `--no-warmup`, whichever came last).
    pub fn warmup_enabled(&self) -> bool {
        !self.no_warmup
    }

    /// The llama.cpp flags that are settled before anything is loaded,
    /// once, for every engine path: refusals, `-sysf` read into
    /// `--system`, `-hft` into `HF_TOKEN`, and the stderr presentation
    /// (`-co`, `--no-show-timings`).
    ///
    /// Called at the top of `run_infer`, before the dispatch to the
    /// dedicated engines, so no path can see the flags unapplied.
    pub fn apply_llama_flags(&mut self) -> anyhow::Result<()> {
        if self.no_jinja {
            anyhow::bail!(
                "--no-jinja: frink has no non-Jinja template mode. It renders the GGUF's own \
                 tokenizer.chat_template, which is what llama.cpp's --jinja turns on, and has \
                 no built-in template matcher to fall back to. Use --no-cnv for a raw completion"
            );
        }
        if let Some(mode) = &self.flash_attn {
            frink_core::argv::check_flash_attn(mode).map_err(|e| anyhow::anyhow!(e))?;
        }
        if let Some(token) = &self.hf_token {
            // SAFETY: single-threaded, before any worker or download.
            unsafe { std::env::set_var("HF_TOKEN", token) };
        }
        self.read_system_prompt_file()?;
        crate::cli_output::set_colour(self.colour_choice()?);
        crate::cli_output::set_show_timings(self.show_timings_enabled());
        Ok(())
    }

    /// `-sysf FILE` read into `--system`, dropping one trailing newline
    /// as llama.cpp does (`common/arg.cpp`, `--system-prompt-file`).
    pub(crate) fn read_system_prompt_file(&mut self) -> anyhow::Result<()> {
        if let Some(path) = self.system_prompt_file.take() {
            let mut text = std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("--system-prompt-file {}: {e}", path.display()))?;
            if text.ends_with('\n') {
                text.pop();
            }
            self.system = Some(text);
        }
        Ok(())
    }

    /// `-co`: `Some(true)` on, `Some(false)` off, `None` for `auto` or
    /// no flag (colour when stderr is a terminal). A value llama.cpp's
    /// `--color` would throw on is refused.
    pub(crate) fn colour_choice(&self) -> anyhow::Result<Option<bool>> {
        use frink_core::argv::Switch;
        let Some(value) = &self.color else {
            return Ok(None);
        };
        match frink_core::argv::llama_switch(value.trim()) {
            Some(Switch::On) => Ok(Some(true)),
            Some(Switch::Off) => Ok(Some(false)),
            Some(Switch::Auto) => Ok(None),
            None => anyhow::bail!("--color {value}: expected on, off or auto"),
        }
    }

    /// Whether timing lines print (`--show-timings`, the default, or
    /// `--no-show-timings`, whichever came last).
    pub fn show_timings_enabled(&self) -> bool {
        !self.no_show_timings
    }

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
            self.json_schema_file.is_some(),
        ]
        .iter()
        .filter(|b| **b)
        .count();
        if given > 1 {
            anyhow::bail!(
                "--grammar, --grammar-file, --json-schema and --json-schema-file are mutually \
                 exclusive; pass exactly one"
            );
        }
        if let Some(path) = &self.json_schema_file {
            let schema = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("--json-schema-file {}: {e}", path.display()))?;
            return Ok(Some(
                frink_models::grammar::json_schema_to_grammar(&schema)
                    .map_err(|e| anyhow::anyhow!("--json-schema-file: {e}"))?,
            ));
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
    /// two chat paths), which is the shape ADR 0004 names as this
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
            sampler_order: self.sampler_seq.unwrap_or(self.samplers),
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

pub use frink_models::gpu_layers::GpuLayers;

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
