//! llama.cpp-style GGUF completion (`-m` / `-p` / `-n` / …).

use std::io::{self, Read, Write};
use std::path::Path;

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use frink_core::cache::KvCache;
use frink_gguf::ShardedGguf;
use frink_models::tokenizer::SpecialTokens;
use frink_models::{
    ensure_generic_decoder, load_gemma4_engine_from_path, load_glm52_engine_from_path,
    load_mla_engine_from_path, select_engine_kind, Decoder, Engine, GgufBpeTokenizer,
    GgufPlamo2Tokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, ModelConfig, PenaltyWindow,
    Sampler, SelectedEngineKind, ServedEngine,
};

mod args;
mod engines;
mod prompt;
mod tokenizer;
pub use args::*;
use engines::{
    refuse_what_dedicated_engines_ignore, run_gemma4_infer, run_glm52_infer, run_mla_infer,
};
use prompt::*;
use tokenizer::*;

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
        // A block-quantized wire on the host store (it keeps f16), or any
        // dtype on CUDA (it keeps f32): `host_kv_dtype` says why.
        format!(
            " (--ctk {requested_ctk} has no {} host store; using {})",
            budget_backend_for(args).as_str(),
            effective_ctk.as_str()
        )
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

/// Width of the KV store the selected backend will really keep: the
/// Metal device wire `--ctk` selects, or the host store
/// [`host_kv_dtype_for`] builds. The same function decides the latter
/// for the budget and for the caches, so the two cannot disagree.
fn kv_elem_for(args: &InferArgs) -> frink_models::KvElem {
    use frink_models::{BudgetBackend, KvElem};
    match budget_backend_for(args) {
        BudgetBackend::Metal => KvElem::from_ctk(&args.ctk),
        BudgetBackend::Cuda | BudgetBackend::Cpu => KvElem::from(host_kv_dtype_for(args)),
    }
}

/// What this run's host K/V caches store (`frink_models::host_kv_dtype`).
fn host_kv_dtype_for(args: &InferArgs) -> frink_core::kv_store::KvDtype {
    frink_models::host_kv_dtype(budget_backend_for(args), &args.ctk)
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
    let mut args = args;
    args.apply_llama_flags()?;
    // `-hf` resolves to a local path before anything else looks at
    // `--model`, so the rest of this function sees one kind of input.
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
    // Parsed for every path, so a malformed spec is refused by name
    // whichever engine the checkpoint selects.
    let lora_specs = frink_models::lora_attach::LoraSpec::from_flags(&args.lora, &args.lora_scaled)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if matches!(
        select_engine_kind(&arch_early),
        Ok(SelectedEngineKind::Mla | SelectedEngineKind::Gemma4)
    ) || arch_early == "glm-dsa"
    {
        refuse_what_dedicated_engines_ignore(&args, &arch_early, &file)?;
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

    let prompt = final_prompt(&args, &file)?;

    let load_t = Instant::now();
    // LongRoPE picks its factor set from the run's context size, not the
    // checkpoint's advertised maximum (llama.cpp does the same, per
    // request, from `cparams.n_ctx_seq`).
    let mut config = config;
    config.apply_runtime_context(ctx_size);
    config.kv_dtype = host_kv_dtype_for(&args);
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

    // Before the timer, never inside it: the first forward pass
    // builds every pipeline and allocates every buffer, and that
    // cost divided by a short prompt is what made this line read
    // 12.64 t/s where llama.cpp reads 197.7 for the same file. Above
    // the speculative branch, so `--warmup` means the same on it.
    crate::cli_output::warm_up(&decoder, args.warmup_enabled());

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

    let mut config =
        frink_models::ModelConfig::from_gguf(&frink_gguf::ShardedGguf::open(draft_path)?)?;
    // The draft's caches at the target's width: one `--ctk` for the run.
    config.kv_dtype = decoder.config.kv_dtype;
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

    if !crate::cli_output::show_timings() {
        return Ok(());
    }
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
mod tests;
