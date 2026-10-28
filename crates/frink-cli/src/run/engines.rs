//! `frink run` for the dedicated engines (MLA, Gemma-4, GLM-5.2), which
//! do not go through `Decoder`. Moved out of `run.rs` verbatim.

use super::*;

/// Dense-lead DeepSeek-2 / Mistral-4 path via [`MlaEngine`].
pub(super) fn run_mla_infer(
    args: InferArgs,
    path: &Path,
    file: &ShardedGguf,
) -> anyhow::Result<()> {
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

    let prompt = final_prompt(&args, file)?;

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
pub(super) fn run_gemma4_infer(
    args: InferArgs,
    path: &Path,
    file: &ShardedGguf,
) -> anyhow::Result<()> {
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

    let prompt = final_prompt(&args, file)?;

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
pub(super) fn run_glm52_infer(
    args: InferArgs,
    path: &Path,
    file: &ShardedGguf,
) -> anyhow::Result<()> {
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

    let prompt = final_prompt(&args, file)?;

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

/// Refuses every flag the dedicated engines (MLA, Gemma-4, GLM-5.2)
/// would accept and not honour. They do not go through `Decoder`, so
/// adapters, a draft model and a partial offload never reach them.
///
/// The destructure has no `..`: a field added to [`InferArgs`] does not
/// compile until somebody decides here whether these engines honour it.
pub(super) fn refuse_what_dedicated_engines_ignore(
    args: &InferArgs,
    arch: &str,
    file: &ShardedGguf,
) -> anyhow::Result<()> {
    let InferArgs {
        model: _,
        hf_repo: _,
        hf_file: _,
        presence_penalty: _,
        frequency_penalty: _,
        prompt: _,
        file: _,
        n_predict: _,
        ctx_size: _,
        strict_budget: _,
        threads: _,
        temperature: _,
        top_k: _,
        top_p: _,
        grammar: _,
        grammar_file: _,
        json_schema: _,
        min_p: _,
        repeat_last_n: _,
        repeat_penalty: _,
        typical_p: _,
        top_n_sigma: _,
        xtc_probability: _,
        xtc_threshold: _,
        dry_multiplier: _,
        dry_base: _,
        dry_allowed_length: _,
        dry_penalty_last_n: _,
        dry_sequence_breaker: _,
        samplers: _,
        sampler_seq: _,
        seed: _,
        device: _,
        list_devices: _,
        n_gpu_layers,
        model_draft,
        lora,
        lora_scaled,
        draft_max: _,
        draft_p_min: _,
        system: _,
        no_cnv: _,
        single_turn: _,
        escape: _,
        no_escape: _,
        ignore_eos: _,
        verbose_prompt: _,
        mtp: _,
        ctk: _,
    } = args;
    if !lora.is_empty() || !lora_scaled.is_empty() {
        anyhow::bail!(
            "--lora is not implemented for the {arch} engine (only the generic decoder \
             attaches adapters); refusing rather than running the base weights"
        );
    }
    if model_draft.is_some() {
        anyhow::bail!(
            "--model-draft is not implemented for the {arch} engine (only the generic decoder \
             speculates); refusing rather than decoding without the draft"
        );
    }
    if let GpuLayers::Count(n) = n_gpu_layers {
        if *n > 0 {
            let n_layers = file
                .metadata_u64(&format!("{arch}.block_count"))
                .ok_or_else(|| anyhow::anyhow!("--n-gpu-layers {n}: {arch}.block_count missing"))?;
            n_gpu_layers.check_supported(n_layers as usize)?;
        }
    }
    Ok(())
}
