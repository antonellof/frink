//! Minimal greedy decode used by [`crate::verify`].
//!
//! Deliberately independent of `run.rs`: this is a reference harness, and
//! it should not drift when the CLI's prompt handling, chat templating or
//! sampling changes. Raw prompt, no chat template, greedy argmax.

use anyhow::Context;
use frink_core::cache::KvCache;
use frink_gguf::ShardedGguf;
use frink_models::config::ModelConfig;
use frink_models::decoder::Decoder;
use frink_models::engine::Engine;
use frink_models::engine_factory::{
    load_gemma4_engine_from_path, load_mla_engine_from_path, select_engine_kind,
    SelectedEngineKind, ServedEngine,
};
use frink_models::tokenizer::{
    GgufBpeTokenizer, GgufPlamo2Tokenizer, GgufSpmTokenizer, GgufUnigramTokenizer, SpecialTokens,
};
use frink_models::GEMMA4_ARCHES;
use std::path::Path;

enum Tok {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    Plamo2(Box<GgufPlamo2Tokenizer>),
}

impl Tok {
    fn encode(&self, text: &str, specials: SpecialTokens) -> Vec<usize> {
        match self {
            Tok::Bpe(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|i| i as usize)
                .collect(),
            Tok::Spm(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|i| i as usize)
                .collect(),
            Tok::Unigram(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|i| i as usize)
                .collect(),
            Tok::Plamo2(t) => t
                .encode(text, specials)
                .into_iter()
                .map(|i| i as usize)
                .collect(),
        }
    }
}

/// Greedy-decodes `n` tokens from `prompt` and returns their ids plus
/// the prompt's tokenized length.
///
/// Stops early at EOS — a shorter sequence still compares correctly,
/// because the caller checks length as well as contents.
///
/// `prompt_tokens`, when set, stretches the prompt to exactly that many
/// tokens by repeating it (BOS kept once, at the front). This exists
/// because the prefill attention kernels — the ones this tool is meant
/// to catch — are gated on batch size (`n_q >= 8` for `fa_ext`), so a
/// short prompt verifies the decode path twice and the prefill path
/// never. Repeated text is fine: the check is kernel agreement, not
/// output quality.
pub fn greedy_token_ids(
    path: &Path,
    prompt: &str,
    specials: SpecialTokens,
    n: usize,
    prompt_tokens: Option<usize>,
) -> anyhow::Result<(Vec<u32>, usize)> {
    let (decoder, tokens, eos) = load_and_tokenize(path, prompt, specials, prompt_tokens)?;

    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();

    let prompt_len = tokens.len();
    let mut logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
    let mut out = Vec::with_capacity(n);
    for pos in (prompt_len..).take(n) {
        let next = argmax(&logits);
        out.push(next as u32);
        if Some(next) == eos {
            break;
        }
        logits = decoder.forward_token(next, pos, &mut caches);
    }
    Ok((out, prompt_len))
}

/// The prompt token ids and the logit vector at the LAST prompt position
/// — i.e. the distribution the model would sample the first generated
/// token from, before anything is sampled.
///
/// Separate from [`greedy_token_ids`] because the question is different:
/// that one asks "do two backends produce the same text", this one hands
/// out the distribution itself so it can be compared against another
/// engine's. The token ids come back too, because any cross-engine
/// comparison must feed the *same* ids to both sides — otherwise it is
/// measuring the two tokenizers.
pub fn prefill_logits(
    path: &Path,
    prompt: &str,
    specials: SpecialTokens,
    prompt_tokens: Option<usize>,
) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    let (file, tokens, _eos, runtime_ctx) =
        tokenize_checkpoint(path, prompt, specials, prompt_tokens)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or_default();
    if GEMMA4_ARCHES.contains(&arch) {
        return prefill_logits_gemma4(path, &tokens);
    }
    if matches!(select_engine_kind(arch), Ok(SelectedEngineKind::Mla)) {
        return prefill_logits_mla(path, &tokens);
    }
    let mut config = ModelConfig::from_gguf(&file).context("reading model config")?;
    config.apply_runtime_context(runtime_ctx);
    // The reference dumper runs llama.cpp's default context, f16 K/V
    // (`tools/llama_logits.c` takes `llama_context_default_params`), so
    // this side keeps the host store llama.cpp would: like for like.
    config.kv_dtype = frink_models::host_kv_dtype_from_env(frink_models::BudgetBackend::Cpu);
    let decoder = Decoder::from_gguf(path, config)?;
    let mut caches: Vec<KvCache> = decoder.config.new_kv_caches();
    let logits = decoder.forward_batch_last(&tokens, 0, &mut caches);
    Ok((tokens.into_iter().map(|t| t as u32).collect(), logits))
}

/// The MLA engine (`deepseek2` / `mistral4` / `plm`) has one body,
/// token by token, so the prefill is a decode loop; the distribution
/// at the last prompt position is the same question. This is what puts
/// a real DeepSeek or PLM file in front of `frink parity`, where the
/// generic-only dispatch used to refuse it as `DedicatedOnly`.
fn prefill_logits_mla(path: &Path, tokens: &[usize]) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    let served = load_mla_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Mla(engine) = served else {
        anyhow::bail!("expected MlaEngine for an MLA checkpoint");
    };
    let mut state = Engine::new_state(&engine);
    let mut logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        logits = Engine::forward_token(&engine, tok, pos, &mut state);
    }
    Ok((tokens.iter().map(|&t| t as u32).collect(), logits))
}

fn prefill_logits_gemma4(path: &Path, tokens: &[usize]) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
    let served = load_gemma4_engine_from_path(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ServedEngine::Gemma4(engine) = served else {
        anyhow::bail!("expected Gemma4Engine for gemma4 checkpoint");
    };
    let mut state = Engine::new_state(engine.as_ref());
    let mut logits = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        logits = Engine::forward_token(engine.as_ref(), tok, pos, &mut state);
    }
    Ok((tokens.iter().map(|&t| t as u32).collect(), logits))
}

/// Tokenize a prompt the same way verify/parity do, without loading weights.
///
/// `specials` is llama.cpp's `parse_special`, and the caller says which
/// because the tools above this differ: a prompt handed to `verify`,
/// `parity`, `layer-divergence` or `quant-sensitivity` is parsed like
/// `llama-completion`'s (`Parse`); the corpus `perplexity` and `imatrix`
/// read is text, and `llama-perplexity` (`common_tokenize(ctx,
/// params.prompt, true)`, default `false`) and `llama-imatrix`
/// (`common.h`: `parse_special = false`) both keep it that way.
fn tokenize_checkpoint(
    path: &Path,
    prompt: &str,
    specials: SpecialTokens,
    prompt_tokens: Option<usize>,
) -> anyhow::Result<(ShardedGguf, Vec<usize>, Option<usize>, usize)> {
    let file = ShardedGguf::open(path)?;
    let tokenizer = match file.metadata_str("tokenizer.ggml.model") {
        Some("gpt2" | "gemma4") => Tok::Bpe(Box::new(GgufBpeTokenizer::from_gguf(&file)?)),
        Some("llama") => Tok::Spm(GgufSpmTokenizer::from_gguf(&file)?),
        Some("t5") => Tok::Unigram(GgufUnigramTokenizer::from_gguf(&file)?),
        Some("plamo2") => Tok::Plamo2(Box::new(GgufPlamo2Tokenizer::from_gguf(&file)?)),
        other => anyhow::bail!("verify does not cover tokenizer {other:?}"),
    };
    let eos = file
        .metadata_u64("tokenizer.ggml.eos_token_id")
        .map(|v| v as usize);
    let bos = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .map(|v| v as usize);

    let mut tokens = tokenizer.encode(prompt, specials);
    if frink_models::tokenizer::should_add_bos_token(&file) {
        if let Some(b) = bos {
            if tokens.first() != Some(&b) {
                tokens.insert(0, b);
            }
        }
    }
    if tokens.is_empty() {
        anyhow::bail!("prompt tokenized to nothing");
    }
    if let Some(want) = prompt_tokens {
        tokens = stretch_prompt(tokens, want, bos)?;
    }
    let runtime_ctx = tokens.len() + 8;
    Ok((file, tokens, eos, runtime_ctx))
}

/// Shared setup: open the GGUF, build the tokenizer the header names,
/// encode the prompt (adding BOS only when the checkpoint says to),
/// stretch it if asked, and build the decoder.
pub(crate) fn load_and_tokenize(
    path: &Path,
    prompt: &str,
    specials: SpecialTokens,
    prompt_tokens: Option<usize>,
) -> anyhow::Result<(Decoder, Vec<usize>, Option<usize>)> {
    let (file, tokens, eos, runtime_ctx) =
        tokenize_checkpoint(path, prompt, specials, prompt_tokens)?;
    let arch = file
        .metadata_str("general.architecture")
        .unwrap_or_default();
    if GEMMA4_ARCHES.contains(&arch) {
        anyhow::bail!(
            "architecture '{arch}' uses Gemma4Engine; use prefill_logits or frink run, not generic Decoder"
        );
    }
    let mut config = ModelConfig::from_gguf(&file).context("reading model config")?;
    config.apply_runtime_context(runtime_ctx);
    let decoder = Decoder::from_gguf(path, config)?;
    Ok((decoder, tokens, eos))
}

/// Repeat `tokens` until it is exactly `want` long, keeping at most one
/// leading BOS. Errors rather than silently shortening when `want` is
/// smaller than the tokenized prompt, so `--prompt-tokens` never reports
/// a length the run did not use.
fn stretch_prompt(
    tokens: Vec<usize>,
    want: usize,
    bos: Option<usize>,
) -> anyhow::Result<Vec<usize>> {
    if want == 0 {
        anyhow::bail!("--prompt-tokens must be at least 1");
    }
    if want < tokens.len() {
        anyhow::bail!(
            "--prompt-tokens {want} is shorter than the tokenized prompt ({}); \
             pass a shorter --prompt instead",
            tokens.len()
        );
    }
    let leading_bos = (bos.is_some() && tokens.first().copied() == bos).then(|| tokens[0]);
    let body = &tokens[leading_bos.iter().count()..];
    if body.is_empty() {
        anyhow::bail!("prompt is BOS only; nothing to repeat");
    }
    let mut out = Vec::with_capacity(want);
    out.extend(leading_bos);
    while out.len() < want {
        let take = (want - out.len()).min(body.len());
        out.extend_from_slice(&body[..take]);
    }
    Ok(out)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::stretch_prompt;

    #[test]
    fn stretch_repeats_the_body_and_keeps_one_bos() {
        // [BOS, a, b, c] stretched to 8 -> BOS once, body cycled.
        let got = stretch_prompt(vec![1, 10, 11, 12], 8, Some(1)).unwrap();
        assert_eq!(got, vec![1, 10, 11, 12, 10, 11, 12, 10]);
        assert_eq!(got.len(), 8);
        assert_eq!(got.iter().filter(|&&t| t == 1).count(), 1);
    }

    #[test]
    fn stretch_without_bos_cycles_the_whole_prompt() {
        assert_eq!(
            stretch_prompt(vec![7, 8], 5, None).unwrap(),
            vec![7, 8, 7, 8, 7]
        );
    }

    #[test]
    fn stretch_is_a_no_op_at_the_current_length() {
        assert_eq!(
            stretch_prompt(vec![1, 4, 5], 3, Some(1)).unwrap(),
            vec![1, 4, 5]
        );
    }

    /// Parity and verify share `load_and_tokenize`; LongRoPE must pick
    /// short factors at small `n_ctx`, not the file's trained length.
    #[test]
    fn parity_context_size_is_tokens_plus_eight() {
        let mut c = frink_models::config::test_dense_fixture();
        c.rope_orig_ctx = Some(4096);
        c.rope_freqs_short = Some(vec![1.0; 48]);
        c.rope_freqs_long = Some((0..48).map(|i| 1.0 + i as f32).collect());
        c.rope_freqs = None;
        // Five prompt tokens + 8, same as llama_logits.c for a 5-token run.
        c.apply_runtime_context(5 + 8);
        assert_eq!(c.rope_freqs.as_ref().unwrap().full[1], 1.0);
    }

    #[test]
    fn stretch_refuses_to_shorten_or_empty_a_prompt() {
        // Silently truncating would report a prompt length the run did
        // not use, which is exactly the vacuous-pass bug this flag fixes.
        assert!(stretch_prompt(vec![1, 4, 5, 6], 2, Some(1)).is_err());
        assert!(stretch_prompt(vec![1, 4], 0, Some(1)).is_err());
        assert!(stretch_prompt(vec![1], 8, Some(1)).is_err());
    }

    #[test]
    fn a_repeated_token_that_equals_bos_is_only_stripped_at_the_front() {
        // Body containing the BOS id mid-prompt must survive: only a
        // leading BOS is special.
        let got = stretch_prompt(vec![1, 9, 1, 9], 6, Some(1)).unwrap();
        assert_eq!(got, vec![1, 9, 1, 9, 9, 1]);
    }
}
