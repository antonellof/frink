//! The one serving path behind llama.cpp's `/completion` and `/infill`.
//!
//! Upstream hands both routes to `handle_completions_impl`
//! (`tools/server/server-context.cpp`): `/infill` assembles its prompt
//! and then IS a completion, with the same options, the same response
//! object and the same stream. So this module is that function once,
//! taking the prompt as a parameter, and the two handlers differ only in
//! how they build it (ADR 0004).

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use super::wire::{final_body, frame, partial_body};
use super::{Budget, CompletionRequest};
use crate::attribution::Attribution;
use crate::decode_task::{self, DecodeHandles};
use crate::generate::GenerationParams;
use crate::prompt_input::Prompt;
use crate::{sse, stats, unsupported_feature, ActiveModel, ApiError, AppState};
use frink_models::tokenizer::SpecialTokens;

/// What the decode prefills, and what the response echoes as `prompt`.
pub(crate) enum CompletionPrompt {
    /// `/completion`'s text, tokenized by the decode.
    Text(String),
    /// `/infill`'s assembled ids, and the text they detokenize to,
    /// which is what upstream echoes for a token prompt
    /// (`server_slot::to_json`, `tokens.detokenize(ctx, true)`).
    Ids { ids: Vec<usize>, echo: String },
}

impl CompletionPrompt {
    fn echo(&self) -> &str {
        match self {
            CompletionPrompt::Text(text) => text,
            CompletionPrompt::Ids { echo, .. } => echo,
        }
    }
}

impl<'a> From<&'a CompletionPrompt> for Prompt<'a> {
    fn from(prompt: &'a CompletionPrompt) -> Self {
        match prompt {
            CompletionPrompt::Text(text) => Prompt::Text(text),
            CompletionPrompt::Ids { ids, .. } => Prompt::Ids(ids),
        }
    }
}

/// The request's bookkeeping, taken when it arrived so the timings and
/// the ring cover the route's own work as well as the decode.
pub(crate) struct Arrival {
    pub(crate) request_id: String,
    pub(crate) started: std::time::Instant,
    pub(crate) attribution: Attribution,
    /// Whichever path the client called: `/completion` and
    /// `/completions` share a handler.
    pub(crate) route: String,
}

impl Arrival {
    pub(crate) fn now(
        matched: Option<&axum::extract::MatchedPath>,
        headers: &axum::http::HeaderMap,
        fallback: &str,
    ) -> Self {
        Arrival {
            request_id: frink_api::next_request_id(),
            started: std::time::Instant::now(),
            attribution: Attribution::from_headers(headers),
            route: matched
                .map(|m| m.as_str().to_string())
                .unwrap_or_else(|| fallback.to_string()),
        }
    }
}

/// Generate `prompt` under `req`'s options and answer in llama.cpp's
/// native shape, buffered or streamed. `req` has been validated.
pub(crate) async fn serve(
    state: Arc<AppState>,
    active: Arc<ActiveModel>,
    handles: DecodeHandles,
    req: CompletionRequest,
    prompt: CompletionPrompt,
    arrival: Arrival,
) -> Result<Response, ApiError> {
    let mut params = GenerationParams {
        cache_salt: None,
        prompt_logprobs: None,
        // Wired per route once a wire renders them; see
        // `docs/plans/per-token-logprobs.md`.
        wants_logprobs: false,
        // `n` is wired in a later step of
        // `docs/plans/done/several-completions-per-request.md`; the field is
        // still refused on the wire by `crate::unimplemented_fields`,
        // so nothing can reach this with anything but 1.
        n: 1,
        // Set by the STREAMING routes, which are the only ones whose
        // delivery order a client can observe (`crate::round_robin`).
        interleave_choices: false,
        // llama.cpp's native wire carries neither field.
        logit_bias: crate::logit_bias::LogitBias::parse(req.logit_bias.as_ref(), "/completion")?,
        keep_special_tokens: false,
        truncate_prompt_tokens: None,
        token_mask: crate::token_mask::TokenMask::default(),
        // This endpoint returns the text verbatim and never splits a
        // reasoning block out of it, so counting one would describe a
        // split that did not happen.
        reasoning: None,
        max_tokens: 0,
        sampling: req
            .sampling_knobs()?
            .resolve(active.sampler_model())
            .map_err(|e| {
                crate::unsupported_feature(&format!("`dry_multiplier` on /completion: {e}"))
            })?,
        seed: req.seed(),
        stop: req.stop.clone().unwrap_or_default(),
        json_object: false,
        grammar: req.constraint()?,
        stop_token_ids: Vec::new(),
        cancel: None,
        ignore_eos: req.ignore_eos.unwrap_or(false),
        // A raw completion has no reasoning format (`reasoning: None`
        // above), so there is no block for a budget to bound.
        reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
        lora: crate::lora::resolve_request(handles.model(), req.lora.as_deref())?,
    };
    params.max_tokens = match req.budget()? {
        Budget::Fixed(n) => n,
        // `-1` means "until the context is full", so it needs a context
        // to be full of. Resolving it against the ceiling is the only
        // reading that is not an invention: the prompt is tokenized
        // with the model's own tokenizer, and what is left of the
        // ceiling is the budget.
        Budget::UntilContextFull => {
            let limit = active
                .ceiling
                .as_ref()
                .and_then(|c| c.limit())
                .ok_or_else(|| {
                    unsupported_feature(
                        "`n_predict: -1` means \"generate until the context is full\", and this \
                         server has no context ceiling for the loaded model to be full of -- it \
                         could not be priced at load (see the startup log). Send an explicit \
                         n_predict, or set FRINK_CB_MAX_CONTEXT so -1 has a bound. Note that \
                         an ABSENT n_predict is -1 too: that is llama.cpp's default, and this \
                         server does not quietly substitute a smaller one",
                    )
                })?;
            let prompt_tokens = match &prompt {
                CompletionPrompt::Text(text) => {
                    handles.model().encode(text, SpecialTokens::Parse).len()
                }
                CompletionPrompt::Ids { ids, .. } => ids.len(),
            };
            limit.saturating_sub(prompt_tokens)
        }
    };

    let model_name = handles.model().name().to_string();
    if req.wants_stream() {
        return stream(state, handles, params, prompt, model_name, arrival).await;
    }

    let echo = prompt.echo().to_string();
    let produced = decode_task::buffered(handles, prompt, params.clone()).await?;
    let usage = produced.usage;
    // Choice 0: this wire has no `n`, and `crate::unimplemented_fields`
    // refuses the field.
    let one = produced
        .choices
        .into_iter()
        .next()
        .expect("a generation produces at least one choice");
    let (finish, content) = (one.finish, one.text);

    state.record_request(stats::Record {
        request_id: &arrival.request_id,
        route: &arrival.route,
        model: Some(model_name.clone()),
        status: 200,
        stream: false,
        duration_ms: arrival.started.elapsed().as_millis() as u64,
        usage: Some(&usage),
        attribution: &arrival.attribution,
    });
    Ok(Json(final_body(
        &content,
        &finish,
        &usage,
        &params,
        &model_name,
        &echo,
    ))
    .into_response())
}

/// The native stream: `data: <partial>` per token, then the same
/// terminal object the buffered response returns with `"stop": true`,
/// and **no `[DONE]`** -- upstream terminates the native stream with
/// nothing at all (`server-context.cpp:4285-4291`, where only the
/// OpenAI dialects get the sentinel). A client waiting for `[DONE]`
/// here would hang, and one that got it would try to parse it as JSON.
async fn stream(
    state: Arc<AppState>,
    handles: DecodeHandles,
    mut params: GenerationParams,
    prompt: CompletionPrompt,
    model_name: String,
    arrival: Arrival,
) -> Result<Response, ApiError> {
    let Arrival {
        request_id,
        started,
        attribution,
        route,
    } = arrival;
    // The same two-tier cancellation the other streams have: the guard
    // rides with the generation task and deregisters however that task
    // ends, panic included.
    let (cancel_token, cancel_guard) = state.cancels.register(&request_id);
    params.cancel = Some(cancel_token.clone());
    let stats_state = Arc::clone(&state);
    let stats_request_id = request_id.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    // A partial with empty content: a concatenating client adds nothing
    // and the transport sees traffic. See `sse::with_keepalive` for why
    // this is a data frame rather than an SSE comment.
    let keepalive = sse::keepalive_event(&partial_body(""));

    tokio::task::spawn_blocking(move || {
        let _cancel_guard = cancel_guard;
        let orphan = sse::orphan_timeout_from_env();
        let send = |event: Event| {
            if sse::send_or_orphan(&tx, Ok(event), orphan).is_err() {
                // The reader is gone or has stopped reading, and this
                // stream keeps no replay buffer, so there is nothing
                // left to generate for.
                cancel_token.cancel();
            }
        };
        let result = handles.run_emit((&prompt).into(), &params, |_choice, chunk| {
            if !chunk.is_empty() {
                send(frame(&partial_body(chunk)));
            }
        });
        match result {
            // Streaming, so exactly one choice.
            Ok(generated) => {
                let usage = generated.usage;
                let one = generated
                    .choices
                    .into_iter()
                    .next()
                    .expect("a generation produces at least one choice");
                let (finish, content) = (one.finish, one.text);
                stats_state.record_request(stats::Record {
                    request_id: &stats_request_id,
                    route: &route,
                    model: Some(model_name.clone()),
                    status: 200,
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: Some(&usage),
                    attribution: &attribution,
                });
                send(frame(&final_body(
                    &content,
                    &finish,
                    &usage,
                    &params,
                    &model_name,
                    prompt.echo(),
                )));
            }
            Err(e) => {
                tracing::warn!("decode error on streamed completion {stats_request_id}: {e}");
                // The socket already carried 200 -- SSE headers precede
                // the first token -- so the failure can only ride IN the
                // stream. Upstream's own shape for that is a frame whose
                // body is `{"error": …}` (`server-context.cpp:4271`).
                let (status, body) = crate::decode_error_response(e);
                stats_state.record_request(stats::Record {
                    request_id: &stats_request_id,
                    route: &route,
                    model: Some(model_name.clone()),
                    status: status.as_u16(),
                    stream: true,
                    duration_ms: started.elapsed().as_millis() as u64,
                    usage: None,
                    attribution: &attribution,
                });
                send(frame(&json!({ "error": body.0["error"] })));
            }
        }
    });

    let stream = sse::with_keepalive(rx, keepalive, sse::KEEPALIVE_INTERVAL);
    Ok((
        // See `chat_completions_stream`: nginx and the proxies that
        // copied it buffer `text/event-stream` by default, which turns
        // a token stream into one silent wait.
        [(
            axum::http::HeaderName::from_static("x-accel-buffering"),
            axum::http::HeaderValue::from_static("no"),
        )],
        Sse::new(stream),
    )
        .into_response())
}
