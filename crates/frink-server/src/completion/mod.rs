//! llama.cpp's native `POST /completion` (and `/completions`).
//!
//! **This is a wire, not an engine.** Sampling, grammars, stop strings,
//! the context ceiling and the decode loop are all the ones
//! `/v1/completions` already uses: [`crate::sampling_knobs`],
//! [`crate::grammar_request`], [`crate::stop`] and
//! [`crate::decode_task`]. What lives here is the translation between
//! llama.cpp's field names and frink's, and llama.cpp's response and
//! SSE shapes -- which are genuinely different from OpenAI's and are
//! why an alias would not have done.
//!
//! # Why the endpoint exists at all
//!
//! `/v1/completions` covers OpenAI clients. It covers nothing else.
//! llama.cpp's own web UI, `llama.vim` and a long tail of wrappers
//! speak this endpoint instead, and against frink they got a 404. The
//! shapes are not interchangeable:
//!
//! | | llama.cpp `/completion` | OpenAI `/v1/completions` |
//! |---|---|---|
//! | budget field | `n_predict`, `-1` = until the context is full | `max_tokens`, default 16 |
//! | repetition | `repeat_penalty`, `repeat_last_n` | `repetition_penalty` |
//! | response | flat object: `content`, `stop`, `stop_type`, `timings` | `choices[].text`, `finish_reason` |
//! | stream frame | `data: {"content":…,"stop":false}` | `data: {"choices":[{"text":…}]}` |
//! | stream end | the final object with `"stop": true`, **no `[DONE]`** | `data: [DONE]` |
//!
//! Transcribed from `tools/server/README.md` and
//! `tools/server/server-task.cpp:368-390` (final) and `:1077-1099`
//! (partial), not inferred from the OpenAI shape. The response side of
//! that lives in [`wire`]; this file is the request side and the two
//! handlers.
//!
//! # The rule every option here is held to
//!
//! A field this server cannot honour is refused **by name**, never
//! dropped. But llama.cpp's request carries two dozen sampler options
//! that are *inert at their defaults*, and refusing a client for
//! sending `mirostat: 0` -- which asks for nothing -- would be a false
//! refusal. So [`UNSUPPORTED`] pairs each option with the value at
//! which it does nothing upstream: at that value it is served, at any
//! other it is a 501 naming the field and what frink is missing.
//!
//! Options that only *parameterise* a switched-off sampler
//! (`mirostat_tau`, `dry_base`, `xtc_threshold`, `dynatemp_exponent`,
//! …) are deliberately absent from that table: they do nothing while
//! their switch is off, and their switch is in it.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::attribution::Attribution;
use crate::decode_task::{self, DecodeHandles};
use crate::generate::GenerationParams;
use crate::sampling_knobs::SamplingKnobs;
use crate::{sse, stats, unsupported_feature, ApiError, AppState};
use frink_models::tokenizer::SpecialTokens;

mod wire;

use wire::{final_body, frame, partial_body};

/// llama.cpp's `n_predict` default and its spelling of "no limit"
/// (`tools/server/README.md`). Kept as a named constant because an
/// absent `n_predict` means exactly this, and reinterpreting it as
/// frink's own default would be the silent-reinterpretation the task
/// this endpoint closes is about.
const N_PREDICT_UNBOUNDED: i64 = -1;

/// One llama.cpp option frink does not implement.
struct Unsupported {
    /// The wire field.
    field: &'static str,
    /// The value at which this option does nothing upstream. A request
    /// that sends it is asking for nothing and is served.
    inert: Inert,
    /// What frink is missing, said in the refusal.
    missing: &'static str,
}

/// The shape of "this option is switched off".
enum Inert {
    Num(f64),
    False,
    /// Absent, or present and empty. Used for list-valued options.
    EmptyList,
}

impl Unsupported {
    fn is_inert(&self, value: &Value) -> bool {
        match &self.inert {
            Inert::Num(off) => value
                .as_f64()
                .is_some_and(|v| (v - off).abs() < f64::EPSILON),
            Inert::False => value.as_bool() == Some(false),
            Inert::EmptyList => match value {
                Value::Null => true,
                Value::Array(items) => items.is_empty(),
                Value::Object(fields) => fields.is_empty(),
                _ => false,
            },
        }
    }
}

const fn off_at(field: &'static str, off: f64, missing: &'static str) -> Unsupported {
    Unsupported {
        field,
        inert: Inert::Num(off),
        missing,
    }
}

const fn off_when_false(field: &'static str, missing: &'static str) -> Unsupported {
    Unsupported {
        field,
        inert: Inert::False,
        missing,
    }
}

const fn off_when_empty(field: &'static str, missing: &'static str) -> Unsupported {
    Unsupported {
        field,
        inert: Inert::EmptyList,
        missing,
    }
}

/// Every llama.cpp `/completion` option this server does not implement,
/// with the value at which it asks for nothing.
///
/// Read this table as the honest support matrix for the endpoint: a
/// field that is neither here nor in [`CompletionRequest`] is one
/// llama.cpp does not define either, and is ignored exactly as upstream
/// ignores it.
const UNSUPPORTED: &[Unsupported] = &[
    off_at(
        "dynatemp_range",
        0.0,
        "dynamic temperature sampling is not implemented",
    ),
    off_at("mirostat", 0.0, "mirostat sampling is not implemented"),
    off_at(
        "n_probs",
        0.0,
        "per-token logprobs are not implemented; the sampler does not publish the \
         candidate distribution",
    ),
    off_when_false(
        "post_sampling_probs",
        "per-token probabilities are not implemented; the sampler does not publish the \
         candidate distribution",
    ),
    off_at(
        "min_keep",
        0.0,
        "the min-keep floor is not implemented; this engine's truncation filters have no \
         minimum-candidate guarantee",
    ),
    off_when_false(
        "return_tokens",
        "raw generated token ids are not returned; the decode loop hands this layer text, \
         not ids",
    ),
    off_at(
        "n_indent",
        0.0,
        "indentation-aware stopping is not implemented",
    ),
    off_at(
        "n_keep",
        0.0,
        "context-shift retention is not implemented: frink refuses a request that does not \
         fit its context rather than discarding tokens from the middle of it, so there is \
         nothing for n_keep to protect",
    ),
    off_at(
        "n_cmpl",
        1.0,
        "more than one completion per prompt is not implemented; send the request again",
    ),
    off_at(
        "n_cache_reuse",
        0.0,
        "cache reuse via KV shifting is not implemented; frink's radix prefix cache reuses \
         a shared PREFIX only",
    ),
    off_at(
        "t_max_predict_ms",
        0.0,
        "a wall-clock limit on generation is not implemented; bound the work with n_predict",
    ),
    off_at(
        "id_slot",
        -1.0,
        "this server has no slots to pin a request to; concurrency is per request, not per \
         slot",
    ),
    off_when_empty(
        "response_fields",
        "response field projection is not implemented; the whole object is returned",
    ),
    off_when_false(
        "return_progress",
        "prompt-processing progress frames are not implemented",
    ),
    off_when_false(
        "timings_per_token",
        "per-token timings are not implemented; `timings` is reported once, on the final \
         frame",
    ),
    off_when_empty(
        "sse_ping_interval",
        "the keepalive interval is fixed at 15s for every stream this server serves and is \
         not settable per request",
    ),
];

/// llama.cpp's native completion request.
///
/// Only the options frink **implements** are named as fields; every
/// option it refuses is matched against [`UNSUPPORTED`] out of
/// `extra`, so adding a refusal is a table row rather than a field plus
/// a branch. Fields llama.cpp itself does not define fall into `extra`
/// and are ignored, which is what upstream does with them too.
#[derive(Debug, Deserialize)]
pub(crate) struct CompletionRequest {
    /// String prompts only. `[int]`, mixed arrays, multiple prompts and
    /// the `{"prompt_string": …, "multimodal_data": […]}` object are all
    /// real llama.cpp shapes and are refused by name below rather than
    /// dying as a serde type error.
    #[serde(default)]
    prompt: Value,
    /// llama.cpp's `max_tokens`. `-1` (and absent) means "until the
    /// context is full"; `0` means "evaluate the prompt and generate
    /// nothing". See [`CompletionRequest::budget`].
    #[serde(default)]
    n_predict: Option<i64>,
    #[serde(default)]
    stream: Option<bool>,
    /// llama.cpp's per-request `lora: [{id, scale}]`; see
    /// `ChatCompletionRequest::lora`. Typed rather than in the
    /// `UNSUPPORTED` table now that adapters are served.
    #[serde(default)]
    lora: Option<Vec<frink_api::LoraScaleRequest>>,
    /// llama.cpp takes an array here and nothing else.
    #[serde(default)]
    stop: Option<Vec<String>>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    /// llama.cpp's `typ_p`, `top_n_sigma`, `xtc_*` and `dry_*`, in ONE
    /// struct shared with the other two routes that take them. See
    /// `sampling_knobs::ExtraSamplerFields`.
    /// See `crate::unimplemented_fields`: shared with both OpenAI
    /// generation routes so no wire can drop what another refuses.
    #[serde(flatten)]
    unimplemented: crate::unimplemented_fields::UnimplementedFields,
    #[serde(flatten)]
    extra_samplers: crate::sampling_knobs::ExtraSamplerFields,
    /// llama.cpp's spelling of `repetition_penalty`.
    #[serde(default)]
    repeat_penalty: Option<f32>,
    /// The window the penalties look back over. `-1` is upstream's
    /// "the whole context", refused below because frink needs a
    /// concrete count.
    #[serde(default)]
    repeat_last_n: Option<i64>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    /// `-1` is upstream's "pick one at random".
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    ignore_eos: Option<bool>,
    /// GBNF, the same field and the same syntax `/v1/completions` takes.
    #[serde(default)]
    grammar: Option<String>,
    /// upstream's bare JSON Schema, compiled to a grammar through the
    /// one site that decides a schema everywhere else. Not a
    /// `response_format`: llama.cpp's request has no such field, and
    /// this one carries the schema directly.
    #[serde(default)]
    json_schema: Option<Value>,
    /// Upstream's default is `true`, and `true` is a PERMISSION to reuse
    /// KV rather than a requirement, so it is always honourable.
    /// `false` is a requirement not to, which frink can only promise
    /// when no prefix cache is configured.
    #[serde(default)]
    cache_prompt: Option<bool>,
    /// Refused through `unsupported_sampling`, shared with both OpenAI
    /// routes: three routes disagreeing about one field is the bug this
    /// server keeps re-fixing.
    #[serde(default)]
    logit_bias: Option<Value>,
    /// The ORDER the sampler chain runs in, upstream's own field.
    /// Decided through `unsupported_sampling`, shared with both OpenAI
    /// routes: it used to be a blanket refusal in [`UNSUPPORTED`], and
    /// is now honoured for the samplers frink has and refused BY NAME
    /// for the ones it does not.
    #[serde(default)]
    samplers: Option<Value>,
    #[serde(flatten)]
    extra: Map<String, Value>,
}

/// How many tokens to generate, and the reason if that cannot be
/// decided.
#[derive(Debug)]
enum Budget {
    Fixed(usize),
    /// `n_predict: -1` with a context ceiling to stop at. The concrete
    /// number needs the prompt's token count, so it is resolved after
    /// tokenizing.
    UntilContextFull,
}

impl CompletionRequest {
    /// The prompt text, or a refusal naming the shape that was sent.
    fn prompt_text(&self) -> Result<&str, ApiError> {
        match &self.prompt {
            Value::String(s) => Ok(s),
            Value::Array(_) => Err(unsupported_feature(
                "`prompt` must be a string on /completion. Token-id prompts, mixed \
                 token/string arrays and multiple prompts in one request are llama.cpp \
                 shapes this server does not implement; the response would have to be an \
                 array, which no part of this engine produces",
            )),
            Value::Object(fields) if fields.contains_key("multimodal_data") => {
                Err(unsupported_feature(
                    "`prompt.multimodal_data` is not implemented: this server has no \
                     multimodal projector, so image or audio input cannot reach the model",
                ))
            }
            Value::Object(fields) => match fields.get("prompt_string") {
                Some(Value::String(s)) => Ok(s),
                _ => Err(crate::invalid_request(
                    "a `prompt` object must carry a string `prompt_string`",
                    "prompt",
                )),
            },
            Value::Null => Err(crate::invalid_request(
                "missing `prompt`: /completion needs the text to continue",
                "prompt",
            )),
            _ => Err(crate::invalid_request(
                "`prompt` must be a string",
                "prompt",
            )),
        }
    }

    /// This request's sampler knobs, in frink's names.
    ///
    /// The mapping is the whole job of this function: `repeat_penalty`
    /// and `repeat_last_n` are llama.cpp's spellings, everything else
    /// happens to agree. Resolution -- what an absent knob means -- is
    /// `SamplingKnobs::resolve`'s, shared with both OpenAI routes.
    pub(crate) fn sampling_knobs(&self) -> Result<SamplingKnobs, ApiError> {
        let penalty_last_n = match self.repeat_last_n {
            None => None,
            Some(n) if n >= 0 => Some(n as usize),
            Some(_) => {
                return Err(unsupported_feature(
                    "`repeat_last_n: -1` (llama.cpp's \"the whole context\") is not \
                     implemented: this server's penalty window is a fixed count, and it has \
                     no context length to expand -1 into on a deployment with no derived \
                     ceiling. Send a concrete window, or 0 to disable the penalties",
                ))
            }
        };
        let mut knobs = SamplingKnobs {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            penalty_last_n,
            sampler_order: crate::unsupported_sampling::parse_sampler_order(
                self.samplers.as_ref(),
                frink_api::routes::COMPLETION,
            )?,
            ..SamplingKnobs::default()
        };
        self.extra_samplers.apply(&mut knobs);
        Ok(knobs)
    }

    /// `n_predict`, read as llama.cpp defines it.
    fn budget(&self) -> Result<Budget, ApiError> {
        match self.n_predict.unwrap_or(N_PREDICT_UNBOUNDED) {
            N_PREDICT_UNBOUNDED => Ok(Budget::UntilContextFull),
            n if n >= 0 => Ok(Budget::Fixed(n as usize)),
            other => Err(crate::invalid_request(
                &format!(
                    "`n_predict` must be -1 (until the context is full) or a non-negative \
                     count; got {other}"
                ),
                "n_predict",
            )),
        }
    }

    /// The seed, with `-1` meaning "choose one".
    ///
    /// A random seed is genuinely random rather than a fixed stand-in:
    /// a client that asked not to pin the sampler and got the same
    /// answer every time would have been given the opposite of what it
    /// asked for.
    fn seed(&self) -> u64 {
        match self.seed {
            None | Some(-1) => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
                .unwrap_or(0),
            Some(s) => s as u64,
        }
    }

    fn wants_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Refuse everything this server does not implement, by name,
    /// before any prompt is tokenized.
    pub(crate) fn validate(&self, has_prefix_cache: bool) -> Result<(), ApiError> {
        crate::unsupported_sampling::refuse_logit_bias(
            self.logit_bias.as_ref(),
            frink_api::routes::COMPLETION,
        )?;
        // Parsed here as well as in `sampling_knobs` -- the same
        // function both times -- so a chain naming a sampler this
        // engine lacks is refused before any prompt is tokenized.
        crate::unsupported_sampling::parse_sampler_order(
            self.samplers.as_ref(),
            frink_api::routes::COMPLETION,
        )?;
        self.unimplemented.refuse(frink_api::routes::COMPLETION)?;
        for option in UNSUPPORTED {
            let sent = self.extra.get(option.field);
            let asked_for_something = match sent {
                None => false,
                Some(value) => !option.is_inert(value),
            };
            if asked_for_something {
                return Err(unsupported_feature(&format!(
                    "`{}` is not implemented on /completion: {}",
                    option.field, option.missing
                )));
            }
        }
        // `cache_prompt: true` is upstream's default and is a permission
        // to reuse KV, which is satisfiable however this server is
        // configured. `false` is a REQUIREMENT not to reuse, and a
        // configured prefix cache will reuse a shared prefix anyway, so
        // that combination has to be refused rather than ignored: a
        // caller disables it precisely to get a deterministic,
        // uncontaminated evaluation.
        if self.cache_prompt == Some(false) && has_prefix_cache {
            return Err(unsupported_feature(
                "`cache_prompt: false` cannot be honoured while a prefix cache is configured \
                 (FRINK_PREFIX_CACHE_ENTRIES): this server's radix cache reuses a shared \
                 prompt prefix for every request and has no per-request opt-out. Unset that \
                 variable to serve requests that require a cold prompt",
            ));
        }
        // Compiled here so an unparseable grammar, or a schema with a
        // keyword that has none, is a 400 before any work happens.
        self.constraint()?;
        Ok(())
    }

    /// The grammar this request runs under, from either field that can
    /// state one.
    ///
    /// One function, called by both [`Self::validate`] and the handler,
    /// because the two of them disagreeing about which field wins is
    /// exactly the shape that lets a request be validated against one
    /// constraint and served under another.
    ///
    /// `grammar` and `json_schema` together are refused rather than
    /// ranked: upstream silently prefers `grammar` and drops the schema,
    /// which is a 200 whose answer need not match the schema the caller
    /// sent.
    fn constraint(&self) -> Result<Option<Arc<frink_models::grammar::Grammar>>, ApiError> {
        match (&self.grammar, &self.json_schema) {
            (Some(g), Some(_)) if !g.trim().is_empty() => Err(crate::invalid_request(
                "a \"grammar\" and a \"json_schema\" are two different constraints on the same \
                 generation; send one",
                "json_schema",
            )),
            (_, Some(schema)) => {
                crate::grammar_request::from_schema(schema, "json_schema").map(Some)
            }
            (grammar, None) => crate::grammar_request::for_request(grammar.as_deref(), None),
        }
    }
}
pub(crate) async fn completion(
    State(state): State<Arc<AppState>>,
    matched: Option<axum::extract::MatchedPath>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    crate::cache_admin::check_admission(&state)?;
    let request_id = frink_api::next_request_id();
    let started = std::time::Instant::now();
    let attribution = Attribution::from_headers(&headers);
    // llama.cpp mounts this handler at both `/completion` and
    // `/completions`; the ring records whichever the client called.
    let route = matched
        .as_ref()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| frink_api::routes::COMPLETION.to_string());

    // Pinned once: this request decodes against exactly this checkpoint
    // even if `/admin/models/load` swaps another in halfway through.
    let active = state.require_active()?;
    let handles = DecodeHandles::take(&state, &active)?;
    req.validate(handles.has_prefix_cache())?;

    let prompt = req.prompt_text()?.to_string();
    let mut params = GenerationParams {
        cache_salt: None,
        prompt_logprobs: None,
        // Wired per route once a wire renders them; see
        // `docs/plans/per-token-logprobs.md`.
        wants_logprobs: false,
        // `n` is wired in a later step of
        // `docs/plans/several-completions-per-request.md`; the field is
        // still refused on the wire by `crate::unimplemented_fields`,
        // so nothing can reach this with anything but 1.
        n: 1,
        // Set by the STREAMING routes, which are the only ones whose
        // delivery order a client can observe (`crate::round_robin`).
        interleave_choices: false,
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
            let prompt_tokens = handles.model().encode(&prompt, SpecialTokens::Parse).len();
            limit.saturating_sub(prompt_tokens)
        }
    };

    let model_name = handles.model().name().to_string();
    if req.wants_stream() {
        return stream(
            state,
            handles,
            params,
            prompt,
            model_name,
            route,
            request_id,
            started,
            attribution,
        )
        .await;
    }

    let produced = decode_task::buffered(handles, prompt.clone(), params.clone()).await?;
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
        request_id: &request_id,
        route: &route,
        model: Some(model_name.clone()),
        status: 200,
        stream: false,
        duration_ms: started.elapsed().as_millis() as u64,
        usage: Some(&usage),
        attribution: &attribution,
    });
    Ok(Json(final_body(
        &content,
        &finish,
        &usage,
        &params,
        &model_name,
        &prompt,
    ))
    .into_response())
}

/// The native stream: `data: <partial>` per token, then the same
/// terminal object the buffered response returns with `"stop": true`,
/// and **no `[DONE]`** -- upstream terminates the native stream with
/// nothing at all (`server-context.cpp:4285-4291`, where only the
/// OpenAI dialects get the sentinel). A client waiting for `[DONE]`
/// here would hang, and one that got it would try to parse it as JSON.
#[allow(clippy::too_many_arguments)] // one request's context, threaded once
async fn stream(
    state: Arc<AppState>,
    handles: DecodeHandles,
    mut params: GenerationParams,
    prompt: String,
    model_name: String,
    route: String,
    request_id: String,
    started: std::time::Instant,
    attribution: Attribution,
) -> Result<Response, ApiError> {
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
        let result = handles.run_emit(&prompt, &params, |_choice, chunk| {
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
                    &prompt,
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use frink_models::sampling::SamplingParams;

    fn request(value: Value) -> CompletionRequest {
        serde_json::from_value(value).expect("request")
    }

    /// The mapping is the whole endpoint. A knob read from the wrong
    /// field name is a knob silently dropped.
    #[test]
    fn llama_cpps_sampler_spellings_reach_frinks_knobs() {
        let knobs = request(json!({
            "prompt": "hi",
            "temperature": 0.7,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 40,
            "repeat_penalty": 1.15,
            "repeat_last_n": 128,
            "presence_penalty": 0.25,
            "frequency_penalty": 0.5,
        }))
        .sampling_knobs()
        .expect("all supported")
        .resolve(crate::sampling_knobs::SamplerModel::absent())
        .expect("no dry");
        assert_eq!(knobs.temperature, 0.7);
        assert_eq!(knobs.top_p, 0.9);
        assert_eq!(knobs.min_p, 0.05);
        assert_eq!(knobs.top_k, 40);
        assert_eq!(knobs.repetition_penalty, 1.15);
        assert_eq!(knobs.penalty_last_n, 128);
        assert_eq!(knobs.presence_penalty, 0.25);
        assert_eq!(knobs.frequency_penalty, 0.5);
    }

    /// An empty request must resolve to exactly what the OpenAI routes
    /// resolve an empty request to: three wires, one set of defaults.
    #[test]
    fn an_empty_request_resolves_to_the_same_defaults_the_openai_routes_use() {
        let mine = request(json!({"prompt": "hi"}))
            .sampling_knobs()
            .expect("nothing to refuse")
            .resolve(crate::sampling_knobs::SamplerModel::absent())
            .expect("no dry");
        let shared = SamplingKnobs::default()
            .resolve(crate::sampling_knobs::SamplerModel::absent())
            .expect("no dry");
        assert_eq!(mine.temperature, shared.temperature);
        assert_eq!(mine.top_p, shared.top_p);
        assert_eq!(mine.min_p, shared.min_p);
        assert_eq!(mine.top_k, shared.top_k);
        assert_eq!(mine.repetition_penalty, shared.repetition_penalty);
        assert_eq!(mine.penalty_last_n, shared.penalty_last_n);
        assert_eq!(
            shared.penalty_last_n,
            SamplingParams::default().penalty_last_n
        );
    }

    /// `n_predict` is llama.cpp's `max_tokens` and `-1` is its
    /// infinity. Reading an absent one as frink's own default would be
    /// the silent reinterpretation this endpoint exists to avoid.
    #[test]
    fn n_predict_keeps_llama_cpps_meaning_including_its_default() {
        assert!(matches!(
            request(json!({"prompt": "hi"})).budget().unwrap(),
            Budget::UntilContextFull
        ));
        assert!(matches!(
            request(json!({"prompt": "hi", "n_predict": -1}))
                .budget()
                .unwrap(),
            Budget::UntilContextFull
        ));
        assert!(matches!(
            request(json!({"prompt": "hi", "n_predict": 0}))
                .budget()
                .unwrap(),
            Budget::Fixed(0)
        ));
        assert!(matches!(
            request(json!({"prompt": "hi", "n_predict": 128}))
                .budget()
                .unwrap(),
            Budget::Fixed(128)
        ));
        // Anything else is a client bug, not a second spelling of
        // infinity.
        let (status, _) = request(json!({"prompt": "hi", "n_predict": -7}))
            .budget()
            .expect_err("-7 means nothing");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// Every option in the table is inert at its documented default, so
    /// a stock llama.cpp client -- which sends most of them explicitly
    /// -- must be served rather than refused.
    #[test]
    fn a_stock_client_sending_every_option_at_its_default_is_served() {
        let stock = request(json!({
            "prompt": "hi",
            "dynatemp_range": 0.0,
            "dynatemp_exponent": 1.0,
            "typical_p": 1.0,
            "xtc_probability": 0.0,
            "xtc_threshold": 0.1,
            "mirostat": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "dry_multiplier": 0.0,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
            "dry_penalty_last_n": -1,
            "dry_sequence_breakers": ["\n", ":", "\"", "*"],
            "samplers": [],
            "n_probs": 0,
            "post_sampling_probs": false,
            "min_keep": 0,
            "return_tokens": false,
            "n_indent": 0,
            "n_keep": 0,
            "n_cmpl": 1,
            "n_cache_reuse": 0,
            "t_max_predict_ms": 0,
            "id_slot": -1,
            "response_fields": [],
            "return_progress": false,
            "timings_per_token": false,
            "cache_prompt": true,
        }));
        stock
            .validate(false)
            .expect("every one of these asks for nothing");
    }

    /// Upstream's bare `json_schema` used to be routed into the 501 that
    /// said the schema-to-grammar converter was missing. It is compiled
    /// now, through the same module `response_format` goes through, and
    /// the evidence is the machine rejecting a document the schema does.
    #[test]
    fn a_bare_json_schema_becomes_the_requests_grammar() {
        let req = request(json!({
            "prompt": "hi",
            "json_schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": false,
            },
        }));
        req.validate(false).expect("this schema converts");
        let grammar = req
            .constraint()
            .expect("and compiles")
            .expect("into a grammar");
        let mut good = (*grammar).clone();
        good.accept_token(0, br#"{"ok": true}"#)
            .expect("a schema-valid document");
        assert!(good.allows_eog(), "and it completes the parse");
        let mut bad = (*grammar).clone();
        assert!(
            bad.accept_token(0, br#"{"ok": "yes""#).is_err(),
            "a property's own type must be enforced"
        );
    }

    /// A schema with no grammar is a 400 naming the keyword, at the same
    /// seam an unparseable `grammar` is refused at.
    #[test]
    fn a_json_schema_with_no_grammar_is_a_400_naming_the_keyword() {
        let (status, body) = request(json!({
            "prompt": "hi",
            "json_schema": {"type": "object", "patternProperties": {"^a": {}}},
        }))
        .validate(false)
        .expect_err("patternProperties has no grammar");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.0["error"]["message"]
                .as_str()
                .unwrap()
                .contains("patternProperties"),
            "the refusal must name the keyword: {body:?}"
        );
    }

    /// Two constraints on one generation. Upstream silently prefers
    /// `grammar` and drops the schema, which is a 200 whose answer need
    /// not match the schema the caller sent.
    #[test]
    fn a_grammar_and_a_json_schema_together_are_refused() {
        let (status, body) = request(json!({
            "prompt": "hi",
            "grammar": "root ::= \"a\"",
            "json_schema": {"type": "boolean"},
        }))
        .validate(false)
        .expect_err("two constraints, one generation");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.0["error"]["param"], "json_schema");

        // Each alone is still served, so the refusal is about the pair.
        assert!(
            request(json!({"prompt": "hi", "grammar": "root ::= \"a\""}))
                .constraint()
                .unwrap()
                .is_some()
        );
        assert!(
            request(json!({"prompt": "hi", "json_schema": {"type": "boolean"}}))
                .constraint()
                .unwrap()
                .is_some()
        );
    }

    /// And each one refused by name the moment it asks for something.
    #[test]
    fn every_unsupported_option_is_refused_by_its_own_name() {
        let asking: &[(&str, Value)] = &[
            ("dynatemp_range", json!(0.5)),
            ("mirostat", json!(2)),
            ("n_probs", json!(5)),
            ("post_sampling_probs", json!(true)),
            ("min_keep", json!(1)),
            ("return_tokens", json!(true)),
            ("n_indent", json!(4)),
            ("n_keep", json!(32)),
            ("n_cmpl", json!(4)),
            ("n_cache_reuse", json!(256)),
            ("t_max_predict_ms", json!(5000)),
            ("id_slot", json!(3)),
            ("response_fields", json!(["content"])),
            ("return_progress", json!(true)),
            ("timings_per_token", json!(true)),
            ("sse_ping_interval", json!(5)),
        ];
        for (field, value) in asking {
            let mut body = json!({"prompt": "hi"});
            body[field] = value.clone();
            let (status, message) = request(body)
                .validate(false)
                .expect_err("{field} asks for something this server lacks");
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{field}");
            assert!(
                message.0["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(field),
                "the refusal for {field} must name it: {message:?}"
            );
        }
        // The table and the test cover the same set, so an option added
        // to one without the other is a failure rather than a silent
        // hole.
        assert_eq!(asking.len(), UNSUPPORTED.len());
        for option in UNSUPPORTED {
            assert!(
                asking.iter().any(|(field, _)| *field == option.field),
                "{} is in the table with no test",
                option.field
            );
        }
    }

    /// A field llama.cpp does not define either is ignored, exactly as
    /// upstream ignores it -- refusing it would break clients over
    /// something neither server has an opinion about.
    #[test]
    fn a_field_neither_server_defines_is_ignored() {
        request(json!({"prompt": "hi", "some_client_extension": 7}))
            .validate(false)
            .expect("nothing to refuse");
    }

    /// `cache_prompt: false` is a requirement, not a preference. It can
    /// be met only where nothing would have been reused anyway.
    #[test]
    fn cache_prompt_false_is_refused_only_where_it_cannot_be_kept() {
        request(json!({"prompt": "hi", "cache_prompt": false}))
            .validate(false)
            .expect("no prefix cache: nothing is reused, so the promise holds");
        let (status, _) = request(json!({"prompt": "hi", "cache_prompt": false}))
            .validate(true)
            .expect_err("a configured prefix cache reuses regardless");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        // The default is a permission and is always servable.
        request(json!({"prompt": "hi", "cache_prompt": true}))
            .validate(true)
            .expect("true is upstream's default");
    }

    /// The prompt shapes llama.cpp accepts and frink does not, each
    /// refused for its own reason rather than as one serde type error.
    #[test]
    fn the_prompt_shapes_this_server_cannot_serve_are_named() {
        assert_eq!(
            request(json!({"prompt": "hello"})).prompt_text().unwrap(),
            "hello"
        );
        assert_eq!(
            request(json!({"prompt": {"prompt_string": "hello"}}))
                .prompt_text()
                .unwrap(),
            "hello"
        );
        for body in [
            json!({"prompt": [12, 34, 56]}),
            json!({"prompt": ["one", "two"]}),
            json!({"prompt": {"prompt_string": "hi", "multimodal_data": ["AAA"]}}),
        ] {
            let (status, _) = request(body.clone())
                .prompt_text()
                .expect_err("not implemented: {body}");
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
        }
        let (status, _) = request(json!({}))
            .prompt_text()
            .expect_err("no prompt at all");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// `repeat_last_n: -1` is upstream's "the whole context". Mapping
    /// it to the default 64 would silently give a caller a window
    /// orders of magnitude smaller than the one it asked for.
    #[test]
    fn the_context_wide_penalty_window_is_refused_rather_than_shrunk() {
        let (status, message) = request(json!({"prompt": "hi", "repeat_last_n": -1}))
            .sampling_knobs()
            .expect_err("no context length to expand -1 into");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(message.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("repeat_last_n"));
        // 0 is a real value: penalties off.
        assert_eq!(
            request(json!({"prompt": "hi", "repeat_last_n": 0}))
                .sampling_knobs()
                .unwrap()
                .resolve(crate::sampling_knobs::SamplerModel::absent())
                .expect("no dry")
                .penalty_last_n,
            0
        );
    }

    /// `seed: -1` is upstream's "choose one", and a fixed stand-in
    /// would hand back the same answer every time to a caller that
    /// asked for the opposite.
    #[test]
    fn an_explicit_seed_is_kept_and_minus_one_is_not_a_constant() {
        assert_eq!(request(json!({"prompt": "hi", "seed": 42})).seed(), 42);
        let a = request(json!({"prompt": "hi", "seed": -1})).seed();
        let b = request(json!({"prompt": "hi"})).seed();
        // Two draws from a nanosecond clock; equality would mean the
        // "random" seed is a constant.
        assert!(a != b || a != 0, "a random seed must vary: {a} {b}");
    }
}
