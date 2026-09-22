//! Extra OpenAI-shaped endpoints: tokenize / detokenize and legacy
//! `/v1/completions`.
//!
//! `/v1/embeddings` used to live here too and now has its own module,
//! [`crate::embeddings`] — this file was 939 lines and the embedding
//! route was about to grow an encoder path.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;

use crate::attribution::Attribution;
use crate::generate::{FinishReason, GenerationParams};
use crate::sampling_knobs::SamplingKnobs;
use crate::{unsupported_feature, ApiError, AppState};
use frink_models::tokenizer::SpecialTokens;

/// What one of the small endpoints knows before it does any work.
///
/// Captured at entry so the ring entry can be written from the same
/// facts however the handler ends -- and so the three of them travel
/// together instead of as three positional arguments that are each
/// easy to pass in the wrong order.
pub(crate) struct Call {
    request_id: String,
    started: std::time::Instant,
    attribution: Attribution,
}

impl Call {
    pub(crate) fn new(headers: &axum::http::HeaderMap) -> Self {
        Call {
            request_id: frink_api::next_request_id(),
            started: std::time::Instant::now(),
            attribution: Attribution::from_headers(headers),
        }
    }

    /// Records a call that reached a response. Spelled out rather than
    /// left as a `&Ok(())` at the call site, which reads like a
    /// mistake.
    fn record_success(
        &self,
        state: &AppState,
        route: &str,
        model: Option<String>,
        usage: Option<&frink_api::Usage>,
    ) {
        self.record(state, route, model, &Ok::<(), ApiError>(()), usage);
    }

    /// Records this call in the `/admin/stats` ring, with the status
    /// the caller actually saw.
    ///
    /// These endpoints used not to be recorded at all, which made the
    /// monitor quietly wrong rather than merely incomplete: an editor
    /// hammering `/v1/embeddings` showed up as an idle server. A
    /// failure is recorded with its own status for the same reason -- a
    /// 400 that leaves no trace is indistinguishable from a request
    /// that was never sent.
    pub(crate) fn record<T>(
        &self,
        state: &AppState,
        route: &str,
        model: Option<String>,
        result: &Result<T, ApiError>,
        usage: Option<&frink_api::Usage>,
    ) {
        let status = match result {
            Ok(_) => 200,
            Err((code, _)) => code.as_u16(),
        };
        state.record_request(crate::stats::Record {
            request_id: &self.request_id,
            route,
            model,
            status,
            stream: false,
            duration_ms: self.started.elapsed().as_millis() as u64,
            usage: result.is_ok().then_some(usage).flatten(),
            attribution: &self.attribution,
        });
    }
}

/// Both dialects of the tokenize request in one struct.
///
/// frink invented `/v1/tokenize` with a `prompt` field; llama.cpp
/// serves `/tokenize` with a `content` field
/// (`tools/server/server-context.cpp:4918-4956`). Two structs would be
/// two handlers, which is how a copied path starts, so the field is
/// accepted under either spelling and resolved once in
/// [`TokenizeRequest::text`].
#[derive(Debug, Deserialize)]
pub(crate) struct TokenizeRequest {
    /// frink's spelling.
    #[serde(default)]
    prompt: Option<String>,
    /// llama.cpp's spelling.
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    model: Option<String>,
    /// llama.cpp: prepend `BOS`. Default `false`, as upstream. Honoured
    /// -- it prepends exactly the id the generation path prepends, so
    /// a caller counting a prompt's tokens gets the count the model
    /// will actually see.
    #[serde(default)]
    add_special: Option<bool>,
    /// llama.cpp: tokenize special-token text as special tokens rather
    /// than as plaintext. Upstream defaults to `true`
    /// (`server-context.cpp`: `json_value(body, "parse_special", true)`)
    /// and so does this server; `false` tokenizes `<|im_start|>` as the
    /// characters it is written with.
    #[serde(default)]
    parse_special: Option<bool>,
    /// llama.cpp: return `{"id", "piece"}` objects instead of bare ids.
    /// Refused by name -- see the refusal message for why.
    #[serde(default)]
    with_pieces: Option<bool>,
}

impl TokenizeRequest {
    /// The text to tokenize, under whichever spelling the client used.
    ///
    /// llama.cpp answers `{"tokens": []}` when `content` is absent.
    /// frink does not: a request that names neither field is far more
    /// likely a typo than a deliberate ask for an empty array, and an
    /// empty array is indistinguishable from tokenizing `""`.
    fn text(&self) -> Result<&str, ApiError> {
        match (&self.prompt, &self.content) {
            (Some(_), Some(_)) => Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message":
                    "give either `prompt` (frink) or `content` (llama.cpp), not both: \
                     they are the same field and this server cannot tell which you meant"}})),
            )),
            (Some(p), None) => Ok(p.as_str()),
            (None, Some(c)) => Ok(c.as_str()),
            (None, None) => Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": {"message":
                    "missing the text to tokenize: send `content` (llama.cpp's spelling) \
                     or `prompt` (frink's). Both are accepted on both /tokenize and \
                     /v1/tokenize"}})),
            )),
        }
    }

    /// Every knob this server does not implement, refused by name
    /// before any work happens.
    fn reject_unsupported(&self) -> Result<(), ApiError> {
        if self.with_pieces == Some(true) {
            return Err(unsupported_feature(
                "`with_pieces: true` is not implemented: frink's tokenizers expose decoded \
                 text, not the raw per-token piece bytes llama.cpp returns, so a \
                 byte-fallback token could not be represented as its `piece` byte array. \
                 Detokenize the ids you want the text for instead",
            ));
        }
        Ok(())
    }

    /// The `parse_special` setting the request asked for, with
    /// llama.cpp's default when it did not say.
    fn specials(&self) -> SpecialTokens {
        match self.parse_special {
            Some(false) => SpecialTokens::AsText,
            Some(true) | None => SpecialTokens::Parse,
        }
    }
}

/// Shared by `/detokenize` and `/v1/detokenize`; the field name is the
/// same in both dialects.
#[derive(Debug, Deserialize)]
pub(crate) struct DetokenizeRequest {
    tokens: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CompletionsRequest {
    /// Accepted as a `Value` rather than a `String` so a token-id prompt
    /// -- `[int]` or `[[int]]`, which OpenAI allows and this server does
    /// not implement -- is REFUSED BY NAME rather than dying as a serde
    /// type error the caller has to guess at.
    prompt: serde_json::Value,
    /// The legacy 16 is right *here*, and only here: a caller asking to
    /// complete a fragment usually wants a fragment back. The chat and
    /// Responses surfaces deliberately keep it out (see
    /// `DEFAULT_CHAT_MAX_TOKENS`).
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    model: Option<String>,
    // Sampler knobs. These four were NOT declared until now, so serde
    // dropped `top_k`, `repetition_penalty`, `presence_penalty` and
    // `frequency_penalty` on this route while the chat route honoured
    // every one of them -- and the `SamplingParams` below hardcoded
    // `top_k: 0` and `repetition_penalty: 1.0` over the top. Two of
    // them (`presence_penalty`, `frequency_penalty`) are OpenAI's own
    // `/v1/completions` fields. Same defect as `logit_bias`, four more
    // times, so they are honoured here rather than refused: the chat
    // route already implements all four.
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    /// llama.cpp's `--min-p`. See `SamplingKnobs::min_p` for why an
    /// absent one is off rather than llama.cpp's CLI default of 0.05.
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    /// llama.cpp's `typ_p`, `top_n_sigma`, `xtc_*` and `dry_*`, in ONE
    /// struct shared with the other two routes that take them. See
    /// `sampling_knobs::ExtraSamplerFields`.
    #[serde(flatten)]
    extra_samplers: crate::sampling_knobs::ExtraSamplerFields,
    /// See `crate::unimplemented_fields`: the same struct the chat and
    /// native routes flatten, so `n` cannot be refused on one wire and
    /// dropped on this one again.
    #[serde(flatten)]
    unimplemented: crate::unimplemented_fields::UnimplementedFields,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Honoured. Silently dropping this is the dangerous one: the caller
    /// believes generation halts at their sentinel and instead gets the
    /// full budget of text past it.
    #[serde(default)]
    stop: Option<StopParam>,
    /// Run past the model's own end-of-generation tokens. See
    /// `ChatCompletionRequest::ignore_eos`.
    #[serde(default)]
    ignore_eos: Option<bool>,
    /// llama.cpp's per-request `lora: [{id, scale}]`; see
    /// `ChatCompletionRequest::lora`.
    #[serde(default)]
    lora: Option<Vec<frink_api::LoraScaleRequest>>,
    // Fields this server does not implement. Deserialized ONLY so they
    // can be refused by name -- serde would otherwise drop each one
    // silently, which is indistinguishable from having honoured it.
    #[serde(default)]
    logprobs: Option<serde_json::Value>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    logit_bias: Option<serde_json::Value>,
    /// llama.cpp's `samplers`: the ORDER the sampler chain runs in.
    /// Decided by `unsupported_sampling::parse_sampler_order`, shared
    /// with `/v1/chat/completions` and `/completion`, so the three
    /// routes cannot disagree about which samplers exist.
    #[serde(default)]
    samplers: Option<serde_json::Value>,
    /// A GBNF grammar every sampled token must keep parseable. The same
    /// field, with the same meaning, as on `/v1/chat/completions`: a
    /// constraint the two routes spelled differently would be a
    /// constraint one of them eventually dropped.
    #[serde(default)]
    grammar: Option<String>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

/// `stop` as OpenAI sends it: one string or a list of them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum StopParam {
    One(String),
    Many(Vec<String>),
}

impl CompletionsRequest {
    /// The prompt text, or a named refusal.
    ///
    /// A token-id prompt is a real OpenAI shape and a real thing this
    /// server cannot serve; saying so beats a deserialization error, and
    /// beats silently treating `[1,2,3]` as the string it is not.
    fn prompt_text(&self) -> Result<&str, ApiError> {
        match &self.prompt {
            serde_json::Value::String(s) => Ok(s),
            serde_json::Value::Array(_) => Err(crate::unsupported_feature(
                "token-id prompts are not implemented; send `prompt` as a string",
            )),
            _ => Err(crate::invalid_request("prompt must be a string", "prompt")),
        }
    }

    /// This request's sampler knobs, resolved by the same function the
    /// chat route uses. See `crate::sampling_knobs`.
    /// Fallible because `samplers` is parsed here; see the chat route's
    /// twin.
    pub(crate) fn sampling_knobs(&self) -> Result<SamplingKnobs, ApiError> {
        let mut knobs = SamplingKnobs {
            temperature: self.temperature,
            top_p: self.top_p,
            min_p: self.min_p,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            // The OpenAI wire has no field for the penalty window; see
            // `SamplingKnobs::penalty_last_n`.
            penalty_last_n: None,
            sampler_order: crate::unsupported_sampling::parse_sampler_order(
                self.samplers.as_ref(),
                "/v1/completions",
            )?,
            ..SamplingKnobs::default()
        };
        self.extra_samplers.apply(&mut knobs);
        Ok(knobs)
    }

    fn stop_sequences(&self) -> Vec<String> {
        match &self.stop {
            Some(StopParam::One(s)) => vec![s.clone()],
            Some(StopParam::Many(v)) => v.clone(),
            None => Vec::new(),
        }
    }

    /// Refuse what this server does not implement, by name.
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        // `logit_bias` decides in `unsupported_sampling`, shared with
        // `/v1/chat/completions`: the two routes disagreed about this
        // field for as long as each held its own copy of the rule.
        crate::unsupported_sampling::refuse_logit_bias(
            self.logit_bias.as_ref(),
            "/v1/completions",
        )?;
        crate::unsupported_sampling::parse_sampler_order(
            self.samplers.as_ref(),
            "/v1/completions",
        )?;
        self.unimplemented.refuse("/v1/completions")?;
        let unsupported = [
            (self.logprobs.is_some(), "logprobs"),
            (self.echo == Some(true), "echo"),
            (self.suffix.is_some(), "suffix"),
        ];
        for (present, name) in unsupported {
            if present {
                return Err(crate::unsupported_feature(&format!(
                    "`{name}` is not implemented on /v1/completions (see docs/API.md)"
                )));
            }
        }
        // Compiled here so an unparseable grammar is a 400 before any
        // prompt is tokenized. `response_format` is not passed: this
        // route refuses every value of it but `text` just below, so
        // there is no `json_schema` for the grammar seam to answer.
        crate::grammar_request::for_request(self.grammar.as_deref(), None)?;
        // `{"type": "text"}` is the default and means nothing to refuse.
        if let Some(fmt) = &self.response_format {
            let kind = fmt.get("type").and_then(|v| v.as_str());
            if kind != Some("text") {
                return Err(crate::unsupported_feature(
                    "only `response_format: {\"type\": \"text\"}` is implemented on \
                     /v1/completions (see docs/API.md)",
                ));
            }
        }
        Ok(())
    }
}

fn default_max_tokens() -> usize {
    16
}

/// The path this request was actually routed on, for the `/admin/stats`
/// ring.
///
/// `/tokenize` and `/v1/tokenize` are one handler, so recording a
/// constant would make every llama.cpp client's traffic show up under
/// the frink spelling and hide the split. `MatchedPath` is the route
/// pattern the router matched, not the raw URI, so it cannot be
/// influenced by the caller.
fn matched_route(matched: &Option<axum::extract::MatchedPath>, fallback: &'static str) -> String {
    matched
        .as_ref()
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| fallback.to_string())
}

pub async fn tokenize(
    State(state): State<Arc<AppState>>,
    matched: Option<axum::extract::MatchedPath>,
    headers: axum::http::HeaderMap,
    Json(req): Json<TokenizeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = tokenize_inner(&state, req);
    // No `usage`, deliberately. Tokenizing runs the tokenizer and not
    // the model, and `prompt_tokens` here feeds `tokens_prompt_total`,
    // which means "tokens this server put through a forward pass".
    // Counting a tokenize call into it would inflate every throughput
    // number derived from it.
    call.record(
        &state,
        &matched_route(&matched, frink_api::routes::V1_TOKENIZE),
        state.active_model_name(),
        &result,
        None,
    );
    result
}

fn tokenize_inner(
    state: &AppState,
    req: TokenizeRequest,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = &req.model;
    req.reject_unsupported()?;
    let text = req.text()?;
    // Tokenizing needs the loaded vocabulary, so this is a 503 like any
    // other generation endpoint when nothing is loaded -- answering
    // with byte-fallback ids would silently be the wrong vocabulary.
    // `require_active`, not `require_model`. Tokenizing is a question
    // about the tokenizer, and an encoder has a real one: routing this
    // through `generative()` made `/v1/tokenize` answer 501 "not a
    // generative model" on an encoder-only server, which is the right
    // refusal for a decode and the wrong one for this (issue #28).
    let active = state.require_active()?;
    let mut tokens = active.encode_any(text, req.specials());
    if req.add_special == Some(true) {
        // The same helper the generation path uses, so `add_special`
        // reports the prompt the model would actually be given --
        // including its no-op behaviour on a checkpoint whose metadata
        // says not to add BOS. An encoder wraps its own specials in
        // `token_ids` already, so there is no BOS to prepend and
        // `bos_id()` is `None` there: the helper is a no-op, which is
        // the honest answer rather than a second special token.
        let bos = active.generative_opt().and_then(|m| m.bos_id());
        frink_models::tokenizer::prepend_bos(&mut tokens, bos);
    }
    let count = tokens.len();
    Ok(Json(serde_json::json!({
        "tokens": tokens,
        // frink's own extra; llama.cpp returns `tokens` alone, and a
        // client of either dialect ignores what it does not read.
        "count": count,
    })))
}

pub async fn detokenize(
    State(state): State<Arc<AppState>>,
    matched: Option<axum::extract::MatchedPath>,
    headers: axum::http::HeaderMap,
    Json(req): Json<DetokenizeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    let result = detokenize_inner(&state, req);
    // Same reasoning as `tokenize`: no forward pass, so no usage.
    call.record(
        &state,
        &matched_route(&matched, frink_api::routes::V1_DETOKENIZE),
        state.active_model_name(),
        &result,
        None,
    );
    result
}

fn detokenize_inner(
    state: &AppState,
    req: DetokenizeRequest,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Same reasoning as `tokenize_inner`: an encoder's vocabulary is a
    // real vocabulary, and asking what a token id means is not asking
    // for a decode.
    let text = state.require_active()?.decode_any(&req.tokens);
    // Both keys, same string: llama.cpp's clients read `content`
    // (`server-context.cpp:4970`), frink's existing ones read `text`,
    // and there is only one answer to disagree about.
    Ok(Json(serde_json::json!({ "text": text, "content": text })))
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<CompletionsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let call = Call::new(&headers);
    crate::cache_admin::check_admission(&state)?;
    req.validate()?;
    let prompt = req.prompt_text()?.to_string();
    let active = state.require_active()?;
    let params = GenerationParams {
        // The prompt is prefilled once and the KV forked per choice
        // (`crate::generate`). Streaming is refused above for `n` > 1,
        // so a streaming request always lands on 1.
        n: req.unimplemented.n.unwrap_or(1).max(1) as usize,
        // This endpoint returns the text verbatim and never splits a
        // reasoning block out of it, so counting one would describe a
        // split that did not happen.
        reasoning: None,
        max_tokens: req.max_tokens,
        sampling: req
            .sampling_knobs()?
            .resolve(active.sampler_model())
            .map_err(|e| {
                crate::unsupported_feature(&format!("`dry_multiplier` on /v1/completions: {e}"))
            })?,
        seed: req.seed.unwrap_or(0),
        stop: req.stop_sequences(),
        json_object: false,
        grammar: crate::grammar_request::for_request(req.grammar.as_deref(), None)?,
        // `/v1/completions` is buffered rather than streamed here, so
        // there is no first chunk on which to state a request id and
        // nothing for a client to name in a cancel.
        stop_token_ids: Vec::new(),
        cancel: None,
        ignore_eos: req.ignore_eos.unwrap_or(false),
        // A raw completion has no reasoning format (`reasoning: None`
        // above), so there is no block for a budget to bound.
        reasoning_budget: crate::reasoning_budget::ReasoningBudget::Unrestricted,
        lora: crate::lora::resolve_request(active.generative()?, req.lora.as_deref())?,
    };
    let (choices, usage) = crate::decode_task::buffered(
        crate::decode_task::DecodeHandles::take(&state, &active)?,
        prompt,
        params,
    )
    .await?;

    // One entry per choice, in order, which is what `n` asked for.
    let rendered: Vec<serde_json::Value> = choices
        .into_iter()
        .enumerate()
        .map(|(index, (finish, text))| {
            let finish_reason = match finish {
                FinishReason::Stop | FinishReason::StopSequence(_) => "stop",
                FinishReason::Length => "length",
                // Unreachable today -- this path passes `cancel: None`
                // -- but written out rather than defaulted so that
                // wiring cancellation into `/v1/completions` later is a
                // compile error here first, instead of a completion
                // silently reported as finished.
                FinishReason::Cancelled => "cancelled",
            };
            serde_json::json!({
                "index": index,
                "text": text,
                "finish_reason": finish_reason,
            })
        })
        .collect();
    let model_name = req.model.unwrap_or_else(|| active.name().to_string());
    call.record_success(
        &state,
        frink_api::routes::V1_COMPLETIONS,
        Some(active.name().to_string()),
        Some(&usage),
    );

    Ok(Json(serde_json::json!({
        "id": call.request_id,
        "object": "text_completion",
        "model": model_name,
        "choices": rendered,
        "usage": usage,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> CompletionsRequest {
        serde_json::from_value(value).expect("request")
    }

    fn tokenize_request(value: serde_json::Value) -> TokenizeRequest {
        serde_json::from_value(value).expect("request")
    }

    /// The llama.cpp knob this server does not implement. It
    /// deserializes, so serde would happily have dropped it; the point
    /// of declaring it is that it is refused BY NAME.
    #[test]
    fn the_tokenize_knob_frink_lacks_is_refused_by_name() {
        let field = "with_pieces";
        let (status, body) =
            tokenize_request(serde_json::json!({"content": "hi", "with_pieces": true}))
                .reject_unsupported()
                .expect_err("this is not implemented");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{field}");
        assert!(
            body.0["error"]["message"].as_str().unwrap().contains(field),
            "the refusal must name {field}: {body:?}"
        );
    }

    /// `n` on the one OpenAI route whose response can carry several
    /// answers. The acceptance property is `prompt_tokens`: three
    /// completions of one prompt are billed for ONE prompt, because
    /// the prefill was shared and the KV forked
    /// (`docs/plans/several-completions-per-request.md`).
    #[tokio::test]
    async fn several_completions_share_one_prefill() {
        let app = crate::tests::test_app();
        let body = |n: u32| {
            serde_json::json!({
                "model": "x",
                "prompt": "hello",
                "max_tokens": 4,
                "n": n,
                "temperature": 1.0
            })
        };

        let (status, one) =
            crate::tests::post_json_uri(&app, frink_api::routes::V1_COMPLETIONS, body(1)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{one}");
        assert_eq!(one["choices"].as_array().unwrap().len(), 1);

        let (status, three) =
            crate::tests::post_json_uri(&app, frink_api::routes::V1_COMPLETIONS, body(3)).await;
        assert_eq!(status, axum::http::StatusCode::OK, "{three}");
        let choices = three["choices"].as_array().expect("an array");
        assert_eq!(choices.len(), 3, "{three}");
        for (i, c) in choices.iter().enumerate() {
            assert_eq!(c["index"], i, "choices carry their own index");
            assert!(c["text"].is_string(), "{c}");
            assert!(c["finish_reason"].is_string(), "{c}");
        }
        // THE property: one prompt, billed once.
        assert_eq!(
            three["usage"]["prompt_tokens"], one["usage"]["prompt_tokens"],
            "n = 3 billed the prompt more than once, so the prefill was not shared"
        );
        assert_eq!(
            three["usage"]["completion_tokens"].as_u64().unwrap(),
            one["usage"]["completion_tokens"].as_u64().unwrap() * 3,
            "completion tokens are the sum over choices"
        );
    }

    /// The wire with no `choices` array still refuses it BY NAME
    /// rather than quietly answering with one.
    #[tokio::test]
    async fn a_wire_without_a_choices_array_still_refuses_n() {
        let app = crate::tests::test_app();
        for (uri, body) in [(
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 2, "n": 3}),
        )] {
            let (status, answer) = crate::tests::post_json_uri(&app, uri, body).await;
            assert_eq!(
                status,
                axum::http::StatusCode::NOT_IMPLEMENTED,
                "{uri} served `n`: {answer}"
            );
            assert!(
                answer["error"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains('n')),
                "{uri}: {answer}"
            );
        }
    }

    /// The values frink *does* honour must not be refused: upstream's
    /// defaults are `parse_special: true` and `with_pieces: false`, so
    /// a llama.cpp client that sends them explicitly must still work.
    #[test]
    fn the_upstream_defaults_are_accepted_rather_than_refused() {
        tokenize_request(
            serde_json::json!({"content": "hi", "parse_special": true, "with_pieces": false}),
        )
        .reject_unsupported()
        .expect("these are the values this server implements");
    }

    /// `parse_special` maps onto the tokenizer's setting, and an absent
    /// field is llama.cpp's default of `true`. `false` used to be
    /// refused by name because frink parsed specials unconditionally.
    #[test]
    fn parse_special_selects_the_tokenizer_setting_and_defaults_to_true() {
        assert_eq!(
            tokenize_request(serde_json::json!({"content": "hi"})).specials(),
            SpecialTokens::Parse
        );
        assert_eq!(
            tokenize_request(serde_json::json!({"content": "hi", "parse_special": true}))
                .specials(),
            SpecialTokens::Parse
        );
        let off = tokenize_request(serde_json::json!({"content": "hi", "parse_special": false}));
        off.reject_unsupported().expect("false is implemented now");
        assert_eq!(off.specials(), SpecialTokens::AsText);
    }

    /// One field under two spellings. Absent means absent -- llama.cpp
    /// answers an empty array, which is indistinguishable from
    /// tokenizing `""`, so frink says what is missing instead.
    #[test]
    fn the_text_is_read_from_either_dialects_field() {
        assert_eq!(
            tokenize_request(serde_json::json!({"prompt": "a"}))
                .text()
                .expect("frink's spelling"),
            "a"
        );
        assert_eq!(
            tokenize_request(serde_json::json!({"content": "b"}))
                .text()
                .expect("llama.cpp's spelling"),
            "b"
        );

        let (status, body) = tokenize_request(serde_json::json!({}))
            .text()
            .expect_err("neither field is present");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = body.0["error"]["message"].as_str().unwrap().to_string();
        assert!(
            message.contains("content") && message.contains("prompt"),
            "{message}"
        );

        // Both at once is a client bug, and guessing which one it meant
        // would tokenize text the caller never asked about.
        let (status, _) = tokenize_request(serde_json::json!({"prompt": "a", "content": "b"}))
            .text()
            .expect_err("ambiguous");
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    /// The dangerous silent drop. A caller who sets `stop` believes
    /// generation halts at their sentinel; dropping it hands them the
    /// full budget of text past it instead.
    #[test]
    fn a_completion_stop_string_reaches_the_generation_params() {
        let one = request(serde_json::json!({"prompt": "hi", "stop": "END"}));
        assert_eq!(one.stop_sequences(), vec!["END".to_string()]);

        let many = request(serde_json::json!({"prompt": "hi", "stop": ["A", "B"]}));
        assert_eq!(
            many.stop_sequences(),
            vec!["A".to_string(), "B".to_string()]
        );

        let none = request(serde_json::json!({"prompt": "hi"}));
        assert!(none.stop_sequences().is_empty());
    }

    /// The same field, on the other route. A constraint honoured by
    /// `/v1/chat/completions` and dropped by `/v1/completions` is the
    /// `logit_bias` bug again, with a different field name.
    #[test]
    fn a_grammar_on_the_completions_wire_is_compiled_rather_than_dropped() {
        let req = request(serde_json::json!({"prompt": "hi", "grammar": "root ::= \"a\"+"}));
        req.validate().expect("a valid grammar is a valid request");
        assert!(
            crate::grammar_request::for_request(req.grammar.as_deref(), None)
                .expect("it compiled during validate too")
                .is_some()
        );

        let bad = request(serde_json::json!({"prompt": "hi", "grammar": "root ::= \"a"}));
        let (status, _) = bad.validate().expect_err("this does not parse");
        assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    }

    /// The knobs this route accepted-and-dropped. `top_k`,
    /// `repetition_penalty`, `presence_penalty` and `frequency_penalty`
    /// were undeclared, so serde discarded them and the request was
    /// served with `top_k: 0` and `repetition_penalty: 1.0` hardcoded
    /// over the top -- while `/v1/chat/completions` honoured all four.
    /// `min_p` was never on either wire.
    ///
    /// Every value here is deliberately different from the default it
    /// replaces, so a knob that stopped being read would show up as the
    /// default rather than as the asked-for number.
    #[test]
    fn every_sampler_knob_reaches_the_sampler_from_the_completions_wire() {
        let resolved = request(serde_json::json!({
            "prompt": "hi",
            "temperature": 0.5,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.25,
            "frequency_penalty": 0.75,
            "samplers": ["penalties", "top_p", "top_k", "min_p", "temperature"],
        }))
        .sampling_knobs()
        .expect("knobs")
        .resolve(crate::sampling_knobs::SamplerModel::absent())
        .expect("no dry");

        assert_eq!(resolved.temperature, 0.5);
        assert_eq!(resolved.top_p, 0.9);
        assert_eq!(resolved.min_p, 0.05);
        assert_eq!(resolved.top_k, 40);
        assert_eq!(resolved.repetition_penalty, 1.1);
        assert_eq!(resolved.presence_penalty, 0.25);
        assert_eq!(resolved.frequency_penalty, 0.75);
        assert_eq!(
            resolved.sampler_order.to_string(),
            "penalties;top_p;top_k;min_p;temperature"
        );
    }

    /// And the two routes resolve one body to the same sampler, which is
    /// the property that kept breaking: each route owning its own
    /// mapping is how `/v1/completions` came to hardcode two of these.
    #[test]
    fn both_routes_resolve_the_same_knobs_to_the_same_sampler() {
        let knobs = serde_json::json!({
            "temperature": 0.5,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 40,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.25,
            "frequency_penalty": 0.75,
            "samplers": ["penalties", "top_p", "top_k", "min_p", "temperature"],
        });

        let mut completion_body = knobs.clone();
        completion_body["prompt"] = serde_json::json!("hi");
        let completion = request(completion_body)
            .sampling_knobs()
            .expect("knobs")
            .resolve(crate::sampling_knobs::SamplerModel::absent())
            .expect("no dry");

        let mut chat_body = knobs;
        chat_body["model"] = serde_json::json!("m");
        chat_body["messages"] = serde_json::json!([{"role": "user", "content": "hi"}]);
        let chat: crate::ChatCompletionRequest =
            serde_json::from_value(chat_body).expect("chat request");
        let chat = chat
            .sampling_params(crate::sampling_knobs::SamplerModel::absent())
            .expect("knobs");

        assert_eq!(completion.temperature, chat.temperature);
        assert_eq!(completion.top_p, chat.top_p);
        assert_eq!(completion.min_p, chat.min_p);
        assert_eq!(completion.top_k, chat.top_k);
        assert_eq!(completion.repetition_penalty, chat.repetition_penalty);
        assert_eq!(completion.penalty_last_n, chat.penalty_last_n);
        assert_eq!(completion.sampler_order, chat.sampler_order);
        assert_eq!(completion.presence_penalty, chat.presence_penalty);
        assert_eq!(completion.frequency_penalty, chat.frequency_penalty);
    }

    /// Each of these is a real OpenAI field this server does not
    /// implement. Serde drops an undeclared field silently, which a
    /// caller cannot tell apart from having had it honoured -- so each
    /// is declared purely in order to be refused BY NAME.
    #[test]
    fn every_unimplemented_completion_field_is_refused_by_name() {
        for (field, value) in [
            ("logprobs", serde_json::json!(5)),
            ("echo", serde_json::json!(true)),
            ("suffix", serde_json::json!("tail")),
            ("logit_bias", serde_json::json!({"5": -100})),
        ] {
            let mut body = serde_json::json!({"prompt": "hi"});
            body[field] = value;
            let (status, payload) = request(body)
                .validate()
                .expect_err(&format!("`{field}` must be refused"));
            assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
            assert!(
                payload["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains(field),
                "the refusal must name `{field}`"
            );
        }
    }

    /// `{"type": "text"}` is the default and means nothing to refuse;
    /// anything else is a promise this endpoint cannot keep.
    #[test]
    fn only_the_text_response_format_is_accepted() {
        let text =
            request(serde_json::json!({"prompt": "hi", "response_format": {"type": "text"}}));
        assert!(text.validate().is_ok());

        let json = request(
            serde_json::json!({"prompt": "hi", "response_format": {"type": "json_object"}}),
        );
        assert!(json.validate().is_err());
    }

    /// A token-id prompt is a real OpenAI shape. Saying so beats a
    /// deserialization error the caller has to guess at, and beats
    /// treating `[1,2,3]` as a string it is not.
    #[test]
    fn a_token_id_prompt_is_refused_by_name_rather_than_mis_parsed() {
        let ids = request(serde_json::json!({"prompt": [1, 2, 3]}));
        let (status, payload) = ids.prompt_text().expect_err("refused");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(payload["error"]["message"]
            .as_str()
            .unwrap()
            .contains("token-id"));

        let text = request(serde_json::json!({"prompt": "hi"}));
        assert_eq!(text.prompt_text().expect("a string prompt"), "hi");
    }
}
