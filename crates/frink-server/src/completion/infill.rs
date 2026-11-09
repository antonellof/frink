//! llama.cpp's `POST /infill`: fill in the middle between a prefix and
//! a suffix, with other files as context.
//!
//! Transcribed from `tools/server/server-context.cpp` (`post_infill`)
//! and `tools/server/server-common.cpp` (`format_prompt_infill`) at the
//! pin. The handler checks the model's FIM tokens, validates the body
//! with upstream's messages, assembles the prompt as upstream does, and
//! then serves it as a `/completion` through the same path
//! ([`super::serve`]): same options, same response object, same stream.
//!
//! # Where this answers differently, on purpose
//!
//! - **The three body checks that upstream does not return from.** A
//!   `prompt` that is not a string, a missing `input_prefix` and a
//!   missing `input_suffix` each set an error upstream and then carry
//!   on: the first is served with an empty prompt, the other two end in
//!   a JSON exception. Here each is the 400 its message describes.
//! - **A dedicated engine** (Kimi, MLA, Gemma-4, GLM-5.2) is refused by
//!   name: the FIM tokens are read for the generic GGUF decoder only.
//! - **A token id in `input_prefix` or `input_suffix`** must be a
//!   non-negative integer; upstream converts any JSON number.
//! - **The extra context needs a context size.** Upstream budgets it
//!   against the slot's `n_ctx`; a deployment with no derived ceiling has
//!   none, and a request whose prompt would carry extra context is then
//!   refused by name rather than budgeted against a guess.

use std::sync::Arc;

use axum::extract::State;
use axum::response::Response;
use axum::Json;
use serde_json::Value;

use super::serve::{serve, Arrival, CompletionPrompt};
use super::CompletionRequest;
use crate::decode_task::DecodeHandles;
use crate::llama_error::{llama_error, LlamaErrorKind};
use crate::{ApiError, AppState, Model};
use frink_models::tokenizer::SpecialTokens;

/// What `--spm-infill` lowers to: the Suffix/Prefix/Middle order.
pub(crate) const SPM_INFILL_ENV: &str = "FRINK_SPM_INFILL";

/// llama-server's `n_predict` with no `-n`: `-1`, no limit
/// (`common/common.h`). `format_prompt_infill` reads it to leave room
/// for the answer, and frink-server takes no `-n`.
const SERVER_N_PREDICT: i64 = -1;

/// The FIM ids a prompt is assembled from. The three upstream requires
/// are plain ids here, so assembly cannot meet a missing one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FimIds {
    pre: usize,
    suf: usize,
    mid: usize,
    rep: Option<usize>,
    sep: Option<usize>,
}

impl FimIds {
    /// The model's FIM ids, or upstream's refusal naming every one that
    /// is missing, in its words and its order.
    fn of(model: &Model) -> Result<Self, ApiError> {
        let Some(facts) = model.gguf_facts() else {
            return Err(llama_error(
                LlamaErrorKind::NotSupported,
                "Infill is implemented for the generic GGUF decoder; this server is serving a \
                 dedicated engine, whose fill-in-the-middle tokens are not read",
            ));
        };
        let fim = facts.fim;
        let mut err = String::new();
        if fim.pre.is_none() {
            err += "prefix token is missing. ";
        }
        if fim.suf.is_none() {
            err += "suffix token is missing. ";
        }
        if fim.mid.is_none() {
            err += "middle token is missing. ";
        }
        match (fim.pre, fim.suf, fim.mid) {
            (Some(pre), Some(suf), Some(mid)) => Ok(FimIds {
                pre: pre as usize,
                suf: suf as usize,
                mid: mid as usize,
                rep: fim.rep.map(|id| id as usize),
                sep: fim.sep.map(|id| id as usize),
            }),
            _ => Err(llama_error(
                LlamaErrorKind::NotSupported,
                format!("Infill is not supported by this model: {err}"),
            )),
        }
    }
}

/// One `input_extra` chunk: another file the model sees first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ExtraChunk {
    text: String,
    /// upstream's default when a chunk names none.
    filename: String,
}

/// The infill fields of a body, validated with upstream's messages.
struct InfillFields<'a> {
    prompt: &'a str,
    input_prefix: &'a Value,
    input_suffix: &'a Value,
    input_extra: Vec<ExtraChunk>,
}

fn invalid(message: &str) -> ApiError {
    llama_error(LlamaErrorKind::InvalidRequest, message)
}

impl<'a> InfillFields<'a> {
    fn parse(body: &'a Value) -> Result<Self, ApiError> {
        let prompt = match body.get("prompt") {
            None | Some(Value::Null) => "",
            Some(Value::String(s)) => s,
            Some(_) => return Err(invalid("\"prompt\" must be a string")),
        };
        let input_prefix = body
            .get("input_prefix")
            .ok_or_else(|| invalid("\"input_prefix\" is required"))?;
        let input_suffix = body
            .get("input_suffix")
            .ok_or_else(|| invalid("\"input_suffix\" is required"))?;
        let chunks =
            match body.get("input_extra") {
                None => &[][..],
                Some(Value::Array(chunks)) => chunks.as_slice(),
                Some(_) => return Err(invalid(
                    "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
                )),
            };
        let mut input_extra = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let Some(Value::String(text)) = chunk.get("text") else {
                return Err(invalid(
                    "extra_context chunk must contain a \"text\" field with a string value",
                ));
            };
            let filename = match chunk.get("filename") {
                None => "tmp".to_string(),
                Some(Value::String(name)) => name.clone(),
                Some(_) => {
                    return Err(invalid(
                        "extra_context chunk's \"filename\" field must be a string",
                    ))
                }
            };
            input_extra.push(ExtraChunk {
                text: text.clone(),
                filename,
            });
        }
        Ok(InfillFields {
            prompt,
            input_prefix,
            input_suffix,
            input_extra,
        })
    }
}

/// upstream's `tokenize_mixed(vocab, value, false, false)`: a string, or
/// an array of strings and token ids, every string WITHOUT
/// `parse_special`.
fn tokenize_mixed(
    encode: &dyn Fn(&str) -> Vec<usize>,
    value: &Value,
    field: &str,
) -> Result<Vec<usize>, ApiError> {
    let refusal = || {
        invalid(&format!(
            "\"{field}\" must be a string, or an array of strings and non-negative token ids"
        ))
    };
    match value {
        Value::String(text) => Ok(encode(text)),
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => out.extend(encode(text)),
                    Value::Number(n) => out.push(n.as_u64().ok_or_else(refusal)? as usize),
                    _ => return Err(refusal()),
                }
            }
            Ok(out)
        }
        _ => Err(refusal()),
    }
}

/// The numbers `format_prompt_infill` sizes the prompt with.
#[derive(Clone, Copy, Debug)]
pub(super) struct InfillBudget {
    /// llama.cpp's `-b`, default 2048.
    pub(super) n_batch: i64,
    /// The server's `-n`.
    pub(super) n_predict: i64,
    /// The per-request context, when this deployment has one.
    pub(super) n_ctx: Option<i64>,
    /// `--spm-infill`.
    pub(super) spm: bool,
}

/// Extra context was given and there is no context size to budget it
/// against.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct NoContextSize;

/// upstream's `format_prompt_infill`, line for line.
///
/// `encode` is `common_tokenize(vocab, text, false, false)`: no BOS and
/// no `parse_special`. `prefix` and `suffix` are already tokenized
/// ([`tokenize_mixed`]) and `prompt` with `parse_special`; `bos` is the
/// BOS this vocabulary adds, if it adds one. The layout, with the
/// repo-level pattern of <https://arxiv.org/pdf/2409.12186>:
///
/// ```text
/// [FIM_REP]myproject\n [FIM_SEP]file\n text ... [FIM_SEP]filename\n
/// [BOS] [FIM_PRE] prefix prompt [FIM_SUF] suffix [FIM_MID]
/// ```
///
/// with prefix and suffix swapped under `--spm-infill`. The prompt sits
/// at the end of the prefix, which is where upstream's code puts it
/// whatever its comment says.
#[allow(clippy::too_many_arguments)] // upstream's own parameter list
pub(super) fn format_infill(
    encode: &dyn Fn(&str) -> Vec<usize>,
    fim: FimIds,
    bos: Option<usize>,
    mut prefix: Vec<usize>,
    mut suffix: Vec<usize>,
    extra: &[ExtraChunk],
    budget: InfillBudget,
    prompt: &[usize],
) -> Result<Vec<usize>, NoContextSize> {
    let mut extra_tokens: Vec<usize> = Vec::new();
    if let Some(rep) = fim.rep {
        extra_tokens.push(rep);
        extra_tokens.extend(encode("myproject\n"));
    }
    for chunk in extra {
        match fim.sep {
            Some(sep) => {
                extra_tokens.push(sep);
                extra_tokens.extend(encode(&format!("{}\n", chunk.filename)));
            }
            // upstream writes this separator out byte by byte "to avoid
            // confusing the AI"; it is this string.
            None => extra_tokens.extend(encode("\n\n--- snippet ---\n\n")),
        }
        extra_tokens.extend(encode(&chunk.text));
    }
    if let Some(sep) = fim.sep {
        extra_tokens.push(sep);
        extra_tokens.extend(encode("filename\n"));
    }

    // Prefix to suffix 3:1, both inside one batch.
    let quarter = budget.n_batch / 4;
    let n_prefix_take = (prefix.len() as i64).min(3 * quarter).max(0) as usize;
    let n_suffix_take = (suffix.len() as i64)
        .min((quarter - (2 + prompt.len() as i64)).max(0))
        .max(0) as usize;
    let n_extra_take = if extra_tokens.is_empty() {
        0
    } else {
        let n_ctx = budget.n_ctx.ok_or(NoContextSize)?;
        (n_ctx - budget.n_batch - 2 * budget.n_predict)
            .max(0)
            .min(extra_tokens.len() as i64) as usize
    };

    prefix.drain(..prefix.len() - n_prefix_take);
    suffix.truncate(n_suffix_take);

    prefix.insert(0, fim.pre);
    prefix.extend_from_slice(prompt);
    suffix.insert(0, fim.suf);

    let (mut inp, end) = if budget.spm {
        (suffix, prefix)
    } else {
        (prefix, suffix)
    };
    if let Some(bos) = bos {
        inp.insert(0, bos);
    }
    let mut out = extra_tokens[extra_tokens.len() - n_extra_take..].to_vec();
    out.append(&mut inp);
    out.extend(end);
    out.push(fim.mid);
    Ok(out)
}

/// The budget this deployment assembles with.
fn deployment_budget(n_ctx: Option<usize>) -> InfillBudget {
    InfillBudget {
        n_batch: crate::prefill_batch::logical_batch() as i64,
        n_predict: SERVER_N_PREDICT,
        n_ctx: n_ctx.map(|n| n as i64),
        spm: std::env::var(SPM_INFILL_ENV).is_ok_and(|v| v == "1"),
    }
}

pub(crate) async fn infill(
    State(state): State<Arc<AppState>>,
    matched: Option<axum::extract::MatchedPath>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    crate::cache_admin::check_admission(&state)?;
    let arrival = Arrival::now(matched.as_ref(), &headers, frink_api::routes::INFILL);
    let active = state.require_active()?;
    let handles = DecodeHandles::take(&state, &active)?;
    let model = handles.model();

    // upstream's order: the model first, then the body.
    let fim = FimIds::of(model)?;
    let fields = InfillFields::parse(&body)?;
    let req: CompletionRequest = serde_json::from_value(body.clone()).map_err(|e| {
        invalid(&format!(
            "the request body is not a completion request: {e}"
        ))
    })?;
    req.validate(handles.has_prefix_cache())?;

    let as_text = |text: &str| model.encode(text, SpecialTokens::AsText);
    let prefix = tokenize_mixed(&as_text, fields.input_prefix, "input_prefix")?;
    let suffix = tokenize_mixed(&as_text, fields.input_suffix, "input_suffix")?;
    let prompt = model.encode(fields.prompt, SpecialTokens::Parse);
    let n_ctx = active.ceiling.as_ref().and_then(|c| c.limit());
    let ids = format_infill(
        &as_text,
        fim,
        model.bos_id(),
        prefix,
        suffix,
        &fields.input_extra,
        deployment_budget(n_ctx),
        &prompt,
    )
    .map_err(|NoContextSize| {
        llama_error(
            LlamaErrorKind::NotSupported,
            "this infill prompt carries repo-level context (input_extra, or the model's FIM \
             repo and file-separator tokens), which llama.cpp budgets against the context \
             size, and this server has no context ceiling for the loaded model: it could not \
             be priced at load (see the startup log). Set -c / FRINK_CB_MAX_CONTEXT",
        )
    })?;
    let echo = model.decode(&ids);
    serve(
        state,
        active,
        handles,
        req,
        CompletionPrompt::Ids { ids, echo },
        arrival,
    )
    .await
}

#[cfg(test)]
mod tests;
