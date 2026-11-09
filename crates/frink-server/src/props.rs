//! llama-server's `GET /props` and `POST /props`: the server's global
//! properties, which llama.cpp's own web UI reads first.
//!
//! Transcribed from `get_res_props` in `tools/server/server-context.cpp`
//! at the pin. Field names, nesting and types are upstream's. The rule
//! for the values is the one `frink bench`'s llama-bench printers follow
//! (`crates/frink-cli/src/bench_model/output.rs`): every field frink can
//! state truthfully is here, and a field that would need a value frink
//! does not have is LEFT OUT rather than defaulted, because a default
//! reads as a fact about this server. `docs/API.md` lists each omitted
//! field and why.
//!
//! `default_generation_settings.params` is what `/completion` resolves
//! an empty request to, read from the code that resolves it
//! ([`SamplingKnobs::resolve`], [`DryRequest::default`],
//! [`SamplerOrder`]) rather than restated, so the two cannot drift.
//! upstream's list is `task_params::to_json(true)`, the short form.
//!
//! **Asleep.** upstream answers from a snapshot taken when it went to
//! sleep, with `is_sleeping: true`, because the model is gone. frink's
//! `POST /sleep` drops the model too, so it takes the same snapshot
//! ([`snapshot`]) and this route answers from it until `/wake_up`.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::{json, Map, Value};

use frink_models::dry::DryRequest;

use crate::llama_error::{llama_error, LlamaErrorKind};
use crate::sampling_knobs::SamplingKnobs;
use crate::{ActiveModel, ApiError, AppState};

/// `build_info`: this server's own build, as upstream reports its own.
const BUILD_INFO: &str = concat!("frink-server ", env!("CARGO_PKG_VERSION"));

/// `GET /props`.
pub(crate) async fn get_props(State(state): State<Arc<AppState>>) -> Result<Json<Value>, ApiError> {
    if let Some(props) = state.slept_props() {
        return Ok(Json(props));
    }
    let active = state.require_active()?;
    Ok(Json(props(&active, false)))
}

/// `POST /props`, which upstream answers only under `--props`.
///
/// frink-server has no `--props`: upstream's handler changes nothing
/// even when the flag is given (`// update any props here`, then
/// `success: true`), and a 200 for a change that did not happen is the
/// one answer this server does not give. So it answers as upstream does
/// without the flag, saying why there is no flag.
pub(crate) async fn post_props() -> ApiError {
    llama_error(
        LlamaErrorKind::NotSupported,
        "This server does not support changing global properties: frink-server has no global \
         property a request can change, so it has no `--props` to enable that with",
    )
}

/// The properties of the model `POST /sleep` is about to drop, with
/// `is_sleeping: true`, for [`get_props`] to answer with while asleep.
pub(crate) fn snapshot(active: &ActiveModel) -> Value {
    props(active, true)
}

/// upstream's `get_res_props`, as far as frink can state it.
fn props(active: &ActiveModel, is_sleeping: bool) -> Value {
    let n_ctx = active.ceiling.as_ref().and_then(|c| c.limit());
    let mut out = Map::new();
    if let Some(model) = active.generative_opt() {
        let mut settings = Map::new();
        settings.insert("params".into(), default_params(active));
        if let Some(n_ctx) = n_ctx {
            settings.insert("n_ctx".into(), json!(n_ctx));
        }
        out.insert(
            "default_generation_settings".into(),
            Value::Object(settings),
        );
        if let Some(n) = total_slots(active) {
            out.insert("total_slots".into(), json!(n));
        }
        out.insert("model_alias".into(), json!(active.name()));
        if let Some(name) = model
            .gguf_facts()
            .and_then(|f| f.file_type)
            .and_then(frink_gguf::ftype::llama_ftype_name)
        {
            out.insert("model_ftype".into(), json!(name));
        }
        insert_path(&mut out, active);
        insert_server_flags(&mut out);
        let template = model.chat_template();
        if let Some(source) = template.source() {
            out.insert("chat_template".into(), json!(source));
        }
        if let Some(bos) = template.bos_token() {
            out.insert("bos_token".into(), json!(bos));
        }
        if let Some(eos) = template.eos_token() {
            out.insert("eos_token".into(), json!(eos));
        }
    } else {
        // An encoder: no sampler, no template, no special tokens to
        // report, and the rest stands.
        out.insert("model_alias".into(), json!(active.name()));
        insert_path(&mut out, active);
        insert_server_flags(&mut out);
    }
    out.insert("build_info".into(), json!(BUILD_INFO));
    out.insert("is_sleeping".into(), json!(is_sleeping));
    out.insert("cors_proxy_enabled".into(), json!(false));
    Value::Object(out)
}

fn insert_path(out: &mut Map<String, Value>, active: &ActiveModel) {
    if let Some(path) = &active.checkpoint_path {
        out.insert("model_path".into(), json!(path.display().to_string()));
    }
}

/// The fields that describe the server rather than the model.
fn insert_server_flags(out: &mut Map<String, Value>) {
    // No image, video or audio input reaches any model here.
    out.insert(
        "modalities".into(),
        json!({"vision": false, "video": false, "audio": false}),
    );
    out.insert(
        "endpoint_slots".into(),
        json!(crate::slots::listing::endpoint_enabled()),
    );
    // No `--props`; see `post_props`.
    out.insert("endpoint_props".into(), json!(false));
    // `/metrics` is always served (`--metrics` changes nothing).
    out.insert("endpoint_metrics".into(), json!(true));
    // This server serves no web UI: `GET /` is a 404.
    out.insert("ui".into(), json!(false));
}

/// upstream's `n_parallel`: the seats a continuous batcher was given
/// with `-np`. A deployment without that cap has no slot count.
pub(crate) fn total_slots(active: &ActiveModel) -> Option<usize> {
    active
        .batcher
        .as_ref()
        .map(|b| b.stats().max_seqs)
        .filter(|&n| n > 0)
}

/// `task_params::to_json(true)` for an empty `/completion` request.
fn default_params(active: &ActiveModel) -> Value {
    // An empty request resolves through the same function a real one
    // does. DRY is off in it, which is why the resolution cannot fail;
    // the wire defaults of DRY's own knobs are `DryRequest`'s.
    let Ok(s) = SamplingKnobs::default().resolve(active.sampler_model()) else {
        return Value::Object(Map::new());
    };
    let dry = DryRequest::default();
    let samplers: Vec<&str> = s
        .sampler_order
        .steps()
        .iter()
        .map(|step| step.name().as_str())
        .collect();
    let mut params = json!({
        "temperature": s.temperature,
        "top_k": s.top_k,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "top_n_sigma": s.top_n_sigma,
        "xtc_probability": s.xtc_probability,
        "xtc_threshold": s.xtc_threshold,
        "typical_p": s.typical_p,
        "repeat_last_n": s.penalty_last_n,
        "repeat_penalty": s.repetition_penalty,
        "presence_penalty": s.presence_penalty,
        "frequency_penalty": s.frequency_penalty,
        "dry_multiplier": dry.multiplier,
        "dry_base": dry.base,
        "dry_allowed_length": dry.allowed_length,
        "dry_penalty_last_n": dry.penalty_last_n,
        // `/completion`'s own default: until the context is full.
        "max_tokens": crate::completion::N_PREDICT_UNBOUNDED,
        "n_predict": crate::completion::N_PREDICT_UNBOUNDED,
        "ignore_eos": false,
        "stream": false,
        "samplers": samplers,
        "lora": [],
    });
    // The samplers and options frink does not have, at the value
    // `/completion` serves them at: the OFF value, which is a true
    // statement about this server (no dynamic temperature, no mirostat
    // ever runs here). Read from `/completion`'s own table, so the
    // default reported is one it accepts.
    let fields = params.as_object_mut().expect("a JSON object literal");
    if let Some(v) = crate::completion::inert_number("dynatemp_range") {
        fields.insert("dynatemp_range".into(), json!(v));
    }
    for int in ["mirostat", "n_keep", "n_probs", "min_keep"] {
        if let Some(v) = crate::completion::inert_number(int) {
            fields.insert(int.into(), json!(v as i64));
        }
    }
    for switch in ["timings_per_token", "post_sampling_probs"] {
        if crate::completion::inert_when_false(switch) {
            fields.insert(switch.into(), json!(false));
        }
    }
    params
}

#[cfg(test)]
mod tests;
