use std::time::Duration;

use axum::http::StatusCode;
use serde_json::json;

use super::*;
use crate::model::GgufFacts;
use crate::response_cache::ResponseCache;
use crate::tests::{
    get_json_uri, post_json_uri, test_app_with_state, test_byte_model_with_facts, test_state_at,
};

fn app(path: Option<&str>) -> axum::Router {
    let model = test_byte_model_with_facts(GgufFacts {
        // Q4_K - Medium, by llama.cpp's number.
        file_type: Some(15),
        ..GgufFacts::default()
    });
    test_app_with_state(Arc::new(test_state_at(
        model,
        ResponseCache::new(16, Duration::from_secs(60)),
        path.map(std::path::PathBuf::from),
    )))
}

/// upstream's field names, nesting and types, for every field frink
/// states.
///
/// Sabotage: rename `endpoint_metrics` in `insert_server_flags`; this
/// goes red.
#[tokio::test]
async fn props_answers_in_upstreams_shape() {
    let (status, body) = get_json_uri(&app(Some("/models/m.gguf")), frink_api::routes::PROPS).await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let params = &body["default_generation_settings"]["params"];
    for float in [
        "temperature",
        "dynatemp_range",
        "top_p",
        "min_p",
        "top_n_sigma",
        "xtc_probability",
        "xtc_threshold",
        "typical_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "dry_multiplier",
        "dry_base",
    ] {
        assert!(params[float].is_f64(), "{float}: {params}");
    }
    for int in [
        "top_k",
        "repeat_last_n",
        "dry_allowed_length",
        "dry_penalty_last_n",
        "mirostat",
        "max_tokens",
        "n_predict",
        "n_keep",
        "n_probs",
        "min_keep",
    ] {
        assert!(params[int].is_i64(), "{int}: {params}");
    }
    for switch in [
        "ignore_eos",
        "stream",
        "timings_per_token",
        "post_sampling_probs",
    ] {
        assert_eq!(params[switch], false, "{switch}");
    }
    assert_eq!(params["lora"], json!([]));
    assert_eq!(
        params["samplers"],
        json!([
            "penalties",
            "dry",
            "top_n_sigma",
            "top_k",
            "typ_p",
            "top_p",
            "min_p",
            "xtc",
            "temperature"
        ]),
        "llama.cpp's default chain, by its names"
    );

    assert!(body["model_alias"].is_string());
    assert_eq!(body["model_ftype"], "Q4_K - Medium");
    assert_eq!(body["model_path"], "/models/m.gguf");
    assert_eq!(
        body["modalities"],
        json!({"vision": false, "video": false, "audio": false})
    );
    assert_eq!(body["endpoint_slots"], true);
    assert_eq!(body["endpoint_props"], false);
    assert_eq!(body["endpoint_metrics"], true);
    assert_eq!(body["ui"], false);
    assert_eq!(body["is_sleeping"], false);
    assert_eq!(body["cors_proxy_enabled"], false);
    assert!(body["build_info"]
        .as_str()
        .is_some_and(|b| b.starts_with("frink-server ")));
}

/// A field frink has no value for is absent, never defaulted. This
/// fixture has no ceiling, no batcher and a builtin template, so the
/// fields that would describe them are absent too.
#[tokio::test]
async fn a_field_frink_cannot_state_is_left_out() {
    let (_, body) = get_json_uri(&app(None), frink_api::routes::PROPS).await;
    for omitted in [
        "media_marker",
        "ui_settings",
        "chat_template_caps",
        "chat_template_tool_use",
        "connect_code",
        // No `-np` batcher, no path, no source text.
        "total_slots",
        "model_path",
        "chat_template",
    ] {
        assert!(body.get(omitted).is_none(), "{omitted}: {body}");
    }
    let settings = &body["default_generation_settings"];
    assert!(settings.get("n_ctx").is_none(), "no ceiling: {settings}");
    for omitted in [
        "seed",
        "dynatemp_exponent",
        "mirostat_tau",
        "mirostat_eta",
        "adaptive_target",
        "adaptive_decay",
        "n_discard",
        "chat_format",
        "reasoning_format",
        "reasoning_in_content",
        "generation_prompt",
        "speculative.types",
        "backend_sampling",
    ] {
        assert!(
            settings["params"].get(omitted).is_none(),
            "{omitted}: {settings}"
        );
    }
}

/// The defaults `/props` reports are ones `/completion` accepts: sent
/// back, they ask for nothing it would refuse.
///
/// Sabotage: report `mirostat` as 2; `/completion` refuses it by name
/// and this goes red.
#[tokio::test]
async fn the_reported_defaults_are_a_request_completion_serves() {
    let (_, body) = get_json_uri(&app(None), frink_api::routes::PROPS).await;
    let mut request = body["default_generation_settings"]["params"].clone();
    request["prompt"] = json!("hi");
    let request: crate::completion::CompletionRequest =
        serde_json::from_value(request).expect("a completion request");
    request
        .validate(false)
        .expect("every reported default is servable");
}

/// upstream answers `POST /props` without `--props` with a 501, and
/// frink has no `--props`.
#[tokio::test]
async fn post_props_is_upstreams_refusal() {
    let (status, body) = post_json_uri(&app(None), frink_api::routes::PROPS, json!({})).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["type"], "not_supported_error");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .starts_with("This server does not support changing global properties"));
}

/// Asleep, the model is gone and `/props` answers from the snapshot the
/// sleep took, saying it is asleep, as upstream does.
///
/// Sabotage: answer `require_active()?` first in `get_props`; asleep
/// that is a 503 and this goes red.
#[tokio::test]
async fn asleep_props_answers_from_the_snapshot_the_sleep_took() {
    let app = app(Some("/models/m.gguf"));
    let (_, awake) = get_json_uri(&app, frink_api::routes::PROPS).await;
    let (status, _) = post_json_uri(&app, frink_api::routes::SLEEP, json!({})).await;
    assert_eq!(status, StatusCode::OK);
    let (status, asleep) = get_json_uri(&app, frink_api::routes::PROPS).await;
    assert_eq!(status, StatusCode::OK, "{asleep}");
    assert_eq!(asleep["is_sleeping"], true);
    assert_eq!(asleep["model_alias"], awake["model_alias"]);
    assert_eq!(
        asleep["default_generation_settings"],
        awake["default_generation_settings"]
    );
}
