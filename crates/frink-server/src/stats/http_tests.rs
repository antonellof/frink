//! `/v1/stats` over HTTP, on both decode paths: the private loop and
//! the continuous batch worker, each asked a buffered and a streamed
//! request.
//!
//! The token checks are against the ring's totals, which are summed
//! from each reply's `usage`: with no prefix cache every prompt token
//! is computed, so the prefill window must hold exactly the prompt
//! tokens and the decode window exactly the completion tokens. A path
//! that counted nothing, or counted a token twice, disagrees with its
//! own usage block.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use frink_api::routes::{V1_CHAT_COMPLETIONS, V1_COMPLETIONS, V1_STATS};

use crate::serving::batch::{BatcherConfig, ContinuousBatcher};
use crate::tests::{
    get_json_uri, greedy_params, post_json_uri, test_app_with_state, test_model_full_byte_vocab,
    test_state,
};
use crate::{ActiveModel, AppState, Loaded, Model, ResponseCache};

/// A server over the byte fixture, decoding on the private loop or,
/// with `batched`, on a continuous batch worker.
fn server(batched: bool) -> Arc<AppState> {
    let model = test_model_full_byte_vocab();
    let state = test_state(model, ResponseCache::new(1000, Duration::from_secs(3600)));
    if batched {
        let current = state.active().expect("the fixture is loaded");
        let model: Arc<Model> = Arc::clone(current.generative().expect("generative"));
        let Model::Gguf(gguf) = &*model else {
            unreachable!("the byte fixture is a GGUF decoder")
        };
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&gguf.decoder),
            Arc::new(|ids: &[usize]| ids.iter().map(|&id| id as u8).collect()),
            BatcherConfig::default(),
        );
        *state.active.write().unwrap() = Some(Arc::new(ActiveModel {
            id: None,
            loaded: Loaded::Generative(model),
            batcher: Some(batcher),
            ceiling: None,
            checkpoint_path: None,
        }));
    }
    Arc::new(state)
}

/// Tokens each window holds: (prefill, decode).
fn windows(state: &AppState) -> (u64, u64) {
    state.serving.lock().window_tokens()
}

/// One buffered and one streamed request, each answered and counted
/// once, with tokens on both windows matching the two replies' usage.
async fn counts_every_request_and_token(batched: bool) {
    let state = server(batched);
    let app = test_app_with_state(Arc::clone(&state));

    let (status, before) = get_json_uri(&app, V1_STATS).await;
    assert_eq!(status, StatusCode::OK);
    let requests = &before["requests"];
    assert_eq!(
        (
            &requests["active"],
            &requests["completed"],
            &requests["aborted"]
        ),
        (&0.into(), &0.into(), &0.into()),
        "{before}"
    );
    assert_eq!(before["throughput"]["decode_tps"], 0.0, "{before}");
    assert_eq!(before["throughput"]["prefill_tps"], 0.0, "{before}");

    let (status, reply) = post_json_uri(
        &app,
        V1_COMPLETIONS,
        serde_json::json!({"model": "x", "prompt": "hello", "max_tokens": 4, "temperature": 0}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["usage"]["prompt_tokens"], 5, "{reply}");
    assert_eq!(reply["usage"]["completion_tokens"], 4, "{reply}");
    assert_eq!(windows(&state), (5, 4), "the buffered request's tokens");

    // Streamed: the body is SSE, so it is read to its end and not parsed.
    let (status, _) = post_json_uri(
        &app,
        V1_CHAT_COMPLETIONS,
        serde_json::json!({
            "model": "x",
            "messages": [{"role": "user", "content": "hi there"}],
            "max_tokens": 3,
            "temperature": 0,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (prefilled, decoded) = windows(&state);
    let (_, after) = get_json_uri(&app, V1_STATS).await;
    let requests = &after["requests"];
    assert_eq!(requests["active"], 0, "{after}");
    assert_eq!(requests["completed"], 2, "{after}");
    assert_eq!(requests["aborted"], 0, "{after}");
    assert_eq!(
        requests["prompt_tokens_total"], prefilled,
        "every prompt token was computed: {after}"
    );
    assert_eq!(requests["completion_tokens_total"], 7, "{after}");
    assert_eq!(decoded, 7, "four buffered and three streamed: {after}");
    for rate in ["decode_tps", "prefill_tps"] {
        assert!(
            after["throughput"][rate].as_f64().is_some_and(|r| r > 0.0),
            "{rate} must be positive right after a request: {after}"
        );
    }
}

#[tokio::test]
async fn stats_count_requests_and_tokens_on_the_private_loop() {
    counts_every_request_and_token(false).await;
}

#[tokio::test]
async fn stats_count_requests_and_tokens_on_the_batch_worker() {
    counts_every_request_and_token(true).await;
}

/// A generation cancelled before it ran is answered (with what it had,
/// which is nothing) and counted as an abort, not a completion, on
/// either path. Driven through the seam itself: an HTTP cancel races
/// the decode, and a pre-cancelled token does not.
#[tokio::test]
async fn a_cancelled_generation_is_counted_as_aborted_on_both_paths() {
    for batched in [false, true] {
        let state = server(batched);
        let active = state.active().expect("loaded");
        let mut params = greedy_params(4);
        let token = crate::cancel::CancelToken::new();
        token.cancel();
        params.cancel = Some(token);
        crate::run_generation(
            active.generative().unwrap(),
            "hello",
            &params,
            None,
            None,
            None,
            active.batcher.as_ref(),
            None,
            None,
            Some(&state.serving),
        )
        .expect("a cancelled generation is still answered");

        let app = test_app_with_state(Arc::clone(&state));
        let (_, body) = get_json_uri(&app, V1_STATS).await;
        let requests = &body["requests"];
        assert_eq!(requests["aborted"], 1, "batched={batched}: {body}");
        assert_eq!(requests["completed"], 0, "batched={batched}: {body}");
        assert_eq!(requests["active"], 0, "batched={batched}: {body}");
    }
}

/// A generation that fails after admission leaves the in-flight count
/// and is neither completed nor aborted.
#[tokio::test]
async fn a_failed_generation_is_neither_completed_nor_aborted() {
    let state = server(false);
    let active = state.active().expect("loaded");
    let out_of_vocab: Vec<usize> = vec![1, 100_000];
    let failed = crate::run_generation(
        active.generative().unwrap(),
        crate::prompt_input::Prompt::Ids(&out_of_vocab),
        &greedy_params(4),
        None,
        None,
        None,
        None,
        None,
        None,
        Some(&state.serving),
    );
    assert!(failed.is_err(), "an id past the vocabulary is refused");

    let app = test_app_with_state(Arc::clone(&state));
    let (_, body) = get_json_uri(&app, V1_STATS).await;
    let requests = &body["requests"];
    assert_eq!(
        (
            &requests["active"],
            &requests["completed"],
            &requests["aborted"]
        ),
        (&0.into(), &0.into(), &0.into()),
        "{body}"
    );
}
