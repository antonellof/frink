//! LoRA adapters in the server: the two routes, the per-request `lora`
//! field, and the one rule that keeps a scale change from racing a
//! generation that is reading the scales.
//!
//! **What llama.cpp does.** The server holds one list of adapters with
//! a scale each (`params_base.lora_adapters`); `POST /lora-adapters`
//! replaces the scales (`server-context.cpp:2705-2711`); a request may
//! carry its own `lora: [{id, scale}]`, which becomes THAT request's
//! list with every unnamed adapter at 0 (`construct_lora_list`,
//! `:1721-1732`); and the scheduler never co-batches two slots whose
//! lists differ (`are_lora_equal`, `:425`), setting the batch's list on
//! the context once per batch (`:2838`).
//!
//! **What frink does.** An adapter's scale is one atomic shared by
//! every projection it decorates (`frink_core::weight_matrix::
//! LoraScale`), read at apply time. So "the scales" are process state
//! rather than a per-slot list, and the equivalent of not co-batching
//! is a reader/writer gate: every generation holds a read lease for
//! its whole run, a `POST /lora-adapters` takes the write side, and a
//! request whose `lora` field asks for scales that differ from the
//! current ones takes the write side too -- waits for the generations
//! in flight, sets the scales, runs alone, and restores them. A
//! request whose `lora` field names exactly the current scales is an
//! ordinary reader, which is the common case for a client that always
//! sends the field.
//!
//! The gate is a `std::sync::RwLock` because both sides are taken on a
//! blocking thread (`run_generation_emit` runs under `spawn_blocking`;
//! the POST handler moves onto one), and a process-wide static rather
//! than a field because the server serves one model at a time and a
//! lease restores through the `Arc<Decoder>` it captured, so a swap
//! during a lease cannot restore the wrong decoder's scales.
//!
//! **Configuration.** `--lora` / `--lora-scaled` are lowered to
//! `FRINK_LORA` (comma-separated `path:scale`) by `cli.rs`, as every
//! server flag is, and `--lora-init-without-apply` to
//! `FRINK_LORA_INIT_WITHOUT_APPLY=1`, which loads every adapter at
//! scale 0 until a `POST` says otherwise. The env is read at every
//! model load, so `/admin/models/load` attaches the same adapters to
//! the next checkpoint or refuses it by name.

use std::sync::{Arc, RwLock};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use frink_api::{LoraAdapterInfo, LoraApplyResponse, LoraScaleRequest};
use frink_gguf::TensorSource;
use frink_models::lora_attach::LoraSpec;
use frink_models::Decoder;

use crate::{invalid_request, unsupported_feature, ApiError, AppState, Model};

/// Comma-separated `path:scale` specs, what `--lora` / `--lora-scaled`
/// lower to.
pub(crate) const ENV_SPECS: &str = "FRINK_LORA";

/// `1` to load every adapter at scale 0 (`--lora-init-without-apply`).
pub(crate) const ENV_INIT_WITHOUT_APPLY: &str = "FRINK_LORA_INIT_WITHOUT_APPLY";

/// The specs the environment names, in order.
pub(crate) fn specs_from_env() -> anyhow::Result<Vec<LoraSpec>> {
    match std::env::var(ENV_SPECS) {
        Ok(raw) => parse_specs(&raw),
        Err(_) => Ok(Vec::new()),
    }
}

/// `FRINK_LORA`'s value: comma-separated `path:scale`.
fn parse_specs(raw: &str) -> anyhow::Result<Vec<LoraSpec>> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| LoraSpec::parse_scaled(s).map_err(|e| anyhow::anyhow!("{ENV_SPECS}: {e}")))
        .collect()
}

fn init_without_apply() -> bool {
    std::env::var(ENV_INIT_WITHOUT_APPLY).is_ok_and(|v| v == "1")
}

/// Attaches the environment's adapters to a freshly loaded decoder.
/// Called by `model.rs` right after `Decoder::from_gguf*`, so a
/// checkpoint the adapters do not fit never becomes the active model.
pub(crate) fn attach_from_env(
    decoder: &mut Decoder,
    base: &impl TensorSource,
) -> anyhow::Result<()> {
    let specs = specs_from_env()?;
    if specs.is_empty() {
        return Ok(());
    }
    decoder
        .attach_lora_specs(base, &specs)
        .map_err(|e| anyhow::anyhow!("lora: {e}"))?;
    if init_without_apply() {
        decoder
            .set_lora_scales(&[])
            .map_err(|e| anyhow::anyhow!("lora: {e}"))?;
        tracing::info!(
            "{} adapter(s) loaded at scale 0 ({ENV_INIT_WITHOUT_APPLY}); apply them with \
             POST /lora-adapters",
            specs.len()
        );
    }
    Ok(())
}

/// The dedicated engines do not go through `Decoder`, and a flag that
/// is accepted must reach the thing it names.
pub(crate) fn refuse_env_for_engine(engine: &str) -> anyhow::Result<()> {
    if specs_from_env()?.is_empty() {
        return Ok(());
    }
    anyhow::bail!(
        "{ENV_SPECS} (--lora) is not implemented for the {engine} engine: only the generic \
         decoder attaches adapters; refusing rather than serving the base weights"
    )
}

// --- the gate --------------------------------------------------------

static GATE: RwLock<()> = RwLock::new(());

/// Held, never read: the guard IS the lease.
enum Guard {
    Read(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>),
    Write(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>),
}

/// Held for the whole of one generation (or one `POST`). Dropping it
/// releases the gate and, for a request that overrode the scales,
/// puts the previous ones back.
pub(crate) struct LoraLease {
    _guard: Guard,
    restore: Option<(Arc<Decoder>, Vec<f32>)>,
}

impl Drop for LoraLease {
    fn drop(&mut self) {
        if let Some((decoder, previous)) = self.restore.take() {
            let scales: Vec<(usize, f32)> = previous.into_iter().enumerate().collect();
            // The ids were this decoder's own a moment ago, so the only
            // way this fails is a decoder with fewer adapters than the
            // lease captured, which cannot happen: adapters are attached
            // at load and never removed.
            let _ = decoder.set_lora_scales(&scales);
        }
    }
}

/// Take the gate for one generation. `scales`, when present, is the
/// FULL per-id vector the generation must run under
/// ([`resolve_request`]). When it is what is already applied the lease
/// is shared; otherwise it is exclusive, sets the scales, and restores
/// them on drop. Blocking: call it on the generation's own thread.
pub(crate) fn lease(model: &Model, scales: Option<&[f32]>) -> LoraLease {
    let shared = || LoraLease {
        _guard: Guard::Read(GATE.read().unwrap_or_else(|e| e.into_inner())),
        restore: None,
    };
    let Some(want) = scales else {
        return shared();
    };
    let Model::Gguf(g) = model else {
        // `resolve_request` refused this combination before the
        // generation started.
        return shared();
    };
    // Read under the shared side first: the common case is a request
    // that names what is already applied, and it must not queue behind
    // the readers it could have joined.
    let read = GATE.read().unwrap_or_else(|e| e.into_inner());
    if g.decoder.lora_scales() == want {
        return LoraLease {
            _guard: Guard::Read(read),
            restore: None,
        };
    }
    drop(read);
    let guard = GATE.write().unwrap_or_else(|e| e.into_inner());
    let previous = g.decoder.lora_scales();
    let scales: Vec<(usize, f32)> = want.iter().copied().enumerate().collect();
    // Validated against this decoder by `resolve_request`.
    let _ = g.decoder.set_lora_scales(&scales);
    LoraLease {
        _guard: Guard::Write(guard),
        restore: Some((Arc::clone(&g.decoder), previous)),
    }
}

/// Resolves a request's `lora` field against the model into the FULL
/// per-id scale vector the generation runs under: the request's own
/// (unnamed adapters at 0) when it sent one, else the scales currently
/// applied. `None` only when the model holds no adapter at all.
///
/// Always the effective vector rather than "only when it differs",
/// because the response cache keys on it: a `POST /lora-adapters`
/// between two identical requests changes the answer, and a key that
/// could not see the current scales would serve the first answer to
/// the second request.
///
/// Refuses, rather than ignoring as upstream does, an id no loaded
/// adapter has; and refuses the field on a model that holds no
/// adapters at all, because a client naming an adapter deserves to
/// hear there is none.
pub(crate) fn resolve_request(
    model: &Model,
    requested: Option<&[LoraScaleRequest]>,
) -> Result<Option<Vec<f32>>, ApiError> {
    let Model::Gguf(g) = model else {
        return match requested {
            Some(_) => Err(unsupported_feature(
                "`lora` is only served by the generic decoder; this checkpoint runs on a \
                 dedicated engine that attaches no adapters",
            )),
            None => Ok(None),
        };
    };
    let n = g.decoder.lora_adapters.len();
    let current = || (n > 0).then(|| g.decoder.lora_scales());
    let Some(requested) = requested else {
        return Ok(current());
    };
    if requested.is_empty() {
        // `"lora": []` is what a stock llama.cpp client sends when it
        // has nothing to say, and upstream reads it as "the server's
        // own list" (`launch_slot_with_task`, `server-context.cpp:
        // 1736-1750`: only a NON-empty list builds a per-request one).
        return Ok(current());
    }
    if n == 0 {
        return Err(invalid_request(
            "`lora` names an adapter but none is loaded (start the server with --lora)",
            "lora",
        ));
    }
    let mut want = vec![0f32; n];
    for entry in requested {
        if entry.id >= n {
            return Err(invalid_request(
                &format!(
                    "lora adapter id {} is out of range: {n} adapter(s) loaded",
                    entry.id
                ),
                "lora",
            ));
        }
        want[entry.id] = entry.scale;
    }
    Ok(Some(want))
}

// --- the routes ------------------------------------------------------

/// The adapters the active model carries, one entry per id.
pub(crate) fn list(model: &Model) -> Vec<LoraAdapterInfo> {
    let Model::Gguf(g) = model else {
        return Vec::new();
    };
    g.decoder
        .lora_adapters
        .iter()
        .enumerate()
        .map(|(id, a)| LoraAdapterInfo {
            id,
            path: a.path.display().to_string(),
            scale: a.scale(),
            task_name: a.task_name.clone(),
            prompt_prefix: a.prompt_prefix.clone(),
        })
        .collect()
}

/// `GET /lora-adapters`.
pub(crate) async fn get_lora_adapters(State(state): State<Arc<AppState>>) -> Response {
    let active = match state.require_active() {
        Ok(a) => a,
        Err(e) => return e.into_response(),
    };
    let model = match active.generative() {
        Ok(m) => m,
        Err(_) => return Json(Vec::<LoraAdapterInfo>::new()).into_response(),
    };
    Json(list(model)).into_response()
}

/// `POST /lora-adapters`: the body is the `[{id, scale}]` array; every
/// adapter not named goes to 0. Waits for the generations in flight,
/// as upstream's task queue orders it after the batch that is running.
pub(crate) async fn post_lora_adapters(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Vec<LoraScaleRequest>>,
) -> Response {
    let active = match state.require_active() {
        Ok(a) => a,
        Err(e) => return e.into_response(),
    };
    let model = match active.generative() {
        Ok(m) => Arc::clone(m),
        Err(e) => return e.into_response(),
    };
    let Model::Gguf(_) = &*model else {
        return unsupported_feature(
            "this checkpoint runs on a dedicated engine that attaches no adapters",
        )
        .into_response();
    };
    let scales: Vec<(usize, f32)> = body.iter().map(|e| (e.id, e.scale)).collect();
    let result = tokio::task::spawn_blocking(move || {
        let _gate = GATE.write().unwrap_or_else(|e| e.into_inner());
        let Model::Gguf(g) = &*model else {
            unreachable!("checked above");
        };
        g.decoder.set_lora_scales(&scales)
    })
    .await;
    match result {
        Ok(Ok(())) => Json(LoraApplyResponse { success: true }).into_response(),
        Ok(Err(msg)) => invalid_request(&msg, "id").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": format!("lora apply task failed: {e}")}})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_specs_parse_in_order_and_refuse_a_bad_scale() {
        let specs = parse_specs("a.gguf:1, b.gguf:0.5 ,").unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[1].scale, 0.5);
        assert!(parse_specs("a.gguf")
            .unwrap_err()
            .to_string()
            .contains("FNAME:SCALE"));
        assert!(parse_specs("").unwrap().is_empty());
    }

    #[test]
    fn a_lease_without_an_override_is_shared_and_restores_nothing() {
        // Two shared leases coexist: the gate is a reader lock.
        let a = LoraLease {
            _guard: Guard::Read(GATE.read().unwrap()),
            restore: None,
        };
        let b = LoraLease {
            _guard: Guard::Read(GATE.read().unwrap()),
            restore: None,
        };
        // While either is held, the write side is not available.
        assert!(GATE.try_write().is_err());
        drop(a);
        drop(b);
        // And the write side is free afterwards. A BLOCKING acquire,
        // not `try_write`: `GATE` is process-wide and the `http_tests`
        // module takes read leases on it from other test threads, so
        // `try_write` here failed whenever one of those happened to be
        // in flight (red on `main` twice on 2026-09-18 with nothing
        // changed near it). What this test proves is that its own two
        // guards were released; a leaked guard would hang this line,
        // and another test's transient lease merely delays it.
        drop(GATE.write().unwrap());
    }
}

#[cfg(test)]
mod http_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::http::StatusCode;
    use frink_core::weight_matrix::{LoraDelta, LoraScale};
    use frink_models::lora_attach::LoraAttached;
    use frink_models::Decoder;

    use crate::response_cache::ResponseCache;
    use crate::tests::{get_json, post_json_uri, test_app_with_state, test_state};
    use crate::{chat_template, GgufModel, Model, ServerTokenizer, StopTokens};

    /// A byte-vocab random decoder with `n` hand-attached adapters on
    /// the output head, each at scale 1: enough to exercise the routes
    /// and the lease without a fixture file. On the head, so that a
    /// scale change is visible in the greedy pick: adapter `i` pushes
    /// logit `10 + i` by `scale * 4 * sum(normed hidden)`.
    fn model_with_adapters(n: usize) -> (Model, Arc<Decoder>) {
        let mut cfg = frink_models::config::test_dense_fixture();
        cfg.vocab_size = 256;
        let mut d = Decoder::new_random_small(cfg, 2, 256);
        for i in 0..n {
            let head = &mut d.output_head;
            let (rows, cols) = (head.rows(), head.cols());
            let scale = LoraScale::new(1.0);
            // A = ones (rank 1), so `A x` is the sum of the normed
            // hidden state; B is one entry of 4 at token `10 + i`.
            let mut b = vec![0.0; rows];
            b[10 + i] = 4.0;
            head.attach_lora(
                LoraDelta::new(vec![1.0; cols], b, 1, rows, cols, 0.0, scale.clone()).unwrap(),
            );
            d.lora_adapters.push(LoraAttached {
                path: format!("adapter_{i}.gguf").into(),
                alpha: 0.0,
                task_name: String::new(),
                prompt_prefix: String::new(),
                scale,
                n_tensors: 1,
            });
        }
        let decoder = Arc::new(d);
        let model = Model::Gguf(GgufModel {
            decoder: Arc::clone(&decoder),
            tokenizer: Arc::new(ServerTokenizer::Byte),
            stop_tokens: StopTokens::default(),
            bos_id: None,
            is_synthetic: true,
            chat_template: chat_template::PromptTemplate::plain(),
            draft: None,
        });
        (model, decoder)
    }

    fn app(model: Model) -> axum::Router {
        test_app_with_state(Arc::new(test_state(
            model,
            ResponseCache::new(16, Duration::from_secs(60)),
        )))
    }

    #[tokio::test]
    async fn a_model_without_adapters_lists_none_and_refuses_the_field_by_name() {
        let (model, _) = model_with_adapters(0);
        let app = app(model);
        let (status, body) = get_json(&app, frink_api::routes::LORA_ADAPTERS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!([]));

        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 1, "lora": [{"id": 0, "scale": 0.5}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("none is loaded"),
            "{body}"
        );

        // An empty list asks for nothing, as upstream reads it.
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::COMPLETION,
            serde_json::json!({"prompt": "hi", "n_predict": 1, "lora": []}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn get_lists_every_adapter_and_post_sets_the_scales_zeroing_the_unnamed() {
        let (model, decoder) = model_with_adapters(2);
        let app = app(model);
        let (status, body) = get_json(&app, frink_api::routes::LORA_ADAPTERS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            serde_json::json!([
                {"id": 0, "path": "adapter_0.gguf", "scale": 1.0, "task_name": "", "prompt_prefix": ""},
                {"id": 1, "path": "adapter_1.gguf", "scale": 1.0, "task_name": "", "prompt_prefix": ""},
            ])
        );

        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::LORA_ADAPTERS,
            serde_json::json!([{"id": 1, "scale": 0.25}]),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body, serde_json::json!({"success": true}));
        assert_eq!(
            decoder.lora_scales(),
            vec![0.0, 0.25],
            "unnamed adapter 0 went to 0"
        );
        let (_, body) = get_json(&app, frink_api::routes::LORA_ADAPTERS).await;
        assert_eq!(body[0]["scale"], 0.0);
        assert_eq!(body[1]["scale"], 0.25);

        // An id no adapter has is refused and changes nothing.
        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::LORA_ADAPTERS,
            serde_json::json!([{"id": 2, "scale": 1.0}]),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("out of range"));
        assert_eq!(decoder.lora_scales(), vec![0.0, 0.25]);

        // `scale` absent reads 0, as upstream's json_value default.
        let (status, _) = post_json_uri(
            &app,
            frink_api::routes::LORA_ADAPTERS,
            serde_json::json!([{"id": 0}]),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(decoder.lora_scales(), vec![0.0, 0.0]);
    }

    #[tokio::test]
    async fn a_per_request_override_runs_alone_and_restores_the_scales_after() {
        let (model, decoder) = model_with_adapters(2);
        let app = app(model);
        assert_eq!(decoder.lora_scales(), vec![1.0, 1.0]);

        let run = |lora: serde_json::Value| {
            let app = app.clone();
            async move {
                post_json_uri(
                    &app,
                    frink_api::routes::COMPLETION,
                    serde_json::json!({
                        "prompt": "abc", "n_predict": 3, "temperature": 0.0, "lora": lora
                    }),
                )
                .await
            }
        };
        let (status, plain) =
            run(serde_json::json!([{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}])).await;
        assert_eq!(status, StatusCode::OK, "{plain}");
        // Adapter 1 alone at a scale that dominates every logit: the
        // greedy pick becomes its token (byte 0x0b) at every step
        // where the hidden state's sum has the sign the scale needs,
        // and one of the two signs has it at the first step.
        let content = |b: &serde_json::Value| b["content"].as_str().unwrap().to_string();
        let mut hit = false;
        for sign in [1000.0, -1000.0] {
            let (status, overridden) = run(serde_json::json!([{"id": 1, "scale": sign}])).await;
            assert_eq!(status, StatusCode::OK, "{overridden}");
            assert_eq!(
                decoder.lora_scales(),
                vec![1.0, 1.0],
                "the override lasted exactly one generation"
            );
            // The synthetic demo banner renders the decoded text with
            // `{:?}`, so byte 0x0b shows as the six characters `\u{b}`.
            hit |= content(&overridden).contains("\\u{b}");
        }
        assert!(hit, "the override must have reached the weights: {plain}");
        assert!(!content(&plain).contains("\\u{b}"));

        // Out of range is refused before anything runs.
        let (status, body) = run(serde_json::json!([{"id": 5, "scale": 1.0}])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(decoder.lora_scales(), vec![1.0, 1.0]);
    }

    #[test]
    fn a_lease_sets_the_scales_for_its_lifetime_and_the_head_sees_them() {
        let (model, decoder) = model_with_adapters(2);
        let mut kv = decoder.config.new_kv_caches();
        let before = decoder.forward_token(3, 0, &mut kv);
        {
            let _lease = super::lease(&model, Some(&[0.0, 1000.0]));
            assert_eq!(decoder.lora_scales(), vec![0.0, 1000.0]);
            let mut kv = decoder.config.new_kv_caches();
            let during = decoder.forward_token(3, 0, &mut kv);
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            };
            assert_ne!(before, during, "the head must see the scale");
            drop(_lease);
            // A second exclusive lease after the first is released (one
            // thread cannot hold two: the gate is a real lock).
            let _neg = super::lease(&model, Some(&[0.0, -1000.0]));
            let mut kv = decoder.config.new_kv_caches();
            let neg = decoder.forward_token(3, 0, &mut kv);
            assert!(
                argmax(&during) == 11 || argmax(&neg) == 11,
                "one sign must make token 11 the greedy pick: {} / {}",
                argmax(&during),
                argmax(&neg)
            );
        }
        assert_eq!(decoder.lora_scales(), vec![1.0, 1.0], "restored on drop");
    }

    /// The same override on `/v1/completions` and `/v1/chat/completions`:
    /// all three routes resolve through one function.
    #[tokio::test]
    async fn the_openai_routes_take_the_same_field() {
        let (model, decoder) = model_with_adapters(1);
        let app = app(model);
        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_COMPLETIONS,
            serde_json::json!({"model": "m", "prompt": "hi", "max_tokens": 1, "lora": [{"id": 3}]}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let (status, body) = post_json_uri(
            &app,
            frink_api::routes::V1_CHAT_COMPLETIONS,
            serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1, "lora": [{"id": 0, "scale": 0.5}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(
            decoder.lora_scales(),
            vec![1.0],
            "restored after the chat turn"
        );
    }
}
