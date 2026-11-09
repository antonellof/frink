//! `GET /slots`: llama-server's live slot listing, behind its `--slots`
//! / `--no-slots` switch.
//!
//! Upstream answers with one object per slot (`server_slot::to_json` in
//! `tools/server/server-context.cpp`): `id`, `n_ctx`, `speculative` and
//! `is_processing` always, and, once a slot has served a task, the
//! task's id, its prompt counts, its sampling `params` and a
//! `next_token` block. `?fail_on_no_slot=1` turns "every slot is busy"
//! into a 503.
//!
//! # What a slot is here
//!
//! A SEAT of the continuous batcher. `-np N` caps it at N concurrent
//! sequences, which is upstream's `n_parallel` and the only fixed slot
//! count this server has, so the listing has N entries and the first
//! `running` of them are busy. That is all the batcher can say about a
//! seat: frink builds KV per request and shares prefixes through one
//! cache, so a seat is not a KV region that keeps a task's state after
//! it ends, and its `id` is a position in this listing, not an address
//! (the same bookkeeping `POST /slots/{id}` gives an id).
//!
//! So every seat carries the four fields that are always present
//! upstream, and none of the per-task ones, which a seat does not hold.
//! A deployment with no seats (no batcher, or a batcher without `-np`)
//! has no slot count to list, and says so by name rather than inventing
//! one.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::llama_error::{llama_error, LlamaErrorKind};
use crate::{ApiError, AppState};

/// What `--slots` / `--no-slots` lower to: `0` turns the listing off.
pub(crate) const ENDPOINT_SLOTS_ENV: &str = "FRINK_ENDPOINT_SLOTS";

/// Whether `GET /slots` is served. On unless `--no-slots`, as upstream
/// (`endpoint_slots = true` in `common/common.h`).
pub(crate) fn endpoint_enabled() -> bool {
    std::env::var(ENDPOINT_SLOTS_ENV).map_or(true, |v| v.trim() != "0")
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlotsQuery {
    #[serde(default)]
    fail_on_no_slot: Option<String>,
}

/// The seats of a batcher capped at `max_seqs`, `running` of them busy.
fn seats(max_seqs: usize, running: usize, n_ctx: Option<usize>) -> Vec<Value> {
    (0..max_seqs)
        .map(|id| {
            let mut slot = Map::new();
            slot.insert("id".into(), json!(id));
            if let Some(n_ctx) = n_ctx {
                slot.insert("n_ctx".into(), json!(n_ctx));
            }
            // A batched request never drafts: speculative decoding is the
            // private decode loop's alone.
            slot.insert("speculative".into(), json!(false));
            slot.insert("is_processing".into(), json!(id < running));
            Value::Object(slot)
        })
        .collect()
}

/// `GET /slots`.
pub(crate) async fn get_slots(
    State(state): State<Arc<AppState>>,
    Query(query): Query<SlotsQuery>,
) -> Result<Json<Value>, ApiError> {
    listing(endpoint_enabled(), &state, query)
}

/// The listing, with the switch passed in: a test that set
/// `FRINK_ENDPOINT_SLOTS` would turn the route off for every other test
/// in the process.
fn listing(enabled: bool, state: &AppState, query: SlotsQuery) -> Result<Json<Value>, ApiError> {
    if !enabled {
        return Err(llama_error(
            LlamaErrorKind::NotSupported,
            "This server does not support slots endpoint. Start it with `--slots`",
        ));
    }
    let active = state.require_active()?;
    let Some(batcher) = active.batcher.as_ref() else {
        return Err(llama_error(
            LlamaErrorKind::NotSupported,
            "this server has no slots to list: a slot is a seat of the continuous batcher, and \
             this deployment decodes every request on its own thread (continuous batching is \
             off, or a KV pool or prefix cache keeps requests on the private decode loop)",
        ));
    };
    let stats = batcher.stats();
    if stats.max_seqs == 0 {
        return Err(llama_error(
            LlamaErrorKind::NotSupported,
            "this server has no slot count to list: its continuous batcher seats any number of \
             sequences. Start it with `-np N` for N slots",
        ));
    }
    let running = stats.running.min(stats.max_seqs);
    if query.fail_on_no_slot.is_some_and(|v| !v.is_empty()) && running == stats.max_seqs {
        return Err(llama_error(
            LlamaErrorKind::Unavailable,
            "no slot available",
        ));
    }
    let n_ctx = active.ceiling.as_ref().and_then(|c| c.limit());
    Ok(Json(Value::Array(seats(stats.max_seqs, running, n_ctx))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// upstream's four always-present fields, with their types, and the
    /// busy seats first.
    #[test]
    fn a_seat_carries_upstreams_four_fields_and_the_busy_ones_come_first() {
        let listed = seats(3, 2, Some(4096));
        assert_eq!(listed.len(), 3);
        for (id, slot) in listed.iter().enumerate() {
            assert_eq!(slot["id"], id);
            assert_eq!(slot["n_ctx"], 4096);
            assert_eq!(slot["speculative"], false);
            assert_eq!(slot["is_processing"], id < 2);
            assert_eq!(slot.as_object().unwrap().len(), 4, "{slot}");
        }
    }

    /// No ceiling, no `n_ctx`: absent rather than a zero.
    #[test]
    fn a_seat_without_a_context_size_omits_n_ctx() {
        let listed = seats(1, 0, None);
        assert!(listed[0].get("n_ctx").is_none());
        assert_eq!(listed[0]["is_processing"], false);
    }

    fn query(fail_on_no_slot: Option<&str>) -> SlotsQuery {
        SlotsQuery {
            fail_on_no_slot: fail_on_no_slot.map(str::to_string),
        }
    }

    /// The fixture state, with its model served through a batcher of
    /// `max_seqs` seats (`0` for upstream's unlimited) and a ceiling.
    fn batched_state(max_seqs: usize) -> AppState {
        use crate::serving::batch::{BatcherConfig, ContinuousBatcher};
        let state = crate::tests::test_state(
            crate::tests::test_byte_model_with_facts(crate::model::GgufFacts::default()),
            crate::response_cache::ResponseCache::new(16, std::time::Duration::from_secs(60)),
        );
        let active = state.active().expect("the fixture is loaded");
        let decoder = Arc::clone(active.generative().unwrap().gguf_decoder().unwrap());
        let batcher = ContinuousBatcher::spawn_with_config(
            Arc::clone(&decoder),
            Arc::new(|ids: &[usize]| ids.iter().map(|&i| i as u8).collect()),
            BatcherConfig {
                max_seqs: if max_seqs == 0 { usize::MAX } else { max_seqs },
                ..BatcherConfig::default()
            },
        );
        state.swap_active(Some(Arc::new(crate::ActiveModel {
            id: None,
            loaded: crate::Loaded::Generative(Arc::clone(active.generative().unwrap())),
            batcher: Some(batcher),
            ceiling: Some(Arc::new(crate::budget::ContextCeiling::new(
                Some(512),
                frink_models::KvShape::host(&decoder.config),
            ))),
            checkpoint_path: None,
        })));
        state
    }

    /// An idle `-np 3` server lists three free seats in upstream's shape.
    ///
    /// Sabotage: list `max_seqs - 1` seats; this goes red.
    #[test]
    fn an_idle_batcher_lists_its_seats_free() {
        let state = batched_state(3);
        let Json(body) = listing(true, &state, query(None)).expect("listed");
        let slots = body.as_array().unwrap();
        assert_eq!(slots.len(), 3);
        assert!(slots
            .iter()
            .all(|s| s["is_processing"] == false && s["n_ctx"] == 512));
        // A free seat exists, so `fail_on_no_slot` does not fire.
        assert!(listing(true, &state, query(Some("1"))).is_ok());
    }

    /// `--no-slots` is upstream's 501, in its words.
    #[test]
    fn the_switch_off_is_upstreams_refusal() {
        let (status, body) = listing(false, &batched_state(3), query(None)).unwrap_err();
        assert_eq!(status.as_u16(), 501);
        assert_eq!(
            body.0["error"]["message"],
            "This server does not support slots endpoint. Start it with `--slots`"
        );
        assert_eq!(body.0["error"]["type"], "not_supported_error");
    }

    /// No batcher, or a batcher without `-np`, has no slot count, and
    /// says which.
    #[test]
    fn a_deployment_without_seats_says_so_rather_than_inventing_them() {
        let unbatched = crate::tests::test_state(
            crate::tests::test_byte_model_with_facts(crate::model::GgufFacts::default()),
            crate::response_cache::ResponseCache::new(16, std::time::Duration::from_secs(60)),
        );
        let (status, body) = listing(true, &unbatched, query(None)).unwrap_err();
        assert_eq!(status.as_u16(), 501);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no slots to list"));

        let (status, body) = listing(true, &batched_state(0), query(None)).unwrap_err();
        assert_eq!(status.as_u16(), 501);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("-np N"));
    }

    /// The route is mounted: a 404 would be indistinguishable from a
    /// build with no listing at all.
    #[tokio::test]
    async fn the_route_is_registered() {
        let (status, body) =
            crate::tests::get_json_uri(&crate::tests::test_app(), frink_api::routes::SLOTS).await;
        assert_eq!(status.as_u16(), 501, "{body}");
        assert!(body["error"]["message"].is_string());
    }
}
