use super::*;
use crate::policy::maintenance::MaintenanceGate;
use crate::tests::{
    get_json_uri, post_json_uri, test_app_with_state, test_model_full_byte_vocab, test_state,
};
use frink_api::routes::{V1_CACHE_REBUILD, V1_COMPLETIONS, V1_STATS};
use frink_core::cache::{KvBlockPool, KvCache};
use std::sync::Mutex;
use std::time::Duration;

/// Tokens per page in the pools below.
const PAGE: u64 = 16;

/// A server whose shared KV pool holds `blocks` pages of [`PAGE`]
/// tokens, with the byte budget it was sized from, if any.
fn pooled(
    blocks: usize,
    byte_budget: Option<crate::generate::KvByteBudget>,
) -> (Arc<AppState>, Arc<Mutex<KvBlockPool>>) {
    let pool = Arc::new(Mutex::new(KvBlockPool::new(PAGE as usize, blocks)));
    let mut state = test_state(
        test_model_full_byte_vocab(),
        crate::ResponseCache::new(1000, Duration::from_secs(3600)),
    );
    state.kv_pool = Some(crate::generate::KvPoolConfig {
        pool: Arc::clone(&pool),
        queue_wait: Duration::ZERO,
        byte_budget,
    });
    (Arc::new(state), pool)
}

fn total(pool: &Mutex<KvBlockPool>) -> usize {
    pool.lock().unwrap().total_blocks()
}

fn serving(state: &AppState) -> bool {
    state.maintenance.lock().unwrap().check_admission().is_ok()
}

/// The pre-check the endpoint exists to run: a re-split the pool's byte
/// budget cannot pay for is refused BEFORE the pool moves, and the gate
/// reopens. A target inside the budget is applied.
#[tokio::test]
async fn a_rebuild_over_the_byte_budget_is_refused_before_the_pool_moves() {
    let (state, pool) = pooled(
        32,
        Some(crate::generate::KvByteBudget {
            budget_bytes: 64 * 1000,
            bytes_per_block: 1000,
        }),
    );
    let app = test_app_with_state(Arc::clone(&state));

    let (status, body) =
        post_json_uri(&app, V1_CACHE_REBUILD, serde_json::json!({"kv": 65 * PAGE})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|e| e.contains("FRINK_KV_BYTE_BUDGET")),
        "{body}"
    );
    assert_eq!(total(&pool), 32, "a refused rebuild changes nothing");
    assert!(serving(&state), "and reopens the gate");

    let (status, body) =
        post_json_uri(&app, V1_CACHE_REBUILD, serde_json::json!({"kv": 64 * PAGE})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["kv"]["num_pages"], 64);
    assert_eq!(total(&pool), 64);
    assert!(serving(&state));
}

/// A pool sized in blocks has no byte figure, so the same size a
/// budget refuses is a valid target there.
#[tokio::test]
async fn a_pool_sized_in_blocks_has_no_byte_ceiling() {
    let (state, pool) = pooled(32, None);
    let app = test_app_with_state(state);
    let (status, body) = post_json_uri(
        &app,
        V1_CACHE_REBUILD,
        serde_json::json!({"kv": 4096 * PAGE}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(total(&pool), 4096);
}

/// Each refusal names what was wrong, and none of them moves the pool
/// or leaves the gate closed.
#[tokio::test]
async fn a_rebuild_the_deployment_cannot_do_is_refused_by_name() {
    let (state, pool) = pooled(32, None);
    let app = test_app_with_state(Arc::clone(&state));
    for (body, says) in [
        (serde_json::json!({"moe": 64}), "no moe pool"),
        (serde_json::json!({"swa": 64, "kv": 64}), "no swa pool"),
        (serde_json::json!({}), "nothing to rebuild"),
        (
            serde_json::json!({"kv": PAGE - 1}),
            "below one 16-token page",
        ),
    ] {
        let (status, reply) = post_json_uri(&app, V1_CACHE_REBUILD, body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {reply}");
        assert!(
            reply["error"].as_str().is_some_and(|e| e.contains(says)),
            "{body} should say {says:?}: {reply}"
        );
        assert_eq!(total(&pool), 32);
        assert!(serving(&state));
    }

    // And a deployment with no shared pool says how to get one.
    let app = test_app_with_state(Arc::new(test_state(
        test_model_full_byte_vocab(),
        crate::ResponseCache::new(1000, Duration::from_secs(3600)),
    )));
    let (status, reply) =
        post_json_uri(&app, V1_CACHE_REBUILD, serde_json::json!({"kv": 64})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        reply["error"]
            .as_str()
            .is_some_and(|e| e.contains("FRINK_KV_POOL_BLOCKS")),
        "{reply}"
    );
}

/// Shrinking under what in-flight requests hold is the retryable
/// refusal: `409`, the floor in tokens, the pool unchanged. And a
/// rebuild through another one is refused by the gate before anything
/// is measured.
#[tokio::test]
async fn a_rebuild_under_what_is_held_or_through_another_is_a_409() {
    let (state, pool) = pooled(32, None);
    let app = test_app_with_state(Arc::clone(&state));
    let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 64).expect("blocks");
    let in_use = {
        let p = pool.lock().unwrap();
        p.total_blocks() - p.free_blocks()
    };
    assert!(in_use > 1);

    let (status, reply) = post_json_uri(
        &app,
        V1_CACHE_REBUILD,
        serde_json::json!({"kv": (in_use as u64 - 1) * PAGE}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{reply}");
    let floor = format!("at least {} tokens", in_use as u64 * PAGE);
    assert!(
        reply["error"].as_str().is_some_and(|e| e.contains(&floor)),
        "{reply}"
    );
    assert_eq!(total(&pool), 32);
    assert!(serving(&state));
    drop(held);

    state.maintenance.lock().unwrap().begin_rebuild().unwrap();
    let (status, reply) =
        post_json_uri(&app, V1_CACHE_REBUILD, serde_json::json!({"kv": 64 * PAGE})).await;
    assert_eq!(status, StatusCode::CONFLICT, "{reply}");
    assert_eq!(reply["status"], "busy");
    assert_eq!(total(&pool), 32);
}

/// `/v1/stats` reports what the paged store's prefix tree holds, once a
/// request has published its prompt there: tokens in whole pages,
/// nothing locked after the request ended, all of it evictable. Without
/// paged KV there is no tree and the gauge is `null`, not zero.
#[tokio::test]
async fn stats_reports_what_the_prefix_tree_holds() {
    let app = test_app_with_state(Arc::new(test_state(
        test_model_full_byte_vocab(),
        crate::ResponseCache::new(1000, Duration::from_secs(3600)),
    )));
    let (_, body) = get_json_uri(&app, V1_STATS).await;
    assert!(body["pools"]["prefix_cache"].is_null(), "{}", body["pools"]);

    let model = test_model_full_byte_vocab();
    let crate::Model::Gguf(gguf) = &model else {
        unreachable!("the byte fixture is a GGUF decoder")
    };
    let block = 4;
    let paged = crate::generate::PagedKvConfig {
        store: Arc::new(gguf.decoder.config.new_paged_kv(block, 64)),
        queue_wait: Duration::ZERO,
        radix: Some(Arc::new(Mutex::new(
            crate::policy::radix::SaltedRadix::new(block),
        ))),
        anchor_token: None,
        slide_interval: crate::policy::pool_budget::DEFAULT_SWA_EVICTION_INTERVAL,
    };
    let mut state = test_state(
        model,
        crate::ResponseCache::new(1000, Duration::from_secs(3600)),
    );
    state.paged_kv = Some(paged);
    let app = test_app_with_state(Arc::new(state));

    let (_, body) = get_json_uri(&app, V1_STATS).await;
    let gauge = &body["pools"]["prefix_cache"];
    assert_eq!(gauge["tokens"], 0, "{gauge}");
    assert_eq!(gauge["page_size"], block, "{gauge}");

    let (status, reply) = post_json_uri(
        &app,
        V1_COMPLETIONS,
        serde_json::json!({
            "model": "x",
            "prompt": "a prompt long enough to fill several pages",
            "max_tokens": 2,
            "temperature": 0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{reply}");

    let (_, body) = get_json_uri(&app, V1_STATS).await;
    let gauge = &body["pools"]["prefix_cache"];
    let tokens = gauge["tokens"].as_u64().expect("a number");
    assert!(tokens >= 8, "the prompt was published: {gauge}");
    assert_eq!(tokens % block as u64, 0, "whole pages only: {gauge}");
    assert_eq!(gauge["locked"], 0, "the request has ended: {gauge}");
    assert_eq!(gauge["evictable"], tokens, "{gauge}");
}

/// The refusal that matters: shrinking below what in-flight
/// requests hold cannot be represented, so the engine is left
/// exactly as it was and the caller is told the floor rather than
/// having to find it by being rejected again.
#[test]
fn a_rebuild_below_what_is_held_leaves_the_pool_untouched() {
    use frink_core::cache::{KvBlockPool, KvCache};
    use std::sync::Mutex;

    let pool = Arc::new(Mutex::new(KvBlockPool::new(16, 64)));
    let held = KvCache::with_pool(2, 4, Arc::clone(&pool), 64).expect("blocks");
    let mut p = pool.lock().unwrap();
    let in_use = p.total_blocks() - p.free_blocks();
    assert!(in_use > 0);

    assert_eq!(p.resize(in_use - 1), Err(in_use));
    assert_eq!(p.total_blocks(), 64, "a refused rebuild changes nothing");
    drop(p);
    drop(held);
}

/// The whole reason a KV-side rebuild has to invalidate the prefix
/// cache: a stored prefix names positions in an allocation that has
/// just stopped existing, so handing one back would restore another
/// request's state into this one -- silently, since a KV cache
/// carries no identity of its own.
#[test]
fn a_kv_rebuild_drops_every_stored_prefix_but_keeps_the_counters() {
    use frink_core::cache::KvCache;
    use frink_models::PrefixCache;

    // The cache must really hold the three positions the prefix
    // names; `find_longest_prefix` truncates to the matched length.
    let mut kv = KvCache::new(2, 4);
    for _ in 0..3 {
        kv.push(&[0.0; 8], &[0.0; 8]).expect("unpooled push");
    }
    let mut pc = PrefixCache::new(4);
    pc.store(vec![1, 2, 3], vec![kv], vec![0.0; 8]);
    assert!(pc.find_longest_prefix(&[1, 2, 3, 4]).matched_len > 0);
    let hits_before = pc.stats().hits;

    pc.clear();
    assert_eq!(
        pc.find_longest_prefix(&[1, 2, 3, 4]).matched_len,
        0,
        "nothing may survive a re-split"
    );
    assert_eq!(
        pc.stats().hits,
        hits_before,
        "the counters describe what this process served, which a \
         re-split does not undo"
    );
}

/// A supervisor whose response was lost retries and must receive
/// exactly the same totals, not a second and larger snapshot taken
/// after more tokens were served.
#[test]
fn sealing_a_stop_is_idempotent_for_the_life_of_the_process() {
    let mut gate = MaintenanceGate::serving();
    gate.begin_stop().expect("stop starts");

    let snapshot = |completion| {
        move || crate::policy::maintenance::SealedAccounting {
            model_id: Some("m".to_string()),
            prompt_tokens_total: 10,
            completion_tokens_total: completion,
            uptime_seconds: 5,
            drain_complete: true,
        }
    };
    let first = gate.seal(0, false, snapshot(20)).expect("seals");
    let again = gate.seal(0, false, snapshot(99_999)).expect("seals again");
    assert_eq!(first, again, "a retry must not re-measure");
}

/// The rule the barrier exists for. A request that will not
/// terminate must not be able to talk the server back into serving,
/// and must not be sealed around.
#[test]
fn a_stop_with_work_still_in_flight_seals_nothing_and_reopens_nothing() {
    let mut gate = MaintenanceGate::serving();
    gate.begin_stop().expect("stop starts");

    let err = gate
        .seal(2, true, || unreachable!("must not snapshot"))
        .expect_err("refused");
    assert_eq!(err, StopRefused::AbortBarrierTimedOut(2));
    assert_eq!(gate.state(), MaintenanceState::Stopping);
    assert!(gate.check_admission().is_err());
    assert!(gate.sealed().is_none());
}

/// The convention the three gauges have to share, asserted on
/// `PoolUsage` directly because frink cannot yet produce a
/// non-zero evictable count (a shared KV pool and the prefix cache
/// never coexist -- see `kv_pages`).
///
/// A healthy idle server FILLS its caches with reusable prefixes.
/// Counting those as occupied puts every gauge near 100% while the
/// machine is doing nothing, and admission -- which is happily
/// seating requests into that memory -- then disagrees with the
/// number the operator is reading and sizes a pool that was never
/// full.
#[test]
fn an_evictable_page_is_memory_and_not_occupancy() {
    // 40 free, 20 evictable, 40 really held.
    let usage = PoolUsage::from_available(100, 40 + 20);
    assert_eq!(usage.used, 40, "evictable pages are not occupancy");
    assert_eq!(usage.total, 100);

    // The saturating half: `available` is summed from two
    // independent sources, so a step that double-counts must read
    // as an empty pool rather than wrap to astronomical nonsense.
    assert_eq!(PoolUsage::from_available(10, 12).used, 0);
}

/// The whole ordering, end to end and on a real filesystem: the
/// receipt is on disk before `prepare_stop` answers, because here
/// the ANSWER is the signal -- a supervisor learns the stop is
/// sealed by reading it, so replying first would leave a crash in
/// between with the supervisor believing a receipt exists that does
/// not.
///
/// And a retry reuses the same document. With a fresh id per
/// attempt, a supervisor whose response was lost writes a SECOND
/// receipt for one engine generation, which downstream is a second
/// billing event for work that happened once.
#[test]
fn a_receipt_lands_on_disk_and_a_retry_reuses_the_same_document() {
    use crate::policy::maintenance::SealedAccounting;

    let dir = std::env::temp_dir().join(format!(
        "frink-outbox-test-{}-{}",
        std::process::id(),
        line!()
    ));
    let _ = std::fs::remove_dir_all(&dir);

    let sealed = SealedAccounting {
        model_id: Some("glm-5.2".to_string()),
        prompt_tokens_total: 100,
        completion_tokens_total: 50,
        uptime_seconds: 7,
        drain_complete: true,
    };
    let receipt = Receipt::from_sealed("stable-identity", &sealed);
    let path = dir.join(format!("{}.json", receipt.id));

    let write = |r: &Receipt| -> Result<(), String> {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        if path.exists() {
            return Ok(());
        }
        let tmp = dir.join(format!("{}.json.partial", r.id));
        std::fs::write(&tmp, b"{}").map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())
    };

    crate::policy::outbox::finish_stop(receipt.clone(), write, |_| Ok(())).expect("first stop");
    assert!(path.exists(), "the receipt must be durable");

    // The retry derives the same id, so it addresses the same file
    // and writes no second document.
    let retry = Receipt::from_sealed("stable-identity", &sealed);
    assert_eq!(retry.id, receipt.id);
    crate::policy::outbox::finish_stop(retry, write, |_| Ok(())).expect("retry");
    let count = std::fs::read_dir(&dir).unwrap().count();
    assert_eq!(count, 1, "one generation, one receipt");

    // No `.partial` survives: a reader must never see a
    // half-written receipt, and the moment it is most likely to
    // look is while the engine is stopping.
    assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".partial")));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Unset means no outbox and no persistence step. Inventing a
/// directory to write accounting documents nobody asked for would
/// be worse than not keeping them.
#[test]
fn no_configured_outbox_means_nothing_is_written_and_that_is_not_a_failure() {
    // `outbox_dir` reads the environment, which tests share, so the
    // assertion is on the shape rather than on a mutated variable:
    // an absent path yields no directory at all.
    assert_eq!(
        std::env::var_os("FRINK_ACCOUNTING_OUTBOX").is_none(),
        outbox_dir().is_none()
    );
}
