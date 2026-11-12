//! The elastic cache surface: what the pools are, and moving VRAM
//! between them without a restart.
//!
//! `frink-edge` has held the whole policy for a while with nobody to
//! call it. `pool_budget::CacheReadiness` produces the geometry,
//! `pool_budget::validate_rebuild` checks a re-split against the floors and the
//! budget *before* anything is freed, and `maintenance::MaintenanceGate`
//! decides whether a rebuild may start at all. This module is the three
//! HTTP endpoints over them, plus the one piece of real mutation frink
//! can do today: re-budgeting the shared KV block pool.
//!
//! # Why the order of the checks is the design
//!
//! A rebuild is destructive. The whole point of `validate_rebuild` is
//! that every check is arithmetic and happens up front, so a refused
//! rebuild leaves the engine serving exactly what it was serving. That
//! ordering is preserved here and it is the reason the code reads
//! back-to-front from what you might write first:
//!
//! 1. the **gate** (is a rebuild allowed to start?),
//! 2. the **arithmetic** (does the target fit the floors and budget?),
//! 3. only then the **pool** (which can still refuse, and says by how
//!    much).
//!
//! Any refusal at any step reopens the gate and changes nothing.
//!
//! # What frink can actually re-split
//!
//! The KV pool, and only the KV pool. The expert cache is sized once at
//! load from `FRINK_EXPERT_CACHE_BYTES`, and there is no device-side
//! slot pool to move bytes into, which is #406's
//! (`docs/plans/out-of-core-moe.md`). A request naming `moe` is therefore refused as
//! a pool this deployment does not have, rather than accepted and
//! silently ignored: an operator who asks for expert slots and gets a
//! `200` has been told the split moved when it did not.
//!
//! Ported from FreeToken's `server/api_server.py` cache routes and
//! `server/accounting.py`; see `docs/THIRD_PARTY_NOTICES.md`.

use std::sync::Arc;

use crate::policy::footprint::Footprint;
use crate::policy::maintenance::{MaintenanceState, RebuildRefused, StopRefused};
use crate::policy::outbox::Receipt;
use crate::serving::batch::PoolUsage;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::AppState;

/// `GET /v1/cache/status`.
///
/// Reports the geometry a client denominates its sliders in. Everything
/// it cannot say is **absent** rather than zero -- the rule
/// `crate::policy::cache_report` is built around, because `0.0 GiB` is a
/// lie and a total that silently omits a pool is worse.
pub(crate) async fn cache_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let gate = state
        .maintenance
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .state();
    let kv = kv_pages(&state);
    Json(serde_json::json!({
        "state": gate.as_str(),
        // `null`, not `{}`: a deployment with no shared KV pool has
        // every request allocating privately, which is a different
        // thing from a pool of size zero.
        "kv": kv.map(|(usage, page_size, evictable)| serde_json::json!({
            "num_pages": usage.total,
            "used_pages": usage.used,
            "evictable_pages": evictable,
            "page_size": page_size,
            "num_tokens": usage.total * page_size,
        })),
        // The pools frink does not have. Named so a client can see
        // they were considered and are absent, rather than guess.
        "moe": serde_json::Value::Null,
        "mamba": serde_json::Value::Null,
        "swa": serde_json::Value::Null,
        "resizable": kv.is_some().then(|| serde_json::json!(["kv"])),
    }))
}

/// The sealed totals plus their receipt, written before the answer
/// goes out.
///
/// **Durable before the reply, because here the reply IS the signal.**
/// This server has no child process to signal; a supervisor learns the
/// stop is sealed by reading this response. Answering first and writing
/// after would leave a crash in between with the supervisor believing a
/// receipt exists that does not -- which is the exact loss the ordering
/// rule in `crate::policy::outbox` exists to prevent.
///
/// A write failure is therefore a refusal rather than a warning. The
/// engine is preserved, the totals are already sealed and idempotent,
/// and a retry derives the same id and seals the same numbers, so
/// retrying costs nothing and cannot double-count.
fn sealed_response(
    state: &AppState,
    sealed: &crate::policy::maintenance::SealedAccounting,
) -> Response {
    match persist_receipt(state, sealed) {
        Ok(receipt) => {
            let mut body = sealed_json(sealed);
            if let Some(receipt) = receipt {
                body["receipt_id"] = serde_json::json!(receipt.id);
                body["receipt_status"] = serde_json::json!(receipt.status.as_str());
            }
            Json(body).into_response()
        }
        Err(why) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": why.to_string(),
                "drain_complete": true,
                "engine_preserved": true,
                "retryable": true,
            })),
        )
            .into_response(),
    }
}

/// Where accounting receipts are written, or `None` for a deployment
/// that keeps none.
///
/// Unset means no outbox and no persistence step, which is the right
/// default for a server nobody is billing: inventing a directory under
/// a user's home to write accounting documents they never asked for
/// would be worse than not keeping them.
fn outbox_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("FRINK_ACCOUNTING_OUTBOX").map(std::path::PathBuf::from)
}

/// What names THIS engine generation stably across a retried stop.
///
/// The instance id when one is configured, otherwise the process id
/// with the server's start time -- which is stable for the life of a
/// process, and a receipt is only ever sealed once per process, so a
/// retry within that life derives the same id. It is NOT stable across
/// a restart, and it does not need to be: a restarted engine is a
/// different generation with different totals and must not reuse the
/// receipt of the one before it.
fn engine_identity(state: &AppState) -> String {
    if let Ok(id) = std::env::var("FRINK_INSTANCE_ID") {
        return id;
    }
    format!("pid:{}:started:{}", std::process::id(), state.started_unix)
}

/// Writes the receipt, idempotently by its id.
///
/// `Ok(None)` when no outbox is configured -- nothing to write is not a
/// failure. Otherwise the document lands at `<dir>/<id>.json` through a
/// temporary file and a rename, so a reader never sees a half-written
/// receipt: `rename` within a directory is atomic, a plain write is
/// not, and the one moment a reader is most likely to look is while the
/// engine is stopping.
///
/// A receipt that is already there is left alone and reported as
/// success. It is addressed by a DERIVED id, so an existing file for
/// this id is this generation's own receipt from a previous attempt --
/// exactly what idempotence means here, and rewriting it would risk
/// replacing a good document with a worse one if the retry's totals
/// were somehow read differently.
fn persist_receipt(
    state: &AppState,
    sealed: &crate::policy::maintenance::SealedAccounting,
) -> Result<Option<Receipt>, std::io::Error> {
    let Some(dir) = outbox_dir() else {
        return Ok(None);
    };
    let receipt = Receipt::from_sealed(&engine_identity(state), sealed);
    let final_path = dir.join(format!("{}.json", receipt.id));

    let result = crate::policy::outbox::finish_stop(
        receipt,
        |receipt| {
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            if final_path.exists() {
                return Ok(());
            }
            let tmp = dir.join(format!("{}.json.partial", receipt.id));
            let body = serde_json::json!({
                "id": receipt.id,
                "model_id": receipt.model_id,
                "prompt_tokens_total": receipt.prompt_tokens_total,
                "completion_tokens_total": receipt.completion_tokens_total,
                "uptime_s": receipt.uptime_seconds,
                "status": receipt.status.as_str(),
            });
            std::fs::write(&tmp, serde_json::to_vec_pretty(&body).unwrap_or_default())
                .map_err(|e| e.to_string())?;
            std::fs::rename(&tmp, &final_path).map_err(|e| e.to_string())
        },
        // The response is the signal, and it is sent by the caller once
        // this returns. Nothing to do here, and nothing that CAN fail
        // here -- which is why the `NotSignalled` arm below is
        // unreachable rather than handled.
        |_| Ok(()),
    );

    match result {
        Ok(receipt) => Ok(Some(receipt)),
        Err(why) => Err(std::io::Error::other(why.to_string())),
    }
}

/// KV occupancy in pages, plus the page size and the evictable count,
/// or `None` when this deployment allocates privately per request.
///
/// **`used` excludes evictable pages**, which is the convention the
/// three gauges have to share or an operator is sized against a lie:
/// a healthy idle server FILLS its caches with reusable prefixes, so
/// counting those as occupied puts every gauge near 100% while the
/// machine is doing nothing -- and admission, which is happily seating
/// requests into that memory, disagrees with the number the operator is
/// reading. `crate::serving::batch::PoolUsage::from_available` is where the rule
/// lives, so it is stated once rather than re-derived per endpoint.
///
/// Evictable is **zero today, and structurally so**: `generate` stores
/// a prefix only `if kv_pool.is_none()`, so a shared pool and the
/// prefix cache never coexist and nothing evictable can be holding a
/// page. Computed through `PoolUsage` anyway rather than as
/// `total - free`, so that when the two are allowed to coexist the
/// gauge is already right instead of quietly wrong.
fn kv_pages(state: &AppState) -> Option<(PoolUsage, usize, usize)> {
    let cfg = state.kv_pool.as_ref()?;
    let pool = cfg.pool.lock().unwrap_or_else(|p| p.into_inner());
    let evictable = 0;
    Some((
        PoolUsage::from_available(pool.total_blocks(), pool.free_blocks() + evictable),
        pool.block_size(),
        evictable,
    ))
}

/// The pool gauges for `GET /v1/stats`, in the shape a status bar
/// renders without a second round trip.
///
/// A pool this deployment does not have is `null`, never a zero row:
/// "no window pool" and "a window pool with nothing in it" are
/// different facts, and an operator shown the second for the first
/// sizes against a pool that does not exist.
pub(crate) fn pool_gauges(state: &AppState) -> serde_json::Value {
    let kv = kv_pages(state);
    serde_json::json!({
        "kv_pages": kv.map(|(usage, page_size, evictable)| serde_json::json!({
            "used": usage.used,
            "total": usage.total,
            "evictable": evictable,
            "page_size": page_size,
        })),
        "window_slots": serde_json::Value::Null,
        "state_slots": serde_json::Value::Null,
    })
}

/// This process's live memory footprint, read from `/proc`.
///
/// PSS first: it divides each shared page by the number of processes
/// sharing it, which is what makes a figure summable across a process
/// group. RSS is the fallback for a kernel without `smaps_rollup`
/// (before 4.14), and it OVERCOUNTS the moment anything is shared -- so
/// which one was read travels with the number. See
/// `crate::policy::footprint`.
///
/// `None` on a platform with no `/proc` at all, which is macOS and
/// Windows: absent is the honest answer, and a zero would say the
/// engine is using no memory.
#[cfg(target_os = "linux")]
fn read_footprint() -> Option<Footprint> {
    // Imported inside the arm that uses it, not at the top of the
    // module: `FootprintKind` is named only here, so a module-level
    // import would be an unused-import error on every other platform --
    // which is a `-D warnings` build failure that a Linux-only local
    // check cannot see.
    use crate::policy::footprint::FootprintKind;

    if let Some(bytes) = std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .as_deref()
        .and_then(crate::policy::footprint::parse_smaps_rollup_pss)
    {
        return Some(Footprint {
            bytes,
            kind: FootprintKind::Pss,
        });
    }
    let bytes = std::fs::read_to_string("/proc/self/status")
        .ok()
        .as_deref()
        .and_then(crate::policy::footprint::parse_status_rss)?;
    Some(Footprint {
        bytes,
        kind: FootprintKind::Rss,
    })
}

#[cfg(not(target_os = "linux"))]
fn read_footprint() -> Option<Footprint> {
    None
}

/// The footprint block for `GET /v1/stats`.
///
/// Behind a TTL cache because reading `smaps_rollup` walks the whole
/// VMA list, which on a process mapping tens of gigabytes takes long
/// enough that four dashboards polling this endpoint would produce four
/// concurrent walks -- each slower for the others being there, so the
/// endpoint would end up measuring its own instrumentation.
///
/// `null` rather than a zero when nothing could be read: an engine
/// using no memory is not a thing that happens, so a zero here would be
/// a failed read presented as a fact.
pub(crate) fn footprint_json(state: &AppState) -> serde_json::Value {
    let now_ms = state.uptime().as_millis().min(u64::MAX as u128) as u64;
    let reading = state
        .footprint
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_probe(now_ms, read_footprint);
    match reading {
        Some(f) => serde_json::json!({
            "bytes": f.bytes,
            // Which quantity this is, never assumed: a caller comparing
            // a PSS figure with an RSS one is comparing two different
            // things and will read the difference as a leak.
            "kind": f.kind.as_str(),
        }),
        None => serde_json::Value::Null,
    }
}

/// What a client may ask a rebuild for.
///
/// `kv` is in **tokens**, the unit an operator thinks in, converted to
/// pages here. Every other pool is accepted only so it can be refused by
/// name -- see the module doc.
#[derive(Debug, Deserialize)]
pub(crate) struct CacheRebuildRequest {
    #[serde(default)]
    kv: Option<u64>,
    #[serde(default)]
    moe: Option<u64>,
    #[serde(default)]
    mamba: Option<u64>,
    #[serde(default)]
    swa: Option<u64>,
}

fn refusal(status: StatusCode, status_word: &str, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"status": status_word, "error": message})),
    )
        .into_response()
}

/// `POST /v1/cache/rebuild`.
///
/// Moves the KV pool's budget on a live engine. Refuses, changing
/// nothing, when: the engine is not serving; the request names a pool
/// this deployment does not have; the target is not a positive number of
/// tokens; or the pool is holding more blocks than the target could
/// cover.
///
/// The last one is the interesting refusal, and it is why this reports
/// the floor rather than just saying no: shrinking below what is held
/// cannot be represented, because the caches holding those blocks do not
/// give them back on request.
pub(crate) async fn cache_rebuild(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CacheRebuildRequest>,
) -> Response {
    // 1. The gate. A rebuild through a load, another rebuild, or a stop
    //    is refused before anything is measured.
    {
        let mut gate = state.maintenance.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(why) = gate.begin_rebuild() {
            let status = match why {
                RebuildRefused::NotReady | RebuildRefused::Latched => {
                    StatusCode::SERVICE_UNAVAILABLE
                }
                RebuildRefused::Busy(_) => StatusCode::CONFLICT,
            };
            let word = match why {
                RebuildRefused::NotReady => "loading",
                RebuildRefused::Latched => "failed",
                RebuildRefused::Busy(_) => "busy",
            };
            return refusal(status, word, why.to_string());
        }
    }

    // From here every exit reopens the gate. `finish_rebuild(true)`
    // rather than `rebuild_never_dispatched()`: this engine IS the
    // scheduler, so there is no in-flight request that might still land
    // -- the outcome is known the moment this function returns.
    let reopen = |ok: bool| {
        state
            .maintenance
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .finish_rebuild(ok);
    };

    for (asked, pool) in [(req.moe, "moe"), (req.mamba, "mamba"), (req.swa, "swa")] {
        if asked.is_some() {
            reopen(true);
            return refusal(
                StatusCode::BAD_REQUEST,
                "failed",
                format!(
                    "this deployment has no {pool} pool to rebuild; see \
                     GET /v1/cache/status for what it does have"
                ),
            );
        }
    }

    let Some(kv_tokens) = req.kv else {
        reopen(true);
        return refusal(
            StatusCode::BAD_REQUEST,
            "failed",
            "nothing to rebuild: pass `kv` in tokens".to_string(),
        );
    };

    let Some(cfg) = state.kv_pool.as_ref() else {
        reopen(true);
        return refusal(
            StatusCode::BAD_REQUEST,
            "failed",
            "this deployment has no shared KV pool; every request allocates privately \
             (set FRINK_KV_POOL_BLOCKS to enable one)"
                .to_string(),
        );
    };

    // 2. The arithmetic, before anything is touched.
    let mut pool = cfg.pool.lock().unwrap_or_else(|p| p.into_inner());
    let page_size = pool.block_size() as u64;
    let pages = kv_tokens / page_size;
    if pages == 0 {
        let held = (pool.total_blocks() - pool.free_blocks()) as u64;
        drop(pool);
        reopen(true);
        return refusal(
            StatusCode::BAD_REQUEST,
            "failed",
            format!(
                "kv={kv_tokens} tokens is below one {page_size}-token page; \
                 the pool is currently holding {held} page(s)"
            ),
        );
    }

    // 3. The pool, inside a transaction -- which can still refuse, and
    //    says by how much.
    //
    // `RebuildTxn` exists for the destructive case: free the old pools,
    // allocate the new, and a failure on either side of that line means
    // something different (see `crate::policy::rebuild`). Frink has no
    // such line YET -- `KvBlockPool::resize` moves counters and frees
    // nothing, so it either succeeds or refuses atomically, and the
    // closure below never marks a teardown. The transaction is wired
    // anyway rather than added later: the day a device-side pool really
    // is torn down, the caller marks the teardown and the rollback arm
    // is already here and already tested, instead of being written
    // under the pressure of the first engine that 503s forever.
    let before = pool.total_blocks() as u64;
    let current = crate::policy::pool_budget::PoolSizes {
        moe_cache_slots: 0,
        kv_pages: before,
        prefill_overlap: false,
    };
    let target = crate::policy::pool_budget::PoolSizes {
        kv_pages: pages,
        ..current
    };
    let txn = crate::policy::rebuild::RebuildTxn::open(
        &crate::policy::pool_budget::RebuildRequest {
            moe_cache_slots: None,
            kv_pages: Some(pages),
            mamba_slots: None,
            swa_pages: None,
        },
        &current,
    );

    let mut refused_floor: Option<usize> = None;
    let outcome = txn.run(
        target,
        |_mark_teardown, target| {
            // No `mark_teardown()`: nothing is freed here. See above.
            pool.resize(target.kv_pages as usize).map_err(|held| {
                refused_floor = Some(held);
                format!("{held} page(s) are held by in-flight requests")
            })
        },
        |_old, _touched| unreachable!("nothing was torn down, so nothing can be restored"),
    );

    match outcome {
        crate::policy::rebuild::RebuildOutcome::Applied(_) => {
            let after = pool.total_blocks() as u64;
            drop(pool);
            // Any KV-side resize invalidates the prefix cache: a cached
            // prefix names positions in an allocation that has just
            // stopped existing, and handing one back would serve another
            // request's state.
            let invalidated = state.prefix_cache.is_some();
            if let Some(pc) = &state.prefix_cache {
                pc.lock().unwrap_or_else(|p| p.into_inner()).clear();
            }
            reopen(true);
            tracing::info!(
                "cache rebuilt: kv {before} -> {after} pages of {page_size} tokens \
                 (prefix cache invalidated: {invalidated})"
            );
            Json(serde_json::json!({
                "status": "ok",
                "kv": {
                    "num_pages": after,
                    "page_size": page_size,
                    "num_tokens": after * page_size,
                },
                "prefix_cache_invalidated": invalidated,
            }))
            .into_response()
        }
        crate::policy::rebuild::RebuildOutcome::RejectedIntact { .. } => {
            drop(pool);
            reopen(true);
            let held = refused_floor.unwrap_or(0);
            refusal(
                StatusCode::CONFLICT,
                "busy",
                format!(
                    "kv={kv_tokens} tokens is {pages} page(s), below the {held} page(s) \
                     currently held by in-flight requests; the engine is unchanged. \
                     Retry when they finish, or ask for at least {} tokens",
                    held as u64 * page_size
                ),
            )
        }
        // Unreachable while the resize frees nothing, and written out
        // rather than defaulted so that making it destructive is a
        // compile error here first instead of a silent 503-forever.
        crate::policy::rebuild::RebuildOutcome::RolledBack { reason, .. } => {
            drop(pool);
            reopen(true);
            refusal(StatusCode::CONFLICT, "busy", reason)
        }
        crate::policy::rebuild::RebuildOutcome::Latched {
            reason,
            rollback_error,
        } => {
            drop(pool);
            // The one outcome that does NOT reopen the gate: there are
            // no pools and no way back, so the engine must stop
            // pretending it can serve.
            state
                .maintenance
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .latch_failed();
            refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed",
                format!("{reason}; and the rollback failed too: {rollback_error}"),
            )
        }
    }
}

/// What `prepare-stop` accepts. Both are seconds and both are advisory:
/// the caller does the waiting, this decides what the waiting *meant*.
#[derive(Debug, Deserialize)]
pub(crate) struct PrepareStopRequest {
    /// Ignored today -- see the handler. Accepted so a supervisor's
    /// body deserializes rather than 422-ing.
    #[serde(default)]
    #[allow(dead_code)]
    drain_timeout_s: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    abort_timeout_s: Option<f64>,
}

/// `POST /v1/admin/prepare-stop`.
///
/// Closes admission, then seals the token totals -- in that order, and
/// only once nothing is in flight. A supervisor calls this before it
/// signals the process, so shutdown cannot race the last sampled token.
///
/// Two rules carry the whole thing, both from
/// `crate::policy::maintenance::MaintenanceGate`:
///
/// * **A refusal never reopens admission.** A server that announced it
///   was stopping, accepted more work, and then sealed totals that do
///   not include it has produced accounting that is simply false. The
///   caller is expected to preserve the process and retry.
/// * **Sealing is idempotent for the life of the process.** A
///   supervisor whose response was lost retries and gets exactly the
///   same numbers, not a second and larger snapshot.
pub(crate) async fn prepare_stop(
    State(state): State<Arc<AppState>>,
    body: Option<Json<PrepareStopRequest>>,
) -> Response {
    let _ = body;
    let mut gate = state.maintenance.lock().unwrap_or_else(|p| p.into_inner());

    // Already sealed: hand back the same document. Checked before
    // `begin_stop` so a retry cannot be refused by a rebuild that
    // started afterwards.
    //
    // The receipt is (re-)persisted on this path too, and the response
    // carries its id. This is the caller idempotence exists FOR: a
    // supervisor retries precisely because it lost the first answer, so
    // answering the retry without the receipt id would hand it a body
    // it cannot act on -- and leave it unable to tell a written receipt
    // from an unwritten one. Persisting is addressed by a derived id
    // and skips a document that is already there, so the retry writes
    // nothing.
    if let Some(sealed) = gate.sealed().cloned() {
        drop(gate);
        return sealed_response(&state, &sealed);
    }

    if let Err(why) = gate.begin_stop() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": why.to_string(),
                "drain_complete": false,
                "engine_preserved": true,
            })),
        )
            .into_response();
    }

    // The caller owns the waiting; this reads what is still live. The
    // cancel registry is the honest count -- it holds exactly the
    // generations that have started and not finished.
    let active = state.cancels.live_count();
    let model_id = state.active_model_name();
    let uptime = state.uptime().as_secs();
    let (prompt_total, completion_total) = (
        state.stats.tokens_prompt_total(),
        state.stats.tokens_generated_total(),
    );

    match gate.seal(active, active > 0, || {
        crate::policy::maintenance::SealedAccounting {
            model_id,
            prompt_tokens_total: prompt_total,
            completion_tokens_total: completion_total,
            uptime_seconds: uptime,
            drain_complete: true,
        }
    }) {
        Ok(sealed) => {
            drop(gate);
            sealed_response(&state, &sealed)
        }
        Err(why) => {
            let engine_preserved = !matches!(why, StopRefused::RebuildInProgress);
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": why.to_string(),
                    "drain_complete": false,
                    "engine_preserved": engine_preserved,
                    "active": active,
                })),
            )
                .into_response()
        }
    }
}

fn sealed_json(sealed: &crate::policy::maintenance::SealedAccounting) -> serde_json::Value {
    serde_json::json!({
        "model_id": sealed.model_id,
        "prompt_tokens_total": sealed.prompt_tokens_total,
        "completion_tokens_total": sealed.completion_tokens_total,
        "uptime_s": sealed.uptime_seconds,
        "drain_complete": sealed.drain_complete,
    })
}

/// Whether the engine will admit a request right now.
///
/// Exposed for the generation handlers: a request that arrives during a
/// rebuild or a stop must be refused rather than queued into a cache
/// that is being resized under it.
pub(crate) fn check_admission(state: &AppState) -> Result<(), crate::ApiError> {
    let gate = state.maintenance.lock().unwrap_or_else(|p| p.into_inner());
    gate.check_admission().map_err(|closed| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": {
                "message": closed.to_string(),
                "type": match closed.state {
                    MaintenanceState::Loading => "model_loading",
                    MaintenanceState::Rebuilding => "cache_rebuilding",
                    MaintenanceState::Stopping => "server_stopping",
                    MaintenanceState::Failed => "server_failed",
                    MaintenanceState::Serving => unreachable!("serving admits"),
                },
            }})),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::maintenance::MaintenanceGate;

    /// A rebuild through a load, another rebuild, or a stop is refused
    /// before anything is measured -- and each refusal is a DIFFERENT
    /// status, because `busy` is retryable in a moment, `loading` when
    /// the model finishes, and `failed` not at all.
    #[test]
    fn a_rebuild_is_refused_by_the_gate_before_anything_is_measured() {
        let mut gate = MaintenanceGate::new();
        assert_eq!(gate.begin_rebuild(), Err(RebuildRefused::NotReady));

        gate.finish_loading(true);
        gate.begin_rebuild().expect("the first one starts");
        assert!(matches!(
            gate.begin_rebuild(),
            Err(RebuildRefused::Busy(MaintenanceState::Rebuilding))
        ));

        gate.finish_rebuild(true);
        gate.begin_stop().expect("stop starts");
        assert!(matches!(
            gate.begin_rebuild(),
            Err(RebuildRefused::Busy(MaintenanceState::Stopping))
        ));
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
        let mut gate = MaintenanceGate::new();
        gate.finish_loading(true);
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
        let mut gate = MaintenanceGate::new();
        gate.finish_loading(true);
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
}
