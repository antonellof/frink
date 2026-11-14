//! The elastic cache surface: what the pools are, and moving VRAM
//! between them without a restart.
//!
//! `maintenance::MaintenanceGate` decides whether a rebuild may start at
//! all, `pool_budget::validate_rebuild` checks a re-split against the
//! floors and the byte budget *before* anything is touched, and
//! `rebuild::RebuildTxn` runs the resize. This module is the three HTTP
//! endpoints over them, plus the one piece of real mutation frink can do
//! today: re-budgeting the shared KV block pool.
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
//! llama.cpp's server has no counterpart: it sizes its KV once at
//! startup (`-c`, `--kv-unified`) and never re-splits it.
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
use crate::policy::pool_budget::{
    validate_rebuild, ByteBudget, PoolFloors, PoolSizes, RebuildRejected, RebuildRequest,
};
use crate::policy::rebuild::{RebuildOutcome, RebuildTxn};
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
        "prefix_cache": prefix_tree_gauge(state),
    })
}

/// What the paged store's prefix tree holds, in tokens, across every
/// cache salt.
///
/// `locked` is what live requests are reading and cannot be reclaimed;
/// `evictable` is what admission may take back from the tree when the
/// store runs short (`generate::acquire_paged_caches`). `null` without
/// paged KV (`FRINK_PAGED_KV_BLOCKS`), which is the only store with a
/// tree, for the same reason as the pools above: no tree and an empty
/// tree are different facts.
fn prefix_tree_gauge(state: &AppState) -> serde_json::Value {
    let Some(radix) = state.paged_kv.as_ref().and_then(|p| p.radix.as_ref()) else {
        return serde_json::Value::Null;
    };
    let tree = radix.lock().unwrap_or_else(|p| p.into_inner());
    serde_json::json!({
        "tokens": tree.total_size(),
        "locked": tree.protected_size(),
        "evictable": tree.evictable_size(),
        "page_size": tree.page_size(),
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

/// What `validate_rebuild`'s refusal means to a client.
///
/// Every one is a `400`: the request cannot succeed by retrying it.
/// The pool being held by in-flight requests is the retryable refusal,
/// and that one comes from the pool itself, after this.
fn rejected(why: &RebuildRejected, kv_tokens: Option<u64>, page_size: u64) -> Response {
    let message = match why {
        RebuildRejected::NoSuchPool("kv") => "this deployment has no shared KV pool; every \
             request allocates privately (set FRINK_KV_POOL_BLOCKS or FRINK_KV_BYTE_BUDGET \
             to enable one)"
            .to_string(),
        RebuildRejected::NoSuchPool(_) => {
            format!("{why}; see GET /v1/cache/status for what it does have")
        }
        RebuildRejected::BelowFloor { pool: "kv", .. } => format!(
            "kv={} tokens is below one {page_size}-token page",
            kv_tokens.unwrap_or(0)
        ),
        RebuildRejected::BelowFloor { .. } => why.to_string(),
        RebuildRejected::TooLarge(too) => format!(
            "kv={} tokens is {} page(s), which needs {} bytes; FRINK_KV_BYTE_BUDGET is {} \
             bytes. The engine is unchanged",
            kv_tokens.unwrap_or(0),
            too.sizes.kv_pages,
            too.needed_bytes,
            too.budget_bytes
        ),
    };
    refusal(StatusCode::BAD_REQUEST, "failed", message)
}

/// `POST /v1/cache/rebuild`.
///
/// Moves the KV pool's budget on a live engine. Refuses, changing
/// nothing, when: the engine is not serving; the request names a pool
/// this deployment does not have, or none at all; the target is under
/// one page; the target would cost more than the pool's byte budget
/// (`FRINK_KV_BYTE_BUDGET`, when the pool was sized from one); or the
/// pool is holding more blocks than the target could cover.
///
/// The last one is the interesting refusal, and it is why this reports
/// the floor rather than just saying no: shrinking below what is held
/// cannot be represented, because the caches holding those blocks do not
/// give them back on request.
pub(crate) async fn cache_rebuild(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CacheRebuildRequest>,
) -> Response {
    // 1. The gate. A rebuild through another rebuild, a stop, or a
    //    latched engine is refused before anything is measured.
    {
        let mut gate = state.maintenance.lock().unwrap_or_else(|p| p.into_inner());
        if let Err(why) = gate.begin_rebuild() {
            let (status, word) = match why {
                RebuildRefused::Latched => (StatusCode::SERVICE_UNAVAILABLE, "failed"),
                RebuildRefused::Busy(_) => (StatusCode::CONFLICT, "busy"),
            };
            return refusal(status, word, why.to_string());
        }
    }

    // From here every exit but a latch reopens the gate: the resize runs
    // here, in-process, so its outcome is known the moment this returns.
    let reopen = || {
        state
            .maintenance
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .finish_rebuild();
    };

    // 2. The arithmetic, before anything is touched. The pool's lock is
    //    taken here and held through the resize, so the size validated
    //    is the size resized.
    let pool = state
        .kv_pool
        .as_ref()
        .map(|cfg| cfg.pool.lock().unwrap_or_else(|p| p.into_inner()));
    let page_size = pool.as_ref().map_or(1, |p| p.block_size() as u64);
    let request = RebuildRequest {
        moe_cache_slots: req.moe,
        kv_pages: req.kv.map(|tokens| tokens / page_size),
        mamba_slots: req.mamba,
        swa_pages: req.swa,
    };
    if request.is_empty() {
        drop(pool);
        reopen();
        return refusal(
            StatusCode::BAD_REQUEST,
            "failed",
            "nothing to rebuild: pass `kv` in tokens".to_string(),
        );
    }
    let current = PoolSizes {
        moe_cache_slots: 0,
        kv_pages: pool.as_ref().map_or(0, |p| p.total_blocks() as u64),
        prefill_overlap: false,
    };
    // Only the KV pool is resizable here (see the module doc), and one
    // page is the least a pool that admits anything can hold. What
    // in-flight requests hold is the pool's own refusal, below, because
    // it is retryable and these are not.
    let floors = PoolFloors {
        kv_pages: pool.as_ref().map(|_| 1),
        ..PoolFloors::default()
    };
    let budget = state
        .kv_pool
        .as_ref()
        .and_then(|cfg| cfg.byte_budget)
        .map(|b| ByteBudget {
            budget_bytes: i64::try_from(b.budget_bytes).unwrap_or(i64::MAX),
            bytes_per_expert: 0,
            bytes_per_page: b.bytes_per_block,
        });
    let target = match validate_rebuild(&request, &current, &floors, budget.as_ref()) {
        Ok(target) => target,
        Err(why) => {
            drop(pool);
            reopen();
            return rejected(&why, req.kv, page_size);
        }
    };
    let mut pool = pool.expect("validate_rebuild refuses a kv target without a kv pool");

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
    let before = current.kv_pages;
    let txn = RebuildTxn::open(&request, &current);

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
        RebuildOutcome::Applied(_) => {
            let after = pool.total_blocks() as u64;
            drop(pool);
            // A KV-side resize invalidates the prefix cache: a cached
            // prefix names positions in an allocation that has just
            // stopped existing, and handing one back would serve another
            // request's state.
            let invalidated = request.invalidates_prefix_cache() && state.prefix_cache.is_some();
            if invalidated {
                if let Some(pc) = &state.prefix_cache {
                    pc.lock().unwrap_or_else(|p| p.into_inner()).clear();
                }
            }
            reopen();
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
        RebuildOutcome::RejectedIntact { .. } => {
            drop(pool);
            reopen();
            let held = refused_floor.unwrap_or(0);
            refusal(
                StatusCode::CONFLICT,
                "busy",
                format!(
                    "kv={} tokens is {} page(s), below the {held} page(s) \
                     currently held by in-flight requests; the engine is unchanged. \
                     Retry when they finish, or ask for at least {} tokens",
                    req.kv.unwrap_or(0),
                    target.kv_pages,
                    held as u64 * page_size
                ),
            )
        }
        // Unreachable while the resize frees nothing, and written out
        // rather than defaulted so that making it destructive is a
        // compile error here first instead of a silent 503-forever.
        RebuildOutcome::RolledBack { reason, .. } => {
            drop(pool);
            reopen();
            refusal(StatusCode::CONFLICT, "busy", reason)
        }
        RebuildOutcome::Latched {
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
mod tests;
