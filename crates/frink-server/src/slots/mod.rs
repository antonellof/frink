//! `POST /slots/{id_slot}?action=save|restore`: persist a prompt's KV
//! state to disk and read it back, so a long system prompt survives a
//! restart.
//!
//! # llama.cpp's shape, and where frink's differs
//!
//! The route, the query parameter, the gate and the response fields are
//! llama.cpp's, so a client written against `llama-server` works
//! unchanged: `tools/server/server.cpp:273` registers
//! `POST /slots/:id_slot`, `server-context.cpp:4536-4567` dispatches on
//! `action` and refuses everything when `--slot-save-path` is unset, and
//! `server-task.cpp:1570-1592` is the `n_saved`/`n_written` /
//! `n_restored`/`n_read` response body reproduced here.
//!
//! Two things genuinely differ, and both are said out loud rather than
//! emulated:
//!
//! - **`id_slot` is bookkeeping.** llama.cpp has N fixed slots, each
//!   owning a KV region for the whole process lifetime, and `-np` sets
//!   N. frink builds a request's KV per request and shares prefixes
//!   through ONE `PrefixCache`, so there is no per-slot region for an
//!   id to select. The id is validated and echoed so llama.cpp clients
//!   and their logs keep working; what it does not do is pick between
//!   independent caches, because there are not any.
//! - **`save` names its prompt.** llama.cpp saves whatever the slot was
//!   last serving. Nothing here is "last serving" anything, so the save
//!   body carries the `prompt` to prefill and store. That is also the
//!   operation an operator actually wants: warm a specific system
//!   prompt, not whichever request happened to land on slot 3.
//!
//! `action=erase` is REFUSED rather than approximated -- see
//! [`erase_refusal`].
//!
//! # Where the state goes
//!
//! Into the server's existing [`PrefixCache`], which the private decode
//! path already consults on every request. That is the whole point of
//! not inventing a second store: a restored slot is found by the same
//! lookup that finds a warm prefix, so there is no second code path to
//! keep in agreement with the first, and no way for a slot to be
//! present-but-unused. It also means slots require
//! `FRINK_PREFIX_CACHE_ENTRIES`, and say so when it is unset.
//!
//! # Safety of a restore
//!
//! A slot file is raw attention state. Restoring it under different
//! weights produces confident wrong tokens, not an error, so identity
//! is checked before anything is stored: see [`identity`] for what is
//! hashed and why the obvious keys (path, size, decoder config) are all
//! insufficient.

mod file;
mod identity;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;

use frink_core::cache::KvCache;
use frink_models::Decoder;

use crate::{ActiveModel, ApiError, AppState};

use file::SlotPayload;
use identity::SlotIdentity;

/// Directory slot files live in: llama.cpp's `--slot-save-path`.
///
/// Read per request rather than cached, because the refusal when it is
/// unset is the documented behaviour and a cached `None` would be
/// indistinguishable from a cached path in the error.
fn slot_dir() -> Option<PathBuf> {
    std::env::var("FRINK_SLOT_SAVE_PATH")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

#[derive(Debug, Deserialize)]
pub(crate) struct SlotQuery {
    #[serde(default)]
    action: String,
}

#[derive(Debug, Deserialize)]
struct SlotBody {
    filename: Option<String>,
    /// The prompt whose KV this slot holds. Required for `save`; see
    /// the module note for why llama.cpp does not need one.
    prompt: Option<String>,
}

/// The crate's small error pair, not a built `Response`: a `Response`
/// carries an `axum::body::Body` and makes every `Result` in this module
/// enormous on its Err side, which clippy's `result_large_err` is right
/// about.
fn error(status: StatusCode, message: String, kind: &str) -> ApiError {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": kind, "code": status.as_u16()}
        })),
    )
}

/// Why `action=erase` is a refusal and not a no-op.
///
/// llama.cpp's erase drops one slot's token cache
/// (`server-context.cpp:2671-2690`). frink's slots share one
/// `PrefixCache`, which has whole-cache `clear()` and no per-entry
/// eviction, so the two things this could do are both wrong: clearing
/// everything would throw away every other slot on an erase of one, and
/// doing nothing while answering 200 would report an eviction that did
/// not happen. A refusal naming the reason is the only answer that is
/// true, and this repo counts a refusal as coverage.
fn erase_refusal() -> ApiError {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "action=erase: frink holds slot KV in one shared prefix cache, which has no per-entry \
         eviction, so erasing one slot would mean clearing all of them. Restart the server, or \
         delete the slot file, to stop a slot being restorable"
            .to_string(),
        "unsupported_feature",
    )
}

/// llama.cpp's `fs_validate_filename` in the strict direction: a slot
/// name is a bare file name, and anything that could leave the
/// configured directory is rejected rather than sanitized.
///
/// Whitelist, not blacklist. A blacklist of `..` and `/` has to be
/// right about every encoding of every separator on every platform; a
/// whitelist of `[A-Za-z0-9._-]` cannot name a parent directory at all.
fn validate_filename(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("filename is empty".to_string());
    }
    if name.len() > 255 {
        return Err(format!(
            "filename is {} bytes; a slot name may be at most 255",
            name.len()
        ));
    }
    if name.starts_with('.') {
        return Err(
            "filename starts with '.'; a slot name may not be a dotfile or a \
                    relative-path component"
                .to_string(),
        );
    }
    if let Some(bad) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '.' || *c == '_' || *c == '-'))
    {
        return Err(format!(
            "filename contains {bad:?}; a slot name may use only ASCII letters, digits, '.', \
             '_' and '-'"
        ));
    }
    Ok(())
}

/// What this server is serving, as a slot identity, or a refusal
/// naming what is missing.
fn serving_identity(active: &ActiveModel) -> Result<(SlotIdentity, Arc<Decoder>), ApiError> {
    let model = active.generative()?;
    let Some(decoder) = model.gguf_decoder() else {
        return Err(error(
            StatusCode::NOT_IMPLEMENTED,
            "slots are implemented for the generic GGUF decoder only; this server is serving a \
             dedicated engine whose KV layout this format does not describe"
                .to_string(),
            "unsupported_feature",
        ));
    };
    let Some(path) = active.checkpoint_path.as_ref() else {
        return Err(error(
            StatusCode::NOT_IMPLEMENTED,
            identity::FingerprintError::NoCheckpoint.to_string(),
            "unsupported_feature",
        ));
    };
    // The slot file header carries ONE `n_kv_heads` for every layer
    // (`file::encode` writes the first layer's), so a model whose layers
    // cache different widths (`frink_models::layer_shapes`, deci /
    // openelm) has no faithful encoding in this format. Refuse rather
    // than write a header that describes only layer 0.
    if !decoder.config.layer_shapes.is_uniform() {
        return Err(error(
            StatusCode::NOT_IMPLEMENTED,
            "slots are implemented for models whose layers all cache the same KV width; this \
             model's layers differ (per-layer head counts), and the slot file header holds one \
             geometry"
                .to_string(),
            "unsupported_feature",
        ));
    }
    // The same header holds ONE `head_dim` for K and V; a model whose V
    // head width differs (MiMo-V2, `frink_models::kv_head_dims`) has no
    // faithful encoding either.
    if decoder.config.kv_head_dims_split() {
        return Err(error(
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "slots are implemented for models whose K and V heads share one width; this \
                 model's are {} and {}, and the slot file header holds one head_dim",
                decoder.config.head_dim,
                decoder.config.v_head_dim()
            ),
            "unsupported_feature",
        ));
    }
    let fingerprint = identity::fingerprint_gguf(path).map_err(|e| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            e.to_string(),
            "checkpoint_unreadable",
        )
    })?;
    let config = &decoder.config;
    Ok((
        SlotIdentity {
            model_name: config.name.to_string(),
            n_layers: decoder.layers.len(),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            dtype: config.kv_dtype,
            fingerprint,
        },
        Arc::clone(decoder),
    ))
}

/// `POST /slots/{id_slot}?action=save|restore|erase`.
pub(crate) async fn post_slot(
    State(state): State<Arc<AppState>>,
    AxumPath(id_slot): AxumPath<String>,
    Query(query): Query<SlotQuery>,
    body: String,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(dir) = slot_dir() else {
        return Err(error(
            StatusCode::NOT_IMPLEMENTED,
            "This server does not support slots action. Start it with `--slot-save-path`"
                .to_string(),
            "unsupported_feature",
        ));
    };
    let Ok(id_slot) = id_slot.parse::<u32>() else {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "Invalid slot ID".to_string(),
            "invalid_request_error",
        ));
    };
    match query.action.as_str() {
        "save" => save(state, dir, id_slot, body).await,
        "restore" => restore(state, dir, id_slot, body).await,
        "erase" => Err(erase_refusal()),
        other => Err(error(
            StatusCode::BAD_REQUEST,
            format!("Invalid action {other:?}: expected save, restore or erase"),
            "invalid_request_error",
        )),
    }
}

fn parse_body(body: &str) -> Result<SlotBody, ApiError> {
    serde_json::from_str::<SlotBody>(body).map_err(|e| {
        error(
            StatusCode::BAD_REQUEST,
            format!("slot request body is not JSON: {e}"),
            "invalid_request_error",
        )
    })
}

fn slot_path(dir: &std::path::Path, body: &SlotBody) -> Result<PathBuf, ApiError> {
    let Some(filename) = body.filename.as_deref() else {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "slot request needs a \"filename\"".to_string(),
            "invalid_request_error",
        ));
    };
    validate_filename(filename).map_err(|why| {
        error(
            StatusCode::BAD_REQUEST,
            format!("Invalid filename: {why}"),
            "invalid_request_error",
        )
    })?;
    Ok(dir.join(filename))
}

/// The prefix cache, or the refusal naming the variable that turns it
/// on. Slots without it would write files nothing could ever use.
fn require_prefix_cache(
    state: &AppState,
) -> Result<Arc<std::sync::Mutex<frink_models::PrefixCache>>, ApiError> {
    state.prefix_cache.clone().ok_or_else(|| {
        error(
            StatusCode::NOT_IMPLEMENTED,
            "slots store and restore into this server's prefix cache, which is off: set \
             FRINK_PREFIX_CACHE_ENTRIES to a positive number of entries"
                .to_string(),
            "unsupported_feature",
        )
    })
}

async fn save(
    state: Arc<AppState>,
    dir: PathBuf,
    id_slot: u32,
    body: String,
) -> Result<Json<serde_json::Value>, ApiError> {
    let started = Instant::now();
    let body = parse_body(&body)?;
    let path = slot_path(&dir, &body)?;
    let Some(prompt) = body.prompt.as_deref() else {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "slot save needs a \"prompt\": frink has no per-slot KV region holding one \
             already, so the save names the prefix to warm"
                .to_string(),
            "invalid_request_error",
        ));
    };
    let active = state.require_active()?;
    let cache = require_prefix_cache(&state)?;
    let (identity, decoder) = serving_identity(&active)?;
    let mut tokens = active.encode_any(prompt, frink_models::tokenizer::SpecialTokens::Parse);
    frink_models::tokenizer::prepend_bos(
        &mut tokens,
        active.generative_opt().and_then(|m| m.bos_id()),
    );
    if tokens.is_empty() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "slot save prompt encodes to no tokens; there is no KV state to save".to_string(),
            "invalid_request_error",
        ));
    }

    let n_tokens = tokens.len();
    let write = tokio::task::spawn_blocking(move || {
        let payload = prefill_slot(&decoder, tokens);
        let bytes = file::encode(&identity, &payload);
        let written = publish(&path, &bytes)?;
        cache.lock().unwrap_or_else(|p| p.into_inner()).store(
            payload.tokens,
            payload.layers,
            payload.pending_logits,
        );
        Ok::<u64, std::io::Error>(written)
    })
    .await;

    match write {
        Ok(Ok(n_written)) => Ok(Json(serde_json::json!({
            "id_slot": id_slot,
            "filename": body.filename,
            "n_saved": n_tokens,
            "n_written": n_written,
            "timings": {"save_ms": started.elapsed().as_secs_f64() * 1000.0},
        }))),
        Ok(Err(e)) => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("writing the slot file: {e}"),
            "slot_write_failed",
        )),
        Err(e) => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("the slot save task did not finish: {e}"),
            "slot_write_failed",
        )),
    }
}

async fn restore(
    state: Arc<AppState>,
    dir: PathBuf,
    id_slot: u32,
    body: String,
) -> Result<Json<serde_json::Value>, ApiError> {
    let started = Instant::now();
    let body = parse_body(&body)?;
    let path = slot_path(&dir, &body)?;
    let active = state.require_active()?;
    let cache = require_prefix_cache(&state)?;
    let (identity, _decoder) = serving_identity(&active)?;
    let vocab = active.generative_opt().and_then(|m| m.vocab_size());

    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(error(
                StatusCode::NOT_FOUND,
                format!("no slot file {}", path.display()),
                "not_found",
            ))
        }
        Err(e) => {
            return Err(error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("reading the slot file: {e}"),
                "slot_read_failed",
            ))
        }
    };
    let n_read = bytes.len();
    let unverified = match file::decode(&bytes) {
        Ok(slot) => slot,
        Err(e) => {
            return Err(error(
                StatusCode::BAD_REQUEST,
                format!("{}: {e}", path.display()),
                "invalid_slot_file",
            ))
        }
    };
    tracing::debug!(
        "restoring {}: saved under model {} with {} layers, checkpoint {}",
        path.display(),
        unverified.identity().model_name,
        unverified.identity().n_layers,
        unverified.identity().fingerprint,
    );
    let payload = match unverified.verify(&identity) {
        Ok(payload) => payload,
        Err(mismatch) => {
            return Err(error(
                StatusCode::BAD_REQUEST,
                format!(
                    "refusing to restore {}: {mismatch}. Restoring attention state computed by \
                     other weights does not fail, it answers wrongly",
                    path.display()
                ),
                "slot_model_mismatch",
            ))
        }
    };
    // The one thing the file's own arithmetic cannot check: a token id
    // is only meaningful against a vocabulary, and this file's ids were
    // written by a tokenizer, not measured from the payload.
    if let Some(vocab) = vocab {
        if let Some(&bad) = payload.tokens.iter().find(|&&t| t >= vocab) {
            return Err(error(
                StatusCode::BAD_REQUEST,
                format!(
                    "refusing to restore {}: it names token id {bad}, and this model's \
                     vocabulary has {vocab} tokens",
                    path.display()
                ),
                "slot_model_mismatch",
            ));
        }
    }

    let n_restored = payload.tokens.len();
    cache.lock().unwrap_or_else(|p| p.into_inner()).store(
        payload.tokens,
        payload.layers,
        payload.pending_logits,
    );

    Ok(Json(serde_json::json!({
        "id_slot": id_slot,
        "filename": body.filename,
        "n_restored": n_restored,
        "n_read": n_read,
        "timings": {"restore_ms": started.elapsed().as_secs_f64() * 1000.0},
    })))
}

/// Runs the prompt through the decoder and returns exactly what the
/// prefix cache stores for a finished request: the tokens, one KV cache
/// per layer, and the logits predicting the token after them.
///
/// `host_kv = true` and the Metal sync are not optional and not
/// belt-and-braces. A Metal prefill leaves the real K/V on the device
/// and the host rows zero-filled; a snapshot taken without them is all
/// zeros, and the request that restores it answers fluent nonsense.
/// That exact bug is recorded at `generate::forward_prompt_batch`, and
/// this path would have reproduced it.
fn prefill_slot(decoder: &Decoder, tokens: Vec<usize>) -> SlotPayload {
    let mut layers: Vec<KvCache> = decoder.config.new_kv_caches();
    let pending_logits =
        crate::generate::forward_prompt_batch(decoder, &tokens, 0, &mut layers, true);
    #[cfg(feature = "metal")]
    decoder.sync_metal_attn_kv_to_host(&mut layers);
    SlotPayload {
        tokens,
        pending_logits,
        layers,
    }
}

/// Temp file, `fsync`, rename. A reader sees the whole slot or no slot,
/// never a half-written one -- the same publish `frink_core::kv_disk`
/// uses, for the same reason.
fn publish(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<u64> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(bytes.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whitelist has to reject every way of naming a parent
    /// directory, including the ones a blacklist of "/" and ".." would
    /// miss.
    #[test]
    fn a_filename_that_could_leave_the_slot_directory_is_refused() {
        for bad in [
            "../escape",
            "..",
            ".",
            "sub/dir",
            "back\\slash",
            "nul\0byte",
            ".hidden",
            "",
        ] {
            assert!(
                validate_filename(bad).is_err(),
                "{bad:?} should not be a slot name"
            );
        }
    }

    #[test]
    fn an_ordinary_slot_name_is_accepted() {
        for good in ["system.fslot", "sys-prompt_v2.fslot", "a", "0"] {
            assert!(validate_filename(good).is_ok(), "{good:?}");
        }
    }

    /// A length read off the wire, bounded by the format rather than by
    /// a chosen constant: 255 is the shortest maximum any filesystem
    /// frink targets imposes on a path component.
    #[test]
    fn an_over_long_filename_is_refused_rather_than_truncated() {
        let name = "a".repeat(256);
        assert!(validate_filename(&name).is_err());
        assert!(validate_filename(&name[..255]).is_ok());
    }
}
