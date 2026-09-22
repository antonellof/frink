//! The `/admin` control surface: model inventory, model swap, the Hub
//! download job, task polling and server counters.
//!
//! ## What discovery is allowed to cost
//!
//! `GET /admin/models` reads **headers only**. `GgufFile::open` mmaps
//! the file and parses the metadata table plus the tensor descriptors;
//! it never touches a weight byte and never dequantizes anything, so
//! listing twenty checkpoints costs twenty header parses rather than
//! twenty loads. Anything that would need real work to answer -- what a
//! loaded model's true resident-set size is, for instance -- is
//! reported as `null`. A plausible number in that slot is worse than
//! no number, because the UI cannot tell it apart from a measured one.
//!
//! ## What the surface deliberately cannot do
//!
//! There is **no load-by-path endpoint**. A client names a model by an
//! id the server itself minted while scanning a directory it chose, so
//! there is no request shape that can make the server open an arbitrary
//! file. The download endpoint is the mirror image: the target filename
//! is sanitized to a bare `*.gguf` name and the resolved path is
//! checked to be a direct child of the model directory, so no `repo` or
//! `file` value can write outside it.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use frink_api::{
    admin::{
        CancelResponse, DownloadRequest, LoadModelRequest, ModelEntry, ModelState, ModelsResponse,
        SleepResponse, StatsResponse, TaskAccepted, TaskKind, TasksResponse, UnloadResponse,
    },
    routes,
};
use frink_gguf::{GgmlType, GgufFile, ShardName};

use crate::tasks::{Task, TaskGuard};
use crate::{ActiveModel, AppState};

/// Hard ceiling on a `.gguf` filename accepted from a download request.
/// Long enough for every real Hub filename, short enough that no
/// filesystem rejects it.
const MAX_FILENAME_LEN: usize = 255;

// ---------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------

/// One checkpoint found on disk, described from its header.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Discovered {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) size_bytes: u64,
    pub(crate) arch: Option<String>,
    pub(crate) quant: Option<String>,
    pub(crate) context_length: Option<u64>,
    pub(crate) param_count: Option<u64>,
}

/// The directories `/admin/models` scans: the directory holding
/// `FRINK_MODEL_PATH`, plus `FRINK_MODEL_DIR` when set. Both, not
/// either -- an operator who points the server at one file and a
/// library at another expects to see both listed.
pub(crate) fn model_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut push = |dir: PathBuf| {
        if dir.is_dir() && !dirs.contains(&dir) {
            dirs.push(dir);
        }
    };
    if let Ok(dir) = std::env::var("FRINK_MODEL_DIR") {
        push(PathBuf::from(dir));
    }
    if let Ok(path) = std::env::var("FRINK_MODEL_PATH") {
        // The parent either way: for a `.gguf` file that is the library
        // directory, and for a Kimi-style checkpoint -- which *is* a
        // directory -- it is the directory that library sits in, which
        // is the one holding sibling checkpoints.
        if let Some(parent) = PathBuf::from(path).parent().map(Path::to_path_buf) {
            push(parent);
        }
    }
    dirs
}

/// llama.cpp's `general.file_type` enum, for the quantization *mix*
/// name (`Q4_K_M`) that no single tensor dtype can express.
///
/// Only the values whose meaning is unambiguous are mapped. An
/// unrecognized file type falls through to the dominant tensor dtype,
/// which is measured rather than looked up -- coarser (`Q4_K`, not
/// `Q4_K_M`) but never wrong.
fn file_type_name(ft: u64) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        _ => return None,
    })
}

/// The ggml name of one tensor dtype, for the fallback above.
fn ggml_type_name(t: GgmlType) -> Option<&'static str> {
    Some(match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::BF16 => "BF16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q8_1 => "Q8_1",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::IQ4NL => "IQ4_NL",
        GgmlType::IQ4XS => "IQ4_XS",
        GgmlType::IQ1S => "IQ1_S",
        GgmlType::IQ2XXS => "IQ2_XXS",
        GgmlType::IQ3XXS => "IQ3_XXS",
        // Landed on main while this file was on a branch: the decode
        // tiers from `coverage-iq-tiers-published`.
        GgmlType::IQ2XS => "IQ2_XS",
        GgmlType::IQ2S => "IQ2_S",
        GgmlType::IQ3S => "IQ3_S",
        GgmlType::IQ1M => "IQ1_M",
        GgmlType::MXFP4 => "MXFP4",
        // Recognized and sized, no execution path. They still name the
        // checkpoint's quantization honestly, which is exactly what an
        // admin listing is for -- the refusal happens at load, not
        // here.
        GgmlType::TQ1_0 => "TQ1_0",
        GgmlType::TQ2_0 => "TQ2_0",
        GgmlType::NVFP4 => "NVFP4",
        GgmlType::Q1_0 => "Q1_0",
        GgmlType::Q2_0 => "Q2_0",
        GgmlType::PQ2_0 => "PQ2_0",
        GgmlType::PTQ1_0 => "PTQ1_0",
        // I32 is a routing table, not a weight format, and `Other` is
        // a tag this build does not recognize. Neither names the
        // checkpoint's quantization.
        GgmlType::I32 | GgmlType::Other(_) => return None,
    })
}

/// The quantization of a GGUF: its declared file type when that maps to
/// a name, otherwise the dtype most of its *weight* tensors use.
///
/// "Most" is counted by tensor, not by byte, and norms/embeddings are
/// not excluded -- this is a label for a human, and the dominant dtype
/// of a real checkpoint is unambiguous either way.
fn quant_of(file: &GgufFile) -> Option<String> {
    if let Some(name) = file
        .metadata_u64("general.file_type")
        .and_then(file_type_name)
    {
        return Some(name.to_string());
    }
    let mut counts: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();
    for t in &file.tensors {
        if let Some(name) = ggml_type_name(t.dtype) {
            *counts.entry(name).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(name, n)| (*n, *name))
        .map(|(name, _)| name.to_string())
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Total on-disk size of a checkpoint directory (a Kimi-style
/// checkpoint), one level deep. Not recursive: the real layout is flat,
/// and walking arbitrary depth from an HTTP handler is an invitation.
fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}

/// Reads one `.gguf`'s header into a [`Discovered`].
///
/// For a split checkpoint this is called on shard 1 only; sibling
/// shards contribute their size and their tensor counts, and nothing
/// else -- every metadata key is required to agree across shards
/// anyway (see `frink_gguf::ShardedGguf`).
fn describe_gguf(id: String, path: &Path) -> Discovered {
    let mut size_bytes = file_size(path);
    let mut siblings: Vec<PathBuf> = Vec::new();
    if let Some(name) = ShardName::parse(path) {
        for no in 2..=name.count {
            let sibling = name.sibling(no);
            if sibling.exists() {
                size_bytes += file_size(&sibling);
                siblings.push(sibling);
            }
        }
    }

    let Ok(file) = GgufFile::open(path) else {
        // Unreadable header: the file is listed (it exists, and hiding
        // it would leave the user wondering where it went) with every
        // derived field null.
        return Discovered {
            id,
            path: path.to_path_buf(),
            size_bytes,
            arch: None,
            quant: None,
            context_length: None,
            param_count: None,
        };
    };

    let arch = file
        .metadata_str("general.architecture")
        .map(str::to_string);
    let context_length = arch
        .as_deref()
        .and_then(|a| file.metadata_u64(&format!("{a}.context_length")));
    let param_count = match file.metadata_u64("general.parameter_count") {
        Some(n) => Some(n),
        None => {
            let mut total: u64 = file.tensors.iter().map(|t| t.n_elements() as u64).sum();
            for sibling in &siblings {
                match GgufFile::open(sibling) {
                    Ok(shard) => {
                        total += shard
                            .tensors
                            .iter()
                            .map(|t| t.n_elements() as u64)
                            .sum::<u64>()
                    }
                    // One unreadable shard makes the sum wrong, and a
                    // wrong parameter count is worse than none.
                    Err(_) => {
                        return Discovered {
                            id,
                            path: path.to_path_buf(),
                            size_bytes,
                            arch,
                            quant: quant_of(&file),
                            context_length,
                            param_count: None,
                        }
                    }
                }
            }
            Some(total)
        }
    };

    Discovered {
        id,
        path: path.to_path_buf(),
        size_bytes,
        arch,
        quant: quant_of(&file),
        context_length,
        param_count,
    }
}

/// True for a directory that looks like a Kimi K3 checkpoint (the one
/// non-GGUF shape `model::load_from_path` accepts).
fn is_checkpoint_dir(dir: &Path) -> bool {
    dir.join("model.safetensors.index.json").is_file()
}

/// Scans `dirs` non-recursively for `.gguf` files and checkpoint
/// directories.
///
/// Continuation shards (`…-00002-of-00003.gguf`) are not listed
/// separately: one logical checkpoint is one entry, named by shard 1.
/// Ids collide across directories often enough to matter (two copies of
/// the same download), so a duplicate id is suffixed with the directory
/// it came from rather than silently dropping one of the two files.
pub(crate) fn discover(dirs: &[PathBuf]) -> Vec<Discovered> {
    let mut out: Vec<Discovered> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            let found = if path.is_dir() {
                if !is_checkpoint_dir(&path) {
                    continue;
                }
                let Some(id) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                Discovered {
                    id: id.to_string(),
                    size_bytes: dir_size(&path),
                    path: path.clone(),
                    arch: None,
                    quant: None,
                    context_length: None,
                    param_count: None,
                }
            } else {
                if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
                    continue;
                }
                // Shard 2..n of a split checkpoint is part of shard 1's
                // entry, not an entry of its own.
                if ShardName::parse(&path).is_some_and(|n| n.no != 1) {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                // `foo-00001-of-00003` reads badly as an id; use the
                // shard prefix, which is the checkpoint's real name.
                let id = match ShardName::parse(&path) {
                    Some(name) => Path::new(&name.prefix)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or(stem)
                        .to_string(),
                    None => stem.to_string(),
                };
                describe_gguf(id, &path)
            };
            let found = disambiguate(found, &out, dir);
            out.push(found);
        }
    }
    out
}

/// Keeps ids unique within one response. A collision means the same
/// filename in two scanned directories; suffixing with the directory
/// name keeps both addressable instead of hiding one.
fn disambiguate(mut found: Discovered, existing: &[Discovered], dir: &Path) -> Discovered {
    if !existing.iter().any(|d| d.id == found.id) {
        return found;
    }
    let dir_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dir")
        .to_string();
    let base = format!("{}@{dir_name}", found.id);
    let mut candidate = base.clone();
    let mut n = 2;
    while existing.iter().any(|d| d.id == candidate) {
        candidate = format!("{base}-{n}");
        n += 1;
    }
    found.id = candidate;
    found
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

fn json_error(status: StatusCode, message: &str, kind: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

pub(crate) async fn models(State(state): State<Arc<AppState>>) -> Response {
    let dirs = model_dirs();
    let active = state.active();
    let active_id = active.as_ref().and_then(|a| a.id.clone());
    let loading_id = state.loading_model_id();

    let found = tokio::task::spawn_blocking(move || discover(&dirs)).await;
    let found = match found {
        Ok(found) => found,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("model discovery panicked: {e}"),
                "internal",
            )
        }
    };

    let last_error = state.last_load_error();
    let models = found
        .into_iter()
        .map(|d| {
            let state_of = if Some(&d.id) == active_id.as_ref() {
                ModelState::Loaded
            } else if Some(&d.id) == loading_id.as_ref() {
                ModelState::Loading
            } else if last_error.as_ref().is_some_and(|(id, _)| *id == d.id) {
                ModelState::Error
            } else {
                ModelState::Available
            };
            ModelEntry {
                id: d.id,
                path: d.path.display().to_string(),
                size_bytes: d.size_bytes,
                arch: d.arch,
                quant: d.quant,
                context_length: d.context_length,
                param_count: d.param_count,
                error: if state_of == ModelState::Error {
                    last_error.as_ref().map(|(_, msg)| msg.clone())
                } else {
                    None
                },
                state: state_of,
                // Deliberately unmeasured: frink keeps checkpoints
                // mmap-resident, so the honest answer is a page-cache
                // property this process cannot read. See ModelEntry.
                resident_bytes: None,
            }
        })
        .collect();

    Json(ModelsResponse {
        // The directory fixed at startup, not a fresh read of the
        // environment: it is also the only directory a download may
        // write into, and the UI should be told the same one.
        model_dir: state.model_dir.as_ref().map(|d| d.display().to_string()),
        active: active_id,
        models,
    })
    .into_response()
}

pub(crate) async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let dirs = model_dirs();
    let id = req.id.clone();
    let found = match tokio::task::spawn_blocking(move || discover(&dirs)).await {
        Ok(found) => found,
        Err(e) => {
            return json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("model discovery panicked: {e}"),
                "internal",
            )
        }
    };
    let Some(target) = found.into_iter().find(|d| d.id == id) else {
        return json_error(
            StatusCode::NOT_FOUND,
            &format!("no model with id '{id}' in the scanned model directories"),
            "not_found",
        );
    };

    // One load at a time. Two concurrent loads would each allocate a
    // checkpoint's worth of memory and then one would win anyway.
    if state
        .load_in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return json_error(
            StatusCode::CONFLICT,
            "a model load is already in progress",
            "load_in_progress",
        );
    }

    let task = state
        .tasks
        .create(TaskKind::Load, format!("Loading {}", target.id));
    state.set_loading_model(Some(target.id.clone()));
    let task_id = task.task_id.clone();

    let state_for_task = Arc::clone(&state);
    let path = target.path.display().to_string();
    let model_id = target.id.clone();
    tokio::task::spawn_blocking(move || {
        // The guard, not the function, is what guarantees a verdict:
        // nothing awaits this handle, so a panic inside would otherwise
        // leave the task at `running` forever.
        let _guard = TaskGuard::new(Arc::clone(&task));
        run_load_task(state_for_task, task, model_id, path);
    });

    (StatusCode::ACCEPTED, Json(TaskAccepted { task_id })).into_response()
}

/// The load worker. Runs on `spawn_blocking` -- `model::load_from_path`
/// mmaps and, for a Kimi checkpoint, touches every expert range.
///
/// Cancellation is checked twice: before starting, and again before
/// publishing. A load in flight cannot be interrupted (there is no
/// cancellation point inside an mmap-and-index pass), so a cancel that
/// arrives mid-load discards the finished result instead of pretending
/// the work stopped early. That costs the wasted load and keeps the
/// reported state true.
fn run_load_task(state: Arc<AppState>, task: Arc<Task>, id: String, path: String) {
    struct Guard(Arc<AppState>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.load_in_progress.store(false, Ordering::SeqCst);
            self.0.set_loading_model(None);
        }
    }
    let _guard = Guard(Arc::clone(&state));

    if task.is_cancelled() {
        task.acknowledge_cancel();
        return;
    }
    task.start();

    let loaded = match crate::model::load_from_path(&path) {
        Ok(loaded) => loaded,
        Err(e) => {
            tracing::warn!("admin load of '{id}' failed: {e}");
            state.set_last_load_error(Some((id, e.to_string())));
            task.fail(e);
            return;
        }
    };

    if task.is_cancelled() {
        tracing::info!("admin load of '{id}' finished but was cancelled; discarding it");
        task.acknowledge_cancel();
        return;
    }

    // The path this task resolved, not `FRINK_MODEL_PATH`: a swapped-in
    // model must be priced against its own weights, or the new model
    // would admit on the startup model's arithmetic.
    let (loaded, batcher, ceiling) = crate::activate_loaded_model(
        loaded,
        state.continuous_batching_enabled,
        Some(path.as_str()),
        // A hot-swapped model reuses the server's paged store: the
        // pages are the deployment's memory, not the model's.
        state.paged_kv.as_ref(),
    );
    let previous = state.swap_active(Some(Arc::new(ActiveModel {
        id: Some(id.clone()),
        loaded,
        batcher,
        ceiling,
        checkpoint_path: Some(PathBuf::from(&path)),
    })));
    state.set_last_load_error(None);
    // Dropped here, outside the swap's write lock and after it: any
    // request still decoding holds its own Arc, so this only frees the
    // old weights once the last of them is finished.
    drop(previous);
    tracing::info!("active model is now '{id}' ({path})");
    task.succeed();
}

pub(crate) async fn unload_model(State(state): State<Arc<AppState>>) -> Response {
    let previous = state.swap_active(None);
    if let Some(previous) = &previous {
        tracing::info!(
            "unloaded model '{}'; in-flight requests keep their handle until they finish",
            previous.id.as_deref().unwrap_or("<startup>")
        );
    }
    drop(previous);
    Json(UnloadResponse {
        ok: true,
        active: None,
    })
    .into_response()
}

/// Free the active model's memory, remembering what it was.
///
/// An UNLOAD THAT REMEMBERS. `/admin/models/unload` leaves the server
/// with nothing to serve and no record of what it served, so only a
/// client that already knows the id can bring it back; a sleeping
/// server can wake itself. That is the difference that makes the pair
/// usable from a scheduler which does not know the deployment.
///
/// **One level, and it is honest about that.** vLLM offers two --
/// offload the weights to host memory, or discard them -- because it
/// holds them in device memory it allocated. frink mmaps its weights,
/// so "discard" is what dropping the handle already does and the OS
/// page cache decides how much of a reload touches disk. What sleep
/// really frees here is the allocations around the weights: the KV
/// pool, the paged store, the repack and expert caches, and any device
/// buffers. A `level` parameter would be a knob with one position.
pub(crate) async fn sleep(State(state): State<Arc<AppState>>) -> Response {
    if state.is_sleeping() {
        // Idempotent: a scheduler that sends two is not an error, and
        // the second must not lose the record the first made.
        return Json(SleepResponse {
            ok: true,
            is_sleeping: true,
        })
        .into_response();
    }
    let Some(active) = state.active() else {
        return json_error(
            StatusCode::CONFLICT,
            "nothing is loaded, so there is nothing to put away",
            "model_not_loaded",
        );
    };
    let Some(path) = active.checkpoint_path.clone() else {
        // A model with no path on record cannot be reloaded, and
        // sleeping it would be a one-way door dressed as a round trip.
        return json_error(
            StatusCode::CONFLICT,
            "the active model has no checkpoint path on record, so it could not be reloaded; \
             refusing to put away something that cannot be brought back",
            "not_reloadable",
        );
    };
    *state.slept.lock().unwrap_or_else(|p| p.into_inner()) = Some(crate::SleptModel {
        id: active.id.clone(),
        path,
    });
    // After the record, so a wake between the two cannot find an empty
    // one.
    let previous = state.swap_active(None);
    drop(previous);
    tracing::info!("asleep; POST /wake_up reloads the model");
    Json(SleepResponse {
        ok: true,
        is_sleeping: true,
    })
    .into_response()
}

/// Reload the model a [`sleep`] put away.
pub(crate) async fn wake_up(State(state): State<Arc<AppState>>) -> Response {
    let slept = state
        .slept
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let Some(slept) = slept else {
        return json_error(
            StatusCode::CONFLICT,
            "this server is not asleep",
            "not_sleeping",
        );
    };
    if state
        .load_in_progress
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return json_error(
            StatusCode::CONFLICT,
            "a model load is already in progress",
            "load_in_progress",
        );
    }
    let id = slept.id.clone().unwrap_or_else(|| "<startup>".to_string());
    let path = slept.path.display().to_string();
    let task = state.tasks.create(TaskKind::Load, &id);
    let handle = task.clone();
    let inner = Arc::clone(&state);
    tokio::task::spawn_blocking(move || run_load_task(inner, handle, id, path));
    // The record is cleared only once the load is under way, so a
    // failed wake leaves the server asleep and retryable rather than
    // awake with nothing loaded.
    *state.slept.lock().unwrap_or_else(|p| p.into_inner()) = None;
    Json(SleepResponse {
        ok: true,
        is_sleeping: false,
    })
    .into_response()
}

/// Whether this server is asleep.
pub(crate) async fn is_sleeping(State(state): State<Arc<AppState>>) -> Response {
    Json(serde_json::json!({ "is_sleeping": state.is_sleeping() })).into_response()
}

pub(crate) async fn tasks(State(state): State<Arc<AppState>>) -> Response {
    Json(TasksResponse {
        tasks: state.tasks.views(),
    })
    .into_response()
}

pub(crate) async fn cancel_task(
    State(state): State<Arc<AppState>>,
    AxumPath(task_id): AxumPath<String>,
) -> Response {
    let Some(task) = state.tasks.get(&task_id) else {
        return json_error(
            StatusCode::NOT_FOUND,
            &format!("no task with id '{task_id}'"),
            "not_found",
        );
    };
    if !task.request_cancel() {
        return json_error(
            StatusCode::CONFLICT,
            &format!("task '{task_id}' has already finished"),
            "task_finished",
        );
    }
    Json(CancelResponse { ok: true }).into_response()
}

pub(crate) async fn stats(State(state): State<Arc<AppState>>) -> Response {
    // The queue gauge exists only where a queue exists. Continuous
    // batching is the one path that makes requests wait for a decode
    // slot; without it every request gets its own blocking thread and
    // nothing is queued in front of anything. Reporting `0` in that
    // case would claim an empty queue was measured, and a UI cannot
    // tell that apart from "there is no queue here".
    let queue = state
        .active()
        .and_then(|a| a.batcher.as_ref().map(|b| b.stats()));
    Json(StatsResponse {
        uptime_seconds: state.uptime().as_secs(),
        requests_total: state.requests_total(),
        errors_total: state.errors_total(),
        cache_hits: state.cache_stats().hits,
        cache_misses: state.cache_stats().misses,
        tokens_prompt_total: state.stats.tokens_prompt_total(),
        tokens_generated_total: state.stats.tokens_generated_total(),
        last_request_age_seconds: state.last_request_age_seconds(),
        generating_now: state.cancels.live_count(),
        queue_depth: queue.as_ref().map(|q| q.queue_depth),
        queue_rejected_total: queue.as_ref().map(|q| q.queue_rejected),
        recent: state.stats.recent(),
    })
    .into_response()
}

// ---------------------------------------------------------------------
// Download
// ---------------------------------------------------------------------

/// Rejects anything that is not a bare `*.gguf` filename.
///
/// This is the whole of the path safety story for downloads and it is
/// deliberately a whitelist: no separators of either kind, no `..`, no
/// leading dot, no NUL, and a `.gguf` extension. A name that survives
/// this cannot escape the model directory when joined to it, which is
/// re-checked in [`target_path`] anyway.
pub(crate) fn sanitize_gguf_filename(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("file must not be empty".to_string());
    }
    if name.len() > MAX_FILENAME_LEN {
        return Err(format!("file name is longer than {MAX_FILENAME_LEN} bytes"));
    }
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return Err("only .gguf targets can be downloaded".to_string());
    }
    if name.contains('/') || name.contains('\\') {
        return Err("file must be a bare filename, not a path".to_string());
    }
    if name.contains('\0') {
        return Err("file name contains a NUL byte".to_string());
    }
    if name == ".." || name.starts_with('.') {
        return Err("file must not start with a dot".to_string());
    }
    // A drive-relative or UNC-ish name on Windows.
    if name.contains(':') {
        return Err("file name must not contain ':'".to_string());
    }
    Ok(name.to_string())
}

/// Joins a sanitized filename to the model directory and re-checks that
/// the result is a direct child of it.
///
/// Belt and braces on purpose: [`sanitize_gguf_filename`] already makes
/// an escape impossible, and this catches the case where it is one day
/// relaxed. `canonicalize` is applied to the *directory* only, since
/// the target file does not exist yet.
pub(crate) fn target_path(dir: &Path, filename: &str) -> Result<PathBuf, String> {
    let filename = sanitize_gguf_filename(filename)?;
    let dir = dir
        .canonicalize()
        .map_err(|e| format!("model directory {} is not readable: {e}", dir.display()))?;
    let target = dir.join(&filename);
    if target.parent() != Some(dir.as_path()) {
        return Err("resolved path escapes the model directory".to_string());
    }
    Ok(target)
}

/// A Hub repo id: `owner/name`, no path traversal, no query string.
pub(crate) fn sanitize_repo(repo: &str) -> Result<String, String> {
    let repo = repo.trim().trim_matches('/');
    if repo.is_empty() {
        return Err("repo must not be empty".to_string());
    }
    if repo.len() > 200 {
        return Err("repo is implausibly long".to_string());
    }
    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 {
        return Err("repo must be of the form owner/name".to_string());
    }
    for part in &parts {
        if part.is_empty() || *part == "." || *part == ".." {
            return Err("repo contains an empty or relative path segment".to_string());
        }
        if !part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        {
            return Err("repo may only contain letters, digits, '-', '_' and '.'".to_string());
        }
    }
    Ok(repo.to_string())
}

pub(crate) async fn download(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DownloadRequest>,
) -> Response {
    let repo = match sanitize_repo(&req.repo) {
        Ok(repo) => repo,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, &e, "invalid_request"),
    };
    // A glob is resolved against the repo's file list before anything
    // touches the filesystem; a literal is validated right here.
    let is_glob = req.file.contains('*');
    if !is_glob {
        if let Err(e) = sanitize_gguf_filename(&req.file) {
            return json_error(StatusCode::BAD_REQUEST, &e, "invalid_request");
        }
    } else if !req.file.to_ascii_lowercase().ends_with(".gguf") {
        return json_error(
            StatusCode::BAD_REQUEST,
            "only .gguf targets can be downloaded",
            "invalid_request",
        );
    }

    // Resolved once at startup and never from the request, so no
    // request body can redirect where bytes land.
    let Some(dir) = state.model_dir.clone() else {
        return json_error(
            StatusCode::PRECONDITION_FAILED,
            "no model directory is configured; set FRINK_MODEL_DIR or FRINK_MODEL_PATH",
            "no_model_dir",
        );
    };

    let label = format!("Downloading {} from {repo}", req.file);
    // Two workers on one target would interleave writes into the same
    // `.part` file and leave a corrupt checkpoint of exactly the right
    // size -- the worst failure mode available here.
    if state.tasks.has_live(TaskKind::Download, &label) {
        return json_error(
            StatusCode::CONFLICT,
            "that file is already downloading",
            "download_in_progress",
        );
    }

    let task = state.tasks.create(TaskKind::Download, label);
    let task_id = task.task_id.clone();
    let pattern = req.file.clone();
    tokio::task::spawn_blocking(move || {
        let _guard = TaskGuard::new(Arc::clone(&task));
        if let Err(e) = run_download_task(&task, &repo, &pattern, &dir) {
            if task.is_cancelled() {
                task.acknowledge_cancel();
            } else {
                tracing::warn!("download of {pattern} from {repo} failed: {e}");
                task.fail(e);
            }
        }
    });

    (StatusCode::ACCEPTED, Json(TaskAccepted { task_id })).into_response()
}

/// Runs one download to completion, reporting through `task`.
///
/// Resumable: bytes land in `<target>.part`, a restart sends
/// `Range: bytes=<already-there>-`, and the partial file is renamed into
/// place only after the last byte arrives. A server that ignores the
/// range header answers 200 instead of 206, in which case the partial
/// file is truncated and the transfer starts over rather than
/// concatenating two copies of the prefix.
fn run_download_task(task: &Task, repo: &str, pattern: &str, dir: &Path) -> Result<(), String> {
    task.start();
    let filename = if pattern.contains('*') {
        frink_models::hub::resolve_glob(repo, pattern)?
    } else {
        pattern.to_string()
    };
    let target = target_path(dir, &filename)?;
    if target.exists() {
        return Err(format!(
            "{} already exists; delete it first",
            target.display()
        ));
    }
    let part = target.with_extension("gguf.part");

    let resume_from = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut response = frink_models::hub::open_file(repo, &filename, resume_from)?;
    let mut file = if response.resumed {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(&part)
            .map_err(|e| format!("reopening {}: {e}", part.display()))?;
        f.seek(SeekFrom::Start(resume_from))
            .map_err(|e| format!("seeking {}: {e}", part.display()))?;
        f
    } else {
        std::fs::File::create(&part).map_err(|e| format!("creating {}: {e}", part.display()))?
    };

    let mut done = if response.resumed { resume_from } else { 0 };
    let total = response.total_bytes;
    task.observe(done, total);

    let mut buf = vec![0u8; 1 << 20];
    loop {
        if task.is_cancelled() {
            // The `.part` file stays: a later retry resumes from it.
            return Err("cancelled".to_string());
        }
        let n = response
            .body
            .read(&mut buf)
            .map_err(|e| format!("reading from the Hub: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("writing {}: {e}", part.display()))?;
        done += n as u64;
        task.observe(done, total);
    }
    file.flush()
        .map_err(|e| format!("flushing {}: {e}", part.display()))?;
    drop(file);

    if let Some(total) = total {
        if done != total {
            return Err(format!(
                "truncated download: {done} of {total} bytes (the partial file was kept for a retry)"
            ));
        }
    }
    std::fs::rename(&part, &target)
        .map_err(|e| format!("renaming {} into place: {e}", part.display()))?;
    tracing::info!("downloaded {} ({done} bytes)", target.display());
    task.succeed();
    Ok(())
}

/// The axum path pattern for [`routes::ADMIN_TASK_CANCEL`].
///
/// axum 0.7 spells a path parameter `:name`; the contract crate spells
/// it `{name}` because it is imported by clients that have never heard
/// of axum. Deriving one from the other keeps them from drifting -- see
/// the test below.
pub(crate) fn cancel_route() -> String {
    routes::ADMIN_TASK_CANCEL.replace("{task_id}", ":task_id")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_axum_cancel_pattern_matches_the_published_template() {
        assert_eq!(cancel_route(), "/admin/tasks/:task_id/cancel");
        assert_eq!(
            routes::admin_task_cancel("abc"),
            cancel_route().replace(":task_id", "abc")
        );
    }

    #[test]
    fn a_traversing_filename_is_rejected_outright() {
        for bad in [
            "../evil.gguf",
            "/etc/evil.gguf",
            "sub/dir/model.gguf",
            "..\\evil.gguf",
            "C:model.gguf",
            ".hidden.gguf",
            "",
        ] {
            assert!(
                sanitize_gguf_filename(bad).is_err(),
                "{bad:?} should have been rejected"
            );
        }
    }

    #[test]
    fn only_gguf_targets_are_accepted() {
        assert!(sanitize_gguf_filename("model.safetensors").is_err());
        assert!(sanitize_gguf_filename("model.gguf.sh").is_err());
        assert!(sanitize_gguf_filename("model.sh").is_err());
        assert_eq!(
            sanitize_gguf_filename("Model-Q4_K_M.gguf").unwrap(),
            "Model-Q4_K_M.gguf"
        );
        // Case-insensitive extension, name preserved verbatim.
        assert_eq!(sanitize_gguf_filename("M.GGUF").unwrap(), "M.GGUF");
    }

    #[test]
    fn a_sanitized_name_resolves_inside_the_model_directory() {
        let dir = std::env::temp_dir().join(format!("frink-admin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = target_path(&dir, "model.gguf").unwrap();
        assert_eq!(target.parent().unwrap(), dir.canonicalize().unwrap());
        assert!(target_path(&dir, "../model.gguf").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repo_ids_must_be_owner_slash_name() {
        assert_eq!(
            sanitize_repo("unsloth/gemma-3-4b").unwrap(),
            "unsloth/gemma-3-4b"
        );
        for bad in [
            "",
            "single",
            "a/b/c",
            "../../etc",
            "owner/name?x=1",
            "owner/na me",
            "owner//name",
        ] {
            assert!(sanitize_repo(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn globs_match_the_patterns_people_actually_type() {
        assert!(frink_models::hub::glob_matches(
            "*.gguf",
            "model-Q4_K_M.gguf"
        ));
        assert!(!frink_models::hub::glob_matches(
            "*.gguf",
            "model.safetensors"
        ));
        assert!(frink_models::hub::glob_matches(
            "*Q4_K_M*.gguf",
            "llama-3.2-3B-Q4_K_M-v2.gguf"
        ));
        assert!(!frink_models::hub::glob_matches(
            "*Q4_K_M*.gguf",
            "llama-3.2-3B-Q8_0.gguf"
        ));
        assert!(frink_models::hub::glob_matches("exact.gguf", "exact.gguf"));
        assert!(!frink_models::hub::glob_matches("exact.gguf", "other.gguf"));
        assert!(frink_models::hub::glob_matches("*", "anything"));
    }

    #[test]
    fn known_file_types_win_over_the_dtype_fallback() {
        assert_eq!(file_type_name(15), Some("Q4_K_M"));
        assert_eq!(file_type_name(1), Some("F16"));
        // Deprecated / unknown enum values must not be invented.
        assert_eq!(file_type_name(4), None);
        assert_eq!(file_type_name(999), None);
    }

    #[test]
    fn tensor_dtype_names_cover_the_formats_this_build_decodes() {
        assert_eq!(ggml_type_name(GgmlType::Q4K), Some("Q4_K"));
        assert_eq!(ggml_type_name(GgmlType::IQ4XS), Some("IQ4_XS"));
        // Not a weight format, so not a quantization label.
        assert_eq!(ggml_type_name(GgmlType::I32), None);
        assert_eq!(ggml_type_name(GgmlType::Other(77)), None);
    }

    // --- discovery fixtures -----------------------------------------
    //
    // A GGUF header written by hand, in the same shape a real writer
    // emits: magic, version 3, tensor and kv counts, the kv table, the
    // tensor descriptors, then 32-byte-aligned tensor data. Enough for
    // discovery, which never reads past the descriptors.

    enum Kv {
        U32(u32),
        Str(&'static str),
    }

    fn push_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn build_gguf(kvs: &[(&str, Kv)], tensors: &[(&str, usize)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&frink_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, val) in kvs {
            push_string(&mut buf, key);
            match val {
                Kv::U32(v) => {
                    buf.extend_from_slice(&4u32.to_le_bytes());
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(v) => {
                    buf.extend_from_slice(&8u32.to_le_bytes());
                    push_string(&mut buf, v);
                }
            }
        }
        let mut offset = 0u64;
        for (name, n) in tensors {
            push_string(&mut buf, name);
            buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            buf.extend_from_slice(&(*n as u64).to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += ((*n as u64) * 4).div_ceil(32) * 32;
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.resize(buf.len() + offset as usize, 0);
        buf
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "frink_admin_{tag}_{}_{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::remove_dir_all(&dir).ok();
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn discovery_reads_arch_quant_context_and_parameters_from_the_header() {
        let dir = TempDir::new("headers");
        std::fs::write(
            dir.0.join("tiny-llama.gguf"),
            build_gguf(
                &[
                    ("general.architecture", Kv::Str("llama")),
                    ("general.file_type", Kv::U32(15)),
                    ("llama.context_length", Kv::U32(8192)),
                ],
                &[("blk.0.attn_q.weight", 64), ("output.weight", 32)],
            ),
        )
        .unwrap();

        let found = discover(std::slice::from_ref(&dir.0));
        assert_eq!(found.len(), 1);
        let m = &found[0];
        assert_eq!(m.id, "tiny-llama");
        assert_eq!(m.arch.as_deref(), Some("llama"));
        // The declared file type beats the dominant tensor dtype: it is
        // the only place the *mix* (`_M`) is stated at all.
        assert_eq!(m.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(m.context_length, Some(8192));
        assert_eq!(m.param_count, Some(96));
        assert!(m.size_bytes > 0);
    }

    #[test]
    fn a_header_without_a_file_type_falls_back_to_the_measured_dtype() {
        let dir = TempDir::new("nofiletype");
        std::fs::write(
            dir.0.join("plain.gguf"),
            build_gguf(
                &[("general.architecture", Kv::Str("qwen2"))],
                &[("blk.0.attn_q.weight", 32)],
            ),
        )
        .unwrap();
        let found = discover(std::slice::from_ref(&dir.0));
        assert_eq!(found[0].quant.as_deref(), Some("F32"));
        // No `{arch}.context_length` key means no answer, not a guess.
        assert_eq!(found[0].context_length, None);
    }

    #[test]
    fn an_unreadable_gguf_is_listed_with_null_fields_rather_than_hidden() {
        let dir = TempDir::new("corrupt");
        std::fs::write(dir.0.join("broken.gguf"), b"not a gguf at all").unwrap();
        let found = discover(std::slice::from_ref(&dir.0));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "broken");
        assert_eq!(found[0].arch, None);
        assert_eq!(found[0].quant, None);
        assert_eq!(found[0].param_count, None);
    }

    #[test]
    fn non_gguf_files_and_continuation_shards_are_not_separate_entries() {
        let dir = TempDir::new("shards");
        let header = build_gguf(
            &[("general.architecture", Kv::Str("llama"))],
            &[("blk.0.attn_q.weight", 32)],
        );
        std::fs::write(dir.0.join("big-00001-of-00002.gguf"), &header).unwrap();
        std::fs::write(dir.0.join("big-00002-of-00002.gguf"), &header).unwrap();
        std::fs::write(dir.0.join("notes.txt"), b"ignore me").unwrap();
        std::fs::write(dir.0.join("model.safetensors"), b"ignore me too").unwrap();

        let found = discover(std::slice::from_ref(&dir.0));
        assert_eq!(found.len(), 1, "{found:?}");
        // Named by the checkpoint, not by shard 1's filename.
        assert_eq!(found[0].id, "big");
        // Both shards contribute their bytes and their tensors.
        assert_eq!(found[0].size_bytes, 2 * header.len() as u64);
        assert_eq!(found[0].param_count, Some(64));
    }

    #[test]
    fn a_checkpoint_directory_is_discovered_and_a_plain_one_is_not() {
        let dir = TempDir::new("dirs");
        let kimi = dir.0.join("kimi-k3");
        std::fs::create_dir_all(&kimi).unwrap();
        std::fs::write(kimi.join("model.safetensors.index.json"), b"{}").unwrap();
        std::fs::create_dir_all(dir.0.join("not-a-checkpoint")).unwrap();

        let found = discover(std::slice::from_ref(&dir.0));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, "kimi-k3");
        // A safetensors checkpoint carries no GGUF header to read.
        assert_eq!(found[0].arch, None);
        assert!(found[0].size_bytes > 0);
    }

    #[test]
    fn the_same_filename_in_two_directories_stays_addressable() {
        let a = TempDir::new("dupa");
        let b = TempDir::new("dupb");
        let header = build_gguf(
            &[("general.architecture", Kv::Str("llama"))],
            &[("blk.0.attn_q.weight", 32)],
        );
        std::fs::write(a.0.join("model.gguf"), &header).unwrap();
        std::fs::write(b.0.join("model.gguf"), &header).unwrap();

        let found = discover(&[a.0.clone(), b.0.clone()]);
        assert_eq!(found.len(), 2);
        assert_ne!(found[0].id, found[1].id);
        assert_eq!(found[0].id, "model");
    }

    #[test]
    fn duplicate_ids_are_disambiguated_rather_than_dropped() {
        let first = Discovered {
            id: "model".into(),
            path: PathBuf::from("/a/model.gguf"),
            size_bytes: 1,
            arch: None,
            quant: None,
            context_length: None,
            param_count: None,
        };
        let second = Discovered {
            path: PathBuf::from("/b/model.gguf"),
            ..first.clone()
        };
        let out = disambiguate(second, std::slice::from_ref(&first), Path::new("/b"));
        assert_eq!(out.id, "model@b");
        assert_ne!(out.id, first.id);
    }
}
