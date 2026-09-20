//! Server-side conversation storage: the transcript tree, held by the
//! server instead of by the browser.
//!
//! Studio's Chat screen kept its transcript in `localStorage` because
//! this server had nothing to keep it in. That is the failure mode the
//! UI plan names outright: a browser-side store is per-browser, dies
//! with a cleared profile, and -- in the product this plan was written
//! from -- silently *pops conversations off the end* when the quota is
//! hit. Silent loss is the part that matters. Everything in this module
//! is arranged so that a write either lands on disk or is refused out
//! loud; nothing is ever dropped to make room.
//!
//! ## The surface
//!
//! ```text
//! GET  /v1/conversations                            summaries, newest first
//! POST /v1/conversations                            create
//! GET  /v1/conversations/{conversation_id}          one conversation, with messages
//! POST /v1/conversations/{conversation_id}          update: title/model/head_id/append
//! POST /v1/conversations/{conversation_id}/delete   delete
//! ```
//!
//! Behind the same `FRINK_API_KEY` gate as `/v1/chat/completions`,
//! because a stored transcript is the text a caller paid to generate.
//! It is mounted into the `protected` router in `lib.rs`, so it
//! inherits the auth, rate-limit and CORS layers applied there and
//! cannot accidentally be reachable when `/v1/chat/completions` is not.
//!
//! ### Why delete is a POST
//!
//! Because the CORS allow-list in `lib.rs` permits `GET` and `POST` and
//! nothing else, a `DELETE` route would be mounted, would work from
//! `curl`, and would fail the preflight for every cross-origin browser
//! client -- which, since the split that made Studio a standalone app,
//! is the normal deployment. `/admin/tasks/{task_id}/cancel` already
//! spells a destructive action as a `POST` suffix for the same reason,
//! so this follows it rather than inventing a second convention.
//!
//! ## The message tree
//!
//! A conversation is a set of nodes, each naming its `parent_id`. That
//! is what makes edit-and-regenerate a *branch* rather than an
//! overwrite: the old answer keeps its node, the new one hangs off the
//! same parent, and `head_id` says which leaf is currently selected.
//!
//! Two rules keep the set an actual tree, checked on the way in:
//!
//! - A node's `parent_id` must already name a node in the conversation
//!   (or one appended earlier in the same batch). A dangling parent is
//!   refused rather than stored, because a transcript that cannot be
//!   walked back to a root cannot be replayed to the model.
//! - Because a parent must *already* exist, a cycle cannot be
//!   constructed. This is a structural property, not a check that could
//!   be forgotten: append-only plus parent-must-exist is a forest by
//!   construction.
//!
//! Message ids are the client's -- a UI already has ids for the
//! messages on its screen, and making it re-key them against
//! server-minted ones is how transcripts get scrambled. Conversation
//! ids are the *server's*, and that is deliberate: see the path note
//! below.
//!
//! ## Storage
//!
//! One JSON file per conversation under `FRINK_CONVERSATIONS_DIR`
//! (default `./frink-conversations`), written by
//! write-temp-fsync-rename, so a crash mid-write leaves the previous
//! version intact rather than a half-written one. One file per
//! conversation rather than one file for all of them: a rewrite then
//! costs one transcript rather than the whole library, and a file that
//! does get corrupted loses one conversation instead of every
//! conversation.
//!
//! No new dependency. `crates/frink-server/src/journal.rs` already
//! establishes the convention this follows -- a flat file, a plain
//! env-var-or-literal-default path, no `dirs`/XDG resolution, and no
//! embedded database. The UI plan asked for SQLite; SQLite in this
//! workspace means `rusqlite` + `libsqlite3-sys`, which is a C
//! toolchain in the release path of a project whose Cargo.toml already
//! spends a paragraph explaining how it kept `aws-lc-sys` out. A
//! transcript store that never needs a query more complex than "give me
//! conversation X" does not earn that. If a real query ever does, this
//! module is the only thing that has to change.
//!
//! **The directory is created on first write, not at startup.** A
//! server nobody stores a conversation on leaves no directory behind.
//!
//! ## Privacy: the exact inverse of the journal
//!
//! `journal.rs` documents that it never contains prompts or generated
//! text. This module is the other kind of file: it contains *only*
//! that. It is plaintext on disk and it is what the user typed and what
//! the model said. Nothing here is written unless a client explicitly
//! asks for it -- no request to `/v1/chat/completions` is captured on
//! its way past -- so an operator who never uses the conversation API
//! stores nothing. Do not add interception here; storing a transcript
//! must stay something a caller asked for.
//!
//! ## Nothing is evicted
//!
//! The store is bounded, and every bound refuses rather than deletes:
//! too many conversations, too many messages in one, or one that has
//! grown past its byte ceiling all answer with a status and a reason.
//! An eviction the user cannot see is the specific bug this module
//! exists to not have.

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::{
    extract::Path as AxumPath,
    http::StatusCode,
    routing::{get, post},
    Extension, Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::ApiError;

// ---------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------

/// The collection.
///
/// Spelled here rather than in `frink_api::routes` on purpose: that
/// crate's stated rule is that a constant is a promise to the clients
/// that import it, and it publishes only what has settled. The
/// conversation tree's shape is expected to move as a second client
/// uses it. These move there when it stops moving, not before.
pub(crate) const CONVERSATIONS: &str = "/v1/conversations";

/// One conversation. **A template, not a literal** -- written in the
/// OpenAPI `{name}` style like every other template in this workspace,
/// and passed through [`crate::axum_path`] before it reaches the
/// router. Mounting it raw is what made `GET /v1/responses/{id}` answer
/// a bodiless 404 for every real id.
pub(crate) const CONVERSATION: &str = "/v1/conversations/{conversation_id}";

/// Deletion, as a `POST` suffix -- see the module docs for why it is
/// not the `DELETE` method.
pub(crate) const CONVERSATION_DELETE: &str = "/v1/conversations/{conversation_id}/delete";

// ---------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------

/// How many conversations one store holds. Past this, creation is
/// refused with a reason; nothing older is deleted to make room.
const MAX_CONVERSATIONS: usize = 500;

/// Nodes in one conversation, branches included.
const MAX_MESSAGES: usize = 2_000;

/// One message's text.
const MAX_CONTENT_BYTES: usize = 1 << 20;

/// One conversation's serialized size. Checked against the bytes that
/// would actually be written, so the limit means what it says.
const MAX_CONVERSATION_BYTES: usize = 8 << 20;

/// A client-minted message id. Long enough for a UUID and then some.
const MAX_MESSAGE_ID_LEN: usize = 128;

/// A title, in characters rather than bytes -- it is displayed, so the
/// unit a user would count in is the right one.
const MAX_TITLE_CHARS: usize = 200;

/// Nodes accepted in one `append`. A batch, not a bulk import.
const MAX_APPEND: usize = 256;

/// Characters of the first user message used for a derived title.
const DERIVED_TITLE_CHARS: usize = 60;

/// The roles this store accepts.
///
/// `tool` is deliberately absent. A tool message without its
/// `tool_call_id` cannot be replayed to `/v1/chat/completions`, so
/// storing one would mean keeping a message no client can send back --
/// which reads as a supported feature and is not one. Refusing it is
/// the honest answer until the tool-call round trip is stored whole.
const ROLES: [&str; 3] = ["user", "assistant", "system"];

// ---------------------------------------------------------------------
// The stored shape
// ---------------------------------------------------------------------

/// One node of the tree.
///
/// `metadata` is opaque and client-owned: the server never reads it,
/// never validates its shape, and hands it back byte-identical. Studio
/// puts the server's own `usage` block there, which is what makes the
/// TTFT / prefill / decode line under an answer survive a reload. A
/// second client can put something else there without this module
/// caring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct MessageNode {
    pub(crate) id: String,
    /// `None` for a root. Every other value names a node that already
    /// existed when this one was appended.
    #[serde(default)]
    pub(crate) parent_id: Option<String>,
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: String,
    /// A reasoning model's chain of thought, kept BESIDE `content` the
    /// way `/v1/chat/completions` returns it, never folded in.
    ///
    /// An R1 answer that ran out of `max_tokens` while thinking is
    /// entirely `reasoning_content`; stored as `content: ""` it came
    /// back from a reload as an empty assistant turn, with the thinking
    /// gone. Absent on the wire and on disk when there is none, so a
    /// record written before this field existed reads back unchanged
    /// (`default`), and a client that does not think never sees the
    /// key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    /// How long the model thought, in milliseconds: the wall-clock
    /// between the first `reasoning_content` delta and the first
    /// `content` delta, as the client that consumed the stream saw it.
    /// Stored here rather than derived from `usage` because usage
    /// measures decode, and a thought's length in seconds is not a
    /// decode figure. Only ever beside a `reasoning_content`; a
    /// duration for a turn that did not think is dropped on append.
    /// Absent on records from before it existed (`default`), which
    /// then show their thought with no time on it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_ms: Option<u64>,
    /// Stamped by the server on append. A client-supplied time would be
    /// a claim; this is a fact about when the server was told.
    pub(crate) created_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) metadata: Option<serde_json::Value>,
}

/// One conversation, as stored on disk.
///
/// No `object` field here: that is a wire convention, not a fact worth
/// writing to every file. [`ConversationBody`] adds it on the way out.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Conversation {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) title: Option<String>,
    /// The model the client says this conversation is being held with.
    /// Recorded, never enforced -- this server decodes against whatever
    /// is loaded, and pretending otherwise would make the transcript
    /// disagree with what happened.
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    /// The currently selected leaf, so a branch choice survives a
    /// reload.
    #[serde(default)]
    pub(crate) head_id: Option<String>,
    #[serde(default)]
    pub(crate) messages: Vec<MessageNode>,
}

impl Conversation {
    fn summary(&self) -> Summary {
        Summary {
            object: "conversation.summary",
            id: self.id.clone(),
            title: self.title.clone(),
            model: self.model.clone(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            head_id: self.head_id.clone(),
            message_count: self.messages.len(),
        }
    }

    fn has(&self, id: &str) -> bool {
        self.messages.iter().any(|m| m.id == id)
    }
}

// ---------------------------------------------------------------------
// Wire bodies
// ---------------------------------------------------------------------

/// A conversation plus its `object` tag.
#[derive(Debug, Serialize)]
struct ConversationBody {
    object: &'static str,
    #[serde(flatten)]
    conversation: Conversation,
}

impl From<Conversation> for ConversationBody {
    fn from(conversation: Conversation) -> Self {
        Self {
            object: "conversation",
            conversation,
        }
    }
}

/// A listed conversation, without its messages.
///
/// Tagged `conversation.summary` rather than `conversation` on purpose:
/// it is missing the `messages` array, and a client that mistook one
/// for the other would render an empty transcript and believe it. Two
/// shapes, two names.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Summary {
    object: &'static str,
    id: String,
    title: Option<String>,
    model: Option<String>,
    created_at: u64,
    updated_at: u64,
    head_id: Option<String>,
    message_count: usize,
}

#[derive(Debug, Serialize)]
struct ListBody {
    object: &'static str,
    data: Vec<Summary>,
}

#[derive(Debug, Serialize)]
struct DeletedBody {
    object: &'static str,
    id: String,
    deleted: bool,
}

/// A message on the way in.
///
/// Separate from [`MessageNode`] so the set of things a client is
/// allowed to state is visible in the type: no `created_at` here, which
/// is what makes "the server stamps the time" structural rather than a
/// line of code somebody could delete.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct NewMessage {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) parent_id: Option<String>,
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: String,
    /// See [`MessageNode::reasoning_content`]. An empty string is
    /// stored as absent: the field means "this turn thought", and an
    /// empty thought is no thought.
    #[serde(default)]
    pub(crate) reasoning_content: Option<String>,
    /// See [`MessageNode::reasoning_ms`]. Kept only when the turn has a
    /// thought to attach it to.
    #[serde(default)]
    pub(crate) reasoning_ms: Option<u64>,
    #[serde(default)]
    pub(crate) metadata: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct CreateRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    head_id: Option<String>,
    #[serde(default)]
    messages: Vec<NewMessage>,
}

/// Every field optional, and absent means "leave it alone".
///
/// There is deliberately no way to *clear* the title back to `null`:
/// send an empty string for that. Distinguishing absent from
/// explicitly-null costs a nested `Option` on the wire and buys a
/// distinction no caller has asked for.
#[derive(Debug, Default, Deserialize)]
pub(crate) struct UpdateRequest {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    head_id: Option<String>,
    #[serde(default)]
    append: Vec<NewMessage>,
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// Everything this store can refuse, with the reason kept intact.
///
/// The status codes are load-bearing for the UI: a `404` on the
/// collection means "this build has no conversation API" and sends
/// Studio back to its browser-local transcript, while a `404` on one
/// conversation means "that one is gone". They must not collapse into
/// one message.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StoreError {
    NotFound,
    Invalid { code: &'static str, message: String },
    TooLarge { code: &'static str, message: String },
    Full(String),
    Io(String),
}

impl StoreError {
    fn invalid(code: &'static str, message: impl Into<String>) -> Self {
        Self::Invalid {
            code,
            message: message.into(),
        }
    }

    fn too_large(code: &'static str, message: impl Into<String>) -> Self {
        Self::TooLarge {
            code,
            message: message.into(),
        }
    }

    fn into_api_error(self) -> ApiError {
        let (status, code, message) = match self {
            Self::NotFound => (
                StatusCode::NOT_FOUND,
                "conversation_not_found",
                "no conversation with that id is stored on this server".to_string(),
            ),
            Self::Invalid { code, message } => (StatusCode::BAD_REQUEST, code, message),
            Self::TooLarge { code, message } => (StatusCode::PAYLOAD_TOO_LARGE, code, message),
            Self::Full(message) => (StatusCode::INSUFFICIENT_STORAGE, "store_full", message),
            Self::Io(message) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "conversation_store_unwritable",
                message,
            ),
        };
        (
            status,
            Json(json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": code,
                }
            })),
        )
    }
}

type StoreResult<T> = Result<T, StoreError>;

// ---------------------------------------------------------------------
// Ids
// ---------------------------------------------------------------------

const ID_PREFIX: &str = "conv_";

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-process stamp, so ids minted by two runs of the same binary in
/// the same directory cannot collide.
///
/// It is `frink_api::request_id`'s, not a second implementation of the
/// same idea: this file used to carry a byte-identical copy whose own
/// comment said it mirrored that one.
use frink_api::request_id::process_stamp;

fn next_conversation_id() -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "{ID_PREFIX}{:012x}{:06x}",
        process_stamp() & 0xffff_ffff_ffff,
        n
    )
}

/// Whether a string is shaped like an id this server minted.
///
/// Used on the *load* path, where filenames come off a disk this
/// process does not exclusively own. It is not what stops path
/// traversal -- nothing built here ever joins a client-supplied string
/// onto the store directory; every path is built from the id inside a
/// [`Conversation`] the server itself minted. That is a structural
/// property, and this check is defence in depth behind it.
pub(crate) fn is_conversation_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix(ID_PREFIX) else {
        return false;
    };
    !rest.is_empty() && rest.len() <= 32 && rest.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------
// The store
// ---------------------------------------------------------------------

/// Every conversation this server holds, in memory and on disk.
///
/// The map is the truth for reads; the files are the truth across
/// restarts. A mutation writes the file *first* and only then commits
/// to the map, so a failed write leaves memory and disk agreeing with
/// each other and the caller holding an error. The alternative --
/// commit then write -- would answer `200` for a conversation that is
/// not stored anywhere, which is the exact lie this module exists to
/// avoid.
///
/// One `Mutex` around the whole map, held across the file write. These
/// are small local files and the store serves one user's Chat screen;
/// a finer-grained lock would buy nothing and would let two writers
/// race on one conversation's file.
pub(crate) struct ConversationStore {
    dir: PathBuf,
    inner: Mutex<BTreeMap<String, Conversation>>,
}

impl ConversationStore {
    /// The store the server runs with: `FRINK_CONVERSATIONS_DIR`, or
    /// `./frink-conversations` beside the working directory. Loads
    /// whatever is already there.
    pub(crate) fn from_env() -> Self {
        Self::open(default_dir())
    }

    /// Open a store over `dir`, reading back everything already in it.
    ///
    /// A missing directory is an empty store, not an error: it is
    /// created by the first write and not before, so a server nobody
    /// stores a conversation on leaves nothing behind.
    pub(crate) fn open(dir: PathBuf) -> Self {
        let inner = Mutex::new(load_dir(&dir));
        Self { dir, inner }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<String, Conversation>> {
        // A panic inside a handler while holding this lock would poison
        // it. The stored map is a plain value with no half-applied
        // state -- every mutation is built and validated before it is
        // committed -- so recovering is correct here, and strictly
        // better than turning one panicked request into a permanently
        // dead conversation API.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Summaries, most recently updated first.
    ///
    /// Ties break on id, descending, so two conversations touched in
    /// the same second still come back in a stable order rather than
    /// whichever the map happened to yield.
    pub(crate) fn list(&self) -> Vec<Summary> {
        let map = self.lock();
        let mut out: Vec<Summary> = map.values().map(Conversation::summary).collect();
        out.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| b.id.cmp(&a.id))
        });
        out
    }

    pub(crate) fn get(&self, id: &str) -> StoreResult<Conversation> {
        self.lock().get(id).cloned().ok_or(StoreError::NotFound)
    }

    pub(crate) fn create(&self, request: CreateRequest) -> StoreResult<Conversation> {
        let mut map = self.lock();
        if map.len() >= MAX_CONVERSATIONS {
            return Err(StoreError::Full(format!(
                "this server stores at most {MAX_CONVERSATIONS} conversations and is holding \
                 {}; delete one before creating another (nothing is evicted automatically)",
                map.len()
            )));
        }
        let now = crate::unix_now();
        let mut conversation = Conversation {
            id: next_conversation_id(),
            title: validated_title(request.title)?,
            model: request.model,
            created_at: now,
            updated_at: now,
            head_id: None,
            messages: Vec::new(),
        };
        append_messages(&mut conversation, request.messages, now)?;
        set_head(&mut conversation, request.head_id)?;
        derive_title(&mut conversation);
        check_size(&conversation)?;

        self.persist(&conversation)?;
        map.insert(conversation.id.clone(), conversation.clone());
        Ok(conversation)
    }

    pub(crate) fn update(&self, id: &str, request: UpdateRequest) -> StoreResult<Conversation> {
        let mut map = self.lock();
        let current = map.get(id).ok_or(StoreError::NotFound)?;
        let mut next = current.clone();
        let mut changed = false;

        if let Some(title) = validated_title(request.title)? {
            if next.title.as_deref() != Some(title.as_str()) {
                next.title = Some(title);
                changed = true;
            }
        }
        if let Some(model) = request.model {
            if next.model.as_deref() != Some(model.as_str()) {
                next.model = Some(model);
                changed = true;
            }
        }
        if !request.append.is_empty() {
            append_messages(&mut next, request.append, crate::unix_now())?;
            changed = true;
        }
        if request.head_id.is_some() {
            let before = next.head_id.clone();
            set_head(&mut next, request.head_id)?;
            changed |= before != next.head_id;
        }
        if derive_title(&mut next) {
            changed = true;
        }

        // Nothing to write means nothing to write. A UI that polls with
        // an empty update must not rewrite the file and bump
        // `updated_at`, which would reorder the list on every tick.
        if !changed {
            return Ok(next);
        }

        next.updated_at = crate::unix_now();
        check_size(&next)?;
        self.persist(&next)?;
        map.insert(next.id.clone(), next.clone());
        Ok(next)
    }

    pub(crate) fn delete(&self, id: &str) -> StoreResult<()> {
        let mut map = self.lock();
        let conversation = map.get(id).ok_or(StoreError::NotFound)?;
        let path = self.file_for(&conversation.id);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            // Already gone on disk is the state we were asking for.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(StoreError::Io(format!(
                    "could not delete {}: {e}",
                    path.display()
                )))
            }
        }
        map.remove(id);
        Ok(())
    }

    /// The file holding one conversation.
    ///
    /// Takes the id off a stored [`Conversation`], never off a request.
    /// That is what makes traversal impossible here rather than merely
    /// unlikely: there is no code path from a client-supplied string to
    /// a filesystem path.
    fn file_for(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    /// Write-temp, fsync, rename.
    ///
    /// The `sync_all` before the rename is what makes the rename mean
    /// anything: without it the directory entry can land ahead of the
    /// bytes, and a crash leaves a file that exists and is empty --
    /// which reads as "the conversation is there but has no messages",
    /// the most confusing of the possible failures.
    fn persist(&self, conversation: &Conversation) -> StoreResult<()> {
        let io = |what: &str, e: std::io::Error| {
            StoreError::Io(format!(
                "conversation store at {} is not writable ({what}: {e})",
                self.dir.display()
            ))
        };
        std::fs::create_dir_all(&self.dir).map_err(|e| io("create directory", e))?;
        let bytes = serde_json::to_vec_pretty(conversation)
            .map_err(|e| StoreError::Io(format!("could not serialize conversation: {e}")))?;
        let final_path = self.file_for(&conversation.id);
        let tmp_path = self.dir.join(format!("{}.json.tmp", conversation.id));
        {
            let mut file = std::fs::File::create(&tmp_path).map_err(|e| io("create", e))?;
            file.write_all(&bytes).map_err(|e| io("write", e))?;
            file.sync_all().map_err(|e| io("sync", e))?;
        }
        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            std::fs::remove_file(&tmp_path).ok();
            io("rename", e)
        })
    }
}

fn default_dir() -> PathBuf {
    std::env::var("FRINK_CONVERSATIONS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./frink-conversations"))
}

/// Read every `conv_*.json` in `dir`.
///
/// A file that will not parse is skipped and logged, never deleted and
/// never rewritten: the bytes stay on disk for a human to look at. A
/// file whose inner `id` disagrees with its name is skipped too, since
/// one of the two is wrong and guessing which would rename someone's
/// conversation.
fn load_dir(dir: &PathBuf) -> BTreeMap<String, Conversation> {
    let mut map = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return map;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if path.extension().and_then(|e| e.to_str()) != Some("json") || !is_conversation_id(stem) {
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("conversation {} could not be read: {e}", path.display());
                continue;
            }
        };
        match serde_json::from_str::<Conversation>(&text) {
            Ok(conversation) if conversation.id == stem => {
                map.insert(conversation.id.clone(), conversation);
            }
            Ok(conversation) => tracing::warn!(
                "conversation file {} holds id {:?}; skipped rather than guessing which is right",
                path.display(),
                conversation.id
            ),
            Err(e) => tracing::warn!(
                "conversation file {} is not readable as a conversation ({e}); left in place, \
                 not loaded",
                path.display()
            ),
        }
        if map.len() >= MAX_CONVERSATIONS {
            tracing::warn!(
                "conversation store at {} holds more than {MAX_CONVERSATIONS} files; the rest \
                 are on disk but not loaded",
                dir.display()
            );
            break;
        }
    }
    map
}

// ---------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------

fn validated_title(title: Option<String>) -> StoreResult<Option<String>> {
    let Some(title) = title else { return Ok(None) };
    if title.chars().count() > MAX_TITLE_CHARS {
        return Err(StoreError::too_large(
            "title_too_long",
            format!("a title is at most {MAX_TITLE_CHARS} characters"),
        ));
    }
    Ok(Some(title))
}

/// Append a batch, checking every rule before any of it is committed.
///
/// Operates on a clone of the conversation, so a batch whose fourth
/// message is bad leaves the first three unstored as well. Half a batch
/// is worse than none: the client would have to diff to find out what
/// landed.
fn append_messages(
    conversation: &mut Conversation,
    messages: Vec<NewMessage>,
    now: u64,
) -> StoreResult<()> {
    if messages.len() > MAX_APPEND {
        return Err(StoreError::too_large(
            "too_many_messages",
            format!("at most {MAX_APPEND} messages may be appended in one request"),
        ));
    }
    if conversation.messages.len() + messages.len() > MAX_MESSAGES {
        return Err(StoreError::too_large(
            "conversation_full",
            format!(
                "a conversation holds at most {MAX_MESSAGES} messages (branches included); \
                 start a new conversation rather than losing the old one"
            ),
        ));
    }

    let existing: HashSet<String> = conversation.messages.iter().map(|m| m.id.clone()).collect();
    let mut staged: Vec<MessageNode> = Vec::with_capacity(messages.len());
    // Ids of the nodes staged so far, so a batch can carry a parent and
    // its child in one request -- which is exactly what a turn is.
    let mut staged_ids: HashSet<String> = HashSet::new();

    for message in messages {
        if message.id.is_empty() || message.id.len() > MAX_MESSAGE_ID_LEN {
            return Err(StoreError::invalid(
                "invalid_message_id",
                format!("a message id must be 1..={MAX_MESSAGE_ID_LEN} bytes"),
            ));
        }
        if existing.contains(&message.id) || staged_ids.contains(&message.id) {
            return Err(StoreError::invalid(
                "duplicate_message_id",
                format!(
                    "message {:?} is already in this conversation; appending is not an update",
                    message.id
                ),
            ));
        }
        if !ROLES.contains(&message.role.as_str()) {
            return Err(StoreError::invalid(
                "unsupported_role",
                format!(
                    "role {:?} is not one of {}; a message this server cannot replay to \
                     /v1/chat/completions is refused rather than stored",
                    message.role,
                    ROLES.join(", ")
                ),
            ));
        }
        if message.content.len() > MAX_CONTENT_BYTES {
            return Err(StoreError::too_large(
                "message_too_large",
                format!("a message holds at most {MAX_CONTENT_BYTES} bytes of content"),
            ));
        }
        // The same ceiling, applied to the thought on its own rather
        // than to the sum: a long thought must not make a short answer
        // unstorable, and vice versa. The conversation-wide byte
        // ceiling still bounds the two together.
        let reasoning_content = message.reasoning_content.filter(|r| !r.is_empty());
        // The clock goes with the thought. A duration on a turn with
        // no reasoning is a number about nothing, and storing it would
        // make a reload show "Thought for 15 seconds" over no thought.
        let reasoning_ms = message.reasoning_ms.filter(|_| reasoning_content.is_some());
        if reasoning_content
            .as_ref()
            .is_some_and(|r| r.len() > MAX_CONTENT_BYTES)
        {
            return Err(StoreError::too_large(
                "message_too_large",
                format!("a message holds at most {MAX_CONTENT_BYTES} bytes of reasoning_content"),
            ));
        }
        if let Some(parent) = message.parent_id.as_deref() {
            if !existing.contains(parent) && !staged_ids.contains(parent) {
                return Err(StoreError::invalid(
                    "unknown_parent",
                    format!(
                        "message {:?} names parent {parent:?}, which is not in this \
                         conversation; a transcript that cannot be walked back to a root \
                         cannot be replayed",
                        message.id
                    ),
                ));
            }
        }
        staged_ids.insert(message.id.clone());
        staged.push(MessageNode {
            id: message.id,
            parent_id: message.parent_id,
            role: message.role,
            content: message.content,
            reasoning_content,
            reasoning_ms,
            created_at: now,
            metadata: message.metadata,
        });
    }

    conversation.messages.extend(staged);
    Ok(())
}

fn set_head(conversation: &mut Conversation, head_id: Option<String>) -> StoreResult<()> {
    let Some(head) = head_id else { return Ok(()) };
    // An empty string clears the selection, which is what a client with
    // nothing selected has to be able to say.
    if head.is_empty() {
        conversation.head_id = None;
        return Ok(());
    }
    if !conversation.has(&head) {
        return Err(StoreError::invalid(
            "unknown_head",
            format!("head_id {head:?} names no message in this conversation"),
        ));
    }
    conversation.head_id = Some(head);
    Ok(())
}

/// Give an untitled conversation the first line of its first user
/// message.
///
/// Server-side rather than client-side so that a transcript stored by
/// something other than Studio still lists with a readable name.
/// Returns whether anything changed.
fn derive_title(conversation: &mut Conversation) -> bool {
    if conversation.title.is_some() {
        return false;
    }
    let Some(first) = conversation
        .messages
        .iter()
        .find(|m| m.role == "user" && !m.content.trim().is_empty())
    else {
        return false;
    };
    let line: String = first
        .content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    conversation.title = Some(truncate_chars(&line, DERIVED_TITLE_CHARS));
    true
}

/// Cut on a character boundary, with an ellipsis when anything was cut.
/// Byte slicing would panic on the first multi-byte character, and a
/// title is the field most likely to hold one.
fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

/// The ceiling is checked against the bytes that would be written, not
/// against a running total of the parts, so it means exactly what it
/// says on disk.
fn check_size(conversation: &Conversation) -> StoreResult<()> {
    let size = serde_json::to_vec(conversation)
        .map(|b| b.len())
        .unwrap_or(0);
    if size > MAX_CONVERSATION_BYTES {
        return Err(StoreError::too_large(
            "conversation_too_large",
            format!(
                "this conversation would be {size} bytes and the ceiling is \
                 {MAX_CONVERSATION_BYTES}; nothing was stored and nothing was dropped"
            ),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

type Store = Extension<Arc<ConversationStore>>;

async fn list(Extension(store): Store) -> Json<ListBody> {
    Json(ListBody {
        object: "list",
        data: store.list(),
    })
}

async fn create(
    Extension(store): Store,
    Json(request): Json<CreateRequest>,
) -> Result<(StatusCode, Json<ConversationBody>), ApiError> {
    let conversation = store.create(request).map_err(StoreError::into_api_error)?;
    Ok((StatusCode::CREATED, Json(conversation.into())))
}

async fn fetch(
    Extension(store): Store,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ConversationBody>, ApiError> {
    let conversation = store.get(&id).map_err(StoreError::into_api_error)?;
    Ok(Json(conversation.into()))
}

async fn update(
    Extension(store): Store,
    AxumPath(id): AxumPath<String>,
    Json(request): Json<UpdateRequest>,
) -> Result<Json<ConversationBody>, ApiError> {
    let conversation = store
        .update(&id, request)
        .map_err(StoreError::into_api_error)?;
    Ok(Json(conversation.into()))
}

async fn delete(
    Extension(store): Store,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<DeletedBody>, ApiError> {
    store.delete(&id).map_err(StoreError::into_api_error)?;
    Ok(Json(DeletedBody {
        object: "conversation.deleted",
        id,
        deleted: true,
    }))
}

/// The conversation routes, over the store `FRINK_CONVERSATIONS_DIR`
/// names.
///
/// Generic over the router's state and carrying the store as an
/// `Extension` instead of reaching into `AppState`: the store has
/// nothing to do with the loaded model, so it does not belong in the
/// state that guards one, and this keeps `lib.rs`'s diff to the module
/// declaration and one `.merge`.
pub(crate) fn router<S>() -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router_with(Arc::new(ConversationStore::from_env()))
}

/// [`router`] over a caller-owned store, which is what the tests below
/// mount so they never touch the process-wide default directory.
pub(crate) fn router_with<S>(store: Arc<ConversationStore>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    Router::new()
        .route(CONVERSATIONS, get(list).post(create))
        .route(&crate::axum_path(CONVERSATION), get(fetch).post(update))
        .route(&crate::axum_path(CONVERSATION_DELETE), post(delete))
        .layer(Extension(store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serde_json::Value;
    use tower::ServiceExt;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "frink_conversations_{tag}_{}_{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::remove_dir_all(&dir).ok();
            std::fs::remove_file(&dir).ok();
            TempDir(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
            std::fs::remove_file(&self.0).ok();
        }
    }

    fn store(dir: &TempDir) -> Arc<ConversationStore> {
        Arc::new(ConversationStore::open(dir.0.clone()))
    }

    /// The concrete paths, spelled the way a client would build them.
    ///
    /// Here rather than beside the constants because nothing in the
    /// server itself ever needs a concrete conversation path -- it
    /// mounts templates. `frink_api::routes` grows the published
    /// `v1_conversation()` when a Rust client needs one; until then a
    /// helper no caller uses is a promise with nobody to keep it.
    fn conversation_path(id: &str) -> String {
        CONVERSATION.replace("{conversation_id}", id)
    }

    fn conversation_delete_path(id: &str) -> String {
        CONVERSATION_DELETE.replace("{conversation_id}", id)
    }

    fn msg(id: &str, parent: Option<&str>, role: &str, content: &str) -> NewMessage {
        NewMessage {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            reasoning_ms: None,
            metadata: None,
        }
    }

    fn created(messages: Vec<NewMessage>) -> CreateRequest {
        CreateRequest {
            messages,
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------
    // The store
    // -----------------------------------------------------------------

    #[test]
    fn a_created_conversation_is_readable_and_has_a_server_minted_id() {
        let dir = TempDir::new("create");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "hello")]))
            .unwrap();
        assert!(is_conversation_id(&conversation.id), "{}", conversation.id);
        assert_eq!(store.get(&conversation.id).unwrap(), conversation);
        assert_eq!(conversation.messages[0].role, "user");
    }

    #[test]
    fn the_store_survives_a_restart() {
        let dir = TempDir::new("restart");
        let id = {
            let store = store(&dir);
            store
                .create(created(vec![msg("m1", None, "user", "remember me")]))
                .unwrap()
                .id
        };
        // A fresh store over the same directory: this is what a server
        // restart looks like, and it is the entire point of storing the
        // transcript here rather than in a browser.
        let reopened = store(&dir);
        let conversation = reopened.get(&id).unwrap();
        assert_eq!(conversation.messages[0].content, "remember me");
        assert_eq!(reopened.list().len(), 1);
    }

    #[test]
    fn a_branch_keeps_both_answers() {
        let dir = TempDir::new("branch");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![
                msg("u1", None, "user", "hi"),
                msg("a1", Some("u1"), "assistant", "first answer"),
            ]))
            .unwrap();
        // Regenerate: a second assistant node under the same parent.
        let updated = store
            .update(
                &conversation.id,
                UpdateRequest {
                    append: vec![msg("a2", Some("u1"), "assistant", "second answer")],
                    head_id: Some("a2".to_string()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(updated.messages.len(), 3);
        assert_eq!(updated.head_id.as_deref(), Some("a2"));
        assert!(
            updated.messages.iter().any(|m| m.content == "first answer"),
            "regenerating must branch, not overwrite"
        );
    }

    #[test]
    fn a_dangling_parent_is_refused_rather_than_stored() {
        let dir = TempDir::new("dangling");
        let store = store(&dir);
        let err = store
            .create(created(vec![msg("a1", Some("nobody"), "assistant", "x")]))
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::Invalid {
                code: "unknown_parent",
                ..
            }
        ));
    }

    #[test]
    fn a_batch_can_carry_a_parent_and_its_child() {
        let dir = TempDir::new("batch");
        let store = store(&dir);
        // One turn is two nodes, and the client sends them together.
        let conversation = store
            .create(created(vec![
                msg("u1", None, "user", "hi"),
                msg("a1", Some("u1"), "assistant", "hello"),
            ]))
            .unwrap();
        assert_eq!(conversation.messages.len(), 2);
    }

    #[test]
    fn a_bad_message_leaves_the_whole_batch_unstored() {
        let dir = TempDir::new("atomic");
        let store = store(&dir);
        let conversation = store.create(CreateRequest::default()).unwrap();
        let err = store
            .update(
                &conversation.id,
                UpdateRequest {
                    append: vec![
                        msg("u1", None, "user", "kept?"),
                        msg("a1", Some("ghost"), "assistant", "no"),
                    ],
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, StoreError::Invalid { .. }));
        assert!(
            store.get(&conversation.id).unwrap().messages.is_empty(),
            "half a batch would make the client diff to find out what landed"
        );
    }

    #[test]
    fn a_duplicate_message_id_is_refused_because_appending_is_not_updating() {
        let dir = TempDir::new("dupe");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "one")]))
            .unwrap();
        let err = store
            .update(
                &conversation.id,
                UpdateRequest {
                    append: vec![msg("m1", None, "user", "two")],
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::Invalid {
                code: "duplicate_message_id",
                ..
            }
        ));
        assert_eq!(
            store.get(&conversation.id).unwrap().messages[0].content,
            "one"
        );
    }

    #[test]
    fn an_unknown_head_is_refused() {
        let dir = TempDir::new("head");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "one")]))
            .unwrap();
        assert!(store
            .update(
                &conversation.id,
                UpdateRequest {
                    head_id: Some("elsewhere".into()),
                    ..Default::default()
                },
            )
            .is_err());
    }

    #[test]
    fn a_role_this_server_cannot_replay_is_refused() {
        let dir = TempDir::new("role");
        let store = store(&dir);
        let err = store
            .create(created(vec![msg("m1", None, "tool", "{}")]))
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::Invalid {
                code: "unsupported_role",
                ..
            }
        ));
    }

    #[test]
    fn the_server_stamps_the_time_and_the_client_cannot() {
        let dir = TempDir::new("time");
        let store = store(&dir);
        // `NewMessage` has no `created_at` field at all, so a client
        // cannot state one; this asserts the value that comes back is a
        // real clock reading rather than a zero default.
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "hi")]))
            .unwrap();
        assert!(conversation.messages[0].created_at > 1_700_000_000);
    }

    #[test]
    fn an_untitled_conversation_takes_its_first_user_line() {
        let dir = TempDir::new("title");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg(
                "m1",
                None,
                "user",
                "  what   is a  \n gguf file? ",
            )]))
            .unwrap();
        assert_eq!(conversation.title.as_deref(), Some("what is a gguf file?"));
    }

    #[test]
    fn a_long_title_is_cut_on_a_character_boundary() {
        // Byte slicing here would panic rather than truncate, and a
        // title is where a multi-byte character is most likely.
        let long = "é".repeat(DERIVED_TITLE_CHARS + 10);
        let cut = truncate_chars(&long, DERIVED_TITLE_CHARS);
        assert_eq!(cut.chars().count(), DERIVED_TITLE_CHARS + 1);
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn an_explicit_title_wins_over_the_derived_one() {
        let dir = TempDir::new("explicit");
        let store = store(&dir);
        let conversation = store
            .create(CreateRequest {
                title: Some("Kernels".into()),
                messages: vec![msg("m1", None, "user", "unrelated question")],
                ..Default::default()
            })
            .unwrap();
        assert_eq!(conversation.title.as_deref(), Some("Kernels"));
    }

    #[test]
    fn an_empty_update_does_not_rewrite_the_conversation() {
        let dir = TempDir::new("noop");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "hi")]))
            .unwrap();
        let again = store
            .update(&conversation.id, UpdateRequest::default())
            .unwrap();
        // A polling client must not reorder the list on every tick.
        assert_eq!(again.updated_at, conversation.updated_at);
    }

    #[test]
    fn delete_removes_the_file_as_well_as_the_entry() {
        let dir = TempDir::new("delete");
        let store = store(&dir);
        let conversation = store
            .create(created(vec![msg("m1", None, "user", "hi")]))
            .unwrap();
        let path = dir.0.join(format!("{}.json", conversation.id));
        assert!(path.exists());
        store.delete(&conversation.id).unwrap();
        assert!(!path.exists());
        assert!(matches!(
            store.get(&conversation.id),
            Err(StoreError::NotFound)
        ));
        assert!(store.list().is_empty());
    }

    #[test]
    fn nothing_is_evicted_when_the_store_is_full() {
        let dir = TempDir::new("full");
        let store = store(&dir);
        for _ in 0..MAX_CONVERSATIONS {
            store.create(CreateRequest::default()).unwrap();
        }
        let err = store.create(CreateRequest::default()).unwrap_err();
        assert!(matches!(err, StoreError::Full(_)));
        // The refusal is the point: a store that made room by dropping
        // the oldest conversation is the bug this module exists to
        // avoid, and it would be invisible from here.
        assert_eq!(store.list().len(), MAX_CONVERSATIONS);
    }

    #[test]
    fn an_oversized_message_is_refused() {
        let dir = TempDir::new("big");
        let store = store(&dir);
        let err = store
            .create(created(vec![msg(
                "m1",
                None,
                "user",
                &"x".repeat(MAX_CONTENT_BYTES + 1),
            )]))
            .unwrap_err();
        assert!(matches!(
            err,
            StoreError::TooLarge {
                code: "message_too_large",
                ..
            }
        ));
    }

    #[test]
    fn a_failed_write_is_not_reported_as_a_stored_conversation() {
        let dir = TempDir::new("unwritable");
        // A plain file where the store wants a directory: `create_dir_all`
        // fails, so `persist` fails, so nothing may be committed.
        std::fs::write(&dir.0, b"not a directory").unwrap();
        let store = store(&dir);
        let err = store
            .create(created(vec![msg("m1", None, "user", "hi")]))
            .unwrap_err();
        assert!(matches!(err, StoreError::Io(_)), "{err:?}");
        assert!(
            store.list().is_empty(),
            "memory must not hold a conversation that never reached disk"
        );
    }

    /// The defect: an R1 answer cut off inside its thought is ALL
    /// `reasoning_content`, and a store with no such field kept an
    /// empty assistant turn. The thought now survives a restart, and
    /// an empty one is stored as nothing rather than as `""`.
    #[test]
    fn a_chain_of_thought_survives_a_restart_beside_its_answer() {
        let dir = TempDir::new("reasoning");
        let id = {
            let store = store(&dir);
            let mut thought = msg("a1", Some("u1"), "assistant", "");
            thought.reasoning_content = Some("Let me think about".to_string());
            let mut blank = msg("a2", Some("a1"), "assistant", "answer");
            blank.reasoning_content = Some(String::new());
            store
                .create(created(vec![
                    msg("u1", None, "user", "why"),
                    thought,
                    blank,
                ]))
                .unwrap()
                .id
        };
        let conversation = store(&dir).get(&id).unwrap();
        assert_eq!(
            conversation.messages[1].reasoning_content.as_deref(),
            Some("Let me think about")
        );
        assert_eq!(conversation.messages[1].content, "");
        assert_eq!(
            conversation.messages[2].reasoning_content, None,
            "an empty thought is no thought"
        );
        let on_disk = std::fs::read_to_string(dir.0.join(format!("{id}.json"))).unwrap();
        assert_eq!(
            on_disk.matches("reasoning_content").count(),
            1,
            "the key is written only where there is a thought: {on_disk}"
        );
    }

    /// The thought's clock is stored beside the thought and only there:
    /// a `reasoning_ms` on a turn with no `reasoning_content` is dropped
    /// on append rather than stored, and the key is written only where
    /// there is a duration.
    #[test]
    fn a_thoughts_duration_survives_a_restart_and_never_without_a_thought() {
        let dir = TempDir::new("reasoning-ms");
        let id = {
            let store = store(&dir);
            let mut timed = msg("a1", Some("u1"), "assistant", "answer");
            timed.reasoning_content = Some("Let me think".to_string());
            timed.reasoning_ms = Some(15_250);
            let mut untimed = msg("a2", Some("a1"), "assistant", "again");
            untimed.reasoning_content = Some("Hmm".to_string());
            let mut thoughtless = msg("a3", Some("a2"), "assistant", "plain");
            thoughtless.reasoning_ms = Some(3_000);
            let mut blank = msg("a4", Some("a3"), "assistant", "blank");
            blank.reasoning_content = Some(String::new());
            blank.reasoning_ms = Some(4_000);
            store
                .create(created(vec![
                    msg("u1", None, "user", "why"),
                    timed,
                    untimed,
                    thoughtless,
                    blank,
                ]))
                .unwrap()
                .id
        };
        let conversation = store(&dir).get(&id).unwrap();
        assert_eq!(conversation.messages[1].reasoning_ms, Some(15_250));
        assert_eq!(conversation.messages[2].reasoning_ms, None);
        assert_eq!(
            conversation.messages[3].reasoning_ms, None,
            "a duration with no thought beside it is a number about nothing"
        );
        assert_eq!(
            conversation.messages[4].reasoning_ms, None,
            "an empty thought is no thought, so it has no duration either"
        );
        let on_disk = std::fs::read_to_string(dir.0.join(format!("{id}.json"))).unwrap();
        assert_eq!(
            on_disk.matches("reasoning_ms").count(),
            1,
            "the key is written only where there is a duration: {on_disk}"
        );
    }

    /// A file written before the field existed carries no
    /// `reasoning_content` at all, and must load as it did then.
    #[test]
    fn a_record_from_before_reasoning_was_stored_loads_unchanged() {
        let dir = TempDir::new("pre-reasoning");
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(
            dir.0.join("conv_0000000000000000.json"),
            br#"{"id":"conv_0000000000000000","created_at":1,"updated_at":1,"head_id":"a1",
                "messages":[{"id":"u1","parent_id":null,"role":"user","content":"hi","created_at":1},
                {"id":"a1","parent_id":"u1","role":"assistant","content":"hello","created_at":1}]}"#,
        )
        .unwrap();
        let conversation = store(&dir).get("conv_0000000000000000").unwrap();
        assert_eq!(conversation.messages.len(), 2);
        assert_eq!(conversation.messages[1].content, "hello");
        assert_eq!(conversation.messages[1].reasoning_content, None);
        assert_eq!(conversation.messages[1].reasoning_ms, None);
    }

    #[test]
    fn an_oversized_thought_is_refused_like_an_oversized_answer() {
        let dir = TempDir::new("big-thought");
        let store = store(&dir);
        let mut thought = msg("m1", None, "assistant", "short");
        thought.reasoning_content = Some("x".repeat(MAX_CONTENT_BYTES + 1));
        let err = store.create(created(vec![thought])).unwrap_err();
        assert!(matches!(
            err,
            StoreError::TooLarge {
                code: "message_too_large",
                ..
            }
        ));
    }

    #[test]
    fn an_unreadable_file_is_skipped_and_left_alone() {
        let dir = TempDir::new("corrupt");
        std::fs::create_dir_all(&dir.0).unwrap();
        let path = dir.0.join("conv_deadbeef.json");
        std::fs::write(&path, b"{ not json").unwrap();
        let store = store(&dir);
        assert!(store.list().is_empty());
        assert!(
            path.exists(),
            "the bytes stay on disk for a human; nothing here deletes a transcript it could \
             not parse"
        );
    }

    #[test]
    fn a_file_whose_inner_id_disagrees_with_its_name_is_skipped() {
        let dir = TempDir::new("mismatch");
        std::fs::create_dir_all(&dir.0).unwrap();
        std::fs::write(
            dir.0.join("conv_00000000.json"),
            serde_json::to_vec(&Conversation {
                id: "conv_11111111".into(),
                title: None,
                model: None,
                created_at: 1,
                updated_at: 1,
                head_id: None,
                messages: vec![],
            })
            .unwrap(),
        )
        .unwrap();
        assert!(store(&dir).list().is_empty());
    }

    #[test]
    fn only_server_minted_ids_are_recognised() {
        assert!(is_conversation_id(&next_conversation_id()));
        assert!(!is_conversation_id("conv_../../etc/passwd"));
        assert!(!is_conversation_id("../conv_dead"));
        assert!(!is_conversation_id("conv_"));
        assert!(!is_conversation_id("chatcmpl-1"));
    }

    #[test]
    fn the_list_is_newest_first() {
        let dir = TempDir::new("order");
        let store = store(&dir);
        let a = store.create(CreateRequest::default()).unwrap();
        let b = store.create(CreateRequest::default()).unwrap();
        let ids: Vec<String> = store.list().into_iter().map(|s| s.id).collect();
        // Same second in a fast test, so the tie-break on id decides --
        // and it must be deterministic rather than map order.
        assert_eq!(ids, vec![b.id, a.id]);
    }

    #[test]
    fn a_summary_is_not_mistakable_for_a_conversation() {
        let dir = TempDir::new("summary");
        let store = store(&dir);
        store
            .create(created(vec![msg("m1", None, "user", "hi")]))
            .unwrap();
        let summary = &store.list()[0];
        assert_eq!(summary.object, "conversation.summary");
        assert_eq!(summary.message_count, 1);
        let json = serde_json::to_value(summary).unwrap();
        assert!(
            json.get("messages").is_none(),
            "a summary carries no transcript, so it must not be tagged as one"
        );
    }

    #[test]
    fn client_metadata_comes_back_byte_identical() {
        let dir = TempDir::new("metadata");
        let store = store(&dir);
        let custom = json!({ "custom": { "stats": { "line": "TTFT 40 ms" } } });
        let conversation = store
            .create(created(vec![NewMessage {
                metadata: Some(custom.clone()),
                ..msg("m1", None, "assistant", "hi")
            }]))
            .unwrap();
        assert_eq!(conversation.messages[0].metadata, Some(custom.clone()));
        // And across a restart, which is the case that matters: the
        // usage line under an answer is in here.
        let reopened = ConversationStore::open(dir.0.clone());
        assert_eq!(
            reopened.get(&conversation.id).unwrap().messages[0].metadata,
            Some(custom)
        );
    }

    // -----------------------------------------------------------------
    // Routing
    // -----------------------------------------------------------------

    /// The published templates and the router's patterns must describe
    /// the same paths. `lib.rs` has the same walk over its own
    /// templates; this is the conversation surface's copy of it, kept
    /// beside the constants it checks.
    #[test]
    fn no_template_reaches_the_router_with_its_braces() {
        for template in [CONVERSATION, CONVERSATION_DELETE] {
            assert!(template.contains('{'), "{template} has no placeholder");
            let mounted = crate::axum_path(template);
            assert!(
                !mounted.contains('{') && !mounted.contains('}'),
                "{template} would be mounted as {mounted}, whose braces axum reads as a \
                 literal segment"
            );
            assert!(mounted.contains(':'), "{template} lost its placeholder");
        }
        assert_eq!(
            crate::axum_path(CONVERSATION),
            "/v1/conversations/:conversation_id"
        );
        assert_eq!(
            conversation_path("conv_1"),
            crate::axum_path(CONVERSATION).replace(":conversation_id", "conv_1")
        );
        assert_eq!(
            conversation_delete_path("conv_1"),
            "/v1/conversations/conv_1/delete"
        );
    }

    /// The collection route must not swallow the item routes.
    #[test]
    fn the_item_routes_sit_under_the_collection() {
        assert!(CONVERSATION.starts_with(CONVERSATIONS));
        assert!(CONVERSATION_DELETE.starts_with(CONVERSATION));
        assert!(CONVERSATIONS.starts_with("/v1/"));
    }

    async fn call(app: &Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, body)
    }

    async fn get_path(app: &Router, path: &str) -> (StatusCode, Value) {
        call(
            app,
            Request::builder().uri(path).body(Body::empty()).unwrap(),
        )
        .await
    }

    async fn post_path(app: &Router, path: &str, body: Value) -> (StatusCode, Value) {
        call(
            app,
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
    }

    fn test_app(dir: &TempDir) -> Router {
        router_with::<()>(store(dir)).with_state(())
    }

    #[tokio::test]
    async fn the_http_surface_round_trips_a_conversation() {
        let dir = TempDir::new("http");
        let app = test_app(&dir);

        let (status, created) = post_path(
            &app,
            CONVERSATIONS,
            json!({ "messages": [{ "id": "u1", "role": "user", "content": "hi" }] }),
        )
        .await;
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(created["object"], "conversation");
        let id = created["id"].as_str().unwrap().to_string();

        let (status, listed) = get_path(&app, CONVERSATIONS).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(listed["object"], "list");
        assert_eq!(listed["data"][0]["id"], id.as_str());

        let (status, fetched) = get_path(&app, &conversation_path(&id)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(fetched["messages"][0]["content"], "hi");

        let (status, updated) = post_path(
            &app,
            &conversation_path(&id),
            json!({
                "append": [
                    { "id": "a1", "parent_id": "u1", "role": "assistant", "content": "hello" }
                ],
                "head_id": "a1",
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["head_id"], "a1");
        assert_eq!(updated["messages"].as_array().unwrap().len(), 2);

        let (status, deleted) = post_path(&app, &conversation_delete_path(&id), json!({})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(deleted["deleted"], true);

        let (status, _) = get_path(&app, &conversation_path(&id)).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    /// A real id must reach the handler rather than axum's catch-all.
    ///
    /// The distinction is the whole reason [`crate::axum_path`] exists:
    /// a route mounted with its braces intact answers every real id
    /// with an empty-bodied 404, which a client cannot tell apart from
    /// a server that does not have the endpoint at all.
    #[tokio::test]
    async fn an_unknown_id_gets_the_handler_not_a_bare_404() {
        let dir = TempDir::new("bare404");
        let app = test_app(&dir);
        let (status, body) = get_path(&app, &conversation_path("conv_nothing")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "conversation_not_found");

        let (status, body) =
            post_path(&app, &conversation_delete_path("conv_nothing"), json!({})).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "conversation_not_found");
    }

    /// An id that would escape the store directory reaches the same
    /// "no such conversation" answer as any other unknown id, and
    /// leaves no file behind. Traversal is structurally impossible here
    /// -- paths are built from stored ids only -- and this pins that.
    #[tokio::test]
    async fn a_traversal_shaped_id_is_just_an_unknown_id() {
        let dir = TempDir::new("traversal");
        let app = test_app(&dir);
        let (status, body) = get_path(&app, "/v1/conversations/..%2F..%2Fetc%2Fpasswd").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "conversation_not_found");
    }

    #[tokio::test]
    async fn a_refused_append_says_which_rule_it_broke() {
        let dir = TempDir::new("refusal");
        let app = test_app(&dir);
        let (_, created) = post_path(&app, CONVERSATIONS, json!({})).await;
        let id = created["id"].as_str().unwrap().to_string();
        let (status, body) = post_path(
            &app,
            &conversation_path(&id),
            json!({ "append": [{ "id": "a1", "parent_id": "ghost", "role": "assistant" }] }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "unknown_parent");
        assert!(body["error"]["message"].as_str().unwrap().contains("ghost"));
    }
}
