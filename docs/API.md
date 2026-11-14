# API

`frink-server` serves an HTTP API in three dialects: OpenAI's,
Anthropic's Messages, and llama-server's native routes.

A refusal is a 400 (the request is wrong) or a 501 (the server does not
implement it), with a message that names the field or the problem.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported (capability handshake, see [Health](#health)) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (JSON + SSE) |
| `POST /v1/completions` | Supported, JSON only (`stream` is not served yet). See [Legacy completions](#legacy-completions) |
| `POST /completion` · `POST /completions` | llama.cpp's native completion endpoint, JSON + its own SSE shape. Not an alias of the line above |
| `POST /v1/tokenize` · `POST /tokenize` | Supported. The unprefixed spelling is llama.cpp's, on the same handler |
| `POST /v1/detokenize` · `POST /detokenize` | Supported, same aliasing |
| `POST /apply-template` | llama-server's prompt preview: a chat completions body in, `{"prompt": ...}` out, nothing generated. Same rendering as `/v1/chat/completions` |
| `POST /v1/score` · `POST /score` | Sentence-pair scoring. One text against many, or two lists of the same length paired element by element; different lengths are a **400** naming both counts. A cross-encoder answers with its head, a bi-encoder with the cosine of two embeddings, and `frink_score_kind` says which. A generative model is a **501** |
| `POST /v1/embeddings` | Supported. A BERT/BGE encoder when one is loaded, otherwise a pool of a GGUF decoder's hidden states (see [Embeddings](#embeddings)) |
| `POST /v1/rerank` · `POST /rerank` | Cross-encoder reranking with the checkpoint's own classification head (see [Reranking](#reranking)) |
| `POST /v1/messages` | Anthropic Messages, streaming and buffered |
| `POST /v1/messages/count_tokens` | Anthropic prompt sizing, no generation |
| `POST /v1/responses` | OpenAI Responses surface (what `codex` speaks), streaming and buffered |
| `GET /v1/responses/{id}` · `POST /v1/responses/{id}/cancel` | Always 404: no responses are stored. Cancel a live generation with `POST /v1/cancel` |
| `GET /v1/stats` · `GET /v1/requests` | Live telemetry and a page of the request ring (see [Serving telemetry](#serving-telemetry)) |
| `POST /v1/cancel` | Stop a streamed generation by `request_id` (see [Cancellation](#cancellation)) |
| `GET /v1/stream/{request_id}` · `GET /v1/stream/{request_id}/poll` | Reconnect into a resumable stream, over SSE or JSON (see [Resumable streams](#resumable-streams)) |
| `GET /v1/cache/status` · `POST /v1/cache/rebuild` | KV pool geometry and re-split (see [Cache pools and a clean stop](#cache-pools-and-a-clean-stop)) |
| `GET /v1/conversations` · `POST /v1/conversations` | Server-side transcripts: list newest first, or create |
| `GET`/`POST /v1/conversations/{conversation_id}` | Read one with its messages, or rename, retarget and append |
| `POST /v1/conversations/{conversation_id}/delete` | Delete. A POST suffix, because the CORS allow-list is `GET, POST` |
| `POST /v1/admin/prepare-stop` | Close admission, seal the accounting, write the receipt |
| `POST /slots/{id_slot}?action=save\|restore\|erase` | llama.cpp's slot save/restore. Needs `--slot-save-path` and `FRINK_PREFIX_CACHE_ENTRIES` (see [Slot save and restore](#slot-save-and-restore)) |
| `GET /slots` | llama.cpp's live slot listing, one entry per continuous-batcher seat under `-np N`. Refused without seats, and under `--no-slots` |
| `GET /props` · `POST /props` | llama.cpp's global properties. `POST` is llama.cpp's refusal without `--props` (see [Server properties](#server-properties-get-props)) |
| `POST /infill` | llama.cpp's fill-in-the-middle completion, in `/completion`'s shape (see [Infill](#infill-post-infill)) |
| `GET /lora-adapters` · `POST /lora-adapters` | llama.cpp's LoRA listing and scale setting; the per-request `lora` field works on `/v1/chat/completions`, `/v1/completions` and `/completion` (see [LoRA adapters](#lora-adapters)) |
| `GET /cache/stats` · `GET /metrics` | Response-cache and prefix-cache counters, and Prometheus metrics |
| `GET /admin/models` | Checkpoint inventory from GGUF headers (see [Admin control surface](#admin-control-surface)) |
| `POST /admin/models/load` · `POST /admin/models/unload` | Swap or drop the served model. A load is a background task |
| `POST /admin/download` | Fetch one `.gguf` from the Hugging Face Hub into the model directory, as a background task |
| `GET /admin/tasks` · `POST /admin/tasks/{task_id}/cancel` | List background tasks, or ask one to stop |
| `GET /admin/stats` | Counters and the recent-request ring |
| `POST /sleep` · `POST /wake_up` · `GET /is_sleeping` | An unload that remembers the checkpoint path so the server can wake itself. Frees the KV pool, the paged store, the repack and expert caches and any device buffers. Idempotent. No checkpoint path on record is a `409`. While asleep, routes needing a model answer `503 server_sleeping` rather than `model_not_loaded`. One sleep level (weights are mmapped) |
| `GET /` | 404. The web UI in [`ui/`](../ui) is a separate app |
| Audio / images | Not supported. An `image_url` content part is a 501 |

That is the complete list. Every public path is one constant in the
`frink-api` crate, except the `/v1/conversations` routes, which are the
server's own (`frink-server/src/conversations.rs`).

## Authentication

Set `FRINK_API_KEY` and every route except `GET /health` needs the key,
`/metrics` and `/cache/stats` included (a Prometheus scraper needs the
header too). Both `Authorization: Bearer <key>` and `x-api-key: <key>`
(what the Anthropic SDKs send) are read; if both are present,
`Authorization` wins.

Request bodies are capped at axum's 2 MiB default; a larger body is a
`413` before any handler sees it.

## Chat completions fields

"All three generation wires" means `/v1/chat/completions`,
`/v1/completions` and `/completion`.

| Field | Status |
|---|---|
| `model`, `messages` | Supported |
| `max_tokens` · `max_completion_tokens` | One budget, read as aliases as llama-server does. Two that disagree are a 400 naming both. Default 32768. An explicit `0` is a 400 |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop` | Supported |
| `typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` | Supported, llama.cpp's spellings; absent means the sampler is off |
| `samplers` | Supported. The chain order, as a list of names or a `;`-separated string. Defaults to llama.cpp's default chain |
| `presence_penalty`, `frequency_penalty` | Supported |
| `stream` | Supported (SSE). `usage` rides the final chunk, so `stream_options` needs nothing more |
| `tools` / `tool_choice: none\|auto` | Supported, parsed in the checkpoint's own wire format (see [Tool-call formats](#tool-call-formats)) |
| `parallel_tool_calls` | Supported. `false` returns only the first call, in the whole answer and on the stream. Absent or `true` returns every call. Also on `/v1/responses` |
| `tool_choice: required` / named function | Supported by a lazy grammar; `muse_glimmer` is a 501 (see [Grammar-constrained decoding](#grammar-constrained-decoding)) |
| `logprobs` | On `/v1/completions` (parallel arrays plus `text_offset`) and on chat as `logprobs: true` + `top_logprobs` (OpenAI's `content[]` shape). Taken over the distribution the sampler drew from, so penalties and a grammar mask are in it; a candidate the chain removed is omitted, not `null`. Cap 5 on completions, 20 on chat; above is a 400. Such a request always misses the response cache |
| `n` (>1) | On `/v1/completions` and chat: one prefill, KV forked per choice, `prompt_tokens` counts the prompt once. Choice `i` uses `seed + i`, so choice 0 equals the `n = 1` answer. Streamed on chat, choices are interleaved and each chunk carries its index. **501** on `/completion` |
| `best_of` (>1) | Where `n` is served: generates `best_of` from one prefill and returns the `n` best by summed log-probability (not per-token mean), as upstream. `usage.completion_tokens` counts discards too. `best_of < n` is a 400 |
| `cache_salt` | Every store. The caller's prefix-cache namespace; the contiguous cache, the response cache and the paged radix tree all key on it. Only its hash is kept. Absent, empty or whitespace is the shared namespace |
| `prompt_logprobs` | On `/v1/completions`: one entry per prompt token, the first `null`, each later one with a 1-based `rank`, scored from the plain softmax. Cap 5. On the paged store such a request skips the prefix tree. **501** on the other wires |
| `logit_bias` | All three generation wires. `{"<token id>": <bias>}`, added before sampling, range `-100..100`; outside is a **400**. `{}` and `null` accepted. Cannot lift a token a grammar, JSON mode, `allowed_token_ids` or `bad_words` forbade |
| `allowed_token_ids`, `bad_words` | All three generation wires, in the grammar's mask. `bad_words` works on tokens: a word's last token is forbidden only after the tokens before it, so it is exact only for the model's own tokenization. An empty `allowed_token_ids` is a **400** |
| `echo` | On `/v1/completions`; **501** on the two message wires. Prompt and completion come back as one `text`; with `logprobs` the arrays cover both. After truncation it echoes the kept tokens |
| `truncate_prompt_tokens` | All three generation wires: keeps the prompt's last `k` tokens before prefill, so KV, `usage.prompt_tokens`, the prefix cache and `echo` see the truncated prompt. Below 1 is a **400** |
| `skip_special_tokens` | All three generation wires. `false` keeps the end-of-generation token in the text and counts it; it still ends the answer. A single-token `stop` string still stops before the token |
| `return_tokens_as_token_ids` | All three generation wires: each reported token is spelled `token_id:123` in `tokens`, the `top_logprobs` keys and chat's `content[].token`. `text` is unchanged |
| `use_beam_search`, `prompt_embeds` | 501 naming the field on all three generation wires. `use_beam_search: false` is served |
| `response_format: json_object` | Supported (character-class mask plus validation) |
| `grammar` | Supported. llama.cpp's GBNF field, enforced per token |
| `response_format: json_schema` | Supported: compiled to GBNF and enforced per token |
| Other `response_format` types | 400 naming the type, `text` included on this route |
| `session_id` | Frink extension (server-side history) |
| `stream_resumable` | Frink extension: keep a replay buffer so a client can reconnect (see [Resumable streams](#resumable-streams)) |
| `lora` | llama.cpp's per-request adapter scales (see [LoRA adapters](#lora-adapters)) |
| `chat_template_kwargs` | Supported (see [Chat templates](#chat-templates)) |
| `reasoning_effort` | Supported, quantized onto what the checkpoint grades; `none`/`off` turn thinking off |
| `thinking: {"type": …}` | Supported (DeepSeek wire): `enabled`/`disabled`, anything else is a 400. On `/v1/messages`, `thinking.budget_tokens` becomes `reasoning_budget_tokens` as llama.cpp's Anthropic lowering does (`server-chat.cpp:585-591`); `enabled` without a number gets 10,000 |
| `ignore_eos` | Frink extension: run past the model's end-of-generation tokens to produce exactly `max_tokens`. A caller's own `stop` strings still end the answer |
| `reasoning_content` (both ways) | Frink extension: chain of thought split out of `content`, and replayable on the way in (`reasoning` is an alias). See [Reasoning content](#reasoning-content) |
| `continue_final_message` | llama.cpp's field and values: `true` (auto), `"reasoning_content"`, `"content"`, or `false`. A trailing assistant message is rendered as a turn still being written, thought included. On by default, as in llama.cpp (`server-common.cpp:1046-1056`). `false` renders it as a closed turn (llama.cpp reads `false` as absence; this is the one intended difference), and `--no-prefill-assistant` turns the default off server-wide. Same rule on `/v1/chat/completions`, `/v1/messages` and `/v1/responses`. The harmony and ATEM channel formats, a content continuation for an always-open family, and a turn with tool calls are 501 by name; a 501 reached through the default says how to switch it off |
| `reasoning_budget_tokens` / `thinking_budget_tokens` | llama.cpp's sampler-level budget (`common/reasoning-budget.cpp`). `-1` or absent takes `--reasoning-budget` (default `-1`, unrestricted); `N` allows N tokens of thought after the opener, then forces the closer, so the answer still ends with `finish_reason: "stop"`; `0` forces the closer immediately. The closer is not counted, waits for a split UTF-8 character, and a second block gets a fresh budget. Below `-1` is a 400. Also accepted at the top level of a `/v1/responses` body. No effect on a checkpoint with no reasoning format; the harmony and ATEM channel formats are 501 by name |

### Where a completion stops

Besides `max_tokens` and `stop`, generation ends on any token in the
checkpoint's end-of-generation set: the `eos`/`eot`/`eom` metadata ids
plus every vocabulary entry on llama.cpp's literal EOG list
(`<|eot_id|>`, `<end_of_turn>`, `<|im_end|>`, `<turn|>`, …). Both decode
paths use the same set. A Kimi K3 checkpoint keeps its vocabulary in
`tokenizer_config.json`, so its set comes from the special-token names
and `[EOT]` ends a turn there too.

BOS follows [CLI.md](CLI.md#who-adds-bos): the template owns it when it
prints one, the loader otherwise, never twice.

### When a request does not fit the context

A prompt that fits is served with `max_tokens` clamped to the room left
under the per-request ceiling. A prompt that fills the ceiling by itself
is refused; see
[Errors that retrying will not fix](#errors-that-retrying-will-not-fix).

## Chat templates

The prompt is rendered from the checkpoint's own
`tokenizer.chat_template` (Jinja2). With no template, it falls back to
ChatML (as llama.cpp `--jinja`), or to role-labeled lines for a byte or
synthetic tokenizer. A template that does not compile, or uses a
construct the evaluator lacks, fails the request with the compiler's
message; there is no guessed framing.

`chat_template_kwargs` entries become top-level template variables
(`enable_thinking` for Qwen3 and Gemma-4, `thinking` for DeepSeek,
`preserve_thinking`). They never shadow `messages`, `tools` or
`add_generation_prompt`.

`preserve_reasoning` is `true` unless the server was started with
`--no-reasoning-preserve` or the request sets it, as in llama-server. A
boolean value also reaches the template as `preserve_thinking`, and its
opposite as `clear_thinking`, `truncate_history_thinking` and
`drop_thinking` (what GLM-4.7, LFM2.5, LFM3, Nemotron-3, Laguna and
DeepSeek-V4 read to keep earlier turns' reasoning). A request that sets
one of those four keeps its own value.

`enable_thinking` is `true` when the template supports thinking and the
server was not started with `-rea off`, as in llama-server, so a
thinking checkpoint (DeepSeek V3.1 to V4, QwQ, R1-Distill, Gemma 4,
Laguna, MiniCPM5) thinks by default. A request's `enable_thinking`,
`reasoning_effort: "none"` or `thinking: {"type": "disabled"}` turns it
off. `date_string` and `datetime` carry today's date in server local
time.

Rules applied before rendering:

* An explicit knob wins wholesale. If `chat_template_kwargs` sets any of
  `enable_thinking`, `thinking`, `thinking_mode` or `reasoning_effort`,
  the protocol-level knobs are ignored rather than merged.
* `none` and `off` turn thinking off and are never quantized to a gear.
  `thinking: {"type": "disabled"}` does the same and beats any effort.
* Effort is quantized onto the gears the template grades (probed once at
  load): a gear outside them maps to the nearest one, or is dropped
  (template default) when nothing is close. `"minimal"` against
  `low`/`medium`/`high` renders as `low`.
* The value is also broadcast as `reasoning_strength` for the
  graded-strength dialect.

`GET /v1/models` reports the probe: `supported_reasoning_efforts`
(least thinking first: `off`, `adaptive`, then the gears, or a bare `on`
for a toggle with no ladder) and `default_reasoning_effort`. A
checkpoint that says nothing about thinking carries neither field.

Offering tools leaves thinking where it was, as in llama-server. Whether
tools are described by the template or by a text preamble is decided at
load by rendering with and without a tool; see
[Tool-call formats](#tool-call-formats).

## Health

`GET /health` returns JSON and is never behind auth or rate limiting.

```jsonc
{"state": "ready",                     // ready | detecting | unavailable
 "model": {"id": "…", "tokenizer": "…", "synthetic_weights": false},
 "capabilities": [{"id": "metal", "available": false,
   "reason": "metal_not_built",        // stable code, safe to switch on
   "detail": "Apple M2 Pro is present but this binary was built without --features metal; rebuild to use it."}],
 "version": "0.63.0", "pid": 4242, "uptime_seconds": 12.5,
 "server_time_unix_ms": 1786000000000,
 "last_request_age_seconds": 0.4}      // absent until one is served
```

- `detecting`: backend probing has not finished and capabilities are
  provisional. Detection gets 1 s, then is filled in with
  `reason: "detection_timed_out"`. The handler never blocks.
- Status is 200 for `ready` and `detecting`, 503 for `unavailable`
  (no model loaded, after `POST /admin/models/unload` or `POST /sleep`;
  `reason: "model_not_loaded"`, no `model`). A non-ready state carries a
  top-level `reason` and `detail`; `ready` carries neither.
- Capability ids: `cpu`, `metal`, `cuda`, `real_weights`,
  `continuous_batching`. Each has a machine `reason` and a human
  `detail`. `metal_unavailable` (no device), `metal_not_built` (this
  binary) and `disabled` (a flag) are distinct; `cuda_*` and `cpu_only`
  follow the same pattern.
- `last_request_age_seconds` shows a busy server is alive when a
  saturated GPU starves the health handler.

## Request ids

Every `/v1/chat/completions` response carries a server-assigned
`chatcmpl-…` id as `id` and again as `request_id`: in the body when
buffered, in the first SSE chunk when streamed. It is the key frink logs
and cancels by. `/v1/messages` states it in a `request-id` response
header (see [Cancellation](#cancellation)).

## Usage timings

`usage` extends the OpenAI shape with llama.cpp-style timings, prefill
and decode reported separately.

| Field | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | OpenAI convention |
| `completion_tokens_details.reasoning_tokens` | completion tokens inside a reasoning block; present only when the reasoning split ran |
| `prompt_eval_duration_ms` | wall time on the prompt |
| `generation_duration_ms` | wall time in the decode loop |
| `time_to_first_token_ms` | prefill start to first token |
| `prompt_per_second`, `predicted_per_second` | per-phase throughput |
| `cached_tokens` | prompt tokens reused from the KV prefix cache. Absent: no prefix cache configured; `0`: consulted and missed |
| `acceptance_length` | completion tokens per target forward pass. 1.0 means speculation saved nothing, 2.0 half the forwards. A round that drafted nothing still counts |
| `draft_tokens` | draft tokens the target evaluated |
| `accepted_draft_tokens` | draft tokens kept |
| `draft_accept_rate_per_position` | entry `i` is the share of this request's rounds that offered position `i` and kept it |

Every timing is omitted, not nulled or zeroed, when it was not measured
(a cached response, a batched decode). Streamed requests get the same
`usage` on the final chunk.

### Speculative decoding

Every generic GGUF decode drafts, on the private loop and in the
continuous batch. The default drafter is prompt lookup (an n-gram match
over the request's own history, no second checkpoint). A server started
with `-md` drafts with that model (plain or DFlash) instead, with KV of
the request's own; [`CLI.md`](CLI.md#server-flags) says where that is
refused. A drafted token is kept only when it equals the server's own
sampler draw, so speculation never changes the answer.

The four speculation fields are present when a request drafted at least
one token, absent otherwise. A request does not draft when:

- it has a `grammar`, a `json_schema` or `json_object` format, or a
  forced `tool_choice`;
- its target is recurrent and it runs in the continuous batch or on the
  paged store (on the private loop's contiguous store it drafts);
- it holds a paged lease under a sliding window;
- its KV lives on the Metal device and the host rows stop at the prompt
  (the private loop with Metal attention);
- it is a streamed `n > 1` request;
- it has no room left to draft.

`/admin/stats` and `/v1/requests` rows carry `acceptance_length` and
`draft_accept_rate_per_position` under the same rule.

## Admin control surface

Everything under `/admin` needs `FRINK_API_KEY`, like `/v1/*`. Shapes
are defined in the `frink-api` crate.

| Endpoint | Answer |
|---|---|
| `GET /admin/models` | inventory (below) |
| `POST /admin/models/load` `{"id":"…"}` | `202 {"task_id":"…"}` |
| `POST /admin/models/unload` | `200 {"ok":true,"active":null}` |
| `POST /admin/download` `{"repo":"…","file":"*.gguf"}` | `202 {"task_id":"…"}` |
| `GET /admin/tasks` | every task, newest last |
| `POST /admin/tasks/{task_id}/cancel` | `200 {"ok":true}` |
| `GET /admin/stats` | counters and the recent-request ring |

`POST /sleep`, `POST /wake_up` and `GET /is_sleeping` sit at the root,
behind the same key.

### Models

```jsonc
{"model_dir": "/models",               // null when none is configured
 "active": "Qwen3-0.6B-Q4_K_M",        // null when nothing is loaded
 "models": [{"id": "Qwen3-0.6B-Q4_K_M", // file stem, the only way to name a model
   "path": "/models/Qwen3-0.6B-Q4_K_M.gguf", "size_bytes": 396705472,
   "arch": "qwen3", "quant": "Q4_K_M", "context_length": 40960, "param_count": 596049920,
   "state": "available",               // loaded | loading | available | error
   "error": null,                      // why the last load attempt failed
   "resident_bytes": null}]}
```

Read from GGUF headers only, never a weight. A field that is expensive
to establish is `null`, and the key is always present. `quant` comes
from `general.file_type` when that maps to a known name, else from the
dominant tensor dtype. `resident_bytes` is always `null` (checkpoints
are mmapped; the figure is a page-cache property).

Discovery scans `FRINK_MODEL_DIR` and the directory holding
`FRINK_MODEL_PATH`, non-recursively, for `*.gguf` plus safetensors-index
checkpoint directories. Split checkpoints fold into one entry named for
the shard prefix.

### Model swap

`POST /admin/models/load` answers at once with a task id and loads on a
blocking thread. It accepts only ids from `GET /admin/models` (unknown
is a 404); there is no load-by-path. A second load while one runs is a
`409`.

In-flight requests finish on the model they started on; the old model
is freed when the last of them ends.

After `POST /admin/models/unload`, generation endpoints answer `503`
with `"type": "model_not_loaded"`, `GET /v1/models` is an empty list,
and `/health` is `unavailable` (503). Adapters named by `--lora` are
attached to every checkpoint a load brings in, or the load is refused by
name.

### Tasks

```jsonc
{"tasks": [{"task_id": "dl-2", "kind": "download",   // download | load
  "label": "Downloading *Q4_K_M.gguf from unsloth/Qwen3-0.6B-GGUF",
  "status": "running",        // queued | running | done | error | cancelled
  "error": null, "started_at_ms": 1786717917646, "updated_at_ms": 1786717925031,
  "progress": {"fraction": 0.126, "bytes_done": 50198634, "bytes_total": 396705472,
    "rate_bytes_per_s": 9802506.2, "eta_seconds": 35.3,
    "state": "stable"}}]}     // warming | stable
```

- Timestamps are server Unix epoch milliseconds. `done`, `error` and
  `cancelled` are terminal.
- `rate_bytes_per_s` and `eta_seconds` are `null` while `state` is
  `warming` (until the rate window holds at least 3 samples spanning at
  least 3 s). `fraction` is available immediately.
- Cancellation is cooperative: the task reaches `cancelled` once a
  worker acknowledges it. A download stops within one chunk and keeps
  its `.part` file for a resume. A load cannot be interrupted mid-mmap,
  so a cancel during one discards the result. Cancelling a finished task
  is a `409`; an unknown id is a `404`.

### Download

Fetches one `.gguf` from the Hugging Face Hub into the model directory.
`file` is a literal name or a `*` glob resolved against the repo's file
list (sorted, first match, root-level only). Bytes land in
`<target>.part` and a restart resumes with `Range`, or restarts cleanly
if the server answers `200` instead of `206`.

Validation: each `repo` segment may contain only letters, digits, `-`,
`_` and `.`; `file` must be a bare `*.gguf` name (no separators, no
leading dot, no `..`, no `:`), and the joined path must be a direct child
of the model directory fixed at startup. No model directory is a `412`.
Two downloads of the same target are a `409`.

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repos, and
`HF_ENDPOINT` for a mirror.

### Stats

```jsonc
{"uptime_seconds": 43, "requests_total": 1, "errors_total": 0,
 "cache_hits": 0, "cache_misses": 1, "tokens_prompt_total": 17,
 "tokens_generated_total": 24, "last_request_age_seconds": 0.02,
 "generating_now": 0,           // decoding right now, not a queue depth
 "queue_depth": null, "queue_rejected_total": null,
 "recent": [{"request_id": "chatcmpl-b28a1aeab8f8000000",
   "at_ms": 1786718007013, "route": "/v1/chat/completions",
   "model": "Qwen3-0.6B-Q4_K_M", "status": 200,
   "prompt_tokens": 17, "completion_tokens": 24,
   "ttft_ms": 6586.2, "duration_ms": 23603, "decode_ms": 17004.8,
   "cached_tokens": 0,          // absent without a prefix cache, as in usage
   "stream": false,
   "acceptance_length": 1.6,    // absent when the request did not draft
   "draft_accept_rate_per_position": [0.81, 0.62, 0.37],
   "via_api_key": "key-4f21a0c3", "client": "frink-studio"}]}
```

- `recent` is the last 200 requests, keyed by the `request_id` the
  response carried.
- `duration_ms` includes queue wait, prefill and decode; `decode_ms` is
  the decode loop alone. `decode_ms` and `ttft_ms` are `null` when not
  timed or served from cache.
- Recorded routes: every generation route (streamed or not, rejections
  included), `/v1/embeddings`, `/v1/rerank`, `/v1/score`, tokenize and
  detokenize (under the path the client called). Tokenize and
  detokenize carry no token counts, because `tokens_prompt_total` counts
  only tokens put through a forward pass. `/v1/embeddings` counts its
  prompt tokens and has `decode_ms: null`.
- `queue_depth` and `queue_rejected_total` (rejected by the queue cap
  since start) are `null` when continuous batching is off.
- `model` is the model that answered, as `/v1/models` reports it, never
  the request's `model` string. `null` means nothing was loaded (a 503
  row).
- `via_api_key` is a per-process salted fingerprint of the
  `Authorization: Bearer` key: stable within one run, meaningless across
  restarts. `null` means no Bearer header (including `x-api-key`
  requests and servers without `FRINK_API_KEY`).
- `client` is the unauthenticated `X-Frink-Client` header, kept to 32
  label characters. Frink Studio sends `frink-studio`.

## Cache pools and a clean stop

`GET /v1/cache/status`:

```json
{"state": "ready",
 "kv": {"num_pages": 512, "used_pages": 40, "evictable_pages": 0, "page_size": 256, "num_tokens": 131072},
 "moe": null, "mamba": null, "swa": null, "resizable": ["kv"]}
```

`kv` is `null` with no shared KV pool (`FRINK_KV_POOL_BLOCKS` unset;
every request allocates privately). `moe`, `mamba` and `swa` are always
`null`: frink has no such pools.

`POST /v1/cache/rebuild` with `{"kv": <tokens>}` re-budgets the shared
KV pool on a live engine. Every check runs before anything is freed, so
a refused rebuild changes nothing. llama.cpp has no counterpart; its KV
is sized once at startup.

| Case | Answer |
|---|---|
| names `moe`, `mamba` or `swa`; missing `kv`; target below one page; no pool | 400 |
| target costs more than `FRINK_KV_BYTE_BUDGET`, when the pool was sized from one | 400 naming both byte figures. A pool sized with `FRINK_KV_POOL_BLOCKS` has no byte ceiling |
| target below the pages in-flight requests hold | 409 naming the floor in tokens |
| a rebuild already running, or a stop | 409 |
| a previous rebuild failed and latched the engine | 503 |
| success | clears the prefix cache, says so in `prefix_cache_invalidated` |

While a rebuild runs, generation requests answer 503 with
`type: "cache_rebuilding"`.

`POST /v1/admin/prepare-stop` closes admission and, once no generation
is in flight, seals the token totals:

```json
{"model_id": "…", "prompt_tokens_total": 1234, "completion_tokens_total": 5678,
 "uptime_s": 3600, "drain_complete": true}
```

Call it before signalling the process. While generations still run it
answers 503 with `drain_complete: false` and the `active` count, and
admission stays closed; retry. Sealing is idempotent, so a retry returns
the same numbers. With `FRINK_ACCOUNTING_OUTBOX` set, the totals are
also written as `<dir>/<id>.json` before the response, which then
carries `receipt_id` and `receipt_status`; a failed write is a 500 with
`retryable: true`. `drain_timeout_s` and `abort_timeout_s` are accepted
and not used. Generation requests after the call answer 503 with
`type: "server_stopping"`.

## Embeddings

Two things can answer `POST /v1/embeddings`:

| Source | How it is loaded | Default pooling | Normalized |
|---|---|---|---|
| An encoder (BERT family) | `FRINK_MODEL_PATH` at an encoder GGUF, `/admin/models/load`, or `FRINK_EMBEDDING_MODEL_PATH` beside a generative model | the checkpoint's own `pooling_type` (`CLS` for every BGE) | yes (L2) |
| A decoder's hidden states | whatever generative model is loaded | `mean` | no |

The decoder path is best effort: that model was not trained for
sentence embeddings. `embedding_type` overrides the default: `mean` and
`last` on both, `cls` on the encoder only. `none` and `rank` are refused
on both (`rank` is served by [Reranking](#reranking)).

### An encoder as the served model

`FRINK_MODEL_PATH=bge-small-en-v1.5-q8_0.gguf` works; the file's
`general.architecture` decides the loader. `GET /v1/models` then reports:

```json
{"id": "bge-small-en-v1.5", "object": "model", "frink_model_kind": "embedding",
 "frink_endpoints": ["/v1/embeddings"], "frink_n_embd": 384, "frink_pooling": "CLS",
 "frink_context_length": 512, "frink_tokenizer": "gguf-wordpiece"}
```

`frink_endpoints` lists `/v1/rerank` instead when `pooling_type` is
RANK. `GET /health` is `ready`. Every generation endpoint
(`/v1/chat/completions`, `/v1/completions`, `/completion`,
`/v1/messages`, `/v1/responses`) answers `501` naming the model:

```json
{"error": {"type": "unsupported", "param": "model",
  "message": "the loaded model 'bge-small-en-v1.5' is an embedding model (bert encoder, 384 dims, pooling CLS). An encoder has no output head, so it cannot generate text at all. There is no next token for it to predict. POST /v1/embeddings to use it, or load a generative checkpoint."}}
```

`/v1/tokenize` and `/v1/detokenize` work on an encoder. `add_special` is
a no-op there (the encoder wraps its own `[CLS]`/`[SEP]`, the tokens
`usage.prompt_tokens` counts). Every route that needs a decode refuses.

Encoder architectures that load
(`frink_models::bert_gguf_loader::ENCODER_ARCHS`): `bert`, `nomic-bert`,
`jina-bert-v2`, `jina-bert-v3`, `neo-bert` and `eurobert`. Only a
WordPiece tokenizer (`tokenizer.ggml.model = "bert"`) is supported;
another is refused by name. The other embedding architectures upstream
builds (`nomic-bert-moe`, `modern-bert`, `t5encoder`, `gemma-embedding`,
`gemma-embedding2`, `llama-embed` and two decision-head scorers) refuse
by name; `frink_models::embedding_model::NOT_YET` is that list.

## Reranking

`POST /v1/rerank`, and llama.cpp's `/rerank` on the same handler, score
with the checkpoint's own classification head (`cls` / `cls.output`).
The route never substitutes cosine similarity of embeddings.

Request `{"model"?, "query", "documents": [...], "top_n"?,
"return_documents"?}`. Response `{"object": "list", "model", "results":
[{"index", "relevance_score", "document"?}], "usage"}`, best first.

- `index` is the position in the request's `documents`.
- Ties keep request order.
- `top_n` clamps to `documents.len()`; `top_n: 0` returns no results.
  An empty `documents` or `query` is a 400.
- Needs an encoder GGUF with a rank head, through `FRINK_MODEL_PATH` or
  `FRINK_EMBEDDING_MODEL_PATH`; anything else is a 501 naming the model.
- `/v1/embeddings` against a rank-head checkpoint refuses and points
  here.

The pair is encoded as `[CLS] query [SEP] document [SEP]`, query in
segment 0 and document in segment 1, as HuggingFace does. llama.cpp
sets every token type to zero, which reorders results (on "How many
people live in Berlin?" the population document ranks first with
segments 0/1 and fifth with all zeros). A reranker whose token-type
table has fewer than two rows, or an encoder with no two-segment input
form, is refused at load.

Verified against `ms-marco-MiniLM-L6-v2`: scores match a NumPy
transcription of `BertForSequenceClassification` to 4e-3 and the order
matches HuggingFace on four query sets.

### Which head ran, and why it changes the scale

Each rerank response reports `frink_score_head`. Read it before
thresholding on an absolute score:

| `frink_score_head` | Meaning | Range on `ms-marco-MiniLM-L6-v2` |
|---|---|---|
| `classifier(tanh(pooler(cls)))` | The full head the checkpoint was trained with | about plus or minus 11 |
| `classifier(cls)` | The classifier alone, no pooler in the file | about plus or minus 0.2 |

The order is the same; the scale is not, so a `relevance_score > 0.5`
threshold from elsewhere never fires in the second regime. A load-time
note says the same to the operator.

Every published GGUF is the second regime: llama.cpp's converter drops
the pooler (`conversion/bert.py`, `BertModel.filter_tensors`). Under
llama.cpp's naming the pooler slot is `cls` and `cls.output` is the
classifier (`src/llama-graph.cpp`). frink reads `cls` when present and
never invents one. A missing pooler is a note, not a refusal, because
some checkpoints (`jina-reranker-v1-tiny-en`) really have none.

### Getting the first regime: `frink splice-pooler`

`frink splice-pooler` (see `docs/CLI.md`) writes a GGUF carrying the
pooler from the checkpoint's own safetensors. The two files are tied by
comparing their classifiers element-wise to within the GGUF's storage
precision, not by `general.name` (the published
`ms-marco-MiniLM-L6-v2-Q8_0.gguf` names the L12 model). A pooler from
another checkpoint is refused naming the element that disagrees.

Measured on `ms-marco-MiniLM-L6-v2`, four query sets, seventeen pairs,
against `scripts/rerank_reference_ms_marco.py` (`hf` rows):

| | `relevance_score` range | largest deviation from HuggingFace | orderings |
|---|---|---|---|
| converter's GGUF, `classifier(cls)` | about `-0.25 .. 0.15` | not comparable (a different function) | 4 of 4 match |
| spliced, `classifier(tanh(pooler(cls)))` | `-11.19 .. 10.93` | `0.051` | 4 of 4 match |

llama.cpp also runs the pooler on the spliced file (its `build_pooling`
RANK arm reads `cls`); its scores still differ by the all-zero token
types.

## Errors that retrying will not fix

Two ceilings do not move with load. Hitting one is a `400`, never a
`503`, because an idle server would answer the same way.

```json
{"error": {
  "message": "context_length_exceeded: prompt is too long: 9000 tokens > 8000 maximum",
  "type": "invalid_request_error", "code": "context_length_exceeded",
  "binding": "context_length_exceeded",
  "estimated_bytes": 4718592000, "limit_bytes": 4194304000,
  "positions": 9000, "positions_limit": 8000}}
```

`code` (copied in `binding`) names the ceiling:

| `code` | When | Fix |
|---|---|---|
| `context_length_exceeded` | The prompt alone reaches the per-request ceiling (`FRINK_CB_MAX_CONTEXT`, or the one derived from device memory at load). The message reads `prompt is too long: N tokens > M maximum`, the wording Claude Code and OpenClaw match on (the Anthropic wire has no code for it). A `prompt + max_tokens` that overflows the position counter gets the same code | A shorter prompt. `max_tokens` is already clamped |
| `device_memory_budget_exceeded` | The prompt plus the clamped `max_tokens` needs more KV than the whole budget (`FRINK_CB_KV_BLOCKS` under continuous batching, `FRINK_KV_POOL_BLOCKS` on the private loop) | A lower `max_tokens` or a shorter prompt; server side, more KV blocks or a smaller model |

`estimated_bytes` and `limit_bytes` are the KV cost of the request and
of the ceiling, priced as `frink inspect-plan` prints. The per-request
ceiling is checked first.

Momentary pressure is a `503` with `Retry-After`, counted separately in
`/metrics`. Every `503` carries `Retry-After: 1` unless the handler set
its own.

## Stop sequences

A stop sequence is matched at two levels:

1. Token level. A stop string that is exactly one token (`<|im_end|>`,
   `<end_of_turn>`, `<|eot_id|>`) is matched on the id before
   detokenizing; the token never enters the answer or `usage`.
2. Text level. Everything else is matched on emitted text. Only an
   actual partial match is withheld, and it is released when the answer
   ends if the match never completes.

A partial match never reaches the wire. On the OpenAI routes a stop
string reports `finish_reason: "stop"` and is trimmed from the answer.

On `/v1/messages` a caller's stop string reports
`stop_reason: "stop_sequence"` with `stop_sequence: "<the string>"`, on
the buffered body and in the terminal `message_delta`. Except:

- a stop the server added (the template's end-of-turn marker) reports
  `end_turn`;
- a truncated generation reports `max_tokens`, and one that produced
  tool calls reports `tool_use`.

## Cancellation

Both ways set the same server-side flag:

1. Drop the connection. A streaming path stops at the next token,
   unless the stream was opened with `stream_resumable` (see
   [Resumable streams](#resumable-streams)).
2. `POST /v1/cancel` with the request's id, behind `FRINK_API_KEY`. From
   a browser, send it with `keepalive: true` so it survives page unload.

```bash
curl -s http://127.0.0.1:8383/v1/cancel \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"chatcmpl-…"}'
```

The answer is `200 {"request_id":"…","cancelled":true,"detail":"…"}`
when a live generation was signalled, and `404` with `"cancelled":
false` otherwise.

A cancelled stream ends with `finish_reason: "cancelled"` (not an OpenAI
value). Tokens already decoded are kept and counted in `usage`.
Continuous-batching requests stop wherever they are (queued,
mid-prefill or decoding) at the scheduler's next step. The flag is read
between tokens, so a forward pass already running completes.

| Route | Id the cancel takes |
|---|---|
| `/v1/chat/completions`, streamed | the `chatcmpl-…` id in the first chunk |
| `/v1/messages`, streamed | the `request-id` response header; the `message_start` id is a per-message `msg_…` the cancel registry does not know |
| `/v1/responses`, `/completion`, streamed | registered, but neither wire states the id; drop the connection |
| `/v1/completions` | none; it is buffered |

## Keepalives

Every streamed endpoint sends a keepalive after 15 seconds of silence,
including the wait before the first token. Every keepalive is a data
frame, never an SSE comment, so a client's stream-idle timeout sees it.

| Endpoint | Keepalive frame |
|---|---|
| `/v1/chat/completions` | a `chat.completion.chunk` with an empty delta |
| `/v1/stream/{id}` (replay) | an id-less chunk with `choices: []` |
| `/v1/responses` | `response.in_progress` |
| `/v1/messages` | `ping` |
| `/completion`, `/infill` | a partial frame with empty `content` and `"stop": false` |

A replay never re-delivers keepalives, and the replay keepalive carries
no `id:`, so it is outside the `Last-Event-ID` sequence.

Streamed responses carry `X-Accel-Buffering: no` beside
`Cache-Control: no-cache`, so nginx-style proxies do not buffer
`text/event-stream`. Because keepalives flow every 15 s, measure a stall
timeout against bytes received, not tokens.

## Resumable streams

`id:` lines are emitted only where a replay buffer exists. A
`/v1/chat/completions` request asks for one:

```jsonc
{"model": "...", "messages": [...], "stream": true, "stream_resumable": true}
```

Every event then carries `id: {request_id}:{n}`, and the first also
carries `retry: 1500`. Two ways back in, both behind `FRINK_API_KEY`:

```bash
# Reconnect over SSE, continuing after the last event seen
curl -N http://127.0.0.1:8383/v1/stream/chatcmpl-… \
  -H 'Last-Event-ID: chatcmpl-…:41'

# Or drain the same buffer over plain JSON
curl "http://127.0.0.1:8383/v1/stream/chatcmpl-…/poll?from=42"
# {"request_id":"…","events":[{"index":42,"data":"{…}"}],
#  "next_index":43,"done":false}
```

`data` is the exact payload the stream sent, `[DONE]` included. `done`
is `true` only once the generation has ended and the buffer is drained.
A poll waits up to 10 s for the next event.

With `stream_resumable`, a dropped socket does not cancel: the
generation keeps running into the buffer, and `POST /v1/cancel` is its
stop path.

| Bound | Answer |
|---|---|
| Replay window 1 MiB per stream; oldest events dropped past it | asking for an evicted position is `410 replay_window_lost` |
| A finished stream is kept 120 s; at most 64 streams, oldest finished evicted first (a live stream never is) | an unknown or forgotten id is `404 stream_not_found` |
| `Last-Event-ID` naming a different request | `400 bad_last_event_id` |

## Continuous batching

Concurrent requests share one batched decode worker (llama.cpp's slot
model): many in-flight sequences, one `forward_multi_seq` step per tick.

Off by default on every backend. `--cont-batching` / `-cb`,
`-np N` / `--parallel N` (which also caps in-flight sequences at N) or
`FRINK_CONTINUOUS_BATCHING=1` turn it on. Without it, requests run one
at a time on the private decode loop, which is faster for one client on
Metal (it attends on the device; the batched decode attends on the
host).

It serves the generic GGUF decoder only; the dedicated engines stay on
the private loop. On the contiguous KV path, a KV pool
(`FRINK_KV_POOL_BLOCKS`) or a prefix cache (`FRINK_PREFIX_CACHE_ENTRIES`)
also keeps requests on the private loop; the paged KV store lifts that
(see [`CONFIG.md`](CONFIG.md)). Streaming emits tokens as sampled, in
the same SSE shape.

Prefill is chunked. Each tick runs one prefill chunk (round-robin across
waiting prompts) plus one decode step over every in-flight row. The
chunk is at most `FRINK_CB_PREFILL_CHUNK` (default 128), further limited
to what is left of `FRINK_CB_MAX_BATCH_TOKENS` (default 256) after the
decoding rows, with a floor of 16 prompt tokens per tick.

`FRINK_CB_MAX_SEQS` caps in-flight sequences, counting prompts still
prefilling. `FRINK_CB_MAX_QUEUE` (default 512) caps requests waiting for
admission; past it a request gets a `503` with `Retry-After`, and the
body names the queue depth, the cap and `retry_after_seconds`.
Admission is first in, first out, except that a job whose prompt the
radix prefix cache already holds may jump ahead: only the first eight
waiting jobs are considered, and no job is passed over more than four
times.

`-b N` / `--batch-size N` and `-ub N` / `--ubatch-size N` set the
prefill chunk on both decode paths (`FRINK_CB_PREFILL_CHUNK` here,
`FRINK_CHUNKED_PREFILL` on the private loop). With one prefill stage,
the two resolve as llama.cpp resolves them
(`src/llama-context.cpp:265`): the smaller of whichever was named.

## Serving telemetry

`GET /v1/stats` is one object for a status bar: `model`, `state`,
`uptime_s`, recent throughput (`throughput.decode_tps`,
`throughput.prefill_tps`), request counts and latencies
(`requests.active`, `completed`, `p95_ms`, `ttft_mean_ms`,
`prompt_tokens_total`, `completion_tokens_total`), pool gauges
(`pools`) and the live memory footprint (`memory`, `null` when it cannot
be read). The two throughput rates read `0.0` today on every reply:
nothing on either decode path feeds them yet.

`pools.kv_pages` is the shared KV pool (`used`, `total`, `evictable`,
`page_size`); `pools.prefix_cache` is what paged KV's prefix tree holds,
in tokens across every cache salt (`tokens`, `locked` by live requests,
`evictable` back to the store when it runs short, `page_size`). Each is
`null`, never a zero row, on a deployment without that pool:
`kv_pages` without `FRINK_KV_POOL_BLOCKS` or `FRINK_KV_BYTE_BUDGET`,
`prefix_cache` without `FRINK_PAGED_KV_BLOCKS`. `window_slots` and
`state_slots` are always `null`.

`GET /v1/requests?since=<cursor>&limit=<n>` pages the `/admin/stats`
request ring (rows as in [Stats](#stats)) and answers
`{"requests", "next_cursor", "missed", "total"}`. Pass `next_cursor`
back to read each row once; `missed` counts rows that left the ring
unread. `limit` defaults to, and is capped at, 200.

`GET /cache/stats` reports the response cache's counters and the prefix
cache's (`null` without `FRINK_PREFIX_CACHE_ENTRIES`).

`GET /metrics` is Prometheus text. With a batcher active it reports
`frink_prefill_chunks_total`, `frink_prefill_tokens_total`,
`frink_decode_steps_total`, `frink_scheduler_queue_depth`,
`frink_scheduler_queue_rejected_total`, and the configured caps
`frink_scheduler_max_seqs` (`-np`; 0 when unlimited) and
`frink_scheduler_prefill_chunk` (`-ub`). On every deployment:

| Family | Type | What |
|---|---|---|
| `frink_prompt_tokens_total` | counter | prompt tokens |
| `frink_prompt_tokens_cached_total` | counter | prompt tokens the prefix cache served. Over `frink_prompt_tokens_total`, the hit rate in tokens |
| `frink_generation_tokens_total` | counter | completion tokens |
| `frink_time_to_first_token_seconds` | histogram | request to first token, for the requests that timed it |
| `frink_time_per_output_token_seconds` | histogram | decode time over completion tokens |
| `frink_request_duration_seconds` | histogram | the whole request: queue wait, prefill and decode |

A histogram counts a request only when the request measured that field.

## Slot save and restore

llama.cpp's `POST /slots/{id_slot}?action=save|restore`, so a long
system prompt survives a restart. Route, `action` query, gate and
response fields are llama.cpp's (`tools/server/server.cpp:273`,
`server-context.cpp:4536-4567`, `server-task.cpp:1570-1592`).

```bash
FRINK_PREFIX_CACHE_ENTRIES=8 frink-server -m model.gguf \
  --slot-save-path ./slots --port 8383

# Prefill a prompt into the prefix cache and write ./slots/system.fslot
curl -s -X POST 'http://127.0.0.1:8383/slots/0?action=save' \
  -H 'content-type: application/json' \
  -d '{"filename":"system.fslot","prompt":"You are a terse assistant ..."}'
# {"id_slot":0,"filename":"system.fslot","n_saved":83,"n_written":5953021,"timings":{"save_ms":297.5}}
# ... restart the server ...
curl -s -X POST 'http://127.0.0.1:8383/slots/0?action=restore' \
  -H 'content-type: application/json' \
  -d '{"filename":"system.fslot"}'
# {"id_slot":0,"filename":"system.fslot","n_restored":83,"n_read":5953021,"timings":{"restore_ms":75.5}}
```

A completion whose prompt starts with those tokens then reports
`cached_tokens: 83` and prefills only the rest, with the same greedy
output as a cold server.

Differences from llama.cpp:

- `id_slot` is bookkeeping: frink builds KV per request and shares
  prefixes through one prefix cache. It is validated (a non-negative
  integer) and echoed.
- `save` takes a `prompt` to prefill (tokenized with BOS as a completion
  would be), instead of saving what the slot last served.

`action=erase` answers llama.cpp's `{"id_slot": 0, "n_erased": 83}`
(`0` if the slot held nothing) and drops the prefix cache entry for the
prompt that slot id last saved or restored. An entry another slot id
also holds stays, and so does a longer entry that only starts with the
prompt. Erase needs `--slot-save-path` and the prefix cache too.

A restore is refused when the file is not of the served model:

```
400 slot_model_mismatch
refusing to restore ./slots/system.fslot: checkpoint: the slot was saved
under 2b77f720... and this server is serving 3a59835e.... Restoring
attention state computed by other weights does not fail, it answers wrongly
```

The identity is a SHA-256 over the GGUF's metadata (sorted, typed), its
full tensor directory, and 4 KiB from the head and tail of every tensor,
beside the layer count, KV head count, head dimension and KV dtype. The
same model at another quantisation is refused on `checkpoint`; a
different model on `model` or the first differing shape field.
llama.cpp's own slot file (`src/llama-context.cpp:3081-3141`) carries no
identity.

The file is framed with a length-and-digest prefix like
`frink_core::kv_disk`'s blocks, so a truncated, edited or foreign file
is refused before parsing. Filenames are a whitelist (`[A-Za-z0-9._-]`,
no leading dot), so a name cannot leave `--slot-save-path`.

Slots work on the generic GGUF decoder. Dedicated engines and a server
without a checkpoint on disk refuse by name. A model whose layers cache
different KV widths, or whose K and V heads differ in width, is a 501.

### The live listing, `GET /slots`

An array with one object per seat of the continuous batcher (`-np N`
gives N); the first ones, as many as sequences decoding or prefilling,
are busy. `?fail_on_no_slot=1` answers `503 no slot available` when
every seat is busy.

```json
[{"id": 0, "n_ctx": 8192, "speculative": true, "is_processing": true},
 {"id": 1, "n_ctx": 8192, "speculative": true, "is_processing": false}]
```

`n_ctx` is the per-request context ceiling, absent when the model could
not be priced. `speculative` is `true`: a batched row drafts (prompt lookup or
`--model-draft`) except as [Speculative decoding](#speculative-decoding)
lists. `id`
is a position in the listing. Left out, because a seat keeps nothing
after its request: `id_task`, `n_prompt_tokens`,
`n_prompt_tokens_processed`, `n_prompt_tokens_cache`, `params`,
`next_token`, and the debug-only `prompt` and `generated`.

A deployment without seats is a 501: no batcher (continuous batching
off, or a KV pool or prefix cache keeping requests on the private loop),
or a batcher without `-np`. `--no-slots` (`LLAMA_ARG_ENDPOINT_SLOTS=0`)
turns the route off with llama.cpp's 501, `This server does not support
slots endpoint. Start it with --slots`. `--slots` is the default.

## LoRA adapters

llama.cpp's `GET /lora-adapters`, `POST /lora-adapters` and the
per-request `lora` field, over adapters loaded with `--lora` /
`--lora-scaled` (see [`CLI.md`](CLI.md#lora-adapters)).

```sh
frink-server -m model.gguf --lora style.gguf --lora-scaled tone.gguf:0.5

curl localhost:8383/lora-adapters
# [{"id":0,"path":"style.gguf","scale":1.0,"task_name":"","prompt_prefix":""},
#  {"id":1,"path":"tone.gguf","scale":0.5,"task_name":"","prompt_prefix":""}]

# Set the server's scales; an adapter not named goes to 0.
curl localhost:8383/lora-adapters -d '[{"id":1,"scale":1.0}]'
# {"success":true}

# One request under its own scales; the server's are untouched after.
curl localhost:8383/v1/chat/completions -d '{"model":"m",
  "messages":[{"role":"user","content":"hi"}], "lora":[{"id":0,"scale":0.25}]}'
```

- `GET` reports the scale currently applied (0 under
  `--lora-init-without-apply`).
- `POST` takes `[{id, scale}]`; an absent `scale` reads 0. An unknown
  `id` is a 400 naming the range (upstream ignores it).
- A request's `lora` list, as upstream's `construct_lora_list`: each
  named adapter gets its scale, every other one 0 for that request; an
  empty list means the server's scales. Naming an adapter on a server
  that loaded none is a 400; the field on a dedicated engine is a 501.
- A `POST`, or a request whose list differs from the applied scales,
  waits for in-flight generations and runs alone (llama.cpp's
  no-co-batching rule, at server granularity). A request naming exactly
  the current scales runs concurrently.
- The response cache keys on the scales a generation ran under.

## MCP

`--mcp-config PATH` loads server metadata under `frink_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Reasoning content

For a family that reasons inside markers (`<think>`, DeepSeek's DSML
variant, Gemma's channels, MiniMax-M3's namespaced pair, the gpt-oss
harmony `analysis` channel), the block is split out of `content` into
`reasoning_content`, on the message and on each streamed delta, with the
same result buffered or streamed.

The family comes from the served model's name, else from the chat
template. A model for which neither implies a family gets no reasoning
parser, so a literal `<think>` in its output stays in `content`.

### Replaying it

`reasoning_content` (alias `reasoning`) is accepted on request
assistant messages and passed to the template under both keys
(`message.reasoning_content` for DeepSeek and GLM, `message.reasoning`
for Qwen and gpt-oss). An empty string is dropped. It is never folded
into `content`.

Whether the block is already open is read from the rendered prompt: a
template asked to think can open it there, and then the model's first
token is reasoning.

## Tool-call formats

When the template does not consume `tools`, the server adds a system
turn of its own that describes them and asks for a Hermes-style
`<tool_call>{"name": …, "arguments": {…}}</tool_call>`. llama-server
renders such a template without describing the tools, so this prompt
differs from llama-server's: on DeepSeek-R1-Distill, DeepSeek V3.1,
SmolLM3, Phi-3.5 and RWKV World, the render census lists it under
`deviation:tool-preamble`. Parsing uses the format the checkpoint
implies: the one its template prints, else the one the model's name
implies. A forced `tool_choice` makes the call schema-constrained (see
[Grammar-constrained decoding](#grammar-constrained-decoding)).

| Family | Shape |
|---|---|
| Hermes / Qwen2.5 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| Llama 3 | `<\|python_tag\|>{"name": …, "parameters": {…}}`, or the same object bare. A bare turn that opens with `{` is read as a call; one naming no offered tool is returned as text |
| Mistral | `[TOOL_CALLS] [{…}]` |
| Qwen3-Coder | `<function=name><parameter=key>value</parameter></function>` |
| GLM-4.7 | `<arg_key>k</arg_key><arg_value>v</arg_value>` |
| DeepSeek | `<｜DSML｜invoke name="…"><｜DSML｜parameter name="…">…` |
| MiniMax | `<minimax:tool_call><invoke name="…"><parameter name="…">…` |
| MiniMax-M3 | `]<]minimax[>[<tool_call>` around a namespaced element grammar |
| gpt-oss | `<\|channel\|>commentary to=functions.name<\|message\|>{…}<\|call\|>` |
| Gemma 4 | `<\|tool_call>call:name{k: v}<tool_call\|>` |
| muse-glimmer | `<atem:invoke>` / `<atem:parameter>` inside an ATEM channel block |

- Every call is returned, with ids `call_0`, `call_1`, …; correlate by
  index.
- On the private decode path, Qwen3-Coder, GLM-4.7, DeepSeek, MiniMax
  and muse-glimmer stream arguments as deltas: the first delta carries
  `index`, `id`, `type` and `function.name` with empty arguments, later
  ones only more `function.arguments` text. Concatenate in `index`
  order and parse. `/v1/messages` streams `content_block_start` with a
  `tool_use` block then `input_json_delta`; `/v1/responses` streams
  `response.output_item.added` then
  `response.function_call_arguments.delta`.
- The JSON-payload families (Hermes, Llama 3, Mistral, MiniMax-M3,
  gpt-oss, Gemma) arrive whole.
- A generation truncated mid-call reports `length`, not `tool_calls`.
- A call naming a tool the request did not offer is dropped; a
  namespaced name (`skills:read`) is forwarded.
- In the XML-like formats a value is typed from the tool's `parameters`
  schema; a declared `string` is passed verbatim (`"018956"` stays a
  string).

## Legacy completions

`/v1/completions` takes `prompt` (a string) and `max_tokens` (default
16). Every sampling field of the chat table is read the same way
(`SamplingKnobs::resolve`), as are `seed`, `stop`, `ignore_eos`,
`grammar`, `lora` and `logit_bias`, plus the rows the chat table marks
as served here (`logprobs`, `prompt_logprobs`, `n`, `best_of`, `echo`,
`cache_salt`, `allowed_token_ids`, `bad_words`,
`truncate_prompt_tokens`, `skip_special_tokens`,
`return_tokens_as_token_ids`). An absent `min_p` is `0.0` (off) on both
routes, not llama.cpp's CLI default of `0.05`.

The route answers with one JSON body. Refused by name: `stream: true`
(use `/completion` or `/v1/chat/completions`), token-id prompts
(`[int]` / `[[int]]`), `suffix`, and any `response_format` other than
`{"type": "text"}`.

## llama.cpp's native `/completion`

`POST /completion` and `POST /completions` (one handler; `/admin/stats`
records the path called) are llama.cpp's completion endpoint, what its
web UI, `llama.vim` and many wrappers speak. Same engine, sampler,
grammar and stop machinery as `/v1/completions`.

| | `/completion` (llama.cpp) | `/v1/completions` (OpenAI) |
|---|---|---|
| budget | `n_predict`; `-1`, and absent, mean "until the context is full" | `max_tokens`, default 16 |
| repetition | `repeat_penalty`, `repeat_last_n` | `repetition_penalty` |
| response | flat object: `content`, `stop`, `stop_type`, `timings`, `generation_settings` | `choices[].text`, `finish_reason` |
| stream frame | `data: {"content":…,"stop":false}` | not streamed |
| stream end | the terminal object with `"stop": true`, and no `[DONE]` | not streamed |

### What it honours

`prompt` (string, or `{"prompt_string": "…"}`) · `n_predict` ·
`stream` · `stop` · `temperature` · `top_p` · `min_p` · `top_k` ·
`typical_p` · `top_n_sigma` · `xtc_probability` · `xtc_threshold` ·
`dry_multiplier` · `dry_base` · `dry_allowed_length` ·
`dry_penalty_last_n` · `dry_sequence_breakers` · `repeat_penalty` ·
`repeat_last_n` · `presence_penalty` · `frequency_penalty` · `samplers`
· `seed` (`-1` draws one) · `ignore_eos` · `grammar` · `json_schema` ·
`logit_bias` · `lora` · `cache_prompt`.

- The nine samplers of llama.cpp's default chain run in its order
  (`penalties, dry, top_n_sigma, top_k, typical_p, top_p, min_p, xtc,
  temperature`). An absent field makes its sampler a no-op.
- `dry_sequence_breakers` are strings tokenised against the model's
  vocabulary (llama.cpp's `get_overlapping_token_sequences`). With no
  real vocabulary (synthetic weights) `dry_multiplier` is refused; send
  `dry_sequence_breakers: []` to run DRY with no breakers on purpose.
- `n_predict` absent is `-1`, resolved against the context ceiling
  minus the prompt. With no ceiling (model not priced), `-1` is a 501
  naming `n_predict` and `FRINK_CB_MAX_CONTEXT`. `n_predict: 0`
  generates nothing.
- `repeat_last_n: -1` ("the whole context") is refused.
- `cache_prompt: true` (default) is always served. `cache_prompt: false`
  is refused when `FRINK_PREFIX_CACHE_ENTRIES` is set.
- `json_schema` is compiled like `response_format: json_schema`. A
  `grammar` and a `json_schema` together are a 400 (upstream drops the
  schema).

### What it refuses, by name

Each option below is served at the value where llama.cpp treats it as
off (stock clients send most of them explicitly) and is a 501 naming the
field at any other value:

`dynatemp_range` · `mirostat` · `n_probs` and `post_sampling_probs` ·
`min_keep` · `return_tokens` · `n_indent` · `n_keep` (frink refuses an
oversized request rather than shifting context) · `n_cmpl` ·
`n_cache_reuse` · `t_max_predict_ms` · `id_slot` ·
`response_fields` · `return_progress` · `timings_per_token` ·
`sse_ping_interval` (the keepalive is fixed at 15 s).

Also refused: token-id prompts, multiple prompts, `prompt.multimodal_data`,
and the fields the chat table marks as 501 on this route, such as `n`
above 1. Parameters of a switched-off sampler (`mirostat_tau`,
`mirostat_eta`, `dynatemp_exponent`) are accepted. A field llama.cpp
does not define is ignored, as upstream.

### Response fields

- `truncated` is always `false`: frink refuses rather than truncates.
- A `timings` value not measured is `null`, not `0`; `timings.cache_n`
  is `-1` (upstream's sentinel) when no prefix cache is configured.
- `stop_type` uses llama.cpp's `eos`, `word`, `limit`, plus `cancelled`
  for an answer stopped by `POST /v1/cancel` or a disconnect. `none`
  means still generating.
- Known inaccuracy: a stop string that is exactly one token is caught at
  the token level and reported as `"eos"` where llama.cpp says `"word"`;
  `/v1/messages` loses its `stop_sequence` attribution on the same
  inputs. Multi-token stop strings are reported correctly.

## Infill, `POST /infill`

llama.cpp's fill-in-the-middle completion: a `/completion` body with
three more fields, answered with `/completion`'s object and stream.

```bash
curl -s http://127.0.0.1:8383/infill -H 'content-type: application/json' -d '{
  "input_prefix": "def add(a, b):\n    ",
  "input_suffix": "\n\nprint(add(1, 2))\n",
  "input_extra": [{"filename": "util.py", "text": "def sub(a, b): return a - b\n"}],
  "n_predict": 32
}'
```

- `input_prefix` and `input_suffix` (required): a string, or an array of
  strings and non-negative token ids; strings are tokenized without
  parsing special markers.
- `input_extra` (optional): an array of `{"filename", "text"}`, other
  files the model reads first. `filename` defaults to `tmp`.
- `prompt` (optional): text appended to the prefix, tokenized with
  special markers parsed.

The prompt is assembled as llama.cpp's `format_prompt_infill` does,
with the repo-level pattern when the vocabulary has its tokens:
`[FIM_REP]myproject\n`, each extra file as `[FIM_SEP]filename\n` and its
text (or llama.cpp's `--- snippet ---` separator without a file
separator), a closing `[FIM_SEP]filename\n`, then BOS where the
vocabulary adds one, `[FIM_PRE]` prefix and prompt, `[FIM_SUF]` suffix,
`[FIM_MID]`. `--spm-infill` puts the suffix first. The prefix keeps its
last `3 * (n_batch / 4)` tokens, the suffix its first
`n_batch / 4 - (2 + prompt)`, where `n_batch` is `-b` (default 2048),
and the extra context its last `n_ctx - n_batch + 2`. The response's
`prompt` is the detokenized assembly.

FIM tokens come from `tokenizer.ggml.fim_*_token_id` (and the deprecated
`prefix_token_id`, `suffix_token_id`, `middle_token_id`), then by text
as llama.cpp finds them (`<|fim_prefix|>`, `<fim_prefix>`, `▁<PRE>`, …).
A missing prefix, suffix or middle token is llama.cpp's 501,
`Infill is not supported by this model: prefix token is missing.` (and
so on). Where two entries of one kind match, the lower id is taken.

The body is checked with llama.cpp's messages (`"input_prefix" is
required`, `"input_extra" must be an array of {"filename": string,
"text": string}`, `extra_context chunk must contain a "text" field with
a string value`, ...), then as a `/completion` request. Differences:

- A non-string `prompt`, or a missing `input_prefix` or `input_suffix`,
  is a 400 (llama.cpp carries on).
- The dedicated engines (Kimi, MLA, Gemma-4, GLM-5.2) are a 501.
- Repo-level context on a deployment where the model could not be priced
  is a 501.
- llama.cpp also ends generation on the FIM pad, repo and file-separator
  tokens; frink's end-of-generation set does not include them yet, on
  any route.

## Server properties, `GET /props`

llama.cpp's global properties, with its names, nesting and types. A
field frink has no value for is left out rather than defaulted.

| Field | Value |
|---|---|
| `default_generation_settings.params` | What `/completion` resolves an empty request to: the samplers in default order by llama.cpp's names, each knob at its default, and the options frink lacks (`dynatemp_range`, `mirostat`, `n_probs`, `min_keep`, `n_keep`, `timings_per_token`, `post_sampling_probs`) at their off value. `n_predict` and `max_tokens` are `-1` |
| `default_generation_settings.n_ctx` | The per-request context ceiling |
| `total_slots` | The `-np` seats of the continuous batcher |
| `model_alias` | The served name (`-a`, or the model's own) |
| `model_ftype` | `general.file_type` in llama.cpp's words (`Q4_K - Medium`), for the generic GGUF decoder |
| `model_path` | The checkpoint path |
| `modalities` | `vision`, `video` and `audio` all `false` |
| `endpoint_slots` | `false` under `--no-slots` |
| `endpoint_props` | `false`: there is no `--props` |
| `endpoint_metrics` | `true` |
| `ui` | `false` |
| `chat_template` | The Jinja source the checkpoint is served with |
| `bos_token` · `eos_token` | The vocabulary entries the GGUF names |
| `build_info` | `frink-server` and its version |
| `is_sleeping` | `true` between `POST /sleep` and `POST /wake_up` |
| `cors_proxy_enabled` | `false` |

Left out:

- `seed` in `params` (llama.cpp's `4294967295` would read here as a fixed
  seed; frink draws one per request).
- `dynatemp_exponent`, `mirostat_tau`, `mirostat_eta`,
  `adaptive_target`, `adaptive_decay`, `n_discard`.
- `chat_format`, `reasoning_format`, `reasoning_in_content`,
  `generation_prompt`, `speculative.types`, `backend_sampling`.
- `n_ctx`, `total_slots`, `model_ftype`, `model_path`, `chat_template`,
  `bos_token`, `eos_token` when the deployment lacks them (no ceiling,
  no `-np`, unknown file type or a dedicated engine, no file on disk, a
  builtin template, no named BOS or EOS).
- `media_marker`, `ui_settings`, `chat_template_caps`,
  `chat_template_tool_use`, `connect_code`.

An encoder model's `/props` carries only the model and server fields.
Asleep, `/props` answers from the snapshot taken at sleep, with
`is_sleeping: true`, as llama.cpp does. `POST /props` is llama.cpp's 501
without `--props`.

## Tokenize / detokenize, in both dialects

`/v1/tokenize` and `/v1/detokenize` (frink's spelling) and llama.cpp's
`/tokenize` and `/detokenize` share one handler; `/admin/stats` records
the path called. Both body dialects work on both paths:

| Field | Status |
|---|---|
| `content` | llama.cpp's name for the text. Supported |
| `prompt` | frink's name for the same text. Supported |
| both at once | 400 |
| neither | 400 naming both (llama.cpp answers an empty array) |
| `add_special` | Supported, default `false`. Prepends the BOS the generation path would, including its no-op when metadata says not to add one |
| `parse_special: true` (upstream's default) | Supported. Special-token markers become the tokens they name |
| `parse_special: false` | Supported. `<|im_start|>` is tokenized as plain characters |
| `with_pieces: true` | 501 by name (no raw per-token piece bytes) |
| `model` | Accepted and ignored |

Tokenize answers `tokens` (llama.cpp) plus `count`. Detokenize answers
the text under both `content` (llama.cpp) and `text` (frink).

## Grammar-constrained decoding

`"grammar": "<GBNF>"` is llama.cpp's field and syntax, accepted on
`/v1/chat/completions`, `/v1/completions` and `/completion`, compiled
once per request and enforced per token by a stack machine.

- `response_format: json_object` is a character-class mask (it cannot
  check that a `}` closes an open object; a grammar can). The two
  compose; neither un-masks what the other forbids.
- A satisfied grammar with no legal continuation ends normally
  (`finish_reason: "stop"`). A grammar that dead-ends unsatisfied is a
  400 with `failure_code: "invalid_grammar"`; under continuous batching
  only that row fails.
- `response_format: {"type": "json_schema", "json_schema": {"name": ...,
  "schema": {...}, "strict": true}}` is compiled to GBNF by the converter
  `tool_choice` uses, so `required` properties and `enum` values are
  enforced. `name` and `description` constrain nothing. `required`
  members are enforced in declared order, as llama.cpp does.
- Inside `json_schema`, an unknown member is a 400 naming it, and
  `strict: false` is a 400 (the schema is always enforced). An omitted
  `strict` is enforced.
- A schema the converter will not compile is a 400 naming the keyword
  and its subschema. This is stricter than llama.cpp, which drops
  untested keywords (`{"type": "string", "pattern": "^[a-z]+$"}`
  compiles upstream to any string).
- A `grammar` and a `json_schema` together, or a forced `tool_choice`
  beside either, is a 400.

### Forced tool calls

`tool_choice: "required"` and a named `tool_choice` use a lazy grammar
built from the request's `tools`; each tool's `arguments` rule is its
`parameters` schema compiled to GBNF, and a named choice narrows the
union to one tool.

Unlike llama.cpp's eager grammar, the trigger is the wire format's
opening marker, because several families open a reasoning block in the
prompt and an eager call would be parsed as thinking. While waiting, the
grammar masks only the end-of-generation tokens: the prefix is free,
but the turn cannot end before a call begins. A model that never opens
a call runs to `max_tokens` and finishes `"length"`.

| Shape | Formats | Root rule |
|---|---|---|
| JSON payload | Hermes/Qwen2.5, Llama 3, Mistral | a marker, a JSON object naming the tool, a closing marker |
| Elements | `qwen3_coder`, `glm47`, `minimax`, `deepseekv32`, `minimax_m3` | an invoke element holding one element per argument |
| Harmony | `gpt_oss` | a channel header addressed to `functions.<name>`, then JSON |
| Pairs | `gemma4` | `call:NAME{k:v,k:v}`, the values in Gemma's own quoting |

The grammar's literals come from the same marker description the
streaming parser reads, so forced output always parses back.

Refused by property name: on `gemma4`, an `object` or `array` argument
(Gemma's DSL spelling would come back as a string); on the element
formats, a property with no declared `type`.

`muse_glimmer` is a 501 naming the format: its call boundary depends on
the channel, and forcing the channel header needs facts about the
checkpoint's template that no available checkpoint supplies.

## Not yet

Image and audio input · MCP tool invocation · multi-GPU, tensor
parallel, prefill/decode disaggregation · streamed argument deltas for
the JSON-payload tool formats · `stream` on `/v1/completions`. Narrower
gaps are noted in their sections: refused embedding architectures,
requests that do not speculate, `/infill` on dedicated engines, the
per-task fields of `GET /slots`.

## Fields that are read and not used

`metadata` on `/v1/messages`, `store` and `metadata` on
`/v1/responses`, `user` on the chat route, and `stream_options` (`usage`
always rides the final chunk) are accepted and change nothing. Fields
that would change the tokens or the text and are not implemented are
refused by name (the 501 rows above). Any field this page does not name
is dropped by the JSON parser without an error.

See [`plans/north-star.md`](plans/north-star.md) for what comes next.
