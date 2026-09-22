# API

`frink-server` exposes an OpenAI-compatible HTTP API for chat serving.

Fields marked **Reject** return HTTP 400 or 501 with an error message
that names the problem. Multimodal input the server does not handle
comes back the same way.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /health` | Supported (capability handshake, see below) |
| `GET /v1/models` | Supported |
| `POST /v1/chat/completions` | Supported (JSON + SSE) |
| `POST /v1/completions` | Supported (`prompt`, `max_tokens`, sampling subset) |
| `POST /completion` · `POST /completions` | llama.cpp's **native** completion endpoint, JSON + its own SSE shape. Not an alias of the line above (see below) |
| `POST /v1/tokenize` · `POST /tokenize` | Supported. The unprefixed spelling is llama.cpp's, on the same handler (see below) |
| `POST /v1/detokenize` · `POST /detokenize` | Supported, same aliasing |
| `POST /v1/embeddings` | Supported. A real BERT/BGE encoder when one is loaded (`cls`/`mean`/`last`, L2-normalized), otherwise a mean/last pool of a GGUF decoder's hidden states (see below) |
| `POST /v1/messages` | Anthropic Messages, streaming and buffered |
| `POST /v1/messages/count_tokens` | Anthropic prompt sizing, no generation |
| `POST /v1/responses` | OpenAI Responses surface (what `codex` speaks), streaming and buffered |
| `GET /v1/responses/{id}` · `POST /v1/responses/{id}/cancel` | Always 404. This server keeps no responses, so there is nothing to fetch or cancel by response id. Cancel a live generation with `POST /v1/cancel` |
| `GET /v1/stats` · `GET /v1/requests` | Live serving telemetry, pool gauges, memory footprint, and the request ring |
| `POST /v1/cancel` | Stop a streamed generation by `request_id` (see below) |
| `GET /v1/stream/{request_id}` · `GET /v1/stream/{request_id}/poll` | Reconnect into a resumable stream, over SSE or plain JSON (see below) |
| `GET /v1/cache/status` · `POST /v1/cache/rebuild` | KV pool geometry and re-split (see below) |
| `GET /v1/conversations` · `POST /v1/conversations` | Server-side transcripts: list newest first, or create |
| `GET`/`POST /v1/conversations/{conversation_id}` | Read one with its messages, or rename, retarget and append |
| `POST /v1/conversations/{conversation_id}/delete` | Delete. Spelled as a POST suffix because the CORS allow-list is `GET, POST`, so a `DELETE` method would work from curl and fail from every cross-origin browser |
| `POST /v1/admin/prepare-stop` | Close admission, seal the accounting, and make the receipt durable (see below) |
| `POST /slots/{id_slot}?action=save\|restore` | llama.cpp's slot save/restore: persist a prompt prefix's KV to disk and load it back after a restart. Needs `--slot-save-path` and `FRINK_PREFIX_CACHE_ENTRIES` (see below). `action=erase` is refused by name |
| `GET /lora-adapters` · `POST /lora-adapters` | llama.cpp's LoRA listing and scale setting; the per-request `lora` field is honoured on `/v1/chat/completions`, `/v1/completions` and `/completion` (see below) |
| `GET /cache/stats` · `GET /metrics` | Frink extensions |
| `/admin/*` | Control surface (see below) |
| `GET /` | 404. The web UI in [`ui/`](../ui) is a separate app and this server does not serve it |
| Audio / images | Not supported |

That is the whole list. Every path lives as one constant in the
`frink-api` crate, and the server mounts nothing that is not in it, so
the UI and the server cannot disagree about a URL.

## Authentication

Set `FRINK_API_KEY` and every route except `GET /health` needs
`Authorization: Bearer <key>`. `/metrics` and `/cache/stats` are in that
set, so a Prometheus scraper needs the header too.

Both spellings are read: `Authorization: Bearer <key>` and
`x-api-key: <key>`, which is what the Anthropic SDKs send. A stock
Anthropic client works against a keyed server without setting a header
by hand. If a request carries both, `Authorization` wins.

Request bodies are capped at axum's 2 MiB default. A long `/v1/messages`
conversation or a large `/v1/embeddings` batch past that comes back
`413` before any handler sees it.

## Chat completions fields

| Field | Status |
|---|---|
| `model`, `messages` | Supported |
| `max_tokens` | Supported; **defaults to 32768**, not OpenAI's legacy 16. An explicit `0` is a 400 |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop` | Supported |
| `typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` | Supported. llama.cpp's own spellings; absent means the sampler is off |
| `samplers` | Supported. The chain order, as a list of names or a `;`-separated string. Defaults to llama.cpp's own default chain |
| `presence_penalty`, `frequency_penalty` | Supported |
| `stream` | Supported (overlapped SSE when tools off and CB off) |
| `tools` / `tool_choice: none\|auto` | Supported (prompt-engineered, parsed in eleven wire formats) |
| `tool_choice: required` / named function | Supported on **ten of the eleven** wire formats, by a lazy grammar built from the same marker description the parser reads with. **501 naming the format** on the remaining one (`muse_glimmer`), for the reason the refusal states |
| `logprobs` | **Served on `/v1/completions`**: `logprobs: N` reports the chosen token's log-probability and up to `N` alternatives per position, over the distribution the sampler actually drew from -- penalties applied, llama.cpp's chain run, a grammar's mask included. A candidate the chain REMOVED is omitted rather than reported as `null` or as a large negative stand-in: `ln(0)` is not a number, and it was not a candidate. `N` above 5 is a **400** naming the field, as upstream caps it. `text_offset` is a byte offset into the returned text, computed from the same pieces `tokens` reports. **Reject** on `/v1/chat/completions` (`top_logprobs`), which caches one answer per key and would replay text without the distributions it came from |
| `n` (>1) | **Served on `/v1/completions` and `/v1/chat/completions`**: the prompt is prefilled once and the KV forked per choice, so `prompt_tokens` counts the prompt ONCE while `completion_tokens` sums. Each choice is parsed for tool calls and reasoning in its own right. Choice `i` samples from `seed + i`, so choice 0 is byte-identical to the `n = 1` answer at the same seed, and at `temperature 0` every choice is the same sequence because the sampler is deterministic. **501 naming the field** on the native `/completion`, which returns a single `content`, and for `n` > 1 together with `stream`, because the choices would arrive one after another rather than interleaved by `choices[].index` |
| `best_of` (>1), `prompt_logprobs`, `echo`, `use_beam_search`, `truncate_prompt_tokens`, `prompt_embeds`, `allowed_token_ids`, `bad_words`, `skip_special_tokens: false`, `return_tokens_as_token_ids` | **Reject, 501 naming the field**, on `/v1/chat/completions`, `/v1/completions` AND llama.cpp's native `/completion`. Each changes the tokens or the text returned, so ignoring one answers a question the caller did not ask. The default a caller may legitimately spell out (`n: 1`, `echo: false`, `skip_special_tokens: true`) is **served**. One table, flattened into all three request bodies, so a field cannot be refused on one wire and dropped on another again |
| `response_format: json_object` | Supported (best-effort character mask + validate) |
| `grammar` | Supported. llama.cpp's own field: a GBNF string, enforced per token by a real parser |
| `response_format: json_schema` | Supported. The `json_schema.schema` is compiled to GBNF and enforced per token. `strict: false` and any unknown member of the `json_schema` object are refused **by name**; a schema the converter cannot compile is a 400 naming the keyword |
| Other `response_format` types | **Reject** |
| `session_id` | Frink extension (server-side history) |
| `chat_template_kwargs` | Supported (see [Chat templates](#chat-templates)) |
| `reasoning_effort` | Supported, quantized onto what the checkpoint grades; `none`/`off` turn thinking off |
| `thinking: {"type": …}` | Supported (DeepSeek wire): `enabled`/`disabled`, anything else is a 400. On `/v1/messages`, `thinking.budget_tokens` becomes `reasoning_budget_tokens` exactly as llama.cpp's Anthropic lowering does (`server-chat.cpp:585-591`): `enabled` without a number gets 10,000 |
| `ignore_eos` | Frink extension: run past the model's own end-of-generation tokens so the request produces exactly `max_tokens`. A serving-benchmark knob. Suppresses the model's set only. A caller's own `stop` strings still end the answer |
| `reasoning_content` (both ways) | Frink extension: a reasoning model's chain of thought, split out of `content` on the way out and replayable on the way in (`reasoning` is accepted as an alias) |
| `continue_final_message` | Supported, llama.cpp's field and value set: `true` (auto), `"reasoning_content"`, `"content"`, or `false`. The trailing assistant message is rendered as a turn still being written, thought included, so the model carries on from where it stopped. **Default on, as llama.cpp's server** (`server-common.cpp:1046-1056`): a request that says nothing and ends in an assistant message is continued; `false` renders it as a closed turn for that request (llama.cpp reads `false` as absence, so this is the one deliberate difference), and `--no-prefill-assistant` turns the default off server-wide. One rule for `/v1/chat/completions`, `/v1/messages` and `/v1/responses`, applied in one renderer. The harmony and ATEM channel formats, a content continuation for an always-open family, and a turn with tool calls are **501 by name**; a 501 reached by the default says how to switch it off |
| `reasoning_budget_tokens` / `thinking_budget_tokens` | Supported, llama.cpp's sampler-level budget (`common/reasoning-budget.cpp`): `-1` or absent takes the server's `--reasoning-budget` (itself `-1`, unrestricted); `N` allows N tokens of thought after the opener and then forces the closer one token per step, so the answer still arrives with `finish_reason: "stop"`; `0` forces the closer the moment the block opens. The closer is never counted, a forced closer waits for a split UTF-8 character to complete, and a second block in the same response gets a fresh budget. Below `-1` is a 400 naming the field. Also accepted at the top level of a `/v1/responses` body, as llama.cpp's lowering passes it. A checkpoint with no reasoning format has no thought to bound (llama.cpp builds no sampler either); the harmony and ATEM channel formats are **501 by name** |

### Where a completion stops

Besides `max_tokens` and the request's own `stop` strings, generation ends
on any token in the checkpoint's **end-of-generation set**: the
`eos`/`eot`/`eom` metadata ids plus every vocabulary entry whose text is on
llama.cpp's literal EOG list (`<|eot_id|>`, `<end_of_turn>`, `<|im_end|>`,
`<turn|>`, …). Stop on `tokenizer.ggml.eos_token_id` alone, as this
server did until `StopTokens` landed, and a Llama-3 or gemma checkpoint
runs past the end of its own turn, because neither one's turn ender *is*
the metadata EOS. Both decode paths (`generate` and the continuous
batcher) use the same set. A Kimi K3 checkpoint keeps its vocabulary in
`tokenizer_config.json` rather than GGUF metadata, so the set is derived
from its special-token names and `[EOT]` ends a turn there too.

BOS follows the rule in [CLI.md](CLI.md#who-adds-bos): the chat template
owns it when it prints one, the loader otherwise, added idempotently so a
template that already emitted BOS never gets a second one.

### When a request does not fit the context

Two outcomes, not one. A **prompt** at or past the deployment's
per-request position ceiling is refused with a 400 whose message reads
`prompt is too long: N tokens > M maximum`, the wording that Claude Code
and OpenClaw match on, because the Anthropic wire carries no error code
for it. A prompt that *fits* is **served**, with `max_tokens` clamped
down to the room that remains; refusing that case would turn a servable
long-prompt request into an error over a `max_tokens` the caller most
likely never set. The clamp is also what makes the 32768 default output
budget safe.

## Chat templates

The prompt is rendered by evaluating the checkpoint's own
`tokenizer.chat_template`, the real Jinja2 source shipped in the GGUF,
rather than by recognising it. A checkpoint that ships no template at all falls
back to ChatML (matching llama.cpp `--jinja`), or to role-labeled lines
for a byte/synthetic tokenizer. A template that does not compile, or that
uses a construct this evaluator does not provide, fails the chat request
with the compiler's own message; it does **not** quietly fall back to a
guessed framing, which is the failure it exists to remove.

`chat_template_kwargs` is passed through: whatever a client puts there
becomes a top-level template variable, which is how `enable_thinking`
(Qwen3, gemma-4), `thinking` (DeepSeek) and `preserve_thinking` are
really driven. It can never shadow `messages`, `tools`, or
`add_generation_prompt`.

Five rules are applied to it before rendering:

* **An explicit knob wins wholesale.** A caller who already set any of
  `enable_thinking`, `thinking`, `thinking_mode` or `reasoning_effort`
  *inside* `chat_template_kwargs` has said what they want; the
  protocol-level knobs then stand down entirely rather than merging, so
  a default can never contradict an explicit request.
* **`none` and `off` are not gears.** `reasoning_effort: "none"` means
  *turn thinking off*, and is broadcast as such. It is not quantized
  onto the nearest gear, which would turn "do not think" into "think a
  little". The DeepSeek-wire `thinking: {"type": "disabled"}` does the
  same and beats any effort in the same request, because the switch is
  what the caller reached for last and the gear is what they would have
  used had thinking been on.

* **Offering tools turns thinking on.** Some encoders emit well-formed
  tool calls only in thinking mode, so `tools` implies
  `enable_thinking` unless the request said otherwise.
* **Effort is quantized onto what this checkpoint grades.** The
  template's real effort vocabulary is probed once at load; a request
  asking for a gear outside it is mapped to the nearest one, or dropped
  (so the template's own default applies) when nothing is close enough.
  `reasoning_effort: "minimal"` against a checkpoint that grades only
  `low`/`medium`/`high` renders as `low` rather than failing or
  interpolating an unknown word into the prompt. Top-level
  `reasoning_effort` loses to an explicit `chat_template_kwargs` entry.
* **One value, every spelling.** The graded-strength dialect reads
  `reasoning_strength`; the value is broadcast to both, and a Jinja
  template ignores variables it does not declare.

`GET /v1/models` advertises what came out of that probe:
`supported_reasoning_efforts` (least thinking first: `off`, `adaptive`,
then the gears, or a bare `on` when there is a toggle but no ladder) and
`default_reasoning_effort` (the gear the checkpoint is already in when
asked for nothing). A checkpoint that says nothing about thinking
carries **neither field**, rather than an empty list: an empty list
would read as "asked, and it has no gears", which is a different claim
from "this is not a reasoning model".

Whether the *tools* are described by the template or by a text preamble
depends on the template, established at load by rendering it with and
without a tool rather than by looking for the word: one that really
consumes `tools` is handed them as structured JSON and owns the whole
grammar; one that does not gets the preamble described under
[Tool-call formats](#tool-call-formats).

## Health

`GET /health` returns JSON (it used to return the string `ok`) and is
never behind auth or rate limiting.

```jsonc
{
  "state": "ready",              // ready | detecting | unavailable
  "model": { "id": "…", "tokenizer": "…", "synthetic_weights": false },
  "capabilities": [
    { "id": "metal", "available": false,
      "reason": "metal_not_built",       // stable code, safe to switch on
      "detail": "Apple M2 Pro is present but this binary was built without --features metal; rebuild to use it." }
  ],
  "version": "0.12.0", "pid": 4242, "uptime_seconds": 12.5,
  "server_time_unix_ms": 1786000000000,
  "last_request_age_seconds": 0.4    // absent until one is served
}
```

There are three states, not two. `detecting` means backend probing has
not finished. The capabilities in that response are guesses, not
measurements, so a client should wait rather than draw a conclusion.
Detection gets a hard 1s budget. Past that the answer is filled in
provisionally with `reason: "detection_timed_out"`. The handler itself
never blocks.

Status is 200 for `ready` and `detecting`, 503 for `unavailable`. You
reach `unavailable` by calling `POST /admin/models/unload`, which
answers with `reason: "model_not_loaded"` and a null `model`. A
supervisor that read 200 there would route traffic straight into a 503.

Every capability carries a machine `reason` and a human `detail`, so a
UI greys the control and shows the sentence without working out why
itself. `metal_unavailable` (no device), `metal_not_built` (this
binary), and `disabled` (a flag turned it off) are three different
problems with three different fixes.

`last_request_age_seconds` exists so nobody reads a busy server as a
dead one. A saturated GPU starves the health handler, and recent
request activity is positive evidence that the process is alive.

## Request ids

Every `/v1/chat/completions` response carries a server-assigned
`chatcmpl-…` id. It is the `id` field, and it is repeated once under
`request_id`. For a non-streamed call it sits in the JSON body. For a
streamed one it arrives in the **first** SSE chunk, before any content.
That is the key frink logs and cancels by, so a client never has to
guess which in-flight request is its own.

## Usage timings

`usage` extends the OpenAI shape with llama.cpp-style timings. Prefill
and decode are reported separately, deliberately: dividing total tokens
by total wall time reads a 50 tok/s model as 5 tok/s on a long prompt.

| Field | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | OpenAI convention |
| `prompt_eval_duration_ms` | wall time spent on the prompt |
| `generation_duration_ms` | wall time spent in the decode loop |
| `time_to_first_token_ms` | prefill start → first token produced |
| `prompt_per_second`, `predicted_per_second` | per-phase throughput |
| `cached_tokens` | prompt tokens reused from the KV prefix cache |
| `acceptance_length` | completion tokens per FORWARD PASS, speculative or not. 1.0 means speculation saved nothing, 2.0 means half the forwards. A round that drafted nothing still costs a forward and is counted, because a denominator of speculative rounds alone reports a number that flatters the drafter |
| `draft_tokens`, `accepted_draft_tokens` | draft tokens evaluated / kept |
| `draft_accept_rate_per_position` | accept rate at each position in the draft block |

Every timing is optional and **omitted** rather than nulled or zeroed
when it was not measured (a cached response, a batched decode). Note
`cached_tokens`: absent means no prefix cache is configured, `0` means
the cache was consulted and missed.

The four speculation fields are absent unless speculative decoding ran,
which is not the same statement as `acceptance_length: 1.0` ("it ran and
never helped"). `draft_accept_rate_per_position` is reported beside the
mean rather than folded into it: a drafter that is right at the first
drafted position and useless at the last has the same mean as a
uniformly mediocre one, and the two call for opposite block sizes.
`/admin/stats` rows carry `acceptance_length` and
`draft_accept_rate_per_position` for the same reason.

**The server speculates now**, so these fields are present whenever a
request actually drafted. The drafter is prompt-lookup: an n-gram match
over the request's own history, with no second checkpoint, so it costs
no memory and no load time and it helps exactly where output quotes
input. `acceptance_length` is tokens per forward pass, and 1.0 means
the drafting bought nothing.

The fields stay ABSENT, rather than reporting zeros, for a request that
did not draft: a grammar-constrained request (the verification draws
through the same grammar machine, and a rejected block would leave it
advanced over tokens that were never emitted, so drafting is refused
rather than silently skipped), a paged or recurrent KV store (a
rejected draft has to be rolled back and neither can), and a request
with no room to draft. An absent field reads as "this request did not
speculate"; a zero would read as "the drafter was useless".

Streamed requests get the same `usage` object on the final chunk.

## Admin / control surface

Everything under `/admin` either changes what the server serves or
writes to disk, so it needs the same `FRINK_API_KEY` as `/v1/*`. None
of it sits on the unauthenticated `/health` side. Paths and payload
shapes are defined once in the `frink-api` crate, so the UI and the
server cannot disagree about them.

| Endpoint | Answer |
|---|---|
| `GET /admin/models` | inventory (below) |
| `POST /admin/models/load` `{"id":"…"}` | `202 {"task_id":"…"}` |
| `POST /admin/models/unload` | `200 {"ok":true,"active":null}` |
| `POST /admin/download` `{"repo":"…","file":"*.gguf"}` | `202 {"task_id":"…"}` |
| `GET /admin/tasks` | every job, newest last |
| `POST /admin/tasks/{task_id}/cancel` | `200 {"ok":true}` |
| `GET /admin/stats` | counters + recent-request ring |

### Models

```jsonc
{
  "model_dir": "/models",          // null when none is configured
  "active": "Qwen3-0.6B-Q4_K_M",   // null when nothing is loaded
  "models": [{
    "id": "Qwen3-0.6B-Q4_K_M",     // file stem, the only way to name a model
    "path": "/models/Qwen3-0.6B-Q4_K_M.gguf",
    "size_bytes": 396705472,
    "arch": "qwen3", "quant": "Q4_K_M",
    "context_length": 40960, "param_count": 596049920,
    "state": "available",          // loaded | loading | available | error
    "error": null,                 // why the last load attempt failed
    "resident_bytes": null
  }]
}
```

This comes from **GGUF headers only**: metadata and tensor descriptors,
never a weight. Listing a directory of checkpoints costs header parses,
not loads. A field that is expensive to establish comes back `null`
rather than guessed, and the key is always present, because an unknown
context length and a zero one are different facts. `quant` comes from
`general.file_type` when that maps to a known name, which is the only
place the `_M` in `Q4_K_M` is stated. Failing that it comes from the
dominant tensor dtype, which is coarser and still measured rather than
invented. `resident_bytes` is always `null`, because frink keeps
checkpoints mmap-resident and the true figure is a page-cache property
this process has no way to read.

Discovery scans `FRINK_MODEL_DIR` and the directory holding
`FRINK_MODEL_PATH`, non-recursively, for `*.gguf` plus any
safetensors-index checkpoint directory. Split checkpoints fold into one
entry named for the shard prefix.

### Model swap

`POST /admin/models/load` answers immediately with a task id, and the
load runs on a blocking thread. It accepts only ids from
`GET /admin/models`. There is deliberately **no load-by-path
endpoint**, so no request talks the server into opening an arbitrary
file. A second load while one is running comes back `409`.

**In-flight requests keep the model they started on.** A request clones
its handle once, up front, and decodes against exactly those weights
even if a different checkpoint is published mid-generation. The old
model is freed when the last such request finishes. Nothing tries to
migrate a running request. Half a completion from one checkpoint and
half from another is worse than either.

After `POST /admin/models/unload`, generation endpoints answer `503`
with `"type": "model_not_loaded"`, `GET /v1/models` returns an empty
list, and `/health` reports `unavailable` (503).

### Tasks

One shape for every long-running job.

```jsonc
{"tasks": [{
  "task_id": "dl-2", "kind": "download",      // download | load
  "label": "Downloading *Q4_K_M.gguf from unsloth/Qwen3-0.6B-GGUF",
  "status": "running",            // queued | running | done | error | cancelled
  "error": null,
  "started_at_ms": 1786717917646, "updated_at_ms": 1786717925031,
  "progress": {
    "fraction": 0.126, "bytes_done": 50198634, "bytes_total": 396705472,
    "rate_bytes_per_s": 9802506.2, "eta_seconds": 35.3,
    "state": "stable"             // warming | stable
  }
}]}
```

Timestamps are the server's Unix epoch milliseconds. The browser clock
is not trusted for ordering. `done`, `error` and `cancelled` are
terminal and never change, so a client stops polling the moment it
reads one.

`rate_bytes_per_s` and `eta_seconds` are `null` while `state` is
`warming`, which lasts until the rate window holds at least 3 samples
spanning at least 3 s. Show "measuring", not a number. The first tick of
any transfer divides out to gigabytes per second. `fraction` is
available immediately, because it is a ratio of two counters rather
than a derivative.

Cancellation is cooperative. `POST …/cancel` raises a flag and returns.
The task reaches `cancelled` only once a worker acknowledges it. A
download stops within one chunk and keeps its `.part` file for a resume.
A model load **cannot be interrupted** mid-mmap, so a cancel that
arrives during one discards the finished result rather than pretending
the work stopped early. Cancelling a finished task is a `409`.

### Download

Fetches one `.gguf` from the Hugging Face Hub into the model directory.
Give `file` a literal name or a `*` glob, which is resolved against the
repo's file list (sorted, first match, root-level names only).
Downloads resume: bytes land in `<target>.part` and a restart sends
`Range`, falling back to a clean restart when the server answers `200`
instead of `206`.

Security here is a whitelist, not a filter. Bare `*.gguf` filenames
only: no separators of either kind, no leading dot, no `..`, no `:`.
The joined path is then re-checked to be a direct child of a model
directory **fixed at startup**, never one named by the request. Two
downloads of the same target come back `409`, because interleaved
writes into one `.part` file produce a corrupt checkpoint of exactly
the right size.

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repos, and
`HF_ENDPOINT` for a mirror.

### Stats

```jsonc
{
  "uptime_seconds": 43, "requests_total": 1, "errors_total": 0,
  "cache_hits": 0, "cache_misses": 1,
  "tokens_prompt_total": 17, "tokens_generated_total": 24,
  "last_request_age_seconds": 0.02,
  "generating_now": 0,               // decoding right now, NOT a queue depth
  "queue_depth": null,               // null unless continuous batching is on
  "queue_rejected_total": null,      // turned away by the queue cap, since start
  "recent": [{
    "request_id": "chatcmpl-b28a1aeab8f8000000",
    "at_ms": 1786718007013, "route": "/v1/chat/completions",
    "model": "Qwen3-0.6B-Q4_K_M",   // what SERVED it, not what it asked for
    "status": 200, "prompt_tokens": 17, "completion_tokens": 24,
    "ttft_ms": 6586.2, "duration_ms": 23603, "decode_ms": 17004.8,
    "stream": false,
    "via_api_key": "key-4f21a0c3",   // fingerprint, never the key; null if none
    "client": "frink-studio"        // SELF-DECLARED (X-Frink-Client); null if absent
  }]
}
```

`recent` is a 200-entry ring keyed by the same `request_id` the response
carried, so a log row joins its message on equality instead of on a
guess. **`duration_ms` and `decode_ms` are separate and must stay
separate.** The first carries queue wait plus prefill plus decode, so
dividing completion tokens by it reads a 50 tok/s model as 5.
`decode_ms` and `ttft_ms` are `null` when the engine did not time itself
or the answer came from cache.

Recorded today for `/v1/chat/completions` (streamed and not, including
rejections), `/v1/completions`, `/v1/messages`, `/v1/embeddings`,
`/v1/tokenize` and `/v1/detokenize`, the last three with their status,
success or failure. `/v1/tokenize` and `/v1/detokenize` carry no token
counts on purpose: they run the tokenizer and not the model, and
`prompt_tokens` here feeds `tokens_prompt_total`, which means "tokens
this server put through a forward pass". `/v1/embeddings` does run one,
so its prompt tokens are counted and its `decode_ms` is `null`, there
is no decode loop to time.

`queue_depth` and `queue_rejected_total` come from the continuous-batching
scheduler's queue and are `null`, not `0`, when batching is off, because
then nothing queues at all: every request gets its own blocking thread. A
gauge reading `0` claims an empty queue was measured.

`model` names the model that answered, as `/v1/models` reports it, and
never the `model` field the request carried, this server decodes
against whatever is loaded and ignores that string, so echoing it back
would make the log agree with the caller's belief rather than with what
happened. `null` means nothing was loaded, which is what a 503 row is.

**Attribution.** `via_api_key` is a short fingerprint of the bearer key
that served the request, never the key and never anything the key can be
recovered from; it is salted per process, so it is stable within one
server run, meaningless across a restart, and useless for testing key
guesses offline. `null` means no `Authorization: Bearer` header was
presented, which on a server started without `FRINK_API_KEY` is every
request. `client` is the caller's own `X-Frink-Client` header, kept to
32 label characters, **a claim, not proof**: Frink Studio sends
`frink-studio` and so could anything else. Nothing authenticates it, and
a UI that shows it must say so.

## Embeddings

Two different things can answer `POST /v1/embeddings`, and they are not
equally good:

| Source | How it is loaded | Default pooling | Normalized |
|---|---|---|---|
| A **BERT/BGE encoder** | `FRINK_MODEL_PATH` at an encoder-only GGUF, `/admin/models/load`, or `FRINK_EMBEDDING_MODEL_PATH` beside a generative model | the checkpoint's own `bert.pooling_type` (`CLS` for every BGE) | yes |
| A **decoder's hidden states** | whatever generative model is loaded | `mean` | no |

The encoder is the one that was trained to put a sentence
representation somewhere; the decoder path is a best-effort reading of a
model that was not. `embedding_type` overrides the default: `mean` and
`last` on both, `cls` on the encoder path only (row 0 of a decoder's
hidden states is its BOS position and means nothing in particular).
`none` and `rank` are refused on both, because this response shape carries one
vector per input, not per token, and `rank` is a reranker
classification head frink does not implement.

### An encoder as the served model

`FRINK_MODEL_PATH=bge-small-en-v1.5-q8_0.gguf` works. The file's
`general.architecture` is what decides: an encoder/embedding
architecture never reaches a decoder loader.

`GET /v1/models` then says what it is, so a client does not have to send
a request to find out:

```json
{"id": "bge-small-en-v1.5", "object": "model",
 "frink_model_kind": "embedding",
 "frink_endpoints": ["/v1/embeddings"],
 "frink_n_embd": 384, "frink_pooling": "CLS",
 "frink_context_length": 512,
 "frink_tokenizer": "gguf-wordpiece"}
```

`GET /health` is `ready`, the server really can serve. Every
generation endpoint (`/v1/chat/completions`, `/v1/completions`,
`/completion`, `/v1/messages`, `/v1/responses`) answers `501` naming the
model rather than blaming a tensor:

```json
{"error": {
  "message": "the loaded model 'bge-small-en-v1.5' is an embedding model (bert encoder, 384 dims, pooling CLS). An encoder has no output head, so it cannot generate text at all. There is no next token for it to predict. POST /v1/embeddings to use it, or load a generative checkpoint.",
  "type": "unsupported",
  "param": "model"
}}
```

`/v1/tokenize` and `/v1/detokenize` are the exception: they ANSWER for
an encoder, because an encoder has a real tokenizer and asking what
tokens it saw is not asking it to generate. They used to refuse with the
rest, which left no way to check a surprising embedding without loading
the checkpoint in a second tool. `add_special` is a no-op there, since
an encoder wraps its own `[CLS]`/`[SEP]` already, and those are the same
tokens `usage.prompt_tokens` counts on an embeddings response.

Every route that needs a decode still refuses.

Only `bert` loads. The other encoder rows upstream builds from
`bert.cpp`: `nomic-bert`, `jina-bert-v2/v3`, `neo-bert`,
`modern-bert`, `eurobert`, `t5encoder`, and the decoder-style
`llama-embed` / `gemma-embedding` all refuse **by name**, saying what
each one needs (RoPE, a gated FFN, per-projection QK norm, its own
graph). `pangu-embedded` was in that list from its name alone and is a
decoder LLM (openPangu-Embedded); it runs on the decode routes since
2026-09-14. `frink_models::embedding_model::NOT_YET` is that
list, and a test pins it against the capability registry so a new row
cannot fall through to a generic refusal.

## Reranking

`POST /v1/rerank`, and llama.cpp's `/rerank` on the same handler.

Cross-encoder reranking against the loaded checkpoint's own
classification head (`cls` / `cls.output`), NOT the cosine of two
embeddings. The cheap version, embed the query and each document and
sort by similarity, is a different model answering a different question,
and it would be served with a 200 and read as a ranking.

Request `{"model"?, "query", "documents": [...], "top_n"?,
"return_documents"?}`. Response `{"object": "list", "model", "results":
[{"index", "relevance_score", "document"?}], "usage"}`, best first.

- `index` is the position in the request's own `documents` array, not
  the position in the sorted result.
- Ties keep request order; the sort is stable.
- `top_n` clamps to `documents.len()`, and `top_n: 0` returns no
  results. An empty `documents` array is a 400.
- Needs a `bert` GGUF carrying a rank head, through `FRINK_MODEL_PATH`
  or `FRINK_EMBEDDING_MODEL_PATH`. Anything else answers **501 naming
  the model** rather than substituting a similarity.

`/v1/embeddings` against a rank-head checkpoint still refuses: its
`pooling_type` is RANK, which is a classification head and not a pooling
rule, and that refusal is what sends a caller here.

The pair is encoded as `[CLS] query [SEP] document [SEP]`, with the
query in segment 0 and the document in segment 1.

**This deliberately differs from llama.cpp**, which hardcodes token
types to zero, and it is one of the few places frink does. The reason
is measured rather than preferred: segment ids do not merely shift the
scores, THEY REORDER THE RESULTS. On "How many people live in Berlin?"
against five documents, the one carrying the population figure ranks
first with segments 0/1 and FIFTH with segments all zero. Matching
HuggingFace is what makes the ranking right, so a reranker checkpoint
whose token-type table has fewer than two rows is refused at load.

Verified end to end against `ms-marco-MiniLM-L6-v2`, with scores
matching an independent NumPy transcription of
`BertForSequenceClassification` to 4e-3 and the order matching the
HuggingFace reference on four query sets.

### Which head ran, and why it changes the scale

**A rerank response reports `frink_score_head`**, and a client that
thresholds on an absolute score has to read it. There are two regimes
and they differ by more than an order of magnitude:

| `frink_score_head` | Meaning | Range on `ms-marco-MiniLM-L6-v2` |
|---|---|---|
| `classifier(tanh(pooler(cls)))` | The full head the checkpoint was trained with | about plus or minus 11 |
| `classifier(cls)` | The classifier alone, no pooler in the file | about plus or minus 0.2 |

The ORDER is the same either way; the scale is not. A client carrying
`relevance_score > 0.5` from another engine gets a threshold that never
fires in the second regime, which is the failure this field exists to
prevent. A load-time NOTE says the same thing to an operator, because a
log is what an operator sees and a JSON field is the only one a client
can act on.

Every published GGUF is the second regime. llama.cpp's converter drops
the pooler by name (`conversion/bert.py`, `BertModel.filter_tensors`,
"we are only using BERT for embeddings so we do not need the pooling
layer"; still unconditional on `master` as of 2026-09-11), and under
llama.cpp's tensor naming the pooler slot is `cls`, with `cls.output`
being the classifier that follows it (`src/llama-graph.cpp`: `cls`,
then bias, then `tanh`, then an optional head norm). frink reads `cls`
when a file carries it and refuses to invent one when it does not.

An absent pooler is a NOTE rather than a refusal on purpose: nothing in
the file distinguishes "trained without a pooler" from "converted
without one", and `jina-reranker-v1-tiny-en` is a real checkpoint whose
head IS a direct projection. Refusing would reject a valid model to
flag a lossy conversion.

### Getting the first regime: `frink splice-pooler`

The pooler is in the checkpoint's own safetensors, and `frink
splice-pooler` writes a GGUF that carries it (`docs/CLI.md`). The tie
between the two files is the classifier they both hold, compared
element-wise to within the GGUF's storage precision; the GGUF's
`general.name` is deliberately NOT trusted, because the published
`ms-marco-MiniLM-L6-v2-Q8_0.gguf` names the L12 model. A pooler from
any other checkpoint is refused naming the element that disagrees.

Measured on `ms-marco-MiniLM-L6-v2`, four query sets, seventeen pairs,
against the NumPy transcription of `BertForSequenceClassification`
(`scripts/rerank_reference_ms_marco.py`, `hf` rows):

| | `relevance_score` range | largest deviation from HuggingFace | orderings |
|---|---|---|---|
| converter's GGUF, `classifier(cls)` | about `-0.25 .. 0.15` | not comparable (a different function) | 4 of 4 match |
| spliced, `classifier(tanh(pooler(cls)))` | `-11.19 .. 10.93` | `0.051` | 4 of 4 match |

The "How many people live in Berlin?" set, converter's file then
spliced, HuggingFace in the last row:

```text
[-0.027, 0.073, -0.175, -0.250, 0.009]
[-4.302, 8.604, -11.100, -11.188, 0.647]
[-4.320, 8.607, -11.101, -11.188, 0.637]
```

llama.cpp loads the spliced file and runs the pooler too (its
`build_pooling` RANK arm reads `cls`); its scores still differ from
HuggingFace by its all-zero token types, described above.

Tracked as [#82](https://github.com/antonellof/frink/issues/82): the
converter's output is still uncalibrated, and the splice is the
in-tree route until upstream keeps the tensor.

## Errors that retrying will not fix

Two ceilings do not move with load. A request that hits one gets a `400`
and never a `503`, because an idle server would answer the same way, and
a `Retry-After` here would be a lie that turns into a retry loop.

```json
{"error": {
  "message": "context_length_exceeded: request asks for 9000 token positions …",
  "type": "invalid_request_error",
  "code": "context_length_exceeded",
  "estimated_bytes": 4718592000,
  "limit_bytes": 4194304000,
  "positions": 9000,
  "positions_limit": 8000
}}
```

`code` names **which** ceiling is binding. The two have different fixes,
and only one of them is in the client's hands:

| `code` | Meaning | Fix |
|---|---|---|
| `context_length_exceeded` | Longer than any single request is allowed to be (`FRINK_CB_MAX_CONTEXT`) | Shorter prompt, lower `max_tokens` |
| `device_memory_budget_exceeded` | Bigger than the server's whole KV budget (`FRINK_CB_KV_BLOCKS`) | More KV blocks, or a smaller model |

`estimated_bytes` and `limit_bytes` are the KV cost of the request and
of the ceiling, priced from the model's own layout. It is the same
arithmetic `frink inspect-plan` prints, so check it yourself instead of
taking it on trust. When a request exceeds both ceilings, the
per-request one is reported, because that is the one the caller acts on.

Momentary pressure gets a different answer: `503` with `Retry-After`,
counted separately, so an operator reading `/metrics` tells "one request
too big" apart from "too many requests".

## Stop sequences

Two layers, because a stop sequence promises something about what
reaches the client, and streaming makes that promise hard to keep.

1. **Token level.** A stop string that is exactly one token in this
   model's vocabulary, the usual case for `<|im_end|>`,
   `<end_of_turn>` and `<|eot_id|>`, is matched on the token id before
   the token is detokenized. That catches control tokens whose rendered
   form is not the string the client asked for, which the text scan
   never sees, and the token never becomes part of the answer. This is
   the same treatment the EOS token gets, and it is left out of `usage`
   too.
2. **Text level.** Everything else is matched on the emitted text, with
   the output buffered so that no trailing text still capable of
   becoming a stop string is ever sent. Only the real partial match is
   withheld, not a
   fixed number of bytes, so ordinary text goes out the moment it is
   provably safe rather than permanently trailing the model by the
   length of the longest stop string.

A partial match therefore never appears on the wire and is never taken
back. SSE has no mechanism for taking anything back. Text withheld
against a match that never arrives is released when the answer ends, so
nothing is lost.

## Cancellation

Two tiers, both ending at the same server-side flag.

1. **Drop the connection.** The streaming path now notices that its SSE
   receiver is gone and stops at the next token. Before this it ignored
   the failed send and generated the rest of the answer into nothing.
2. **`POST /v1/cancel`** with the `request_id` the first SSE chunk
   states. Behind `FRINK_API_KEY` like the endpoint that started the
   work. A browser should send it with `keepalive: true` so it survives
   the page unload that killed the stream, which is the case tier 1
   handles worst.

```bash
curl -s http://127.0.0.1:8383/v1/cancel \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"chatcmpl-…"}'
```

`200 {"request_id":"…","cancelled":true,"detail":"…"}` when a live
generation was signalled, `404` with `"cancelled": false` when the id
names nothing that is running. The two are different facts: only one of
them saved any work, and a client told `ok` for both would claim it
stopped something it did not.

### Stop strings

A generation that runs into one of the caller's `stop` strings reports
`finish_reason: "stop"` here. OpenAI's vocabulary has no other value,
and inventing one would make a completed answer look like a failure to
a client checking against the documented set. The stop string itself is
trimmed from the answer.

`/v1/messages` says more, because Anthropic's protocol has room for it:
a caller's stop string that fires reports `stop_reason: "stop_sequence"`
with `stop_sequence: "<the string>"` beside it, on the buffered body and
in the terminal `message_delta` alike. The two are only ever reported
together. Two things are deliberately *not* reported that way:

- A stop the **server** added, the served template's own end-of-turn
  marker, reports the ordinary `end_turn`. Telling an agent it hit a
  fence it never put up is worse than telling it nothing.
- A **truncated** generation reports `max_tokens` even if a stop
  matched, and a generation that produced tool calls reports
  `tool_use`. What the client does next is decided by those, not by the
  fence.

A cancelled stream ends with `finish_reason: "cancelled"`. OpenAI does
not define that value, because OpenAI has no cancel endpoint, but it is
*a* finish reason, so no client reads the stream as truncation. Tokens
already decoded are kept and reported in `usage`.

Continuous-batching requests are covered too, by a different route. The
cancel enqueues an id, then the scheduler thread drains that queue
between steps and drops the sequence itself. Removing a sequence means
touching KV buffers a forward pass might be reading, so the mutation
happens on the thread that owns the step rather than wherever the cancel
arrived. A request stops wherever it has got to, whether still queued,
mid-prefill, or decoding, and keeps whatever it produced.

The mechanism has one honest edge: the flag is read between decoded
tokens, so a prefill already inside a forward pass runs to completion.
`/v1/completions` is buffered and registers no id. `/v1/messages` does
register one, but the Anthropic protocol has no field to state it in --
the `message_start` id is a per-message `msg_...` the cancel registry
does not know -- so the server puts it in a `request-id` response
header, spelled as the upstream API spells it. Cancel a streamed
`/v1/messages` with that value.

## Keepalives

Every streamed endpoint sends a keepalive after 15 seconds of silence,
and every one of them is a **data frame**, never an SSE comment. axum's
own `Sse::keep_alive` writes `: ping`, which a proxy sees but a client's
event handler never does. A client's stream-idle timeout is armed
on received events, so a comment-kept stream gets torn down and
reconnected in the middle of a long prefill, which is exactly when a
keepalive was supposed to help.

| Endpoint | Keepalive frame |
|---|---|
| `/v1/chat/completions` | a `chat.completion.chunk` with an empty delta |
| `/v1/stream/{id}` (replay) | an id-less chunk with `choices: []` |
| `/v1/responses` | `response.in_progress` |
| `/v1/messages` | `ping` |

The silence *before* the first token counts: that window is the queue
wait plus the prefill, and on a long prompt it is the longest quiet
stretch of the request.

A replayed stream never re-delivers keepalives, and the replay
keepalive carries no `id:`, so it does not enter the sequence a
`Last-Event-ID` resumes from.

## Streaming behind a proxy

Streamed completions carry `X-Accel-Buffering: no` beside axum's
`Cache-Control: no-cache`. nginx, and the proxies that copied its
convention, buffer `text/event-stream` by default. That turns a
token-by-token stream into one silent wait followed by the whole answer,
which looks exactly like a hung backend from the client side.

A keepalive data frame goes out every 15 s, so an idle but healthy
stream still puts bytes on the wire. Measure your stall timeout against
bytes received rather than tokens. A long prefill then never trips it,
and a swallowed connection does.

### Resumable streams

`id:` is a promise that a reconnect can pick up where the connection
stopped, so it is emitted **only** where a replay buffer exists. Ask for
one per request:

```jsonc
{"model": "...", "messages": [...], "stream": true, "stream_resumable": true}
```

Then every event carries `id: {request_id}:{n}` and the first one also
carries `retry: 1500`. Two ways back in, both behind `FRINK_API_KEY`
like the request that filled the buffer:

```bash
# Reconnect over SSE, continuing after the last event seen
curl -N http://127.0.0.1:8383/v1/stream/chatcmpl-… \
  -H 'Last-Event-ID: chatcmpl-…:41'

# Or drain the same buffer over plain JSON
curl "http://127.0.0.1:8383/v1/stream/chatcmpl-…/poll?from=42"
# {"request_id":"…","events":[{"index":42,"data":"{…}"}],
#  "next_index":43,"done":false}
```

`data` is the exact payload the stream sent, `[DONE]` included, so a
client feeds replayed events to the same parser as live ones. `done` is
`true` only once the generation has ended *and* the buffer is drained,
so a client that stops on it never discards events it was not given.
A poll parks for up to 10 s waiting for the next event, which keeps it
reading like a stream while staying a short JSON response, the thing a
buffering proxy cannot hold back, which is the whole point of the
fallback.

**`stream_resumable` also changes what a dropped socket means.** Without
it, the connection closing cancels the generation (tier 1 above). With
it, the generation keeps running into the replay buffer, that is the
point of asking for one, and `POST /v1/cancel` is its stop path. The
caller decides because neither answer suits both: a tab that navigated
away wants the CPU back, and a tab whose proxy dropped a 90-second
answer wants the answer.

Both bounds fail closed rather than quietly. The replay window is 1 MiB
per stream; past it the oldest events are dropped and asking for an
evicted position answers `410 replay_window_lost` rather than a stream
with a hole in it. A finished stream stays reconnectable for 120 s and
at most 64 streams are remembered, oldest finished evicted first, a
live stream is never evicted out from under its reader. A `Last-Event-ID`
naming a different request is `400 bad_last_event_id`, not silently
rounded down to a full replay of the wrong answer. An unknown or
forgotten id is `404 stream_not_found`.

## Continuous batching

Concurrent requests share one batched decode worker (llama.cpp's slot
model): many in-flight sequences, one `forward_multi_seq` step per tick.
This is the supported multi-client path on Metal.

Enable explicitly with `FRINK_CONTINUOUS_BATCHING=1`, `--cont-batching`
/ `-cb`, or `-np N` / `--parallel N` (which also sets
`FRINK_CB_MAX_SEQS`). Disable with `FRINK_CONTINUOUS_BATCHING=0` or
`--no-cont-batching`. When unset on Metal builds with fused attention,
continuous batching is **on by default** unless a KV pool or prefix cache
forces the private decode path.

Mutually exclusive with `FRINK_KV_POOL_BLOCKS` and
`FRINK_PREFIX_CACHE_ENTRIES` on the contiguous path (paged KV lifts
this; see [`CONFIG.md`](CONFIG.md)).

Streaming under continuous batching emits tokens incrementally as they are
sampled (same SSE shape as the private decode path), not one string at
the end.

Prefill is chunked. Per tick the scheduler runs one bounded prefill
chunk (`FRINK_CB_PREFILL_CHUNK`, default 128 tokens, round-robin
across waiting prompts) plus one batched decode step. A long prompt
joining the batch therefore costs an in-flight decode one chunk instead
of the whole prompt.

`FRINK_CB_MAX_SEQS` caps in-flight sequences, counting prompts still
prefilling. `FRINK_CB_MAX_QUEUE` (default 512) caps how many requests
wait for admission. Past that cap a request gets a `503` with a
`Retry-After` header instead of joining an unbounded queue, and the JSON
body names the queue depth, the cap and `retry_after_seconds`.

`-b N` / `--batch-size N` and `-ub N` / `--ubatch-size N` set the
prefill chunk on both decode paths at once (`FRINK_CB_PREFILL_CHUNK`
here and `FRINK_CHUNKED_PREFILL` on the private loop, which used to be
two independent knobs for one number). frink has one prefill stage, so
the two flags resolve the way llama.cpp resolves them
(`src/llama-context.cpp:265`): the smaller of whichever was named.

While a batcher is active, `GET /metrics` reports
`frink_prefill_chunks_total`, `frink_prefill_tokens_total`,
`frink_decode_steps_total`, `frink_scheduler_queue_depth`,
`frink_scheduler_queue_rejected_total`, and the configured caps
`frink_scheduler_max_seqs` (`-np`; 0 when unlimited) and
`frink_scheduler_prefill_chunk` (`-ub`), so a flag can be read back
from the process that received it rather than trusted.

Every `503` this server returns carries `Retry-After: 1`. It is a fixed
hint, not a computed one. An honest estimate would need the caller's
throughput, and the server does not know it.

## Slot save and restore

llama.cpp's `POST /slots/{id_slot}?action=save|restore`, so a long
system prompt survives a restart instead of being prefilled again. The
route, the `action` query, the gate and the response fields are
llama.cpp's (`tools/server/server.cpp:273`,
`server-context.cpp:4536-4567`, `server-task.cpp:1570-1592`); a client
written against `llama-server` works unchanged.

```bash
# Start with somewhere to put the files, and the prefix cache the
# slots restore into.
FRINK_PREFIX_CACHE_ENTRIES=8 frink-server -m model.gguf \
  --slot-save-path ./slots --port 8383

# Prefill a prompt, store its KV in the prefix cache, and write it to
# ./slots/system.fslot
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

Any completion whose prompt starts with those tokens then reports
`cached_tokens: 83` in `usage` and prefills only the remainder, with
the same greedy output the cold server gave.

Two things differ from llama.cpp, and are said rather than emulated:

- **`id_slot` is bookkeeping.** llama.cpp has N fixed slots each owning
  a KV region, and `-np` sets N. frink builds KV per request and
  shares prefixes through one `PrefixCache`, so there is no per-slot
  region for the id to select. It is validated (a non-negative
  integer) and echoed.
- **`save` names its `prompt`.** llama.cpp saves whatever the slot was
  last serving; nothing here is "last serving" anything. The body
  carries the prompt to prefill, tokenized with BOS exactly as a
  completion would be.

`action=erase` is `501`, not a no-op: the prefix cache has whole-cache
clearing and no per-entry eviction, so erasing one slot would mean
erasing all of them. Delete the file, or restart.

**A restore is refused, by name, when the file is not of the model
being served.** A slot file is raw attention state; restoring it under
other weights does not fail, it answers wrongly. So the file carries
the checkpoint's identity and a restore compares it before storing
anything:

```
400 slot_model_mismatch
refusing to restore ./slots/system.fslot: checkpoint: the slot was saved
under 2b77f720... and this server is serving 3a59835e.... Restoring
attention state computed by other weights does not fail, it answers wrongly
```

The identity is a SHA-256 over the GGUF's metadata (sorted, typed),
its full tensor directory, and 4 KiB from the head and tail of every
tensor, beside the layer count, KV head count, head dimension and KV
dtype. Path, size and hyper-parameters were each rejected as the key
because each admits the likely pair, two fine-tunes of one base at one
quantisation. The same model at a different quantisation is refused on
`checkpoint`; a different model is refused on `model` or the first
shape field that differs. llama.cpp's own slot file
(`src/llama-context.cpp:3081-3141`) carries no identity at all.

The file is framed with a length-and-digest prefix like
`frink_core::kv_disk`'s blocks: a truncated, edited or foreign file is
refused before anything is parsed or allocated. Filenames are a strict
whitelist (`[A-Za-z0-9._-]`, no leading dot), so a name cannot leave
`--slot-save-path`.

Slots are implemented for the generic GGUF decoder. The dedicated
engines (Kimi, MLA, Gemma-4, GLM-5.2) refuse by name; so does a server
without a checkpoint on disk.

## LoRA adapters

llama.cpp's `GET /lora-adapters`, `POST /lora-adapters` and the
per-request `lora` field, over adapters loaded with `--lora` /
`--lora-scaled` (see [`CLI.md`](CLI.md#lora-adapters)).

```sh
frink-server -m model.gguf --lora style.gguf --lora-scaled tone.gguf:0.5

curl localhost:8383/lora-adapters
# [{"id":0,"path":"style.gguf","scale":1.0,"task_name":"","prompt_prefix":""},
#  {"id":1,"path":"tone.gguf","scale":0.5,"task_name":"","prompt_prefix":""}]

# Set the scales for every request that does not override them. An
# adapter not named goes to 0, as upstream's construct_lora_list sets it.
curl localhost:8383/lora-adapters -d '[{"id":1,"scale":1.0}]'
# {"success":true}

# One request under its own scales; the server's are untouched after.
curl localhost:8383/v1/chat/completions -d '{"model":"m",
  "messages":[{"role":"user","content":"hi"}], "lora":[{"id":0,"scale":0.25}]}'
```

`GET` reports the scale currently applied, which is what
`--lora-init-without-apply` starts at (0) rather than the flag's
number. `POST` takes the `[{id, scale}]` array (`scale` absent reads
0, as upstream's `json_value` default); an `id` no adapter has is a
400 naming the range, where upstream ignores it. A request's `lora`
list means what `construct_lora_list` makes it mean -- every adapter
named gets its scale, every other adapter 0 for that request -- and
an EMPTY list means the server's own scales, as upstream reads it.
Naming an adapter on a server that loaded none is a 400; the field on
a checkpoint served by a dedicated engine (MLA, Gemma-4, GLM-5.2,
Kimi) is a 501.

An adapter's scale is one atomic read by every projection at apply
time, so a change costs nothing and recomputes nothing. What it costs
instead is exclusivity: a `POST`, or a request whose `lora` list
differs from what is applied, waits for the generations in flight and
runs alone, then the request's override is restored. A request whose
list names exactly the current scales is an ordinary concurrent
request. This is llama.cpp's rule -- two slots with different LoRA
lists are never co-batched -- at the granularity of the whole server
rather than the batch. The response cache keys on the scales a
generation ran under, so a `POST` between two identical requests does
not serve the first answer to the second.

## MCP

`--mcp-config PATH` loads server metadata under `frink_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Reasoning content

A checkpoint whose family reasons inside markers (`<think>`, DeepSeek's
DSML variant, Gemma's channels, MiniMax-M3's namespaced pair, the
gpt-oss harmony `analysis` channel) has that block split out of
`content` and returned as `reasoning_content`, on the message and on
each streamed delta. The split runs *as tokens arrive*, so an
overlapped SSE stream and a buffered one report the same fields for the
same request rather than differing by transport.

Which family applies is inferred from the served model's name, which is
all there is: frink carries no per-checkpoint parser declaration. A
name that implies nothing gets no reasoning parser at all, the right
answer for a model that does not reason, since an unconditional
splitter would eat a literal `<think>` written in a code block.

### Replaying it

`reasoning_content` is accepted on a request's assistant messages, not
only returned on responses, and `reasoning` is accepted as an alias so a
turn can be replayed in the shape `/v1/responses` or `/v1/messages`
handed it back. It is passed to the template on its own key, under
*both* spellings, since the DeepSeek and GLM lineages iterate
`message.reasoning_content` while Qwen and gpt-oss read
`message.reasoning`, and a template treats the key it does not know as
undefined. An empty string is dropped rather than passed, so a template
that opens its thinking markers on `if message.reasoning_content` does
not wrap them around nothing.

It is never folded into `content`. A past chain of thought shown as
content is something the model believes it said out loud, which is the
whole reason the family markers exist. A template that knows nothing
about reasoning ignores the key and renders exactly what it did before.

Whether the block is *already open* is read off the rendered prompt, not
guessed from the family: a template asked to think can open the block in
the prompt itself, and then the model's first token is reasoning and no
opening marker ever arrives. Same checkpoint, same request text,
different answer depending on what `chat_template_kwargs` rendered, so
the prompt is what gets consulted.

## Tool-call formats

`tools` is still prompt-engineered. A grammar engine exists now, but
and it is now wired to `tools`: a forced `tool_choice` compiles the
offered tools into a lazy grammar, so the call itself is
schema-constrained. The preamble asks for a Hermes-style
`<tool_call>{"name": …, "arguments": {…}}</tool_call>`. But a model
trained on a different format frequently answers in its own, correctly
and in its own terms. Parsing tries the format the served checkpoint's
family implies, then the one the preamble asked for:

| Family | Shape |
|---|---|
| Hermes / Qwen2.5 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
| Llama 3 | `<\|python_tag\|>{"name": …, "parameters": {…}}` |
| Mistral | `[TOOL_CALLS] [{…}]` |
| Qwen3-Coder | `<function=name><parameter=key>value</parameter></function>` |
| GLM-4.7 | `<arg_key>k</arg_key><arg_value>v</arg_value>` |
| DeepSeek | `<｜DSML｜invoke name="…"><｜DSML｜parameter name="…">…` |
| MiniMax | `<minimax:tool_call><invoke name="…"><parameter name="…">…` |
| MiniMax-M3 | `]<]minimax[>[<tool_call>` around a namespaced element grammar, a different protocol rather than renamed tags |
| gpt-oss | `<\|channel\|>commentary to=functions.name<\|message\|>{…}<\|call\|>` |
| Gemma 4 | `<\|tool_call>call:name{k: v}<tool_call\|>` |
| muse-glimmer | `<atem:invoke>` / `<atem:parameter>` inside an ATEM channel block |

Eleven formats, and the parser chooses among them from the served
model's name. Every call in a response is returned, not just the first, with ids
`call_0`, `call_1`, and so on. Nothing in these formats carries an id, so a
client correlates by index.

**Streaming.** On the non-batched path five invoke/parameter families
stream their arguments as deltas: Qwen3-Coder, GLM-4.7, DeepSeek,
MiniMax and muse-glimmer. The first delta of a call carries `index`,
`id`, `type` and `function.name` with empty arguments, and every delta
after it carries only more `function.arguments` text. The fragments are
literal continuations, so a client concatenates them in `index` order
and parses the result, which is what lets a coding agent watch a file
argument arrive instead of waiting for it.

`/v1/messages` and `/v1/responses` stream the same events in their own
protocols rather than in OpenAI's: Anthropic sends `content_block_start`
with a `tool_use` block and then `input_json_delta` fragments, and the
Responses API sends `response.output_item.added` followed by
`response.function_call_arguments.delta`.

The JSON-payload families (Hermes, Llama 3, Mistral, MiniMax-M3,
gpt-oss, Gemma) still arrive whole, because a half-written JSON object
is not a fragment anyone can use. A generation truncated mid-call
reports `length`, not `tool_calls`: a half-written call must not be
executed. A call naming a tool the request never
offered is dropped rather than forwarded; a namespaced name
(`skills:read`) is forwarded, because the client is what resolves the
namespace.

The XML-ish formats state no types, so a parameter's value is typed
from the tool's own `parameters` schema: a declared `string` is handed
over verbatim, which is what keeps a zero-padded id like `"018956"`
from arriving as a number.

## Legacy completions

`/v1/completions` honours `stop`, `max_tokens` (default 16, because the legacy
floor is right *here*, where a caller completing a fragment usually
wants a fragment back), `temperature`, `top_p`, `min_p`, `top_k`,
`typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, the five
`dry_*` fields, `samplers`, `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `ignore_eos` and `grammar`.

Four of those are recent. `top_k`, `repetition_penalty`,
`presence_penalty` and `frequency_penalty` were undeclared on this
route, so serde dropped all four while the chat route honoured every
one, and the sampler then had `top_k: 0` and `repetition_penalty: 1.0`
hardcoded over the top. Both routes now read the same
`SamplingKnobs::resolve`, so which knobs exist and what an absent one
means are stated once rather than twice. An absent `min_p` resolves to
`0.0` (off) rather than llama.cpp's CLI default of `0.05`: an HTTP
caller is not running llama.cpp's command line, and quietly truncating
every unconfigured request is a behaviour change nobody asked for.

Everything it does not implement comes back as an error **naming the
field**, rather than being dropped: token-id prompts (`[int]` / `[[int]]`), `logprobs`, `echo`,
`suffix`, `logit_bias`, and any `response_format` other than
`{"type": "text"}`. Serde drops an undeclared field silently, and a
caller cannot tell that apart from having had it honoured, which for
`stop` in particular means believing generation will halt at a sentinel
and instead getting the full budget of text past it.

**`logit_bias` is refused on `/v1/chat/completions` too**, and by the
same rule rather than a second copy of it. Until recently only
`/v1/completions` refused it: the chat request struct had no such field,
so serde dropped it and the caller got a 200 with unbiased output. Two
routes disagreeing about one parameter is the shape of bug this API
surface keeps producing, so both now call one `refuse_logit_bias`.

One deliberate exception: **an empty `logit_bias: {}` is served, not
refused**, on both routes. Several OpenAI clients send an empty map on
every request, and there is no token whose logit it would move, so
refusing it is a false refusal. A non-object (`[]`, `"none"`) is still
refused, so nothing falls through the empty-map hole.

## llama.cpp's native `/completion`

`POST /completion` and `POST /completions` are llama.cpp's own
completion endpoint, which is **not** `/v1/completions` with a shorter
path. Different request fields, a different response object, and a
different stream. It is what llama.cpp's own web UI, `llama.vim` and a
long tail of wrappers speak; before it existed here they got a 404.

| | `/completion` (llama.cpp) | `/v1/completions` (OpenAI) |
|---|---|---|
| budget | `n_predict`; `-1`, and absent, mean "until the context is full" | `max_tokens`, default 16 |
| repetition | `repeat_penalty`, `repeat_last_n` | `repetition_penalty` |
| response | flat object: `content`, `stop`, `stop_type`, `timings`, `generation_settings` | `choices[].text`, `finish_reason` |
| stream frame | `data: {"content":…,"stop":false}` | `data: {"choices":[{"text":…}]}` |
| stream end | the terminal object with `"stop": true`, and **no `[DONE]`** | `data: [DONE]` |

Both routes are the same handler; `/admin/stats` records whichever one
the client called. Generation itself is the same engine, sampler,
grammar seam and stop machinery `/v1/completions` uses. This endpoint
is a wire, not a second implementation.

### What it honours

`prompt` (string, or `{"prompt_string": "…"}`) · `n_predict` ·
`stream` · `stop` · `temperature` · `top_p` · `min_p` · `top_k` ·
`typical_p` · `top_n_sigma` · `xtc_probability` · `xtc_threshold` ·
`dry_multiplier` · `dry_base` · `dry_allowed_length` ·
`dry_penalty_last_n` · `dry_sequence_breakers` ·
`repeat_penalty` · `repeat_last_n` · `presence_penalty` ·
`frequency_penalty` · `samplers` · `seed` (`-1` draws one) ·
`ignore_eos` · `grammar` · `cache_prompt`.

The nine samplers of llama.cpp's default chain all run, in llama.cpp's
order (`penalties, dry, top_n_sigma, top_k, typical_p, top_p, min_p,
xtc, temperature`). An absent field resolves to the value that makes its
sampler a no-op, so a request that names none of them is sampled exactly
as it was before they existed.

`dry_sequence_breakers` are STRINGS, tokenised against the loaded
model's own vocabulary (llama.cpp's `get_overlapping_token_sequences`).
A checkpoint with no real vocabulary — the synthetic-weight fallback —
**refuses** `dry_multiplier` rather than running DRY with no breakers,
because a breaker-less DRY looks like working DRY and penalises across
every boundary the caller named. Send `dry_sequence_breakers: []` to ask
for DRY with no breakers deliberately.

`n_predict` keeps llama.cpp's meaning exactly, including the part that
is easy to get wrong: **an absent `n_predict` is `-1`**, which means
"generate until the context is full", and this server does not quietly
substitute a smaller budget. `-1` resolves against the derived context
ceiling (`crate::budget`) minus the tokenized prompt. On a deployment
where the model could not be priced there is no ceiling to be full of,
and `-1` is a **501** naming `n_predict` and `FRINK_CB_MAX_CONTEXT`
rather than a silent 16. `n_predict: 0` generates nothing, as upstream.

`repeat_last_n: -1` ("the whole context") is refused for the same
reason: shrinking it to the default 64 would give a caller a window
orders of magnitude smaller than the one it asked for.

`cache_prompt: true` is upstream's default and is a *permission* to
reuse KV, so it is always servable. `cache_prompt: false` is a
*requirement* not to, which this server can only keep when no prefix
cache is configured; with `FRINK_PREFIX_CACHE_ENTRIES` set it is
refused rather than ignored.

### What it refuses, by name

Every option below deserializes and is checked against the value at
which llama.cpp itself treats it as off. **At that value it is served**,
because a stock client sends most of them explicitly and refusing
`mirostat: 0` would be a false refusal. At any other value it is a 501
naming the field:

`dynatemp_range` · `mirostat` ·
`n_probs` and `post_sampling_probs` (no per-token logprobs) ·
`min_keep` · `return_tokens` (the decode loop hands this layer text,
not ids) · `n_indent` · `n_keep` (frink refuses an oversized request
rather than shifting context, so there is nothing to protect) ·
`n_cmpl` · `n_cache_reuse` · `t_max_predict_ms` · `id_slot` (no slots)
· `response_fields` · `return_progress` · `timings_per_token`
· `sse_ping_interval` (the keepalive is fixed at 15s).

Also refused: `logit_bias` (through the same rule both OpenAI routes
use), token-id prompts, multiple prompts in one request, and
`prompt.multimodal_data`.

`json_schema` is llama.cpp's bare-schema field and is **supported**,
compiled through the same site as `response_format: json_schema`. A
`grammar` and a `json_schema` in one request are refused (400) rather
than ranked; upstream prefers `grammar` and drops the schema, which is
a 200 whose answer need not match the schema the caller sent.

Options that only *parameterise* a switched-off sampler (
`mirostat_tau`, `mirostat_eta`, `dynatemp_exponent`) are deliberately
not in that list: they do nothing while their switch is off, and their
switch is. A field
llama.cpp does not define either is ignored, exactly as upstream
ignores it.

### Response fields, and two honest ones

`truncated` is always `false`, and that is a statement rather than a
placeholder: frink refuses a request that does not fit its context
instead of discarding tokens to make it fit, so a served answer was
never truncated. A `timings` value this server did not measure is
`null` rather than `0`, which would read as an instantaneous prefill,
and `cache_n` is `-1` (upstream's sentinel) when no prefix cache is
configured, which says something different from "the cache missed".

`stop_type` uses llama.cpp's vocabulary (`eos`, `word`, `limit`) plus
one value it has no word for: **`cancelled`**, for an answer stopped
through `POST /v1/cancel` or by the client disconnecting. None of
upstream's three is true of an interrupted answer, and `none` means
"still generating", so folding it into one of them would report an
interruption as a normal finish.

**One known inaccuracy in `stop_type`.** A caller stop string that is
exactly *one token* in the model's vocabulary is caught by the token
layer of the stop machinery, which matches on the id before
detokenizing and does not carry back which string it was; it is
reported as `"eos"` where llama.cpp would say `"word"`. This is not
local to this endpoint (`/v1/messages` loses its `stop_sequence`
attribution on the same inputs), and closing it means giving the stop
matcher the id-to-string mapping it currently discards. Multi-token
stop strings, which is what a `/completion` caller normally sets, are
reported correctly.

## Tokenize / detokenize, in both dialects

These two are the one place frink invented a path OpenAI does not have.
OpenAI has no tokenize endpoint at all, so `/v1/tokenize` was frink's
own spelling, while llama.cpp serves `/tokenize` and `/detokenize`
unprefixed. Every llama.cpp client therefore asked for a URL that did
not exist and got a 404 naming nothing.

**Both spellings now answer, on the same handler.** `/tokenize` and
`/v1/tokenize` are one function, not two implementations, so they cannot
drift; `/admin/stats` records whichever path the client actually called,
so the split between dialects stays visible.

The request body accepts both dialects on both paths:

| Field | Status |
|---|---|
| `content` | llama.cpp's name for the text. Supported |
| `prompt` | frink's name for the same text. Supported |
| both at once | **400.** They are one field, and guessing which was meant would tokenize text the caller did not ask about |
| neither | **400** naming both. llama.cpp answers an empty array here; frink does not, because an empty array cannot be told apart from tokenizing `""` |
| `add_special` | Supported. Prepends the same BOS id the generation path prepends, including its no-op on a checkpoint whose metadata says not to add one, so the count matches the prompt the model would really see |
| `parse_special: true` (upstream's default) | Supported. Special-token markers in the text are the tokens they name |
| `parse_special: false` | Supported. `<|im_start|>` is tokenized as the characters it is written with, as llama.cpp does |
| `with_pieces` | **501 by name.** frink's tokenizers expose decoded text, not the raw per-token piece bytes, so a byte-fallback token could not be given llama.cpp's `piece` byte array |
| `model` | Accepted and ignored; this server serves one model at a time |

The tokenize response carries `tokens` (llama.cpp's key) plus frink's
own `count`. The detokenize response carries the text under **both**
`content` (llama.cpp's key) and `text` (frink's), same string, so
neither dialect's client reads a null.

## Grammar-constrained decoding

`"grammar": "<GBNF>"` is llama.cpp's own field and takes the same
syntax. It is accepted on `/v1/chat/completions` and `/v1/completions`,
compiled once per request, and enforced on **every** token by a real
stack machine, not by masking characters.

That distinction is the point. `response_format: json_object` masks
character classes, so it cannot know whether a `}` closes an object that
was actually opened; a grammar can. The two compose, and neither can
un-mask what the other forbids.

A grammar that is **satisfied** and has no legal continuation ends the
response normally (`finish_reason: "stop"`). Only a grammar that dead-ends
while still unsatisfied is an error, and it is a 400 with
`failure_code: "invalid_grammar"`. Under continuous batching that failure
is scoped to the row that hit it; other requests in the batch keep
running.

`response_format: {"type": "json_schema", "json_schema": {"name": ...,
"schema": {...}, "strict": true}}` is compiled to GBNF by the same
converter `tool_choice` uses and enforced by the same stack machine, so
a `required` property cannot be omitted and an `enum` cannot be
invented. `name` and `description` are labels and constrain nothing.

Everything inside `json_schema` that this server does not act on is
refused **by name** rather than dropped: an unknown member is a 400
naming it, and `strict: false` is a 400 too. OpenAI's `strict: false`
asks for best-effort guidance and permits schemas that strict mode
rejects. frink has one behaviour for a schema, which is to enforce it
token by token, so serving that under `strict: false` would report a
guarantee the caller declined. An **omitted** `strict` is enforced.

A schema the converter will not compile is a 400 naming the keyword and
the subschema it sits in, never a 500 and never a grammar that is only
nearly the caller's schema. The converter is stricter than llama.cpp's
on purpose: upstream discards a keyword no branch tested, so
`{"type": "string", "pattern": "^[a-z]+$"}` compiles upstream to a
grammar for *any* string.

A `grammar` and a `json_schema` in one request are two different
constraints on one generation and are refused (400), rather than one of
them being quietly honoured. A forced `tool_choice` beside either is
refused the same way.

One observable difference from llama.cpp: `required` members are
enforced in lexicographic order rather than the schema's declaration
order, because the request body reaches the converter as a parsed value
and this workspace builds `serde_json` without `preserve_order`. Both
are valid grammars for the schema.

`tool_choice: "required"` and a named `tool_choice` are supported, by a
lazy grammar built from the request's own `tools`. Each tool's
`arguments` rule is its `parameters` JSON Schema compiled to GBNF, so a
`required` property cannot be omitted and an `enum` cannot be invented.
A named choice is the same grammar with the union narrowed to one
alternative.

**The trigger is mandatory, and that is a deliberate departure from
llama.cpp.** Upstream forces a call with an *eager* grammar. That does
not survive this server: several families open the reasoning block in
the prompt, so the model's first token is already inside `<think>`, and
a call forced there is read back by frink's own reasoning parser as
thinking, so the caller who demanded a call would get `reasoning_content`
and no call. So the grammar stays lazy, triggers on the wire format's
opening marker, and while awaiting masks *only* the end-of-generation
tokens: the prefix is free, but the turn cannot END until a call has
begun.

The cost, stated: a model that never opens a call runs to `max_tokens`
and finishes `"length"`, a visible failure rather than prose served as
the call that was asked for.

**Ten of the eleven** wire formats are supported. One root rule per
SHAPE, not one per format:

| Shape | Formats | Root rule |
|---|---|---|
| JSON payload | Hermes/Qwen2.5, Llama 3, Mistral | a marker, a JSON object naming the tool, a closing marker |
| Elements | `qwen3_coder`, `glm47`, `minimax`, `deepseekv32`, `minimax_m3` | an invoke element holding one element per argument |
| Harmony | `gpt_oss` | a channel header addressed to `functions.<name>`, then JSON |
| Pairs | `gemma4` | `call:NAME{k:v,k:v}`, the values in gemma's own quoting |

The grammar's literals are built from the SAME marker description the
streaming parser reads with, which is the only reason this is safe to
widen. Two hand-kept tables, one for writing a call and one for reading
it, would drift, and the symptom would be output the engine forced and
then could not parse back.

An argument whose declared type the family's own spelling and this
server's own reader disagree about is refused BY PROPERTY NAME rather
than written approximately. On `gemma4` that is an `object` or `array`
argument: the template writes a composite in gemma's DSL (bare keys,
gemma-quoted strings) and frink reads a value with `serde_json`, so the
spelling the checkpoint was trained to write comes back as a string. On
the element formats it is a property with no declared `type` at all.

One still returns **501 naming the format**:

- `muse_glimmer`: the call boundary is not syntactic. The same ATEM
  block is a call in a tool channel and prose in a user-facing one, and
  forcing the channel HEADER instead would need two facts about the
  checkpoint's template rather than about the format: which recipient
  name it addresses a tool with (the parser accepts any name that is not
  `self` or `user`), and how much of that header the rendered prompt
  already wrote, since a muse-glimmer prompt ends *inside* one. No
  muse-glimmer checkpoint or template is on hand to read either off, and
  guessing is the thing this refusal exists to prevent.

## Not yet

Image and audio input · MCP tool
invocation
(the config is read, nothing is called) · embedding architectures other
than `bert` (they refuse by name, see above) ·
multi-GPU, tensor parallel, prefill/decode disaggregation · streamed
argument deltas for the JSON-payload tool formats (they arrive whole) ·
streamed tool calls on the continuous-batching path · a speculative
decode path for grammar-constrained or paged requests, so their
speculation fields in `usage` are
absent today · llama.cpp's `/infill`, `/props`, `GET /slots` (the live
slot listing; `POST /slots/{id}` save/restore is supported, see above),
and `/apply-template`, none of which has a frink counterpart.

A few request fields deserialize and then go nowhere, accepted so a
stock client's body does not fail validation over something this server
has no use for: `metadata` and `thinking.budget_tokens` on
`/v1/messages`, and `store`, `metadata` and `parallel_tool_calls` on
`/v1/responses`. Everything else this server does not implement is
refused by name.

See [`ROADMAP.md`](ROADMAP.md).
