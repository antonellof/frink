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
| `POST /completion` · `POST /completions` | llama.cpp's native completion endpoint, JSON + its own SSE shape. Not an alias of the line above (see below) |
| `POST /v1/tokenize` · `POST /tokenize` | Supported. The unprefixed spelling is llama.cpp's, on the same handler (see below) |
| `POST /v1/detokenize` · `POST /detokenize` | Supported, same aliasing |
| `POST /v1/score` · `POST /score` | Sentence-pair scoring. One text against many, or two lists of the same length paired element by element; different lengths are a **400** naming both counts. A cross-encoder answers with its head, a bi-encoder with the cosine of two embeddings, and `frink_score_kind` says which. A generative model is a **501** |
| `POST /v1/embeddings` | Supported. A real BERT/BGE encoder when one is loaded (`cls`/`mean`/`last`, L2-normalized), otherwise a mean/last pool of a GGUF decoder's hidden states (see below) |
| `POST /v1/rerank` · `POST /rerank` | Cross-encoder reranking with the checkpoint's own classification head. The unprefixed spelling is llama.cpp's, on the same handler (see below) |
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
| `GET /admin/models` | Checkpoint inventory from GGUF headers (see [Admin control surface](#admin-control-surface)) |
| `POST /admin/models/load` · `POST /admin/models/unload` | Swap or drop the served model. A load is a background task |
| `POST /admin/download` | Fetch one `.gguf` from the Hugging Face Hub into the model directory, as a background task |
| `GET /admin/tasks` · `POST /admin/tasks/{task_id}/cancel` | List background tasks, or ask one to stop |
| `GET /admin/stats` | Counters and the recent-request ring |
| `POST /sleep` · `POST /wake_up` · `GET /is_sleeping` | An unload that remembers the checkpoint path, so the server can wake itself; `/admin/models/unload` cannot. Frees the KV pool, the paged store, the repack and expert caches and any device buffers. Idempotent. A model with no checkpoint path on record is a `409`. While asleep, routes needing a model answer `503 server_sleeping` rather than `model_not_loaded`. There is one sleep level, because frink mmaps its weights |
| `GET /` | 404. The web UI in [`ui/`](../ui) is a separate app and this server does not serve it |
| Audio / images | Not supported |

That is the complete list. Every path is defined as one constant in the
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
| `max_tokens` | Supported; defaults to 32768 (OpenAI's legacy default is 16). An explicit `0` is a 400 |
| `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `stop` | Supported |
| `typical_p`, `top_n_sigma`, `xtc_probability`, `xtc_threshold`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers` | Supported. llama.cpp's own spellings; absent means the sampler is off |
| `samplers` | Supported. The chain order, as a list of names or a `;`-separated string. Defaults to llama.cpp's own default chain |
| `presence_penalty`, `frequency_penalty` | Supported |
| `stream` | Supported (overlapped SSE when tools off and CB off) |
| `tools` / `tool_choice: none\|auto` | Supported (prompt-engineered, parsed in eleven wire formats) |
| `tool_choice: required` / named function | Supported on ten of the eleven wire formats, by a lazy grammar built from the same marker description the parser reads with. **501 naming the format** on the remaining one (`muse_glimmer`), for the reason the refusal states |
| `logprobs` | Served on `/v1/completions` (parallel arrays plus `text_offset`) and on `/v1/chat/completions` as `logprobs: true` + `top_logprobs` (OpenAI's `content[]` shape). Over the distribution the sampler drew from, so penalties and a grammar mask are in it; a candidate the chain removed is omitted rather than reported as `null`. Cap 5 on completions, 20 on chat, above which is a 400. Such a request always misses the response cache, which stores text and not distributions |
| `n` (>1) | Served on `/v1/completions` and `/v1/chat/completions`: prefilled once and the KV forked per choice, so `prompt_tokens` counts the prompt once. Choice `i` samples from `seed + i`, so choice 0 is byte-identical to the `n = 1` answer. With `stream` on the chat route the choices are interleaved, and each chunk carries its own index. **501** on the native `/completion`, which returns a single `content` |
| `best_of` (>1) | Served where `n` is: generates `best_of` from one prefill, returns the `n` best by summed log-probability, as upstream does. The sum favours short answers; a per-token mean would rank differently and is deliberately not used. `usage.completion_tokens` counts every token generated, discards included. `best_of < n` is a 400 |
| `cache_salt` | Served on every store. Names the caller's prefix-cache namespace; the contiguous cache, the response cache and the paged radix tree all key on it, so two callers sending the same prompt cannot be served each other's prefix. Hashed, so only the hash is kept. Absent, empty or whitespace is the shared namespace |
| `prompt_logprobs` | Served on `/v1/completions`: one entry per prompt token, the first `null` because nothing predicted it, each later one with a 1-based `rank`. Scored from the plain softmax instead of the sampler's chain, because a prompt token was supplied rather than drawn. Cap 5. On the paged store such a request declines the prefix tree and runs its whole prompt. **501** on the other wires, which have no field for it |
| `logit_bias` | Served on all three generation wires. `{"<token id>": <bias>}`, added before sampling, range `-100..100`; outside it is a **400** rather than a clamp. `{}` and `null` are accepted. A bias cannot lift a token a grammar, JSON mode, `allowed_token_ids` or `bad_words` forbade |
| `allowed_token_ids`, `bad_words` | Served on all three generation wires, in the same mask as the grammar. They steer the draw; `stop` ends a generation. `bad_words` works on tokens: a word's last token is forbidden only when the tokens before it are what was just generated, so it is exact only for the tokenization the model would have produced. An empty `allowed_token_ids` is a **400** |
| `echo` | Served on `/v1/completions`, the wire that returns a continuation of the prompt; **501** on the two that return a message. Prompt and completion come back as one `text`, and with `logprobs` the arrays cover both. After a truncation it echoes the tokens that were kept |
| `truncate_prompt_tokens` | Served on all three generation wires: keeps the prompt's last `k` tokens, applied before anything is prefilled, so the KV, `usage.prompt_tokens`, the prefix cache and `echo` all see the prompt that was answered. Below 1 is a **400** |
| `skip_special_tokens` | Served on all three generation wires. `false` keeps the model's end-of-generation token in the text and counts it; it still ends the answer. A caller's own `stop_token_ids` entry is unaffected: stopping on a token means stopping before it |
| `return_tokens_as_token_ids` | Served on all three generation wires: each reported token is spelled `token_id:123` instead of its text, in the `tokens` array, the `top_logprobs` keys and chat's `content[].token`. The completion's `text` is unchanged. It exists because two ids can detokenize to one string, and a map keyed by text loses one of them |
| `use_beam_search`, `prompt_embeds` | **Reject, 501 naming the field**, on `/v1/chat/completions`, `/v1/completions` and llama.cpp's native `/completion`. Each changes the tokens or the text returned, so ignoring one answers a question the caller did not ask. The default a caller may legitimately spell out (`n: 1`, `echo: false`, `skip_special_tokens: true`) is served. One table, flattened into all three request bodies, so a field cannot be refused on one wire and dropped on another again |
| `response_format: json_object` | Supported (a character-class mask plus validation; see [Grammar-constrained decoding](#grammar-constrained-decoding)) |
| `grammar` | Supported. llama.cpp's own field: a GBNF string, enforced per token by a parser |
| `response_format: json_schema` | Supported. `json_schema.schema` is compiled to GBNF and enforced per token. `strict: false` and any unknown member of the `json_schema` object are refused by name; a schema the converter cannot compile is a 400 naming the keyword |
| Other `response_format` types | Reject |
| `session_id` | Frink extension (server-side history) |
| `stream_resumable` | Frink extension: keep a replay buffer for this stream so a client can reconnect (see [Resumable streams](#resumable-streams)) |
| `lora` | llama.cpp's per-request adapter scales (see [LoRA adapters](#lora-adapters)) |
| `chat_template_kwargs` | Supported (see [Chat templates](#chat-templates)) |
| `reasoning_effort` | Supported, quantized onto what the checkpoint grades; `none`/`off` turn thinking off |
| `thinking: {"type": …}` | Supported (DeepSeek wire): `enabled`/`disabled`, anything else is a 400. On `/v1/messages`, `thinking.budget_tokens` becomes `reasoning_budget_tokens` as llama.cpp's Anthropic lowering does (`server-chat.cpp:585-591`): `enabled` without a number gets 10,000 |
| `ignore_eos` | Frink extension: run past the model's own end-of-generation tokens so the request produces exactly `max_tokens`. A serving-benchmark knob. It suppresses the model's set only; a caller's own `stop` strings still end the answer |
| `reasoning_content` (both ways) | Frink extension: a reasoning model's chain of thought, split out of `content` on the way out and replayable on the way in (`reasoning` is accepted as an alias) |
| `continue_final_message` | Supported, with llama.cpp's field and value set: `true` (auto), `"reasoning_content"`, `"content"`, or `false`. The trailing assistant message is rendered as a turn still being written, thought included, so the model carries on from where it stopped. On by default, as in llama.cpp's server (`server-common.cpp:1046-1056`): a request that says nothing and ends in an assistant message is continued. `false` renders it as a closed turn for that request (llama.cpp reads `false` as absence, so this is the one intended difference), and `--no-prefill-assistant` turns the default off server-wide. One rule for `/v1/chat/completions`, `/v1/messages` and `/v1/responses`, applied in one renderer. The harmony and ATEM channel formats, a content continuation for an always-open family, and a turn with tool calls are 501 by name; a 501 reached through the default says how to switch it off |
| `reasoning_budget_tokens` / `thinking_budget_tokens` | Supported, llama.cpp's sampler-level budget (`common/reasoning-budget.cpp`). `-1` or absent takes the server's `--reasoning-budget` (itself `-1`, unrestricted); `N` allows N tokens of thought after the opener and then forces the closer one token per step, so the answer still arrives with `finish_reason: "stop"`; `0` forces the closer as soon as the block opens. The closer is never counted, a forced closer waits for a split UTF-8 character to complete, and a second block in the same response gets a fresh budget. Below `-1` is a 400 naming the field. Also accepted at the top level of a `/v1/responses` body, as llama.cpp's lowering passes it. A checkpoint with no reasoning format has no thought to bound (llama.cpp builds no sampler either); the harmony and ATEM channel formats are 501 by name |

### Where a completion stops

Besides `max_tokens` and the request's own `stop` strings, generation
ends on any token in the checkpoint's end-of-generation set: the
`eos`/`eot`/`eom` metadata ids plus every vocabulary entry whose text is
on llama.cpp's literal EOG list (`<|eot_id|>`, `<end_of_turn>`,
`<|im_end|>`, `<turn|>`, …). Stopping on `tokenizer.ggml.eos_token_id`
alone would let a Llama-3 or Gemma checkpoint run past the end of its
own turn, because neither one's turn ender is the metadata EOS. Both
decode paths (the private loop and the continuous batcher) use the same
set. A Kimi K3 checkpoint keeps its vocabulary in
`tokenizer_config.json` rather than GGUF metadata, so the set is derived
from its special-token names and `[EOT]` ends a turn there too.

BOS follows the rule in [CLI.md](CLI.md#who-adds-bos): the chat template
owns it when it prints one, the loader otherwise, added idempotently so
a template that already emitted BOS never gets a second one.

### When a request does not fit the context

There are two outcomes. A prompt at or past the deployment's
per-request position ceiling is refused with a 400 whose message reads
`prompt is too long: N tokens > M maximum`, the wording Claude Code and
OpenClaw match on, because the Anthropic wire carries no error code for
it. A prompt that fits is served, with `max_tokens` clamped down to the
room that remains; refusing that case would turn a servable long-prompt
request into an error over a `max_tokens` the caller most likely never
set. The clamp is also what makes the 32768 default output budget safe.

## Chat templates

The prompt is rendered by evaluating the checkpoint's own
`tokenizer.chat_template`, the Jinja2 source shipped in the GGUF. A
checkpoint that ships no template falls back to ChatML (matching
llama.cpp `--jinja`), or to role-labeled lines for a byte or synthetic
tokenizer. A template that does not compile, or that uses a construct
this evaluator does not provide, fails the chat request with the
compiler's own message. There is no fallback to a guessed framing.

`chat_template_kwargs` is passed through: whatever a client puts there
becomes a top-level template variable, which is how `enable_thinking`
(Qwen3, Gemma-4), `thinking` (DeepSeek) and `preserve_thinking` are
driven. It can never shadow `messages`, `tools`, or
`add_generation_prompt`.

Five rules are applied to it before rendering:

* An explicit knob wins wholesale. A caller who already set any of
  `enable_thinking`, `thinking`, `thinking_mode` or `reasoning_effort`
  inside `chat_template_kwargs` has said what they want; the
  protocol-level knobs then stand down entirely rather than merging, so
  a default can never contradict an explicit request.
* `none` and `off` turn thinking off. `reasoning_effort: "none"` is
  broadcast as "thinking off" and is never quantized onto the nearest
  gear, which would turn "do not think" into "think a little". The
  DeepSeek-wire `thinking: {"type": "disabled"}` does the same and beats
  any effort in the same request.
* Offering tools turns thinking on. Some encoders emit well-formed tool
  calls only in thinking mode, so `tools` implies `enable_thinking`
  unless the request said otherwise.
* Effort is quantized onto what this checkpoint grades. The template's
  effort vocabulary is probed once at load; a request asking for a gear
  outside it is mapped to the nearest one, or dropped (so the
  template's own default applies) when nothing is close enough.
  `reasoning_effort: "minimal"` against a checkpoint that grades only
  `low`/`medium`/`high` renders as `low`. Top-level `reasoning_effort`
  loses to an explicit `chat_template_kwargs` entry.
* One value, every spelling. The graded-strength dialect reads
  `reasoning_strength`; the value is broadcast to both, and a Jinja
  template ignores variables it does not declare.

`GET /v1/models` advertises what came out of that probe:
`supported_reasoning_efforts` (least thinking first: `off`, `adaptive`,
then the gears, or a bare `on` when there is a toggle but no ladder) and
`default_reasoning_effort` (the gear the checkpoint is in when asked for
nothing). A checkpoint that says nothing about thinking carries neither
field. An empty list would read as "asked, and it has no gears", which
is a different claim from "this is not a reasoning model".

Whether the tools are described by the template or by a text preamble
depends on the template, established at load by rendering it with and
without a tool. One that consumes `tools` is handed them as structured
JSON and owns the whole grammar; one that does not gets the preamble
described under [Tool-call formats](#tool-call-formats).

## Health

`GET /health` returns JSON and is never behind auth or rate limiting.

```jsonc
{
  "state": "ready",              // ready | detecting | unavailable
  "model": { "id": "…", "tokenizer": "…", "synthetic_weights": false },
  "capabilities": [
    { "id": "metal", "available": false,
      "reason": "metal_not_built",       // stable code, safe to switch on
      "detail": "Apple M2 Pro is present but this binary was built without --features metal; rebuild to use it." }
  ],
  "version": "0.50.0", "pid": 4242, "uptime_seconds": 12.5,
  "server_time_unix_ms": 1786000000000,
  "last_request_age_seconds": 0.4    // absent until one is served
}
```

`detecting` means backend probing has not finished, so the capabilities
in that response are provisional and a client should wait before
drawing a conclusion. Detection gets a 1 s budget. Past that the answer
is filled in provisionally with `reason: "detection_timed_out"`. The
handler itself never blocks.

Status is 200 for `ready` and `detecting`, 503 for `unavailable`. A
server reaches `unavailable` when no model is loaded, after
`POST /admin/models/unload` or `POST /sleep`; it then answers with
`reason: "model_not_loaded"` and no `model`. A non-ready state carries
a top-level `reason` and `detail` (`detecting` or `model_not_loaded`);
`ready` carries neither. A supervisor that read
200 there would route traffic straight into a 503.

Every capability carries a machine `reason` and a human `detail`, so a
UI can grey the control and show the sentence without working out why
itself. The capability ids are `cpu`, `metal`, `cuda`, `real_weights`
and `continuous_batching`. `metal_unavailable` (no device),
`metal_not_built` (this binary) and `disabled` (a flag turned it off)
are three different problems with three different fixes; `cuda_*` and
`cpu_only` follow the same pattern.

`last_request_age_seconds` exists so a busy server is not read as a dead
one. A saturated GPU starves the health handler, and recent request
activity is positive evidence that the process is alive.

## Request ids

Every `/v1/chat/completions` response carries a server-assigned
`chatcmpl-…` id. It is the `id` field, and it is repeated under
`request_id`. For a non-streamed call it sits in the JSON body; for a
streamed one it arrives in the first SSE chunk, before any content. That
is the key frink logs and cancels by, so a client never has to guess
which in-flight request is its own. `/v1/messages` states the same id in
a `request-id` response header (see [Cancellation](#cancellation)).

## Usage timings

`usage` extends the OpenAI shape with llama.cpp-style timings. Prefill
and decode are reported separately: dividing total tokens by total wall
time reads a 50 tok/s model as 5 tok/s on a long prompt.

| Field | Meaning |
|---|---|
| `prompt_tokens`, `completion_tokens`, `total_tokens` | OpenAI convention |
| `completion_tokens_details.reasoning_tokens` | completion tokens spent inside a reasoning block; present only when the reasoning split ran |
| `prompt_eval_duration_ms` | wall time spent on the prompt |
| `generation_duration_ms` | wall time spent in the decode loop |
| `time_to_first_token_ms` | prefill start to first token produced |
| `prompt_per_second`, `predicted_per_second` | per-phase throughput |
| `cached_tokens` | prompt tokens reused from the KV prefix cache |
| `acceptance_length` | completion tokens per forward pass, speculative or not. 1.0 means speculation saved nothing, 2.0 means half the forwards. A round that drafted nothing still costs a forward and is counted |
| `draft_tokens`, `accepted_draft_tokens` | draft tokens evaluated and kept |
| `draft_accept_rate_per_position` | accept rate at each position in the draft block |

Every timing is optional and omitted, rather than nulled or zeroed, when
it was not measured (a cached response, a batched decode). For
`cached_tokens`, absent means no prefix cache is configured and `0`
means the cache was consulted and missed.

The server drafts with prompt lookup: an n-gram match over the
request's own history, with no second checkpoint, so it costs no memory
and no load time and helps where output quotes input. Verification
draws with the server's own sampler and keeps a drafted token only when
it equals that draw, so speculation never changes the answer. The four
speculation fields are present whenever a request drafted.
`draft_accept_rate_per_position` is reported beside the mean because a
drafter that is right at the first position and useless at the last has
the same mean as a uniformly mediocre one, and the two call for
opposite block sizes. `/admin/stats` rows carry `acceptance_length` and
`draft_accept_rate_per_position` for the same reason.

The speculation fields are absent for a request that did not draft: a
grammar-constrained or `json_object` request (verification draws
through the same grammar machine, and a rejected block would leave it
advanced over tokens that were never emitted), a paged or recurrent KV
store (a rejected draft has to be rolled back and neither can), a
streamed `n > 1` request (its choices are interleaved a token at a
time), and a request with no room to draft. An absent field reads as
"this request did not speculate"; a zero would read as "the drafter was
useless".

Streamed requests get the same `usage` object on the final chunk.

## Admin control surface

Everything under `/admin` either changes what the server serves or
writes to disk, so it needs the same `FRINK_API_KEY` as `/v1/*`. None of
it sits on the unauthenticated `/health` side. Paths and payload shapes
are defined once in the `frink-api` crate.

| Endpoint | Answer |
|---|---|
| `GET /admin/models` | inventory (below) |
| `POST /admin/models/load` `{"id":"…"}` | `202 {"task_id":"…"}` |
| `POST /admin/models/unload` | `200 {"ok":true,"active":null}` |
| `POST /admin/download` `{"repo":"…","file":"*.gguf"}` | `202 {"task_id":"…"}` |
| `GET /admin/tasks` | every task, newest last |
| `POST /admin/tasks/{task_id}/cancel` | `200 {"ok":true}` |
| `GET /admin/stats` | counters and the recent-request ring |

`POST /sleep`, `POST /wake_up` and `GET /is_sleeping` sit at the root
rather than under `/admin`, behind the same key; the endpoint table
describes them.

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

This comes from GGUF headers only: metadata and tensor descriptors,
never a weight, so listing a directory of checkpoints costs header
parses rather than loads. A field that is expensive to establish comes
back `null` rather than guessed, and the key is always present, because
an unknown context length and a zero one are different facts. `quant`
comes from `general.file_type` when that maps to a known name, which is
the only place the `_M` in `Q4_K_M` is stated; failing that it comes
from the dominant tensor dtype. `resident_bytes` is always `null`,
because frink keeps checkpoints mmap-resident and the true figure is a
page-cache property this process cannot read.

Discovery scans `FRINK_MODEL_DIR` and the directory holding
`FRINK_MODEL_PATH`, non-recursively, for `*.gguf` plus any
safetensors-index checkpoint directory. Split checkpoints fold into one
entry named for the shard prefix.

### Model swap

`POST /admin/models/load` answers immediately with a task id, and the
load runs on a blocking thread. It accepts only ids from
`GET /admin/models`; an unknown id is a 404. There is no load-by-path
endpoint, so no request can make the server open an arbitrary file. A
second load while one is running comes back `409`.

In-flight requests keep the model they started on. A request clones its
handle once, up front, and decodes against those weights even if a
different checkpoint is published mid-generation. The old model is
freed when the last such request finishes. Nothing migrates a running
request.

After `POST /admin/models/unload`, generation endpoints answer `503`
with `"type": "model_not_loaded"`, `GET /v1/models` returns an empty
list, and `/health` reports `unavailable` (503). The adapters named by
`--lora` are attached to every checkpoint a load brings in, or the load
is refused by name.

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

Timestamps are the server's Unix epoch milliseconds; the browser clock
is not used for ordering. `done`, `error` and `cancelled` are terminal
and never change, so a client can stop polling once it reads one.

`rate_bytes_per_s` and `eta_seconds` are `null` while `state` is
`warming`, which lasts until the rate window holds at least 3 samples
spanning at least 3 s; the first tick of a transfer would otherwise
divide out to gigabytes per second. `fraction` is available
immediately, because it is a ratio of two counters.

Cancellation is cooperative. `POST …/cancel` raises a flag and returns,
and the task reaches `cancelled` only once a worker acknowledges it. A
download stops within one chunk and keeps its `.part` file for a resume.
A model load cannot be interrupted mid-mmap, so a cancel that arrives
during one discards the finished result. Cancelling a finished task is a
`409`; an unknown task id is a `404`.

### Download

Fetches one `.gguf` from the Hugging Face Hub into the model directory.
`file` is a literal name or a `*` glob, which is resolved against the
repo's file list (sorted, first match, root-level names only). Downloads
resume: bytes land in `<target>.part` and a restart sends `Range`,
falling back to a clean restart when the server answers `200` instead
of `206`.

The checks are a whitelist. Each segment of `repo` may contain only
letters, digits, `-`, `_` and `.`, and `file` must be a bare `*.gguf`
name: no separators of either kind, no leading dot, no `..`, no `:`.
The joined path is then re-checked to be a direct child of the model
directory fixed at startup, never one named by the request. With no
model directory configured the request is a `412`. Two downloads of the
same target come back `409`, because interleaved writes into one
`.part` file would produce a corrupt checkpoint of exactly the right
size.

Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for gated repos, and
`HF_ENDPOINT` for a mirror.

### Stats

```jsonc
{
  "uptime_seconds": 43, "requests_total": 1, "errors_total": 0,
  "cache_hits": 0, "cache_misses": 1,
  "tokens_prompt_total": 17, "tokens_generated_total": 24,
  "last_request_age_seconds": 0.02,
  "generating_now": 0,               // decoding right now, not a queue depth
  "queue_depth": null,               // null unless continuous batching is on
  "queue_rejected_total": null,      // turned away by the queue cap, since start
  "recent": [{
    "request_id": "chatcmpl-b28a1aeab8f8000000",
    "at_ms": 1786718007013, "route": "/v1/chat/completions",
    "model": "Qwen3-0.6B-Q4_K_M",   // the model that served it
    "status": 200, "prompt_tokens": 17, "completion_tokens": 24,
    "ttft_ms": 6586.2, "duration_ms": 23603, "decode_ms": 17004.8,
    "stream": false,
    "via_api_key": "key-4f21a0c3",   // fingerprint, never the key; null if none
    "client": "frink-studio"        // self-declared (X-Frink-Client); null if absent
  }]
}
```

`recent` is a 200-entry ring keyed by the same `request_id` the response
carried, so a log row joins its message on equality. `duration_ms` and
`decode_ms` are separate: the first carries queue wait plus prefill plus
decode, so dividing completion tokens by it reads a 50 tok/s model as 5.
`decode_ms` and `ttft_ms` are `null` when the engine did not time itself
or the answer came from cache.

Every generation route is recorded (`/v1/chat/completions`,
`/v1/completions`, `/completion`, `/v1/messages`, `/v1/responses`,
streamed and not, including rejections), and so are `/v1/embeddings`,
`/v1/rerank`, `/v1/score`, tokenize and detokenize, each with its
status, success or failure. Tokenize and detokenize are recorded under
the path the client called, and carry no token counts: they run the
tokenizer and not the model, and `prompt_tokens` here feeds
`tokens_prompt_total`, which counts tokens this server put through a
forward pass. `/v1/embeddings` does run one, so its prompt tokens are
counted and its `decode_ms` is `null`, since there is no decode loop to
time.

`queue_depth` and `queue_rejected_total` come from the
continuous-batching scheduler's queue and are `null` when batching is
off, because then nothing queues: every request gets its own blocking
thread. A gauge reading `0` would claim an empty queue was measured.

`model` names the model that answered, as `/v1/models` reports it, and
never the `model` field the request carried; this server decodes
against whatever is loaded and ignores that string. `null` means
nothing was loaded, which is what a 503 row is.

`via_api_key` is a short fingerprint of the `Authorization: Bearer` key
that served the request, never the key and never anything the key can
be recovered from. It is salted per process, so it is stable within one
server run, meaningless across a restart, and useless for testing key
guesses offline. `null` means no `Authorization: Bearer` header was
presented, which includes a request authenticated with `x-api-key` and
every request on a server started without `FRINK_API_KEY`. `client` is
the caller's own `X-Frink-Client` header, kept to 32 label characters.
It is a claim and nothing authenticates it: Frink Studio sends
`frink-studio`, and so could anything else.

## Cache pools and a clean stop

`GET /v1/cache/status` reports the KV pool geometry a client sizes
against:

```jsonc
{
  "state": "ready",
  "kv": { "num_pages": 512, "used_pages": 40, "evictable_pages": 0,
          "page_size": 256, "num_tokens": 131072 },
  "moe": null, "mamba": null, "swa": null,
  "resizable": ["kv"]
}
```

`kv` is `null` when the deployment has no shared KV pool
(`FRINK_KV_POOL_BLOCKS` unset), because every request then allocates
privately, which is a different fact from a pool of size zero. `moe`,
`mamba` and `swa` are always `null`: frink has no such pools, and they
are named so a client can see they were considered.

`POST /v1/cache/rebuild` with `{"kv": <tokens>}` re-budgets the shared
KV pool on a live engine. Every check runs before anything is freed, so
a refused rebuild leaves the engine serving what it was serving. A
request naming `moe`, `mamba` or `swa`, a missing `kv`, a target below
one page, or a deployment with no pool is a 400. A target below the
pages in-flight requests hold is a 409 that names the floor in tokens.
A rebuild that is not allowed to start (the engine is loading, or a
previous rebuild failed) is a 503, and one already running is a 409. A
successful rebuild clears the prefix cache and says so in
`prefix_cache_invalidated`. While a rebuild runs, generation requests
answer 503 with `type: "cache_rebuilding"`.

`POST /v1/admin/prepare-stop` closes admission and, once no generation
is in flight, seals the token totals:

```json
{"model_id": "…", "prompt_tokens_total": 1234, "completion_tokens_total": 5678,
 "uptime_s": 3600, "drain_complete": true}
```

A supervisor calls it before it signals the process, so shutdown cannot
race the last sampled token. While generations are still running the
answer is a 503 with `drain_complete: false` and the `active` count, and
admission stays closed; the caller retries. Sealing is idempotent for
the life of the process, so a retried call returns the same numbers.
With `FRINK_ACCOUNTING_OUTBOX` set, the totals are also written as a
receipt `<dir>/<id>.json` before the response goes out, and the
response carries `receipt_id` and `receipt_status`; a failed write is a
500 with `retryable: true`. `drain_timeout_s` and `abort_timeout_s` are
accepted and not used. Generation requests after the call answer 503
with `type: "server_stopping"`.

## Embeddings

Two different things can answer `POST /v1/embeddings`:

| Source | How it is loaded | Default pooling | Normalized |
|---|---|---|---|
| An encoder (BERT family) | `FRINK_MODEL_PATH` at an encoder GGUF, `/admin/models/load`, or `FRINK_EMBEDDING_MODEL_PATH` beside a generative model | the checkpoint's own `pooling_type` (`CLS` for every BGE) | yes |
| A decoder's hidden states | whatever generative model is loaded | `mean` | no |

The encoder is the model trained to produce a sentence representation;
the decoder path is a best-effort reading of a model that was not.
`embedding_type` overrides the default: `mean` and `last` on both, `cls`
on the encoder path only (row 0 of a decoder's hidden states is its BOS
position). `none` and `rank` are refused on both, because this response
carries one vector per input, and `rank` is a classification head that
[Reranking](#reranking) serves.

### An encoder as the served model

`FRINK_MODEL_PATH=bge-small-en-v1.5-q8_0.gguf` works. The file's
`general.architecture` decides: an encoder architecture never reaches a
decoder loader.

`GET /v1/models` then says what it is, so a client does not have to
send a request to find out:

```json
{"id": "bge-small-en-v1.5", "object": "model",
 "frink_model_kind": "embedding",
 "frink_endpoints": ["/v1/embeddings"],
 "frink_n_embd": 384, "frink_pooling": "CLS",
 "frink_context_length": 512,
 "frink_tokenizer": "gguf-wordpiece"}
```

`frink_endpoints` lists `/v1/rerank` instead for a checkpoint whose
`pooling_type` is RANK. `GET /health` is `ready`, since the server can
serve. Every generation endpoint (`/v1/chat/completions`,
`/v1/completions`, `/completion`, `/v1/messages`, `/v1/responses`)
answers `501` naming the model:

```json
{"error": {
  "message": "the loaded model 'bge-small-en-v1.5' is an embedding model (bert encoder, 384 dims, pooling CLS). An encoder has no output head, so it cannot generate text at all. There is no next token for it to predict. POST /v1/embeddings to use it, or load a generative checkpoint.",
  "type": "unsupported",
  "param": "model"
}}
```

`/v1/tokenize` and `/v1/detokenize` answer for an encoder, because an
encoder has a tokenizer and asking what tokens it saw is not asking it
to generate. `add_special` is a no-op there, since an encoder wraps its
own `[CLS]`/`[SEP]`, and those are the same tokens
`usage.prompt_tokens` counts on an embeddings response. Every route
that needs a decode refuses.

The encoder architectures that load are listed in
`frink_models::bert_gguf_loader::ENCODER_ARCHS`: `bert`, `nomic-bert`,
`jina-bert-v2`, `jina-bert-v3`, `neo-bert` and `eurobert`. The encoder
path has a WordPiece tokenizer only (`tokenizer.ggml.model = "bert"`),
and a file declaring another tokenizer is refused by name. The other
embedding rows upstream builds (`nomic-bert-moe`, `modern-bert`,
`t5encoder`, `gemma-embedding`, `gemma-embedding2` and two decision-head
scorers) refuse by name, each saying what it needs;
`frink_models::embedding_model::NOT_YET` is that list. `llama-embed`
is a Llama decoder with the output head skipped, so it loads on the
decoder path and `/v1/embeddings` pools its hidden states.

## Reranking

`POST /v1/rerank`, and llama.cpp's `/rerank` on the same handler.

Cross-encoder reranking with the loaded checkpoint's own classification
head (`cls` / `cls.output`). Embedding the query and each document and
sorting by cosine similarity would be a different model answering a
different question, so the route never substitutes it.

Request `{"model"?, "query", "documents": [...], "top_n"?,
"return_documents"?}`. Response `{"object": "list", "model", "results":
[{"index", "relevance_score", "document"?}], "usage"}`, best first.

- `index` is the position in the request's own `documents` array, not
  the position in the sorted result.
- Ties keep request order; the sort is stable.
- `top_n` clamps to `documents.len()`, and `top_n: 0` returns no
  results. An empty `documents` array or an empty `query` is a 400.
- It needs an encoder GGUF carrying a rank head, through
  `FRINK_MODEL_PATH` or `FRINK_EMBEDDING_MODEL_PATH`. Anything else
  answers 501 naming the model.

`/v1/embeddings` against a rank-head checkpoint refuses: its
`pooling_type` is RANK, which is a classification head and not a
pooling rule, and that refusal is what sends a caller here.

The pair is encoded as `[CLS] query [SEP] document [SEP]`, with the
query in segment 0 and the document in segment 1. llama.cpp sets every
token type to zero here, and frink differs on purpose: segment ids
reorder the results as well as shifting the scores. On "How many people
live in Berlin?" against five documents, the one carrying the
population figure ranks first with segments 0/1 and fifth with segments
all zero. Matching HuggingFace is what makes the ranking right, so a
reranker checkpoint whose token-type table has fewer than two rows is
refused at load, and so is an encoder with no two-segment input form.

Verified end to end against `ms-marco-MiniLM-L6-v2`, with scores
matching an independent NumPy transcription of
`BertForSequenceClassification` to 4e-3 and the order matching the
HuggingFace reference on four query sets.

### Which head ran, and why it changes the scale

A rerank response reports `frink_score_head`, and a client that
thresholds on an absolute score has to read it. There are two regimes,
and they differ by more than an order of magnitude:

| `frink_score_head` | Meaning | Range on `ms-marco-MiniLM-L6-v2` |
|---|---|---|
| `classifier(tanh(pooler(cls)))` | The full head the checkpoint was trained with | about plus or minus 11 |
| `classifier(cls)` | The classifier alone, no pooler in the file | about plus or minus 0.2 |

The order is the same either way; the scale is not. A client carrying
`relevance_score > 0.5` from another engine gets a threshold that never
fires in the second regime, which is what this field is for. A
load-time note says the same thing to an operator.

Every published GGUF is the second regime. llama.cpp's converter drops
the pooler by name (`conversion/bert.py`, `BertModel.filter_tensors`),
and under llama.cpp's tensor naming the pooler slot is `cls`, with
`cls.output` being the classifier that follows it (`src/llama-graph.cpp`:
`cls`, then bias, then `tanh`, then an optional head norm). frink reads
`cls` when a file carries it and does not invent one when it does not.

An absent pooler is a note rather than a refusal because nothing in the
file distinguishes "trained without a pooler" from "converted without
one", and `jina-reranker-v1-tiny-en` is a real checkpoint whose head is
a direct projection.

### Getting the first regime: `frink splice-pooler`

The pooler is in the checkpoint's own safetensors, and `frink
splice-pooler` writes a GGUF that carries it (`docs/CLI.md`). The tie
between the two files is the classifier they both hold, compared
element-wise to within the GGUF's storage precision; the GGUF's
`general.name` is not trusted, because the published
`ms-marco-MiniLM-L6-v2-Q8_0.gguf` names the L12 model. A pooler from any
other checkpoint is refused naming the element that disagrees.

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

Tracked as [#82](https://github.com/antonellof/frink/issues/82).

## Errors that retrying will not fix

Two ceilings do not move with load. A request that hits one gets a `400`
and never a `503`, because an idle server would answer the same way and
a `Retry-After` would only start a retry loop.

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

`code` names the ceiling that is binding. The two have different fixes,
and only one of them is in the client's hands:

| `code` | Meaning | Fix |
|---|---|---|
| `context_length_exceeded` | Longer than any single request may be (`FRINK_CB_MAX_CONTEXT`) | Shorter prompt, lower `max_tokens` |
| `device_memory_budget_exceeded` | Bigger than the server's whole KV budget (`FRINK_CB_KV_BLOCKS`) | More KV blocks, or a smaller model |

`estimated_bytes` and `limit_bytes` are the KV cost of the request and
of the ceiling, priced from the model's own layout with the same
arithmetic `frink inspect-plan` prints. When a request exceeds both
ceilings, the per-request one is reported, because that is the one the
caller acts on.

Momentary pressure gets a different answer: `503` with `Retry-After`,
counted separately, so an operator reading `/metrics` can tell "one
request too big" from "too many requests". Every `503` this server
returns carries `Retry-After: 1` unless the handler set its own. It is a
fixed hint: an estimate would need the caller's throughput, which the
server does not know.

## Stop sequences

A stop sequence is matched at two levels, because it promises something
about what reaches the client, and streaming makes that promise hard to
keep.

1. Token level. A stop string that is exactly one token in this model's
   vocabulary, the usual case for `<|im_end|>`, `<end_of_turn>` and
   `<|eot_id|>`, is matched on the token id before the token is
   detokenized. That catches control tokens whose rendered form is not
   the string the client asked for, and the token never becomes part of
   the answer or of `usage`.
2. Text level. Everything else is matched on the emitted text, with the
   output buffered so that no trailing text that could still become a
   stop string is sent. Only the actual partial match is withheld, so
   ordinary text goes out as soon as it is provably safe.

A partial match therefore never appears on the wire and is never taken
back; SSE has no way to take anything back. Text withheld against a
match that never arrives is released when the answer ends.

A generation that runs into one of the caller's `stop` strings reports
`finish_reason: "stop"` on the OpenAI routes, which have no other value
for it, and the stop string is trimmed from the answer.

`/v1/messages` reports more, because Anthropic's protocol has room for
it: a caller's stop string that fires reports
`stop_reason: "stop_sequence"` with `stop_sequence: "<the string>"`
beside it, on the buffered body and in the terminal `message_delta`
alike. Two cases are reported differently:

- A stop the server added, the served template's own end-of-turn
  marker, reports the ordinary `end_turn`.
- A truncated generation reports `max_tokens` even if a stop matched,
  and a generation that produced tool calls reports `tool_use`.

## Cancellation

There are two ways to stop a generation, and both end at the same
server-side flag.

1. Drop the connection. A streaming path notices that its SSE receiver
   is gone and stops at the next token. A stream opened with
   `stream_resumable` is the exception (see
   [Resumable streams](#resumable-streams)).
2. `POST /v1/cancel` with the request's id. It sits behind
   `FRINK_API_KEY` like the endpoint that started the work. A browser
   should send it with `keepalive: true` so it survives the page unload
   that killed the stream.

```bash
curl -s http://127.0.0.1:8383/v1/cancel \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"chatcmpl-…"}'
```

The answer is `200 {"request_id":"…","cancelled":true,"detail":"…"}`
when a live generation was signalled, and `404` with
`"cancelled": false` when the id names nothing that is running.

A cancelled stream ends with `finish_reason: "cancelled"`. OpenAI does
not define that value, since OpenAI has no cancel endpoint, but it is a
finish reason, so no client reads the stream as truncation. Tokens
already decoded are kept and reported in `usage`.

Continuous-batching requests are covered too. The cancel enqueues an id,
and the scheduler thread drains that queue between steps and drops the
sequence itself, because removing a sequence touches KV buffers a
forward pass might be reading. A request stops wherever it has got to,
whether queued, mid-prefill or decoding, and keeps what it produced.

The flag is read between decoded tokens, so a prefill already inside a
forward pass runs to completion. Which routes can be cancelled by id:

| Route | Id the cancel takes |
|---|---|
| `/v1/chat/completions`, streamed | the `chatcmpl-…` id in the first chunk |
| `/v1/messages`, streamed | the `request-id` response header, as the Anthropic API spells it; the `message_start` id is a per-message `msg_…` the cancel registry does not know |
| `/v1/responses`, `/completion`, streamed | registered, but neither wire states the id, so dropping the connection is the way to stop them |
| `/v1/completions` | none; it is buffered and registers no id |

## Keepalives

Every streamed endpoint sends a keepalive after 15 seconds of silence,
and every keepalive is a data frame, never an SSE comment. axum's own
`Sse::keep_alive` writes `: ping`, which a proxy sees and a client's
event handler does not. A client's stream-idle timeout is armed on
received events, so a stream kept alive by comments gets torn down and
reconnected in the middle of a long prefill.

| Endpoint | Keepalive frame |
|---|---|
| `/v1/chat/completions` | a `chat.completion.chunk` with an empty delta |
| `/v1/stream/{id}` (replay) | an id-less chunk with `choices: []` |
| `/v1/responses` | `response.in_progress` |
| `/v1/messages` | `ping` |
| `/completion` | a partial frame with empty `content` and `"stop": false` |

The silence before the first token counts: that window is the queue
wait plus the prefill, and on a long prompt it is the longest quiet
stretch of the request.

A replayed stream never re-delivers keepalives, and the replay keepalive
carries no `id:`, so it does not enter the sequence a `Last-Event-ID`
resumes from.

## Streaming behind a proxy

Streamed responses carry `X-Accel-Buffering: no` beside axum's
`Cache-Control: no-cache`. nginx, and the proxies that copied its
convention, buffer `text/event-stream` by default, which turns a
token-by-token stream into one silent wait followed by the whole answer.

Because a keepalive data frame goes out every 15 s, an idle but healthy
stream still puts bytes on the wire. Measure a stall timeout against
bytes received rather than tokens; a long prefill then never trips it,
and a swallowed connection does.

### Resumable streams

`id:` promises that a reconnect can pick up where the connection
stopped, so it is emitted only where a replay buffer exists. A
`/v1/chat/completions` request asks for one:

```jsonc
{"model": "...", "messages": [...], "stream": true, "stream_resumable": true}
```

Then every event carries `id: {request_id}:{n}` and the first one also
carries `retry: 1500`. There are two ways back in, both behind
`FRINK_API_KEY` like the request that filled the buffer:

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
`true` only once the generation has ended and the buffer is drained, so
a client that stops on it never discards events it was not given. A
poll waits up to 10 s for the next event, which keeps it reading like a
stream while staying a short JSON response that a buffering proxy
cannot hold back.

`stream_resumable` also changes what a dropped socket means. Without
it, the connection closing cancels the generation. With it, the
generation keeps running into the replay buffer, and `POST /v1/cancel`
is its stop path. The caller decides because neither answer suits both
cases: a tab that navigated away wants the CPU back, and a tab whose
proxy dropped a 90-second answer wants the answer.

Both bounds fail closed. The replay window is 1 MiB per stream; past it
the oldest events are dropped and asking for an evicted position
answers `410 replay_window_lost`. A finished stream stays reconnectable
for 120 s and at most 64 streams are remembered, oldest finished
evicted first; a live stream is never evicted out from under its
reader. A `Last-Event-ID` naming a different request is
`400 bad_last_event_id`. An unknown or forgotten id is
`404 stream_not_found`.

## Continuous batching

Concurrent requests share one batched decode worker (llama.cpp's slot
model): many in-flight sequences, one `forward_multi_seq` step per tick.
This is the supported multi-client path on Metal.

Enable it with `FRINK_CONTINUOUS_BATCHING=1`, `--cont-batching` / `-cb`,
or `-np N` / `--parallel N` (which also sets `FRINK_CB_MAX_SEQS`).
Disable it with `FRINK_CONTINUOUS_BATCHING=0` or `--no-cont-batching`.
When unset on Metal builds with fused attention, continuous batching is
on by default unless a KV pool or prefix cache forces the private
decode path.

On the contiguous KV path it is mutually exclusive with
`FRINK_KV_POOL_BLOCKS` and `FRINK_PREFIX_CACHE_ENTRIES`; the paged KV
store lifts this (see [`CONFIG.md`](CONFIG.md)).

Streaming under continuous batching emits tokens as they are sampled,
in the same SSE shape as the private decode path.

Prefill is chunked. Each tick the scheduler runs one bounded prefill
chunk, taken round-robin across waiting prompts, plus one batched decode
step over every in-flight row. The chunk is at most
`FRINK_CB_PREFILL_CHUNK` (default 128) and is further limited to what is
left of `FRINK_CB_MAX_BATCH_TOKENS` (default 256) after the rows
decoding in the same tick, with a floor of 16 prompt tokens per tick so
a saturated batch cannot deadlock. A long prompt joining the batch
therefore costs an in-flight decode one chunk instead of the whole
prompt.

`FRINK_CB_MAX_SEQS` caps in-flight sequences, counting prompts still
prefilling. `FRINK_CB_MAX_QUEUE` (default 512) caps how many requests
wait for admission. Past that cap a request gets a `503` with a
`Retry-After` header, and the JSON body names the queue depth, the cap
and `retry_after_seconds`. Admission is first in, first out, except that
a job whose prompt the radix prefix cache already holds may be admitted
ahead of the next one: only the first eight waiting jobs are
considered, and no job is passed over more than four times.

`-b N` / `--batch-size N` and `-ub N` / `--ubatch-size N` set the
prefill chunk on both decode paths at once (`FRINK_CB_PREFILL_CHUNK`
here and `FRINK_CHUNKED_PREFILL` on the private loop). frink has one
prefill stage, so the two flags resolve as llama.cpp resolves them
(`src/llama-context.cpp:265`): the smaller of whichever was named.

While a batcher is active, `GET /metrics` reports
`frink_prefill_chunks_total`, `frink_prefill_tokens_total`,
`frink_decode_steps_total`, `frink_scheduler_queue_depth`,
`frink_scheduler_queue_rejected_total`, and the configured caps
`frink_scheduler_max_seqs` (`-np`; 0 when unlimited) and
`frink_scheduler_prefill_chunk` (`-ub`), so a flag can be read back from
the process that received it.

## Slot save and restore

llama.cpp's `POST /slots/{id_slot}?action=save|restore`, so a long
system prompt survives a restart instead of being prefilled again. The
route, the `action` query, the gate and the response fields are
llama.cpp's (`tools/server/server.cpp:273`,
`server-context.cpp:4536-4567`, `server-task.cpp:1570-1592`), so a client
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
`cached_tokens: 83` in `usage` and prefills only the remainder, with the
same greedy output the cold server gave.

Two things differ from llama.cpp:

- `id_slot` is bookkeeping. llama.cpp has N fixed slots each owning a
  KV region, and `-np` sets N. frink builds KV per request and shares
  prefixes through one prefix cache, so there is no per-slot region for
  the id to select. It is validated (a non-negative integer) and
  echoed.
- `save` names its `prompt`. llama.cpp saves whatever the slot was last
  serving, and nothing here is "last serving" anything. The body
  carries the prompt to prefill, tokenized with BOS as a completion
  would be.

`action=erase` is `501`: the prefix cache has whole-cache clearing and
no per-entry eviction, so erasing one slot would mean erasing all of
them. Delete the file, or restart.

A restore is refused by name when the file is not of the model being
served. A slot file is raw attention state, and restoring it under
other weights does not fail, it answers wrongly. So the file carries
the checkpoint's identity and a restore compares it before storing
anything:

```
400 slot_model_mismatch
refusing to restore ./slots/system.fslot: checkpoint: the slot was saved
under 2b77f720... and this server is serving 3a59835e.... Restoring
attention state computed by other weights does not fail, it answers wrongly
```

The identity is a SHA-256 over the GGUF's metadata (sorted, typed), its
full tensor directory, and 4 KiB from the head and tail of every tensor,
beside the layer count, KV head count, head dimension and KV dtype.
Path, size and hyper-parameters were each rejected as the key because
each admits the likely pair, two fine-tunes of one base at one
quantisation. The same model at a different quantisation is refused on
`checkpoint`; a different model is refused on `model` or the first shape
field that differs. llama.cpp's own slot file
(`src/llama-context.cpp:3081-3141`) carries no identity.

The file is framed with a length-and-digest prefix like
`frink_core::kv_disk`'s blocks, so a truncated, edited or foreign file
is refused before anything is parsed or allocated. Filenames are a
whitelist (`[A-Za-z0-9._-]`, no leading dot), so a name cannot leave
`--slot-save-path`.

Slots are implemented for the generic GGUF decoder. The dedicated
engines refuse by name, as does a server without a checkpoint on disk.
The slot file header holds one KV geometry, so a model whose layers
cache different KV widths, or whose K and V heads differ in width, is
refused with a 501 as well.

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
`--lora-init-without-apply` starts at (0) rather than the flag's number.
`POST` takes the `[{id, scale}]` array (`scale` absent reads 0, as
upstream's `json_value` default); an `id` no adapter has is a 400
naming the range, where upstream ignores it. A request's `lora` list
means what `construct_lora_list` makes it mean: every adapter named
gets its scale, every other adapter 0 for that request, and an empty
list means the server's own scales, as upstream reads it. Naming an
adapter on a server that loaded none is a 400; the field on a
checkpoint served by a dedicated engine is a 501.

An adapter's scale is one atomic value read by every projection at
apply time, so a change recomputes nothing. What it costs is
exclusivity: a `POST`, or a request whose `lora` list differs from what
is applied, waits for the generations in flight and runs alone, then
the request's override is restored. A request whose list names exactly
the current scales is an ordinary concurrent request. This is
llama.cpp's rule (two slots with different LoRA lists are never
co-batched) at the granularity of the whole server rather than the
batch. The response cache keys on the scales a generation ran under, so
a `POST` between two identical requests does not serve the first answer
to the second.

## MCP

`--mcp-config PATH` loads server metadata under `frink_mcp` in
`GET /v1/models`. Tool invocation is not wired yet.

## Reasoning content

A checkpoint whose family reasons inside markers (`<think>`, DeepSeek's
DSML variant, Gemma's channels, MiniMax-M3's namespaced pair, the
gpt-oss harmony `analysis` channel) has that block split out of
`content` and returned as `reasoning_content`, on the message and on
each streamed delta. The split runs as tokens arrive, so an overlapped
SSE stream and a buffered one report the same fields for the same
request.

The family is inferred from the served model's name first, and from
what the checkpoint's chat template implies when the name says nothing.
frink carries no per-checkpoint parser declaration. A model for which
neither implies a family gets no reasoning parser, which is the right
answer for a model that does not reason, since an unconditional
splitter would eat a literal `<think>` written in a code block.

### Replaying it

`reasoning_content` is accepted on a request's assistant messages as
well as returned on responses, and `reasoning` is accepted as an alias
so a turn can be replayed in the shape `/v1/responses` or
`/v1/messages` handed it back. It is passed to the template on its own
key, under both spellings, since the DeepSeek and GLM lineages iterate
`message.reasoning_content` while Qwen and gpt-oss read
`message.reasoning`, and a template treats a key it does not know as
undefined. An empty string is dropped rather than passed, so a template
that opens its thinking markers on `if message.reasoning_content` does
not wrap them around nothing.

It is never folded into `content`: a past chain of thought shown as
content is something the model believes it said aloud. A template that
knows nothing about reasoning ignores the key and renders what it did
before.

Whether the block is already open is read off the rendered prompt: a
template asked to think can open the block in the prompt itself, and
then the model's first token is reasoning and no opening marker ever
arrives. The same checkpoint and request text can go either way
depending on what `chat_template_kwargs` rendered, so the prompt is
what gets consulted.

## Tool-call formats

When the template does not consume `tools`, the server describes them
in a preamble that asks for a Hermes-style
`<tool_call>{"name": …, "arguments": {…}}</tool_call>`. A model trained
on a different format frequently answers in its own, so parsing uses
the format the checkpoint implies: the one its chat template prints
when it prints one, otherwise the one the served model's name implies.
A forced `tool_choice` compiles the offered tools into a lazy grammar,
so the call itself is schema-constrained (see
[Grammar-constrained decoding](#grammar-constrained-decoding)).

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

Every call in a response is returned, with ids `call_0`, `call_1`, and
so on. Nothing in these formats carries an id, so a client correlates
by index.

On the private decode path five invoke/parameter families stream their
arguments as deltas: Qwen3-Coder, GLM-4.7, DeepSeek, MiniMax and
muse-glimmer. The first delta of a call carries `index`, `id`, `type`
and `function.name` with empty arguments, and every delta after it
carries only more `function.arguments` text. The fragments are literal
continuations, so a client concatenates them in `index` order and
parses the result.

`/v1/messages` and `/v1/responses` stream the same events in their own
protocols: Anthropic sends `content_block_start` with a `tool_use` block
and then `input_json_delta` fragments, and the Responses API sends
`response.output_item.added` followed by
`response.function_call_arguments.delta`.

The JSON-payload families (Hermes, Llama 3, Mistral, MiniMax-M3,
gpt-oss, Gemma) arrive whole, because a half-written JSON object is not
a usable fragment. A generation truncated mid-call reports `length`,
not `tool_calls`, so a half-written call is never executed. A call
naming a tool the request never offered is dropped; a namespaced name
(`skills:read`) is forwarded, because the client resolves the
namespace.

The XML-like formats state no types, so a parameter's value is typed
from the tool's own `parameters` schema: a declared `string` is handed
over verbatim, which keeps a zero-padded id like `"018956"` from
arriving as a number.

## Legacy completions

`/v1/completions` honours `prompt` (a string), `max_tokens` (default
16, because a caller completing a fragment usually wants a fragment
back), `stop`, `temperature`, `top_p`, `min_p`, `top_k`, `typical_p`,
`top_n_sigma`, `xtc_probability`, `xtc_threshold`, the five `dry_*`
fields, `samplers`, `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `ignore_eos`, `grammar`, `lora` and
`logit_bias`, plus the fields the table above marks as served here:
`logprobs`, `prompt_logprobs`, `n`, `best_of`, `echo`, `cache_salt`,
`allowed_token_ids`, `bad_words`, `truncate_prompt_tokens`,
`skip_special_tokens` and `return_tokens_as_token_ids`.

The chat route and this one read the same `SamplingKnobs::resolve`, so
which knobs exist and what an absent one means are stated once. An
absent `min_p` resolves to `0.0` (off) rather than llama.cpp's CLI
default of `0.05`: an HTTP caller is not running llama.cpp's command
line.

The route answers with one JSON body; it does not read `stream`. What
it does not implement comes back as an error naming the field:
token-id prompts (`[int]` / `[[int]]`), `suffix`, and any
`response_format` other than `{"type": "text"}`.

## llama.cpp's native `/completion`

`POST /completion` and `POST /completions` are llama.cpp's own
completion endpoint, which differs from `/v1/completions` in its
request fields, its response object and its stream. It is what
llama.cpp's own web UI, `llama.vim` and many wrappers speak.

| | `/completion` (llama.cpp) | `/v1/completions` (OpenAI) |
|---|---|---|
| budget | `n_predict`; `-1`, and absent, mean "until the context is full" | `max_tokens`, default 16 |
| repetition | `repeat_penalty`, `repeat_last_n` | `repetition_penalty` |
| response | flat object: `content`, `stop`, `stop_type`, `timings`, `generation_settings` | `choices[].text`, `finish_reason` |
| stream frame | `data: {"content":…,"stop":false}` | not streamed |
| stream end | the terminal object with `"stop": true`, and no `[DONE]` | not streamed |

Both paths are the same handler; `/admin/stats` records whichever one
the client called. Generation uses the same engine, sampler, grammar
seam and stop machinery as `/v1/completions`.

### What it honours

`prompt` (string, or `{"prompt_string": "…"}`) · `n_predict` ·
`stream` · `stop` · `temperature` · `top_p` · `min_p` · `top_k` ·
`typical_p` · `top_n_sigma` · `xtc_probability` · `xtc_threshold` ·
`dry_multiplier` · `dry_base` · `dry_allowed_length` ·
`dry_penalty_last_n` · `dry_sequence_breakers` · `repeat_penalty` ·
`repeat_last_n` · `presence_penalty` · `frequency_penalty` · `samplers`
· `seed` (`-1` draws one) · `ignore_eos` · `grammar` · `json_schema` ·
`logit_bias` · `lora` · `cache_prompt`.

The nine samplers of llama.cpp's default chain all run, in llama.cpp's
order (`penalties, dry, top_n_sigma, top_k, typical_p, top_p, min_p,
xtc, temperature`). An absent field resolves to the value that makes its
sampler a no-op.

`dry_sequence_breakers` are strings, tokenised against the loaded
model's own vocabulary (llama.cpp's `get_overlapping_token_sequences`).
A checkpoint with no real vocabulary (the synthetic-weight fallback)
refuses `dry_multiplier` rather than running DRY with no breakers. Send
`dry_sequence_breakers: []` to ask for DRY with no breakers on purpose.

`n_predict` keeps llama.cpp's meaning: an absent `n_predict` is `-1`,
which means "generate until the context is full". `-1` resolves against
the derived context ceiling minus the tokenized prompt. On a deployment
where the model could not be priced there is no ceiling, and `-1` is a
501 naming `n_predict` and `FRINK_CB_MAX_CONTEXT`. `n_predict: 0`
generates nothing, as upstream.

`repeat_last_n: -1` ("the whole context") is refused for the same
reason: shrinking it to the default 64 would give a caller a much
smaller window than the one it asked for.

`cache_prompt: true` is upstream's default and is a permission to reuse
KV, so it is always servable. `cache_prompt: false` is a requirement not
to, which this server can keep only when no prefix cache is configured;
with `FRINK_PREFIX_CACHE_ENTRIES` set it is refused.

`json_schema` is llama.cpp's bare-schema field, compiled through the
same site as `response_format: json_schema`. A `grammar` and a
`json_schema` in one request are refused (400); upstream prefers
`grammar` and drops the schema.

### What it refuses, by name

Every option below is checked against the value at which llama.cpp
itself treats it as off. At that value it is served, because a stock
client sends most of them explicitly and refusing `mirostat: 0` would be
a false refusal. At any other value it is a 501 naming the field:

`dynatemp_range` · `mirostat` · `n_probs` and `post_sampling_probs` ·
`min_keep` · `return_tokens` (the decode loop hands this layer text, not
ids) · `n_indent` · `n_keep` (frink refuses an oversized request rather
than shifting context, so there is nothing to protect) · `n_cmpl` ·
`n_cache_reuse` · `t_max_predict_ms` · `id_slot` (no slots) ·
`response_fields` · `return_progress` · `timings_per_token` ·
`sse_ping_interval` (the keepalive is fixed at 15 s).

Also refused: token-id prompts, multiple prompts in one request,
`prompt.multimodal_data`, and the fields the table above marks as 501
on this route, such as `n` above 1.

Options that only parameterise a switched-off sampler (`mirostat_tau`,
`mirostat_eta`, `dynatemp_exponent`) are not in that list: they do
nothing while their switch is off, and their switch is. A field
llama.cpp does not define either is ignored, as upstream ignores it.

### Response fields

`truncated` is always `false`: frink refuses a request that does not fit
its context instead of discarding tokens to make it fit, so a served
answer was never truncated. A `timings` value this server did not
measure is `null` rather than `0`, and `timings.cache_n` is `-1`
(upstream's sentinel) when no prefix cache is configured, which is
different from "the cache missed".

`stop_type` uses llama.cpp's vocabulary (`eos`, `word`, `limit`) plus
`cancelled`, for an answer stopped through `POST /v1/cancel` or by the
client disconnecting. None of upstream's three is true of an
interrupted answer, and `none` means "still generating".

One known inaccuracy: a caller stop string that is exactly one token in
the model's vocabulary is caught by the token layer of the stop
machinery, which matches on the id and does not carry back which string
it was, so it is reported as `"eos"` where llama.cpp would say `"word"`.
`/v1/messages` loses its `stop_sequence` attribution on the same
inputs. Multi-token stop strings, which is what a `/completion` caller
normally sets, are reported correctly.

## Tokenize / detokenize, in both dialects

OpenAI has no tokenize endpoint, so `/v1/tokenize` is frink's own
spelling, while llama.cpp serves `/tokenize` and `/detokenize`
unprefixed. Both spellings answer, on the same handler, and
`/admin/stats` records whichever path the client called.

The request body accepts both dialects on both paths:

| Field | Status |
|---|---|
| `content` | llama.cpp's name for the text. Supported |
| `prompt` | frink's name for the same text. Supported |
| both at once | 400. They are one field, and guessing which was meant would tokenize text the caller did not ask about |
| neither | 400 naming both. llama.cpp answers an empty array here; frink does not, because an empty array cannot be told apart from tokenizing `""` |
| `add_special` | Supported, default `false` as upstream. Prepends the same BOS id the generation path prepends, including its no-op on a checkpoint whose metadata says not to add one, so the count matches the prompt the model would see |
| `parse_special: true` (upstream's default) | Supported. Special-token markers in the text are the tokens they name |
| `parse_special: false` | Supported. `<|im_start|>` is tokenized as the characters it is written with, as llama.cpp does |
| `with_pieces: true` | 501 by name. frink's tokenizers expose decoded text, not raw per-token piece bytes, so a byte-fallback token could not be given llama.cpp's `piece` byte array |
| `model` | Accepted and ignored; this server serves one model at a time |

The tokenize response carries `tokens` (llama.cpp's key) plus frink's
own `count`. The detokenize response carries the text under both
`content` (llama.cpp's key) and `text` (frink's), the same string, so
neither dialect's client reads a null.

## Grammar-constrained decoding

`"grammar": "<GBNF>"` is llama.cpp's own field and takes the same
syntax. It is accepted on `/v1/chat/completions`, `/v1/completions` and
`/completion`, compiled once per request, and enforced on every token by
a stack machine.

`response_format: json_object` masks character classes, so it cannot
know whether a `}` closes an object that was opened; a grammar can. The
two compose, and neither can un-mask what the other forbids.

A grammar that is satisfied and has no legal continuation ends the
response normally (`finish_reason: "stop"`). Only a grammar that
dead-ends while still unsatisfied is an error, a 400 with
`failure_code: "invalid_grammar"`. Under continuous batching that
failure is scoped to the row that hit it; other requests in the batch
keep running.

`response_format: {"type": "json_schema", "json_schema": {"name": ...,
"schema": {...}, "strict": true}}` is compiled to GBNF by the same
converter `tool_choice` uses and enforced by the same stack machine, so
a `required` property cannot be omitted and an `enum` cannot be
invented. `name` and `description` are labels and constrain nothing.

Everything inside `json_schema` that this server does not act on is
refused by name: an unknown member is a 400 naming it, and
`strict: false` is a 400 too. OpenAI's `strict: false` asks for
best-effort guidance and permits schemas that strict mode rejects;
frink has one behaviour for a schema, which is to enforce it token by
token, so serving it under `strict: false` would report a guarantee the
caller declined. An omitted `strict` is enforced.

A schema the converter will not compile is a 400 naming the keyword and
the subschema it sits in. The converter is stricter than llama.cpp's on
purpose: upstream discards a keyword no branch tested, so
`{"type": "string", "pattern": "^[a-z]+$"}` compiles upstream to a
grammar for any string.

A `grammar` and a `json_schema` in one request are two constraints on
one generation and are refused (400). A forced `tool_choice` beside
either is refused the same way.

One observable difference from llama.cpp: `required` members are
enforced in lexicographic order rather than the schema's declaration
order, because the request body reaches the converter as a parsed value
and this workspace builds `serde_json` without `preserve_order`. Both
are valid grammars for the schema.

`tool_choice: "required"` and a named `tool_choice` are supported by a
lazy grammar built from the request's own `tools`. Each tool's
`arguments` rule is its `parameters` JSON Schema compiled to GBNF. A
named choice is the same grammar with the union narrowed to one
alternative.

The trigger is mandatory, which departs from llama.cpp. Upstream forces
a call with an eager grammar. Several families open the reasoning block
in the prompt, so the model's first token is already inside `<think>`,
and a call forced there would be read back by frink's reasoning parser
as thinking: the caller who demanded a call would get
`reasoning_content` and no call. So the grammar stays lazy, triggers on
the wire format's opening marker, and while waiting masks only the
end-of-generation tokens: the prefix is free, but the turn cannot end
until a call has begun. A model that never opens a call runs to
`max_tokens` and finishes `"length"`.

Ten of the eleven wire formats are supported, with one root rule per
shape:

| Shape | Formats | Root rule |
|---|---|---|
| JSON payload | Hermes/Qwen2.5, Llama 3, Mistral | a marker, a JSON object naming the tool, a closing marker |
| Elements | `qwen3_coder`, `glm47`, `minimax`, `deepseekv32`, `minimax_m3` | an invoke element holding one element per argument |
| Harmony | `gpt_oss` | a channel header addressed to `functions.<name>`, then JSON |
| Pairs | `gemma4` | `call:NAME{k:v,k:v}`, the values in Gemma's own quoting |

The grammar's literals are built from the same marker description the
streaming parser reads with, so the engine cannot force output that it
then fails to parse back.

An argument whose declared type the family's own spelling and this
server's reader disagree about is refused by property name. On `gemma4`
that is an `object` or `array` argument: the template writes a
composite in Gemma's DSL (bare keys, Gemma-quoted strings) and frink
reads a value with `serde_json`, so the spelling the checkpoint was
trained to write would come back as a string. On the element formats it
is a property with no declared `type` at all.

`muse_glimmer` returns 501 naming the format. Its call boundary is not
syntactic: the same ATEM block is a call in a tool channel and prose in
a user-facing one, and forcing the channel header instead would need
two facts about the checkpoint's template rather than the format (which
recipient name it addresses a tool with, and how much of that header
the rendered prompt already wrote, since a muse-glimmer prompt ends
inside one). No muse-glimmer checkpoint or template is on hand to read
either off.

## Not yet

Image and audio input · MCP tool invocation (the config is read,
nothing is called) · the embedding architectures listed above as
refused by name · multi-GPU, tensor parallel, prefill/decode
disaggregation · streamed argument deltas for the JSON-payload tool
formats (they arrive whole) · `stream` on `/v1/completions` ·
speculative decoding for grammar-constrained or paged requests, so
their speculation fields in `usage` are absent · llama.cpp's `/infill`,
`/props`, `GET /slots` (the live slot listing; `POST /slots/{id}` save
and restore is supported, see above) and `/apply-template`, none of
which has a frink counterpart.

A few request fields deserialize and then go nowhere, accepted so a
stock client's body does not fail validation over something this server
has no use for: `metadata` on `/v1/messages`, and `store`, `metadata`
and `parallel_tool_calls` on `/v1/responses`. Everything else this
server does not implement is refused by name.

See [`ROADMAP.md`](ROADMAP.md).
