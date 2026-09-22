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
| `logprobs` | **Served on `/v1/completions`**: `logprobs: N` reports the chosen token's log-probability and up to `N` alternatives per position, over the distribution the sampler actually drew from -- penalties applied, llama.cpp's chain run, a grammar's mask included. A candidate the chain REMOVED is omitted rather than reported as `null` or as a large negative stand-in: `ln(0)` is not a number, and it was not a candidate. `N` above 5 is a **400** naming the field, as upstream caps it. `text_offset` is a byte offset into the returned text, computed from the same pieces `tokens` reports. Also served on `/v1/chat/completions` as `logprobs: true` plus `top_logprobs: N` (cap 20), in OpenAI's CHAT shape -- `choices[].logprobs.content[]` of `{token, logprob, bytes, top_logprobs[]}`, not the completions wire's parallel arrays. `top_logprobs` without `logprobs: true` is a **400**: guessing which of two fields the caller meant would answer a question nobody asked. Such a request is **uncacheable** and always misses the response cache, which stores text and finish reasons and never distributions |
| `n` (>1) | **Served on `/v1/completions` and `/v1/chat/completions`**: the prompt is prefilled once and the KV forked per choice, so `prompt_tokens` counts the prompt ONCE while `completion_tokens` sums. Each choice is parsed for tool calls and reasoning in its own right. Choice `i` samples from `seed + i`, so choice 0 is byte-identical to the `n = 1` answer at the same seed, and at `temperature 0` every choice is the same sequence because the sampler is deterministic. **501 naming the field** on the native `/completion`, which returns a single `content`, and for `n` > 1 together with `stream`, because the choices would arrive one after another rather than interleaved by `choices[].index` |
| `best_of` (>1) | **Served on `/v1/completions` and `/v1/chat/completions`**: generates `best_of` completions from ONE shared prefill and returns the `n` best by **summed log-probability**, upstream's rule. The sum is length-sensitive and so favours short answers; a per-token mean would rank differently and is deliberately not used, because a different ranking from every other engine is worse than a known bias. Ties keep generation order, so a seeded request is reproducible. `usage.completion_tokens` counts **every** token generated including the discarded ones, while `prompt_tokens` counts the prompt once. `best_of` below `n` is a **400** naming both numbers. **501** on the native `/completion`, which has no `choices` array |
| `best_of` (>1), `prompt_logprobs`, `echo`, `use_beam_search`, `truncate_prompt_tokens`, `prompt_embeds`, `allowed_token_ids`, `bad_words`, `skip_special_tokens: false`, `return_tokens_as_token_ids` | **Reject, 501 naming the field**, on `/v1/chat/completions`, `/v1/completions` AND llama.cpp's native `/completion`. Each changes the tokens or the text returned, so ignoring one answers a question the caller did not ask. The default a caller may legitimately spell out (`n: 1`, `echo: false`, `skip_special_tokens: true`) is **served**. One table, flattened into all three request bodies, so a field cannot be refused on one wire and dropped on another again |
