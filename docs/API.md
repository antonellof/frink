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
| `POST /v1/score` · `POST /score` | Sentence-pair scoring. One text against many, or two lists of the same length paired element by element; different lengths are a **400** naming both counts. A cross-encoder answers with its head, a bi-encoder with the cosine of two embeddings, and `frink_score_kind` says which. A generative model is a **501** |
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
| `POST /sleep` · `POST /wake_up` · `GET /is_sleeping` | An unload that REMEMBERS, so the server can wake itself; `/admin/models/unload` cannot. Frees the KV pool, the paged store, the repack and expert caches and any device buffers. Idempotent. A model with no checkpoint path on record is a `409`. While asleep, routes needing a model answer `503 server_sleeping` rather than `model_not_loaded`. One level, not two: frink mmaps its weights |
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
| `logprobs` | Served on `/v1/completions` (parallel arrays plus `text_offset`) and on `/v1/chat/completions` as `logprobs: true` + `top_logprobs` (OpenAI's `content[]` shape). Over the distribution the sampler drew from, so penalties and a grammar mask are in it; a candidate the chain REMOVED is omitted rather than reported as `null`. Cap 5 on completions, 20 on chat, above which is a 400. Such a request always misses the response cache, which stores text and not distributions |
| `n` (>1) | Served on `/v1/completions` and `/v1/chat/completions`: prefilled ONCE and the KV forked per choice, so `prompt_tokens` counts the prompt once. Choice `i` samples from `seed + i`, so choice 0 is byte-identical to the `n = 1` answer. With `stream` the choices INTERLEAVE, each chunk carrying its own index. **501** on the native `/completion`, which returns a single `content` |
| `best_of` (>1) | Served where `n` is: generates `best_of` from one prefill, returns the `n` best by **summed log-probability**, upstream's rule. The sum favours short answers; a per-token mean would rank differently and is deliberately not used. `usage.completion_tokens` counts every token generated, discards included. `best_of < n` is a 400 |
| `cache_salt` | Served on every store. Names the caller's prefix-cache namespace; the contiguous cache, the response cache and the paged radix tree all key on it, so two callers sending the same prompt cannot be served each other's prefix. Hashed, so only the hash is kept. Absent, empty or whitespace is the shared namespace |
| `prompt_logprobs` | Served on `/v1/completions`: one entry per prompt token, the first `null` because nothing predicted it, each later one with a 1-based `rank`. Scored from the **plain softmax**, not the sampler's chain, because a prompt token was supplied rather than drawn. Cap 5. On the paged store such a request declines the prefix tree and runs its whole prompt. **501** on the other wires, which have no field for it |
| `logit_bias` | Served on all three generation wires. `{"<token id>": <bias>}`, added before sampling, range `-100..100`; outside it is a **400** rather than a clamp. `{}` and `null` are accepted. A bias cannot lift a token a grammar, JSON mode, `allowed_token_ids` or `bad_words` forbade |
| `allowed_token_ids`, `bad_words` | Served on all three generation wires, in the same mask as the grammar. They steer the draw; `stop` ends a generation. `bad_words` is TOKENIZED: a word's last token is forbidden only when the tokens before it are what was just generated, so it is exact only for the tokenization the model would have produced. An empty `allowed_token_ids` is a **400** |
| `echo` | Served on `/v1/completions`, the wire that returns a continuation of the prompt; **501** on the two that return a message. Prompt and completion come back as one `text`, and with `logprobs` the arrays cover both. After a truncation it echoes the tokens that were KEPT |
| `truncate_prompt_tokens` | Served on all three generation wires: keeps the prompt's LAST `k` tokens, applied before anything is prefilled, so the KV, `usage.prompt_tokens`, the prefix cache and `echo` all see the prompt that was answered. Below 1 is a **400** |
| `skip_special_tokens` | Served on all three generation wires. `false` keeps the model's end-of-generation token in the text and counts it; it still ENDS the answer. A caller's own `stop_token_ids` entry is unaffected: they asked to stop ON it, which means before it |
| `return_tokens_as_token_ids` | Served on all three generation wires: a REPORTED token is spelled `token_id:123` instead of its text, in the `tokens` array, the `top_logprobs` keys and chat's `content[].token`. The completion's `text` is unchanged. It exists because two ids can detokenize to one string, and a map keyed by text loses one of them |
| `use_beam_search`, `prompt_embeds` | **Reject, 501 naming the field**, on `/v1/chat/completions`, `/v1/completions` AND llama.cpp's native `/completion`. Each changes the tokens or the text returned, so ignoring one answers a question the caller did not ask. The default a caller may legitimately spell out (`n: 1`, `echo: false`, `skip_special_tokens: true`) is **served**. One table, flattened into all three request bodies, so a field cannot be refused on one wire and dropped on another again |
