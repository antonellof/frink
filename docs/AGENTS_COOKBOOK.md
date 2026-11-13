# Agents & IDEs

Point coding agents at a running Frink server (`frink serve` or the
standalone `frink-server`; both take the same flags).

```bash
cargo build -p frink-cli --release --features metal   # serve is a default feature
./target/release/frink serve -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383
```

The OpenAI-compatible base URL is `http://127.0.0.1:8383/v1`.

| Client | Setting |
|---|---|
| Cursor / OpenAI SDK | `baseURL` → `http://127.0.0.1:8383/v1` |
| OpenCode / Cline / similar | OpenAI-compatible provider → same URL |
| Anthropic SDK | `baseURL` → `http://127.0.0.1:8383`, use `POST /v1/messages` |
| codex | Responses provider → `POST /v1/responses` |
| curl | `POST /v1/chat/completions` (see [CLI.md](CLI.md)) |

With `FRINK_API_KEY` set, send `Authorization: Bearer …` or
`x-api-key: …`; both work, so an Anthropic SDK needs no change.

Tool calls parse in eleven wire formats, five of which stream their
arguments as deltas. Reasoning arrives separately as
`reasoning_content` (`thinking` blocks on `/v1/messages`).
`POST /v1/cancel` stops a generation by the `request_id` its first
streamed chunk carries. Every route is in [API.md](API.md).

Frink Studio, a browser client, is a separate app in [`ui/`](../ui)
(`npm run dev` there); `GET /` on the server is a 404.

## One agent: the prefix cache

By default requests run one at a time on the private decode loop, the
fast path for a single agent (on Metal it attends on the device). An
agent resends the whole conversation every turn; keep its KV so a
follow-up prefills only the new message:

```bash
FRINK_PREFIX_CACHE_ENTRIES=2 ./target/release/frink-server -m model.gguf
```

Each entry is a full copy of one conversation's KV; one or two is
enough for one client. `GET /cache/stats` reports hits.

## Several clients: continuous batching

Off by default; turn it on for concurrent clients:

```bash
./target/release/frink-server -m model.gguf -cb      # or FRINK_CONTINUOUS_BATCHING=1
./target/release/frink-server -m model.gguf -np 4
```

- Not with `FRINK_KV_POOL_BLOCKS` or `FRINK_PREFIX_CACHE_ENTRIES`, unless
  paged KV is on. With either set, the server keeps the private loop
  and logs a warning
- The generic GGUF decoder only. The dedicated engines (MLA, Gemma 4,
  GLM-5.2, Kimi) serve on the private loop
- A draft model (`-md`) and prompt-lookup speculation work inside the
  batch as on the private loop

## Several clients: sharing a system prompt

Paged KV stores a shared preamble once and lets each conversation
point at its pages:

```bash
FRINK_PAGED_KV_BLOCKS=4096 FRINK_PAGED_KV_BLOCK_SIZE=16 \
  ./target/release/frink-server -m model.gguf -dev metal -ngl all
```

`usage.cached_tokens` says how much of the prompt was already
computed. Paged KV composes with continuous batching and excludes
`FRINK_PREFIX_CACHE_ENTRIES`. Metal is pinned on hardware by
`paged_metal_parity`; CUDA has no hardware run behind it.

Shared pages across tenants are a disclosure. `cache_salt` on the
request names a namespace that the prefix cache, the response cache and
the radix tree all key on, so a salted caller matches only its own
prefixes; absent means the shared namespace. Switches are in
[CONFIG.md](CONFIG.md).

## Putting a model away without losing it

Free a model's memory and bring it back without knowing the
deployment:

```bash
curl -X POST http://127.0.0.1:8383/sleep
curl     http://127.0.0.1:8383/is_sleeping     # {"is_sleeping": true}
curl -X POST http://127.0.0.1:8383/wake_up
```

`/sleep` frees the KV pool, paged store, repack and expert caches and
device buffers but keeps the checkpoint path, so the server can reload
itself; `/admin/models/unload` keeps nothing. While asleep, model routes
answer `503 server_sleeping` rather than `model_not_loaded`.
