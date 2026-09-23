# Agents & IDEs

Point coding agents at a running Frink server. Start it either way:
`frink serve` from the main binary (needs `--features serve` at build
time) or the standalone `frink-server`. Both take the same flags.

```bash
cargo build -p frink-cli --release --features "serve metal"
./target/release/frink serve -m /path/to/model.gguf \
  --host 127.0.0.1 --port 8383
```

**OpenAI-compatible base URL:** `http://127.0.0.1:8383/v1`

| Client | Setting |
|---|---|
| Cursor / OpenAI SDK | `baseURL` → `http://127.0.0.1:8383/v1` |
| OpenCode / Cline / similar | OpenAI-compatible provider → same URL |
| Anthropic SDK | `baseURL` → `http://127.0.0.1:8383`, use `POST /v1/messages` |
| codex | Responses provider → `POST /v1/responses` |
| curl | `POST /v1/chat/completions` (see [CLI.md](CLI.md)) |

If `FRINK_API_KEY` is set, send `Authorization: Bearer …` or
`x-api-key: …`. Both are accepted, so an Anthropic SDK works unchanged
against a keyed server. Leave the key unset on a loopback bind if you
would rather not send one at all.

Tool calls come back parsed in eleven wire formats rather than only the
one the prompt asks for, and five of those stream their arguments as
deltas, so an agent watching a file path arrive does not wait for the
whole call. A reasoning model's chain of thought arrives separately as
`reasoning_content` (`thinking` blocks on `/v1/messages`).

Also available: `POST /v1/tokenize`, `/v1/detokenize`, `/v1/embeddings`
(Decoder pool), `POST /v1/messages/count_tokens`, and `POST /v1/cancel`
to stop a generation by the `request_id` its first streamed chunk
carries. Full list: [API.md](API.md).

For a browser client, Frink Studio lives in [`ui/`](../ui) as a
separate app. This server does not serve it, and `GET /` here is a 404.
Run `npm run dev` from that directory.

## Continuous batching

```bash
FRINK_CONTINUOUS_BATCHING=1 ./target/release/frink-server -m model.gguf …
```

- Mutually exclusive with `FRINK_KV_POOL_BLOCKS` and `FRINK_PREFIX_CACHE_ENTRIES`
- GGUF Decoder only (Kimi / MLA ignore CB)

## Sharing a system prompt between conversations

Agents send the same long preamble on every turn. Paged KV stores it
once and lets each conversation point at those pages, instead of every
request holding a copy:

```bash
FRINK_PAGED_KV_BLOCKS=4096 FRINK_PAGED_KV_BLOCK_SIZE=16 \
  ./target/release/frink-server -m model.gguf -dev metal -ngl all
```

`usage.cached_tokens` on each response says how much of that prompt was
already computed. This used to be refused on a GPU backend, because a
Metal prefill left K/V on the device and the paged prefill copied host
placeholders into the page store. Fixed, and pinned on hardware by
`paged_metal_parity`, so Metal is supported. CUDA has no equivalent
hardware run behind it. See [CONFIG.md](CONFIG.md).

**Sharing pages is sharing, so say whose they are.** Two callers
sending the same preamble get the same pages, which is the point and
also a disclosure if they are different tenants. `cache_salt` on the
request names a namespace: the contiguous prefix cache, the response
cache and the paged radix tree all key on it, so a salted caller can
match only its own prefixes. Absent means the shared namespace, which
is what every request got before the field existed.

## Putting a model away without losing it

A scheduler that runs several models on one box can free a model's
memory and bring it back without knowing the deployment:

```bash
curl -X POST http://127.0.0.1:8383/sleep
curl     http://127.0.0.1:8383/is_sleeping     # {"is_sleeping": true}
curl -X POST http://127.0.0.1:8383/wake_up
```

`/sleep` is an unload that REMEMBERS: it frees the KV pool, the paged
store, the repack and expert caches and any device buffers, and keeps
the checkpoint path, so the server can reload itself.
`/admin/models/unload` leaves nothing behind. While asleep, routes that
need a model answer `503 server_sleeping` rather than
`model_not_loaded`, so a caller can tell "put away, ask again" from
"nothing here".

Full API matrix: [API.md](API.md).
