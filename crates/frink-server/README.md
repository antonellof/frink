# frink-server

OpenAI-compatible HTTP server for Frink (`frink-server` binary).

Serves chat/completions against a GGUF (or Kimi safetensors dir) via
`FRINK_MODEL_PATH`. See [`docs/API.md`](../../docs/API.md).

## Parallel serving (Metal)

Multiple concurrent clients are supported via **continuous batching**
(llama.cpp slots + one batched decode worker). On Metal, this is **on by
default** when KV pool and prefix cache are not configured.

```bash
frink serve -m model.gguf -dev metal -ngl all              # CB auto-on
frink serve -m model.gguf -dev metal -cb -np 4             # explicit slot cap
frink serve -m model.gguf -dev metal --no-cont-batching      # private path (serialized on Metal)
```

See [`docs/plans/metal-parallel-concurrency.md`](../../docs/plans/metal-parallel-concurrency.md).

Serving measurements (continuous batching, paged KV, prefix cache)
are in
[`benchmarks/RESULTS.md`](../../benchmarks/RESULTS.md#serving-features-measured),
with the raw receipts under
[`benchmarks/receipts/serving/`](../../benchmarks/receipts/serving/).
To measure your own deployment, point `frink serve-bench` at a running
server.

## The web UI is a separate app

This binary serves the HTTP API and nothing else. `GET /` is a 404 like
any other unknown path. **Frink Studio**, the chat / models / activity /
connect frontend, lives at [`ui/`](../../ui) in the repository root and
reaches this server over the same public API an editor would use.

```bash
cargo run -p frink-server -- -m model.gguf     # terminal 1
cd ui && npm install && npm run dev             # terminal 2 -> :5173
```

`npm run dev` proxies `/v1`, `/admin`, `/health`, `/metrics` and
forwards to the server address printed at startup.
