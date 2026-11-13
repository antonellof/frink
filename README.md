<div align="center">

<img src="docs/assets/frink-logo.webp" alt="Frink" width="38%" />

**A pure-Rust GGUF inference engine. Dense and MoE, on CPU, Apple Metal, or CUDA.**

[![CI][ci-badge]][ci-workflow]
[![Latest release][release-badge]][latest-release]
[![crates.io][crates-badge]][crates-url]
[![docs.rs][docs-badge]][docs-url]
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built_with-Rust-dea584.svg)](https://www.rust-lang.org/)
[![Backends](https://img.shields.io/badge/backends-CPU%20%7C%20Metal%20%7C%20CUDA-64748b.svg)](docs/FEATURES.md)

**[Features](docs/FEATURES.md)** ·
**[Models](docs/MODELS.md)** ·
**[CLI](docs/CLI.md)** ·
**[API](docs/API.md)** ·
**[Config](docs/CONFIG.md)** ·
**[Benchmarks](benchmarks/README.md)** ·
**[Studio UI](ui/)** ·
**[Agents](docs/AGENTS_COOKBOOK.md)** ·
**[Roadmap](docs/plans/north-star.md)** ·
**[Changelog](CHANGELOG.md)** ·
**[Contributing](CONTRIBUTING.md)**

</div>

---

Frink loads GGUF checkpoints and runs them on the hardware you already
own. It does not bind to llama.cpp or wrap ggml: the loader, the
quantized kernels, attention and expert routing are written here, in
Rust.

- **One binary, no Python.** The macOS arm64 prebuilt has Metal and the
  server built in, plus `download`, `bench`, `batched-bench`,
  `quantize`, `imatrix`, `gguf-split` and `verify`.
- **Weights stay quantized.** Memory-mapped and dequantized inside the
  matmul. K-quants, the IQ tiers, MXFP4, F16 and BF16 all load.
- **A drop-in for llama.cpp.** Same flags, sampler order and tools.
  `quantize` writes Q8_0 and the K-quants byte-identically to
  `llama-quantize`. `frink parity` checks tokens, logits and a greedy
  decode against libllama, and CI runs it on every fixture. 101
  architectures run with a logit comparison behind them; any other
  stops with an error naming what is missing.
- **llama-server's chat prompts.** The GGUF's Jinja template renders
  with llama-server's variables, so a thinking model thinks by default.
  `--chat-template`, `--chat-template-kwargs`, `-rea` and
  `--reasoning-effort` keep llama-server's spelling.
- **An OpenAI-compatible server**, plus the Anthropic and Responses
  APIs, paged KV shared across conversations, model swap, sleep and
  wake, slot save and restore, and resumable streams. With
  `--cont-batching` or `--parallel N`, clients share one batched decode
  worker. [Measured](benchmarks/RESULTS.md#serving-features-measured):
  1.28x aggregate throughput at eight concurrent requests, and 736 of
  757 prompt tokens reused on a shared system prompt.
- **The full sampling API and structured output.** `n`, `best_of`,
  `logprobs`, `logit_bias` and the rest; a field the server does not
  implement is refused by name. A GBNF grammar, JSON schema or forced
  `tool_choice` masks every token that would break it.
- **Built for agents.** Reasoning streams into `reasoning_content`, and
  tool calls parse in the eleven formats real checkpoints emit.
- **Lossless speculative decoding** with a draft model (`-md`), a
  DFlash block draft, or prompt lookup in the server.
- **Mixture of experts** with GPU routing, indexed expert GEMMs, and
  experts streamed from the checkpoint only when they do not fit.
- **Embeddings, reranking and scoring** on BGE, E5, GTE and
  cross-encoder checkpoints.

The full list is in [docs/FEATURES.md](docs/FEATURES.md).

## Performance

`frink bench` runs llama-bench's workload (`pp512`, `tg128`) on the
same GGUF and machine as `llama-bench`. Every published number has a
receipt in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

- **Metal (Apple M2 Pro, 0.20.0).** Prefill within 11% of llama.cpp,
  decode level with it or ahead on every model in the suite.
- **CUDA (NVIDIA L40S, 0.59.0).** Dense 3B to 12B: decode 1.2x to 1.4x
  slower, prefill 3x to 4x. Qwen3-30B-A3B: 2x on decode, 5.5x on
  prefill; gpt-oss still runs on the host path
  ([the L40S rows](benchmarks/RESULTS.md#amd-epyc-7r13-processor-4c-linux-612113-138209amzn2023x86_64--nvidia-l40s)).
  Closing that tops the [roadmap](docs/plans/north-star.md) (#133, #259).
- **CPU.** Behind llama.cpp, by less on decode than on prefill.

Every row was measured on an older build than the current one, and the
table marks each one so.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/frink/main/scripts/install.sh | bash
```

Installs `frink` (built with `serve`) and `frink-server` into
`~/.local/bin` (override with `FRINK_INSTALL_DIR`, pin with
`FRINK_VERSION=v0.63.0`). Prebuilts are macOS arm64 with Metal and
Linux x86_64 with CPU.

From crates.io or source:

```bash
cargo install frink-cli --features metal      # one binary, everything
cargo install frink-server --features metal   # the server on its own
cargo build --release -p frink-cli --features metal
```

The only feature to choose is the GPU backend: `metal` on Apple
silicon, `cuda` on Linux with NVIDIA, none for CPU only. `vulkan` is
early and runs some decode work when `FRINK_VULKAN=1` is set.

## Quick start

Commands are spelled as in llama.cpp. Frink uses the GPU it was built
for; add `-ngl 0` for the CPU. `-hf` fetches from Hugging Face and the
prompt is wrapped in the model's chat template (`--no-cnv` for a raw
completion).

```bash
frink -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
  -p "Explain quantization in two sentences" -n 128

# Or download to a folder first (hf download's syntax, no Python).
frink download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
frink -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "Hi" -n 64
```

Serve it on `127.0.0.1:8383` and point any OpenAI client at `/v1`.
Most llama-server flags work as spelled (`-c`, `--api-key`, `--alias`,
`--jinja`); `frink-server` is the same server as its own binary.

```bash
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 8192 --alias local &
curl -s http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64}'
```

Compare with llama.cpp on the same file (`--compare` also runs
`llama-bench`), or check numerics with `frink parity`:

```bash
frink bench -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 --compare
```

The numeric deviations from ggml that remain are listed with their size
in [ADR 0009](docs/adr/0009-numerics-follow-ggml.md#known-deviations).
[docs/MODELS.md](docs/MODELS.md) lists what runs and what refuses.
Speculative decoding, sampling and every other flag are in
[docs/CLI.md](docs/CLI.md).

## Frink Studio

A web UI for the server (chat, model inventory, live activity,
connection snippets) that uses only the public API.
[`ui/README.md`](ui/README.md) has more.

<img src="docs/assets/studio-chat.webp" alt="Frink Studio chat" width="100%" />

```bash
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl all &
cd ui && npm install && npm run dev      # http://localhost:5173
```

## Use it as a library

Published as [`frink-inference`](https://crates.io/crates/frink-inference),
a facade over the workspace (the `frink` crate on crates.io is
unrelated). Features, none on by default: `metal`, `cuda`, `api`.

```toml
[dependencies]
frink-inference = "0.63"
```

```rust
use frink_inference::gguf::ShardedGguf;
use frink_inference::models::{Decoder, ModelConfig};

let path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
let file = ShardedGguf::open(path)?;              // metadata only
let config = ModelConfig::from_gguf(&file)?;      // guesses in config.best_effort_fields
let decoder = Decoder::from_gguf(path, config)?;
```

Each layer is also published on its own, at one shared version:
[gguf](https://crates.io/crates/frink-gguf), [quant](https://crates.io/crates/frink-quant),
[safetensors](https://crates.io/crates/frink-safetensors), [core](https://crates.io/crates/frink-core),
[moe](https://crates.io/crates/frink-moe), [models](https://crates.io/crates/frink-models),
[api](https://crates.io/crates/frink-api), [metal](https://crates.io/crates/frink-metal),
[cuda](https://crates.io/crates/frink-cuda), [vulkan](https://crates.io/crates/frink-vulkan) (early).

## AI full disclosure

This software is developed with heavy assistance from Cursor, Grok 4.5,
GPT 5.6, and Claude Fable 5. Humans lead the ideas, the testing, and the
debugging. If you don't want AI-developed code, this software is not
for you. It also would not exist without
[llama.cpp](https://github.com/ggerganov/llama.cpp) and GGML, which
were largely written by hand.

## Acknowledgements to llama.cpp and GGML

Frink does not link against GGML, but it exists because of llama.cpp:
its kernels, quantization formats, the GGUF ecosystem and years of work
done in the open were our main reference. Some pieces are retained or
adapted under the MIT license, notably the IQ codebook tables; others
(GGUF layouts, quant and dot semantics, CLI and server conventions) were
written independently against that public design. We keep the GGML
authors' copyright notice in
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).

[ci-badge]: https://github.com/antonellof/frink/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/antonellof/frink/actions/workflows/ci.yml
[release-badge]: https://img.shields.io/github/v/release/antonellof/frink?display_name=tag
[latest-release]: https://github.com/antonellof/frink/releases/latest
[crates-badge]: https://img.shields.io/crates/v/frink-inference.svg
[crates-url]: https://crates.io/crates/frink-inference
[docs-badge]: https://docs.rs/frink-inference/badge.svg
[docs-url]: https://docs.rs/frink-inference
