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
**[Roadmap](docs/ROADMAP.md)** ·
**[Changelog](CHANGELOG.md)** ·
**[Contributing](CONTRIBUTING.md)**

</div>

---

Frink loads GGUF checkpoints and runs them on the hardware you already
own. It does not bind to llama.cpp or wrap ggml: the loader, the
quantized kernels, attention and expert routing are all written here,
in Rust.

- Everything ships as one 23 MB binary (macOS arm64, with Metal and the
  server): completions, the API server, `download`, `bench`,
  `batched-bench`, `quantize`, `imatrix`, `gguf-split` and `verify`.
  There are no Python wheels to install and no CUDA userspace to match
  against a driver. For comparison, PyTorch alone is 402 MB before a
  serving stack sits on top.
- Weights stay quantized on mmap and are dequantized inside the matmul,
  so an 8B model fits on a laptop. Supported formats are the K-quants,
  the IQ tiers, MXFP4, F16 and BF16.
- It works as a drop-in for llama.cpp. The flags are the same, the
  sampler chain runs in the same order, and the tools match: `quantize`
  writes Q8_0 and the K-quants byte-identically to `llama-quantize`,
  with or without an importance matrix, and `gguf-split`, `imatrix` and
  `batched-bench` are ports. Tokenization is verified against libllama
  on twenty checkpoints under both special-token settings, and every
  engine number in [the speed table](benchmarks/RESULTS.md) was measured
  against llama.cpp on the same host with the same file. 100
  architectures run with a logit comparison behind them. Any other
  architecture stops with an error that says what is missing.
- The server is OpenAI-compatible. On Metal, concurrent clients share
  one batched decode worker (llama.cpp slots plus continuous batching,
  on by default). It also has paged KV shared across conversations,
  runtime model swap and sleep/wake, slot save and restore that names
  the checkpoint when it refuses a mismatch, resumable streams,
  Anthropic and Responses endpoints, and speculative decoding that
  stays lossless at any temperature. Your existing client can point at
  it unchanged.
  [Measured](benchmarks/RESULTS.md#serving-features-measured), that comes
  to 1.28x aggregate throughput at eight concurrent requests, and 736 of
  757 prompt tokens reused on a shared system prompt.
- The full sampling API is there: `n` and `best_of` from one shared
  prefill, interleaved when streaming; `logprobs` and
  `prompt_logprobs`; `logit_bias`, `allowed_token_ids` and `bad_words`;
  `echo`, `truncate_prompt_tokens` and `cache_salt`. A field the server
  does not implement gets an error naming that field. Silently dropping
  it would look the same to the client as honouring it.
- Structured output is enforced per token. Given a GBNF grammar, a
  forced `tool_choice` in ten of the eleven tool-call wire formats, or a
  tool's own `parameters` schema, a stack machine masks every token that
  would break the constraint, so the model cannot produce an invalid
  answer and there is nothing to retry or repair.
- For agents, reasoning streams into `reasoning_content`, tool calls
  parse in the eleven formats real checkpoints emit, and prompts are
  framed by the GGUF's own `tokenizer.chat_template`, which is compiled
  and evaluated.
- Mixture-of-experts models get GPU routing, indexed expert GEMMs and
  residency planning. Experts stream from the checkpoint when they do
  not fit in memory, and never when they do.
- Embeddings, reranking and scoring run on real encoder models. Point
  `-m` at a BGE, E5 or GTE checkpoint and `/v1/embeddings` serves it,
  pooled the way the file specifies, with no decoder hidden states
  standing in. A cross-encoder with a rank head answers `/v1/rerank`
  through that head; `/v1/score` accepts either kind and reports which
  one answered.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/frink/main/scripts/install.sh | bash
```

Installs `frink` and `frink-server` into `~/.local/bin` (override with
`FRINK_INSTALL_DIR`, pin with `FRINK_VERSION=v0.50.0`). The downloaded
`frink` is built with `serve`, so one binary runs completions and
serves the API. `frink-server` ships alongside it so an existing one on
your PATH keeps working. Prebuilts are macOS arm64 with Metal and Linux
x86_64 with CPU.

From crates.io or source instead:

```bash
# One binary, everything: completions, `frink serve`, `frink download`,
# bench, verify. Use --features cuda on Linux+NVIDIA.
cargo install frink-cli --features metal

# Or the server on its own, if you prefer two binaries.
cargo install frink-server --features metal

# From source.
cargo build --release -p frink-cli --features metal
```

The only feature flag to choose is the GPU backend: `--features metal`
on Apple silicon, `--features cuda` on Linux with an NVIDIA card, and
none on a CPU-only machine. `frink serve`, `frink download`,
`frink bench`, `frink verify` and completions are included by default.

Using frink as a Rust library rather than a command? Depend on
[`frink-inference`](https://crates.io/crates/frink-inference), or on
`frink-models` / `frink-core` for just the engine. None of them pull
in the CLI or the server.

## Quick start

```bash
# 0. Or skip step 1 entirely: -hf is llama.cpp's, and fetches on first use.
#    Most llama-server flags work as spelled: -c, --api-key, --alias, --jinja.
frink serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M -c 8192 --alias local

# 1. Get a model. No Python, no huggingface_hub: same syntax as `hf download`.
#    The `:QUANT` tag works here too and picks the file for you.
frink download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models

# 2. Run it. Frink evaluates the GGUF's own chat template and wraps your
#    prompt in it. Add --no-cnv for a raw completion.
frink -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -p "Explain quantization in two sentences" -n 128 -dev metal -ngl all

# 3. Or serve it on 127.0.0.1:8383 and point any OpenAI client at /v1.
#    On Metal, continuous batching is on by default, so several clients can
#    stream in parallel. `frink-server` is the same server standalone.
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -dev metal -ngl all &
curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'

# 4. Check it against llama.cpp on your own machine.
frink bench -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 --compare
```

On `Q8_0` and `IQ4_NL`, frink's logits match llama.cpp's. On K-quants
they drift, for a [known reason](docs/plans/llama-cpp-gap-inventory.md)
that is not a frink bug: llama.cpp quantizes activations to `Q8_K`
before the dot product, and frink keeps them in f32.

Despite the similar name, `IQ4_XS` drifts too. ggml declares
`vec_dot_type = Q8_K` for `IQ4_XS` and `Q8_0` for `IQ4_NL`, so
`frink parity` on an `IQ4_XS` checkpoint reports `DRIFT` and explains
why in its output. Use whichever quant you would use with llama.cpp.
If you want to compare the two engines, use `Q8_0`, which takes
activation quantization out of the comparison.
[docs/MODELS.md](docs/MODELS.md) lists what runs today and which
checkpoints stop with an error instead.

## Frink Studio

A web UI for the server: chat, the model inventory, live request
activity, and copy-pasteable connection snippets. It is a separate app
that talks to `frink-server` over the same public API as any other
client, so everything it does is something the API exposes.

<img src="docs/assets/studio-chat.webp" alt="Frink Studio chat" width="100%" />

```bash
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -ngl all &
cd ui && npm install && npm run dev      # http://localhost:5173
```

More screenshots and how it is put together: [`ui/README.md`](ui/README.md).

## Use it as a library

Published as [`frink-inference`](https://crates.io/crates/frink-inference),
a facade re-exporting the workspace. The name `frink` on crates.io
belongs to an unrelated crate.

```toml
[dependencies]
frink-inference = "0.50"
```

```rust
use frink_inference::gguf::ShardedGguf;
use frink_inference::models::{Decoder, ModelConfig};

let path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

// Read metadata without loading a single weight.
let file = ShardedGguf::open(path)?;
println!("{} tensors", file.tensor_count());

// Hyperparameters come from the file. Anything guessed is listed in
// `config.best_effort_fields`.
let config = ModelConfig::from_gguf(&file)?;
let decoder = Decoder::from_gguf(path, config)?;
```

Features, none on by default: `metal`, `cuda`, `api`. Every layer is
published separately if you want one rather than the stack:
[gguf](https://crates.io/crates/frink-gguf),
[quant](https://crates.io/crates/frink-quant),
[safetensors](https://crates.io/crates/frink-safetensors),
[core](https://crates.io/crates/frink-core),
[moe](https://crates.io/crates/frink-moe),
[models](https://crates.io/crates/frink-models),
[api](https://crates.io/crates/frink-api),
[metal](https://crates.io/crates/frink-metal),
[cuda](https://crates.io/crates/frink-cuda),
[vulkan](https://crates.io/crates/frink-vulkan) (an early Vulkan
backend, one kernel wide). All share one version.

## AI full disclosure

This software is developed with heavy assistance from Cursor, Grok 4.5,
GPT 5.6, and Claude Fable 5. Humans lead the ideas, the testing, and the
debugging. We say so up front because it shaped how the project was
built, and if you don't want AI-developed code, this software is not
for you. It also would not exist without
[llama.cpp](https://github.com/ggerganov/llama.cpp) and GGML, which
were largely written by hand.

## Acknowledgements to llama.cpp and GGML

Frink does not link against GGML, but it exists because of llama.cpp:
its kernels, its quantization formats, the GGUF ecosystem, and years of
engineering knowledge worked out in the open. We are thankful to
llama.cpp and its contributors. Their implementation, kernels, tests,
and design choices were our main reference while building this
pure-Rust GGUF / MoE inference path. Some source-level pieces are
retained or adapted here under the MIT license, notably the IQ
quantization codebook tables. Other pieces (GGUF layouts, quant/dot
semantics, CLI and server conventions) were written independently
against that public design. For that reason, and out of gratitude, we
keep the GGML authors' copyright notice in
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
