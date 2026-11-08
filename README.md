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
quantized kernels, attention and expert routing are all written here,
in Rust.

- **One binary, no Python.** On macOS arm64 it is 23 MB, with Metal and
  the server built in: completions, the API server, `download`, `bench`,
  `batched-bench`, `quantize`, `imatrix`, `gguf-split` and `verify`.
  There are no wheels to install and no CUDA userspace to match against
  a driver.
- **Weights stay quantized.** They are memory-mapped and dequantized
  inside the matmul, so an 8B model fits on a laptop. K-quants, the IQ
  tiers, MXFP4, F16 and BF16 all load.
- **A drop-in for llama.cpp.** The flags, the sampler order and the
  tools are the same. `quantize` writes Q8_0 and the K-quants
  byte-identically to `llama-quantize`, with or without an importance
  matrix. Tokenization is checked against libllama on twenty
  checkpoints, and 101 architectures run with a logit comparison
  behind them. Any other architecture stops with an error that says
  what is missing.
- **llama-server's chat prompts.** The GGUF's own Jinja chat template
  renders with the variables llama-server passes (`enable_thinking`,
  today's date, `preserve_reasoning`), so a thinking model thinks by
  default and earlier turns keep their reasoning. CI renders every
  template llama.cpp bundles through both engines and lists each prompt
  that still differs. `--chat-template`, `--chat-template-kwargs`,
  `-rea` and `--reasoning-effort` are accepted with llama-server's spelling.
- **An OpenAI-compatible server.** Point your existing client at it.
  It also serves the Anthropic and Responses APIs, with paged KV shared
  across conversations, model swap, sleep and wake, slot save and
  restore, and resumable streams. On Metal, concurrent clients share
  one batched decode worker. [Measured](benchmarks/RESULTS.md#serving-features-measured),
  that is 1.28x aggregate throughput at eight concurrent requests, and
  736 of 757 prompt tokens reused on a shared system prompt.
- **The full sampling API.** `n` and `best_of` from one prefill,
  `logprobs` and `prompt_logprobs`, `logit_bias`, `allowed_token_ids`,
  `bad_words`, `echo`, `truncate_prompt_tokens` and `cache_salt`. A
  field the server does not implement is refused by name, never
  silently dropped.
- **Structured output, enforced per token.** A GBNF grammar, a JSON
  schema or a forced `tool_choice` (in ten of the eleven tool-call
  formats) masks every token that would break it, so there is nothing
  to retry or repair.
- **Built for agents.** Reasoning streams into `reasoning_content`, and
  tool calls parse in the eleven formats real checkpoints emit.
- **Lossless speculative decoding.** A draft model (`-md`, as in
  llama.cpp) proposes tokens and the target checks them with the
  speculative-sampling rejection rule, so the text is exactly the
  target's at any temperature. The server takes `-md` too, or
  speculates on its own with prompt lookup, and reports what that saved
  in `usage`, on its private decode loop. Inside the continuous batch it
  does not yet.
- **Mixture of experts.** GPU routing, indexed expert GEMMs and
  residency planning. Experts stream from the checkpoint only when they
  do not fit in memory.
- **Embeddings, reranking and scoring** on real encoder models. A BGE,
  E5 or GTE checkpoint serves `/v1/embeddings`, pooled the way the file
  says. A cross-encoder answers `/v1/rerank` through its rank head, and
  `/v1/score` takes either.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/antonellof/frink/main/scripts/install.sh | bash
```

Installs `frink` and `frink-server` into `~/.local/bin` (override with
`FRINK_INSTALL_DIR`, pin with `FRINK_VERSION=v0.58.0`). The downloaded
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

Every command here is spelled as it is in llama.cpp. Frink uses the GPU
it was built for by default. Add `-ngl 0` to run on the CPU.

### Run a model

`-hf` fetches the GGUF from Hugging Face on first use. The prompt is
wrapped in the model's own chat template. Add `--no-cnv` for a raw
completion.

```bash
frink -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
  -p "Explain quantization in two sentences" -n 128
```

To keep the file in a folder of your own, download it first. The syntax
is `hf download`'s, and needs no Python:

```bash
frink download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
frink -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p "Hi" -n 64
```

### Serve it

The server listens on `127.0.0.1:8383`. Point any OpenAI client at
`/v1`. Most llama-server flags work as spelled (`-c`, `--api-key`,
`--alias`, `--jinja`). `frink-server` is the same server as its own
binary.

```bash
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 8192 --alias local &

curl -s http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64}'
```

### Compare it with llama.cpp

`--compare` also runs `llama-bench` on the same file and prints the
gap:

```bash
frink bench -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -p 512 -n 128 -r 3 --compare
```

On `Q8_0` and `IQ4_NL` frink's logits match llama.cpp's. On the
K-quants and `IQ4_XS` they drift, for a
[known reason](docs/plans/archive/llama-cpp-gap-inventory.md) that is not a
frink bug: llama.cpp quantizes the activations to `Q8_K` before those
dot products, and frink keeps them in f32. `frink parity` says so when
it reports `DRIFT`. To compare the two engines, use `Q8_0`.
[docs/MODELS.md](docs/MODELS.md) lists what runs today, and which
checkpoints stop with an error instead.

### Speculative decoding

A 1B drafter from the same family proposes and the 3B target verifies,
so the output is the target's own. The drafter runs on the CPU, so the
whole run does (`-dev cpu`).

```bash
frink download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir models
frink -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -md models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --draft-max 5 \
  -p "Write the numbers from 1 to 30 separated by commas:" \
  -n 96 --temp 0 --no-cnv -dev cpu
```

The run ends with lines like these, from one CPU run of that command.
The token rate is left out, because one laptop run is not a benchmark.

```
frink: predict 96 tokens, … t/s over 18 verification steps
frink: acceptance length 5.33 tokens/step, accepted 77 of 81 drafted
frink: accept rate per position [0.89, 1.00, 1.00, 0.94, 0.93]
```

Run the same command without `-md` to see whether the drafter helps on
your text. [docs/CLI.md](docs/CLI.md) covers `--draft-p-min` and the
cases it refuses: grammar-constrained runs, recurrent models, and a
drafter placed on the GPU.

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
frink-inference = "0.58"
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
[vulkan](https://crates.io/crates/frink-vulkan) (early and opt-in:
decode matvecs only). All share one version.

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
