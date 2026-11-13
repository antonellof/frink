# frink-inference

Facade crate for [Frink](https://github.com/antonellof/frink), a
pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Apple Metal + CUDA kernels, and an OpenAI-compatible server.

It contains no logic: it re-exports the workspace under one name (the
`frink` crate on crates.io is unrelated).

```toml
[dependencies]
frink-inference = "0.63"
```

```rust
use frink_inference::gguf::ShardedGguf;

let file = ShardedGguf::open("model.gguf")?;
```

## The binaries are elsewhere

```bash
cargo install frink-cli      # installs the `frink` binary
cargo install frink-server   # OpenAI-compatible HTTP server
```

Not from this crate, so two crates never install a `frink` binary to
the same path.

## Features

| Feature | Effect |
|---|---|
| `metal` | Apple Metal kernels. Apple Silicon only. |
| `cuda` | CUDA/NVRTC kernels. Builds without a CUDA toolkit; the driver and NVRTC libraries are loaded at run time. |
| `api` | Re-export `frink-api` (route constants + wire DTOs). |

Neither GPU feature is on by default. `cuda` compiles anywhere
(`cudarc` opens the driver with `dlopen`) but runs only with an NVIDIA
GPU and driver. What each backend covers is in
[`docs/FEATURES.md`](https://github.com/antonellof/frink/blob/main/docs/FEATURES.md).

## The rest of the workspace

| Crate | What it is |
|---|---|
| [`frink-gguf`](https://crates.io/crates/frink-gguf) | GGUF mmap reader, sharded checkpoints |
| [`frink-quant`](https://crates.io/crates/frink-quant) | Block layouts, fused dequant+dot |
| [`frink-safetensors`](https://crates.io/crates/frink-safetensors) | SafeTensors mmap reader |
| [`frink-core`](https://crates.io/crates/frink-core) | Tensor ops, RoPE, GQA, KV cache, MoE expert residency |
| [`frink-moe`](https://crates.io/crates/frink-moe) | Expert routing and dispatch |
| [`frink-models`](https://crates.io/crates/frink-models) | Loaders and decoder stacks |
| [`frink-api`](https://crates.io/crates/frink-api) | Route constants + wire DTOs |
| [`frink-metal`](https://crates.io/crates/frink-metal) | Apple Metal kernels |
| [`frink-cuda`](https://crates.io/crates/frink-cuda) | CUDA/NVRTC kernels |
| [`frink-vulkan`](https://crates.io/crates/frink-vulkan) | Early, opt-in Vulkan backend |

The serving policy (prefix cache, scheduler, output parsers) lives in
`frink-server`'s `policy` module. Speed against llama.cpp, same host and
GGUF, is in
[`benchmarks/RESULTS.md`](https://github.com/antonellof/frink/blob/main/benchmarks/RESULTS.md).

Apache-2.0.
