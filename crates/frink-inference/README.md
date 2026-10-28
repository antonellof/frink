# frink-inference

Facade crate for [Frink](https://github.com/antonellof/frink), a
pure-Rust GGUF / MoE inference engine: mmap loaders, quantized CPU +
Apple Metal + CUDA kernels, and an OpenAI-compatible server.

It contains no logic. It re-exports the workspace under one name, so a
dependent writes one line instead of six, and so the project is
findable on crates.io, since the name `frink` belongs to an unrelated
crate.

```toml
[dependencies]
frink-inference = "0.50"
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

They are not shipped from this crate on purpose: two crates installing
a binary called `frink` would fight over the same path in
`~/.cargo/bin`.

## Features

| Feature | Effect |
|---|---|
| `metal` | Apple Metal kernels. Apple Silicon only. |
| `cuda` | CUDA/NVRTC kernels. Builds without a CUDA toolkit; the driver and NVRTC libraries are loaded at run time. |
| `api` | Re-export `frink-api` (route constants + wire DTOs). |

Neither GPU feature is on by default. `metal` does not build off Apple
Silicon. `cuda` compiles anywhere, because `cudarc` opens the driver
and NVRTC with `dlopen`, but it only runs on a machine with an NVIDIA
GPU and driver.

CUDA timings against llama.cpp are published for an RTX 3090 and an
RTX 3060 in
[`benchmarks/RESULTS.md`](https://github.com/antonellof/frink/blob/main/benchmarks/RESULTS.md),
measured on rented hosts. See
[`docs/FEATURES.md`](https://github.com/antonellof/frink/blob/main/docs/FEATURES.md)
for what the CUDA backend covers.

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
| [`frink-vulkan`](https://crates.io/crates/frink-vulkan) | Early Vulkan backend: one Q8_0 compute shader, end to end |

The serving policy (prefix cache, scheduler, output parsers) lives in
`frink-server`'s `policy` module. It was a crate of its own,
`frink-edge`, which has been dissolved into `frink-core` and
`frink-server`.

Speed claims live in
[`benchmarks/RESULTS.md`](https://github.com/antonellof/frink/blob/main/benchmarks/RESULTS.md),
measured against llama.cpp on the same host and the same GGUF. That
table is generated from the raw timing files each run writes, and it
says so wherever nothing has been measured.

Apache-2.0.
