# frink-vulkan

Matvecs and a prefill GEMM on any Vulkan device (#409): every K-quant and legacy
kind, every IQ kind, MXFP4, TQ1_0 and PTQ1_0, written in GLSL and
committed as SPIR-V; `docs/KERNELS.md` is the list. The codebook
kinds read their grids from one buffer filled from
`frink_quant::iq_tables`. Each matvec is also compiled as its
embedding gather (`runtime::gather`). Weights are uploaded once
and stay resident. A prefill batch is the same shader dispatched over the
batch (`Runtime::gemm`), and a decode token or a prefill batch of a
plain llama-family model runs 26 layers per submission, attention
included (`runtime::ops`). `frink-core` reaches it behind its `vulkan` feature, and
only with `--device vulkan` (or `FRINK_VULKAN=1`). The device is
llama.cpp's `Vulkan0`: the first discrete or integrated GPU, or the
first index in `GGML_VK_VISIBLE_DEVICES` (`src/select.rs`).

The beachhead verdict is in
[`vulkan-beachhead-verdict.md`](https://github.com/antonellof/frink/blob/75d2769d/bc5074aa:docs/plans/archive/vulkan-beachhead-verdict.md).

```bash
cargo test -p frink-vulkan                      # SPIR-V half, no GPU needed
cargo run  -p frink-vulkan --features vulkan --example probe
cargo test -p frink-vulkan --features vulkan    # runs every shader on a real device
FRINK_REQUIRE_VULKAN=1 cargo test -p frink-vulkan --features vulkan  # a skip is a failure
```

Device tests are not `#[ignore]`d: they skip loudly without a device.
With `FRINK_REQUIRE_VULKAN=1` (CI's lavapipe job), a missing device or
`spirv-val` fails instead.

## Why a separate crate

As with `frink-metal` and `frink-cuda`: the backend's dependency (`ash`)
stays out of the default build, its kernels do not enlarge
`frink-core`, and `cargo test -p frink-vulkan` stays under a second.
`frink-core` is its one caller (optional `vulkan` feature,
`weight_matrix/gpu_backend.rs`), which is why it is published:
Cargo requires every dependency of a published crate on the registry.

## Getting a Vulkan device on macOS

No Vulkan ships on a Mac; these formulae are enough, with no environment
variable:

```bash
brew install molten-vk vulkan-loader glslang spirv-tools
```

`glslang` rebuilds the SPIR-V (`shaders/build.sh`); a test checks the
committed modules against it when its version matches `spv/GENERATOR`.
`spirv-val` (from `spirv-tools`) validates every module. When either is
absent the test says so and passes.

If `libvulkan` is installed somewhere the dynamic linker does not look,
set `FRINK_VULKAN_LOADER` to its full path.

## Layout

| File | What |
|---|---|
| `shaders/*.comp`, `shaders/matvec_common.glsl` | GLSL source, one matvec per kind |
| `shaders/spv/` | committed SPIR-V, rebuilt by `shaders/build.sh`; `GENERATOR` pins the glslang version |
| `src/shaders.rs` | the one table (`MATVEC`); validated and drift-checked with no GPU |
| `src/device.rs` | `ash` loader, instance, device and compute queue, incl. MoltenVK portability |
| `src/runtime.rs` | `Runtime`: cached pipelines, resident weights, `matvec`, `matvec_many` |
| `src/runtime/ffn.rs` | the fused dense SwiGLU FFN, one submission |
| `src/runtime/ops.rs` | layer ops (RMSNorm, add, RoPE, f16 KV append, attention) recorded into a `Batch`, one submission |
| `src/dispatch.rs` | one typed entry point per kind, tested against its `frink_quant` scalar twin |

`shaders.rs` compiles and is tested unconditionally, with no Vulkan
driver present.

## Getting a Vulkan device on Linux without a GPU

Mesa's lavapipe is a CPU Vulkan driver, and CI uses it:

```bash
sudo apt-get install libvulkan1 mesa-vulkan-drivers spirv-tools glslang-tools
```
