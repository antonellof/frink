# frink-vulkan

Decode matvecs on any Vulkan device (#409): every K-quant and legacy
kind, IQ4_NL, IQ4_XS and MXFP4, written in GLSL and committed as
SPIR-V. The codebook kinds (IQ1_S to IQ3_S), TQ1_0 and PTQ1_0 have no
shader yet; `docs/KERNELS.md` is the list. Weights are uploaded once
and stay resident. There is no GEMM (prefill runs on the host) and no
attention yet. `frink-core` reaches it behind its `vulkan` feature, and
only when `FRINK_VULKAN=1` is set.

It began as a GO/NO-GO beachhead; the verdict, its costs and what it did
not establish are in
[`docs/plans/archive/vulkan-beachhead-verdict.md`](../../docs/plans/archive/vulkan-beachhead-verdict.md).

```bash
cargo test -p frink-vulkan                      # SPIR-V half, no GPU needed
cargo run  -p frink-vulkan --features vulkan --example probe
cargo test -p frink-vulkan --features vulkan    # runs every shader on a real device
FRINK_REQUIRE_VULKAN=1 cargo test -p frink-vulkan --features vulkan  # a skip is a failure
```

The device tests are not `#[ignore]`d: they skip loudly when there is no
device, so they are free on a driverless machine and automatic
everywhere else. With `FRINK_REQUIRE_VULKAN=1`, as CI's lavapipe job
sets, a missing device or a missing `spirv-val` fails the test instead.

## Why a separate crate

`CLAUDE.md` says "no new crate for something one crate uses", and
`frink-core` is this crate's one caller (behind its optional `vulkan`
feature, in `weight_matrix/gpu_backend.rs`),
so the rule deserves an answer rather than a precedent.

- `frink-metal` and `frink-cuda` are both separate crates for the same
  reason: the backend's third-party dependency must be absent from the
  default build, and the backend's kernels must not enlarge the crate
  that dispatches to them. `ash` is optional here exactly as `cudarc` is
  there.
- The beachhead must not touch `frink-core`. `weight_matrix.rs` is the
  file `backend-seam-refactor` has to rewrite, and adding a third
  backend to it *before* that refactor is the copy-a-code-path-to-vary-it
  mistake `CLAUDE.md` names by name.
- Gates stay cheap. `cargo test -p frink-vulkan` is under a second; the
  same tests inside `frink-core` would ride behind that crate's whole
  suite.

If the verdict had been NO-GO, the honest disposition would be to delete
this crate rather than leave it as dead code. It is GO, and the crate is
the evidence, so it stays. It is published to crates.io because
`frink-core` depends on it: Cargo requires every dependency of a
published crate to be on the registry, optional ones included.

## Getting a Vulkan device on macOS

There is no Vulkan on a stock Mac. Three bottled Homebrew formulae are
enough, and after them the loader finds the ICD with no environment
variable:

```bash
brew install molten-vk vulkan-loader glslang spirv-tools
```

`glslang` rebuilds the SPIR-V (`shaders/build.sh`), and a test checks
the committed modules against it when its version matches
`spv/GENERATOR`. `spirv-tools` supplies `spirv-val`, which the shader
tests use as an outside authority on whether a module is legal SPIR-V.
When either is absent the test says so and passes.

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
| `src/dispatch.rs` | one typed entry point per kind, tested against its `frink_quant` scalar twin |

`shaders.rs` compiles and is tested unconditionally, with no Vulkan
driver present.

## Getting a Vulkan device on Linux without a GPU

Mesa's lavapipe is a CPU Vulkan driver, and CI uses it:

```bash
sudo apt-get install libvulkan1 mesa-vulkan-drivers spirv-tools glslang-tools
```
