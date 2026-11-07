# frink-rocm

The ROCm/HIP beachhead for frink (`docs/plans/amd.md`, phase R1).

frink's CUDA kernels are runtime-compiled text. This crate puts that
text behind a small HIP prelude and compiles it with `hiprtc` for AMD
GPUs, the way llama.cpp's `ggml-hip` compiles `ggml-cuda`'s sources. It
adds no AMD kernels of its own.

- `prelude`: the CUDA spellings HIP's runtime compiler lacks.
- `targets`: the AMD architectures served (RDNA, wave32) and refused
  (CDNA, wave64, until each 32-lane assumption is made width-aware).
- `census`: every `frink-cuda` kernel that is compiled for AMD.
- `rtc` and `hip` (feature `rocm`): `hiprtc` and the HIP runtime,
  opened with `dlopen`, and a Q8_0 matvec launch checked against its
  scalar twin.

```bash
cargo test -p frink-rocm                                  # prelude, targets, census
cargo test -p frink-rocm --features rocm -- --ignored census   # needs libhiprtc (no GPU)
cargo test -p frink-rocm --features rocm -- --ignored gpu      # needs an AMD GPU
```
