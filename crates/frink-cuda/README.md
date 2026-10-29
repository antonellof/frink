# frink-cuda

CUDA/NVRTC kernels for Frink.

Capability detection always builds. Real CUDA dispatch sits behind
`--features cuda`. Loading is dynamic, through `cudarc`, so no toolkit
is needed to compile; the driver and NVRTC libraries are opened at run
time. Run the hardware tests only on a machine with a
GPU.

`mul_mm`, the batched quantized GEMM, is wired into batched prefill:
`frink-core`'s `weight_matrix.rs` calls
`mul_mm_launch::launch_mul_mm` when the batch is wide enough. On
`sm_80` and newer it runs the tensor-core body (`mul_mm_tc`), and on
older devices or with `FRINK_CUDA_MUL_MM=simt` the SIMT body.
The CHANGELOG entry for 0.22.0 records it running on an RTX 3090. It ships with a scalar twin (`mul_mm_ref`) that the default
`cargo test -p frink-cuda` exercises, and with
`tools/mul_mm_host_check/run.sh`, which compiles the emitted CUDA C
against a barrier shim and runs it on this CPU to check it against that
twin. The decode matvec kernels for Q4_K, Q5_K, Q6_K and Q8_0
(`matvec_kinds/coalesced.rs`) have the same pair: twins in
`coalesced_twin.rs` and `tools/matvec_host_check/run.sh`, which runs
both widths (`FRINK_CUDA_MATVEC`, see `docs/CONFIG.md`) against them.
Neither of those is a measurement. A kernel change merges only
after `cargo test -p frink-cuda --features cuda -- --ignored` has run
on a real device.
