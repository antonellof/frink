//! The gated activation between a fused FFN's gate/up projections and
//! its down projection, on the device: SwiGLU, Gemma's GeGLU and
//! gpt-oss's clamped SwiGLU, one launch for all three
//! ([`glu_mul_device`]). Re-exported from [`crate::gpu`], where it lived.

use crate::gpu::{alloc_out, ensure_module_loaded, CudaError};

/// The gated activation a fused FFN applies to its gate before the
/// product with `up`: SwiGLU (`silu(gate) * up`), Gemma's GeGLU
/// (`gelu(gate) * up`), or gpt-oss's clamped SwiGLU ([`SWIGLU_OAI_SRC`]).
/// One launch serves all three ([`glu_mul_device`]), so they cannot
/// disagree about anything but the function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GluCuda {
    Silu,
    Gelu,
    /// `min(g, L) * sigmoid(A * min(g, L)) * (clamp(u, -L, L) + 1)` at
    /// [`SWIGLU_OAI_ALPHA`] and [`SWIGLU_OAI_LIMIT`].
    SwigluOai,
}

/// gpt-oss's sigmoid steepness and clamp, `llama-graph.cpp`'s
/// `LLM_FFN_SWIGLU_OAI_MOE` constants. The decoder holds them equal to
/// `frink_moe`'s, which its host path computes with.
pub const SWIGLU_OAI_ALPHA: f32 = 1.702;
pub const SWIGLU_OAI_LIMIT: f32 = 7.0;

impl GluCuda {
    fn module(self) -> (&'static str, &'static str, &'static str) {
        match self {
            GluCuda::Silu => (SILU_MUL_KERNEL_SRC, "frink_silu_mul", "silu_mul_f32"),
            GluCuda::Gelu => (GELU_MUL_KERNEL_SRC, "frink_gelu_mul", "gelu_mul_f32"),
            GluCuda::SwigluOai => (SWIGLU_OAI_SRC, "frink_swiglu_oai", "swiglu_oai_f32"),
        }
    }
}

/// gpt-oss's clamped SwiGLU, `frink_moe::swiglu_oai` (and ggml's
/// `swiglu_oai`) in its order: the gate clamped from above only, the
/// sigmoid steepened by `alpha`, and `up` clamped both ways with a `+1`.
pub const SWIGLU_OAI_SRC: &str = r#"
extern "C" __global__ void swiglu_oai_f32(
    const float* gate,
    const float* up,
    float* out,
    int n,
    float alpha,
    float limit
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = fminf(gate[i], limit);
        const float y = fmaxf(fminf(up[i], limit), -limit);
        const float out_glu = x / (1.0f + expf(alpha * -x));
        out[i] = out_glu * (y + 1.0f);
    }
}
"#;

/// CUDA C source for an elementwise SwiGLU pair fuse:
/// `out[i] = silu(gate[i]) * up[i]`, `silu(x) = x / (1 + exp(-x))`.
/// Mirrors `frink_core::matmul::swiglu` (which is `silu(gate)*up`) and
/// `frink-metal/src/elem.rs`'s `silu_mul_f32` kernel exactly, so the
/// fused dense-FFN path keeps activations on-device between the
/// gate/up matvecs and the down matvec instead of downloading two
/// vectors, computing SwiGLU on the host, and re-uploading.
pub const SILU_MUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void silu_mul_f32(
    const float* gate,
    const float* up,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}
"#;

/// CUDA C source for the GeGLU pair, `out[i] = gelu(gate[i]) * up[i]`,
/// in ggml-cuda's tanh form (`ggml_cuda_op_gelu_single`,
/// `ggml/src/ggml-cuda/unary.cuh`), which is what llama.cpp's CUDA
/// backend runs for Gemma's FFN. The host's `frink_core::matmul::geglu`
/// evaluates the same function through `x / (1 + exp(-2u))`, so the
/// two agree to rounding, not bit for bit.
pub const GELU_MUL_KERNEL_SRC: &str = r#"
extern "C" __global__ void gelu_mul_f32(
    const float* gate,
    const float* up,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float GELU_COEF_A = 0.044715f;
        const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        float g = gate[i];
        float gelu = 0.5f * g * (1.0f + tanhf(SQRT_2_OVER_PI * g * (1.0f + GELU_COEF_A * g * g)));
        out[i] = up[i] * gelu;
    }
}
"#;

/// Enqueues the elementwise `act(gate) * up` kernel on the shared
/// device's default stream (no host sync), returning the device output
/// slice. `gate`/`up` are device slices of length `n` produced by the
/// two FFN input projections; the result feeds straight into the down
/// projection without ever touching host memory.
pub(crate) fn glu_mul_device(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    glu: GluCuda,
    gate: &cudarc::driver::CudaSlice<f32>,
    up: &cudarc::driver::CudaSlice<f32>,
    n: usize,
) -> Result<cudarc::driver::CudaSlice<f32>, CudaError> {
    use cudarc::driver::LaunchAsync;

    let (src, module, name) = glu.module();
    ensure_module_loaded(dev, src, module, name)?;
    let func = dev.get_func(module, name).ok_or_else(|| {
        CudaError::KernelCompile(format!("function '{name}' not found after load_ptx"))
    })?;
    // SAFETY: the grid below is `ceil(n / 256)` blocks of 256 threads
    // and thread `i < n` writes `out[i]`.
    let mut d_out = unsafe { alloc_out(dev, n, name) }?;
    let block = 256u32;
    let grid = (n as u32).div_ceil(block);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the kernel reads `n` floats from `gate` and `up` and writes
    // `n` to `d_out`; the caller passes slices of length `n` and `d_out`
    // was just allocated at `n`. The grid covers `n` threads, each
    // bounds-checked against `n`. The clamped form takes its two
    // constants after `n`, matched positionally.
    unsafe {
        match glu {
            GluCuda::Silu | GluCuda::Gelu => func.launch(cfg, (gate, up, &mut d_out, n as i32)),
            GluCuda::SwigluOai => func.launch(
                cfg,
                (
                    gate,
                    up,
                    &mut d_out,
                    n as i32,
                    SWIGLU_OAI_ALPHA,
                    SWIGLU_OAI_LIMIT,
                ),
            ),
        }
        .map_err(|e| CudaError::Launch(format!("{name} launch: {e:?}")))?;
    }
    Ok(d_out)
}
