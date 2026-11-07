//! The CUDA spellings HIP's runtime compiler does not have.
//!
//! `frink-cuda`'s kernels are written against NVRTC. `hiprtc` takes the
//! same C++ and supplies the HIP runtime header implicitly, so almost
//! everything compiles as written: `__syncthreads`, `__int_as_float`,
//! `ldexpf`, `extern "C" __global__`, templates and `#pragma unroll`.
//! What it lacks, measured by compiling the census under ROCm 6.4's
//! `hiprtc` (`error: use of undeclared identifier '__shfl_down_sync';
//! did you mean '__shfl_down'?`), is the `_sync` family of warp
//! shuffles. They are mapped the way llama.cpp maps them
//! (`ggml/src/ggml-cuda/vendors/hip.h:32-34`): the mask is dropped,
//! because a HIP shuffle is over the whole wavefront, which on the RDNA
//! targets this crate serves is the 32 lanes the kernels assume.

/// Prepended to every `frink-cuda` translation unit compiled for AMD.
pub const HIP_PRELUDE: &str = r#"
// frink-rocm prelude (crates/frink-rocm/src/prelude.rs).
#define __shfl_sync(mask, var, lane, ...) __shfl(var, lane, ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, delta, ...) __shfl_up(var, delta, ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down(var, delta, ##__VA_ARGS__)
#define __shfl_xor_sync(mask, var, lane, ...) __shfl_xor(var, lane, ##__VA_ARGS__)
#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif
"#;

/// `src`, a `frink-cuda` translation unit, as `hiprtc` should see it.
pub fn hip_translation_unit(src: &str) -> String {
    format!("{HIP_PRELUDE}{src}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `_sync` shuffle a kernel in the census spells has its
    /// mapping, so a kernel cannot reach `hiprtc` with an unmapped one.
    #[test]
    fn every_sync_shuffle_the_census_uses_is_mapped() {
        for k in crate::census::kernels() {
            for name in [
                "__shfl_sync",
                "__shfl_up_sync",
                "__shfl_down_sync",
                "__shfl_xor_sync",
            ] {
                if k.source.contains(&format!("{name}(")) {
                    assert!(
                        HIP_PRELUDE.contains(&format!("#define {name}(")),
                        "{} uses {name}, which the prelude does not map",
                        k.name
                    );
                }
            }
        }
    }

    #[test]
    fn the_prelude_comes_first() {
        let tu = hip_translation_unit("extern \"C\" __global__ void k() {}");
        assert!(tu.starts_with(HIP_PRELUDE));
        assert!(tu.ends_with("void k() {}"));
    }
}
