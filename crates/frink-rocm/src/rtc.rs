//! `hiprtc`: compiling a `frink-cuda` translation unit to an AMD code
//! object, through ROCm's runtime compiler opened with `dlopen`.

use std::ffi::{c_char, c_int, c_void, CString};

use libloading::Library;

use crate::prelude::hip_translation_unit;
use crate::RocmError;

type Program = *mut c_void;
type CreateFn = unsafe extern "C" fn(
    *mut Program,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> c_int;
type CompileFn = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int;
type SizeFn = unsafe extern "C" fn(Program, *mut usize) -> c_int;
type BytesFn = unsafe extern "C" fn(Program, *mut c_char) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut Program) -> c_int;

const LIB: &str = "libhiprtc";
const ENV: &str = "FRINK_HIPRTC_LIB";
const CANDIDATES: &[&str] = &[
    "libhiprtc.so.6",
    "libhiprtc.so.7",
    "libhiprtc.so",
    "/opt/rocm/lib/libhiprtc.so.6",
    "/opt/rocm/lib/libhiprtc.so",
    "hiprtc0605.dll",
    "hiprtc0604.dll",
    "hiprtc0603.dll",
];

/// ROCm's runtime compiler.
pub struct Hiprtc {
    lib: Library,
}

fn sym<T: Copy>(lib: &Library, name: &'static str) -> Result<T, RocmError> {
    let cname = format!("{name}\0");
    // SAFETY: `T` is the C signature ROCm's `hip/hiprtc.h` declares for
    // `name`; the symbol lives as long as `lib`, which outlives every
    // call made through the copy returned here (both are owned by the
    // same `Hiprtc`).
    unsafe {
        lib.get::<T>(cname.as_bytes())
            .map(|s| *s)
            .map_err(|e| RocmError::Symbol {
                lib: LIB,
                symbol: name,
                detail: e.to_string(),
            })
    }
}

impl Hiprtc {
    /// Opens `libhiprtc` (`FRINK_HIPRTC_LIB` overrides the search).
    pub fn open() -> Result<Self, RocmError> {
        Ok(Self {
            lib: crate::open_library(LIB, ENV, CANDIDATES)?,
        })
    }

    /// Compiles `src` (a `frink-cuda` translation unit, without the
    /// prelude) for `gfx` and returns the code object `hipModuleLoadData`
    /// takes.
    pub fn compile(&self, name: &str, src: &str, gfx: &str) -> Result<Vec<u8>, RocmError> {
        let create: CreateFn = sym(&self.lib, "hiprtcCreateProgram")?;
        let compile: CompileFn = sym(&self.lib, "hiprtcCompileProgram")?;
        let log_size: SizeFn = sym(&self.lib, "hiprtcGetProgramLogSize")?;
        let log: BytesFn = sym(&self.lib, "hiprtcGetProgramLog")?;
        let code_size: SizeFn = sym(&self.lib, "hiprtcGetCodeSize")?;
        let code: BytesFn = sym(&self.lib, "hiprtcGetCode")?;
        let destroy: DestroyFn = sym(&self.lib, "hiprtcDestroyProgram")?;

        let tu = CString::new(hip_translation_unit(src)).expect("kernel text has no NUL");
        let file = CString::new(format!("{}.cu", name.replace(' ', "_"))).expect("no NUL");
        let opts: Vec<CString> = [format!("--offload-arch={gfx}"), "-O3".to_string()]
            .into_iter()
            .map(|o| CString::new(o).expect("no NUL"))
            .collect();
        let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();

        let mut prog: Program = std::ptr::null_mut();
        // SAFETY: every pointer is a live, NUL-terminated C string owned
        // above; zero headers means the two header arrays are not read.
        let rc = unsafe {
            create(
                &mut prog,
                tu.as_ptr(),
                file.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if rc != 0 {
            return Err(RocmError::Compile {
                name: name.into(),
                gfx: gfx.into(),
                log: format!("hiprtcCreateProgram returned {rc}"),
            });
        }
        // SAFETY: `prog` was created above; `opt_ptrs` holds
        // `opt_ptrs.len()` live C strings.
        let rc = unsafe { compile(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr()) };
        let result = if rc == 0 {
            let mut n = 0usize;
            // SAFETY: `prog` compiled successfully; `buf` is sized to the
            // length hiprtc reports before it writes that many bytes.
            unsafe {
                code_size(prog, &mut n);
                let mut buf = vec![0u8; n];
                code(prog, buf.as_mut_ptr() as *mut c_char);
                Ok(buf)
            }
        } else {
            let mut n = 0usize;
            // SAFETY: as above, for the compile log.
            let text = unsafe {
                log_size(prog, &mut n);
                let mut buf = vec![0u8; n.max(1)];
                log(prog, buf.as_mut_ptr() as *mut c_char);
                String::from_utf8_lossy(&buf)
                    .trim_end_matches('\0')
                    .to_string()
            };
            Err(RocmError::Compile {
                name: name.into(),
                gfx: gfx.into(),
                log: text,
            })
        };
        // SAFETY: `prog` is destroyed exactly once, after its last use.
        unsafe { destroy(&mut prog) };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::{kernels, Kernel};

    /// The kernels `frink-cuda` keeps behind its `cuda` feature, which
    /// this test sees through the dev-dependency that enables it.
    fn feature_gated() -> Vec<Kernel> {
        let k = |name: &str, src: &str| Kernel::new(name, src);
        use frink_cuda::prefill::kernels as p;
        vec![
            k("gqa_decode", frink_cuda::attn::GQA_DECODE_KERNEL_SRC),
            k("silu_mul", frink_cuda::gpu::SILU_MUL_KERNEL_SRC),
            k("swiglu_oai", frink_cuda::gpu::SWIGLU_OAI_SRC),
            k(
                "fused_add_rmsnorm",
                frink_cuda::gpu::FUSED_ADD_RMSNORM_KERNEL_SRC,
            ),
            k("prefill rmsnorm_rows", p::RMSNORM_ROWS_KERNEL_SRC),
            k("prefill add_bias_rows", p::ADD_BIAS_ROWS_KERNEL_SRC),
            k("prefill add_rows", p::ADD_ROWS_KERNEL_SRC),
            k("prefill rope_rows", p::ROPE_ROWS_KERNEL_SRC),
            k("prefill causal_gqa", p::CAUSAL_GQA_PREFILL_KERNEL_SRC),
            k("prefill add_bias_tab_rows", p::ADD_BIAS_TAB_ROWS_KERNEL_SRC),
            k("prefill moe_topk", p::MOE_TOPK_KERNEL_SRC),
            k("prefill gather_ptrs", p::GATHER_PTRS_KERNEL_SRC),
            k("prefill moe_group", p::MOE_GROUP_KERNEL_SRC),
        ]
    }

    /// **The R1 gate.** Every `frink-cuda` kernel frink would run on an
    /// AMD GPU compiles under ROCm's `hiprtc` for every served target.
    /// Needs `libhiprtc` and no GPU: CI runs it in a ROCm container.
    /// Reports every failure, not the first, so one run is the list.
    #[test]
    #[ignore = "needs libhiprtc (ROCm); no GPU. Run: cargo test -p frink-rocm --features rocm -- --ignored census"]
    fn census_every_kernel_compiles_for_every_served_target() {
        let rtc = Hiprtc::open().expect("libhiprtc");
        let mut all = kernels();
        all.extend(feature_gated());
        let mut failures = Vec::new();
        let mut compiled = 0usize;
        for arch in crate::targets::served() {
            for k in &all {
                match rtc.compile(&k.name, &k.source, arch.gfx) {
                    Ok(code) => {
                        assert!(
                            !code.is_empty(),
                            "{} for {}: empty code object",
                            k.name,
                            arch.gfx
                        );
                        compiled += 1;
                    }
                    Err(e) => failures.push(e.to_string()),
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} compiles failed:\n{}",
            failures.len(),
            compiled + failures.len(),
            failures.join("\n\n")
        );
        eprintln!("census: {compiled} kernels compiled across the served targets");
    }

    /// The prelude is load-bearing: a `_sync` shuffle compiles with it
    /// and not without it, so a refactor that dropped it fails here.
    #[test]
    #[ignore = "needs libhiprtc (ROCm); no GPU. Run: cargo test -p frink-rocm --features rocm -- --ignored census"]
    fn census_the_prelude_is_what_makes_a_sync_shuffle_compile() {
        let rtc = Hiprtc::open().expect("libhiprtc");
        let src = r#"extern "C" __global__ void k(float* v) {
            float x = v[threadIdx.x];
            x += __shfl_down_sync(0xffffffff, x, 1);
            v[threadIdx.x] = x;
        }"#;
        assert!(rtc.compile("with_prelude", src, "gfx1100").is_ok());
        let bare = Hiprtc::open().expect("libhiprtc");
        let err = bare
            .compile_without_prelude("no_prelude", src, "gfx1100")
            .unwrap_err();
        assert!(err.to_string().contains("__shfl_down_sync"), "{err}");
    }

    impl Hiprtc {
        /// Test-only: compile without [`crate::prelude::HIP_PRELUDE`].
        fn compile_without_prelude(
            &self,
            name: &str,
            src: &str,
            gfx: &str,
        ) -> Result<Vec<u8>, RocmError> {
            // The prelude is `#define`s only, so undefining them after it
            // is the same translation unit as compiling without it.
            let undo = "#undef __shfl_sync\n#undef __shfl_up_sync\n#undef __shfl_down_sync\n#undef __shfl_xor_sync\n";
            self.compile(name, &format!("{undo}{src}"), gfx)
        }
    }
}
