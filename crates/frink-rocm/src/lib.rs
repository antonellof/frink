//! frink-rocm: the ROCm/HIP beachhead (`docs/plans/amd.md`, phase R1).
//!
//! `frink-cuda`'s kernels are runtime-compiled text. This crate puts
//! that text behind a small HIP prelude ([`prelude`]) and compiles it
//! with `hiprtc` for AMD GPUs ([`rtc`]), the way llama.cpp's `ggml-hip`
//! compiles `ggml-cuda`'s sources; it adds no AMD kernels of its own.
//! [`targets`] says which architectures are served (RDNA, wave32) and
//! which are refused by name (CDNA, wave64). [`census`] is every kernel
//! the `hiprtc` gate compiles.
//!
//! What is verified, and how:
//!
//! - **Compiles for AMD**: every census kernel, for every served target,
//!   under ROCm's `hiprtc`, in CI (a ROCm container, no GPU) and by
//!   `cargo test -p frink-rocm --features rocm -- --ignored census`.
//! - **Runs on AMD**: NOT YET. [`hip`] launches the Q8_0 matvec against
//!   its scalar twin, and that test is `#[ignore]`d until it runs on an
//!   RDNA GPU (`-- --ignored gpu`). No frink model runs on ROCm until
//!   phase R2 wires this into `frink-core`.

pub mod census;
pub mod prelude;
pub mod targets;

#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "rocm")]
pub mod rtc;

pub use targets::{lookup, AmdArch, ArchRefused, Served};

/// What went wrong reaching ROCm.
#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    #[error("could not open {lib} (tried {tried}); install ROCm, or point {env} at it")]
    LibraryNotFound {
        lib: &'static str,
        tried: String,
        env: &'static str,
    },
    #[error("{lib} has no symbol {symbol}: {detail}")]
    Symbol {
        lib: &'static str,
        symbol: &'static str,
        detail: String,
    },
    #[error("hiprtc could not compile {name} for {gfx}: {log}")]
    Compile {
        name: String,
        gfx: String,
        log: String,
    },
    #[error("{call} failed: HIP error {code} ({message})")]
    Hip {
        call: &'static str,
        code: i32,
        message: String,
    },
    #[error(transparent)]
    Arch(#[from] ArchRefused),
}

/// Opens the first library that loads: the path in `env` when it is
/// set, else each candidate name in turn.
#[cfg(feature = "rocm")]
pub(crate) fn open_library(
    lib: &'static str,
    env: &'static str,
    candidates: &[&str],
) -> Result<libloading::Library, RocmError> {
    let mut tried: Vec<String> = Vec::new();
    let names: Vec<String> = match std::env::var(env) {
        Ok(path) if !path.is_empty() => vec![path],
        _ => candidates.iter().map(|s| s.to_string()).collect(),
    };
    for name in &names {
        // SAFETY: loading a shared library runs its initialisers; ROCm's
        // libraries are the intended, trusted target of this call.
        match unsafe { libloading::Library::new(name) } {
            Ok(l) => return Ok(l),
            Err(_) => tried.push(name.clone()),
        }
    }
    Err(RocmError::LibraryNotFound {
        lib,
        tried: tried.join(", "),
        env,
    })
}
