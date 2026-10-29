//! Which backend a test process computes on, said out loud.
//!
//! The backend is decided ONCE per process: `dense_enabled` caches the
//! `FRINK_METAL` read in a `OnceLock`, and with the variable unset it
//! asks the device probe. So a golden suite that never mentioned a
//! backend tested the CPU on a build without `--features metal` and
//! the GPU on a build with it, under the same name and the same
//! tolerance. On 2026-10-28 that was 163 failing goldens nobody saw,
//! because CI never builds this crate's tests with the feature.
//!
//! Every decoder the shared helpers build goes through [`backend`],
//! and so does every [`super::assert_close`]. The first call PINS the
//! process: `FRINK_GOLDEN_BACKEND=metal` asks for the deliberate Metal
//! golden mode, anything else (or nothing) is the CPU. The CPU pin is
//! `frink_core::weight_matrix::pin_host_backend`, which holds whatever
//! a test racing ahead of it had decided, so the order tests run in
//! cannot change which engine a golden checks.
//!
//! A CPU-pinned process on a Metal build also proves it encodes no
//! Metal work: `frink_metal::metal_encode_stats` counts every dispatch
//! the crate makes, and it must not move after the pin. That is what
//! caught the chunked delta rule launching on the GPU under
//! `FRINK_METAL=0`.

use std::sync::OnceLock;

/// The engine a test process runs its decoders on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The host path, which is the definition every device kernel is
    /// checked against.
    Cpu,
    /// Dense Metal matvecs plus the fused attention block, as `frink
    /// run --ngl 99` serves.
    Metal,
}

/// The pinned backend, and for the CPU the Metal dispatch count at the
/// moment of the pin.
static PINNED: OnceLock<(Backend, u64)> = OnceLock::new();

/// What this process was asked to test. Read once, by the first pin.
fn requested() -> Backend {
    match std::env::var("FRINK_GOLDEN_BACKEND").as_deref() {
        Err(_) | Ok("") | Ok("cpu") => Backend::Cpu,
        Ok("metal") => Backend::Metal,
        Ok(other) => panic!("FRINK_GOLDEN_BACKEND={other:?}: expected `cpu` or `metal`"),
    }
}

fn dispatches() -> u64 {
    #[cfg(feature = "metal")]
    return frink_metal::metal_encode_stats().dispatches;
    #[cfg(not(feature = "metal"))]
    0
}

/// Turn the pinned backend on. Runs once, inside the `OnceLock`.
fn apply(b: Backend) -> (Backend, u64) {
    match b {
        Backend::Cpu => {
            frink_core::weight_matrix::pin_host_backend();
            // The environment too, for any child process a test spawns.
            std::env::set_var("FRINK_METAL", "0");
            std::env::set_var("FRINK_METAL_ATTN", "0");
            std::env::set_var("FRINK_CUDA", "0");
            std::env::set_var("FRINK_VULKAN", "0");
        }
        Backend::Metal => {
            #[cfg(not(feature = "metal"))]
            panic!(
                "the Metal golden mode needs `--features metal`; without it every \
                 comparison would be the CPU's under the Metal tolerance"
            );
            #[cfg(feature = "metal")]
            {
                std::env::set_var("FRINK_METAL", "1");
                std::env::set_var("FRINK_METAL_ATTN", "1");
            }
        }
    }
    (b, dispatches())
}

/// The pinned backend must be the one that runs.
fn verify(b: Backend, dispatches_at_pin: u64) {
    #[cfg(feature = "metal")]
    {
        let dense = frink_core::weight_matrix::metal_dense_enabled();
        let attn = frink_metal::attn::metal_attn_enabled();
        match b {
            Backend::Cpu => {
                assert!(!dense, "pinned to the CPU, but Metal dense reports enabled");
                let n = dispatches() - dispatches_at_pin;
                assert_eq!(
                    n, 0,
                    "a CPU-pinned process encoded {n} Metal dispatches since the pin: \
                     some device path does not honour FRINK_METAL=0"
                );
            }
            Backend::Metal => assert!(
                dense && attn,
                "pinned to Metal, but this process decided dense={dense} attn={attn} \
                 before the pin; build the decoder through tests/common"
            ),
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = (b, dispatches_at_pin);
}

/// The backend this process tests, pinning the requested one on first
/// use and checking the pin still holds on every later one.
pub fn backend() -> Backend {
    let &(b, at_pin) = PINNED.get_or_init(|| apply(requested()));
    verify(b, at_pin);
    b
}

/// Pin `want` for this process, for a test that is ABOUT one backend
/// (a Metal fence test, the CPU child of a Metal comparison). Fails if
/// the process is already pinned to the other one: two backends in one
/// process is two answers under one name.
pub fn pin_backend(want: Backend) {
    let &(got, at_pin) = PINNED.get_or_init(|| apply(want));
    assert_eq!(
        got, want,
        "this process is already pinned to {got:?}; run the {want:?} test in its own \
         process (`--exact`)"
    );
    verify(got, at_pin);
}

/// The worst |Metal - llama.cpp| a last-position golden may show when a
/// whole suite runs under `FRINK_GOLDEN_BACKEND=metal`.
///
/// Measured (2026-10-28, M2 Pro, the 631 `metal-golden` comparisons the
/// graph suites print): p50 2.9e-5, p90 2.9e-3, p99 6.4e-2, continuous
/// to 6.4e-2 (`hrm_text` prefill), then nothing until 2.3e-1, where the
/// fixtures `decode_step_graphs`' `KNOWN_METAL_DIVERGENCES` lists begin.
/// The continuous part is f16 rounding: the prefill GEMM stages both
/// operands as `half` (prefill p50 6.4e-4) and the device KV is f16,
/// where the goldens were made with f32 KV.
///
/// That mode is a MEASUREMENT, not a gate: suites also assert host
/// properties (bit-identity between two host bodies, KL lines measured
/// on the CPU, sabotage distances smaller than f16 rounding) that do not
/// hold on a GPU and are not meant to. The Metal gate is
/// `decode_step_graphs::metal_decode_matches_cpu_and_llama_cpp`.
pub const METAL_GOLDEN_TOL: f32 = 0.1;

/// The tolerance a comparison runs at on this process's backend: the
/// suite's own on the CPU, never tighter than [`METAL_GOLDEN_TOL`] on
/// Metal.
pub fn tolerance(cpu_tol: f32) -> f32 {
    match backend() {
        Backend::Cpu => cpu_tol,
        Backend::Metal => cpu_tol.max(METAL_GOLDEN_TOL),
    }
}
