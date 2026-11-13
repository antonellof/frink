//! `frink parity` compares the arithmetic `frink run` performs (#584).
//!
//! It used to leave the integer-dot tier at the library default, off,
//! while `frink run` and the server turn it on wherever it wins. So on
//! an aarch64 host every parity verdict was about f32 activations
//! against dequantized weights, a path no user takes, and on a pure
//! Q2_K file that path is 1.9e-2 relative away from llama.cpp's Q8_K
//! one at the first V projection. The report also said frink quantized
//! activations "the same, bit for bit", which it did not.
//!
//! Each test runs the binary in its own process, because the tier is
//! decided once per process. The dumper is `/usr/bin/false`: frink's
//! prefill runs and reports its activations before the reference is
//! asked anything, and the run then fails on the reference, which is
//! not what these tests are about.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../frink-models/tests/fixtures/llama_biases_tiny.gguf")
}

/// `frink parity` on the fixture with `FRINK_CPU_INT_DOT` set to
/// `int_dot` (unset when `None`); its stdout and stderr.
fn parity_output(int_dot: Option<&str>) -> Option<String> {
    let dumper = Path::new("/usr/bin/false");
    if !dumper.exists() {
        eprintln!("no /usr/bin/false on this host; skipping");
        return None;
    }
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_frink"));
    cmd.args([
        "parity",
        "--allow-multiple-instances",
        "--tokens",
        "1,2,3",
        "-m",
    ])
    .arg(fixture())
    .arg("--dumper")
    .arg(dumper)
    .env("FRINK_METAL", "0")
    .env("FRINK_CUDA", "0")
    .env("FRINK_VULKAN", "0")
    .env_remove("FRINK_CPU_INT_DOT");
    if let Some(v) = int_dot {
        cmd.env("FRINK_CPU_INT_DOT", v);
    }
    let out = cmd.output().expect("running frink parity");
    Some(format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    ))
}

/// On aarch64 `frink run` takes the int-dot matvec (decode) on every
/// host, so parity must too. x86 keeps the matvec off in `frink run`
/// (measured to lose there), so the assertion is aarch64's.
#[cfg(target_arch = "aarch64")]
#[test]
fn parity_takes_the_int_dot_tier_frink_run_takes() {
    let Some(out) = parity_output(None) else {
        return;
    };
    assert!(
        out.contains("frink activations prefill") && out.contains("decode int8"),
        "parity's frink side must quantize decode activations as `frink run` does:\n{out}"
    );
}

/// `FRINK_CPU_INT_DOT=0` still selects f32, on every host.
#[test]
fn parity_honours_an_explicit_int_dot_off() {
    let Some(out) = parity_output(Some("0")) else {
        return;
    };
    assert!(
        out.contains("frink activations prefill f32, decode f32"),
        "FRINK_CPU_INT_DOT=0 must leave parity on f32 activations:\n{out}"
    );
}
