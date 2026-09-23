//! Build-time facts the version line needs.
//!
//! `frink version` matches llama.cpp's shape, which names the commit
//! that produced the binary. Nothing in `std` can answer that at
//! runtime, so it is captured here.
//!
//! Absence is reported as absence: a build from a tarball with no git
//! metadata prints no commit rather than a placeholder that reads like
//! one.

use std::process::Command;

fn main() {
    // Re-run when HEAD moves, so the stamped commit is the one that
    // built the binary rather than whenever the cache was last cold.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");

    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    if let Some(sha) = sha {
        println!("cargo:rustc-env=FRINK_GIT_SHA={sha}");
    }
    println!(
        "cargo:rustc-env=FRINK_TARGET={}",
        std::env::var("TARGET").unwrap_or_else(|_| "unknown".into())
    );
}
