//! What the Metal and Vulkan decode gates share: the CPU child that
//! prints a fixture's host rows, and how a device decode is compared
//! with them. One copy, so the two gates cannot judge by different
//! rules.

use super::*;

/// The CPU's rows for `name`, or `None` for a fixture the loader
/// refuses (it never decodes, on any backend). `kv` is the child's host
/// KV dtype, `f32` or `f16`.
pub fn cpu_rows_from_child(name: &str, test: &str, kv: &str) -> Option<Vec<Vec<f32>>> {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["--include-ignored", "--exact", "--nocapture", test])
        .env("FRINK_DECODE_STEP_CHILD", name)
        .env("FRINK_DECODE_STEP_KV", kv)
        .env("FRINK_GOLDEN_BACKEND", "cpu")
        .env("FRINK_METAL", "0")
        .env("FRINK_METAL_ATTN", "0")
        .env("FRINK_VULKAN", "0")
        .output()
        .expect("spawning the CPU child");
    assert!(
        out.status.success(),
        "{name}: CPU child failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    if stdout.lines().any(|l| l.starts_with("FRINK-ROWS refused ")) {
        return None;
    }
    Some(
        stdout
            .lines()
            .filter_map(|l| l.strip_prefix("FRINK-ROWS logits "))
            .map(|l| l.split_whitespace().map(|w| w.parse().unwrap()).collect())
            .collect(),
    )
}

/// How one fixture's device decode compared with its CPU decode.
pub struct Agreement {
    /// Worst |device - CPU| over the steps where both decoded the
    /// same sequence.
    pub worst: f32,
    /// Steps compared before the sequences parted (or all of them).
    pub steps: usize,
    /// `Some(lead)` when the tokens parted: the CPU's top-1 lead over
    /// the token the device picked at that step.
    pub parted: Option<f32>,
}

pub fn agreement(gpu: &[Vec<f32>], cpu: &[Vec<f32>]) -> Agreement {
    let mut worst = 0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let delta = common::worst_vs(g, c);
        worst = if delta.is_nan() {
            f32::NAN
        } else {
            worst.max(delta)
        };
        let (picked, want) = (argmax(g), argmax(c));
        if picked != want {
            return Agreement {
                worst,
                steps: i + 1,
                parted: Some(c[want] - c[picked]),
            };
        }
    }
    Agreement {
        worst,
        steps: gpu.len(),
        parted: None,
    }
}
