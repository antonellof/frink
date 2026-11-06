//! The envelope every timed engine run shares: what is checked before
//! the clock starts, what is reported after it stops, and what a
//! receipt must carry to be worth reading later.
//!
//! `frink bench` and `frink batched-bench` measure different
//! workloads through different engine seams, but the measurement
//! contract around them is ONE thing: a quiet, cool host with the
//! weights in RAM; the load and thermal state recorded on both sides
//! of the run; a receipt whose backend label is the backend that ran
//! and whose GPU row names its card. Each of those rules exists
//! because it was broken once and published (#126 is the label one).
//! Having them here, as functions both tools call, is what keeps the
//! second tool from drifting away from the first the way copied code
//! does in this repo.

use crate::bench_guard;
use crate::host_state::{self, ThermalReading};
use std::path::Path;

/// What the host was doing around this run, for the receipt.
#[derive(Clone, Copy)]
pub struct HostState {
    pub load_start: Option<f64>,
    pub load_end: Option<f64>,
    pub thermal_start: ThermalReading,
    pub thermal_end: ThermalReading,
}

/// The checks that run BEFORE anything is loaded: a timed run on a
/// busy or hot host produces a number that looks like a measurement
/// and is not one.
///
/// `--max-load 0` is the documented "measure anyway, not publishable"
/// escape; the thermal bar shares it rather than growing a second flag
/// that has to be remembered separately.
///
/// Returns the starting load average and thermal reading so the
/// after-run report can show the delta.
pub fn preflight_host(max_load: f64) -> anyhow::Result<(Option<f64>, ThermalReading)> {
    let load_start = host_state::ensure_quiet_enough(max_load)?;
    let thermal_start = host_state::thermal_reading();
    host_state::ensure_cool_enough(&thermal_start, max_load > 0.0)?;
    Ok((load_start, thermal_start))
}

/// A busy host and a hot host are refused by [`preflight_host`]. A FULL
/// host is the same failure with a different cause: the weights page
/// to disk and the run times the page file. The file size is the floor
/// on the footprint, `extra_gb` is whatever the workload will allocate
/// on top of it (KV caches, for a run that knows its context up
/// front), and `--max-load 0` waives this the same way it waives the
/// other two.
pub fn ensure_weights_fit(max_load: f64, weights: &Path, extra_gb: f64) -> anyhow::Result<()> {
    if max_load <= 0.0 {
        return Ok(());
    }
    if let Ok(meta) = std::fs::metadata(weights) {
        let weights_gb = meta.len() as f64 / 1024.0 / 1024.0 / 1024.0;
        host_state::ensure_fits_in_ram(weights_gb + extra_gb, 2.0)?;
    }
    Ok(())
}

/// What the host looked like when the run finished, plus the engine
/// knobs that were in effect for it.
pub struct HostAfter {
    pub load_end: Option<f64>,
    pub thermal_end: ThermalReading,
    /// Non-default `FRINK_*` variables, see
    /// [`bench_guard::nondefault_engine_env`].
    pub engine_env: Vec<(String, String)>,
}

/// Reads the host again after the run and prints the before/after pair
/// to stderr under `tool`'s name, warning when the run started cool and
/// finished thermally limited -- its later repetitions ran under
/// different physics than its first ones.
pub fn report_host_after(
    tool: &str,
    load_start: Option<f64>,
    thermal_start: &ThermalReading,
) -> HostAfter {
    let load_end = host_state::load_average_1min();
    let thermal_end = host_state::thermal_reading();
    eprintln!(
        "{tool}: host 1-min load {} -> {}, {} -> {}",
        fmt_load(load_start),
        fmt_load(load_end),
        thermal_start.describe(),
        thermal_end.describe(),
    );
    if !thermal_start.is_degraded() && thermal_end.is_degraded() {
        eprintln!(
            "{tool}: WARNING -- the host became thermally limited during this \
             run ({}); the later repetitions did not run under the same conditions \
             as the first",
            thermal_end.describe()
        );
    }
    let engine_env = bench_guard::nondefault_engine_env(std::env::vars());
    if !engine_env.is_empty() {
        eprintln!(
            "{tool}: non-default engine env in effect: {}",
            engine_env
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    HostAfter {
        load_end,
        thermal_end,
        engine_env,
    }
}

/// Serializes a thermal reading so that "we did not measure" and
/// "we measured, and it was nominal" can never be read as the same
/// thing. `measured` is stated outright rather than left to be
/// inferred from a null, because a field that is always null while
/// implying it was measured is exactly the lie this receipt exists to
/// prevent.
fn thermal_json(r: &ThermalReading) -> serde_json::Value {
    serde_json::json!({
        "measured": r.measured(),
        "pressure": r.pressure.map(|p| p.as_str()),
        "source": r.source,
        "cpu_speed_limit_percent": r.cpu_speed_limit_percent,
        "degraded": r.measured().then(|| r.is_degraded()),
    })
}

/// A load average we could not read prints as `?`, never as `0.00`.
pub fn fmt_load(l: Option<f64>) -> String {
    l.map(|l| format!("{l:.2}"))
        .unwrap_or_else(|| "?".to_string())
}

/// Whether a receipt's `backend` label describes the backend that ran.
///
/// The label is lowercase and comes from the suite (`cpu`, `metal`,
/// `cuda`); the active backend is what `active_backend` resolved.
/// Compared case-insensitively and nothing else: a mismatch is always
/// a defect, never a naming convention.
pub fn backend_label_agrees(label: &str, active: &str) -> bool {
    label.eq_ignore_ascii_case(active)
}

/// The accelerator a non-CPU run executed on, as the backend itself
/// reports it.
///
/// Returns `None` on CPU, where `host_spec` already describes the
/// machine, and `None` when a GPU backend cannot name its device --
/// which the receipt writer treats as a refusal rather than a blank,
/// because an unnamed card is what made ten CUDA rows unattributable.
pub(crate) fn accelerator_name(backend: &str) -> Option<String> {
    match backend {
        #[cfg(feature = "cuda")]
        "CUDA" => frink_cuda::gpu::probe().and_then(|i| i.first_device_name),
        #[cfg(feature = "metal")]
        "Metal" => frink_metal::gpu::probe(),
        _ => None,
    }
}

/// The label a receipt is about to be written under, checked against
/// the backend that ran, BEFORE anything is timed.
///
/// The write-time check in [`receipt_common`] is the discipline; this
/// is the same predicate asked early so a mislabelled run is refused
/// in a millisecond rather than after the whole sweep. Both call the
/// one function, so they cannot disagree.
pub fn ensure_label_matches_backend(label: &str, active: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        backend_label_agrees(label, active),
        "refusing to write a receipt labelled `{label}` for a run that executed on \
         {active}. The label is what the ledger publishes; the active backend is what \
         ran (#126)."
    );
    Ok(())
}

/// The fields every engine receipt carries, and the two refusals that
/// decide whether it is written at all.
///
/// The caller merges its own workload-specific fields in on top;
/// nothing here depends on what was measured, only on where and how.
pub fn receipt_common(
    label: &str,
    backend_active: &str,
    threads: usize,
    load_s: f64,
    host: HostState,
    engine_env: &[(String, String)],
) -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
    // WHAT the machine was, never WHICH machine it was. A receipt is
    // public, and `--render` reads every receipt in the directory:
    // without this, rows from two hosts merge into one table with no
    // column saying so.
    let host_spec_json = {
        let spec = host_state::host_spec();
        serde_json::json!({
            "label": host_state::host_label(&spec),
            "cpu": spec.cpu,
            "arch": spec.arch,
            "cores": spec.cores,
            "perf_cores": spec.perf_cores,
            "ram_gb": spec.ram_gb.map(|g| (g * 10.0).round() / 10.0),
            "os": spec.os,
        })
    };
    // A receipt may not claim one backend and record another. Every
    // `cpu` row in the ledger did exactly that (#126), and the
    // contradiction sat in the file: `backend: "cpu"` beside
    // `backend_active: "Metal"`. Two fields that must agree, with
    // nothing comparing them, which is this repo's oldest bug shape.
    // Asked first so a mislabelled receipt is refused AS mislabelled
    // in every build, not as "no accelerator" in a CPU-only one.
    ensure_label_matches_backend(label, backend_active)?;

    // `host_spec` names the CPU. For a `cpu` row that is the whole
    // machine; for a `cuda` or `metal` row it is the least interesting
    // part. Ten published CUDA rows carried a Xeon's model number and
    // NOTHING about the card, in a file whose own header says rows are
    // never compared across machines -- there was no field to compare
    // them by, so a Pascal row and an Ampere row grouped as one host.
    let accelerator = accelerator_name(backend_active);
    if backend_active != "CPU" && accelerator.is_none() {
        anyhow::bail!(
            "refusing to write a `{label}` receipt that does not name the accelerator it \
             ran on. A GPU gap is meaningless without the card: the ledger groups rows by \
             host, and two different GPUs in one host section are two different machines."
        );
    }

    let serde_json::Value::Object(map) = serde_json::json!({
        "backend": label,
        "backend_active": backend_active,
        "threads": threads,
        "warmup_reps": bench_guard::WARMUP_REPS,
        "load_s": load_s,
        // Null, not zero, when the platform would not say -- see host_state.
        "host_load_1min_start": host.load_start,
        "host_load_1min_end": host.load_end,
        "host_thermal_start": thermal_json(&host.thermal_start),
        "host_thermal_end": thermal_json(&host.thermal_end),
        // The single field an auditor reads first: was this row taken
        // under the measurement contract's quiet-host bar at all?
        "quiet_host": host.load_start.map(|l| l < host_state::DEFAULT_MAX_LOAD),
        "host_spec": host_spec_json,
        // The card, for a row that ran on one. `null` for `cpu` rows,
        // where `host_spec` already is the machine.
        "accelerator": accelerator,
        // Non-default `FRINK_*` knobs in effect. Some of them (MoE
        // stage ablation, the fail-closed loader override) change how
        // much work the engine does, so a row taken under one is not
        // comparable to a row taken without it.
        "engine_env": engine_env
            .iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect::<serde_json::Map<_, _>>(),
        "frink_version": env!("CARGO_PKG_VERSION"),
    }) else {
        unreachable!("json! with braces is an object");
    };
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The receipt guard that #126 needed and did not have.
    #[test]
    fn a_cpu_label_does_not_describe_a_metal_run() {
        assert!(backend_label_agrees("cpu", "CPU"));
        assert!(backend_label_agrees("metal", "Metal"));
        assert!(backend_label_agrees("cuda", "CUDA"));
        assert!(
            !backend_label_agrees("cpu", "Metal"),
            "this exact pair is what all 13 published cpu receipts recorded"
        );
        assert!(!backend_label_agrees("metal", "CPU"));
        assert!(!backend_label_agrees("cuda", "Metal"));
    }

    /// The early check and the write-time check are one predicate;
    /// this pins that the early one refuses the same pair.
    #[test]
    fn the_early_label_check_refuses_the_published_mismatch() {
        let err = ensure_label_matches_backend("cpu", "Metal")
            .unwrap_err()
            .to_string();
        assert!(err.contains("labelled `cpu`"), "{err}");
        assert!(err.contains("executed on Metal"), "{err}");
        assert!(ensure_label_matches_backend("cpu", "CPU").is_ok());
    }

    /// A `cpu` row has no accelerator and must not be refused for it.
    #[test]
    fn the_cpu_backend_names_no_accelerator() {
        assert_eq!(accelerator_name("CPU"), None);
    }

    fn unmeasured_host() -> HostState {
        HostState {
            load_start: None,
            load_end: None,
            thermal_start: host_state::ThermalReading::default(),
            thermal_end: host_state::ThermalReading::default(),
        }
    }

    /// The write-time refusal, exercised through the function both
    /// receipt writers call rather than through either writer.
    #[test]
    fn a_receipt_labelled_for_another_backend_is_not_written() {
        let err = receipt_common("cpu", "Metal", 4, 1.0, unmeasured_host(), &[])
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("#126"), "{err}");
    }

    /// A GPU row that cannot name its card is refused too. In a build
    /// without that backend the probe always answers `None`, which is
    /// exactly the "cannot name it" case.
    #[test]
    fn a_gpu_receipt_without_an_accelerator_name_is_not_written() {
        let err = receipt_common("metal", "Metal", 4, 1.0, unmeasured_host(), &[])
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not name the accelerator"), "{err}");
    }

    #[test]
    fn a_cpu_receipt_carries_the_shared_fields_and_no_accelerator() {
        let map = receipt_common("cpu", "CPU", 4, 1.5, unmeasured_host(), &[]).unwrap();
        assert_eq!(map["backend"], "cpu");
        assert_eq!(map["backend_active"], "CPU");
        assert_eq!(map["threads"], 4);
        assert_eq!(map["accelerator"], serde_json::Value::Null);
        // Unmeasured, not "quiet": a null load must not read as passing.
        assert_eq!(map["quiet_host"], serde_json::Value::Null);
        assert_eq!(map["host_thermal_start"]["measured"], false);
        assert_eq!(map["warmup_reps"], bench_guard::WARMUP_REPS);
    }

    /// The fits-in-RAM check is waived by `--max-load 0`, the same
    /// switch that waives the load and thermal bars, and only by it.
    /// An exabyte of KV on top of any real file does not fit any host
    /// that reports its free memory.
    #[test]
    fn the_ram_check_is_waived_with_the_load_check() {
        let me = std::env::current_exe().expect("the test binary exists");
        assert!(ensure_weights_fit(0.0, &me, 1e9).is_ok());
        if host_state::free_ram_gb().is_some() {
            let err = ensure_weights_fit(2.0, &me, 1e9).unwrap_err().to_string();
            assert!(err.contains("would run from swap"), "{err}");
        }
    }
}
