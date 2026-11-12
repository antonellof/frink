//! The Vulkan decode gate's body (`vulkan_decode_matches_cpu_and_llama_cpp`).

use super::*;
use frink_core::kv_store::KvDtype;

/// The worst |Vulkan - CPU| logit delta any step may show over a
/// 32-step greedy decode (the prefill batch and then the row body, each
/// on the Vulkan stack where it serves a layer and on the matvec and
/// GEMM offload elsewhere).
///
/// Both sides keep an f16 KV: the device's is f16, the stack declines a
/// host cache that is not, and the CPU child is given the same dtype. So
/// what is left is the order a device kernel accumulates in.
///
/// Measured (2026-11-12, M2 Pro through MoltenVK, the 28 fixtures the
/// stack serves): min 1.2e-7, p50 3.1e-6, p90 4.3e-4, then the 64-layer
/// `exaone4_32b` pair at 9.0e-4 and `glm4` at 1.1e-3. The tail is where
/// a norm sits on a block's output: its sum of squares is an f32 tree on
/// the device and an f64 loop on the CPU.
///
/// `glm4` (pre-norms and post-norms) is the fixture's own sensitivity
/// and not the device's error: against llama.cpp's decode golden the
/// device is at 7.3e-4 and the CPU with the same f16 KV at 1.2e-3, so
/// the device is the closer of the two to the oracle. The bound is 2.7x
/// that distance.
///
/// With an f32 KV on the CPU side an earlier run showed 8.7e-2 on
/// `lora_base`, which is the KV's rounding and not the stack's, and is
/// why the comparison is f16 against f16.
pub const VULKAN_DECODE_TOL: f32 = 3e-3;

/// The bound against llama.cpp's decode goldens, which keep an f32 KV:
/// the Metal gate's, for the same f16 rounding. Measured at 1.8e-4 on
/// `mistral3`, the one served fixture with a decode golden.
const VULKAN_GOLDEN_TOL: f32 = 0.2;

/// Fixtures the stack served, and agreed, at the measurement above. The
/// gate fails below it: fewer means a refusal moved and a fixture now
/// runs on the host, which this gate would otherwise pass in silence.
const MIN_SERVED: usize = 28;

/// Fixtures whose Vulkan decode is not within rounding of the CPU's,
/// each with what was measured. The gate fails if this set and the
/// measured one differ in either direction.
pub const KNOWN_VULKAN_DIVERGENCES: &[&str] = &[];

/// Layers one submission holds: `vulkan_stack`'s `MAX_DISPATCHES /
/// OPS_PER_LAYER`.
const MAX_RUN: usize = frink_vulkan::runtime::MAX_DISPATCHES / 19;

/// A served fixture with F32 weights, which the matvec offload does not
/// take: with an f32 host KV the stack declines and nothing is submitted.
const F32_KV_FIXTURE: &str = "smollm3";

/// The prompt prefilled as two batches (two positions, then the rest
/// over that prefix) gives the logits of one batch, bit for bit: the
/// second batch attends over a device KV the first one wrote, at a
/// start position that is not zero.
fn a_split_prefill_is_the_prefill(d: &Decoder) -> Result<(), String> {
    let prompt = &common::GRAPH_PROMPT;
    let mut kv = d.config.new_kv_caches();
    let whole = d.forward_batch_last(prompt, 0, &mut kv);
    let mut kv = d.config.new_kv_caches();
    d.forward_batch_last(&prompt[..2], 0, &mut kv);
    let split = d.forward_batch_last(&prompt[2..], 2, &mut kv);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&whole) != bits(&split) {
        return Err(format!(
            "a prefill split at position 2 differs from the whole one by {:.3e}",
            common::worst_vs(&split, &whole)
        ));
    }
    Ok(())
}

fn submissions() -> u64 {
    frink_core::weight_matrix::gpu_backend::vulkan_runtime()
        .and_then(|rt| rt.lock().ok().map(|r| r.submissions()))
        .unwrap_or(0)
}

pub fn run() {
    if let Err(e) = frink_vulkan::device::probe() {
        frink_vulkan::skipped(&format!("no Vulkan device: {e}"));
        return;
    }
    common::pin_backend(common::Backend::Vulkan);
    let mut failures = Vec::new();
    let mut refused = Vec::new();
    let mut on_host = Vec::new();
    let mut spread = Vec::new();
    let mut checked_f32_kv = false;
    for name in common::graph_fixture_names() {
        let Some(cpu) = device_gate::cpu_rows_from_child(
            &name,
            "vulkan_decode_matches_cpu_and_llama_cpp",
            "f16",
        ) else {
            refused.push(name);
            continue;
        };
        assert_eq!(cpu.len(), STEPS, "{name}: CPU child row count");
        let mut d = common::load_graph_fixture(&name);
        d.config.kv_dtype = KvDtype::F16;
        let before = submissions();
        let run = std::panic::AssertUnwindSafe(|| prefill_then_row(&d, &common::GRAPH_PROMPT));
        let gpu = match std::panic::catch_unwind(run) {
            Ok(rows) => rows,
            Err(e) => {
                let why = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                failures.push((name.clone(), format!("the Vulkan decode panicked: {why}")));
                continue;
            }
        };
        let submitted = submissions() - before;
        if submitted == 0 {
            // No Vulkan path serves this fixture (F32 weights, a kind
            // with no shader): Vulkan against CPU would be CPU against
            // CPU.
            on_host.push(name);
            continue;
        }
        // The stand-in for speed, since nothing here is timed: the
        // prefill batch and each decode token wait once per run of
        // layers, never once per op. `STEPS` forward passes: the
        // prefill, then `STEPS - 1` tokens.
        //
        // A layer with a sliding window stays on the host, so it cuts
        // the run on either side of it: each one allows a run more.
        let windowed = (0..d.layers.len())
            .filter(|&li| d.config.layer_sliding_window(li).is_some())
            .count();
        let per_pass = submitted as usize / STEPS;
        if !(submitted as usize).is_multiple_of(STEPS)
            || per_pass > d.layers.len().div_ceil(MAX_RUN) + windowed
        {
            failures.push((
                name.clone(),
                format!(
                    "{submitted} submissions over a prefill and {} tokens of {} layers, \
                     {windowed} of them windowed; the stack is one per {MAX_RUN} layers \
                     and per run between windowed layers",
                    STEPS - 1,
                    d.layers.len()
                ),
            ));
        }
        if let Err(why) = a_split_prefill_is_the_prefill(&d) {
            failures.push((name.clone(), why));
        }
        if name == F32_KV_FIXTURE {
            d.config.kv_dtype = KvDtype::F32;
            let before = submissions();
            prefill_then_row(&d, &common::GRAPH_PROMPT);
            let n = submissions() - before;
            if n != 0 {
                failures.push((
                    name.clone(),
                    format!(
                        "{n} submissions with an f32 host KV, which the f16 device KV must decline"
                    ),
                ));
            }
            checked_f32_kv = true;
        }
        let a = device_gate::agreement(&gpu, &cpu);
        let vs_llama = GOLDENS.iter().find(|(g, _)| *g == name).map(|_| {
            let golden = read_golden(&name);
            (
                worst_over_steps(&gpu, &golden),
                first_divergence(
                    &gpu,
                    &golden.logits,
                    Some(&golden.tokens),
                    VULKAN_GOLDEN_TOL,
                ),
            )
        });
        eprintln!(
            "vulkan-decode {name:>28}: {submitted:>5} submissions, worst |vulkan - cpu| = \
             {:.3e} over {} steps{}{}",
            a.worst,
            a.steps,
            a.parted
                .map(|lead| format!(", PARTED (cpu lead {lead:.3e})"))
                .unwrap_or_default(),
            vs_llama
                .as_ref()
                .map(|(w, _)| format!(", worst |vulkan - llama.cpp| = {w:.3e}"))
                .unwrap_or_default(),
        );
        if a.worst.is_nan() || a.worst > VULKAN_DECODE_TOL {
            failures.push((
                name.clone(),
                format!(
                    "worst |vulkan - cpu| {:.3e} > {VULKAN_DECODE_TOL:.0e} within {} steps",
                    a.worst, a.steps
                ),
            ));
        }
        if let Some(lead) = a.parted {
            if lead > 2.0 * VULKAN_DECODE_TOL {
                failures.push((
                    name.clone(),
                    format!(
                        "greedy token parted from the CPU's at step {} where the CPU led \
                         by {lead:.3e}, more than a {VULKAN_DECODE_TOL:.0e} error can flip",
                        a.steps - 1
                    ),
                ));
            }
        }
        if let Some((_, Some(why))) = vs_llama {
            failures.push((name.clone(), format!("Vulkan vs llama.cpp: {why}")));
        }
        spread.push((a.worst, name));
    }
    for (name, why) in &failures {
        eprintln!("vulkan-decode diverges {name}: {why}");
    }
    let diverged: std::collections::BTreeSet<&str> =
        failures.iter().map(|(n, _)| n.as_str()).collect();
    let mut within: Vec<(f32, &str)> = spread
        .iter()
        .filter(|(_, n)| !diverged.contains(n.as_str()))
        .map(|(w, n)| (*w, n.as_str()))
        .collect();
    within.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(
        within.len() >= MIN_SERVED,
        "only {} fixtures reached a Vulkan path and agreed, under the {MIN_SERVED} measured; \
         a refusal moved, or the switch is not live in this process",
        within.len()
    );
    assert!(
        checked_f32_kv,
        "{F32_KV_FIXTURE} was not served, so the f32 KV refusal went unchecked"
    );
    let q = |p: f64| within[((within.len() - 1) as f64 * p).round() as usize].0;
    eprintln!(
        "vulkan-decode spread over the {} fixtures within rounding: min {:.3e}, p50 {:.3e}, \
         p90 {:.3e}, max {:.3e} ({}); {} diverge, {} refused, {} never submitted",
        within.len(),
        q(0.0),
        q(0.5),
        q(0.9),
        q(1.0),
        within.last().map(|s| s.1).unwrap_or(""),
        diverged.len(),
        refused.len(),
        on_host.len(),
    );
    let known: std::collections::BTreeSet<&str> =
        KNOWN_VULKAN_DIVERGENCES.iter().copied().collect();
    let new: Vec<_> = diverged.difference(&known).collect();
    let fixed: Vec<_> = known.difference(&diverged).collect();
    assert!(
        new.is_empty() && fixed.is_empty(),
        "Vulkan decode against the CPU's: NEW divergences {new:?} (see the `vulkan-decode \
         diverges` lines), and listed ones that now agree {fixed:?}"
    );
}
