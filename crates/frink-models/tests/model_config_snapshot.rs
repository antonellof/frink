//! The snapshot guard for the architecture registry migration (#403,
//! `docs/plans/arch-registry-migration.md`, step 0).
//!
//! Every later step of that plan moves architecture-keyed table rows
//! into per-architecture files and claims "no behaviour change". This
//! test is what makes the claim checkable: for EVERY GGUF fixture the
//! repo carries it records what the loaders resolve, and compares the
//! record against a checked-in snapshot.
//!
//! Per fixture, one text file under `tests/snapshots/model_config/`:
//!
//! - the full resolved [`ModelConfig`] (`{:#?}`), or the exact refusal
//!   text when `ModelConfig::from_gguf` refuses;
//! - every per-layer accessor the decoder bodies read
//!   (`layer_sliding_window`, `layer_rope`, `layer_rotates`,
//!   `layer_rope_theta`, `layer_attention_kind`, `layer_is_dense`,
//!   `layer_shape`, `layer_ffn_acts`, `layer_kv_dtype`,
//!   `layer_cache_geometry`);
//! - after `Decoder::from_gguf`, or its refusal text, which optional
//!   weights each layer holds and how each norm site resolved. Weight
//!   VALUES are reduced to a length, an f64 sum and a hash of the bits,
//!   so a loader that rewrites a vector (an offset, a permutation, a
//!   sign) shows up without the snapshot carrying the tensor. Per-layer
//!   lines whose value is `None` or `false` are left out to keep the
//!   snapshots small; one that becomes present appears as a new line.
//!
//! The weight structs are destructured with no `..`, so a new field on
//! `AttnWeights`, `MoeWeights` or `LayerWeights` does not compile here
//! until it is given a line in the digest.
//!
//! # Regenerating
//!
//! ```text
//! FRINK_UPDATE_SNAPSHOTS=1 cargo test -p frink-models --test model_config_snapshot
//! ```
//!
//! rewrites every snapshot and deletes the ones whose fixture is gone.
//! Nothing regenerates them otherwise: a mismatch fails, naming the
//! fixture and the field, and the fix is either the code or a
//! deliberate regeneration whose diff is reviewed in the PR.
//!
//! # Determinism
//!
//! The digest carries no absolute path (the workspace root is replaced
//! by `<root>`), no address and no time. `ModelConfig` holds no hash
//! map. Floats print through Rust's shortest round-trip formatting,
//! which is platform independent. The two environment switches that
//! change what the loaders admit must be unset, and the test says so
//! rather than recording a different answer. The snapshots are of a
//! default-feature build; the one backend probe in the digest (the
//! execution plan's fused-op caps) is masked, and Metal's packed expert
//! planes are not recorded, so a `metal` build digests the same.

use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::ShardedGguf;
use frink_models::decoder::{AttnWeights, ExpertBacking, LayerWeights, MoeWeights};
use frink_models::{Decoder, ModelConfig, NormOp};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const UPDATE_ENV: &str = "FRINK_UPDATE_SNAPSHOTS";

/// Environment switches the loaders read that change what they admit.
const LOADER_ENV: [&str; 2] = ["FRINK_ALLOW_UNKNOWN_TENSORS", "FRINK_ALLOW_UNAUDITED_ARCH"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/model_config")
}

/// `(snapshot name, fixture path)` for every fixture GGUF, sorted.
///
/// The two directories any test in the workspace loads GGUFs from. A
/// fixture added to either is covered without editing this file.
fn fixtures(root: &Path) -> Vec<(String, PathBuf)> {
    let dirs: [(&str, &str); 2] = [
        ("", "crates/frink-models/tests/fixtures"),
        ("workspace-", "tests/fixtures"),
    ];
    let mut out = Vec::new();
    for (prefix, dir) in dirs {
        let mut found = Vec::new();
        collect_gguf(&root.join(dir), &mut found);
        assert!(!found.is_empty(), "no fixtures under {dir}");
        for p in found {
            let stem = p.file_stem().unwrap().to_str().unwrap().to_string();
            out.push((format!("{prefix}{stem}"), p));
        }
    }
    out.sort();
    for w in out.windows(2) {
        assert_ne!(w[0].0, w[1].0, "two fixtures share a snapshot name");
    }
    out
}

fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_gguf(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("gguf") {
            out.push(p);
        }
    }
}

// --- digest ----------------------------------------------------------

/// `len`, sequential f64 sum and FNV-1a over the bit patterns. Both are
/// exact functions of the bits, so they agree on every platform.
fn vec_digest(v: &[f32]) -> String {
    let sum: f64 = v.iter().map(|&x| x as f64).sum();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("len={} sum={sum:?} fnv={h:016x}", v.len())
}

fn opt_vec(v: &Option<Vec<f32>>) -> String {
    match v {
        Some(v) => vec_digest(v),
        None => "None".into(),
    }
}

fn wm(m: &WeightMatrix) -> String {
    let storage = match m {
        WeightMatrix::F32(_) => "F32",
        WeightMatrix::Quantized { .. } => "Quantized",
        WeightMatrix::Mxfp4 { .. } => "Mxfp4",
        WeightMatrix::Adapted { .. } => "Adapted",
        WeightMatrix::Folded { .. } => "Folded",
    };
    format!("{}x{} {storage} {:?}", m.rows(), m.cols(), m.quant_kind())
}

fn norm(n: &NormOp) -> String {
    match n {
        NormOp::Rms(w) => format!("Rms({})", vec_digest(w)),
        NormOp::LayerNormNoParams => "LayerNormNoParams".into(),
        NormOp::RmsNoParams => "RmsNoParams".into(),
        NormOp::LayerNorm(w) => format!("LayerNorm({})", vec_digest(w)),
        NormOp::LayerNormBias { weight, bias } => format!(
            "LayerNormBias(weight {}, bias {})",
            vec_digest(weight),
            vec_digest(bias)
        ),
        NormOp::RmsBias { weight, bias } => format!(
            "RmsBias(weight {}, bias {})",
            vec_digest(weight),
            vec_digest(bias)
        ),
        NormOp::None => "None".into(),
    }
}

fn config_layers(c: &ModelConfig, out: &mut String) {
    for l in 0..c.n_layers {
        let p = format!("cfg.layer[{l}]");
        let rope = match c.layer_rope(l) {
            Some(r) => format!(
                "Some(theta={:?} rot_dim={:?} freq_factors={})",
                r.theta,
                r.rot_dim,
                r.freq_factors
                    .map(vec_digest)
                    .unwrap_or_else(|| "None".into())
            ),
            None => "None".into(),
        };
        let _ = writeln!(out, "{p}.sliding_window: {:?}", c.layer_sliding_window(l));
        let _ = writeln!(out, "{p}.rope: {rope}");
        let _ = writeln!(out, "{p}.rotates: {:?}", c.layer_rotates(l));
        let _ = writeln!(out, "{p}.rope_theta: {:?}", c.layer_rope_theta(l));
        let _ = writeln!(out, "{p}.attention_kind: {:?}", c.layer_attention_kind(l));
        let _ = writeln!(out, "{p}.is_dense: {:?}", c.layer_is_dense(l));
        let _ = writeln!(out, "{p}.shape: {:?}", c.layer_shape(l));
        let _ = writeln!(out, "{p}.ffn_acts: {:?}", c.layer_ffn_acts(l));
        let _ = writeln!(out, "{p}.kv_dtype: {:?}", c.layer_kv_dtype(l));
        let _ = writeln!(out, "{p}.cache_geometry: {:?}", c.layer_cache_geometry(l));
    }
}

fn attn_digest(p: &str, a: &AttnWeights, out: &mut String) {
    let AttnWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        norm_weight,
        q_norm,
        k_norm,
        q_bias,
        k_bias,
        v_bias,
        post_attn_norm,
        post_ffn_norm,
        output_gate,
        sinks,
        attn_sub_norm,
        o_scale,
        o_bias,
        shortconv,
        ssm,
        q_gate_interleaved,
    } = a;
    let _ = writeln!(out, "{p}.q_proj: {}", wm(q_proj));
    let _ = writeln!(out, "{p}.k_proj: {}", wm(k_proj));
    let _ = writeln!(out, "{p}.v_proj: {}", wm(v_proj));
    let _ = writeln!(out, "{p}.o_proj: {}", wm(o_proj));
    let _ = writeln!(out, "{p}.norm_weight: {}", norm(norm_weight));
    let _ = writeln!(out, "{p}.q_norm: {}", opt_vec(q_norm));
    let _ = writeln!(out, "{p}.k_norm: {}", opt_vec(k_norm));
    let _ = writeln!(out, "{p}.q_bias: {}", opt_vec(q_bias));
    let _ = writeln!(out, "{p}.k_bias: {}", opt_vec(k_bias));
    let _ = writeln!(out, "{p}.v_bias: {}", opt_vec(v_bias));
    let _ = writeln!(out, "{p}.post_attn_norm: {}", opt_vec(post_attn_norm));
    let _ = writeln!(out, "{p}.post_ffn_norm: {}", opt_vec(post_ffn_norm));
    let _ = writeln!(out, "{p}.output_gate: {output_gate:?}");
    let _ = writeln!(out, "{p}.sinks: {}", opt_vec(sinks));
    let _ = writeln!(out, "{p}.attn_sub_norm: {}", opt_vec(attn_sub_norm));
    let _ = writeln!(out, "{p}.o_scale: {o_scale:?}");
    let _ = writeln!(out, "{p}.o_bias: {}", opt_vec(o_bias));
    let sc = match shortconv {
        Some(s) => format!(
            "Some(l_cache={} conv {} in_proj {} out_proj {})",
            s.l_cache,
            vec_digest(&s.conv),
            wm(&s.in_proj),
            wm(&s.out_proj)
        ),
        None => "None".into(),
    };
    let _ = writeln!(out, "{p}.shortconv: {sc}");
    let ssm = match ssm {
        Some(frink_models::ssm_block::SsmBlock::Mamba1(_)) => "Mamba1",
        Some(frink_models::ssm_block::SsmBlock::Mamba2(_)) => "Mamba2",
        Some(frink_models::ssm_block::SsmBlock::Gdn(_)) => "Gdn",
        Some(frink_models::ssm_block::SsmBlock::Plamo2(_)) => "Plamo2",
        Some(frink_models::ssm_block::SsmBlock::Lightning(_)) => "Lightning",
        None => "None",
    };
    let _ = writeln!(out, "{p}.ssm: {ssm}");
    let _ = writeln!(out, "{p}.q_gate_interleaved: {q_gate_interleaved:?}");
}

fn moe_digest(p: &str, m: &MoeWeights, out: &mut String) {
    let MoeWeights {
        router,
        experts,
        shared_experts,
        shared_expert_gate,
        norm_weight,
        exp_probs_bias,
        ffn_sub_norm,
        down_scale,
        dense_bias,
        exps_norm,
        parallel_sum_scale,
        parallel,
        activation_counts,
        // Metal's repacked copy of the experts, built only on a
        // `metal` build. Left out of the digest so the snapshot is
        // the same with and without the feature.
        #[cfg(feature = "metal")]
            packed_q4: _,
    } = m;
    let _ = writeln!(out, "{p}.router: {}", wm(router));
    let experts = match experts {
        ExpertBacking::Resident(v) => {
            let mut shapes: Vec<String> = v
                .iter()
                .map(|e| format!("gate {} up {} down {}", wm(&e.gate), wm(&e.up), wm(&e.down)))
                .collect();
            shapes.dedup();
            format!("Resident(n={} [{}])", v.len(), shapes.join("; "))
        }
        ExpertBacking::Stored { layouts, .. } => format!("Stored(n={})", layouts.len()),
    };
    let _ = writeln!(out, "{p}.experts: {experts}");
    for (i, e) in shared_experts.iter().enumerate() {
        let _ = writeln!(
            out,
            "{p}.shared_experts[{i}]: gate {} up {} down {}",
            wm(&e.gate),
            wm(&e.up),
            wm(&e.down)
        );
    }
    let _ = writeln!(
        out,
        "{p}.shared_expert_gate: {}",
        opt_vec(shared_expert_gate)
    );
    let _ = writeln!(out, "{p}.norm_weight: {}", norm(norm_weight));
    let _ = writeln!(out, "{p}.exp_probs_bias: {}", opt_vec(exp_probs_bias));
    let _ = writeln!(out, "{p}.ffn_sub_norm: {}", opt_vec(ffn_sub_norm));
    let _ = writeln!(out, "{p}.down_scale: {down_scale:?}");
    let db = match dense_bias {
        Some(b) => format!(
            "Some(gate {} up {} down {})",
            opt_vec(&b.gate),
            opt_vec(&b.up),
            opt_vec(&b.down)
        ),
        None => "None".into(),
    };
    let _ = writeln!(out, "{p}.dense_bias: {db}");
    let _ = writeln!(out, "{p}.exps_norm: {}", opt_vec(exps_norm));
    let _ = writeln!(out, "{p}.parallel_sum_scale: {parallel_sum_scale:?}");
    let _ = writeln!(out, "{p}.parallel: {parallel:?}");
    let _ = writeln!(
        out,
        "{p}.activation_counts: len={}",
        activation_counts.len()
    );
}

fn decoder_digest(d: &Decoder, out: &mut String) {
    let _ = writeln!(out, "dec.embedding: {}", wm(&d.embedding));
    let _ = writeln!(
        out,
        "dec.position_embd: {}",
        d.position_embd
            .as_ref()
            .map(wm)
            .unwrap_or_else(|| "None".into())
    );
    let _ = writeln!(out, "dec.embedding_norm: {}", norm(&d.embedding_norm));
    let _ = writeln!(out, "dec.hrm_z_l_init: {}", opt_vec(&d.hrm_z_l_init));
    let _ = writeln!(out, "dec.final_norm: {}", norm(&d.final_norm));
    let _ = writeln!(out, "dec.output_head: {}", wm(&d.output_head));
    let _ = writeln!(out, "dec.output_bias: {}", opt_vec(&d.output_bias));
    let _ = writeln!(out, "dec.alibi_slopes: {}", opt_vec(&d.alibi_slopes));
    let _ = writeln!(
        out,
        "dec.gpu_vram_budget_bytes: {:?}",
        d.gpu_vram_budget_bytes
    );
    let _ = writeln!(out, "dec.qk_norm_after_rope: {:?}", d.qk_norm_after_rope);
    let _ = writeln!(out, "dec.lora_adapters: {}", d.lora_adapters.len());
    // The fused-op caps are a probe of the backend this build has, not
    // a fact of the file: masked so a `metal` build digests the same.
    let mut plan = d.execution_plan.clone();
    plan.fused = Default::default();
    let _ = writeln!(out, "dec.execution_plan: {plan:?}");
    match &d.gpt_oss {
        Some(g) => {
            for (l, layer) in g.layers.iter().enumerate() {
                let biases: Vec<f32> = layer
                    .expert_bias
                    .iter()
                    .flat_map(|b| b.gate.iter().chain(&b.up).chain(&b.down).copied())
                    .collect();
                let _ = writeln!(
                    out,
                    "dec.gpt_oss[{l}]: router_bias {} expert_bias n={} {}",
                    vec_digest(&layer.router_bias),
                    layer.expert_bias.len(),
                    vec_digest(&biases)
                );
            }
        }
        None => {
            let _ = writeln!(out, "dec.gpt_oss: None");
        }
    }
    let _ = writeln!(out, "dec.layers: {}", d.layers.len());
    for (l, layer) in d.layers.iter().enumerate() {
        let LayerWeights {
            attn,
            moe,
            out_scale,
        } = layer;
        attn_digest(&format!("dec.layer[{l}].attn"), attn, out);
        moe_digest(&format!("dec.layer[{l}].moe"), moe, out);
        let _ = writeln!(out, "dec.layer[{l}].out_scale: {out_scale:?}");
    }
}

fn panic_text(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic>".into()
    }
}

/// Drops the per-layer and decoder lines whose value is `None` or
/// `false`: most layers of most fixtures have most options absent, and
/// a field that BECOMES present still shows as an added line naming
/// itself. The `ModelConfig` dump is kept whole.
fn omit_absent(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        let per_layer = line.starts_with("cfg.") || line.starts_with("dec.");
        if per_layer && (line.ends_with(": None") || line.ends_with(": false")) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// What the loaders resolve for one fixture. Each stage that refuses
/// or panics is recorded as text and ends the digest there.
fn digest(path: &Path) -> String {
    let mut out = String::new();
    let file = match ShardedGguf::open(path) {
        Ok(f) => f,
        Err(e) => {
            let _ = writeln!(out, "open: REFUSED: {e}");
            return out;
        }
    };
    let config = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ModelConfig::from_gguf(&file)
    })) {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            let _ = writeln!(out, "config: REFUSED: {e}");
            return out;
        }
        Err(p) => {
            let _ = writeln!(out, "config: PANICKED: {}", panic_text(p));
            return out;
        }
    };
    drop(file);
    let _ = writeln!(out, "config: {config:#?}");
    if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut s = String::new();
        config_layers(&config, &mut s);
        out.push_str(&s);
    })) {
        let _ = writeln!(out, "cfg.layers: PANICKED: {}", panic_text(p));
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Decoder::from_gguf(path, config)
    })) {
        Ok(Ok(d)) => {
            let _ = writeln!(out, "decoder: LOADED");
            decoder_digest(&d, &mut out);
        }
        Ok(Err(e)) => {
            let _ = writeln!(out, "decoder: REFUSED: {e}");
        }
        Err(p) => {
            let _ = writeln!(out, "decoder: PANICKED: {}", panic_text(p));
        }
    }
    omit_absent(&out)
}

// --- comparison ------------------------------------------------------

/// The dotted field path of line `i` of a `{:#?}` dump: each enclosing
/// line with smaller indentation contributes its key.
fn field_path(lines: &[&str], i: usize) -> String {
    let indent = |s: &str| s.len() - s.trim_start().len();
    let key = |s: &str| {
        let t = s.trim();
        t.split_once(": ")
            .map(|(k, _)| k)
            .unwrap_or(t.trim_end_matches([',', '{', '(', '[', ' ']))
            .to_string()
    };
    // A bare value line (`0.6,` inside `Some(`) names nothing itself.
    let mut parts = if lines[i].contains(": ") {
        vec![key(lines[i])]
    } else {
        Vec::new()
    };
    let mut level = indent(lines[i]);
    for j in (0..i).rev() {
        if level == 0 {
            break;
        }
        let l = indent(lines[j]);
        if l < level && !lines[j].trim().is_empty() {
            parts.push(key(lines[j]));
            level = l;
        }
    }
    parts.reverse();
    parts.join(".")
}

/// The first differing hunk between two snapshots, named by field.
fn describe_diff(name: &str, want: &str, got: &str) -> String {
    let w: Vec<&str> = want.lines().collect();
    let g: Vec<&str> = got.lines().collect();
    let pre = w.iter().zip(&g).take_while(|(a, b)| a == b).count();
    let suf = w[pre..]
        .iter()
        .rev()
        .zip(g[pre..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    let (w_mid, g_mid) = (&w[pre..w.len() - suf], &g[pre..g.len() - suf]);
    let at = if pre < w.len() {
        pre
    } else {
        w.len().saturating_sub(1)
    };
    let field = if w.is_empty() {
        String::new()
    } else {
        field_path(&w, at)
    };
    let mut s = format!(
        "fixture `{name}`: field `{field}` (line {}) differs (- snapshot, + now)\n",
        pre + 1
    );
    const SHOW: usize = 12;
    for l in w_mid.iter().take(SHOW) {
        let _ = writeln!(s, "  - {l}");
    }
    if w_mid.len() > SHOW {
        let _ = writeln!(s, "  - ... {} more", w_mid.len() - SHOW);
    }
    for l in g_mid.iter().take(SHOW) {
        let _ = writeln!(s, "  + {l}");
    }
    if g_mid.len() > SHOW {
        let _ = writeln!(s, "  + ... {} more", g_mid.len() - SHOW);
    }
    s
}

/// Every fixture's resolved config and per-layer weights match the
/// checked-in snapshot.
///
/// This is the net under every "no behaviour change" step of #403: the
/// registry migration moves about 110 architecture-keyed tables, and a
/// row lost or misread in the move changes what one fixture resolves to
/// without necessarily changing any logit a graph test compares.
#[test]
fn every_fixture_resolves_to_its_snapshot() {
    for var in LOADER_ENV {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set; it changes what the loaders admit, so the snapshots would not mean what they say"
        );
    }
    let update = std::env::var_os(UPDATE_ENV).is_some();
    let root = workspace_root();
    let root_str = root.to_str().expect("utf-8 root").to_string();
    let dir = snapshot_dir();
    let fixtures = fixtures(&root);

    if update {
        std::fs::create_dir_all(&dir).expect("create snapshot dir");
    }
    let mut failures = Vec::new();
    let mut expected_files = std::collections::BTreeSet::new();
    for (name, path) in &fixtures {
        let rel = path.strip_prefix(&root).unwrap().to_str().unwrap();
        let rel = rel.replace('\\', "/");
        let body = digest(path).replace(&root_str, "<root>");
        let got = format!("fixture: {rel}\n{body}");
        let file = dir.join(format!("{name}.txt"));
        expected_files.insert(format!("{name}.txt"));
        if update {
            std::fs::write(&file, &got).expect("write snapshot");
            continue;
        }
        match std::fs::read_to_string(&file) {
            Ok(want) if want == got => {}
            Ok(want) => failures.push(describe_diff(name, &want, &got)),
            Err(_) => failures.push(format!(
                "fixture `{name}`: no snapshot at {}",
                file.strip_prefix(&root).unwrap_or(&file).display()
            )),
        }
    }
    // A snapshot whose fixture is gone is a fixture deleted without
    // anyone noticing what it covered.
    for e in std::fs::read_dir(&dir).expect("snapshot dir").flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if !expected_files.contains(&n) {
            if update {
                std::fs::remove_file(e.path()).expect("remove stale snapshot");
            } else {
                failures.push(format!("snapshot `{n}` has no fixture"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} fixture snapshot(s) differ. If the change is intended, regenerate with\n  \
         {UPDATE_ENV}=1 cargo test -p frink-models --test model_config_snapshot\nand review the diff.\n\n{}",
        failures.len(),
        fixtures.len(),
        failures.join("\n")
    );
}
