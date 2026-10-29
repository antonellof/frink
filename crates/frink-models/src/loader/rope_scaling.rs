//! The two RoPE scalings read from a GGUF's metadata: `linear` and
//! YaRN.

use frink_gguf::TensorSource;

use super::model_config::metadata_f32_any;

/// The `linear` RoPE scaling factor, if this file declares one.
///
/// Deliberately separate from [`yarn_scaling_from_gguf`]: YaRN needs an
/// original context length and per-band betas, and linear needs neither.
/// Any factor at or below one is not a correction, and is treated as
/// absent rather than applied as a no-op.
pub(super) fn linear_scaling_from_gguf(file: &impl TensorSource, arch: &str) -> Option<f32> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let scaling_type = file.metadata_str(&key("rope.scaling.type"))?;
    if !scaling_type.eq_ignore_ascii_case("linear") {
        return None;
    }
    metadata_f32_any(file, &[key("rope.scaling.factor")]).filter(|f| f.is_finite() && *f > 1.0)
}

/// Architectures whose `load_arch_hparams` reads
/// `rope.scaling.yarn_beta_fast` / `_slow`
/// (`LLM_KV_ROPE_SCALING_YARN_BETA_FAST` / `_SLOW`). Everywhere else the
/// two keys are dead metadata upstream: `llama-hparams.h` seeds 32 / 1
/// and nothing overrides them.
///
/// Measured 2026-10-28 at the `604cc430` pin: `grep -l
/// YARN_BETA_FAST src/models/*.cpp` is `grok.cpp:26`, `mistral3.cpp:7`,
/// `laguna.cpp:10` (new since `5b59b83`) and `k2-horizon.cpp:4` (a new,
/// refused graph). frink read both keys for EVERY architecture until
/// then, while `conversion/base.py:1653-1656` writes them for any HF
/// config whose YaRN block carries `beta_fast` / `beta_slow` -- the
/// `expert_weights_scale` shape again: a key honoured where upstream
/// ignores it. `laguna`'s reading is the one that moved upstream; for it
/// frink was already right.
///
/// `grok` states its row in `crate::models::grok` (#403 step 6).
pub(crate) const YARN_BETA_READERS: &[&str] = &["mistral3", "laguna"];

/// The YaRN RoPE scaling a GGUF declares, or `None` when this file
/// declares none that changes the rotation.
///
/// llama.cpp's key names (`llama-arch.cpp`
/// `LLM_KV_ROPE_SCALING_TYPE` / `_FACTOR`): `<arch>.rope.scaling.type`
/// is a string (`"none"`, `"linear"`, `"yarn"`, `"longrope"`) and
/// `<arch>.rope.scaling.factor` the ratio of served to trained context.
/// `beta_fast` / `beta_slow` are read from both the plain and the
/// `yarn_`-prefixed spelling ONLY where the architecture's graph reads
/// them ([`YARN_BETA_READERS`]), and otherwise fall back to the
/// reference's own defaults (32.0 / 1.0, or Grok's seeded 8.0).
///
/// `None` is returned for every case where applying YaRN would be a
/// guess or a no-op rather than a correction, so that no checkpoint's
/// rotation moves without the file having asked for it:
///
/// * a scaling type other than `yarn` (`linear` divides positions,
///   `longrope` rides the `rope_factors_long`/`_short` tensors this
///   loader already reads -- neither is this rewrite, and treating them
///   as YaRN would rope them wrong in a *new* way instead of leaving
///   them as they are),
/// * a missing, non-finite or `<= 1.0` factor (the reference's own
///   `get_mscale` treats `scale <= 1` as unscaled, and a factor of 1.0
///   makes every band's divisor exactly 1.0 anyway),
/// * a missing `rope.scaling.original_context_length` -- the trained
///   context is what the correction range is measured against, and
///   inventing one (say, from `context_length`, which on a YaRN file is
///   the *extended* length) would put the ramp in the wrong place and
///   quietly rope the checkpoint at frequencies nobody trained.
pub(crate) fn yarn_scaling_from_gguf(
    file: &impl TensorSource,
    arch: &str,
    orig_ctx: Option<usize>,
) -> Option<frink_core::attention::YarnScaling> {
    let key = |suffix: &str| format!("{arch}.{suffix}");
    let scaling_type = file.metadata_str(&key("rope.scaling.type"))?;
    if !scaling_type.eq_ignore_ascii_case("yarn") {
        return None;
    }
    let factor = metadata_f32_any(file, &[key("rope.scaling.factor")])
        .filter(|f| f.is_finite() && *f > 1.0)?;
    let orig_max_pos = orig_ctx?;
    let reads_beta = crate::models::ArchFacts::of(arch).scaling.reads_yarn_beta;
    let beta = |suffix: &str, default: f32| -> f32 {
        if !reads_beta {
            return default;
        }
        metadata_f32_any(
            file,
            &[
                key(&format!("rope.scaling.{suffix}")),
                key(&format!("rope.scaling.yarn_{suffix}")),
            ],
        )
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
    };
    // `llama-hparams.h:137` seeds `yarn_beta_fast = 32.0f` for every
    // architecture and `grok.cpp:5` reseeds it to 8.0 before the key is
    // read; the table that holds Grok's other defaults holds that one
    // too, so it is not a second literal here.
    let beta_fast_default = crate::scalar_multipliers::multiplier_support(arch)
        .defaults
        .yarn_beta_fast()
        .unwrap_or(32.0);
    Some(frink_core::attention::YarnScaling {
        factor,
        beta_fast: beta("beta_fast", beta_fast_default),
        beta_slow: beta("beta_slow", 1.0),
        orig_max_pos,
        // No GGUF key carries the reference's `truncate` flag, and its
        // default is `true`; a file that wanted the fractional range
        // would have no way to say so here.
        truncate: true,
    })
}
