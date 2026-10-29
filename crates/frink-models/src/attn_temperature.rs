//! **Per-position attention temperature** -- llama.cpp's
//! `llm_graph_input_attn_temp`, as one value a `ModelConfig` carries
//! and one table that says which architectures compute it.
//!
//! # What it is
//!
//! `llama-graph.cpp:155-172` fills an `F32 [n_tokens]` input with one
//! scalar per token,
//!
//! ```text
//! attn_scale[i] = log(floor((pos[i] + offset) / floor_scale) + 1) * temp_scale + 1
//! ```
//!
//! and the graphs that use it multiply Q by that vector -- broadcast
//! over every head and every channel of a token's Q -- AFTER RoPE and
//! BEFORE `build_attn`, with `kq_scale` untouched
//! (`mistral3.cpp:153-156`). The scale is 1 at every position below
//! `floor_scale`, so the effect is invisible on a short prompt and
//! grows stepwise with the position: it is Llama-4's "attention
//! temperature tuning" (`llama4.cpp:15-17` seeds the three constants
//! and calls the input by that name), which mistral3 and deepseek2
//! adopted through a GGUF key.
//!
//! # Who reads it -- MEASURED, not read off one file
//!
//! `grep -ln 'attn_temp\|temperature_scale\|build_inp_attn_scale'
//! src/models/*.cpp` over all 155 graphs, 2026-09-11:
//!
//! | arch | scale | floor | offset | layers | line |
//! |---|---|---|---|---|---|
//! | `mistral3` | `attention.temperature_scale` | `n_ctx_orig_yarn` | 0 | all | `mistral3.cpp:5,14-17,109-111,153-156` |
//! | `deepseek2` (+ `mistral4`, `models.h:1311` reuses its hparams) | `attention.temperature_scale` | `attention.temperature_length` | 0 | all | `deepseek2.cpp:46-49,457-460,595-598,632-635` |
//! | `llama4` | literal `0.1` | literal `8192` | literal `1.0` | the NON-ROPE layers only, and only on its chunked-SWA branch | `llama4.cpp:15-17,122-123,175-176` |
//!
//! Two other hits are NOT this feature and are recorded so nobody
//! re-derives it: `grok.cpp:23` reads `attention.temperature_length`
//! into `hparams.attn_temp_length` and applies it nowhere
//! (`crate::scalar_multipliers` already says so), and `dflash.cpp:133`
//! / `deepseek4.cpp:124` name a `blk.N.hc_attn_scale` TENSOR, which is
//! a hyper-connection weight. `plamo3.cpp:140` is a local variable.
//!
//! So `mistral3` is the ONLY generic-path reader of a KEY, and
//! [`resolve_attn_temperature`] is written for the two that read one.
//! `llama4` seeds LITERALS ([`LLAMA4`]) on the
//! branch every real export takes (`llama4.cpp:13-17`; the other
//! branch is refused by name in `crate::chunked_swa`) and gates the
//! multiply on `!use_rope` (`:175` is an `else if` on the RoPE branch),
//! so [`AttnTemperature::unrotated_layers_only`] is the per-layer
//! variant, read against `ModelConfig::layer_rotates` by the one
//! helper. It landed with its one caller, as the OLMo lesson says a
//! variant should.
//!
//! # The floor is `n_ctx_orig_yarn`, which is NOT a key
//!
//! `mistral3.cpp:15` takes the floor from `hparams.n_ctx_orig_yarn`,
//! and `llama-model.cpp:1164-1165` seeds that from `n_ctx_train`
//! (`{arch}.context_length`, REQUIRED) before letting
//! `rope.scaling.original_context_length` override it. So a Ministral
//! file without the YaRN key floors on its context length, not on
//! nothing, and [`resolve_attn_temperature`] takes the already-resolved
//! value rather than the key. `mistral3.cpp:16-17` then throws on a
//! zero floor; frink refuses the same file.
//!
//! # Why a nonzero key on any other architecture is IGNORED here
//!
//! `build_qkv` and `build_attn` never look at `f_attn_temp_scale`; only
//! the three graphs above build the input. A `llama` file carrying
//! `llama.attention.temperature_scale = 0.5` runs unscaled in llama.cpp
//! and must run unscaled here, and
//! `a_nonzero_key_on_an_architecture_whose_graph_never_reads_it_is_dead_metadata`
//! pins that. The defect this module closes is the other direction:
//! before it, a `mistral3` file carrying the key loaded and ran at the
//! wrong temperature with no error.
//!
//! # Where it is applied
//!
//! ONE helper, `Decoder::apply_attn_temperature`, called from the CPU
//! row body and both batched host bodies at the point
//! `attention_scale` is applied (after RoPE and the post-RoPE QK-norm,
//! before attention). No fused Metal launch has a per-token Q scale
//! uniform, so `Decoder::metal_can_serve_model` keeps a model with one
//! on the host bodies -- the same fence `clamp_kqv` and
//! `residual_scale` share, for the same reason.
//!
//! The MLA engine (`deepseek2` / `mistral4`) REFUSES a file that
//! declares a nonzero scale rather than implementing it: that engine
//! has no libllama-golden fixture at all, so an implementation there
//! would be a guess with nothing to check it against.

use std::num::NonZeroU32;

/// llama.cpp's three attention-temperature hyper-parameters, for a
/// model whose graph applies them.
///
/// Only ever `Some` in a `ModelConfig` when `scale != 0`, which is
/// llama.cpp's own gate (`mistral3.cpp:14,109`): a zero scale builds
/// no input tensor there and builds no value here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnTemperature {
    /// `hparams.f_attn_temp_scale`, the multiplier on the log term.
    pub scale: f32,
    /// `hparams.n_attn_temp_floor_scale`, the position period at which
    /// the log's argument steps. `llama-graph.cpp:160` asserts it is
    /// nonzero, so the type says so.
    pub floor_scale: NonZeroU32,
    /// `hparams.f_attn_temp_offset`, added to the position before the
    /// division. `0.0` for both key-driven readers; `llama4` uses `1.0`.
    pub offset: f32,
    /// Multiply Q on the layers that do NOT rotate and leave the
    /// rotating ones alone: `llama4.cpp:163-177` is `if (use_rope) {
    /// rope } else if (inp_attn_scale) { mul }`. The two key-driven
    /// readers scale every layer (`mistral3.cpp:153-156` has no
    /// branch).
    pub unrotated_layers_only: bool,
}

impl AttnTemperature {
    /// The multiplier for a token at `pos`, in the precision
    /// `llama-graph.cpp:163-167` computes it.
    ///
    /// Transcribed rather than simplified: `pos` is a `float` there,
    /// the offset add and the division by the (integer-promoted) floor
    /// are single precision, `std::floor` keeps single precision, and
    /// the `+ 1.0`, `std::log`, `* scale` and `+ 1.0` are double before
    /// the store to `float`. Evaluating it all in `f64` would agree to
    /// ~1e-7 on any real file and disagree by a whole step on a position
    /// that lands exactly on a floor boundary after single-precision
    /// rounding.
    #[inline]
    pub fn scale_at(self, pos: usize) -> f32 {
        let pos = pos as f32;
        let floored = ((pos + self.offset) / (self.floor_scale.get() as f32)).floor();
        ((f64::from(floored) + 1.0).ln() * f64::from(self.scale) + 1.0) as f32
    }

    /// Multiplies each row of a `[rows, q_width]` Q batch by that row's
    /// temperature.
    ///
    /// One body for the row path (`rows == 1`) and both batched bodies,
    /// taking the position as a function of the row so that the prefill
    /// body's `start_pos + b` and the multi-sequence body's `positions[b]`
    /// are two callers of one loop rather than two loops.
    pub fn apply_rows(self, q: &mut [f32], q_width: usize, pos_of_row: impl Fn(usize) -> usize) {
        for (b, row) in q.chunks_mut(q_width).enumerate() {
            let s = self.scale_at(pos_of_row(b));
            // `scale_at` is exactly 1.0 below the first floor step
            // (`ln(1) * scale + 1`), which is every position of a
            // prompt shorter than `floor_scale`; skipping the multiply
            // there is a no-op made cheaper, not a different answer.
            if s != 1.0 {
                for v in row.iter_mut() {
                    *v *= s;
                }
            }
        }
    }
}

/// Where a reader's floor comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloorSource {
    /// `hparams.n_ctx_orig_yarn` -- `rope.scaling.original_context_length`
    /// with `context_length` as its default (`llama-model.cpp:1164-1165`).
    /// `mistral3.cpp:15`.
    OrigCtxYarn,
    /// `{arch}.attention.temperature_length`, read as optional and
    /// defaulting to 0 -- which `llama-graph.cpp:160` then aborts on.
    /// `deepseek2.cpp:47`.
    TemperatureLengthKey,
}

/// Every architecture whose graph multiplies Q by the temperature
/// input, with where its floor comes from and the line that applies it.
///
/// This is the CENSUS and [`resolve_attn_temperature`] is the
/// implementation; `every_key_driven_reader_resolves_to_a_temperature`
/// checks one against the other so a name added to one without the
/// other fails. `llama4` is absent because it reads no key (see the
/// module doc); `capability::unaudited_triage` carries it.
pub const ATTN_TEMPERATURE_READERS: &[(&str, FloorSource, &str)] = &[
    (
        "mistral3",
        FloorSource::OrigCtxYarn,
        "src/models/mistral3.cpp:5,14-17,153-156",
    ),
    (
        "deepseek2",
        FloorSource::TemperatureLengthKey,
        "src/models/deepseek2.cpp:46-49,595-598,632-635",
    ),
    (
        "mistral4",
        FloorSource::TemperatureLengthKey,
        "src/models/models.h:1311-1318 (reuses deepseek2's hparams and graph)",
    ),
];

/// `llama4.cpp:15-17,175-176`: a LITERAL temperature, read from no key
/// (scale 0.1 over a floor of 8192, offset 1, on the unrotated layers
/// only). `crate::models::llama4` states it; it is the one such graph.
pub const LLAMA4: AttnTemperature = AttnTemperature {
    scale: 0.1,
    floor_scale: NonZeroU32::new(8192).unwrap(),
    offset: 1.0,
    unrotated_layers_only: true,
};

/// What the file declares, gathered by the loader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclaredTemperature {
    /// `{arch}.attention.temperature_scale`.
    pub scale: Option<f32>,
    /// `{arch}.attention.temperature_length`.
    pub length: Option<u64>,
    /// `hparams.n_ctx_orig_yarn` as llama.cpp resolves it: the YaRN
    /// original-context key, else `context_length`.
    pub n_ctx_orig_yarn: Option<u64>,
}

/// Why a declared temperature cannot be honoured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnTemperatureError {
    /// The floor resolved to zero or is absent, which llama.cpp refuses
    /// at load (`mistral3.cpp:16-17`) or aborts on at the first batch
    /// (`llama-graph.cpp:160`).
    ZeroFloor(FloorSource),
    /// The floor does not fit the `uint32_t` llama.cpp stores it in.
    FloorTooLarge(u64),
}

impl AttnTemperatureError {
    /// The sentence the loader puts in its error.
    pub fn message(&self, arch: &str) -> String {
        match self {
            AttnTemperatureError::ZeroFloor(FloorSource::OrigCtxYarn) => format!(
                "`{arch}.attention.temperature_scale` is nonzero but the floor it divides \
                 positions by -- `{arch}.rope.scaling.original_context_length`, else \
                 `{arch}.context_length` (llama-model.cpp:1164-1165) -- is zero or absent; \
                 llama.cpp refuses the same file (`invalid n_ctx_orig_yarn for attention \
                 temperature scaling`, src/models/mistral3.cpp:16-17)"
            ),
            AttnTemperatureError::ZeroFloor(FloorSource::TemperatureLengthKey) => format!(
                "`{arch}.attention.temperature_scale` is nonzero but \
                 `{arch}.attention.temperature_length` is zero or absent; llama.cpp reads \
                 the floor from that key (src/models/deepseek2.cpp:47) and aborts on a zero \
                 one at the first batch (llama-graph.cpp:160)"
            ),
            AttnTemperatureError::FloorTooLarge(v) => format!(
                "`{arch}`'s attention-temperature floor {v} does not fit the uint32 \
                 llama.cpp stores `n_attn_temp_floor_scale` in"
            ),
        }
    }
}

/// The temperature the graph applies for `arch`, from what the file
/// declares.
///
/// `Ok(None)` is "no temperature": an architecture whose graph never
/// builds the input (every one not in [`ATTN_TEMPERATURE_READERS`],
/// whatever the file says), or a reader whose scale is absent or
/// exactly zero (`mistral3.cpp:14` and `deepseek2.cpp:458` both test
/// `!= 0.0f`).
pub fn resolve_attn_temperature(
    arch: &str,
    declared: DeclaredTemperature,
) -> Result<Option<AttnTemperature>, AttnTemperatureError> {
    resolve_attn_temperature_for(
        arch,
        crate::models::ArchFacts::of(arch)
            .scaling
            .attn_temperature_literal,
        declared,
    )
}

/// [`resolve_attn_temperature`] given the literal row (`ArchFacts`);
/// the key readers ([`ATTN_TEMPERATURE_READERS`]) are still keyed on
/// `arch`.
pub fn resolve_attn_temperature_for(
    arch: &str,
    literal: Option<AttnTemperature>,
    declared: DeclaredTemperature,
) -> Result<Option<AttnTemperature>, AttnTemperatureError> {
    // The literal seeders first: the file's keys are not read for them
    // (`llama4.cpp:3-26` reads no `temperature_*` key), so a declared
    // value is dead metadata as it is on every non-reader.
    if let Some(literal) = literal {
        return Ok(Some(literal));
    }
    let Some((_, floor_source, _)) = ATTN_TEMPERATURE_READERS
        .iter()
        .find(|(name, _, _)| *name == arch)
    else {
        return Ok(None);
    };
    let scale = match declared.scale {
        Some(s) if s != 0.0 => s,
        _ => return Ok(None),
    };
    let floor = match floor_source {
        FloorSource::OrigCtxYarn => declared.n_ctx_orig_yarn,
        FloorSource::TemperatureLengthKey => declared.length,
    }
    .unwrap_or(0);
    let floor = u32::try_from(floor).map_err(|_| AttnTemperatureError::FloorTooLarge(floor))?;
    let floor_scale =
        NonZeroU32::new(floor).ok_or(AttnTemperatureError::ZeroFloor(*floor_source))?;
    Ok(Some(AttnTemperature {
        scale,
        floor_scale,
        // Both key-driven readers assign 0.0 (`mistral3.cpp:11`,
        // `deepseek2.cpp:49`); only llama4's literal is 1.0.
        offset: 0.0,
        unrotated_layers_only: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(scale: f32, floor: u32) -> AttnTemperature {
        AttnTemperature {
            scale,
            floor_scale: NonZeroU32::new(floor).unwrap(),
            offset: 0.0,
            unrotated_layers_only: false,
        }
    }

    /// The literal seeder answers its literals whatever the file says,
    /// including a declared key it never reads, and no key-driven
    /// reader is in the literal table.
    #[test]
    fn llama4_takes_its_literals_and_ignores_a_declared_key() {
        let declared = DeclaredTemperature {
            scale: Some(0.5),
            length: Some(16),
            n_ctx_orig_yarn: Some(16),
        };
        let got = resolve_attn_temperature("llama4", declared)
            .unwrap()
            .expect("llama4 always scales");
        assert_eq!(got, LLAMA4);
        assert!(got.unrotated_layers_only);
        assert_eq!(got.scale_at(8190), 1.0);
        assert!(got.scale_at(8191) > 1.0);
        for (name, _, _) in ATTN_TEMPERATURE_READERS {
            assert!(
                crate::models::ArchFacts::of(name)
                    .scaling
                    .attn_temperature_literal
                    .is_none(),
                "{name} reads a key and cannot also seed a literal"
            );
        }
    }

    /// The formula, position by position, against a hand evaluation of
    /// `llama-graph.cpp:165-167`: exactly 1 below the first floor, then
    /// `1 + scale * ln(k + 1)` on the k-th period.
    #[test]
    fn the_scale_steps_at_every_floor_boundary_and_is_one_before_the_first() {
        let t = temp(0.5, 2);
        for pos in 0..2 {
            assert_eq!(t.scale_at(pos), 1.0, "position {pos} is below the floor");
        }
        for pos in 2..4 {
            let want = (2f64.ln() * 0.5 + 1.0) as f32;
            assert_eq!(t.scale_at(pos), want, "position {pos} is on period 1");
        }
        for pos in 4..6 {
            let want = (3f64.ln() * 0.5 + 1.0) as f32;
            assert_eq!(t.scale_at(pos), want, "position {pos} is on period 2");
        }
        // Not monotone-trivial: a sign error on the scale would still
        // step, so pin the direction.
        assert!(t.scale_at(5) > t.scale_at(3) && t.scale_at(3) > t.scale_at(1));
    }

    /// Llama-4's literal offset: `floor((pos + 1) / 8192)` steps one
    /// position EARLIER than `floor(pos / 8192)`, at 8191 rather than
    /// 8192. The offset is a field so that the day llama4 is served the
    /// value is a table entry and not a second formula.
    #[test]
    fn the_offset_moves_the_step_by_one_position() {
        let no_offset = temp(0.1, 8192);
        let llama4 = AttnTemperature {
            offset: 1.0,
            ..no_offset
        };
        assert_eq!(no_offset.scale_at(8191), 1.0);
        assert!(llama4.scale_at(8191) > 1.0, "llama4 steps at 8191");
        assert_eq!(llama4.scale_at(8191), no_offset.scale_at(8192));
    }

    /// `apply_rows` scales every channel of a row by that ROW's
    /// position, and a row below the floor is bit-identical.
    #[test]
    fn apply_rows_scales_each_row_by_its_own_position() {
        let t = temp(0.5, 2);
        let mut q = vec![1.0f32; 3 * 4];
        // Rows at positions 1, 2, 5: below the floor, period 1, period 2.
        let positions = [1usize, 2, 5];
        t.apply_rows(&mut q, 4, |b| positions[b]);
        assert_eq!(&q[..4], &[1.0; 4], "position 1 is untouched");
        for v in &q[4..8] {
            assert_eq!(*v, t.scale_at(2));
        }
        for v in &q[8..] {
            assert_eq!(*v, t.scale_at(5));
        }
        assert_ne!(
            q[4], q[8],
            "the two scaled rows must differ, or this saw one scale"
        );
    }

    /// The census and the resolver agree: every key-driven reader gets
    /// a temperature from a file that declares one, and the floor comes
    /// from the source the census names.
    #[test]
    fn every_key_driven_reader_resolves_to_a_temperature() {
        for (arch, source, line) in ATTN_TEMPERATURE_READERS {
            let declared = DeclaredTemperature {
                scale: Some(0.25),
                length: Some(64),
                n_ctx_orig_yarn: Some(4096),
            };
            let got = resolve_attn_temperature(arch, declared)
                .unwrap_or_else(|e| panic!("{arch} ({line}): {e:?}"))
                .unwrap_or_else(|| panic!("{arch} ({line}) reads the key and must resolve"));
            let want_floor = match source {
                FloorSource::OrigCtxYarn => 4096,
                FloorSource::TemperatureLengthKey => 64,
            };
            assert_eq!(got.floor_scale.get(), want_floor, "{arch}'s floor source");
            assert_eq!(got.scale, 0.25);
            assert_eq!(got.offset, 0.0);
        }
    }

    /// The other direction, and the one the oracle decides: only three
    /// graphs build the input, so the key on any other architecture is
    /// dead metadata in llama.cpp and stays dead here.
    #[test]
    fn a_nonzero_key_on_an_architecture_whose_graph_never_reads_it_is_dead_metadata() {
        let declared = DeclaredTemperature {
            scale: Some(0.5),
            length: Some(64),
            n_ctx_orig_yarn: Some(4096),
        };
        for arch in ["llama", "qwen3", "gemma3", "olmo2", "granite", "smollm3"] {
            assert_eq!(
                resolve_attn_temperature(arch, declared),
                Ok(None),
                "{arch} has no build_inp_attn_scale in src/models/"
            );
        }
    }

    /// `!= 0.0f` is llama.cpp's gate on both readers: an absent key and
    /// an explicit zero both build no input.
    #[test]
    fn a_zero_or_absent_scale_is_no_temperature() {
        for scale in [None, Some(0.0)] {
            let declared = DeclaredTemperature {
                scale,
                length: Some(64),
                n_ctx_orig_yarn: Some(4096),
            };
            assert_eq!(resolve_attn_temperature("mistral3", declared), Ok(None));
            assert_eq!(resolve_attn_temperature("deepseek2", declared), Ok(None));
        }
    }

    /// `mistral3.cpp:15` floors on `n_ctx_orig_yarn` and NOT on
    /// `attention.temperature_length`; `deepseek2.cpp:47` the reverse.
    /// A file declaring only the other one's floor is refused, as
    /// llama.cpp refuses (`mistral3`) or aborts on (`deepseek2`) it.
    #[test]
    fn each_reader_takes_its_own_floor_and_refuses_the_other_ones() {
        let only_length = DeclaredTemperature {
            scale: Some(0.5),
            length: Some(64),
            n_ctx_orig_yarn: None,
        };
        assert_eq!(
            resolve_attn_temperature("mistral3", only_length),
            Err(AttnTemperatureError::ZeroFloor(FloorSource::OrigCtxYarn))
        );
        let only_ctx = DeclaredTemperature {
            scale: Some(0.5),
            length: None,
            n_ctx_orig_yarn: Some(4096),
        };
        assert_eq!(
            resolve_attn_temperature("deepseek2", only_ctx),
            Err(AttnTemperatureError::ZeroFloor(
                FloorSource::TemperatureLengthKey
            ))
        );
        // And an explicit zero floor is the same refusal, not a
        // division by zero.
        let zero_len = DeclaredTemperature {
            length: Some(0),
            ..only_length
        };
        assert_eq!(
            resolve_attn_temperature("deepseek2", zero_len),
            Err(AttnTemperatureError::ZeroFloor(
                FloorSource::TemperatureLengthKey
            ))
        );
        // Each message names the key the user has to look at.
        let msg = AttnTemperatureError::ZeroFloor(FloorSource::OrigCtxYarn).message("mistral3");
        assert!(
            msg.contains("mistral3.rope.scaling.original_context_length"),
            "{msg}"
        );
        assert!(msg.contains("mistral3.cpp:16-17"), "{msg}");
        let msg =
            AttnTemperatureError::ZeroFloor(FloorSource::TemperatureLengthKey).message("deepseek2");
        assert!(
            msg.contains("deepseek2.attention.temperature_length"),
            "{msg}"
        );
    }
}
