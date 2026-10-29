//! **Which layers slide.** llama.cpp's `hparams.is_swa_impl[il]`, as
//! one value a `ModelConfig` carries, with the per-layer ARRAY form of
//! `{arch}.attention.sliding_window_pattern` read the way each
//! architecture's graph reads it.
//!
//! Until 2026-09-11 frink carried the layout as a scalar period plus a
//! phase, read the key as a scalar, and REFUSED the array form for every
//! architecture. llama.cpp reads the key with `get_key_or_arr`, and that
//! name hides THREE behaviours, decided per architecture by which
//! overload the graph's `load_arch_hparams` calls:
//!
//! | mode | call | scalar in the file | ARRAY in the file |
//! |---|---|---|---|
//! | [`PatternKeyRead::ScalarPeriod`] | `get_key_or_arr(kid, swa_period, false)` | overrides the seeded period | **IGNORED**: the scalar overload returns `false` on an array when `required` is false (`llama-model-loader.cpp:490-512`), and the seeded period stands |
//! | [`PatternKeyRead::PerLayerBool`] | `get_arr(kid, hparams.is_swa_impl)` (required) | **refused**: `array key not found in model` (`:356-360`) | the per-layer truth at any length up to `LLAMA_MAX_LAYERS` (below) |
//! | [`PatternKeyRead::ScalarThenArray`] | the first, then the second on `false` | a period | the per-layer truth |
//!
//! The census is measured, not remembered: `grep -n
//! LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN src/models/*.cpp` over all
//! 155 graphs. Twenty-two read the key. Five read only the array
//! ([`PER_LAYER_ARRAY_READERS`]), two try the scalar and fall back to
//! the array ([`SCALAR_THEN_ARRAY_READERS`]), and the other fifteen read
//! only the scalar. An architecture that reads the key in NO form
//! (`llama`, `qwen2`, ...) is treated as the scalar-only mode here,
//! which is what the loader did before: a scalar period in such a file
//! is honoured, an array is ignored.
//!
//! **The IGNORED cell is the one that matters for real files.** The
//! converters write the array for `exaone4`, `exaone-moe`
//! (`conversion/exaone.py:84`), `olmo2` (`olmo.py:59-66`) and `gemma3n`
//! (`gemma.py:532-535`), and all four graphs read the SCALAR overload,
//! so every real EXAONE-4 32B, EXAONE-MoE and Olmo-3 export carries an
//! array that llama.cpp never looks at and runs on the literal period
//! (`exaone4.cpp:7`, `exaone-moe.cpp:6`, `olmo2.cpp:9`). frink refused
//! all of them. That the array and the literal AGREE for every real
//! checkpoint is why upstream gets away with it; `tests/
//! window_array_graphs.rs` measures that libllama's logits do not move
//! when the array is rewritten to DISAGREE, and frink matches both.
//!
//! **The array's length is not checked.** At the pin every array read
//! goes through `get_arr` into the fixed `is_swa_impl` array
//! (`mimo2.cpp:12`, `step35.cpp:26`, `maple.cpp:10`, ..., and
//! `load_swa_pattern`, `llama-model.cpp:3952-3959`). `get_arr`
//! (`llama-model-loader.cpp:352-405`) checks the element type and
//! `length <= LLAMA_MAX_LAYERS` (`:382-384`, a throw past it), copies
//! the file's entries from index 0, and leaves every later entry at the
//! zero `llama-model.cpp:1465` filled it with. So entry `il` is layer
//! `il`'s flag, a SHORTER array leaves the remaining layers full
//! attention, and a LONGER one is read past the trunk by no graph
//! (`is_swa` is asked only for `il < n_layer()`). [`read_swa_layers`]
//! does exactly that (#452). The converters write `block_count` entries
//! (`mimo.py:146-153`, `step3.py:164-173`) or one per trunk layer
//! (`command_r.py:98`), and both read the same here.
//!
//! A SCALAR where only `get_arr` reads the key is `array key not found
//! in model` (`:356-360`, `required` defaults to true), refused here.

use std::num::NonZeroUsize;
use std::sync::Arc;

use frink_gguf::{GgufValue, TensorSource};

use crate::capability::SwaPattern;
use crate::loader::LoadError;
use crate::mtp_blocks::TrunkLayers;

/// llama.cpp's `is_swa_impl`, as a rule or as the array itself.
///
/// Three spellings, one accessor ([`Self::slides`]). Every consumer --
/// the CPU attention mask, the KV block layout, the fused Metal
/// launches, the per-layer RoPE gate -- asks `ModelConfig::
/// layer_sliding_window(il)`, which asks this. Replacing the two fields
/// it used to be (`swa_pattern: Option<usize>`, `swa_dense_first:
/// bool`) with one enum is what makes a fourth spelling a compile error
/// at every match rather than a silently unhandled case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwaLayers {
    /// Every layer slides. llama.cpp's `set_swa_pattern(0)`
    /// (`deepseek4.cpp:68`, `dflash.cpp:54`), and frink's answer for a
    /// declared window on an architecture with no seeded period.
    All,
    /// `set_swa_pattern(period, dense_first)`, `llama-hparams.cpp:8-22`:
    ///
    /// - `dense_first = false`: `is_swa[il] = il % p < p - 1`, the LAST
    ///   layer of every period is full attention;
    /// - `dense_first = true`: `is_swa[il] = il % p != 0`, the FIRST.
    ///
    /// A period of 1 windows NOTHING under either phase, which is
    /// `phi3.cpp:23`'s spelling and the opposite of [`Self::All`]; it
    /// is why the period is `NonZeroUsize` rather than `usize` with a
    /// zero that means "all".
    Period {
        period: NonZeroUsize,
        dense_first: bool,
    },
    /// The file's own per-layer answer, one entry per TRUNK layer.
    /// Indexing past the end answers `false`, which is what
    /// `set_swa_pattern` writes for every layer past `n_layer()`.
    PerLayer(Arc<[bool]>),
}

impl SwaLayers {
    /// `set_swa_pattern(period, dense_first)` with llama.cpp's own
    /// degenerate case folded in: a period of 0 is [`Self::All`].
    pub fn period(period: usize, dense_first: bool) -> Self {
        match NonZeroUsize::new(period) {
            Some(period) => Self::Period {
                period,
                dense_first,
            },
            None => Self::All,
        }
    }

    /// The seeded layout for an architecture, or every layer when it
    /// seeds none.
    pub fn from_default(layout: Option<SwaPattern>) -> Self {
        match layout {
            Some(p) => Self::period(p.period, p.dense_first),
            None => Self::All,
        }
    }

    /// Does layer `layer_idx` slide? llama.cpp's `hparams.is_swa(il)`
    /// for a model that has a window at all; the window's presence is
    /// `ModelConfig::sliding_window`'s question, not this one's.
    #[inline]
    pub fn slides(&self, layer_idx: usize) -> bool {
        match self {
            Self::All => true,
            Self::Period {
                period,
                dense_first,
            } => {
                let period = period.get();
                if *dense_first {
                    !layer_idx.is_multiple_of(period)
                } else {
                    layer_idx % period < period - 1
                }
            }
            Self::PerLayer(layers) => layers.get(layer_idx).copied().unwrap_or(false),
        }
    }
}

/// Which `get_key_or_arr` overload an architecture's `load_arch_hparams`
/// reads `{arch}.attention.sliding_window_pattern` through. See the
/// module doc for what each does with each shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKeyRead {
    /// `get_key_or_arr(kid, uint32_t & swa_period, false)`: scalar
    /// honoured as the period, array ignored.
    ScalarPeriod,
    /// `get_arr(kid, hparams.is_swa_impl)`: array honoured, a scalar
    /// or an absent key refused (REQUIRED).
    PerLayerBool,
    /// The scalar overload first, the array on its `false`. The array
    /// read is REQUIRED, so an absent key refuses.
    ScalarThenArray,
}

/// Every graph that reads the key ONLY as the per-layer array, with the
/// line. `gemma4` and `gemma4-assistant` run on their own engine
/// (`gemma4_gguf_loader.rs` has read the array since it existed);
/// `step35` and `mimo2` closed on other seams and run on this one;
/// `dflash` is deferred.
pub const PER_LAYER_ARRAY_READERS: &[(&str, &str)] = &[
    ("gemma4", "src/models/gemma4.cpp:5"),
    ("gemma4-assistant", "src/models/gemma4-assistant.cpp:7"),
    ("dflash", "src/models/dflash.cpp:69"),
    ("step35", "src/models/step35.cpp:26"),
    ("mimo2", "src/models/mimo2.cpp:12"),
    // Landed upstream after the 2026-08-04 pin; `ml.get_arr` with no
    // scalar attempt, like the four above.
    ("spark2_5", "src/models/spark2-5.cpp:8"),
    ("maple", "src/models/maple.cpp:8"),
    ("granite_swa", "src/models/granite-swa.cpp:17"),
    // Landed upstream after the 2026-09-19 pin.
    ("lfm3", "src/models/lfm3.cpp:13"),
    // Also after it. `crate::models::kolibri1` cannot state this row:
    // `ArchFacts` does not carry the pattern read yet.
    ("kolibri1", "src/models/kolibri1.cpp:17"),
];

/// `LLAMA_MAX_LAYERS` (`llama-hparams.h:11`): the longest array `get_arr`
/// accepts into `is_swa_impl`.
pub const LLAMA_MAX_LAYERS: usize = 512;

/// Every graph that tries the scalar and falls back to the array, with
/// the line.
pub const SCALAR_THEN_ARRAY_READERS: &[(&str, &str)] = &[
    ("mellum", "src/models/mellum.cpp:12-17"),
    ("cohere2moe", "src/models/cohere2moe.cpp:32-36"),
    // `muse-glimmer.cpp:26` calls `load_swa_pattern(ml, 4)`, which
    // tries the ARRAY first (`llama-model.cpp:3309`) and falls back to
    // the scalar and then to a seeded period of 4 with `dense_first =
    // false`. Array-first versus scalar-first is not observable: one
    // key holds either a scalar or an array, and each overload returns
    // false for the other's spelling.
    ("muse-glimmer", "src/models/muse-glimmer.cpp:26"),
];

/// THE table. Derived from the two censuses above rather than restated
/// beside them: an architecture in neither is the scalar mode, which is
/// also what an architecture that never reads the key gets.
pub fn pattern_key_read(arch: &str) -> PatternKeyRead {
    if PER_LAYER_ARRAY_READERS.iter().any(|(a, _)| *a == arch) {
        PatternKeyRead::PerLayerBool
    } else if SCALAR_THEN_ARRAY_READERS.iter().any(|(a, _)| *a == arch) {
        PatternKeyRead::ScalarThenArray
    } else {
        PatternKeyRead::ScalarPeriod
    }
}

/// Reads `{arch}.attention.sliding_window_pattern` the way `arch`'s
/// graph does, and falls back to `seeded` -- the architecture's literal
/// period from `capability::default_swa_layout`, or the family default
/// -- where llama.cpp would keep its seed.
///
/// Called only for a file that HAS a window: for one that does not, no
/// graph consults `is_swa` and the answer is irrelevant.
///
/// `trunk` is here for the array length. See the module doc.
pub fn read_swa_layers(
    file: &impl TensorSource,
    arch: &str,
    key: &str,
    trunk: &TrunkLayers,
    seeded: Option<SwaPattern>,
) -> Result<SwaLayers, LoadError> {
    let value = file.metadata(key);
    let period_from_scalar = |v: &GgufValue| -> Result<SwaLayers, LoadError> {
        let period = v.as_u64().ok_or_else(|| {
            LoadError::UnsupportedFeature(
                arch.to_string(),
                format!("{key} is neither an unsigned integer nor an array: {v:?}"),
            )
        })?;
        Ok(SwaLayers::period(
            period as usize,
            seeded.is_some_and(|p| p.dense_first),
        ))
    };
    let per_layer = |items: &[GgufValue]| -> Result<SwaLayers, LoadError> {
        // `get_arr` (`llama-model-loader.cpp:382-384`): no length check
        // but the cap; entry `il` is layer `il`, and a layer past the
        // array's end keeps the zero seed (module doc).
        if items.len() > LLAMA_MAX_LAYERS {
            return Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "{key} has {} entries; llama.cpp refuses an array longer than \
                     LLAMA_MAX_LAYERS = {LLAMA_MAX_LAYERS} (llama-model-loader.cpp:382-384)",
                    items.len()
                ),
            ));
        }
        let mut out = vec![false; trunk.n_layers];
        for (il, item) in items.iter().take(trunk.n_layers).enumerate() {
            // `get_arr` admits BOOL, UINT32 and INT32 arrays into the
            // `uint32_t` layer array (`:368-371`) and tests a bool entry
            // as `x != 0` (`:394-401`); `as_bool` is that rule for every
            // integer width.
            out[il] = item.as_bool().ok_or_else(|| {
                LoadError::UnsupportedFeature(
                    arch.to_string(),
                    format!("{key} entry {il} is not a bool or integer: {item:?}"),
                )
            })?;
        }
        Ok(SwaLayers::PerLayer(out.into()))
    };
    match pattern_key_read(arch) {
        PatternKeyRead::ScalarPeriod => match value {
            // `get_key_or_arr(kid, swa_period, false)` returns false on
            // an array and the seed stands. NOT a refusal, because every
            // real EXAONE-4 32B / EXAONE-MoE / Olmo-3 export carries one;
            // see the module doc.
            None | Some(GgufValue::Array(_)) => Ok(SwaLayers::from_default(seeded)),
            Some(scalar) => period_from_scalar(scalar),
        },
        PatternKeyRead::ScalarThenArray => match value {
            // The array read is REQUIRED on the fallback path
            // (`mellum.cpp:16`, `cohere2moe.cpp:35` pass no `required`),
            // so an absent key is llama.cpp's `key not found in model`.
            None => Err(LoadError::MissingHparam(key.to_string())),
            Some(GgufValue::Array(items)) => per_layer(items),
            Some(scalar) => period_from_scalar(scalar),
        },
        PatternKeyRead::PerLayerBool => match value {
            // REQUIRED: the array overload defaults `required` to true
            // and `mimo2.cpp:12` / `step35.cpp:26` pass nothing.
            None => Err(LoadError::MissingHparam(key.to_string())),
            Some(GgufValue::Array(items)) => per_layer(items),
            // `get_arr` refuses anything but an array
            // (`llama-model-loader.cpp:356-360`): `array key not found in
            // model`. The graphs pass no `required`, so it throws.
            Some(scalar) => Err(LoadError::UnsupportedFeature(
                arch.to_string(),
                format!(
                    "{key} is a scalar ({scalar:?}), and llama.cpp reads it for this \
                     architecture only as an array (`get_arr`, required); a file with a scalar \
                     does not load upstream"
                ),
            )),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trunk(block_count: usize, n_layers: usize) -> TrunkLayers {
        TrunkLayers {
            block_count,
            n_layers,
            n_mtp_blocks: block_count - n_layers,
        }
    }

    struct Meta(Vec<(String, GgufValue)>);
    impl TensorSource for Meta {
        fn metadata(&self, key: &str) -> Option<&GgufValue> {
            self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }
        fn find_tensor(&self, _: &str) -> Option<&frink_gguf::TensorInfo> {
            None
        }
        fn tensor_bytes(&self, name: &str) -> Result<&[u8], frink_gguf::GgufError> {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<(Arc<frink_gguf::MmapHandle>, std::ops::Range<usize>), frink_gguf::GgufError>
        {
            Err(frink_gguf::GgufError::TensorNotFound(name.to_string()))
        }
    }

    const KEY: &str = "x.attention.sliding_window_pattern";

    fn with(value: Option<GgufValue>) -> Meta {
        Meta(value.into_iter().map(|v| (KEY.to_string(), v)).collect())
    }

    fn bools(v: &[bool]) -> GgufValue {
        GgufValue::Array(v.iter().map(|&b| GgufValue::Bool(b)).collect())
    }

    const LAST_DENSE_4: Option<SwaPattern> = Some(SwaPattern {
        period: 4,
        dense_first: false,
    });
    const DENSE_FIRST_4: Option<SwaPattern> = Some(SwaPattern {
        period: 4,
        dense_first: true,
    });

    /// The two phases of `set_swa_pattern` and its two degenerate
    /// periods, layer by layer against `llama-hparams.cpp:8-22`.
    #[test]
    fn period_matches_set_swa_pattern_in_both_phases() {
        let last_dense = SwaLayers::period(4, false);
        let dense_first = SwaLayers::period(4, true);
        let got: Vec<(bool, bool)> = (0..8)
            .map(|il| (last_dense.slides(il), dense_first.slides(il)))
            .collect();
        let want: Vec<(bool, bool)> = (0..8u32).map(|il| (il % 4 < 3, il % 4 != 0)).collect();
        assert_eq!(got, want);
        assert_eq!(SwaLayers::period(0, false), SwaLayers::All);
        assert!((0..8).all(|il| SwaLayers::All.slides(il)));
        assert!((0..8).all(|il| !SwaLayers::period(1, false).slides(il)));
        assert!((0..8).all(|il| !SwaLayers::period(1, true).slides(il)));
    }

    /// Past the array's end is "does not slide", which is what
    /// `set_swa_pattern` writes for `il >= n_layer()`.
    #[test]
    fn per_layer_answers_false_past_the_trunk() {
        let layers = SwaLayers::PerLayer(vec![true, false].into());
        assert!(layers.slides(0));
        assert!(!layers.slides(1));
        assert!(!layers.slides(2));
    }

    /// The census and the table are one thing: every listed reader
    /// answers its listed mode, and an unlisted name answers the
    /// scalar mode.
    #[test]
    fn every_listed_reader_answers_its_mode() {
        for (arch, _) in PER_LAYER_ARRAY_READERS {
            assert_eq!(
                pattern_key_read(arch),
                PatternKeyRead::PerLayerBool,
                "{arch}"
            );
        }
        for (arch, _) in SCALAR_THEN_ARRAY_READERS {
            assert_eq!(
                pattern_key_read(arch),
                PatternKeyRead::ScalarThenArray,
                "{arch}"
            );
        }
        for arch in ["exaone4", "exaone-moe", "olmo2", "gemma3", "llama"] {
            assert_eq!(
                pattern_key_read(arch),
                PatternKeyRead::ScalarPeriod,
                "{arch}"
            );
        }
    }

    /// The IGNORED cell: an array in a scalar-mode file leaves the
    /// seeded period standing, even when the array DISAGREES with it.
    /// This is the EXAONE / Olmo-3 over-refusal, lifted.
    #[test]
    fn a_scalar_mode_architecture_ignores_the_array_and_keeps_its_seed() {
        // Disagrees with last-dense-4 on every layer.
        let file = with(Some(bools(&[false, false, false, true])));
        let got = read_swa_layers(&file, "exaone-moe", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
        assert_eq!(got, SwaLayers::period(4, false));
        // And an absent key is the same answer.
        let got =
            read_swa_layers(&with(None), "exaone-moe", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
        assert_eq!(got, SwaLayers::period(4, false));
        // A scalar still overrides the seed, keeping the seed's phase.
        let file = with(Some(GgufValue::U32(2)));
        let got = read_swa_layers(&file, "exaone-moe", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
        assert_eq!(got, SwaLayers::period(2, false));
    }

    /// The array mode honours the array at ANY length up to
    /// `LLAMA_MAX_LAYERS`, as `get_arr` does (#452): entry `il` is layer
    /// `il`, a layer past the end is full attention, entries past the
    /// trunk are never read. The key is REQUIRED.
    #[test]
    fn an_array_mode_architecture_reads_the_array_at_any_length() {
        // block_count 5, trunk 4: the MTP entry is dropped.
        let file = with(Some(bools(&[false, true, true, false, true])));
        let got = read_swa_layers(&file, "mimo2", KEY, &trunk(5, 4), None).unwrap();
        assert_eq!(
            got,
            SwaLayers::PerLayer(vec![false, true, true, false].into())
        );
        // Shorter than the trunk: the rest keep the zero seed.
        let file = with(Some(bools(&[true, true])));
        let got = read_swa_layers(&file, "mimo2", KEY, &trunk(5, 4), None).unwrap();
        assert_eq!(
            got,
            SwaLayers::PerLayer(vec![true, true, false, false].into())
        );
        // Longer: the tail is never read, up to the cap.
        let file = with(Some(bools(&[true; LLAMA_MAX_LAYERS])));
        let got = read_swa_layers(&file, "mimo2", KEY, &trunk(4, 4), None).unwrap();
        assert_eq!(got, SwaLayers::PerLayer(vec![true; 4].into()));
        let file = with(Some(bools(&[true; LLAMA_MAX_LAYERS + 1])));
        assert!(matches!(
            read_swa_layers(&file, "mimo2", KEY, &trunk(4, 4), None),
            Err(LoadError::UnsupportedFeature(a, m)) if a == "mimo2" && m.contains("LLAMA_MAX_LAYERS")
        ));
        // Absent is REQUIRED.
        assert!(matches!(
            read_swa_layers(&with(None), "mimo2", KEY, &trunk(4, 4), None),
            Err(LoadError::MissingHparam(k)) if k == KEY
        ));
    }

    /// `cohere2moe`'s converter writes one entry per TRUNK layer and
    /// `mimo2`'s one per block; both read the same, as in libllama.
    #[test]
    fn a_trunk_length_and_a_block_count_length_array_read_alike() {
        for arr in [
            &[false, true, true, false][..],
            &[false, true, true, false, true][..],
        ] {
            let got = read_swa_layers(
                &with(Some(bools(arr))),
                "cohere2moe",
                KEY,
                &trunk(5, 4),
                DENSE_FIRST_4,
            )
            .unwrap();
            assert_eq!(
                got,
                SwaLayers::PerLayer(vec![false, true, true, false].into())
            );
        }
    }

    /// In the array-only mode a scalar is refused: `get_arr` throws
    /// `array key not found in model` on a non-array value.
    #[test]
    fn an_array_mode_architecture_refuses_a_scalar() {
        for v in [6, 0] {
            assert!(matches!(
                read_swa_layers(&with(Some(GgufValue::U32(v))), "step35", KEY, &trunk(3, 3), None),
                Err(LoadError::UnsupportedFeature(a, m)) if a == "step35" && m.contains("only as an array")
            ));
        }
    }

    /// The scalar-then-array mode takes whichever shape the file has,
    /// and refuses an absent key because its fallback read is REQUIRED.
    #[test]
    fn a_scalar_then_array_architecture_takes_either_shape_and_requires_one() {
        let got = read_swa_layers(
            &with(Some(GgufValue::U32(2))),
            "mellum",
            KEY,
            &trunk(4, 4),
            LAST_DENSE_4,
        )
        .unwrap();
        assert_eq!(got, SwaLayers::period(2, false));
        let got = read_swa_layers(
            &with(Some(bools(&[true, true, false, true]))),
            "mellum",
            KEY,
            &trunk(4, 4),
            LAST_DENSE_4,
        )
        .unwrap();
        assert_eq!(
            got,
            SwaLayers::PerLayer(vec![true, true, false, true].into())
        );
        assert!(matches!(
            read_swa_layers(&with(None), "mellum", KEY, &trunk(4, 4), LAST_DENSE_4),
            Err(LoadError::MissingHparam(k)) if k == KEY
        ));
    }
}
