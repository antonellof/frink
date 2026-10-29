//! **Which layers slide.** llama.cpp's `hparams.is_swa_impl[il]`, as
//! one value a `ModelConfig` carries, with the per-layer ARRAY form of
//! `{arch}.attention.sliding_window_pattern` read the way each
//! architecture's graph reads it.
//!
//! Until 2026-09-11 frink carried the layout as a scalar period plus a
//! phase, read the key as a scalar, and REFUSED the array form for every
//! architecture. At the pin (`604cc430`, re-derived for #420) a graph
//! reads the key in one of two ways, or not at all:
//!
//! | mode | call | absent | scalar in the file | ARRAY in the file |
//! |---|---|---|---|---|
//! | [`PatternKeyRead::ArrayThenScalar`] | `load_swa_pattern(ml, n, dense_first)` (`llama-model.cpp:3952-3959`) | the seeded period `n` | the period, with the seed's phase | the per-layer truth |
//! | [`PatternKeyRead::PerLayerBool`] | `get_arr(kid, hparams.is_swa_impl)` (required) | refused: `array key not found in model` (`llama-model-loader.cpp:356-360`) | refused, the same throw | the per-layer truth |
//! | [`PatternKeyRead::Unread`] | none | the seed | ignored | ignored |
//!
//! `load_swa_pattern` tries the ARRAY first, optional (`:3953`), then
//! the scalar, optional (`:3957`), then keeps the seed: so the array is
//! honoured on every graph that calls it. Until #420 frink ignored it on
//! the rows the older pin read through the scalar overload (`exaone4`,
//! `exaone-moe`, `olmo2`, `gemma2`, `gemma3`, ...) and REQUIRED it on
//! `mellum`, `cohere2moe` and `muse-glimmer`, which run on the seed
//! upstream when the key is absent.
//!
//! The census is measured, not remembered: `grep -nE
//! 'SLIDING_WINDOW_PATTERN|load_swa_pattern' src/models/*.cpp`.
//! Nineteen graphs call `load_swa_pattern` ([`LOAD_SWA_PATTERN_CALLERS`])
//! and fifteen read the array with `get_arr` ([`PER_LAYER_ARRAY_READERS`]
//! has thirteen; `beam.cpp:23-25` and `maion-coder.cpp:15-17` also throw
//! unless the array has `n_layer_all` entries, and both are refused by
//! name for other things). Every other graph reads the key in no form.
//!
//! Real exports write an array that AGREES with the seeded period
//! (`conversion/exaone.py:84`, `olmo.py:59-66`, `gemma.py:532-535`), so a
//! converter-made file reads the same either way; `tests/
//! window_array_graphs.rs` measures a file whose array DISAGREES, and
//! libllama's logits follow the array.
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

/// How an architecture's `load_arch_hparams` reads
/// `{arch}.attention.sliding_window_pattern`. See the module doc for
/// what each does with each shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternKeyRead {
    /// Not read: the seeded layout, whatever the file says.
    Unread,
    /// `get_arr(kid, hparams.is_swa_impl)`: array honoured, a scalar
    /// or an absent key refused (REQUIRED).
    PerLayerBool,
    /// `load_swa_pattern`: the array if the file has one, else the
    /// scalar as a period, else the seed. Neither read is required.
    ArrayThenScalar,
}

/// Every graph that reads the key ONLY as the per-layer array, with the
/// line. `gemma4` and `gemma4-assistant` run on their own engine
/// (`gemma4_gguf_loader.rs` has read the array since it existed);
/// `step35` and `mimo2` closed on other seams and run on this one;
/// `dflash` is deferred.
pub const PER_LAYER_ARRAY_READERS: &[(&str, &str)] = &[
    ("gemma4", "src/models/gemma4.cpp:6"),
    ("gemma4-assistant", "src/models/gemma4-assistant.cpp:7"),
    ("dflash", "src/models/dflash.cpp:119"),
    ("dots3note", "src/models/dots3note.cpp:34"),
    ("flint", "src/models/flint.cpp:13"),
    ("muse-spark", "src/models/muse-spark.cpp:17"),
    ("step35", "src/models/step35.cpp:26"),
    ("mimo2", "src/models/mimo2.cpp:12"),
    // Landed upstream after the 2026-08-04 pin; `ml.get_arr` with no
    // scalar attempt, like the four above.
    ("spark2_5", "src/models/spark2-5.cpp:8"),
    ("maple", "src/models/maple.cpp:10"),
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

/// Every graph that calls `load_swa_pattern`, with the line. The seed
/// each passes is `capability::default_swa_layout`'s row (or the
/// registry's `SwaFacts::default_layout`).
pub const LOAD_SWA_PATTERN_CALLERS: &[(&str, &str)] = &[
    ("afmoe", "src/models/afmoe.cpp:17"),
    ("cohere2", "src/models/cohere2.cpp:5"),
    ("cohere2moe", "src/models/cohere2moe.cpp:28"),
    ("exaone-moe", "src/models/exaone-moe.cpp:6"),
    ("exaone4", "src/models/exaone4.cpp:7"),
    ("gemma-embedding", "src/models/gemma-embedding.cpp:5"),
    ("gemma-embedding2", "src/models/gemma-embedding2.cpp:5"),
    ("gemma2", "src/models/gemma2.cpp:6"),
    ("gemma3", "src/models/gemma3.cpp:7"),
    ("gemma3n", "src/models/gemma3n.cpp:5"),
    ("gpt-oss", "src/models/openai-moe.cpp:9"),
    ("laguna", "src/models/laguna.cpp:42"),
    ("llama4", "src/models/llama4.cpp:19"),
    ("mellum", "src/models/mellum.cpp:16"),
    ("modern-bert", "src/models/modern-bert.cpp:11"),
    ("muse-glimmer", "src/models/muse-glimmer.cpp:13"),
    ("olmo2", "src/models/olmo2.cpp:9"),
    ("plamo3", "src/models/plamo3.cpp:9"),
    ("smallthinker", "src/models/smallthinker.cpp:9"),
];

/// THE table, derived from the two censuses above.
pub fn pattern_key_read(arch: &str) -> PatternKeyRead {
    if PER_LAYER_ARRAY_READERS.iter().any(|(a, _)| *a == arch) {
        PatternKeyRead::PerLayerBool
    } else if LOAD_SWA_PATTERN_CALLERS.iter().any(|(a, _)| *a == arch) {
        PatternKeyRead::ArrayThenScalar
    } else {
        PatternKeyRead::Unread
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
        // No read: the file's key, scalar or array, is dead metadata.
        PatternKeyRead::Unread => Ok(SwaLayers::from_default(seeded)),
        // `load_swa_pattern` (`llama-model.cpp:3952-3959`): the array if
        // there is one (`:3953`, optional), else the scalar (`:3957`,
        // optional), else the seed. No absent key refuses.
        PatternKeyRead::ArrayThenScalar => match value {
            None => Ok(SwaLayers::from_default(seeded)),
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
    /// answers its listed mode, and an unlisted name reads nothing.
    #[test]
    fn every_listed_reader_answers_its_mode() {
        for (arch, _) in PER_LAYER_ARRAY_READERS {
            assert_eq!(
                pattern_key_read(arch),
                PatternKeyRead::PerLayerBool,
                "{arch}"
            );
        }
        for (arch, _) in LOAD_SWA_PATTERN_CALLERS {
            assert_eq!(
                pattern_key_read(arch),
                PatternKeyRead::ArrayThenScalar,
                "{arch}"
            );
        }
        for arch in ["llama", "qwen2", "phi3"] {
            assert_eq!(pattern_key_read(arch), PatternKeyRead::Unread, "{arch}");
        }
    }

    /// `load_swa_pattern` honours an array even where it DISAGREES with
    /// the seeded period (#420), takes a scalar as a period with the
    /// seed's phase, and keeps the seed when the key is absent.
    #[test]
    fn a_load_swa_pattern_architecture_takes_the_array_then_the_scalar_then_the_seed() {
        // Disagrees with last-dense-4 on every layer.
        let file = with(Some(bools(&[false, false, false, true])));
        let got = read_swa_layers(&file, "exaone-moe", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
        assert_eq!(
            got,
            SwaLayers::PerLayer(vec![false, false, false, true].into())
        );
        let file = with(Some(GgufValue::U32(2)));
        let got = read_swa_layers(&file, "exaone-moe", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
        assert_eq!(got, SwaLayers::period(2, false));
        let got = read_swa_layers(&file, "cohere2moe", KEY, &trunk(4, 4), DENSE_FIRST_4).unwrap();
        assert_eq!(got, SwaLayers::period(2, true));
        // Absent: the seed, on every caller, `mellum` included.
        for arch in ["exaone-moe", "mellum"] {
            let got = read_swa_layers(&with(None), arch, KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
            assert_eq!(got, SwaLayers::period(4, false), "{arch}");
        }
    }

    /// A graph that reads no key keeps its seed whatever the file says.
    #[test]
    fn an_unread_key_is_dead_metadata() {
        for v in [GgufValue::U32(2), bools(&[true, false, true, false])] {
            let got =
                read_swa_layers(&with(Some(v)), "llama", KEY, &trunk(4, 4), LAST_DENSE_4).unwrap();
            assert_eq!(got, SwaLayers::period(4, false));
        }
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
}
