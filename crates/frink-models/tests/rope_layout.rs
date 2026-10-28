//! frink's per-architecture RoPE layout, pinned against llama.cpp's
//! `llama_model_rope_type`.
//!
//! Why a whole test for one enum: getting this wrong is silent. A NEOX
//! model rotated as NORM (or the reverse) still loads, still runs at
//! full speed, and still emits fluent text — it just rotates the wrong
//! pairs of every Q and K head, so the answers are wrong and nothing
//! says so. It has already happened twice in this repo: gpt-oss was on
//! the interleaved list until the gpt-oss coverage work, and an audit of
//! the whole table against the reference then found **24 more** archs on
//! the generic GQA path with the same defect (afmoe, apertus,
//! bailingmoe2, codeshell, dots1, exaone, exaone-moe, grovemoe,
//! hunyuan-dense, hunyuan-moe, laguna, mellum, mimo2, minicpm3,
//! nemotron, openelm, orion, plamo, plamo3, seed_oss, smallthinker,
//! starcoder2, step35, talkie).
//!
//! `LLAMA_ROPE_TYPES` below is a mechanical transcription of the
//! `LLAMA_ROPE_TYPE_NORM` and `LLAMA_ROPE_TYPE_NEOX` groups of that
//! switch (`src/llama-model.cpp`), keyed by the GGUF architecture string
//! from `LLM_ARCH_NAMES` (`src/llama-arch.cpp`). Architectures llama.cpp
//! returns `MROPE` / `IMROPE` for, or decides per-checkpoint (`glm4`,
//! `dflash`), are deliberately absent: frink does not implement those
//! layouts, and a placeholder here would assert agreement that does not
//! exist.
//!
//! `LLAMA_NO_ROPE` is the switch's *first* group, `LLAMA_ROPE_TYPE_NONE`,
//! and it exists because leaving it out left a hole exactly where the
//! stakes are highest. `rope_layout_matches_llama_cpp` skips any
//! architecture it has no entry for, so an arch that must not be rotated
//! at all could sit on the generic (rotating) path and be *skipped* by
//! the test that exists to catch that. Five did: `gpt2` (learned
//! absolute position embeddings) and `bloom` / `mpt` / `refact` / `jais`
//! (ALiBi). `bloom` and `refact` hardcode `f_max_alibi_bias = 8.0f` in
//! `load_arch_hparams` and carry no GGUF key at all, so no
//! metadata-presence gate could have found them either.
//!
//! Regenerate by re-reading the reference; do not edit an entry to make
//! a failing test pass.

use frink_models::capability::{architecture_catalog, resolve_profile, ArchPath};
use frink_models::config::RopeLayout;

const LLAMA_ROPE_TYPES: &[(&str, RopeLayout)] = &[
    // Added 2026-09-19 with the pin move to `5b59b83`: seven of the
    // fifteen graphs that landed upstream since 2026-08-04 are in the
    // NORM or NEOX group (`llama-model.cpp`'s switch). All seven
    // refuse today, so the layout here is what frink WOULD rotate
    // them as if the refusal lifted -- which is exactly what this test
    // is for. `qwen4exp` and `qwen3tts` answer IMROPE and `kimi-k3`
    // NONE, so they stay out of the table as the other MROPE rows do.
    // Added 2026-10-28 with the pin move to `604cc430`: ten of the
    // thirteen new names are in the NORM or NEOX group (`beam`,
    // `deepseek41`, `flint`, `gemma-embedding2`, `k2-horizon`,
    // `kolibri1`, `lfm3`, `maion-coder`, `muse-spark`,
    // `vllm-decision2`); `clef` answers IMROPE, `sensenova_u1` MROPE
    // and `glm5-next` NONE. `bailingmoe3` moved from the NORM group to
    // a per-checkpoint arm (MROPE when the file carries sections) and
    // was never in this table, because it is `dedicated`.
    ("afmoe", RopeLayout::Neox),
    ("apertus", RopeLayout::Neox),
    ("arcee", RopeLayout::Norm),
    ("arctic", RopeLayout::Norm),
    ("baichuan", RopeLayout::Norm),
    ("bailingmoe", RopeLayout::Norm),
    ("bailingmoe2", RopeLayout::Neox),
    ("beam", RopeLayout::Neox),
    ("bert", RopeLayout::Neox),
    ("bitnet", RopeLayout::Neox),
    ("chameleon", RopeLayout::Norm),
    ("chatglm", RopeLayout::Norm),
    ("codeshell", RopeLayout::Neox),
    ("cogvlm", RopeLayout::Neox),
    ("cohere2", RopeLayout::Norm),
    ("cohere2moe", RopeLayout::Norm),
    ("command-r", RopeLayout::Norm),
    ("dbrx", RopeLayout::Neox),
    ("deci", RopeLayout::Norm),
    ("deepseek", RopeLayout::Norm),
    ("deepseek2", RopeLayout::Norm),
    ("deepseek2-ocr", RopeLayout::Norm),
    ("deepseek32", RopeLayout::Norm),
    ("deepseek4", RopeLayout::Norm),
    ("deepseek41", RopeLayout::Norm),
    ("dots1", RopeLayout::Neox),
    ("dream", RopeLayout::Neox),
    ("eagle3", RopeLayout::Norm),
    ("ernie4_5", RopeLayout::Norm),
    ("ernie4_5-moe", RopeLayout::Norm),
    ("eurobert", RopeLayout::Neox),
    ("exaone", RopeLayout::Neox),
    ("exaone-moe", RopeLayout::Neox),
    ("exaone4", RopeLayout::Neox),
    ("falcon", RopeLayout::Neox),
    ("falcon-h1", RopeLayout::Neox),
    ("flint", RopeLayout::Neox),
    ("gemma", RopeLayout::Neox),
    ("gemma-embedding", RopeLayout::Neox),
    ("gemma-embedding2", RopeLayout::Neox),
    ("gemma2", RopeLayout::Neox),
    ("gemma3", RopeLayout::Neox),
    ("gemma3n", RopeLayout::Neox),
    ("gemma4", RopeLayout::Neox),
    ("gemma4-assistant", RopeLayout::Neox),
    ("glm-dsa", RopeLayout::Norm),
    ("gpt-oss", RopeLayout::Neox),
    ("gptneox", RopeLayout::Neox),
    ("granite", RopeLayout::Norm),
    ("granite_swa", RopeLayout::Norm),
    ("granitehybrid", RopeLayout::Norm),
    ("granitemoe", RopeLayout::Norm),
    ("graniteswitch", RopeLayout::Norm),
    ("grok", RopeLayout::Neox),
    ("grovemoe", RopeLayout::Neox),
    ("hrm_text", RopeLayout::Neox),
    ("hunyuan-dense", RopeLayout::Neox),
    ("hunyuan-moe", RopeLayout::Neox),
    ("hy_v3", RopeLayout::Neox),
    ("internlm2", RopeLayout::Norm),
    ("jais2", RopeLayout::Neox),
    ("jina-bert-v3", RopeLayout::Neox),
    ("k2-horizon", RopeLayout::Neox),
    ("kolibri1", RopeLayout::Neox),
    ("laguna", RopeLayout::Neox),
    ("lfm2", RopeLayout::Neox),
    ("lfm2moe", RopeLayout::Neox),
    ("lfm3", RopeLayout::Neox),
    ("llada", RopeLayout::Norm),
    ("llada-moe", RopeLayout::Neox),
    ("llama", RopeLayout::Norm),
    ("llama-embed", RopeLayout::Norm),
    ("llama4", RopeLayout::Norm),
    ("maincoder", RopeLayout::Norm),
    ("maion-coder", RopeLayout::Neox),
    ("maple", RopeLayout::Neox),
    ("mellum", RopeLayout::Neox),
    ("mimo2", RopeLayout::Neox),
    ("minicpm", RopeLayout::Norm),
    ("minicpm3", RopeLayout::Neox),
    ("minimax-01", RopeLayout::Neox),
    ("minimax-m2", RopeLayout::Neox),
    ("minimax-m3", RopeLayout::Neox),
    ("mistral3", RopeLayout::Norm),
    ("mistral4", RopeLayout::Norm),
    ("modern-bert", RopeLayout::Neox),
    ("muse-glimmer", RopeLayout::Norm),
    ("muse-spark", RopeLayout::Norm),
    ("nanbeige", RopeLayout::Norm),
    ("nemotron", RopeLayout::Neox),
    ("neo-bert", RopeLayout::Norm),
    ("nomic-bert", RopeLayout::Neox),
    ("nomic-bert-moe", RopeLayout::Neox),
    ("olmo", RopeLayout::Norm),
    ("olmo2", RopeLayout::Neox),
    ("olmoe", RopeLayout::Neox),
    ("openelm", RopeLayout::Neox),
    ("orion", RopeLayout::Neox),
    ("pangu-embedded", RopeLayout::Neox),
    ("phi2", RopeLayout::Neox),
    ("phi3", RopeLayout::Neox),
    ("phimoe", RopeLayout::Neox),
    ("plamo", RopeLayout::Neox),
    ("plamo2", RopeLayout::Neox),
    ("plamo3", RopeLayout::Neox),
    ("plm", RopeLayout::Norm),
    ("qwen", RopeLayout::Neox),
    ("qwen2", RopeLayout::Neox),
    ("qwen2moe", RopeLayout::Neox),
    ("qwen3", RopeLayout::Neox),
    ("qwen3moe", RopeLayout::Neox),
    ("qwen3next", RopeLayout::Neox),
    ("rnd1", RopeLayout::Neox),
    ("seed_oss", RopeLayout::Neox),
    ("smallthinker", RopeLayout::Neox),
    ("smollm3", RopeLayout::Norm),
    ("spark2_5", RopeLayout::Neox),
    ("stablelm", RopeLayout::Neox),
    ("starcoder", RopeLayout::Norm),
    ("starcoder2", RopeLayout::Neox),
    ("step35", RopeLayout::Neox),
    ("talkie", RopeLayout::Neox),
    ("vllm-decision2", RopeLayout::Neox),
    ("xverse", RopeLayout::Norm),
];

/// `llama_model_rope_type`'s `LLAMA_ROPE_TYPE_NONE` group, transcribed
/// from the same switch. For every one of these the correct rotation is
/// *none*: the position information lives in ALiBi biases, in a learned
/// `position_embd` table, in a recurrent state, or nowhere.
///
/// No entry carries a `RopeLayout`, because there is no honest value to
/// put there. What the test asserts instead is structural: none of them
/// may reach a path that rotates.
const LLAMA_NO_ROPE: &[&str] = &[
    "clip",
    "gpt2",
    "gptj",
    "mpt",
    "refact",
    "bloom",
    "mamba",
    "mamba2",
    "jamba",
    "jina-bert-v2",
    "t5",
    "t5encoder",
    "jais",
    "rwkv6",
    "rwkv6qwen2",
    "rwkv7",
    "arwkv7",
    "wavtokenizer-dec",
    "nemotron_h",
    "nemotron_h_moe",
    "kimi-linear",
    // Added 2026-10-28 (`604cc430`); a `dedicated` refusal.
    "glm5-next",
];

/// Every architecture frink actually routes through a RoPE kernel must
/// agree with the reference.
///
/// `ArchPath::Deferred` entries are exempt: `capability::deferred_scope`
/// gives all of them the same placeholder layout, and no graph ever
/// reads it because the load refuses first. Making them agree would be
/// asserting something about code that does not run.
#[test]
fn rope_layout_matches_llama_cpp() {
    let expected: std::collections::HashMap<&str, RopeLayout> =
        LLAMA_ROPE_TYPES.iter().copied().collect();

    let mut checked = 0usize;
    let mut wrong = Vec::new();
    for p in architecture_catalog() {
        if matches!(
            p.path,
            ArchPath::Deferred { .. } | ArchPath::TestFixture { .. }
        ) {
            continue;
        }
        let Some(&want) = expected.get(p.gguf_name) else {
            // A frink-only alias (`phi4`, `kimi_k3`, the `granite-*`
            // spellings) has no entry in llama.cpp's own name table, so
            // there is nothing to pin it against.
            // `a_frink_only_name_on_the_generic_path_is_declared`
            // below is the other half of this skip: it is why the skip
            // cannot silently grow.
            continue;
        };
        checked += 1;
        if p.rope != want {
            wrong.push(format!(
                "{}: frink {:?}, llama.cpp {:?}",
                p.gguf_name, p.rope, want
            ));
        }
    }

    assert!(
        checked >= 85,
        "only {checked} architectures were actually compared -- the pin has gone vacuous"
    );
    assert!(
        wrong.is_empty(),
        "RoPE layout disagrees with llama.cpp:\n  {}",
        wrong.join("\n  ")
    );
}

/// A frink-only name that reaches a rotating path must be on a short,
/// declared list.
///
/// `rope_layout_matches_llama_cpp`'s lookup miss is a `continue`, so
/// every name absent from `LLAMA_ROPE_TYPES` is silently exempt from
/// the only test that compares RoPE layouts. `mistral`, `mixtral` and
/// `yi` sat in that blind spot as `GenericGqa { rope: Neox }` while the
/// graph they claim to be -- `llama` -- is NORM, which is the
/// wrong-pairs defect behind the Llama-3.1-8B bug, latent only because
/// the rows refused for another reason. They are `DedicatedOnly` now.
///
/// This is what stops the blind spot from growing again: a frink-only
/// name may exist, but if it is on the generic path somebody has to add
/// it here and say why.
#[test]
fn a_frink_only_name_on_the_generic_path_is_declared() {
    let known: std::collections::HashMap<&str, RopeLayout> =
        LLAMA_ROPE_TYPES.iter().copied().collect();
    // Two, both deliberate, both still refusing:
    //
    //  * `phi4` -- unlike the three alias rows it names a concrete
    //    hypothesis a real file would settle (phi3's fused-QKV graph,
    //    NEOX). `capability::phi4`'s UNKNOWN verdict carries it.
    //  * `granite-moe` -- a spelling variant of llama.cpp's
    //    `granitemoe`, and it carries the SAME layout (NORM, the
    //    `LLM_ARCH_GRANITE_MOE` arm of `llama_model_rope_type`) and the
    //    same `GRANITE_MULTIPLIERS` verdict, which is the point of
    //    keeping one blocker string for all three granite rows.
    //  * `glm4moe` -- llama.cpp decides it per checkpoint
    //    (`llama-model.cpp:2700`: `MROPE` when `rope.dimension_sections`
    //    says so, `NEOX` otherwise), so it is deliberately absent from
    //    the table above. frink rotates NEOX, which is the text-tower
    //    answer in both cases: `tests/glm4moe_graphs.rs` pins that
    //    libllama's logits under MROPE with text positions are byte
    //    for byte the NEOX ones.
    //  * `glm4` -- the same per-checkpoint arm one line up
    //    (`llama-model.cpp:2699`: `MROPE` with sections, `NORM`
    //    without). frink rotates NORM and REFUSES a sectioned file
    //    (`frink_models::mrope`), because for this one the converter
    //    permuted the weights and the two rotations differ (measured,
    //    `tests/glm4_graphs.rs`).
    //  * `granite-hybrid` -- the same frink-only alias as `granite-moe`,
    //    of `granitehybrid` (`llama-arch.cpp:102`); NORM like its row.
    //  * `qwen35`, `qwen35moe` -- `llama_model_rope_type` answers
    //    `IMROPE` (`llama-model.cpp:2694-2696`) unconditionally, a
    //    layout this table has no column for. On text positions it is
    //    NEOX band for band (`frink_models::mrope`), which
    //    `tests/qwen35_graphs.rs` pins against libllama with the
    //    sections in the file.
    //  * `qwen4exp` -- `llama_model_rope_type` answers `IMROPE` for it
    //    as it does for `qwen35` (`llama-model.cpp`), and the row is a
    //    NEW CODE refusal on its delta-net and its hybrid memory index,
    //    so nothing rotates today; the NEOX layout it carries is what
    //    IMROPE is on text positions (`frink_models::mrope`).
    const DECLARED: &[&str] = &[
        "qwen4exp",
        "phi4",
        "granite-moe",
        "granite-hybrid",
        "glm4moe",
        "glm4",
        "qwen35",
        "qwen35moe",
    ];
    let mut undeclared = Vec::new();
    for p in architecture_catalog() {
        let ArchPath::GenericGqa { rope } = p.path else {
            continue;
        };
        if known.contains_key(p.gguf_name) || DECLARED.contains(&p.gguf_name) {
            continue;
        }
        // A `LLAMA_ROPE_TYPE_NONE` row on the generic path rotates
        // nothing (`gpt2`, `frink_models::position_embd`); the layout
        // it carries is a filler, and `no_rope_architectures_never_
        // reach_a_rotating_path` below is the test that pins that.
        if frink_models::rope_layers::rope_layers(p.gguf_name, 12, false, 0)
            == frink_models::rope_layers::RopeLayers::Never
        {
            continue;
        }
        undeclared.push(format!("{}: frink rotates it as {rope:?}", p.gguf_name));
    }
    assert!(
        undeclared.is_empty(),
        "on the generic path, rotated, and absent from llama.cpp's name table, so \
         `rope_layout_matches_llama_cpp` skips them:\n  {}",
        undeclared.join("\n  ")
    );
}

/// An architecture llama.cpp does not rotate must never reach a frink
/// path that does.
///
/// This is the hole `rope_layout_matches_llama_cpp` had: its lookup
/// miss is a `continue`, so an arch absent from the reference table is
/// silently exempt, and the `LLAMA_ROPE_TYPE_NONE` group was absent by
/// design. `gpt2`, `bloom`, `mpt`, `refact` and `jais` were all admitted
/// as `GenericGqa { rope: Neox }` and skipped by the only test that
/// could have said so.
#[test]
fn no_rope_architectures_never_reach_a_rotating_path() {
    let mut rotated = Vec::new();
    let mut unknown = Vec::new();
    for &name in LLAMA_NO_ROPE {
        let Some(p) = resolve_profile(name) else {
            unknown.push(name);
            continue;
        };
        // `TestFixture` is not in this group and would be a mistake here.
        // A generic-path row is admissible ONLY under the rule that
        // rotates nothing (`gpt2`: a learned position table,
        // `frink_models::position_embd`); its layout is then a filler
        // no rotation site reads.
        if let ArchPath::GenericGqa { rope } | ArchPath::TestFixture { rope } = p.path {
            if frink_models::rope_layers::rope_layers(name, 12, false, 0)
                != frink_models::rope_layers::RopeLayers::Never
            {
                rotated.push(format!("{name}: frink rotates it as {rope:?}"));
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "not in frink's registry, so a checkpoint tagged with it is refused for the \
         wrong reason: {unknown:?}"
    );
    assert!(
        rotated.is_empty(),
        "llama.cpp applies no RoPE to these:\n  {}",
        rotated.join("\n  ")
    );
}

/// The transcription itself must not silently shrink.
#[test]
fn the_reference_table_is_complete_enough_to_be_worth_pinning() {
    assert!(
        LLAMA_ROPE_TYPES.len() >= 100,
        "transcribed {} entries from llama_model_rope_type",
        LLAMA_ROPE_TYPES.len()
    );
    assert!(
        LLAMA_NO_ROPE.len() >= 21,
        "transcribed {} entries from the LLAMA_ROPE_TYPE_NONE group",
        LLAMA_NO_ROPE.len()
    );
    // Every name in either transcription has to resolve, or the two
    // tests above quietly compare nothing.
    for &name in LLAMA_NO_ROPE
        .iter()
        .chain(LLAMA_ROPE_TYPES.iter().map(|(n, _)| n))
    {
        assert!(
            resolve_profile(name).is_some(),
            "{name} is in llama.cpp's inventory but not in frink's capability registry"
        );
    }
}
