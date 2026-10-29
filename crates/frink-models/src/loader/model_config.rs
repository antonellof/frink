//! `ModelConfig::from_gguf`: every hyper-parameter, refusal and per-
//! architecture default resolved from a GGUF's metadata.
//!
//! One function, kept whole: it is a single ordered reading of the
//! file, and the order is load-bearing (several keys are read only
//! after another has decided whether they apply), so splitting it would
//! be a rewrite rather than a move.

use frink_gguf::{GgufValue, TensorSource};
use frink_moe::{GatingFunction, MoeLayerConfig};

use crate::config::ModelConfig;

use super::rope_scaling::{linear_scaling_from_gguf, yarn_scaling_from_gguf};
use super::tensors::load_f32_vec_optional;
use super::LoadError;

pub(super) fn metadata_u64_any(file: &impl TensorSource, keys: &[String]) -> Option<u64> {
    keys.iter().find_map(|k| file.metadata_u64(k))
}

pub(super) fn metadata_f32_any(file: &impl TensorSource, keys: &[String]) -> Option<f32> {
    keys.iter()
        .find_map(|k| file.metadata(k).and_then(GgufValue::as_f32))
}

impl ModelConfig {
    /// Derives a `ModelConfig` from a real GGUF file's own hyperparameter
    /// metadata, following llama.cpp's `general.architecture`-prefixed key
    /// convention (`{arch}.block_count`, `{arch}.embedding_length`,
    /// `{arch}.attention.head_count`, `{arch}.expert_count`, ...) rather
    /// than requiring a hand-written preset to already match the file's
    /// shape exactly. This is what lets `frink-server` (and `frink
    /// run-real`) load an arbitrary checkpoint, not just the three
    /// hand-tuned presets in `config.rs`.
    ///
    /// Fields with no corresponding metadata key fall back to widely-used
    /// llama.cpp defaults (documented inline) and are listed in the
    /// returned config's `best_effort_fields`, following the same
    /// confirmed-vs-estimated discipline as the hand-written presets.
    pub fn from_gguf(file: &impl TensorSource) -> Result<Self, LoadError> {
        let arch = file
            .metadata_str("general.architecture")
            .ok_or_else(|| LoadError::MissingHparam("general.architecture".to_string()))?
            .to_string();
        // Every table row this function reads, resolved once from the
        // RAW name (`crate::models::facts`); the order of the reads
        // below is unchanged.
        let facts = crate::models::ArchFacts::from_tables(&arch);
        let arch_profile = facts
            .profile
            .ok_or_else(|| LoadError::UnsupportedArchitecture(arch.clone()))?;
        let rope_layout = match arch_profile.path {
            crate::capability::ArchPath::GenericGqa { rope }
            | crate::capability::ArchPath::TestFixture { rope } => rope,
            crate::capability::ArchPath::DedicatedOnly { reason } => {
                return Err(LoadError::DedicatedArchitectureRequired(
                    arch.clone(),
                    reason,
                ));
            }
            crate::capability::ArchPath::Deferred { reason } => {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!("architecture deferred from Frink text-generation scope: {reason}"),
                ));
            }
        };
        let qk_norm_style = arch_profile.qk_norm;
        // A vision export's text tower declaring M-RoPE sections on an
        // architecture whose text rotation is NORM (`crate::mrope`).
        if let Some(reason) = crate::mrope::mrope_refusal(file, &arch) {
            return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
        }
        for (meta_key, feature) in facts.unsupported_feature_keys(&arch) {
            if let Some(v) = metadata_f32_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0.0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!("{feature} (metadata {meta_key}={v})"),
                    ));
                }
            }
            if let Some(v) = metadata_u64_any(file, std::slice::from_ref(&meta_key)) {
                if v > 0 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        feature.to_string(),
                    ));
                }
            }
        }
        // `{arch}.attention.causal = false`. llama.cpp reads it for EVERY
        // architecture (`llama-model.cpp:1400`, optional, default true)
        // and a graph built with it attends both ways; `lfm3.cpp:22-25,
        // 92-96` is the case a converter really writes (the d1 decision
        // checkpoints, `conversion/lfm3.py:167`), with no memory at all
        // (`llama-model.cpp:2808`). Every body on this path masks
        // causally, so such a file would compute a different graph.
        if file.metadata_bool(&format!("{arch}.attention.causal")) == Some(false) {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "`{arch}.attention.causal` is false: the file's graph attends to the whole \
                     prompt both ways (llama-model.cpp:1400; for lfm3 the d1 decision graph, \
                     lfm3.cpp:22-25,92-96, which keeps no memory, llama-model.cpp:2808), and \
                     the generic decoder only masks causally"
                ),
            ));
        }
        // Metadata-declared multipliers the generic decoder does not
        // apply. Unlike the tensor-consumption gate, nothing about these
        // is visible in the weights, so a Granite checkpoint would load
        // and answer at the wrong scale. See
        // `ArchFacts::unsupported_scaling_keys`.
        for (meta_key, feature, no_op) in facts.unsupported_scaling_keys(&arch) {
            if let Some(v) = metadata_f32_any(file, std::slice::from_ref(&meta_key)) {
                if (v - no_op).abs() > 1e-6 {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!("{feature} (metadata {meta_key}={v})"),
                    ));
                }
            }
        }
        let key = |suffix: &str| format!("{arch}.{suffix}");

        let name: &'static str = Box::leak(
            file.metadata_str("general.name")
                .unwrap_or(&arch)
                .to_string()
                .into_boxed_str(),
        );

        let block_count =
            file.metadata_u64(&key("block_count"))
                .ok_or_else(|| LoadError::MissingHparam(key("block_count")))? as usize;
        // llama.cpp's `n_layer()` is `block_count` MINUS the NextN/MTP
        // blocks the converter appended inside it, for the graphs that
        // read `nextn_predict_layers` (`crate::mtp_blocks`). `n_layers`
        // is the trunk from here on; `block_count` is handed ONLY to the
        // two things llama.cpp decides before it has read the key --
        // `exaone4.cpp:4`'s layer-count gate and the per-layer array
        // lengths -- and nowhere else.
        let trunk = crate::mtp_blocks::trunk_layers(file, &arch, block_count)?;
        // Nanbeige's `num_loops`: the trunk is the PHYSICAL count and
        // `n_layers` the logical one from here on (`crate::layer_loops`);
        // the per-layer arrays below are read at physical length and
        // replicated per pass, as `nanbeige.cpp:24-26` replicate them.
        let layer_loops = crate::layer_loops::read_layer_loops(file, &arch, trunk.n_layers)?;
        let n_layers = layer_loops
            .map(|l| l.logical_layers())
            .unwrap_or(trunk.n_layers);
        // Baichuan-13B (block_count 40) used to be refused HERE: one
        // architecture string, two positional schemes, decided by layer
        // count with no key (`baichuan.cpp:11-14`). It is served now
        // through `crate::alibi` (the bias) and `crate::rope_layers`
        // (no rotation), both keyed on the same layer count.
        // EXAONE-4 32B used to be refused HERE, on the same shape:
        // `exaone4.cpp:4-14` switches the whole SWA machinery on inside
        // `if (hparams.n_layer() == 64)` and :116 then ropes only the
        // sliding layers, so its full-attention layers get no rotation
        // at all and no GGUF key says so. That is now IMPLEMENTED rather
        // than refused -- `capability::swa_disabled_by_arch` carries the
        // layer-count gate and `crate::rope_layers` the per-layer
        // rotation rule it feeds -- so the two EXAONE-4 sizes are one
        // code path with two answers instead of one running and one
        // stopping. `tests/no_rope_layer_graphs.rs` has a 64-layer
        // fixture against libllama's own logits.
        let hidden_dim = file
            .metadata_u64(&key("embedding_length"))
            .ok_or_else(|| LoadError::MissingHparam(key("embedding_length")))?
            as usize;
        // Scalar OR per-layer array, as llama.cpp reads all three
        // (`get_key_or_arr`, llama-model.cpp:1149-1158). The scalars
        // below are the WIDEST layer's; `ModelConfig::layer_shape` is
        // what a layer body reads. See `crate::layer_shapes`.
        let heads_per_layer =
            crate::layer_shapes::read_u64_trunk_layers(file, &key("attention.head_count"), &trunk)?
                .ok_or_else(|| LoadError::MissingHparam(key("attention.head_count")))?;
        let n_heads = heads_per_layer.iter().copied().max().unwrap_or(0) as usize;

        let mut best_effort_fields: Vec<&'static str> = Vec::new();

        let kv_heads_per_layer = match crate::layer_shapes::read_u64_trunk_layers(
            file,
            &key("attention.head_count_kv"),
            &trunk,
        )? {
            Some(v) => v,
            None => {
                best_effort_fields.push("n_kv_heads (no attention.head_count_kv key; assumed equal to n_heads, i.e. plain MHA)");
                heads_per_layer.clone()
            }
        };
        let n_kv_heads = kv_heads_per_layer.iter().copied().max().unwrap_or(0) as usize;
        let head_dim = match file.metadata_u64(&key("attention.key_length")) {
            Some(v) => v as usize,
            None => {
                // llama.cpp derives it from LAYER 0's head count
                // (`n_embd / n_head()`, llama-model.cpp:1195), which on
                // a file whose layer 0 has none is a division by zero
                // there and a refusal here.
                let h0 = heads_per_layer.first().copied().unwrap_or(0) as usize;
                // A pure recurrent model has no heads and no head width
                // (`layer_shapes::PURE_RECURRENT`); nothing reads one.
                match hidden_dim.checked_div(h0) {
                    _ if h0 == 0 && facts.topology.pure_recurrent_block.is_some() => 0,
                    None => {
                        return Err(LoadError::MissingHparam(format!(
                            "{} (layer 0 declares head_count 0, so it cannot be derived as \
                             hidden_dim / n_heads)",
                            key("attention.key_length")
                        )));
                    }
                    Some(derived) => {
                        best_effort_fields.push(
                            "head_dim (no attention.key_length key; derived as hidden_dim / n_heads)",
                        );
                        derived
                    }
                }
            }
        };
        let v_head_dim = crate::kv_head_dims::resolve_v_head_dim(
            &arch,
            head_dim,
            file.metadata_u64(&key("attention.value_length"))
                .map(|v| v as usize),
        )?;
        // `Some` only when it differs: see `ModelConfig::v_head_dim`.
        let v_head_dim = (v_head_dim != head_dim).then_some(v_head_dim);
        let vocab_size = file
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(items) => Some(items.len()),
                _ => None,
            })
            .or_else(|| file.metadata_u64(&key("vocab_size")).map(|v| v as usize))
            .unwrap_or_else(|| {
                best_effort_fields.push("vocab_size (no tokenizer.ggml.tokens array or {arch}.vocab_size key; fell back to output.weight's own row count)");
                // `output.weight`'s real raw shape is `[hidden_dim,
                // vocab_size]` (ggml's fastest-first `ne[]` order --
                // see `load_weight_matrix`'s doc comment), so vocab_size
                // is the *last* element, not the first.
                file.find_tensor("output.weight")
                    .and_then(|t| t.shape.last().copied())
                    .unwrap_or(0) as usize
            });
        let rope_theta = metadata_f32_any(file, &[key("rope.freq_base")]).unwrap_or_else(|| {
            best_effort_fields.push("rope_theta (no rope.freq_base key; defaulted to 10000.0)");
            10000.0
        });
        // NTK-alpha: `{arch}.rope.scaling.alpha` is read for every
        // architecture (llama-model.cpp:1186) and APPLIED by two
        // (`hunyuan-vl.cpp:8-12`, inherited by `hunyuan-dense`). The
        // list and the arithmetic live together in one module so the
        // key's readers and its appliers cannot drift apart -- see
        // `crate::rope_ntk_alpha`, which also records why a converted
        // `hunyuan-dense` file carries the already-scaled base instead.
        let rope_theta = crate::rope_ntk_alpha::ntk_alpha_scaled_rope_base(
            &arch,
            rope_theta,
            head_dim,
            metadata_f32_any(file, &[key("rope.scaling.alpha")]),
        );
        // The norm FUNCTION is the architecture's, except where the file
        // decides it (`crate::norm::NORM_BY_RMS_EPS_KEY`): a present and
        // nonzero RMS epsilon means RMSNorm there, and a zero one is
        // llama.cpp's "absent", so it is dropped before the epsilon
        // itself is read below.
        let declared_rms_eps = metadata_f32_any(file, &[key("attention.layer_norm_rms_epsilon")])
            .filter(|eps| *eps != 0.0 || !facts.norm.norm_by_rms_eps_key);
        let norm_function = crate::norm::norm_function_for_file(&arch, declared_rms_eps);
        let rms_norm_eps = declared_rms_eps
            .or_else(|| metadata_f32_any(file, &[key("attention.layer_norm_epsilon")]))
            .unwrap_or_else(|| {
                best_effort_fields
                    .push("rms_norm_eps (no layer_norm_rms_epsilon key; defaulted to 1e-5)");
                1e-5
            });

        let n_experts = metadata_u64_any(file, &[key("expert_count")]).unwrap_or(0) as usize;
        let is_moe = n_experts > 1;

        // `expert_used_count` is scalar OR an array at `block_count`
        // length: `llama-model.cpp:1266` reads it with `get_key_or_arr`
        // in the COMMON loader, for every architecture, and
        // `gguf_writer.py:869-873` writes whichever it is handed.
        // `conversion/nemotron.py:574` hands it a LIST (Nemotron-H
        // Puzzle, one entry per block), and `nemotron_h` is an
        // architecture frink serves -- so before 2026-09-19 such a
        // file read no scalar here, fell into the default, and routed
        // top-2 on every layer whatever the file said. A uniform array
        // is that one value; a varying one needs a per-layer top-k the
        // MoE layer does not have and stops by name rather than
        // picking a number.
        let n_experts_active = if is_moe {
            match crate::layer_shapes::read_u64_per_layer(
                file,
                &key("expert_used_count"),
                // `n_layer_all`, i.e. `block_count` including any MTP
                // blocks, which is the length llama.cpp asks for at
                // `llama-model.cpp:1266` -- BEFORE `n_layer()` drops
                // them.
                block_count,
            )? {
                Some(per_layer) => {
                    let first = per_layer[0];
                    if per_layer.iter().any(|v| *v != first) {
                        return Err(LoadError::UnsupportedFeature(
                            key("expert_used_count"),
                            format!(
                                "a PER-LAYER expert count ({per_layer:?}). llama.cpp reads this \
                                 key with `get_key_or_arr` for every architecture \
                                 (llama-model.cpp:1266) and routes layer `il` to \
                                 `n_expert_used_arr[il]` experts; frink carries one top-k for \
                                 the model, so it would route every layer to {first} and answer \
                                 something else. conversion/nemotron.py:574 writes the array for \
                                 Nemotron-H Puzzle"
                            ),
                        ));
                    }
                    first as usize
                }
                None => {
                    best_effort_fields
                        .push("moe.n_experts_active (no expert_used_count key; defaulted to 2)");
                    2
                }
            }
        } else {
            1
        };
        // Read here, ahead of the shared-expert inference below, because
        // the tensor that inference probes lives on the first MoE
        // layer, not on layer 0.
        let n_dense_leading_layers = if facts.moe.leading_dense_key_is_inert {
            0
        } else {
            metadata_u64_any(file, &[key("leading_dense_block_count")]).unwrap_or(0) as usize
        };
        // Prefer the GGUF hparam when present. Qwen2MoE (and some other
        // HF→GGUF exports) omit `expert_shared_count` but still ship
        // `blk.N.ffn_{gate,up,down}_shexp.weight` -- without a tensor-
        // presence fallback those weights are silently dropped and the
        // model runs with a large chunk of active FFN missing.
        //
        // The probe is the FIRST MoE LAYER, not `blk.0`: a leading-dense
        // model has no shared expert on layer 0, and probing there
        // answered 0 for every such file. `laguna` is the case that
        // found it -- `laguna.cpp:20` assigns `n_expert_shared = 1`
        // before reading the key, `conversion/laguna.py` never writes
        // the key, and its layer 0 is dense (:105), so a real Laguna
        // export loaded with its three REQUIRED `_shexp` tensors
        // (:138-140) unread on every MoE layer.
        // The interleave step the loader honours (`crate::moe_interleave`,
        // `llama4.cpp:64`), read here because the shared-expert probe
        // below needs the first layer it makes MoE.
        let moe_interleave_step = crate::moe_interleave::interleave_step(
            &arch,
            metadata_u64_any(file, &[key("interleave_moe_layer_step")]),
            n_experts,
        )
        .map_err(|reason| LoadError::UnsupportedFeature(arch.clone(), reason))?;
        let first_moe_layer = (n_dense_leading_layers..n_layers)
            .find(|&il| !moe_interleave_step.is_some_and(|step| !(il + 1).is_multiple_of(step)))
            .unwrap_or(n_layers.saturating_sub(1));
        let shexp_probe = format!("blk.{first_moe_layer}.ffn_gate_shexp.weight");
        let n_shared_experts = match metadata_u64_any(file, &[key("expert_shared_count")]) {
            Some(n) => n as usize,
            None if is_moe && file.find_tensor(&shexp_probe).is_some() => {
                best_effort_fields.push(
                    "moe.n_shared_experts (no expert_shared_count; inferred 1 from the first \
                     MoE layer's ffn_gate_shexp.weight)",
                );
                1
            }
            None => 0,
        };
        // MoE GGUFs often only set `feed_forward_length` (OLMoE=1024,
        // Qwen2-MoE=5632 for the shared expert). `expert_feed_forward_length`
        // is optional. llama.cpp `qwen2moe.cpp` uses
        // `n_ff_exp = n_ff_exp ? n_ff_exp : n_ff / n_expert_used` (1408 for
        // Qwen1.5-MoE); the shared expert keeps the full `n_ff` (5632).
        let ffn_per_layer =
            crate::layer_shapes::read_u64_trunk_layers(file, &key("feed_forward_length"), &trunk)?
                // Qwen-1 declares gate and up as one number; see
                // `FFN_LENGTH_COUNTS_GATE_AND_UP`.
                .map(|v| {
                    if facts.ffn.length_counts_gate_and_up {
                        v.into_iter().map(|ff| ff / 2).collect()
                    } else {
                        v
                    }
                });
        let feed_forward_length = ffn_per_layer.as_ref().and_then(|v| v.iter().copied().max());
        // Scalar OR an array, exactly as `expert_used_count` above:
        // llama.cpp reads it with `get_key_or_arr` (`maple.cpp:6`,
        // `dots3note.cpp:11`, `nemotron-h.cpp`), and
        // `conversion/nemotron.py:573` writes a LIST for Nemotron-H
        // Puzzle -- an architecture frink serves. Read as a scalar
        // alone, an array answered `None` here and the fallback below
        // silently sized every expert at `feed_forward_length /
        // n_experts_used`, which is a different FFN and loads without
        // complaint when the tensors happen to be that wide.
        let expert_ffn_per_layer = crate::layer_shapes::read_u64_per_layer(
            file,
            &key("expert_feed_forward_length"),
            block_count,
        )?;
        if let Some(per_layer) = expert_ffn_per_layer.as_ref() {
            let first = per_layer[0];
            if per_layer.iter().any(|v| *v != first) {
                return Err(LoadError::UnsupportedFeature(
                    key("expert_feed_forward_length"),
                    format!(
                        "a PER-LAYER expert FFN width ({per_layer:?}). llama.cpp sizes the \
                         expert tensors from layer 0's entry and `LayerShapes` carries one \
                         expert width for the model, so frink would build every layer at \
                         {first} and read the others' weights at the wrong stride"
                    ),
                ));
            }
        }
        let expert_ffn_dim = expert_ffn_per_layer
            .map(|v| v[0])
            .or_else(|| {
                feed_forward_length.map(|ff| {
                    if is_moe && n_experts_active > 0 {
                        ff / n_experts_active as u64
                    } else {
                        ff
                    }
                })
            })
            .unwrap_or_else(|| {
                best_effort_fields.push(
                    "moe.expert_ffn_dim (no expert_feed_forward_length/feed_forward_length; defaulted to 4x hidden_dim)",
                );
                (hidden_dim * 4) as u64
            }) as usize;
        // `{arch}.attention.rope_pattern`: one entry per layer, nonzero
        // meaning "this layer rotates" (`llama-hparams.cpp:333-343`).
        // Read only where llama.cpp reads it, because
        // `llama-model.cpp:1314` seeds the array with 1 for every
        // architecture and only `granite-swa.cpp:43` reads it back --
        // so honouring it elsewhere would answer differently from
        // upstream on a file that carries it as dead metadata.
        let rope_pattern: Option<std::sync::Arc<[bool]>> = if facts.topology.reads_rope_pattern {
            crate::layer_shapes::read_u64_per_layer(
                file,
                &key("attention.rope_pattern"),
                block_count,
            )?
            .map(|v| v.into_iter().map(|x| x != 0).collect())
        } else {
            None
        };

        // Qwen3.5's recurrent layers come from two keys, not from the
        // head counts (`crate::gdn::recurrent_layers`).
        let recurrent_layers =
            crate::gdn::recurrent_layers(file, &arch, trunk.block_count, n_layers)?;
        let layer_shapes = crate::layer_shapes::LayerShapes::resolve(
            &arch,
            &heads_per_layer,
            &kv_heads_per_layer,
            ffn_per_layer.as_deref(),
            expert_ffn_dim,
            recurrent_layers.as_ref(),
        )?
        // `nanbeige.cpp:24-26` copies each physical layer's shape
        // arrays to every logical slot; HRM-Text's two stacks are
        // uniform and its arrays are scalars, so the replication is
        // spelled for the one schedule that needs it rather than
        // divided out of the other's counts.
        .replicated(match layer_loops {
            Some(crate::layer_loops::LayerLoops::Repeat { n_loops, .. }) => n_loops,
            _ => 1,
        });
        // The OTHER half of llama.cpp's dense-vs-MoE rule.
        // `ModelConfig::layer_is_dense` implements the leading-dense
        // prefix and not the `(il + 1) % n_moe_layer_step == 0` at
        // `src/models/ernie4-5-moe.cpp:64`, so a file whose step would
        // change the answer stops here rather than looking for expert
        // tensors on a layer that stores dense ones. `moe_interleave`
        // records what building the fixture found: llama.cpp cannot load
        // such a file either, because its own tensor loader has no step
        // in it.
        if let Some(reason) = crate::moe_interleave::interleave_step_refusal(
            &arch,
            metadata_u64_any(file, &[key("interleave_moe_layer_step")]),
        ) {
            return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
        }

        // ik_llama.cpp's real gating-function hparam
        // (LLM_KV_EXPERT_GATING_FUNC: 1=softmax, 2=sigmoid) if the file
        // carries it; otherwise fall back to the same architecture-name
        // convention the hand-written presets in config.rs use (see
        // docs/MODELS.md for the citations behind that list).
        let gating = match (
            facts.moe.gating_literal,
            metadata_u64_any(file, &[key("expert_gating_func")]),
        ) {
            (Some(literal), _) => literal,
            (None, Some(2)) => GatingFunction::Sigmoid,
            (None, Some(1)) => GatingFunction::Softmax,
            (None, _) => {
                if facts.moe.sigmoid_by_default {
                    GatingFunction::Sigmoid
                } else {
                    if is_moe {
                        best_effort_fields.push(
                            "moe.gating (no expert_gating_func key and architecture not in the known-sigmoid list; defaulted to softmax)",
                        );
                    }
                    GatingFunction::Softmax
                }
            }
        };

        // `{arch}.expert_weights_norm` (llama.cpp
        // `LLM_KV_EXPERT_WEIGHTS_NORM`) is the real metadata key for
        // whether the selected experts' weights are renormalised. Most
        // checkpoints do not carry it, which is why the fallback below
        // exists at all -- but when one does, the file's own answer wins
        // over an architecture-name guess.
        // The key only where llama.cpp reads it (`EXPERT_WEIGHTS_NORM_READERS`);
        // everywhere else the graph's literal, which the table below
        // transcribes, whatever the file says.
        let norm_key = if facts.moe.reads_expert_weights_norm {
            file.metadata_bool(&key("expert_weights_norm"))
        } else {
            None
        };
        let norm_topk_prob = match norm_key {
            Some(v) => v,
            None => {
                // See `NO_TOPK_RENORMALIZE_ARCHITECTURES`'s doc comment:
                // an architecture-name lookup, the same convention
                // `gating`'s fallback above uses.
                if is_moe && matches!(gating, GatingFunction::Softmax) {
                    best_effort_fields.push(
                        "moe.norm_topk_prob (no expert_weights_norm key; defaulted by architecture-name lookup against NO_TOPK_RENORMALIZE_ARCHITECTURES)",
                    );
                }
                !facts.moe.no_topk_renormalize
            }
        };

        // `{arch}.expert_weights_scale` (`LLM_KV_EXPERT_WEIGHTS_SCALE`).
        // llama.cpp's `build_moe_ffn` skips the multiply for both 0.0 and
        // 1.0, so both mean "no scaling" and both land on 1.0 here.
        // And only where llama.cpp reads it (`EXPERT_WEIGHTS_SCALE_READERS`).
        let expert_weights_scale = if facts.moe.reads_expert_weights_scale {
            metadata_f32_any(file, &[key("expert_weights_scale")])
                .filter(|s| *s != 0.0)
                .unwrap_or(1.0)
        } else {
            1.0
        };

        // Real GGUF key (`{arch}.attention.sliding_window`, confirmed
        // against `gguf-py/gguf/constants.py`'s real
        // `LLM_KV_ATTENTION_SLIDING_WINDOW`). Some checkpoints
        // (confirmed for real published Qwen1.5-MoE/Qwen2-MoE GGUFs)
        // carry a nonzero window value even when the model's own
        // config disables sliding-window attention entirely
        // (`use_sliding_window: false`) -- llama.cpp's own convention
        // is that a window of 0 means "unused," so only a real nonzero
        // value here is treated as active.
        let declared_window = metadata_u64_any(file, &[key("attention.sliding_window")]);
        let sliding_window = declared_window
            .map(|v| v as usize)
            .filter(|&w| w > 0)
            // `phi3` declares a window that llama.cpp deliberately does
            // NOT honour -- see `capability::swa_disabled_by_arch`. This
            // has to drop the window rather than pick a period, because
            // upstream is declining to use the file's value, not
            // choosing a different one.
            //
            // `block_count`, NOT `n_layers`: `exaone4.cpp:4` tests
            // `n_layer() == 64` at a point where `n_layer_nextn` has not
            // been read yet (`:18`), so a 64-trunk EXAONE-4 with an MTP
            // block appended sees 65 there and gets no window.
            //
            // `smallthinker` declares a window that llama.cpp REPLACES:
            // `smallthinker.cpp:8` assigns `n_swa = 4096` on the branch
            // the file's nonzero value selected. One table decides all
            // three answers (`capability::swa_window_override`), so a
            // row cannot be dropped by one reader and pinned by another.
            .and_then(
                |w| match (facts.rules.swa_window_override)(&arch, trunk.block_count) {
                    crate::capability::SwaWindowOverride::Honour => Some(w),
                    crate::capability::SwaWindowOverride::Drop => None,
                    crate::capability::SwaWindowOverride::Pin(pinned) => Some(pinned),
                },
            );

        // A CHUNKED window (`crate::chunked_swa`): the literal chunk on
        // the branch the file takes, whatever nonzero value it declares
        // and whether it declares one at all; a declared ZERO is the
        // branch libllama aborts on, refused there by name.
        let swa_chunked = crate::chunked_swa::chunked_window(&arch, declared_window)?;
        let sliding_window = swa_chunked.or(sliding_window);
        let swa_chunked = swa_chunked.is_some();

        // A window llama.cpp REQUIRES (`crate::swa_geometry::
        // window_required`): without it the file does not load upstream,
        // and here it would run with no layer rotated.
        if let (None, Some(line)) = (sliding_window, facts.swa.window_required) {
            return Err(LoadError::UnsupportedFeature(
                arch.clone(),
                format!(
                    "`{arch}.attention.sliding_window` is absent or zero, and llama.cpp reads it \
                     as a REQUIRED key for this architecture (src/models/{line}); every real \
                     export writes it, and a file without it does not load upstream"
                ),
            ));
        }

        // A sliding-layer RoPE base llama.cpp REQUIRES
        // (`crate::swa_geometry::swa_rope_base_required`): without it the
        // file does not load upstream, and here the sliding layers would
        // rotate at a fallback base.
        if let Some(line) = facts.swa.rope_base_required {
            if metadata_f32_any(file, &[key("rope.freq_base_swa")]).is_none() {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!(
                        "`{arch}.rope.freq_base_swa` is absent, and llama.cpp reads it as a \
                         REQUIRED key for this architecture (src/models/{line}); a file without \
                         it does not load upstream"
                    ),
                ));
            }
        }

        // Three graphs rope their SLIDING layers with the scaling
        // switched off -- freq_scale = 1, ext_factor = 0, attn_factor =
        // 1 -- while the full-attention layers use the model's:
        // `olmo2` (Olmo-3), `mellum` and `laguna` (Laguna-XS.2), each
        // at the lines `crate::swa_geometry` cites. frink's
        // `RopeFreqs` already keeps their sliding layers' divisors
        // unscaled, but `rope_attn_factor` is one value for the whole
        // model, so honouring the file would mean rotating half the
        // layers at a magnitude the checkpoint never trained at.
        //
        // A window with NO scaling is not this case and is not refused:
        // both branches then reduce to the same plain RoPE, and the
        // difference is masking alone, which frink implements.
        if let (Some(lines), true) = (facts.swa.layers_unscaled_rope, sliding_window.is_some()) {
            let scaling_type = file
                .metadata_str(&key("rope.scaling.type"))
                .unwrap_or("none")
                .to_string();
            if !scaling_type.eq_ignore_ascii_case("none") {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!(
                        "this {arch} checkpoint declares BOTH a sliding window and \
                         rope.scaling.type = \"{scaling_type}\". llama.cpp ropes the \
                         sliding layers with the scaling switched off (freq_scale = 1, \
                         ext_factor = 0, attn_factor = 1; {lines}) and the \
                         full-attention layers with it on, and frink carries one RoPE \
                         scaling for the whole model. A {arch} file with a window and no \
                         scaling, or with scaling and no window, is unaffected"
                    ),
                ));
            }
        }

        // WHICH LAYERS SLIDE (`attention.sliding_window_pattern`).
        //
        // llama.cpp seeds a period per architecture (gemma2 2, gemma3
        // 6, exaone4 4, ...) and only then reads the key, so a missing
        // key must NOT mean "all SWA": the seed is
        // `capability::default_swa_layout`, with the gemma3+ period for
        // any Gemma variant the table does not name. The phase is a
        // property of the architecture -- `dense_first` is an argument
        // to `set_swa_pattern`, not a GGUF key -- so it comes from the
        // seed either way.
        //
        // The key itself is a scalar period OR a per-layer bool array,
        // and which of the two an architecture's graph honours -- and
        // what it does with the other -- is `crate::swa_layers`'s
        // table, transcribed from the `get_key_or_arr` overload each
        // `load_arch_hparams` calls. The array used to be REFUSED here
        // for every architecture, which stopped every real EXAONE-4
        // 32B, EXAONE-MoE and Olmo-3 export at the door over a value
        // llama.cpp never reads for them.
        let swa_seed = facts.swa.default_layout.or(match arch_profile.family {
            crate::capability::DecoderFamily::GemmaFamily => Some(crate::capability::SwaPattern {
                period: 6,
                dense_first: false,
            }),
            _ => None,
        });
        let swa_layers = match sliding_window {
            // No window: no graph consults `is_swa`, and reading the
            // key would only refuse a file over a value nothing uses.
            None => crate::swa_layers::SwaLayers::All,
            Some(_) => crate::swa_layers::read_swa_layers(
                file,
                &arch,
                &key("attention.sliding_window_pattern"),
                &trunk,
                swa_seed,
            )?,
        };
        // A window on a short-conv architecture, read the way its
        // generation's graph reads it (`crate::shortconv`'s module doc):
        // lfm2's scalar is refused, lfm3's array is served unless it
        // marks a conv layer or no layer at all.
        if let Some(reason) = crate::shortconv::shortconv_window_refusal(
            &arch,
            sliding_window,
            trunk.n_layers,
            |il| swa_layers.slides(il),
            &kv_heads_per_layer,
        ) {
            return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
        }

        // The metadata-declared scalar multipliers, resolved once for
        // whichever subset this architecture's reference graph applies.
        // See `crate::scalar_multipliers`; the keys the graph does NOT
        // apply were already refused above, by a list derived from the
        // same table. Read first because its `defaults` also seed the
        // softcap below: `grok.cpp:5-12` assigns all of them in one
        // place, and so does this.
        let multiplier_support = facts.scaling.multipliers;

        // The file's softcap, then the architecture's default for a
        // file that declares none (`grok.cpp:9`), then llama.cpp's own
        // "off". `> 0.0` is what every graph tests before applying one.
        let attn_logit_softcap = metadata_f32_any(
            file,
            &[
                key("attention.logit_softcapping"),
                key("attn_logit_softcapping"),
            ],
        )
        .or(multiplier_support.defaults.attn_logit_softcap())
        .filter(|&v| v > 0.0);
        let final_logit_softcap =
            metadata_f32_any(file, &[key("final_logit_softcapping")]).filter(|&v| v > 0.0);

        let declared = crate::scalar_multipliers::DeclaredMultipliers {
            logit: metadata_f32_any(file, &[key("logit_scale")]),
            residual: metadata_f32_any(file, &[key("residual_scale")]),
            embedding: metadata_f32_any(file, &[key("embedding_scale")]),
            // Exactly the spelling this architecture's graph reads --
            // `attention.scale` for Granite, `attention.output_scale`
            // for Grok -- and nothing for the rest. The other spelling
            // was refused above, by the same table.
            attention: multiplier_support
                .attention
                .suffix()
                .and_then(|suffix| metadata_f32_any(file, &[key(suffix)])),
        };
        let multipliers = crate::scalar_multipliers::resolve(
            multiplier_support,
            declared,
            // `n_layer` and `n_embd` are here for MiniCPM's defaults
            // (`minicpm.cpp:6-7`), which are computed from the model's
            // own shape rather than declared: an older MiniCPM export
            // carries none of the three keys and is still scaled by all
            // three.
            crate::scalar_multipliers::MultiplierDims {
                head_dim,
                n_layer: n_layers,
                n_embd: hidden_dim,
            },
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;

        // Gemma and afmoe: embeddings are scaled by sqrt(hidden_dim) at
        // input. That is ARITHMETIC, not a key -- those graphs read no
        // `embedding_scale` at all -- so it comes from the table and a
        // file declaring the key on one of them is refused above rather
        // than honoured. Granite's comes out of `{arch}.embedding_scale`.
        let embedding_scale = if facts.scaling.embeddings_scaled_by_sqrt_n_embd {
            Some((hidden_dim as f32).sqrt())
        } else {
            multipliers.embedding_scale
        };

        // llama.cpp's `f_attention_scale`, and ONLY where it differs from
        // the `1/sqrt(head_dim)` frink's attention kernels already
        // apply -- `Some` here means "pre-scale Q", so restating the
        // kernels' own scale would double-scale every score.
        //
        // For Gemma-2 and Gemma-3 that difference is real at 27B and
        // nowhere else (`capability::attention_scale_override` carries
        // the llama.cpp lines). This used to be a hardcoded `None` under
        // a comment that NAMED the 27B exception without implementing
        // it, so Gemma-2-27B scored 1.061x and Gemma-3-27B 1.146x too
        // large on every layer: a sharper softmax than the trained one,
        // fluent and wrong, with no error.
        //
        // Granite reaches the same slot from the file's own
        // `{arch}.attention.scale` (`granite.cpp:225`, whose `0.0f`
        // sentinel means "use the kernels' scale"). The two sources
        // cannot both be live on one architecture: `attention_scale_override`
        // covers the architectures that COMPUTE the scale and
        // `scalar_multipliers` the ones that READ it, and no llama.cpp
        // architecture does both. `.or` rather than a match because the
        // computed one is the one that cannot be turned off by a file.
        let attention_scale =
            (facts.rules.attention_scale_override)(&arch, n_layers, hidden_dim, n_heads, head_dim)
                .or(multipliers.attention_scale);

        // Granite reads `{arch}.rope.scaling.finetuned` as a switch for
        // RoPE itself, not as a note about the scaling: a file declaring
        // it false runs UNROTATED in llama.cpp (every Granite-4.0 hybrid
        // export), which is `RopeLayers::Never` below
        // (`crate::rope_finetuned`).
        let rope_switched_off = crate::rope_finetuned::unrotated(
            &arch,
            file.metadata(&key("rope.scaling.finetuned"))
                .and_then(GgufValue::as_bool),
        );

        // OLMo-1 and DBRX clamp Q, K and V by `{arch}.attention.clamp_kqv`
        // inside the shared `build_qkv`. Resolved here for the
        // architectures whose loader reads the key, REQUIRED where
        // llama.cpp's is (`dbrx.cpp:5`), and applied by the one helper
        // every host body shares (`decoder/qkv_bias.rs`). See
        // `crate::clamp_kqv`, which also records that both converters
        // really write this key.
        let clamp_kqv = crate::clamp_kqv::resolve_clamp(
            &arch,
            metadata_f32_any(file, &[key("attention.clamp_kqv")]),
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;

        // SWA-layer RoPE base. `llama_hparams` defaults it to 10000 and
        // the Gemma-3 lineage relies on that default; the architectures
        // in `swa_rope_base_follows_model` instead seed it from the
        // model's own base before the key can override.
        let rope_theta_swa = if sliding_window.is_some() {
            let fallback = if facts.swa.rope_base_follows_model {
                rope_theta
            } else {
                10_000.0
            };
            Some(
                metadata_f32_any(
                    file,
                    &[key("rope.freq_base_swa"), key("rope_freq_base_swa")],
                )
                .unwrap_or(fallback),
            )
        } else {
            None
        };

        let ffn_activation = match arch_profile.family {
            // Per-ARCHITECTURE first, because llama.cpp's choice is per
            // architecture and the family partition does not match it:
            // `grok` is StandardGqa and passes `LLM_FFN_GELU`.
            _ if facts.ffn.geglu => crate::config::FfnActivation::Gelu,
            _ if facts.ffn.relu_sqr => crate::config::FfnActivation::ReluSqr,
            _ if facts.ffn.gelu_ungated => crate::config::FfnActivation::GeluUngated,
            // The GATED ReLU (`ggml_reglu_split`), a real gate tensor:
            // NOT the row above, which aliases gate to up.
            _ if facts.ffn.reglu => crate::config::FfnActivation::Reglu,
            // The four per-layer arrays travel IN the variant, read as
            // `apertus.cpp:6-9` reads them (`crate::act_layers`).
            _ if facts.ffn.xielu => crate::config::FfnActivation::Xielu(
                crate::act_layers::read_xielu_layers(file, trunk.n_layers)?,
            ),
            // The two clamp arrays, read as `step35.cpp:28-29` read them
            // (optional; a file with neither is plain SwiGLU).
            _ if facts.ffn.reads_swiglu_clamps => {
                match crate::act_layers::read_swiglu_clamps(file, &arch, &trunk)? {
                    Some(clamps) => crate::config::FfnActivation::SwigluClamped(clamps),
                    None => crate::config::FfnActivation::Swiglu,
                }
            }
            crate::capability::DecoderFamily::GemmaFamily => crate::config::FfnActivation::Gelu,
            crate::capability::DecoderFamily::PhiFamily => {
                crate::config::FfnActivation::SwigluFused
            }
            _ => crate::config::FfnActivation::Swiglu,
        };

        // Llama 3/3.1/3.2's real per-band RoPE frequency correction: one
        // model-level tensor (`TENSOR_NOT_REQUIRED`, `TENSOR_DUPLICATED`
        // for every layer but the first in the real llama.cpp source --
        // i.e. every layer shares this same array), not per-layer. See
        // `frink_core::attention::apply_rope_with_freq_factors`'s doc
        // comment for why this matters.
        let rope_freqs = load_f32_vec_optional(file, "rope_freqs.weight")?;

        // Phi-3/Phi-4 LongRoPE: two per-band factor tensors instead of
        // Llama's single `rope_freqs.weight`, selected by context size
        // (llama.cpp `llama_model::get_rope_factors`: `rope_freqs` wins if
        // present, else `rope_long` when the run's context exceeds
        // `rope.scaling.original_context_length`, else `rope_short`).
        //
        // Provisional pick from the checkpoint's advertised context length
        // (llama.cpp's default `n_ctx`). The definitive pick happens in
        // `ModelConfig::apply_runtime_context`, called from `frink run`
        // (`--ctx-size`) and from `verify_engine::load_and_tokenize`
        // (`n_tokens + 8`, matching `tools/llama_logits.c`).
        let rope_orig_ctx = metadata_u64_any(file, &[key("rope.scaling.original_context_length")])
            .map(|v| v as usize);

        // The per-position attention temperature (`crate::attn_temperature`).
        // `mistral3.cpp:15` floors it on `hparams.n_ctx_orig_yarn`, which
        // `llama-model.cpp:1164-1165` seeds from `context_length` BEFORE
        // the YaRN key overrides it -- so a Ministral file with no YaRN
        // key floors on its context length, and the resolver is handed
        // that value rather than the key. Before this existed the key
        // loaded and was silently dropped on the one generic-path
        // architecture whose graph applies it.
        let attn_temperature = crate::attn_temperature::resolve_attn_temperature(
            &arch,
            crate::attn_temperature::DeclaredTemperature {
                scale: metadata_f32_any(file, &[key("attention.temperature_scale")]),
                length: metadata_u64_any(file, &[key("attention.temperature_length")]),
                n_ctx_orig_yarn: rope_orig_ctx
                    .map(|v| v as u64)
                    .or_else(|| metadata_u64_any(file, &[key("context_length")])),
            },
        )
        .map_err(|e| LoadError::UnsupportedFeature(arch.clone(), e.message(&arch)))?;
        // `rope_freqs.weight` outranks the LongRoPE pair (llama.cpp
        // `get_rope_factors` checks it first), so a checkpoint carrying
        // it never populates these and the runtime re-pick below cannot
        // overwrite a Llama-3 correction with a Phi one.
        let (rope_freqs_long, rope_freqs_short) = if rope_freqs.is_some() {
            (None, None)
        } else {
            (
                load_f32_vec_optional(file, "rope_factors_long.weight")?,
                load_f32_vec_optional(file, "rope_factors_short.weight")?,
            )
        };
        // Provisional pick from the checkpoint's own advertised context;
        // `ModelConfig::apply_runtime_context` re-picks once the run's
        // `--ctx-size` is known, which is the number llama.cpp decides on.
        let rope_freqs = match (rope_freqs, rope_orig_ctx) {
            (Some(f), _) => Some(f),
            (None, Some(orig)) => {
                let model_ctx = metadata_u64_any(file, &[key("context_length")])
                    .unwrap_or(orig as u64) as usize;
                if model_ctx > orig {
                    rope_freqs_long.clone().or_else(|| rope_freqs_short.clone())
                } else {
                    rope_freqs_short.clone().or_else(|| rope_freqs_long.clone())
                }
            }
            (None, None) => None,
        };

        // Partial rotary: the file's `rope.dimension_count`, or the head
        // width when absent -- llama.cpp's seeded `n_rot_full`
        // (`llama-model.cpp:1200-1202`).
        let rope_dim_seeded = metadata_u64_any(file, &[key("rope.dimension_count")])
            .map(|d| d as usize)
            .filter(|d| *d > 0)
            .unwrap_or(head_dim);

        // The sliding layers' OWN rotary and head widths
        // (`llama-model.cpp:1215-1223`). The rotary one is honoured --
        // `crate::swa_geometry` resolves the key and step35's halving
        // into the two widths `ModelConfig::layer_rope` hands out -- and
        // the head ones are refused, since frink carries one head width
        // in every cache. Only a model with a sliding layer reads the
        // keys; on one that has none they are dead metadata, as they
        // are upstream (`n_rot(il)` never takes the `_swa` branch). The
        // halving is not a key and applies regardless.
        let geometry = crate::swa_geometry::SwaGeometry {
            rope_dim_swa: metadata_u64_any(file, &[key("rope.dimension_count_swa")]),
            key_length_swa: metadata_u64_any(file, &[key("attention.key_length_swa")]),
            value_length_swa: metadata_u64_any(file, &[key("attention.value_length_swa")]),
            rope_dim_full: rope_dim_seeded as u64,
            head_dim: head_dim as u64,
        };
        let widths = if sliding_window.is_some() || facts.swa.full_layers_rotate_half.is_some() {
            if let Some(reason) = crate::swa_geometry::swa_geometry_refusal(&arch, geometry) {
                return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
            }
            crate::swa_geometry::rotary_widths(&arch, geometry)
        } else {
            crate::swa_geometry::RotaryWidths {
                full: Some(rope_dim_seeded).filter(|d| *d < head_dim),
                swa: None,
            }
        };
        // Equal values mean "whole head", which is the same thing as
        // `None` and stays `None` so nothing downstream has to
        // special-case it.
        let rope_dim = widths.full;
        let rope_dim_swa = widths.swa;

        // See `ModelConfig::rope_attn_factor`. `mut` because YaRN's
        // magnitude term is folded into it below.
        let mut rope_attn_factor = metadata_f32_any(file, &[key("rope.scaling.attn_factor")])
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(1.0);

        // YaRN long-context scaling. `rope.scaling.attn_factor` above is
        // only YaRN's *magnitude* term (ggml `rope_yarn`'s `mscale`); the
        // frequency half -- which bands get interpolated toward the
        // trained context and which stay extrapolated -- lives in
        // `rope.scaling.type` + `rope.scaling.factor`, and frink read
        // neither before this. A YaRN checkpoint was therefore roped as
        // if it declared no scaling at all: right near position 0 and
        // progressively wrong further in, i.e. the failure that reads as
        // long-prompt quality decay rather than as a bug.
        //
        // The rewrite is folded into `rope_freqs`, the same per-band
        // divisor array Llama-3's `rope_freqs.weight` supplies (ggml
        // divides each band's theta by it), so it rides the existing CPU
        // and Metal RoPE paths unchanged. When a file carries both, the
        // two corrections compose by multiplication, as they do in
        // llama.cpp (`ggml_rope_cache_init` divides by `freq_factors`
        // *and then* runs `rope_yarn`).
        // Linear scaling, which was silently DROPPED before this.
        //
        // `rope.scaling.type = "linear"` with factor s means rotating
        // position `p/s` instead of `p`. Since the angle is `p * freq`,
        // that is exactly `p * (freq / s)`, and `rope_freqs` already
        // divides each band's frequency. So a uniform vector of `s`
        // expresses it exactly and rides the existing CPU and Metal RoPE
        // paths unchanged, the same way YaRN does below.
        //
        // Before this, the type was compared against "yarn" and anything
        // else returned None, so a checkpoint declaring linear scaling
        // with factor 4 loaded and roped at UNSCALED positions where
        // llama.cpp divides them by 4. It answered as a different model
        // with no error. Affects the long-context community rescales
        // (`*-16k`, `*-32k` Llama-2 derivatives).

        // The file's own per-band factors, BEFORE any position-scaling
        // fold. That is what a sliding layer uses on an architecture
        // whose SWA layers do not inherit the trained scale -- llama.cpp
        // keeps the two apart as `freq_factors` (a tensor, the same for
        // every layer) and `freq_scale` (per layer,
        // `llama-model.cpp:2033`), while frink folds them into one
        // vector. See `config::RopeFreqs`.
        let rope_freqs_unscaled = rope_freqs.clone();

        let rope_freqs = match linear_scaling_from_gguf(file, &arch) {
            None => rope_freqs,
            Some(factor) => {
                let rotary_dim = rope_dim.unwrap_or(head_dim);
                if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
                    best_effort_fields.push(
                        "rope_freqs (linear scaling declared but the rotary width is odd; \
                         scaling not applied)",
                    );
                    rope_freqs
                } else {
                    let linear = vec![factor; rotary_dim / 2];
                    match rope_freqs {
                        None => Some(linear),
                        // Compose by multiplication, as a file carrying
                        // its own `rope_freqs.weight` tensor and a
                        // declared linear factor means both.
                        Some(own) if own.len() == linear.len() => {
                            Some(own.iter().zip(linear.iter()).map(|(a, b)| a * b).collect())
                        }
                        Some(own) => {
                            best_effort_fields.push(
                                "rope_freqs (linear scaling declared but the file's own \
                                 rope_freqs tensor has a different width; scaling not applied)",
                            );
                            Some(own)
                        }
                    }
                }
            }
        };
        let rope_freqs = match yarn_scaling_from_gguf(file, &arch, rope_orig_ctx) {
            None => rope_freqs,
            Some(scaling) => {
                // YaRN's MAGNITUDE half (`crate::yarn_magnitude`):
                // llama.cpp multiplies the rotated channels of q and k
                // by `get_mscale(factor, 1) / get_mscale(factor,
                // log_mul)` on top of `rope.scaling.attn_factor`
                // (`llama-context.cpp:196-231` with ggml's `rope_yarn`
                // term cancelled), and frink applied only the key.
                // Folded into the same field so it reaches the CPU
                // helper and the Metal `mscale` uniform through one
                // value. Gated on the same `Some(scaling)` as the
                // frequency half, so a file frink does not rewrite
                // (no `original_context_length`) takes neither half.
                rope_attn_factor *= crate::yarn_magnitude::yarn_attn_magnitude(
                    scaling.factor,
                    crate::yarn_magnitude::yarn_log_mul_for(
                        &arch,
                        metadata_f32_any(file, &[key("rope.scaling.yarn_log_multiplier")]),
                    ),
                );
                let rotary_dim = rope_dim.unwrap_or(head_dim);
                if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
                    best_effort_fields.push(
                        "rope_freqs (YaRN declared but the rotary width is odd; scaling not applied)",
                    );
                    rope_freqs
                } else {
                    let yarn =
                        frink_core::attention::yarn_freq_factors(scaling, rotary_dim, rope_theta);
                    match rope_freqs {
                        None => Some(yarn),
                        Some(own) if own.len() == yarn.len() => {
                            Some(own.iter().zip(yarn.iter()).map(|(a, b)| a * b).collect())
                        }
                        Some(own) => {
                            best_effort_fields.push(
                                "rope_freqs (YaRN declared alongside a per-band factor tensor of a \
                                 different width; the file's own tensor is used unscaled)",
                            );
                            Some(own)
                        }
                    }
                }
            }
        };

        // The SWA half of the split. llama.cpp defaults
        // `rope_freq_scale_train_swa` to `1.0f`
        // (`src/llama-hparams.h:129`) and only the architectures in
        // `swa_rope_scale_follows_model` assign it from
        // `rope_freq_scale_train`; `get_rope_freq_scale`
        // (`llama-model.cpp:2033-2035`) then picks between them per
        // layer. `gemma3.cpp` is not on that list and its converter
        // writes the FULL-ATTENTION factor
        // (`conversion/base.py:1222-1230`), so a Gemma-3 4B/12B/27B was
        // rotating five layers in six at `p/8` where llama.cpp rotates
        // at `p`.
        //
        // "No scaling" is spelled as an all-ones divisor vector, which
        // is what dividing by nothing is, so the sliding layers need no
        // second code path anywhere downstream.
        // With TWO rotary widths, one divisor vector cannot serve both
        // kinds of layer. `step35.cpp:247` passes NO factors to its
        // sliding layers (`crate::swa_geometry::swa_layers_drop_rope_
        // factors`), so for it the full layers take the first
        // `rope_dim/2` bands of the tensor -- ggml reads only that many
        // -- and the sliding layers divide by nothing at their own
        // width; every other architecture is refused by name, because
        // nothing upstream says which layers would take which.
        if rope_freqs.is_some() {
            if let Some(reason) =
                crate::swa_geometry::two_widths_with_factors_refusal(&arch, widths)
            {
                return Err(LoadError::UnsupportedFeature(arch.clone(), reason));
            }
        }
        let rope_freqs = rope_freqs
            .map(|full| -> Result<crate::config::RopeFreqs, LoadError> {
                if let Some(swa_width) = rope_dim_swa {
                    let full_width = rope_dim.unwrap_or(head_dim);
                    if full.len() < full_width / 2 {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "rope_freqs.weight has {} bands; the full-attention layers rotate \
                                 {full_width} dims and need {}",
                                full.len(),
                                full_width / 2
                            ),
                        ));
                    }
                    let full: Vec<f32> = full[..full_width / 2].to_vec();
                    return Ok(crate::config::RopeFreqs {
                        full,
                        swa: Some(vec![1.0; swa_width / 2]),
                    });
                }
                // One width: the sliding layers take the file's own
                // factors unscaled, or NONE where the graph passes them
                // nothing (`lfm3.cpp:113`, the same table as above).
                let drops = facts.swa.layers_drop_rope_factors.is_some();
                let swa = (sliding_window.is_some()
                    && (drops || !facts.swa.rope_scale_follows_model))
                    .then(|| match rope_freqs_unscaled {
                        Some(own) if !drops => own,
                        _ => vec![1.0; full.len()],
                    })
                    .filter(|swa| *swa != full);
                Ok(crate::config::RopeFreqs { full, swa })
            })
            .transpose()?;

        // RoPE layout comes from the capability registry above (fail-
        // closed). Getting this wrong for `llama` (needs Norm) was the
        // real root cause of the Llama-3.1-8B early-stop/wrong-logits bug.

        if best_effort_fields.is_empty() {
            best_effort_fields.push(
                "none -- every field above was read directly from this file's own GGUF metadata",
            );
        }

        // LAST, deliberately. The generic path is a GUESS, so it has to
        // be opted into rather than fallen onto: it assumes plain GQA
        // with no ALiBi, no learned position embeddings and no
        // per-layer rope skipping, and that assumption was already
        // wrong for gpt2, mpt, refact, bloom and jais.
        //
        // But it runs AFTER every architecture-specific refusal, so a
        // checkpoint with a NAMED problem still reports that problem.
        // Checking first would have replaced "this uses ALiBi" with
        // "this is unaudited", which is true and much less useful.
        if matches!(
            arch_profile.path,
            crate::capability::ArchPath::GenericGqa { .. }
        ) && !facts.audited
            && !matches!(
                std::env::var("FRINK_ALLOW_UNAUDITED_ARCH").ok().as_deref(),
                Some("1") | Some("true") | Some("on")
            )
        {
            return Err(LoadError::UnauditedArchitecture(
                arch.clone(),
                rope_layout,
                crate::capability::unaudited_refusal_detail(&arch),
            ));
        }

        Ok(ModelConfig {
            kv_dtype: frink_core::kv_store::KvDtype::F32,
            name,
            n_layers,
            n_mtp_blocks: trunk.n_mtp_blocks,
            layer_loops,
            skip_stream: facts.topology.skip_stream,
            parallel_ssm: facts.topology.parallel_ssm,
            swa_chunked,
            weightless_qk_norm: crate::weightless_qk_norm::weightless_qk_norm(&arch, n_experts),
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            v_head_dim,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            // `crate::norm::POST_NORM_EPS_LITERAL`: the architecture's, which
            // is the model's for all but one graph of 155.
            post_norm_eps: crate::norm::post_norm_eps(&arch, rms_norm_eps),
            // No GGUF file encodes a hybrid KDA/Gated-MLA attention
            // topology today; every real checkpoint loaded this way
            // runs the standard Gqa path.
            attention: crate::config::AttentionKind::Gqa,
            sliding_window,
            swa_layers,
            // llama.cpp's per-layer `use_rope`. Fed the POST-gate window
            // answer (`sliding_window`, not the raw key), because
            // `exaone4` decides both off the same layer count and the
            // two must not be able to disagree.
            rope_layers: if rope_switched_off {
                crate::rope_layers::RopeLayers::Never
            } else if let Some(mask) = rope_pattern.clone() {
                // The FILE's answer, for the one architecture that
                // reads the key (`rope_layers::ROPE_PATTERN_READERS`).
                crate::rope_layers::RopeLayers::FileMask(mask)
            } else {
                (facts.rules.rope_layers)(
                    &arch,
                    n_layers,
                    sliding_window.is_some(),
                    n_dense_leading_layers,
                )
            },
            router_input: facts.moe.router_input,
            block_sub_norms: facts.norm.block_sub_norms,
            parallel_residual: crate::parallel_residual::model_has_parallel_layer(
                file, &arch, n_layers,
            ),
            learned_positions: facts.topology.learned_positions,
            attn_value_scale: crate::attn_value_scale::resolve_attn_value_scale(
                &arch,
                file.metadata_f32(&key("attention.value_scale")),
            ),
            alibi_max_bias: crate::alibi::max_alibi_bias(
                &arch,
                n_layers,
                file.metadata_f32(&key("attention.max_alibi_bias")),
            ),
            layer_shapes,
            moe: MoeLayerConfig {
                n_experts: n_experts.max(1),
                n_experts_active,
                n_shared_experts,
                hidden_dim,
                expert_ffn_dim,
                gating,
                norm_topk_prob,
                expert_group_count: metadata_u64_any(file, &[key("expert_group_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 1),
                expert_group_used_count: metadata_u64_any(file, &[key("expert_group_used_count")])
                    .map(|v| v as usize)
                    .filter(|&c| c > 0),
                expert_weights_scale,
                routed_weight_before_ffn: facts.moe.routed_weight_before_ffn,
            },
            n_dense_leading_layers,
            moe_interleave_step,
            norm_function,
            rope_freqs,
            rope_layout,
            qk_norm_style,
            attn_logit_softcap,
            final_logit_softcap,
            embedding_scale,
            residual_scale: multipliers.residual_scale,
            normed_residual_scale: multipliers.normed_residual_scale,
            clamp_kqv,
            attn_temperature,
            logit_multiplier: multipliers.logit_multiplier,
            attention_scale,
            rope_attn_factor,
            rope_dim,
            rope_dim_swa,
            rope_freqs_long,
            rope_freqs_short,
            rope_orig_ctx,
            rope_theta_swa,
            ffn_activation,
            best_effort_fields: Box::leak(best_effort_fields.into_boxed_slice()),
        })
    }
}
