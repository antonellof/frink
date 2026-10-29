//! `Decoder::from_gguf` and `from_gguf_with_expert_cache`: every
//! layer's tensors loaded against a resolved `ModelConfig`.
//!
//! `from_gguf_with_expert_cache` is one function of some 800 lines,
//! kept whole: its per-layer loop shares state across the attention,
//! norm and FFN halves, and splitting it would be a rewrite rather than
//! a move.

use frink_core::expert_store::ExpertKey;
use frink_core::tensor::Tensor;
use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::{GgufError, ShardedGguf};
use frink_moe::ExpertWeights;

use crate::config::ModelConfig;
use crate::decoder::{AttnWeights, Decoder, ExpertBacking, LayerWeights, MoeWeights};
use crate::norm::NormOp;

use super::expert_source::{stored_expert_specs, GgufExpertSource, StoredExpertLayout};
use super::layer_parts::{load_attn_sinks, load_dense_expert, load_gpt_oss_layer};
use super::model_config::metadata_u64_any;
#[cfg(feature = "metal")]
use super::moe_planes::try_build_moe_packed_q4_planes;
use super::tensor_audit::assert_every_tensor_consumed;
use super::tensors::{
    load_f32_vec, load_f32_vec_optional, load_weight_matrix, split_expert_tensor,
};
use super::LoadError;

impl Decoder {
    /// Loads real weights from `path` for the given `config`. `config`
    /// supplies the architecture shape (layer count, head counts, MoE
    /// topology); tensor names are resolved against it using the
    /// llama.cpp naming convention described in the module docs.
    ///
    /// A `config.moe.n_experts <= 1` model is treated as dense: expert
    /// weights are read from the plain `blk.N.ffn_{gate,up,down}.weight`
    /// tensor names rather than the packed 3D `_exps` variant.
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        config: ModelConfig,
    ) -> Result<Self, LoadError> {
        Self::from_gguf_with_expert_cache(path, config, None)
    }

    /// Like `from_gguf`, but with `expert_cache_bytes: Some(budget)`
    /// routed experts are NOT loaded resident: each layer holds only
    /// byte-range layouts, and expert bytes are read on demand through
    /// one bounded, lease-protected `ExpertStore` shared by every
    /// layer (a single global byte budget; see
    /// `frink_core::expert_store`). Dense layers, shared experts,
    /// attention, embeddings, and the output head stay resident/mapped
    /// exactly as before -- only routed experts stream. Layers whose
    /// expert tensors are F32/BF16 fall back to resident loading (the
    /// store exists for the quantized case). Output is bit-identical
    /// to the resident path -- same bytes, same kernels -- pinned by
    /// the roundtrip suite's equivalence test.
    pub fn from_gguf_with_expert_cache(
        path: impl AsRef<std::path::Path>,
        mut config: ModelConfig,
        expert_cache_bytes: Option<u64>,
    ) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let file = ShardedGguf::open(path)?;

        // gpt-oss carries four per-layer tensors the generic GQA layer
        // structs have no home for (its fifth, the attention sinks, is
        // `AttnWeights::sinks` and loads by tensor presence). That is
        // decided by the architecture string, so resolve it once here.
        // See `crate::decoder::GptOssWeights`.
        //
        // This used to be ONE flag with the norm-slot fact below,
        // `arch == "gpt-oss"`, standing for two unrelated facts.
        // Splitting them is what let `seed_oss` -- which shares the norm
        // slot and has none of the extra tensors -- be admitted without
        // also being handed attention sinks.
        // Canonicalised: an architecture that llama.cpp computes
        // another one's graph for reads that one's tables, so an alias
        // needs one entry rather than a row in each of a dozen
        // per-architecture lists (`capability::canonical_architecture`).
        let arch = crate::capability::canonical_architecture(
            file.metadata_str("general.architecture")
                .unwrap_or_default(),
        )
        .to_string();
        // The tensor-table rows, from the CANONICAL name above
        // (`crate::models::facts`); `family` / `memory` below still come
        // from the raw one.
        let facts = crate::models::ArchFacts::of(&arch);
        let is_gpt_oss = facts.tensors.gpt_oss_layers;
        // A `<projection>.scale` companion is a multiply llama.cpp
        // applies and frink does not; refused by name here, before
        // the unread-tensor gate can be talked past
        // (`crate::weight_scales`).
        crate::weight_scales::refuse_weight_scale_tensors(
            &arch,
            file.tensors().map(|(_, t)| t.name.as_str()),
        )?;
        // Which tensor each of the five norm sites is stored under and
        // which FUNCTION norms it, resolved ONCE. Four shapes reach this
        // loader -- the plain pre-norm layer, the post-norm-only
        // topology (`olmo2`, `exaone4`), the non-parametric LayerNorm
        // (`olmo`) and the weighted LayerNorm (`dbrx`) -- and three
        // architectures keep a norm under a name another architecture
        // uses for a different site (`gpt-oss` / `seed_oss`, `dbrx`,
        // `grok`). `crate::norm_sites` is the one table for all of it;
        // this loader used to restate the decision at every site.
        let norm_sites = crate::norm_sites::NormSites {
            function: config.norm_function,
            ..facts.tensors.norm_sites
        };
        let mut gpt_oss_layers: Vec<crate::decoder::GptOssLayer> = Vec::new();

        // One store for the whole model (keys are (layer, expert)),
        // built up-front with every stored expert's segments; created
        // only when the cache is enabled AND some layer can use it.
        let mut store_segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]> =
            std::collections::HashMap::new();
        let mut stored_layouts: Vec<Option<Vec<StoredExpertLayout>>> = Vec::new();

        // Loaded like any other weight matrix: a quantized embedding
        // table stays quantized (zero-copy mmap) and token lookup
        // dequantizes one row via `WeightMatrix::dequant_row`, instead
        // of the whole vocabulary tensor being widened to f32 up front.
        let embedding = load_weight_matrix(&file, "token_embd.weight")?;
        // The learned position table (`crate::position_embd`), one row
        // per trained position, for the graphs that add one.
        let position_embd = crate::position_embd::load_position_embd(
            &file,
            &arch,
            config.hidden_dim,
            metadata_u64_any(&file, &[format!("{arch}.context_length")]).map(|v| v as usize),
        )?;

        // PHYSICAL layers: the blocks the file holds tensors for. A
        // looped model (`crate::layer_loops`) has more logical layers
        // than this, and they run these same weights.
        let n_physical = config
            .layer_loops
            .map_or(config.n_layers, |loops| loops.physical_layers());
        let mut layers = Vec::with_capacity(n_physical);
        let mut refined_qk_norm = config.qk_norm_style;
        for l in 0..n_physical {
            // THIS layer's head counts and FFN width. Uniform for every
            // architecture but the per-layer ones (`crate::layer_shapes`),
            // and the loader reads the shape rather than the scalars so
            // that a deci / openelm layer is sized by its own header.
            let shape = config.layer_shape(l);
            // Whether THIS layer's FFN reads the layer input rather
            // than the post-attention residual, and under which norm
            // (`crate::parallel_residual`).
            let parallel = crate::parallel_residual::layer_parallel_norm(&file, &arch, l);
            // THIS layer's norm slots: the architecture's row, with
            // Falcon-40B's `attn_norm_2` crossing the two pre-norm names
            // on the layers that carry it (`crate::norm_sites`).
            let layer_sites = norm_sites.for_layer(&arch, &file, l);
            // BitNet's two inner norms, REQUIRED when the architecture
            // has them and untouched otherwise (`crate::sub_norms`).
            let sub_norms = crate::sub_norms::load_sub_norms(
                &file,
                &arch,
                config.block_sub_norms,
                l,
                config.hidden_dim,
                shape.ffn_dim,
            )?;
            let attn = match shape.attention {
                crate::layer_shapes::AttnShape::Gqa { n_heads, .. } => {
                    // Q/K/V and their biases come out of ONE decision about
                    // which spelling this layer uses -- see `qkv_fused`. They
                    // used to be resolved independently, and a checkpoint that
                    // fused both (ChatGLM, Qwen-1) had its bias dropped.
                    let crate::qkv_fused::QkvProjections {
                        q: q_proj,
                        k: k_proj,
                        v: v_proj,
                        q_bias,
                        k_bias,
                        v_bias,
                    } = crate::qkv_fused::load_fused_or_split_qkv(
                        &file,
                        l,
                        &config,
                        facts.tensors.qkv_tensors,
                    )?;
                    let q_norm =
                        load_f32_vec_optional(&file, &format!("blk.{l}.attn_q_norm.weight"))?;
                    let k_norm =
                        load_f32_vec_optional(&file, &format!("blk.{l}.attn_k_norm.weight"))?;
                    // The per-head LAYERNORM (`crate::qk_layer_norm`), whose
                    // weight is `n_heads * head_dim` long and would pass
                    // the length rule below as `WholeVector`.
                    if let Some(reason) = crate::qk_layer_norm::per_head_layer_norm_refusal_for(
                        facts.tensors.per_head_layer_norm_qk,
                        l,
                        q_norm.is_some() || k_norm.is_some(),
                    ) {
                        return Err(LoadError::UnsupportedFeature(
                            config.name.to_string(),
                            reason,
                        ));
                    }
                    // Refine WholeVector vs PerHead from the first observed norm length.
                    // The per-head SCALAR gain is decided by architecture first
                    // (`capability::PER_HEAD_SCALAR_QK_GAIN`): its length is
                    // `n_heads`, which a length test alone could confuse with
                    // `head_dim`.
                    if let Some(ref w) = q_norm {
                        if facts.tensors.per_head_scalar_qk_gain {
                            if w.len() != n_heads {
                                return Err(LoadError::UnsupportedFeature(
                                    config.name.to_string(),
                                    format!(
                                        "blk.{l}.attn_q_norm.weight length {} is not one gain per \
                                         head (n_heads={n_heads}; talkie.cpp:26 creates it {{1, \
                                         n_head}})",
                                        w.len()
                                    ),
                                ));
                            }
                            refined_qk_norm = crate::capability::QkNormStyle::PerHeadScalar;
                        } else if facts.tensors.per_head_distinct_qk_norm {
                            // `plamo2.cpp:92-93`: `{head_dim, n_head}`, one
                            // row per head, RMS per head. The same length as
                            // a whole-vector weight, so the architecture
                            // decides (`capability::PER_HEAD_DISTINCT_QK_NORM`).
                            if w.len() != n_heads * config.head_dim {
                                return Err(LoadError::UnsupportedFeature(
                                    config.name.to_string(),
                                    format!(
                                        "blk.{l}.attn_q_norm.weight length {} is not one row per \
                                         head (n_heads={n_heads} x head_dim={}; plamo2.cpp:92 \
                                         creates it {{head_dim, n_head}})",
                                        w.len(),
                                        config.head_dim
                                    ),
                                ));
                            }
                            refined_qk_norm = crate::capability::QkNormStyle::PerHeadDistinct;
                        } else if w.len() == config.head_dim {
                            refined_qk_norm = crate::capability::QkNormStyle::PerHead;
                        } else if w.len() == n_heads * config.head_dim {
                            refined_qk_norm = crate::capability::QkNormStyle::WholeVector;
                        } else {
                            return Err(LoadError::UnsupportedFeature(
                                config.name.to_string(),
                                format!(
                                    "blk.{l}.attn_q_norm.weight length {} matches neither \
                                     head_dim={} nor n_heads*head_dim={}",
                                    w.len(),
                                    config.head_dim,
                                    n_heads * config.head_dim
                                ),
                            ));
                        }
                    }
                    let attn = AttnWeights {
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj: load_weight_matrix(&file, &format!("blk.{l}.attn_output.weight"))?,
                        // Which tensor, which function, and whether there is a
                        // norm here at all: all three answered by the table.
                        norm_weight: layer_sites.load_pre_norm(layer_sites.attn, &file, Some(l))?,
                        q_norm,
                        k_norm,
                        // Qwen2/Qwen2-MoE-family real QKV bias (`attn_{q,k,v}.bias`,
                        // real config `qkv_bias`, `o_proj` has none) -- see
                        // `AttnWeights::q_bias`'s doc comment. Resolved above,
                        // alongside the projections they belong to, because a
                        // file that fuses the weight fuses the bias too.
                        q_bias,
                        k_bias,
                        v_bias,
                        post_attn_norm: crate::norm_sites::NormSites::load_post_norm(
                            norm_sites.post_attn,
                            &file,
                            l,
                        )?,
                        post_ffn_norm: crate::norm_sites::NormSites::load_post_norm(
                            norm_sites.post_ffn,
                            &file,
                            l,
                        )?,
                        // The architecture table decides whether there is
                        // a gate and how it is applied; the tensor decides
                        // its width. See `crate::attn_gate`.
                        output_gate: crate::attn_gate::AttnGate::load(
                            &file,
                            &arch,
                            l,
                            n_heads,
                            config.head_dim,
                            config.hidden_dim,
                        )?,
                        // The TENSOR decides. Four llama.cpp graphs pass it
                        // into the one `build_attn_mha`; on the generic
                        // path a file that has it gets the sink term and
                        // a file that does not gets none, whatever the
                        // architecture string. gpt-oss's requirement is
                        // checked where its side table loads.
                        sinks: load_attn_sinks(&file, l, n_heads)?,
                        attn_sub_norm: sub_norms.as_ref().map(|n| n.attn.clone()),
                        o_scale: crate::weight_scales::load_projection_gain(
                            &file,
                            &arch,
                            l,
                            "attn_output",
                        )?,
                        o_bias: crate::proj_bias::load_attn_out_bias(
                            &file,
                            &arch,
                            facts.tensors.attn_out_bias,
                            l,
                            config.hidden_dim,
                        )?,
                        shortconv: None,
                        // falcon-h1.cpp:55-71: the Mamba-2 block beside
                        // attention on every layer (`crate::mamba2::
                        // PARALLEL_WITH_ATTENTION`).
                        ssm: if config.parallel_ssm {
                            Some(crate::ssm_block::SsmBlock::Mamba2(
                                crate::mamba2::Mamba2::load(&file, &arch, l, config.hidden_dim)?,
                            ))
                        } else {
                            None
                        },
                        q_gate_interleaved: facts.tensors.q_gate_interleaved,
                    };
                    crate::layer_shapes::check_gqa_projection_widths(
                        l,
                        shape.attention,
                        config.head_dim,
                        config.v_head_dim(),
                        config.hidden_dim,
                        &attn,
                    )?;
                    attn
                }
                other => crate::layer_shapes::load_non_gqa_attention(
                    other,
                    &file,
                    &arch,
                    l,
                    &norm_sites,
                    &config,
                )?,
            };
            if let Some(reason) = crate::norm_sites::post_norm_pairing_refusal_for(
                &arch,
                facts.tensors.post_norms_both_or_neither,
                l,
                attn.post_attn_norm.is_some(),
                attn.post_ffn_norm.is_some(),
            ) {
                return Err(LoadError::UnsupportedFeature(
                    config.name.to_string(),
                    reason,
                ));
            }

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) load from the plain dense tensor names
            // regardless of this model's global MoE topology, matching
            // the DeepSeek-2/3-family convention found in
            // ik_llama.cpp's source. A model with n_experts<=1
            // globally (the dense test fixture) is dense on every
            // layer either way.
            // A layer with NO FFN at all (`ffn_dim 0`: deci's, Nemotron-H's
            // block-only layers) takes the dense arm, whose loader answers
            // `absent_ffn` for that width, whatever the model's MoE says.
            let is_dense_layer = config.layer_is_dense(l)
                || config.moe.n_experts <= 1
                || shape.ffn_dim == 0
                || crate::moe_interleave::dense_by_router_absence(&arch, &file, l);
            // The ungated experts (`nemotron-h.cpp:82-86,209-215`: a null
            // gate into `build_moe_ffn`, `LLM_FFN_RELU_SQR`) are spelled
            // the way the dense ungated FFN is (`load_dense_expert`): the
            // gate ALIASED to `up`, so `relu(up)^2` runs through the gated
            // body with no branch. A file that carries a gate anyway is
            // refused, as the dense loader refuses one.
            let routed_gate_name = if config.ffn_is_ungated() && !is_dense_layer {
                if file
                    .find_tensor(&format!("blk.{l}.ffn_gate_exps.weight"))
                    .is_some()
                {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.ffn_gate_exps.weight is present but this architecture's \
                             experts are ungated ({:?}: a null gate into build_moe_ffn, \
                             nemotron-h.cpp:212)",
                            config.ffn_activation
                        ),
                    ));
                }
                format!("blk.{l}.ffn_up_exps.weight")
            } else {
                format!("blk.{l}.ffn_gate_exps.weight")
            };
            // The inner FFN norm has a site in the dense body only
            // (`build_ffn` with a NULL down, `bitnet.cpp:127-141`);
            // `build_moe_ffn` has none, so a routed layer that carried
            // one would have nowhere to apply it.
            if sub_norms.is_some() && !is_dense_layer {
                return Err(LoadError::UnsupportedFeature(
                    arch.clone(),
                    format!(
                        "blk.{l}.ffn_sub_norm on a MoE layer: llama.cpp applies the inner FFN \
                         norm in the dense `build_ffn` body only (bitnet.cpp:127-141), and no \
                         routed-expert graph has that site"
                    ),
                ));
            }
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let experts: ExpertBacking = if is_dense_layer {
                ExpertBacking::Resident(vec![load_dense_expert(&file, l, &config, shape.ffn_dim)?])
            } else {
                // Try store-backed layouts first when the cache is
                // enabled; fall back to resident when any of the three
                // tensors isn't a supported quantized dtype.
                let stored = if expert_cache_bytes.is_some() {
                    let g = stored_expert_specs(&file, &routed_gate_name, n_experts)?;
                    let u = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        n_experts,
                    )?;
                    let d = stored_expert_specs(
                        &file,
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        n_experts,
                    )?;
                    match (g, u, d) {
                        (Some(gt), Some(ut), Some(dt)) => {
                            let mut layouts = Vec::with_capacity(n_experts);
                            for e in 0..n_experts {
                                let key = ExpertKey {
                                    layer: l as u32,
                                    expert: e as u32,
                                };
                                store_segments.insert(
                                    key,
                                    [
                                        (gt.shard, gt.per_expert[e].0, gt.per_expert[e].1),
                                        (ut.shard, ut.per_expert[e].0, ut.per_expert[e].1),
                                        (dt.shard, dt.per_expert[e].0, dt.per_expert[e].1),
                                    ],
                                );
                                let mut gate = gt.spec;
                                let mut up = ut.spec;
                                let mut down = dt.spec;
                                gate.offset = 0;
                                up.offset = gate.len;
                                down.offset = gate.len + up.len;
                                layouts.push(StoredExpertLayout { gate, up, down });
                            }
                            Some(layouts)
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                match stored {
                    Some(layouts) => {
                        // Placeholder; the shared store is attached in a
                        // second pass below once every layer's segments
                        // are collected.
                        stored_layouts.push(Some(layouts));
                        ExpertBacking::Resident(Vec::new())
                    }
                    None => {
                        let gates = split_expert_tensor(&file, &routed_gate_name, n_experts)?;
                        let ups = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_up_exps.weight"),
                            n_experts,
                        )?;
                        let downs = split_expert_tensor(
                            &file,
                            &format!("blk.{l}.ffn_down_exps.weight"),
                            n_experts,
                        )?;
                        ExpertBacking::Resident(
                            gates
                                .into_iter()
                                .zip(ups)
                                .zip(downs)
                                .map(|((gate, up), down)| ExpertWeights { gate, up, down })
                                .collect(),
                        )
                    }
                }
            };
            if stored_layouts.len() < layers.len() + 1 {
                stored_layouts.push(None);
            }

            let mut shared_experts: Vec<ExpertWeights> =
                if config.moe.n_shared_experts > 0 && !is_dense_layer {
                    // The shared expert takes the architecture's dense
                    // activation, so an ungated one aliases its gate as
                    // `load_dense_expert` does (`nemotron-h.cpp:222-227`).
                    let shexp_gate = if config.ffn_is_ungated() {
                        if file
                            .find_tensor(&format!("blk.{l}.ffn_gate_shexp.weight"))
                            .is_some()
                        {
                            return Err(LoadError::UnsupportedFeature(
                                arch.clone(),
                                format!(
                                    "blk.{l}.ffn_gate_shexp.weight is present but this \
                                     architecture's shared expert is ungated"
                                ),
                            ));
                        }
                        format!("blk.{l}.ffn_up_shexp.weight")
                    } else {
                        format!("blk.{l}.ffn_gate_shexp.weight")
                    };
                    vec![ExpertWeights {
                        gate: load_weight_matrix(&file, &shexp_gate)?,
                        up: load_weight_matrix(&file, &format!("blk.{l}.ffn_up_shexp.weight"))?,
                        down: load_weight_matrix(&file, &format!("blk.{l}.ffn_down_shexp.weight"))?,
                    }]
                } else {
                    Vec::new()
                };
            // A dense FFN SUMMED with the experts (Grok-2, Arctic) is the
            // shared-expert slot under the dense names, plus the row's
            // scale on the sum (`crate::parallel_dense_ffn`). Decided per
            // layer: Grok-1's layers have no triple and take neither.
            // ...and the same scale under the `_shexp` names
            // (`SHARED_EXPERT_SUM_SCALE`, cohere2moe's `* 0.5`), on a
            // layer that loaded a shared expert above.
            let parallel_sum_scale = if is_dense_layer {
                None
            } else {
                match crate::parallel_dense_ffn::parallel_dense_for_layer(&arch, &file, l)? {
                    Some(row) => {
                        shared_experts.push(ExpertWeights {
                            gate: load_weight_matrix(&file, &format!("blk.{l}.ffn_gate.weight"))?,
                            up: load_weight_matrix(&file, &format!("blk.{l}.ffn_up.weight"))?,
                            down: load_weight_matrix(&file, &format!("blk.{l}.ffn_down.weight"))?,
                        });
                        row.sum_scale
                    }
                    None => crate::parallel_dense_ffn::sum_scale_on_layer(
                        facts.tensors.shared_expert_sum_scale,
                        !shared_experts.is_empty(),
                    ),
                }
            };
            // Arctic's second per-layer norm, the routed branch's operand
            // (`crate::router_input::RouterInput::NormedLayerInput`):
            // REQUIRED on its routed layers, unread everywhere else.
            let exps_norm = if config.router_input.needs_exps_norm() && !is_dense_layer {
                Some(load_f32_vec(
                    &file,
                    &format!("blk.{l}.ffn_norm_exps.weight"),
                )?)
            } else {
                None
            };

            let router = if !is_dense_layer {
                load_weight_matrix(&file, &format!("blk.{l}.ffn_gate_inp.weight"))?
            } else {
                // dense layer: no real router; a zero [1, hidden] matrix
                // always selects the single expert deterministically.
                WeightMatrix::F32(Tensor::zeros(vec![1, config.hidden_dim]))
            };

            let n_for_counts = match &experts {
                ExpertBacking::Resident(v) if v.is_empty() => n_experts,
                other => other.n_experts(),
            };
            let activation_counts = (0..n_for_counts)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect();
            // Qwen2-MoE-specific real tensor (`blk.N.ffn_gate_inp_shexp.weight`,
            // real on-disk shape `[hidden_dim]`, confirmed against
            // llama.cpp's real `qwen2moe.cpp`) -- see
            // `MoeWeights::shared_expert_gate`'s doc comment. Presence
            // of the tensor itself is the real signal (not an
            // architecture-name list): every other supported
            // architecture's checkpoints simply don't carry this
            // tensor, so this naturally stays `None` there.
            let shared_expert_gate = if is_dense_layer {
                None
            } else {
                load_f32_vec_optional(&file, &format!("blk.{l}.ffn_gate_inp_shexp.weight"))?
            };
            #[cfg(feature = "metal")]
            let packed_q4 = match &experts {
                ExpertBacking::Resident(v) if !v.is_empty() => try_build_moe_packed_q4_planes(v),
                _ => None,
            };
            // DeepSeek-V3's aux-loss-free selection bias. The on-disk
            // name carries no `ffn_` prefix -- llama.cpp's
            // `LLM_TENSOR_FFN_EXP_PROBS_B` maps to `blk.%d.exp_probs_b`
            // (`llama-arch.cpp:416`, `gguf-py/gguf/constants.py:1240`).
            // Optional: only the DeepSeek-V3-lineage MoE recipes carry
            // it, and this same generic loader serves OLMoE / Qwen2-MoE /
            // Mixtral, which do not.
            let exp_probs_bias = if is_dense_layer {
                None
            } else {
                load_f32_vec_optional(&file, &format!("blk.{l}.exp_probs_b.bias"))?
            };
            if let Some(bias) = &exp_probs_bias {
                if bias.len() != config.moe.n_experts {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.exp_probs_b.bias has {} entries but the model has {} experts",
                            bias.len(),
                            config.moe.n_experts
                        ),
                    ));
                }
                // Grouped selection masks the *biased* scores before the
                // global top-k (`build_moe_ffn`, the `n_expert_groups > 1`
                // block). frink's `route_top_k_grouped` takes a fixed
                // count from every group instead, which is a different
                // algorithm, so combining the two here would be a guess.
                // Refuse rather than route wrongly.
                if config.moe.expert_group_count.is_some() {
                    return Err(LoadError::UnsupportedFeature(
                        arch.clone(),
                        format!(
                            "blk.{l}.exp_probs_b.bias together with expert groups \
                             ({:?}): llama.cpp masks the biased scores per group \
                             before a global top-k, which is not the per-group \
                             top-k frink implements",
                            config.moe.expert_group_count
                        ),
                    ));
                }
            }
            let moe = MoeWeights {
                router,
                experts,
                shared_experts,
                shared_expert_gate,
                exp_probs_bias,
                exps_norm,
                parallel_sum_scale,
                // The dense FFN's biases (`crate::proj_bias`), on a dense
                // layer; a routed layer's experts carry none on the
                // generic path (gpt-oss's are its side table's).
                dense_bias: if is_dense_layer && shape.ffn_dim > 0 {
                    let bias = crate::proj_bias::load_dense_ffn_bias(
                        &file,
                        &arch,
                        facts.tensors.ffn_bias,
                        l,
                        config.hidden_dim,
                        shape.ffn_dim,
                        config.ffn_is_ungated(),
                    )?;
                    if bias.is_some() && sub_norms.is_some() {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "layer {l} has both an inner FFN norm and FFN biases; no llama.cpp \
                                 graph has both and the dense body has one arm for each"
                            ),
                        ));
                    }
                    bias
                } else {
                    None
                },
                ffn_sub_norm: sub_norms.map(|n| n.ffn),
                down_scale: {
                    let gain =
                        crate::weight_scales::load_projection_gain(&file, &arch, l, "ffn_down")?;
                    if gain.is_some() && !is_dense_layer {
                        return Err(LoadError::UnsupportedFeature(
                            arch.clone(),
                            format!(
                                "blk.{l}.ffn_down.scale on a MoE layer: the routed experts' \
                                 scales are `ffn_down_exps.scale`, one per expert, which is \
                                 not applied here"
                            ),
                        ));
                    }
                    gain
                },
                // The same table as the attention slot, so the two
                // pre-norms cannot disagree about the function, and the
                // pre-FFN tensor's NAME comes from the same row that
                // decided the post-attention slot must not read it. An
                // FFN-free layer (`deci.cpp:52-54`) has no such tensor.
                // A parallel layer with ONE shared norm has no pre-FFN
                // tensor and no pre-FFN norm: the FFN reads the vector
                // attention read (`crate::parallel_residual`).
                norm_weight: if shape.ffn_dim == 0
                    || parallel == Some(crate::parallel_residual::ParallelNorm::SharedNorm)
                {
                    NormOp::None
                } else {
                    layer_sites.load_pre_norm(layer_sites.ffn, &file, Some(l))?
                },
                parallel,
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4,
            };

            if is_gpt_oss {
                gpt_oss_layers.push(load_gpt_oss_layer(&file, l, &config, attn.sinks.is_some())?);
            }

            // Talkie's per-layer skip scalar (`crate::skip_stream`);
            // REQUIRED there, untouched everywhere else.
            let out_scale =
                crate::skip_stream::load_out_scale(&file, &arch, config.skip_stream, l)?;
            layers.push(LayerWeights {
                attn,
                moe,
                out_scale,
            });
        }

        // `olmo.cpp:15-36` creates no `output_norm` at all and
        // `:128-130` norms the final hidden state with a null weight, so
        // asking for the tensor would refuse every real OLMo-1 file;
        // the table's function decides whether the read happens.
        let final_norm = norm_sites.load_pre_norm(norm_sites.output, &file, None)?;
        // `hrm-text.cpp:46` creates `hrm_z_l_init` REQUIRED, and only
        // that graph does (`crate::hrm`): the learned LOW stream, one
        // `[n_embd]` row broadcast over the tokens at `:182`.
        let hrm_z_l_init = match config.layer_loops {
            Some(crate::layer_loops::LayerLoops::Hrm { .. }) => {
                Some(load_f32_vec(&file, "hrm.z_l_init")?)
            }
            _ => None,
        };
        // The embedding norm (`norm_sites::EMBEDDING_NORM_ARCHITECTURES`),
        // `NormOp::None` where the site is absent.
        // Two answers, one field: a STORED embedding norm (`bloom`) or
        // a weightless one (`muse-glimmer.cpp:69`), and the tables that
        // decide them are disjoint by construction
        // (`norm_sites::WEIGHTLESS_EMBEDDING_NORM`).
        let embedding_norm = if facts.tensors.weightless_embedding_norm {
            crate::norm::NormOp::RmsNoParams
        } else {
            norm_sites.load_pre_norm(norm_sites.embedding, &file, None)?
        };
        // Many small Llama/Gemma-family GGUFs tie the lm-head to
        // `token_embd.weight` and omit `output.weight` (llama.cpp
        // `llama_model_loader` falls back the same way). Prefer the
        // explicit head when present.
        let output_head = match load_weight_matrix(&file, "output.weight") {
            Ok(w) => w,
            Err(_) => load_weight_matrix(&file, "token_embd.weight")?,
        };
        // `output.bias` for the graphs that create it (`crate::proj_bias`).
        let output_bias = crate::proj_bias::load_output_bias(
            &file,
            &arch,
            facts.tensors.output_bias,
            output_head.rows(),
        )?;

        // Second pass: attach the one shared store to every
        // store-backed layer. Opening the shard files fresh (plain
        // `File` handles for positional reads, not mmaps) keeps the
        // stored experts' bytes out of the process's mapped footprint
        // entirely.
        if !store_segments.is_empty() {
            let budget = expert_cache_bytes
                .expect("store_segments only populated when a cache budget is set")
                as usize;
            let files: Result<Vec<std::fs::File>, std::io::Error> =
                file.shard_paths().iter().map(std::fs::File::open).collect();
            let files = files.map_err(GgufError::from)?;
            // The policy ranks by the counters these layers' routing
            // bumps, so it reads the same `Arc`s rather than a copy.
            let hotness = super::expert_store_build::LayerActivationCounts::new(
                layers
                    .iter()
                    .enumerate()
                    .map(|(l, layer)| {
                        stored_layouts
                            .get(l)
                            .is_some_and(Option::is_some)
                            .then(|| std::sync::Arc::clone(&layer.moe.activation_counts))
                    })
                    .collect(),
            );
            let all_layouts: Vec<&[StoredExpertLayout]> =
                stored_layouts.iter().flatten().map(Vec::as_slice).collect();
            let store = std::sync::Arc::new(super::expert_store_build::build_expert_store(
                GgufExpertSource {
                    files,
                    segments: store_segments,
                },
                &all_layouts,
                layers.len(),
                budget,
                hotness,
            ));
            for (l, layer) in layers.iter_mut().enumerate() {
                if let Some(layouts) = stored_layouts.get_mut(l).and_then(Option::take) {
                    layer.moe.experts = ExpertBacking::Stored {
                        store: std::sync::Arc::clone(&store),
                        layouts,
                        layer: l as u32,
                    };
                }
            }
        }

        config.qk_norm_style = refined_qk_norm;

        // The RAW name (an alias keeps its own catalog row), and "llama"
        // for a file with none, as before.
        let raw_facts = crate::models::ArchFacts::of(
            file.metadata_str("general.architecture").unwrap_or("llama"),
        );
        let family = raw_facts
            .family()
            .unwrap_or(crate::capability::DecoderFamily::StandardGqa);
        let memory_kind = raw_facts
            .profile
            .map(|p| p.memory)
            .unwrap_or(crate::capability::MemoryKind::KvGqa);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            family,
            memory_kind,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        let alibi_slopes = crate::decoder::config_alibi_slopes(&config);
        let decoder = Decoder {
            config,
            embedding,
            position_embd,
            embedding_norm,
            // `hrm-text.cpp:46` creates it REQUIRED, and only that
            // graph does (`crate::hrm`); the loader reads it for the
            // architecture whose schedule needs it and for no other.
            hrm_z_l_init,
            alibi_slopes,
            layers,
            final_norm,
            output_head,
            output_bias,
            gpu_vram_budget_bytes: None,
            gpt_oss: if is_gpt_oss {
                Some(crate::decoder::GptOssWeights {
                    layers: gpt_oss_layers,
                })
            } else {
                None
            },
            qk_norm_after_rope: facts.tensors.qk_norm_after_rope,
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            kv_window: crate::decoder::KvWindowPolicy::from_env(),
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            lora_adapters: Vec::new(),
        };
        // Resolve every kernel the model will need while we still have a
        // load-time error path to report it on, then seal: from here a
        // lookup that misses is an unpredicted slow path and says so.
        decoder.probe_kernels();
        frink_core::kernel_registry::seal_or_error()
            .map_err(|e| LoadError::StrictKernels(e.to_string()))?;
        // `ModelConfig` is parsed from a *different* handle on the same
        // file (the CLI opens its own `GgufFile`, then hands the config
        // here), so the model-level tensors it consumed were recorded on
        // that handle, not this one. Replay them before the gate, or
        // every Llama-3.x checkpoint reads as carrying an unread
        // `rope_freqs.weight` it in fact uses on every RoPE call.
        for name in crate::config::MODEL_LEVEL_TENSORS_READ_BY_CONFIG {
            file.note_consumed(name);
        }
        // The NextN/MTP blocks llama.cpp creates `TENSOR_SKIP` and never
        // runs (`crate::mtp_blocks`): deliberately unread, and said so,
        // rather than left for the gate below to report as a term the
        // graph is missing. The range is the config's, so the layer
        // loop above and this mark cannot disagree about where the
        // trunk ends.
        let skipped = crate::mtp_blocks::note_mtp_blocks_skipped(
            &file,
            &crate::mtp_blocks::TrunkLayers {
                block_count: n_physical + decoder.config.n_mtp_blocks,
                n_layers: n_physical,
                n_mtp_blocks: decoder.config.n_mtp_blocks,
            },
        );
        if skipped > 0 {
            eprintln!(
                "frink: skipping {} NextN/MTP block(s) after layer {} ({skipped} tensors), as \
                 llama.cpp does",
                decoder.config.n_mtp_blocks,
                n_physical - 1
            );
        }
        // Slots llama.cpp creates and never reads (`crate::unread_tensors`):
        // ignored as upstream ignores them, and said so.
        let ignored = crate::unread_tensors::note_unread_layer_tensors(&file, &arch, n_physical);
        if !ignored.is_empty() {
            eprintln!(
                "frink: ignoring {} tensor(s) llama.cpp creates and never reads for `{}` \
                 (first: {}), as llama.cpp does",
                ignored.len(),
                arch,
                ignored[0]
            );
        }
        assert_every_tensor_consumed(&file)?;
        Ok(decoder)
    }
}
