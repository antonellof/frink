//! Building a decoder without a checkpoint
//! (`Decoder::new_random_small`) and probing the kernels a loaded one
//! will dispatch.

use std::sync::atomic::AtomicU64;

use frink_core::tensor::Tensor;
use frink_core::weight_matrix::WeightMatrix;
use frink_moe::ExpertWeights;

use crate::config::ModelConfig;
use crate::norm::NormOp;

use super::config_alibi_slopes;
use super::{AttnWeights, Decoder, ExpertBacking, KvWindowPolicy, LayerWeights, MoeWeights};

/// Simple deterministic pseudo-random generator so tests are
/// reproducible without pulling in an external `rand` dependency.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * 0.1).collect()
    }
}

impl Decoder {
    /// Eagerly resolve every kernel lookup this model's dispatch paths
    /// will make, and record it in
    /// [`frink_core::kernel_registry`] before anything runs.
    ///
    /// Call once, at the end of loading, immediately before
    /// [`frink_core::kernel_registry::seal`]. Nothing here dispatches
    /// or decides anything: it asks the same predicates the hot path
    /// asks and writes the answers down, so a kernel that is missing
    /// becomes a startup line instead of an unexplained benchmark row.
    ///
    /// Routed experts held in an [`ExpertBacking::Stored`] layer are not
    /// probed -- they exist only as byte ranges until a token routes to
    /// them, and materialising every expert here would defeat the
    /// bounded expert store. Their kinds are the same as the resident
    /// case, and a dispatch-site miss still trips the sealed registry.
    pub fn probe_kernels(&self) {
        use frink_core::kernel_registry as reg;

        if !reg::enabled() {
            return;
        }
        self.embedding.probe_kernels("token_embd");
        self.output_head.probe_kernels("output_head");
        for layer in &self.layers {
            layer.attn.q_proj.probe_kernels("attn_q");
            layer.attn.k_proj.probe_kernels("attn_k");
            layer.attn.v_proj.probe_kernels("attn_v");
            layer.attn.o_proj.probe_kernels("attn_o");
            layer.moe.router.probe_kernels("moe_router");
            for e in &layer.moe.shared_experts {
                e.gate.probe_kernels("shexp_gate");
                e.up.probe_kernels("shexp_up");
                e.down.probe_kernels("shexp_down");
            }
            if let ExpertBacking::Resident(experts) = &layer.moe.experts {
                for e in experts {
                    e.gate.probe_kernels("ffn_gate");
                    e.up.probe_kernels("ffn_up");
                    e.down.probe_kernels("ffn_down");
                }
            }
        }
        // The generic decoder has a real batched prefill
        // (`forward_hidden_batch`), so a `pp512` here is one GEMM per
        // projection, not 512 matvecs. Recorded as a hit so that an
        // engine which lacks it stands out as a miss rather than as an
        // absence.
        reg::record_build(
            reg::Lookup::new(
                frink_core::weight_matrix::active_backend(),
                reg::op::ENGINE_PREFILL_BATCH,
                None,
            )
            .with_role("generic_decoder"),
            reg::Outcome::Hit,
        );
    }

    /// Builds a decoder with correctly-shaped, randomly initialized
    /// weights for `config`, but overrides `n_layers` and `vocab_size`
    /// with small test-scale numbers so it can actually be allocated and
    /// run inside a CI sandbox. Use this to validate the forward-pass
    /// plumbing only, never to draw conclusions about real model
    /// quality.
    pub fn new_random_small(config: ModelConfig, n_layers: usize, vocab_size: usize) -> Self {
        let mut rng = Lcg::new(42);
        let mut config = config;
        config.n_layers = n_layers;
        config.vocab_size = vocab_size;
        let hidden = config.hidden_dim;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;

        let embedding = WeightMatrix::F32(Tensor::new(
            rng.vec(vocab_size * hidden),
            vec![vocab_size, hidden],
        ));

        let wm = |data: Vec<f32>, shape: Vec<usize>| WeightMatrix::F32(Tensor::new(data, shape));

        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let attn = AttnWeights {
                q_proj: wm(
                    rng.vec(n_heads * head_dim * hidden),
                    vec![n_heads * head_dim, hidden],
                ),
                k_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                v_proj: wm(
                    rng.vec(n_kv_heads * head_dim * hidden),
                    vec![n_kv_heads * head_dim, hidden],
                ),
                o_proj: wm(
                    rng.vec(hidden * n_heads * head_dim),
                    vec![hidden, n_heads * head_dim],
                ),
                norm_weight: NormOp::Rms(vec![1.0; hidden]),
                q_norm: None,
                k_norm: None,
                q_bias: None,
                k_bias: None,
                v_bias: None,
                post_attn_norm: None,
                post_ffn_norm: None,
                output_gate: None,
                sinks: None,
                attn_sub_norm: None,
                o_scale: None,
                o_bias: None,
                shortconv: None,
                ssm: None,
                q_gate_interleaved: false,
            };

            // Leading dense layers (see ModelConfig::layer_is_dense's
            // doc comment) get a single-expert, no-shared-expert
            // dense-equivalent FFN regardless of this model's global
            // MoE topology, matching the DeepSeek-2/3-family
            // convention found in ik_llama.cpp's source.
            let is_dense_layer = config.layer_is_dense(layer_idx);
            let n_experts = if is_dense_layer {
                1
            } else {
                config.moe.n_experts
            };
            let n_shared = if is_dense_layer {
                0
            } else {
                config.moe.n_shared_experts
            };
            let ffn_dim = config.moe.expert_ffn_dim;
            let make_expert = |rng: &mut Lcg| ExpertWeights {
                gate: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                up: WeightMatrix::F32(Tensor::new(
                    rng.vec(ffn_dim * hidden),
                    vec![ffn_dim, hidden],
                )),
                down: WeightMatrix::F32(Tensor::new(
                    rng.vec(hidden * ffn_dim),
                    vec![hidden, ffn_dim],
                )),
            };
            let experts: Vec<ExpertWeights> =
                (0..n_experts).map(|_| make_expert(&mut rng)).collect();
            let shared_experts = (0..n_shared).map(|_| make_expert(&mut rng)).collect();
            let activation_counts = (0..experts.len()).map(|_| AtomicU64::new(0)).collect();

            let moe = MoeWeights {
                exp_probs_bias: None,
                ffn_sub_norm: None,
                down_scale: None,
                exps_norm: None,
                parallel_sum_scale: None,
                parallel: None,
                dense_bias: None,
                router: wm(rng.vec(n_experts * hidden), vec![n_experts, hidden]),
                experts: ExpertBacking::Resident(experts),
                shared_experts,
                shared_expert_gate: None,
                norm_weight: NormOp::Rms(vec![1.0; hidden]),
                activation_counts,
                #[cfg(feature = "metal")]
                packed_q4: None,
            };

            layers.push(LayerWeights {
                attn,
                moe,
                out_scale: None,
            });
        }

        let final_norm = NormOp::Rms(vec![1.0; hidden]);
        let output_head = wm(rng.vec(vocab_size * hidden), vec![vocab_size, hidden]);
        let execution_plan = crate::execution_plan::ExecutionPlan::from_config(
            &config,
            crate::capability::DecoderFamily::StandardGqa,
            crate::capability::MemoryKind::KvGqa,
            crate::execution_plan::ExecutionPlan::probe_metal_caps(),
        );

        let alibi_slopes = config_alibi_slopes(&config);
        Decoder {
            config,
            embedding,
            position_embd: None,
            embedding_norm: NormOp::None,
            hrm_z_l_init: None,
            layers,
            final_norm,
            output_head,
            output_bias: None,
            alibi_slopes,
            gpu_vram_budget_bytes: None,
            // Synthetic-weights constructor: no checkpoint, no gpt-oss.
            gpt_oss: None,
            // Synthetic-weights constructor: the preset families it
            // serves all norm before RoPE.
            qk_norm_after_rope: false,
            #[cfg(feature = "metal")]
            metal_attn_kv: std::sync::Mutex::new(None),
            execution_plan,
            kv_window: KvWindowPolicy::from_env(),
            plan_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            lora_adapters: Vec::new(),
        }
    }
}
