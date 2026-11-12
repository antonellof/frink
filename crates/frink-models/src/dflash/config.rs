//! The hyperparameters `llama_model_dflash::load_arch_hparams` reads
//! (`dflash.cpp:7-125`), and the refusals for the graphs this file does
//! not build.

use frink_gguf::{GgufValue, TensorSource};

use super::DFLASH_ARCH;

/// Why a DFlash draft cannot be loaded or paired with a target.
#[derive(Debug, thiserror::Error)]
pub enum DFlashError {
    #[error("not a DFlash draft: general.architecture is {0:?}, not \"dflash\"")]
    NotDFlash(Option<String>),
    #[error("the DFlash draft is missing metadata key '{0}'")]
    MissingKey(String),
    #[error("the DFlash draft is missing tensor '{0}'")]
    MissingTensor(String),
    #[error(
        "this DFlash draft is {0}, which frink does not build yet (llama.cpp's dflash.cpp \
         runs a different graph for it); use a plain DFlash or DFlash2 draft"
    )]
    Unsupported(String),
    #[error("the DFlash draft does not fit its target: {0}")]
    Mismatch(String),
    #[error(transparent)]
    Load(#[from] crate::loader::LoadError),
    #[error(transparent)]
    Taps(#[from] crate::decoder::LayerTapError),
}

/// A DFlash draft's shape, as `dflash.cpp` reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct DFlashConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Leading dimensions of each head that rotate (NeoX layout).
    pub n_rot: usize,
    /// `dflash.block_size`: the trained block, anchor included.
    pub block_size: usize,
    /// `dflash.conv_kernel_size` / `conv_group_size`, both zero for a
    /// DFlash1 draft.
    pub conv_kernel: usize,
    pub conv_group: usize,
    /// `dflash.selector_rank` / `selector_top_k`, zero without a
    /// selector (DFlash1).
    pub selector_rank: usize,
    pub selector_top_k: usize,
    /// Target layers whose INPUT the draft reads, ascending
    /// (`dflash.target_layers`, already the HF ids plus one).
    pub target_layers: Vec<usize>,
    /// `attention.sliding_window` and the per-layer pattern, when set.
    pub sliding_window: Option<usize>,
    pub swa_layers: Vec<bool>,
    /// `dflash.attention.causal`; false unless the file says otherwise
    /// (`common_speculative_impl_draft_dflash`, `causal_attn`).
    pub causal: bool,
    pub mask_token: usize,
    /// `1/sqrt(head_dim)` unless `attention.scale` sets it.
    pub kq_scale: f32,
    pub embedding_scale: Option<f32>,
    pub value_scale: Option<f32>,
    pub logit_scale: Option<f32>,
    pub final_logit_softcap: Option<f32>,
}

fn key(name: &str) -> String {
    format!("{DFLASH_ARCH}.{name}")
}

fn req_u64(file: &impl TensorSource, k: &str) -> Result<usize, DFlashError> {
    file.metadata_u64(k)
        .map(|v| v as usize)
        .ok_or_else(|| DFlashError::MissingKey(k.to_string()))
}

fn opt_u64(file: &impl TensorSource, name: &str) -> Option<usize> {
    file.metadata_u64(&key(name)).map(|v| v as usize)
}

fn opt_f32(file: &impl TensorSource, name: &str) -> Option<f32> {
    file.metadata_f32(&key(name))
}

fn u64_array(file: &impl TensorSource, k: &str) -> Option<Vec<usize>> {
    match file.metadata(k)? {
        GgufValue::Array(items) => items
            .iter()
            .map(|v| v.as_u64().map(|x| x as usize))
            .collect(),
        _ => None,
    }
}

impl DFlashConfig {
    /// Reads the draft's hyperparameters, refusing every variant whose
    /// graph `dflash.cpp` builds differently from the one here.
    pub fn from_gguf(file: &impl TensorSource) -> Result<Self, DFlashError> {
        let arch = file.metadata_str("general.architecture");
        if arch != Some(DFLASH_ARCH) {
            return Err(DFlashError::NotDFlash(arch.map(str::to_string)));
        }

        // dflash.cpp:66-111: a hyper-connected DeepSeek-V4 backbone.
        if opt_u64(file, "hyper_connection.count").unwrap_or(0) > 0 {
            return Err(DFlashError::Unsupported(
                "a DSpark draft with a DeepSeek-V4 backbone".into(),
            ));
        }
        // dflash.cpp:36-48: a LiLiCorr head.
        if file.metadata("lilicorr.hidden_size").is_some() {
            return Err(DFlashError::Unsupported("a LiLiCorr draft".into()));
        }
        // dflash.cpp:15-21: GELU is the only other activation.
        if let Some(act) = file.metadata_str(&key("hidden_activation")) {
            if act != "silu" {
                return Err(DFlashError::Unsupported(format!(
                    "a draft with the {act} activation"
                )));
            }
        }
        // dflash.cpp:29 and llama-model.cpp:3670: sections make it M-RoPE.
        if u64_array(file, &key("rope.dimension_sections"))
            .is_some_and(|s| s.iter().any(|&v| v != 0))
        {
            return Err(DFlashError::Unsupported(
                "a draft for an M-RoPE target (rope sections)".into(),
            ));
        }
        if let Some(kind) = file.metadata_str(&key("rope.scaling.type")) {
            if kind != "none" {
                return Err(DFlashError::Unsupported(format!(
                    "a draft with {kind} rope scaling"
                )));
            }
        }

        let n_embd = req_u64(file, &key("embedding_length"))?;
        let n_layer = req_u64(file, &key("block_count"))?;
        let n_head = req_u64(file, &key("attention.head_count"))?;
        let n_head_kv = opt_u64(file, "attention.head_count_kv").unwrap_or(n_head);
        let head_dim = opt_u64(file, "attention.key_length").unwrap_or(n_embd / n_head.max(1));
        let v_dim = opt_u64(file, "attention.value_length").unwrap_or(head_dim);
        if v_dim != head_dim || n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
            return Err(DFlashError::Unsupported(format!(
                "a draft with {n_head} query heads, {n_head_kv} KV heads, key width {head_dim} and \
                 value width {v_dim}"
            )));
        }
        let n_ff = req_u64(file, &key("feed_forward_length"))?;
        let rms_eps = file
            .metadata_f32(&key("attention.layer_norm_rms_epsilon"))
            .ok_or_else(|| DFlashError::MissingKey(key("attention.layer_norm_rms_epsilon")))?;
        let rope_theta = opt_f32(file, "rope.freq_base").unwrap_or(10_000.0);
        let n_rot = opt_u64(file, "rope.dimension_count").unwrap_or(head_dim);
        if n_rot > head_dim || !n_rot.is_multiple_of(2) {
            return Err(DFlashError::Unsupported(format!(
                "a draft rotating {n_rot} of {head_dim} head dimensions"
            )));
        }

        // `common_speculative_impl_draft_dflash`: 16 when the key is absent.
        let block_size = opt_u64(file, "block_size").unwrap_or(16);
        let conv_kernel = opt_u64(file, "conv_kernel_size").unwrap_or(0);
        let conv_group = opt_u64(file, "conv_group_size").unwrap_or(0);
        let selector_rank = opt_u64(file, "selector_rank").unwrap_or(0);
        let selector_top_k = opt_u64(file, "selector_top_k").unwrap_or(0);
        if block_size < 2 {
            return Err(DFlashError::Unsupported(format!(
                "a draft with block size {block_size}"
            )));
        }
        let has_conv = conv_kernel > 0 && conv_group > 0;
        // dflash.cpp:160-161.
        if has_conv && n_embd % conv_group != 0 {
            return Err(DFlashError::Unsupported(format!(
                "a draft whose hidden size {n_embd} is not a multiple of its conv group {conv_group}"
            )));
        }

        let target_layers = u64_array(file, &key("target_layers"))
            .ok_or_else(|| DFlashError::MissingKey(key("target_layers")))?;
        if target_layers.is_empty() {
            return Err(DFlashError::MissingKey(key("target_layers")));
        }

        // dflash.cpp:117-121: a window needs its per-layer pattern.
        let sliding_window = opt_u64(file, "attention.sliding_window").filter(|&w| w > 0);
        let swa_layers = match sliding_window {
            None => vec![false; n_layer],
            Some(_) => match file.metadata(&key("attention.sliding_window_pattern")) {
                Some(GgufValue::Array(items)) if items.len() == n_layer => {
                    items.iter().map(|v| v.as_bool().unwrap_or(false)).collect()
                }
                _ => {
                    return Err(DFlashError::MissingKey(key(
                        "attention.sliding_window_pattern",
                    )))
                }
            },
        };

        let causal = file
            .metadata_bool(&key("attention.causal"))
            .unwrap_or(false);
        let mask_token = file
            .metadata_u64("tokenizer.ggml.mask_token_id")
            .ok_or_else(|| DFlashError::MissingKey("tokenizer.ggml.mask_token_id".into()))?
            as usize;

        let nonzero = |v: Option<f32>| v.filter(|&x| x != 0.0);
        Ok(Self {
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            rms_eps,
            rope_theta,
            n_rot,
            block_size,
            conv_kernel: if has_conv { conv_kernel } else { 0 },
            conv_group: if has_conv { conv_group } else { 0 },
            selector_rank,
            selector_top_k,
            target_layers,
            sliding_window,
            swa_layers,
            causal,
            mask_token,
            kq_scale: nonzero(opt_f32(file, "attention.scale"))
                .unwrap_or(1.0 / (head_dim as f32).sqrt()),
            embedding_scale: nonzero(opt_f32(file, "embedding_scale")),
            value_scale: nonzero(opt_f32(file, "attention.value_scale")),
            logit_scale: nonzero(opt_f32(file, "logit_scale")),
            final_logit_softcap: opt_f32(file, "final_logit_softcapping").filter(|&x| x > 0.0),
        })
    }

    /// DFlash2: the selector walks the block (`dflash.cpp:1111`).
    pub fn has_selector(&self) -> bool {
        self.selector_rank > 0 && self.selector_top_k > 0
    }

    /// Most tokens one block can draft: the anchor fills one of the
    /// trained block's slots (`speculative.cpp:1099`).
    pub fn max_draft(&self) -> usize {
        self.block_size - 1
    }

    /// Values per position the draft reads from the target.
    pub fn target_features(&self, target_hidden: usize) -> usize {
        self.target_layers.len() * target_hidden
    }
}
