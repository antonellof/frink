//! BERT GGUF → [`BertEncoder`].
//!
//! Tensor names and requiredness follow llama.cpp
//! `llama_model_bert::load_arch_tensors` (`src/models/bert.cpp`), and
//! hparam keys follow its `load_arch_hparams` plus the shared
//! `LLM_KV_*` set.
//!
//! # `bert.cpp` upstream is five architectures; this is one of them
//!
//! `nomic-bert`, `nomic-bert-moe`, `jina-bert-v2`, `jina-bert-v3` and
//! `neo-bert` all build their graph from the same file, each switching
//! on `model.arch` for RoPE, a gated FFN, expert layers or a second
//! attention norm. [`crate::bert_encoder`] implements only the plain
//! `bert` shape, so every one of those differences is checked for here
//! and **refused by name**: a checkpoint that carries `ffn_gate` or
//! `attn_q_norm` is not silently run without it.
//!
//! The last line of defence is
//! [`crate::loader::assert_every_tensor_consumed`], which the load ends
//! with: for `bge-small-en-v1.5-q8_0.gguf` this graph reads all 197
//! tensors, so any weight a variant adds and this module has no home
//! for stops the load instead of being ignored.

use frink_gguf::{ShardedGguf, TensorSource};

use crate::bert_encoder::{BertEncoder, BertFfn, BertHparams, BertLayer, BertTopology};
use crate::loader::{
    assert_every_tensor_consumed, load_f32_vec, load_f32_vec_optional, load_weight_matrix,
    LoadError,
};
use crate::pooling::PoolingType;

/// The architecture string this loader implements.
pub const BERT_ARCH: &str = "bert";

/// llama.cpp's `tokenizer_model == "bert"` defaults, applied when the
/// GGUF carries no explicit id (`llama-vocab.cpp`).
const DEFAULT_CLS_ID: u32 = 101;
const DEFAULT_SEP_ID: u32 = 102;

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn refuse(what: &str) -> LoadError {
    LoadError::UnsupportedFeature(BERT_ARCH.to_string(), what.to_string())
}

/// Refuses if `name` exists, naming the upstream variant that carries it.
fn reject_tensor(file: &ShardedGguf, name: &str, why: &str) -> Result<(), LoadError> {
    if file.find_tensor(name).is_some() {
        return Err(refuse(&format!("checkpoint carries '{name}': {why}")));
    }
    Ok(())
}

/// The whole architecture policy of this loader, in one place so it can
/// be tested without a GGUF: `bert` and nothing else.
pub fn check_arch(arch: &str) -> Result<(), LoadError> {
    if ENCODER_ARCHS.iter().any(|(a, _)| *a == arch) {
        Ok(())
    } else {
        Err(LoadError::UnsupportedArchitecture(arch.to_string()))
    }
}

/// The architectures this loader builds, with the FFN each one runs.
///
/// Both share `bert.cpp`'s graph; what differs is two lines of it,
/// and both are read from the architecture because upstream reads
/// them that way: the rotation at `:126-133` and the FFN at
/// `:179-201`. A row here is a promise that every OTHER line of that
/// graph is the same, which is why `nomic-bert-moe` is not in it (its
/// `moe_every_n_layers` layers are a second FFN shape) and
/// `jina-bert-v2` is not either (a second attention norm).
/// One architecture's row on this loader: the FFN, where the norms
/// sit, and which rotation (if any) the attention uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderSpec {
    pub ffn: BertFfn,
    pub topology: BertTopology,
    /// `true` for the NORM (interleaved) rotation: `llama_model_rope_type`
    /// answers that for `neo-bert` and NEOX for the others.
    pub rope_interleaved: bool,
    /// The tensor the FINAL norm is stored under, for the pre-norm
    /// shape. `neo-bert.cpp:23` uses `enc.output_norm` and
    /// `eurobert.cpp:16` plain `output_norm` -- one fact, two
    /// spellings, which is the `attn_output_norm` case again.
    pub final_norm_name: &'static str,
}

const POST: EncoderSpec = EncoderSpec {
    ffn: BertFfn::GeluSeq,
    topology: BertTopology::PostNormLayerNorm,
    rope_interleaved: false,
    final_norm_name: "",
};

pub const ENCODER_ARCHS: &[(&str, EncoderSpec)] = &[
    ("bert", POST),
    (
        "nomic-bert",
        EncoderSpec {
            ffn: BertFfn::SwigluPar,
            ..POST
        },
    ),
    // `jina-bert-v3.cpp` reuses `llama_model_bert::graph` verbatim
    // (`models.h:314-322`) and creates no position table and no
    // QK-norm tensors, so it is `nomic-bert`'s rotation with `bert`'s
    // ungated GELU FFN.
    ("jina-bert-v3", POST),
    // The one row whose position is neither a table nor a rotation:
    // ALiBi at a literal 8.0 (`jina-bert-v2.cpp:5`). Its GEGLU has two
    // spellings and the loader narrows this one per file.
    (
        "jina-bert-v2",
        EncoderSpec {
            ffn: BertFfn::GegluFusedUp,
            ..POST
        },
    ),
    // The PRE-NORM pair. `neo-bert.cpp:59-118` and
    // `eurobert.cpp:55-114` are the same topology -- RMSNorm before
    // each block, a bare residual after it, one final norm -- and
    // differ in three columns: the QKV spelling (fused vs split, which
    // the loader reads off the file), the FFN's (fused vs a separate
    // gate), and the rotation.
    (
        "neo-bert",
        EncoderSpec {
            ffn: BertFfn::SwigluFusedUp,
            topology: BertTopology::PreNormRms,
            rope_interleaved: true,
            final_norm_name: "enc.output_norm.weight",
        },
    ),
    (
        "eurobert",
        EncoderSpec {
            ffn: BertFfn::SwigluPar,
            topology: BertTopology::PreNormRms,
            rope_interleaved: false,
            final_norm_name: "output_norm.weight",
        },
    ),
];

/// `jina-bert-v2.cpp:5` assigns this as a literal; no key carries it.
const JINA_V2_ALIBI_MAX_BIAS: f32 = 8.0;

/// A `WeightMatrix` copy for the three slices of a split fused QKV.
///
/// `WeightMatrix` is not `Clone` (a quantized one owns its bytes and a
/// folded one an `Arc`), and the encoder needs three owned matrices out
/// of one fused tensor, so this dequantizes into an owned F32 matrix.
/// Only the fused path reaches it, and only at load.
fn clone_matrix(m: &frink_core::WeightMatrix) -> frink_core::WeightMatrix {
    let cols = m.cols();
    let mut data = Vec::with_capacity(m.rows() * cols);
    for r in 0..m.rows() {
        data.extend_from_slice(&m.dequant_row(r));
    }
    frink_core::WeightMatrix::F32(frink_core::Tensor::new(data, vec![m.rows(), cols]))
}

/// Reads and checks `bert.*` hparams. Fails closed on anything the
/// graph in [`crate::bert_encoder`] does not implement.
pub fn read_bert_hparams(file: &impl TensorSource) -> Result<BertHparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    check_arch(&arch)?;
    let p = |suffix: &str| format!("{arch}.{suffix}");

    let spec = ENCODER_ARCHS
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|(_, s)| *s)
        .expect("check_arch admitted this architecture");
    let ffn = spec.ffn;
    let n_layer = crate::mtp_blocks::checked_block_count(
        file,
        &p("block_count"),
        meta_u64(file, &p("block_count"))? as usize,
    )?;
    let n_embd = meta_u64(file, &p("embedding_length"))? as usize;
    let n_ff = meta_u64(file, &p("feed_forward_length"))? as usize;
    let n_head = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_head_kv = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(n_head as u64) as usize;
    let n_ctx_train = meta_u64(file, &p("context_length"))? as usize;

    // `LLM_KV_ATTENTION_LAYERNORM_EPS` is read with `get_key(..., true)`
    // upstream, i.e. required: there is no sane default for a norm this
    // small (this checkpoint's is 1e-12, a thousand times tighter than
    // any RMSNorm eps in the rest of this codebase).
    // The two topologies read DIFFERENT keys, because they are
    // different norm functions: `bert.cpp:4` reads
    // `attention.layer_norm_epsilon` and `neo-bert.cpp:4` /
    // `eurobert.cpp:4` read `attention.layer_norm_rms_epsilon`. Both
    // are required by upstream, so neither gets a default here.
    let eps_key = match spec.topology {
        BertTopology::PostNormLayerNorm => p("attention.layer_norm_epsilon"),
        BertTopology::PreNormRms => p("attention.layer_norm_rms_epsilon"),
    };
    let layer_norm_eps = match file.metadata_f32(&eps_key) {
        Some(eps) => eps,
        None => return Err(LoadError::MissingHparam(eps_key)),
    };

    // `n_token_types` is required by upstream's own loader, which
    // throws "model needs to define token type count".
    let n_token_types = meta_u64(file, "tokenizer.ggml.token_type_count")? as usize;
    if n_token_types == 0 {
        return Err(refuse("tokenizer.ggml.token_type_count is 0"));
    }

    // An encoder is bidirectional by construction. If a checkpoint ever
    // says otherwise, this graph is the wrong one for it.
    if file.metadata_bool(&p("attention.causal")).unwrap_or(false) {
        return Err(refuse(
            "bert.attention.causal is true, but this graph applies no mask — \
             a causal BERT would need a decoder path",
        ));
    }

    if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
        return Err(refuse(&format!(
            "head_count {n_head} is not a multiple of head_count_kv {n_head_kv}"
        )));
    }
    if !n_embd.is_multiple_of(n_head) {
        return Err(refuse(&format!(
            "embedding_length {n_embd} is not divisible by head_count {n_head}"
        )));
    }
    if file.metadata_u64(&p("expert_count")).unwrap_or(0) != 0
        || file.metadata_u64(&p("moe_every_n_layers")).unwrap_or(0) != 0
    {
        return Err(refuse(
            "expert layers (nomic-bert-moe's moe_every_n_layers) are not implemented",
        ));
    }

    // Upstream defaults `hparams.pooling_type` to NONE and reads the key
    // as optional, so an absent key means "return every row", not
    // "guess CLS".
    let pooling = PoolingType::from_gguf(file, &arch)
        .map_err(|e| refuse(&e.to_string()))?
        .unwrap_or(PoolingType::None);

    let cls_id = file
        .metadata_u64("tokenizer.ggml.bos_token_id")
        .unwrap_or(u64::from(DEFAULT_CLS_ID)) as u32;
    let sep_id = file
        .metadata_u64("tokenizer.ggml.seperator_token_id")
        .unwrap_or(u64::from(DEFAULT_SEP_ID)) as u32;

    // `bert.cpp:126-133` rotates for the architectures listed there and
    // adds no position table for them (`:90` is gated on `bert`); the
    // two facts are one field on `BertHparams`.
    // `bert.cpp:189` decides jina-bert-v2's FFN spelling from the
    // FILE, not from the architecture: `up_contains_gate` is true when
    // there is no `ffn_gate` and `ffn_up` is wider than `n_ff`. The
    // table above carries the fused spelling and this narrows it, so
    // the two cannot disagree about a file that has the gate.
    let ffn = if ffn == BertFfn::GegluFusedUp && file.find_tensor("blk.0.ffn_gate.weight").is_some()
    {
        BertFfn::GegluPar
    } else {
        ffn
    };
    // `bert.cpp:78-80` builds no positions at all for `jina-bert-v2`,
    // `:126-133` rotates for the others, and `bert` itself takes the
    // learned table. Three architectures, three answers, one place.
    let rope_theta = (arch != BERT_ARCH && arch != "jina-bert-v2")
        .then(|| file.metadata_f32(&p("rope.freq_base")).unwrap_or(10_000.0));
    // The pre-norm rows read the RMS epsilon key, not the LayerNorm
    // one (`neo-bert.cpp:4`, `eurobert.cpp:4`).
    let _ = &spec;
    let alibi_slopes = (arch == "jina-bert-v2")
        .then(|| frink_core::alibi::slopes(n_head, JINA_V2_ALIBI_MAX_BIAS))
        .flatten();
    let head_dim = n_embd / n_head;
    let rope_dim = file
        .metadata_u64(&p("rope.dimension_count"))
        .map(|v| v as usize)
        .unwrap_or(head_dim);
    if rope_theta.is_some() && (rope_dim == 0 || rope_dim > head_dim || !rope_dim.is_multiple_of(2))
    {
        return Err(refuse(&format!(
            "rope.dimension_count is {rope_dim}, which is not an even width at or under \
             the {head_dim}-wide head"
        )));
    }

    Ok(BertHparams {
        arch,
        topology: spec.topology,
        rope_interleaved: spec.rope_interleaved,
        alibi_slopes,
        rope_theta,
        rope_dim,
        ffn,
        n_layer,
        n_embd,
        n_ff,
        n_head,
        n_head_kv,
        n_ctx_train,
        n_token_types,
        layer_norm_eps,
        pooling,
        cls_id,
        sep_id,
    })
}

/// Loads a `bert` GGUF into a runnable encoder.
pub fn load_bert_encoder_from_path(
    path: impl AsRef<std::path::Path>,
) -> Result<BertEncoder, LoadError> {
    load_bert_encoder(&ShardedGguf::open(path.as_ref())?)
}

/// Same, from an already-open file — so a caller that also needs the
/// tokenizer out of it ([`crate::embedding_model`]) mmaps it once.
pub fn load_bert_encoder(file: &ShardedGguf) -> Result<BertEncoder, LoadError> {
    let hp = read_bert_hparams(file)?;

    let tok_embd = load_weight_matrix(file, "token_embd.weight")?;
    // `bert.cpp:32` creates the table for every architecture on the
    // graph, but `:90` reads it only for `bert` -- and libllama's own
    // load log never names it for `nomic-bert`, measured. So a
    // rotating file carries none and a rotating file that DOES carry
    // one is refused rather than silently ignored.
    // The learned table belongs to `bert` alone: the rotating rows
    // carry none, and `jina-bert-v2` carries none either because its
    // position is ALiBi (`bert.cpp:78-80` builds no `inp_pos` for it).
    let pos_embd = match (hp.rope_theta, hp.alibi_slopes.is_some()) {
        (None, false) => Some(load_weight_matrix(file, "position_embd.weight")?),
        _ => {
            if file.find_tensor("position_embd.weight").is_some() {
                return Err(refuse(
                    "this encoder's graph carries its position in the attention (RoPE at \
                     bert.cpp:126-133, or ALiBi for jina-bert-v2) and never reads \
                     position_embd.weight; llama.cpp refuses the file as carrying an \
                     unread tensor and so does frink",
                ));
            }
            None
        }
    };
    if let Some(table) = &pos_embd {
        if table.rows() != hp.n_ctx_train {
            return Err(refuse(&format!(
                "position_embd.weight has {} rows but {}.context_length says {} — the \
                 learned position table and the advertised context disagree",
                table.rows(),
                hp.arch,
                hp.n_ctx_train
            )));
        }
    }
    if pos_embd.as_ref().is_some_and(|t| t.cols() != hp.n_embd) || tok_embd.cols() != hp.n_embd {
        return Err(refuse(&format!(
            "embedding tables are {} / {} wide but embedding_length is {}",
            tok_embd.cols(),
            pos_embd.as_ref().map_or(hp.n_embd, |t| t.cols()),
            hp.n_embd
        )));
    }

    // The WHOLE table, not row 0. Upstream views `type_embd` at offset
    // 0 and adds it everywhere, because `llama_batch` carries no
    // segment ids; frink's encoder takes them as a parameter, so a
    // cross-encoder pair can put its document half on row 1 the way the
    // checkpoint was trained. Loading row 0 alone is what made
    // `/v1/rerank` rank the relevant document last (see
    // `bert_encoder`'s module docs). The tensor is
    // `TENSOR_NOT_REQUIRED` upstream, so its absence is not an error —
    // it means no segment embedding is added at all, and a reranker
    // checkpoint in that state is refused by `EmbeddingModel`.
    let type_embd = match file.find_tensor("token_types.weight") {
        Some(_) => {
            let table = load_weight_matrix(file, "token_types.weight")?;
            if table.rows() != hp.n_token_types || table.cols() != hp.n_embd {
                return Err(refuse(&format!(
                    "token_types.weight is {}x{}, expected {}x{}",
                    table.rows(),
                    table.cols(),
                    hp.n_token_types,
                    hp.n_embd
                )));
            }
            Some((0..table.rows()).map(|r| table.dequant_row(r)).collect())
        }
        None => None,
    };

    // The post-norm shape norms the embeddings before layer 0
    // (`bert.cpp:96`); the pre-norm one feeds them in raw
    // (`neo-bert.cpp:52`, `eurobert.cpp:48`) and norms once at the end
    // instead.
    let (tok_norm_w, tok_norm_b, final_norm) = match hp.topology {
        BertTopology::PostNormLayerNorm => (
            Some(load_f32_vec(file, "token_embd_norm.weight")?),
            Some(load_f32_vec(file, "token_embd_norm.bias")?),
            None,
        ),
        BertTopology::PreNormRms => {
            let name = ENCODER_ARCHS
                .iter()
                .find(|(a, _)| *a == hp.arch)
                .map(|(_, s)| s.final_norm_name)
                .expect("a loaded architecture is in the table");
            (None, None, Some(load_f32_vec(file, name)?))
        }
    };

    let mut layers = Vec::with_capacity(hp.n_layer);
    for l in 0..hp.n_layer {
        let b = format!("blk.{l}");
        // `neo-bert.cpp:29` stores one `n_embd + 2 * n_embd_gqa`-wide
        // matrix where the other rows store three; every other
        // architecture on this loader is refused for carrying it,
        // because their graphs read the three.
        let fused_qkv = match hp.topology {
            BertTopology::PreNormRms => file.find_tensor(&format!("{b}.attn_qkv.weight")).is_some(),
            BertTopology::PostNormLayerNorm => {
                reject_tensor(
                    file,
                    &format!("{b}.attn_qkv.weight"),
                    "a fused QKV projection; this graph reads separate attn_q/attn_k/attn_v",
                )?;
                false
            }
        };
        let split = if fused_qkv {
            let fused = load_weight_matrix(file, &format!("{b}.attn_qkv.weight"))?;
            let q_rows = hp.n_head * hp.head_dim();
            let kv_rows = hp.n_head_kv * hp.head_dim();
            if fused.rows() != q_rows + 2 * kv_rows {
                return Err(refuse(&format!(
                    "blk.{l}.attn_qkv.weight has {} rows, expected {} (q {q_rows} + 2 x kv \
                     {kv_rows})",
                    fused.rows(),
                    q_rows + 2 * kv_rows
                )));
            }
            Some(crate::qkv_fused::split_fused_weight(
                &fused,
                crate::qkv_fused::FusedQkvRows::from_widths(q_rows, kv_rows),
            )?)
        } else {
            None
        };
        if hp.arch != "jina-bert-v2" {
            reject_tensor(
                file,
                &format!("{b}.attn_q_norm.weight"),
                "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
            )?;
            reject_tensor(
                file,
                &format!("{b}.attn_k_norm.weight"),
                "per-projection QK normalization (jina-bert-v3 / neo-bert), not implemented",
            )?;
            reject_tensor(
                file,
                &format!("{b}.attn_norm_2.weight"),
                "jina-bert-v2's second attention norm, not implemented",
            )?;
        }
        // The gate belongs to `BertFfn::SwigluPar` and to nothing
        // else: a `bert` file that carries one is a file this graph
        // would run as an ungated GELU while llama.cpp ran it gated,
        // which is the silent-wrong shape the refusal exists for.
        if hp.ffn == BertFfn::GeluSeq {
            reject_tensor(
                file,
                &format!("{b}.ffn_gate.weight"),
                "a gated FFN (jina-bert-v2 GEGLU); this architecture's graph runs a plain \
                 GELU MLP (bert.cpp:179-187)",
            )?;
        }
        reject_tensor(
            file,
            &format!("{b}.ffn_up_exps.weight"),
            "MoE expert tensors (nomic-bert-moe), not implemented",
        )?;

        layers.push(BertLayer {
            wq: match split.as_ref() {
                Some((q, _, _)) => clone_matrix(q),
                None => load_weight_matrix(file, &format!("{b}.attn_q.weight"))?,
            },
            bq: load_f32_vec_optional(file, &format!("{b}.attn_q.bias"))?,
            wk: match split.as_ref() {
                Some((_, k, _)) => clone_matrix(k),
                None => load_weight_matrix(file, &format!("{b}.attn_k.weight"))?,
            },
            bk: load_f32_vec_optional(file, &format!("{b}.attn_k.bias"))?,
            wv: match split.as_ref() {
                Some((_, _, v)) => clone_matrix(v),
                None => load_weight_matrix(file, &format!("{b}.attn_v.weight"))?,
            },
            bv: load_f32_vec_optional(file, &format!("{b}.attn_v.bias"))?,
            wo: load_weight_matrix(file, &format!("{b}.attn_output.weight"))?,
            bo: load_f32_vec_optional(file, &format!("{b}.attn_output.bias"))?,
            // The two topologies read different norm slots, and the
            // pair is exclusive by construction: a post-norm layer has
            // `attn_output_norm` / `layer_output_norm` and no
            // `attn_norm`, a pre-norm layer the other way round.
            pre_attn_norm: match hp.topology {
                BertTopology::PostNormLayerNorm => None,
                BertTopology::PreNormRms => {
                    Some(load_f32_vec(file, &format!("{b}.attn_norm.weight"))?)
                }
            },
            pre_ffn_norm: match hp.topology {
                BertTopology::PostNormLayerNorm => None,
                BertTopology::PreNormRms => {
                    Some(load_f32_vec(file, &format!("{b}.ffn_norm.weight"))?)
                }
            },
            attn_out_norm_w: match hp.topology {
                BertTopology::PostNormLayerNorm => {
                    Some(load_f32_vec(file, &format!("{b}.attn_output_norm.weight"))?)
                }
                BertTopology::PreNormRms => None,
            },
            attn_out_norm_b: match hp.topology {
                BertTopology::PostNormLayerNorm => {
                    Some(load_f32_vec(file, &format!("{b}.attn_output_norm.bias"))?)
                }
                BertTopology::PreNormRms => None,
            },
            qk_norm: match load_f32_vec_optional(file, &format!("{b}.attn_q_norm.weight"))? {
                None => None,
                Some(q_w) => Some(crate::bert_encoder::QkLayerNorm {
                    q_w,
                    q_b: load_f32_vec(file, &format!("{b}.attn_q_norm.bias"))?,
                    k_w: load_f32_vec(file, &format!("{b}.attn_k_norm.weight"))?,
                    k_b: load_f32_vec(file, &format!("{b}.attn_k_norm.bias"))?,
                }),
            },
            attn_norm_2: match load_f32_vec_optional(file, &format!("{b}.attn_norm_2.weight"))? {
                None => None,
                Some(w) => Some((w, load_f32_vec(file, &format!("{b}.attn_norm_2.bias"))?)),
            },
            ffn_up: load_weight_matrix(file, &format!("{b}.ffn_up.weight"))?,
            ffn_up_b: load_f32_vec_optional(file, &format!("{b}.ffn_up.bias"))?,
            ffn_gate: match hp.ffn {
                BertFfn::GeluSeq | BertFfn::GegluFusedUp | BertFfn::SwigluFusedUp => None,
                BertFfn::SwigluPar | BertFfn::GegluPar => {
                    Some(load_weight_matrix(file, &format!("{b}.ffn_gate.weight"))?)
                }
            },
            ffn_down: load_weight_matrix(file, &format!("{b}.ffn_down.weight"))?,
            ffn_down_b: load_f32_vec_optional(file, &format!("{b}.ffn_down.bias"))?,
            layer_out_norm_w: match hp.topology {
                BertTopology::PostNormLayerNorm => Some(load_f32_vec(
                    file,
                    &format!("{b}.layer_output_norm.weight"),
                )?),
                BertTopology::PreNormRms => None,
            },
            layer_out_norm_b: match hp.topology {
                BertTopology::PostNormLayerNorm => {
                    Some(load_f32_vec(file, &format!("{b}.layer_output_norm.bias"))?)
                }
                BertTopology::PreNormRms => None,
            },
        });
    }

    let kv_dim = hp.n_head_kv * hp.head_dim();
    for (l, layer) in layers.iter().enumerate() {
        for (name, m, rows) in [
            ("attn_q", &layer.wq, hp.n_embd),
            ("attn_k", &layer.wk, kv_dim),
            ("attn_v", &layer.wv, kv_dim),
            ("attn_output", &layer.wo, hp.n_embd),
            // Twice as wide for the fused GEGLU spelling, where the
            // first half of every row is the gate (`bert.cpp:189`).
            (
                "ffn_up",
                &layer.ffn_up,
                match hp.ffn {
                    BertFfn::GegluFusedUp | BertFfn::SwigluFusedUp => 2 * hp.n_ff,
                    _ => hp.n_ff,
                },
            ),
            ("ffn_down", &layer.ffn_down, hp.n_embd),
        ] {
            if m.rows() != rows {
                return Err(refuse(&format!(
                    "blk.{l}.{name}.weight has {} output rows, expected {rows}",
                    m.rows()
                )));
            }
        }
    }

    assert_every_tensor_consumed(file)?;

    Ok(BertEncoder {
        hp,
        tok_embd,
        type_embd,
        pos_embd,
        tok_norm_w,
        tok_norm_b,
        final_norm,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only `bert`. The other eleven encoder rows in the catalog share
    /// `bert.cpp` upstream, and each of them differs from this graph in
    /// a way that would load clean and embed wrong.
    #[test]
    fn an_architecture_outside_the_table_is_refused_by_name() {
        for arch in ["nomic-bert-moe", "modern-bert", "t5encoder", "llama"] {
            assert!(
                !ENCODER_ARCHS.iter().any(|(a, _)| *a == arch),
                "`{arch}` is in the table; the refusal below would be wrong"
            );
            let err = check_arch(arch).unwrap_err();
            assert!(
                matches!(&err, LoadError::UnsupportedArchitecture(a) if a == arch),
                "{arch} was not refused: {err}"
            );
        }
        assert!(check_arch(BERT_ARCH).is_ok());
    }
}
