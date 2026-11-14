//! Gemma-4 GGUF → [`crate::gemma4_engine::Gemma4Engine`].
//!
//! Reads `{arch}.*` hparams (SWA pattern array, shared KV, split head
//! dims, per-layer emb) and loads tensors named as in llama.cpp
//! `gemma4.cpp` / `LLM_TENSOR_NAMES`.

use frink_gguf::{GgufValue, TensorSource};

use crate::gemma4_engine::{
    Gemma4AttnWeights, Gemma4Engine, Gemma4Hparams, Gemma4LayerWeights, Gemma4Weights,
    GEMMA4_ARCHES,
};
use crate::loader::LoadError;
use crate::loader::{load_f32_vec, load_weight_matrix};

fn meta_u64(file: &impl TensorSource, key: &str) -> Result<u64, LoadError> {
    file.metadata_u64(key)
        .ok_or_else(|| LoadError::MissingHparam(key.to_string()))
}

fn meta_f32(file: &impl TensorSource, key: &str, default: f32) -> f32 {
    file.metadata_f32(key).unwrap_or(default)
}

fn meta_usize_array(file: &impl TensorSource, key: &str) -> Option<Vec<usize>> {
    match file.metadata(key)? {
        GgufValue::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(v.as_u64()? as usize);
            }
            Some(out)
        }
        v => Some(vec![v.as_u64()? as usize]),
    }
}

fn meta_bool_array(file: &impl TensorSource, key: &str, n: usize) -> Result<Vec<bool>, LoadError> {
    match file.metadata(key) {
        Some(GgufValue::Array(items)) if items.len() == n => {
            let mut out = Vec::with_capacity(n);
            for v in items {
                match v {
                    GgufValue::Bool(b) => out.push(*b),
                    GgufValue::U8(u) => out.push(*u != 0),
                    GgufValue::U32(u) => out.push(*u != 0),
                    GgufValue::U64(u) => out.push(*u != 0),
                    GgufValue::I32(i) => out.push(*i != 0),
                    other => {
                        return Err(LoadError::MissingHparam(format!(
                            "{key}: expected bool array element, got {other:?}"
                        )))
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(LoadError::MissingHparam(format!(
            "{key}: expected bool array of length {n}"
        ))),
        None => Err(LoadError::MissingHparam(key.to_string())),
    }
}

/// Parse Gemma-4 hparams; fail-closed on wrong arch or missing required keys.
pub fn read_gemma4_hparams(file: &impl TensorSource) -> Result<Gemma4Hparams, LoadError> {
    let arch = file
        .metadata_str("general.architecture")
        .ok_or_else(|| LoadError::MissingHparam("general.architecture".into()))?
        .to_string();
    if !GEMMA4_ARCHES.contains(&arch.as_str()) {
        return Err(LoadError::UnsupportedArchitecture(arch));
    }
    let p = |suffix: &str| format!("{arch}.{suffix}");
    // The trunk: `block_count` minus the NextN/MTP blocks llama.cpp
    // never runs, decided once for every loader in `crate::mtp_blocks`.
    let n_layer =
        crate::mtp_blocks::trunk_layers(file, &arch, meta_u64(file, &p("block_count"))? as usize)?
            .n_layers;
    let hidden_dim = meta_u64(file, &p("embedding_length"))? as usize;
    let ffn_dims = meta_usize_array(file, &p("feed_forward_length"))
        .ok_or_else(|| LoadError::MissingHparam(p("feed_forward_length")))?;
    if ffn_dims.len() != 1 && ffn_dims.len() != n_layer {
        return Err(LoadError::MissingHparam(format!(
            "{}: expected 1 or {n_layer} feed_forward_length entries, got {}",
            p("feed_forward_length"),
            ffn_dims.len()
        )));
    }
    let ffn_dims = if ffn_dims.len() == 1 {
        vec![ffn_dims[0]; n_layer]
    } else {
        ffn_dims
    };
    let n_heads = meta_u64(file, &p("attention.head_count"))? as usize;
    let n_kv_heads = file
        .metadata_u64(&p("attention.head_count_kv"))
        .unwrap_or(1) as usize;
    let head_dim_full = meta_u64(file, &p("attention.key_length"))? as usize;
    let head_dim_swa = file
        .metadata_u64(&p("attention.key_length_swa"))
        .unwrap_or(head_dim_full as u64) as usize;
    let sliding_window = meta_u64(file, &p("attention.sliding_window"))? as usize;
    let is_swa = meta_bool_array(file, &p("attention.sliding_window_pattern"), n_layer)?;
    let shared_kv = file
        .metadata_u64(&p("attention.shared_kv_layers"))
        .unwrap_or(0) as usize;
    let n_layer_kv_from_start = if shared_kv > 0 && shared_kv < n_layer {
        n_layer - shared_kv
    } else {
        n_layer
    };
    let embd_per_layer = file
        .metadata_u64(&p("embedding_length_per_layer_input"))
        .unwrap_or(0) as usize;
    let final_logit_softcap = file
        .metadata_f32(&p("final_logit_softcapping"))
        .filter(|&v| v > 0.0);
    let rms_norm_eps = meta_f32(file, &p("attention.layer_norm_rms_epsilon"), 1e-6);
    let (rot_dim_full, rot_dim_swa) =
        read_rot_dims(file, &arch, &is_swa, head_dim_full, head_dim_swa)?;
    let rope_theta = meta_f32(file, &p("rope.freq_base"), 1_000_000.0);
    let rope_theta_swa = meta_f32(file, &p("rope.freq_base_swa"), 10_000.0);
    Ok(Gemma4Hparams {
        arch,
        n_layer,
        hidden_dim,
        ffn_dims,
        n_heads,
        n_kv_heads,
        head_dim_full,
        head_dim_swa,
        rot_dim_full,
        rot_dim_swa,
        sliding_window,
        is_swa,
        n_layer_kv_from_start,
        embd_per_layer,
        rms_norm_eps,
        rope_theta,
        rope_theta_swa,
        final_logit_softcap,
        attention_scale: 1.0,
    })
}

/// The RoPE widths as `llama_model::load_hparams` reads them:
/// `rope.dimension_count` defaults to the full head (`n_embd_head_k`),
/// and `rope.dimension_count_swa` defaults to THAT width, not to the
/// sliding head. Each layer then rotates `n_rot(il)` channels
/// (`gemma4.cpp`, `n_rot_l`) and passes the rest through.
///
/// A width ggml cannot take is refused here by key: wider than the head
/// it rotates, or odd, aborts in `ggml_compute_forward_rope_flt`
/// (`n_dims <= ne0`, `n_dims % 2 == 0`). Zero rotates nothing in ggml;
/// the Metal kernels read zero as "whole head", so it is refused rather
/// than given a meaning one backend would not keep (ADR 0002). Only a
/// layer kind the file has is checked, as ggml only ever sees those.
fn read_rot_dims(
    file: &impl TensorSource,
    arch: &str,
    is_swa: &[bool],
    head_dim_full: usize,
    head_dim_swa: usize,
) -> Result<(usize, usize), LoadError> {
    let key_full = format!("{arch}.rope.dimension_count");
    let key_swa = format!("{arch}.rope.dimension_count_swa");
    let full = file
        .metadata_u64(&key_full)
        .map_or(head_dim_full, |v| v as usize);
    let swa = file.metadata_u64(&key_swa).map_or(full, |v| v as usize);
    let check = |rot: usize, head: usize, kind: &str, key: &str| {
        if rot == 0 || !rot.is_multiple_of(2) || rot > head {
            Err(LoadError::ImplausibleHparam(
                key.to_string(),
                rot as u64,
                format!(
                    "a {kind} layer rotates an even, non-zero width no wider than its \
                     {head}-wide head"
                ),
            ))
        } else {
            Ok(())
        }
    };
    if is_swa.iter().any(|&s| !s) {
        check(full, head_dim_full, "full-attention", &key_full)?;
    }
    if is_swa.iter().any(|&s| s) {
        // Without `_swa` the sliding width is the full one, and the
        // error names the key that set it.
        let key = if file.metadata_u64(&key_swa).is_some() {
            &key_swa
        } else {
            &key_full
        };
        check(swa, head_dim_swa, "sliding", key)?;
    }
    Ok((full, swa))
}

fn load_layer(
    file: &impl TensorSource,
    hp: &Gemma4Hparams,
    il: usize,
) -> Result<Gemma4LayerWeights, LoadError> {
    // First, before a single tensor is read. A MoE Gemma-4 layer has no
    // `ffn_gate.weight` at all -- its feed-forward is `ffn_gate_exps`
    // and friends -- so leaving this until the dense loads would report
    // whichever tensor happened to be missing first, which reads like a
    // corrupt file rather than an unsupported architecture.
    refuse_moe_layer(file, hp, il)?;
    let head_dim = hp.head_dim(il);
    let has_kv = hp.has_kv(il);
    let k_name = format!("blk.{il}.attn_k.weight");
    let kn_name = format!("blk.{il}.attn_k_norm.weight");
    let (k_proj, k_norm) = if has_kv || file.find_tensor(&k_name).is_some() {
        (
            Some(load_weight_matrix(file, &k_name)?),
            Some(load_f32_vec(file, &kn_name)?),
        )
    } else {
        (None, None)
    };
    let v_name = format!("blk.{il}.attn_v.weight");
    let v_proj = if file.find_tensor(&v_name).is_some() {
        Some(load_weight_matrix(file, &v_name)?)
    } else {
        None
    };
    let out_scale = if file
        .find_tensor(&format!("blk.{il}.layer_output_scale.weight"))
        .is_some()
    {
        let v = load_f32_vec(file, &format!("blk.{il}.layer_output_scale.weight"))?;
        Some(v.first().copied().unwrap_or(1.0))
    } else {
        None
    };
    let (per_layer_inp_gate, per_layer_proj, per_layer_post_norm) = if hp.embd_per_layer > 0 {
        (
            Some(load_weight_matrix(
                file,
                &format!("blk.{il}.inp_gate.weight"),
            )?),
            Some(load_weight_matrix(file, &format!("blk.{il}.proj.weight"))?),
            Some(load_f32_vec(file, &format!("blk.{il}.post_norm.weight"))?),
        )
    } else {
        (None, None, None)
    };
    let _ = head_dim;
    Ok(Gemma4LayerWeights {
        attn_norm: load_f32_vec(file, &format!("blk.{il}.attn_norm.weight"))?,
        attn: Gemma4AttnWeights {
            q_proj: load_weight_matrix(file, &format!("blk.{il}.attn_q.weight"))?,
            k_proj,
            v_proj,
            o_proj: load_weight_matrix(file, &format!("blk.{il}.attn_output.weight"))?,
            q_norm: load_f32_vec(file, &format!("blk.{il}.attn_q_norm.weight"))?,
            k_norm,
            post_attn_norm: load_f32_vec(file, &format!("blk.{il}.post_attention_norm.weight"))?,
        },
        ffn_norm: load_f32_vec(file, &format!("blk.{il}.ffn_norm.weight"))?,
        ffn_gate: load_weight_matrix(file, &format!("blk.{il}.ffn_gate.weight"))?,
        ffn_up: load_weight_matrix(file, &format!("blk.{il}.ffn_up.weight"))?,
        ffn_down: load_weight_matrix(file, &format!("blk.{il}.ffn_down.weight"))?,
        ffn_post_norm: load_f32_vec(file, &format!("blk.{il}.post_ffw_norm.weight"))?,
        per_layer_inp_gate,
        per_layer_proj,
        per_layer_post_norm,
        out_scale,
    })
}

/// Stops a MoE Gemma-4 checkpoint with an error that says what it is.
///
/// # Why this refuses instead of falling back to the dense path
///
/// Gemma-4's MoE layer is not a router bolted onto the dense FFN. Three
/// of its differences change the numbers without changing any shape, so
/// a stack that ignored them would load, run at full speed, and return
/// fluent text computed the wrong way, with nothing in the output to say
/// so:
///
/// - the router normalizes with a **weightless** RMSNorm, then scales by
///   a learned vector **and** by `hidden^-0.5`;
/// - selection is by raw logit with the softmax over only the selected
///   `k`, which sums to one, unlike a slice of a full softmax;
/// - each routing weight is then multiplied by
///   `per_expert_scale[expert_id]`, after which the weights deliberately
///   no longer sum to one.
///
/// The routing arithmetic itself is ported and tested --
/// [`frink_moe::route_gemma4_moe`] and
/// [`frink_moe::gemma4_router_logits`]. What is missing is the tensor
/// names those two scale vectors and the layer's three feed-forward
/// norms carry in a real GGUF, which cannot be guessed: the feed-forward
/// runs two branches from two different pre-norms, post-norms each with
/// its own weight, RMS-norms their sum with a third, adds the residual,
/// and scales the layer output by a scalar.
///
/// So the error names what was found and what is still needed, which is
/// also how those names get settled the first time someone points a real
/// checkpoint at it.
fn refuse_moe_layer(
    file: &impl TensorSource,
    hp: &Gemma4Hparams,
    il: usize,
) -> Result<(), LoadError> {
    let expert_tensor = format!("blk.{il}.ffn_gate_exps.weight");
    let declared = file
        .metadata_u64(&format!("{}.expert_count", hp.arch))
        .unwrap_or(0);
    if declared == 0 && file.find_tensor(&expert_tensor).is_none() {
        return Ok(());
    }
    let active = file
        .metadata_u64(&format!("{}.expert_used_count", hp.arch))
        .unwrap_or(0);
    let found = file.find_tensor(&expert_tensor).is_some();
    Err(LoadError::UnsupportedFeature(
        hp.arch.clone(),
        format!(
            "this is a MoE Gemma-4 checkpoint ({declared} experts, {active} active per token, \
             blk.{il}.ffn_gate_exps.weight {}), and frink loads only the dense Gemma-4 \
             feed-forward. The routing arithmetic is implemented \
             (frink_moe::route_gemma4_moe) but the loader cannot place it: Gemma-4's router \
             needs a learned input scale and a per-expert output scale, and its feed-forward \
             needs three separate norms, none of whose GGUF tensor names are known here. \
             Loading it as dense would drop the per-expert scale and collapse the norms, which \
             produces fluent, wrong output rather than an error. Report the tensor names in \
             this file to have them wired up",
            if found { "present" } else { "absent" }
        ),
    ))
}

/// Load a Gemma-4 GGUF into [`Gemma4Engine`].
pub fn load_gemma4_engine(file: &impl TensorSource) -> Result<Gemma4Engine, LoadError> {
    let hp = read_gemma4_hparams(file)?;
    let token_embd = load_weight_matrix(file, "token_embd.weight")?;
    let output_head = match load_weight_matrix(file, "output.weight") {
        Ok(w) => w,
        Err(_) => load_weight_matrix(file, "token_embd.weight")?,
    };
    let (per_layer_token_embd, per_layer_model_proj, per_layer_proj_norm) = if hp.embd_per_layer > 0
    {
        (
            Some(load_weight_matrix(file, "per_layer_token_embd.weight")?),
            Some(load_weight_matrix(file, "per_layer_model_proj.weight")?),
            Some(load_f32_vec(file, "per_layer_proj_norm.weight")?),
        )
    } else {
        (None, None, None)
    };
    let rope_freqs = if file.find_tensor("rope_freqs.weight").is_some() {
        let f = load_f32_vec(file, "rope_freqs.weight")?;
        // `load_arch_tensors` creates it `{n_embd_head/2}` on the full
        // layers; ggml reads its first `n_rot/2` bands.
        if hp.is_swa.iter().any(|&s| !s) && f.len() != hp.head_dim_full / 2 {
            return Err(LoadError::UnsupportedFeature(
                hp.arch.clone(),
                format!(
                    "rope_freqs.weight has {} entries, llama.cpp requires head_dim_full/2 = {}",
                    f.len(),
                    hp.head_dim_full / 2
                ),
            ));
        }
        Some(f)
    } else {
        None
    };
    let mut layers = Vec::with_capacity(hp.n_layer);
    for il in 0..hp.n_layer {
        layers.push(load_layer(file, &hp, il)?);
    }
    let engine = Gemma4Engine {
        weights: Gemma4Weights {
            token_embd,
            per_layer_token_embd,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            output_norm: load_f32_vec(file, "output_norm.weight")?,
            output_head,
            rope_freqs,
        },
        hp,
        // llama.cpp's default; callers set the run's `--ctk` over it.
        kv_dtype: frink_core::kv_store::KvDtype::F16,
        #[cfg(feature = "metal")]
        device_aux: Default::default(),
    };
    // Same contract as the generic loader: resolve every kernel now,
    // then seal. This engine has no batched prefill, which the probe
    // records explicitly -- see `Gemma4Engine::probe_kernels`.
    engine.probe_kernels();
    frink_core::kernel_registry::seal_or_error()
        .map_err(|e| LoadError::StrictKernels(e.to_string()))?;
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use crate::gemma4_engine::Gemma4Hparams;
    use byteorder::{LittleEndian, WriteBytesExt};
    use frink_gguf::GgufFile;
    use std::io::Write;

    struct FixtureTensor {
        name: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.write_f32::<LittleEndian>(*x).unwrap();
        }
        b
    }

    fn f32_tensor(name: &str, shape: Vec<u64>, values: Vec<f32>) -> FixtureTensor {
        FixtureTensor {
            name: name.into(),
            shape,
            bytes: f32_bytes(&values),
        }
    }

    fn build_gguf(
        arch: &str,
        kv: &[(&str, u64)],
        fkv: &[(&str, f32)],
        bool_arr: Option<(&str, &[bool])>,
        tensors: &[FixtureTensor],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
            .unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();
        buf.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
        let kv_count = 1 + kv.len() + fkv.len() + usize::from(bool_arr.is_some());
        buf.write_u64::<LittleEndian>(kv_count as u64).unwrap();
        let write_string = |buf: &mut Vec<u8>, s: &str| {
            buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
        };
        write_string(&mut buf, "general.architecture");
        buf.write_u32::<LittleEndian>(8).unwrap();
        write_string(&mut buf, arch);
        for &(k, v) in kv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(10).unwrap();
            buf.write_u64::<LittleEndian>(v).unwrap();
        }
        for &(k, v) in fkv {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(6).unwrap();
            buf.write_f32::<LittleEndian>(v).unwrap();
        }
        if let Some((k, arr)) = bool_arr {
            write_string(&mut buf, k);
            buf.write_u32::<LittleEndian>(9).unwrap(); // ARRAY
            buf.write_u32::<LittleEndian>(7).unwrap(); // BOOL
            buf.write_u64::<LittleEndian>(arr.len() as u64).unwrap();
            for &b in arr {
                buf.write_u8(u8::from(b)).unwrap();
            }
        }
        let mut offset = 0u64;
        let mut offsets = Vec::with_capacity(tensors.len());
        for t in tensors {
            write_string(&mut buf, &t.name);
            buf.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
            for &d in t.shape.iter().rev() {
                buf.write_u64::<LittleEndian>(d).unwrap();
            }
            buf.write_u32::<LittleEndian>(0).unwrap();
            offsets.push(offset);
            buf.write_u64::<LittleEndian>(offset).unwrap();
            offset += (t.bytes.len().div_ceil(32) * 32) as u64;
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        let data_start = buf.len();
        for (t, &off) in tensors.iter().zip(offsets.iter()) {
            while buf.len() < data_start + off as usize {
                buf.push(0);
            }
            buf.extend_from_slice(&t.bytes);
            while buf.len() % 32 != 0 {
                buf.push(0);
            }
        }
        buf
    }

    const HP_SWA: [bool; 5] = [true, true, true, true, false];

    /// `read_gemma4_hparams` on a tiny file with a 16-wide full head and
    /// an 8-wide sliding one, plus `rope` keys (`dimension_count` and
    /// its `_swa` twin, by suffix).
    fn hparams_with_rope(rope: &[(&str, u64)]) -> Result<Gemma4Hparams, LoadError> {
        let n_layer = HP_SWA.len();
        let h = 32usize;
        let tensors = vec![
            f32_tensor("token_embd.weight", vec![4u64, h as u64], vec![0.01; 4 * h]),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        ];
        // Minimal tensors — only testing hparam parse via read_gemma4_hparams
        // through a tiny file that still opens.
        let keys: Vec<String> = rope
            .iter()
            .map(|(k, _)| format!("gemma4.rope.{k}"))
            .collect();
        let mut kv = vec![
            ("gemma4.block_count", n_layer as u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", 64u64),
            ("gemma4.attention.head_count", 4u64),
            ("gemma4.attention.head_count_kv", 1u64),
            ("gemma4.attention.key_length", 16u64),
            ("gemma4.attention.key_length_swa", 8u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 2u64),
            ("gemma4.embedding_length_per_layer_input", 4u64),
        ];
        kv.extend(keys.iter().zip(rope).map(|(k, &(_, v))| (k.as_str(), v)));
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-6f32),
            ("gemma4.rope.freq_base", 1e6f32),
            ("gemma4.rope.freq_base_swa", 1e4f32),
            ("gemma4.final_logit_softcapping", 30.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &HP_SWA)),
            &tensors,
        );
        let tag: String = rope.iter().map(|(k, v)| format!("_{k}{v}")).collect();
        let path =
            std::env::temp_dir().join(format!("frink_gemma4_hp_{}{tag}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let hp = read_gemma4_hparams(&file);
        let _ = std::fs::remove_file(&path);
        hp
    }

    #[test]
    fn parse_hparams_routing_from_synthetic_gguf() {
        let hp = hparams_with_rope(&[("dimension_count", 16), ("dimension_count_swa", 8)])
            .expect("hparams");
        assert_eq!(hp.n_layer, 5);
        assert_eq!(hp.n_layer_kv_from_start, 3);
        assert_eq!(hp.head_dim_full, 16);
        assert_eq!(hp.head_dim_swa, 8);
        assert_eq!((hp.rot_dim(4), hp.rot_dim(0)), (16, 8), "whole heads");
        assert_eq!(hp.is_swa, HP_SWA.to_vec());
        assert!(hp.has_kv(2));
        assert!(!hp.has_kv(3));
        assert_eq!(hp.kv_reuse_layer(3), 1); // SWA → 3-2? n_kv=3, swa → 3-2=1
        assert_eq!(hp.kv_reuse_layer(4), 2); // full → 3-1=2
        let _ = Gemma4Hparams { ..hp };
    }

    /// #635: the RoPE widths as llama.cpp's `load_hparams` reads them.
    /// `dimension_count` defaults to the full head and `_swa` to
    /// `dimension_count`, NOT to the sliding head; a partial width is
    /// honoured per layer kind.
    #[test]
    fn rope_widths_follow_llama_defaults() {
        let hp = hparams_with_rope(&[("dimension_count", 4)]).expect("partial");
        assert_eq!(
            (hp.rot_dim(4), hp.rot_dim(0)),
            (4, 4),
            "_swa defaults to n_rot_full"
        );
        let hp = hparams_with_rope(&[("dimension_count", 8), ("dimension_count_swa", 4)])
            .expect("two partial widths");
        assert_eq!((hp.rot_dim(4), hp.rot_dim(0)), (8, 4));
    }

    /// A width ggml aborts on (`n_dims <= ne0`, even), or zero, is
    /// refused at load, naming the key that set it. With no RoPE keys at
    /// all the sliding width defaults to the 16-wide full head, wider
    /// than the 8-wide sliding one: llama.cpp aborts in
    /// `ggml_compute_forward_rope_flt` on that file, and frink refuses it.
    #[test]
    fn a_rope_width_ggml_cannot_take_is_refused_by_key() {
        let refused = |rope: &[(&str, u64)], key: &str, val: u64| match hparams_with_rope(rope) {
            Err(LoadError::ImplausibleHparam(k, v, _)) => {
                assert_eq!((k.as_str(), v), (key, val), "{rope:?}")
            }
            other => panic!("{rope:?}: expected a refusal, got {other:?}"),
        };
        refused(&[], "gemma4.rope.dimension_count", 16);
        refused(
            &[("dimension_count", 32), ("dimension_count_swa", 8)],
            "gemma4.rope.dimension_count",
            32,
        );
        refused(
            &[("dimension_count", 16), ("dimension_count_swa", 10)],
            "gemma4.rope.dimension_count_swa",
            10,
        );
        refused(
            &[("dimension_count", 6), ("dimension_count_swa", 5)],
            "gemma4.rope.dimension_count_swa",
            5,
        );
        refused(&[("dimension_count", 0)], "gemma4.rope.dimension_count", 0);
    }

    #[test]
    fn load_tiny_gemma4_and_forward() {
        let n_layer = 2usize;
        let h = 16usize;
        let n_heads = 2usize;
        let n_kv = 1usize;
        let hd_swa = 8usize;
        let hd_full = 8usize; // same dim so shared-KV reuse is safe in tiny fixture
        let ffn = 32usize;
        let n_pl = 4usize;
        let vocab = 4usize;
        let swa = [true, false];

        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; vocab * h],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
            f32_tensor(
                "per_layer_token_embd.weight",
                vec![vocab as u64, (n_pl * n_layer) as u64],
                vec![0.01; vocab * n_pl * n_layer],
            ),
            f32_tensor(
                "per_layer_model_proj.weight",
                vec![(n_pl * n_layer) as u64, h as u64],
                vec![0.01; n_pl * n_layer * h],
            ),
            f32_tensor(
                "per_layer_proj_norm.weight",
                vec![n_pl as u64],
                vec![1.0; n_pl],
            ),
        ];
        for (il, &is_swa) in swa.iter().enumerate().take(n_layer) {
            let hd = if is_swa { hd_swa } else { hd_full };
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_q.weight"),
                vec![(n_heads * hd) as u64, h as u64],
                vec![0.01; n_heads * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_k.weight"),
                vec![(n_kv * hd) as u64, h as u64],
                vec![0.01; n_kv * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_v.weight"),
                vec![(n_kv * hd) as u64, h as u64],
                vec![0.01; n_kv * hd * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_output.weight"),
                vec![h as u64, (n_heads * hd) as u64],
                vec![0.01; h * n_heads * hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_q_norm.weight"),
                vec![hd as u64],
                vec![1.0; hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.attn_k_norm.weight"),
                vec![hd as u64],
                vec![1.0; hd],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_attention_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_gate.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; ffn * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_up.weight"),
                vec![ffn as u64, h as u64],
                vec![0.01; ffn * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.ffn_down.weight"),
                vec![h as u64, ffn as u64],
                vec![0.01; h * ffn],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_ffw_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.inp_gate.weight"),
                vec![n_pl as u64, h as u64],
                vec![0.01; n_pl * h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.proj.weight"),
                vec![h as u64, n_pl as u64],
                vec![0.01; h * n_pl],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.post_norm.weight"),
                vec![h as u64],
                vec![1.0; h],
            ));
            tensors.push(f32_tensor(
                &format!("blk.{il}.layer_output_scale.weight"),
                vec![1u64],
                vec![1.0],
            ));
        }
        let kv = [
            ("gemma4.block_count", n_layer as u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", ffn as u64),
            ("gemma4.attention.head_count", n_heads as u64),
            ("gemma4.attention.head_count_kv", n_kv as u64),
            ("gemma4.attention.key_length", hd_full as u64),
            ("gemma4.attention.key_length_swa", hd_swa as u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 0u64),
            ("gemma4.embedding_length_per_layer_input", n_pl as u64),
        ];
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("gemma4.rope.freq_base", 10000.0f32),
            ("gemma4.rope.freq_base_swa", 10000.0f32),
            ("gemma4.final_logit_softcapping", 30.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &swa)),
            &tensors,
        );
        let path =
            std::env::temp_dir().join(format!("frink_gemma4_fwd_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let engine = load_gemma4_engine(&file).expect("load");
        assert_eq!(engine.hp.n_layer, 2);
        assert_eq!(engine.vocab_size(), vocab);
        let mut state = engine.new_state();
        let logits = engine.forward_token(0, 0, &mut state);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|x| x.is_finite()));
        // The host caches store what `kv_dtype` says: llama.cpp's f16
        // from the loader, and whatever a run's `--ctk` sets over it
        // (#634: they were f32 whatever `--ctk` said).
        use frink_core::kv_store::KvDtype;
        let dtypes = |s: &crate::gemma4_engine::Gemma4DecodeState| -> Vec<KvDtype> {
            s.kv.iter().flatten().map(|c| c.k.dtype()).collect()
        };
        assert_eq!(engine.kv_dtype, KvDtype::F16);
        assert_eq!(dtypes(&state), vec![KvDtype::F16; 2]);
        let mut engine = engine;
        engine.kv_dtype = KvDtype::F32;
        assert_eq!(dtypes(&engine.new_state()), vec![KvDtype::F32; 2]);
        let _ = std::fs::remove_file(&path);
    }
    /// A MoE Gemma-4 checkpoint is refused by NAME, not by a missing
    /// dense tensor.
    ///
    /// The reported symptom was a bare
    /// `TensorNotFound(blk.0.ffn_gate.weight)`, which reads like a
    /// corrupt file. A MoE layer simply has no `ffn_gate.weight`: its
    /// feed-forward is `ffn_gate_exps` and friends. The error now says
    /// what the file is, how many experts it has, and what frink still
    /// needs -- and crucially it REFUSES rather than falling back to the
    /// dense path, because dropping Gemma-4's per-expert scale and
    /// collapsing its three feed-forward norms changes no shape at all
    /// and yields fluent, wrong output.
    #[test]
    fn a_moe_gemma4_checkpoint_is_refused_by_name_not_by_a_missing_dense_tensor() {
        let (h, n_experts, ffn, vocab) = (16usize, 4usize, 32usize, 4usize);
        let swa = [false];
        let mut tensors = vec![
            f32_tensor(
                "token_embd.weight",
                vec![vocab as u64, h as u64],
                vec![0.01; vocab * h],
            ),
            f32_tensor("output_norm.weight", vec![h as u64], vec![1.0; h]),
        ];
        // Only the attention side plus the MoE feed-forward: exactly the
        // tensor set that made the dense loader report a missing gate.
        for name in [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_norm.weight",
            "post_ffw_norm.weight",
        ] {
            tensors.push(f32_tensor(
                &format!("blk.0.{name}"),
                vec![h as u64],
                vec![1.0; h],
            ));
        }
        // A real MoE checkpoint carries its attention side unchanged;
        // included so the refusal is proven to be about the FEED-FORWARD
        // rather than about a fixture that is merely incomplete.
        let (n_heads, n_kv, hd) = (2usize, 1usize, 8usize);
        for (name, rows, cols) in [
            ("attn_q", n_heads * hd, h),
            ("attn_k", n_kv * hd, h),
            ("attn_v", n_kv * hd, h),
            ("attn_output", h, n_heads * hd),
        ] {
            tensors.push(f32_tensor(
                &format!("blk.0.{name}.weight"),
                vec![rows as u64, cols as u64],
                vec![0.01; rows * cols],
            ));
        }
        for name in ["attn_q_norm", "attn_k_norm"] {
            tensors.push(f32_tensor(
                &format!("blk.0.{name}.weight"),
                vec![hd as u64],
                vec![1.0; hd],
            ));
        }
        tensors.push(f32_tensor(
            "blk.0.ffn_gate_inp.weight",
            vec![n_experts as u64, h as u64],
            vec![0.01; n_experts * h],
        ));
        for name in ["ffn_gate_exps", "ffn_up_exps"] {
            tensors.push(f32_tensor(
                &format!("blk.0.{name}.weight"),
                vec![n_experts as u64, ffn as u64, h as u64],
                vec![0.01; n_experts * ffn * h],
            ));
        }

        let kv = [
            ("gemma4.block_count", 1u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", ffn as u64),
            ("gemma4.attention.head_count", 2u64),
            ("gemma4.attention.head_count_kv", 1u64),
            ("gemma4.attention.key_length", 8u64),
            ("gemma4.attention.key_length_swa", 8u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 0u64),
            ("gemma4.embedding_length_per_layer_input", 0u64),
            ("gemma4.expert_count", n_experts as u64),
            ("gemma4.expert_used_count", 2u64),
        ];
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("gemma4.rope.freq_base", 10000.0f32),
            ("gemma4.rope.freq_base_swa", 10000.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &swa)),
            &tensors,
        );
        let path =
            std::env::temp_dir().join(format!("frink_gemma4_moe_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();

        let err = match load_gemma4_engine(&file) {
            Err(e) => e,
            Ok(_) => panic!("a MoE Gemma-4 must not load as dense"),
        };
        let msg = err.to_string();
        assert!(
            matches!(err, LoadError::UnsupportedFeature(ref a, _) if a == "gemma4"),
            "wrong error kind: {msg}"
        );
        assert!(msg.contains("MoE Gemma-4"), "{msg}");
        assert!(msg.contains("4 experts"), "the count it found: {msg}");
        assert!(msg.contains("2 active"), "the top-k it found: {msg}");
        assert!(
            msg.contains("route_gemma4_moe"),
            "point at the routing that IS implemented: {msg}"
        );
        assert!(
            !msg.contains("ffn_gate.weight'"),
            "must not report the dense tensor a MoE layer never has: {msg}"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// And the detection must not fire on a dense checkpoint, which has
    /// neither the metadata nor the expert tensors.
    #[test]
    fn a_dense_gemma4_checkpoint_is_untouched_by_the_moe_check() {
        // `load_tiny_gemma4_and_forward` is the positive case in full;
        // this pins the predicate itself, so a future edit that made the
        // check fire on `expert_count == 0` would fail here rather than
        // by breaking every dense load.
        let (h, ffn) = (16usize, 32usize);
        let tensors = vec![f32_tensor(
            "token_embd.weight",
            vec![4u64, h as u64],
            vec![0.01; 4 * h],
        )];
        let kv = [
            ("gemma4.block_count", 1u64),
            ("gemma4.embedding_length", h as u64),
            ("gemma4.feed_forward_length", ffn as u64),
            ("gemma4.attention.head_count", 2u64),
            ("gemma4.attention.head_count_kv", 1u64),
            ("gemma4.attention.key_length", 8u64),
            ("gemma4.attention.key_length_swa", 8u64),
            ("gemma4.attention.sliding_window", 4u64),
            ("gemma4.attention.shared_kv_layers", 0u64),
            ("gemma4.embedding_length_per_layer_input", 0u64),
        ];
        let fkv = [
            ("gemma4.attention.layer_norm_rms_epsilon", 1e-5f32),
            ("gemma4.rope.freq_base", 10000.0f32),
            ("gemma4.rope.freq_base_swa", 10000.0f32),
        ];
        let bytes = build_gguf(
            "gemma4",
            &kv,
            &fkv,
            Some(("gemma4.attention.sliding_window_pattern", &[false])),
            &tensors,
        );
        let path =
            std::env::temp_dir().join(format!("frink_gemma4_dense_{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let hp = read_gemma4_hparams(&file).expect("hparams");
        assert!(
            refuse_moe_layer(&file, &hp, 0).is_ok(),
            "a dense checkpoint must pass the MoE check untouched"
        );
        let _ = std::fs::remove_file(&path);
    }
}
