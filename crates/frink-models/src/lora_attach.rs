//! Attaching a parsed [`LoraAdapter`] to a [`Decoder`]: the walk from
//! each `<base name>` in the adapter to the `WeightMatrix` (or
//! matrices) frink holds for it, the shape checks llama.cpp makes
//! against the base tensor (`llama-adapter.cpp:346-367`), and the
//! refusals for what frink holds differently.
//!
//! llama.cpp keys an adapter on the base tensor's NAME: `get_weight(w)`
//! looks `w->name` up in `ab_map` at every `build_lora_mm` call. A
//! `WeightMatrix` carries no name (`frink_core::activation_tap` says
//! why), so the name is resolved ONCE here, at attach, into the field
//! the loader put that tensor in -- and where the loader split one
//! file tensor into several matrices (a fused `attn_qkv.weight` into
//! Q/K/V, Phi-3's fused `ffn_up.weight` into gate/up), the adapter's
//! `lora_b` rows are split the same way, in the same order, with one
//! copy of `lora_a` each: `B (A x)` over stacked rows IS the stacked
//! `B_i (A x)`, exactly. Where the loader ALIASED one file tensor into
//! two matrices (the ungated FFN's gate is its up, `loader/layer_parts.rs:191`),
//! the whole delta goes on both, so the two stay one tensor.
//!
//! What is refused, by name, rather than approximated:
//!
//! * a routed-expert tensor (`*_exps`): upstream applies those through
//!   `build_lora_mm_id` and frink has no per-expert delta;
//! * a tensor frink holds as something other than a `WeightMatrix`
//!   (a norm vector, a bias): the delta would be dropped;
//! * `token_embd.weight` on a base whose output head is tied to the
//!   embedding: upstream aborts in `ggml_mul_mat` on that pair, so no
//!   engine serves it;
//! * an expert layer whose weights are leased from a store per use.
//!
//! Every fused Metal launch is fenced off a decoder with ANY adapter
//! attached, through the one predicate they share
//! (`Decoder::metal_can_serve_model`), because the stacks read weight
//! bytes past the `WeightMatrix` methods that serve the delta. The
//! per-matrix GPU launches (`apply_gpu`, `apply_gpu_multi`,
//! `apply_gpu_batch`, CUDA's `mul_mm`) run the base on the device and
//! add the rank-sized delta on the host, so Metal and CUDA still serve
//! an adapted model -- on the per-matrix path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use frink_core::weight_matrix::{LoraDelta, LoraScale, WeightMatrix};
use frink_gguf::TensorSource;

use crate::decoder::{Decoder, ExpertBacking};
use crate::lora::{LoraAdapter, LoraError, LoraPair};

/// One adapter after attach: what `GET /lora-adapters` lists and what
/// `POST /lora-adapters` / a per-request `lora` list changes.
#[derive(Debug, Clone)]
pub struct LoraAttached {
    pub path: PathBuf,
    pub alpha: f32,
    pub task_name: String,
    pub prompt_prefix: String,
    /// Shared by every delta this adapter attached.
    pub scale: Arc<LoraScale>,
    /// Base tensors this adapter decorates.
    pub n_tensors: usize,
}

impl LoraAttached {
    pub fn scale(&self) -> f32 {
        self.scale.get()
    }
}

/// One `--lora FNAME` or `--lora-scaled FNAME:SCALE`, before the file
/// is opened. Parsed in ONE place for the CLI and the server, so the
/// two cannot read `a:b:0.5` differently.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub scale: f32,
}

impl LoraSpec {
    /// `--lora FNAME`: scale 1, as `arg.cpp:2869` pushes it.
    pub fn plain(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            scale: 1.0,
        }
    }

    /// `--lora-scaled FNAME:SCALE` (`arg.cpp:2878-2885`). The LAST colon
    /// splits, so a path with a drive letter or a colon in a directory
    /// name still parses; llama.cpp splits on every colon and refuses
    /// those.
    pub fn parse_scaled(spec: &str) -> Result<Self, String> {
        let (path, scale) = spec
            .rsplit_once(':')
            .ok_or_else(|| format!("lora-scaled format: FNAME:SCALE (got {spec:?})"))?;
        let scale: f32 = scale
            .trim()
            .parse()
            .map_err(|_| format!("lora-scaled format: FNAME:SCALE ({scale:?} is not a number)"))?;
        if path.is_empty() {
            return Err(format!("lora-scaled format: FNAME:SCALE (got {spec:?})"));
        }
        Ok(Self {
            path: PathBuf::from(path),
            scale,
        })
    }

    /// The CLI's two flags as one list, in the order given: every
    /// `--lora` value (comma-separated, as upstream's `parse_csv_row`)
    /// then every `--lora-scaled` value.
    pub fn from_flags(plain: &[String], scaled: &[String]) -> Result<Vec<Self>, String> {
        let mut out = Vec::new();
        for item in plain.iter().flat_map(|v| v.split(',')) {
            let item = item.trim();
            if !item.is_empty() {
                out.push(Self::plain(item));
            }
        }
        for item in scaled.iter().flat_map(|v| v.split(',')) {
            let item = item.trim();
            if !item.is_empty() {
                out.push(Self::parse_scaled(item)?);
            }
        }
        Ok(out)
    }
}

impl Decoder {
    /// Opens and attaches every spec in order. ONE function for the CLI
    /// and the server, so a refusal reads the same from both.
    pub fn attach_lora_specs(
        &mut self,
        base: &impl TensorSource,
        specs: &[LoraSpec],
    ) -> Result<(), LoraError> {
        for spec in specs {
            let adapter = LoraAdapter::open(&spec.path)?;
            let id = self.attach_lora(base, adapter, spec.scale)?;
            let a = &self.lora_adapters[id];
            eprintln!(
                "frink: lora adapter {id}: {} ({} tensor(s), alpha {}, scale {})",
                spec.path.display(),
                a.n_tensors,
                a.alpha,
                spec.scale
            );
        }
        Ok(())
    }
}

/// Where a base tensor's rows went.
enum Parts<'a> {
    /// One matrix, or several holding consecutive row blocks of the
    /// file tensor, in file order.
    Split(Vec<&'a mut WeightMatrix>),
    /// Several matrices that are each the WHOLE file tensor.
    Alias(Vec<&'a mut WeightMatrix>),
}

enum Target {
    Embedding,
    Output,
    Layer(usize, LayerTarget),
}

enum LayerTarget {
    Q,
    K,
    V,
    Qkv,
    O,
    AttnGate,
    FfnGate,
    FfnUp,
    FfnDown,
    ShexpGate,
    ShexpUp,
    ShexpDown,
    Router,
}

fn target_of(name: &str, path: &Path) -> Result<Target, LoraError> {
    let no_projection = || LoraError::NoProjection {
        path: path.to_path_buf(),
        name: name.to_string(),
    };
    match name {
        "token_embd.weight" => return Ok(Target::Embedding),
        "output.weight" => return Ok(Target::Output),
        _ => {}
    }
    let rest = name.strip_prefix("blk.").ok_or_else(no_projection)?;
    let (il, tensor) = rest.split_once('.').ok_or_else(no_projection)?;
    let il: usize = il.parse().map_err(|_| no_projection())?;
    let t = tensor.strip_suffix(".weight").ok_or_else(no_projection)?;
    if t.ends_with("_exps") {
        return Err(LoraError::RoutedExperts {
            path: path.to_path_buf(),
            name: name.to_string(),
        });
    }
    let kind = match t {
        "attn_q" => LayerTarget::Q,
        "attn_k" => LayerTarget::K,
        "attn_v" => LayerTarget::V,
        "attn_qkv" => LayerTarget::Qkv,
        "attn_output" => LayerTarget::O,
        "attn_gate" => LayerTarget::AttnGate,
        "ffn_gate" => LayerTarget::FfnGate,
        "ffn_up" => LayerTarget::FfnUp,
        "ffn_down" => LayerTarget::FfnDown,
        "ffn_gate_shexp" => LayerTarget::ShexpGate,
        "ffn_up_shexp" => LayerTarget::ShexpUp,
        "ffn_down_shexp" => LayerTarget::ShexpDown,
        "ffn_gate_inp" => LayerTarget::Router,
        _ => return Err(no_projection()),
    };
    Ok(Target::Layer(il, kind))
}

impl Decoder {
    /// Is any adapter attached? The fact every fused Metal launch is
    /// fenced on.
    pub fn lora_attached(&self) -> bool {
        !self.lora_adapters.is_empty()
    }

    /// Attaches `adapter` at `scale` and returns its id (its index in
    /// [`Decoder::lora_adapters`]). `base` is the model's own GGUF: the
    /// architecture and every tensor's presence and shape are checked
    /// against it, as `llama_adapter_lora_init_impl` checks them against
    /// the loaded model.
    ///
    /// On an error the decoder may already carry some of this adapter's
    /// deltas; a caller must discard it rather than serve it, which is
    /// what both the CLI and the server do (the load fails).
    pub fn attach_lora(
        &mut self,
        base: &impl TensorSource,
        adapter: LoraAdapter,
        scale: f32,
    ) -> Result<usize, LoraError> {
        let path = adapter.path.to_path_buf();
        let base_arch = base.metadata_str("general.architecture").unwrap_or("");
        if adapter.arch != base_arch {
            return Err(LoraError::ArchMismatch {
                path,
                adapter: adapter.arch,
                base: base_arch.to_string(),
            });
        }
        let handle = LoraScale::new(scale);
        let n_tensors = adapter.pairs.len();
        for (name, pair) in &adapter.pairs {
            let info = base.find_tensor(name).ok_or_else(|| LoraError::NotInBase {
                path: path.to_path_buf(),
                name: name.clone(),
            })?;
            let target = target_of(name, &path)?;
            // Checked HERE, in name order, rather than up front: on an
            // adapter that also names `output.weight`, libllama's map
            // walk reaches that pair's "does not exist in base model"
            // first (`output.weight` sorts before `token_embd.weight`),
            // and so does this loop.
            if matches!(target, Target::Embedding) && base.find_tensor("output.weight").is_none() {
                return Err(LoraError::TiedHead { path });
            }
            if info.shape.len() != 2 {
                return Err(LoraError::NoProjection {
                    path: path.to_path_buf(),
                    name: name.clone(),
                });
            }
            let (rows, cols) = (info.shape[1] as usize, info.shape[0] as usize);
            let whole = checked_pair(&path, name, pair, &target, rows, cols)?;
            let shape_err = || LoraError::Shape {
                path: path.to_path_buf(),
                name: name.clone(),
                rows,
                cols,
                a: pair.a.shape,
                b: pair.b.shape,
            };
            let alpha = adapter.alpha;
            match self.parts_mut(&target, &path, name, rows)? {
                Parts::Split(parts) => {
                    // `B (A x)` over stacked rows is the stacked
                    // `B_i (A x)`: each part takes its block of B's
                    // rows and a copy of A.
                    if parts.iter().map(|m| m.rows()).sum::<usize>() != rows
                        || parts.iter().any(|m| m.cols() != cols)
                    {
                        return Err(shape_err());
                    }
                    let mut row0 = 0;
                    for m in parts {
                        let n = m.rows();
                        let b = whole.b[row0 * whole.rank..(row0 + n) * whole.rank].to_vec();
                        row0 += n;
                        let delta = LoraDelta::new(
                            whole.a.clone(),
                            b,
                            whole.rank,
                            n,
                            cols,
                            alpha,
                            Arc::clone(&handle),
                        )
                        .map_err(|_| shape_err())?;
                        m.attach_lora(delta);
                    }
                }
                Parts::Alias(parts) => {
                    for m in parts {
                        if m.rows() != rows || m.cols() != cols {
                            return Err(shape_err());
                        }
                        let delta = LoraDelta::new(
                            whole.a.clone(),
                            whole.b.clone(),
                            whole.rank,
                            rows,
                            cols,
                            alpha,
                            Arc::clone(&handle),
                        )
                        .map_err(|_| shape_err())?;
                        m.attach_lora(delta);
                    }
                }
            }
        }
        self.lora_adapters.push(LoraAttached {
            path,
            alpha: adapter.alpha,
            task_name: adapter.task_name,
            prompt_prefix: adapter.prompt_prefix,
            scale: handle,
            n_tensors,
        });
        Ok(self.lora_adapters.len() - 1)
    }

    /// Sets every adapter's scale from `scales` (`id -> scale`); an
    /// adapter not listed goes to `0`, as `construct_lora_list`
    /// (`server-context.cpp:1721-1732`) sets it. An unknown id is an
    /// error naming the range, where upstream ignores it.
    pub fn set_lora_scales(&self, scales: &[(usize, f32)]) -> Result<(), String> {
        for &(id, _) in scales {
            if id >= self.lora_adapters.len() {
                return Err(format!(
                    "lora adapter id {id} is out of range: {} adapter(s) loaded",
                    self.lora_adapters.len()
                ));
            }
        }
        for (id, a) in self.lora_adapters.iter().enumerate() {
            let s = scales
                .iter()
                .rev()
                .find(|(i, _)| *i == id)
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            a.scale.set(s);
        }
        Ok(())
    }

    /// The current scale of every adapter, by id.
    pub fn lora_scales(&self) -> Vec<f32> {
        self.lora_adapters.iter().map(LoraAttached::scale).collect()
    }

    /// The matrices holding base tensor `name`'s rows, and how.
    fn parts_mut(
        &mut self,
        target: &Target,
        path: &Path,
        name: &str,
        file_rows: usize,
    ) -> Result<Parts<'_>, LoraError> {
        let no_projection = || LoraError::NoProjection {
            path: path.to_path_buf(),
            name: name.to_string(),
        };
        let ungated = self.config.ffn_is_ungated();
        Ok(match target {
            Target::Embedding => Parts::Split(vec![&mut self.embedding]),
            Target::Output => Parts::Split(vec![&mut self.output_head]),
            Target::Layer(il, kind) => {
                let layer = self.layers.get_mut(*il).ok_or_else(no_projection)?;
                match kind {
                    LayerTarget::Q => Parts::Split(vec![&mut layer.attn.q_proj]),
                    LayerTarget::K => Parts::Split(vec![&mut layer.attn.k_proj]),
                    LayerTarget::V => Parts::Split(vec![&mut layer.attn.v_proj]),
                    LayerTarget::Qkv => Parts::Split(vec![
                        &mut layer.attn.q_proj,
                        &mut layer.attn.k_proj,
                        &mut layer.attn.v_proj,
                    ]),
                    LayerTarget::O => Parts::Split(vec![&mut layer.attn.o_proj]),
                    LayerTarget::AttnGate => match layer.attn.output_gate.as_mut() {
                        Some(g) => Parts::Split(vec![&mut g.proj]),
                        None => return Err(no_projection()),
                    },
                    LayerTarget::FfnGate | LayerTarget::FfnUp | LayerTarget::FfnDown => {
                        let ex = match &mut layer.moe.experts {
                            ExpertBacking::Resident(v) if v.len() == 1 => &mut v[0],
                            ExpertBacking::Resident(_) => return Err(no_projection()),
                            ExpertBacking::Stored { .. } => {
                                return Err(LoraError::StoredExperts {
                                    path: path.to_path_buf(),
                                    name: name.to_string(),
                                })
                            }
                        };
                        match kind {
                            LayerTarget::FfnGate => Parts::Split(vec![&mut ex.gate]),
                            LayerTarget::FfnDown => Parts::Split(vec![&mut ex.down]),
                            // `ffn_up` is three things across the loader:
                            // the up matrix; Phi-3's fused gate+up (first
                            // half gate, `loader/layer_parts.rs:208`); or the ungated
                            // FFN's up, which is ALSO its gate.
                            _ if ungated => Parts::Alias(vec![&mut ex.up, &mut ex.gate]),
                            _ if file_rows == ex.gate.rows() + ex.up.rows()
                                && file_rows != ex.up.rows() =>
                            {
                                Parts::Split(vec![&mut ex.gate, &mut ex.up])
                            }
                            _ => Parts::Split(vec![&mut ex.up]),
                        }
                    }
                    LayerTarget::ShexpGate | LayerTarget::ShexpUp | LayerTarget::ShexpDown => {
                        let sh = layer
                            .moe
                            .shared_experts
                            .first_mut()
                            .ok_or_else(no_projection)?;
                        Parts::Split(vec![match kind {
                            LayerTarget::ShexpGate => &mut sh.gate,
                            LayerTarget::ShexpUp => &mut sh.up,
                            _ => &mut sh.down,
                        }])
                    }
                    LayerTarget::Router => Parts::Split(vec![&mut layer.moe.router]),
                }
            }
        })
    }
}

/// A pair checked against the base tensor and brought into frink's one
/// layout: `a` is `[rank][cols]`, `b` is `[rows][rank]`.
struct CheckedPair {
    a: Vec<f32>,
    b: Vec<f32>,
    rank: usize,
}

/// The shape checks of `llama-adapter.cpp:354-367`.
fn checked_pair(
    path: &Path,
    name: &str,
    pair: &LoraPair,
    target: &Target,
    rows: usize,
    cols: usize,
) -> Result<CheckedPair, LoraError> {
    let shape_err = || LoraError::Shape {
        path: path.to_path_buf(),
        name: name.to_string(),
        rows,
        cols,
        a: pair.a.shape,
        b: pair.b.shape,
    };
    if matches!(target, Target::Embedding) {
        // `:355-359`: B is `[n_embd, rank]`, A is `[n_vocab, rank]`
        // (flipped and transposed by the converter), and the graph
        // gathers a row of A per token and multiplies by B. In this
        // module's one layout that row gather IS `lora_b`'s role and
        // B transposed is `lora_a`'s.
        if cols != pair.b.rows() || rows != pair.a.rows() {
            return Err(shape_err());
        }
        let rank = pair.a.cols();
        if rank != pair.b.cols() {
            return Err(shape_err());
        }
        let n_embd = cols;
        let mut a_eff = vec![0f32; rank * n_embd];
        for e in 0..n_embd {
            for k in 0..rank {
                a_eff[k * n_embd + e] = pair.b.data[e * rank + k];
            }
        }
        return Ok(CheckedPair {
            a: a_eff,
            b: pair.a.data.clone(),
            rank,
        });
    }
    // `:361-363`: `ne[0]` (cols) against A's, `ne[1]` (rows) against B's.
    if cols != pair.a.cols() || rows != pair.b.rows() {
        return Err(shape_err());
    }
    // `:364-366`: A's rank against B's.
    let rank = pair.a.rows();
    if rank != pair.b.cols() {
        return Err(LoraError::NotTransposed {
            path: path.to_path_buf(),
            name: name.to_string(),
            a: pair.a.shape,
            b: pair.b.shape,
        });
    }
    Ok(CheckedPair {
        a: pair.a.data.clone(),
        b: pair.b.data.clone(),
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_specs_split_on_the_last_colon() {
        let s = LoraSpec::parse_scaled("a/b.gguf:0.5").unwrap();
        assert_eq!(s.path, PathBuf::from("a/b.gguf"));
        assert_eq!(s.scale, 0.5);
        let s = LoraSpec::parse_scaled("C:/x/y.gguf:2").unwrap();
        assert_eq!(s.path, PathBuf::from("C:/x/y.gguf"));
        assert_eq!(s.scale, 2.0);
        let s = LoraSpec::parse_scaled("z.gguf:-1").unwrap();
        assert_eq!(s.scale, -1.0);
        assert!(LoraSpec::parse_scaled("z.gguf")
            .unwrap_err()
            .contains("FNAME:SCALE"));
        assert!(LoraSpec::parse_scaled("z.gguf:abc")
            .unwrap_err()
            .contains("not a number"));
        assert!(LoraSpec::parse_scaled(":0.5").is_err());
    }

    #[test]
    fn the_two_flags_become_one_ordered_list() {
        let specs = LoraSpec::from_flags(
            &["a.gguf,b.gguf".to_string(), "c.gguf".to_string()],
            &["d.gguf:0.25".to_string()],
        )
        .unwrap();
        assert_eq!(
            specs,
            vec![
                LoraSpec::plain("a.gguf"),
                LoraSpec::plain("b.gguf"),
                LoraSpec::plain("c.gguf"),
                LoraSpec {
                    path: "d.gguf".into(),
                    scale: 0.25
                },
            ]
        );
        assert!(LoraSpec::from_flags(&[], &["x".to_string()]).is_err());
    }
}
