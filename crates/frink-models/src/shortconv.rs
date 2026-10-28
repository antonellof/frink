//! LFM2's short convolution: the recurrent block that stands where
//! attention stands on a layer whose `head_count_kv` is zero.
//!
//! `lfm2.cpp:9-11` marks layer `il` recurrent when `n_head_kv(il) == 0`
//! (`conversion/lfm2.py:37-40` writes the array with a `0` on every
//! `conv` layer), and `:192-208` runs ONE residual topology for both
//! kinds: `attn_norm`, then the short convolution or attention, the
//! residual add, `ffn_norm`, the FFN. So the block is a third answer to
//! "what is this layer's attention" ([`crate::layer_shapes::AttnShape::
//! ShortConv`]) beside deci's two, and NOT a second engine: the loader,
//! the three host bodies and the KV plumbing are the generic path's,
//! with one arm each.
//!
//! # The op (`lfm2.cpp:139-189`)
//!
//! ```text
//! bcx    = in_proj(normed)                 {3 n_embd}       :151
//! b, c, x = bcx split in three             {n_embd} each    :156-160
//! bx     = b * x                                            :162
//! y[t]   = sum_{i < L} conv[i] * bx[t - (L - 1) + i]        :164-186, ggml_ssm_conv
//! out    = out_proj(c * y)                                  :187-188
//! ```
//!
//! `L` is `{arch}.shortconv.l_cache` (3 on every LFM2 export), the
//! conv kernel is `{L, n_embd}` -- one `L`-tap filter per channel, tap
//! `L - 1` on the current token (`ggml-cpu/ops.cpp:9601-9603` dots the
//! window against the kernel in order, the window's LAST entry being
//! the newest input) -- and the state is the previous `L - 1` inputs,
//! zero before the sequence starts (`build_rs` zeroes a new sequence's
//! state, and `:172` prepends it).
//!
//! # Where the state lives
//!
//! In the layer's [`frink_core::KvCache`], as ONE "KV head" of width
//! `n_embd` with an empty V ([`crate::layer_shapes::AttnShape::
//! cache_geometry`]): every row is one token's `bx`, and the conv reads
//! the last `L` of them. llama.cpp keeps only the `L - 1` it needs
//! (`n_embd_r()`, `llama-hparams.cpp:189-192`); frink keeps the whole
//! history because that is what every consumer of a per-layer cache
//! here assumes it can do -- truncate to a position, snapshot, share a
//! prefix, page -- and a conv layer whose state was not a history would
//! have been the one layer none of them could serve. The cost is
//! `n_embd` floats per token per conv layer, beside an attention
//! layer's `2 * n_kv_heads * head_dim`.
//!
//! # Reach
//!
//! `grep -l shortconv src/models/*.cpp` over all 155 llama.cpp graphs:
//! `lfm2.cpp` and `lfm2moe.cpp`, and `models.h:1899` gives the second
//! the first's graph (`using graph = llama_model_lfm2::graph`). At the
//! `604cc430` pin the block moved into `lfm-base.cpp`
//! (`llm_build_lfm_base::build_lfm_shortconv`) and a third graph,
//! `lfm3.cpp`, calls it (`:165-171`). So [`SHORTCONV_ARCHITECTURES`] is
//! three rows, and what `head_count_kv 0` means on the OTHER hybrid rows
//! (a Mamba-2 block, a gated delta-net block, a KDA block) is recorded
//! in `layer_shapes::ZeroKvLayer` and refused by name until a body
//! exists for it.
//!
//! # Windows
//!
//! The two generations read the window differently, and
//! [`shortconv_window_refusal`] is the one place that says how:
//!
//! - `lfm2.cpp:24-29` honours a scalar `attention.sliding_window` on the
//!   attention layers ALONE (`is_swa_impl[il] = !is_recr_impl[il]`),
//!   which `crate::swa_layers` has no variant for, and no published LFM2
//!   export writes the key (the converter does not; measured over the
//!   four sizes' configs). REFUSED. The fixture that evidences the
//!   refusal runs in libllama and its logits differ from the unwindowed
//!   file's.
//! - `lfm3.cpp:13-18` reads the per-layer ARRAY with `get_arr`, which
//!   `crate::swa_layers` honours, and REQUIRES at least one sliding
//!   layer (`:14-16`). Its converter marks `sliding_attention` layers
//!   only (`conversion/lfm3.py:95`), never a conv layer. Served, with
//!   two refusals: a file with no sliding layer (llama.cpp throws), and
//!   one that marks a CONV layer as sliding, which upstream ignores (no
//!   conv code reads `is_swa`) but which here would arm `FRINK_KV_
//!   WINDOW`'s eviction against the history the conv indexes by row.
//!   No converter writes the second; refusing it is cheaper than a
//!   mask that only matters for a hand-written file.
//!
//! # Post norms on the block
//!
//! `lfm3.cpp:178-181` norms the branch output with `attn_post_norm`
//! OUTSIDE the `is_recr` branch, so a conv layer's output is normed as
//! an attention layer's is ([`CONV_OUTPUT_POST_NORMED`]); `lfm2.cpp`
//! creates no such tensor.

use frink_core::weight_matrix::WeightMatrix;
use frink_gguf::TensorSource;

use crate::loader::{load_f32_vec, load_weight_matrix, LoadError};

/// The two graphs that build the block, with the lines that decide
/// which layers run it.
pub const SHORTCONV_ARCHITECTURES: &[(&str, &str)] = &[
    (
        "lfm2",
        "lfm2.cpp:9-11 (is_recr), :80-82 (tensors), :139-189 (block)",
    ),
    (
        "lfm2moe",
        "lfm2moe.cpp:12-14, :66-69; the graph is lfm2's (models.h:1899)",
    ),
    (
        "lfm3",
        "lfm3.cpp:8-10 (is_recr), :84-86 (tensors), :165-171 (lfm-base.cpp:9-112, the block)",
    ),
];

/// The short-conv graphs that norm the conv block's output with
/// `attn_post_norm` before the residual add (`lfm3.cpp:178-181`, outside
/// the `is_recr` branch). See the module doc.
pub const CONV_OUTPUT_POST_NORMED: &[&str] = &["lfm3"];

/// Why a short-conv architecture's window cannot be served, or `None`
/// when it can (or the architecture is not a short-conv one). `window`
/// is the declared `attention.sliding_window`; `slides` and `kv_heads`
/// are the per-layer `is_swa` answer and `head_count_kv` (`0` is a conv
/// layer). See the module doc for the two generations.
pub fn shortconv_window_refusal(
    arch: &str,
    window: Option<usize>,
    n_layers: usize,
    slides: impl Fn(usize) -> bool,
    kv_heads: &[u64],
) -> Option<String> {
    if !is_shortconv_architecture(arch) {
        return None;
    }
    let reads_array = matches!(
        crate::swa_layers::pattern_key_read(arch),
        crate::swa_layers::PatternKeyRead::PerLayerBool
    );
    match (window, reads_array) {
        (None, _) => None,
        (Some(w), false) => Some(format!(
            "`{arch}.attention.sliding_window` {w}: lfm2.cpp:24-29 windows the attention \
             layers and not the conv layers, which `swa_layers` cannot yet spell, and a \
             window on a conv layer would evict the history its convolution reads \
             (`crate::shortconv`); no published LFM2 export writes the key"
        )),
        (Some(_), true) => {
            let is_conv = |il: usize| kv_heads.get(il).copied() == Some(0);
            if let Some(il) = (0..n_layers).find(|&il| is_conv(il) && slides(il)) {
                return Some(format!(
                    "`{arch}.attention.sliding_window_pattern` marks conv layer {il} as \
                     sliding: llama.cpp's conv block never reads `is_swa` (lfm-base.cpp:9-112) \
                     and its converter never writes it (conversion/lfm3.py:95), while here a \
                     window on a conv layer would evict the history its convolution reads"
                ));
            }
            if !(0..n_layers).any(&slides) {
                return Some(format!(
                    "`{arch}.attention.sliding_window_pattern` marks no layer as sliding, and \
                     llama.cpp refuses that file (lfm3.cpp:14-16, \"LFM3 requires sliding \
                     window attention layers\")"
                ));
            }
            None
        }
    }
}

/// True for an architecture whose zero-KV layers are short convolutions.
pub fn is_shortconv_architecture(arch: &str) -> bool {
    SHORTCONV_ARCHITECTURES.iter().any(|(a, _)| *a == arch)
}

/// One conv layer's weights.
pub struct ShortConv {
    /// `blk.N.shortconv.conv.weight`, `[n_embd][l_cache]`: channel `c`'s
    /// taps, oldest first.
    pub conv: Vec<f32>,
    /// `{arch}.shortconv.l_cache`, the conv's width.
    pub l_cache: usize,
    /// `blk.N.shortconv.in_proj.weight`, `[3 n_embd, n_embd]`.
    pub in_proj: WeightMatrix,
    /// `blk.N.shortconv.out_proj.weight`, `[n_embd, n_embd]`.
    pub out_proj: WeightMatrix,
}

impl ShortConv {
    /// Loads layer `layer`'s three tensors and checks them against
    /// `lfm2.cpp:80-82`'s shapes.
    pub fn load(
        file: &impl TensorSource,
        arch: &str,
        layer: usize,
        hidden_dim: usize,
    ) -> Result<Self, LoadError> {
        let key = format!("{arch}.shortconv.l_cache");
        let l_cache = file
            .metadata_u64(&key)
            .ok_or_else(|| LoadError::MissingHparam(key.clone()))? as usize;
        // :171 `GGML_ASSERT(hparams.n_shortconv_l_cache > 1)`.
        if l_cache < 2 {
            return Err(LoadError::UnsupportedFeature(
                key,
                format!("{l_cache}: lfm2.cpp:171 asserts a conv of width at least 2"),
            ));
        }
        let conv_name = format!("blk.{layer}.shortconv.conv.weight");
        let conv = load_f32_vec(file, &conv_name)?;
        if conv.len() != l_cache * hidden_dim {
            return Err(LoadError::UnsupportedFeature(
                conv_name,
                format!(
                    "{} elements; lfm2.cpp:80 sizes the kernel {{l_cache {l_cache}, n_embd \
                     {hidden_dim}}}",
                    conv.len()
                ),
            ));
        }
        let in_proj = load_weight_matrix(file, &format!("blk.{layer}.shortconv.in_proj.weight"))?;
        let out_proj = load_weight_matrix(file, &format!("blk.{layer}.shortconv.out_proj.weight"))?;
        for (name, m, rows) in [
            ("in_proj", &in_proj, 3 * hidden_dim),
            ("out_proj", &out_proj, hidden_dim),
        ] {
            if m.rows() != rows || m.cols() != hidden_dim {
                return Err(LoadError::UnsupportedFeature(
                    format!("blk.{layer}.shortconv.{name}.weight"),
                    format!(
                        "{}x{}; lfm2.cpp:81-82 size it {rows}x{hidden_dim}",
                        m.rows(),
                        m.cols()
                    ),
                ));
            }
        }
        Ok(Self {
            conv,
            l_cache,
            in_proj,
            out_proj,
        })
    }

    /// The channel count.
    pub fn hidden_dim(&self) -> usize {
        self.out_proj.rows()
    }

    /// `rows` consecutive tokens of ONE sequence (`normed` is
    /// `[rows][n_embd]`, the `attn_norm` output), through the block.
    ///
    /// `history` is the state: called once per row in order with that
    /// row's `bx`, it appends it to the sequence's cache and returns the
    /// window the conv reads -- the last `l_cache` inputs INCLUDING the
    /// one just appended, oldest first, zero-padded at the front while
    /// the sequence is shorter than the conv. The three cache backings
    /// each spell that closure once (`Decoder::shortconv_block`); this
    /// body never sees which.
    pub fn forward_rows(
        &self,
        normed: &[f32],
        rows: usize,
        mut history: impl FnMut(&[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        let n = self.hidden_dim();
        assert_eq!(normed.len(), rows * n);
        let bcx = if rows == 1 {
            self.in_proj.apply(normed)
        } else {
            self.in_proj.apply_batch(normed, rows)
        };
        let l = self.l_cache;
        let mut y = vec![0.0f32; rows * n];
        for r in 0..rows {
            let row = &bcx[r * 3 * n..(r + 1) * 3 * n];
            let (b, c, x) = (&row[..n], &row[n..2 * n], &row[2 * n..]);
            let bx: Vec<f32> = b.iter().zip(x).map(|(b, x)| b * x).collect();
            let window = history(&bx);
            assert_eq!(window.len(), l * n, "the window is l_cache rows of n_embd");
            // ops.cpp:9601-9603: a float accumulator, deliberately not
            // double ("not using ggml_vec_dot_f32, because its sum is
            // in double precision").
            let out = &mut y[r * n..(r + 1) * n];
            for ch in 0..n {
                let taps = &self.conv[ch * l..(ch + 1) * l];
                let mut acc = 0.0f32;
                for (i, tap) in taps.iter().enumerate() {
                    acc += window[i * n + ch] * tap;
                }
                out[ch] = c[ch] * acc;
            }
        }
        if rows == 1 {
            self.out_proj.apply(&y)
        } else {
            self.out_proj.apply_batch(&y, rows)
        }
    }
}

/// The conv's window from a history of `bx` rows: the last `l_cache`
/// rows of `rows` (each `n_embd` wide), oldest first, zero-padded at
/// the front when fewer exist. `row_at(i)` is row `i` of the history,
/// `n_rows` how many there are AFTER the current token's push.
///
/// One function for the contiguous cache (`KvCache::k`, rows adjacent)
/// and the paged store (rows behind a block table), so the padding
/// rule cannot differ between them.
pub fn window_from_history<'a>(
    l_cache: usize,
    n_embd: usize,
    n_rows: usize,
    row_at: impl Fn(usize) -> &'a [f32],
) -> Vec<f32> {
    let mut window = vec![0.0f32; l_cache * n_embd];
    for i in 0..l_cache {
        // Window slot `i` is history row `n_rows - l_cache + i`.
        let Some(row) = (n_rows + i).checked_sub(l_cache) else {
            continue;
        };
        window[i * n_embd..(i + 1) * n_embd].copy_from_slice(row_at(row));
    }
    window
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_core::Tensor;

    fn identity(n: usize) -> WeightMatrix {
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        WeightMatrix::F32(Tensor::new(v, vec![n, n]))
    }

    /// The window rule: three rows of history, `l_cache 3`, so the
    /// first token sees two zero rows and itself, the third sees all
    /// three.
    #[test]
    fn window_is_zero_padded_then_the_last_l_rows() {
        let hist: Vec<Vec<f32>> = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let w = window_from_history(3, 2, 1, |i| &hist[i]);
        assert_eq!(w, vec![0.0, 0.0, 0.0, 0.0, 1.0, 10.0]);
        let w = window_from_history(3, 2, 3, |i| &hist[i]);
        assert_eq!(w, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        // A history longer than the conv: only the tail is read.
        let w = window_from_history(2, 2, 3, |i| &hist[i]);
        assert_eq!(w, vec![2.0, 20.0, 3.0, 30.0]);
    }

    /// The op against a hand computation: identity projections, one
    /// channel of the two carrying a filter `[1, 2, 3]`, so `y[t] =
    /// 3 bx[t] + 2 bx[t-1] + bx[t-2]`, scaled by `c`, and the newest
    /// input is on the LAST tap (ops.cpp:9601-9603).
    #[test]
    fn newest_input_is_on_the_last_tap() {
        let n = 2;
        // in_proj is 3n x n: b = x_in, c = x_in, x = x_in (three stacked
        // identities), so bx = x_in^2 and the output is c * y = x_in * y.
        let mut in_v = vec![0.0f32; 3 * n * n];
        for blk in 0..3 {
            for i in 0..n {
                in_v[(blk * n + i) * n + i] = 1.0;
            }
        }
        let sc = ShortConv {
            conv: vec![1.0, 2.0, 3.0, 0.0, 0.0, 1.0],
            l_cache: 3,
            in_proj: WeightMatrix::F32(Tensor::new(in_v, vec![3 * n, n])),
            out_proj: identity(n),
        };
        let mut hist: Vec<Vec<f32>> = Vec::new();
        let inputs = [[1.0f32, 1.0], [2.0, 1.0], [1.0, 1.0]];
        let flat: Vec<f32> = inputs.concat();
        let out = sc.forward_rows(&flat, 3, |bx| {
            hist.push(bx.to_vec());
            window_from_history(3, n, hist.len(), |i| &hist[i])
        });
        // bx = [1,1], [4,1], [1,1]. Channel 0: y = 3*bx[t] + 2*bx[t-1] + bx[t-2].
        // t0: 3;  t1: 12 + 2 = 14;  t2: 3 + 8 + 1 = 12. Times c (= x_in ch 0).
        // Channel 1: y = bx[t] = 1, times c = 1.
        assert_eq!(out, vec![3.0, 1.0, 28.0, 1.0, 12.0, 1.0]);
        // The same three tokens one at a time agree with the batch.
        let mut hist2: Vec<Vec<f32>> = Vec::new();
        let mut one_at_a_time = Vec::new();
        for row in inputs {
            one_at_a_time.extend(sc.forward_rows(&row, 1, |bx| {
                hist2.push(bx.to_vec());
                window_from_history(3, n, hist2.len(), |i| &hist2[i])
            }));
        }
        assert_eq!(one_at_a_time, out);
    }

    #[test]
    fn the_table_is_the_three_lfm_graphs() {
        assert!(is_shortconv_architecture("lfm2"));
        assert!(is_shortconv_architecture("lfm2moe"));
        assert!(is_shortconv_architecture("lfm3"));
        assert!(!is_shortconv_architecture("deci"));
        assert!(!is_shortconv_architecture("jamba"));
    }

    /// The two generations: lfm2's scalar window is refused whatever
    /// it says, lfm3's array is served unless it marks a conv layer or
    /// no layer at all, and a windowless file is never refused.
    #[test]
    fn each_generation_reads_its_window_as_its_graph_does() {
        let kv = [0u64, 2, 2, 0, 2];
        let slides = |p: &'static [bool]| move |il: usize| p[il];
        assert!(shortconv_window_refusal("lfm2", Some(4), 5, |_| true, &kv)
            .is_some_and(|m| m.contains("lfm2.cpp:24-29")));
        assert_eq!(
            shortconv_window_refusal("lfm2", None, 5, |_| true, &kv),
            None
        );
        const SERVED: [bool; 5] = [false, true, false, false, true];
        assert_eq!(
            shortconv_window_refusal("lfm3", Some(3), 5, slides(&SERVED), &kv),
            None
        );
        const ON_CONV: [bool; 5] = [true, true, false, false, true];
        assert!(
            shortconv_window_refusal("lfm3", Some(3), 5, slides(&ON_CONV), &kv)
                .is_some_and(|m| m.contains("conv layer 0"))
        );
        assert!(shortconv_window_refusal("lfm3", Some(3), 5, |_| false, &kv)
            .is_some_and(|m| m.contains("lfm3.cpp:14-16")));
        // Not a short-conv graph: nothing to say.
        assert_eq!(
            shortconv_window_refusal("llama", Some(3), 5, |_| true, &kv),
            None
        );
    }
}
