//! What a captured decode token froze into its graph, as words.
//!
//! A graph's kernel arguments are the values the capture saw: every
//! weight's device address, every shape, every scalar, the mirrors'
//! addresses and capacities. A token may replay a graph only when all of
//! those are what it would launch with, so the key is all of them, and a
//! token whose key differs runs eagerly (and its successor captures
//! again). The words are built from exhaustive destructures of every
//! type a layer is described by (ADR 0004): a field added to any of them
//! does not compile until it is keyed here or named as read per token.
//!
//! A host slice is keyed by its address and length, which is what the
//! resident caches key its device copy by. The graph holds those device
//! copies alive (`super::DecodeGraph::held`), so a key that matches
//! never points the graph at freed memory; and a graph lives in its
//! sequence's KV mirror, which is rebuilt, graph and all, the moment
//! the host history stops being the one it mirrors, so the weights
//! behind a matching key are the ones the history was computed with.

use crate::gpu::MatvecLaunch;
use crate::prefill::{
    AttnExtrasCuda, DeviceGating, DeviceRouting, ExpertBiasCuda, ExpertProjCuda, GluCuda,
    LayerRopeCuda, MulMmWeights, PrefillDenseLayerCuda, PrefillParams, ProjWeights, RopeLayoutCuda,
    RouterCuda, StackFfn, StackInput,
};

/// The words of a key, in the order they were pushed.
#[derive(Default)]
pub(crate) struct Words(pub(crate) Vec<u64>);

impl Words {
    pub(crate) fn n(&mut self, v: usize) {
        self.0.push(v as u64);
    }

    fn f(&mut self, v: f32) {
        self.0.push(u64::from(v.to_bits()));
    }

    fn slice<T>(&mut self, s: &[T]) {
        self.n(s.as_ptr() as usize);
        self.n(s.len());
    }

    fn opt_slice<T>(&mut self, s: Option<&[T]>) {
        match s {
            None => self.n(0),
            Some(s) => {
                self.n(1);
                self.slice(s);
            }
        }
    }

    fn text(&mut self, s: &'static str) {
        self.slice(s.as_bytes());
    }

    fn mul_mm(&mut self, m: &MulMmWeights<'_>) {
        let MulMmWeights {
            kind,
            data,
            rows,
            cols,
            row_bytes,
        } = *m;
        self.n(std::ptr::from_ref(kind) as usize);
        self.slice(data);
        self.n(rows);
        self.n(cols);
        self.n(row_bytes);
    }

    fn matvec(&mut self, m: &MatvecLaunch<'_>) {
        let MatvecLaunch {
            kernel_src,
            module_name,
            fn_name,
            weights,
            rows,
            row_bytes,
            n_blocks_per_row,
        } = *m;
        self.text(kernel_src);
        self.text(module_name);
        self.text(fn_name);
        self.slice(weights);
        self.n(rows);
        self.n(row_bytes);
        self.n(n_blocks_per_row);
    }

    fn proj(&mut self, p: &ProjWeights<'_>) {
        let ProjWeights { gemm, matvec } = p;
        self.mul_mm(gemm);
        match matvec {
            None => self.n(0),
            Some(m) => {
                self.n(1);
                self.matvec(m);
            }
        }
    }

    fn expert(&mut self, e: &ExpertProjCuda<'_>) {
        let ExpertProjCuda {
            gate,
            up,
            down,
            bias,
        } = e;
        self.proj(gate);
        self.proj(up);
        self.proj(down);
        match bias {
            None => self.n(0),
            Some(ExpertBiasCuda { gate, up, down }) => {
                self.n(1);
                self.slice(gate);
                self.slice(up);
                self.slice(down);
            }
        }
    }

    fn device_routing(&mut self, d: &DeviceRouting<'_>) {
        let DeviceRouting {
            key,
            probe,
            n_experts,
            k,
            gating,
            router_bias,
            // Asked only for tables not yet cached; the graph holds the
            // tables its capture read.
            source: _,
            layer,
        } = *d;
        self.n(key);
        for p in probe {
            self.slice(p);
        }
        self.n(n_experts);
        self.n(k);
        match gating {
            DeviceGating::Softmax { norm } => {
                self.n(0);
                self.n(usize::from(norm));
            }
            DeviceGating::SoftmaxWeight => self.n(1),
        }
        self.opt_slice(router_bias);
        self.n(layer);
    }

    fn ffn(&mut self, f: &StackFfn<'_>) {
        match f {
            StackFfn::Dense(e) => {
                self.n(0);
                self.expert(e);
            }
            StackFfn::Moe {
                router,
                ffn_dim,
                device,
            } => {
                self.n(1);
                match router {
                    RouterCuda::F32 { data, rows } => {
                        self.n(0);
                        self.slice(data);
                        self.n(*rows);
                    }
                    RouterCuda::Quant(p) => {
                        self.n(1);
                        self.proj(p);
                    }
                }
                self.n(*ffn_dim);
                match device {
                    None => self.n(0),
                    Some(d) => {
                        self.n(1);
                        self.device_routing(d);
                    }
                }
            }
        }
    }

    fn layer(&mut self, l: &PrefillDenseLayerCuda<'_>) {
        let PrefillDenseLayerCuda {
            attn_norm_w,
            ffn_norm_w,
            q,
            k,
            v,
            o,
            ffn,
            glu,
            post_attn_norm,
            post_ffn_norm,
            extras,
            rope,
        } = l;
        self.slice(attn_norm_w);
        self.slice(ffn_norm_w);
        for p in [q, k, v, o] {
            self.proj(p);
        }
        self.ffn(ffn);
        self.n(match glu {
            GluCuda::Silu => 0,
            GluCuda::Gelu => 1,
            GluCuda::SwigluOai => 2,
        });
        self.opt_slice(*post_attn_norm);
        self.opt_slice(*post_ffn_norm);
        let AttnExtrasCuda {
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
            sinks,
            o_bias,
        } = *extras;
        for v in [q_bias, k_bias, v_bias, q_norm, k_norm, sinks, o_bias] {
            self.opt_slice(v);
        }
        match rope {
            None => self.n(0),
            Some(LayerRopeCuda {
                theta,
                freq_factors,
                rot_dim,
                layout,
                mscale,
            }) => {
                self.n(1);
                self.f(*theta);
                self.opt_slice(*freq_factors);
                self.n(*rot_dim);
                self.n(match layout {
                    RopeLayoutCuda::Norm => 0,
                    RopeLayoutCuda::Neox => 1,
                });
                self.f(*mscale);
            }
        }
    }

    fn params(&mut self, p: &PrefillParams<'_>) {
        let PrefillParams {
            n_heads,
            n_kv_heads,
            head_dim,
            rms_eps,
            attn_scale,
            attn_softcap,
            window,
            // Per token: the history is read from the mirrors, and the
            // position from the word each replay writes.
            prefix_k: _,
            prefix_v: _,
            start_pos: _,
        } = *p;
        self.n(n_heads);
        self.n(n_kv_heads);
        self.n(head_dim);
        self.f(rms_eps);
        self.f(attn_scale);
        match attn_softcap {
            None => self.n(0),
            Some(c) => {
                self.n(1);
                self.f(c);
            }
        }
        self.n(window.map_or(0, |w| w + 1));
    }

    /// The input's kind and shape (not its values, which every replay
    /// uploads), then every layer of the run.
    pub(crate) fn stack(
        &mut self,
        input: &StackInput<'_>,
        layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    ) {
        match input {
            StackInput::Hidden(h) => {
                self.n(0);
                self.n(h.len());
            }
            StackInput::Gather { table, ids } => {
                self.n(1);
                self.mul_mm(table);
                self.n(ids.len());
            }
        }
        self.n(layers.len());
        for (layer, params) in layers {
            self.layer(layer);
            self.params(params);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefill::tests::{q8_0_proj, small_q8_0};

    fn key(
        input: &StackInput<'_>,
        layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    ) -> Vec<u64> {
        let mut w = Words::default();
        w.stack(input, layers);
        w.0
    }

    /// The key is what a capture froze, and only that: two tokens of one
    /// run at different positions over different host histories share
    /// it, and changing any one frozen value changes it.
    #[test]
    fn a_key_is_every_frozen_value_and_no_per_token_one() {
        let (hidden, ffn) = (64usize, 96usize);
        let w: Vec<&'static [u8]> = (0..7).map(|s| small_q8_0(64, 64, 40 + s)).collect();
        let up = small_q8_0(ffn, hidden, 50);
        let gate = small_q8_0(ffn, hidden, 51);
        let down = small_q8_0(hidden, ffn, 52);
        let leak = |v: Vec<f32>| -> &'static [f32] { Box::leak(v.into_boxed_slice()) };
        let norm = leak(vec![1.0f32; hidden]);
        let other_norm = leak(vec![1.0f32; hidden]);
        let ff = leak(vec![1.0f32; 16]);
        let layer = PrefillDenseLayerCuda {
            attn_norm_w: norm,
            ffn_norm_w: norm,
            q: q8_0_proj(w[0], 64, hidden),
            k: q8_0_proj(w[1], 64, hidden),
            v: q8_0_proj(w[2], 64, hidden),
            o: q8_0_proj(w[3], hidden, 64),
            ffn: StackFfn::Dense(ExpertProjCuda {
                gate: q8_0_proj(gate, ffn, hidden),
                up: q8_0_proj(up, ffn, hidden),
                down: q8_0_proj(down, hidden, ffn),
                bias: None,
            }),
            glu: GluCuda::Silu,
            post_attn_norm: None,
            post_ffn_norm: None,
            extras: AttnExtrasCuda::default(),
            rope: Some(LayerRopeCuda {
                theta: 10000.0,
                freq_factors: Some(ff),
                rot_dim: 32,
                layout: RopeLayoutCuda::Norm,
                mscale: 1.0,
            }),
        };
        let hist_a = leak(vec![0.5f32; 70 * 32]);
        let hist_b = leak(vec![0.25f32; 71 * 32]);
        let params = |pk: &'static [f32], pos: usize| PrefillParams {
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            rms_eps: 1e-5,
            attn_scale: 0.125,
            attn_softcap: None,
            window: None,
            prefix_k: pk,
            prefix_v: pk,
            start_pos: pos,
        };
        let (pa, pb) = (params(hist_a, 70), params(hist_b, 71));
        let row_a = vec![0.1f32; hidden];
        let row_b = vec![0.9f32; hidden];
        let base = key(&StackInput::Hidden(&row_a), &[(&layer, &pa)]);
        // Another position, history and hidden row: the same graph.
        assert_eq!(base, key(&StackInput::Hidden(&row_b), &[(&layer, &pb)]));

        let differs = |what: &str, k: Vec<u64>| assert_ne!(base, k, "{what} is frozen");
        differs(
            "a window",
            key(
                &StackInput::Hidden(&row_a),
                &[(
                    &layer,
                    &PrefillParams {
                        window: Some(100),
                        ..pa
                    },
                )],
            ),
        );
        differs(
            "the softcap",
            key(
                &StackInput::Hidden(&row_a),
                &[(
                    &layer,
                    &PrefillParams {
                        attn_softcap: Some(30.0),
                        ..pa
                    },
                )],
            ),
        );
        let with = |f: &dyn Fn(&mut PrefillDenseLayerCuda<'static>)| {
            let mut l = PrefillDenseLayerCuda { ..layer };
            f(&mut l);
            key(&StackInput::Hidden(&row_a), &[(&l, &pa)])
        };
        differs("a norm's buffer", with(&|l| l.ffn_norm_w = other_norm));
        differs("a projection", with(&|l| l.q = q8_0_proj(w[4], 64, hidden)));
        differs("the activation", with(&|l| l.glu = GluCuda::Gelu));
        differs(
            "a sink",
            with(&|l| {
                l.extras = AttnExtrasCuda {
                    sinks: Some(&ff[..2]),
                    ..AttnExtrasCuda::default()
                }
            }),
        );
        differs("the rotation", with(&|l| l.rope = None));
        differs(
            "the rotation's layout",
            with(&|l| {
                l.rope = Some(LayerRopeCuda {
                    layout: RopeLayoutCuda::Neox,
                    ..l.rope.unwrap()
                })
            }),
        );
        differs("one layer more", {
            key(&StackInput::Hidden(&row_a), &[(&layer, &pa), (&layer, &pa)])
        });
        let table = MulMmWeights {
            kind: &crate::mul_mm::Q8_0,
            data: w[5],
            rows: 64,
            cols: hidden,
            row_bytes: hidden / 32 * 34,
        };
        differs(
            "the input's kind",
            key(
                &StackInput::Gather {
                    table,
                    ids: &[3u32],
                },
                &[(&layer, &pa)],
            ),
        );
        // A gather of another id is the same graph.
        assert_eq!(
            key(
                &StackInput::Gather {
                    table,
                    ids: &[3u32]
                },
                &[(&layer, &pa)]
            ),
            key(
                &StackInput::Gather {
                    table,
                    ids: &[9u32]
                },
                &[(&layer, &pb)]
            ),
        );
    }
}
