//! The id kernels against a CPU MoE FFN built from `frink_quant`'s
//! dequant, on real hardware (`--ignored`), through the launch the
//! decoder's prefill calls (`gpu::launch_moe_prefill_q4_0`). It takes
//! `mul_mv_id` when no expert gathers `gpu::MOE_MM_ID_TOKEN_MIN` tokens
//! and the indexed GEMM `mul_mm_id` otherwise, so each kind is run both
//! ways, and the test asserts which one each case took.

use super::*;
use crate::gpu::{launch_moe_prefill_q4_0, mul_mm_id_meta, MoePackedQ4};

/// Every entry point [`dispatch`] names is generated in the source it
/// names, from the `MOE_MV_ID_ENTRIES` line for its prefix. Runs
/// without a device.
#[test]
fn every_dispatch_names_an_entry_its_source_generates() {
    for kind in KINDS {
        for down in [false, true] {
            let d = dispatch(kind, down).unwrap_or_else(|| panic!("{kind}"));
            let (src, prefix) = kind_entry(kind).unwrap();
            assert!(
                src.contains(&format!("MOE_MV_ID_ENTRIES({prefix}, ")),
                "{kind}: the matvec unit generates no id entries"
            );
            let suffix = if down {
                "_moe_down_id"
            } else {
                "_moe_matvec_id"
            };
            assert_eq!(d.fn_name, format!("{prefix}{suffix}"), "{kind}");
            assert!(src.contains("#define MOE_MV_ID_ENTRIES"), "{kind}");
        }
    }
    assert!(dispatch("Q4_1", false).is_none());
}

/// Each of the four has both id paths, so a pack of it never falls
/// back to one matvec per expert. Runs without a device.
#[test]
fn every_kind_has_both_id_paths() {
    for kind in KINDS {
        assert!(
            crate::gpu::moe_packed_mul_mv_id_supported(kind, kind, kind),
            "{kind}: no mul_mv_id"
        );
        assert!(mul_mm_id_meta(kind).is_some(), "{kind}: no mul_mm_id");
        assert!(
            crate::gpu::mul_mm_id_f16_meta(kind).is_some_and(|m| m.0.ends_with("_f16")),
            "{kind}: no f16-activation mul_mm_id"
        );
    }
}

/// The decode stack's refusal names the first plane with no id kernel,
/// so a pack it cannot run is refused before encoding (#448), and a
/// pack of these kinds is not. Runs without a device.
#[test]
fn the_decode_stack_refuses_exactly_the_packs_with_no_id_kernel() {
    use crate::gpu::moe_packed_mul_mv_id_missing as missing;
    for kind in KINDS {
        assert_eq!(missing(kind, kind, kind), None, "{kind}");
    }
    assert_eq!(missing("Q4_1", "Q4_0", "Q4_0"), Some("Q4_1"));
    assert_eq!(missing("Q4_0", "Q4_1", "Q4_0"), Some("Q4_1"));
    assert_eq!(missing("Q4_0", "Q4_0", "PTQ1_0"), Some("PTQ1_0"));
}

/// `(block bytes, block elems, CPU dequant)`.
type Kind = (
    usize,
    usize,
    fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>,
);

fn kind(name: &str) -> Kind {
    match name {
        "Q2_K" => (84, 256, frink_quant::dequant_q2_k),
        "Q3_K" => (110, 256, frink_quant::dequant_q3_k),
        "IQ4_NL" => (18, 32, frink_quant::dequant_iq4_nl),
        "MXFP4" => (17, 32, frink_quant::dequant_mxfp4_gguf),
        "Q5_K" => (176, 256, frink_quant::dequant_q5_k),
        "Q6_K" => (210, 256, frink_quant::dequant_q6_k),
        "IQ4_XS" => (136, 256, frink_quant::dequant_iq4_xs),
        other => panic!("{other}"),
    }
}

/// `n_blocks` blocks of pseudo-random bytes with each format's scale
/// fields forced to realistic magnitudes.
fn blocks(name: &str, n_blocks: usize, seed: u32) -> Vec<u8> {
    let (bb, _, _) = kind(name);
    let mut w = crate::k_quant_check::pseudo_bytes(seed, n_blocks * bb);
    for b in 0..n_blocks {
        let at = b * bb;
        let h = |v: f32| half::f16::from_f32(v).to_le_bytes();
        match name {
            "Q2_K" => {
                w[at + 80..at + 82].copy_from_slice(&h(0.010 + 0.002 * (b % 3) as f32));
                w[at + 82..at + 84].copy_from_slice(&h(0.004 + 0.001 * (b % 5) as f32));
            }
            "Q3_K" => w[at + 108..at + 110].copy_from_slice(&h(0.003 + 0.0005 * (b % 4) as f32)),
            "IQ4_NL" => w[at..at + 2].copy_from_slice(&h(0.004 + 0.001 * (b % 9) as f32)),
            // The three with 6-bit (or 6-bit-like) sub-block scales take
            // a smaller super-scale, so the activations stay inside the
            // `half` range `mul_mm_id` stages them in.
            "IQ4_XS" => w[at..at + 2].copy_from_slice(&h(1e-4 + 2e-5 * (b % 9) as f32)),
            "Q5_K" => {
                w[at..at + 2].copy_from_slice(&h(1e-4 + 2e-5 * (b % 3) as f32));
                w[at + 2..at + 4].copy_from_slice(&h(5e-5 + 1e-5 * (b % 5) as f32));
            }
            "Q6_K" => w[at + 208..at + 210].copy_from_slice(&h(1e-4 + 2e-5 * (b % 4) as f32)),
            "MXFP4" => w[at] = 118 + (b % 6) as u8,
            other => panic!("{other}"),
        }
    }
    w
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// One pack: `n_experts` experts of `gate`, `up` (`ffn x hidden`) and
/// `down` (`hidden x ffn`), all `name`.
struct Pack {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    n_experts: usize,
    ffn: usize,
    hidden: usize,
}

impl Pack {
    fn new(name: &str, n_experts: usize, ffn: usize, hidden: usize) -> Self {
        let (_, be, _) = kind(name);
        let gu_blocks = n_experts * ffn * hidden / be;
        Pack {
            gate: blocks(name, gu_blocks, 3),
            up: blocks(name, gu_blocks, 5),
            down: blocks(name, n_experts * hidden * ffn / be, 7),
            n_experts,
            ffn,
            hidden,
        }
    }

    fn view<'a>(&'a self, name: &'static str) -> MoePackedQ4<'a> {
        let (bb, be, _) = kind(name);
        let gate_row_bytes = self.hidden / be * bb;
        let down_row_bytes = self.ffn / be * bb;
        MoePackedQ4 {
            gate: &self.gate,
            up: &self.up,
            down: &self.down,
            gate_stride: self.ffn * gate_row_bytes,
            up_stride: self.ffn * gate_row_bytes,
            down_stride: self.hidden * down_row_bytes,
            n_experts: self.n_experts,
            ffn_rows: self.ffn,
            hidden_rows: self.hidden,
            gate_row_bytes,
            down_row_bytes,
            gate_kind: name,
            up_kind: name,
            down_kind: name,
        }
    }

    /// The routed SwiGLU FFN in f64 over the CPU dequant: per token,
    /// `sum_k route * down_e(silu(gate_e x) * up_e x)`.
    fn reference(
        &self,
        name: &str,
        x: &[f32],
        ids: &[i32],
        route: &[f32],
        top_k: usize,
    ) -> Vec<f64> {
        let (_, _, dequant) = kind(name);
        let (g, u, d) = (
            dequant(&self.gate).unwrap(),
            dequant(&self.up).unwrap(),
            dequant(&self.down).unwrap(),
        );
        let (h, f) = (self.hidden, self.ffn);
        let dot = |w: &[f32], v: &[f64]| -> f64 {
            w.iter().zip(v).map(|(&a, &b)| f64::from(a) * b).sum()
        };
        let mut out = Vec::new();
        for (t, xt) in x.chunks_exact(h).enumerate() {
            let xt: Vec<f64> = xt.iter().map(|&v| f64::from(v)).collect();
            let mut acc = vec![0f64; h];
            for k in 0..top_k {
                let e = ids[t * top_k + k] as usize;
                let act: Vec<f64> = (0..f)
                    .map(|r| {
                        let row = (e * f + r) * h;
                        silu(dot(&g[row..row + h], &xt)) * dot(&u[row..row + h], &xt)
                    })
                    .collect();
                for (r, a) in acc.iter_mut().enumerate() {
                    let row = (e * h + r) * f;
                    *a += f64::from(route[t * top_k + k]) * dot(&d[row..row + f], &act);
                }
            }
            out.extend(acc);
        }
        out
    }
}

/// Routing for `n_tokens`: slot `k` of token `t` is expert
/// `(t + 3k) % n_experts`, which reaches the first and the last expert,
/// or (`crowd`) expert 0 for every token's first slot and one of the
/// others for the rest, so expert 0 gathers `n_tokens` and the launch
/// takes `mul_mm_id`. No token picks one expert twice.
fn routing(n_tokens: usize, top_k: usize, n_experts: usize, crowd: bool) -> (Vec<i32>, Vec<f32>) {
    let mut ids = Vec::new();
    let mut route = Vec::new();
    for t in 0..n_tokens {
        for k in 0..top_k {
            let e = match (crowd, k) {
                (true, 0) => 0,
                (true, _) => 1 + (t + k) % (n_experts - 1),
                (false, _) => (t + 3 * k + 1) % n_experts,
            };
            ids.push(e as i32);
            route.push(if k == 0 {
                0.7
            } else {
                0.3 / (top_k - 1) as f32
            });
        }
    }
    (ids, route)
}

/// Each kind, at `(experts, ffn, hidden)` of odd block counts, through
/// `mul_mv_id` (3 tokens, top-2, no expert past two tokens) and
/// `mul_mm_id` (11 tokens, one expert gathering all 11). The bound is a
/// fraction of the largest output: `mul_mv_id` is f32 throughout,
/// `mul_mm_id` stages weights and activations as `half` (as ggml's
/// does). Prints the worst error per kind and path.
#[test]
#[ignore = "needs a real Metal GPU"]
fn id_kernels_match_the_cpu_moe_ffn() {
    // 3 and 5 blocks a row; 6 experts. Every pack is built first and
    // lives to the end, so no two kinds' planes share an address; the
    // device-plane cache's handling of one that does (#449) is
    // `a_packed_plane_reused_at_the_same_address_is_rebuilt`'s.
    let n_experts = 6usize;
    let packs: Vec<(&str, usize, usize, Pack)> = KINDS
        .into_iter()
        .map(|name| {
            let (_, be, _) = kind(name);
            let (ffn, hidden) = (5 * be, 3 * be);
            (name, ffn, hidden, Pack::new(name, n_experts, ffn, hidden))
        })
        .collect();
    for (name, _, hidden, pack) in &packs {
        let (name, hidden) = (*name, *hidden);
        let packed = pack.view(name);
        for (n_tokens, crowd, path, rel) in [
            (3usize, false, "mul_mv_id", 1e-5),
            (11, true, "mul_mm_id", 2e-3),
        ] {
            let top_k = 2;
            let (ids, route) = routing(n_tokens, top_k, n_experts, crowd);
            let max_batch = (0..n_experts as i32)
                .map(|e| ids.iter().filter(|&&i| i == e).count())
                .max()
                .unwrap();
            assert_eq!(
                crate::gpu::moe_use_mm_id(max_batch, name, name, name),
                crowd,
                "{name}: the {path} case takes the other path ({max_batch} tokens on one expert, \
                 threshold {})",
                crate::gpu::MOE_MM_ID_TOKEN_MIN
            );
            let x: Vec<f32> = (0..n_tokens * hidden)
                .map(|i| ((i as f32) * 0.037).sin() * 0.5)
                .collect();
            let want = pack.reference(name, &x, &ids, &route, top_k);
            let got = launch_moe_prefill_q4_0(&x, n_tokens, &packed, &ids, &route, top_k)
                .unwrap_or_else(|e| panic!("{name} {path}: {e:?}"));
            assert_eq!(got.len(), want.len());
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-6);
            let mut worst = 0f64;
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let err = (f64::from(g) - w).abs();
                assert!(
                    err <= rel * scale,
                    "{name} {path} output {i}: metal {g} cpu {w} (|err| {err:e}, max |out| {scale:e})"
                );
                worst = worst.max(err);
            }
            eprintln!("{name} {path}: max |metal - cpu| = {worst:.3e} (max |out| {scale:.3e})");
        }
    }
}

/// The packed planes' device copy follows the host bytes, not the host
/// address (#449). The address reuse is forced rather than hoped for:
/// the planes are overwritten in place, which is what an allocator
/// handing a freed plane's address to the next one looks like to a
/// cache keyed on it.
#[test]
#[ignore = "needs a real Metal GPU"]
fn a_packed_plane_reused_at_the_same_address_is_rebuilt() {
    use crate::gpu::{moe_packed_resident, shared_metal};
    use objc2_metal::MTLBuffer;

    let device_bytes = |r: &crate::gpu::ResidentWeightBuffer, len: usize| -> Vec<u8> {
        // SAFETY: an owned plane's entry is a StorageModeShared buffer
        // holding a copy of exactly the `len` host bytes at
        // `weight_offset`, and no GPU work is in flight against it.
        unsafe {
            std::slice::from_raw_parts(
                (r.buffer.contents().as_ptr() as *const u8).add(r.weight_offset),
                len,
            )
            .to_vec()
        }
    };
    let device = &shared_metal().expect("metal device").device;
    let mut pack = Pack::new("IQ4_NL", 4, 64, 64);
    let first = pack.gate.clone();
    let bound = moe_packed_resident(device, &pack.view("IQ4_NL")).expect("first bind");
    assert_eq!(device_bytes(&bound.gate, first.len()), first);

    let ptr = pack.gate.as_ptr();
    for (dst, src) in pack
        .gate
        .iter_mut()
        .zip(blocks("IQ4_NL", first.len() / 18, 99))
    {
        *dst = src;
    }
    assert_eq!(pack.gate.as_ptr(), ptr, "the address is the one cached");
    assert_ne!(pack.gate, first);
    let again = moe_packed_resident(device, &pack.view("IQ4_NL")).expect("second bind");
    assert_eq!(
        device_bytes(&again.gate, pack.gate.len()),
        pack.gate,
        "the cache served the previous plane's bytes at a reused address"
    );
}
