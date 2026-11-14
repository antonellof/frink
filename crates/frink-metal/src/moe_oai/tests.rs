//! The gpt-oss FFN kernels against scalar transcriptions of ggml's
//! `ggml_compute_forward_swiglu_oai_f32` and `add_id`.

use super::*;
use crate::elem::tests::{alloc, read_f32, upload};
use crate::gpu::shared_metal;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLResourceOptions};

/// ggml-cpu `swiglu_oai`: the gate clamped from above only, the up
/// branch from both sides and offset by one.
fn swiglu_oai(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
    let x = g.min(limit);
    let y = u.clamp(-limit, limit);
    x / (1.0 + (alpha * -x).exp()) * (y + 1.0)
}

fn fill(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.731).sin() * scale)
        .collect()
}

fn upload_u32(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &[u32],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    let mut owned = data.to_vec();
    // SAFETY: `owned` is `len * 4` initialised bytes, copied by Metal
    // before the call returns.
    unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(owned.as_mut_ptr() as *mut _).unwrap(),
            owned.len() * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .expect("ids buffer")
}

/// Four slots over five experts, an expert used twice, and inputs wide
/// enough (|x| up to 12) that both clamps bind: a kernel that clamped
/// the gate from below, or forgot the `+ 1`, or read the bias of the
/// slot instead of its expert, is off by far more than the tolerance.
#[test]
#[ignore = "needs a real Metal GPU"]
fn swiglu_oai_and_add_id_match_ggml() {
    let shared = shared_metal().expect("metal");
    let device = &shared.device;
    let (n_experts, ffn, hidden) = (5usize, 96usize, 40usize);
    let ids: Vec<u32> = vec![3, 0, 3, 4];
    let n_slots = ids.len();
    let gate = fill(n_slots * ffn, 1.0, 12.0);
    let up = fill(n_slots * ffn, 7.0, 12.0);
    let router_bias = fill(n_experts, 2.0, 1.0);
    let gate_bias: Arc<[f32]> = fill(n_experts * ffn, 3.0, 2.0).into();
    let up_bias: Arc<[f32]> = fill(n_experts * ffn, 5.0, 2.0).into();
    let down_bias: Arc<[f32]> = fill(n_experts * hidden, 11.0, 2.0).into();
    let down_out = fill(n_slots * hidden, 13.0, 3.0);
    let (alpha, limit) = (1.702f32, 7.0f32);
    let oai = MoeOaiMetal {
        router_bias: &router_bias,
        gate_bias: &gate_bias,
        up_bias: &up_bias,
        down_bias: &down_bias,
        alpha,
        limit,
    };
    oai.check(n_experts, ffn, hidden);
    let res = MoeOaiResident::new(device, &oai).unwrap();

    let ids_buf = upload_u32(device, &ids);
    let (g_buf, u_buf) = (upload(device, &gate).unwrap(), upload(device, &up).unwrap());
    let act_buf = alloc(device, n_slots * ffn).unwrap();
    let act16_buf = alloc(device, n_slots * ffn).unwrap();
    let d_buf = upload(device, &down_out).unwrap();
    let cmd = shared.queue.commandBuffer().unwrap();
    let enc = cmd.computeCommandEncoder().unwrap();
    let whole = IdsBinding::whole(&ids_buf);
    encode_swiglu_oai_id(
        &enc, device, &res, &g_buf, &u_buf, whole, &act_buf, ffn, n_slots, false,
    )
    .unwrap();
    encode_swiglu_oai_id(
        &enc, device, &res, &g_buf, &u_buf, whole, &act16_buf, ffn, n_slots, true,
    )
    .unwrap();
    encode_add_id(
        &enc,
        device,
        &d_buf,
        whole,
        &res.down_bias.buffer,
        hidden,
        n_slots,
    )
    .unwrap();
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();

    let act = read_f32(&act_buf, n_slots * ffn);
    // SAFETY: the f16 output is `n_slots * ffn` halves the GPU finished.
    let act16: Vec<f32> = unsafe {
        std::slice::from_raw_parts(act16_buf.contents().as_ptr() as *const u16, n_slots * ffn)
    }
    .iter()
    .map(|&b| half::f16::from_bits(b).to_f32())
    .collect();
    let mut clamped = 0;
    for (s, &e) in ids.iter().enumerate() {
        for r in 0..ffn {
            let i = s * ffn + r;
            let b = e as usize * ffn + r;
            let (g, u) = (gate[i] + gate_bias[b], up[i] + up_bias[b]);
            if g > limit || u.abs() > limit {
                clamped += 1;
            }
            let want = swiglu_oai(g, u, alpha, limit);
            assert!(
                (act[i] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "slot {s} row {r}: f32 {} vs {want}",
                act[i]
            );
            assert!(
                (act16[i] - want).abs() <= 2e-3 * want.abs().max(1.0),
                "slot {s} row {r}: f16 {} vs {want}",
                act16[i]
            );
        }
    }
    assert!(clamped > n_slots * ffn / 8, "the clamps never bound");
    let down = read_f32(&d_buf, n_slots * hidden);
    for (s, &e) in ids.iter().enumerate() {
        for r in 0..hidden {
            let want = down_out[s * hidden + r] + down_bias[e as usize * hidden + r];
            assert!(
                (down[s * hidden + r] - want).abs() <= 1e-6,
                "add_id slot {s} row {r}"
            );
        }
    }
}

/// The expert-bias planes are cached by address with no byte compare,
/// on the strength of the entry holding the plane. A plane its owner
/// drops must therefore stay allocated, so the next plane of the same
/// size cannot land on its address and be served the old upload: the
/// allocator hands a just-freed block straight back, so an entry that
/// proved itself by address alone, without the hold, goes red here on
/// the second iteration.
#[test]
#[ignore = "needs a real Metal GPU"]
fn a_dropped_bias_plane_is_never_served_to_the_next_one() {
    let shared = shared_metal().expect("metal");
    let device = &shared.device;
    for i in 0..32u32 {
        let plane: Arc<[f32]> = vec![i as f32; 4096].into();
        let res = crate::gpu::resident_f32_shared(device, &plane).unwrap();
        let got = read_f32(&res.buffer, plane.len());
        assert!(
            got.iter().all(|&x| x == i as f32),
            "plane {i} was served plane {}'s upload",
            got[0]
        );
    }
}

/// Run `f` in a command buffer of its own and wait for it, so each step
/// of a two-dispatch pair sees the previous one's writes.
fn run(f: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>)) {
    let shared = shared_metal().expect("metal");
    let cmd = shared.queue.commandBuffer().unwrap();
    let enc = cmd.computeCommandEncoder().unwrap();
    f(&enc);
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The decode stack folds gpt-oss's router bias into the top-k kernel
/// and its `down` expert bias into the weighted sum, where they used to
/// be a `vec_add_rows` and a `moe_add_id` dispatch of their own. Each
/// fused kernel must give exactly what the pair gave (ADR 0009: the pair
/// is the one checked against ggml above), and the bias must matter in
/// both, or the comparison proves nothing.
#[test]
#[ignore = "needs a real Metal GPU"]
fn the_fused_bias_kernels_match_the_dispatch_pairs_they_replaced_bit_for_bit() {
    let shared = shared_metal().expect("metal");
    let device = &shared.device;

    // Top-k over 32 router logits, a bias large enough to reorder them.
    let (n_experts, top_k) = (32usize, 4usize);
    let logits = fill(n_experts, 1.0, 1.0);
    let router_bias = upload(device, &fill(n_experts, 9.0, 3.0)).unwrap();
    let topk = |logits_buf: &ProtocolObject<dyn MTLBuffer>,
                bias: Option<&ProtocolObject<dyn MTLBuffer>>| {
        let (ids, w) = (alloc(device, top_k).unwrap(), alloc(device, top_k).unwrap());
        run(|enc| {
            crate::gpu::encode_moe_topk_softmax_batch(
                enc,
                device,
                logits_buf,
                IdsBinding::whole(&ids),
                &w,
                n_experts as u32,
                top_k as u32,
                true,
                1,
                bias,
            )
            .unwrap()
        });
        (bits(&read_f32(&ids, top_k)), bits(&read_f32(&w, top_k)))
    };
    let pair_logits = upload(device, &logits).unwrap();
    run(|enc| {
        crate::elem::encode_vec_add_rows(
            enc,
            device,
            &pair_logits,
            &router_bias,
            n_experts as u32,
            1,
        )
        .unwrap()
    });
    let pair = topk(&pair_logits, None);
    let fused = topk(&upload(device, &logits).unwrap(), Some(&router_bias));
    let unbiased = topk(&upload(device, &logits).unwrap(), None);
    assert_ne!(unbiased.0, pair.0, "the bias must change the winners");
    assert_eq!(
        fused, pair,
        "fused top-k (ids, weights) differ from the pair"
    );

    // The weighted sum over four slots, an expert used twice.
    let (n_experts, hidden) = (5usize, 40usize);
    let ids = upload_u32(device, &[3, 0, 3, 4]);
    let n_slots = 4usize;
    let expert_out = fill(n_slots * hidden, 13.0, 3.0);
    let route = upload(device, &[0.4, 0.3, 0.2, 0.1]).unwrap();
    let down_bias = upload(device, &fill(n_experts * hidden, 11.0, 2.0)).unwrap();
    let h0 = fill(hidden, 17.0, 1.0);
    let whole = IdsBinding::whole(&ids);
    let combine =
        |eo: &ProtocolObject<dyn MTLBuffer>,
         add_id: Option<(IdsBinding<'_>, &ProtocolObject<dyn MTLBuffer>)>| {
            let h = upload(device, &h0).unwrap();
            run(|enc| {
                crate::gpu::encode_moe_weighted_sum_residual(
                    enc,
                    device,
                    eo,
                    &route,
                    &h,
                    hidden,
                    n_slots as u32,
                    add_id,
                )
                .unwrap()
            });
            bits(&read_f32(&h, hidden))
        };
    let pair_eo = upload(device, &expert_out).unwrap();
    run(|enc| encode_add_id(enc, device, &pair_eo, whole, &down_bias, hidden, n_slots).unwrap());
    let pair = combine(&pair_eo, None);
    let fused = combine(
        &upload(device, &expert_out).unwrap(),
        Some((whole, &down_bias)),
    );
    let unbiased = combine(&upload(device, &expert_out).unwrap(), None);
    assert_ne!(unbiased, pair, "the bias must change the sum");
    assert_eq!(fused, pair, "fused weighted sum differs from add_id + sum");
}
