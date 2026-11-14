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
