//! `frink_vulkan`'s layer ops held to the CPU functions a device layer
//! will replace (#409, step 5): RMSNorm to [`crate::matmul::rms_norm`]
//! and `rms_norm_per_head`, RoPE through [`crate::rope_table`] to the
//! rotations in [`crate::attention`], and the f16 KV append plus
//! attention to [`crate::attention::causal_gqa_attention_row`] over the
//! CPU's own f16 store.
//!
//! These open their own `Runtime` rather than the process-wide one, so
//! they need no `FRINK_VULKAN`; with no device they skip through
//! `frink_vulkan::skipped`, which `FRINK_REQUIRE_VULKAN=1` turns red.

use frink_vulkan::runtime::ops::{AttnShape, Batch, DeviceBuf};
use frink_vulkan::runtime::Runtime;
use half::f16;

fn runtime() -> Option<Runtime> {
    match Runtime::new() {
        Ok(rt) => Some(rt),
        Err(e) => {
            frink_vulkan::skipped(&format!("no Vulkan device: {e}"));
            None
        }
    }
}

fn data(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((((i + seed) * 2654435761) % 1999) as f32 - 999.0) / 600.0)
        .collect()
}

fn upload(rt: &mut Runtime, v: &[f32]) -> DeviceBuf {
    let b = rt.alloc_f32(v.len()).unwrap();
    rt.write_f32(b, 0, v).unwrap();
    b
}

fn close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}[{i}]: vulkan {g} vs cpu {w}"
        );
    }
}

#[test]
fn the_vulkan_rms_norm_is_the_cpu_rms_norm() {
    let Some(mut rt) = runtime() else { return };
    let x = data(1, 2048);
    let w: Vec<f32> = data(2, 2048).iter().map(|v| 1.0 + v * 0.1).collect();
    let (xb, wb, ob) = (
        upload(&mut rt, &x),
        upload(&mut rt, &w),
        rt.alloc_f32(2048).unwrap(),
    );
    let mut b = Batch::new();
    b.rms_norm(wb, xb, ob, 2048, 1, 1e-5);
    rt.submit_batch(&b).unwrap();
    close(
        "rms_norm",
        &rt.read_f32(ob, 0, 2048).unwrap(),
        &crate::matmul::rms_norm(&x, &w, 1e-5),
        1e-5,
    );

    // Per head: Qwen3's attn_q_norm shape.
    let (heads, hd) = (16usize, 128usize);
    let q = data(3, heads * hd);
    let wq: Vec<f32> = data(4, hd).iter().map(|v| 1.0 + v * 0.1).collect();
    let (qb, wqb, oqb) = (
        upload(&mut rt, &q),
        upload(&mut rt, &wq),
        rt.alloc_f32(heads * hd).unwrap(),
    );
    let mut b = Batch::new();
    b.rms_norm(wqb, qb, oqb, hd, heads, 1e-6);
    rt.submit_batch(&b).unwrap();
    close(
        "rms_norm_per_head",
        &rt.read_f32(oqb, 0, heads * hd).unwrap(),
        &crate::matmul::rms_norm_per_head(&q, &wq, hd, 1e-6),
        1e-5,
    );
}

/// Llama 3.2's RoPE: Norm pairing with `rope_freqs` divisors, at a
/// position whose angles are far outside [-pi, pi], per head.
#[test]
fn the_vulkan_rope_is_the_cpu_rope() {
    let Some(mut rt) = runtime() else { return };
    let (heads, hd, pos, theta) = (8usize, 64usize, 70_001usize, 500_000.0f32);
    let ff: Vec<f32> = (0..hd / 2).map(|i| 1.0 + (i % 4) as f32 * 2.0).collect();
    let x = data(5, heads * hd);
    for (neox, factors) in [(false, Some(&ff[..])), (true, None)] {
        let tab = crate::rope_table::rope_cos_sin(pos, theta, hd, factors);
        let (tb, xb, ob) = (
            upload(&mut rt, &tab),
            upload(&mut rt, &x),
            rt.alloc_f32(x.len()).unwrap(),
        );
        let mut b = Batch::new();
        b.rope(tb, xb, ob, heads, hd, hd, neox);
        rt.submit_batch(&b).unwrap();
        let mut want = x.clone();
        for head in want.chunks_mut(hd) {
            match (neox, factors) {
                (false, Some(f)) => {
                    crate::attention::apply_rope_interleaved_with_freq_factors(head, pos, theta, f)
                }
                _ => crate::attention::apply_rope(head, pos, theta),
            }
        }
        // The device rotates with the CPU's angles, so the only
        // difference left is one rounding in each product.
        close(
            &format!("rope neox={neox}"),
            &rt.read_f32(ob, 0, x.len()).unwrap(),
            &want,
            1e-6,
        );
    }
}

/// Every position appended through the device's f16 store, then
/// attention, against the CPU's attention over its own f16 rows.
#[test]
fn the_vulkan_attention_over_its_f16_cache_is_the_cpu_attention() {
    let Some(mut rt) = runtime() else { return };
    let (n_heads, n_kv, hd, seq) = (32usize, 8usize, 64usize, 300usize);
    let row = n_kv * hd;
    let (k, v, q) = (
        data(6, seq * row),
        data(7, seq * row),
        data(8, n_heads * hd),
    );
    let half_words = seq * row / 2;
    let cache_words = 2 * half_words;
    let cache = rt.alloc_f32(cache_words).unwrap();
    let scores = rt.alloc_f32(n_heads * seq).unwrap();
    let out = rt.alloc_f32(n_heads * hd).unwrap();
    let (kb, vb, qb) = (
        upload(&mut rt, &k),
        upload(&mut rt, &v),
        upload(&mut rt, &q),
    );
    let mut b = Batch::new();
    b.kv_store(kb, cache, seq * row, 0, cache_words);
    b.kv_store(vb, cache, seq * row, half_words, cache_words);
    b.attention(
        cache,
        cache_words,
        qb,
        scores,
        out,
        AttnShape {
            n_heads,
            n_kv_heads: n_kv,
            head_dim: hd,
            seq_len: seq,
            k_off: 0,
            v_off: half_words,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    rt.submit_batch(&b).unwrap();

    let k16: Vec<f16> = k.iter().map(|x| f16::from_f32(*x)).collect();
    let v16: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
    let want = crate::attention::causal_gqa_attention_row(
        &q,
        &k16[..],
        &v16[..],
        n_heads,
        n_kv,
        hd,
        hd,
        seq,
        None,
        None,
        None,
        None,
    );
    close(
        "attention",
        &rt.read_f32(out, 0, n_heads * hd).unwrap(),
        &want,
        1e-4,
    );
}
