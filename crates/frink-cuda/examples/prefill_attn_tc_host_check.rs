//! Emits everything `tools/prefill_attn_tc_host_check/run.sh` needs to
//! run the *real* tensor-core prefill attention body
//! (`frink_cuda::prefill_attn_tc`) on a host CPU and compare it bit for
//! bit against `prefill_attn_tc_twin`: one unit per head width, a
//! fixture per case, the twin's output, and a manifest.
//!
//! Not a substitute for a GPU. NVRTC never sees the text, the four PTX
//! helpers are CPU stand-ins written from the ISA (`tc_shim.h`), and a
//! race a real scheduler would expose cannot show. It does check that
//! the body compiles as C++, that its fragment indexing, staging, key
//! walk, tile skipping and masking, online softmax, sink fold and
//! stores compute what the twin computes, at every head width and
//! group size, and that it writes every output element.
//!
//! It first emits the tensor-core GEMM body, which passed on real
//! hardware, with the exact f16 product it must reproduce through the
//! same stand-ins (`tc_shim.h` says why).
//!
//! Run it through `tools/prefill_attn_tc_host_check/run.sh`, not
//! directly.

use frink_cuda::mul_mm::{Q8_0, SUB};
use frink_cuda::mul_mm_tc::{tc_kernel_src, tc_names, TC_PTX_SRC};
use frink_cuda::prefill_attn_tc::{
    grid, heads_per_block, kernel::SMEM_DECL, prefill_attn_tc_twin, round_f16, variant_for,
    AttnArgs, VARIANTS,
};

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// The Q8_0 tensor-core GEMM over rows and tokens that leave both tiles
/// ragged, and the exact product its stand-in run must equal: each
/// dequantized weight and each activation rounded to f16 (what the body
/// and the launch's conversion store), summed over k in order in f32.
fn emit_gemm_calibration(dir: &str) {
    let kind = &Q8_0;
    let src = tc_kernel_src(kind);
    assert!(
        src.contains(TC_PTX_SRC),
        "the GEMM unit no longer carries the shared PTX"
    );
    std::fs::write(format!("{dir}/gemm.cu"), src.replace(TC_PTX_SRC, "")).unwrap();
    let (n_rows, n_cols, batch) = (165usize, 96usize, 178usize);
    let row_bytes = n_cols / kind.block_elems * kind.block_bytes;
    let mut state = 99u32;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        state
    };
    let mut w = Vec::with_capacity(n_rows * row_bytes);
    for _ in 0..n_rows * n_cols / kind.block_elems {
        let d = half::f16::from_f32(0.001 + (next() >> 20) as f32 / 4096.0 * 0.05);
        w.extend_from_slice(&d.to_le_bytes());
        w.extend((0..kind.block_elems).map(|_| (next() >> 16) as u8));
    }
    let x16: Vec<half::f16> = (0..batch * n_cols)
        .map(|i| half::f16::from_f32(((i as f32) * 0.023).sin()))
        .collect();
    let mut want = vec![0f32; batch * n_rows];
    let mut reg = [0f32; SUB];
    for row in 0..n_rows {
        let rp = &w[row * row_bytes..(row + 1) * row_bytes];
        let mut wr = Vec::with_capacity(n_cols);
        for sub in 0..n_cols / SUB {
            let block = &rp[(sub / kind.nl()) * kind.block_bytes..];
            (kind.dequant_twin)(block, sub % kind.nl(), &mut reg);
            wr.extend(reg.iter().map(|&v| round_f16(v)));
        }
        for t in 0..batch {
            let mut acc = 0f32;
            for (a, b) in wr.iter().zip(&x16[t * n_cols..(t + 1) * n_cols]) {
                acc += a * b.to_f32();
            }
            want[t * n_rows + row] = acc;
        }
    }
    std::fs::write(format!("{dir}/gemm.w"), &w).unwrap();
    let xb: Vec<u8> = x16.iter().flat_map(|h| h.to_le_bytes()).collect();
    std::fs::write(format!("{dir}/gemm.x"), xb).unwrap();
    std::fs::write(format!("{dir}/gemm.want"), bytes(&want)).unwrap();
    std::fs::write(
        format!("{dir}/gemm.txt"),
        format!(
            "{} {n_rows} {n_cols} {batch} {row_bytes}\n",
            tc_names(kind).1
        ),
    )
    .unwrap();
}

/// `(n_q, n_heads, n_kv_heads, head_dim, start_pos, window, softcap,
/// sink, group)`, `group` 0 for the launch's own choice.
type Case = (
    usize,
    usize,
    usize,
    usize,
    usize,
    Option<usize>,
    Option<f32>,
    Option<f32>,
    usize,
);

/// Every head width and group size, and GQA 1, 2, 4, 7 (rows padded at
/// G = 8) and 8. Then a prefix, batches that are not a multiple of 16 or
/// of a block, windows that start mid-tile and leave whole tiles out of
/// a warp, softcaps that bind, and sinks above and below the scores.
const CASES: &[Case] = &[
    (13, 4, 4, 64, 0, None, None, None, 1),
    (13, 4, 4, 64, 0, None, None, None, 0),
    (40, 8, 4, 64, 9, None, None, Some(-1.0), 2),
    (77, 8, 2, 128, 0, None, None, None, 4),
    (77, 8, 2, 128, 45, Some(40), None, Some(3.0), 0),
    (21, 7, 1, 64, 5, Some(9), None, None, 8),
    (21, 7, 1, 64, 5, Some(9), None, Some(0.5), 0),
    (70, 8, 1, 128, 33, Some(23), Some(1.5), None, 8),
    (70, 8, 1, 128, 33, None, None, None, 1),
    (33, 8, 1, 128, 70, None, Some(1.5), Some(-2.0), 2),
    (19, 4, 1, 256, 40, Some(45), Some(2.0), None, 4),
    (40, 2, 1, 256, 3, None, None, Some(1.0), 1),
    (9, 2, 1, 256, 0, None, None, None, 0),
    (100, 2, 1, 256, 10, Some(20), None, None, 1),
    (130, 2, 2, 128, 0, Some(64), None, None, 1),
    (2, 32, 8, 128, 6, None, None, None, 0),
    (1, 8, 1, 64, 100, Some(30), None, None, 8),
];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: prefill_attn_tc_host_check <output-dir>");
    emit_gemm_calibration(&dir);
    for v in &VARIANTS {
        let body = v.body();
        assert!(
            body.contains(SMEM_DECL),
            "the shared-memory declaration moved"
        );
        let body = body.replace(
            SMEM_DECL,
            "unsigned short* const fx_tc_smem = g_tc_dyn_smem;",
        );
        std::fs::write(
            format!("{dir}/d{}.cu", v.head_dim),
            format!("{}{body}", v.defines()),
        )
        .unwrap();
    }
    let wave = |n: usize, seed: f32, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|i| amp * ((i as f32) * 0.173 + seed).sin())
            .collect()
    };
    let mut manifest = String::new();
    for (i, &(n_q, n_heads, n_kv, hd, start_pos, window, softcap, sink, group)) in
        CASES.iter().enumerate()
    {
        let total = start_pos + n_q;
        let q = wave(n_q * n_heads * hd, 0.3 + i as f32, 1.5);
        let k = wave(total * n_kv * hd, 0.7, 1.0);
        let v = wave(total * n_kv * hd, 1.1, 1.0);
        let args = AttnArgs {
            n_heads,
            n_kv_heads: n_kv,
            head_dim: hd,
            start_pos,
            window,
            scale: (hd as f32).sqrt().recip(),
            softcap,
        };
        let var = variant_for(&args).expect("a shape the kernel takes");
        let group = if group == 0 {
            heads_per_block(args.gqa(), n_q)
        } else {
            group
        };
        let sinks: Option<Vec<f32>> =
            sink.map(|s| (0..n_heads).map(|h| s + 0.25 * h as f32).collect());
        let want = prefill_attn_tc_twin(&q, &k, &v, n_q, &args, sinks.as_deref(), group);
        let (gx, gy, _) = grid(n_q, n_kv, args.gqa(), group);
        let tag = format!("case{i}");
        std::fs::write(format!("{dir}/{tag}.q"), bytes(&q)).unwrap();
        std::fs::write(format!("{dir}/{tag}.k"), bytes(&k)).unwrap();
        std::fs::write(format!("{dir}/{tag}.v"), bytes(&v)).unwrap();
        std::fs::write(format!("{dir}/{tag}.want"), bytes(&want)).unwrap();
        // Empty when the case has no sinks: the C side passes null.
        std::fs::write(
            format!("{dir}/{tag}.sinks"),
            bytes(sinks.as_deref().unwrap_or(&[])),
        )
        .unwrap();
        // Floats as their bits, so the C side reads exactly the value
        // the twin used.
        manifest.push_str(&format!(
            "{tag} {hd} {} {n_q} {n_heads} {n_kv} {start_pos} {} {} {} {group} {gx} {gy} {}\n",
            var.names.1,
            args.window_or_zero(),
            args.q_factor().to_bits(),
            args.softcap_or_zero().to_bits(),
            var.smem_bytes(),
        ));
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
