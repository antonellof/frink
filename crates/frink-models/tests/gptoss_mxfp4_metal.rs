//! gpt-oss with MXFP4 experts, on Metal against the CPU (#405).
//!
//! gpt-oss ships its routed experts as MXFP4, and until Metal had an
//! MXFP4 matvec every expert product of a Metal decode ran on the host
//! with the GPU idle. `gptoss_tiny.gguf` is an all-F32 file, so it never
//! reached that gap; this test re-stores the fixture's routed expert
//! gate and up projections as MXFP4 in memory (a nearest-value E2M1 quantizer below, any byte
//! pattern being a valid block) and runs the same decode twice:
//!
//! - in a CPU-pinned child process, which prints its logits, and
//! - in this Metal-pinned process, where the logits must agree with the
//!   child's and every expert product must have been a Metal dispatch.
//!
//! The second half is the claim. An expert kind with no Metal kernel
//! falls to the host without failing, so the test counts: the all-F32
//! fixture's decode encodes no Metal work, and the MXFP4 one must
//! encode one dispatch per MXFP4 expert product.
//!
//! `cargo test -p frink-models --features metal --test gptoss_mxfp4_metal -- --ignored --nocapture`

mod common;
use frink_core::cache::KvCache;
use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use frink_models::{Decoder, ModelConfig};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/gptoss_tiny.gguf"
);

/// `gptoss_graph`'s prompt: six positions cross the window of four.
const PROMPT: [usize; 6] = [3, 7, 11, 19, 23, 5];

const CHILD: &str = "FRINK_GPTOSS_MXFP4_CHILD";

/// E2M1, as `frink_quant` and ggml's Metal shaders read it.
const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// One row of f32 as GGUF MXFP4 blocks: an E8M0 exponent putting the
/// block's largest magnitude at or under 6, then each value's nearest
/// E2M1 code, low nibble element `j`, high nibble `j + 16`.
fn mxfp4_row(row: &[f32]) -> Vec<u8> {
    assert!(row.len().is_multiple_of(32), "MXFP4 needs whole blocks");
    let mut out = Vec::with_capacity(row.len() / 32 * 17);
    for block in row.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let e = if amax == 0.0 {
            1
        } else {
            ((amax / 6.0).log2().ceil() as i32 + 127).clamp(1, 254)
        };
        let d = f32::from_bits((e as u32) << 23);
        let code = |v: f32| {
            (0..16u8)
                .min_by(|&a, &b| {
                    let da = (E2M1[a as usize] * d - v).abs();
                    let db = (E2M1[b as usize] * d - v).abs();
                    da.total_cmp(&db)
                })
                .expect("sixteen codes")
        };
        out.push(e as u8);
        for j in 0..16 {
            out.push(code(block[j]) | (code(block[j + 16]) << 4));
        }
    }
    out
}

fn to_mxfp4(m: &mut WeightMatrix) {
    let WeightMatrix::F32(t) = m else {
        panic!("the fixture is all-F32; a routed expert is not");
    };
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let data: Vec<u8> = t.data.chunks_exact(cols).flat_map(mxfp4_row).collect();
    *m = WeightMatrix::Quantized {
        data: WeightBytes::Owned(data),
        rows,
        cols,
        kind: QuantKind::Mxfp4Gguf,
    };
}

fn load(mxfp4: bool) -> Decoder {
    let file = frink_gguf::GgufFile::open(FIXTURE).expect("fixture opens");
    let config = ModelConfig::from_gguf(&file).expect("fixture config parses");
    let mut d = common::load_decoder(FIXTURE, config);
    if mxfp4 {
        let mut n = 0;
        for layer in &mut d.layers {
            let frink_models::decoder::ExpertBacking::Resident(experts) = &mut layer.moe.experts
            else {
                panic!("the fixture's experts are resident");
            };
            // `down` reads the 16-wide expert FFN, half an MXFP4 block,
            // so it stays F32 (on Metal in both runs, which the count
            // below needs). `gate` and `up` read the 32-wide hidden.
            for ex in experts {
                to_mxfp4(&mut ex.gate);
                to_mxfp4(&mut ex.up);
                n += 2;
            }
        }
        assert!(n > 0, "no routed expert was re-stored");
    }
    d
}

/// Prefill the prompt (the GEMM path), then decode it token by token
/// from an empty cache (the matvec path). Both rows are returned.
fn run(d: &Decoder) -> (Vec<f32>, Vec<f32>) {
    let mut caches: Vec<KvCache> = d.config.new_kv_caches();
    let prefill = d.forward_batch_last(&PROMPT, 0, &mut caches);
    let mut caches: Vec<KvCache> = d.config.new_kv_caches();
    let mut decode = Vec::new();
    for (pos, &tok) in PROMPT.iter().enumerate() {
        decode = d.forward_token(tok, pos, &mut caches);
    }
    (prefill, decode)
}

fn print_row(tag: &str, row: &[f32]) {
    let vals: Vec<String> = row.iter().map(|v| format!("{v:e}")).collect();
    println!("FRINK-ROW {tag} {}", vals.join(" "));
}

#[test]
#[ignore = "needs Apple Metal GPU"]
fn gpt_oss_mxfp4_experts_decode_on_metal_and_match_the_cpu() {
    if std::env::var_os(CHILD).is_some() {
        common::pin_backend(common::Backend::Cpu);
        let (prefill, decode) = run(&load(true));
        print_row("prefill", &prefill);
        print_row("decode", &decode);
        return;
    }
    #[cfg(not(feature = "metal"))]
    eprintln!("skip: built without --features metal");
    #[cfg(feature = "metal")]
    metal::run();
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// Worst |Metal - CPU| logit delta. The prefill GEMM stages weights
    /// and activations as `half`; the decode matvec is f32 throughout.
    /// Both sit well under this; a wrong nibble or scale moves the
    /// logits by their own magnitude (~0.5).
    const TOL: f32 = 2e-2;

    fn cpu_rows() -> (Vec<f32>, Vec<f32>) {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "--nocapture",
                "gpt_oss_mxfp4_experts_decode_on_metal_and_match_the_cpu",
            ])
            .env(CHILD, "1")
            .env("FRINK_GOLDEN_BACKEND", "cpu")
            .env("FRINK_METAL", "0")
            .env("FRINK_METAL_ATTN", "0")
            .output()
            .expect("spawning the CPU child");
        assert!(
            out.status.success(),
            "CPU child failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let row = |tag: &str| -> Vec<f32> {
            let prefix = format!("FRINK-ROW {tag} ");
            stdout
                .lines()
                .find_map(|l| l.strip_prefix(prefix.as_str()))
                .unwrap_or_else(|| panic!("the CPU child printed no {tag} row"))
                .split_whitespace()
                .map(|w| w.parse().expect("a float"))
                .collect()
        };
        (row("prefill"), row("decode"))
    }

    /// Dispatches encoded by `f`.
    fn dispatched<T>(f: impl FnOnce() -> T) -> (T, u64) {
        let before = frink_metal::metal_encode_stats().dispatches;
        let out = f();
        (out, frink_metal::metal_encode_stats().dispatches - before)
    }

    pub fn run() {
        if frink_metal::gpu::probe().is_none() {
            eprintln!("skip: no Metal GPU detected");
            return;
        }
        common::pin_backend(common::Backend::Metal);
        let (cpu_prefill, cpu_decode) = cpu_rows();

        let mx = load(true);
        let ((prefill, decode), n_run) = dispatched(|| super::run(&mx));
        let worst_prefill = common::worst_vs(&prefill, &cpu_prefill);
        let worst_decode = common::worst_vs(&decode, &cpu_decode);
        eprintln!(
            "gpt-oss MXFP4 experts: worst |metal - cpu| prefill {worst_prefill:.3e}, \
             decode {worst_decode:.3e} ({n_run} dispatches)"
        );
        assert!(
            worst_prefill <= TOL && worst_decode <= TOL,
            "Metal against CPU: prefill {worst_prefill:.3e}, decode {worst_decode:.3e} \
             (tol {TOL:.0e})"
        );

        // The rest of the fixture is F32, which `WeightMatrix::apply`
        // keeps on the host, so the decode's Metal work is the expert
        // products and nothing else: each position, each layer, each of
        // top-k, one dispatch for `gate` and one for `up`. An MXFP4
        // product that fell to the host makes the count short; the
        // all-F32 file is the zero it is measured from.
        let f32_model = load(false);
        let (_, n_f32) = dispatched(|| decode_only(&f32_model));
        let (_, n_mx) = dispatched(|| decode_only(&mx));
        let want = PROMPT.len() * mx.layers.len() * mx.config.moe.n_experts_active * 2;
        eprintln!(
            "decode dispatches: all-F32 {n_f32}, MXFP4 experts {n_mx} (expert products {want})"
        );
        assert_eq!(
            n_f32, 0,
            "the all-F32 decode is not the host baseline it was"
        );
        assert_eq!(
            n_mx as usize, want,
            "the MXFP4 decode encoded {n_mx} Metal dispatches for {want} MXFP4 expert \
             products: some of them ran on the host"
        );
    }

    fn decode_only(d: &Decoder) -> Vec<f32> {
        let mut caches: Vec<KvCache> = d.config.new_kv_caches();
        let mut out = Vec::new();
        for (pos, &tok) in PROMPT.iter().enumerate() {
            out = d.forward_token(tok, pos, &mut caches);
        }
        out
    }
}
