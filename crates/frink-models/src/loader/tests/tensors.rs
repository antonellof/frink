//! Loading single tensors end to end from hand-built GGUFs, one per
//! storage type, and the refusal for a type that is recognised but not
//! implemented.

use super::*;

/// A quantized 1-D tensor loads through the shared helper.
///
/// This used to be six copies of `load_f32_vec`, and they had
/// drifted badly: this one decoded twenty dtypes while the five
/// architecture loaders decoded three (F32/F16/BF16). A quantizer
/// that emits a Q8_0 norm or bias -- ordinary for aggressive
/// quants -- loaded on the generic path and was rejected with
/// `UnsupportedDtype` on GLM-5.2, Kimi, DeepSeek-MLA, Gemma-4 and
/// the hybrid stack.
///
/// This file's own comment predicted exactly that, about the same
/// split one level down: "a dtype frink can decode should never be
/// rejected here just because the *other* dispatch table below
/// knows it -- that split is how a supported format turns into a
/// load failure on the one checkpoint that uses it."
#[test]
fn a_quantized_one_dimensional_tensor_widens_through_the_shared_helper() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.25).collect();
    let quantized = frink_quant::quantize_q8_0(&values);

    struct OneTensor {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for OneTensor {
        fn metadata(&self, _key: &str) -> Option<&frink_gguf::GgufValue> {
            None
        }
        fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }
        fn tensor_bytes(&self, _name: &str) -> Result<&[u8], GgufError> {
            Ok(&self.bytes)
        }
        fn tensor_mapped_range(
            &self,
            name: &str,
        ) -> Result<
            (
                std::sync::Arc<frink_gguf::MmapHandle>,
                std::ops::Range<usize>,
            ),
            GgufError,
        > {
            // Never reached: `load_f32_vec` widens from bytes.
            Err(GgufError::TensorNotFound(name.to_string()))
        }
    }

    let source = OneTensor {
        info: TensorInfo {
            name: "blk.0.attn_norm.weight".to_string(),
            shape: vec![64],
            dtype: GgmlType::Q8_0,
            offset: 0,
        },
        bytes: quantized,
    };

    let widened = load_f32_vec(&source, "blk.0.attn_norm.weight")
        .expect("a Q8_0 norm must load, not report an unsupported dtype");
    assert_eq!(widened.len(), values.len());
    for (got, want) in widened.iter().zip(values.iter()) {
        assert!(
            (got - want).abs() < 0.05,
            "q8_0 round trip: got {got}, want {want}"
        );
    }
}

/// The same Q5_K block bytes cross-validated against an independent
/// Python reference in `frink-quant`'s own tests, reused here for
/// the same full-path proof as the Q6_K test below.
#[rustfmt::skip]
const Q5_K_TEST_BLOCK: [u8; 176] = [
    0x66, 0x2a, 0x66, 0x2a, 0x01, 0x01, 0x01, 0x01, 0x4f, 0x4b, 0x10, 0x12, 0x41, 0xe2, 0xc1,
    0xb1, 0x72, 0x2f, 0x20, 0x07, 0x31, 0x0c, 0x38, 0xb3, 0x9c, 0xb8, 0xad, 0x2f, 0x9a, 0xea,
    0x17, 0xd0, 0xee, 0x93, 0x9e, 0x3e, 0x74, 0xbb, 0x28, 0x18, 0x39, 0x25, 0xb6, 0x09, 0x18,
    0x29, 0x1c, 0x1d, 0x29, 0x41, 0x40, 0x0a, 0x74, 0x7d, 0xfd, 0x21, 0xdd, 0x6d, 0x45, 0x73,
    0x0e, 0x1e, 0xc0, 0x4a, 0xfc, 0xf3, 0x8e, 0x24, 0x6b, 0x34, 0x7d, 0xbe, 0x94, 0xde, 0x59,
    0x7a, 0x35, 0x30, 0x36, 0x0a, 0xf9, 0x4a, 0x9b, 0xa2, 0x26, 0x21, 0xa2, 0xfa, 0xdf, 0x4b,
    0x29, 0x64, 0x6f, 0xbb, 0xca, 0x0f, 0x3c, 0xda, 0x20, 0xf4, 0x93, 0x86, 0xab, 0x6e, 0xb9,
    0xe5, 0xd5, 0xa0, 0x82, 0xd6, 0x41, 0xff, 0x12, 0xbc, 0x34, 0xbb, 0xab, 0xb8, 0x20, 0x2f,
    0xbb, 0x5f, 0x0c, 0x10, 0xcf, 0x49, 0xc5, 0x86, 0x5c, 0xdf, 0xff, 0x78, 0x44, 0x26, 0x3b,
    0xc2, 0x23, 0x3d, 0x2b, 0xe9, 0x00, 0x12, 0xf8, 0xea, 0xe2, 0x9e, 0x5e, 0x50, 0x20, 0x9f,
    0x9d, 0x8d, 0x7d, 0x7f, 0xcc, 0x1d, 0x0e, 0x13, 0xf8, 0xc2, 0xf1, 0x3d, 0x08, 0x2f, 0x23,
    0x13, 0xac, 0x0d, 0xa7, 0xe7, 0x20, 0xa3, 0x90, 0xb7, 0xc8, 0x28,
];

fn build_single_q5_k_tensor_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-q5k-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols,
                                               // rows] -- reversed from the semantic [rows, cols] this tensor
                                               // represents (1 row, 256 cols / 1 Q5_K block).
    buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q5_K block)
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(13).unwrap(); // dtype tag: Q5_K
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&Q5_K_TEST_BLOCK);
    buf
}

/// How far a fused dot may sit from an exact dequantized dot.
///
/// Two regimes, and one fixed number cannot describe both. With
/// `FRINK_CPU_INT_DOT` off the activation stays f32 and only
/// rounding separates the two. With it on, the activation is
/// quantized to int8 at `d = amax / 127`, which is the flag both
/// binaries turn on by default and the reason the Q5_K and Q6_K
/// cases failed against a flat `1e-2`.
///
/// The bound grows with the L2 norm of the row, NOT the L1. Each
/// element carries an independent rounding of up to `d/2`, so the
/// dot's error is a sum of independent terms whose standard
/// deviation is `d/sqrt(12) * ||w||_2`. Bounding by the worst case
/// `d/2 * ||w||_1` instead assumes every rounding aligns with its
/// weight's sign, which on this fixture gives 0.347 against a dot
/// of 2.77: 12% of the value, loose enough that injecting a 5%
/// error still passed. Measured here, the real error is 1.8 sigma,
/// so four sigma keeps better than 2x headroom while still failing
/// that 5% injection.
fn fused_dot_tolerance(weights: &[f32], x: &[f32], exact_bound: f32) -> f32 {
    if !frink_core::weight_matrix::cpu_int_dot_for(frink_core::weight_matrix::IntDotShape::Matvec) {
        return exact_bound;
    }
    let amax = x.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let l2 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
    4.0 * (amax / 127.0) / 12f32.sqrt() * l2 + exact_bound
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_q5_k_tensor_end_to_end() {
    let tmp =
        std::env::temp_dir().join(format!("frink_test_q5k_tensor_{}.gguf", std::process::id()));
    std::fs::write(&tmp, build_single_q5_k_tensor_gguf()).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_K GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_K tensor must load");
    assert_eq!(matrix.rows(), 1);
    assert_eq!(matrix.cols(), 256);
    match &matrix {
        WeightMatrix::Quantized { kind, data, .. } => {
            assert_eq!(*kind, QuantKind::Q5K);
            assert!(
                data.is_mapped(),
                "Q5_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
            );
        }
        _ => panic!("expected a Quantized matrix for a Q5_K tensor"),
    }

    let expected = frink_quant::dequant_q5_k(&Q5_K_TEST_BLOCK).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
    let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

    let got = matrix.apply(&x);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
        "end-to-end loaded+applied Q5_K matrix diverged from direct dequant: got={} expected={}",
        got[0],
        expected_dot
    );
}

/// The same Q6_K block bytes cross-validated against an independent
/// Python reference in `frink-quant`'s own tests; reused here to
/// prove the *full*
/// path -- real on-disk GGUF bytes, parsed by `frink-gguf`, read
/// through `GgufFile::tensor_mapped_range`, dispatched by
/// `WeightMatrix::apply` to `frink_quant::dot_q6_k_f32` -- produces
/// the same result as directly dequantizing those bytes, not just
/// that the isolated kernel is correct in unit-test isolation.
#[rustfmt::skip]
const Q6_K_TEST_BLOCK: [u8; 210] = [
    0xe0, 0xa5, 0x40, 0x5c, 0x8d, 0x3a, 0x0a, 0x26, 0xfb, 0x4b, 0x6e, 0x9a, 0xdf, 0x3e, 0xa3,
    0xc4, 0xf8, 0x2b, 0x1d, 0x95, 0x76, 0x7d, 0x3b, 0xcd, 0xfd, 0xef, 0xc2, 0x0b, 0x07, 0x63,
    0x29, 0xfb, 0x81, 0x57, 0xbe, 0xbe, 0x06, 0xf7, 0x3a, 0x92, 0xc4, 0x43, 0xff, 0xad, 0xac,
    0x7e, 0x0f, 0x00, 0x2a, 0x4f, 0xf0, 0xf8, 0xa9, 0xfa, 0x3c, 0x90, 0x6d, 0x73, 0x2d, 0x5a,
    0xe6, 0xc6, 0x46, 0xf2, 0x0d, 0x55, 0x4c, 0x25, 0x38, 0x71, 0x2b, 0x35, 0x38, 0x82, 0x16,
    0x37, 0x5f, 0x32, 0x61, 0x02, 0xdd, 0x2f, 0x6f, 0x7b, 0x1f, 0xb4, 0x1a, 0x1b, 0x3e, 0x4f,
    0x11, 0xa3, 0x17, 0x40, 0x5a, 0x5f, 0x76, 0xcd, 0x19, 0x27, 0x9b, 0xc7, 0xc8, 0xf7, 0xf7,
    0xee, 0xf4, 0x86, 0xd9, 0xfd, 0xa7, 0xfe, 0x9e, 0xac, 0x70, 0x53, 0x5b, 0x76, 0xfb, 0x39,
    0xf8, 0x4b, 0x98, 0xfe, 0xd0, 0x06, 0x21, 0x4c, 0x4d, 0xbe, 0x10, 0x2b, 0x06, 0x65, 0xc9,
    0x5e, 0xf9, 0x95, 0x72, 0xae, 0x99, 0xd9, 0x7e, 0x15, 0xbd, 0x5e, 0x6d, 0xe8, 0x25, 0x8a,
    0xd5, 0x99, 0xc6, 0x6b, 0x69, 0xc7, 0x84, 0xc6, 0xa4, 0xf7, 0xb9, 0x6d, 0x68, 0x45, 0x0e,
    0x65, 0x69, 0xeb, 0xe6, 0xeb, 0xe9, 0x28, 0xa6, 0xb9, 0x96, 0xf2, 0xe8, 0xa7, 0x9b, 0x6e,
    0x79, 0x8a, 0x68, 0x65, 0x59, 0x98, 0x8b, 0x44, 0x41, 0x98, 0x9a, 0x56, 0x01, 0x01, 0x01,
    0x02, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x02, 0x02, 0x01, 0x01, 0x01, 0x02, 0x1f, 0x25,
];

fn build_single_q6_k_tensor_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-q6k-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
    buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q6_K block)
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(14).unwrap(); // dtype tag: Q6_K
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&Q6_K_TEST_BLOCK);
    buf
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_q6_k_tensor_end_to_end() {
    let tmp =
        std::env::temp_dir().join(format!("frink_test_q6k_tensor_{}.gguf", std::process::id()));
    std::fs::write(&tmp, build_single_q6_k_tensor_gguf()).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real Q6_K GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("Q6_K tensor must load");
    assert_eq!(matrix.rows(), 1);
    assert_eq!(matrix.cols(), 256);
    match &matrix {
        WeightMatrix::Quantized { kind, data, .. } => {
            assert_eq!(*kind, QuantKind::Q6K);
            assert!(
                data.is_mapped(),
                "Q6_K tensors should take the zero-copy mmap path, same as Q8_0/Q4_0"
            );
        }
        _ => panic!("expected a Quantized matrix for a Q6_K tensor"),
    }

    let expected = frink_quant::dequant_q6_k(&Q6_K_TEST_BLOCK).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
    let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

    let got = matrix.apply(&x);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-2),
        "end-to-end loaded+applied Q6_K matrix diverged from direct dequant: got={} expected={}",
        got[0],
        expected_dot
    );
}

fn build_single_bf16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-bf16-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
    buf.write_u64::<LittleEndian>(cols).unwrap();
    buf.write_u64::<LittleEndian>(rows).unwrap();
    buf.write_u32::<LittleEndian>(30).unwrap(); // dtype tag: BF16
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    for &v in values {
        // Real bf16 truncation (round-toward-zero, matching a real
        // writer closely enough for round-trip test purposes): top
        // 16 bits of the f32 bit pattern.
        let bf16_bits = (v.to_bits() >> 16) as u16;
        buf.extend_from_slice(&bf16_bits.to_le_bytes());
    }
    buf
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_bf16_tensor_end_to_end() {
    // Values with zero low-mantissa bits, so f32->bf16 truncation
    // is lossless and this is an exact-equality check.
    let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
    let tmp = std::env::temp_dir().join(format!(
        "frink_test_bf16_tensor_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&tmp, build_single_bf16_tensor_gguf(2, 3, &values)).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real BF16 GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("BF16 tensor must load");
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    match &matrix {
        WeightMatrix::F32(tensor) => {
            assert_eq!(tensor.data, values, "BF16 must widen to f32 exactly");
        }
        _ => panic!("expected an F32 matrix for a BF16 tensor (no fused dot kernel for it)"),
    }
}

fn build_single_f16_tensor_gguf(rows: u64, cols: u64, values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-f16-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(cols).unwrap();
    buf.write_u64::<LittleEndian>(rows).unwrap();
    buf.write_u32::<LittleEndian>(1).unwrap(); // dtype tag: F16
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    for &v in values {
        buf.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
    }
    buf
}

/// `GgmlType::F16` was parsed and sized but had no dequant arm in any
/// of the seven loaders, so every `*-f16.gguf` was a hard
/// `UnsupportedDtype`. Values are exactly representable in f16, so
/// this is an exact-equality check.
#[test]
fn load_weight_matrix_handles_a_real_on_disk_f16_tensor_end_to_end() {
    let values: Vec<f32> = vec![1.0, -2.5, 0.0, 4.0, -8.0, 16.0];
    let tmp =
        std::env::temp_dir().join(format!("frink_test_f16_tensor_{}.gguf", std::process::id()));
    std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("F16 tensor must load");
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.cols(), 3);
    match &matrix {
        WeightMatrix::F32(tensor) => {
            assert_eq!(tensor.data, values, "F16 must widen to f32 exactly");
        }
        _ => panic!("expected an F32 matrix for an F16 tensor (no fused dot kernel for it)"),
    }

    // The same tensor read as a plain vector (norm weights, biases and
    // the router all take this path, not `load_weight_matrix`).
    let tmp = std::env::temp_dir().join(format!("frink_test_f16_vec_{}.gguf", std::process::id()));
    std::fs::write(&tmp, build_single_f16_tensor_gguf(2, 3, &values)).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real F16 GGUF file must parse");
    std::fs::remove_file(&tmp).ok();
    assert_eq!(load_f32_vec(&file, "test.weight").unwrap(), values);
}

fn build_single_q5_1_tensor_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-q5-1-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
    buf.write_u64::<LittleEndian>(32).unwrap(); // cols (1 Q5_1 block)
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(7).unwrap(); // dtype tag: Q5_1
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    // d=0.25 (f16 0x3400), m=1.5 (f16 0x3E00) -- both exact in f16,
    // hand-verified bit patterns to avoid pulling in the `half`
    // crate just for two test constants. qh varied, qs a real
    // (non-degenerate) pattern.
    buf.extend_from_slice(&0x3400u16.to_le_bytes());
    buf.extend_from_slice(&0x3E00u16.to_le_bytes());
    buf.extend_from_slice(&[0x9au8, 0x3c, 0xf0, 0x0f]);
    buf.extend_from_slice(&(0..16u8).map(|i| i | ((15 - i) << 4)).collect::<Vec<u8>>());
    buf
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_q5_1_tensor_end_to_end() {
    let tmp = std::env::temp_dir().join(format!(
        "frink_test_q5_1_tensor_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&tmp, build_single_q5_1_tensor_gguf()).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real Q5_1 GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("Q5_1 tensor must load");
    assert_eq!(matrix.rows(), 1);
    assert_eq!(matrix.cols(), 32);
    let raw = file.tensor_bytes("test.weight").unwrap();
    let expected = frink_quant::dequant_q5_1(raw).unwrap();
    match &matrix {
        WeightMatrix::Quantized { kind, data, .. } => {
            assert_eq!(*kind, QuantKind::Q5_1);
            assert!(data.is_mapped());
        }
        _ => panic!("expected a Quantized matrix for a Q5_1 tensor"),
    }

    let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.017).cos()).collect();
    let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    let got = matrix.apply(&x);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expected_dot).abs() < 1e-2,
        "end-to-end loaded+applied Q5_1 matrix diverged from direct dequant: got={} expected={}",
        got[0],
        expected_dot
    );
}

// Same bytes as frink-quant's own Q3_K_TEST_BLOCK (Python-cross-
// validated there); duplicated here to build a real on-disk GGUF
// file, matching this file's existing per-format test convention
// (see Q6_K_TEST_BLOCK above).
const Q3_K_TEST_BLOCK: [u8; 110] = [
    0x56, 0xf2, 0xb4, 0x2b, 0xd5, 0x6f, 0x51, 0x71, 0x3c, 0x0a, 0xb9, 0x1d, 0xd0, 0xb9, 0x3b, 0xb3,
    0x0f, 0xff, 0x8c, 0xb2, 0x83, 0x3a, 0x3d, 0x24, 0xb1, 0x12, 0x56, 0xe3, 0x23, 0x54, 0xf2, 0xfa,
    0x7f, 0xdf, 0x31, 0xe1, 0x18, 0x26, 0x6e, 0xcd, 0x5b, 0x38, 0xee, 0xbd, 0x9f, 0x8c, 0x57, 0x47,
    0x0b, 0x11, 0xcb, 0xfb, 0xb4, 0x83, 0xa0, 0x4e, 0x0b, 0xd4, 0xa7, 0x85, 0xe0, 0x60, 0xf3, 0xb3,
    0xe3, 0x95, 0x43, 0xc6, 0x05, 0x05, 0x77, 0x53, 0xed, 0x23, 0xcc, 0x6a, 0x0e, 0x89, 0xa1, 0x79,
    0x85, 0xf6, 0x6e, 0x5a, 0x23, 0x63, 0xbe, 0x53, 0xfa, 0xa2, 0x2b, 0xe9, 0xcd, 0xce, 0xf8, 0x3d,
    0x6f, 0xd0, 0x42, 0x6e, 0x3b, 0x7f, 0x23, 0x26, 0xd3, 0xb9, 0x18, 0xbf, 0xa4, 0x34,
];

fn build_single_q3_k_tensor_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-q3k-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
    buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 Q3_K block)
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(11).unwrap(); // dtype tag: Q3_K
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&Q3_K_TEST_BLOCK);
    buf
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_q3_k_tensor_end_to_end() {
    let tmp =
        std::env::temp_dir().join(format!("frink_test_q3k_tensor_{}.gguf", std::process::id()));
    std::fs::write(&tmp, build_single_q3_k_tensor_gguf()).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real Q3_K GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("Q3_K tensor must load");
    assert_eq!(matrix.rows(), 1);
    assert_eq!(matrix.cols(), 256);
    match &matrix {
        WeightMatrix::Quantized { kind, data, .. } => {
            assert_eq!(*kind, QuantKind::Q3K);
            assert!(data.is_mapped());
        }
        _ => panic!("expected a Quantized matrix for a Q3_K tensor"),
    }

    let expected = frink_quant::dequant_q3_k(&Q3_K_TEST_BLOCK).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
    let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

    let got = matrix.apply(&x);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expected_dot).abs() < fused_dot_tolerance(&expected, &x, 1e-1),
        "end-to-end loaded+applied Q3_K matrix diverged from direct dequant: got={} expected={}",
        got[0],
        expected_dot
    );
}

// Same bytes as frink-quant's own IQ4_XS_TEST_BLOCK (Python-cross-
// validated there); duplicated here to build a real on-disk GGUF
// file, matching this file's existing per-format test convention.
const IQ4_XS_TEST_BLOCK: [u8; 136] = [
    0x5c, 0x33, 0xb4, 0x39, 0xd1, 0x64, 0x97, 0x82, 0xcb, 0xbd, 0x88, 0x95, 0xf3, 0x60, 0x2a, 0xb5,
    0xe7, 0x24, 0xd3, 0xee, 0xfe, 0x71, 0x13, 0xbe, 0x70, 0x84, 0x48, 0x79, 0x7b, 0x3e, 0xf0, 0x55,
    0xdc, 0xb2, 0xb2, 0xde, 0x32, 0xa1, 0x5b, 0x02, 0x01, 0xdc, 0x2a, 0xbb, 0xf7, 0x0b, 0x8a, 0x88,
    0xdd, 0x0b, 0x02, 0x7e, 0x5e, 0x76, 0x87, 0x30, 0x1e, 0x1c, 0xcf, 0x48, 0xd7, 0x61, 0xf3, 0x51,
    0x52, 0x17, 0x98, 0x0a, 0x87, 0xcf, 0x02, 0x91, 0xc8, 0xee, 0xc0, 0x91, 0x69, 0x2a, 0x4f, 0x64,
    0x68, 0xa7, 0xb2, 0xe6, 0x98, 0x21, 0x81, 0x75, 0x53, 0x2a, 0x8d, 0x12, 0xae, 0xe0, 0xea, 0x0c,
    0x75, 0xff, 0x22, 0x5e, 0x25, 0x19, 0xda, 0x2e, 0x51, 0x4e, 0x81, 0xdc, 0x0e, 0x78, 0x86, 0xd7,
    0x58, 0xb5, 0xb7, 0xf6, 0x45, 0xa9, 0x0a, 0x83, 0xfd, 0x2a, 0x12, 0x7d, 0xf0, 0x12, 0x97, 0xe2,
    0xfe, 0xf4, 0xd0, 0xa2, 0x11, 0x14, 0x78, 0xdb,
];

fn build_single_iq4_xs_tensor_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count

    write_kv_str(&mut buf, "general.architecture", "frink-iq4xs-test");

    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
                                               // Real GGUF ne[] order is fastest-varying-first, i.e. [cols, rows].
    buf.write_u64::<LittleEndian>(256).unwrap(); // cols (1 IQ4_XS block)
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(23).unwrap(); // dtype tag: IQ4_XS
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(&IQ4_XS_TEST_BLOCK);
    buf
}

#[test]
fn load_weight_matrix_handles_a_real_on_disk_iq4_xs_tensor_end_to_end() {
    let tmp = std::env::temp_dir().join(format!(
        "frink_test_iq4xs_tensor_{}.gguf",
        std::process::id()
    ));
    std::fs::write(&tmp, build_single_iq4_xs_tensor_gguf()).unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("real IQ4_XS GGUF file must parse");
    std::fs::remove_file(&tmp).ok();

    let matrix = load_weight_matrix(&file, "test.weight").expect("IQ4_XS tensor must load");
    assert_eq!(matrix.rows(), 1);
    assert_eq!(matrix.cols(), 256);
    match &matrix {
        WeightMatrix::Quantized { kind, data, .. } => {
            assert_eq!(*kind, QuantKind::IQ4XS);
            assert!(data.is_mapped());
        }
        _ => panic!("expected a Quantized matrix for an IQ4_XS tensor"),
    }

    let expected = frink_quant::dequant_iq4_xs(&IQ4_XS_TEST_BLOCK).unwrap();
    let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.013).sin()).collect();
    let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

    let got = matrix.apply(&x);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expected_dot).abs() < 1e-1,
        "end-to-end loaded+applied IQ4_XS matrix diverged from direct dequant: got={} expected={}",
        got[0],
        expected_dot
    );
}

// Same bytes as frink-quant's own IQ low-bit test blocks
// (Python-cross-validated there against the real compiled ggml
// implementation), duplicated as literals for the same reason as
// IQ4_XS_TEST_BLOCK above.
const IQ1_S_TEST_BLOCK: [u8; 50] = [
    0x0a, 0x2f, 0xfa, 0x06, 0x1e, 0x37, 0x6f, 0xe3, 0x62, 0xd0, 0xb6, 0xa4, 0x25, 0xae, 0x76, 0x14,
    0x72, 0x5b, 0xfa, 0x05, 0xd1, 0xf1, 0x2a, 0x4c, 0xad, 0x29, 0xae, 0xf4, 0xcf, 0x0c, 0x96, 0x51,
    0x58, 0x03, 0x6d, 0xd3, 0x10, 0x92, 0x70, 0xff, 0x61, 0x58, 0xc8, 0x30, 0x25, 0x64, 0x49, 0x85,
    0xc0, 0x24,
];

const IQ2_XXS_TEST_BLOCK: [u8; 66] = [
    0x29, 0x30, 0xd9, 0x33, 0x95, 0x4c, 0x08, 0x1e, 0xad, 0x79, 0x49, 0xf2, 0x8d, 0x5f, 0x93, 0xea,
    0x78, 0x18, 0x98, 0xb9, 0x94, 0x14, 0xad, 0xce, 0xca, 0x1d, 0xab, 0x81, 0x53, 0x4a, 0x68, 0xd0,
    0x59, 0x96, 0x36, 0x5d, 0xbe, 0x20, 0xc4, 0xff, 0xe4, 0x2c, 0xcd, 0x2f, 0x4f, 0x4f, 0x67, 0x53,
    0xc6, 0xd5, 0xa2, 0xfb, 0xc7, 0xf3, 0xe2, 0x6b, 0xf1, 0x99, 0x23, 0x1e, 0x2d, 0x5e, 0x8c, 0x78,
    0xc2, 0x31,
];

const IQ3_XXS_TEST_BLOCK: [u8; 98] = [
    0x71, 0x31, 0x16, 0x0a, 0x79, 0x04, 0x5d, 0x87, 0xae, 0x2a, 0x4a, 0x43, 0xfd, 0x02, 0xba, 0x6c,
    0x10, 0x42, 0x80, 0xe5, 0x1d, 0x08, 0x22, 0xcb, 0x21, 0x54, 0xf9, 0xaa, 0x8e, 0xc2, 0xf2, 0x34,
    0x66, 0x1e, 0x2a, 0xef, 0x19, 0xae, 0x48, 0x47, 0x29, 0xa0, 0x72, 0xd1, 0x31, 0xc0, 0x65, 0x49,
    0xde, 0x79, 0x32, 0xe6, 0x4d, 0xb6, 0x55, 0x3f, 0x4d, 0xf1, 0x18, 0xbb, 0x18, 0x59, 0x4c, 0x31,
    0xa3, 0xb2, 0x34, 0xdd, 0xf6, 0x4a, 0x91, 0x51, 0x3f, 0x3e, 0x40, 0x69, 0xad, 0xbf, 0x1a, 0xd0,
    0x05, 0xfb, 0xbe, 0x8b, 0x0b, 0xdd, 0xdf, 0x7d, 0x94, 0x74, 0x92, 0x3e, 0xff, 0x04, 0x2a, 0xc4,
    0xea, 0xc9,
];

#[rustfmt::skip]
const MXFP4_GGUF_TEST_BLOCKS: [u8; 68] = [0x79, 0xb4, 0x8d, 0xe2, 0x62, 0x5d, 0xbb, 0x9d, 0x54, 0xe6, 0xdb, 0x94, 0x59, 0x7d, 0x28, 0xf9, 0x79, 0x7a, 0xfc, 0xc1, 0xfa, 0x1e, 0x53, 0x5b, 0x0e, 0xc2, 0x5a, 0x2f, 0x0c, 0x82, 0x4d, 0xcb, 0x11, 0x28, 0x7b, 0x7c, 0xb6, 0x45, 0xe0, 0xb0, 0x52, 0x40, 0x51, 0xec, 0x30, 0x1a, 0xd2, 0x17, 0xf3, 0xbb, 0xfc, 0x7c, 0x8f, 0xf0, 0x67, 0x83, 0x88, 0x9d, 0x79, 0xdb, 0xf4, 0x45, 0x29, 0x78, 0xe6, 0xf4, 0x99, 0xea];

/// A live ggml type this build has no kernel for must be REFUSED BY
/// NAME at execution, having been sized correctly at parse.
///
/// Before `TQ2_0` was recognized, tag 35 was `Other(35)`, which had
/// no block layout: the tensor's size was unknown, so `tensor_bytes`
/// could not even hand back the row, and the error named a number.
/// Now the file parses, the tensor measures 66 bytes per 256
/// elements, and the stop happens where it belongs -- at the point
/// something wants to multiply by it -- naming `TQ2_0`.
#[test]
fn a_recognized_but_unimplemented_ggml_type_refuses_by_name_after_sizing_correctly() {
    // 256 elements of TQ2_0 = one 66-byte block.
    let block = pseudo_iq_block(66, 0x0720_5eed);
    let tmp = std::env::temp_dir().join(format!("frink_test_tq2_0_{}.gguf", std::process::id()));
    std::fs::write(
        &tmp,
        build_single_iq_lowbit_tensor_gguf("tq2test", 35, 256, &block),
    )
    .unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("a TQ2_0 file must still parse");
    std::fs::remove_file(&tmp).ok();

    // Sized, not zero: the size estimate is right even though the
    // kernel is missing.
    let info = file.find_tensor("test.weight").expect("tensor present");
    assert_eq!(info.dtype, GgmlType::TQ2_0);
    assert_eq!(info.byte_len(), Some(66));
    assert_eq!(
        file.tensor_bytes("test.weight").map(<[u8]>::len).ok(),
        Some(66)
    );

    match load_weight_matrix(&file, "test.weight") {
        Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
            assert_eq!(name, "test.weight");
        }
        Err(other) => panic!("TQ2_0 must be refused by name, got {other:?}"),
        Ok(_) => panic!("TQ2_0 must be refused, not loaded as some other kind"),
    }
}

/// An MXFP4 norm/bias must widen, not be refused.
///
/// `load_weight_matrix` accepts MXFP4 as a 2-D weight and
/// `load_moe_expert_matrices` accepts it as an expert tensor, and
/// `WeightMatrix::dequant` calls `dequant_mxfp4_gguf` on both. One
/// missing arm in `widen_plain_float` made the *1-D* tensors of the
/// exact same dtype a hard `UnsupportedDtype` -- the split that
/// turns a supported format into a load failure on the one
/// checkpoint that uses it.
#[test]
fn an_mxfp4_one_dimensional_tensor_widens_instead_of_being_refused() {
    let expected = frink_quant::dequant_mxfp4_gguf(&MXFP4_GGUF_TEST_BLOCKS)
        .expect("the fixture blocks must dequantize");
    let cols = expected.len();
    let tmp =
        std::env::temp_dir().join(format!("frink_test_mxfp4_norm_{}.gguf", std::process::id()));
    std::fs::write(
        &tmp,
        build_single_iq_lowbit_tensor_gguf("mxfp4norm", 39, cols as u64, &MXFP4_GGUF_TEST_BLOCKS),
    )
    .unwrap();
    let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
    std::fs::remove_file(&tmp).ok();

    let got = load_f32_vec(&file, "test.weight")
        .expect("an MXFP4 norm must load, not report an unsupported dtype");
    assert_eq!(got, expected);

    // Same arm, reached directly: `widen_plain_float` is the shared
    // helper the six architecture loaders call, so its table is the
    // one that has to know MXFP4.
    let direct = widen_plain_float(GgmlType::MXFP4, &MXFP4_GGUF_TEST_BLOCKS, "test.weight")
        .expect("widen_plain_float must widen MXFP4");
    assert_eq!(direct, expected);

    // And the refusal still works for a dtype that genuinely has no
    // widening path, so this test cannot pass by making everything
    // succeed.
    match widen_plain_float(GgmlType::TQ2_0, &MXFP4_GGUF_TEST_BLOCKS, "test.weight") {
        Err(LoadError::UnsupportedDtype(name, GgmlType::TQ2_0)) => {
            assert_eq!(name, "test.weight");
        }
        other => panic!("TQ2_0 must be refused by name, got {other:?}"),
    }
}

fn build_single_iq_lowbit_tensor_gguf(arch: &str, tag: u32, cols: u64, block: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.write_u32::<LittleEndian>(frink_gguf::GGUF_MAGIC)
        .unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap(); // version
    buf.write_u64::<LittleEndian>(1).unwrap(); // tensor_count
    buf.write_u64::<LittleEndian>(1).unwrap(); // kv_count
    write_kv_str(&mut buf, "general.architecture", arch);
    write_string(&mut buf, "test.weight");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(cols).unwrap();
    buf.write_u64::<LittleEndian>(1).unwrap(); // rows
    buf.write_u32::<LittleEndian>(tag).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset
    while buf.len() % 32 != 0 {
        buf.push(0);
    }
    buf.extend_from_slice(block);
    buf
}

/// A structurally valid block of `len` bytes for any of the
/// codebook-grid formats: every bit pattern is a legal code in all
/// of them (the grid indices are bounded by their own bit widths),
/// so a deterministic byte fill is a real block, not a fixture that
/// happens to avoid the interesting paths. Only the f16 scale needs
/// pinning, and only so the comparison below can't be NaN-vs-NaN.
fn pseudo_iq_block(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        out.push((s >> 24) as u8);
    }
    out
}

/// End-to-end load+apply for the codebook-grid low-bit formats the
/// published Dynamic GGUFs are built from: a real on-disk tensor of
/// each type must load zero-copy as the right `QuantKind` and
/// produce the same matvec result as dequantizing the block
/// directly. That is the property this test exists for -- the
/// *values* are pinned against real ggml in `frink-quant`; what
/// can only break here is the tag -> kind -> block-stride chain,
/// and a wrong stride silently reads the neighbouring row.
/// Dtype tags (19/29/16/17/22/18/21/39) verified against ggml.h's
/// enum ggml_type.
#[test]
fn load_weight_matrix_handles_real_on_disk_iq_lowbit_tensors_end_to_end() {
    type DequantFn = fn(&[u8]) -> Result<Vec<f32>, frink_quant::QuantError>;
    // IQ1_M carries no f16 scale field; its scale is reassembled
    // from the four scale words' top nibbles, and the top nibble of
    // the last one supplies the f16 sign + high exponent bits.
    // Pinning it to 0x2 keeps the exponent out of the all-ones
    // NaN/Inf pattern whatever the rest of the fill does. The other
    // three do carry a leading f16 `d`, pinned for the same reason.
    let mut iq1m = pseudo_iq_block(frink_quant::IQ1_M_BLOCK_BYTES, 0x2907_31A0);
    iq1m[55] = (iq1m[55] & 0x0F) | 0x20;
    let mut iq2xs = pseudo_iq_block(frink_quant::IQ2_XS_BLOCK_BYTES, 0x2107_31A1);
    let mut iq2s = pseudo_iq_block(frink_quant::IQ2_S_BLOCK_BYTES, 0x2207_31A2);
    let mut iq3s = pseudo_iq_block(frink_quant::IQ3_S_BLOCK_BYTES, 0x2307_31A3);
    for blk in [&mut iq2xs, &mut iq2s, &mut iq3s] {
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.115).to_le_bytes());
    }
    let cases: [(&str, u32, &[u8], QuantKind, DequantFn); 8] = [
        (
            "iq1s",
            19,
            &IQ1_S_TEST_BLOCK,
            QuantKind::IQ1S,
            frink_quant::dequant_iq1_s,
        ),
        (
            "iq1m",
            29,
            &iq1m,
            QuantKind::IQ1M,
            frink_quant::dequant_iq1_m,
        ),
        (
            "iq2xxs",
            16,
            &IQ2_XXS_TEST_BLOCK,
            QuantKind::IQ2XXS,
            frink_quant::dequant_iq2_xxs,
        ),
        (
            "iq2xs",
            17,
            &iq2xs,
            QuantKind::IQ2XS,
            frink_quant::dequant_iq2_xs,
        ),
        (
            "iq2s",
            22,
            &iq2s,
            QuantKind::IQ2S,
            frink_quant::dequant_iq2_s,
        ),
        (
            "iq3xxs",
            18,
            &IQ3_XXS_TEST_BLOCK,
            QuantKind::IQ3XXS,
            frink_quant::dequant_iq3_xxs,
        ),
        (
            "iq3s",
            21,
            &iq3s,
            QuantKind::IQ3S,
            frink_quant::dequant_iq3_s,
        ),
        (
            "mxfp4_gguf",
            39,
            &MXFP4_GGUF_TEST_BLOCKS,
            QuantKind::Mxfp4Gguf,
            frink_quant::dequant_mxfp4_gguf,
        ),
    ];
    for (name, tag, block, kind, dequant) in cases {
        let expected = dequant(block).unwrap();
        let cols = expected.len();
        let tmp = std::env::temp_dir().join(format!("frink_test_{name}_tensor.gguf"));
        std::fs::write(
            &tmp,
            build_single_iq_lowbit_tensor_gguf(name, tag, cols as u64, block),
        )
        .unwrap();
        let file = frink_gguf::GgufFile::open(&tmp).expect("file must parse");
        std::fs::remove_file(&tmp).ok();

        let matrix = load_weight_matrix(&file, "test.weight").expect("low-bit tensor must load");
        assert_eq!((matrix.rows(), matrix.cols()), (1, cols), "{name}");
        match &matrix {
            WeightMatrix::Quantized { kind: k, data, .. } => {
                assert_eq!(*k, kind, "{name}");
                assert!(data.is_mapped(), "{name} must load zero-copy");
            }
            _ => panic!("expected a Quantized matrix for {name}"),
        }

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.013).sin()).collect();
        let expected_dot: f32 = expected.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let got = matrix.apply(&x);
        assert!(
            (got[0] - expected_dot).abs() < 1e-1,
            "{name}: loaded+applied diverged from direct dequant: got={} expected={}",
            got[0],
            expected_dot
        );
    }
}
