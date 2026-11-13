//! Prints one Q4_K sub-block's `(scale, min)` as raw bits, to compare
//! against llama.cpp's `make_qkx2_quants` on the same 32 floats.
//!
//! Reads 32 lines of 8 hex digits (f32 bit patterns) from the path in
//! `argv[1]`, or `/tmp/blk.in`. See `1ebda4fc:docs/plans/llama-cpp-gap-inventory.md`
//! on why frink's Q4_K matches a reference built with
//! `-ffp-contract=off` exactly and an optimised one only approximately.
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/blk.in".to_string());
    let hex: Vec<u32> = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {path}: {e}"))
        .lines()
        .map(|l| u32::from_str_radix(l.trim(), 16).expect("8 hex digits per line"))
        .collect();
    assert_eq!(hex.len(), 32, "a Q4_K sub-block is 32 values");
    let x: Vec<f32> = hex.iter().map(|&b| f32::from_bits(b)).collect();
    let (scale, min) = frink_quant::encode::q4_k::probe_sub_block(&x);
    println!("scale={:08x} min={:08x}", scale.to_bits(), min.to_bits());
}
