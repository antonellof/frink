//! Emits everything `tools/moe_id_host_check/run.sh` needs to execute
//! the routed-expert CUDA C (`mul_mv_id`, `mul_mm_id`) on a host CPU
//! and compare it bit for bit against the twins in `frink_cuda::moe_id`:
//! one `.cu` per (kind, kernel), the fixtures, the twins' output, and a
//! manifest.
//!
//! Not a substitute for a GPU. It does check that each unit compiles as
//! C++, and that the expert offset, the activation-row and output-row
//! indirection and the per-expert early return address what the twins
//! do, against the kernel text.
//!
//! Run it through `tools/moe_id_host_check/run.sh`, not directly.

use frink_cuda::moe_id::{
    mm_id_names, mm_id_reference, mm_id_src, mv_id_names, mv_id_reference, mv_id_src, IdMap,
    IdPlane,
};
use frink_cuda::mul_mm::{BN, KINDS};

/// Five experts, seven tokens, top 2: expert 4 is never routed, expert
/// 1 gathers more rows than one column tile holds once the mm fixture
/// repeats it.
const N_EXPERTS: usize = 5;
const TOP_K: usize = 2;
const IDS: [u32; 14] = [1, 3, 0, 1, 1, 2, 3, 1, 1, 0, 2, 1, 3, 3];
const ROWS: usize = 70;

fn bytes_f(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn bytes_i(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: moe_id_host_check <output-dir>");
    let mut manifest = String::new();
    for k in KINDS {
        let cols = 256usize.max(k.block_elems).next_multiple_of(k.block_elems);
        let row_bytes = (cols / k.block_elems) * k.block_bytes;
        let stride = ROWS * row_bytes;
        // Arbitrary bytes, NaN and Inf scales included: both sides
        // must agree on those too.
        let mut state = 4242u32;
        let w: Vec<u8> = (0..N_EXPERTS * stride)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        let p = IdPlane {
            kind: k,
            data: &w,
            n_experts: N_EXPERTS,
            expert_stride: stride,
            rows: ROWS,
            cols,
        };
        let n_tokens = IDS.len() / TOP_K;
        let x: Vec<f32> = (0..n_tokens * cols)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();

        // mul_mv_id, gate/up shape: activation row = token.
        let unit = format!("{}_mv", k.name);
        std::fs::write(format!("{dir}/{unit}.cu"), mv_id_src(k)).unwrap();
        let want = mv_id_reference(&p, &IDS, &x, TOP_K);
        std::fs::write(format!("{dir}/{unit}.w"), &w).unwrap();
        std::fs::write(format!("{dir}/{unit}.x"), bytes_f(&x)).unwrap();
        let ids_i: Vec<i32> = IDS.iter().map(|&e| e as i32).collect();
        std::fs::write(format!("{dir}/{unit}.ids"), bytes_i(&ids_i)).unwrap();
        std::fs::write(format!("{dir}/{unit}.want"), bytes_f(&want)).unwrap();
        manifest.push_str(&format!(
            "{unit} MV {} {unit} {ROWS} {cols} {row_bytes} {} {} {TOP_K} {stride}\n",
            mv_id_names(k).1,
            cols / k.block_elems,
            IDS.len()
        ));

        // mul_mm_id, down shape (activation row = slot), with the ids
        // repeated so one expert gathers more than one column tile.
        let reps = BN / 8 + 1;
        let ids_mm: Vec<u32> = (0..reps).flat_map(|_| IDS).collect();
        let x_mm: Vec<f32> = (0..ids_mm.len() * cols)
            .map(|i| ((i as f32) * 0.023).sin())
            .collect();
        let map = IdMap::build(&ids_mm, TOP_K, N_EXPERTS, false);
        let want = mm_id_reference(&p, &map, &x_mm, ids_mm.len());
        let unit = format!("{}_mm", k.name);
        std::fs::write(format!("{dir}/{unit}.cu"), mm_id_src(k)).unwrap();
        std::fs::write(format!("{dir}/{unit}.w"), &w).unwrap();
        std::fs::write(format!("{dir}/{unit}.x"), bytes_f(&x_mm)).unwrap();
        std::fs::write(format!("{dir}/{unit}.mb"), bytes_i(&map.map_b)).unwrap();
        std::fs::write(format!("{dir}/{unit}.md"), bytes_i(&map.map_d)).unwrap();
        std::fs::write(format!("{dir}/{unit}.experts"), bytes_i(&map.experts())).unwrap();
        std::fs::write(format!("{dir}/{unit}.want"), bytes_f(&want)).unwrap();
        manifest.push_str(&format!(
            "{unit} MM {} {unit} {ROWS} {cols} {row_bytes} {} {} {} {stride}\n",
            mm_id_names(k).1,
            ids_mm.len(),
            map.eids.len(),
            map.max_count()
        ));
    }
    std::fs::write(format!("{dir}/manifest.txt"), manifest).unwrap();
}
