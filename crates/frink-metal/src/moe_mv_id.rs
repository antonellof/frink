//! The slot-parallel MoE matvecs (`mul_mv_id`) built from a kind's own
//! decode matvec (#405).
//!
//! ggml's `kernel_mul_mv_id` is one template over the plain matvec's
//! `_impl`: slot `tgpig.z` reads its expert id, offsets the weights to
//! that expert, the activation to its token and the output to its slot,
//! and runs the same body. The Q2_K, Q3_K, IQ4_NL and MXFP4 matvecs are
//! written as such an `_impl` (`crate::q2_k`, `crate::q3_k`,
//! `crate::codebook4`), so their id kernels are generated here from
//! that one function: the dequant and the dot exist once per kind, and
//! the decode matvec's own kernel-vs-CPU tests cover the body the id
//! kernel runs.
//!
//! Each `_impl` takes `(weights, x, out, row_bytes, n_blocks_per_row,
//! n_rows, tgpig, tiisg, sgitg, table)`; `table` is 32 floats of
//! threadgroup memory the codebook formats stage their values in and
//! the K-quants ignore. The entries' buffer layout is the one
//! `gpu::encode_moe_matvec_id` and `gpu::encode_moe_down_id` bind.

use crate::gpu::matvec_launch_meta;

/// The two entry points for one kind: `{PREFIX}_moe_matvec_id` (gate
/// and up: activation row `token`, output row `slot`) and
/// `{PREFIX}_moe_down_id` (activation row `slot`, `ffn_rows` wide;
/// output row `slot`, `hidden_rows` wide). The slot is uniform over a
/// threadgroup, so the early return cannot split a barrier.
pub(crate) const MOE_MV_ID_MSL: &str = r#"
#define MOE_MV_ID_ENTRIES(PREFIX, IMPL)                                     \
kernel void PREFIX##_moe_matvec_id(                                         \
    device const uchar* w_all [[buffer(0)]],                                \
    device const float* x [[buffer(1)]],                                    \
    device float* out [[buffer(2)]],                                        \
    device const int* ids [[buffer(3)]],                                    \
    constant uint& row_bytes [[buffer(4)]],                                 \
    constant uint& n_blocks_per_row [[buffer(5)]],                          \
    constant uint& n_rows [[buffer(6)]],                                    \
    constant uint& top_k [[buffer(7)]],                                     \
    constant uint& expert_stride [[buffer(8)]],                             \
    constant uint& n_tokens [[buffer(9)]],                                  \
    constant uint& x_stride [[buffer(10)]],                                 \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    uint tiisg [[thread_index_in_simdgroup]],                               \
    uint sgitg [[simdgroup_index_in_threadgroup]]                           \
) {                                                                         \
    threadgroup float table[32];                                            \
    const uint slot = tgpig.z;                                              \
    if (slot >= n_tokens * top_k) return;                                   \
    const uint token = slot / top_k;                                        \
    const uint eid = uint(ids[slot]);                                       \
    IMPL(w_all + (size_t)eid * expert_stride,                               \
         x + (size_t)token * x_stride,                                      \
         out + (size_t)slot * n_rows,                                       \
         row_bytes, n_blocks_per_row, n_rows, tgpig.x, tiisg, sgitg, table); \
}                                                                           \
kernel void PREFIX##_moe_down_id(                                           \
    device const uchar* down_all [[buffer(0)]],                             \
    device const float* act [[buffer(1)]],                                  \
    device float* expert_out [[buffer(2)]],                                 \
    device const int* ids [[buffer(3)]],                                    \
    constant uint& row_bytes [[buffer(4)]],                                 \
    constant uint& n_blocks_per_row [[buffer(5)]],                          \
    constant uint& hidden_rows [[buffer(6)]],                               \
    constant uint& ffn_rows [[buffer(7)]],                                  \
    constant uint& top_k [[buffer(8)]],                                     \
    constant uint& expert_stride [[buffer(9)]],                             \
    constant uint& n_tokens [[buffer(10)]],                                 \
    uint3 tgpig [[threadgroup_position_in_grid]],                           \
    uint tiisg [[thread_index_in_simdgroup]],                               \
    uint sgitg [[simdgroup_index_in_threadgroup]]                           \
) {                                                                         \
    threadgroup float table[32];                                            \
    const uint slot = tgpig.z;                                              \
    if (slot >= n_tokens * top_k) return;                                   \
    const uint eid = uint(ids[slot]);                                       \
    IMPL(down_all + (size_t)eid * expert_stride,                            \
         act + (size_t)slot * ffn_rows,                                     \
         expert_out + (size_t)slot * hidden_rows,                           \
         row_bytes, n_blocks_per_row, hidden_rows, tgpig.x, tiisg, sgitg,   \
         table);                                                            \
}
"#;

/// Threads per threadgroup of every generated id kernel: two
/// simdgroups, as each kind's decode matvec runs.
const TG_THREADS: usize = 64;

/// The kinds whose id kernels this module generates, the matvec
/// translation unit that holds them, and their entry-point prefix.
fn kind_entry(kind: &str) -> Option<(&'static str, &'static str)> {
    match kind {
        "Q2_K" => Some((crate::q2_k::q2_k_matvec_src(), "q2_k")),
        "Q3_K" => Some((crate::q3_k::q3_k_matvec_src(), "q3_k")),
        "IQ4_NL" => Some((crate::codebook4::codebook4_matvec_src(), "iq4_nl")),
        "MXFP4" => Some((crate::codebook4::codebook4_matvec_src(), "mxfp4")),
        _ => None,
    }
}

/// The kinds [`dispatch`] answers for.
#[cfg(test)]
pub(crate) const KINDS: [&str; 4] = ["Q2_K", "Q3_K", "IQ4_NL", "MXFP4"];

/// One generated id kernel's launch: source, entry point, rows per
/// threadgroup (the decode matvec's, from `matvec_launch_meta`) and
/// threadgroup shape. `down` picks `_moe_down_id` over `_moe_matvec_id`.
pub(crate) fn dispatch(kind: &str, down: bool) -> Option<crate::gpu::MoeMatvecIdDispatch> {
    let (kernel_src, prefix) = kind_entry(kind)?;
    let (_, _, _, _, rows_per_tg) = matvec_launch_meta(kind)?;
    let fn_name = match (prefix, down) {
        ("q2_k", false) => "q2_k_moe_matvec_id",
        ("q2_k", true) => "q2_k_moe_down_id",
        ("q3_k", false) => "q3_k_moe_matvec_id",
        ("q3_k", true) => "q3_k_moe_down_id",
        ("iq4_nl", false) => "iq4_nl_moe_matvec_id",
        ("iq4_nl", true) => "iq4_nl_moe_down_id",
        ("mxfp4", false) => "mxfp4_moe_matvec_id",
        ("mxfp4", true) => "mxfp4_moe_down_id",
        _ => return None,
    };
    Some(crate::gpu::MoeMatvecIdDispatch {
        kernel_src,
        fn_name,
        rows_per_tg,
        tg_threads: (TG_THREADS, 1, 1),
        tg_mem: 0,
    })
}

#[cfg(test)]
mod tests;
