//! The stack's first layer reading a gathered embedding row
//! ([`StackInput::Gather`]) instead of an uploaded one.

use super::tests::{q8_0_proj, small_q8_0, wave};
use super::*;
use crate::mul_mm::Q8_0;

const HIDDEN: usize = 64;
const N_VOCAB: usize = 40;

fn table() -> MulMmWeights<'static> {
    MulMmWeights {
        kind: &Q8_0,
        data: small_q8_0(N_VOCAB, HIDDEN, 71),
        rows: N_VOCAB,
        cols: HIDDEN,
        row_bytes: HIDDEN / 32 * 34,
    }
}

/// One small dense layer, every projection Q8_0 with both views.
fn layer(norm: &'static [f32]) -> PrefillDenseLayerCuda<'static> {
    let (n_heads, n_kv_heads, head_dim, ffn) = (2usize, 1usize, 32usize, 96usize);
    let (q_w, kv_w) = (n_heads * head_dim, n_kv_heads * head_dim);
    PrefillDenseLayerCuda {
        attn_norm_w: norm,
        ffn_norm_w: norm,
        q: q8_0_proj(small_q8_0(q_w, HIDDEN, 21), q_w, HIDDEN),
        k: q8_0_proj(small_q8_0(kv_w, HIDDEN, 22), kv_w, HIDDEN),
        v: q8_0_proj(small_q8_0(kv_w, HIDDEN, 23), kv_w, HIDDEN),
        o: q8_0_proj(small_q8_0(HIDDEN, q_w, 24), HIDDEN, q_w),
        gate: q8_0_proj(small_q8_0(ffn, HIDDEN, 25), ffn, HIDDEN),
        up: q8_0_proj(small_q8_0(ffn, HIDDEN, 26), ffn, HIDDEN),
        down: q8_0_proj(small_q8_0(HIDDEN, ffn, 27), HIDDEN, ffn),
        post_attn_norm: None,
        post_ffn_norm: None,
        extras: AttnExtrasCuda::default(),
        rope: None,
    }
}

fn params() -> PrefillParams<'static> {
    PrefillParams {
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        rms_eps: 1e-5,
        attn_scale: 32f32.sqrt().recip(),
        attn_softcap: None,
        window: None,
        prefix_k: &[],
        prefix_v: &[],
        start_pos: 0,
    }
}

fn norm() -> &'static [f32] {
    Box::leak(
        wave(HIDDEN, 0.4)
            .iter()
            .map(|v| 1.0 + 0.1 * v)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    )
}

/// A gather whose id is past the table, whose table is another width,
/// or whose id count is not the batch is refused before the device:
/// these run on a host with no GPU because nothing is launched.
///
/// Sabotage: drop the `validate` call in `StackInput::check` and the
/// out-of-range id reaches the device (a launch error on a GPU-less
/// host, not this refusal).
#[test]
fn a_gather_the_table_cannot_serve_is_refused_before_the_device() {
    let (layer, p) = (layer(norm()), params());
    let refuse = |input: StackInput<'_>, batch: usize, why: &str| {
        let err = launch_prefill_dense_stack(input, &[(&layer, &p)], batch, StackKv::UploadPrefix)
            .err()
            .unwrap_or_else(|| panic!("{why}: was not refused"));
        assert!(
            matches!(&err, CudaError::Unsupported(m) if m.contains("get_rows") || m.contains("input")),
            "{why}: {err:?}"
        );
    };
    let t = table();
    refuse(
        StackInput::Gather {
            table: t,
            ids: &[N_VOCAB as u32],
        },
        1,
        "id past the table",
    );
    refuse(
        StackInput::Gather {
            table: t,
            ids: &[1, 2],
        },
        1,
        "two ids for one row",
    );
    let narrow = MulMmWeights {
        cols: 32,
        row_bytes: 34,
        ..t
    };
    refuse(
        StackInput::Gather {
            table: narrow,
            ids: &[0],
        },
        1,
        "a table narrower than the layer",
    );
}

/// The stack started from a gathered row is the stack started from the
/// twin's row uploaded, bit for bit: after the first kernel the two
/// runs are the same launches on the same bytes. One row (the decode
/// path, matvecs) and four (a prefill batch, the GEMM).
#[test]
#[ignore = "requires real CUDA hardware; not yet run"]
fn a_gathered_stack_input_is_the_uploaded_host_row() {
    let (layer, p) = (layer(norm()), params());
    let t = table();
    for ids in [&[7u32][..], &[3, 39, 0, 3]] {
        let rows =
            crate::get_rows::get_rows_reference(t.kind, t.data, t.rows, ids, t.cols, t.row_bytes)
                .unwrap();
        let run = |input| {
            launch_prefill_dense_stack(input, &[(&layer, &p)], ids.len(), StackKv::UploadPrefix)
                .unwrap()
        };
        let gathered = run(StackInput::Gather { table: t, ids });
        let uploaded = run(StackInput::Hidden(&rows));
        assert_eq!(
            gathered.hidden,
            uploaded.hidden,
            "{} rows: hidden",
            ids.len()
        );
        assert_eq!(
            gathered.kv_rows,
            uploaded.kv_rows,
            "{} rows: K/V",
            ids.len()
        );
    }
}
