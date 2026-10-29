#!/usr/bin/env python3
"""Generate the tiny synthetic `cohere2moe` GGUFs used by frink's Cohere2
MoE coverage test (`crates/frink-models/tests/cohere2moe_graphs.rs`).

`cohere2moe.cpp` is `cohere2.cpp` (the shared-norm PARALLEL residual, a
REQUIRED window and `logit_scale`, NORM RoPE) with routed experts:

  * `:4-11,166`: the norm FUNCTION comes from which epsilon key the
    file carries -- `layer_norm_rms_epsilon` present and nonzero is
    `LLM_NORM_RMS`, otherwise `LLM_NORM` (the weighted LayerNorm).
    `conversion/command_r.py` writes `layer_norm_epsilon` for every real
    export (`crate::norm::NORM_BY_RMS_EPS_KEY`).
  * `:13-22`: `sliding_window` and `logit_scale` REQUIRED,
    `leading_dense_block_count`, `expert_feed_forward_length`, the
    shared expert's count and width, `expert_weights_norm` / `_scale`,
    `expert_gating_func` defaulting to SIGMOID (`:27-29`).
  * `:31-37`: `set_swa_pattern(period, true)` (dense FIRST) from the
    scalar key, else the per-layer bool ARRAY the converter writes
    (`command_r.py:98`).
  * `:177-179,192`: a layer rotates when `is_swa(il) || il <
    n_layer_dense_lead` (`rope_layers::RopeLayers::SlidingOrLeadingDense`).
  * `:90-107`: the leading dense layers carry the dense triple, the
    rest the routed triple plus `_shexp` when `n_expert_shared > 0`.
  * `:234-260`: the router reads `ffn_inp = attn_norm(inpL)` (the
    parallel branch's one normed input); with a shared expert,
    `(moe_out + shexp) * 0.5` (`parallel_dense_ffn::
    SHARED_EXPERT_SUM_SCALE`).
  * `:23-24,380-420`: an MTP block inside `block_count`
    (`nextn_predict_layers`), skipped by the main graph
    (`crate::mtp_blocks`).

Shapes (four trunk layers, window 3 over a 6-token prompt, pattern
array `[F, T, T, F]`, dense lead 1): layer 0 is dense, full attention
AND rotated (the `il < n_layer_dense_lead` arm); layers 1-2 slide and
rotate; layer 3 is full, unrotated, MoE. 4 experts, 2 used, one shared
expert at `n_ff_exp`, `expert_weights_norm = false`, no gating key
(SIGMOID by default), `logit_scale 0.25`.

Variants:
  * default        `attention.layer_norm_epsilon` (LayerNorm, as every
                   real export)
  * `--rms`        `attention.layer_norm_rms_epsilon` instead (RMSNorm)
  * `--mtp`        one NextN block appended inside `block_count`
                   (`nextn_predict_layers = 1`), as a real export ships it
  * `--norm-w`     `expert_weights_norm = true` and a SOFTMAX gating key
  * `--no-eps`     neither norm epsilon key: refusal evidence, `:4-9`
                   throw `missing Cohere2 MoE norm epsilon` (#464)
  * `--no-dense-lead`  no `leading_dense_block_count`: refusal
                   evidence, `:15` reads it REQUIRED (#464)

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_cohere2moe_fixture.py OUT.gguf [--rms] [--mtp] [--norm-w]
        [--no-eps] [--no-dense-lead]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "cohere2moe"

N_LAYER = 4
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 6
N_FF = 40
N_FF_EXP = 16
N_EXPERT = 4
N_EXPERT_USED = 2
N_SHARED = 1
N_VOCAB = 48
CTX = 64
EPS = 1e-5
ROPE_BASE = 50_000.0
LOGIT_SCALE = 0.25
WINDOW = 3
DENSE_LEAD = 1
PATTERN = [False, True, True, False]


def main(
    out_path: str,
    rms: bool,
    mtp: bool,
    norm_w: bool,
    eps: bool = True,
    dense_lead: bool = True,
) -> None:
    rng = np.random.default_rng(0xC02E)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_weight(n: int) -> np.ndarray:
        return (1.0 + rng.standard_normal(n) * 0.3).astype(np.float32)

    n_blocks = N_LAYER + (1 if mtp else 0)
    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-cohere2moe-fixture")
    w.add_block_count(n_blocks)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_rope_freq_base(ROPE_BASE)
    if not eps:
        pass
    elif rms:
        w.add_layer_norm_rms_eps(EPS)
    else:
        w.add_layer_norm_eps(EPS)
    w.add_logit_scale(LOGIT_SCALE)
    w.add_sliding_window(WINDOW)
    # The converter's spelling: one bool per TRUNK layer (`n_layer()`).
    w.add_sliding_window_pattern(PATTERN)
    w.add_vocab_size(N_VOCAB)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    if dense_lead:
        w.add_leading_dense_block_count(DENSE_LEAD)
    w.add_expert_weights_norm(norm_w)
    if norm_w:
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
    w.add_expert_shared_count(N_SHARED)
    w.add_expert_shared_feed_forward_length(N_FF_EXP * N_SHARED)
    if mtp:
        w.add_nextn_predict_layers(1)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(3, N_VOCAB)]
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * N_VOCAB)
    types = [gguf.TokenType.CONTROL if i < 3 else gguf.TokenType.NORMAL for i in range(N_VOCAB)]
    w.add_token_types([int(t) for t in types])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    def attention(p: str) -> None:
        w.add_tensor(p + "attn_norm.weight", norm_weight(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

    def routed(p: str) -> None:
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP) * 2.0)
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_EXP * N_SHARED, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_EXP * N_SHARED, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_EXP * N_SHARED) * 2.0)

    for il in range(N_LAYER):
        p = f"blk.{il}."
        attention(p)
        if il < DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)
        else:
            routed(p)

    w.add_tensor("output_norm.weight", norm_weight(N_EMBD))

    if mtp:
        # cohere2moe.cpp:110-146: a full MoE block plus the NextN head,
        # inside `block_count`, never run by the main graph. Drawn AFTER
        # every trunk tensor so the trunk is byte-identical to the
        # default file's and libllama's logits for the two can be
        # compared directly.
        p = f"blk.{N_LAYER}."
        attention(p)
        routed(p)
        w.add_tensor(p + "nextn.eh_proj.weight", rnd(N_EMBD, 2 * N_EMBD))
        w.add_tensor(p + "nextn.enorm.weight", norm_weight(N_EMBD))
        w.add_tensor(p + "nextn.hnorm.weight", norm_weight(N_EMBD))
        w.add_tensor(p + "nextn.shared_head_norm.weight", norm_weight(N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (rms={rms}, mtp={mtp}, norm_w={norm_w})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "cohere2moe-fixture.gguf",
        "--rms" in sys.argv[1:],
        "--mtp" in sys.argv[1:],
        "--norm-w" in sys.argv[1:],
        "--no-eps" not in sys.argv[1:],
        "--no-dense-lead" not in sys.argv[1:],
    )
