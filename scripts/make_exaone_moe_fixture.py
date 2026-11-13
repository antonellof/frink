#!/usr/bin/env python3
"""Generate the tiny synthetic `exaone-moe` GGUF used by frink's
per-layer-RoPE coverage test.

`exaone-moe` refused as UNAUDITED, triaged NEW CODE, for exactly one
blocker: **its GLOBAL layers get no RoPE.** `src/models/exaone-moe.cpp`
:136 is `const bool is_local_layer = hparams.is_swa(il);` and :155-161
wraps BOTH `ggml_rope_ext` calls in `if (is_local_layer)`, so on the
full-attention layer of every period Q and K are never rotated. :4 sets
`swa_type = LLAMA_SWA_TYPE_STANDARD` unconditionally and :6-8 gives it
`set_swa_pattern(4)`, last-dense, so it is three sliding layers and one
global, forever.

That is the SAME RULE as `exaone4`, not a similar one: `exaone4.cpp:116`
is `use_rope = is_swa(il) || swa_type == NONE`, and `exaone-moe`'s
`swa_type` is nailed to `STANDARD`, which makes the second disjunct
false and the two predicates identical. frink implements it once
(`crates/frink-models/src/rope_layers.rs`); this fixture and
`exaone4_32b_tiny.gguf` are what prove the shared body is right for both
rows rather than right for one and plausible for the other.

**FOUR LAYERS ARE NOT AN ACCIDENT.** With a period of 4, a 2- or
3-layer fixture would slide EVERY layer, every layer would rotate, and
the file would be evidence about nothing. Layer 3 is the global one:
unrotated, and (with `leading_dense_block_count = 1`) an MoE layer too,
so the row's two halves meet in it.

The rest of the graph is machinery frink already had, and the fixture
carries all of it so that "the MoE half is fine" stops being an
assertion:

  * a leading DENSE layer (`leading_dense_block_count = 1`, :72)
  * `blk.N.exp_probs_b.bias` -- the DeepSeek-V3 selection bias (:78),
    drawn large enough to REORDER the top-k against the unbiased scores
  * a shared expert added to the routed output (:91-93, :208-217), sized
    by `expert_shared_feed_forward_length` (:37)
  * `expert_weights_scale` != 1 and `expert_weights_norm` (:19-20)
  * sigmoid gating, a REQUIRED key here (:18)
  * PER-HEAD Q/K norms, `{n_embd_head_k}` wide (:66-67), applied before
    RoPE (:150-151 precede :155)
  * a sliding window NARROWER than the six-token prompt, so the mask is
    exercised rather than being a no-op. :13 reads
    `{arch}.attention.sliding_window` as a REQUIRED key, so the file's
    own value always wins over the `n_swa = 128` seeded at :5.

Two things a REAL export carries on top, each behind a flag so the
base fixture stays byte-identical to the one that audited the row:

  * `--window-array agree|disagree` writes `attention.sliding_window_
    pattern` as the per-layer BOOL ARRAY `conversion/exaone.py:84`
    writes from `layer_types` (K-EXAONE's is "LLLG" repeated), at
    `num_hidden_layers` length -- the TRUNK, not `block_count`.
    `exaone-moe.cpp:7` reads the key through the SCALAR overload of
    `get_key_or_arr`, which returns false on an array
    (llama-model-loader.cpp:502-507), so llama.cpp never looks at it
    and :8 keeps `set_swa_pattern(4)`. `agree` writes the array a real
    converter would ([T, T, T, F]); `disagree` writes its inverse, so
    that the golden logits being IDENTICAL for the two is the
    measurement that upstream ignores the value, and frink matching
    both is the evidence that it does too.
  * `--mtp` appends ONE NextN/MTP block after the trunk, inside
    `block_count` (5 blocks, `nextn_predict_layers = 1`), exactly as
    `conversion/exaone.py:132,146` does for K-EXAONE-236B-A23B
    (`num_hidden_layers = 48`, `num_nextn_predict_layers = 1`, 49
    blocks). llama.cpp creates its tensors `TENSOR_SKIP` (:52-57) and
    runs `n_layer() = 4` (llama-hparams.cpp:280-282); the block is
    REQUIRED to be present (the missing-tensor throw at
    llama-model-loader.cpp:1092-1095 precedes the skip at :1116, and
    this loader has no trunk-only probe), and it is a DENSE block
    (:73 `i >= n_layer` takes the dense branch) with the `nextn.*`
    tensors of :96-104. Its weights are drawn LARGE so that running it
    as a fifth layer is not a near miss.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_exaone_moe_fixture.py OUT.gguf \\
            [--window-array agree|disagree] [--mtp]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "exaone-moe"

# Four, so that layer 3 is the global (unrotated) one. See the docstring.
N_LAYER = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_FF_EXP = 16
N_FF_SHEXP = 16
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_DENSE_LEAD = 1
N_VOCAB = 48
CTX = 64
# Narrower than the six-token prompt, so the sliding layers really mask.
SWA_WINDOW = 3
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str, window_array: str | None, mtp: bool, no_gating: bool = False) -> None:
    rng = np.random.default_rng(0xE4A0E)
    n_mtp = 1 if mtp else 0
    block_count = N_LAYER + n_mtp

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-exaone-moe-fixture")
    w.add_block_count(block_count)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(SWA_WINDOW)
    if window_array is not None:
        # exaone.py:84: `[t == "sliding_attention" for t in layer_types]`,
        # one entry per HIDDEN layer. `set_swa_pattern(4)` last-dense is
        # [T, T, T, F] over four layers.
        agree = [il % 4 != 3 for il in range(N_LAYER)]
        w.add_sliding_window_pattern(agree if window_array == "agree" else [not b for b in agree])
    if mtp:
        w.add_nextn_predict_layers(n_mtp)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID; REQUIRED at exaone-moe.cpp:18.
    # `exaone-moe.cpp:16` reads the gating REQUIRED: `--no-gating-func`
    # leaves it out, and libllama refuses the file (#620).
    if not no_gating:
        w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    # Minimal SPM-flavoured vocab: llama.cpp needs tokens/scores/types to
    # build a vocab at all, but the fixture is always driven by explicit
    # token ids, never by tokenizing text.
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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Drawn wider than the rest of the file so the softmax over a
        # six-token prompt is not near-uniform; a near-uniform attention
        # cannot see a RoPE sabotage on the layers that DO rotate.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # PER-HEAD: `{n_embd_head_k}` wide (exaone-moe.cpp:66-67).
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        if il < N_DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # Large enough to reorder the top-k: sigmoid scores live in
        # (0, 1), so a bias spread of ~1 changes which experts win.
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )

        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    for il in range(N_LAYER, block_count):
        # The NextN block, exaone-moe.cpp:52-104 with `i >= n_layer`:
        # the attention set, a DENSE FFN, and the `nextn.*` head. Drawn
        # 8x wider than the trunk so that a loader which runs it as a
        # layer moves the logits by far more than any tolerance.
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) * 8.0)
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 8.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 8.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 8.0)
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q) * 8.0)
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) * 8.0)
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD) * 8.0)
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD) * 8.0)
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 8.0)
        # ne = [2 * n_embd, n_embd] -> numpy [n_embd, 2 * n_embd]
        w.add_tensor(p + "nextn.eh_proj.weight", rnd(N_EMBD, 2 * N_EMBD))
        w.add_tensor(p + "nextn.enorm.weight", rnd(N_EMBD))
        w.add_tensor(p + "nextn.hnorm.weight", rnd(N_EMBD))
        w.add_tensor(p + "nextn.shared_head_norm.weight", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="exaone-moe-fixture.gguf")
    ap.add_argument("--window-array", choices=["agree", "disagree"], default=None)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--no-gating-func", action="store_true")
    args = ap.parse_args()
    main(args.out, args.window_array, args.mtp, args.no_gating_func)
