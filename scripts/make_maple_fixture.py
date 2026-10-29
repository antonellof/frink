#!/usr/bin/env python3
"""Generate the tiny synthetic `maple` GGUF used by frink's
per-layer-RoPE coverage test.

`maple` landed upstream after the 2026-08-04 llama.cpp pin and was
triaged ONE MATCH ARM on 2026-09-19: `src/models/maple.cpp:88` calls
`ggml_rope_ext` inside `if (hparams.is_swa(il))` and nowhere else, so
its FULL-attention layers get no rotation at all. That is
`RopeLayers::SlidingOnly`, the rule `exaone-moe` and `cohere2` already
use, and the row needed the NAME in that table.

Everything else in the graph is a seam that already landed, and this
fixture carries each rather than assuming it:

  * the sliding-window BOOL ARRAY (`:8`) with its own RoPE base
    (`:10-12`, `rope.freq_base_swa`), so the rotated layers and the
    unrotated ones are told apart by the FILE,
  * the per-layer `expert_feed_forward_length` ARRAY (`:6`,
    `crate::layer_shapes`),
  * a per-head QK RMSNorm at `{head_dim}` (`:49-50,84-88`),
  * softmax routing with `norm_w = true` (`:128`),
  * the `swiglu_clamp_exp` arrays llama.cpp's GENERIC `build_moe_ffn`
    applies above 1e-6 (`:11`, `crate::act_layers`), with a zero beside
    each nonzero entry so both branches of that `if` are exercised,
  * NEOX RoPE (`llama_model_rope_type`).

Layer 2 is the FULL-attention layer: it is the one whose logits move if
`SlidingOnly` is read as "every layer rotates", and its window-free
attention also sees the whole six-token prompt where the sliding layers
see three.

**`--model-rope`** writes NO `rope.freq_base_swa`, a model base other
than 10000 and a linear `rope.scaling.factor` of 2 (#446). `maple.cpp:12-13`
seed the sliding layers' base AND scale from the model's before the
optional key (`:14`), so libllama rotates every layer at the model's
base and scale; a reader that fell back to `llama_hparams`' defaults
(base 10000, scale 1) would rotate the sliding layers differently.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_maple_fixture.py OUT.gguf [--model-rope]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "maple"

N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
# `--model-rope`: the model's own base and a linear factor, both away
# from `llama_hparams`' SWA defaults (10000, 1).
MODEL_ROPE_BASE = 40000.0
MODEL_ROPE_LINEAR = 2.0
RMS_EPS = 1e-5
N_EXPERT = 6
N_EXPERT_USED = 2

IS_SWA = [True, True, False, True]
# Per-layer expert width (`maple.cpp:6` reads the array); the tensors
# are sized from `n_ff_exp()`, layer 0's entry, so the array is uniform
# here and what it evidences is that frink READS it as an array rather
# than failing on one.
FF_EXP = [16, 16, 16, 16]
# A zero beside each nonzero entry, so the `> 1e-6` branch of
# llama-graph.cpp:2146-2164 runs on two layers and is skipped on two.
CLAMP_EXP = [2.0, 0.0, 1.5, 0.0]


def main(
    out_path: str, no_clamp: bool = False, all_swa: bool = False, model_rope: bool = False
) -> None:
    n_layer = len(IS_SWA)
    rng = np.random.default_rng(0x4A9E)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-maple-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    if model_rope:
        w.add_rope_freq_base(MODEL_ROPE_BASE)
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        w.add_rope_scaling_factor(MODEL_ROPE_LINEAR)
    else:
        w.add_rope_freq_base(ROPE_BASE)
        w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern([True] * len(IS_SWA) if all_swa else IS_SWA)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(FF_EXP)
    w.add_swiglu_clamp_exp([0.0] * len(CLAMP_EXP) if no_clamp else CLAMP_EXP)
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

    for il in range(n_layer):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, FF_EXP[il], N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, FF_EXP[il], N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, FF_EXP[il]))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "maple-fixture.gguf",
        "--no-clamp" in sys.argv[2:],
        "--all-swa" in sys.argv[2:],
        "--model-rope" in sys.argv[2:],
    )
