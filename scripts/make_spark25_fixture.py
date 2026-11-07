#!/usr/bin/env python3
"""Generate the tiny synthetic `spark2_5` GGUF used by frink's
gated-attention coverage test.

`spark2_5` landed upstream after the 2026-08-04 llama.cpp pin and was
triaged ONE MATCH ARM on 2026-09-19 (`capability::NEOX_ROPE_TRIAGED`):
`src/models/spark2-5.cpp:41` creates `attn_gate` at `{n_embd, n_head}`
and `:97-105` sigmoids it and multiplies it into the attention output
per HEAD, before `wo` -- which is `GateAct::Sigmoid` with
`GateWidth::PerHead`, the pair `crate::attn_gate` already admits for
`step35`. The row needed the NAME in that table and nothing else.

Everything else in the graph is a seam that already landed, and this
fixture carries each rather than assuming it:

  * the sliding-window BOOL ARRAY (`:8`, `crate::swa_layers`) with a
    window narrower than the prompt, and its own RoPE base (`:10-12`,
    `rope.freq_base_swa`),
  * per-layer head counts (`:33-37`, `crate::layer_shapes`), so the
    per-head gate is sized by each layer's own count and a loader that
    read layer 0's count would build the wrong gate on layers 1..3,
  * a gated GELU FFN (`:124`, `LLM_FFN_GELU` under `LLM_FFN_PAR`),
  * NEOX RoPE (`llama_model_rope_type`), and a tied lm_head is
    permitted (`:26-29`) but this file carries `output.weight` so the
    head is exercised as its own matrix.

The window is 3 against a six-token prompt so the mask is doing work,
and the gate weights are drawn WIDE so the sigmoid spans its saturated
and linear regimes rather than sitting near 0.5 everywhere: a gate that
is 0.5 on every head is indistinguishable from a missing gate scaled by
two.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_spark25_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "spark2_5"

N_EMBD = 32
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
RMS_EPS = 1e-5

# `spark2-5.cpp:8` reads the pattern as an ARRAY, so it is the file that
# decides which layers slide -- layer 2 is full attention here, and it
# is the one whose logits move if `swa_layers` reads the array wrong.
IS_SWA = [True, True, False, True]
# Per-layer head counts (`:33`), deliberately non-uniform: the gate is
# `{n_embd, n_head(il)}`, so a loader that sized it from layer 0 would
# fail to load layers 1..3 rather than answer quietly.
HEADS = [4, 2, 4, 2]


# `--model-rope`: no `rope.freq_base_swa`, a model base other than 10000
# and a linear scaling factor of 2. `spark2-5.cpp:10-11` seed the sliding
# layers' base AND scale from the model's before the optional key (`:12`),
# so libllama rotates every layer at the model's base and scale (#470).
MODEL_ROPE_BASE = 40000.0
MODEL_ROPE_LINEAR = 2.0


def main(
    out_path: str,
    simple: bool = False,
    mode: str = "",
    no_window: bool = False,
    model_rope: bool = False,
) -> None:
    n_layer = len(HEADS)
    rng = np.random.default_rng(0x59A24)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-spark25-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(HEADS[0] if (simple or mode == "swaonly") else HEADS)
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
    # `spark2-5.cpp:5` reads the window REQUIRED: `--no-window` leaves it
    # out, and libllama refuses the file, as frink must (#470).
    if not no_window:
        w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern([True] * len(HEADS) if (simple or mode == "headsonly") else IS_SWA)
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

    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(n_layer):
        p = f"blk.{il}."
        n_head_il = HEADS[0] if (simple or mode == "swaonly") else HEADS[il]
        n_embd_q = n_head_il * HEAD_DIM
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) + 1.0)
        # x2 rather than the x4 the laguna fixture uses: the FFN here
        # is a GATED GELU, llama.cpp computes GELU from a 65536-entry
        # f16 table, and the table's error grows with the magnitude of
        # its argument. Amplifying the attention amplifies the residual
        # stream the FFN reads, and at x4 over four layers the
        # reference's own approximation moves the logits by 1.8e-3 --
        # nine times `GELU_TABLE_TOL`, which would have meant either a
        # weaker tolerance for this row or a golden that hides a real
        # defect behind the table. The gate weights below stay wide:
        # they feed a SIGMOID, which ggml computes exactly.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # `{n_embd, n_head}`: one scalar per head, drawn wide so the
        # sigmoid saturates on some heads and not others.
        w.add_tensor(p + "attn_gate.weight", rnd(n_head_il, N_EMBD) * 6.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "spark25-fixture.gguf",
        "--simple" in sys.argv[2:],
        "swaonly" if "--swaonly" in sys.argv[2:] else ("headsonly" if "--headsonly" in sys.argv[2:] else ""),
        no_window="--no-window" in sys.argv[2:],
        model_rope="--model-rope" in sys.argv[2:],
    )
