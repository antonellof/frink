#!/usr/bin/env python3
"""Generate the tiny synthetic `minimax-01` GGUFs used by frink's
MiniMax-Text-01 coverage test
(`crates/frink-models/tests/minimax_01_graphs.rs`).

`minimax-01` is MiniMax-Text-01 (456B-A45B): lightning attention on the
layers the recurrent mask names and ordinary GQA on the rest, with a
softmax MoE on every layer.

The mask is Qwen3.5's, read from the same two keys
(`minimax-01.cpp:11-17`): `attention.recurrent_layers` when present,
else layer `i` is recurrent iff `(i + 1) % full_attention_interval != 0`
with the interval defaulting to 8 instead of Qwen3.5's 4.

The lightning block (`:293-420`), per token and head:

```text
  qkv      = silu(attn_qkv(x))            # :303, BEFORE the split
  q, k, v  = qkv[h*3d .. ]                # :305-309, HEAD-major
  KV      <- KV * exp(-c s_h) + k^T v
  o        = q @ KV_after_decay + (q.k) v
  out      = attn_output(rms_norm(o, attn_norm_2) * sigmoid(attn_gate(x)))
```

with `s_h` the geometric slope ladder (`:97-103`) and
`c = 1 - il/(n_layer - 1) + 1e-5` this layer's scale (`:288`).

The residual is NOT the layer input (`:249,428-431,440,455-458`): each
sublayer's own pre-norm output, times a REQUIRED `{arch}.residual_scale`,
replaces the stream its branch joins. The fixture declares a scale far
from 1 so that a reader who treated it as Granite's branch multiplier
gets different logits.

Two heads at least, because the head-major split of the fused
projection is the identity at one head and a permutation at two; this
uses four.

Variants:
  * default          `full_attention_interval 2` (layers 0 and 2 recurrent)
  * `--array`        the same layout declared with `attention.recurrent_layers`
  * `--fused-qkv`    the full-attention layers carry one `attn_qkv`
                     instead of separate `attn_q` / `attn_k` / `attn_v`
                     (`llama-model.cpp:3289`, `create_tensor_qkv`)
  * `--output`       a separate `output.weight`
  * `--unit-scale`   `residual_scale = 1.0`: still this topology, because
                     the layer input is discarded whatever the multiplier
                     is -- the file that tells the seam apart from
                     `scale_or_none`

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_minimax_01_fixture.py OUT.gguf [--array] \\
            [--fused-qkv] [--output] [--unit-scale]

Weights are pseudo-random from a fixed seed so the files are byte-stable.
The golden logits that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "minimax-01"

N_EMBD = 32
N_HEAD = 4
N_KV = 2
HEAD_DIM = 8  # n_head * head_dim == n_embd, as MiniMax-Text-01 has it
ROPE_DIM = 4  # minimax-01.cpp:195: head_dim 128, n_rot 64
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
RESIDUAL_SCALE = 0.35

N_LAYER = 4
FULL_ATTENTION_INTERVAL = 2
RECURRENT = [(i + 1) % FULL_ATTENTION_INTERVAL != 0 for i in range(N_LAYER)]

# MoE on every layer (`minimax-01.cpp:58-62`).
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16

WIDTH = N_HEAD * HEAD_DIM


def main(
    out_path: str,
    as_array: bool,
    fused_qkv: bool,
    separate_output: bool,
    unit_scale: bool,
) -> None:
    rng = np.random.default_rng(0x0101)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"frink-{ARCH}-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    # minimax-01.cpp:6, REQUIRED and with no default.
    w.add_residual_scale(1.0 if unit_scale else RESIDUAL_SCALE)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(ROPE_DIM)
    if as_array:
        w.add_array(f"{ARCH}.attention.recurrent_layers", RECURRENT)
    else:
        w.add_full_attention_interval(FULL_ATTENTION_INTERVAL)
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if RECURRENT[il]:
            # minimax-01.cpp:48-50.
            w.add_tensor(p + "attn_norm_2.weight", (1.0 + rnd(WIDTH)).astype(np.float32))
            w.add_tensor(p + "attn_qkv.weight", rnd(3 * WIDTH, N_EMBD) * 2.0)
            w.add_tensor(p + "attn_gate.weight", rnd(WIDTH, N_EMBD) * 2.0)
        elif fused_qkv:
            w.add_tensor(
                p + "attn_qkv.weight",
                rnd(WIDTH + 2 * N_KV * HEAD_DIM, N_EMBD) * 2.0,
            )
        else:
            w.add_tensor(p + "attn_q.weight", rnd(WIDTH, N_EMBD) * 2.0)
            w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD) * 2.0)
            w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        # minimax-01.cpp:53: the SAME tensor on both kinds of layer.
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, WIDTH))

        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {out_path} ({ARCH}, array={as_array}, fused_qkv={fused_qkv}, "
        f"output={separate_output}, unit_scale={unit_scale})"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "minimax-01-fixture.gguf",
        "--array" in sys.argv[1:],
        "--fused-qkv" in sys.argv[1:],
        "--output" in sys.argv[1:],
        "--unit-scale" in sys.argv[1:],
    )
