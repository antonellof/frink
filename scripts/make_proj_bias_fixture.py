#!/usr/bin/env python3
"""Generate the tiny synthetic `starcoder2`, `codeshell`, `jais2` and
biased-`llama` GGUFs used by frink's projection-bias coverage test.

The three were the rows of the "LayerNorm-with-bias group" whose OTHER
blocker was the projection biases: every one REQUIRES `attn_output.bias`,
`ffn_up.bias` and `ffn_down.bias` beside the six LayerNorm biases, and
the generic dense path had no slot for the first three
(`frink_models::proj_bias` is that slot; `NormOp::LayerNormBias` the
norm). One script, because the three graphs differ in three cells:

| | `starcoder2.cpp` | `codeshell.cpp` | `jais2.cpp` |
|---|---|---|---|
| FFN | `LLM_FFN_GELU, LLM_FFN_SEQ` (`:125-131`), ungated | the same (`:120-126`) | `LLM_FFN_RELU_SQR, LLM_FFN_SEQ` (`:130-136`), ungated |
| rope keys | `rope.dimension_count` (`conversion/starcoder.py`, the base class) | none: `n_rot = head_dim`; `rope.scaling.type = linear`, factor 1.0 (`conversion/codeshell.py:19-21`) | `rope.dimension_count = head_dim` (`conversion/jais.py:20-21`) |
| lm_head | `output` optional, `tok_embd` when absent (`:24-27`) | `output` REQUIRED (`:22`) | `output` optional (`:16-19`) |

`llama` is the fourth shape, and it is the OPTIONAL case: `llama.cpp`'s
own graph creates `attn_output.bias` and all three FFN biases as
`TENSOR_NOT_REQUIRED` and applies them when present, with a GATED
SwiGLU, so this file carries `ffn_gate.bias` too -- the one bias the
three ungated rows cannot exercise -- and RMSNorm rather than the biased
LayerNorm. frink used to refuse such a file as carrying unread
tensors.

The three biased-LayerNorm rows agree on everything else: Q/K/V biases present (`create_tensor_qkv`
in the first two; `jais2.cpp:37-39`, which size the K and V biases
`{n_embd}` and so admit only an MHA file), `wo` + `wo_b`,
the biased LayerNorm at all three sites (`attention.layer_norm_epsilon`),
NEOX RoPE (llama-model.cpp:2649,2652,2662), `kq_scale = 1/sqrt(head_dim)`.
The biases are drawn AWAY from zero, so any one dropped moves the logits
by far more than the tolerance; the GELU rows are compared at the
GeGLU tolerance because llama.cpp's f16 GELU table is the approximate
side.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_proj_bias_fixture.py {starcoder2,codeshell,jais2,llama} OUT.gguf

The golden values that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5
ROPE_BASE = 10000.0

SEEDS = {"starcoder2": 0x5C02, "codeshell": 0xC0DE, "jais2": 0x3A15, "llama": 0x11A3}

# `jais2.cpp:37-39` size the K and V biases `{n_embd}` -- "all have shape
# n_embd" -- rather than `{n_embd_k_gqa}`, so a grouped-query Jais-2 file
# cannot load in llama.cpp at all (`check_tensor_dims: tensor
# 'blk.0.attn_k.bias' has wrong shape; expected 32, got 16`, measured):
# the graph is MHA by construction and the fixture is too.
KV_HEADS = {"starcoder2": N_HEAD_KV, "codeshell": N_HEAD_KV, "jais2": N_HEAD, "llama": N_HEAD_KV}


def main(arch: str, out_path: str, spelled: str | None = None) -> None:
    # `spelled` writes a DIFFERENT architecture string into the file
    # while keeping this row's seed, shapes and weights. That is how an
    # alias is evidenced: a file byte-identical to the row it aliases
    # but for the name, asserted against that row's own golden. Without
    # it the two fixtures would differ in their random draws and the
    # comparison would prove nothing.
    rng = np.random.default_rng(SEEDS[arch])

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w() -> np.ndarray:
        return (1.5 + rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    def away_from_zero(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, spelled or arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    n_head_kv = KV_HEADS[arch]
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(n_head_kv)
    if arch == "llama":
        w.add_layer_norm_rms_eps(LN_EPS)
    else:
        w.add_layer_norm_eps(LN_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    if arch == "codeshell":
        # conversion/codeshell.py:19-21: no dimension count, a linear
        # scaling of exactly 1.
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        w.add_rope_scaling_factor(1.0)
    else:
        w.add_rope_dimension_count(HEAD_DIM)
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

    gated = arch == "llama"
    layer_norm = arch != "llama"

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = n_head_kv * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w())
        if layer_norm:
            w.add_tensor(p + "attn_norm.bias", away_from_zero(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 2.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", away_from_zero(N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm_w())
        if layer_norm:
            w.add_tensor(p + "ffn_norm.bias", away_from_zero(N_EMBD))
        if gated:
            # llama.cpp:`build_ffn(up, up_b, gate, gate_b, down, down_b,
            # ..., LLM_FFN_SILU, LLM_FFN_PAR)`, every bias optional.
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_gate.bias", away_from_zero(N_FF))
        # `ffn_up` / `ffn_down` and their biases.
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.bias", away_from_zero(N_FF))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)
        w.add_tensor(p + "ffn_down.bias", away_from_zero(N_EMBD))

    w.add_tensor("output_norm.weight", norm_w())
    if layer_norm:
        w.add_tensor("output_norm.bias", away_from_zero(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arch", choices=sorted(SEEDS))
    ap.add_argument("out")
    ap.add_argument(
        "--spelled",
        default=None,
        help="write this architecture string instead, keeping the row's weights",
    )
    args = ap.parse_args()
    main(args.arch, args.out, args.spelled)
