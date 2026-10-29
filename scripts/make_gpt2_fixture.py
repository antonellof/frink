#!/usr/bin/env python3
"""Generate the tiny synthetic `gpt2` and `starcoder` GGUFs used by frink's
learned-position-embedding coverage test.

Both refused as `dedicated` rows for a learned `position_embd` the
generic decoder had no slot for, and for their REQUIRED biases. Read
side by side, `gpt2.cpp` and `starcoder.cpp` are ONE graph: the
LayerNorm with a bias at every site (`gpt2.cpp:22-23,34-35,43-44`;
`NormOp::LayerNormBias`), a fused `attn_qkv.weight` / `.bias` (`:37-38`,
`qkv_fused`), REQUIRED `attn_output.bias`, `ffn_up.bias`, `ffn_down.bias`
(`:41,47,50`, `crate::proj_bias`), the ungated GELU FFN (`:57-62`),
a sequential residual, `output` optional and tied when absent
(`:24-28`), and `pos_embd` `{n_embd, n_ctx_train}` REQUIRED (`:19`),
gathered at `inp_pos` and ADDED to the token embedding before layer 0
(`:74-77`, `inpL = inpL + pos`) in place of any rotation
(`llama_model_rope_type`: `LLAMA_ROPE_TYPE_NONE`). `starcoder.cpp`
differs in `head_count_kv 1` (multi-query) and its size table.

Keys as `conversion/gpt2.py` and `conversion/starcoder.py` write them:
`block_count`, `context_length` (`n_ctx` / `n_positions`; the position
table's row count), `embedding_length`, `feed_forward_length = 4 *
n_embd`, `head_count`, `attention.layer_norm_epsilon`; `head_count_kv
1` for starcoder and NO `head_count_kv` for gpt2.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_gpt2_fixture.py OUT.gguf [--starcoder]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
N_FF = 4 * N_EMBD
N_VOCAB = 48
CTX = 64
LN_EPS = 1e-5


def main(out_path: str, starcoder: bool, split: bool = False) -> None:
    # `split` (`--split`) stores Q/K/V as `attn_{q,k,v}.weight` / `.bias`:
    # refusal evidence for #461. `gpt2.cpp:37-38` creates ONLY the fused
    # tensors, so libllama fails on the file (`wrong number of
    # tensors`), and frink refuses it by name.
    arch = "starcoder" if starcoder else "gpt2"
    n_head_kv = 1 if starcoder else N_HEAD
    rng = np.random.default_rng(0x6972 if starcoder else 0x6902)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    if starcoder:
        w.add_head_count_kv(n_head_kv)
    w.add_layer_norm_eps(LN_EPS)
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
    # :19 -- one row per trained position, drawn at the embedding's scale
    # so the position term is as visible as the token term.
    w.add_tensor("position_embd.weight", rnd(CTX, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = n_head_kv * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_norm.bias", norm_b(N_EMBD))
        q = rnd(n_embd_q, N_EMBD) * 2.0
        k = rnd(n_embd_kv, N_EMBD) * 2.0
        v = rnd(n_embd_kv, N_EMBD)
        b = rnd(n_embd_q + 2 * n_embd_kv)
        if split:
            w.add_tensor(p + "attn_q.weight", q)
            w.add_tensor(p + "attn_k.weight", k)
            w.add_tensor(p + "attn_v.weight", v)
            w.add_tensor(p + "attn_q.bias", b[:n_embd_q])
            w.add_tensor(p + "attn_k.bias", b[n_embd_q : n_embd_q + n_embd_kv])
            w.add_tensor(p + "attn_v.bias", b[n_embd_q + n_embd_kv :])
        else:
            w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
            w.add_tensor(p + "attn_qkv.bias", b)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "ffn_norm.bias", norm_b(N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)
        w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="gpt2-fixture.gguf")
    ap.add_argument("--starcoder", action="store_true")
    ap.add_argument("--split", action="store_true")
    args = ap.parse_args()
    main(args.out, args.starcoder, args.split)
