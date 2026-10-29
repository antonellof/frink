#!/usr/bin/env python3
"""Generate the tiny synthetic `falcon` GGUFs used by frink's Falcon
coverage test.

`falcon` (Falcon-7B, Falcon-40B, Falcon-180B) refused as a `dedicated`
row for its parallel residual. `falcon.cpp` is, line by line:

  * `:32-33,20-21` the biased LayerNorm (`NormOp::LayerNormBias`).
  * `:38` a FUSED `attn_qkv.weight` with NO bias, Q rows then K then V
    (`conversion/falcon.py:27-46` re-orders HF's per-kv-group layout
    into that), multi-query at 7B (`n_head_kv = 1`), grouped at 40B.
  * `:71-74,124-135` the shared-norm PARALLEL residual: the FFN reads
    `attn_norm(x)`, the tensor attention read, and `ffn_out + attn_out +
    inpL` are summed once (`crate::parallel_residual`, `SharedNorm`).
  * `:35-36,79-85` Falcon-40B's `attn_norm_2`, OPTIONAL: when present,
    ATTENTION reads `attn_norm_2(x)` and the FFN keeps reading
    `attn_norm(x)` -- two norms of the layer input, `TwoNorms`, with the
    tensor NAMES crossed relative to `gptneox` (`crate::norm_sites`).
  * `:127-131` the ungated GELU FFN with no biases; `:23-26` `output`
    optional, tied to the embeddings when absent; NEOX RoPE over the
    whole head (`:51` asserts `n_embd_head == n_rot`).

Keys as `conversion/falcon.py:10-24` writes them: `context_length 2048`
(hard-coded), `embedding_length`, `feed_forward_length = 4 * hidden`,
`block_count`, `head_count`, `head_count_kv`, `attention.layer_norm_
epsilon`, `general.tensor_data_layout = jploski`; no `rope.freq_base`
(10000), no `rope.dimension_count`.

Shapes:

  * (default) Falcon-7B: `head_count_kv 1`, no `attn_norm_2`.
  * `--40b`: `head_count_kv 2` of 4 and `attn_norm_2.{weight,bias}` on
    every layer.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_falcon_fixture.py OUT.gguf [--40b]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import argparse

import numpy as np

import gguf

ARCH = "falcon"

N_LAYER = 3
N_EMBD = 32
N_HEAD = 4
HEAD_DIM = 8
N_FF = 4 * N_EMBD
N_VOCAB = 48
CTX = 2048
LN_EPS = 1e-5


def main(out_path: str, forty_b: bool, qkv_bias: bool = False) -> None:
    # `qkv_bias` (`--qkv-bias`) adds an `attn_qkv.bias` per layer:
    # refusal evidence for #447. `falcon.cpp:38` creates the fused
    # weight and no bias, so libllama fails on the extra tensors
    # (`wrong number of tensors`) and frink leaves them unread.
    rng = np.random.default_rng(0xFA1C)
    n_head_kv = 2 if forty_b else 1

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    def norm_w(n: int) -> np.ndarray:
        return (1.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    def norm_b(n: int) -> np.ndarray:
        return (0.5 + rng.standard_normal(n) * 0.5).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-falcon-fixture")
    w.add_context_length(CTX)
    w.add_tensor_data_layout("jploski")
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_block_count(N_LAYER)
    w.add_head_count(N_HEAD)
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

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = n_head_kv * HEAD_DIM
    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", norm_w(N_EMBD))
        w.add_tensor(p + "attn_norm.bias", norm_b(N_EMBD))
        if forty_b:
            w.add_tensor(p + "attn_norm_2.weight", norm_w(N_EMBD))
            w.add_tensor(p + "attn_norm_2.bias", norm_b(N_EMBD))
        q = rnd(n_embd_q, N_EMBD) * 2.0
        k = rnd(n_embd_kv, N_EMBD) * 2.0
        v = rnd(n_embd_kv, N_EMBD)
        w.add_tensor(p + "attn_qkv.weight", np.concatenate([q, k, v], axis=0))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF) * 2.0)

    w.add_tensor("output_norm.weight", norm_w(N_EMBD))
    w.add_tensor("output_norm.bias", norm_b(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))
    if qkv_bias:
        for il in range(N_LAYER):
            w.add_tensor(f"blk.{il}.attn_qkv.bias", rnd(n_embd_q + 2 * n_embd_kv))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="falcon-fixture.gguf")
    ap.add_argument("--40b", dest="forty_b", action="store_true")
    ap.add_argument("--qkv-bias", action="store_true")
    args = ap.parse_args()
    main(args.out, args.forty_b, args.qkv_bias)
