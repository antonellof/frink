#!/usr/bin/env python3
"""Generate the tiny synthetic DFlash pair: a `qwen3` target and a
`dflash` draft that drafts for it.

    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_dflash_fixture.py OUT.gguf --target
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_dflash_fixture.py OUT.gguf [--dflash1] [--window N]

The draft is what `src/models/dflash.cpp` loads (llama.cpp#27342 for the
DFlash2 parts). What it is built to tell apart:

  * TARGET LAYER INPUTS, not outputs: `dflash.target_layers` is [1, 3],
    the stream ENTERING layers 1 and 3 of a 4-layer target
    (`t_layer_inp[il] = inpL`, `qwen3.cpp:86`). Reading the output of
    those layers instead is the input of 2 and 4, a different vector.
  * The ORDER the two layers are concatenated in: `fc` is not
    symmetric in its two halves.
  * The two-tap convolution's layout (`build_dflash2_conv`): bases and
    projections are far from zero, so swapping the side, the tap or the
    group index moves every drafted score.
  * Non-causal block attention: the mask rows attend to each other in
    both directions; a causal mask changes every row but the last.
  * Shared V (`attn_v` absent): V is the RMS-normed raw K.
  * The selector's predecessor: position 1 scores against the anchor,
    later positions against the candidate chosen before them.
  * No `token_embd` / `output` in the draft: both are the target's.

`--dflash1` writes a draft without convolutions or selector (plain
DFlash, an argmax read-out). `--window N` gives every draft layer an N-
token sliding window, which masks the oldest context rows.

Weights are pseudo-random from fixed seeds, so the files are byte-stable.
The goldens come from libllama (`scripts/dflash_reference.cpp`), never
from frink.
"""

import argparse

import numpy as np

import gguf

N_VOCAB = 48
MASK_TOKEN = 47
N_EMBD = 32
HEAD_DIM = 8
N_HEAD = 4
N_HEAD_KV = 2
RMS_EPS = 1e-6
CTX = 128

TARGET_LAYERS = 4
TARGET_FF = 48

DRAFT_LAYERS = 2
DRAFT_FF = 40
BLOCK = 5
CONV_KERNEL = 2
CONV_GROUP = 8
SELECTOR_RANK = 6
SELECTOR_TOP_K = 4
# The stream entering target layers 1 and 3 (HF ids 0 and 2, plus one).
EXTRACT = [1, 3]


def vocab(w):
    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(3, N_VOCAB)]
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * N_VOCAB)
    types = [gguf.TokenType.CONTROL if i < 3 else gguf.TokenType.NORMAL for i in range(N_VOCAB)]
    w.add_token_types([int(t) for t in types])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_mask_token_id(MASK_TOKEN)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)


def common(w, n_layer, n_ff):
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    vocab(w)


def attn_ffn(w, rnd, il, n_ff, with_v=True):
    p = f"blk.{il}."
    w.add_tensor(p + "attn_norm.weight", (rnd(N_EMBD) + 1.0).astype(np.float32))
    w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
    w.add_tensor(p + "attn_k.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
    if with_v:
        w.add_tensor(p + "attn_v.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
    w.add_tensor(p + "attn_q_norm.weight", (rnd(HEAD_DIM) + 1.0).astype(np.float32))
    w.add_tensor(p + "attn_k_norm.weight", (rnd(HEAD_DIM) + 1.0).astype(np.float32))
    w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
    w.add_tensor(p + "ffn_norm.weight", (rnd(N_EMBD) + 1.0).astype(np.float32))
    w.add_tensor(p + "ffn_gate.weight", rnd(n_ff, N_EMBD))
    w.add_tensor(p + "ffn_up.weight", rnd(n_ff, N_EMBD))
    w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, n_ff))


def write(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def target(out):
    rng = np.random.default_rng(0xD7A5)

    def rnd(*shape):
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    w = gguf.GGUFWriter(out, "qwen3")
    w.add_name("frink-dflash-target-fixture")
    common(w, TARGET_LAYERS, TARGET_FF)
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))
    for il in range(TARGET_LAYERS):
        attn_ffn(w, rnd, il, TARGET_FF)
    w.add_tensor("output_norm.weight", (rnd(N_EMBD) + 1.0).astype(np.float32))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))
    write(w)


def draft(out, dflash1, window):
    rng = np.random.default_rng(0xDF2 if not dflash1 else 0xDF1)

    def rnd(*shape):
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    w = gguf.GGUFWriter(out, "dflash")
    w.add_name("frink-dflash-draft-fixture")
    common(w, DRAFT_LAYERS, DRAFT_FF)
    w.add_block_size(BLOCK)
    w.add_target_layers(EXTRACT)
    w.add_causal_attention(False)
    if window:
        w.add_sliding_window(window)
        w.add_sliding_window_pattern([True] * DRAFT_LAYERS)
    if not dflash1:
        w.add_conv_kernel_size(CONV_KERNEL)
        w.add_conv_group_size(CONV_GROUP)
        w.add_selector_rank(SELECTOR_RANK)
        w.add_selector_top_k(SELECTOR_TOP_K)

    groups = N_EMBD // CONV_GROUP
    for il in range(DRAFT_LAYERS):
        # Layer 1 shares K as V; layer 0 has its own V.
        attn_ffn(w, rnd, il, DRAFT_FF, with_v=(il == 0))
        if not dflash1:
            p = f"blk.{il}."
            # ne = {n_embd, kernel, 2} -> numpy [2, kernel, n_embd].
            w.add_tensor(p + "attn_conv_base", (rnd(2, CONV_KERNEL, N_EMBD) + 0.5).astype(np.float32))
            w.add_tensor(p + "attn_conv_proj.weight", rnd(2 * CONV_KERNEL * groups, N_EMBD))
            w.add_tensor(p + "ffn_conv_base", (rnd(2, CONV_KERNEL, N_EMBD) + 0.5).astype(np.float32))
            w.add_tensor(p + "ffn_conv_proj.weight", rnd(2 * CONV_KERNEL * groups, N_EMBD))

    # ne = {n_embd_inp, n_embd}: the two target layers side by side,
    # with different weights on each half.
    w.add_tensor("fc.weight", rnd(N_EMBD, len(EXTRACT) * N_EMBD))
    w.add_tensor("enc.output_norm.weight", (rnd(N_EMBD) + 1.0).astype(np.float32))
    w.add_tensor("output_norm.weight", (rnd(N_EMBD) + 1.0).astype(np.float32))
    if not dflash1:
        # ne = {rank, n_vocab} codebooks and {n_embd, rank} gate.
        w.add_tensor("selector_predecessor.weight", rnd(N_VOCAB, SELECTOR_RANK) * 4)
        w.add_tensor("selector_successor.weight", rnd(N_VOCAB, SELECTOR_RANK) * 4)
        w.add_tensor("selector_hidden.weight", rnd(SELECTOR_RANK, N_EMBD))
    write(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--dflash1", action="store_true")
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    if a.target:
        target(a.out)
    else:
        draft(a.out, a.dflash1, a.window)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
