#!/usr/bin/env python3
"""Generate the tiny synthetic `lfm3` GGUF used by frink's LFM3 coverage
test (`crates/frink-models/tests/lfm3_graphs.rs`).

`lfm3` (LFM3-1.7B, LFM3-VL / Omni's text tower) is LFM2's hybrid layer
stack (`src/models/lfm3.cpp`, `llm_build_lfm_base`) with four seams that
each already exist in frink for another architecture:

  * the window ARRAY, read with `ml.get_arr` (`:13`) and REQUIRED to
    mark at least one layer (`:14-16`); the converter marks
    `sliding_attention` layers only (`conversion/lfm3.py:95`), never a
    conv layer, and `rope.freq_base_swa` is REQUIRED (`:20`);
  * a PER-HEAD sigmoid attention gate, `attn_gate.weight` `{n_embd,
    n_head}`, projected from the `attn_norm` output (`:77,132-139`);
  * `rope_freqs.weight` on the FULL layers only (`:79-82,113`): the
    converter writes it as a partial-RoPE mask, `1` on the rotated pairs
    and `1e30` on the rest (`conversion/lfm3.py:101-111`), so the full
    layers rotate a fraction of the head and the sliding layers all of
    it;
  * optional sandwich post norms, both or neither (`:50-55`), applied
    to the CONV block's output as well as attention's (`:178-181` sit
    outside the `is_recr` branch).

Layers (five, so each kind appears and the sliding layers are not a
fixed phase):

    blk.0  conv
    blk.1  sliding attention   (window 3)
    blk.2  full attention      (rope_freqs, rope.freq_base)
    blk.3  conv
    blk.4  sliding attention

The two RoPE bases differ (25 full, 500 sliding) and head_dim is 8
with the full layers rotating 2 of 4 pairs, so a loader that hands the
sliding layers the full base, the full layers the sliding base, the
mask to the sliding layers, or no mask at all, each moves the logits.
The full base is low on purpose: at 10000 the two pairs the mask
removes turn by under 0.01 rad over a six-token prompt, and dropping
the mask moved the logits by 1e-2, which is no evidence at all.

Variants:
  * default          sandwich post norms, tied output
  * `--no-post-norm` neither post norm (`:50-55`, absent together)
  * `--non-causal`   `attention.causal = false`, the d1 decision graph
                     (`:22-25,92-96`, no memory at `llama-model.cpp:2808`);
                     frink refuses this file by name

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_lfm3_fixture.py OUT.gguf [--no-post-norm] [--non-causal]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "lfm3"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = 8
N_FF = 40
N_VOCAB = 48
CTX = 64
L_CACHE = 3
WINDOW = 3
ROPE_BASE = 25.0
ROPE_BASE_SWA = 500.0
# The full layers rotate this many of HEAD_DIM / 2 pairs
# (`partial_rotary_factor` 0.5, conversion/lfm3.py:104-109).
ROPE_ANGLES = 2
RMS_EPS = 1e-5

# head_count_kv per layer; 0 is a conv layer (lfm3.cpp:8-10).
KV_PER_LAYER = [0, N_KV, N_KV, 0, N_KV]
SLIDING = [False, True, False, False, True]


def main(out_path: str, post_norm: bool, non_causal: bool) -> None:
    n_layer = len(KV_PER_LAYER)
    rng = np.random.default_rng(0x1F03)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"frink-{ARCH}-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(KV_PER_LAYER)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(WINDOW)
    w.add_sliding_window_pattern(SLIDING)
    w.add_shortconv_l_cache(L_CACHE)
    w.add_vocab_size(N_VOCAB)
    if non_causal:
        w.add_causal_attention(False)
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
    # conversion/lfm3.py:101-111: one tensor, read by the full layers.
    mask = [1.0] * ROPE_ANGLES + [1e30] * (HEAD_DIM // 2 - ROPE_ANGLES)
    w.add_tensor("rope_freqs.weight", np.array(mask, dtype=np.float32))

    for il, nkv in enumerate(KV_PER_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if post_norm:
            # Drawn away from one so a dropped post norm is visible.
            w.add_tensor(
                p + "post_attention_norm.weight", (1.0 + 2.0 * rnd(N_EMBD)).astype(np.float32)
            )
            w.add_tensor(p + "post_ffw_norm.weight", (1.0 + 2.0 * rnd(N_EMBD)).astype(np.float32))
        if nkv == 0:
            # lfm3.cpp:84-86, the layout lfm2's converter writes.
            w.add_tensor(p + "shortconv.conv.weight", rnd(N_EMBD, L_CACHE) * 2.0)
            w.add_tensor(p + "shortconv.in_proj.weight", rnd(3 * N_EMBD, N_EMBD))
            w.add_tensor(p + "shortconv.out_proj.weight", rnd(N_EMBD, N_EMBD))
        else:
            w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_k.weight", rnd(nkv * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_v.weight", rnd(nkv * HEAD_DIM, N_EMBD))
            # Per head (:70-71), away from one.
            w.add_tensor(p + "attn_q_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_k_norm.weight", (1.0 + rnd(HEAD_DIM)).astype(np.float32))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
            # :77, {n_embd, n_head}: one sigmoid gate per head. Scaled up
            # so the gates spread across (0, 1) rather than sit at 0.5.
            w.add_tensor(p + "attn_gate.weight", rnd(N_HEAD, N_EMBD) * 6.0)
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

    # LLM_TENSOR_OUTPUT_NORM, not lfm2's `token_embd_norm` spelling.
    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {out_path} ({ARCH}, kv per layer {KV_PER_LAYER}, sliding {SLIDING}, "
        f"post_norm={post_norm}, non_causal={non_causal})"
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "lfm3-fixture.gguf",
        "--no-post-norm" not in sys.argv[1:],
        "--non-causal" in sys.argv[1:],
    )
