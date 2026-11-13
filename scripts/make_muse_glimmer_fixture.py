#!/usr/bin/env python3
"""Generate the tiny synthetic `muse-glimmer` GGUF.

`muse-glimmer` landed upstream after the 2026-08-04 llama.cpp pin and
was triaged NEW CODE on 2026-09-19 for TWO facts about norms, both of
which this fixture has to bind for the golden to mean anything:

  1. **A weightless RMS on the EMBEDDINGS.** `muse-glimmer.cpp:69` is
     `build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS, -1)` before
     layer 0, with no `token_embd_norm` tensor in the file --
     `bloom`'s embedding norm has a weight, and every other weightless
     RMS in llama.cpp is a LAYER slot.
  2. **A post-norm epsilon that is a literal.** `:63` is `const float
     post_norm_eps = 1e-8f` with the comment "Different to
     f_norm_rms_eps for post-attn / post-FFN norms", and `:140-141,
     166-167` use it while `:90,153` use the model's. The fixture
     declares `attention.layer_norm_rms_epsilon = 1e-3`, four orders
     larger, so the two epsilons are not interchangeable here: a
     post-norm run at the model's would move the logits.

The rest of the graph is a seam that already landed, and this fixture
carries each:

  * the per-element sigmoid attention gate (`:46,100-135`, "same as
    afmoe", `frink_models::attn_gate`),
  * RoPE on the SLIDING layers only (`:88`, `RopeLayers::SlidingOnly`)
    with its own base (`:13`, `rope.freq_base_swa`),
  * the window pattern read scalar-then-array through
    `load_swa_pattern(ml, 4)` (`:26`) -- this file declares the SCALAR
    2, so layers 1 and 3 slide and 0 and 2 are full attention,
  * a per-head QK RMSNorm at `{head_dim}` (`:40-41,106-107`),
  * `logit_scale` REQUIRED and MULTIPLIED (`:8,186`) with the final
    tanh softcap on top of it (`:189-193`),
  * Gemma-style post-attention and post-FFN norms with their own
    weights (`:31,49`).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_muse_glimmer_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "muse-glimmer"

N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
SWA_PATTERN = 2
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
MODEL_ROPE_BASE = 40000.0
# Four orders larger than the post-norm literal of 1e-8, so the two
# epsilons cannot be swapped without moving the logits.
RMS_EPS = 1e-3
LOGIT_SCALE = 0.5
FINAL_SOFTCAP = 8.0


def main(out_path: str, variant: str = "") -> None:
    n_layer = 4
    rng = np.random.default_rng(0x9115)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-muse-glimmer-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    # `--model-rope`: no `rope.freq_base_swa` and a model base that is
    # not llama.cpp's 10000 default, so a reader that falls back to the
    # default rotates the sliding layers differently from
    # `muse-glimmer.cpp:9-10`, which seeds them from the model's (#620).
    if variant == "--model-rope":
        w.add_rope_freq_base(MODEL_ROPE_BASE)
    else:
        w.add_rope_freq_base(ROPE_BASE)
        w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    # `muse-glimmer.cpp:5` reads the window REQUIRED: `--no-window`
    # leaves it out, and libllama refuses the file (#620).
    if variant != "--no-window":
        w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(SWA_PATTERN)
    w.add_logit_scale(LOGIT_SCALE)
    w.add_final_logit_softcapping(FINAL_SOFTCAP)
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
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        # Per element, drawn wide so the sigmoid saturates on some
        # channels and not others.
        w.add_tensor(p + "attn_gate.weight", rnd(n_embd_q, N_EMBD) * 6.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)
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
        sys.argv[1] if len(sys.argv) > 1 else "muse-glimmer-fixture.gguf",
        sys.argv[2] if len(sys.argv) > 2 else "",
    )
