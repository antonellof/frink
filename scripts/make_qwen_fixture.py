#!/usr/bin/env python3
"""Generate the tiny synthetic `qwen` GGUF used by frink's Qwen-1
coverage test.

`qwen` is the ORIGINAL Qwen (`QWenLMHeadModel`), not Qwen-2 and not
Qwen-3; those are separate GGUF architecture strings with separate
frink rows. Two things separate it from every audited row, and the
fixture exists to make both visible:

  * **The fused `attn_qkv.bias`** (`src/models/qwen.cpp:28`, REQUIRED,
    not `TENSOR_NOT_REQUIRED`). `build_qkv` adds it to the fused
    projection before splitting (llama-graph.cpp:1605-1609). This is
    the same arm `chatglm` needed, and it is the reason both rows were
    refused: frink split the fused WEIGHT and read bias only under the
    split `attn_q.bias` names.
  * **`n_ff` counts gate and up TOGETHER.** `qwen.cpp:33-35` sizes
    `ffn_gate`, `ffn_up` and `ffn_down` at `n_ff / 2`, because Qwen-1's
    `config.intermediate_size` is the *combined* width (HF's
    `QWenMLP` sets `ff_dim_in = intermediate_size // 2` and builds `w1`
    and `w2` at that width), and `conversion/qwen.py`'s `QwenModel`
    inherits the base `set_gguf_parameters`, which writes
    `intermediate_size` through unchanged (`conversion/base.py:1206`).
    So this fixture declares `feed_forward_length = 2 * FF` and ships
    matrices of width `FF`, exactly as a converted Qwen-1 does.

The rest, read off `src/models/qwen.cpp`:

  * `load_arch_hparams` (:3-11) reads only the RMS epsilon.
  * `load_arch_tensors` (:16-36): `attn_norm`, fused `attn_qkv` of
    `{n_embd, n_embd * 3}` -- so Qwen-1 is MHA, `n_head_kv == n_head`,
    and `head_dim * n_head == n_embd` -- `wo` `{n_embd, n_embd}`,
    `ffn_norm`, and split gate/up/down. `output` is REQUIRED (:20),
    unlike chatglm's tied fallback.
  * The graph (:60-123) is a plain sequential residual with
    `1/sqrt(n_embd_head)` (:92) and `LLM_FFN_SILU, LLM_FFN_PAR` (:113)
    -- ordinary SwiGLU with a separate gate.
  * RoPE is **NEOX** (`LLM_ARCH_QWEN` in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp:2626) over a WHOLE head: no converter writes
    `qwen.rope.dimension_count`, and llama-model.cpp:1200-1202 then
    defaults `n_rot` to `n_embd_head_k`. The fixture writes no such key
    for that reason -- writing one would test a file no converter can
    produce.

`{arch}.rope.freq_base` is deliberately SMALL (500 rather than 10000)
so six positions span radians rather than milliradians and a positional
sabotage is visible.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_qwen_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "qwen"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
# MHA: qwen.cpp:27 sizes the fused QKV as {n_embd, n_embd * 3}, which is
# only consistent with n_head_kv == n_head.
N_HEAD_KV = N_HEAD
HEAD_DIM = N_EMBD // N_HEAD
# The REAL per-matrix FFN width. The file declares twice this; see the
# module docstring.
FF = 40
N_VOCAB = 48
CTX = 64
ROPE_BASE = 500.0
RMS_EPS = 1e-5


def main(out_path: str, qkv_bias: bool = True) -> None:
    # `qkv_bias=False` (`--no-qkv-bias`) drops the fused bias: refusal
    # evidence for #447. `qwen.cpp:28` creates it REQUIRED, so libllama
    # does not load the file, and frink refuses it by name.
    rng = np.random.default_rng(0x9E71)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-qwen-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    # TWICE the real width. qwen.cpp:33-35 halves it.
    w.add_feed_forward_length(2 * FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    # NOTE: no `qwen.rope.dimension_count`. See the module docstring.
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

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_qkv = 3 * N_EMBD

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        # Fused QKV in llama.cpp's order: Q rows, then K rows, then V
        # rows. Q and K are drawn WIDER than the rest so the attention
        # scores over a six-token prompt are not nearly uniform -- a
        # fixture whose softmax is flat cannot see a positional
        # sabotage.
        qkv = np.concatenate(
            [
                rnd(N_EMBD, N_EMBD) * 4.0,
                rnd(N_EMBD, N_EMBD) * 4.0,
                rnd(N_EMBD, N_EMBD),
            ]
        ).astype(np.float32)
        w.add_tensor(p + "attn_qkv.weight", qkv)
        # THE ARM. Centred at 1.0, not 0.0: a shifted bias moves every
        # head's Q and K the same way and so changes the attention
        # pattern as well as the values.
        bias = (rnd(n_embd_qkv) + 1.0).astype(np.float32)
        if qkv_bias:
            w.add_tensor(p + "attn_qkv.bias", bias)

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_EMBD))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        # Width FF, i.e. HALF the declared feed_forward_length.
        w.add_tensor(p + "ffn_gate.weight", rnd(FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, FF))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    # REQUIRED for qwen (qwen.cpp:20) -- no tied-embedding fallback.
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-qkv-bias"]
    main(args[0] if args else "qwen-fixture.gguf", "--no-qkv-bias" not in sys.argv[1:])
