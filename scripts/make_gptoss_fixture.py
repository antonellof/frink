#!/usr/bin/env python3
"""Generate the tiny synthetic `gpt-oss` GGUF used by frink's gpt-oss
coverage tests, plus its metadata sidecar.

The fixture is deliberately small (2 layers, 32-wide, 4 experts) but
carries *every* structural feature of the real gpt-oss graph that the
generic GQA decoder does not otherwise see:

  * `blk.N.attn_sinks.weight`   -- per-head attention sink logits
  * `blk.N.attn_output.bias`    -- attention output bias
  * `blk.N.post_attention_norm` -- gpt-oss's *pre-FFN* norm (not Gemma's
                                   post-attention norm; see the frink
                                   loader for why the distinction bites)
  * `blk.N.ffn_gate_inp.bias`   -- router bias
  * `blk.N.ffn_{gate,up,down}_exps.bias` -- per-expert biases
  * alternating sliding window  -- layer 0 windowed, layer 1 full

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_gptoss_fixture.py OUT.gguf

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "gpt-oss"

N_LAYER = 2
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16
N_VOCAB = 48
SLIDING_WINDOW = 4
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def main(out_path: str, window: bool = True) -> None:
    # `window=False` (`--no-window`) drops `attention.sliding_window`:
    # refusal evidence for #458. `openai-moe.cpp:6` reads it REQUIRED,
    # so libllama does not load the file, and frink refuses it by name.
    rng = np.random.default_rng(0xF3110)

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic logits in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-gptoss-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    if window:
        w.add_sliding_window(SLIDING_WINDOW)
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

    # ne = [n_embd, n_vocab] -> numpy [n_vocab, n_embd]
    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))

        # One sink logit per query head. Spread across zero so the sink
        # both wins and loses the running max somewhere in the fixture.
        w.add_tensor(p + "attn_sinks.weight", rnd(N_HEAD) * 4.0)

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "ffn_gate_inp.bias", rnd(N_EXPERT))

        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        w.add_tensor(p + "ffn_gate_exps.bias", rnd(N_EXPERT, N_FF_EXP))
        w.add_tensor(p + "ffn_up_exps.bias", rnd(N_EXPERT, N_FF_EXP))
        w.add_tensor(p + "ffn_down_exps.bias", rnd(N_EXPERT, N_EMBD))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-window"]
    main(args[0] if args else "gptoss-fixture.gguf", "--no-window" not in sys.argv[1:])
