#!/usr/bin/env python3
"""Generate the tiny synthetic `kolibri1` GGUF for tests/kolibri1_graphs.rs.

`src/models/kolibri1.cpp` (llama.cpp 604cc430) ranks its experts by
`logits + exp_probs_b` BEFORE the sigmoid (:138) and weighs them by the
unbiased sigmoid, where every other biased router ranks
`sigmoid(logits) + exp_probs_b` (llama-graph.cpp:2279). The fixture is
shaped so the two rules pick different experts:

  * the router logits spread wider than the sigmoid's (0, 1) range
    (`ffn_gate_inp` drawn large), so a bias of about one reorders the
    gated scores almost entirely and the raw logits only partly;
  * `exp_probs_b` is drawn uniform in [-1.5, 1.5] per expert and layer.

tests/kolibri1_graphs.rs measures that flipping the rule moves the
logits by more than 1e-2; this script only writes the file.

The rest of the graph, each carried rather than assumed:

  * the sliding-window bool ARRAY (:17) with layer 2 FULL, which is NoPE
    (:88), and a window of 3 under the six-token prompt;
  * per-head QK RMS norms (:47-48), Gemma-2 sandwich norms with the
    post-FFN norm over routed + shared (:41,51,165-167);
  * one ungated shared expert (:59-61) of its own width;
  * sigmoid gating and `expert_weights_norm` from the file (:8-9);
  * three keys the graph never reads, at values that would move the
    logits if they were honoured: `rope.freq_base_swa` (:19 seeds the
    sliding base from the model's), `leading_dense_block_count` and
    `expert_weights_scale` (the converter's base class writes both,
    conversion/afmoe.py:30,36; :52-61 make every layer MoE and :150
    passes `w_scale = 1.0`);
  * NEOX RoPE (llama-model.cpp:3667) at a base of 50, low enough that
    the rotation, and the dead SWA base beside it, is visible.

`--softmax` writes softmax gating with `expert_weights_norm = false`,
the other branch of both keys. `--no-bias` drops `exp_probs_b`, which
:54 creates as REQUIRED: libllama refuses that file, and so does frink.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_kolibri1_fixture.py OUT.gguf [--softmax | --no-bias]
"""

import sys

import numpy as np

import gguf

ARCH = "kolibri1"

N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 50.0
DEAD_ROPE_BASE_SWA = 7.0
RMS_EPS = 1e-5
N_EXPERT = 8
N_EXPERT_USED = 2
FF_EXP = 16
FF_SHEXP = 12

IS_SWA = [True, True, False, True]


def main(out_path: str, softmax: bool, no_bias: bool) -> None:
    n_layer = len(IS_SWA)
    rng = np.random.default_rng(0x0414)

    def rnd(*shape: int, scale: float = 0.25) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-kolibri1-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(FF_EXP)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(DEAD_ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(IS_SWA)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(FF_EXP)
    w.add_expert_shared_count(1)
    w.add_expert_shared_feed_forward_length(FF_SHEXP)
    w.add_expert_weights_norm(not softmax)
    w.add_expert_gating_func(
        gguf.ExpertGatingFuncType.SOFTMAX if softmax else gguf.ExpertGatingFuncType.SIGMOID
    )
    w.add_leading_dense_block_count(1)
    w.add_expert_weights_scale(2.5)
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
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 3.0)
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        # Logits with a spread of about 2.5, against a bias of about one.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD, scale=0.45))
        bias = rng.uniform(-1.5, 1.5, N_EXPERT).astype(np.float32)
        if not no_bias:
            w.add_tensor(p + "exp_probs_b.bias", bias)
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, FF_EXP))
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, FF_SHEXP))
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "kolibri1-fixture.gguf",
        "--softmax" in sys.argv[2:],
        "--no-bias" in sys.argv[2:],
    )
