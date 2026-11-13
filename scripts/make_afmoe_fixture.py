#!/usr/bin/env python3
"""Generate the tiny synthetic `afmoe` GGUF used by frink's
gated-attention coverage test.

`afmoe` refused as UNAUDITED, triaged NEW CODE, for one blocker after
the per-layer RoPE gate closed: **the learned attention output gate.**
`src/models/afmoe.cpp:73` creates `wqkv_gate` (`blk.N.attn_gate.weight`,
`{n_embd, n_head * n_embd_head}`), `:154` projects it from the SAME
normed input Q/K/V read, and `:183-185` multiplies the attention output
by `sigmoid(gate)` BEFORE `wo`. `laguna` and `step35` carry the same
op with a different activation or width; frink implements the three
once (`crates/frink-models/src/attn_gate.rs`) and this fixture is the
sigmoid / per-element / required corner of it.

The one other afmoe-specific fact in the graph is `:120`: the
embeddings are scaled by `sqrt(n_embd)` in the graph, from arithmetic
rather than from a key. Measured over all 140 `src/models/*.cpp`,
`afmoe` is the ONLY non-Gemma graph that does this
(`capability::embeddings_scaled_by_sqrt_n_embd`).

**FOUR LAYERS ARE NOT AN ACCIDENT.** `:15-19` gives a windowed afmoe
`set_swa_pattern(4)`, last-dense, and `:137-138` skips RoPE where
`(il + 1) % 4 == 0`. With fewer than four layers every layer would
slide and rotate and the file would be evidence about neither. Layer 3
is the full-attention, unrotated, MoE layer.

The rest of the graph is machinery frink already had, and the fixture
carries all of it so that "the rest is generic" is measured rather than
asserted:

  * dual norms on both blocks: `attn_norm` + `post_attention_norm`
    (:61-62, :141-144, :194-197) and `ffn_norm` + `post_ffw_norm`
    (:76-77, :208-211, :257-260)
  * PER-HEAD Q/K norms, `{n_embd_head_k}` wide (:69-70), before RoPE
  * a leading DENSE layer (`leading_dense_block_count = 1`, :79)
  * `blk.N.exp_probs_b.bias`, REQUIRED at :82, drawn large enough to
    reorder the top-k against the unbiased scores
  * one shared expert sized `n_ff_exp * n_expert_shared` (:91)
  * `expert_weights_scale` != 1 and `expert_weights_norm` (:9-10)
  * NO `expert_gating_func` key: :8 reads it as optional and :29-30
    defaults it to SIGMOID, so the file declaring nothing is the case
    that tells the default from a key
  * a sliding window NARROWER than the six-token prompt, and a
    `rope.freq_base_swa` different from the model's base (:21-23)

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_afmoe_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "afmoe"

N_LAYER = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_FF_EXP = 16
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_DENSE_LEAD = 1
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str, no_shared_count: bool = False) -> None:
    rng = np.random.default_rng(0xAF0E)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-afmoe-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_sliding_window(SWA_WINDOW)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    # `afmoe.cpp:7` reads the count REQUIRED: `--no-shared-count` leaves
    # it out, and libllama refuses the file (#620).
    if not no_shared_count:
        w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    # Deliberately NO add_expert_gating_func: afmoe.cpp:29-30 defaults
    # to SIGMOID when the key is absent.
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
    n_ff_shexp = N_FF_EXP * N_EXPERT_SHARED

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # ne = {n_embd, n_head * head_dim} -> numpy [n_head * head_dim, n_embd].
        # Drawn wide so the sigmoid leaves the ~0.5 plateau: a gate that
        # is 0.5 everywhere is a uniform scale the test could not see.
        w.add_tensor(p + "attn_gate.weight", rnd(n_embd_q, N_EMBD) * 6.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)

        if il < N_DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(n_ff_shexp, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(n_ff_shexp, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, n_ff_shexp))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "afmoe-fixture.gguf",
        no_shared_count="--no-shared-count" in sys.argv[2:],
    )
