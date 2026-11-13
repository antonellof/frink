#!/usr/bin/env python3
"""Generate the tiny synthetic `laguna` GGUFs used by frink's
gated-attention coverage test: one per gate width.

`laguna` refused as UNAUDITED, triaged NEW CODE, on the learned
attention output gate `src/models/laguna.cpp:124` creates
(`blk.N.attn_gate.weight`) and on a second rotary width (`:50`). The
gate is the same op `afmoe` and `step35` carry -- projected from the
normed input Q/K/V read (`:203,211`), multiplied into the attention
output before `wo` (`:246-260`) -- with two differences that this
fixture pair pins:

  * the activation is SOFTPLUS (`:246`), not sigmoid;
  * the width is read off the stored tensor (`:110-124`): `{n_embd,
    n_head}` is one scalar per head broadcast over `head_dim`
    (`:250-254`, Laguna-XS.2), `{n_embd, n_head * head_dim}` is one
    value per channel (`:256`, Laguna-M.1), and any other width aborts.

Default: the **M.1 shape**. No sliding window (`:34-35` leaves
`swa_type = NONE`), uniform heads, a per-ELEMENT gate, three leading
dense layers in the real model (one here), sigmoid-routed MoE with
`exp_probs_b`, a shared expert sized by
`expert_shared_feed_forward_length` (`:22`), `expert_weights_norm` and
`expert_weights_scale`, per-head QK norm before RoPE (`:215-216`).

`--swa`: the **XS.2 shape**. A window narrower than the prompt with
`set_swa_pattern(4, dense_first=true)` (`:41`: full at `il % 4 == 0`),
`head_count` as a per-layer ARRAY (`conversion/laguna.py:79`;
`laguna.cpp:87-88,176-177` size and run each layer by its own count,
which `crate::layer_shapes` carries), a per-HEAD gate sized by that
layer's own count (`:110`), and a `rope.freq_base_swa` of its own
(`:49`). FOUR LAYERS so that layers 1-3 slide and layer 0 does not.
Deliberately WITHOUT the two things real XS.2 has that frink refuses
by name -- `rope.dimension_count_swa` differing from
`rope.dimension_count` (`:50`) and a YaRN scaling that `:48,184-192`
switch off on the sliding layers -- so that what the golden evidences
is the gate, the width, the per-layer heads and the window, and the
refusals are evidenced by `--swa-rot` and `--swa-yarn` variants that
add exactly one of the two.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_laguna_fixture.py OUT.gguf [--swa|--swa-rot|--swa-yarn]

The golden values that go with them are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "laguna"

N_EMBD = 32
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_FF_EXP = 16
N_FF_SHEXP = 24
N_EXPERT = 6
N_EXPERT_USED = 2
N_DENSE_LEAD = 1
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
ROPE_BASE_SWA = 5000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str, variant: str) -> None:
    swa = variant in ("--swa", "--swa-rot", "--swa-yarn")
    if swa:
        n_layer = 4
        # Full layer 0 has fewer heads than the sliding ones, as XS.2
        # (48 vs 64). KV heads are uniform (laguna.cpp:86).
        heads = [2, 4, 4, 4]
    else:
        n_layer = 3
        heads = [4, 4, 4]

    rng = np.random.default_rng(0x1A60A + (1 if swa else 0))

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name(f"frink-laguna-fixture{'-swa' if swa else ''}")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    if swa:
        w.add_head_count(heads)
    else:
        w.add_head_count(heads[0])
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    if swa:
        w.add_sliding_window(SWA_WINDOW)
        w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    if variant == "--swa-rot":
        # A second rotary width: llama-hparams.cpp:85-91 rotates the
        # sliding layers over this many dims and the full ones over
        # rope.dimension_count. Half the head, so the difference is
        # visible in libllama's own logits.
        w.add_rope_dimension_count_swa(HEAD_DIM // 2)
    if variant == "--swa-yarn":
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(4.0)
        w.add_rope_scaling_orig_ctx_len(CTX // 4)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    # `laguna.cpp:14` reads the count REQUIRED: `--no-dense-lead` leaves
    # it out, and libllama refuses the file (#620).
    if variant != "--no-dense-lead":
        w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
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

    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(n_layer):
        p = f"blk.{il}."
        n_head_il = heads[il]
        n_embd_q = n_head_il * HEAD_DIM
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        # Per head (XS.2): ne = {n_embd, n_head_il}; per element (M.1):
        # ne = {n_embd, n_head_il * head_dim}. Drawn wide so softplus
        # spans both its linear and its log regimes rather than sitting
        # near ln 2 everywhere.
        n_gate = n_head_il if swa else n_embd_q
        w.add_tensor(p + "attn_gate.weight", rnd(n_gate, N_EMBD) * 6.0)

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)

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

        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "laguna-fixture.gguf",
        sys.argv[2] if len(sys.argv) > 2 else "",
    )
