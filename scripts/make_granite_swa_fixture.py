#!/usr/bin/env python3
"""Generate the tiny synthetic `granite_swa` GGUF used by frink's
per-layer-RoPE-pattern coverage test.

`granite_swa` (Granite 4.1) landed upstream after the 2026-08-04
llama.cpp pin and was triaged NEW CODE on 2026-09-19 for TWO small
per-layer tables, of which only one was new here:

  * `{arch}.expert_used_count` read as an ARRAY (`granite-swa.cpp:14`),
    which the loader reads scalar-or-array since the same day, and
  * `{arch}.attention.rope_pattern` (`:43`), one entry per layer,
    nonzero meaning "this layer rotates" (`llama-hparams.cpp:333-343`,
    `llama_hparams::has_rope`, called at `:212`). It is the FIRST
    upstream graph that lets the FILE decide which layers rotate --
    every other per-layer RoPE gate in `frink_models::rope_layers` is
    a rule derived from a literal -- and `grep -rn
    LLM_KV_ATTENTION_ROPE_PATTERN src/models/*.cpp` over the 155 graphs
    is this one line.

Everything else is a seam that already landed, and this fixture carries
each rather than assuming it:

  * Granite's four scalar multipliers (`:7-10`, `logit_scale` REQUIRED
    and DIVIDED at `:192`; `frink_models::scalar_multipliers`),
  * the sliding-window BOOL ARRAY read with `get_arr` (`:17`) and a
    window narrower than the prompt,
  * REQUIRED per-layer attention sinks `{n_head}` (`:81`), which enter
    the softmax as one extra logit per head,
  * the OPTIONAL projection biases (`:79,100-102`,
    `frink_models::proj_bias`): `attn_output.bias` and the three FFN
    ones, all four carried here,
  * NORM RoPE, and a `kq_scale` that comes from `attention.scale`
    rather than `1/sqrt(head_dim)` (`:231`).

Layer 2 is the one the rope pattern switches OFF, and layer 1 is the
one the window array leaves at full attention, so the two arrays
disagree about which layer is special: a loader that read either array
into the other's field would move the logits.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \
        python3 scripts/make_granite_swa_fixture.py OUT.gguf

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "granite_swa"

N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_VOCAB = 48
CTX = 64
SWA_WINDOW = 3
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

LOGIT_SCALE = 2.0
RESIDUAL_SCALE = 0.75
EMBEDDING_SCALE = 1.5
ATTENTION_SCALE = 0.2

IS_SWA = [True, False, True, True]
ROPE_PATTERN = [1, 1, 0, 1]


def main(out_path: str, no_window: bool = False) -> None:
    n_layer = len(IS_SWA)
    rng = np.random.default_rng(0x67A17)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-granite-swa-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(HEAD_DIM)
    # `granite-swa.cpp:18` reads the window REQUIRED; `--no-window` leaves
    # it out, and libllama refuses the file, as frink must (#470).
    if not no_window:
        w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(IS_SWA)
    w.add_logit_scale(LOGIT_SCALE)
    w.add_residual_scale(RESIDUAL_SCALE)
    w.add_embedding_scale(EMBEDDING_SCALE)
    w.add_attention_scale(ATTENTION_SCALE)
    w.add_array(f"{ARCH}.attention.rope_pattern", ROPE_PATTERN)
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
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "attn_output.bias", rnd(N_EMBD))
        # One sink logit per head, drawn wide so it is not a no-op: a
        # sink at -10 leaves the softmax alone and would pass with the
        # feature unimplemented.
        w.add_tensor(p + "attn_sinks.weight", (rng.standard_normal(N_HEAD) * 1.5).astype(np.float32))

        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "ffn_gate.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_up.bias", rnd(N_FF))
        w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD))

    w.add_tensor("output_norm.weight", rnd(N_EMBD) + 1.0)
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "granite-swa-fixture.gguf",
        no_window="--no-window" in sys.argv[2:],
    )
