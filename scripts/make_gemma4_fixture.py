#!/usr/bin/env python3
"""Generate the tiny synthetic `gemma4` GGUF that holds frink's
dedicated Gemma-4 engine to libllama across the sliding window (#634).

Gemma-4 runs on `frink_models::gemma4_engine`, not the generic decoder,
and until this file no libllama answer covered it below a real
checkpoint. A real checkpoint cannot arbitrate a long prompt: on
Gemma-4-E2B Q4_K_M a 1343-token prose prompt reads KL 0.12 against
libllama at the pin, while on the same file dequantized to f32 the two
engines agree to 2.8e-6 in prefill and 8.2e-5 over 16 decode steps.
The quantized distance is the 8-bit activation quantizers: an input
an ulp apart (from a different summation order) moves a whole quant
step, and `attention_scale = 1` amplifies each flip layer by layer.
This fixture is f32 throughout, so what it compares is the graph.

What `src/models/gemma4.cpp` reads, and how this fixture makes each
one bind:

  * `sliding_window_pattern`, a per-layer BOOL array (`:6`), and a
    window of 4 (`:16`), narrower than the oracle's six-token prompt,
    so prefill already masks keys and every one of the 32 decode steps
    reads past the window (`is_masked_swa`, STANDARD).
  * `shared_kv_layers` (`:8-11`): the last three layers own no KV and
    read the cache of `n_layer_kv_from_start - (is_swa ? 2 : 1)`
    (`llama-model.cpp`, the `reuse` callback): layers 3 and 4
    (sliding) read layer 1, layer 5 (full) reads layer 2. A reader
    sent to the wrong slot reads a different head width and refuses,
    or the same width and moves the logits.
  * different head widths for sliding and full layers
    (`key_length_swa`, `:19-20`), 8 against 16.
  * `rope_freqs` on the full layers only (`:94-97`, `:210-213`), with
    factors that are not all one, and `rope.freq_base_swa` (`:14`)
    different from `rope.freq_base`.
  * the Q/K RMS norms with weights, V's weightless RMS norm (`:264`),
    and `f_attention_scale = 1` (`:12`).
  * per-layer embeddings (`:59-61`, `:137-139`, `:375-396`): the token's
    `per_layer_token_embd` row, `per_layer_model_proj`, its norm, and
    each layer's `inp_gate` / `proj` / `post_norm`.
  * `layer_output_scale` (`:400-403`) and `final_logit_softcapping`.

`--no-full-attn-v` leaves `attn_v` out of the full layer that owns a
cache (layer 2), which `load_arch_tensors` allows (`TENSOR_NOT_REQUIRED`)
and the graph then reads K's projection as V, ahead of K's norm and
RoPE (`Vcur = Kcur`, `:253-255`). The host body used K after both.

Every head width is also the RoPE width (`rope.dimension_count` and
`_swa`), unless `--rope-dim N` is given (#635). It writes
`rope.dimension_count = N` and NO `_swa` key, so llama.cpp rotates the
first N channels of every head: `n_rot_full = N`, and `n_rot_swa`
defaults to `n_rot_full` (`llama-model.cpp`, `load_hparams`), not to
the sliding head width. At N = 4 the full layers rotate 4 of 16 and the
sliding ones 4 of 8, so a body that rotates whole heads, defaults the
sliding width to its head, or reads the wrong `rope_freqs` bands (ggml
reads the first N/2) moves the logits.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_gemma4_fixture.py OUT.gguf [--no-full-attn-v] [--rope-dim N]
"""

import sys

import numpy as np

import gguf

ARCH = "gemma4"

N_EMBD = 32
N_HEAD = 2
N_HEAD_KV = 1
HEAD_DIM_FULL = 16
HEAD_DIM_SWA = 8
N_FF = 48
N_VOCAB = 48
N_EMBD_PER_LAYER = 4
CTX = 64
SWA_WINDOW = 4
ROPE_BASE = 1_000_000.0
ROPE_BASE_SWA = 10_000.0
RMS_EPS = 1e-6
SOFTCAP = 30.0

# sliding, sliding, full, twice; the last three share KV.
IS_SWA = [True, True, False, True, True, False]
SHARED_KV_LAYERS = 3


def main(out_path: str, no_full_attn_v: bool = False, rope_dim: int | None = None) -> None:
    n_layer = len(IS_SWA)
    n_kv_from_start = n_layer - SHARED_KV_LAYERS
    rng = np.random.default_rng(0x6E4A4)

    def rnd(*shape: int, scale: float = 0.25) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n: int, centre: float = 1.0) -> np.ndarray:
        return (centre + rng.standard_normal(n) * 0.1).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-gemma4-fixture")
    w.add_block_count(n_layer)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM_FULL)
    w.add_value_length(HEAD_DIM_FULL)
    w.add_key_length_swa(HEAD_DIM_SWA)
    w.add_value_length_swa(HEAD_DIM_SWA)
    if rope_dim is None:
        w.add_rope_dimension_count(HEAD_DIM_FULL)
        w.add_uint32(f"{ARCH}.rope.dimension_count_swa", HEAD_DIM_SWA)
    else:
        w.add_rope_dimension_count(rope_dim)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_freq_base_swa(ROPE_BASE_SWA)
    w.add_sliding_window(SWA_WINDOW)
    w.add_sliding_window_pattern(IS_SWA)
    w.add_shared_kv_layers(SHARED_KV_LAYERS)
    w.add_embedding_length_per_layer_input(N_EMBD_PER_LAYER)
    w.add_final_logit_softcapping(SOFTCAP)
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
    ple = N_EMBD_PER_LAYER * n_layer
    w.add_tensor("per_layer_token_embd.weight", rnd(N_VOCAB, ple))
    w.add_tensor("per_layer_model_proj.weight", rnd(ple, N_EMBD))
    w.add_tensor("per_layer_proj_norm.weight", norm(N_EMBD_PER_LAYER))
    # Half the bands rotate at their own rate and half three times
    # slower: a body that drops the factors, or hands them to a sliding
    # layer, rotates differently.
    w.add_tensor(
        "rope_freqs.weight",
        np.array([1.0 if i % 2 == 0 else 3.0 for i in range(HEAD_DIM_FULL // 2)], dtype=np.float32),
    )

    for il in range(n_layer):
        p = f"blk.{il}."
        hd = HEAD_DIM_SWA if IS_SWA[il] else HEAD_DIM_FULL
        w.add_tensor(p + "attn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * hd, N_EMBD))
        # Small Q norm: with `attention_scale = 1` a unit-RMS Q.K scores
        # ~sqrt(head_dim), and a softmax that peaked on one key would
        # hide which keys the window let in.
        w.add_tensor(p + "attn_q_norm.weight", norm(hd, 0.3))
        if il < n_kv_from_start:
            w.add_tensor(p + "attn_k.weight", rnd(N_HEAD_KV * hd, N_EMBD))
            if not (no_full_attn_v and not IS_SWA[il]):
                w.add_tensor(p + "attn_v.weight", rnd(N_HEAD_KV * hd, N_EMBD))
            w.add_tensor(p + "attn_k_norm.weight", norm(hd, 0.6))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * hd))
        w.add_tensor(p + "post_attention_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "ffn_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "post_ffw_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "inp_gate.weight", rnd(N_EMBD_PER_LAYER, N_EMBD))
        w.add_tensor(p + "proj.weight", rnd(N_EMBD, N_EMBD_PER_LAYER))
        w.add_tensor(p + "post_norm.weight", norm(N_EMBD))
        w.add_tensor(p + "layer_output_scale.weight", np.array([0.6 + 0.05 * il], dtype=np.float32))

    w.add_tensor("output_norm.weight", norm(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


def parse(argv: list[str]) -> tuple[bool, int | None]:
    no_full_attn_v = False
    rope_dim = None
    i = 0
    while i < len(argv):
        if argv[i] == "--no-full-attn-v":
            no_full_attn_v = True
            i += 1
        elif argv[i] == "--rope-dim" and i + 1 < len(argv):
            rope_dim = int(argv[i + 1])
            i += 2
        else:
            raise ValueError(argv[i])
    return no_full_attn_v, rope_dim


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("no output path")
        flags = parse(sys.argv[2:])
    except ValueError:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], *flags)
