#!/usr/bin/env python3
"""Generate the tiny synthetic `exaone4` GGUF used by frink's EXAONE-4
coverage test.

`exaone4` is LG AI Research's EXAONE 4.0, and it is NOT the audited
`exaone` row (EXAONE 3.x, `src/models/exaone.cpp`, a plain pre-norm
llama). It refused as UNAUDITED, triaged NEW CODE, for the same blocker
as `olmo2`: `src/models/exaone4.cpp:60-67` is the complete per-layer norm
list and it creates `attn_post_norm`, `attn_q_norm`, `attn_k_norm` and
`ffn_post_norm` and **no `attn_norm` and no `ffn_norm`**. The graph reads
the raw residual at both sublayers -- `cur = inpL` before `build_qkv`
(:118, :124) and `build_ffn(ffn_inp, ...)` on the un-normed
post-attention residual (:159) -- and applies its two norms to each
branch's OUTPUT before the residual add (:152-155, :166-169).

    ffn_inp = x       + attn_post_norm(attn(x))
    out     = ffn_inp + ffn_post_norm(ffn(ffn_inp))

Identical to `olmo2.cpp:92,160-165,169,177-182`. The two rows share ONE
implementation in frink (`crates/frink-models/src/norm.rs`), and
this fixture and `olmo2_tiny.gguf` are what prove they are the same
graph rather than two that look alike.

What differs from `olmo2`, and why this row needs its own fixture:

  * **PER-HEAD QK-norm.** :61-62 sizes `attn_q_norm` and `attn_k_norm`
    `{n_embd_head_k}`, and :127-128 applies them to what `build_qkv` has
    ALREADY reshaped to 3-D (llama-graph.cpp:1656-1658), so the RMS is
    taken per head. `olmo2` norms the 2-D projection over its whole
    width. Same topology, opposite QK-norm style -- and frink picks
    between the two off the weight LENGTH, so this fixture is what
    checks the length it derives from a real per-head file.
  * Applied BEFORE RoPE (:127-128 precede :132-138).
  * `1.0f / sqrtf(float(n_embd_head))` at :145, so
    `ModelConfig::attention_scale` stays `None`.
  * `LLM_FFN_SILU, LLM_FFN_PAR` (:163): plain SwiGLU with `ffn_gate`.
  * NEOX RoPE (`LLM_ARCH_EXAONE4` is in `llama_model_rope_type`'s NEOX
    group, llama-model.cpp).

**THE LAYER COUNT IS THE ARGUMENT**, because llama.cpp switches the
whole SWA machinery on off it with no GGUF key involved: `:4-14` is
`if (hparams.n_layer() == 64)` around `swa_type =
LLAMA_SWA_TYPE_STANDARD`, `n_swa`, `set_swa_pattern(4)` and both SWA
RoPE fields, and `:116` then makes RoPE conditional --
`use_rope = hparams.is_swa(il) || swa_type == NONE`. So a 64-layer
EXAONE-4 (the 32B) gives its FULL-ATTENTION layers **no rotation at
all**, and a 30-layer one (the 1.2B) rotates everything and ignores any
window its file declares.

That is TWO GRAPHS behind one architecture string, so this script emits
both from ONE body rather than growing a copy per size: `--layers 30`
is the 1.2B and `--layers 64` the 32B, and the 32B also gets
`{arch}.attention.sliding_window` -- which the 30-layer file
deliberately does NOT carry, so the pair separates "llama.cpp ignores
the key below 64 layers" from "the key was absent". The 32B's window is
deliberately NARROWER than the six-token prompt, so the mask is
exercised too rather than being a no-op at this size.

The rule itself is `frink-models/src/rope_layers.rs`, shared with
`exaone-moe`, `smollm3`, `smallthinker`, `afmoe` and `llama4`.

Two variants of the 64-layer file, both against libllama at the pin:

  * `--no-window` drops `attention.sliding_window`. `:6` pins `n_swa =
    4096` BEFORE `:14` reads the key as OPTIONAL, so the file still
    slides, at 4096, and its full-attention layers still rotate nothing
    (#440). Over this prompt a 4096 window masks nothing; what the
    variant tells apart is the rotation, which is the visible half.
  * `--mtp` appends ONE NextN/MTP block (`blk.64`, with the `nextn.*`
    head) and writes `nextn_predict_layers = 1`, so `block_count` is
    65. The common loader reads that key BEFORE `load_arch_hparams`
    (`llama-model.cpp:1411`), so `:4`'s `n_layer()` is the TRUNK, 64,
    and the file is the 32B graph; the tensor loader creates the block
    with `TENSOR_SKIP` (`:38-68`), so libllama's logits are the plain
    64-layer file's (#441). The block's weights are drawn after every
    other tensor so the trunk is byte-identical to `--layers 64`.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_exaone4_fixture.py OUT.gguf [--layers 30|64]
        [--no-window] [--mtp]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "exaone4"

# The 1.2B's layer count, and the default. 64 is EXAONE-4 32B, where
# llama.cpp switches SWA on and stops rotating the full-attention
# layers; see the module docstring.
DEFAULT_N_LAYER = 2
# exaone4.cpp:4 -- equality, not a threshold.
SWA_N_LAYER = 64
# exaone4.cpp:6 hardcodes 4096; this fixture declares a window narrower
# than GRAPH_PROMPT so the mask is not a no-op over six tokens. The key
# is read at :16, but only a 64-layer file ever reaches
# `set_swa_pattern`, so only the 64-layer fixture carries it.
SWA_WINDOW = 3
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5


def widths(n_layer: int) -> dict:
    """Per-layer shapes.

    The 32B fixture has to be exactly 64 layers -- `exaone4.cpp:4` tests
    EQUALITY, so no smaller file is that graph -- and 64 layers at the
    1.2B fixture's widths is a 2 MB checked-in file, seven times the
    largest fixture in the tree. So the deep one is drawn narrower. It
    is the same body either way; only the shapes differ, and the graph
    the fixture exercises does not depend on them.
    """
    n_embd, n_head, n_ff, n_vocab = (
        (16, 2, 16, 32) if n_layer == SWA_N_LAYER else (32, 4, 48, 48)
    )
    return {
        "n_embd": n_embd,
        "n_head": n_head,
        "n_head_kv": n_head // 2,
        # exaone4.cpp:54 sizes `wo` `{n_embd, n_embd}` and :92 asserts
        # `n_embd_head_k == n_rot`, so head_dim * n_head must be n_embd.
        "head_dim": n_embd // n_head,
        "n_ff": n_ff,
        "n_vocab": n_vocab,
    }


def main(
    out_path: str, n_layer: int = DEFAULT_N_LAYER, window: bool = True, mtp: bool = False
) -> None:
    rng = np.random.default_rng(0x0E4A04)
    shapes = widths(n_layer)
    N_EMBD = shapes["n_embd"]
    N_HEAD = shapes["n_head"]
    N_HEAD_KV = shapes["n_head_kv"]
    HEAD_DIM = shapes["head_dim"]
    N_FF = shapes["n_ff"]
    N_VOCAB = shapes["n_vocab"]

    def rnd(*shape: int) -> np.ndarray:
        # Small magnitudes keep the synthetic values in a range where a
        # float32 comparison against llama.cpp is meaningful rather than
        # dominated by catastrophic cancellation.
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-exaone4-fixture")
    w.add_block_count(n_layer + (1 if mtp else 0))
    if mtp:
        w.add_nextn_predict_layers(1)
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
    if n_layer == SWA_N_LAYER and window:
        w.add_sliding_window(SWA_WINDOW)
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

    for il in range(n_layer):
        p = f"blk.{il}."

        # NO `attn_norm` and NO `ffn_norm`. See the module docstring.
        # Projections drawn wider than the rest of the file so the
        # softmax over a six-token prompt is not near-uniform; a
        # near-uniform attention cannot see a RoPE sabotage.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # PER-HEAD: head_dim wide, unlike olmo2's whole-vector norms.
        # Centred near 1.5 so applying them whole-vector, or after RoPE,
        # is far from the truth.
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)

        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    if mtp:
        # `exaone4.cpp:38-68` at `i = n_layer`: the attention set, the
        # dense FFN and the post norms with no `rope_freqs`, plus the
        # `nextn.*` head. All TENSOR_SKIP when no MTP context is open.
        p = f"blk.{n_layer}."
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
        w.add_tensor(p + "post_ffw_norm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.eh_proj.weight", rnd(N_EMBD, 2 * N_EMBD))
        w.add_tensor(p + "nextn.enorm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.hnorm.weight", rnd(N_EMBD) + 1.0)
        w.add_tensor(p + "nextn.shared_head_norm.weight", rnd(N_EMBD) + 1.0)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    n_layer = DEFAULT_N_LAYER
    if "--layers" in args:
        i = args.index("--layers")
        n_layer = int(args[i + 1])
        del args[i : i + 2]
    window = "--no-window" not in args
    mtp = "--mtp" in args
    args = [a for a in args if a not in ("--no-window", "--mtp")]
    main(args[0] if args else "exaone4-fixture.gguf", n_layer, window, mtp)
