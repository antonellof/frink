#!/usr/bin/env python3
"""Generate the tiny synthetic `nemotron_h` GGUFs used by frink's
Nemotron-H coverage test (`crates/frink-models/tests/nemotron_h_graphs.rs`).

`nemotron_h` is NVIDIA's Nemotron-H (8B / 47B / 56B) and Nemotron-3
Nano dense: a hybrid whose every layer is ONE block with one residual
add (`nemotron-h.cpp:143-158`):

    cur = attn_norm(inpL)                          # :145, every kind
    cur = is_recr(il)   ? mamba2(cur)              # :146-148
        : n_ff(il) == 0 ? attention(cur)           # :149-151
        :                 ffn(cur)                 # :152-153
    inpL = inpL + cur                              # :157

`:9-11` mark a layer recurrent iff `n_head_kv(i) == 0 && n_ff(i) == 0`;
the converter (`conversion/nemotron.py:196-256`, a `GraniteHybridModel`)
writes `head_count` as a scalar, `head_count_kv` as an array with 0 off
the attention layers and `feed_forward_length` as an array with 0 off
the MLP layers. So the three kinds are `(kv > 0, ff 0)`, `(kv 0, ff 0)`
and `(kv 0, ff > 0)`, and on frink's generic layer they are "attention
with no FFN", "Mamba-2 with no FFN" and "no attention, an FFN" -- the
FFN-only layer's pre-norm being `attn_norm` (`:52`, "all blocks use the
attn norm"). The attention layer has NO RoPE (`:181-193` never calls
`ggml_rope_ext`), an optional `attn_output.bias` (`:75`), `kq_scale =
1/sqrt(head_dim)` unless `attention.scale` (`:186-187`). The MLP is the
UNGATED ReLU-squared `down(relu(up(x))^2)` (`:227-231`) with optional
`ffn_up.bias` / `ffn_down.bias` (`:96-97`). The Mamba-2 block is
`build_mamba2_layer` with `ssm_norm` REQUIRED (`:61`) and
`ssm_conv1d.bias` optional-to-the-loader (`:56`; the graph adds it
unconditionally, so frink requires it).

Layers (four): mamba, attention, mlp, mamba (`hybrid_override_pattern
"M*-M"`).

Variants:
  * default        no optional biases, tied output
  * `--biases`     `attn_output.bias`, `ffn_up.bias`, `ffn_down.bias`
  * `--output`     a separate `output.weight`
  * `--moe`        the `nemotron_h_moe` architecture (Nemotron-3 Nano
                   30B-A3B): the FFN layer is a sigmoid MoE (`:206-231`,
                   the gating function a LITERAL) of UNGATED ReLU-squared
                   experts (`ffn_up_exps` / `ffn_down_exps`, no gate;
                   `:82-86`) with the REQUIRED router bias `exp_probs_b`
                   (`:80`), `expert_weights_norm` and
                   `expert_weights_scale` read from the file (`:18-19`;
                   the converter writes `norm_topk_prob` / `routed_
                   scaling_factor`, `conversion/nemotron.py:245-246`),
                   plus an ungated ReLU-squared shared expert
                   (`ffn_up_shexp` / `ffn_down_shexp`, `:88-89,222-227`)
                   added to the routed sum. No latent projection
                   (`moe_latent_size` absent, as Nano's config has it).
  * `--mtp`        with `--moe`: ONE NextN/MTP block appended (`blk.4`,
                   `nemotron-h.cpp:147-195`: the `nextn.*` head, one
                   attention and one MoE sub-layer) and
                   `nextn_predict_layers = 1`, the per-layer arrays at
                   `block_count`. The common loader subtracts it before
                   the graph (`llama-model.cpp:1411`) and the tensor
                   loader creates it `TENSOR_SKIP`, so libllama's logits
                   are the `--moe` file's (#441). Drawn after every
                   other tensor, so the trunk is byte-identical.

Weights are pseudo-random from a fixed seed so the files are byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_nemotron_h_fixture.py OUT.gguf [--biases] [--output] [--moe [--mtp]]

The golden logits that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "nemotron_h"
MOE_ARCH = "nemotron_h_moe"

N_EMBD = 24
N_HEAD = 4
N_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
N_FF = 40
N_VOCAB = 48
CTX = 64
RMS_EPS = 1e-5

# Mamba-2.
D_CONV = 4
D_INNER = 48
D_STATE = 8
SSM_HEADS = 4
N_GROUP = 2
CONV_W = D_INNER + 2 * N_GROUP * D_STATE
D_IN_PROJ = 2 * D_INNER + 2 * N_GROUP * D_STATE + SSM_HEADS

# MoE (`--moe`).
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXP = 16
N_FF_SHEXP = 12
EXPERT_WEIGHTS_SCALE = 2.5

# "M*-M": (head_count_kv, feed_forward_length) per layer.
LAYERS = [(0, 0), (N_KV, 0), (0, N_FF), (0, 0)]


def main(out_path: str, biases: bool, separate_output: bool, moe: bool, mtp: bool = False) -> None:
    if mtp and not moe:
        # `nemotron-h.cpp:157-160` throws on an MTP block with no expert
        # width, and a dense file has none.
        raise SystemExit("--mtp needs --moe")
    layers = [(kv, (N_FF_EXP if moe else N_FF) if ff else 0) for kv, ff in LAYERS]
    n_layer = len(layers)
    # The MTP block's entries in the per-layer arrays: attention heads,
    # and the expert width (`:12-15` keep it out of `is_recr` anyway).
    arrays = layers + ([(N_KV, N_FF_EXP)] if mtp else [])
    rng = np.random.default_rng(0x4E3A)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    arch = MOE_ARCH if moe else ARCH
    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
    w.add_block_count(len(arrays))
    if mtp:
        w.add_nextn_predict_layers(1)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    # conversion/nemotron.py:232-241: `moe_intermediate_size` on the E
    # layers of a MoE export, `intermediate_size` on the - layers.
    w.add_feed_forward_length([ff for _, ff in arrays])
    w.add_head_count(N_HEAD)
    w.add_head_count_kv([kv for kv, _ in arrays])
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    # The converter's key, not the RMS one: NemotronH's config spells it
    # `layer_norm_epsilon`, so `conversion/base.py:1688-1690` writes
    # `attention.layer_norm_epsilon` and no `layer_norm_rms_epsilon`.
    # Since llama.cpp 604cc430 `nemotron-h.cpp:18` REQUIRES that key and
    # falls back to it for the RMS norms (`:19-21`); a fixture carrying
    # only the RMS key no longer loads in libllama at all.
    w.add_layer_norm_eps(RMS_EPS)
    w.add_ssm_conv_kernel(D_CONV)
    w.add_ssm_inner_size(D_INNER)
    w.add_ssm_state_size(D_STATE)
    w.add_ssm_time_step_rank(SSM_HEADS)
    w.add_ssm_group_count(N_GROUP)
    if moe:
        # conversion/nemotron.py:238-250.
        w.add_expert_used_count(N_EXPERT_USED)
        w.add_expert_feed_forward_length(N_FF_EXP)
        w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
        w.add_expert_count(N_EXPERT)
        w.add_expert_shared_count(1)
        w.add_expert_weights_norm(True)
        w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
        w.add_expert_group_count(1)
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

    for il, (nkv, nff) in enumerate(layers):
        p = f"blk.{il}."
        # :52: every kind of layer.
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        if nkv == 0 and nff == 0:
            w.add_tensor(p + "ssm_in.weight", rnd(D_IN_PROJ, N_EMBD) * 2.0)
            w.add_tensor(p + "ssm_conv1d.weight", rnd(CONV_W, D_CONV) * 2.0)
            w.add_tensor(p + "ssm_conv1d.bias", rnd(CONV_W))
            w.add_tensor(p + "ssm_dt.bias", rnd(SSM_HEADS) * 2.0)
            w.add_tensor(
                p + "ssm_a",
                (-(0.5 + 2.0 * rng.random(SSM_HEADS))).astype(np.float32).reshape(SSM_HEADS, 1),
            )
            w.add_tensor(p + "ssm_d", (1.0 + rnd(SSM_HEADS)).astype(np.float32).reshape(SSM_HEADS, 1))
            w.add_tensor(
                p + "ssm_norm.weight",
                (1.0 + rnd(N_GROUP, D_INNER // N_GROUP)).astype(np.float32),
            )
            w.add_tensor(p + "ssm_out.weight", rnd(N_EMBD, D_INNER))
        elif nkv > 0:
            w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_k.weight", rnd(nkv * HEAD_DIM, N_EMBD) * 4.0)
            w.add_tensor(p + "attn_v.weight", rnd(nkv * HEAD_DIM, N_EMBD))
            w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
            if biases:
                w.add_tensor(p + "attn_output.bias", rnd(N_EMBD) * 4.0)
        elif moe:
            # :78-89: the router with its REQUIRED bias, ungated experts,
            # the ungated shared expert. No latent projection.
            w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
            w.add_tensor(
                p + "exp_probs_b.bias",
                (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
            )
            w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, nff, N_EMBD))
            w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, nff))
            w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
            w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))
        else:
            # :94-97: up and down only, ReLU squared.
            w.add_tensor(p + "ffn_up.weight", rnd(nff, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, nff))
            if biases:
                w.add_tensor(p + "ffn_up.bias", rnd(nff) * 2.0)
                w.add_tensor(p + "ffn_down.bias", rnd(N_EMBD) * 2.0)

    w.add_tensor("output_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
    if separate_output:
        w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    if mtp:
        p = f"blk.{n_layer}."
        w.add_tensor(p + "nextn.enorm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "nextn.hnorm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "nextn.eh_proj.weight", rnd(N_EMBD, 2 * N_EMBD))
        w.add_tensor(p + "nextn.shared_head_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "attn_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(N_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(p + "post_attention_norm.weight", (1.0 + rnd(N_EMBD)).astype(np.float32))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "exp_probs_b.bias", rnd(N_EXPERT))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} ({arch}, layers {layers}, biases={biases}, output={separate_output})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "nemotron_h-fixture.gguf",
        "--biases" in sys.argv[1:],
        "--output" in sys.argv[1:],
        "--moe" in sys.argv[1:],
        "--mtp" in sys.argv[1:],
    )
