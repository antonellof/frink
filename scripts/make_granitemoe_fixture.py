#!/usr/bin/env python3
"""Generate the tiny synthetic `granitemoe` GGUF used by frink's
Granite coverage test -- and, under `--arch`, the byte-for-byte
equivalent under frink's `granite-moe` alias.

`granitemoe` is IBM's Granite 3.x MoE decoder (and, with
`expert_shared_feed_forward_length`, GraniteMoeShared).
`src/models/granite-moe.cpp` has NO graph of its own: `models.h:1583-1591`
declares `using graph = llama_model_granite::graph`, so `granitemoe` runs
the DENSE row's graph and takes its MoE branch on
`layers[il].ffn_gate_inp != nullptr` (granite.cpp:246). The two rows
differ in the FFN and in nothing else -- same four multipliers, same
residual placement, same `1.0f / f_logit_scale` on the logits -- which is
why frink implements the multipliers once, parameterised, rather than
once per architecture.

What this fixture drives beyond `make_granite_fixture.py`:

  * the MoE branch (granite.cpp:266-277): `build_moe_ffn` with
    `LLM_FFN_SILU`, `norm_w = true` (the top-k weights ARE renormalised
    after selection), `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, and
    `hparams.expert_weights_scale` -- which `granite-moe.cpp:3-24` never
    reads, so it keeps its `0.0f` default and `llama-graph.cpp:2070`
    treats that as no scaling at all. The fixture therefore writes NO
    `expert_weights_scale` key and frink must default it to 1.0.
  * the expert width. granite.cpp:99-102 sizes the expert tensors from
    `n_ff`, NOT from `n_ff_exp`, so this file writes only
    `feed_forward_length` and frink has to size its experts from that.
  * the SHARED expert (granite.cpp:105-110), created only when
    `n_ff_shexp > 0` and added to the routed output UNGATED
    (:280-287) -- unlike qwen2moe, whose shared expert has a sigmoid
    gate. `expert_shared_count` is never written by the Granite
    converter, so frink has to infer one shared expert from
    `blk.0.ffn_gate_shexp.weight`.

**The `granite-moe` alias.** No llama.cpp GGUF spells the architecture
that way -- `llama-arch.cpp:101` is `granitemoe` -- so there is no
libllama golden for it and there never can be. `--arch granite-moe`
writes the SAME weights and the SAME hyper-parameters under the alias's
key prefix, and `tests/granite_family_graphs.rs` asserts it produces
`granitemoe`'s libllama-golden logits to the same tolerance. That is what
carries the evidence across: the alias is only ever safe because it is
the same row, and a test that reads both files against one golden is the
only thing that can keep it that way.

**`--no-rope`** writes the same file plus `rope.scaling.finetuned =
false`, the key `granite.cpp:33-37` reads as a switch for RoPE itself.
`granite-moe.cpp`'s `load_arch_hparams` does NOT read it, so libllama
keeps `rope_pattern` at the all-ones seed (`llama-model.cpp:1464`) and
rotates this file exactly as it rotates the one without the key (#439).
A reader that honoured the key here would run it unrotated, and the
golden is far from that.

Weights are pseudo-random from a fixed seed so the file is byte-stable,
and the seed does not depend on the architecture string.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_granitemoe_fixture.py OUT.gguf
    PYTHONPATH=... python3 scripts/make_granitemoe_fixture.py OUT.gguf \\
        --arch granite-moe
    PYTHONPATH=... python3 scripts/make_granitemoe_fixture.py OUT.gguf --no-rope

The golden logits that go with the `granitemoe` file are produced by
llama.cpp itself (see `scripts/gptoss_reference_logits.cpp`), not by
this script.
"""

import sys

import numpy as np

import gguf

N_LAYER = 2
N_EMBD = 24
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = N_EMBD // N_HEAD
# The ROUTED expert width. granite.cpp:99-102 sizes the expert tensors
# from n_ff; there is deliberately no `expert_feed_forward_length` key.
N_FF = 16
N_FF_SHEXP = 20
N_EXPERT = 4
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# The same four multipliers as the dense row, at the same values, so a
# divergence between the two fixtures cannot be blamed on the scaling.
LOGIT_SCALE = 2.5
RESIDUAL_SCALE = 0.6
EMBEDDING_SCALE = 2.0
ATTENTION_SCALE = 0.9


def main(out_path: str, arch: str, no_rope: bool = False) -> None:
    rng = np.random.default_rng(0x62471)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, arch)
    w.add_name(f"frink-{arch}-fixture")
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
    w.add_rope_dimension_count(HEAD_DIM)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    # GraniteMoeShared's one extra key (granite-moe.cpp:23).
    # `expert_shared_count` is deliberately absent: the Granite converter
    # never writes it, so frink has to infer the single shared expert
    # from `blk.0.ffn_gate_shexp.weight`.
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_logit_scale(LOGIT_SCALE)
    w.add_residual_scale(RESIDUAL_SCALE)
    w.add_embedding_scale(EMBEDDING_SCALE)
    w.add_attention_scale(ATTENTION_SCALE)
    if no_rope:
        w.add_rope_scaling_finetuned(False)

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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        # Wide Q/K: see `make_granite_fixture.py` for why.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))

        # Router logits drawn wide enough that the top-2 of four is a
        # real decision: a near-uniform router would make the whole
        # routed half of this fixture untestable.
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD) * 4.0)
        # gate/up: ne = [n_embd, n_ff, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF, N_EMBD))
        # down: ne = [n_ff, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF))

        # The shared expert, ungated (granite.cpp:280-287).
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(N_FF_SHEXP, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, N_FF_SHEXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (arch={arch})")


if __name__ == "__main__":
    argv = sys.argv[1:]
    arch = "granitemoe"
    if "--arch" in argv:
        i = argv.index("--arch")
        arch = argv[i + 1]
        del argv[i : i + 2]
    no_rope = "--no-rope" in argv
    argv = [a for a in argv if a != "--no-rope"]
    main(argv[0] if argv else "granitemoe-fixture.gguf", arch, no_rope)
