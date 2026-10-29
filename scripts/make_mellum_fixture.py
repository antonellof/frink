#!/usr/bin/env python3
"""Generate the tiny synthetic `mellum` GGUF used by frink's
per-layer sliding-window ARRAY coverage test.

`mellum` refused as UNAUDITED, triaged NEW CODE, for two things. The
first is the Olmo-3 rule -- `src/models/mellum.cpp:128-142` ropes the
sliding layers with the model's YaRN switched OFF (`freq_scale = 1.0`,
`ext_factor = 0.0`, `attn_factor = 1.0`) while :143-154 rope the full
layers with it on -- and that stays a REFUSAL BY NAME
(`crates/frink-models/src/swa_geometry.rs`) for a file that declares
both a window and a RoPE scaling, which every real Mellum2 export does
(`JetBrains/Mellum2-12B-A2.5B-*`: `sliding_window: 1024`, YaRN factor
16 on `full_attention`). The second is THIS fixture's subject:
`mellum.cpp:12-17` reads `attention.sliding_window_pattern` through the
SCALAR overload of `get_key_or_arr` first and, when that returns false,
through the ARRAY overload -- so a per-layer bool array is the layout
the graph runs, and `conversion/mellum.py:28` ALWAYS writes one
(`[t == "sliding_attention" for t in layer_types]`). Until 2026-09-11
frink refused the array form for every architecture.

`mellum` is the only architecture on the generic path whose graph
HONOURS the array (`crates/frink-models/src/swa_layers.rs` has the
census: the other readers are `gemma4` / `gemma4-assistant` on their
own engine and `dflash`, `step35`, `mimo2`, `cohere2moe`, which refuse
for other things), which is why the seam's honoured branch is
evidenced on a Mellum without a RoPE scaling rather than on one of the
rows this task was aimed at.

**THE ARRAY IS NOT THE PERIOD.** `mellum.cpp:11` seeds `swa_period = 4`
last-dense, [T, T, T, F] over four layers; this file writes
[T, T, F, T], which differs on layers 2 AND 3. A loader that ignored the
array and kept the seed -- which is exactly what llama.cpp does for
`exaone-moe` and what frink must NOT do here -- would window layer 2
and not layer 3, and the golden sees it: the window is 3 tokens against
a six-token prompt, so which layers mask is visible in the logits.

The rest is machinery frink already had, carried so that the row's
other half is measured rather than assumed:

  * NEOX RoPE (`llama_model_rope_type`, LLM_ARCH_MELLUM in the
    `LLAMA_ROPE_TYPE_NEOX` group), `n_rot == head_dim` (:84 asserts it)
  * PER-HEAD Q/K RMSNorms, `{n_embd_head_k}` wide (:50-51), applied
    BEFORE RoPE (:120-124 precede :128)
  * softmax top-k routing with the selected weights RENORMALISED
    (:186 passes `norm_w = true`), no selection bias, no shared expert,
    no scale (`expert_weights_scale` is never read and defaults to
    0.0, which `build_moe_ffn` treats as "none")
  * `expert_feed_forward_length` (:5), which :64 otherwise derives as
    `n_ff / n_expert_used`
  * a sliding window NARROWER than the six-token prompt (:6)

**`--mtp`** appends ONE NextN/MTP block (`blk.4`: a whole Mellum block
plus the `nextn.*` head, `mellum.cpp:78-89`) and writes
`nextn_predict_layers = 1`, with the window array at `block_count`
(the MTP entry false). The common loader subtracts the block before
any graph runs (`llama-model.cpp:1411`, `n_layer()` is the trunk) and
the tensor loader creates it `TENSOR_SKIP` (`nextn_flags`, :35-37), so
libllama's logits are the four-layer file's (#441). The block is drawn
after every other tensor, so the trunk is byte-identical.

**`--no-pattern`** drops `attention.sliding_window_pattern`.
`mellum.cpp:16` reads it through `load_swa_pattern(ml, 4)`
(`llama-model.cpp:3952-3959`), where neither read is required, so the
file runs on the seeded period 4 [T, T, T, F] (#420).

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_mellum_fixture.py OUT.gguf [--mtp] [--no-pattern]

The golden values that go with it are produced by llama.cpp itself
(see `scripts/gptoss_reference_logits.cpp`), not by this script.
"""

import sys

import numpy as np

import gguf

ARCH = "mellum"

N_LAYER = 4
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 8
N_FF = 48
N_FF_EXP = 16
N_EXPERT = 6
N_EXPERT_USED = 2
N_VOCAB = 48
CTX = 64
# Narrower than the six-token prompt, so the sliding layers really mask.
SWA_WINDOW = 3
ROPE_BASE = 10000.0
RMS_EPS = 1e-6
# NOT the seeded period-4 last-dense [T, T, T, F]: see the docstring.
SWA_LAYERS = [True, True, False, True]


def main(out_path: str, mtp: bool = False, pattern: bool = True) -> None:
    rng = np.random.default_rng(0x3E11)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("frink-mellum-fixture")
    w.add_block_count(N_LAYER + (1 if mtp else 0))
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
    w.add_sliding_window(SWA_WINDOW)
    # mellum.py:28, verbatim: a bool per layer.
    if pattern:
        w.add_sliding_window_pattern(SWA_LAYERS + ([False] if mtp else []))
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_feed_forward_length(N_FF_EXP)
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

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        # Wide, so the softmax over six tokens is peaked enough that
        # which positions the window admits is visible.
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD) * 4.0)
        # PER-HEAD: `{n_embd_head_k}` wide (mellum.cpp:50-51).
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.5)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        # gate/up: ne = [n_embd, n_ff_exp, n_expert]
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        # down: ne = [n_ff_exp, n_embd, n_expert]
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    if mtp:
        p = f"blk.{N_LAYER}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM) + 1.0)
        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))
        w.add_tensor(p + "ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))
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
    flags = ("--mtp", "--no-pattern")
    args = [a for a in sys.argv[1:] if a not in flags]
    main(
        args[0] if args else "mellum-fixture.gguf",
        "--mtp" in sys.argv[1:],
        "--no-pattern" not in sys.argv[1:],
    )
