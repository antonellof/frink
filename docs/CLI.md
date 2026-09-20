# CLI

`frink` accepts common [llama.cpp](https://github.com/ggerganov/llama.cpp)
completion flags. Top-level `-m` / `-p` work without typing `run`
(rewritten to `frink run …`).

```bash
cargo build --release -p frink-cli --features metal   # macOS Metal + CPU
cargo build --release -p frink-cli                    # CPU only
```

Binary: `./target/release/frink`. One executable covers every backend
compiled into it. Pick one at runtime with `-dev` / `-ngl`.

## Completion (`run`)

### Quick examples

```bash
# Greedy completion (raw prompt)
./target/release/frink -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat-tuned wrap (default when GGUF has tokenizer.chat_template)
./target/release/frink -m models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf \
  -p "What is 2+2?" -n 64 --temp 0 \
  --system "Answer briefly."

# Prompt from file + escapes
./target/release/frink -m model.gguf -f prompt.txt -e -n 128

# Sampling
./target/release/frink -m model.gguf -p "Once upon a time" \
  -n 256 --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 -s 42

# Threads + context
./target/release/frink -m model.gguf -p "Hi" -n 64 -t 8 -c 4096

# Largest context that fits, chosen before the weights load. The
# arithmetic behind the number is printed to stderr.
./target/release/frink -m model.gguf -p "Hi" -n 64 -c auto

# List devices, then select Metal (requires a --features metal build)
./target/release/frink --list-devices
./target/release/frink -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 64 --temp 0 -dev metal -ngl all

# Force CPU with the same Metal-capable executable
./target/release/frink -m model.gguf -p "Hello" -n 64 -dev none -ngl 0
```

Same via explicit subcommand: `frink run -m …`.

### Completion flags

| Flag | Notes |
|---|---|
| `-m` / `--model` | GGUF path |
| `-hf` / `--hf-repo` | Hugging Face repo, `user/repo[:QUANT]`. Fetched to the cache on first use. Mutually exclusive with `-m`, see below |
| `-p` / `--prompt` | Prompt string |
| `-f` / `--file` | Prompt from file |
| `-n` / `--n-predict` | `-1` = fill remaining context |
| `-c` / `--ctx-size` | `auto` = largest that fits the device memory budget, `0` = GGUF `{arch}.context_length` (else 4096), or a token count |
| `--strict-budget` | Stop with an error when the pre-load budget says the context will not fit (default: warn and continue) |
| `-t` / `--threads` | Sets `RAYON_NUM_THREADS` |
| `--temp` | `0` = greedy. Default `0.8`, llama.cpp's |
| `--top-k` | `0` = off. Default `40`, llama.cpp's |
| `--top-p` | Nucleus sampling. Default `0.95`, llama.cpp's |
| `--min-p` | Drop every candidate less than this fraction as likely as the most likely one. `0.0` = off. Default `0.05`, llama.cpp's (`common/common.h:231`) |
| `--repeat-penalty` | `1.0` = off. Default `1.1`; llama.cpp defaults this one to `1.0` |
| `--presence-penalty` | Penalise a token for having appeared at all. `0.0` = off, llama.cpp's default. The engine and the HTTP API always supported this; the CLI used to hardcode it to zero |
| `--frequency-penalty` | Penalise a token in proportion to how often it has appeared. `0.0` = off |
| `--hf-file` | Exact filename inside `--hf-repo`, llama.cpp's `-hff`. Skips quant resolution entirely |
| `--repeat-last-n` | How many recent tokens the repetition / presence / frequency penalties consider. `0` = penalties off. Default `64`, llama.cpp's (`common/common.h:238`) |
| `--typical` / `--typical-p` | Locally typical sampling. Keeps the candidates nearest the distribution's entropy, so it can drop the MOST likely token. `1.0` = off, llama.cpp's default (`common/common.h:230`) |
| `--top-nsigma` / `--top-n-sigma` | Mask every candidate more than `n` standard deviations of the logits below the maximum. `-1.0` = off, llama.cpp's default. `0.0` is a no-op, not greedy |
| `--xtc-probability` | Chance that XTC removes the top candidates on a token. `0.0` = off, llama.cpp's default |
| `--xtc-threshold` | Probability a candidate must reach before XTC may remove it. Default `0.1`; **above `0.5` disables XTC**, as upstream |
| `--dry-multiplier` | DRY sequence-repetition penalty. `0.0` = off, llama.cpp's default |
| `--dry-base` | Base of DRY's exponential. Default `1.75`; below `1.0` disables DRY |
| `--dry-allowed-length` | Repetitions this long or shorter are free. Default `2` |
| `--dry-penalty-last-n` | How many recent tokens DRY scans. `0` = off, `-1` = the context size (default) |
| `--dry-sequence-breaker` | Repeatable. A string DRY refuses to look past. Giving any CLEARS llama.cpp's defaults (`\n`, `:`, `"`, `*`), and the literal `none` clears them outright |
| `-s` / `--seed` | `-1` = time-based |
| `--samplers` / `--sampler-seq` | Order the chain runs in, semicolon-separated. A sampler frink lacks is refused by name, see below |
| `--grammar` | Constrain generation to a GBNF grammar, llama.cpp's `--grammar` |
| `--grammar-file` | Read the GBNF grammar from a file, llama.cpp's `--grammar-file` |
| `-j` / `--json-schema` | Constrain generation to a JSON Schema, converted to GBNF. llama.cpp's `-j` |
| `-dev` / `--device` | `auto`, `none`, `cpu`, `metal`, or `cuda` |
| `--list-devices` | Print compiled, detected devices and exit |
| `-ngl` / `--gpu-layers` / `--n-gpu-layers` | `0`, `auto`, `all`, or a count at/above the layer count. A *partial* count is refused, see below |
| `--ctk` | KV dtype, llama.cpp's set: `f32`, `f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, plus frink's `fp8`. Served: `f16`, `q8_0`, `fp8` (the Q8_0 wire) and `q4_0` (4 bits with a Hadamard rotation on K where the head width allows it); the rest are accepted and reported as falling back. A value outside the set is refused, as llama.cpp refuses it. **Metal only**, see below. Sets `FRINK_CTK` |
| `-d` / `--model-draft FILE` | A smaller checkpoint from the SAME family and tokenizer as the target, used as a speculative drafter. The output is exactly what the target would have written alone. Refused with a grammar, without a prompt, for a recurrent (Mamba) target or draft, and for a device-resident draft KV, see below |
| `--draft-max N` / `--draft` | Tokens the drafter proposes per verification step (llama.cpp's spelling). Default 5 |
| `--draft-p-min P` | Stop drafting once the drafter's own probability for the token it just sampled falls below `P` (llama.cpp's spelling). Default 0.75 |
| `--lora FILE` | A LoRA adapter GGUF (what `convert_lora_to_gguf.py` writes), applied at scale 1. Repeatable; comma-separated as llama.cpp accepts it. See below |
| `--lora-scaled FILE:SCALE` | The same with a scale. Adapters are numbered in the order given, every `--lora` before every `--lora-scaled` |
| `--system` | Chat mode only |
| `--no-cnv` | Skip chat-template wrap |
| `-e` / `--escape` | Expand `\n` `\t` `\r` `\\` in `-p`. **On by default**, as in llama.cpp |
| `--no-escape` | Pass `-p` through literally |
| `--ignore-eos` | Always emit up to `-n` |
| `--verbose-prompt` | Print final prompt to stderr |
| `--mtp` | Errors: MTP draft heads not loaded from GGUF yet |

Stderr prints load and throughput timings. Generated text goes to stdout.

**Structured output.** `--grammar`, `--grammar-file` and `-j` all end
at the same stack machine, which masks every token that cannot continue
a valid string. A schema is compiled to GBNF first, so the two paths
share one enforcer rather than two that drift. The constraint holds per
token, so there is no retry loop and no repair pass; the same machine
serves `response_format` and `tool_choice` on the HTTP API
([docs/API.md](API.md)).

**`--ctk` only binds on Metal.** Only the Metal KV store has a
selectable dtype. On CPU and CUDA the KV cache is the host `Vec<f32>`,
so `--ctk f16` there is accepted, ignored, and reported as ignored by
the startup banner. That matters for memory: f32 doubles the KV bytes
per token, which is why a model that fits at its full context on Metal
can need `--ctx-size auto` on CPU. `frink inspect-plan` prices both.

`--samplers` (llama.cpp's, also `--sampler-seq`) chooses the ORDER, as a
semicolon-separated list. The default is llama.cpp's own default chain,
spelled out:
`--samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"`.
llama.cpp's aliases parse, so `top-k`, `nucleus`, `temp` and `typical`
all work, and an upstream command line pastes in unchanged.

A sampler frink does not implement is **refused by name with the
reason**, never skipped. That is now only `mirostat` and `infill`:
`mirostat` REPLACES the chain upstream rather than joining it, so there
is no position in this order that would honour it, and `infill` needs
the model's FIM tokens. A caller who asked for one and silently got a
chain without it was handed a different sampler than the one they
requested.

Order is not cosmetic, which is why it is worth exposing and why getting
it wrong is a silent quality regression rather than an error. Each
filter renormalises over the survivors of the last, so moving a step
changes what the next step can see. This project shipped that bug once:
temperature ran first, and top-p then summed probabilities temperature
had already reshaped.

**The sampler chain is llama.cpp's, in llama.cpp's order.** Penalties,
DRY, top-n-sigma, top-k, typical-p, top-p, min-p, XTC, and
**temperature last** (`common/common.h:259-269`; frink
`crates/frink-models/src/sampling.rs`'s `filtered_distribution`). Every
one of those nine runs by default, and the four with no OpenAI
equivalent sit at neutral values that make them exact no-ops, so a
command line that does not name them samples what it always did.
Frink used to divide by the temperature first and filter afterwards,
which keeps a different candidate set for the same flags: top-p selects
the smallest set summing to `p`, and temperature changes the
probabilities being summed. The repetition penalty is applied **once per
candidate**, not once per occurrence in the history, so a token seen `n`
times is no longer scaled by `penalty^n`. Both were live on every
`frink run` at the defaults above.

**Speculative decoding** comes in two shapes, and only one of them is
a demo.

`-d` / `--model-draft FILE` is the real one: a second, smaller GGUF
from the same family and tokenizer proposes tokens and the target
verifies them in one batched pass. Decode reads every weight of the
target per token, so bandwidth divided by model bytes is a hard
ceiling; a drafter changes what is read per token rather than how fast.
**The text is exactly what the target would have written alone** -- the
rejection rule is lossless at every temperature, and the drafter can
only ever save a forward pass.

`--draft-max` (default 5) sets how many tokens the drafter proposes
per verification step, and `--draft-p-min` (default 0.75) stops it
early once its own confidence drops. The second is not a micro-tuning
knob: a guessing drafter is worse than no drafter, because the target
pays for the position either way AND a rejection discards every
position after it, so proposing a token the drafter does not believe
in costs twice.

It refuses rather than guessing in five cases, each for a reason:

- with `--grammar` / `--grammar-file` / `-j`, because the draft is
  verified against the target's own sampler and the grammar machine's
  state would have to be rolled back with it;
- without a prompt, since there is no history to draft from;
- when the TARGET has recurrent (Mamba) layers, because such a state is
  a reduction over the whole prefix and cannot be rolled back to a
  middle position (`docs/MODELS.md` names the affected families);
- when the DRAFT does, for the same reason;
- when the drafter's KV would live on the DEVICE (a Metal or CUDA
  drafter), because a drafter that cannot see its own rows cannot roll
  back the ones the target rejected. `--device cpu` runs it. This one
  was found by running it rather than by reading: on Metal it panicked
  mid-answer, after the first block had already been printed.

`frink speculative` is the demo: it matches n-grams against the history,
has no draft model, and runs on synthetic random weights, so the hit
rate it prints tells you nothing about a real drafter. What it does
report honestly is acceptance length and the per-position accept rate
alongside the call counts.

Neither reaches `frink-server` yet: the HTTP API's speculation fields
in `usage` are a wire contract with nothing populating them
(`docs/API.md`), and the row that changes that is
`docs/plans/server-speculative-decoding.md`.

Verification uses the speculative-sampling rejection rule, so it stays
lossless at any temperature rather than only at `--temp 0`. Real drafters
plug in through the `Drafter` trait in `frink_models::speculative`.
`--mtp` is reserved for MiniMax/GLM MTP draft heads
(`num_nextn_predict_layers`) and errors today.

`--device none` (or `cpu`) and `-ngl 0` force CPU. Default is
`--device auto -ngl auto`. `auto`, `all`, or a count at or above the
model's layer count enable all supported ops on the selected backend.

**A partial `-ngl` is refused, deliberately.** llama.cpp's `-ngl N` puts
exactly `N` layers in VRAM and runs the rest on the CPU, which is how
you fit a model that does not otherwise fit. frink has no partial layer
placement, and it used to accept the count and then offload *everything*:
same flag, same value, no error, and an out-of-memory on exactly the
machine the flag existed to accommodate. It now stops and says so. Use
`-ngl 0` for CPU or `-ngl all` for the whole model.

### Chat vs completion

- **Default:** the prompt is rendered through the GGUF's own
  `tokenizer.chat_template`, evaluated as Jinja2 by the same evaluator
  `frink-server` uses, so the CLI and `/v1/chat/completions` frame a
  conversation identically. A checkpoint that ships no template falls
  back to ChatML (matching llama.cpp `--jinja`), or to role-labeled
  lines for a byte tokenizer. A template that does not compile is an
  error, not a fallback to a guessed framing.
- **Whitespace:** the evaluator runs with `trim_blocks` and
  `lstrip_blocks` on, which is how HuggingFace's `apply_chat_template`
  and llama.cpp's Jinja engine both compile a chat template. Templates
  that use explicit `{%- … -%}` control render the same either way;
  TinyLlama's does not, and with the flags off every turn gained blank
  lines. 15 real templates are pinned byte-for-byte against goldens
  generated by jinja2 itself
  (`cargo test -p frink-models --test chat_template_real_gguf`,
  regenerate with `python3 scripts/chat_template_goldens.py`).
- **One disclosed deviation:** `{{ x | tojson }}` sorts object keys.
  That is stock jinja2's policy, but transformers and llama.cpp both
  preserve the author's order. Frink cannot: `serde_json::Map` is a
  `BTreeMap` here, so the order is gone before the filter runs. It
  changes the order of keys inside a `<tools>` block, nothing else.
- **`--no-cnv`:** raw prompt (classic completion). BOS is still added under the
  same rule.

### Who adds BOS

**The chat template owns BOS when it prints one. Otherwise the loader
owns it.** Which of the two applies is a property of the individual
checkpoint, not of the model family, so frink adds the id
*idempotently* (`frink_models::tokenizer::prepend_bos`) rather than
picking a side:

- Most upstream templates open with `{{ bos_token }}`: gemma-2/3/4
  (`<bos>`), Mistral-Instruct and Phi-3 (`<s>`), Llama-3
  (`<|begin_of_text|>`), DeepSeek-R1-Distill. Rendering one puts BOS in the
  *text*, and a rendered prompt is encoded with special-token markers
  parsed (llama.cpp's `parse_special = true`, as its server does), so it
  comes back as the BOS *id* in position 0.
- Unsloth deliberately **strips** `{{ bos_token }}` from the templates it
  bakes into its GGUF exports, so that a runtime adding BOS itself does not
  double it. TinyLlama's checked-in template is the local example.

Whether BOS is added at all is llama.cpp's `add_bos` rule
(`tokenizer.ggml.add_bos_token` if present, else SPM → yes / BPE → no):
Qwen2 ships a `bos_token_id` of `<|endoftext|>` that it never prepends, and
prepending it poisons greedy decode.

Measured over every local checkpoint by
`cargo test -p frink-models --test bos_policy -- --ignored --nocapture`,
which renders each GGUF's own template, encodes it with that GGUF's own
tokenizer, and asserts at most one leading BOS id.

### When generation stops

On the whole end-of-generation set, not `tokenizer.ggml.eos_token_id`
alone: the `eos`/`eot`/`eom` metadata ids plus every vocabulary entry whose
text is on llama.cpp's literal EOG list (`<|eot_id|>`, `<end_of_turn>`,
`<|im_end|>`, `<turn|>`, …). A Llama-3 checkpoint's `eos_token_id` is
`<|end_of_text|>` while its turns end with `<|eot_id|>`. Stop on the
metadata EOS alone and the model runs past its own turn, then starts
interviewing itself. `--ignore-eos` disables all of it. `frink-server`
uses the same set.

## Other commands

```bash
./target/release/frink inspect models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
./target/release/frink inspect-plan models/olmoe-1b-7b-0924-q4_0.gguf --strict
# Plan against a backend's real memory budget (Metal
# recommendedMaxWorkingSetSize / free VRAM / host RAM minus a reserve).
# Always reports the largest context that fits and the arithmetic:
./target/release/frink inspect-plan model.gguf --backend metal --ctk f16
./target/release/frink caps
./target/release/frink archs
./target/release/frink presets
./target/release/frink smoke glm-5.2

# Kimi K3 safetensors directory (large checkpoint, see MODELS.md)
./target/release/frink run-kimi /path/to/kimi --prompt "Hi" --max-new-tokens 32
```

## Serving benchmark (`serve-bench`)

`frink bench` is single-stream and HTTP-free: it measures kernels
against `llama-bench`. `frink serve-bench` answers the other question:
what a running `frink-server` does under concurrency.

```bash
# Start the server first.
FRINK_MODEL_PATH=model.gguf ./target/release/frink-server &

./target/release/frink serve-bench --requests 64 --concurrency 8 --output-len 128
./target/release/frink serve-bench --concurrency 16 --json
```

Four rules decide whether the numbers mean anything, and all four are
in `frink_edge::bench_client` with no socket in them, so each is
covered by a test rather than inferred from a live run:

- **Every request does exactly the requested work.** Temperature 0,
  top-k 1, `ignore_eos`, and an exact output length. Without
  `ignore_eos` the requests finish at different lengths and the slowest
  percentile is whichever prompt happened to run longest, a fact about
  the prompts reported as a fact about the server.
- **The TTFT/TPOT split is positional.** The first token-bearing chunk
  is time-to-first-token; every later one is an inter-token sample.
  Keepalives and the terminal `finish_reason` frame are excluded: a
  keepalive arrives during exactly the window TTFT measures, and the
  terminal frame carries no token.
- **Percentiles are nearest-rank over samples pooled across requests**,
  never per-request means percentiled afterwards. One request that
  stalled mid-answer has to reach the p99, and inside its own mean it
  never does.
- **Throughput is total tokens over the whole run's span**, not the sum
  of per-request rates, which gets *better* the worse the queueing is.

Token counts come from the server's own `usage.completion_tokens`, not
from the chunk count: a buffered answer arrives as one chunk and was
still N tokens of work. A buffered stream therefore reports TTFT and
end-to-end but no TPOT. That detail does not exist, and it is left
blank rather than invented.

## Bandwidth profile (`bench-bw`)

`frink-core`'s `qstar` decides how much of a MoE layer to fetch across
the link and how much to compute on the CPU. Without a measured
profile it falls back to an unbenchmarked default of one fetch per layer
per step, so every deployment gets a split nobody measured.

```bash
cargo build --release -p frink-cli --features cuda
./target/release/frink bench-bw --format q4_k
./target/release/frink bench-bw --dry-run          # measure, write nothing
```

It writes `$XDG_CACHE_HOME/frink/benchbw/<gpu-uuid>.json`, which the
loader finds on its own. A profile is keyed to the card it was taken
on: another machine's split is worse than no split, so a profile whose
recorded GPU name disagrees is ignored rather than approximated.

It refuses to write in two cases, both deliberate:

- **Only one side measured.** The fetch fraction is a *ratio*, so one
  number says nothing about the split. The PCIe half needs a CUDA
  build; without one the command measures the CPU side, says so, and
  writes nothing rather than half a profile that `policy_for` would
  consult as though it were whole.
- **An unoptimized build.** A debug binary measures its own code
  generation, and since the device copy is driver-performed and
  unaffected, that moves the *ratio* rather than merely lowering both
  numbers. A verdict that flips with `--release` measures nothing.
  `--allow-debug-build` overrides it if you know why you want that.

The device-side measurement is a documented stub pending a benchmark
host, see `docs/plans/archive/freetoken-parity.md`. It must be timed with CUDA
events rather than a wall clock, and repeated under contention, because
the number the policy wants is the *contended* pair: standalone
bandwidths assume each side owns the machine and neither does once they
run together.

## Correctness (`verify`, `parity`)

Two different questions, and only the second one involves llama.cpp.

```bash
# Do frink's own backends agree? (CPU reference vs Metal/CUDA)
./target/release/frink verify -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --backend metal --prompt-tokens 64

# Does frink agree with llama.cpp? (first-token distribution, CPU vs CPU)
./target/release/frink parity -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --prompt-tokens 64
```

`verify` greedy-decodes the same prompt on two frink backends and diffs
the token ids. It cannot catch a bug both backends share.

`parity` runs **two** comparisons against llama.cpp on the same GGUF: the
tokenizer first, then the graph.

### The tokenizer half

Frink's token ids against llama.cpp's, for a fixed 19-case corpus, on
the same file. It prints one line per checkpoint and, for each case that
diverges, the token index, the approximate byte offset, the input either
side of that offset, and both engines' ids and decoded pieces in a
window around it:

```
tokenizer Phi-4-mini-instruct-Q4_K_M: DIVERGES (19 cases / 350 tokens, pre=gpt-4o, ...)
  vocab  llama 200064 / frink 200064     add_bos  llama false / frink false
  7/19 cases diverge:

  [digit-runs] token 2 of 23 (llama) / 35 (frink), byte ~6 of 61
      input around it: "Build " >|< "1234567 of 89 took 10000"
      llama  12893:"Build" 220:" " *7633:"123" 19354:"456" 22:"7" 328:" of"
      frink 12893:"Build" 220:" " *16:"1" 17:"2" 18:"3" 19:"4"
```

The corpus is built out of the clauses llama.cpp's pre-tokenizer regexes
actually differ on, not out of prose: long digit runs, runs of two or
more spaces, 4- and 8-space indents, tabs, blank lines, CRLF, trailing
whitespace, uppercase and stacked contractions, CJK, emoji and ZWJ
sequences, Unicode whitespace, control bytes, punctuation runs and
version/address strings. Ordinary English is exactly what a wrong
pre-tokenizer still gets right, which is why running it proved nothing
for years.

Ids are compared with `add_special = false` on both sides; the add-BOS
*decision* is compared separately, as a flag, so that one policy
disagreement does not misreport all 19 cases. Vocab sizes are compared
first, because two different id spaces make everything below them
meaningless.

This half runs before the logit half and both are always reported, but
either one diverging exits non-zero. A tokenizer divergence means the
logit numbers underneath were computed from two different prompts.

### The logit half

`parity` compares the logit distribution at the last prompt position
against llama.cpp's, feeding **the same token ids to both** so the
tokenizer is not part of *that* experiment. It reports KL in both
directions, total variation, max |delta p|, top-k overlap, and where
llama's top-1 ranks for frink, then gives one of four verdicts:

| Verdict | Meaning |
|---|---|
| `MATCH` | same distribution to within f32 accumulation-order noise |
| `DRIFT` | same top-1, distributions moved further than reordering explains |
| `TIE-FLIP` | top-1 differs, but llama's own top-2 margin is under the observed noise, so a tie swapped rather than the graph being wrong |
| `WRONG` | the graphs disagree, and the command exits non-zero |

Comparing greedy *text* would not work here. A chain of argmaxes turns
one last-bit difference into a different sentence, so a text diff cannot
tell `TIE-FLIP` from `WRONG`.

A `WRONG` on a quantized file is a distance between two points and does
not say which one moved. llama.cpp quantizes activations to 8 bits for
its quantized matmuls and frink keeps them in f32, and on a graph that
amplifies that loss the two disagree while frink is the closer of the
two to the f32 answer -- PLM-1.8B Q8_0 reads `WRONG` at 3.5e-2 for
exactly that reason (`docs/plans/llama-cpp-gap-inventory.md` §10.1).
The arbiter is the dequantized file:

```bash
PYTHONPATH=$LLAMA/gguf-py python3 scripts/dequantize_gguf.py model-Q8_0.gguf /tmp/model-f32.gguf
./target/release/frink parity -m /tmp/model-f32.gguf   --dumper target/llama_logits --dump-logits /tmp/f32
./target/release/frink parity -m model-Q8_0.gguf       --dumper target/llama_logits --dump-logits /tmp/q8
# KL(f32.llama || q8.llama) is the reference's own quantization loss;
# KL(f32.llama || q8.frink) is frink's; KL(f32.llama || f32.frink) is the graph.
```

`LLAMA_LOGITS_FLASH_ATTN=0` keeps the reference off its flash-attention
path (llama.cpp itself aborts under the default on a real PLM file; the
published numbers were measured with the default).

### The reference dumper

Both halves need it, built once. It is C, not Rust, and it lives outside
the cargo workspace on purpose. It exists to give llama.cpp's own
answer, so it links llama.cpp's own library:

```bash
./tools/build_llama_logits.sh          # -> target/llama_logits
LLAMA_CPP_PREFIX=/path/to/llama.cpp ./tools/build_llama_logits.sh
```

It lands in `target/`, so `cargo clean` removes it; rebuild rather than
assuming `parity` broke. Point `--dumper` or `FRINK_LLAMA_LOGITS` at it
if you build it elsewhere. A dumper built before the tokenizer half
existed has no `--tokenize` mode, and `parity` says so and names the
rebuild.

To sweep the tokenizer half across every checkpoint under `models/`
without running any prefill:

```bash
./tools/build_llama_logits.sh
cargo test -p frink-cli -- --ignored frink_and_llama_cpp_tokenize_the_corpus_identically --nocapture
```

That test is `#[ignore]`d because it needs the dumper and real
checkpoints. Checkpoints that are missing, or that the installed
`libllama` cannot load, are skipped by name. A reference with no answer
is not a verdict either way.

## Diagnostics (`layer-divergence`, `quant-sensitivity`)

`verify` says *which token* two backends stopped agreeing on.
`layer-divergence` says *which layer*.

```bash
./target/release/frink layer-divergence -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --backend metal --prompt-tokens 16
```

It runs one prefill per backend (a child process each, because the
backend is a process-lifetime choice), then reads every layer's KV cache
back and scores the per-head magnitudes. What it prints per layer is the
**spread** of the per-head ratios, not the mean: one wrong head in
thirty-two leaves the mean at 1.0, and a single bad head is the shape of
every simdgroup-indexing bug this project has hit. The mean is printed
next to it so the reader can watch it fail to notice.

Read a first divergence at layer L as "at or immediately before layer
L": layer L's K and V come from layer L's input, so the fault is in
layer L's norm/QKV projection or in whatever produced its input. Layers
below it are exonerated.

Measured noise floor between CPU and Metal on a healthy model
(Llama-3.2-1B Q4_K_M, 16 tokens): spread 1.6e-5 to 1.3e-4. The default
`--tol 1e-3` sits about 8x above the worst of that.

MoE checkpoints also get a routing column: the total-variation distance
between the two backends' expert-selection histograms. `no counts` there
means one side never recorded a selection, which is not agreement.

```bash
./target/release/frink quant-sensitivity -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --candidate q4_0 --prompt-tokens 16 --top 10
```

`inspect-plan` prices a checkpoint from static type rules.
`quant-sensitivity` measures the same question on the checkpoint in
front of it: it round-trips **one tensor at a time** through a candidate
format, scores `relative_mse` per block, swaps the result into the
loaded model and reports how far the next-token distribution moved (KL,
nats). Every other weight stays as the checkpoint shipped it, so no
tensor inherits damage from the layers above it.

Both columns are printed because they disagree, and the disagreement is
the point: a tensor can round-trip badly and barely move the logits, or
round-trip cleanly and move them a lot. Only the second is a reason to
spend bits. The rollup at the bottom gives each tensor family's share of
the total measured KL, which is what a static quant rule is guessing at.

It runs on CPU by construction and refuses to start with
`FRINK_CPU_INT_DOT=1`, whose repack cache is keyed by buffer address
and would hand a swapped-in tensor another tensor's repacked bytes.
Cost is one forward pass per tensor: about two minutes for a 1B model's
112 tensors at 16 prompt tokens. `--layers 0:4` restricts the sweep.

## Perplexity (`frink perplexity`)

Corpus evaluation, llama.cpp's `perplexity` tool.

```bash
frink perplexity -m model.gguf -f corpus.txt --ctx-size 512
```

This is the quality axis the project did not have. `frink parity`
compares first-token distributions and `frink bench` measures speed;
neither answers "is this quantization worse, and by how much". It is
also the acceptance test the quantizer needs, because a bad K-quant
encoder produces a file that loads fine and generates measurably worse
text.

**Measured against `llama-perplexity` on the same corpus and
checkpoint**, both engines on CPU:

| Checkpoint | frink | llama.cpp | Gap |
|---|---|---|---|
| SmolLM2-135M Q8_0 | 14.7284 | 14.7529 | -0.17% |
| SmolLM2-135M Q4_K_M | 15.0896 | 15.1274 | -0.25% |
| SmolLM2-135M IQ3_M | 16.5144 | 16.6004 | -0.52% |
| Qwen3-0.6B Q8_0 | 19.9805 | 19.9799 | +0.003% |
| TinyLlama-1.1B Q8_0 | 11.9852 | 12.0190 | -0.28% |

Every gap is under a fifth of one standard error, and the per-window
running estimates track window for window, which is what says
tokenization and chunking agree.

**The gaps are not noise, and their shape is the interesting part.**
frink sits below llama.cpp on every quantized checkpoint and the gap
widens as the quant coarsens. That is the `vec_dot_type` difference this
repo already documents
([`plans/llama-cpp-gap-inventory.md`](plans/llama-cpp-gap-inventory.md)
§10) showing up on a second axis, with the sign it should have:
llama.cpp quantizes the activation to the weight's vec_dot type and
frink keeps it in f32, so frink is slightly less surprised. Qwen3 is
the control, straddling zero. A difference in METHOD would not produce a
gap that is monotone in the quant.

The method is llama.cpp's, verified against `tools/perplexity/perplexity.cpp`
rather than assumed, because getting any of it wrong makes the number
incomparable to every published figure while still looking reasonable.
Non-overlapping windows of `--ctx-size`; `first = n_ctx/2` so 255
positions are scored at 512, not 256; BOS at the front of the corpus and
the first token of each window overwritten with it, never scored;
natural log; `exp` of the unweighted mean over all scored tokens pooled
across windows, not a mean of per-window perplexities.

Deviations, all recorded in the module doc: one `forward_batch` per
window rather than an `n_batch` split, which changes the f32 reduction
grouping and not the causal context; the output head runs at every
position rather than the scored half, which costs memory and not
accuracy; and `--ppl-stride`, HellaSwag, WinoGrande, multiple-choice and
KL-divergence are not implemented.

Every number above is CPU on both sides. Metal and CUDA perplexity is
unevidenced.

## Quantize (`frink quantize`)

Writes a `Q8_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M` or `Q6_K` GGUF
from an F32/F16/BF16 one, **byte-identical to `llama-quantize`'s**,
and refuses every other target by name.

```bash
frink quantize model-f16.gguf model-q4_k_m.gguf --type q4_k_m
frink quantize model-f16.gguf model-q4_k_m.gguf --type q4_k_m --imatrix imatrix.gguf
frink quantize model-f16.gguf model-q4_k.gguf   --type q4_k_m --pure   # no per-tensor mix
```

The refusal is the point rather than a limitation to apologise for.
Q2_K/Q3_K, the IQ tiers, MXFP4 and the legacy Q4_0 family each need
their own transcription of an iterative fit, and a K-quant encoder
that takes min and max over a block where llama.cpp does an iterative
scale-and-min fit produces a file that loads and generates measurably
worse text. A `quantize` whose name implied llama.cpp's whole range
while approximating half of it would be worse than the gap.

Which tensors are quantized is transcribed from llama.cpp's
`tensor_allows_quantization` rather than reinvented: everything 2-D
ending in `weight`, except norms, router gates, position and token-type
embeddings, SSM and shortconv kernels, RWKV time-mix, T5 position bias,
multimodal patch tables and audio codebooks. The per-tensor MIX is
llama.cpp's too: a `Q4_K_M` file has a Q6_K output head and Q6_K
`ffn_down` on a quarter of its layers, and `--pure` is
`llama-quantize --pure`, which skips the mix and not the keep-list.

**Every target is byte-identical, and the claim that Q4_K could never
be was wrong.** This document previously said the difference was a
property of the compiler that built the reference: clang contracts
`a*b+c` into a fused multiply-add for C and Rust does not, so a strict
transcription could not match. The first half of that is true and the
conclusion was not. The fix is to spell the contraction out. `sumlx +=
w*x[i]*l` in `ggml-quants.c` is **one** FMA, and writing it as
`mul_add` in Rust reproduces it exactly; one unit in the last place
flips `sumlx*sumlx > best*suml2` and rewrites an entire super-block,
which is why 1.15% of super-blocks differed rather than a
rounding-sized fraction.

Measured against `llama-quantize` b7650 over an F16 Llama-3.2-1B, whole
model rather than a fixture:

| Target | Tensors identical | Super-blocks differing |
|---|---|---|
| Q8_0 (control) | 147 / 147 | 0 |
| Q4_K_M | 147 / 147 | 0 of 3,244,032 Q4_K, 0 of 1,583,104 Q6_K |
| Q5_K_M | 147 / 147 | 0 of 3,244,032 Q5_K, 0 of 1,583,104 Q6_K |
| Q6_K | 147 / 147 | 0 of 4,827,136 Q6_K |

And with an importance matrix (`--imatrix`, below), against
`llama-quantize --imatrix` b7650 over a BF16 Qwen3-0.6B with
`llama-imatrix`'s own file, so the weighted fit is measured on
llama.cpp's input and not on frink's:

| Target | Tensors identical | Super-blocks differing |
|---|---|---|
| Q8_0 (control; ignores the imatrix) | 311 / 311 | 0 of 23,486,464 |
| Q4_K_S | 311 / 311 | 0 of 2,274,816 Q4_K, 0 of 53,248 Q5_K, 0 of 607,744 Q6_K |
| Q4_K_M | 311 / 311 | 0 of 2,098,688 Q4_K, 0 of 837,120 Q6_K |
| Q5_K_M | 311 / 311 | 0 of 2,098,688 Q5_K, 0 of 837,120 Q6_K |
| Q6_K | 311 / 311 | 0 of 2,935,808 Q6_K |

The metadata matches too, including the four `quantize.imatrix.*` keys
llama.cpp records; the only difference between the files is the order
of the header's key-value pairs, which frink writes sorted.

Two things that discipline needs. Goldens must come from the
**installed** binary: a local release build of the same b7650 source
disagrees on exactly these knife-edge blocks, and pinned the wrong
bytes once. And the fixture must be able to fail: the original
synthetic one stayed green with every `mul_add` removed, so it now
carries eight real weight blocks, one per fusion site, of which nine of
thirteen sites redden a golden. The imatrix goldens
(`encode/imatrix_golden.rs`) are two real rows from the run above and
go red when the weight rule, the candidate grid or the `make_qp_quants`
stage 2 is replaced by the plain path's.

### `--imatrix`

The importance-matrix fit is llama.cpp's `quantize_row_q4_K_impl`,
`quantize_row_q5_K_impl` and `quantize_row_q6_K_impl`
(`ggml/src/ggml-quants.c:1376`, `:1581`, `:1793` at b7650), and it is
not the plain fit with a weight added. For Q4_K and Q5_K three things
change: the per-element weight is `qw * sqrt(sigma2 + x^2)` with
`sigma2 = 2 * mean(x^2)` over the super-block instead of
`sqrt(mean(x^2)) + |x|` over the sub-block; the candidate grid is
`(-0.9, 0.05, 36)` for both formats instead of each format's own; and
the 6-bit scales and mins are fitted by `make_qp_quants` (`:899`), a
weighted grid search with a greedy per-code refinement, instead of
`63/max`. Q6_K's change is one argument: the raw imatrix slice goes to
`make_qx_quants` as its `qw`. Q8_0 discards the imatrix (`:2089`). All
of that lives in `frink-quant`'s `encode/fit.rs` and `encode/qp_quants.rs`
as a parameter on the SAME super-block fit the plain path uses, not a
second transcription.

The consumer side is `llama-quant.cpp:913-934`: each tensor looks up its
own name, a tensor with no entry is quantized unweighted with a printed
notice (`output.weight` and `token_embd.weight` are the usual ones,
since `llama-imatrix` collects neither without `--process-output`), and
an entry of the wrong width is a refusal except on `token_embd.weight`.
Either file format is accepted: the GGUF one current `llama-imatrix`
writes, or the legacy `.dat` binary older builds wrote.

One refusal to know about: a tensor whose row width is not a multiple of
256 stops the run. llama.cpp answers that case by changing the tensor's
TYPE, to Q5_0 or F16, and frink has neither encoder; padding the row
would shift every following row on decode. SmolLM2-135M cannot be Q4_K
quantized here for that reason, its embedding being 576 wide.

## Importance matrix (`frink imatrix`)

llama.cpp's `llama-imatrix`: runs a calibration text through the model
and writes, per weight, the per-column sum of squared activations that
the quantizer above weights its fit by. Same flags where they exist on
both, same file format in both directions: a frink file feeds
`llama-quantize --imatrix` and a `llama-imatrix` file feeds `frink
quantize --imatrix`.

```bash
frink imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.gguf
frink imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.dat --output-format dat
frink imatrix -m model-bf16.gguf -f calibration.txt --chunks 64 -c 512 --process-output
frink imatrix -m model-bf16.gguf -f calibration.txt -o mine.gguf --compare theirs.gguf
```

The method is `tools/imatrix/imatrix.cpp` at b7650, cited line by line
in the module doc. What is collected (`:229-237`): the f32 input of
every matrix multiplication whose weight is under `blk.`, plus
`output.weight` with `--process-output`; expert weights per expert with
one count each (`:302-317`). The rule (`:365-372`): `values[j] +=
x[j]*x[j]` per row, `counts += rows`. The chunking (`:909-1013`): the
whole file tokenized once with the checkpoint's BOS rule, non-overlapping
chunks of `--ctx-size`, each chunk's first token overwritten with BOS
when the vocabulary adds one, each chunk a forward pass over a fresh KV
cache, at least two chunks' worth of tokens required. The file
(`:507-615`): `general.type = imatrix`, `imatrix.datasets`,
`imatrix.chunk_count`, `imatrix.chunk_size`, and per weight a
`<name>.in_sum2` F32 `[n_per_row, n_mat]` and a `<name>.counts` F32
`[1, n_mat]`, names sorted, trailing unit dimensions trimmed as ggml
trims them.

frink has no compute graph to hang a callback on, so the activations
are observed at the two functions every projection goes through
(`frink_core::activation_tap`), keyed by the weight's address and
named by walking the decoder's public weight fields against the GGUF's
tensor names. That seam exists on the CPU path only, so the run pins
the CPU backend the way `frink bench --n-gpu-layers 0` does, and
after the run every dense entry's row count is checked against the
token count: a weight whose decoder path bypassed the tap, or was
observed twice, is a refusal rather than a wrong file. An expert the
text never routed to is reported as partial data, as upstream reports
it.

Deviations, all stated: one chunk per forward pass where `llama-imatrix`
folds `n_batch / n_ctx` chunks into one batch as separate sequences
(same rows in the same order, so the same sums); no perplexity printed,
because `frink perplexity` already computes that number by llama.cpp's
method; no `--in-file` combining of earlier matrices; and
expert streaming (`Stored` experts) is refused because a streamed
expert's weight view has no stable identity.

**What matches llama.cpp's file and what does not, measured.** The
names, the counts, the shapes and the accumulation rule are the same,
so the files are interchangeable and `frink quantize --imatrix` on a
`llama-imatrix` file is byte-identical to `llama-quantize` (table
above). The sums are NOT bit-identical, because the activations are
not, and `--compare` prints the gap per entry. On Qwen3-0.6B (BF16 and
an F32 copy, same result), 8 chunks of 512 tokens of plain prose on
which both tokenizers agree exactly:

| Where | frink vs `llama-imatrix`, per entry |
|---|---|
| layer 0 `attn_q/k/v` input (RMSNorm of the embedding, no matmul yet) | max per-column relative 7e-6 |
| layer 0 `attn_output` input (after the first attention) | 3e-3 |
| layer 27 `ffn_down` input | 2.7e-2 (the worst of 196 entries) |
| all entries, L2-relative | median 7.4e-4, max 6.0e-3 |

So the gap enters at the attention block and compounds with depth,
and it is the forward-pass difference between the two engines, not
the accumulation: it is unchanged by `llama-imatrix -ctk f32 -ctv f32`
and by BF16 versus F32 weights (llama.cpp's own two runs are
byte-identical to each other), which rules out the KV cache type and
the `vec_dot_type` rounding as the cause on this checkpoint. Where the
attention arithmetic diverges is a `frink parity` question, not an
imatrix one; on a K-quant checkpoint the documented Q8_K activation
rounding would add to whatever it is. Percent-level differences in an
importance weight are far below what moves a quantized super-block --
the weights enter the fit as relative importances -- but that is a
statement about the effect, not a claim the files match.

Two things to check before trusting a comparison. The text must
tokenize identically, and it once did not: on the repo's own markdown
docs frink's Qwen2-style BPE produced 17208 tokens where
`llama-tokenize --no-escape` produced 17209, one fewer at each
mention of `<s>` -- Qwen2.5's vocabulary carries `<s>` as an ordinary
entry that llama.cpp never treats as special, and frink promoted it
on its shape. That is fixed, and `frink imatrix` now tokenizes its
text with special-token markers left as text, which is
`llama-imatrix`'s own default (`parse_special = false`); a doc that
mentions `<|im_end|>` is six characters on both engines. `frink
parity`'s tokenizer sweep carries a case of markers-as-prose under
both `parse_special` settings so the class stays closed. Either kind
of difference shifts every chunk boundary and turns a 1e-3 comparison
into a 1e-1 one. And it must be the same text through the same number
of chunks, since a chunk count is a token count.

## Split and merge GGUF (`frink gguf-split`)

llama.cpp's `llama-gguf-split`, same flags, same shard names, same
metadata keys. Splitting is the default operation; `--merge` is the
other direction.

```bash
# By tensor count (llama.cpp's default limit is 128)
frink gguf-split --split-max-tensors 128 model.gguf out/model

# By size. Units are DECIMAL, as in llama.cpp: 4G is 4,000,000,000
frink gguf-split --split-max-size 4G model.gguf out/model

# Metadata-only first shard, the layout most published checkpoints use
frink gguf-split --split-max-size 4G --no-tensor-first-split model.gguf out/model

# Plan only: shard count, tensors and bytes per shard, nothing written
frink gguf-split --split-max-size 4G --dry-run model.gguf out/model

# Back to one file. The input is the FIRST shard
frink gguf-split --merge out/model-00001-of-00003.gguf model.gguf
```

Shards are named `<prefix>-NNNNN-of-MMMMM.gguf`, 1-based, and carry
llama.cpp's three keys with llama.cpp's types: `split.no` (u16, 0-based),
`split.count` (u16) and `split.tensors.count` (i32, the total across the
whole set). The first shard holds the complete source metadata and the
rest hold only those three, so a set written here is one `frink run -m
out/model-00001-of-00003.gguf` away from running, and one llama.cpp
reads too.

Tensor bytes are copied straight from the source's mmap into the shard,
never buffered, so a 400 GB checkpoint splits in the memory a header
takes.

Four things differ from llama.cpp's tool, all of them refusals it does
not make:

* Splitting a file that is **already a shard** stops. llama.cpp would
  write a set whose `split.tensors.count` covered that one shard, which
  no loader can reassemble.
* A first tensor **larger than `--split-max-size`** is named, with both
  numbers. llama.cpp prints "one of splits have 0 tensors" and exits.
* `--merge` **refuses the `--split-*` options** instead of parsing and
  ignoring them.
* A missing shard names the file it wanted, at plan time, before the
  output is opened.

Two byte-level differences, both inherited from choices this crate
already made: metadata keys are written sorted, and shards are padded
with the alignment the source declares where llama.cpp's tool always
pads with 32. A merge of a set this tool wrote reproduces the source
byte for byte when the source itself carried the `split.*` keys a
previous merge leaves behind; otherwise the merged file gains exactly
those three keys and nothing else changes.

Cross-checked against the installed `llama-gguf-split` (build 7650, 68b4d516c) on the
21-tensor test fixture at `--split-max-tensors 4`. All **6 of 6** shards
came out the identical SIZE, byte for byte, and differ only where the
metadata keys are ordered: 54 bytes on each later shard (the three
`split.*` keys) and 754 on the first (the whole header). Both directions
work across the two tools: llama.cpp merges a set frink split, and
frink merges a set llama.cpp split, each producing a 24,032-byte file.

## Put a reranker's pooler back (`frink splice-pooler`)

Every `BertForSequenceClassification` reranker GGUF in circulation is
missing its pooler: llama.cpp's converter deletes `bert.pooler.dense`
by name (`conversion/bert.py`, `BertModel.filter_tensors`, "we are only
using BERT for embeddings so we don't need the pooling layer";
unconditional on `master` as of 2026-09-11). The head then runs as
`classifier(cls)` where the checkpoint was trained as
`classifier(tanh(pooler(cls)))`: same ORDER, a score range about fifty
times narrower, so a threshold copied from the model card never fires
(issue #82). This writes a GGUF that carries the tensor.

```bash
frink download cross-encoder/ms-marco-MiniLM-L6-v2 model.safetensors --local-dir models/ms-marco-MiniLM-L6-v2
frink splice-pooler -m models/ms-marco-MiniLM-L6-v2-Q8_0.gguf \
    --safetensors models/ms-marco-MiniLM-L6-v2/model.safetensors \
    -o models/ms-marco-MiniLM-L6-v2-Q8_0-pooled.gguf
```

The output is the input, every key and every tensor byte for byte, plus
`cls.weight` / `cls.bias` (F32, under llama.cpp's own names) and one
provenance key, `frink.rerank.pooler_source`. frink loads it with no
further change and `/v1/rerank` reports
`frink_score_head: classifier(tanh(pooler(cls)))`; llama.cpp loads it
too, and its `build_pooling` RANK arm runs the pooler as well.

**The pooler is tied to the checkpoint by the classifier, not by a
name.** The GGUF's own metadata is not evidence: the published
`ms-marco-MiniLM-L6-v2-Q8_0.gguf` says `general.name = Ms Marco MiniLM L
12 v2` and points `base_model.0.repo_url` at the L12 repo, while its six
layers and its scores are L6's. So the one tensor BOTH files carry,
`cls.output.*` in the GGUF and `classifier.*` in the safetensors, must
agree element-wise to within the GGUF's own storage precision (1/128 of
the tensor's largest magnitude, which admits F32, F16, BF16 and Q8_0
rounding and nothing coarser; a head stored coarser is refused by
dtype). A safetensors whose classifier the GGUF does not contain is
refused naming the tensor, the element and both values, and nothing is
written. On the published file the measured agreement is `1.5e-5`
against a bound of `4.8e-4`.

Also refused: a GGUF that is not `bert`, one that already carries
`cls.weight` (the refusal says where it was spliced from), one with no
`cls.output` at all, a split file (merge it first), and a pooler of
another width. The written file is reopened and passed through the
rerank head loader before the command returns, so the only file it
leaves behind is one the loader has accepted with the pooler in place.

Measured on `ms-marco-MiniLM-L6-v2`, four query sets, seventeen pairs,
against the NumPy transcription of HuggingFace's
`BertForSequenceClassification` (`scripts/rerank_reference_ms_marco.py`):
before, scores in about `-0.25..0.15`; after, `-11.19..10.93`, the
largest deviation from HuggingFace `0.051`, every ordering identical.

## Hugging Face Hub (`download`, `pull`)

Fetches a GGUF over HTTPS directly. No Python and no
`huggingface_hub` install: this used to shell out to the `hf` CLI, so a
Rust engine could not fetch its own weights without a Python
toolchain.

### `-hf`, llama.cpp's one-command form

`-hf user/repo[:QUANT]` fetches on first use and serves or runs
straight away, so nothing has to be downloaded by hand first:

```bash
frink serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
frink -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M -p "Hi" -n 64
```

The tag after the colon is a **quant label, not a git revision**, which
is worth saying because `repo:thing` means a revision nearly everywhere
else. It matches without regard to case, because repos spell it
`Q4_K_M` and `q4_k_m` about equally often. A tag the repo does not
publish is refused with the list of quants it does publish, since you
cannot see a repo's file list from a command line.

`-hf` is one token in llama.cpp's hand-written parser, and clap reads
`-hf` as `-h` followed by `f`. Both frink binaries rewrite `-hf` and
`-hff` before parsing, so the llama.cpp spelling works; `--hf-repo` is
the same flag.

Downloads land in the frink cache, not in `./models`: a model fetched
by `-hf` is not part of the project directory you happen to be standing
in. `FRINK_CACHE`, else `$XDG_CACHE_HOME/frink`, else
`~/.cache/frink`, under `hub/<owner>__<repo>/`. A second run says
`using cached` instead of fetching again, and an interrupted download
resumes by byte range rather than starting over.

`frink download` takes the same `repo:QUANT` shape and puts the file
where you ask instead of in the cache. Before that it sent the whole
string to the Hub as a repo id and returned a bare `401`, which reads
like an auth problem and is not one.

`download` otherwise takes the same arguments as `hf download`, so a
command copied off a model card runs unchanged:

```bash
frink download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
```

`pull` is the older spelling, and prints the local path so it can be
substituted into another command:

```bash
frink pull org/model --file '*.gguf'
frink -m org/model      # downloads when the path is missing
```

Cache default for `pull`: `~/.cache/frink/hf/<org--model>/`.
`download` defaults to `models/`.

| Variable | Effect |
|---|---|
| `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` | Sent as a bearer token, for gated or private repos |
| `HF_ENDPOINT` | Mirror to fetch from instead of `huggingface.co` |

An interrupted download resumes. The bytes land under
`<name>.partial` and are renamed only once the last one arrives, so a
truncated file is never left under a name the loader would open as a
whole GGUF. If the server ignores the range request and restarts the
body, that is detected from the response rather than assumed, so the
file is rewritten instead of being appended to itself.

A file already present is left alone rather than fetched again. When
the pattern matches more than one file, it says so instead of picking.

## Interactive chat (`chat`)

Multi-turn REPL against a running `frink-server` (reuses chat-template + SSE):

```bash
FRINK_MODEL_PATH=model.gguf FRINK_ADDR=127.0.0.1:8383 ./target/release/frink-server
./target/release/frink chat --url http://127.0.0.1:8383 --system "Be brief."
# Commands: /quit  /clear
```

| Flag | Notes |
|---|---|
| `--url` | Server base URL (default `http://127.0.0.1:8383`) |
| `--system` | Optional system message |
| `--max-tokens` / `--temperature` / `--top-p` | Sampling |
| `--no-stream` | Wait for full JSON instead of SSE |

## Server

Two ways to start the same server. `frink serve` is a subcommand of the
main binary and needs the optional `serve` feature at build time.
`frink-server` is that same server as its own executable, and both
parse identical arguments through the same code.

### LoRA adapters

`--lora adapter.gguf` and `--lora-scaled adapter.gguf:0.5` load the
file llama.cpp's `convert_lora_to_gguf.py` writes from a PEFT adapter
directory and apply it exactly as `build_lora_mm` does: every
projection the adapter names computes `W x + scale * alpha / rank *
B (A x)`, `token_embd` and `output` included, and two adapters on one
weight are two terms of the sum. Checked against libllama with the
same adapter: KL at or under 5.0e-13 on the fixture graph (five adapter and scale combinations), and on
Llama-3.2-1B-Instruct Q8_0 with a rank-8 adapter 5.2e-4 against a
base-only floor of 1.9e-4 (the adapter itself moves the distribution
by 1.5e-1).

What is refused, by name, rather than approximated: an adapter for
another architecture, one naming a tensor the base does not carry or
of a shape it does not fit (the three checks `llama-adapter.cpp`
makes), a routed-expert (`*_exps`) target, an activated LoRA
(`adapter.alora.invocation_tokens`), the embedding pair on a model
whose output head is tied to its embedding (libllama aborts in
`ggml_mul_mat` on that pair), and the flag on the MLA, Gemma-4,
GLM-5.2 and Kimi engines.

On Metal an adapted model runs on the per-matrix path -- each
projection's matvec on the device, the rank-sized delta on the host --
because the fused stacks read weight bytes past the seam the delta
lives in and are fenced off for the whole model. The output is the
same tokens as CPU (measured), at per-matrix speed: Llama-3.2-1B Q8_0
decodes at 44 tok/s with an adapter against 117 tok/s fused without
one on an M2 Pro. CUDA and CPU serve the adapter on every path they
have.

### Server flags, and llama.cpp's spellings

`llama-server` commands mostly run unchanged:

| Flag | Notes |
|---|---|
| `-m` / `--model` | GGUF path or Kimi directory |
| `-hf` / `--hf-repo`, `--hf-file` | Fetch from the Hub, see above |
| `-c` / `--ctx-size` | Positions any one request may ask for. Sets `FRINK_CB_MAX_CONTEXT`. Unset means the ceiling is derived at load from weights and per-token KV against the device budget, capped at the model's trained context |
| `--api-key`, `--api-key-file` | Require `Authorization: Bearer`. Also gates `/admin`. Prefer the file form on a shared host: an argument is visible in `ps` to every user on the machine. An empty key file is refused rather than leaving every route open |
| `--alias` | What the model is called in `/v1/models` and in every response's `model` field |
| `--ctk` / `--cache-type-k` | KV dtype. **Metal only**, the CPU and CUDA cache is the host `Vec<f32>` |
| `--host`, `--port` | `--port 0` asks the kernel for a free one and announces it on stdout |
| `-t`, `-ngl`, `-dev` | Threads, GPU layers, device |
| `-cb` / `--cont-batching`, `-np` / `--parallel` | Continuous batching and its sequence cap. Read back as `frink_scheduler_max_seqs` on `GET /metrics` |
| `-b` / `--batch-size`, `-ub` / `--ubatch-size` | Prompt tokens per forward pass, on both decode paths. Resolved to one number the way llama.cpp does (the smaller of whichever was named); read back as `frink_scheduler_prefill_chunk` |
| `--slot-save-path DIR` | Directory for `POST /slots/{id}?action=save\|restore`. Refused at startup when it is not a directory; without it the route answers 501 naming this flag, as llama.cpp does. Slots restore into the prefix cache, so `FRINK_PREFIX_CACHE_ENTRIES` must be set too |
| `--lora FILE`, `--lora-scaled FILE:SCALE` | LoRA adapters, as on the completion side (above). Sets `FRINK_LORA`; every model load, including `/admin/models/load`, attaches the same adapters or refuses the checkpoint by name. `GET /lora-adapters` lists them, `POST /lora-adapters` and a request's `lora` field set their scales, see [`API.md`](API.md#lora-adapters) |
| `--lora-init-without-apply` | Load the adapters at scale 0 until a `POST /lora-adapters` sets them. Sets `FRINK_LORA_INIT_WITHOUT_APPLY` |
| `--reasoning-budget N` | Token budget for thinking, llama.cpp's flag and range: `-1` unrestricted (default), `0` immediate end, `N>0` a budget. The server default a request's `reasoning_budget_tokens` falls back to when absent or `-1`. Enforced in the sampler: after N tokens of thought the closing tag is forced, so the answer still arrives. Sets `FRINK_REASONING_BUDGET` |
| `--prefill-assistant` / `--no-prefill-assistant` | Whether a trailing assistant message is continued rather than closed, llama.cpp's flag; on by default. A request's own `continue_final_message` (including `false`) still wins. Sets `FRINK_PREFILL_ASSISTANT` |
| `--jinja` | Accepted, and already the default: frink always compiles and evaluates the GGUF's own `tokenizer.chat_template` |
| `--no-warmup` | Accepted; there is no warm-up pass to skip |
| `--flash-attn` / `-fa` | Accepted. Fused attention is a backend property here, not a per-run switch |

Two are **refused by name** rather than ignored, because ignoring them
would change the answer without saying so:

- `--no-jinja`. There is no template-free mode to fall back to, and a
  prompt framed by a guess instead of the checkpoint's own template
  reads as a model-quality problem rather than a flag that was dropped.
- `--flash-attn off`. Set `FRINK_METAL_ATTN=0` or `--device cpu`.

`serve` is on by default, so a stock `cargo install frink-cli` has it:

```bash
cargo build --release -p frink-cli --features "serve metal"

./target/release/frink serve \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all -cb -np 4
```

On Metal, continuous batching turns on by default when compatible; use
`-cb` / `--cont-batching` and `-np` / `--parallel N` (llama.cpp slot
cap) to set it explicitly. `--no-cont-batching` keeps the private
decode loop (Metal serializes concurrent requests on that path).

Without the feature, `frink serve` still exists and explains itself
rather than reporting an unknown subcommand, since a compiled-out
feature and a missing one look identical from the outside otherwise.

The standalone binary takes the same flags:

```bash
./target/release/frink-server \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all

# MCP config (metadata under GET /v1/models, invocation is not wired up)
./target/release/frink-server -m model.gguf --mcp-config mcp.json

curl -s -X POST http://127.0.0.1:8383/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"temperature":0}'
```

### Running under a supervisor

```bash
# The kernel picks the port. The bound address is announced on stdout.
./target/release/frink-server -m model.gguf --port 0 --exit-on-stdin-close
{"event":"frink.server.ready","addr":"127.0.0.1:52091","port":52091,"scheme":"http","pid":4242,"version":"0.15.2"}
```

`--port 0` plus that one line saves a parent process from probing
whether a port is free, or working out whether an existing listener is
a stale copy of itself or a stranger's server. Read stdout line by line
and ignore anything that is not the ready event. The tracing subscriber
shares the stream.

`--exit-on-stdin-close` (or `FRINK_EXIT_ON_STDIN_CLOSE=1`) exits when
stdin reaches EOF, which is the one orphan-prevention mechanism that
behaves identically on macOS, Windows and Linux and survives a parent
that dies rather than exiting cleanly. It is **opt-in**: a server
started with stdin redirected from `/dev/null` (systemd, cron, `nohup`)
sees EOF immediately, so the parent that wants the guarantee is the one
that asks for it and keeps the pipe open.

The server accepts `-m/--model`, `--host`, `--port`, `-t/--threads`,
`-dev/--device`, `-ngl/--n-gpu-layers`, `--cont-batching` / `-cb`,
`--no-cont-batching`, `-np` / `--parallel N`, `-b` / `--batch-size N`,
`-ub` / `--ubatch-size N`, `--slot-save-path DIR`, `--reasoning-budget N`,
`--prefill-assistant` / `--no-prefill-assistant`, `--exit-on-stdin-close`,
and `--list-devices`. Existing
`FRINK_MODEL_PATH`, `FRINK_ADDR`, and the backend environment
variables all still work. Command-line values win over them. Keep
secrets such as `FRINK_API_KEY` in the environment.

The web UI is a separate app. See [`ui/`](../ui) and
[`crates/frink-server/README.md`](../crates/frink-server/README.md).
`GET /` on this server is a 404 like any other unknown path.

## Benchmark (`frink bench`)

With `-m`, `bench` works like [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench).
Same workload names (`pp<N>` batched prefill, `tg<N>` decode), same
reporting (median ± population stddev over `-r` reps, one warmup
discarded), same flag names. Put the two outputs side by side and they
line up.

```bash
# one GGUF (CPU). Prints the exact llama-bench command to compare against.
./target/release/frink bench -m model.gguf -p 512 -n 128 -r 3 --compare

# Metal
./target/release/frink bench -m model.gguf --n-gpu-layers 99 -p 512 -n 128 --compare

# multi-model suite (same models list as benchmarks/suite.json)
./target/release/frink bench --suite --fit-host --skip-missing
./target/release/frink bench --suite --id tinyllama_q8 --backend metal
./target/release/frink bench --render
```

| Flag | Meaning |
|---|---|
| `-m/--model` | GGUF to benchmark. Without it, `bench` runs the synthetic matvec microbenchmark instead |
| `-p/--n-prompt` | Prefill tokens (default 512, `0` skips the `pp` row) |
| `-n/--n-gen` | Decode steps (default 128, `0` skips the `tg` row) |
| `-r/--repetitions` | Timed reps (default 3), plus one discarded warmup |
| `-t/--threads` | CPU threads (`0` = performance-core default) |
| `--n-gpu-layers` | `0` forces CPU, anything else offloads |
| `--compare` | Also run `llama-bench` on the same GGUF and print the gap |
| `--suite` | Run every [`benchmarks/suite.json`](../benchmarks/suite.json) entry in its own process, write a timing file per run, re-render [`RESULTS.md`](../benchmarks/RESULTS.md) |
| `--render` | Re-render the RESULTS table from the timing files already on disk, measuring nothing |
| `--id` / `--backend` | Restrict `--suite` to one entry / backend |
| `--fit-host` / `--skip-missing` | Skip entries too large for the host / with no GGUF present |
| `--max-load` | The 1-minute load average a timed run needs to be under (default `2.0`, raw, not per core). `0` waives this and the thermal and free-memory checks with it. `--suite` checks once up front, forwards the bar to every child, and waits between entries for the previous entry's own load to decay |
| `--bench-dir` | Where `suite.json`, `RESULTS.md` and `receipts/` live (default `benchmarks`) |
| `--receipt` | Write a single run's raw timings to this path |

A run stops before the timer starts when the host is busy, thermally
limited, or short enough on free memory that the weights would page to
disk. [`benchmarks/README.md`](../benchmarks/README.md) has each check,
what it reads, and what `--max-load 0` waives.

## Batched benchmark (`frink batched-bench`)

Throughput as a function of batch size, like
[`llama-batched-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/batched-bench):
for every combination of prompt length (`-npp`), generation length
(`-ntg`) and parallel sequences (`-npl`), one row with the prompt
speed, the decode speed and the total. Same ten columns, same widths
(`tools/batched-bench/batched-bench.cpp:128-129,245`), so the two
tables paste side by side; `--output-format jsonl` prints one object
per row with upstream's per-row keys.

```bash
./target/release/frink batched-bench -m model.gguf -c 2048 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32
./target/release/frink batched-bench -m model.gguf -c 2048 -npp 512 -ntg 128 -npl 1,4,16 -pps      # shared prompt
./target/release/frink batched-bench -m model.gguf -ngl 99 -npp 128 -ntg 128 -npl 8 --output-format jsonl
# the llama.cpp line to put beside it
llama-batched-bench -m model.gguf -c 2048 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32
```

```
|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    0.108 |  1186.64 |    3.079 |    41.57 |    3.187 |    80.32 |
```

`PP`/`TG` are per sequence, `B` is the sequence count, `N_KV = B*(PP+TG)`
is the KV the row needs, `S_PP` is `B*PP/T_PP` (or `PP/T_PP` with
`-pps`), `S_TG` is `B*TG/T_TG`, `S` is all tokens over `T_PP + T_TG`.
That is upstream's arithmetic (`batched-bench.cpp:229-235`), pinned by
a test.

What it drives is the continuous batcher's own engine seams, without
the server around them: each prompt goes through
`Decoder::forward_batch_last_host_kv` in `-ub` chunks, and every decode
step is one `Decoder::forward_multi_seq` call across the `B` sequences,
the same call `frink-server` makes per tick under
`FRINK_CONTINUOUS_BATCHING=1`. Two honest differences from upstream:
frink has no cross-sequence prefill, so `B` prompts are `B` calls
rather than one batch (the number reported is still the time to
prefill all of them); and the batched decode attends on the host on
every backend, so `-ngl` offloads the projections and not the
attention. `frink serve-bench` measures the same batcher over HTTP.

| Flag | Meaning |
|---|---|
| `-m/--model` | GGUF to benchmark (generic decoder architectures only; the dedicated engines have no multi-sequence step and are refused by name) |
| `-npp`, `-ntg`, `-npl` | Comma-separated sweeps; all three required, every value > 0 (upstream prints a `NaN` row for `0`, this refuses it) |
| `-pps` | One prompt shared by every sequence: prefilled once, its KV copied to the others between the two timers (`batched-bench.cpp:168-185`) |
| `-tgs` | Decode each sequence to completion in turn instead of one step across all of them per call (`:189-223`) |
| `-c` | `n_kv_max`; a combination needing more is skipped, and the skip is printed rather than silent. `0` = the GGUF's `{arch}.context_length` |
| `-ub` | Prompt tokens per forward call (default 512, upstream's `n_ubatch`) |
| `-t`, `-ngl` | As `frink bench` |
| `--output-format` | `md` (default) or `jsonl` |
| `--receipt`, `--backend-label` | Write a JSON receipt; the label must name the backend that ran or the receipt is refused, before the sweep and again at write time |
| `--max-load` | The same quiet-host bar as `frink bench`, waiving the thermal and free-memory checks with it at `0`. The free-memory check counts the largest row's KV on top of the weights |

Flags `llama-batched-bench` takes that this tool **refuses by name**
rather than accepting and ignoring: `-b` (`n_batch`; frink has no
logical batch distinct from `-ub`), `-kvu` (every sequence has its own
cache here, so `N_KV` is `B*(PP+TG)` with or without `-pps`), `-fa`
(no flash-attention switch to honour) and `-tb` (one thread pool).
For the same reason the JSONL rows omit `n_batch`, `flash_attn` and
`n_threads_batch` instead of printing a made-up value for them.

Every row runs twice: one discarded warmup pass and one timed pass,
and the two must have fed the same tokens and produced the same greedy
picks, per sequence, for prompt and decode. That is `frink bench`'s
determinism check, and it is the reason each row costs two passes
where upstream pays one global 16-token warmup. The other `bench_guard`
checks run too: cold caches per pass, prompt and decode lengths
re-read from every sequence's KV afterwards (the copied caches under
`-pps` included), and a rate that is not finite refuses the row.

### One model at a time

Every command that loads weights (`run`, `bench`, `verify`, `smoke`,
`run-kimi`, and `frink-server`) registers itself, and **stops with an
error when another frink process is already holding a model**:

```
$ frink -m model.gguf -p "hi"
Error: 1 frink instance(s) are already running a model on this host:
  - server pid 59667, metal, models/SmolLM2-135M-Instruct-Q8_0.gguf
Running several models at once does not share the machine -- it thrashes
it, and any timing either process reports is noise. Stop the other
instance, or pass --allow-multiple-instances (or set
FRINK_ALLOW_MULTIPLE_INSTANCES=1) to start anyway.
```

Prefill is a dense GEMM across every core, and the decode pool spins.
Two instances do not run at half speed each. They fight over the same
cores. Pass `--allow-multiple-instances` (or set
`FRINK_ALLOW_MULTIPLE_INSTANCES=1`) when you want them anyway.

Header-only commands (`inspect`, `inspect-plan`, `presets`, `archs`,
`caps`), the HTTP client (`chat`), the downloader (`pull`) and
`bench --suite` / `--render` are exempt. None of them puts weights in
memory, and `--suite` is a supervisor whose children each register on
their own.

The registry is a directory of one small file per live process
(`$FRINK_INSTANCE_DIR`, default `~/.cache/frink/instances`). When a
process is gone, after a `kill -9` or a crash, the next run prunes its
entry instead of being blocked by it. This is **advisory, not a lock**.
Two processes starting in the same instant each see the other and both
stop, which is the safe direction, and nothing here holds back a
determined caller.

Add or change models in [`benchmarks/suite.json`](../benchmarks/suite.json)
(`id`, `name`, `gguf`, `backends`, `estimated_ram_gb`). No HTTP, no
chat template, no tokenizer, no sampling. That is the same line
llama.cpp draws between `llama-bench` and `llama-server`. Details:
[`benchmarks/README.md`](../benchmarks/README.md).

See also: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md) · [`API.md`](API.md).
