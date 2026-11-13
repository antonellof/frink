# CLI

`frink` takes [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
command lines (ADR 0005): `frink run` is `llama-cli`, `frink serve` /
`frink-server` is `llama-server`, `frink bench` is `llama-bench`, with
the same flag names and meanings. Top-level `-m` / `-p` work without
typing `run`. This page lists where frink differs, and frink's own
features; it does not restate what a llama.cpp flag means.

`crates/frink-cli/tests/llama_cpp_flags.txt` is every flag of the three
tools at the llama.cpp pin. `crates/frink-cli/src/llama_flags.rs`
classifies each as accepted, deliberately different (the table below),
or not accepted, and pins the not-accepted count per tool so it only
goes down. `cargo test -p frink-cli llama_flags -- --nocapture` prints
the list.

Two rules hold for every flag:

- A single-dash name is ONE flag, as in llama.cpp: `-ngl`, `-hff`,
  `-ctk`, `-np` are read whole. One that names no flag is an error, not
  a cluster of short options.
- A flag frink accepts and cannot honour is an error, never ignored.

```bash
cargo build --release -p frink-cli --features metal   # macOS Metal + CPU
cargo build --release -p frink-cli                    # CPU only
```

Binary: `./target/release/frink`. One executable covers every backend
compiled into it; pick one at runtime with `-dev` / `-ngl`.

## Deliberate differences from llama.cpp

Each row is a llama.cpp flag frink accepts with a different value
grammar, default or reach. A test requires every row to name a real
flag of its tool.

| Tool | Flag | frink | Why |
|---|---|---|---|
| cli | `-n` | Default `128`; llama.cpp's is `-1` (to EOS) | A bare `frink -m x -p hi` should end. `-n -1` is accepted and fills the context |
| cli | `--repeat-penalty` | Default `1.1`; llama.cpp's is `1.0` (off) | A deliberate default (`docs/FEATURES.md`); `--repeat-penalty 1.0` gives upstream's tokens |
| cli, server | `-dev` | One of `auto`, `none`, `cpu`, `metal`, `cuda`, `vulkan`; llama.cpp takes a device list (`CUDA0,Vulkan1`) | frink places a model on one backend. `vulkan` runs on `Vulkan0`; pick another with `GGML_VK_VISIBLE_DEVICES`, read as llama.cpp reads it (unset is every discrete and integrated GPU, so a CPU driver such as lavapipe must be named). Unlike llama.cpp, a token that is not an index is an error, a device without 16-bit storage buffers is not dropped, and one GPU exposed by two drivers (RADV and AMDVLK) is not merged |
| cli, server | `-ngl` | `0`, `auto`, `all`, or a count at or above the layer count. A PARTIAL count is refused | frink has no partial layer placement. Use `-ngl 0` for CPU or `-ngl all` for the whole model |
| bench | `-ngl` | A count: `0` (the default) is CPU, the layer count or more offloads every layer, a partial count is refused | Same |
| cli, server | `-ctk` | llama.cpp's types are accepted; `f16`, `q8_0`, `q4_0` and frink's `fp8` are served, the rest fall back and the banner says so. See [`--ctk` on each backend](#completion-flags) | Only Metal's device store takes a quantized type |
| server | `-c` | `0` is refused. Unset derives the ceiling from the device budget, capped at the trained context | llama.cpp's `-c 0` means the trained context, which may not fit; omit the flag instead |
| server | `-np` | The cap on concurrent sequences under continuous batching; implies `-cb`. `-1` (auto) is refused, omit it | frink has no fixed slots |
| server | `-b` | `-b` and `-ub` resolve to ONE prefill chunk (the smaller named). `/infill` sizes its prefix and suffix by `-b` alone, as llama.cpp does | No logical/physical batch split |
| server | `-a` | One name; a comma-separated list is refused | `/v1/models` serves one model name |
| cli, server | `--jinja` | Always on; `--no-jinja` is refused | frink has no template-free mode |
| cli, server | `--chat-template` | Jinja source, or `chatml`. Any other bare name is refused, and `--chat-template-file` follows the same rule | Under `--jinja` a name like `llama3` compiles as plain text; llama.cpp's named templates need `--no-jinja` |
| cli, server | `--chat-template-kwargs` | A JSON object, repeatable. On the server the flag replaces `LLAMA_ARG_CHAT_TEMPLATE_KWARGS` rather than merging with it | A variable stands in for its flag only when the flag is absent |
| cli, server | `-rea` | Refused beside a `--chat-template-kwargs` that sets `enable_thinking` to the other value | frink does not keep the order across two flags |
| cli, server | `--reasoning-effort` | Refused beside a `--chat-template-kwargs` that sets `reasoning_effort` to another value | Same |
| cli, server | `--reasoning-preserve` | Refused beside a `--chat-template-kwargs` that sets `preserve_reasoning` to the other value. `--no-reasoning-preserve` likewise | Same |
| cli, server | `-fa` | `on` / `auto` accepted, `off` refused. A bare `-fa` means `auto` | Fused attention is a backend property; use `FRINK_METAL_ATTN=0` or `--device cpu` |
| bench | `-p` | One value per run; lists and ranges are refused | Run the command once per value |
| bench | `-n` | One value per run, as `-p` | Same |
| bench | `-d` | One value per run, as `-p` | Same |
| bench | `-b` | One value per run. Leaving both `-b` and `-ub` out means the whole prompt in one pass, not llama-bench's 2048 / 512 | Keeps the rows every published receipt measured |
| bench | `-ub` | One value per run; see `-b` | Same |
| bench | `-pg` | One pair per run; refused on the Gemma-4 engine, which has no batched prefill to time | Same |
| bench | `-m` | One model per run | `frink bench --suite` runs several, each in its own process |
| bench | `-fa` | One value per run: `on` / `auto` accepted, `off` and comma-separated sweeps refused | One run measures one configuration |
| bench | `-ctk` | `f16` (the default) or `f32` on CPU and Metal; a type the KV store would fall back from (`q8_0`, `q4_0`, ...) is refused, as is a sweep | A row labelled with a type it did not measure is not a measurement |
| bench | `-ctv` | Must equal `-ctk` | The KV store holds K and V in one type |
| bench | `-o` | `csv`, `json`, `jsonl` and `sql` carry llama-bench's field names and statistics (`avg_ts` the mean, `stddev_ts` the sample stddev), but llama.cpp-only fields (`cpu_mask`, `poll`, `split_mode`, `tensor_split`, `n_batch`, `n_ubatch`, `build_number`, ...) are absent rather than defaulted. `md` is frink's own table | A defaulted field would read as a measurement |
| cli | `-co` | Colours the prompt echo and the timing lines on stderr; stdout is never coloured. A bare `-co` means `on` | stdout stays plain so it pipes; `NO_COLOR` still wins under `auto` |
| server | `--metrics` | Accepted and changes nothing: `/metrics` is always served | No switch to turn the endpoint off |
| server | `--cors-origins` | Exact origins only. llama.cpp's default `*` and its special value `localhost` are refused | No wildcard; without the flag CORS is off |

## Completion (`run`)

### Quick examples

```bash
# Greedy completion (raw prompt)
./target/release/frink -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p "The capital of France is" -n 32 --temp 0 --no-cnv

# Chat-tuned wrap (default when GGUF has tokenizer.chat_template)
./target/release/frink -m models/hf_test/SmolLM2-135M-Instruct-Q8_0.gguf \
  -p "What is 2+2?" -n 64 --temp 0 --system "Answer briefly."

# Prompt from file + escapes; sampling; threads + context
./target/release/frink -m model.gguf -f prompt.txt -e -n 128
./target/release/frink -m model.gguf -p "Once upon a time" \
  -n 256 --temp 0.8 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 -s 42
./target/release/frink -m model.gguf -p "Hi" -n 64 -t 8 -c 4096
./target/release/frink -m model.gguf -p "Hi" -n 64 -c auto   # largest context that fits

# List devices, then select Metal (requires a --features metal build)
./target/release/frink --list-devices
./target/release/frink -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p "Hello" -n 64 --temp 0 -dev metal -ngl all

# Force CPU with the same Metal-capable executable
./target/release/frink -m model.gguf -p "Hello" -n 64 -dev none -ngl 0
```

Same via explicit subcommand: `frink run -m …`.

### Completion flags

These have llama.cpp's meaning (differences above): `-m` / `--model`,
`-hf` / `-hfr` / `--hf-repo`, `-hff` / `--hf-file`, `-p` / `--prompt`,
`-f` / `--file`, `-n` / `--n-predict` / `--predict`, `-c` / `--ctx-size`,
`-t` / `--threads`, `--temp` / `--temperature`, `--top-k`, `--top-p`,
`--min-p`, `--repeat-penalty`, `--repeat-last-n`, `--presence-penalty`,
`--frequency-penalty`, `--typical` / `--typical-p`, `--top-nsigma` /
`--top-n-sigma`, `--xtc-probability`, `--xtc-threshold`,
`--dry-multiplier`, `--dry-base`, `--dry-allowed-length`,
`--dry-penalty-last-n`, `--dry-sequence-breaker`, `--samplers`,
`--sampler-seq` / `--sampling-seq`, `-s` / `--seed`, `--grammar`,
`--grammar-file`, `-j` / `--json-schema`, `-dev` / `--device`,
`--list-devices`, `-ngl` / `--gpu-layers` / `--n-gpu-layers`, `-ctk` /
`--ctk` / `--cache-type-k`, `-md` / `-d` / `--model-draft` /
`--spec-draft-model`, `--spec-draft-n-max` (old spellings `--draft-max` /
`--draft` / `--draft-n` kept), `--spec-draft-n-min`, `--spec-type`,
`--spec-draft-p-min` / `--draft-p-min`, `--lora`, `--lora-scaled`,
`-sys` / `--system` / `--system-prompt`, `-sysf` /
`--system-prompt-file` (one trailing newline dropped, as upstream),
`-jf` / `--json-schema-file`, `-hft` / `--hf-token` (sets `HF_TOKEN`),
`--offline` (`-hf` resolves from the cache only, no network),
`--warmup` / `--no-warmup`, `--show-timings` / `--no-show-timings`,
`-co` / `--color`, `--jinja`, `--chat-template`, `--chat-template-file`,
`--chat-template-kwargs`, `-rea` / `--reasoning`, `--reasoning-effort`,
`--reasoning-preserve` / `--no-reasoning-preserve`,
`-fa` / `--flash-attn`, `-st` /
`--single-turn`, `-e` / `--escape`, `--no-escape`, `--ignore-eos`,
`--verbose-prompt`, and `--usage` (llama.cpp's third spelling of
help) and `--version` on every subcommand.

frink's own:

| Flag | Notes |
|---|---|
| `-c auto` | The largest context that fits the device memory budget, decided before the weights load; the arithmetic is printed to stderr |
| `--strict-budget` | Stop with an error when the pre-load budget says the context will not fit (default: warn and continue, since mmapped weights can page) |
| `--no-cnv` | Raw prompt, no chat template (llama-completion's spelling). Refused together with `--system`, which only the template can place |
| `--ctk fp8` | The Q8_0 wire under another name |
| `--mtp` | Errors: MTP draft heads (`num_nextn_predict_layers`) are not loaded from GGUF yet |

The dedicated engines (MLA, Gemma-4, GLM-5.2) refuse `--lora` and
`--model-draft` by name.

Generated text goes to stdout; load and throughput timings go to stderr
(`--no-show-timings` drops the timing lines, the speculative summary
included). `--no-warmup` skips the throwaway forward pass before the
timed prompt, so the prompt rate then includes first-dispatch cost.

**Structured output.** `--grammar`, `--grammar-file` and `-j` all end
at one stack machine that masks every token that cannot continue a
valid string; a schema is compiled to GBNF first. There is no retry or
repair pass. The same machine serves `response_format` and
`tool_choice` on the HTTP API ([docs/API.md](API.md)).

**`--ctk` on each backend.** The host KV cache (CPU, and the host side
of Metal) keeps `f16` by default and `f32` under `--ctk f32`, as
llama.cpp does. The block-quantized wires (`q8_0`, `fp8`, `q4_0`) exist
only in the Metal device store; on the host they fall back to `f16` and
the startup banner says so. CUDA keeps an `f32` host store, and the
banner says that too. `--ctx-size auto` prices the store the run really
builds.

**Sampler order.** llama.cpp's chain in llama.cpp's order: penalties,
DRY, top-n-sigma, top-k, typical-p, top-p, min-p, XTC, temperature
(`crates/frink-models/src/sampling.rs`). All nine run by default; the
four with no OpenAI equivalent sit at neutral values that make them
no-ops. The repetition penalty is applied once per candidate, not once
per occurrence.

`--samplers` sets the order as a semicolon-separated list; the default
is `--samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"`.
llama.cpp's aliases parse (`top-k`, `nucleus`, `temp`, `typical`).
`--sampler-seq` gives the chain in llama.cpp's characters
(`edskypmxt`); an unknown character is refused where llama.cpp warns
and drops it. `penalties` must come first and `temperature` must be
present. A sampler frink does not implement is refused by name, never
skipped: `mirostat` (upstream runs it in place of the chain) and
`infill` (needs the model's FIM tokens).

**Speculative decoding.** `-md` / `-d` / `--model-draft FILE` loads a
smaller GGUF of the same family and tokenizer that proposes tokens; the
target verifies them in one batched pass. The output is exactly what
the target would write alone, at every temperature.

`--spec-draft-n-max` (default 3, as upstream) sets tokens proposed per
verification step, `--spec-draft-n-min` (default 0) drops a shorter
draft, and `--spec-draft-p-min` (default 0, never) stops a draft once
its own confidence drops below it. The pre-rename spellings
(`--draft-max`, `--draft`, `--draft-n`, `--draft-p-min`) still work.

A DFlash or DFlash2 draft (`general.architecture` = `dflash`, as
llama.cpp's converter writes it) is detected from the file.
`--spec-type draft-dflash` asks for it explicitly, `draft-simple` asks
for an autoregressive draft, `none` turns speculation off, and every
other upstream type is refused by name. A DFlash draft reads the
target's residual stream and drafts a whole block in one pass;
`--spec-draft-n-max` is clamped to its trained block minus one. Refused:
a draft whose widths, token table or target layers do not fit the
target, and the DSpark, LiLiCorr and reduced-vocabulary variants.

A target with recurrent (Mamba) layers is served (its caches are
checkpointed around each verify batch). `-md` is refused with
`--grammar` / `--grammar-file` / `-j`, without a prompt, when the draft
has recurrent layers, and when an autoregressive drafter's KV would
live on the device (a Metal or CUDA drafter; `--device cpu` runs it).
`frink-server` takes `-md` too, and drafts by prompt lookup without it
(see [Server flags](#server-flags)).

`frink speculative` is a demo on synthetic random weights that drafts
by n-gram lookup; its hit rate says nothing about a real drafter.
`--ngram-size` (default 4) is the n-gram length and `--max-draft-len`
(default 8) caps tokens drafted per step.

**Devices.** `--device none` (or `cpu`) and `-ngl 0` force CPU, Vulkan
included. `--device vulkan` runs on Vulkan (a build with cargo's
`vulkan` feature). The default is `--device auto -ngl auto`, under which
Vulkan stays opt-in (`FRINK_VULKAN=1`). `auto`, `all`, or a count at or
above the layer count enable all supported ops on the selected backend;
a partial count is refused.

### Chat vs completion

- Default: the prompt is rendered through the GGUF's own
  `tokenizer.chat_template` by the same Jinja2 evaluator `frink-server`
  uses, so the CLI and `/v1/chat/completions` frame a conversation
  identically. No template falls back to ChatML (as llama.cpp
  `--jinja`), or to role-labeled lines for a byte tokenizer. A template
  that does not compile is an error.
- `trim_blocks` and `lstrip_blocks` are on, as in llama.cpp. Real
  templates are pinned against jinja2 goldens
  (`python3 scripts/chat_template_goldens.py` regenerates them), and
  CI's oracle job renders every template llama.cpp bundles; refusals
  there are listed in `EXPECTED_REFUSALS`.
- Every render carries what llama.cpp's do, under the request's own
  kwargs: `enable_thinking`, `date_string` and `datetime` (local time,
  as is `strftime_now`), `reasoning_strength` from `reasoning_effort`,
  and the four spellings of `preserve_reasoning`. `enable_thinking` is
  `true` where the template supports thinking, on `frink run` and
  `frink-server` alike; `-rea off` turns it off.
- One deviation: whether a template supports thinking is decided by
  asking whether turning thinking on changes what follows the user's
  turn, a simpler test than llama.cpp's template analysis. On every
  template llama.cpp bundles the answers agree wherever it reaches the
  prompt (`llama_renders::census` checks this in CI). A template outside
  that set can get the other default.
- Maps keep their key order, as in llama.cpp. `{{ x }}` and
  `x | string` print a list or map in Python literal syntax
  (`{'type': 'function'}`); `x ~ y` joins minijinja's JSON-like text.
- A conversation is reshaped where llama.cpp reshapes it
  (`frink_models::chat_template::reshape`): a system turn the template
  does not show folds into the next message; a replayed call's
  `arguments` stay a JSON string unless the template reads them as an
  object; Ministral 3, Devstral, K2 Horizon, Gemma 4's older template
  and TranslateGemma get their llama.cpp handler's shape. frink learns
  both properties at load by checking what a render prints, where
  llama.cpp checks what it reads.
- One deviation: a string plus an undefined value, or plus `none`, is
  an error, as in jinja2, where llama.cpp reads the missing side as
  empty. frink refuses the request with a 400 naming the template line.
  Two bundled templates reach it: Firefunction v2 on every request, and
  Command R+ (tool use) when a tool parameter has no description. The
  render census lists them under `deviation:string-plus-undefined`
  (#522).
- An unimplemented Python method is an error, as is mutating a list
  outside a `namespace(...)`. `.format()` takes bare `{}` placeholders
  and `.replace()` takes no count, as in llama.cpp.
- `-st` / `--single-turn`: accepted so a llama.cpp command line runs
  unchanged; a no-op, since `frink run` generates once and exits. It
  keeps the chat template; it is not `--no-cnv`.
- `--no-cnv`: raw prompt (classic completion). BOS is still added under
  the same rule.

### Who adds BOS

The chat template owns BOS when it prints one (most upstream templates
open with `{{ bos_token }}`, which encodes to the BOS *id* because
special-token markers are parsed, as llama.cpp's `parse_special = true`);
otherwise the loader does (Unsloth exports strip it, as does
TinyLlama's template). frink adds the id *idempotently*
(`frink_models::tokenizer::prepend_bos`), since this varies per
checkpoint. Whether BOS is added at all is llama.cpp's `add_bos` rule
(`tokenizer.ggml.add_bos_token` if present, else SPM → yes / BPE → no).
Qwen2's `bos_token_id` is never prepended.
`cargo test -p frink-models --test bos_policy -- --ignored --nocapture`
asserts at most one leading BOS id on every local checkpoint.

### When generation stops

On the whole end-of-generation set: the `eos`/`eot`/`eom` metadata ids
plus every vocabulary entry on llama.cpp's literal EOG list
(`<|eot_id|>`, `<end_of_turn>`, `<|im_end|>`, `<turn|>`, …), not
`tokenizer.ggml.eos_token_id` alone (a Llama-3 checkpoint's is
`<|end_of_text|>` while its turns end with `<|eot_id|>`).
`--ignore-eos` disables all of it. `frink-server` uses the same set.

## Matching llama.cpp's launcher

llama.cpp's 0.4 launcher is `llama <command>`; the same command lines
work here:

| `llama` | `frink` |
|---|---|
| `llama cli -m model.gguf -p "Hi" -st` | `frink cli -m model.gguf -p "Hi" -st` |
| `llama serve -m model.gguf` | `frink serve -m model.gguf` |
| `llama download …` | `frink download …` |
| `llama version` | `frink version` |
| `llama licenses` | `frink licenses` |
| `llama help` | `frink help` |

`cli` is an alias of `run`. `version` and `licenses` exist as
subcommands as well as `--version`.

`frink serve` logs to stderr in llama.cpp's shape
(`0.00.910.529 I srv  frink_server: model loaded`); the
`frink.server.ready` JSON line stays on stdout (see
[Running under a supervisor](#running-under-a-supervisor)).

The run output carries llama.cpp's block (`build`, `model`, `ftype`,
`modalities`), plus a `device` row, and closes with
`[ Prompt: X t/s | Generation: Y t/s ]`. The prompt rate is timed after
the warm-up pass, as llama.cpp's is.

## Other commands

```bash
./target/release/frink version
./target/release/frink licenses
./target/release/frink inspect models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
./target/release/frink inspect-plan models/olmoe-1b-7b-0924-q4_0.gguf --strict
# Plan against a backend's real memory budget; reports the largest context that fits
./target/release/frink inspect-plan model.gguf --backend metal --ctk f16
./target/release/frink caps
./target/release/frink caps --kernels      # kernel coverage per kind and backend
./target/release/frink archs
./target/release/frink presets
./target/release/frink smoke glm-5.2

# Kimi K3 safetensors directory (large checkpoint, see MODELS.md)
./target/release/frink run-kimi /path/to/kimi --prompt "Hi" --max-new-tokens 32
```

- `inspect-plan --context N` sizes each request's KV cache for `N`
  tokens (default: the model's `{arch}.context_length`).
  `--concurrency N` (default 1) budgets that many caches.
  `--expert-cache-bytes N` streams routed experts through a bounded
  cache of `N` bytes. The output ends with the kernel coverage of the
  file's quantized tensors on `--backend`: one line per kind and op
  that falls back to the host (or, on CPU, has no integer `vec_dot`).
- `caps --kernels` prints the kernel coverage matrix (every quant kind,
  backend and op) from the dispatch tables, so it is the same on every
  machine. `caps --kernels --write PATH` regenerates `docs/KERNELS.md`.
- `archs --write PATH` regenerates
  `docs/manifests/architecture_manifest.md`.
- `smoke --steps N` sets the smoke test's decode steps (default 8).
- `run-kimi` samples with `--temperature` (default `0.0`, greedy),
  `--top-p` (default `1.0`), `--top-k` (default `0`, off),
  `--repetition-penalty` (default `1.0`, off; looks back 64 tokens) and
  `--seed` (default `0`); `--max-new-tokens` defaults to 64.
- `run-real PATH --fixture dense|moe --token N --pos N` prints one
  decode step's logits for the synthetic test fixtures.

## Serving benchmark (`serve-bench`)

`frink bench` measures kernels with no HTTP; `frink serve-bench`
measures a running `frink-server` under concurrency.

```bash
# Start the server first.
FRINK_MODEL_PATH=model.gguf ./target/release/frink-server &

./target/release/frink serve-bench --requests 64 --concurrency 8 --output-len 128
./target/release/frink serve-bench --concurrency 16 --json

# What the prefix cache saves: one system prompt, many questions.
./target/release/frink serve-bench --shared-prefix 4000 --requests 32
./target/release/frink serve-bench --shared-prefix 4000 --cache-salt tenant-a
```

Every request runs at temperature 0, top-k 1, `ignore_eos` and an exact
output length. The first token-bearing chunk is TTFT, later ones are
inter-token samples (keepalives and the `finish_reason` frame
excluded). Percentiles are nearest-rank over samples pooled across
requests; throughput is total tokens over the run's span. Token counts
and prefix reuse come from the server's `usage.completion_tokens` and
`usage.cached_tokens`; a buffered stream reports no TPOT, and a server
with no prefix cache says so. The rules live in
`frink_cli::bench_client`, unit-tested.

`--shared-prefix N` puts an identical `system` message of N characters
ahead of every request; without it each request gets its own filler.
`--cache-salt` puts the run in its own cache namespace.
`--prompt-chars N` (default 512) sets each prompt's length in
characters; the report gives the prompt-token count the server saw.

## Bandwidth profile (`bench-bw`)

Measures the CPU and PCIe bandwidth `frink-core`'s `qstar` uses to split
a MoE layer between fetching and computing on the CPU. Without a
profile it falls back to one fetch per layer per step.

```bash
cargo build --release -p frink-cli --features cuda
./target/release/frink bench-bw --format q4_k
./target/release/frink bench-bw --dry-run          # measure, write nothing
```

`--bytes` (default 536870912, 512 MiB, must exceed the last-level
cache) is moved per timed pass; `--reps` (default 5) passes, best kept;
`--threshold` (default `1.0`): compute on the CPU when
`cpu_bw > threshold * pcie_bw`. It writes `$XDG_CACHE_HOME/frink/benchbw/<gpu-uuid>.json`, which the
loader finds on its own; a profile whose recorded GPU name disagrees is
ignored. It refuses to write when only one side was measured (the PCIe
half needs a CUDA build), and from an unoptimized build unless
`--allow-debug-build` is given. The device-side measurement is a stub
pending a benchmark host.

## Correctness (`verify`, `parity`)

```bash
# Do frink's own backends agree? (CPU reference vs Metal/CUDA)
./target/release/frink verify -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --backend metal --prompt-tokens 64

# Does frink agree with llama.cpp? (first-token distribution, CPU vs CPU)
./target/release/frink parity -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --prompt-tokens 64
```

`verify` greedy-decodes the same prompt on two frink backends and diffs
the token ids; it cannot catch a bug both share. `parity` compares
against llama.cpp on the same GGUF: the tokenizer first, then the graph.

### The tokenizer half

Frink's token ids against llama.cpp's for a fixed 22-case corpus built
from what pre-tokenizer regexes differ on (digit runs, space runs,
indents, tabs, CRLF, contractions, CJK, emoji and ZWJ, Unicode
whitespace, control bytes, punctuation runs, version strings, non-Latin
scripts and non-ASCII digits, special-token markers). Each
diverging case prints the token index, the byte offset, the input
around it and both engines' ids and pieces. Ids are compared with `add_special = false`; the add-BOS decision and
the vocab sizes are compared separately. Both halves are always
reported, and either diverging exits non-zero.

### The logit half

`parity` feeds the same token ids to both engines and compares the
logit distribution at the last prompt position: KL both ways, total
variation, max |delta p|, top-k overlap, and where llama's top-1 ranks
for frink. Verdicts:

| Verdict | Meaning |
|---|---|
| `MATCH` | same distribution to within f32 accumulation-order noise |
| `DRIFT` | same top-1, distributions moved further than reordering explains |
| `TIE-FLIP` | top-1 differs, but llama's own top-2 margin is under the observed noise, so a tie swapped rather than the graph being wrong |
| `WRONG` | the graphs disagree, and the command exits non-zero |

On a quantized file, `WRONG` does not say which side moved (llama.cpp
quantizes activations to 8 bits, frink keeps f32); the dequantized file
arbitrates:

```bash
PYTHONPATH=$LLAMA/gguf-py python3 scripts/dequantize_gguf.py model-Q8_0.gguf /tmp/model-f32.gguf
./target/release/frink parity -m /tmp/model-f32.gguf   --dumper target/llama_logits --dump-logits /tmp/f32
./target/release/frink parity -m model-Q8_0.gguf       --dumper target/llama_logits --dump-logits /tmp/q8
# KL(f32.llama || q8.llama) is the reference's own quantization loss;
# KL(f32.llama || q8.frink) is frink's; KL(f32.llama || f32.frink) is the graph.
```

`LLAMA_LOGITS_FLASH_ATTN=0` keeps the reference off its flash-attention
path. `FRINK_CTK=f32` runs an f32 K/V cache on BOTH sides; the dumper
reads the same variable.

`parity` leaves the integer-dot tier off unless `FRINK_CPU_INT_DOT=1`,
while `frink run` and the server turn it on where it wins. Set it to
compare the path they take; only then does frink quantize activations
the way llama.cpp's CPU backend does.

### Per layer (`--dump-tensors`)

`--dump-tensors DIR` makes both engines write the prompt's
intermediates under llama.cpp's tensor names (`attn_norm-N`, `Qcur-N`,
`kqv_out-N`, `attn_out-N`, `ffn_inp-N`, `ffn_norm-N`, `ffn_out-N`,
`l_out-N`, `result_norm`) and prints, per tensor, the relative L2
distance of frink's from llama.cpp's: over all rows, for the worst row,
and for the output row. The dumper writes through llama.cpp's eval
callback (`LLAMA_LOGITS_DUMP_TENSORS`, `tools/llama_logits.c`), and
`parity` checks its logits are unchanged by it. Generic decoder only,
on the CPU.

An op whose distance jumps from its input's is where the difference
enters. On int8-activation paths it grows layer by layer after the
first rounding flip; compare the second dumper's library the same way
to see the spread between two llama.cpp builds.

### The decode half, and token ids

`--decode-steps N` adds a greedy decode of N steps: llama.cpp feeds back
its own argmax and frink is fed the same tokens, so every step compares
two distributions over one context. The worst step is the verdict.

`--tokens 3,7,11` compares on those ids and skips the tokenizer half,
for synthetic fixtures whose vocabularies libllama cannot load.

### The oracle over every fixture

`scripts/oracle.sh` is CI's `oracle` job, and runs on a laptop (CPU
only, a few minutes once libllama is built):

```bash
scripts/oracle.sh                       # llama.cpp at tools/llama_cpp_pin
scripts/oracle.sh --llama-ref master    # drift report; add --report-only to exit 0
scripts/oracle.sh --golden-tol 1e-3     # on a laptop: ignore the ISA spread
```

It builds libllama at the pin under `target/oracle/<sha>/` (from a
local `.scratch/llama.cpp*` clone holding the commit, else GitHub),
regenerates every fixture and golden, then runs `parity --tokens
--decode-steps 32` on every fixture. It fails when a regenerated file
differs from the committed one, or an outcome differs from
`tools/oracle/fixtures.tsv`. `tools/oracle/goldens.tsv` names each
golden's dumper invocation; `tests/oracle_manifest.rs` fails one with
no row. The committed goldens are CI's answer (`GGML_NATIVE=OFF`);
another host differs by up to about 4e-4, so commit goldens only from
CI's `oracle-rewritten-goldens` artifact.

### The reference dumper

Both halves need it. It is C, outside the cargo workspace, linked
against llama.cpp's own library:

```bash
./tools/build_llama_logits.sh          # -> target/llama_logits
LLAMA_CPP_PREFIX=/path/to/llama.cpp ./tools/build_llama_logits.sh
```

`cargo clean` removes it. Point `--dumper` or `FRINK_LLAMA_LOGITS` at a
build elsewhere. A dumper with no `--tokenize` mode is too old, and
`parity` says so. `cargo test -p frink-cli -- --ignored
frink_and_llama_cpp_tokenize_the_corpus_identically --nocapture` sweeps
the tokenizer half over every checkpoint under `models/`, skipping by
name those missing or that `libllama` cannot load.

## Diagnostics (`layer-divergence`, `quant-sensitivity`)

`verify` says *which token* two backends stopped agreeing on;
`layer-divergence` says *which layer*.

```bash
./target/release/frink layer-divergence -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --backend metal --prompt-tokens 16
```

It runs one prefill per backend (a child process each), reads every
layer's KV back and prints the spread and mean of the per-head
magnitude ratios. A first divergence at layer L puts the fault in layer
L's norm/QKV projection or in its input. `--tol` defaults to `1e-3`.
`--at all` (default) scores every prompt position, `--at last` only the
final one. MoE checkpoints also get the total-variation distance between
the backends' expert-selection histograms (`no counts`: one side
recorded none).

```bash
./target/release/frink quant-sensitivity -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --candidate q4_0 --prompt-tokens 16 --top 10
```

`quant-sensitivity` round-trips one tensor at a time through a
candidate format, scores `relative_mse` per block, swaps it into the
loaded model and reports the next-token KL (nats), with a rollup of each
tensor family's share. It runs on CPU, one forward pass per tensor, and
refuses to start with `FRINK_CPU_INT_DOT=1`. `--layers 0:4` restricts
the sweep; on a MoE checkpoint `--experts N` (default 1) probes the
first `N` routed experts of each layer.

## Perplexity (`frink perplexity`)

llama.cpp's `perplexity` tool: corpus evaluation, and the acceptance
test for the quantizer.

```bash
frink perplexity -m model.gguf -f corpus.txt --ctx-size 512
```

On CPU it lands within a fifth of one standard error of
`llama-perplexity` on the same corpus and checkpoint (SmolLM2-135M,
Qwen3-0.6B, TinyLlama-1.1B). frink reads slightly lower on quantized
files because it keeps activations in f32.

The method is `tools/perplexity/perplexity.cpp`'s (non-overlapping
windows of `--ctx-size`, the second half of each scored). Deviations: one `forward_batch` per window rather than
an `n_batch` split, and the output head runs at every position.
`--ppl-stride`, HellaSwag, WinoGrande, multiple-choice and
KL-divergence are not implemented. Metal and CUDA perplexity is
unevidenced.

## Quantize (`frink quantize`)

Writes a `Q8_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q5_K_M` or `Q6_K` GGUF
from an F32/F16/BF16 one, byte-identical to `llama-quantize`'s, and
refuses every other target by name (Q2_K/Q3_K, the IQ tiers, MXFP4 and
the legacy Q4_0 family need their own iterative fits).

```bash
frink quantize model-f16.gguf model-q4_k_m.gguf --type q4_k_m
frink quantize model-f16.gguf model-q4_k_m.gguf --type q4_k_m --imatrix imatrix.gguf
frink quantize model-f16.gguf model-q4_k.gguf   --type q4_k_m --pure   # no per-tensor mix
```

An existing output file is not overwritten without `--force`.

Which tensors are quantized follows llama.cpp's
`tensor_allows_quantization`: everything 2-D ending in `weight`, except
norms, router gates, position and token-type embeddings, SSM and
shortconv kernels, RWKV time-mix, T5 position bias, multimodal patch
tables and audio codebooks. The per-tensor mix is llama.cpp's too (a
`Q4_K_M` file has a Q6_K output head and Q6_K `ffn_down` on a quarter
of its layers); `--pure` is `llama-quantize --pure`, which skips the mix
and not the keep-list.

Every tensor of every target matches `llama-quantize` b7650 byte for
byte, with and without an imatrix; only the header key order differs
(frink writes it sorted).

### `--imatrix`

The weighted fit is llama.cpp's `quantize_row_q{4,5,6}_K_impl` at
b7650, implemented as a parameter on the same super-block fit
(`frink-quant`'s `encode/fit.rs` and `encode/qp_quants.rs`). Q8_0
ignores the imatrix.

Each tensor looks up its own name. A tensor with no entry is quantized
unweighted with a printed notice (usually `output.weight` and
`token_embd.weight`, which `llama-imatrix` collects only with
`--process-output`). An entry of the wrong width is refused except on
`token_embd.weight`. Both the GGUF format and the legacy `.dat` binary
are accepted.

A tensor whose row width is not a multiple of 256 stops the run:
llama.cpp switches such a tensor to Q5_0 or F16, and frink has neither
encoder. SmolLM2-135M (embedding 576 wide) cannot be Q4_K quantized
here.

## Importance matrix (`frink imatrix`)

llama.cpp's `llama-imatrix`: runs calibration text through the model
and writes, per weight, the per-column sum of squared activations the
quantizer weights its fit by. Same flags where both have them, same
file format both ways: a frink file feeds `llama-quantize --imatrix`
and a `llama-imatrix` file feeds `frink quantize --imatrix`.

```bash
frink imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.gguf
frink imatrix -m model-bf16.gguf -f calibration.txt -o imatrix.dat --output-format dat
frink imatrix -m model-bf16.gguf -f calibration.txt --chunks 64 -c 512 --process-output
frink imatrix -m model-bf16.gguf -f calibration.txt -o mine.gguf --compare theirs.gguf
```

The method is `tools/imatrix/imatrix.cpp` at b7650 (cited in the module
doc): the input of every matmul whose weight is under `blk.`, plus
`output.weight` with `--process-output`, experts per expert;
non-overlapping chunks of `--ctx-size` with BOS over each chunk's first
token, at least two chunks of tokens required. It runs on the CPU, and
a weight whose row count disagrees with the token count is a refusal.
An expert never routed to is reported as partial data.

Deviations: one chunk per forward pass (same sums); no perplexity
printed (use `frink perplexity`); no `--in-file` combining of earlier
matrices; expert streaming is refused.

Names, counts and shapes match llama.cpp's, so the files are
interchangeable; the sums differ slightly because the forward passes
do. `--compare` prints the gap per entry. For a fair comparison the
text must tokenize identically on both engines (`frink imatrix` leaves
special-token markers as text, as `llama-imatrix` does) and run the
same number of chunks.

## Split and merge GGUF (`frink gguf-split`)

llama.cpp's `llama-gguf-split`: same flags, shard names and metadata
keys. Splitting is the default; `--merge` is the other direction.

```bash
frink gguf-split --split-max-tensors 128 model.gguf out/model   # by tensor count
frink gguf-split --split-max-size 4G model.gguf out/model       # by size; 4G is 4,000,000,000
frink gguf-split --split-max-size 4G --no-tensor-first-split model.gguf out/model  # metadata-only first shard
frink gguf-split --split-max-size 4G --dry-run model.gguf out/model  # plan only, nothing written
frink gguf-split --merge out/model-00001-of-00003.gguf model.gguf   # input is the FIRST shard
```

Shards are `<prefix>-NNNNN-of-MMMMM.gguf`, 1-based, with `split.no`
(u16, 0-based), `split.count` (u16) and `split.tensors.count` (i32, the
whole set's total). The first shard holds the full source metadata.
`frink run -m out/model-00001-of-00003.gguf` runs the set, and
llama.cpp reads it too. Tensor bytes are copied straight from the mmap,
so a split needs no more memory than a header.

Refusals llama.cpp's tool does not make:

* Splitting a file that is already a shard.
* A first tensor larger than `--split-max-size` (named, with both
  numbers).
* `--merge` with any `--split-*` option.
* A missing shard, named at plan time, before the output is opened.

Byte-level differences: metadata keys are written sorted, and shards
are padded with the source's alignment rather than always 32. Sets
split by either tool merge in the other.

## Put a reranker's pooler back (`frink splice-pooler`)

`BertForSequenceClassification` reranker GGUFs lack their pooler
(llama.cpp's converter deletes `bert.pooler.dense`), so the head runs
as `classifier(cls)` instead of `classifier(tanh(pooler(cls)))` and its
scores are about fifty times narrower than the model card's (issue
#82). This writes a GGUF that carries it.

```bash
frink download cross-encoder/ms-marco-MiniLM-L6-v2 model.safetensors --local-dir models/ms-marco-MiniLM-L6-v2
frink splice-pooler -m models/ms-marco-MiniLM-L6-v2-Q8_0.gguf \
    --safetensors models/ms-marco-MiniLM-L6-v2/model.safetensors \
    -o models/ms-marco-MiniLM-L6-v2-Q8_0-pooled.gguf
```

The output is the input byte for byte plus `cls.weight` / `cls.bias`
(F32, llama.cpp's names) and one provenance key,
`frink.rerank.pooler_source`. frink's `/v1/rerank` then reports
`frink_score_head: classifier(tanh(pooler(cls)))`; llama.cpp's
`build_pooling` RANK arm runs the pooler too.

The safetensors is matched by its classifier, not by name or metadata:
`cls.output.*` in the GGUF and `classifier.*` in the safetensors must
agree element-wise to within 1/128 of the tensor's largest magnitude,
or nothing is written and the refusal names the element and both
values. Also refused: a GGUF that is not `bert`, one that already
carries `cls.weight`, one with no `cls.output`, a split file (merge it
first), and a pooler of another width. The written file is reopened
through the rerank head loader before the command returns.

## Hugging Face Hub (`download`, `pull`)

Fetches a GGUF over HTTPS with no Python. The token and mirror
variables are in [CONFIG.md](CONFIG.md#hugging-face).

### `-hf`, llama.cpp's one-command form

`-hf user/repo[:QUANT]` fetches on first use and serves or runs
straight away:

```bash
frink serve -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
frink -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M -p "Hi" -n 64
```

The tag after the colon is a quant label, not a git revision, matched
without regard to case. A tag the repo does not publish is refused with
the list it does publish. `--hf-repo` / `-hfr` is the same flag;
`-hff` / `--hf-file` names an exact file.

Downloads land in the frink cache under `hub/<owner>__<repo>/` (root:
[`FRINK_CACHE`](CONFIG.md#server)), not in `./models`. A second run says
`using cached`.

### `download` and `pull`

`frink download` takes `hf download`'s arguments, and also the
`repo:QUANT` shape. The file goes to `--local-dir` (default `models/`):

```bash
frink download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models
```

`pull` is the older spelling and prints the local path. It stores under
`~/.cache/frink/hf/<org--model>/`, or `--dir DIR`:

```bash
frink pull org/model --file '*.gguf'
frink -m org/model      # downloads when the path is missing
```

Downloads resume by byte range into `<name>.partial`, renamed only when
complete; a server that ignores the range gets the file rewritten, not
appended. A file already present is left alone. A pattern matching more
than one file is refused.

## Interactive chat (`chat`)

Multi-turn REPL against a running `frink-server`:

```bash
FRINK_MODEL_PATH=model.gguf FRINK_ADDR=127.0.0.1:8383 ./target/release/frink-server
./target/release/frink chat --url http://127.0.0.1:8383 --system "Be brief."
# Commands: /quit  /clear
```

| Flag | Notes |
|---|---|
| `--url` | Server base URL (default `http://127.0.0.1:8383`) |
| `--model` | The request's `model` field (default `default`) |
| `--system` | Optional system message |
| `--max-tokens` / `--temperature` / `--top-p` | Sampling (defaults 256, 0.7, 0.95) |
| `--stream` / `--no-stream` | The reply streams over SSE by default; `--no-stream` waits for the full JSON |

## Server

`frink serve` (built in by default, the `serve` feature) and the
standalone `frink-server` are the same server and parse identical
arguments.

### LoRA adapters

`--lora adapter.gguf` and `--lora-scaled adapter.gguf:0.5` load the
file llama.cpp's `convert_lora_to_gguf.py` writes and apply it as
`build_lora_mm` does: every projection the adapter names computes
`W x + scale * alpha / rank * B (A x)`, `token_embd` and `output`
included, and two adapters on one weight are two terms of the sum.
Checked against libllama with the same adapter.

Refused by name: an adapter for another architecture, one naming a
tensor the base does not carry or of a shape it does not fit, a
routed-expert (`*_exps`) target, an activated LoRA
(`adapter.alora.invocation_tokens`), the embedding pair on a model
whose output head is tied to its embedding, and the flag on the MLA,
Gemma-4, GLM-5.2 and Kimi engines.

On Metal an adapted model runs on the slower per-matrix path, with the
same tokens as CPU. CUDA and CPU serve the adapter on every path.

### Server flags

`llama-server` command lines run unchanged where the flag is accepted
(differences in the table at the top). Accepted with llama.cpp's
meaning: `-m` / `--model` (a GGUF or a Kimi directory), `-hf` / `-hfr`
/ `--hf-repo`, `-hff` / `--hf-file`, `-c` / `--ctx-size` (sets
`FRINK_CB_MAX_CONTEXT`), `--api-key`, `--api-key-file`, `-a` /
`--alias`, `-ctk` / `--ctk` / `--cache-type-k`, `--host`, `--port`,
`-t` / `--threads`, `-dev` / `--device`, `--list-devices`, `-ngl` /
`--n-gpu-layers`, `-cb` / `--cont-batching`, `-nocb` /
`--no-cont-batching`, `-np` / `--parallel`, `-b` / `--batch-size`,
`-ub` / `--ubatch-size`, `--slot-save-path DIR`, `--lora`,
`--lora-scaled`, `--lora-init-without-apply`, `-md` / `--model-draft` /
`--spec-draft-model`, `--draft-max` / `--draft` / `--draft-n`,
`--draft-p-min` / `--spec-draft-p-min`, `--reasoning-budget N`,
`--prefill-assistant` / `--no-prefill-assistant`, `--jinja`,
`--chat-template`, `--chat-template-file`, `--chat-template-kwargs`,
`-rea` / `--reasoning`, `--reasoning-effort`,
`--reasoning-preserve` / `--no-reasoning-preserve`,
`--no-warmup` (there is no warm-up pass), `-fa` / `--flash-attn`,
`-hft` / `--hf-token` (sets `HF_TOKEN` before `-hf` downloads),
`--offline` / `LLAMA_ARG_OFFLINE` (`-hf` resolves from the cache only),
`--ssl-key-file` / `--ssl-cert-file` (set `FRINK_TLS_KEY` /
`FRINK_TLS_CERT`; one without the other is refused), `--cors-origins`
(sets `FRINK_CORS_ORIGINS`), `--threads-http N` (sets
`FRINK_TOKIO_WORKERS`; below 1 keeps the default, as upstream),
`--metrics`, `--slots` / `--no-slots` (sets `FRINK_ENDPOINT_SLOTS`;
`GET /slots` is on by default, as upstream), `--spm-infill` (sets
`FRINK_SPM_INFILL`; `/infill` puts the suffix first), `--usage` and
`--version`.

frink's own behaviour:

- `--port 0` asks the kernel for a free port and announces it on stdout
  ([Running under a supervisor](#running-under-a-supervisor)).
- `--api-key` also gates `/admin`. An empty `--api-key-file` is refused.
- `--slot-save-path` is refused at startup when it is not a directory.
  Slots restore into the prefix cache, so `FRINK_PREFIX_CACHE_ENTRIES`
  must be set too.
- `--lora` sets `FRINK_LORA`; every model load, `/admin/models/load`
  included, attaches the same adapters or refuses the checkpoint by
  name ([`API.md`](API.md#lora-adapters)).
- `-cb`/`-np` read back as `frink_scheduler_max_seqs`, and `-b`/`-ub` as
  `frink_scheduler_prefill_chunk`, on `GET /metrics`. `-np N` is also
  the N slots `GET /slots` lists and `/props` reports as `total_slots`;
  without it `GET /slots` says there is no slot count.
- `-b` also sets `FRINK_LOGICAL_BATCH`, the `n_batch` `/infill` budgets
  its prompt with (default 2048, llama.cpp's).
- `--chat-template` and `--chat-template-file` set
  `FRINK_CHAT_TEMPLATE`, used on every load path. The template is
  checked at startup by rendering one user turn, as llama.cpp does.
- `--chat-template-kwargs`, `-rea`, `--reasoning-effort` and
  `--reasoning-preserve` set `FRINK_CHAT_TEMPLATE_KWARGS`. A request's
  `chat_template_kwargs` and its top-level `reasoning_effort` or
  `thinking` win over them; `reasoning_effort: "none"` clears a default
  effort.
- `preserve_reasoning` is on unless `--no-reasoning-preserve` or a
  kwarg turns it off, as in llama.cpp. Templates read it as
  `preserve_thinking`, `clear_thinking`, `truncate_history_thinking` or
  `drop_thinking` (GLM-4.7, LFM2.5, LFM3, Nemotron-3, Laguna,
  DeepSeek-V4).
- `--reasoning-budget` sets `FRINK_REASONING_BUDGET`, the default for a
  request's `reasoning_budget_tokens`.
- Without `-md` the server drafts by prompt lookup where verification
  can reproduce the sampling; `usage` reports what it bought
  ([`API.md`](API.md#usage-timings)). `-md` (sets `FRINK_MODEL_DRAFT`)
  loads one draft model (autoregressive, DFlash or DFlash2) for every
  request, on both decode paths and also for models swapped in through
  `/admin/models/load`. Grammar and `json_object` requests do not
  speculate. Refused at startup: a dedicated-engine model, a recurrent
  draft, a drafter of another vocabulary size, an autoregressive
  drafter whose KV lands on the device (`--device cpu`), and a DFlash
  draft beside `FRINK_PREFIX_CACHE_ENTRIES` or paged KV. `--draft-max`
  (default 3) and `--draft-p-min` (default 0) are llama.cpp's.

**Environment.** llama-server's own variables (`LLAMA_ARG_MODEL`,
`LLAMA_ARG_CTX_SIZE`, `LLAMA_ARG_N_GPU_LAYERS`, `LLAMA_API_KEY`, ...)
are read as the flags they are declared on
(`frink_server::llama_env::LLAMA_ENV_FLAGS`): a set variable becomes
its flag unless the command line gave that flag. Switches take `on/off`,
`true/false`, `1/0`, `enabled/disabled`, and `LLAMA_ARG_NO_*` means
off. frink's `FRINK_*` variables ([CONFIG.md](CONFIG.md)) still work; a
flag or its `LLAMA_ARG_*` variable wins over them. Keep secrets such as
`FRINK_API_KEY` in the environment, not on the command line.

```bash
cargo build --release -p frink-cli --features "serve metal"

./target/release/frink serve \
  -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  --host 127.0.0.1 --port 8383 -dev metal -ngl all -cb -np 4
```

Continuous batching is off on every backend unless `-cb` /
`--cont-batching` or `-np` / `--parallel N` asks for it; `-nocb` keeps
it off even with `-np`. Without it requests run one at a time on the
private decode loop (on Metal, attention on the device); the batched
decode attends on the host, slower for one client at long context but
serving many at once. `-cb` beside
`FRINK_PREFIX_CACHE_ENTRIES` or the KV pool (without paged KV) is
refused at startup ([CONFIG.md](CONFIG.md#tuning)).

Built without the `serve` feature, `frink serve` exists and says so.

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
{"event":"frink.server.ready","addr":"127.0.0.1:52091","port":52091,"scheme":"http","pid":4242,"version":"0.63.0"}
```

Read stdout line by line and ignore anything that is not the ready
event; the tracing subscriber shares the stream.

`--exit-on-stdin-close` (or `FRINK_EXIT_ON_STDIN_CLOSE=1`) exits when
stdin reaches EOF, which works the same on macOS, Windows and Linux. It
is opt-in, because a server started with stdin from `/dev/null`
(systemd, cron, `nohup`) sees EOF at once.

The web UI is a separate app: see [`ui/`](../ui) and
[`crates/frink-server/README.md`](../crates/frink-server/README.md).
`GET /` on this server is a 404.

## Benchmark (`frink bench`)

With `-m`, `bench` works like [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench):
same workload names (`pp<N>` batched prefill, `tg<N>` decode), mean ±
sample stddev over `-r` reps (llama-bench's `avg_ts` and `stdev_ts`),
one warmup discarded, same flag names. (Receipts before schema 4 hold
the median and population stddev; each keeps its raw `frink_samples`.)

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
| `-m/--model` | GGUF to benchmark. Without it, `bench` runs the synthetic matvec microbenchmark |
| `-p/--n-prompt` | Prefill tokens (default 512, `0` skips the `pp` row) |
| `-n/--n-gen` | Decode steps (default 128, `0` skips the `tg` row) |
| `-b/--batch-size`, `-ub/--ubatch-size` | Submit the prompt in pieces of the smaller of the two (either alone stands for both), as llama-bench does. Neither given is the whole prompt in one pass. One value each per run |
| `-pg PP,TG` | Also run one combined test, named `pp512+tg128` as llama-bench names it. One pair per run |
| `-d/--n-depth` | Context tokens put in the KV, untimed, before every repetition; rows are named `pp512 @ d4096`. One value per run |
| `-r/--repetitions` | Timed reps (default 3), plus one discarded warmup |
| `-t/--threads` | CPU threads (`0` = performance-core default) |
| `-ngl/--n-gpu-layers` | `0` (the default) forces CPU; the layer count or more offloads every layer; a partial count is refused |
| `-c/--ctx-size` | Context size (`0`, the default, is the GGUF's own) |
| `-fa/--flash-attn` | `on` or `auto` (the default); `off` and sweeps are refused |
| `--list-devices` | Print the devices this build can run on and exit; with the `vulkan` feature, one `Vulkan<i>` row per visible device (frink uses `Vulkan0`), or why there is none |
| `--no-warmup` | Time every repetition, with no discarded warmup. Refused with `--receipt` |
| `--delay` | Seconds to sleep before each test |
| `--prio` | Process priority: `-1` low, `0` normal, `1` medium, `2` high, `3` realtime (`nice` 5, 0, -5, -10, -20, as llama.cpp sets it). A priority the OS refuses is an error |
| `--progress` | Print each test's progress to stderr, in llama-bench's words |
| `-hf/--hf-repo`, `-hff/--hf-file`, `-hft/--hf-token` | Benchmark a Hugging Face repo (`user/repo[:QUANT]`), fetched into the frink cache first |
| `--offline` | Resolve `-hf` from the frink cache alone, no network |
| `-ctk`, `-ctv` | KV cache type; only types the active backend's store holds, and `-ctv` equal to `-ctk` |
| `-o/--output` | `md` (the default), `csv`, `json`, `jsonl` or `sql` on stdout, in llama-bench's field names. Under anything but `md`, every other line goes to stderr |
| `-oe/--output-err` | The same formats (or `none`, the default) on stderr |
| `--compare` | Also run `llama-bench -o json` on the same GGUF, print the gap, and record the llama.cpp build it reports |
| `--suite` | Run every [`benchmarks/suite.json`](../benchmarks/suite.json) entry in its own process, write a timing file per run, re-render [`RESULTS.md`](../benchmarks/RESULTS.md) |
| `--render` | Re-render the RESULTS table from the timing files on disk, measuring nothing |
| `--id` / `--backend` | Restrict `--suite` to one entry / backend |
| `--fit-host` / `--skip-missing` | Skip entries too large for the host / with no GGUF present |
| `--max-load` | The 1-minute load average a timed run needs to be under (default `2.0`, raw, not per core). `0` waives this and the thermal and free-memory checks. `--suite` checks once up front, forwards the bar to every child, and waits between entries for the load to decay |
| `--bench-dir` | Where `suite.json`, `RESULTS.md` and `receipts/` live (default `benchmarks`) |
| `--receipt` | Write a single run's raw timings to this path |
| `--suite-id` / `--backend-label` | Internal: what a `--suite` child records in its receipt |
| `--hidden` / `--ffn-dim` | Synthetic microbenchmark only (no `-m`): the weight matrix is `[ffn-dim x hidden]`. Defaults 4096 and 14336 |
| `--iters` | Synthetic microbenchmark only: timed iterations per variant (default 20) |

With `--compare --receipt`, the receipt records the llama.cpp build
under `llama_cpp` (`build_commit`, `build_number`, `backends`,
`cpu_info`, `gpu_info`, the command); one missing its commit or
backends is refused at write time. `--render` gives each (machine,
llama.cpp build) pair its own section; older receipts render under
**unknown**.

A run stops before the timer starts on a busy, hot or memory-short host
([`benchmarks/README.md`](../benchmarks/README.md) lists the checks).
Suite entries in `benchmarks/suite.json` carry `id`, `name`, `gguf`,
`backends`, `estimated_ram_gb`.

## Batched benchmark (`frink batched-bench`)

Throughput as a function of batch size, like
[`llama-batched-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/batched-bench):
one row per combination of prompt length (`-npp`), generation length
(`-ntg`) and parallel sequences (`-npl`). Same ten columns and widths,
so the tables paste side by side; `--output-format jsonl` prints one
object per row with upstream's keys.

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

`PP`/`TG` are per sequence, `B` is the sequence count, `N_KV = B*(PP+TG)`,
`S_PP` is `B*PP/T_PP` (or `PP/T_PP` with `-pps`), `S_TG` is
`B*TG/T_TG`, `S` is all tokens over `T_PP + T_TG`, as upstream.

It drives the engine calls `frink-server` makes under
`FRINK_CONTINUOUS_BATCHING=1`, without the server. Differences from
upstream: `B` prompts are `B` prefill calls, and the batched decode
attends on the host on every backend, so `-ngl` offloads the
projections only.

| Flag | Meaning |
|---|---|
| `-m/--model` | GGUF to benchmark (generic decoder architectures only; the dedicated engines are refused by name) |
| `-npp`, `-ntg`, `-npl` | Comma-separated sweeps; all three required, every value > 0 (`0` is refused) |
| `-pps` | One prompt shared by every sequence: prefilled once, its KV copied to the others between the two timers |
| `-tgs` / `--tg-separate` | Decode each sequence to completion in turn instead of one step across all of them per call |
| `-c` | `n_kv_max`; a combination needing more is skipped, and the skip is printed. `0` = the GGUF's `{arch}.context_length` |
| `-ub` | Prompt tokens per forward call (default 512, upstream's `n_ubatch`) |
| `-t`, `-ngl` | As `frink bench` |
| `--output-format` | `md` (default) or `jsonl` |
| `--receipt`, `--backend-label` | Write a JSON receipt; the label must name the backend that ran or the receipt is refused, before the sweep and again at write time |
| `--max-load` | The same quiet-host bar as `frink bench`; `0` waives it and the thermal and free-memory checks. The free-memory check counts the largest row's KV on top of the weights |

Refused by name, accepted only so a llama.cpp command line fails with a
reason (and hidden from the short `-h`): `-b` (no logical batch
distinct from `-ub`), `-kvu` / `--kv-unified` (every sequence has its
own cache), `-fa` and `-tb` / `--threads-batch`. JSONL rows omit
`n_batch`, `flash_attn` and `n_threads_batch`.

Every row runs a discarded warmup pass and a timed pass, which must
produce the same greedy picks; `bench_guard`'s other checks (cold
caches, lengths re-read from the KV, finite rates) refuse a bad row.

## One model at a time

Every command that loads weights (`run`, `bench -m`, `batched-bench`,
`perplexity`, `quant-sensitivity`, `smoke`, `run-kimi`, the per-backend
children of `verify` and `layer-divergence`, and `frink-server`)
registers itself, and stops when another frink process already holds a
model, since two instances fight over the same cores;
the error names each running instance and points at
`--allow-multiple-instances` (or `FRINK_ALLOW_MULTIPLE_INSTANCES=1`).

Exempt: header-only commands (`inspect`, `inspect-plan`, `presets`,
`archs`, `caps`), the HTTP clients (`chat`, `serve-bench`), the
downloaders and `bench --suite` / `--render`. `--suite`, `verify` and
`layer-divergence` are supervisors whose children register on their
own.

The registry is a directory of one small file per live process
(`$FRINK_INSTANCE_DIR`, default `~/.cache/frink/instances`); a dead
process's entry is pruned by the next run. It is advisory, not a lock:
two processes starting in the same instant both stop.

See also: [`FEATURES.md`](FEATURES.md) · [`MODELS.md`](MODELS.md) · [`API.md`](API.md).
