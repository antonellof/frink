# Benchmarks

The published table is **[`RESULTS.md`](RESULTS.md)**, generated from
receipts; never edit it by hand. Measurements without a receipt and
before/after studies are in **[`HISTORY.md`](HISTORY.md)**.

Every timed run writes a JSON receipt of raw numbers to
[`receipts/engine/`](receipts/engine/) (kernel, rendered into
`RESULTS.md`) or [`receipts/serving/`](receipts/serving/) (HTTP,
checked in by hand), named `{id}_{backend}.json` or
`{id}_{backend}_{workload}_{version}.json`.

The setup follows llama.cpp's [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench):
`frink bench` runs `pp512` and `tg128` on synthetic tokens, compared
against `llama-bench` on the same GGUF, with no HTTP, chat template,
tokenizer or sampler in the loop. The suite is [`suite.json`](suite.json).

Host B: Apple M2 Pro, 6 performance and 4 efficiency cores, 32 GiB
unified memory.

## Run

```bash
cargo build -p frink-cli --release --features metal

# one model, with the llama.cpp comparison run for you
./target/release/frink bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -p 512 -n 128 -r 3 --compare

# every suite.json entry × backends, then re-render RESULTS.md
./target/release/frink bench --suite --fit-host --skip-missing

# one suite id (or one backend across the suite)
./target/release/frink bench --suite --id llama32_3b_q4km --backend metal
./target/release/frink bench --suite --backend cpu --fit-host --skip-missing

# re-render RESULTS.md from existing receipts, measuring nothing
./target/release/frink bench --render
```

`--suite` runs each entry in a fresh child process, because backend
selection reads process-global environment and the rayon pool is built
once.

`--fit-host` skips an entry when its `estimated_ram_gb` exceeds ~75% of
physical RAM (macOS only) or the host lacks `estimated_ram_gb + 2 GiB`
free. `--skip-missing` skips GGUFs absent from `models/`. A failed run
keeps the previous receipt. `--render` rewrites the engine table between
HTML markers and leaves the Open notes alone.

Each engine receipt (schema 3) names the llama.cpp build behind its
`llama_tps` in a `llama_cpp` field from `llama-bench -o json`
(`build_commit`, `build_number`, `backends`, `cpu_info`, `gpu_info`,
the command). A receipt with a llama.cpp number and no commit or
backends is refused at write time. Rows are grouped by machine and
llama.cpp build, and two builds never share a table. Schema-2 receipts
render with the build marked **unknown**.

## Batched throughput (`frink batched-bench`)

`frink batched-bench` is `llama-batched-bench`: throughput against the
number of parallel sequences through the continuous batcher's engine
seams (`forward_batch_last_host_kv`, `forward_multi_seq`), no HTTP, with
upstream's ten columns.

```bash
./target/release/frink batched-bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -c 2048 -npp 128,512 -ntg 128 -npl 1,2,4,8
llama-batched-bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -c 2048 -npp 128,512 -ntg 128 -npl 1,2,4,8
```

Same contract as `frink bench`: the host checks below (memory counts the
largest row's KV), one discarded warmup per row that must match the
timed pass, and a `--receipt` refused when `--backend-label` is not the
backend that ran. `-b`, `-kvu`, `-fa` and `-tb` are refused by name
([`docs/CLI.md`](../docs/CLI.md)). No batched rows are published yet.

## When a run stops instead of printing a number

Four checks run before the timer starts; `--max-load 0` turns all of
them off, for numbers nobody publishes.

| Check | Rule |
|---|---|
| Host busy | The 1-minute load average must be below `--max-load` (default `2.0`, raw). Rows read 25-45% low under load |
| Thermal | On macOS, thermal state `serious` or `critical`, or an Intel `CPU_Speed_Limit` under 100%, stops the run; `fair` is recorded. Nothing is read on Linux or Windows. Heating mid-run prints a warning |
| Free memory | GGUF size plus 2 GiB must be free (`vm_stat` free + inactive on macOS, `MemAvailable` on Linux). In `--suite` the entry is skipped and its old receipt stands |
| Previous entry | `--suite` waits silently, polling every 5 s for up to 3 minutes, for the load average to drop; past that it skips the entry |

The load average cannot see one pegged core (on 2026-09-04 `suggestd`
held ~97% of one core for a day under the bar). Before a run that
matters, check `ps -eo pcpu,comm | sort -rn | head` and treat any
process above ~90% as disqualifying.

Checks with no override: exactly one discarded warmup, caches cold
before the prompt, identical workload digest and greedy output across
repetitions, and no `inf`, `NaN` or non-positive rate. The KV position
count that catches skipped steps runs only where the host cache is the
record; under GPU offload the receipt says it did not run.

### Adding a model

Append an object to `models[]` in [`suite.json`](suite.json):

```json
{
  "id": "my_model_q4km",
  "name": "My-Model Q4_K_M",
  "gguf": "models/My-Model-Q4_K_M.gguf",
  "backends": ["metal", "cpu"],
  "estimated_ram_gb": 4,
  "expect": "ok",
  "notes": "optional"
}
```

Put the GGUF under `models/` (path is repo-relative), then:

```bash
./target/release/frink bench --suite --id my_model_q4km --fit-host --skip-missing
```

### Thread counts are not forced, on purpose

llama.cpp defaults to `hw.perflevel0.physicalcpu` (6 on Host B) and
degrades above it: `llama-bench` on SmolLM2-135M Q8_0 measures 346
tok/s at `-t 4` and 176 at `-t 10`. This suite once forced `-t 10` on
both, handicapping llama.cpp 2-4x; **CPU comparisons taken before that
fix are not usable.**

### Run-to-run variance is ±20%

The same binary spreads ±20% on Host B (SmolLM2 `pp512`: 205, 284,
306). For a change under ~20%, interleave the two binaries in one
session and count rounds won:

```bash
for round in 1 2 3 4; do
  for bin in ./frink-base ./frink-new; do
    $bin bench -m model.gguf -p 512 -n 0 -r 3
  done
done
```

## Gap convention

`Gap` = `llama / frink`.

| Gap | Meaning |
|---|---|
| &lt; 1.0 | frink faster |
| ~1.00× | parity (within ~5%) |
| &gt; 1.0 | frink slower |

Prose elsewhere ("1.56× faster") states the inverse when frink wins.
Never quote a ratio that has no matching file under
[`receipts/engine/`](receipts/engine/).

## Env (frink)

| Var | Typical |
|---|---|
| `FRINK_METAL` | `1` (Metal), `0` for CPU runs |
| `FRINK_METAL_ATTN` | `1` |
| `FRINK_METAL_FA_VEC` | default **on** for `head_dim` in {64,96,128,256}, `0` = legacy GQA |
| `FRINK_CTK` | `f16` unless the row says otherwise |
| `FRINK_CPU_INT_DOT` | **on by default** in both binaries, `0` opts out |
| `FRINK_CPU_THREADS` | unset = performance cores (6 on Host B) |
| `FRINK_CUDA_GQA` / `FRINK_CUDA_GRAPH` | CUDA path (not Host B) |

Full list: [`docs/CONFIG.md`](../docs/CONFIG.md).

## CUDA

Host B is Metal/CPU only. On a CUDA host:

```bash
cargo build -p frink-cli --release --features cuda
./target/release/frink bench -m model.gguf --n-gpu-layers 99 --compare
./target/release/frink bench --suite --fit-host --skip-missing --backend cuda
```

Record the GPU and driver with any CUDA number. CUDA receipts are the
`*_cuda__*.json` files under [`receipts/engine/`](receipts/engine/); the
RTX 3090 and RTX 3060 rows predate the resident stack and are marked
stale.

The current rows ran on 2026-11-09 on a rented NVIDIA L40S (48 GB,
driver 580, CUDA 13.0, 8 vCPU of an AMD EPYC 7R13, via Hugging Face
Jobs) against llama.cpp at the pin built for `sm_89`. Each timed run was
preceded by `frink verify` (24 greedy tokens) and `frink parity`; every
model matched, gpt-oss-20b at the DRIFT its CPU path also shows.
[`tools/hf_jobs/cuda_bench.sh`](../tools/hf_jobs/cuda_bench.sh) is the
whole job:

```bash
hf jobs run --flavor l40sx1 --timeout 3h --secrets HF_TOKEN \
  -e MODELS="llama31_8b_q4km|bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c "$(cat tools/hf_jobs/cuda_bench.sh)"
```

## Serving (HTTP)

`frink serve-bench` measures a running server end to end over HTTP
(chat template, tokenizer, sampler, SSE), with receipts in
[`receipts/serving/`](receipts/serving/). There is no llama.cpp HTTP
twin in-tree to compare against yet.

```bash
# install script binary (Metal build on macOS)
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -dev metal -ngl all &

./target/release/frink serve-bench --requests 64 --concurrency 8 --output-len 128
./target/release/frink serve-bench --concurrency 16 --json
```

Temperature 0, exact output length and the positional TTFT/TPOT split
are enforced by `bench_client`; the rules are in
[`docs/CLI.md`](../docs/CLI.md).

### Reproducing a concurrency sweep

Current serving numbers are in
[`RESULTS.md`](RESULTS.md#serving-features-measured); the 0.15.3
receipts are in [`HISTORY.md`](HISTORY.md#metal-continuous-batching-0153).
The same shapes:

```bash
# parallel stream burst, 64 output tokens, two requests per client
for c in 1 2 4 8; do
  ./target/release/frink serve-bench --concurrency $c --requests $((2 * c)) --output-len 64
done

# sequential stream, 128 output tokens
./target/release/frink serve-bench --concurrency 1 --requests 8 --output-len 128
```

`--url` points it at a server other than `http://127.0.0.1:8383`, and
`--json` writes the report as JSON for a receipt.
