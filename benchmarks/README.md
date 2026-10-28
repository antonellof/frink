# Benchmarks

The published table is **[`RESULTS.md`](RESULTS.md)**. It is generated
and holds nothing else, so never edit it by hand.

Everything a generator cannot produce — measurements taken without a
receipt, before/after studies against frink itself, and the sections
that predate the code they describe — is in
**[`HISTORY.md`](HISTORY.md)**.

Every timed run writes one small JSON file of raw numbers into
[`receipts/engine/`](receipts/engine/) (kernel) or
[`receipts/serving/`](receipts/serving/) (HTTP), named
`{id}_{backend}.json` or `{id}_{backend}_{workload}_{version}.json`.
Engine receipts are checked in and `RESULTS.md` is rendered from them.
Serving receipts are checked in by hand. The rest of this repo calls
them receipts.

The setup follows llama.cpp's [`llama-bench`](https://github.com/ggerganov/llama.cpp/tree/master/tools/llama-bench):
a native CLI, the `pp512` and `tg128` workloads, compared against
`llama-bench` on the same GGUF. No HTTP, no chat template, no tokenizer
and no sampler in the loop.

| | |
|---|---|
| Measures | kernels alone |
| Driver | `frink bench` (Rust) |
| Compared against | `llama-bench` |
| Raw numbers | [`receipts/engine/`](receipts/engine/) |
| Workload | `pp512` / `tg128`, synthetic tokens |

Host B: Apple M2 Pro, 6 performance and 4 efficiency cores, 32 GiB
unified memory.

Suite definition: [`suite.json`](suite.json).

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

`--suite` reads [`suite.json`](suite.json) and runs each entry in a
**fresh child process**. That is required rather than tidy. Backend
selection reads process-global environment, and the rayon pool is built
once, so benchmarking several backends inside one process would measure
the first one's configuration for all of them and say nothing about
it.

`--fit-host` skips an entry two ways: when its `estimated_ram_gb`
exceeds ~75% of physical RAM (macOS only, since that is where the total
is read), and when the host does not have `estimated_ram_gb + 2 GiB`
free right now. `--skip-missing` skips GGUFs absent from `models/`.

Each run drops its numbers in [`receipts/engine/`](receipts/engine/)
as `{id}_{backend}.json`. A run that fails leaves the previous receipt
in place instead of overwriting it. `--render` rewrites the engine table
in `RESULTS.md` between HTML markers and leaves the Open notes alone.

## Batched throughput (`frink batched-bench`)

`frink bench` is one sequence. `frink batched-bench` is
`llama-batched-bench`: throughput against the number of parallel
sequences, through the continuous batcher's engine seams
(`forward_batch_last_host_kv` per prompt, `forward_multi_seq` per
decode step) with no HTTP. Same ten columns as upstream, so a frink
table and a llama.cpp table paste side by side.

```bash
./target/release/frink batched-bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -c 2048 -npp 128,512 -ntg 128 -npl 1,2,4,8
llama-batched-bench -m models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf \
  -c 2048 -npp 128,512 -ntg 128 -npl 1,2,4,8
```

It runs under the same contract as `frink bench`: the quiet-host,
thermal and free-memory bars below (the memory bar counts the largest
row's KV on top of the weights), one discarded warmup per row that
must agree with the timed pass on both input and output, and a receipt
(`--receipt`) that is refused when its `--backend-label` is not the
backend that ran. `-b`, `-kvu`, `-fa` and `-tb` are refused by name;
`docs/CLI.md` has the flag table. No batched rows are published in
`RESULTS.md` yet.

## When a run stops instead of printing a number

Four checks run before the timer starts. Each one exists because the
number you would otherwise get looks real and measures the wrong thing.
All four are turned off together by `--max-load 0`, which is the switch
for "measure anyway, and do not publish this".

**The host is busy.** The 1-minute load average has to be *below*
`--max-load` (default `2.0`, raw, not divided by core count). Known-good
rows read 25-45% low under load. Message:

```
host 1-minute load average is 3.10, above the 2.00 bar: a timed run
here is noise, not a measurement …
```

**A load average cannot see one busy core.** The guard above is
necessary and not sufficient. Every CPU measurement taken on 2026-09-04
ran while `suggestd` held ~97% of one core, for over a day, and the
1-minute load stayed under the 2.0 bar the whole time: one pegged core
on a six-core box does not move the average enough to trip it. The
comparison being run that day was thread-scheduling sensitive, which is
exactly the kind a stolen core distorts unevenly. So before a run that
matters, check `ps -eo pcpu,comm | sort -rn | head` as well, and treat
a single process above ~90% as disqualifying even when the guard passes.

**The host is thermally limited.** On macOS, `NSProcessInfo`'s thermal
state at `serious` or `critical`, or an Intel Mac reporting a
`CPU_Speed_Limit` under 100%, stops the run: the OS is cutting sustained
performance, so the timing describes the cooling system. `fair` is
recorded and does not stop anything, or a laptop would never run one.
Nothing is read on Linux or Windows, because a temperature in
`/sys/class/thermal` is not a pressure level and inventing a mapping
would manufacture exactly the false precision this check exists to
avoid. A host that starts cool and heats up mid-run finishes and prints
a warning that the later repetitions did not run under the same
conditions as the first.

**The model would not fit in free memory.** The GGUF's size on disk is
the floor on its footprint, so a single-model run needs that plus 2 GiB
free (`vm_stat` free + inactive pages on macOS, `MemAvailable` on
Linux). Without this, a model that pages to disk still finishes and
still reports a real-looking number for work the disk did. In `--suite`
the same arithmetic skips the entry rather than stopping the whole run,
which keeps a previous receipt that is stale and says so instead of
replacing it with a paged one that does not.

**The previous entry is still in the load average.** `--suite` waits
between entries for the 1-minute average to fall back below the bar,
polling every 5 seconds for up to 3 minutes. It prints nothing while it
waits, so a quiet pause between two entry headers is this and not a
hang. Without the wait the suite locked itself out with its own
benchmark: a 21-entry run wrote 2 receipts. Past 3 minutes it skips that
entry and moves on, the same way a missing GGUF is skipped, and the
entry's previous receipt stands.

A separate set of checks has no override at all, because failing one
means the two repetitions were not the same experiment: exactly one
discarded warmup, caches cold before the prompt, identical workload
digest across repetitions, identical greedy output across repetitions,
and no `inf`, `NaN` or non-positive rate. One of them counts KV
positions after a decode run to catch skipped steps, and it runs only
where the host cache is the record. Under GPU offload the device holds
the KV and the host struct stays empty, so the receipt records that the
check did not run rather than that it passed.

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

Neither engine is pinned to a thread count. That is a correction, not
an oversight. llama.cpp defaults to `hw.perflevel0.physicalcpu` (6 on
this host) and degrades sharply above it, because it splits each graph
node into equal static slices and the 4 efficiency cores then stall
every barrier. On Host B, `llama-bench` on SmolLM2-135M Q8_0 measures
346 tok/s at `-t 4` and 176 at `-t 10`.

This suite used to force `-t 10` on both engines, which handicapped
llama.cpp by 2–4× and flattered frink. **CPU comparisons taken before
that fix are not usable.** Letting each engine choose its own default is
the comparison that means something.

### Run-to-run variance is ±20%

Sequential runs of the *same* binary spread ±20% on this host.
SmolLM2 `pp512` measured 205, 284 and 306 on three consecutive runs.
One before/after pair will not resolve a change smaller than that.

For anything under ~20%, **interleave the two binaries** round by round
in one session and count rounds won, instead of comparing two batches:

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

Record the GPU and driver whenever you quote a CUDA number. CUDA rows
come from rented hosts (an RTX 3090 and an RTX 3060 so far); their
receipts are the `*_cuda__*.json` files under
[`receipts/engine/`](receipts/engine/), and `RESULTS.md` renders them
beside the Metal and CPU rows.

## Serving (HTTP)

Engine benches measure kernels alone. Serving benches measure a running
`frink-server` over the OpenAI-compatible HTTP API: chat template,
tokenizer, sampler, SSE streaming, and (on Metal by default) continuous
batching.

| | |
|---|---|
| Measures | end-to-end HTTP latency and throughput |
| Driver | `frink serve-bench` |
| Compared against | — (no llama.cpp HTTP twin in-tree) |
| Raw numbers | [`receipts/serving/`](receipts/serving/) |
| Workload | streaming chat/completions, concurrency sweeps |

```bash
# install script binary (Metal build on macOS)
frink serve -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf -dev metal -ngl all &

./target/release/frink serve-bench --requests 64 --concurrency 8 --output-len 128
./target/release/frink serve-bench --concurrency 16 --json
```

Rules for meaningful numbers: see [`docs/CLI.md`](../docs/CLI.md)
(`serve-bench` section). Temperature 0, exact output length, and
positional TTFT/TPOT split are enforced in `frink-cli`'s `bench_client`
module, which `serve-bench` drives.

### Metal continuous batching (0.15.3)

Host B, **Llama-3.2-3B-Instruct Q4_K_M**, CB auto-on, `frink 0.15.3`:

| Workload | Concurrency | OK | Aggregate tok/s | Mean TTFT |
|---|---|---|---|---|
| parallel stream burst (`max_tokens=64`) | 1 | 2/2 | 17.4 | 157 ms |
| | 2 | 4/4 | 19.0 | 272 ms |
| | 4 | 8/8 | 22.7 | 450 ms |
| | 8 | 16/16 | **24.4** | 957 ms |
| sequential stream (`max_tokens=128`) | 1 | 8/8 | — | **118 ms** |

These numbers were recorded with frink 0.15.3 by an external harness
that is not part of this repository. Receipts:
[`llama32_3b_q4km_metal_cb_parallel_0.15.3.json`](receipts/serving/llama32_3b_q4km_metal_cb_parallel_0.15.3.json),
[`llama32_3b_q4km_metal_cb_stream_0.15.3.json`](receipts/serving/llama32_3b_q4km_metal_cb_stream_0.15.3.json).
To reproduce the same shapes with the in-repo client, start the server
as above and run:

```bash
# parallel stream burst, 64 output tokens, two requests per client
for c in 1 2 4 8; do
  ./target/release/frink serve-bench --concurrency $c --requests $((2 * c)) --output-len 64
done

# sequential stream, 128 output tokens
./target/release/frink serve-bench --concurrency 1 --requests 8 --output-len 128
```

`--url` points it at a server other than `http://127.0.0.1:8383`, and
`--json` writes the report as JSON for a receipt. The prompts differ
from the 0.15.3 harness's, so expect the shape of the results to
match rather than the exact figures. Current serving numbers are in
[`RESULTS.md`](RESULTS.md#serving-features-measured).

0.15.2 measured similar aggregate throughput at concurrency 8 but returned
garbled text under CB on Metal until the 0.15.3 host-K/V prefill fix.
See [`docs/plans/archive/metal-parallel-concurrency.md`](../docs/plans/archive/metal-parallel-concurrency.md).
