# Hugging Face Jobs scripts

Each script runs on one rented GPU box from Hugging Face Jobs and uploads its
logs and receipts to a results dataset (`RESULTS_REPO`, by default
`antonellof/frink-cuda-runs`, under `runs/<RUN_ID>/`). The job reads
`HF_TOKEN` as a secret. Locally, the helpers read it from the repository's
`.env` (or `FRINK_ENV`) and never print it.

| script | what it does |
|---|---|
| `cuda_validate.sh` | Release validation on a tag: hardware tests, `frink verify`, `frink parity` against llama.cpp at the pin (two dumpers), the timed bench with `--compare`, and an nsys profile. Its receipts are the ones `benchmarks/RESULTS.md` publishes. `MAX_LOAD` (default 2.0) is the bench's load gate. |
| `cuda_bench.sh` | The bench job behind the 0.58.0 rows of `benchmarks/RESULTS.md`. |
| `lab.sh` | The kernel lab, which picks stages from `STAGES`. Its numbers are lab numbers, not receipts. |
| `launch_release.sh <tag>` | Starts `cuda_validate.sh` on an L40S and an RTX PRO 6000. |
| `watch.sh <job> <run>` | Prints a run's new summary lines until the job ends. |
| `fetch.sh <run>` | Downloads a run and prints its receipts as a table. |
| `models10.txt` | The ten models the validation runs (`id\|repo\|file;...`). |

`lab.sh`'s stages, as `STAGES=a,b,...`:

- `tests`: the frink-cuda hardware suite (`--include-ignored`).
- `gemm`: `mul_mm_bench`.
- `mvsweep`: the decode matvec sweep.
- `mmqab`: the int8 GEMM against the f16 tensor-core one, for each of `MMQ_KINDS`.
- `gemmab`: the int8 GEMM with `AB_VAR` set to each of `AB_VALS`.
- `gemmrev`: `mul_mm_bench` built at each of `GEMM_REVS` and run on the same box. Cross-box A/Bs mislead by 10 to 20% on A10Gs.
- `cmds`: each of `CMDS`, separated by `;`.
- Per model in `MODELS`:
  - `verify`;
  - `ab`: a verify and a bench with `AB_VAR` set to each of `AB_VALS`;
  - `mmqmodel`;
  - `time`: `FRINK_CUDA_TIME=1`;
  - `bench`;
  - `nsys`.
- `ncu` fails on Hugging Face Jobs: its containers block GPU performance counters (`ERR_NVGPUCTRPERM`).

Example: the attention A/B behind #568.

```bash
set -a; . ./.env; set +a
hf jobs run --flavor a100-large --timeout 3h -d --secrets HF_TOKEN \
  -e RESULTS_REPO=antonellof/frink-cuda-runs -e RUN_ID=fatc-a100 \
  -e BRANCH=perf/cuda-prefill-attn-tc -e STAGES=tests,ab \
  -e AB_VAR=FRINK_CUDA_PREFILL_ATTN -e "AB_VALS=simt tc" \
  -e "MODELS=llama31_8b_q4km|bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c "$(cat tools/hf_jobs/lab.sh)"
tools/hf_jobs/watch.sh <job-id> fatc-a100
```

Jobs need prepaid credit. When it runs out, Hugging Face cancels every running job with no message, and new ones are refused with "Pre-paid credit balance is insufficient".
