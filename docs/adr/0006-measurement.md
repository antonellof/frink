# 0006. Performance claims come from quiet, rented hardware

Status: Accepted
Date: 2026-10-28

## Context

In one week, wrong numbers reached `benchmarks/RESULTS.md` four ways: a
load guard that could not see one pegged core, a single GPU utilisation
sample taken after the run ended, a flag (`--n-gpu-layers 0`) that did
not change the backend, and 13 receipts labelled CPU that had run on
Metal. The development laptop runs a background process at ~97% of a
core.

## Decision

- CPU and CUDA numbers come from rented vast.ai hosts with
  `cpu_cores_effective == cpu_cores`. The M2 Pro is used for Metal
  only. Agents never benchmark; the parent session does.
- Before a run: `ps -eo pcpu,comm | sort -rn | head`; any process above
  ~90% disqualifies the host, whatever the load average says.
- Trust the receipt, not the flag: `backend_active` is what ran. A
  receipt whose label disagrees is refused at write time.
- A gap is only meaningful against a named reference. The receipt
  records the llama.cpp commit and build flags behind `llama_tps`, and
  rows compared against different builds are not rendered together.
  Two llama.cpp builds have been measured apart on the same file.
- Compare builds interleaved (`main, branch, main, branch`) and report
  the raw sequence. Sample utilisation repeatedly, during the run.
- Before optimising, convert the measurement into the limiting resource
  (for decode, `weight_bytes * tok/s` against memory bandwidth) and
  check the change can close the gap even in its best case.
- Check kernel coverage before profiling: a gap column cannot show a
  missing kernel.

## Consequences

Numbers are fewer and slower to get, and every one of them can be
re-run from its receipt.
