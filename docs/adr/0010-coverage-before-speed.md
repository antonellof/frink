# 0010. Coverage before speed: every kind has a kernel, or says it has none

Status: Accepted
Date: 2026-10-28

## Context

A CUDA K-quant prefill measured 4.88 tok/s against llama.cpp's 1586.80
and looked like a tuning problem. There was no GEMM for that kind at
all: the layer fell back to the host and still answered correctly. The
coverage matrix that would have shown it lived as prose in a plan, and
a host-check tool once checked nothing for a day because its shape list
disagreed with the kernel table.

## Decision

- Each backend has one table of the quant kinds it runs, for matvec and
  for GEMM. The coverage matrix is generated from those tables and
  checked in; a test fails when they disagree.
- A layer that falls back to the host says so: in `frink inspect-plan`,
  in the receipt, and once in the log at load.
- A performance item for a backend starts by checking the matrix.
  Missing kernels are closed before existing ones are tuned.

## Consequences

The scoreboard can show "runs on the backend" separately from "not
slower", so a missing kernel never hides in a gap column. Tracked in
#405.
