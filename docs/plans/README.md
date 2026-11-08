# Plans

Every file in this directory is listed here once.

## The goal

- **[`north-star.md`](north-star.md)** is the goal, where the project
  stands against it, and the only ranking of the work. Open items are
  GitHub issues.

Decisions live in [`../adr/`](../adr/README.md). A plan here is the
design of one item in flight. When it lands, its lasting decisions
become ADRs and the plan moves to `archive/`
([ADR 0007](../adr/0007-issues-plans-adrs.md)).

## Design docs

- **[`arch-registry-migration.md`](arch-registry-migration.md)**, one
  file per architecture (#403, ADR 0003): the flow of architecture
  identity today, the target types, and the steps that each change no
  behaviour.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a MoE whose
  weights exceed RAM (#406). The Metal and host expert slots have
  landed. Asynchronous reads, hotness decay and a run that really
  exceeds memory have not.
- **[`amd.md`](amd.md)**, ROCm on RDNA and CDNA, the Ryzen AI NPU spike,
  and Vulkan for every Radeon (#494).

## Audits

- **[`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)**, the
  llama.cpp surface at the pinned commit: its new architecture names,
  the served behaviour that moved upstream, and what was fixed.
  `tests/oracle_manifest.rs` holds its title to `tools/llama_cpp_pin`,
  so it stays here until the pin moves and a new audit replaces it.
- **[`serving-parity-audit.md`](serving-parity-audit.md)**, continuous
  batching, paged KV, prefix caching and cache-aware admission read
  against the code, every claim carrying the line that decides it
  (`documented_serving.rs` holds the citations).

## Process

- **[`contribution-workflow.md`](contribution-workflow.md)**, the rule
  that a completed feature is a branch and a pull request and a defect
  is a GitHub issue. It also carries the parallel-agent rules and the
  release recipe.

## History

- **[`archive/`](archive/README.md)** holds superseded plans and dated
  audits, each with where its open items went.
- **[`done/`](done/README.md)** holds plans that finished.
- **[`on-hold/`](on-hold/)** holds work ranked below the goal, each with
  the condition that brings it back: `dflash-speculative-decoding.md`
  and `frink-ui.md`.

The counts in the `done/` index are derived from the files by
`crates/frink-models/tests/documented_counts.rs`, because two of them
had already drifted.

## The rules that keep being re-learned

**A plan's own status field is a claim, not evidence.** Verify against
the code. A merged PR once marked `paged-decode-path` complete while it
returned wrong tokens on Metal.

**One agent owns a file.** Two branches editing the same file produce a
merge nobody can review, and this project has already had one branch
silently revert three others.

**No agent runs benchmarks.** Measurement needs a quiet host, and a
loaded run reads 25-45% low.

**Refusing is not a defect.** llama.cpp will often run something
approximately. This project stops and names what is missing. A refusal
is a gap in coverage, not a bug.
