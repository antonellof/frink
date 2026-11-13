# Plans

Every file in this directory is listed here once.

## The roadmap

- **[`north-star.md`](north-star.md)** is the goal, where the project
  stands against it, and the only ranking of the work. Each item is a
  GitHub issue.

Decisions live in [`../adr/`](../adr/README.md). A plan here is the
design of one item in flight. When it lands, its lasting decisions
become ADRs and the plan is deleted; git history is the archive
([ADR 0007](../adr/0007-issues-plans-adrs.md)).

## Design docs, one per item in flight

- **[`arch-registry-migration.md`](arch-registry-migration.md)**, one
  file per architecture (#403, ADR 0003): the flow of architecture
  identity, the target types, and the steps that each change no
  behaviour.
- **[`out-of-core-moe.md`](out-of-core-moe.md)**, running a MoE whose
  weights exceed RAM (#406).
- **[`amd.md`](amd.md)**, ROCm on RDNA and CDNA, the Ryzen AI NPU spike,
  and Vulkan for every Radeon (#494).

## Audits

Each is held to the code by a test, so it stays until it is replaced.

- **[`parity-audit-2026-10-28.md`](parity-audit-2026-10-28.md)**, the
  llama.cpp surface at the pinned commit. `tests/oracle_manifest.rs`
  holds its title to `tools/llama_cpp_pin`; the next pin move replaces
  it with that pin's audit.
- **[`serving-parity-audit.md`](serving-parity-audit.md)**, continuous
  batching, paged KV, prefix caching and cache-aware admission read
  against the code, every claim carrying the line that decides it
  (`documented_serving.rs` holds the citations).

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
