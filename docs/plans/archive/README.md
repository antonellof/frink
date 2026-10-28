# Archive

Superseded plans. Their open items were merged into
[`../roadmap.md`](../roadmap.md) by theme; these files are kept because
the reasoning in them is worth not re-deriving, and because several
todos carry measurements and design arguments that the merged summary
compresses.

Do not add work here. If something in one of these is still live and
the roadmap missed it, put it in the roadmap.

| File | What it held | Where it went |
|---|---|---|
| `llama-cpp-parity-push.md` | 16 items: the audit's silently-wrong findings, CPU kernels, coverage, quality gates | Themes B, E, F, G |
| `freetoken-parity.md` | 11 items: MoE residency and executor, radix and window policy, bench-bw, real checkpoints | Themes C, H |
| `amd-strix-halo.md` | 14 items: x86 measurement, AVX-512, the backend seam, four Vulkan items | Theme D |
| `serving-and-tiered-kv.md` | 1 item: time-debt prefill/decode interleaving | Theme C3 |
| `one-binary-serve.md` | 1 item: the cross-target gate, half landed | Theme F |
| `project-status-2026-09.md` | Not a plan: the September closure log, moved out of `../README.md` | The dated status in `../README.md` |
| `completion-push.md` | A schedule, not a feature plan: file ownership and merge discipline | The rules at the end of `roadmap.md` and `README.md` |

`completion-push.md` is worth reading before dispatching parallel work
even though its waves are spent. Its merge discipline is the reason
this project stopped losing branches to each other.
