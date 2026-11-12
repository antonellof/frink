# Archive

Superseded plans. Their open items were merged into the September
roadmap by theme, and that roadmap is archived here too
([`roadmap-2026-09.md`](roadmap-2026-09.md)); these files are kept because
the reasoning in them is worth not re-deriving, and because several
todos carry measurements and design arguments that the merged summary
compresses.

Do not add work here. If something in one of these is still live, open
a GitHub issue for it; [`../north-star.md`](../north-star.md) ranks the
open work.

| File | What it held | Where it went |
|---|---|---|
| `llama-cpp-parity-push.md` | 16 items: the audit's silently-wrong findings, CPU kernels, coverage, quality gates | Themes B, E, F, G |
| `freetoken-parity.md` | 11 items: MoE residency and executor, radix and window policy, bench-bw, real checkpoints | Themes C, H |
| `amd-strix-halo.md` | 14 items: x86 measurement, AVX-512, the backend seam, four Vulkan items | Theme D |
| `serving-and-tiered-kv.md` | 1 item: time-debt prefill/decode interleaving | Theme C3 |
| `one-binary-serve.md` | 1 item: the cross-target gate, half landed | Theme F |
| `project-status-2026-09.md` | Not a plan: the September closure log, moved out of `../README.md` | The dated status in `../README.md` |
| `roadmap-2026-09.md` | The single roadmap of 2026-09-02: 20 items in themes A to H, merged from the first five plans in this table | Archived 2026-10-28 with several pending items landed; open work is GitHub issues, ranked by `../north-star.md` |
| `model-layer-reorg.md` | Making an architecture a new file: the decoration audit, the shared `attn_block`, the 2026-10-28 file splits | Superseded by `../arch-registry-migration.md` (ADR 0003) |
| `metal-parallel-concurrency.md` | Metal parallel decode and the continuous-batching prefill fix | Shipped in 0.15.2 (#47) and 0.15.3 |
| `llama-cpp-full-parity-audit-2026-09-02.md` | Not a plan: a file-by-file llama.cpp to frink map, a parity sweep and a priority list | Superseded by `../parity-audit-2026-09-19.md` and `../parity-audit-2026-10-28.md` |
| `llama-cpp-parity-review-2026-09-02.md` | Not a plan: the ranked next actions for llama.cpp parity as a serving and CLI stack | Its actions landed or moved to the roadmap; the current audits supersede it |
| `llama-cpp-parity-update-2026-09-03.md` | Not a plan: the post-merge delta to the 2026-09-02 audit | Superseded by the same two audits |
| `closures-2026-09.md` | Not a plan: the dated record of how each generic-path architecture closed and what building it found, 2026-08 to 2026-10 | Moved out of `CLAUDE.md` on 2026-10-28 so that file stays short guidance |
| `completion-push.md` | A schedule, not a feature plan: file ownership and merge discipline | The rules at the end of `roadmap-2026-09.md` and `../README.md` |
| `cpu-cuda-parity.md` | The CPU and CUDA gaps ordered by what rented hosts measured, the kernel-coverage matrix, and a "Traps" section worth reading before CUDA work | Archived 2026-11-08: the matrix is `../../KERNELS.md`, the gaps are #133, #259 and #405 |
| `gdn-resident-state.md` | A recurrent layer that stays on the device: Bonsai-2-27B decode at parity with its reference fork, with the per-token ledger and the measured non-results | Archived 2026-11-08: decode is done, the PTQ1_0 prefill GEMM is priority 12 in `../north-star.md` |
| `llama-cpp-gap-inventory.md` | Not a plan: the evidence-backed llama.cpp differential of 2026-09-01. Its paths predate the 2026-10 splits | Its top ten closed. The K-quant drift explanation is still cited by the README |
| `parity-audit-2026-09-19.md` | Not a plan: the llama.cpp surface at the `5b59b83` pin, with the serving gaps an architecture count cannot see | Superseded by `../parity-audit-2026-10-28.md` |
| `vulkan-beachhead-verdict.md` | The hardware-reach GO/NO-GO: a Q8_0 matvec on a real device, and the backend-seam survey | GO. The seam exists now (`weight_matrix/gpu_backend.rs`), and the rest is #409 |
| `speculative-decoding.md` | The lossless rejection rule, a second GGUF as the drafter, `frink run -md` | Steps 1 to 3 shipped. The server draft model and MTP heads are priority 4 in `../north-star.md` |
| `server-speculative-decoding.md` | The server half: prompt lookup ships on the private loop, a draft model does not reach the server | Priority 4 in `../north-star.md` |

`completion-push.md` is worth reading before dispatching parallel work
even though its waves are spent. Its merge discipline is the reason
this project stopped losing branches to each other.
