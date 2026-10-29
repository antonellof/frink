#!/usr/bin/env bash
# The llama.cpp oracle (ADR 0001, ADR 0011), the same script on CI and a laptop.
#
#   scripts/oracle.sh [--llama-ref <sha|master>] [--report-only] [--summary FILE]
#                     [--golden-tol T]
#
# 1. builds libllama at the pin (tools/llama_cpp_pin) or --llama-ref, CPU
#    only, under target/oracle/<sha>/, and the two reference dumpers;
# 2. regenerates every synthetic fixture with that commit's gguf-py;
# 3. regenerates every checked-in llama.cpp golden from that libllama;
# 4. runs `frink parity` on every fixture (first-token logits and a
#    32-step greedy decode, on token ids) and holds each to the outcome
#    tools/oracle/fixtures.tsv expects;
# 5. fails if 2 or 3 changed a tracked file or 4 found an unexpected
#    outcome. A golden is a cache of the oracle's answer: when this
#    fails on a diff, the diff IS the new answer, and committing it is
#    the fix only after reading why it moved.
#
# --report-only (the scheduled run against master) does all of it and
# exits 0, leaving the diff and the summary as the report.
#
# ONLY CI'S ANSWER IS COMMITTED. libllama is built portable at a fixed
# ISA (tools/oracle/llama_build.sh), so every GitHub runner gives the
# same answer, but another host (an arm64 laptop, an AVX-512 box) gives
# its own, up to ~4e-4 away on these fixtures. Run locally, a golden
# rewrite at the default --golden-tol (3e-6) is that host's last bits,
# not a finding: pass `--golden-tol 1e-3` to see only moves bigger
# than the ISA spread, and when CI's oracle job rewrites goldens, take
# its `oracle-rewritten-goldens` artifact rather than a local rewrite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REF=""
REPORT_ONLY=0
SUMMARY="${GITHUB_STEP_SUMMARY:-}"
GOLDEN_TOL=3e-6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-ref) REF="$2"; shift 2 ;;
    --report-only) REPORT_ONLY=1; shift ;;
    --summary) SUMMARY="$2"; shift 2 ;;
    --golden-tol) GOLDEN_TOL="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
PIN="$(tr -d '[:space:]' < tools/llama_cpp_pin)"
REF="${REF:-$PIN}"
if [[ ! "$REF" =~ ^[0-9a-f]{40}$ ]]; then
  REF="$(git ls-remote https://github.com/ggml-org/llama.cpp "$REF" | awk 'NR==1{print $1}')"
  [[ "$REF" =~ ^[0-9a-f]{40}$ ]] || { echo "cannot resolve the llama.cpp ref" >&2; exit 2; }
fi
WORK="$ROOT/target/oracle/$REF"
mkdir -p "$WORK"
: > "$WORK/summary.md"

note() { printf '%s\n' "$@" | tee -a "$WORK/summary.md"; }

note "## llama.cpp oracle at \`$REF\`"
if [[ "$REF" == "$PIN" ]]; then
  note "" "The pin (\`tools/llama_cpp_pin\`)."
else
  note "" "NOT the pin (\`$PIN\`): a drift report, not a verdict on any PR."
fi
note ""

bash tools/oracle/llama_build.sh "$REF" "$WORK"
PYTHONPATH="$WORK/src/gguf-py" python3 tools/oracle/check_quant.py
cargo build --release -p frink-cli

status=0
python3 tools/oracle/fixtures.py "$WORK/src" || status=1
if [[ -z "${CI:-}" && "$GOLDEN_TOL" == 3e-6 ]]; then
  note "Not CI: a golden rewrite at tol 3e-6 may be this host's ISA, not llama.cpp." \
    "Commit only CI's answer; see the header of scripts/oracle.sh." ""
fi
python3 tools/oracle/goldens.py "$WORK" --tol "$GOLDEN_TOL" --summary "$WORK/summary.md" || status=1
python3 tools/oracle/parity.py target/release/frink "$WORK/llama_logits" \
  --summary "$WORK/summary.md" || status=1

changed="$(git status --porcelain -- crates/frink-models/tests | awk '{print $2}')"
if [[ -n "$changed" ]]; then
  status=1
  note "### Tracked files the oracle rewrote"
  note ""
  while read -r f; do note "- \`$f\`"; done <<< "$changed"
  note ""
  git --no-pager diff --stat -- crates/frink-models/tests
fi

if [[ "$REPORT_ONLY" == 1 && -n "$changed" ]]; then
  # Against master, what the rewritten goldens mean for frink is the
  # report: each failing test binary is an architecture that drifts.
  note "### frink against the regenerated goldens"
  note ""
  cargo test --release -p frink-models --no-fail-fast 2>&1 | tee "$WORK/test.log" >/dev/null || true
  grep -E "^test .* FAILED$" "$WORK/test.log" | sed 's/^/- /' | tee -a "$WORK/summary.md" || true
  note ""
fi

if [[ -n "$SUMMARY" ]]; then cat "$WORK/summary.md" >> "$SUMMARY"; fi
echo "summary: $WORK/summary.md"
if [[ "$REPORT_ONLY" == 1 ]]; then exit 0; fi
exit "$status"
