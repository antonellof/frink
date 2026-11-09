#!/usr/bin/env bash
# fetch.sh <run-id>: downloads the run's results and prints the bench table.
R=$1
D=${FRINK_RUNS_DIR:-${TMPDIR:-/tmp}/frink-hf-runs}
set -a; . "${FRINK_ENV:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.env}"; set +a
mkdir -p "$D"
hf download "${RESULTS_REPO:-antonellof/frink-cuda-runs}" --repo-type dataset --include "runs/$R/*" --local-dir "$D" >/dev/null 2>&1
echo "$D/runs/$R"
python3 - "$D/runs/$R/receipts" <<'EOF'
import json, sys, glob, os
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    d = json.load(open(p))
    for t in d.get('tests', []):
        rows.append((d.get('id'), t['test'], t.get('frink_tps'), t.get('llama_tps')))
print(f"{'model':24} {'test':8} {'frink':>10} {'llama':>10} {'frink/llama':>11}")
for id_, test, f, l in rows:
    r = f / l if f and l else float('nan')
    print(f"{id_:24} {test:8} {f or 0:10.1f} {l or 0:10.1f} {r:11.2f}")
EOF
