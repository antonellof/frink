#!/usr/bin/env bash
# watch.sh <job-id> <run-id>: prints new summary.txt lines and the job's terminal state.
J=$1; R=$2
set -a; . "${FRINK_ENV:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.env}"; set +a
seen=0
while true; do
  s=$(curl -sf -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/datasets/${RESULTS_REPO:-antonellof/frink-cuda-runs}/resolve/main/runs/$R/summary.txt" 2>/dev/null || true)
  if [[ -n "$s" ]]; then
    n=$(printf '%s\n' "$s" | wc -l)
    if (( n > seen )); then printf '%s\n' "$s" | tail -n $((n - seen)); seen=$n; fi
  fi
  st=$(curl -sf -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/jobs/${HF_USER:-antonellof}/$J" |
    python3 -c "import json,sys; d=json.load(sys.stdin)['status']; print(d['stage'], d.get('message') or '')" 2>/dev/null || echo "?")
  case "$st" in
    COMPLETED*|ERROR*|CANCELED*|DELETED*) echo "JOB $J $st"; exit 0 ;;
  esac
  sleep 60
done
