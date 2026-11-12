#!/usr/bin/env bash
# launch_release.sh <tag>: the full validation on the tag, L40S and RTX PRO 6000.
set -euo pipefail
TAG=$1
SP=$(cd "$(dirname "$0")" && pwd)
set -a; . "${FRINK_ENV:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.env}"; set +a
MODELS=$(cat $SP/models10.txt)
STAMP=$(date -u +%Y%m%dT%H%M)
L_RUN=release-${TAG}-l40s-$STAMP
R_RUN=release-${TAG}-rtx6000-$STAMP
echo "L40S run $L_RUN"
hf jobs run --flavor l40sx1 --timeout 6h -d --secrets HF_TOKEN \
  -e RESULTS_REPO=antonellof/frink-cuda-runs -e RUN_ID=$L_RUN -e BRANCH=$TAG \
  -e "MODELS=$MODELS" nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -c "$(cat $SP/cuda_validate.sh)" 2>&1 | grep 'ID:'
echo "RTX run $R_RUN"
SCRIPT="$(cat $SP/cuda_validate.sh)" RUN=$R_RUN TAG=$TAG MODELS="$MODELS" python3 -I - <<'PY'
import os
from huggingface_hub import HfApi
j = HfApi(token=os.environ["HF_TOKEN"]).run_job(
    image="nvidia/cuda:12.8.1-devel-ubuntu22.04",
    command=["bash", "-c", os.environ["SCRIPT"]],
    flavor="rtx-pro-6000",
    env={"RESULTS_REPO": "antonellof/frink-cuda-runs", "RUN_ID": os.environ["RUN"],
         "BRANCH": os.environ["TAG"], "MODELS": os.environ["MODELS"]},
    secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    timeout="6h",
)
print("Job started with ID:", j.id)
PY
