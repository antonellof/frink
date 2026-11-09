#!/usr/bin/env bash
# frink CUDA validation and benchmark against llama.cpp on one HF Jobs GPU box.
# Env: RESULTS_REPO (dataset), RUN_ID, BRANCH, STAGES, MODELS (id|repo|file;...), HF_TOKEN (secret).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
BRANCH=${BRANCH:-main}
STAGES=${STAGES:-hwtest,verify,parity,profile,bench,nsys}
W=/work
OUT=$W/out
mkdir -p "$OUT/logs" "$OUT/receipts" "$OUT/profiles" "$W/models"
SUM=$OUT/summary.txt
log() { echo "=== [$(date -u +%H:%M:%S)] $*" | tee -a "$SUM"; }
has() { [[ ",$STAGES," == *",$1,"* ]]; }
push() {
  [[ -n "${RESULTS_REPO:-}" ]] || return 0
  hf upload "$RESULTS_REPO" "$OUT" "runs/$RUN_ID" --repo-type dataset \
    --commit-message "run $RUN_ID: $1" >/dev/null 2>&1 || echo "upload failed ($1)"
}
trap 'log "job exiting"; push final' EXIT
# run <logname> <timeout> cmd...: logs to a file, records the exit code in the summary.
run() {
  local name=$1 to=$2; shift 2
  local t0=$SECONDS
  timeout "$to" "$@" >"$OUT/logs/$name.log" 2>&1
  local rc=$?
  log "$name rc=$rc ($((SECONDS - t0))s)"
  return $rc
}

log "run $RUN_ID branch $BRANCH stages $STAGES"
nvidia-smi | tee "$OUT/logs/nvidia-smi.txt"
nproc; free -g; uptime

log "setup"
apt-get update -qq >/dev/null && apt-get install -y -qq git curl ca-certificates build-essential \
  cmake pkg-config libssl-dev python3 python3-pip >/dev/null 2>&1
pip3 install -q -U huggingface_hub hf_xet >/dev/null 2>&1
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null 2>&1
. "$HOME/.cargo/env"
# cudarc dlopens libcuda.so and libnvrtc.so by their unversioned names.
if ! ldconfig -p | grep -q 'libcuda.so '; then
  so1=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
  [[ -n "$so1" ]] && ln -sf "$so1" "$(dirname "$so1")/libcuda.so" && ldconfig
fi
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
ldconfig -p | grep -E 'libcuda.so |libnvrtc' | head -5
push setup

git clone -q --depth 1 -b "$BRANCH" https://github.com/antonellof/frink "$W/frink" || exit 1
cd "$W/frink"
log "frink $(git rev-parse HEAD)"
git rev-parse HEAD >"$OUT/frink_commit.txt"
PIN=$(tr -d '[:space:]' <tools/llama_cpp_pin)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')

# Models download in the background while everything builds.
MODELS=${MODELS:-"llama31_8b_q4km|bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"}
IFS=';' read -r -a ENTRIES <<<"$MODELS"
(
  for e in "${ENTRIES[@]}"; do
    IFS='|' read -r id repo file <<<"$e"
    hf download "$repo" "$file" --local-dir "$W/models" >/dev/null 2>&1 &&
      echo "ok $id" || echo "FAILED $id"
  done
) >"$OUT/logs/downloads.log" 2>&1 &
DL=$!

log "build llama.cpp $PIN (CUDA sm_$CAP)"
git init -q "$W/llama" && git -C "$W/llama" fetch -q --depth 1 https://github.com/ggml-org/llama.cpp "$PIN" &&
  git -C "$W/llama" -c advice.detachedHead=false checkout -q FETCH_HEAD
run build-llama-cuda 60m bash -c "cmake -S $W/llama -B $W/llama/build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CAP}-real -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF && cmake --build $W/llama/build-cuda --target llama-bench llama-cli -j$(nproc)"
export PATH=$W/llama/build-cuda/bin:$PATH
command -v llama-bench || log "NO llama-bench"
push build-llama

if has parity; then
  log "build the oracle dumpers"
  run build-oracle 30m env LLAMA_CPP_LOCAL="$W/llama" bash tools/oracle/llama_build.sh "$PIN" "$W/oracle"
  run build-native-cpu 30m bash -c "cmake -S $W/oracle/src -B $W/llama-native -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON \
    && cmake --build $W/llama-native --target llama -j$(nproc) \
    && LLAMA_CPP_PREFIX=$W/llama-native LLAMA_CPP_SOURCE=$W/oracle/src \
       LLAMA_LOGITS_OUT=$W/llama_logits_native bash tools/build_llama_logits.sh"
fi

log "build frink (release, cuda)"
run build-frink 60m cargo build --release -p frink-cli --features cuda || { push build-frink; exit 1; }
FRINK=$W/frink/target/release/frink
push build-frink

if has hwtest; then
  log "CUDA hardware tests"
  run test-frink-cuda 40m cargo test -p frink-cuda --features cuda -- --include-ignored
  grep -E '^test result|FAILED|panicked' "$OUT/logs/test-frink-cuda.log" | tee -a "$SUM"
  run test-frink-moe 30m cargo test -p frink-moe --features cuda -- --ignored
  grep -E '^test result|FAILED|panicked' "$OUT/logs/test-frink-moe.log" | tee -a "$SUM"
  push hwtest
fi

log "waiting for downloads"
wait "$DL"
cat "$OUT/logs/downloads.log" | tee -a "$SUM"
ls -la "$W/models" | tee -a "$SUM"

quiet() { # bench refuses a busy host: let the 1-minute load decay
  for _ in $(seq 1 40); do
    awk '{exit !($1 < 1.5)}' /proc/loadavg && return 0
    sleep 10
  done
}

for e in "${ENTRIES[@]}"; do
  IFS='|' read -r id repo file <<<"$e"
  m="$W/models/$file"
  [[ -f "$m" ]] || { log "$id: missing $file"; continue; }
  log "---- $id ($file)"
  if has verify; then
    run "verify-$id" 20m "$FRINK" verify -m "$m" --backend cuda --prompt-tokens 64
    tail -4 "$OUT/logs/verify-$id.log" | sed "s/^/  [$id verify] /" | tee -a "$SUM"
  fi
  if has parity; then
    run "parity-cuda-$id" 20m env FRINK_CUDA=1 "$FRINK" parity -m "$m" \
      --dumper "$W/oracle/llama_logits" --dumper "$W/llama_logits_native" --prompt-tokens 64
    grep -E '^tokenizer|^parity |^  KL\(llama|^  top-1  ' "$OUT/logs/parity-cuda-$id.log" | head -8 |
      sed "s/^/  [$id parity cuda] /" | tee -a "$SUM"
    if ! grep -q '^parity .*: MATCH' "$OUT/logs/parity-cuda-$id.log"; then
      run "parity-cpu-$id" 30m env FRINK_CUDA=0 "$FRINK" parity -m "$m" \
        --dumper "$W/oracle/llama_logits" --dumper "$W/llama_logits_native" --prompt-tokens 64
      grep -E '^parity |^  KL\(llama|^  top-1  ' "$OUT/logs/parity-cpu-$id.log" | head -6 |
        sed "s/^/  [$id parity cpu] /" | tee -a "$SUM"
    fi
  fi
  if has profile; then
    run "count-$id" 10m env FRINK_CUDA_COUNT=1 "$FRINK" -m "$m" -p "The capital of France is" -n 32 \
      --temp 0 --no-cnv -ngl 99
    grep -i -E 'host wait|h2d|d2h|copies|tok/s|t/s' "$OUT/logs/count-$id.log" | head -6 |
      sed "s/^/  [$id count] /" | tee -a "$SUM"
  fi
  if has bench; then
    quiet
    run "bench-$id" 30m "$FRINK" bench -m "$m" -p 512 -n 128 -r 3 --n-gpu-layers 99 --compare --max-load "${MAX_LOAD:-2.0}" \
      --suite-id "$id" --backend-label cuda --receipt "$OUT/receipts/${id}_cuda.json"
    grep -E 'pp512|tg128|gap|llama' "$OUT/logs/bench-$id.log" | head -10 |
      sed "s/^/  [$id bench] /" | tee -a "$SUM"
  fi
  push "$id"
done

if has nsys; then
  log "nsys kernel profile"
  apt-get install -y -qq cuda-nsight-systems-12-8 >/dev/null 2>&1 || apt-get install -y -qq nsight-systems-cli >/dev/null 2>&1
  NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
  e=${ENTRIES[0]}; IFS='|' read -r id repo file <<<"$e"; m="$W/models/$file"
  if [[ -n "$NSYS" && -f "$m" ]]; then
    run nsys-frink-$id 20m "$NSYS" profile -o "$OUT/profiles/frink-$id" --stats=true -t cuda \
      "$FRINK" bench -m "$m" -p 512 -n 128 -r 1 --n-gpu-layers 99 --max-load 0
    run nsys-llama-$id 20m "$NSYS" profile -o "$OUT/profiles/llama-$id" --stats=true -t cuda \
      llama-bench -m "$m" -p 512 -n 128 -r 1 -ngl 99
  else
    log "no nsys"
  fi
fi
log "done"
