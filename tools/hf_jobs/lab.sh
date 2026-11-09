#!/usr/bin/env bash
# Kernel lab on one HF Jobs GPU box: hardware tests, GEMM microbench old vs new, frink bench.
# Env: RESULTS_REPO, RUN_ID, BRANCH, MODELS (id|repo|file;...), BASE_TC (git ref whose mul_mm_tc.rs is the baseline),
#      STAGES (tests,gemm,verify,bench,nsys), HF_TOKEN (secret).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
STAGES=${STAGES:-tests,gemm,verify,bench,nsys}
W=/work
OUT=$W/out
mkdir -p "$OUT/logs" "$OUT/receipts" "$OUT/profiles" "$W/models"
SUM=$OUT/summary.txt
log() { echo "=== [$(date -u +%H:%M:%S)] $*" | tee -a "$SUM"; }
has() { [[ ",$STAGES," == *",$1,"* ]]; }
push() {
  hf upload "$RESULTS_REPO" "$OUT" "runs/$RUN_ID" --repo-type dataset \
    --commit-message "run $RUN_ID: $1" >/dev/null 2>&1 || echo "upload failed ($1)"
}
trap 'log "job exiting"; push final' EXIT
run() {
  local name=$1 to=$2; shift 2
  local t0=$SECONDS
  timeout "$to" "$@" >"$OUT/logs/$name.log" 2>&1
  local rc=$?
  log "$name rc=$rc ($((SECONDS - t0))s)"
  return $rc
}

log "run $RUN_ID branch $BRANCH stages $STAGES"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv | tee -a "$SUM"
apt-get update -qq >/dev/null && apt-get install -y -qq git curl ca-certificates build-essential \
  cmake pkg-config libssl-dev python3 python3-pip >/dev/null 2>&1
pip3 install -q -U huggingface_hub hf_xet >/dev/null 2>&1
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null 2>&1
. "$HOME/.cargo/env"
if ! ldconfig -p | grep -q 'libcuda.so '; then
  so1=$(ldconfig -p | awk '/libcuda\.so\.1 /{print $NF; exit}')
  [[ -n "$so1" ]] && ln -sf "$so1" "$(dirname "$so1")/libcuda.so" && ldconfig
fi
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

git clone -q -b "$BRANCH" https://github.com/antonellof/frink "$W/frink" || exit 1
cd "$W/frink"
log "frink $(git rev-parse HEAD)"

MODELS=${MODELS:-}
IFS=';' read -r -a ENTRIES <<<"$MODELS"
(
  for e in "${ENTRIES[@]}"; do
    [[ -n "$e" ]] || continue
    IFS='|' read -r id repo file <<<"$e"
    hf download "$repo" "$file" --local-dir "$W/models" >/dev/null 2>&1 && echo "ok $id" || echo "FAILED $id"
  done
) >"$OUT/logs/downloads.log" 2>&1 &
DL=$!

if has tests; then
  run test-frink-cuda 40m cargo test -p frink-cuda --features cuda -- --include-ignored
  grep -E '^test result|FAILED|panicked|^test .*FAILED' "$OUT/logs/test-frink-cuda.log" | head -20 | tee -a "$SUM"
  push tests
fi

if has gemm; then
  run gemm-new 20m cargo run --release -p frink-cuda --features cuda --example mul_mm_bench
  cat "$OUT/logs/gemm-new.log" | grep -E '^(kind|Q|IQ)' | sed 's/^/  [new] /' | tee -a "$SUM"
  if [[ -n "${BASE_BRANCH:-}" ]]; then
    git clone -q -b "$BASE_BRANCH" https://github.com/antonellof/frink "$W/base" &&
      (cd "$W/base" && CARGO_TARGET_DIR=$W/frink/target-base timeout 20m cargo run --release -p frink-cuda --features cuda --example mul_mm_bench) \
        >"$OUT/logs/gemm-basebranch.log" 2>&1
    grep -E '^(kind|Q|IQ)' "$OUT/logs/gemm-basebranch.log" | sed 's/^/  [base] /' | tee -a "$SUM"
  fi
  if [[ -n "${BASE_TC:-}" ]]; then
    cp crates/frink-cuda/src/mul_mm_tc.rs /tmp/mul_mm_tc.new
    git fetch -q origin "$BASE_TC" && git show FETCH_HEAD:crates/frink-cuda/src/mul_mm_tc.rs >crates/frink-cuda/src/mul_mm_tc.rs
    run gemm-base 20m cargo run --release -p frink-cuda --features cuda --example mul_mm_bench
    cat "$OUT/logs/gemm-base.log" | grep -E '^(kind|Q|IQ)' | sed 's/^/  [base] /' | tee -a "$SUM"
    cp /tmp/mul_mm_tc.new crates/frink-cuda/src/mul_mm_tc.rs
  fi
  push gemm
fi

if has cmds; then
  # Each of CMDS (separated by ';') run in the checkout, its test summary lines kept.
  i=0
  IFS=';' read -ra CMDLIST <<<"${CMDS:-}"
  for c in "${CMDLIST[@]}"; do
    i=$((i + 1))
    run "cmd$i" 30m bash -c "$c"
    echo "  [cmd$i] $c" | tee -a "$SUM"
    grep -E '^test result|FAILED|panicked' "$OUT/logs/cmd$i.log" | head -6 | sed "s/^/  [cmd$i] /" | tee -a "$SUM"
  done
  push cmds
fi
if has gemmrev; then
  # mul_mm_bench built at each of GEMM_REVS on this one box, run with AB_VAR set to each of AB_VALS.
  for rev in ${GEMM_REVS:-}; do
    git -C "$W/frink" worktree add -q "$W/rev-$rev" "$rev" 2>/dev/null
    (cd "$W/rev-$rev" && CARGO_TARGET_DIR="$W/frink/target-rev" cargo build -q --release -p frink-cuda --features cuda --example mul_mm_bench) >"$OUT/logs/build-rev-$rev.log" 2>&1 || log "build $rev failed"
    cp "$W/frink/target-rev/release/examples/mul_mm_bench" "$W/mm-$rev"
  done
  for kind in ${MMQ_KINDS:-Q4_K}; do
    for round in 1 2; do
      for rev in ${GEMM_REVS:-}; do
        for val in ${AB_VALS:-x}; do
          run "gemmrev-$rev-$val-$kind-$round" 10m env FRINK_CUDA_MUL_MM=mmq "${AB_VAR:-NONE}=$val" "$W/mm-$rev" "$kind" ${GEMM_BATCH:-512}
          grep -E '^(Q|IQ|MX)' "$OUT/logs/gemmrev-$rev-$val-$kind-$round.log" | sed "s/^/  [r$round $rev $val] /" | tee -a "$SUM"
        done
      done
    done
  done
  push gemmrev
fi
if has gemmab; then
  # mul_mm_bench on the int8 GEMM with AB_VAR set to each of AB_VALS.
  for kind in ${MMQ_KINDS:-Q4_K}; do
    for val in ${AB_VALS:-}; do
      run "gemm-ab$val-$kind" 20m env FRINK_CUDA_MUL_MM=mmq "$AB_VAR=$val" cargo run -q --release -p frink-cuda --features cuda --example mul_mm_bench -- "$kind" ${GEMM_BATCH:-512}
      grep -E '^(kind|Q|IQ|MX)' "$OUT/logs/gemm-ab$val-$kind.log" | sed "s/^/  [$AB_VAR=$val] /" | tee -a "$SUM"
    done
  done
  push gemmab
fi
if has mmqab; then
  for kind in ${MMQ_KINDS:-Q8_0}; do
    run "gemm-tc-$kind" 20m cargo run -q --release -p frink-cuda --features cuda --example mul_mm_bench -- "$kind" 512
    grep -E '^(kind|Q|IQ|MX)' "$OUT/logs/gemm-tc-$kind.log" | sed "s/^/  [tc] /" | tee -a "$SUM"
    run "gemm-mmq-$kind" 20m env FRINK_CUDA_MUL_MM=mmq cargo run -q --release -p frink-cuda --features cuda --example mul_mm_bench -- "$kind" 512
    grep -E '^(kind|Q|IQ|MX)' "$OUT/logs/gemm-mmq-$kind.log" | sed "s/^/  [mmq] /" | tee -a "$SUM"
  done
  push mmqab
fi

if has mvsweep; then
  cargo build -q --release -p frink-cuda --features cuda --example matvec_bench 2>&1 | tail -3
  for cfg in wide:1 wide:2 wide:4 wide:8 narrow:1 narrow:4; do
    w=${cfg%%:*}; r=${cfg##*:}
    run "mv-$w-$r" 10m env FRINK_CUDA_MATVEC=$w FRINK_CUDA_MATVEC_WPR=$r \
      cargo run -q --release -p frink-cuda --features cuda --example matvec_bench
    grep -E '^Q' "$OUT/logs/mv-$w-$r.log" | sed "s/^/  [mv $w x$r] /" | tee -a "$SUM"
  done
  push mvsweep
fi

if has ncu; then
  apt-get install -y -qq cuda-nsight-compute-12-8 >/dev/null 2>&1 || true
  NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)
  if [[ -n "$NCU" ]]; then
    cargo build -q --release -p frink-cuda --features cuda --example mul_mm_bench --example matvec_bench 2>&1 | tail -2
    SECT="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis --section SchedulerStats"
    run ncu-gemm 15m env ${NCU_GEMM_ENV:-FRINK_CUDA_MUL_MM=tc} "$NCU" $SECT -k regex:${NCU_GEMM_K:-q4_k_mul_mm_tc} -c 2 target/release/examples/mul_mm_bench ${NCU_GEMM_ARGS:-Q4_K 512}
    grep -E 'Duration|Compute \(SM\) Throughput|Memory Throughput|Registers Per|Achieved Occupancy|Issue Slots Busy|No Eligible|Stall|Theoretical Occupancy|Block Limit' "$OUT/logs/ncu-gemm.log" | head -40 | tee -a "$SUM"
    grep -E 'Error|ERR_NVGPU|==PROF==' "$OUT/logs/ncu-gemm.log" | head -3 | tee -a "$SUM"
    run ncu-matvec 15m "$NCU" $SECT -k regex:q4_k_matvec_coalesced -c 3 target/release/examples/matvec_bench
    grep -E 'Error|ERR_NVGPU' "$OUT/logs/ncu-matvec.log" | head -3 | tee -a "$SUM"
    grep -E 'Duration|Memory Throughput|DRAM Throughput|Registers Per|Achieved Occupancy|No Eligible|Theoretical Occupancy|Block Limit' "$OUT/logs/ncu-matvec.log" | head -30 | tee -a "$SUM"
  else
    log "no ncu"
  fi
  push ncu
fi

if has verify || has bench || has nsys || has time || has mmqmodel || has ab; then
  run build-frink 60m cargo build --release -p frink-cli --features cuda || { push build; exit 1; }
  FRINK=$W/frink/target/release/frink
  wait "$DL"; cat "$OUT/logs/downloads.log" | tee -a "$SUM"
  quiet() { for _ in $(seq 1 40); do awk '{exit !($1 < 1.5)}' /proc/loadavg && return 0; sleep 10; done; }
  for e in "${ENTRIES[@]}"; do
    [[ -n "$e" ]] || continue
    IFS='|' read -r id repo file <<<"$e"
    m="$W/models/$file"
    [[ -f "$m" ]] || { log "$id: missing"; continue; }
    if has verify; then
      run "verify-$id" 20m "$FRINK" verify -m "$m" --backend cuda --prompt-tokens 64
      tail -3 "$OUT/logs/verify-$id.log" | sed "s/^/  [$id verify] /" | tee -a "$SUM"
      run "count-$id" 10m env FRINK_CUDA_COUNT=1 "$FRINK" -m "$m" -p "The capital of France is" -n 16 --temp 0 --no-cnv -ngl 99
      grep -E 'cuda decode token 2|prefill step 0|declined' "$OUT/logs/count-$id.log" | head -4 | sed "s/^/  [$id count] /" | tee -a "$SUM"
    fi
    if has mmqmodel; then
      run "verify-mmq-$id" 20m env FRINK_CUDA_MUL_MM=mmq "$FRINK" verify -m "$m" --backend cuda --prompt-tokens 64
      tail -1 "$OUT/logs/verify-mmq-$id.log" | sed "s/^/  [$id verify mmq] /" | tee -a "$SUM"
      for mode in tc mmq; do
        run "bench-$mode-$id" 30m env FRINK_CUDA_MUL_MM=$mode "$FRINK" bench -m "$m" -p 512 -n 128 -r 3 --n-gpu-layers 99 --max-load 0
        grep -E 'pp512|tg128' "$OUT/logs/bench-$mode-$id.log" | sed "s/^/  [$id bench $mode] /" | tee -a "$SUM"
      done
    fi
    if has ab; then
      # AB_VAR set to each of AB_VALS (space-separated): a verify at a 512-token prompt and a bench per value.
      for val in ${AB_VALS:-}; do
        run "verify-ab$val-$id" 30m env "$AB_VAR=$val" "$FRINK" verify -m "$m" --backend cuda --prompt-tokens ${AB_PROMPT:-512}
        tail -1 "$OUT/logs/verify-ab$val-$id.log" | sed "s/^/  [$id verify $AB_VAR=$val] /" | tee -a "$SUM"
        run "bench-ab$val-$id" 30m env "$AB_VAR=$val" "$FRINK" bench -m "$m" -p 512 -n 128 -r 3 --n-gpu-layers 99 --max-load 0
        grep -E 'pp512|tg128' "$OUT/logs/bench-ab$val-$id.log" | sed "s/^/  [$id bench $AB_VAR=$val] /" | tee -a "$SUM"
      done
    fi
    if has time; then
      run "time-$id" 30m env FRINK_CUDA_TIME=1 FRINK_CUDA_COUNT=1 "$FRINK" bench -m "$m" -p 512 -n 128 -r 2 --n-gpu-layers 99 --max-load 0
      grep -E 'step [12] time|decode time' "$OUT/logs/time-$id.log" | head -6 | sed "s/^/  [$id time] /" | tee -a "$SUM"
      grep -E 'pp512|tg128' "$OUT/logs/time-$id.log" | sed "s/^/  [$id time] /" | tee -a "$SUM"
    fi
    if has bench; then
      for r in ${WPRS:-1}; do
        [[ "${MAX_LOAD:-2.0}" == "0" ]] || quiet
        run "bench-$id-wpr$r" 30m env FRINK_CUDA_MATVEC_WPR=$r "$FRINK" bench -m "$m" -p 512 -n 128 -r 3 --n-gpu-layers 99 --max-load "${MAX_LOAD:-2.0}" \
          --suite-id "$id" --backend-label cuda --receipt "$OUT/receipts/${id}_cuda_wpr$r.json"
        grep -E 'pp512|tg128' "$OUT/logs/bench-$id-wpr$r.log" | head -4 | sed "s/^/  [$id bench x$r] /" | tee -a "$SUM"
      done
    fi
    push "$id"
  done
  if has nsys; then
    apt-get install -y -qq cuda-nsight-systems-12-8 >/dev/null 2>&1 || apt-get install -y -qq nsight-systems-cli >/dev/null 2>&1
    NSYS=$(command -v nsys || ls /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1)
    e=${ENTRIES[0]}; IFS='|' read -r id repo file <<<"$e"; m="$W/models/$file"
    if [[ -n "$NSYS" && -f "$m" ]]; then
      run nsys-frink-$id 20m "$NSYS" profile -o "$OUT/profiles/frink-$id" --stats=true -t cuda \
        "$FRINK" bench -m "$m" -p 512 -n 128 -r 1 --n-gpu-layers 99 --max-load 0
      grep -A25 'cuda_gpu_kern_sum' "$OUT/logs/nsys-frink-$id.log" | head -30 | tee -a "$SUM"
    else
      log "no nsys"
    fi
  fi
fi
log "done"
