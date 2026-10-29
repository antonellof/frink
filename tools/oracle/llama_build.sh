#!/usr/bin/env bash
# Checks llama.cpp out at one commit and builds the two reference
# dumpers against its libllama, CPU only.
#
#   tools/oracle/llama_build.sh <full-sha> <work-dir>
#
# Leaves, under <work-dir>:
#   src/          llama.cpp at <sha> (gguf-py is what the fixture scripts import)
#   build/        the cmake build; libllama under build/bin
#   llama_logits  tools/llama_logits.c, what `frink parity` runs
#   ref_logits    scripts/gptoss_reference_logits.cpp, what writes the goldens
#
# Source, in order: an existing <work-dir>/src at <sha>; a clone of
# $LLAMA_CPP_LOCAL (default .scratch/llama.cpp, read only: --shared
# borrows its objects and writes nothing into it); a shallow fetch of
# the commit from GitHub.
#
# ONE build, not the two `tools/build_llama_logits.sh` asks for. The
# second dumper calibrates the WRONG line on K-quant checkpoints, where
# two libllama builds disagree with each other (#111). Every fixture
# here is F32 / F16 / Q8_0 (asserted by tools/oracle/check_quant.py, run
# from oracle.sh), whose answers are identical across libllama builds,
# so a second build would measure a spread of zero at the cost of a
# second C++ compile.
set -euo pipefail

SHA="$1"
WORK="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# A worktree's .scratch is its main checkout's. A tree named for the
# commit is preferred; any other is only a source of objects, since
# what gets built is checked out at <sha> and verified below.
MAIN="$(cd "$(git -C "$ROOT" rev-parse --git-common-dir)/.." && pwd)"
LOCAL="${LLAMA_CPP_LOCAL:-}"
if [[ -z "$LOCAL" ]]; then
  for c in "$ROOT/.scratch/llama.cpp-${SHA:0:8}" "$MAIN/.scratch/llama.cpp-${SHA:0:8}" \
    "$ROOT/.scratch/llama.cpp" "$MAIN/.scratch/llama.cpp"; do
    if [[ -e "$c/.git" ]]; then LOCAL="$c"; break; fi
  done
fi
SRC="$WORK/src"
BUILD="$WORK/build"
mkdir -p "$WORK"

if [[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)" != "$SHA" ]]; then
  rm -rf "$SRC"
  # The local clone can hold the commit without every tree under it (a
  # partial clone), so a failed checkout falls through to the fetch.
  if ! { [[ -n "$LOCAL" ]] &&
    git clone --quiet --shared --no-checkout "$LOCAL" "$SRC" 2>/dev/null &&
    git -C "$SRC" -c advice.detachedHead=false checkout --quiet "$SHA" 2>/dev/null; }; then
    rm -rf "$SRC"
    git init --quiet "$SRC"
    git -C "$SRC" fetch --quiet --depth 1 https://github.com/ggml-org/llama.cpp "$SHA"
    git -C "$SRC" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
  fi
  rm -rf "$BUILD"
fi
# THE REFERENCE'S VINTAGE IS PART OF THE RESULT: a libllama built from
# any other commit answers for that commit, silently (ADR 0011).
HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$HEAD" != "$SHA" ]]; then
  echo "llama.cpp source at $SRC is $HEAD, not $SHA; refusing to build it" >&2
  exit 1
fi

# Metal and BLAS off: the reference is llama.cpp's plain CPU path,
# the one frink's CPU path is cross-validated against, on every OS.
#
# GGML_NATIVE OFF and a FIXED instruction set. The answer is a
# function of the ISA ggml is compiled for: its dot products reduce in
# an order set by the SIMD width, and on the pin a native M2 Pro build
# and a portable one differ by up to 3.9e-4 (`bloom`). GitHub's
# ubuntu runners are not one CPU model, so a native build would answer
# differently from runner to runner. On x86-64 the set is the
# AVX2 baseline every runner has (SSE4.2, AVX, AVX2, BMI2, FMA, F16C),
# with AVX-512, AVX-VNNI and AMX off even where the runner has them.
# The goldens are this build's answer on CI (scripts/oracle.sh); any
# other host's build differs from it in the last bits.
isa=(-DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_BACKEND_DL=OFF)
if [[ "$(uname -m)" == x86_64 ]]; then
  isa+=(-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_BMI2=ON
    -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF
    -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
    -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF)
fi
# A build made with other flags (an older native one) is rebuilt.
STAMP="$BUILD/oracle-cmake-flags"
if [[ "$(cat "$STAMP" 2>/dev/null || true)" != "${isa[*]}" ]] ||
  ! ls "$BUILD"/bin/libllama.* >/dev/null 2>&1; then
  rm -rf "$BUILD"
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF "${isa[@]}" >/dev/null
  cmake --build "$BUILD" --target llama -j "$(getconf _NPROCESSORS_ONLN)" >/dev/null
  printf '%s\n' "${isa[*]}" > "$STAMP"
fi

LLAMA_CPP_PREFIX="$BUILD" LLAMA_CPP_SOURCE="$SRC" LLAMA_LOGITS_OUT="$WORK/llama_logits" \
  bash "$ROOT/tools/build_llama_logits.sh" >/dev/null
c++ -std=c++17 -O2 "$ROOT/scripts/gptoss_reference_logits.cpp" \
  -I"$SRC/include" -I"$SRC/ggml/include" \
  -L"$BUILD/bin" -lllama -Wl,-rpath,"$BUILD/bin" -o "$WORK/ref_logits"
echo "llama.cpp $SHA built under $WORK"
