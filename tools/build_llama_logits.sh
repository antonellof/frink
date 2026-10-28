#!/usr/bin/env bash
# Builds the `frink parity` reference dumper against an installed
# libllama. The dumper is C, not Rust, and is not part of the cargo
# workspace on purpose: it exists to be llama.cpp's own answer, so it
# links llama.cpp's own library rather than reimplementing anything.
#
# Override the llama.cpp prefix with LLAMA_CPP_PREFIX when it is not a
# Homebrew install (a source build works too — point at the directory
# holding include/llama.h and lib/libllama.*, OR a cmake build dir whose
# libllama lives under bin/ with headers in LLAMA_CPP_SOURCE).
#
# BUILD IT TWICE. `frink parity` takes `--dumper` more than once, and
# the second one is not a nicety: with a single reference it cannot
# render a WRONG verdict on any K-quant checkpoint at all, because two
# builds of llama.cpp have been measured 3.5e-2 apart in KL from an
# identical graph and no constant can sit above that and still mean
# anything (issue #111). Two dumpers give the run a line it measured
# for itself. LLAMA_LOGITS_OUT names the binary so the second build does
# not overwrite the first:
#
#   bash tools/build_llama_logits.sh                      # target/llama_logits
#   LLAMA_CPP_PREFIX=/tmp/llamabuild \
#     LLAMA_LOGITS_OUT=target/llama_logits_scratch \
#     bash tools/build_llama_logits.sh
#
#   frink parity -m model.gguf \
#     --dumper target/llama_logits --dumper target/llama_logits_scratch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${LLAMA_CPP_PREFIX:-}"
SOURCE="${LLAMA_CPP_SOURCE:-$ROOT/.scratch/llama.cpp}"

if [[ -z "$PREFIX" ]]; then
# THE REFERENCE'S VINTAGE IS PART OF THE RESULT.
#
# Homebrew's `llama.cpp` bottle can be months behind `.scratch/llama.cpp`,
# and a stale reference does not fail loudly -- it disagrees, and the
# disagreement reads as a frink bug. Both known cases were the bottle:
#
#   * bottle 7650 predates `BertNormalizer`'s `strip_accents` switch, so
#     it welds a standalone combining acute into the word and yields
#     `[UNK]` where current llama.cpp, HuggingFace and frink all drop
#     it. That showed up as a WordPiece "divergence" that was not one.
#   * the same bottle cannot LOAD a gemma-4 checkpoint at all, so
#     `frink parity` skipped that model entirely and its tokenizer went
#     unchecked.
#   * Qwen2.5-1.5B Q4_K_M parity used to be reference-dependent: DRIFT
#     against Homebrew (KL ~7.7e-3) but WRONG against a fresh scratch
#     build (KL ~2.7e-2, same top-1). The verdict no longer moves with
#     the bottle -- `parity` measures the two builds against EACH OTHER
#     and judges frink against that, which is why you want both
#     dumpers. But every printed KL still belongs to the reference the
#     report names, so a NUMBER quoted without its libllama is still
#     half an experiment.
#
# Building llama.cpp from `.scratch/llama.cpp` and pointing this script
# at it fixed both: BGE went DIVERGES to MATCH, and gemma-4 joined the
# sweep. If a single model disagrees and the rest match, SUSPECT THE
# REFERENCE before the engine:
#
#   cmake -B /tmp/llamabuild -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
#     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
#     .scratch/llama.cpp
#   cmake --build /tmp/llamabuild --target llama -j8
#   LLAMA_CPP_PREFIX=/tmp/llamabuild bash tools/build_llama_logits.sh

  if command -v brew >/dev/null 2>&1 && brew --prefix llama.cpp >/dev/null 2>&1; then
    PREFIX="$(brew --prefix llama.cpp)"
  else
    echo "cannot find llama.cpp. Install it (brew install llama.cpp) or set" >&2
    echo "LLAMA_CPP_PREFIX to a directory holding include/llama.h and lib/libllama.*" >&2
    exit 1
  fi
fi

# A relative LLAMA_LOGITS_OUT is taken against the repo root, so the
# same command works from anywhere; an absolute one is used as given.
OUT="${LLAMA_LOGITS_OUT:-target/llama_logits}"
case "$OUT" in
  /*) ;;
  *) OUT="$ROOT/$OUT" ;;
esac
mkdir -p "$(dirname "$OUT")"

# The library goes AFTER the source: GNU ld resolves left to right and
# drops a -l nothing before it needs yet (macOS's linker does not care).
#
# Homebrew-style layout: PREFIX/include + PREFIX/lib
if [[ -f "$PREFIX/include/llama.h" ]]; then
  cc -std=c11 -O2 -Wall -Wextra \
    -I"$PREFIX/include" \
    "$ROOT/tools/llama_logits.c" -o "$OUT" \
    -L"$PREFIX/lib" -lllama -Wl,-rpath,"$PREFIX/lib"
  echo "built $OUT (llama.cpp at $PREFIX)"
# CMake build layout: libllama in PREFIX/bin, headers in SOURCE tree
elif [[ -f "$SOURCE/include/llama.h" ]] && ls "$PREFIX"/bin/libllama.* >/dev/null 2>&1; then
  cc -std=c11 -O2 -Wall -Wextra \
    -I"$SOURCE/include" \
    -I"$SOURCE/ggml/include" \
    "$ROOT/tools/llama_logits.c" -o "$OUT" \
    -L"$PREFIX/bin" -lllama -Wl,-rpath,"$PREFIX/bin"
  echo "built $OUT (libllama at $PREFIX/bin, headers at $SOURCE)"
else
  echo "no llama.h under $PREFIX/include and no cmake layout at $PREFIX/bin + $SOURCE/include" >&2
  echo "Set LLAMA_CPP_PREFIX to the cmake build dir and LLAMA_CPP_SOURCE to the llama.cpp checkout." >&2
  exit 1
fi

echo
echo "It writes into target/, so 'cargo clean' removes it — rebuild with this"
echo "script rather than assuming frink parity lost its reference."
echo "Build it a second time against another libllama (LLAMA_CPP_PREFIX=… "
echo "LLAMA_LOGITS_OUT=…) and pass both to 'frink parity --dumper': with one"
echo "reference no K-quant checkpoint can be called WRONG (issue #111)."
