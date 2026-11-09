#!/usr/bin/env bash
# Executes the tensor-core prefill attention body
# (`frink_cuda::prefill_attn_tc`) on this host, with the four PTX
# helpers replaced by CPU stand-ins written from the PTX ISA
# (tc_shim.h), and compares it bit for bit against the Rust twin. No
# GPU is involved. ../mul_mm_host_check/shim.h and tc_shim.h say what
# this does and does not establish.
#
#   crates/frink-cuda/tools/prefill_attn_tc_host_check/run.sh
#
# Needs a C++14 compiler with `_Float16` (clang on arm64 or x86-64, GCC
# 12 and up). `-ffp-contract=off` is required: contraction fuses
# `acc * a + b` into an FMA and the result stops being bit-comparable
# to the twin. A real GPU does contract and its tensor cores add in
# their own order, which is why the on-device test compares against the
# definition within the f16 bound instead.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
shim="$here/../mul_mm_host_check"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example prefill_attn_tc_host_check -- "$out"

flags=(-std=c++14 -O1 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra -Wno-unused-function
    -I"$here" -I"$shim")

# The stand-ins first, on a body that ran on hardware: if they are not
# the instructions, nothing below means anything.
read -r gfn g_rows g_cols g_batch g_row_bytes < "$out/gemm.txt"
{ echo '#include "tc_shim.h"'; cat "$out/gemm.cu"; cat "$here/gemm.cpp"; } > "$out/tu_gemm.cpp"
"$cxx" "${flags[@]}" -DFX_KERNEL="$gfn" "$out/tu_gemm.cpp" -o "$out/run_gemm"
if ! "$out/run_gemm" "$out/gemm" "$g_rows" "$g_cols" "$g_batch" "$g_row_bytes"; then
    echo "FAIL: the PTX stand-ins do not reproduce the hardware-proven GEMM body" >&2
    exit 1
fi

status=0
while read -r tag hd fn n_q n_heads n_kv start window qscale cap group gx gy smem; do
    bin="$out/run_d$hd"
    if [ ! -x "$bin" ]; then
        { echo '#include "tc_shim.h"'; cat "$out/d$hd.cu"; cat "$here/main.cpp"; } > "$out/tu_d$hd.cpp"
        "$cxx" "${flags[@]}" -DFX_KERNEL="$fn" "$out/tu_d$hd.cpp" -o "$bin"
    fi
    "$bin" "$out/$tag" "$n_q" "$n_heads" "$n_kv" "$start" "$window" "$qscale" "$cap" \
        "$group" "$gx" "$gy" "$smem" || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: the tensor-core prefill attention matches its scalar twin bit for bit"
else
    echo "FAIL: the tensor-core prefill attention CUDA C diverges from its scalar twin" >&2
fi
exit "$status"
