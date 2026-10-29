#!/usr/bin/env bash
# Executes the coalesced matvec CUDA C, at both widths, on this host
# and compares it bit for bit against the Rust scalar twins. No GPU
# involved; see ../mul_mm_host_check/shim.h for what this does and does
# not establish.
#
#   crates/frink-cuda/tools/matvec_host_check/run.sh
#
# Needs a C++14 compiler. Every row of
# `frink_cuda::matvec_kinds::coalesced::COALESCED` is covered at every
# width automatically; adding a kind needs no edit here.
#
# `-ffp-contract=off` is required: contraction fuses `acc += a * b`
# into an FMA and the result stops being bit-comparable to the twin. A
# real GPU does contract, which is why the on-device test compares with
# a relative tolerance instead. `-fno-strict-aliasing` because the
# kernels read byte buffers through `uint4` / `unsigned short` /
# `float4` pointers, which CUDA defines and ISO C++ does not.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
shim="$here/../mul_mm_host_check"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example matvec_host_check -- "$out"

status=0
while read -r unit fn tag rows row_bytes n_blocks warps; do
    bin="$out/run_$unit"
    if [ ! -x "$bin" ]; then
        { echo '#include "shim.h"'; cat "$out/$unit.cu"; cat "$here/main.cpp"; } > "$out/tu_$unit.cpp"
        "$cxx" -std=c++14 -O1 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra \
            -Wno-unused-function -I"$shim" -DFX_KERNEL="$fn" \
            "$out/tu_$unit.cpp" -o "$bin"
    fi
    "$bin" "$out/$tag" "$rows" "$row_bytes" "$n_blocks" "$warps" || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: every coalesced matvec kernel matches its scalar twin bit for bit at both widths"
else
    echo "FAIL: the coalesced matvec CUDA C diverges from its scalar twin" >&2
fi
exit "$status"
