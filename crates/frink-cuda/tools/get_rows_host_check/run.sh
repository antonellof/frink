#!/usr/bin/env bash
# Executes the get_rows (embedding gather) CUDA C on this host and
# compares it bit for bit against the Rust scalar twin
# (`frink_cuda::get_rows::get_rows_reference`). No GPU involved; see
# ../mul_mm_host_check/shim.h for what this does and does not establish.
#
#   crates/frink-cuda/tools/get_rows_host_check/run.sh
#
# Needs a C++14 compiler. Every kind in `frink_cuda::mul_mm::KINDS` is
# covered automatically: the gather is built from each kind's dequant.
# `-ffp-contract=off` so a contracted FMA cannot separate the kernel
# from the twin; a gather has no accumulation, so on a GPU the result
# is expected to be the same bits too.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
shim="$here/../mul_mm_host_check"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example get_rows_host_check -- "$out"

status=0
while read -r kind fn tag n_ids n_cols row_bytes threads; do
    bin="$out/run_$kind"
    if [ ! -x "$bin" ]; then
        { echo '#include "shim.h"'; cat "$out/$kind.cu"; cat "$here/main.cpp"; } > "$out/tu_$kind.cpp"
        "$cxx" -std=c++14 -O1 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra \
            -Wno-unused-function -Wno-unused-variable -I"$shim" -DFX_KERNEL="$fn" \
            "$out/tu_$kind.cpp" -o "$bin"
    fi
    "$bin" "$out/$tag" "$n_ids" "$n_cols" "$row_bytes" "$threads" || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: every get_rows kernel matches its scalar twin bit for bit"
else
    echo "FAIL: the get_rows CUDA C diverges from its scalar twin" >&2
fi
exit "$status"
