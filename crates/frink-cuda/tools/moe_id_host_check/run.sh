#!/usr/bin/env bash
# Executes the routed-expert CUDA C (`mul_mv_id` and `mul_mm_id`) for
# every kind on this host and compares it bit for bit against the Rust
# twins in `frink_cuda::moe_id`. No GPU involved; see
# ../mul_mm_host_check/shim.h for what this does and does not establish.
#
#   crates/frink-cuda/tools/moe_id_host_check/run.sh
#
# Needs a C++14 compiler. Every kind in `frink_cuda::mul_mm::KINDS` is
# covered automatically. `-ffp-contract=off` for the reason the other
# host checks give.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
shim="$here/../mul_mm_host_check"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example moe_id_host_check -- "$out"

status=0
while read -r unit mode fn tag a1 a2 a3 a4 a5 a6 a7; do
    bin="$out/run_$unit"
    if [ ! -x "$bin" ]; then
        { echo '#include "shim.h"'; cat "$out/$unit.cu"; cat "$here/main.cpp"; } > "$out/tu_$unit.cpp"
        "$cxx" -std=c++14 -O1 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra \
            -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -I"$shim" \
            -DFX_KERNEL="$fn" -DFX_MODE_"$mode" "$out/tu_$unit.cpp" -o "$bin"
    fi
    "$bin" "$out/$tag" "$a1" "$a2" "$a3" "$a4" "$a5" "$a6" "$a7" || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: every mul_mv_id and mul_mm_id kernel matches its scalar twin bit for bit"
else
    echo "FAIL: the routed-expert CUDA C diverges from its scalar twin" >&2
fi
exit "$status"
