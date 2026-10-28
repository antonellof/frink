#!/usr/bin/env bash
# Executes the one-row attention CUDA C (`frink_cuda::decode_attn`) on
# this host and compares it bit for bit against the Rust scalar twin.
# No GPU involved; see ../mul_mm_host_check/shim.h for what this does
# and does not establish.
#
#   crates/frink-cuda/tools/decode_attn_host_check/run.sh
#
# Needs a C++14 compiler. `-ffp-contract=off` is required: contraction
# fuses `acc * a + b * v` into an FMA and the result stops being
# bit-comparable to the twin. A real GPU does contract, which is why the
# on-device test compares with a tolerance instead.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
shim="$here/../mul_mm_host_check"
cxx="${CXX:-c++}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT

cargo run -q --manifest-path "$root/Cargo.toml" -p frink-cuda \
    --example decode_attn_host_check -- "$out"

{ echo '#include "shim.h"'; cat "$out/decode_attn.cu"; cat "$here/main.cpp"; } > "$out/tu.cpp"
"$cxx" -std=c++14 -O1 -ffp-contract=off -fno-strict-aliasing -Wall -Wextra \
    -Wno-unused-function -I"$shim" "$out/tu.cpp" -o "$out/run"

status=0
while read -r tag n_heads n_kv hd lo hi chunks scale cap; do
    "$out/run" "$out/$tag" "$n_heads" "$n_kv" "$hd" "$lo" "$hi" "$chunks" "$scale" "$cap" \
        || status=1
done < "$out/manifest.txt"

if [ "$status" -eq 0 ]; then
    echo "OK: the one-row attention kernels match their scalar twin bit for bit"
else
    echo "FAIL: the one-row attention CUDA C diverges from its scalar twin" >&2
fi
exit "$status"
