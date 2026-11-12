#!/usr/bin/env bash
# Compiles every shader here to SPIR-V under spv/. The .spv files are
# COMMITTED, so a frink build needs no shader compiler; this script and
# the `shaders::tests::committed_spirv_is_what_the_source_compiles_to`
# test (which runs it into a temp dir and compares) keep the two from
# drifting. Needs glslangValidator (Homebrew/apt `glslang`).
set -euo pipefail
cd "$(dirname "$0")"
out="${1:-spv}"
mkdir -p "$out"
for src in *.comp; do
  glslangValidator -V --target-env vulkan1.0 -Os -o "$out/${src%.comp}.spv" "$src" >/dev/null
done
# Every matvec again as its embedding gather (matvec_common.glsl).
for src in *_matvec.comp; do
  glslangValidator -V --target-env vulkan1.0 -Os -DGATHER \
    -o "$out/${src%_matvec.comp}_gather.spv" "$src" >/dev/null
done
