#!/usr/bin/env python3
"""Fails if any fixture holds a tensor type the one-libllama oracle cannot judge.

    tools/oracle/check_quant.py   (with llama.cpp's gguf-py on PYTHONPATH)

`tools/oracle/llama_build.sh` builds ONE libllama, where
`tools/build_llama_logits.sh` asks for two. The second build exists to
measure how far two libllama builds disagree on a K-quant file, where
llama.cpp quantizes the activation to Q8_K and builds differ by up to
3.5e-2 in KL (#111). On F32, F16, BF16 and Q8_0 two builds agree, so one
reference is the whole experiment. This check is what keeps that true:
a fixture in any other type stops the run instead of being judged
against a line nobody measured.
"""

import glob
import os
import sys

from gguf import GGUFReader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALLOWED = {"F32", "F16", "BF16", "Q8_0"}


def main():
    bad = []
    for path in sorted(glob.glob(os.path.join(ROOT, "crates/frink-models/tests/fixtures/*.gguf"))):
        kinds = {t.tensor_type.name for t in GGUFReader(path).tensors}
        if kinds - ALLOWED:
            bad.append(f"{os.path.basename(path)}: {sorted(kinds - ALLOWED)}")
    for b in bad:
        print(f"fixture type outside {sorted(ALLOWED)}: {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
