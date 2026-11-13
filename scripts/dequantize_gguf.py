#!/usr/bin/env python3
"""Rewrite a quantized GGUF as ALL_F32, tensor by tensor, metadata copied.

The arbiter for a `frink parity` WRONG on a quantized file. `parity`
measures the distance between two points and cannot say which moved:
llama.cpp quantizes ACTIVATIONS to 8 bits for its quantized matmuls and
frink keeps them in f32, so on a graph that amplifies that loss the
two disagree while frink is the closer of the two to the f32 answer
(`1ebda4fc:docs/plans/llama-cpp-gap-inventory.md` section 10). Running both
engines on the dequantized file removes the quantized matmuls from both
sides; then

    KL(llama_f32 || frink_f32)   is the graph
    KL(llama_f32 || llama_q)      is llama.cpp's own quantization loss
    KL(llama_f32 || frink_q)     is frink's

which `frink parity --dump-logits PREFIX` on each file and a few lines
of numpy give. Measured first on PLM-1.8B-Instruct Q8_0 (2026-09-12):
parity 3.54e-2 WRONG; against the dequantized file the graph is 4.5e-5,
frink's Q8_0 loss 4.5e-5 (2.8e-9 from its own f32), llama.cpp's 3.7e-2.
The whole verdict was the reference's quantization.

The output is `4 * parameters` bytes (PLM-1.8B: 6.8 GiB); write it to
scratch, not to `models/`.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/dequantize_gguf.py IN.gguf OUT_f32.gguf
"""

import sys

import numpy as np

import gguf
from gguf.quants import dequantize

SKIP = {
    "general.architecture",
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.file_type",
}

SCALAR_WRITERS = {
    gguf.GGUFValueType.UINT8: "add_uint8",
    gguf.GGUFValueType.INT8: "add_int8",
    gguf.GGUFValueType.UINT16: "add_uint16",
    gguf.GGUFValueType.INT16: "add_int16",
    gguf.GGUFValueType.UINT32: "add_uint32",
    gguf.GGUFValueType.INT32: "add_int32",
    gguf.GGUFValueType.FLOAT32: "add_float32",
    gguf.GGUFValueType.BOOL: "add_bool",
    gguf.GGUFValueType.UINT64: "add_uint64",
    gguf.GGUFValueType.INT64: "add_int64",
    gguf.GGUFValueType.FLOAT64: "add_float64",
}


def main(src: str, dst: str) -> None:
    r = gguf.GGUFReader(src)
    arch_field = r.fields["general.architecture"]
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode()
    w = gguf.GGUFWriter(dst, arch)
    for key, f in r.fields.items():
        if key in SKIP:
            continue
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            if f.types[1] == gguf.GGUFValueType.STRING:
                w.add_array(key, [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data])
            else:
                w.add_array(key, [f.parts[i].tolist()[0] for i in f.data])
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(key, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"))
        else:
            getattr(w, SCALAR_WRITERS[t])(key, f.parts[f.data[0]].tolist()[0])
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    for t in r.tensors:
        shape = tuple(int(x) for x in reversed(t.shape.tolist()))
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = np.asarray(t.data, dtype=np.float32).reshape(shape)
        else:
            arr = dequantize(np.asarray(t.data), t.tensor_type).reshape(shape).astype(np.float32)
        w.add_tensor(t.name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
