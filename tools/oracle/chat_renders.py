#!/usr/bin/env python3
"""libllama's prompt for each (chat template, request), the answer
frink-server's render is held to.

  tools/oracle/chat_renders.py <chat_render> <templates-dir> <out-dir> [--all]

For each template, copies <templates-dir>/<name>.jinja to
<out-dir>/<name>.jinja and writes <out-dir>/<name>/<shape>.prompt with
the bytes llama-server would render, or <shape>.error with llama.cpp's
refusal. The shapes, and the bos/eos every render uses, are in
crates/frink-server/tests/llama_renders/shapes.json, which frink's side
reads too, so the two cannot drift apart.

Without --all the templates are the ones already in <out-dir>: the
checked-in set (crates/frink-server/tests/llama_renders), regenerated
from the pin by scripts/oracle.sh, which fails if a file moved. With
--all it is every template in <templates-dir>, the census that
`llama_renders::census` reads from FRINK_LLAMA_RENDERS.

A template that prints the date (strftime_now) renders today's, in
local time upstream, so the renders run under TZ=UTC and <out-dir>/
rendered_at holds the Unix time they ran at, which frink's side pins its
clock to. The checked-in set refuses such a template: its answer would
move every day.

<chat_render> is built by tools/oracle/llama_build.sh from
tools/oracle/chat_render.cpp.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

SHAPES = pathlib.Path(__file__).resolve().parents[2] / "crates/frink-server/tests/llama_renders/shapes.json"


def render(binary, template, spec, shape):
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "prompt"
        args = [binary, str(out), spec["bos_token"], spec["eos_token"],
                "--chat-template-file", str(template), *shape["flags"]]
        env = dict(os.environ, TZ="UTC")
        run = subprocess.run(args, input=json.dumps(shape["body"]).encode(), capture_output=True, env=env)
        if run.returncode == 0:
            return out.read_bytes(), None
        lines = [l for l in run.stderr.decode(errors="replace").splitlines() if l.startswith("chat_render:")]
        return None, (lines[-1] if lines else run.stderr.decode(errors="replace").strip()[-400:]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chat_render")
    ap.add_argument("templates", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SHAPES.read_text())
    if args.all:
        names = sorted(p.stem for p in args.templates.glob("*.jinja"))
    else:
        names = sorted(p.stem for p in args.out.glob("*.jinja"))
    if not names:
        sys.exit(f"no templates to render in {args.templates if args.all else args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.all:
        (args.out / "rendered_at").write_text(f"{int(time.time())}\n")
    refused = 0
    for name in names:
        source = args.templates / f"{name}.jinja"
        if not source.exists():
            sys.exit(f"{name}.jinja is not in {args.templates}: llama.cpp no longer bundles it")
        if not args.all and "strftime_now" in source.read_text():
            sys.exit(f"{name}.jinja prints today's date, so its render cannot be checked in")
        shutil.copyfile(source, args.out / f"{name}.jinja")
        rendered = args.out / name
        if rendered.exists():
            shutil.rmtree(rendered)
        rendered.mkdir()
        for shape in spec["shapes"]:
            prompt, error = render(args.chat_render, source, spec, shape)
            if prompt is not None:
                (rendered / f"{shape['name']}.prompt").write_bytes(prompt)
            else:
                refused += 1
                (rendered / f"{shape['name']}.error").write_text(error)
    print(f"{len(names)} templates x {len(spec['shapes'])} requests rendered by libllama "
          f"into {args.out}, {refused} refused")


if __name__ == "__main__":
    main()
