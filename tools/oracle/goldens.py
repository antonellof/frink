#!/usr/bin/env python3
"""Regenerates every checked-in llama.cpp golden from libllama, in place.

    tools/oracle/goldens.py <work-dir> [--tol T] [--summary FILE]

<work-dir> is what `tools/oracle/llama_build.sh` built (its `ref_logits`
and `llama_logits`). `tools/oracle/goldens.tsv` says which dumper
invocation answers each golden. A golden whose fresh answer differs
from the checked-in one by more than `--tol` (max abs) is REWRITTEN,
so `git diff` afterwards is the list of goldens that are not libllama's
current answer. Exit status is 1 only when a dumper could not answer
at all; a rewrite is left for the caller's diff to report.

Why a tolerance and not bit equality: the checked-in goldens are CI's
answer (the portable, fixed-ISA libllama of tools/oracle/llama_build.sh
on a GitHub ubuntu runner), and the tolerance leaves room for a printed
literal's rounding. The default is a tenth of `GRAPH_TOL` (3e-5), the
line every graph test holds frink to: a golden within it of libllama's
answer cannot move any of those verdicts. Another host's libllama
answers differently in the last bits (up to ~4e-4 here), so a local run
passes a looser `--tol` and never commits its own rewrite.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, "crates/frink-models/tests")
FIXTURES = os.path.join(TESTS, "fixtures")
MANIFEST = os.path.join(ROOT, "tools/oracle/goldens.tsv")

GRAPH_PROMPT = "3 7 11 19 23 5"


def llama4_long_prompt():
    # `long_prompt()` in tests/llama4_graphs.rs.
    return " ".join(str(3 + (i * 7919 + 13) % 45) for i in range(8200))


MACROS = {"@GRAPH": lambda: GRAPH_PROMPT, "@LLAMA4_LONG": llama4_long_prompt}


def expand(arg):
    if arg in MACROS:
        return MACROS[arg]().split()
    return [arg.replace("$F", FIXTURES)]


def read_manifest():
    rows = []
    with open(MANIFEST) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = shlex.split(line)
            rows.append((n, words))
    return rows


def run(work, words):
    """words: <kind> [ENV=V ...] <dumper args...>. Returns stdout."""
    kind, rest = words[0], words[1:]
    env = dict(os.environ)
    while rest and re.match(r"^[A-Z_]+=", rest[0]):
        k, v = rest.pop(0).split("=", 1)
        env[k] = v
    args = [a for w in rest for a in expand(w)]
    if kind == "logits" or kind == "decode":
        cmd = [os.path.join(work, "ref_logits")] + args
    elif kind == "embed":
        cmd = [os.path.join(work, "llama_logits"), "--embed"] + args
    else:
        raise SystemExit(f"{MANIFEST}: unknown dumper kind {kind!r}")
    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        return None, out.stderr.strip().splitlines()[-1:] or ["exit %d" % out.returncode]
    return out.stdout, None


def f32_literal(v):
    import numpy as np

    x = np.float32(v)
    a = abs(float(x))
    if a != 0.0 and (a < 1e-4 or a >= 1e16):
        return np.format_float_scientific(x, unique=True, trim="-", exp_digits=1).replace("e+", "e")
    s = np.format_float_positional(x, unique=True, trim="-")
    return s if "." in s else s + ".0"


ARRAY = r"(const {name}: \[f32; (\d+)\] = \[)([^\]]*)(\];)"


def rust_golden(path, name):
    with open(path) as f:
        text = f.read()
    m = re.search(ARRAY.format(name=re.escape(name)), text)
    if not m:
        raise SystemExit(f"{path}: no `const {name}: [f32; N]`")
    values = [float(v) for v in re.findall(r"-?[0-9][0-9.eE+-]*", m.group(3))]
    return text, m, values


def rewrite_rust(path, text, m, fresh):
    body = m.group(3)
    if "\n" in body:
        inner = "\n" + "".join(f"    {f32_literal(v)},\n" for v in fresh)
    else:
        inner = ", ".join(f32_literal(v) for v in fresh)
    with open(path, "w") as f:
        f.write(text[: m.start(3)] + inner + text[m.end(3) :])


def parse_decode(text):
    tokens, logits = None, []
    for line in text.splitlines():
        if line.startswith("tokens"):
            tokens = line.split()[1:]
        elif line.startswith("logits"):
            logits.extend(float(v) for v in line.split()[1:])
    return tokens, logits


def check(work, words, tol):
    """Returns (label, verdict, max_delta, note)."""
    target, kind = words[0], words[1]
    if kind == "numpy":
        return target, "SKIP", None, " ".join(words[2:])
    out, err = run(work, words[1:])
    if out is None:
        return target, "FAIL", None, "dumper: " + " ".join(err)
    if kind == "decode":
        path = os.path.join(TESTS, target)
        with open(path) as f:
            old = f.read()
        t0, l0 = parse_decode(old)
        t1, l1 = parse_decode(out)
        delta = max(abs(a - b) for a, b in zip(l0, l1)) if len(l0) == len(l1) else float("inf")
        if t0 != t1 or delta > tol:
            with open(path, "w") as f:
                f.write(out)
            return target, "REWRITTEN", delta, "tokens differ" if t0 != t1 else ""
        return target, "SAME", delta, ""
    rs, name = target.split(":")
    path = os.path.join(TESTS, rs)
    text, m, old = rust_golden(path, name)
    fresh = [float(v) for v in out.split()]
    if name.endswith("_HEAD"):
        # A golden that pins only the leading values of a longer vector.
        fresh = fresh[: len(old)]
    if len(fresh) != len(old):
        return target, "FAIL", None, f"dumper printed {len(fresh)} values, golden has {len(old)}"
    delta = max(abs(a - b) for a, b in zip(old, fresh))
    if delta > tol:
        rewrite_rust(path, text, m, fresh)
        return target, "REWRITTEN", delta, ""
    return target, "SAME", delta, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--tol", type=float, default=3e-6)
    ap.add_argument("--summary")
    a = ap.parse_args()
    lines = []
    bad = failed = 0
    for _, words in read_manifest():
        label, verdict, delta, note = check(a.work, words, a.tol)
        d = "-" if delta is None else f"{delta:.2e}"
        lines.append(f"| `{label}` | {verdict} | {d} | {note} |")
        bad += verdict in ("REWRITTEN", "FAIL")
        failed += verdict == "FAIL"
        print(f"golden {verdict:9} {d:>9}  {label}  {note}", flush=True)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(f"### Goldens ({bad} not libllama's answer at tol {a.tol:g})\n\n")
            f.write("| golden | verdict | max abs delta | note |\n|---|---|---|---|\n")
            f.write("\n".join(lines) + "\n\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
