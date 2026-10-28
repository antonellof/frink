#!/usr/bin/env python3
"""Runs `frink parity` on every synthetic fixture and holds each to the
outcome `tools/oracle/fixtures.tsv` expects for it.

    tools/oracle/parity.py <frink> <llama_logits> [--summary FILE] [--observe]

Each fixture is compared on the token ids every golden uses
(`GRAPH_PROMPT`), first-token logits and a 32-step greedy decode. The
tokenizer half is skipped by `--tokens`: libllama's tokenizer cannot
load the fixtures' synthetic vocabularies, and the real-checkpoint
tokenizer sweep is where that half is held (`docs/MODELS.md`).

An outcome is one of:
  match       prefill MATCH and decode MATCH
  refused     frink refuses the file by name (the refusal is the row)
  ref-fails   libllama cannot load or run the file
  both-fail   neither engine runs it (a vocab or adapter file)
  timeout     no answer within a minute (a hang, killed with its group)
  skip        not run: the row's comment says why
  <verdict>   anything else, e.g. `prefill=DRIFT,decode=MATCH`

A fixture whose observed outcome differs from its expected one fails
the run, in either direction: a `refused` row that starts matching is
as much news as a `match` row that stops. `--observe` prints the
observed column instead, for writing the table the first time.
"""

import argparse
import os
import re
import signal
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES = os.path.join(ROOT, "crates/frink-models/tests/fixtures")
MANIFEST = os.path.join(ROOT, "tools/oracle/fixtures.tsv")
GRAPH_PROMPT = "3,7,11,19,23,5"
STEPS = "32"


def expectations():
    rows = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                name, expect = line.split()[:2]
                rows[name] = expect
    return rows


def observe(frink, dumper, path):
    cmd = [frink, "parity", "-m", path, "--dumper", dumper, "--tokens", GRAPH_PROMPT,
           "--decode-steps", STEPS]
    # The goldens' configuration on both sides: f32 K/V (FRINK_CTK is
    # read by frink AND the dumper) and llama.cpp off its flash-attention
    # path, which accumulates V in f16. Either alone moves llama.cpp's
    # own answer on these fixtures by more than the MATCH line.
    env = dict(os.environ, FRINK_METAL="0", FRINK_CUDA="0", FRINK_CTK="f32",
               LLAMA_LOGITS_FLASH_ATTN="0")
    # Its own process group, so a reference that never returns (libllama
    # spins in `ggml_compute_forward_rope_flt` on `plamo2_tiny` at
    # 604cc430) is killed with frink rather than orphaned. Every fixture
    # finishes in seconds; a minute is a hang, not a slow file.
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        return "timeout", "no answer within 60 s"

    prefill = re.search(r"^parity \S+: (\S+)", out, re.M)
    decode = re.search(r"^decode: (\S+)", out, re.M)
    if prefill and decode and prefill.group(1) == "MATCH" and decode.group(1) == "MATCH":
        return "match", ""
    if prefill:
        return f"prefill={prefill.group(1)},decode={decode.group(1) if decode else 'none'}", ""
    last = out.strip().splitlines()[-1] if out.strip() else f"exit {p.returncode}"
    ref_failed = "reference dumper failed" in out or "reference decode failed" in out
    frink_failed = "frink prefill" in out or "frink teacher-forced" in out
    if ref_failed:
        # The reference runs second, so frink got through its own load.
        return "ref-fails", last
    if frink_failed:
        try:
            probe = subprocess.run([dumper, path, os.devnull] + GRAPH_PROMPT.split(","),
                                   capture_output=True, text=True, timeout=60)
            ref_ok = probe.returncode == 0
        except subprocess.TimeoutExpired:
            ref_ok = False
        if not ref_ok:
            return "both-fail", last
        # A refusal BY NAME is coverage; any other frink error where
        # libllama answers is a failure to load a file it should run.
        named = "requires unimplemented feature" in out
        return ("refused" if named else "frink-fails"), last
    return "error", last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frink")
    ap.add_argument("dumper")
    ap.add_argument("--summary")
    ap.add_argument("--observe", action="store_true")
    a = ap.parse_args()
    expect = {} if a.observe else expectations()
    names = sorted(f[:-5] for f in os.listdir(FIXTURES) if f.endswith(".gguf"))
    missing = sorted(set(names) - set(expect)) if not a.observe else []
    stale = sorted(set(expect) - set(names))
    rows, bad = [], 0
    for name in names:
        if expect.get(name) == "skip":
            rows.append(f"| `{name}` | skip | skip | |")
            print(f"fixture skip {name}", flush=True)
            continue
        got, why = observe(a.frink, a.dumper, os.path.join(FIXTURES, name + ".gguf"))
        if a.observe:
            print(f"{name}\t{got}\t# {why}" if why else f"{name}\t{got}", flush=True)
            continue
        want = expect.get(name, "(not listed)")
        ok = got == want
        bad += not ok
        rows.append(f"| `{name}` | {got} | {want} | {'' if ok else 'DIFFERS'} {why} |")
        print(f"fixture {'ok  ' if ok else 'FAIL'} {name}: {got} (expected {want}) {why}", flush=True)
    for name in stale:
        bad += 1
        print(f"fixture FAIL {name}: listed in {MANIFEST} but no such fixture")
    if a.summary and not a.observe:
        with open(a.summary, "a") as f:
            f.write(f"### Fixtures through `frink parity` ({bad} not as expected)\n\n")
            f.write("| fixture | observed | expected | |\n|---|---|---|---|\n")
            f.write("\n".join(rows) + "\n\n")
    if missing:
        print(f"{len(missing)} fixture(s) not listed in {MANIFEST}: {' '.join(missing)}")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
