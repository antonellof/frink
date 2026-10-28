#!/usr/bin/env python3
"""Extract llama.cpp's command-line flags into crates/frink-cli/tests/llama_cpp_flags.txt.

ADR 0005 says frink accepts llama.cpp's flags with the same names and
meanings, and that the gap is a number a test pins. This is the half
that reads llama.cpp: every `common_arg(...)` in `common/arg.cpp`
(spellings, negated spellings, `set_examples`, `set_excludes`,
`set_env`), plus `llama-bench`'s hand-written parser, which does not go
through `common_arg` at all.

Committed output rather than a parse at test time, for the same reason
`llama_cpp_arch_names.txt` is: CI has no llama.cpp checkout.

    python3 scripts/llama_cpp_flags.py .scratch/llama.cpp-604cc430 \
        --commit 604cc430239c76ff636c443e3400bf552b750421 \
        > crates/frink-cli/tests/llama_cpp_flags.txt

The tree is read as plain files, so an exported checkout with no `.git`
works; `--commit` is the full SHA it was exported at, and goes in the
header. `--rev REV` instead reads the files out of a git checkout at
REV (and takes the SHA from it).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# llama.cpp example -> the frink tool column it lands in. ADR 0005's
# table: llama-cli is `frink run`, llama-server is `frink-server`.
# llama-bench is not here: it has its own parser (see bench_rows).
TOOLS = {"LLAMA_EXAMPLE_CLI": "cli", "LLAMA_EXAMPLE_SERVER": "server"}


def read(root: Path, rel: str, rev: str | None) -> str:
    if rev is None:
        return (root / rel).read_text()
    return subprocess.run(
        ["git", "-C", str(root), "show", f"{rev}:{rel}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def skip_literal(src: str, i: int) -> int:
    """Index just past the string or char literal starting at src[i]."""
    quote = src[i]
    i += 1
    while src[i] != quote:
        i += 2 if src[i] == "\\" else 1
    return i + 1


def matching_paren(src: str, open_at: int) -> int:
    """Index of the `)` closing the `(` at `open_at`, skipping literals and comments."""
    depth = 0
    i = open_at
    while i < len(src):
        c = src[i]
        if c in "\"'":
            i = skip_literal(src, i)
            continue
        if src.startswith("//", i):
            i = src.index("\n", i)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced paren at {open_at}")


STRING_LIST = re.compile(r'\s*\{\s*((?:"[^"]*"\s*,?\s*)+)\}')


def string_list(text: str, at: int) -> tuple[list[str], int] | None:
    # A line comment may sit between `common_arg(` and its spellings.
    while True:
        stripped = len(text) - len(text[at:].lstrip())
        if text.startswith("//", stripped):
            at = text.index("\n", stripped)
        else:
            break
    m = STRING_LIST.match(text, at)
    if not m:
        return None
    return re.findall(r'"([^"]*)"', m.group(1)), m.end()


def common_rows(src: str) -> list[dict]:
    rows: list[dict] = []
    for m in re.finditer(r"add_opt\(common_arg\(", src):
        # Skip commented-out declarations.
        line_start = src.rfind("\n", 0, m.start()) + 1
        if src[line_start : m.start()].lstrip().startswith("//"):
            continue
        outer_open = m.start() + len("add_opt")
        body = src[outer_open : matching_paren(src, outer_open) + 1]
        inner = body.index("common_arg(") + len("common_arg(")
        first = string_list(body, inner)
        if first is None:
            raise ValueError(f"no spellings near offset {m.start()}")
        spellings, after = first
        neg: list[str] = []
        rest = body[after:]
        if rest.lstrip().startswith(","):
            second = string_list(rest, rest.index(",") + 1)
            if second is not None:
                neg = second[0]
        if ".set_preset_only()" in body:
            continue

        def enum_list(method: str) -> list[str]:
            mm = re.search(rf"\.{method}\(\s*\{{([^}}]*)\}}", body)
            return re.findall(r"LLAMA_EXAMPLE_[A-Z_]+", mm.group(1)) if mm else []

        examples = enum_list("set_examples") or ["LLAMA_EXAMPLE_COMMON"]
        excludes = enum_list("set_excludes")
        env = re.search(r'\.set_env\("([^"]+)"\)', body)
        tools = []
        for ex, tool in TOOLS.items():
            # `add_opt`'s rule: in the example, or COMMON and not excluded.
            if (ex in examples or "LLAMA_EXAMPLE_COMMON" in examples) and ex not in excludes:
                tools.append(tool)
        if not tools:
            continue
        rows.append(
            {
                "tools": set(tools),
                "spellings": spellings,
                "neg": neg,
                "env": env.group(1) if env else None,
            }
        )
    return merge(rows)


def merge(rows: list[dict]) -> list[dict]:
    """One row per spelling set. `-np` is declared twice in arg.cpp (a
    server variant and a common one), and is one flag to a user."""
    out: dict[tuple, dict] = {}
    for r in rows:
        key = tuple(r["spellings"])
        if key in out:
            prev = out[key]
            prev["tools"] |= r["tools"]
            if prev["env"] != r["env"] or prev["neg"] != r["neg"]:
                raise ValueError(f"{key}: two declarations disagree")
        else:
            out[key] = r
    return list(out.values())


def bench_rows(src: str) -> list[dict]:
    body = src[src.index("static cmd_params parse_cmd_params") :]
    body = body[: body.index("\n}\n")]
    rows = []
    for cond in re.finditer(r"(?:if|else if) \((.*)\) \{\n", body):
        spellings = re.findall(r'arg == "([^"]+)"', cond.group(1))
        if not spellings or spellings == ["-h", "--help"]:
            continue
        rows.append({"tools": {"bench"}, "spellings": spellings, "neg": [], "env": None})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("llama_cpp", type=Path)
    ap.add_argument("--rev", default=None)
    ap.add_argument("--commit", default=None)
    a = ap.parse_args()
    rev = a.rev
    if (rev is None) == (a.commit is None):
        ap.error("pass exactly one of --rev (a git checkout) or --commit (a plain tree)")
    commit = a.commit or subprocess.run(
        ["git", "-C", str(a.llama_cpp), "rev-parse", f"{rev}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        ap.error(f"{commit!r} is not a full 40-character SHA")
    rows = common_rows(read(a.llama_cpp, "common/arg.cpp", rev))
    rows += bench_rows(read(a.llama_cpp, "tools/llama-bench/llama-bench.cpp", rev))
    counts = {t: sum(t in r["tools"] for r in rows) for t in ("cli", "server", "bench")}
    print("# llama.cpp command-line flags, pinned.")
    print("#")
    print(f"# Source: common/arg.cpp and tools/llama-bench/llama-bench.cpp at {commit}")
    print("# (the pin docs/plans/parity-audit-2026-10-28.md records).")
    print("# Regenerate with scripts/llama_cpp_flags.py; the command is in its docstring.")
    print("#")
    print("# Committed rather than derived at test time because CI has no llama.cpp")
    print("# checkout. Read by crates/frink-cli/src/llama_flags.rs, which classifies")
    print("# every row against frink's own clap commands.")
    print("#")
    print("# Columns: tools (cli = llama-cli, server = llama-server, bench = llama-bench),")
    print("# spellings, negated spellings, env var. `-` is empty.")
    print("#")
    print(f"# Count: cli {counts['cli']}, server {counts['server']}, bench {counts['bench']}")
    for r in rows:
        tools = ",".join(t for t in ("cli", "server", "bench") if t in r["tools"])
        print(
            "\t".join(
                [
                    tools,
                    ",".join(r["spellings"]),
                    ",".join(r["neg"]) or "-",
                    r["env"] or "-",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
