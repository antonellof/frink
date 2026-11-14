#!/usr/bin/env python3
"""Split-level oracle for the BPE pre-tokenizers (#624, #629).

For every `tokenizer.ggml.pre` that `llama-vocab.cpp` at the pin maps
onto a regex list, cut a corpus with libllama's own
`unicode_regex_split` (tools/oracle/regex_split.cpp) and with frink's
port (the ignored test `pretokenize::oracle::dump_splits`), and report
every text the two cut differently.

The regex lists and the `pre` -> arm mapping are read out of
llama-vocab.cpp mechanically, not copied, so a transcription slip in
frink's table cannot be echoed into the oracle's input.

    python3 tools/oracle/regex_split.py \
        --llama-src ~/.cache/frink/llama-src \
        --llama-bin target/oracle/<pin>/build/bin \
        --work target/regex_split

Exit status is 1 when any arm differs. Arms frink refuses are listed
and skipped. Run it on Linux as well: upstream runs two of its three
engines through std::regex, so libstdc++ is a separate oracle.
"""

import argparse
import os
import random
import re
import subprocess
import sys


def c_strings(line):
    """The C string literals on one source line, decoded, ignoring a
    trailing `//` comment that is outside every literal."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c == "/" and line.startswith("//", i):
            break
        if c != '"':
            i += 1
            continue
        i += 1
        buf = []
        while line[i] != '"':
            if line[i] == "\\":
                e = line[i + 1]
                buf.append({"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}[e])
                i += 2
            else:
                buf.append(line[i])
                i += 1
        i += 1
        out.append("".join(buf))
    return out


def regex_lists(src):
    """`LLAMA_VOCAB_PRE_TYPE_*` (or "DEFAULT") -> regex_exprs, from the
    `llm_tokenizer_bpe` constructor's switch."""
    start = src.index("llm_tokenizer_bpe(const llama_vocab & vocab) {")
    end = src.index("std::vector<std::string> regex_exprs;", start)
    lists, labels, collecting, current = {}, [], False, []
    for line in src[start:end].splitlines():
        s = line.strip()
        m = re.match(r"case LLAMA_VOCAB_PRE_TYPE_(\w+):", s)
        if m:
            labels.append(m.group(1))
            continue
        if s.startswith("default:"):
            labels.append("DEFAULT")
            continue
        if s.startswith("//"):
            continue
        if "regex_exprs = {" in s:
            collecting, current = True, []
            s = s.split("regex_exprs = {", 1)[1]
        if collecting:
            current += c_strings(s)
            if "};" in s:
                collecting = False
                for label in labels:
                    lists[label] = current
        if s.startswith("break;"):
            labels = []
    return lists


def pre_map(src):
    """`tokenizer.ggml.pre` -> (pre type, ignore_merges), from the BPE
    branch of `load`. The empty string is upstream's missing key."""
    start = src.index("if (tokenizer_pre.empty()) {")
    end = src.index('unknown pre-tokenizer type', start)
    out, names = {"": ["DEFAULT", False]}, []
    current = None
    for line in src[start:end].splitlines():
        for name in re.findall(r'tokenizer_pre == "([^"]+)"', line):
            if current is not None:
                names, current = [], None
            names.append(name)
        m = re.search(r"pre_type = LLAMA_VOCAB_PRE_TYPE_(\w+);", line)
        if m and names:
            current = names
            for name in names:
                out[name] = [m.group(1), False]
        if "ignore_merges = true;" in line and current:
            for name in current:
                out[name][1] = True
    return out


def corpus(llama_src, seed=629, n_random=3000):
    texts = [
        "Hello world", "DON'T you dare, I'm sure we'll've", "It's 1234567 o'clock",
        "a\tb c\nd\u000ce   \r\n\tg\t", "   leading and trailing   ",
        "日本語のテキスト和中文!", "漢字ひらがなカタカナ한국어ㄱㄴㄷㄅㄆ", "“quoted”—dash…：；，、。〈〉︰﹏！",
        "<sentinel:12> IMGIMGABCDZ IMGIMGAZ IMGIMGJZ", "x    y  z\t\nw", "ab1234cd ١٢٣٤٥ x²³",
        "Ελληνικά Кириллица עברית العربية हिन्दी।", "été ñ", "zw‌nj zw‍j",
        "CamelCaseWord HTTPServer iPhone", "a" + " " * 600 + "b", " " * 70 + "\n", "12,345.678 $5+3<4>2^1~0|9`",
        "<|im_start|>user<|im_end|>", "path/to/file.rs:10:5", "def f(x):\n    return x**2\n",
        "ＡＢＣ！？ １２３", "😀👍🏽 emoji", "İstanbul ǅemal", "a.\n/b", "'S 'Ll 'Re 'VE",
    ]
    for name in sorted(os.listdir(os.path.join(llama_src, "models"))):
        if name.startswith("ggml-vocab-") and name.endswith(".gguf.inp"):
            with open(os.path.join(llama_src, "models", name), encoding="utf-8") as f:
                for case in f.read().split("\n__ggml_vocab_test__\n"):
                    if case and case not in texts:
                        texts.append(case)
    pools = [
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " \t\n\r",
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", "  　\u0085\u000b\u000c ",
        "一二三中文字漢語龥㐀豈", "あいうえおアイウエオー", "가나다라한국ㄱㄴㅏㅓ", "ㄅㄆㄇㄈ",
        "αβγΩΣКириллaעבר", "عربيةابت", "हिन्दीक", "́̃̈ि्",
        "😀👍🎉", "١٢٣٤٥٦²³¹⁴", "ＡＢｃｄ！？１２", "​‌‍﻿",
        "…“”‘’—：；，、。〈〉︰﹏", "'sStTmMdDlLvVrReE",
    ]
    rng = random.Random(seed)
    for _ in range(n_random):
        k = rng.randint(1, 40)
        texts.append("".join(rng.choice(rng.choice(pools)) for _ in range(k)))
    return texts


def hexs(s):
    return s.encode("utf-8").hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-src", required=True)
    ap.add_argument("--llama-bin", required=True, help="directory holding libllama")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cargo-target-dir", default=None)
    args = ap.parse_args()
    # cargo runs frink's side from the crate directory, so every path
    # handed to it must not depend on this script's working directory.
    for name in ("llama_src", "llama_bin", "work", "cargo_target_dir"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    os.makedirs(args.work, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with open(os.path.join(args.llama_src, "src", "llama-vocab.cpp"), encoding="utf-8") as f:
        src = f.read()
    lists, pres = regex_lists(src), pre_map(src)

    texts = corpus(args.llama_src)
    corpus_path = os.path.join(args.work, "corpus.hex")
    with open(corpus_path, "w") as f:
        f.writelines(hexs(t) + "\n" for t in texts)

    dumper = os.path.join(args.work, "regex_split")
    subprocess.run(
        ["c++", "-std=c++17", "-O2", os.path.join(root, "tools", "oracle", "regex_split.cpp"),
         "-L" + args.llama_bin, "-lllama", "-Wl,-rpath," + args.llama_bin, "-o", dumper],
        check=True)

    # frink's side: every `pre` at once, written as `<pre>.frink`, or
    # `<pre>.refused` holding the reason.
    env = dict(os.environ, FRINK_SPLIT_CORPUS=corpus_path, FRINK_SPLIT_OUT=args.work,
               FRINK_SPLIT_PRES=",".join(sorted(pres)))
    if args.cargo_target_dir:
        env["CARGO_TARGET_DIR"] = args.cargo_target_dir
    subprocess.run(
        ["cargo", "test", "-q", "-p", "frink-models", "--lib", "--",
         "--ignored", "--exact", "tokenizer::pretokenize::oracle::dump_splits"],
        check=True, cwd=root, env=env)

    failed = False
    for pre in sorted(pres):
        ptype, ignore = pres[pre]
        label = pre or "<missing>"
        refused = os.path.join(args.work, f"{pre or '_missing'}.refused")
        if os.path.exists(refused):
            with open(refused) as f:
                print(f"{label:28} {ptype:20} refused: {f.read().strip()[:90]}")
            continue
        pat_path = os.path.join(args.work, f"{pre or '_missing'}.patterns")
        with open(pat_path, "w") as f:
            f.writelines(hexs(p) + "\n" for p in lists[ptype])
        with open(corpus_path) as inp:
            oracle = subprocess.run([dumper, pat_path], stdin=inp, capture_output=True,
                                    text=True, check=True).stdout.splitlines()
        with open(os.path.join(args.work, f"{pre or '_missing'}.frink")) as f:
            frink = f.read().splitlines()
        with open(os.path.join(args.work, f"{pre or '_missing'}.ignore_merges")) as f:
            frink_ignore = f.read().strip() == "true"
        differ = [i for i, (a, b) in enumerate(zip(oracle, frink)) if a != b]
        if len(oracle) != len(frink):
            differ.append(-1)
        chunks = sum(len(line.split()) for line in oracle)
        im = "" if frink_ignore == ignore else f"  ignore_merges frink={frink_ignore} upstream={ignore}"
        print(f"{label:28} {ptype:20} {len(texts)} cases, {chunks:6} chunks, {len(differ)} differ{im}")
        if differ or im:
            failed = True
        for i in differ[:3]:
            if i < 0:
                continue
            show = lambda line: [bytes.fromhex(p).decode("utf-8", "replace") for p in line.split()]
            print(f"    text   {texts[i]!r}\n    oracle {show(oracle[i])}\n    frink  {show(frink[i])}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
