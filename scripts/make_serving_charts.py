"""Render the serving-feature charts from the committed receipt.

One source of numbers, three consumers: this script, the prose in
`docs/plans/serving-parity-audit.md`, and the tables in
`benchmarks/RESULTS.md`. A chart drawn from numbers typed in by hand
is two structures that must agree with nothing enforcing it, which is
this repo's dominant bug shape -- so the SVGs are generated and
`documented_serving.rs` checks the prose against the same receipt.

Two files per chart, `-light` and `-dark`, because GitHub strips CSS
media queries out of embedded SVG: the markdown uses <picture> with
`prefers-color-scheme` to pick one.

Palette is the validated categorical pair (checked for CVD separation
and contrast against both surfaces); every series is also labelled, so
identity is never colour alone.

Usage: python3 scripts/make_serving_charts.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECEIPT = ROOT / "benchmarks/receipts/serving/serving_features_m2pro_0.49.0.json"
OUT = ROOT / "docs/assets"

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#17191c", ink2="#4a5159", ink3="#757d86",
                  rule="#dfe1de", rule2="#ecedea", a="#2a78d6", b="#eb6834",
                  none="#9aa1a8"),
    "dark": dict(bg="#1a1a19", ink="#eceeea", ink2="#a8aeb4", ink3="#7b828a",
                 rule="#2e302e", rule2="#232523", a="#3987e5", b="#d95926",
                 none="#6c737a"),
}

FONT = "ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "ui-sans-serif, system-ui, -apple-system, sans-serif"


def head(w, h, t, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{title}">'
        f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>'
    )


def txt(x, y, s, t, size=11, fill=None, anchor="start", font=FONT, weight="400"):
    return (
        f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill or t["ink2"]}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def scaling_chart(d, t):
    pts = d["concurrency_scaling"]["points"]
    w, h = 760, 320
    x0, x1, ytop, ybot = 70, 700, 50, 250
    top = 80.0
    s = [head(w, h, t, "Aggregate throughput against concurrent requests")]
    s.append(txt(x0, 26, "Aggregate throughput, tokens/second", t,
                 size=13, fill=t["ink"], font=SANS, weight="600"))
    for v in (0, 20, 40, 60, 80):
        y = ybot - v / top * (ybot - ytop)
        s.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                 f'stroke="{t["rule2"]}" stroke-width="1"/>')
        s.append(txt(x0 - 10, y + 4, str(v), t, fill=t["ink3"], anchor="end"))
    s.append(f'<line x1="{x0}" y1="{ybot}" x2="{x1}" y2="{ybot}" '
             f'stroke="{t["rule"]}" stroke-width="1"/>')
    step = (x1 - x0 - 40) / (len(pts) - 1)
    xy = []
    for i, p in enumerate(pts):
        x = x0 + 30 + i * step
        y = ybot - p["tok_s"] / top * (ybot - ytop)
        xy.append((x, y, p))
    s.append('<polyline fill="none" stroke="{}" stroke-width="2" '
             'stroke-linejoin="round" points="{}"/>'.format(
                 t["a"], " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in xy)))
    for x, y, p in xy:
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{t["a"]}" '
                 f'stroke="{t["bg"]}" stroke-width="2"/>')
        s.append(txt(x, y - 13, f'{p["tok_s"]:.1f}', t, fill=t["ink"],
                     anchor="middle", weight="500"))
        s.append(txt(x, ybot + 22, str(p["concurrency"]), t, fill=t["ink3"],
                     anchor="middle"))
    s.append(txt((x0 + x1) / 2, ybot + 46, "concurrent requests", t,
                 size=12, font=SANS, anchor="middle"))
    lo, hi = pts[0]["tok_s"], pts[-1]["tok_s"]
    s.append(txt(x0, h - 12,
                 f"{lo:.1f} to {hi:.1f} tok/s, {hi / lo:.2f}x, "
                 f"one decode loop over every in-flight row", t,
                 size=11, fill=t["ink3"], font=SANS))
    return "".join(s) + "</svg>"


def bars_chart(rows, title, caption, t, unit="tok/s"):
    w = 760
    h = 90 + 52 * len(rows)
    x0, x1 = 210, 660
    top = max(r[1] for r in rows) * 1.18
    s = [head(w, h, t, title)]
    s.append(txt(24, 28, title, t, size=13, fill=t["ink"], font=SANS,
                 weight="600"))
    y = 52
    for label, val, colour in rows:
        bw = (val / top) * (x1 - x0)
        s.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="30" rx="4" '
                 f'fill="{colour}"/>')
        s.append(txt(x0 - 12, y + 20, label, t, size=12, font=SANS,
                     anchor="end"))
        s.append(txt(x0 + bw + 10, y + 20,
                     f"{val:g} {unit}".strip(), t, fill=t["ink"],
                     weight="500"))
        y += 52
    s.append(f'<line x1="{x0}" y1="44" x2="{x0}" y2="{y - 14}" '
             f'stroke="{t["rule"]}" stroke-width="1"/>')
    s.append(txt(24, h - 14, caption, t, size=11, fill=t["ink3"], font=SANS))
    return "".join(s) + "</svg>"


def main():
    d = json.loads(RECEIPT.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    ab = d["continuous_batching_ab"]
    pr = d["prefix_reuse"]
    written = []
    for name, theme in THEMES.items():
        t = theme
        charts = {
            f"serving-scaling-{name}.svg": scaling_chart(d, t),
            f"serving-batching-{name}.svg": bars_chart(
                [("batching off", ab["off"]["tok_s"], t["none"]),
                 ("batching on", ab["on"]["tok_s"], t["a"])],
                f'Continuous batching, {ab["concurrency"]} concurrent requests',
                f'{ab["on"]["tok_s"] / ab["off"]["tok_s"]:.2f}x aggregate '
                f'throughput. Same prompts, server restarted between arms.',
                t),
            f"serving-prefix-{name}.svg": bars_chart(
                [("cold", pr["cold"]["latency_ms"], t["b"]),
                 ("warm", pr["warm"]["latency_ms"], t["a"]),
                 ("different cache_salt", pr["isolation"]["latency_ms"],
                  t["b"])],
                "Time to answer, shared system prompt",
                f'{pr["warm"]["cached_tokens"]} of {pr["prompt_tokens"]} '
                f'prompt tokens reused when warm '
                f'({pr["warm"]["cached_tokens"] / pr["prompt_tokens"] * 100:.1f}%). '
                f'A different cache_salt with identical text reuses nothing.',
                t, unit="ms"),
        }
        for fname, svg in charts.items():
            (OUT / fname).write_text(svg + "\n")
            written.append(fname)
    for f in sorted(written):
        print(f"wrote docs/assets/{f}")


if __name__ == "__main__":
    main()
