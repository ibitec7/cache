"""Build every figure in the blog: five Mermaid schematics and three data charts.

Input contract
--------------
`blog/diagrams/src/*.mmd` holds the five structural diagrams as Mermaid source. The three data
charts come from `scripts/blog_figures.py`, which reads `artifacts/` through `blog_build`.

Output contract
---------------
`blog/diagrams/<name>.svg` for all eight, plus `manifest.json` (name to file and pixel size) that
`blog_build.py` uses to place them. Every SVG must draw on a white page and on a near-black one:

- the Mermaid schematics come from `pretty-mermaid` (Node) as two fixed-theme renders, and this
  script folds them into one file by appending a `@media (prefers-color-scheme: dark)` block that
  overrides the renderer's own custom properties with `!important`;
- the charts are already transparent with a mid-tone palette, so they need no second version.

One asset per figure, so the Markdown stays plain `![alt](diagrams/x.svg)` and a renderer that
cannot read the media query still shows a correct light figure instead of a broken one.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import tempfile

import blog_figures

HERE = pathlib.Path(__file__).resolve().parent
BLOG = HERE.parent / "blog"
SRC = BLOG / "diagrams" / "src"
OUT = BLOG / "diagrams"
RENDER = pathlib.Path("/home/ibrahim/.hermes/skills/pretty-mermaid/scripts/render.mjs")

STRUCTURAL = ("path", "episode", "layers", "pipeline", "validate")
THEMES = {"light": "github-light", "dark": "github-dark"}
FLAGS = ["--padding", "18", "--node-spacing", "46", "--layer-spacing", "46"]


def render(name: str, workdir: pathlib.Path) -> dict[str, pathlib.Path]:
    out = {}
    for mode, theme in THEMES.items():
        target = workdir / f"{name}.{mode}.svg"
        cmd = ["node", str(RENDER), "--input", str(SRC / f"{name}.mmd"),
               "--output", str(target), "--theme", theme, *FLAGS]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode or not target.exists() or target.stat().st_size < 500:
            raise RuntimeError(f"{name} {mode}: {done.stderr.strip()[-400:]}")
        out[mode] = target
    return out


def palette(svg: str) -> dict[str, str]:
    """The custom properties the renderer wrote onto the root element."""
    root = svg[: svg.index(">") + 1]
    return dict(re.findall(r"--([a-z-]+):(#[0-9a-fA-F]{3,8})", root))


def adaptive(light: str, dark: str) -> str:
    """Light render plus a dark-scheme override of the same custom properties."""
    vars_dark = palette(dark)
    if not vars_dark:
        raise RuntimeError("no palette found in the dark render")
    css = ["  @media (prefers-color-scheme: dark) {", "    svg {"]
    css += [f"      --{key}: {value} !important;" for key, value in vars_dark.items()]
    css += ["    }", "  }"]
    cut = light.index("</style>")
    return light[:cut] + "\n".join(css) + "\n" + light[cut:]


def size(svg: str) -> tuple[int, int]:
    view = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
    return (round(float(view.group(1))), round(float(view.group(2)))) if view else (0, 0)


def main() -> int:
    if not RENDER.exists():
        raise SystemExit(f"renderer not found: {RENDER}")
    blog_figures.main()
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        for name in STRUCTURAL:
            files = render(name, work)
            (OUT / f"{name}.svg").write_text(
                adaptive(files["light"].read_text(), files["dark"].read_text()))
    manifest = {}
    for path in sorted(OUT.glob("*.svg")):
        svg = path.read_text()
        w, h = size(svg)
        manifest[path.stem] = {"file": f"diagrams/{path.name}", "w": w, "h": h,
                               "bytes": path.stat().st_size}
        print(f"  {path.stem:<9} {w:>5} x {h:<4} {path.stat().st_size:>7} bytes")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} adaptive SVGs and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
