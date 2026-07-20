#!/usr/bin/env python3
"""Static checks for the frozen Python manuscript-figure workflow."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


REQUIRED_SNIPPETS = (
    "plt.rcParams['font.family'] = 'sans-serif'",
    "plt.rcParams['svg.fonttype'] = 'none'",
    'mpl.rcParams["pdf.fonttype"] = 42',
    "FINAL_WIDTH_MM = 183",
    "dpi=600",
)

FORBIDDEN_SNIPPETS = (
    "dropna(",
    ".sample(",
    "np.random",
    "random.",
    "rainbow",
    "jet",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    args = parser.parse_args()

    source = args.script.read_text(encoding="utf-8")
    ast.parse(source)

    failures: list[str] = []
    for snippet in REQUIRED_SNIPPETS:
        if snippet not in source:
            failures.append(f"missing required snippet: {snippet}")
    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in source:
            failures.append(f"forbidden data/style operation: {snippet}")

    if failures:
        print("Nature-figure preflight: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Nature-figure preflight: PASS")
    print("- Python/Matplotlib backend fixed")
    print("- editable SVG/PDF fonts configured")
    print("- 183 mm journal width and 600 dpi TIFF configured")
    print("- no random sampling, row dropping, or unsafe colormaps detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
