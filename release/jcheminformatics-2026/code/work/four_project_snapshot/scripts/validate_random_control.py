#!/usr/bin/env python
"""
Random-validation control for DynamicWeightController.

This command has the same CSV interface as validate_qsar.py, but it deliberately
breaks the relationship between the current validation-only molecules and the
scores passed to the controller. In `shuffle_history` mode it samples one
previous validation-score record from a controller history JSONL and repeats
that score for every input molecule, preserving the marginal scale of the real
validation signal while removing current-batch meaning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_component_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise RuntimeError("No component names were provided")
    return names


def stable_seed(base_seed: int, rows: list[dict[str, str]]) -> int:
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode())
    for row in rows:
        digest.update((row.get("smiles") or "").encode())
        digest.update(b"\n")
    return int.from_bytes(digest.digest()[:8], "big")


def read_history_scores(path: Path, component_names: list[str]) -> list[dict[str, float]]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores = row.get("validation_scores", {})
        if all(name in scores for name in component_names):
            records.append({name: float(scores[name]) for name in component_names})
    if not records:
        raise RuntimeError(f"No usable validation_scores found in {path}")
    return records


def parse_ranges(raw: str, component_names: list[str]) -> dict[str, tuple[float, float]]:
    ranges = {name: (0.0, 1.0) for name in component_names}
    if not raw.strip():
        return ranges
    for item in raw.split(","):
        if not item.strip():
            continue
        name, values = item.split("=", 1)
        low, high = values.split(":", 1)
        ranges[name.strip()] = (float(low), float(high))
    return ranges


def choose_scores(args, rows: list[dict[str, str]], component_names: list[str]) -> dict[str, float]:
    rng = random.Random(stable_seed(args.seed, rows))
    if args.mode == "shuffle_history":
        if not args.history:
            raise RuntimeError("--history is required for shuffle_history mode")
        history = read_history_scores(Path(args.history), component_names)
        return dict(rng.choice(history))

    ranges = parse_ranges(args.ranges, component_names)
    return {name: rng.uniform(*ranges[name]) for name in component_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--component-names", required=True)
    parser.add_argument("--mode", choices=["shuffle_history", "uniform"], default="shuffle_history")
    parser.add_argument("--history", default=None)
    parser.add_argument("--ranges", default="")
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    component_names = parse_component_names(args.component_names)
    rows = read_rows(Path(args.input))
    scores = choose_scores(args, rows, component_names)

    output_fields = ["smiles", *component_names, "status", "control_mode"]
    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            out = {
                "smiles": (row.get("smiles") or "").strip(),
                "status": "random_control",
                "control_mode": args.mode,
            }
            out.update(scores)
            writer.writerow(out)


if __name__ == "__main__":
    main()
