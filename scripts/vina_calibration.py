#!/usr/bin/env python
"""
Calibrate a Vina validation oracle with known actives and decoys.

The pass/fail gate is intentionally simple:

- actives should have higher `validation_score = -vina_score` than decoys;
- AUROC should exceed a configurable threshold;
- a permutation test on the active/decoy mean gap should be significant enough
  for a pilot-quality sanity check.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def read_smiles(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
            return [row[col].strip() for row in reader if row.get(col, "").strip()]

    smiles = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        first_field = line.split()[0]
        if first_field.lower() != "smiles":
            smiles.append(first_field)
    return smiles


def sample(values: list[str], n: int | None, seed: int) -> list[str]:
    if n is None or n >= len(values):
        return list(values)
    rng = random.Random(seed)
    return rng.sample(values, n)


def write_input(path: Path, actives: list[str], decoys: list[str]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles", "label"])
        writer.writeheader()
        for i, smi in enumerate(actives):
            writer.writerow({"id": f"active_{i}", "smiles": smi, "label": "active"})
        for i, smi in enumerate(decoys):
            writer.writerow({"id": f"decoy_{i}", "smiles": smi, "label": "decoy"})


def read_scores(path: Path) -> tuple[list[float], list[float]]:
    active_scores, decoy_scores = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("", None, "ok"):
                continue
            score_raw = row.get("validation_score") or row.get("score") or row.get("vina_score")
            if score_raw in ("", None):
                continue
            score = float(score_raw)
            if "validation_score" not in row and "vina_score" in row:
                score = -score
            if row.get("label") == "active":
                active_scores.append(score)
            elif row.get("label") == "decoy":
                decoy_scores.append(score)
    return active_scores, decoy_scores


def auc(active: list[float], decoy: list[float]) -> float:
    wins = ties = total = 0
    for a in active:
        for d in decoy:
            total += 1
            if a > d:
                wins += 1
            elif a == d:
                ties += 1
    return (wins + 0.5 * ties) / total if total else 0.0


def permutation_pvalue(active: list[float], decoy: list[float], permutations: int, seed: int) -> float:
    observed = statistics.mean(active) - statistics.mean(decoy)
    pooled = active + decoy
    n_active = len(active)
    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        delta = statistics.mean(pooled[:n_active]) - statistics.mean(pooled[n_active:])
        if delta >= observed:
            hits += 1
    return (hits + 1) / (permutations + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--actives", required=True)
    parser.add_argument("--decoys", required=True)
    parser.add_argument("--dock-command", required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--max-pvalue", type=float, default=0.10)
    parser.add_argument("--output-dir", default="vina_calibration")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actives = sample(read_smiles(Path(args.actives)), args.sample_size, args.seed)
    decoys = sample(read_smiles(Path(args.decoys)), args.sample_size, args.seed + 1)

    with tempfile.TemporaryDirectory(prefix=f"calib_{args.target.lower()}_") as tmpdir:
        input_csv = Path(tmpdir) / "calibration_input.csv"
        docked_csv = out_dir / f"{args.target.upper()}_vina_calibration_scores.csv"
        write_input(input_csv, actives, decoys)

        command = args.dock_command.format(input=input_csv, output=docked_csv, target=args.target.upper())
        t0 = time.time()
        subprocess.run(command, shell=True, check=True)
        elapsed = time.time() - t0

    active_scores, decoy_scores = read_scores(docked_csv)
    if not active_scores or not decoy_scores:
        raise RuntimeError("No usable active/decoy docking scores were produced")

    summary = {
        "target": args.target.upper(),
        "n_active": len(active_scores),
        "n_decoy": len(decoy_scores),
        "active_mean": statistics.mean(active_scores),
        "decoy_mean": statistics.mean(decoy_scores),
        "active_median": statistics.median(active_scores),
        "decoy_median": statistics.median(decoy_scores),
        "auc": auc(active_scores, decoy_scores),
        "permutation_pvalue": permutation_pvalue(
            active_scores,
            decoy_scores,
            permutations=args.permutations,
            seed=args.seed,
        ),
        "seconds": elapsed,
        "seconds_per_molecule": elapsed / (len(active_scores) + len(decoy_scores)),
    }

    passed = summary["auc"] >= args.min_auc and summary["permutation_pvalue"] <= args.max_pvalue
    summary["passed"] = passed

    summary_path = out_dir / f"{args.target.upper()}_vina_calibration_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(summary)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
