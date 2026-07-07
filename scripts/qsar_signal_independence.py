#!/usr/bin/env python
"""Quantify independence between training-reward and validation QSAR signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path


def load_model(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def read_split_rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("split") == split]


def fingerprint(smiles: str, radius: int, n_bits: int):
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((1, n_bits), dtype="float32")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr[0])
    return arr


def predict_probability(payload: dict, smiles: str) -> float | None:
    x = fingerprint(smiles, radius=int(payload["radius"]), n_bits=int(payload["n_bits"]))
    if x is None:
        return None
    model = payload["model"]
    if hasattr(model, "predict_proba"):
        value = float(model.predict_proba(x)[0, 1])
    else:
        raw = float(model.decision_function(x)[0])
        value = 1.0 / (1.0 + math.exp(-raw))
    return value if math.isfinite(value) else None


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return None
    return num / (da * db)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            result[indexed[k][0]] = rank
        i = j
    return result


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(ranks(a), ranks(b))


def mean_abs_diff(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def top_fraction_indices(values: list[float], fraction: float) -> set[int]:
    n = max(1, int(round(len(values) * fraction)))
    return {idx for idx, _value in sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:n]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--validation-model", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reward_payload = load_model(Path(args.reward_model))
    validation_payload = load_model(Path(args.validation_model))
    rows = read_split_rows(Path(args.splits), args.split)

    scored = []
    for row in rows:
        smiles = row["smiles"]
        reward = predict_probability(reward_payload, smiles)
        validation = predict_probability(validation_payload, smiles)
        if reward is None or validation is None:
            continue
        scored.append(
            {
                "smiles": smiles,
                "label": int(row["label"]),
                "reward_rf": reward,
                "validation_svm": validation,
                "abs_diff": abs(reward - validation),
            }
        )

    reward_scores = [row["reward_rf"] for row in scored]
    validation_scores = [row["validation_svm"] for row in scored]
    top10_reward = top_fraction_indices(reward_scores, 0.10)
    top10_validation = top_fraction_indices(validation_scores, 0.10)
    union = top10_reward | top10_validation
    intersection = top10_reward & top10_validation

    result = {
        "split": args.split,
        "n_scored": len(scored),
        "reward_model": reward_payload.get("role", "reward"),
        "validation_model": validation_payload.get("role", "validation"),
        "pearson": pearson(reward_scores, validation_scores),
        "spearman": spearman(reward_scores, validation_scores),
        "mean_abs_diff": mean_abs_diff(reward_scores, validation_scores),
        "reward_mean": sum(reward_scores) / len(reward_scores),
        "validation_mean": sum(validation_scores) / len(validation_scores),
        "top10_fraction_jaccard": len(intersection) / len(union) if union else None,
        "top10_fraction_overlap_count": len(intersection),
        "top10_fraction_size": len(top10_reward),
        "interpretation": (
            "Signals are algorithmically and split independent, but high "
            "correlation means the validation oracle is not an orthogonal signal."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
