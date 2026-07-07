#!/usr/bin/env python
"""
Posthoc analysis for DRD2 activity-vs-SA controller runs.

Validation activity is scored with the independent SVM QSAR model. Validation
SA is exact RDKit Contrib SA converted to a 0..1 "easier synthesis is better"
reward. The primary aggregate is the geometric mean of the two validation
components.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa_score_utils import exact_sa_raw_from_mol, mol_from_smiles, sa_reward_from_raw


def load_model(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def fingerprint(smiles: str, radius: int, n_bits: int):
    import numpy as np
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    mol = mol_from_smiles(smiles)
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


def scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def to_float(value: str | None) -> float | None:
    try:
        if value in ("", None):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def validation_sa(smiles: str) -> tuple[float | None, float | None]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None, None
    raw = exact_sa_raw_from_mol(mol)
    reward = sa_reward_from_raw(raw)
    return reward, raw


def read_run(path: Path, payload: dict) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get("SMILES") or "").strip()
            if not smiles or row.get("SMILES_state") != "1":
                continue
            activity = predict_probability(payload, smiles)
            sa_reward, raw_sa = validation_sa(smiles)
            if activity is None or sa_reward is None:
                continue
            joint = math.sqrt(max(0.0, activity) * max(0.0, sa_reward))
            rows.append(
                {
                    "smiles": smiles,
                    "scaffold": scaffold(smiles),
                    "step": int(float(row.get("step") or 0)),
                    "training_score": to_float(row.get("Score")),
                    "training_activity": to_float(row.get("DRD2_activity (raw)") or row.get("DRD2_activity")),
                    "training_sa": to_float(row.get("SA_score (raw)") or row.get("SA_score")),
                    "validation_activity": activity,
                    "validation_sa": sa_reward,
                    "validation_raw_sa": raw_sa,
                    "validation_joint": joint,
                }
            )
    return rows


def mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rows": 0}

    final_step = max(row["step"] for row in rows)
    final_rows = [row for row in rows if row["step"] == final_step]
    last_50 = [row for row in rows if row["step"] >= final_step - 49]
    by_joint = sorted(rows, key=lambda row: row["validation_joint"], reverse=True)

    def block(prefix: str, selected: list[dict]) -> dict:
        return {
            f"{prefix}_mean_validation_joint": mean([row["validation_joint"] for row in selected]),
            f"{prefix}_mean_validation_activity": mean([row["validation_activity"] for row in selected]),
            f"{prefix}_mean_validation_sa": mean([row["validation_sa"] for row in selected]),
            f"{prefix}_mean_validation_raw_sa": mean([row["validation_raw_sa"] for row in selected]),
            f"{prefix}_mean_training_score": mean([row["training_score"] for row in selected]),
            f"{prefix}_mean_training_activity": mean([row["training_activity"] for row in selected]),
            f"{prefix}_mean_training_sa": mean([row["training_sa"] for row in selected]),
        }

    result = {
        "n_rows": len(rows),
        "n_unique_smiles": len({row["smiles"] for row in rows}),
        "n_unique_scaffolds": len({row["scaffold"] for row in rows if row["scaffold"]}),
        "final_step": final_step,
    }
    result.update(block("all", rows))
    result.update(block("last50", last_50))
    result.update(block("final_step", final_rows))
    result.update(block("top10", by_joint[:10]))
    result.update(block("top100", by_joint[:100]))
    return result


def parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise RuntimeError("--run must be NAME=CSV_PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = load_model(Path(args.model))
    result = {}
    for raw in args.run:
        name, path = parse_run_arg(raw)
        result[name] = summarize(read_run(path, payload))

    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
