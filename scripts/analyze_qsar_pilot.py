#!/usr/bin/env python
"""
Posthoc analysis for DRD2 QSAR pilot runs.

Scores generated SMILES with the independent validation QSAR model and reports
simple pilot metrics for dynamic-vs-static sanity checks.
"""

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


def scaffold(smiles: str) -> str:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
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


def read_run(path: Path, payload: dict) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get("SMILES") or "").strip()
            if not smiles or row.get("SMILES_state") != "1":
                continue
            validation_activity = predict_probability(payload, smiles)
            if validation_activity is None:
                continue
            rows.append(
                {
                    "smiles": smiles,
                    "scaffold": scaffold(smiles),
                    "step": int(float(row.get("step") or 0)),
                    "training_score": to_float(row.get("Score")),
                    "training_activity": to_float(row.get("DRD2_activity (raw)") or row.get("DRD2_activity")),
                    "qed": to_float(row.get("QED (raw)") or row.get("QED")),
                    "mw_score": to_float(row.get("Molecular weight")),
                    "mw": to_float(row.get("Molecular weight (raw)")),
                    "validation_activity": validation_activity,
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
    by_validation = sorted(rows, key=lambda row: row["validation_activity"], reverse=True)

    return {
        "n_rows": len(rows),
        "n_unique_smiles": len({row["smiles"] for row in rows}),
        "n_unique_scaffolds": len({row["scaffold"] for row in rows if row["scaffold"]}),
        "final_step": final_step,
        "all_mean_validation_activity": mean([row["validation_activity"] for row in rows]),
        "last50_mean_validation_activity": mean([row["validation_activity"] for row in last_50]),
        "final_step_mean_validation_activity": mean([row["validation_activity"] for row in final_rows]),
        "top10_mean_validation_activity": mean([row["validation_activity"] for row in by_validation[:10]]),
        "top100_mean_validation_activity": mean([row["validation_activity"] for row in by_validation[:100]]),
        "all_mean_training_score": mean([row["training_score"] for row in rows]),
        "last50_mean_training_score": mean([row["training_score"] for row in last_50]),
        "final_step_mean_training_score": mean([row["training_score"] for row in final_rows]),
        "all_mean_training_activity": mean([row["training_activity"] for row in rows]),
        "last50_mean_training_activity": mean([row["training_activity"] for row in last_50]),
        "final_step_mean_training_activity": mean([row["training_activity"] for row in final_rows]),
        "final_step_mean_qed": mean([row["qed"] for row in final_rows]),
        "final_step_mean_mw_score": mean([row["mw_score"] for row in final_rows]),
    }


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
