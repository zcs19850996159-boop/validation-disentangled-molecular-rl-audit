#!/usr/bin/env python
"""
Score validation-only SMILES with an independent QSAR fallback model.

This command is designed for `REINVENT_VALIDATION_COMMAND` in
03_dynamic_weight_controller.py. It reads an input CSV with `smiles` and writes
one output row per molecule with columns matching controller component names.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa_score_utils import score_sa


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_component_names(raw: str | None, default_activity: str) -> list[str]:
    if raw is None or not raw.strip():
        return [default_activity]
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise RuntimeError("No component names were provided")
    return names


def load_model(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    for key in ("model", "radius", "n_bits", "component_name"):
        if key not in payload:
            raise RuntimeError(f"Model payload {path} is missing {key}")
    return payload


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
        return float(model.predict_proba(x)[0, 1])
    score = float(model.decision_function(x)[0])
    return 1.0 / (1.0 + math.exp(-score))


def property_score(component_name: str, smiles: str) -> float | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    key = component_name.lower()
    if key == "qed":
        return float(QED.qed(mol))
    if key in {"molecular weight", "mw", "mol_weight"}:
        mw = float(Descriptors.MolWt(mol))
        return 1.0 if 200.0 <= mw <= 500.0 else 0.0
    if key in {"sa", "sa_score", "synthetic accessibility", "synthetic_accessibility"}:
        return score_sa(smiles, mode="exact")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--component-names", default=None)
    args = parser.parse_args()

    payload = load_model(Path(args.model))
    activity_name = str(payload["component_name"])
    component_names = parse_component_names(args.component_names, activity_name)
    rows = read_rows(Path(args.input))

    output_fields = ["smiles", *component_names, "status"]
    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            smiles = (row.get("smiles") or "").strip()
            out = {"smiles": smiles, "status": "ok"}

            activity_score = predict_probability(payload, smiles) if smiles else None
            for name in component_names:
                if name == activity_name:
                    score = activity_score
                else:
                    score = property_score(name, smiles)

                if score is None or not math.isfinite(float(score)):
                    out[name] = ""
                    out["status"] = "invalid_or_unscored"
                else:
                    out[name] = float(score)

            writer.writerow(out)


if __name__ == "__main__":
    main()
