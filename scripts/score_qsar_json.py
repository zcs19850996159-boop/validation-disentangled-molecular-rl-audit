#!/usr/bin/env python
"""
ExternalProcess scorer for REINVENT4 QSAR reward components.

Reads SMILES from stdin and writes JSON:
  {"version": 1, "payload": {"<component_name>": [score, ...]}}

Use this with the RandomForest reward model from train_qsar_fallback.py. The
independent controller validator should use validation_svm.pkl through
validate_qsar.py instead.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


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


def predict_probability(payload: dict, smiles: str) -> float:
    x = fingerprint(smiles, radius=int(payload["radius"]), n_bits=int(payload["n_bits"]))
    if x is None:
        return 0.0
    model = payload["model"]
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(x)[0, 1])
    else:
        raw = float(model.decision_function(x)[0])
        score = 1.0 / (1.0 + math.exp(-raw))
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--property", default=None)
    args = parser.parse_args()

    payload = load_model(Path(args.model))
    property_name = args.property or str(payload["component_name"])
    smiles = [line.strip() for line in sys.stdin if line.strip()]
    scores = [predict_probability(payload, smi) for smi in smiles]

    print(json.dumps({"version": 1, "payload": {property_name: scores}}))


if __name__ == "__main__":
    main()
