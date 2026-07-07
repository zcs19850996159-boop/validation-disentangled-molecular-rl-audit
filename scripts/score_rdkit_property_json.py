#!/usr/bin/env python
"""
ExternalProcess scorer for lightweight RDKit property reward components.

Reads SMILES from stdin and writes JSON:
  {"version": 1, "payload": {"<component_name>": [score, ...]}}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa_score_utils import clip01, mol_from_smiles, score_sa  # noqa: E402


def property_score(smiles: str, metric: str) -> float:
    from rdkit.Chem import Descriptors, QED

    mol = mol_from_smiles(smiles)
    if mol is None:
        return 0.0

    if metric == "qed":
        return clip01(float(QED.qed(mol)))
    if metric == "mw_window":
        mw = float(Descriptors.MolWt(mol))
        return 1.0 if 200.0 <= mw <= 500.0 else 0.0
    if metric == "sa_exact":
        value = score_sa(smiles, mode="exact")
        return 0.0 if value is None else clip01(value)
    if metric == "sa_proxy":
        value = score_sa(smiles, mode="proxy")
        return 0.0 if value is None else clip01(value)

    raise ValueError(f"Unsupported metric: {metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", required=True)
    parser.add_argument(
        "--metric",
        required=True,
        choices=["qed", "mw_window", "sa_exact", "sa_proxy"],
    )
    args = parser.parse_args()

    smiles = [line.strip() for line in sys.stdin if line.strip()]
    scores = []
    for smi in smiles:
        score = property_score(smi, args.metric)
        if not math.isfinite(float(score)):
            score = 0.0
        scores.append(clip01(score))

    print(json.dumps({"version": 1, "payload": {args.property: scores}}))


if __name__ == "__main__":
    main()
