#!/usr/bin/env python
"""Synthetic-accessibility scoring helpers for DRD2 controller experiments."""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path


def _import_sascorer():
    try:
        return importlib.import_module("rdkit.Contrib.SA_Score.sascorer")
    except Exception:
        pass

    candidates = []
    for root in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if root:
            candidates.append(Path(root) / "share" / "RDKit" / "Contrib" / "SA_Score")
            candidates.append(Path(root) / "Lib" / "site-packages" / "rdkit" / "Contrib" / "SA_Score")

    for path in candidates:
        scorer = path / "sascorer.py"
        if scorer.exists():
            sys.path.insert(0, str(path))
            return importlib.import_module("sascorer")

    raise RuntimeError(
        "RDKit SA_Score sascorer.py was not found. Install/copy RDKit Contrib "
        "SA_Score before running exact SA validation."
    )


def mol_from_smiles(smiles: str):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    return Chem.MolFromSmiles(smiles)


def clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def sa_reward_from_raw(raw_sa: float) -> float:
    """Convert Ertl-Schuffenhauer SA score, 1 easy to 10 hard, to 0..1 reward."""

    return clip01((10.0 - float(raw_sa)) / 9.0)


def exact_sa_raw_from_mol(mol) -> float:
    sascorer = _import_sascorer()
    return float(sascorer.calculateScore(mol))


def exact_sa_reward_from_mol(mol) -> float:
    return sa_reward_from_raw(exact_sa_raw_from_mol(mol))


def proxy_sa_reward_from_mol(mol) -> float:
    """
    Cheap SA proxy for the training reward.

    This deliberately avoids RDKit Contrib fragment scores so the training-side
    SA objective is not identical to the exact SA validation signal.
    """

    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    heavy = float(Descriptors.HeavyAtomCount(mol))
    rot = float(Descriptors.NumRotatableBonds(mol))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    spiro = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    bridge = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    stereo = float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)))
    macro = float(sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8))

    complexity = (
        0.030 * heavy
        + 0.045 * rot
        + 0.100 * rings
        + 0.120 * spiro
        + 0.120 * bridge
        + 0.080 * stereo
        + 0.180 * macro
    )
    return clip01(1.0 / (1.0 + math.exp(1.35 * (complexity - 2.10))))


def score_sa(smiles: str, mode: str = "exact") -> float | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    if mode == "exact":
        return exact_sa_reward_from_mol(mol)
    if mode == "proxy":
        return proxy_sa_reward_from_mol(mol)
    raise ValueError(f"Unknown SA scoring mode: {mode}")
