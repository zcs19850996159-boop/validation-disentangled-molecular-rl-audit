#!/usr/bin/env python
"""
Build property-matched active/negative sets for docking-oracle calibration.

This is meant to avoid analogue/property bias when we filter DRD2 actives toward
an antagonist-like chemotype. Negatives are selected to match simple
physicochemical properties while avoiding the same Murcko scaffold and high
pairwise fingerprint similarity to the matched active.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DESCRIPTOR_FIELDS = [
    "mw",
    "logp",
    "tpsa",
    "hba",
    "hbd",
    "rotb",
    "heavy_atoms",
    "formal_charge",
]


@dataclass
class MoleculeRecord:
    row: dict[str, str]
    label: str
    mol: object
    fp: object
    scaffold: str
    descriptors: dict[str, float]
    max_ref_similarity: float


def find_obabel() -> str:
    obabel = shutil.which("obabel")
    if obabel is None:
        candidate = Path(sys.executable).with_name("obabel")
        if candidate.exists():
            obabel = str(candidate)
    if obabel is None:
        raise RuntimeError("obabel is required for --reference-pdb")
    return obabel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_reference_mols(reference_smiles: list[str], reference_sdf: list[str], reference_pdb: list[str]):
    from rdkit import Chem

    mols = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"Could not parse reference SMILES: {smi}")
        mols.append(mol)

    for sdf in reference_sdf:
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=True)
        mol = supplier[0] if supplier and len(supplier) else None
        if mol is None:
            raise RuntimeError(f"Could not parse reference SDF: {sdf}")
        mols.append(mol)

    if reference_pdb:
        with tempfile.TemporaryDirectory(prefix="refs_") as tmpdir:
            tmp = Path(tmpdir)
            for idx, pdb in enumerate(reference_pdb):
                sdf = tmp / f"ref_{idx}.sdf"
                subprocess.run([find_obabel(), str(pdb), "-O", str(sdf), "-h"], check=True)
                supplier = Chem.SDMolSupplier(str(sdf), removeHs=True)
                mol = supplier[0] if supplier and len(supplier) else None
                if mol is None:
                    raise RuntimeError(f"Could not parse converted reference PDB: {pdb}")
                mols.append(mol)

    if not mols:
        raise RuntimeError("At least one reference molecule is required")
    return mols


def fingerprint(mol):
    from rdkit.Chem import AllChem

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def descriptors(mol) -> dict[str, float]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    return {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "rotb": float(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
    }


def scaffold_smiles(mol) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def make_record(row: dict[str, str], ref_fps: list[object]) -> MoleculeRecord | None:
    from rdkit import Chem, DataStructs

    smiles = (row.get("smiles") or "").strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = fingerprint(mol)
    max_ref_similarity = max(float(DataStructs.TanimotoSimilarity(fp, ref_fp)) for ref_fp in ref_fps)
    return MoleculeRecord(
        row=row,
        label=row.get("label", ""),
        mol=mol,
        fp=fp,
        scaffold=scaffold_smiles(mol),
        descriptors=descriptors(mol),
        max_ref_similarity=max_ref_similarity,
    )


def zscore_params(records: list[MoleculeRecord]) -> dict[str, tuple[float, float]]:
    params = {}
    for field in DESCRIPTOR_FIELDS:
        values = [rec.descriptors[field] for rec in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        std = math.sqrt(variance) or 1.0
        params[field] = (mean, std)
    return params


def property_distance(a: MoleculeRecord, b: MoleculeRecord, params: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for field in DESCRIPTOR_FIELDS:
        _, std = params[field]
        total += ((a.descriptors[field] - b.descriptors[field]) / std) ** 2
    return math.sqrt(total)


def tanimoto(a: MoleculeRecord, b: MoleculeRecord) -> float:
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(a.fp, b.fp))


def select_actives(records: list[MoleculeRecord], similarity_threshold: float, n: int) -> list[MoleculeRecord]:
    candidates = [rec for rec in records if rec.max_ref_similarity >= similarity_threshold]
    candidates.sort(
        key=lambda rec: (
            rec.max_ref_similarity,
            float(rec.row.get("pchembl_value") or 0.0),
        ),
        reverse=True,
    )
    return candidates[:n]


def match_negatives(
    actives: list[MoleculeRecord],
    negatives: list[MoleculeRecord],
    params: dict[str, tuple[float, float]],
    max_pair_tanimoto: float,
    max_property_distance: float,
) -> tuple[list[tuple[MoleculeRecord, MoleculeRecord, float, float]], list[str]]:
    used: set[int] = set()
    pairs = []
    warnings = []

    for active in actives:
        ranked = []
        for idx, negative in enumerate(negatives):
            if idx in used:
                continue
            pair_sim = tanimoto(active, negative)
            if pair_sim > max_pair_tanimoto:
                continue
            if active.scaffold and active.scaffold == negative.scaffold:
                continue
            distance = property_distance(active, negative, params)
            ranked.append((distance, pair_sim, idx, negative))

        ranked.sort(key=lambda item: (item[0], item[1]))
        if not ranked:
            warnings.append(f"No scaffold-different negative found for {active.row.get('molecule_chembl_id')}")
            continue

        distance, pair_sim, idx, negative = ranked[0]
        if distance > max_property_distance:
            warnings.append(
                f"Matched negative for {active.row.get('molecule_chembl_id')} has property distance {distance:.3f}"
            )
        used.add(idx)
        pairs.append((active, negative, distance, pair_sim))

    return pairs, warnings


def write_smi(path: Path, records: list[MoleculeRecord]):
    with path.open("w", newline="") as f:
        for i, rec in enumerate(records):
            mol_id = rec.row.get("molecule_chembl_id") or f"mol_{i}"
            f.write(f"{rec.row['smiles']} {mol_id}\n")


def write_audit(path: Path, pairs: list[tuple[MoleculeRecord, MoleculeRecord, float, float]]):
    fieldnames = [
        "pair_id",
        "role",
        "smiles",
        "molecule_chembl_id",
        "pchembl_value",
        "max_ref_similarity",
        "murcko_scaffold",
        "matched_property_distance",
        "pair_tanimoto",
        *DESCRIPTOR_FIELDS,
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair_id, (active, negative, distance, pair_sim) in enumerate(pairs):
            for role, rec in [("active", active), ("negative", negative)]:
                row = {
                    "pair_id": pair_id,
                    "role": role,
                    "smiles": rec.row.get("smiles", ""),
                    "molecule_chembl_id": rec.row.get("molecule_chembl_id", ""),
                    "pchembl_value": rec.row.get("pchembl_value", ""),
                    "max_ref_similarity": round(rec.max_ref_similarity, 4),
                    "murcko_scaffold": rec.scaffold,
                    "matched_property_distance": round(distance, 4),
                    "pair_tanimoto": round(pair_sim, 4),
                }
                row.update({field: round(rec.descriptors[field], 4) for field in DESCRIPTOR_FIELDS})
                writer.writerow(row)


def group_stats(records: list[MoleculeRecord]) -> dict[str, dict[str, float]]:
    stats = {}
    for field in DESCRIPTOR_FIELDS:
        values = [rec.descriptors[field] for rec in records]
        mean = sum(values) / len(values) if values else float("nan")
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1) if values else float("nan")
        stats[field] = {"mean": mean, "std": math.sqrt(variance) if values else float("nan")}
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="data/chembl_validation_matched")
    parser.add_argument("--name", default="DRD2_7DFP_antagonist_matched")
    parser.add_argument("--reference-smiles", action="append", default=[])
    parser.add_argument("--reference-sdf", action="append", default=[])
    parser.add_argument("--reference-pdb", action="append", default=[])
    parser.add_argument("--active-sim-threshold", type=float, default=0.20)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--max-pair-tanimoto", type=float, default=0.45)
    parser.add_argument("--max-property-distance", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_mols = read_reference_mols(args.reference_smiles, args.reference_sdf, args.reference_pdb)
    ref_fps = [fingerprint(mol) for mol in ref_mols]

    records = []
    for row in read_csv(Path(args.input_csv)):
        rec = make_record(row, ref_fps)
        if rec is not None:
            records.append(rec)

    active_pool = [rec for rec in records if rec.label == "active"]
    negative_pool = [rec for rec in records if rec.label in {"negative", "decoy"}]
    selected_actives = select_actives(active_pool, args.active_sim_threshold, args.pairs)
    params = zscore_params(selected_actives + negative_pool)
    pairs, warnings = match_negatives(
        selected_actives,
        negative_pool,
        params,
        max_pair_tanimoto=args.max_pair_tanimoto,
        max_property_distance=args.max_property_distance,
    )

    matched_actives = [pair[0] for pair in pairs]
    matched_negatives = [pair[1] for pair in pairs]
    if len(pairs) < args.pairs:
        warnings.append(f"Requested {args.pairs} pairs but built {len(pairs)} pairs")

    active_path = out_dir / f"{args.name}_actives.smi"
    negative_path = out_dir / f"{args.name}_negatives.smi"
    audit_path = out_dir / f"{args.name}_audit.csv"
    summary_path = out_dir / f"{args.name}_summary.json"

    write_smi(active_path, matched_actives)
    write_smi(negative_path, matched_negatives)
    write_audit(audit_path, pairs)

    summary = {
        "name": args.name,
        "input_csv": args.input_csv,
        "reference_count": len(ref_mols),
        "active_pool": len(active_pool),
        "negative_pool": len(negative_pool),
        "active_similarity_threshold": args.active_sim_threshold,
        "requested_pairs": args.pairs,
        "matched_pairs": len(pairs),
        "max_pair_tanimoto": args.max_pair_tanimoto,
        "max_property_distance": args.max_property_distance,
        "active_stats": group_stats(matched_actives),
        "negative_stats": group_stats(matched_negatives),
        "mean_pair_property_distance": sum(pair[2] for pair in pairs) / len(pairs) if pairs else None,
        "mean_pair_tanimoto": sum(pair[3] for pair in pairs) / len(pairs) if pairs else None,
        "warnings": warnings,
        "outputs": {
            "actives": str(active_path),
            "negatives": str(negative_path),
            "audit": str(audit_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
