#!/usr/bin/env python
"""
Audit receptor-preparation side effects around the co-crystal ligand.

This checks whether residues deleted by Meeko's --allow_bad_res are actually
near the ligand, which is more informative than distance to the grid center.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def xyz(line: str) -> tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def parse_deleted_residues(log_path: Path) -> set[str]:
    text = log_path.read_text(errors="replace")
    return {
        residue
        for match in re.findall(r"Template matching failed for: \[([^\]]*)\]", text)
        for residue in re.findall(r"'([^']+)'", match)
    }


def ligand_coords(path: Path) -> list[tuple[float, float, float]]:
    coords = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip().upper()
            if not atom_name.startswith("H"):
                coords.append(xyz(line))
    return coords


def receptor_residue_coords(path: Path) -> dict[str, dict[str, object]]:
    residues: dict[str, dict[str, object]] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21].strip()
        resseq = line[22:26].strip()
        insertion = line[26].strip()
        key = f"{chain}:{resseq}{insertion}"
        resname = line[17:20].strip()
        record = residues.setdefault(key, {"residue": key, "resname": resname, "coords": []})
        record["coords"].append(xyz(line))
    return residues


def min_distance(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> float:
    return min(math.dist(x, y) for x in a for y in b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--config-dir", default="receptors")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pocket-cutoff", type=float, default=6.0)
    args = parser.parse_args()

    target = args.target.upper()
    config_path = Path(args.config_dir) / f"{target}.json"
    config = json.loads(config_path.read_text())
    prep = config.get("receptor_preparation", {})
    log_path = Path(prep.get("log_path") or Path(config["receptor_pdbqt"]).with_suffix(".prep.log"))

    deleted = parse_deleted_residues(log_path)
    lig_xyz = ligand_coords(Path(config["co_crystal_ligand_pdb"]))
    residues = receptor_residue_coords(Path(config["receptor_pdb"]))

    rows = []
    for key, record in residues.items():
        distance = min_distance(record["coords"], lig_xyz)
        rows.append(
            {
                "target": target,
                "residue": key,
                "resname": record["resname"],
                "min_ligand_distance": round(distance, 3),
                "deleted_by_meeko": key in deleted,
                "within_pocket_cutoff": distance <= args.pocket_cutoff,
            }
        )

    rows.sort(key=lambda row: float(row["min_ligand_distance"]))
    output_path = Path(args.output or f"receptor_audit/{target}_receptor_pocket_audit.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    deleted_near = [row for row in rows if row["deleted_by_meeko"] and row["within_pocket_cutoff"]]
    summary = {
        "target": target,
        "deleted_residue_count": len(deleted),
        "pocket_cutoff": args.pocket_cutoff,
        "pocket_residue_count": sum(1 for row in rows if row["within_pocket_cutoff"]),
        "deleted_pocket_residue_count": len(deleted_near),
        "closest_deleted_residues": [
            {
                "residue": row["residue"],
                "resname": row["resname"],
                "min_ligand_distance": row["min_ligand_distance"],
            }
            for row in rows
            if row["deleted_by_meeko"]
        ][:10],
        "deleted_pocket_residues": [
            {
                "residue": row["residue"],
                "resname": row["resname"],
                "min_ligand_distance": row["min_ligand_distance"],
            }
            for row in deleted_near
        ],
        "output": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
