#!/usr/bin/env python
"""
Redock a co-crystal ligand and report direct and symmetry-aware pose RMSD.

This is a validation-oracle diagnostic, not part of REINVENT training.
The primary docking RMSD is computed in the receptor coordinate frame without
fitting, because pose reproduction should not be rescued by aligning the
predicted ligand onto the crystal ligand after docking.
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
import time
from pathlib import Path

from dock_with_vina import parse_vina_log, run_vina


def find_obabel() -> str:
    obabel = shutil.which("obabel")
    if obabel is None:
        candidate = Path(sys.executable).with_name("obabel")
        if candidate.exists():
            obabel = str(candidate)
    if obabel is None:
        raise RuntimeError("obabel is required to convert the co-crystal ligand to PDBQT")
    return obabel


def ligand_pdb_to_pdbqt(ligand_pdb: Path, ligand_pdbqt: Path, protonate_ph: float | None):
    cmd = [
        find_obabel(),
        str(ligand_pdb),
        "-O",
        str(ligand_pdbqt),
    ]
    if protonate_ph is None:
        cmd.append("-h")
    else:
        cmd.extend(["-p", str(protonate_ph)])
    cmd.extend(["--partialcharge", "gasteiger"])
    subprocess.run(cmd, check=True)


def ligand_pdb_to_sdf(ligand_pdb: Path, ligand_sdf: Path, protonate_ph: float | None):
    cmd = [
        find_obabel(),
        str(ligand_pdb),
        "-O",
        str(ligand_sdf),
    ]
    if protonate_ph is None:
        cmd.append("-h")
    else:
        cmd.extend(["-p", str(protonate_ph)])
    subprocess.run(cmd, check=True)


def obabel_convert(input_path: Path, output_path: Path):
    subprocess.run([find_obabel(), str(input_path), "-O", str(output_path)], check=True)


def pdbqt_heavy_xyz(path: Path) -> list[tuple[float, float, float]]:
    coords = []
    for line in path.read_text(errors="replace").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip().upper()
        atom_type = line.split()[-1].upper()
        if atom_name.startswith("H") or atom_type.startswith("H"):
            continue
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def direct_rmsd(a: list[tuple[float, float, float]], b: list[tuple[float, float, float]]) -> float:
    if len(a) != len(b):
        raise RuntimeError(f"Heavy atom count mismatch: {len(a)} vs {len(b)}")
    if not a:
        raise RuntimeError("No heavy atoms found for RMSD calculation")
    squared = [sum((x - y) ** 2 for x, y in zip(pa, pb)) for pa, pb in zip(a, b)]
    return math.sqrt(sum(squared) / len(squared))


def read_sdf(path: Path, remove_hs: bool, sanitize: bool = True):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sanitize)
    mol = supplier[0] if supplier and len(supplier) else None
    if mol is None:
        raise RuntimeError(f"Could not read molecule from {path}")
    if not sanitize:
        mol.UpdatePropertyCache(strict=False)
    if remove_hs:
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
        except TypeError:
            mol = Chem.RemoveHs(mol)
    return mol


def mol_with_replaced_coords(reference, coords: list[tuple[float, float, float]]):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    if reference.GetNumAtoms() != len(coords):
        raise RuntimeError(f"Coordinate count mismatch: {reference.GetNumAtoms()} vs {len(coords)}")
    mol = Chem.Mol(reference)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def symmetry_rmsds(
    reference_sdf: Path,
    crystal_coords: list[tuple[float, float, float]],
    probe_coords: list[tuple[float, float, float]],
) -> tuple[float, float, int, int]:
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    ref = read_sdf(reference_sdf, remove_hs=True)
    ref = Chem.RemoveHs(ref)
    ref = mol_with_replaced_coords(ref, crystal_coords)
    probe = mol_with_replaced_coords(ref, probe_coords)
    no_fit = rdMolAlign.CalcRMS(probe, ref, maxMatches=1000000)
    best_fit = rdMolAlign.GetBestRMS(probe, ref, maxMatches=1000000)
    return float(no_fit), float(best_fit), ref.GetNumHeavyAtoms(), probe.GetNumHeavyAtoms()


def formal_charge(path: Path) -> int:
    from rdkit import Chem

    mol = read_sdf(path, remove_hs=False)
    return int(Chem.GetFormalCharge(mol))


def box_margin(coords: list[tuple[float, float, float]], config: dict) -> float:
    center = [config["center_x"], config["center_y"], config["center_z"]]
    half = [config["size_x"] / 2, config["size_y"] / 2, config["size_z"] / 2]
    margins = []
    for point in coords:
        for axis in range(3):
            margins.append(half[axis] - abs(point[axis] - center[axis]))
    return min(margins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--config-dir", default="receptors")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--exhaustiveness", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--protonate-ph", type=float, default=None)
    parser.add_argument("--rmsd-threshold", type=float, default=2.0)
    args = parser.parse_args()

    target = args.target.upper()
    config_path = Path(args.config_dir) / f"{target}.json"
    config = json.loads(config_path.read_text())
    exhaustiveness = args.exhaustiveness or int(config.get("vina_defaults", {}).get("exhaustiveness", 8))
    seed = args.seed or int(config.get("vina_defaults", {}).get("seed", 17))
    output_path = Path(args.output or f"redocking/{target}_redocking_exh{exhaustiveness}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pose_dir = Path(args.pose_dir or output_path.with_suffix(""))
    pose_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"redock_{target.lower()}_") as tmpdir:
        tmp = Path(tmpdir)
        ligand_pdb = Path(config["co_crystal_ligand_pdb"])
        ligand_pdbqt = tmp / f"{target}_cocrystal_input.pdbqt"
        out_pdbqt = tmp / f"{target}_redocked.pdbqt"
        log_path = tmp / f"{target}_redocked.log"
        reference_sdf = tmp / f"{target}_cocrystal_reference.sdf"
        redocked_sdf = tmp / f"{target}_redocked.sdf"
        redocked_pdb = tmp / f"{target}_redocked.pdb"

        ligand_pdb_to_pdbqt(ligand_pdb, ligand_pdbqt, args.protonate_ph)
        ligand_pdb_to_sdf(ligand_pdb, reference_sdf, args.protonate_ph)
        crystal_xyz = pdbqt_heavy_xyz(ligand_pdbqt)
        score = run_vina(config, ligand_pdbqt, out_pdbqt, log_path, exhaustiveness, seed)
        if score is None:
            score = parse_vina_log(log_path)
        obabel_convert(out_pdbqt, redocked_sdf)
        obabel_convert(out_pdbqt, redocked_pdb)
        docked_xyz = pdbqt_heavy_xyz(out_pdbqt)
        rmsd = direct_rmsd(crystal_xyz, docked_xyz)
        symmetry_no_fit = ""
        symmetry_best_fit = ""
        symmetry_error = ""
        ref_heavy = ""
        probe_heavy = ""
        try:
            symmetry_no_fit, symmetry_best_fit, ref_heavy, probe_heavy = symmetry_rmsds(
                reference_sdf,
                crystal_xyz,
                docked_xyz,
            )
        except Exception as exc:
            symmetry_error = f"{type(exc).__name__}: {str(exc)[:120]}"

        ligand_formal_charge = ""
        formal_charge_error = ""
        try:
            ligand_formal_charge = formal_charge(reference_sdf)
        except Exception as exc:
            formal_charge_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        crystal_box_margin = box_margin(crystal_xyz, config)
        docked_box_margin = box_margin(docked_xyz, config)

        for path in [ligand_pdbqt, out_pdbqt, log_path, reference_sdf, redocked_sdf, redocked_pdb]:
            shutil.copy2(path, pose_dir / path.name)

    row = {
        "target": target,
        "pdb_id": config.get("pdb_id", ""),
        "co_crystal_ligand": config.get("co_crystal_ligand", ""),
        "exhaustiveness": exhaustiveness,
        "seed": seed,
        "protonate_ph": args.protonate_ph if args.protonate_ph is not None else "",
        "heavy_atom_count": len(crystal_xyz),
        "rdkit_ref_heavy_atom_count": ref_heavy,
        "rdkit_probe_heavy_atom_count": probe_heavy,
        "ligand_formal_charge": ligand_formal_charge,
        "vina_score": score,
        "direct_heavy_rmsd": rmsd,
        "symmetry_no_fit_rmsd": symmetry_no_fit,
        "symmetry_best_fit_rmsd": symmetry_best_fit,
        "symmetry_error": symmetry_error,
        "rmsd_threshold": args.rmsd_threshold,
        "passed": (symmetry_no_fit if symmetry_no_fit != "" else rmsd) <= args.rmsd_threshold,
        "formal_charge_error": formal_charge_error,
        "crystal_min_box_margin": crystal_box_margin,
        "redocked_min_box_margin": docked_box_margin,
        "pose_dir": str(pose_dir),
        "seconds": round(time.time() - t0, 3),
    }
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(row)
    raise SystemExit(0 if row["passed"] else 2)


if __name__ == "__main__":
    main()
