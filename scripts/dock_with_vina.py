#!/usr/bin/env python
"""
Dock SMILES with AutoDock Vina using a locked receptor JSON.

Input CSV must contain a `smiles` column. Optional columns such as `id` and
`label` are copied to the output. Output contains `vina_score` where lower is
better, plus `validation_score = -vina_score` where higher is better for the RL
controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_sdf_from_smiles(smiles: str, sdf_path: Path, seed: int) -> bool:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return True


def ligand_pdbqt_has_atoms(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(errors="replace") as f:
            return any(line.startswith(("ATOM", "HETATM")) for line in f)
    except OSError:
        return False


def sdf_to_pdbqt(sdf_path: Path, pdbqt_path: Path):
    obabel = shutil.which("obabel")
    if obabel is None:
        candidate = Path(sys.executable).with_name("obabel")
        if candidate.exists():
            obabel = str(candidate)
    if obabel is None:
        raise RuntimeError("obabel is required to convert ligand SDF to PDBQT")
    result = subprocess.run(
        [obabel, str(sdf_path), "-O", str(pdbqt_path), "--partialcharge", "gasteiger"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = " ".join((result.stderr or result.stdout or "").split())[:180]
        raise RuntimeError(f"obabel_failed:{details}")
    if not ligand_pdbqt_has_atoms(pdbqt_path):
        details = " ".join((result.stderr or result.stdout or "").split())[:180]
        raise RuntimeError(f"obabel_empty_ligand_pdbqt:{details}")


def parse_vina_log(log_path: Path) -> float | None:
    for line in log_path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def run_vina(config: dict, ligand_pdbqt: Path, out_pdbqt: Path, log_path: Path, exhaustiveness: int, seed: int):
    if shutil.which("vina") is not None:
        cmd = [
            "vina",
            "--receptor", str(config["receptor_pdbqt"]),
            "--ligand", str(ligand_pdbqt),
            "--center_x", str(config["center_x"]),
            "--center_y", str(config["center_y"]),
            "--center_z", str(config["center_z"]),
            "--size_x", str(config["size_x"]),
            "--size_y", str(config["size_y"]),
            "--size_z", str(config["size_z"]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", "1",
            "--seed", str(seed),
            "--out", str(out_pdbqt),
            "--log", str(log_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return parse_vina_log(log_path)

    from vina import Vina

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(str(config["receptor_pdbqt"]))
    v.set_ligand_from_file(str(ligand_pdbqt))
    v.compute_vina_maps(
        center=[config["center_x"], config["center_y"], config["center_z"]],
        box_size=[config["size_x"], config["size_y"], config["size_z"]],
    )
    v.dock(exhaustiveness=exhaustiveness, n_poses=1)
    energies = v.energies(n_poses=1)
    v.write_poses(str(out_pdbqt), n_poses=1, overwrite=True)
    score = float(energies[0][0])
    log_path.write_text(f"1 {score:.6f} 0.000 0.000\n")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-dir", default="receptors")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    target = args.target.upper()
    config_path = Path(args.config_dir) / f"{target}.json"
    config = json.loads(config_path.read_text())

    rows = read_rows(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]

    output_fields = ["id", "smiles", "label", "target", "vina_score", "validation_score", "status", "seconds"]
    start_all = time.time()

    with Path(args.output).open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fields)
        writer.writeheader()
        f_out.flush()

        with tempfile.TemporaryDirectory(prefix=f"dock_{target.lower()}_") as tmpdir:
            tmp = Path(tmpdir)
            for idx, row in enumerate(rows):
                smiles = row.get("smiles", "").strip()
                mol_id = row.get("id") or row.get("name") or str(idx)
                label = row.get("label", "")
                t0 = time.time()

                result = {
                    "id": mol_id,
                    "smiles": smiles,
                    "label": label,
                    "target": target,
                    "vina_score": "",
                    "validation_score": "",
                    "status": "failed",
                    "seconds": "",
                }

                try:
                    sdf = tmp / f"lig_{idx}.sdf"
                    ligand_pdbqt = tmp / f"lig_{idx}.pdbqt"
                    out_pdbqt = tmp / f"lig_{idx}_out.pdbqt"
                    log_path = tmp / f"lig_{idx}.log"

                    if not write_sdf_from_smiles(smiles, sdf, seed=args.seed + idx):
                        result["status"] = "invalid_smiles_or_embed_failed"
                    else:
                        sdf_to_pdbqt(sdf, ligand_pdbqt)
                        score = run_vina(config, ligand_pdbqt, out_pdbqt, log_path, args.exhaustiveness, args.seed + idx)
                        if score is None:
                            score = parse_vina_log(log_path)
                        if score is None or not math.isfinite(score):
                            result["status"] = "vina_score_missing"
                        else:
                            result["vina_score"] = score
                            result["validation_score"] = -score
                            result["status"] = "ok"
                except Exception as exc:
                    message = str(exc).replace("\n", " ")[:80]
                    result["status"] = f"error:{type(exc).__name__}:{message}"

                result["seconds"] = round(time.time() - t0, 3)
                writer.writerow(result)
                f_out.flush()

    print(f"Docked {len(rows)} rows in {time.time() - start_all:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
