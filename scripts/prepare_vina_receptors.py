#!/usr/bin/env python
"""
Prepare reproducible receptor/grid metadata for Vina validation.

This script downloads the locked PDB structures, strips a protein chain into a
clean receptor PDB, extracts the co-crystal ligand coordinates, and writes a
target JSON containing the deterministic docking box.

It intentionally does not pretend receptor preparation is fully solved. Review
the generated receptor PDBQT before using it in experiments, especially
protonation and protonation-dependent side chains.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetSpec:
    target: str
    pdb_id: str
    protein_chain: str
    ligand_chain: str
    ligand_resname: str
    default_altloc: str
    description: str
    source_url: str


TARGETS = {
    "DRD2": TargetSpec(
        target="DRD2",
        pdb_id="6CM4",
        protein_chain="A",
        ligand_chain="A",
        ligand_resname="8NU",
        default_altloc="A",
        description="D2 dopamine receptor bound to risperidone",
        source_url="https://www.rcsb.org/structure/6CM4",
    ),
    "DRD2_7DFP": TargetSpec(
        target="DRD2_7DFP",
        pdb_id="7DFP",
        protein_chain="A",
        ligand_chain="A",
        ligand_resname="SIP",
        default_altloc="A",
        description="Human D2 dopamine receptor in complex with spiperone",
        source_url="https://www.rcsb.org/structure/7DFP",
    ),
    "GSK3B": TargetSpec(
        target="GSK3B",
        pdb_id="1Q5K",
        protein_chain="A",
        ligand_chain="A",
        ligand_resname="TMU",
        default_altloc="A",
        description="GSK3 beta in complex with AR-A014418-like inhibitor",
        source_url="https://www.rcsb.org/structure/1Q5K",
    ),
    "JNK3": TargetSpec(
        target="JNK3",
        pdb_id="4WHZ",
        protein_chain="A",
        ligand_chain="A",
        ligand_resname="3NL",
        default_altloc="A",
        description="JNK3 in complex with aminopyrazole inhibitor 26k",
        source_url="https://www.rcsb.org/structure/4WHZ",
    ),
}


def download_pdb(pdb_id: str, out_path: Path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())


def atom_xyz(line: str) -> tuple[float, float, float]:
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def keep_altloc(line: str, default_altloc: str) -> str | None:
    altloc = line[16].strip()
    if altloc and altloc != default_altloc:
        return None
    if altloc == default_altloc:
        return line[:16] + " " + line[17:]
    return line


def strip_receptor_and_ligand(
    pdb_path: Path,
    receptor_pdb: Path,
    ligand_pdb: Path,
    protein_chain: str,
    ligand_chain: str,
    ligand_resname: str,
    default_altloc: str,
) -> list[tuple[float, float, float]]:
    ligand_xyz = []
    fallback_ligand_lines = []

    with pdb_path.open() as src, receptor_pdb.open("w") as receptor, ligand_pdb.open("w") as ligand:
        for line in src:
            record = line[:6].strip()
            chain = line[21].strip()
            resname = line[17:20].strip()

            if record == "ATOM" and chain == protein_chain:
                kept = keep_altloc(line, default_altloc)
                if kept is not None:
                    receptor.write(kept)
            elif record == "HETATM" and resname == ligand_resname and chain == ligand_chain:
                kept = keep_altloc(line, default_altloc)
                if kept is not None:
                    ligand.write(kept)
                    ligand_xyz.append(atom_xyz(kept))
            elif record == "HETATM" and resname == ligand_resname:
                kept = keep_altloc(line, default_altloc)
                if kept is not None:
                    fallback_ligand_lines.append(kept)

        receptor.write("TER\nEND\n")

        if not ligand_xyz and fallback_ligand_lines:
            for line in fallback_ligand_lines:
                ligand.write(line)
                ligand_xyz.append(atom_xyz(line))
        ligand.write("END\n")

    if not ligand_xyz:
        raise RuntimeError(f"No ligand atoms found for residue {ligand_resname} in {pdb_path}")

    return ligand_xyz


def compute_box(
    ligand_xyz: list[tuple[float, float, float]],
    padding: float,
    min_size: float,
) -> tuple[dict[str, float], dict[str, float]]:
    xs, ys, zs = zip(*ligand_xyz)
    center = {
        "center_x": sum(xs) / len(xs),
        "center_y": sum(ys) / len(ys),
        "center_z": sum(zs) / len(zs),
    }
    size = {
        "size_x": max(min_size, max(xs) - min(xs) + padding),
        "size_y": max(min_size, max(ys) - min(ys) + padding),
        "size_z": max(min_size, max(zs) - min(zs) + padding),
    }
    return center, size


def try_meeko_receptor_to_pdbqt(
    receptor_pdb: Path,
    receptor_pdbqt: Path,
    center: dict[str, float],
    size: dict[str, float],
    default_altloc: str,
) -> dict[str, object]:
    log_path = receptor_pdbqt.with_suffix(".prep.log")
    if shutil.which("mk_prepare_receptor.py") is None:
        log_path.write_text("mk_prepare_receptor.py not found on PATH\n")
        return {"ok": False, "tool": "meeko", "log_path": str(log_path), "returncode": None}

    basename = receptor_pdbqt.with_suffix("")
    cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb",
        str(receptor_pdb),
        "-o",
        str(basename),
        "-p",
        str(receptor_pdbqt),
        "--box_center",
        str(center["center_x"]),
        str(center["center_y"]),
        str(center["center_z"]),
        "--box_size",
        str(size["size_x"]),
        str(size["size_y"]),
        str(size["size_z"]),
        "--default_altloc",
        default_altloc,
        "--allow_bad_res",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_text = (
        "$ " + " ".join(cmd) + "\n\n"
        + "STDOUT:\n" + result.stdout + "\n\n"
        + "STDERR:\n" + result.stderr + "\n"
    )
    log_path.write_text(log_text)
    ok = result.returncode == 0 and receptor_pdbqt.exists() and receptor_pdbqt.stat().st_size > 0
    deleted_residues = sorted(
        {
            residue
            for match in re.findall(r"Template matching failed for: \[([^\]]*)\]", log_text)
            for residue in re.findall(r"'([^']+)'", match)
        }
    )
    return {
        "ok": ok,
        "tool": "meeko",
        "log_path": str(log_path),
        "returncode": result.returncode,
        "allow_bad_res": True,
        "default_altloc": default_altloc,
        "deleted_residue_count": len(deleted_residues),
        "deleted_residues_preview": deleted_residues[:20],
    }


def write_config(
    spec: TargetSpec,
    out_dir: Path,
    receptor_pdb: Path,
    receptor_pdbqt: Path,
    ligand_pdb: Path,
    center: dict[str, float],
    size: dict[str, float],
    prep: dict[str, object],
):
    config = {
        "target": spec.target,
        "pdb_id": spec.pdb_id,
        "description": spec.description,
        "source_url": spec.source_url,
        "protein_chain": spec.protein_chain,
        "ligand_chain": spec.ligand_chain,
        "default_altloc": spec.default_altloc,
        "co_crystal_ligand": spec.ligand_resname,
        "receptor_pdb": str(receptor_pdb),
        "receptor_pdbqt": str(receptor_pdbqt),
        "receptor_pdbqt_generated": bool(prep.get("ok")),
        "receptor_preparation": prep,
        "co_crystal_ligand_pdb": str(ligand_pdb),
        "box_rule": "center = co-crystal ligand centroid; size = ligand span + padding, with minimum size",
        "padding_angstrom": 12.0,
        "min_box_size_angstrom": 22.0,
        **{k: round(v, 3) for k, v in center.items()},
        **{k: round(v, 3) for k, v in size.items()},
        "vina_defaults": {
            "exhaustiveness": 8,
            "num_modes": 1,
            "seed": 17,
        },
        "review_required": [
            "Confirm protonation states at docking pH.",
            "Confirm receptor PDBQT hydrogens and Gasteiger charges.",
            "For GPCR DRD2, review conserved Asp and binding-site protonation.",
            "Keep this JSON unchanged after pilot calibration unless the receptor is intentionally re-prepared.",
        ],
    }
    (out_dir / f"{spec.target}.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="DRD2,GSK3B,JNK3")
    parser.add_argument("--out-dir", default="receptors")
    parser.add_argument("--padding", type=float, default=12.0)
    parser.add_argument("--min-size", type=float, default=22.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for target in [t.strip().upper() for t in args.targets.split(",") if t.strip()]:
        spec = TARGETS[target]
        target_dir = out_dir / target
        target_dir.mkdir(exist_ok=True)

        raw_pdb = target_dir / f"{spec.pdb_id}.pdb"
        receptor_pdb = target_dir / f"{target}_receptor_chain_{spec.protein_chain}.pdb"
        receptor_pdbqt = target_dir / f"{target}_receptor.pdbqt"
        ligand_pdb = target_dir / f"{target}_{spec.ligand_resname}_cocrystal_ligand.pdb"

        download_pdb(spec.pdb_id, raw_pdb)
        ligand_xyz = strip_receptor_and_ligand(
            raw_pdb,
            receptor_pdb,
            ligand_pdb,
            spec.protein_chain,
            spec.ligand_chain,
            spec.ligand_resname,
            spec.default_altloc,
        )
        center, size = compute_box(ligand_xyz, padding=args.padding, min_size=args.min_size)
        prep = try_meeko_receptor_to_pdbqt(
            receptor_pdb,
            receptor_pdbqt,
            center,
            size,
            spec.default_altloc,
        )
        write_config(spec, out_dir, receptor_pdb, receptor_pdbqt, ligand_pdb, center, size, prep)

        status = "with receptor PDBQT" if prep.get("ok") else "without receptor PDBQT"
        print(f"{target}: wrote {out_dir / (target + '.json')} ({status})")


if __name__ == "__main__":
    main()
