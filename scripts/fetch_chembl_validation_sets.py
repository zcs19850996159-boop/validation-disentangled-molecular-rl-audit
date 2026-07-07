#!/usr/bin/env python
"""
Fetch simple ChEMBL active/low-activity sets for Vina oracle calibration.

These files are for validation-oracle sanity checks only. They are not the
training reward data and should not be used to fit the fast QSAR reward model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


BASE_URL = "https://www.ebi.ac.uk"


@dataclass(frozen=True)
class Target:
    name: str
    chembl_id: str
    description: str


TARGETS = {
    "DRD2": Target("DRD2", "CHEMBL217", "D(2) dopamine receptor, Homo sapiens"),
    "GSK3B": Target("GSK3B", "CHEMBL262", "Glycogen synthase kinase-3 beta, Homo sapiens"),
    "JNK3": Target("JNK3", "CHEMBL2637", "Mitogen-activated protein kinase 10/JNK3, Homo sapiens"),
}


def open_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def chembl_activity_pages(target: Target, limit: int, max_pages: int):
    params = urllib.parse.urlencode(
        {
            "target_chembl_id": target.chembl_id,
            "limit": limit,
        }
    )
    url = f"{BASE_URL}/chembl/api/data/activity.json?{params}"
    pages = 0
    while url and pages < max_pages:
        data = open_json(url)
        yield data.get("activities", [])
        next_url = data.get("page_meta", {}).get("next")
        url = BASE_URL + next_url if next_url else None
        pages += 1
        time.sleep(0.1)


def usable_activity(row: dict, assay_types: set[str], standard_types: set[str]) -> tuple[str, float] | None:
    smiles = (row.get("canonical_smiles") or "").strip()
    if not smiles:
        return None

    if assay_types and row.get("assay_type") not in assay_types:
        return None

    standard_type = (row.get("standard_type") or "").upper()
    if standard_types and standard_type not in standard_types:
        return None

    try:
        pchembl = float(row.get("pchembl_value"))
    except (TypeError, ValueError):
        return None

    return smiles, pchembl


def collect_sets(
    target: Target,
    active_threshold: float,
    negative_threshold: float,
    max_per_class: int,
    assay_types: set[str],
    standard_types: set[str],
    limit: int,
    max_pages: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    best_by_smiles: dict[str, dict[str, object]] = {}

    for page in chembl_activity_pages(target, limit=limit, max_pages=max_pages):
        for row in page:
            usable = usable_activity(row, assay_types, standard_types)
            if usable is None:
                continue

            smiles, pchembl = usable
            old = best_by_smiles.get(smiles)
            if old is None or pchembl > float(old["pchembl_value"]):
                best_by_smiles[smiles] = {
                    "smiles": smiles,
                    "pchembl_value": pchembl,
                    "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                    "standard_type": row.get("standard_type", ""),
                    "standard_value": row.get("standard_value", ""),
                    "standard_units": row.get("standard_units", ""),
                    "assay_type": row.get("assay_type", ""),
                    "target_chembl_id": target.chembl_id,
                }

    active = [r for r in best_by_smiles.values() if float(r["pchembl_value"]) >= active_threshold]
    negative = [r for r in best_by_smiles.values() if float(r["pchembl_value"]) <= negative_threshold]
    active.sort(key=lambda r: float(r["pchembl_value"]), reverse=True)
    negative.sort(key=lambda r: float(r["pchembl_value"]))
    return active[:max_per_class], negative[:max_per_class]


def write_smi(path: Path, rows: list[dict[str, object]]):
    with path.open("w", newline="") as f:
        for row in rows:
            f.write(f"{row['smiles']} {row['molecule_chembl_id']}\n")


def write_csv(path: Path, actives: list[dict[str, object]], negatives: list[dict[str, object]]):
    fieldnames = [
        "id",
        "smiles",
        "label",
        "pchembl_value",
        "molecule_chembl_id",
        "standard_type",
        "standard_value",
        "standard_units",
        "assay_type",
        "target_chembl_id",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in [("active", actives), ("negative", negatives)]:
            for i, row in enumerate(rows):
                out = {k: row.get(k, "") for k in fieldnames}
                out["id"] = f"{label}_{i}"
                out["label"] = label
                writer.writerow(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--output-dir", default="data/chembl_validation")
    parser.add_argument("--active-threshold", type=float, default=7.0)
    parser.add_argument("--negative-threshold", type=float, default=5.0)
    parser.add_argument("--max-per-class", type=int, default=256)
    parser.add_argument("--assay-types", default="B")
    parser.add_argument("--standard-types", default="IC50,KI,KD,EC50")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=60)
    args = parser.parse_args()

    target = TARGETS[args.target]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assay_types = {x.strip().upper() for x in args.assay_types.split(",") if x.strip()}
    standard_types = {x.strip().upper() for x in args.standard_types.split(",") if x.strip()}

    actives, negatives = collect_sets(
        target,
        active_threshold=args.active_threshold,
        negative_threshold=args.negative_threshold,
        max_per_class=args.max_per_class,
        assay_types=assay_types,
        standard_types=standard_types,
        limit=args.limit,
        max_pages=args.max_pages,
    )

    prefix = out_dir / target.name
    write_smi(prefix.with_name(f"{target.name}_actives.smi"), actives)
    write_smi(prefix.with_name(f"{target.name}_negatives.smi"), negatives)
    write_csv(prefix.with_name(f"{target.name}_chembl_validation.csv"), actives, negatives)

    summary = {
        "target": target.name,
        "target_chembl_id": target.chembl_id,
        "description": target.description,
        "n_actives": len(actives),
        "n_negatives": len(negatives),
        "active_threshold": args.active_threshold,
        "negative_threshold": args.negative_threshold,
        "assay_types": ",".join(sorted(assay_types)),
        "standard_types": ",".join(sorted(standard_types)),
    }
    (prefix.with_name(f"{target.name}_chembl_validation_summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
