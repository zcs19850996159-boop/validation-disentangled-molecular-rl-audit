#!/usr/bin/env python
"""Export row-level records for the frozen DRD2 confirmatory tail pools."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path


ENDPOINTS = [
    "svm_activity_mean",
    "mpn_probability_all_valid_mean",
    "mpn_probability_ad_mean",
    "ad_coverage",
    "qed_mean",
    "mw_target_score_mean",
    "validity",
    "uniqueness",
    "scaffold_diversity",
    "scaffold_diversity_per_unique",
]
SUMMARY_TOLERANCE = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_analysis(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen analysis module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arm_metadata(run_id: str) -> dict[str, str | int]:
    if match := re.fullmatch(r"real_seed(\d+)", run_id):
        return {"arm": "real", "seed": int(match.group(1)), "circular_replicate": "", "donor_seed": ""}
    if match := re.fullmatch(r"circular_r(\d+)_seed(\d+)", run_id):
        return {
            "arm": "circular",
            "seed": int(match.group(2)),
            "circular_replicate": int(match.group(1)),
            "donor_seed": "",
        }
    if match := re.fullmatch(r"cross_seed(\d+)_donor(\d+)", run_id):
        return {
            "arm": "cross_seed",
            "seed": int(match.group(1)),
            "circular_replicate": "",
            "donor_seed": int(match.group(2)),
        }
    if match := re.fullmatch(r"frozen_(equal|tuned)_seed(\d+)", run_id):
        return {
            "arm": f"frozen_{match.group(1)}",
            "seed": int(match.group(2)),
            "circular_replicate": "",
            "donor_seed": "",
        }
    raise ValueError(f"Unknown run id: {run_id}")


def compare_summaries(observed: dict, frozen: dict) -> None:
    for run_id, expected in frozen.items():
        current = observed[run_id]
        for field in ("n_rows", "n_valid", "n_unique", "n_unique_scaffolds"):
            if current[field] != expected[field]:
                raise RuntimeError(
                    f"Frozen summary mismatch for {run_id}/{field}: "
                    f"{current[field]} != {expected[field]}"
                )
        for field in ENDPOINTS:
            left, right = current[field], expected[field]
            if left is None or right is None:
                if left != right:
                    raise RuntimeError(f"Frozen summary mismatch for {run_id}/{field}")
            elif abs(float(left) - float(right)) > SUMMARY_TOLERANCE:
                raise RuntimeError(
                    f"Frozen summary mismatch for {run_id}/{field}: {left} != {right}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-script", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--frozen-results", required=True)
    parser.add_argument("--oracle-data", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--svm-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    analysis_path = Path(args.analysis_script)
    analysis = load_frozen_analysis(analysis_path)
    root = Path(args.results_root)
    frozen_path = Path(args.frozen_results)
    frozen = json.loads(frozen_path.read_text())
    names = analysis.arm_names()
    manifests = analysis.verify_runs(root, names)

    oracle_report_path = Path(args.oracle_report)
    oracle_checkpoint = Path(args.oracle_checkpoint)
    oracle_report = json.loads(oracle_report_path.read_text())
    if not oracle_report.get("passed"):
        raise RuntimeError("Frozen untouched oracle did not pass its calibration gate")
    if analysis.digest(oracle_checkpoint) != oracle_report["checkpoint_sha256"]:
        raise RuntimeError("Untouched oracle checkpoint hash mismatch")

    oracle_data = Path(args.oracle_data)
    with oracle_data.open(newline="") as handle:
        oracle_rows = list(csv.DictReader(handle))
    splits, used_seed = analysis.scaffold_split(oracle_rows, int(oracle_report["split_seed"]))
    if used_seed != int(oracle_report["split_seed"]):
        raise RuntimeError("Could not reproduce frozen untouched-oracle split")

    arm_rows, properties = analysis.read_last50(root, names)
    canonical_smiles = sorted(properties)
    if len(canonical_smiles) != frozen["molecule_audit"]["n_unique_valid_across_all_arms"]:
        raise RuntimeError("Unique-valid molecule count differs from frozen analysis")
    print(f"Unique valid molecules: {len(canonical_smiles)}", flush=True)

    svm = analysis.score_svm(canonical_smiles, Path(args.svm_model))
    mpn, ad, max_similarity = analysis.score_mpn_and_ad(
        canonical_smiles,
        oracle_checkpoint,
        [row["smiles"] for row in splits[0]],
        args.device,
    )
    summaries = analysis.summarize_arms(arm_rows, properties, svm, mpn, ad)
    compare_summaries(summaries, frozen["arm_summaries"])
    print("Frozen arm summaries match exactly.", flush=True)

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecular_weight = {
        smiles: float(Descriptors.MolWt(Chem.MolFromSmiles(smiles))) for smiles in canonical_smiles
    }
    fields = [
        "run_id",
        "arm",
        "seed",
        "circular_replicate",
        "donor_seed",
        "step",
        "sample_index",
        "raw_smiles",
        "valid",
        "canonical_smiles",
        "svm_activity",
        "mpn_probability",
        "ad_covered",
        "max_training_tanimoto",
        "qed",
        "molecular_weight",
        "mw_target_score",
        "bemis_murcko_scaffold",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = valid_count = 0
    with gzip.open(output, "wt", newline="", compresslevel=9) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run_id in names:
            metadata = arm_metadata(run_id)
            per_step: dict[int, int] = {}
            with (root / f"{run_id}_1.csv").open(newline="") as source:
                for raw in csv.DictReader(source):
                    step = int(raw["step"])
                    if step < 251:
                        continue
                    sample_index = per_step.get(step, 0)
                    per_step[step] = sample_index + 1
                    raw_smiles = raw["SMILES"].strip()
                    mol = Chem.MolFromSmiles(raw_smiles) if raw_smiles else None
                    canonical = Chem.MolToSmiles(mol, canonical=True) if mol is not None else ""
                    valid = int(bool(canonical))
                    row = {
                        "run_id": run_id,
                        **metadata,
                        "step": step,
                        "sample_index": sample_index,
                        "raw_smiles": raw_smiles,
                        "valid": valid,
                        "canonical_smiles": canonical,
                        "svm_activity": format(svm[canonical], ".10g") if valid else "",
                        "mpn_probability": format(mpn[canonical], ".10g") if valid else "",
                        "ad_covered": int(ad[canonical]) if valid else "",
                        "max_training_tanimoto": format(max_similarity[canonical], ".10g") if valid else "",
                        "qed": format(float(properties[canonical]["qed"]), ".10g") if valid else "",
                        "molecular_weight": format(molecular_weight[canonical], ".10g") if valid else "",
                        "mw_target_score": format(float(properties[canonical]["mw_target_score"]), ".10g") if valid else "",
                        "bemis_murcko_scaffold": properties[canonical]["scaffold"] if valid else "",
                    }
                    writer.writerow(row)
                    row_count += 1
                    valid_count += valid

    if row_count != len(names) * 50 * 64:
        raise RuntimeError(f"Unexpected exported row count: {row_count}")
    metadata = {
        "description": "Row-level generated-molecule records for frozen steps 251-300.",
        "row_count": row_count,
        "valid_row_count": valid_count,
        "unique_valid_canonical_smiles": len(canonical_smiles),
        "n_runs": len(names),
        "n_verified_run_manifests": len(manifests),
        "tail_steps": [251, 300],
        "frozen_summary_match_tolerance": SUMMARY_TOLERANCE,
        "frozen_summary_match": True,
        "analysis_script_sha256": sha256(analysis_path),
        "frozen_results_sha256": sha256(frozen_path),
        "oracle_data_sha256": sha256(oracle_data),
        "oracle_checkpoint_sha256": sha256(oracle_checkpoint),
        "oracle_report_sha256": sha256(oracle_report_path),
        "svm_model_sha256": sha256(Path(args.svm_model)),
        "output_sha256": sha256(output),
        "columns": fields,
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
