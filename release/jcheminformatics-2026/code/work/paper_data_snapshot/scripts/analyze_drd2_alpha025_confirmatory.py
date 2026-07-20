#!/usr/bin/env python
"""Frozen final evaluation for the DRD2 alpha=0.25 confirmatory experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import statistics
from pathlib import Path


SEEDS = list(range(6101, 6111))
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
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scaffold_split(rows: list[dict[str, str]], seed: int):
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["scaffold"], []).append(row)
    items = list(groups.values())
    for trial in range(200):
        random.Random(seed + trial).shuffle(items)
        items.sort(key=len, reverse=True)
        out = [[], [], []]
        targets = [0.6 * len(rows), 0.2 * len(rows), 0.2 * len(rows)]
        for group in items:
            out[min(range(3), key=lambda i: len(out[i]) / targets[i])].extend(group)
        if all(len({int(row["label"]) for row in part}) == 2 for part in out):
            return out, seed + trial
    raise RuntimeError("Could not reproduce frozen scaffold split")


def arm_names() -> list[str]:
    names = [f"real_seed{seed}" for seed in SEEDS]
    names.extend(
        f"circular_r{replicate}_seed{seed}"
        for seed in SEEDS
        for replicate in range(3)
    )
    names.extend(
        f"cross_seed{seed}_donor{SEEDS[(index + 1) % len(SEEDS)]}"
        for index, seed in enumerate(SEEDS)
    )
    names.extend(f"frozen_equal_seed{seed}" for seed in SEEDS)
    names.extend(f"frozen_tuned_seed{seed}" for seed in SEEDS)
    return names


def verify_runs(root: Path, names: list[str]) -> dict[str, dict]:
    manifests = {}
    for arm in names:
        manifest_path = root / f"{arm}.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if arm.startswith("real_seed"):
            expected_seed = int(arm.removeprefix("real_seed"))
            identity_ok = manifest["arm"] == "real" and manifest["seed"] == expected_seed
        else:
            identity_ok = manifest["arm"] == arm
        if not identity_ok or manifest["final_step"] != 300:
            raise RuntimeError(f"Invalid manifest for {arm}")
        expected = {
            "csv_sha256": root / f"{arm}_1.csv",
            "history_sha256": root / f"{arm}.history.jsonl",
            "checkpoint_sha256": root / f"{arm}.stage1.chkpt",
        }
        for field, path in expected.items():
            if digest(path) != manifest[field]:
                raise RuntimeError(f"Hash mismatch for {arm}: {field}")
        manifests[arm] = manifest
    return manifests


def read_last50(root: Path, names: list[str]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    arm_rows: dict[str, list[str | None]] = {}
    molecule_properties: dict[str, dict[str, float | str]] = {}
    for arm in names:
        selected = []
        with (root / f"{arm}_1.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["step"]) < 251:
                    continue
                smiles = row["SMILES"].strip()
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is None:
                    selected.append(None)
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                selected.append(canonical)
                if canonical not in molecule_properties:
                    mw = float(Descriptors.MolWt(mol))
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "[ACYCLIC]"
                    molecule_properties[canonical] = {
                        "qed": float(QED.qed(mol)),
                        "mw_target_score": 1.0 if 200.0 <= mw <= 500.0 else 0.0,
                        "scaffold": scaffold,
                    }
        if not selected or len(selected) != 50 * 64:
            raise RuntimeError(f"Expected 3200 last-50 rows for {arm}; found {len(selected)}")
        arm_rows[arm] = selected
    return arm_rows, molecule_properties


def score_svm(canonical_smiles: list[str], model_path: Path) -> dict[str, float]:
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    payload = pickle.loads(model_path.read_bytes())
    model = payload["model"]
    radius, n_bits = int(payload["radius"]), int(payload["n_bits"])
    result = {}
    for start in range(0, len(canonical_smiles), 4096):
        chunk = canonical_smiles[start : start + 4096]
        matrix = np.zeros((len(chunk), n_bits), dtype="float32")
        for index, smiles in enumerate(chunk):
            fp = AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles), radius=radius, nBits=n_bits
            )
            DataStructs.ConvertToNumpyArray(fp, matrix[index])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(matrix)[:, 1]
        else:
            decision = model.decision_function(matrix)
            probabilities = 1.0 / (1.0 + np.exp(-decision))
        result.update((smiles, float(value)) for smiles, value in zip(chunk, probabilities))
        print(f"SVM {min(start + len(chunk), len(canonical_smiles))}/{len(canonical_smiles)}", flush=True)
    return result


def graph(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    src, dst, bond = [], [], []
    bond_types = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    for edge in mol.GetBonds():
        left, right = edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()
        edge_type = bond_types.get(str(edge.GetBondType()), 0)
        src += [left, right]
        dst += [right, left]
        bond += [edge_type, edge_type]
    return atoms, src, dst, bond


def score_mpn_and_ad(
    canonical_smiles: list[str],
    checkpoint_path: Path,
    training_smiles: list[str],
    device_name: str,
):
    import numpy as np
    import torch
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from torch import nn

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    class MPNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = nn.Embedding(119, 64)
            self.be = nn.Embedding(4, 64)
            self.selfs = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
            self.neis = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
            self.out = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 1)
            )

        def forward(self, batch):
            atoms, src, dst, bond, graph_index, n_graphs = batch
            h = self.e(atoms)
            for self_layer, neighbor_layer in zip(self.selfs, self.neis):
                aggregate = torch.zeros_like(h)
                if len(src):
                    aggregate.index_add_(0, dst, h[src] + self.be(bond))
                h = torch.relu(self_layer(h) + neighbor_layer(aggregate))
            pooled = torch.zeros((n_graphs, h.shape[1]), device=h.device)
            pooled.index_add_(0, graph_index, h)
            counts = torch.bincount(graph_index, minlength=n_graphs).clamp_min(1).unsqueeze(1)
            return self.out(pooled / counts).squeeze(1)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MPNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    temperature = float(checkpoint["temperature"])
    graphs = [graph(smiles) for smiles in canonical_smiles]
    probabilities = {}
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            current = graphs[start : start + batch_size]
            atoms, src, dst, bond, graph_index = [], [], [], [], []
            offset = 0
            for graph_id, (g_atoms, g_src, g_dst, g_bond) in enumerate(current):
                atoms.extend(g_atoms)
                src.extend(index + offset for index in g_src)
                dst.extend(index + offset for index in g_dst)
                bond.extend(g_bond)
                graph_index.extend([graph_id] * len(g_atoms))
                offset += len(g_atoms)
            tensors = (
                torch.tensor(atoms, dtype=torch.long, device=device),
                torch.tensor(src, dtype=torch.long, device=device),
                torch.tensor(dst, dtype=torch.long, device=device),
                torch.tensor(bond, dtype=torch.long, device=device),
                torch.tensor(graph_index, dtype=torch.long, device=device),
                len(current),
            )
            values = torch.sigmoid(model(tensors) / temperature).cpu().numpy()
            probabilities.update(
                (smiles, float(value))
                for smiles, value in zip(canonical_smiles[start : start + len(current)], values)
            )
            print(f"MPN {min(start + len(current), len(graphs))}/{len(graphs)}", flush=True)

    training_fps = [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
        for smiles in training_smiles
    ]
    ad_covered = {}
    max_similarity = {}
    for index, smiles in enumerate(canonical_smiles, start=1):
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)
        similarity = max(DataStructs.BulkTanimotoSimilarity(fp, training_fps))
        max_similarity[smiles] = float(similarity)
        ad_covered[smiles] = bool(similarity >= 0.30)
        if index % 10000 == 0:
            print(f"AD {index}/{len(canonical_smiles)}", flush=True)
    return probabilities, ad_covered, max_similarity


def summarize_arms(arm_rows, properties, svm, mpn, ad):
    summaries = {}
    for arm, rows in arm_rows.items():
        valid = [smiles for smiles in rows if smiles is not None]
        unique = set(valid)
        covered = [smiles for smiles in valid if ad[smiles]]
        scaffolds = {str(properties[smiles]["scaffold"]) for smiles in unique}
        summaries[arm] = {
            "arm": arm,
            "n_rows": len(rows),
            "n_valid": len(valid),
            "n_unique": len(unique),
            "n_unique_scaffolds": len(scaffolds),
            "svm_activity_mean": statistics.fmean(svm[smiles] for smiles in valid),
            "mpn_probability_all_valid_mean": statistics.fmean(mpn[smiles] for smiles in valid),
            "mpn_probability_ad_mean": statistics.fmean(mpn[smiles] for smiles in covered)
            if covered
            else None,
            "ad_coverage": len(covered) / len(valid),
            "qed_mean": statistics.fmean(float(properties[smiles]["qed"]) for smiles in valid),
            "mw_target_score_mean": statistics.fmean(
                float(properties[smiles]["mw_target_score"]) for smiles in valid
            ),
            "validity": len(valid) / len(rows),
            "uniqueness": len(unique) / len(valid),
            "scaffold_diversity": len(scaffolds) / len(valid),
            "scaffold_diversity_per_unique": len(scaffolds) / len(unique),
        }
    return summaries


def mean_endpoint(summaries, arms, endpoint):
    values = [summaries[arm][endpoint] for arm in arms]
    if any(value is None for value in values):
        return None
    return statistics.fmean(values)


def paired_statistics(differences: list[float], bootstrap_seed: int):
    import numpy as np
    from scipy.stats import binomtest

    array = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(array), size=(100000, len(array)))
    bootstrap = array[indices].mean(axis=1)
    positive = int((array > 0).sum())
    negative = int((array < 0).sum())
    nonzero = positive + negative
    return {
        "n": len(differences),
        "differences": differences,
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sample_sd": float(array.std(ddof=1)),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": int((array == 0).sum()),
        "exact_sign_test_two_sided_p": float(binomtest(positive, nonzero, 0.5).pvalue)
        if nonzero
        else 1.0,
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "bootstrap_resamples": 100000,
        "bootstrap_seed": bootstrap_seed,
        "practical_threshold": 0.015,
        "mean_exceeds_practical_threshold": bool(array.mean() >= 0.015),
    }


def build_contrasts(summaries):
    comparators = {
        "circular_mean": lambda seed, index: [
            f"circular_r{replicate}_seed{seed}" for replicate in range(3)
        ],
        "cross_seed": lambda seed, index: [
            f"cross_seed{seed}_donor{SEEDS[(index + 1) % len(SEEDS)]}"
        ],
        "frozen_equal": lambda seed, index: [f"frozen_equal_seed{seed}"],
        "frozen_tuned": lambda seed, index: [f"frozen_tuned_seed{seed}"],
    }
    paired_rows, results = [], {}
    for comparator, arm_fn in comparators.items():
        results[comparator] = {}
        for endpoint in ENDPOINTS:
            differences = []
            for index, seed in enumerate(SEEDS):
                real = summaries[f"real_seed{seed}"][endpoint]
                comparison = mean_endpoint(summaries, arm_fn(seed, index), endpoint)
                if real is None or comparison is None:
                    raise RuntimeError(f"Missing endpoint {endpoint} for seed {seed}")
                difference = float(real) - float(comparison)
                differences.append(difference)
                paired_rows.append(
                    {
                        "comparator": comparator,
                        "endpoint": endpoint,
                        "seed": seed,
                        "real": real,
                        "comparison": comparison,
                        "difference": difference,
                    }
                )
            results[comparator][endpoint] = paired_statistics(differences, 51001)
    return results, paired_rows


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(result: dict) -> str:
    contrasts = result["contrasts"]
    lines = [
        "# DRD2 alpha=0.25 Confirmatory Final Evaluation",
        "",
        "All endpoints use steps 251-300. Circular replicate indices 0, 1, and 2 are averaged "
        "within seed before paired inference. The analysis unit is the seed (n=10).",
        "",
        "| Comparator | Endpoint | Mean difference | 95% bootstrap CI | Positive | Sign-test p | >=0.015 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    display_endpoints = [
        "svm_activity_mean",
        "mpn_probability_all_valid_mean",
        "mpn_probability_ad_mean",
        "ad_coverage",
        "qed_mean",
        "validity",
        "uniqueness",
        "scaffold_diversity",
    ]
    for comparator, endpoint_results in contrasts.items():
        for endpoint in display_endpoints:
            stats = endpoint_results[endpoint]
            low, high = stats["bootstrap_95_ci"]
            lines.append(
                f"| {comparator} | {endpoint} | {stats['mean']:+.6f} | "
                f"[{low:+.6f}, {high:+.6f}] | {stats['positive_count']}/10 | "
                f"{stats['exact_sign_test_two_sided_p']:.6f} | "
                f"{'yes' if stats['mean_exceeds_practical_threshold'] else 'no'} |"
            )
    primary = contrasts["circular_mean"]
    lines.extend(
        [
            "",
            "## Frozen Primary Readout",
            "",
            f"Validation-SVM utilization (real - circular): `{primary['svm_activity_mean']['mean']:+.6f}`.",
            f"Untouched MPN all-valid (real - circular): "
            f"`{primary['mpn_probability_all_valid_mean']['mean']:+.6f}`.",
            f"Untouched MPN AD-covered (real - circular): "
            f"`{primary['mpn_probability_ad_mean']['mean']:+.6f}`.",
            "",
            "These values are reported without changing the predeclared +0.015 practical threshold. "
            "Interpretation should follow the confidence intervals and seed-level paired differences, "
            "not molecule-row pseudoreplication.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--oracle-protocol", required=True)
    parser.add_argument("--oracle-data", required=True)
    parser.add_argument("--oracle-checkpoint", required=True)
    parser.add_argument("--oracle-report", required=True)
    parser.add_argument("--svm-model", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = Path(args.results_root)
    names = arm_names()
    manifests = verify_runs(root, names)
    oracle_checkpoint = Path(args.oracle_checkpoint)
    oracle_report_path = Path(args.oracle_report)
    oracle_report = json.loads(oracle_report_path.read_text())
    if not oracle_report.get("passed"):
        raise RuntimeError("Frozen untouched oracle did not pass its calibration gate")
    if digest(oracle_checkpoint) != oracle_report["checkpoint_sha256"]:
        raise RuntimeError("Untouched oracle checkpoint hash mismatch")

    data_path = Path(args.oracle_data)
    oracle_rows = list(csv.DictReader(data_path.open(newline="")))
    splits, used_seed = scaffold_split(oracle_rows, int(oracle_report["split_seed"]))
    if used_seed != int(oracle_report["split_seed"]) or len(splits[0]) != oracle_report["n"]["train"]:
        raise RuntimeError("Could not reproduce untouched-oracle training split")

    arm_rows, properties = read_last50(root, names)
    canonical_smiles = sorted(properties)
    print(f"Unique valid molecules: {len(canonical_smiles)}", flush=True)
    svm = score_svm(canonical_smiles, Path(args.svm_model))
    mpn, ad, max_similarity = score_mpn_and_ad(
        canonical_smiles,
        oracle_checkpoint,
        [row["smiles"] for row in splits[0]],
        args.device,
    )
    summaries = summarize_arms(arm_rows, properties, svm, mpn, ad)
    contrasts, paired_rows = build_contrasts(summaries)

    prefix = Path(args.output_prefix)
    arm_summary_rows = [summaries[name] for name in names]
    write_csv(prefix.with_suffix(".arm_summaries.csv"), arm_summary_rows)
    write_csv(prefix.with_suffix(".paired_differences.csv"), paired_rows)
    result = {
        "analysis": {
            "last_steps": [251, 300],
            "n_seeds": 10,
            "bootstrap_resamples": 100000,
            "bootstrap_seed": 51001,
            "practical_threshold": 0.015,
            "circular_rule": "mean of replicate indices 0,1,2 within seed",
            "analysis_script_sha256": digest(Path(__file__)),
        },
        "provenance": {
            "protocol_sha256": digest(Path(args.protocol)),
            "oracle_protocol_sha256": digest(Path(args.oracle_protocol)),
            "oracle_data_sha256": digest(data_path),
            "oracle_checkpoint_sha256": digest(oracle_checkpoint),
            "oracle_report_sha256": digest(oracle_report_path),
            "svm_model_sha256": digest(Path(args.svm_model)),
            "n_verified_run_manifests": len(manifests),
        },
        "oracle_gate": oracle_report,
        "molecule_audit": {
            "n_unique_valid_across_all_arms": len(canonical_smiles),
            "ad_similarity_summary": {
                "min": min(max_similarity.values()),
                "median": statistics.median(max_similarity.values()),
                "max": max(max_similarity.values()),
            },
        },
        "arm_summaries": summaries,
        "contrasts": contrasts,
    }
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    prefix.with_suffix(".md").write_text(markdown_report(result))
    print(json.dumps({
        "json": str(json_path),
        "primary_svm": contrasts["circular_mean"]["svm_activity_mean"],
        "primary_mpn": contrasts["circular_mean"]["mpn_probability_all_valid_mean"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
