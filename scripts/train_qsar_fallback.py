#!/usr/bin/env python
"""
Train independent QSAR fallback models for controller validation.

The intended pilot setup is:

- training reward proxy: RandomForest trained on one scaffold-disjoint split;
- controller validation proxy: SVM trained on a different scaffold-disjoint
  split;
- self-check: both models are evaluated on a third held-out scaffold split.

This is weaker evidence than docking, but it preserves the key design rule that
the controller must not be updated with the same signal that trains the agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MolRecord:
    row: dict[str, str]
    smiles: str
    label: int
    scaffold: str


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def label_from_row(row: dict[str, str]) -> int | None:
    label = (row.get("label") or "").strip().lower()
    if label in {"active", "1", "true"}:
        return 1
    if label in {"negative", "inactive", "decoy", "0", "false"}:
        return 0
    return None


def scaffold_smiles(mol) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "NO_SCAFFOLD"
    except Exception:
        return "NO_SCAFFOLD"


def make_records(rows: list[dict[str, str]]) -> list[MolRecord]:
    from rdkit import Chem

    records = []
    seen: set[str] = set()
    for row in rows:
        smiles = (row.get("smiles") or row.get("canonical_smiles") or "").strip()
        label = label_from_row(row)
        if not smiles or label is None or smiles in seen:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        seen.add(smiles)
        records.append(MolRecord(row=row, smiles=smiles, label=label, scaffold=scaffold_smiles(mol)))
    return records


def scaffold_split(
    records: list[MolRecord],
    reward_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[MolRecord]]:
    groups: dict[str, list[MolRecord]] = {}
    for record in records:
        groups.setdefault(record.scaffold, []).append(record)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    n_total = len(records)
    targets = {
        "reward_train": reward_fraction * n_total,
        "validation_train": validation_fraction * n_total,
        "test": (1.0 - reward_fraction - validation_fraction) * n_total,
    }
    splits: dict[str, list[MolRecord]] = {name: [] for name in targets}

    for _scaffold, group in group_items:
        split_name = min(
            splits,
            key=lambda name: len(splits[name]) / max(targets[name], 1.0),
        )
        splits[split_name].extend(group)

    return splits


def split_has_two_classes(records: list[MolRecord]) -> bool:
    return len({record.label for record in records}) == 2


def stratified_fallback_split(
    records: list[MolRecord],
    reward_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, list[MolRecord]]:
    by_label = {0: [], 1: []}
    rng = random.Random(seed)
    for record in records:
        by_label[record.label].append(record)
    for values in by_label.values():
        rng.shuffle(values)

    splits = {"reward_train": [], "validation_train": [], "test": []}
    for values in by_label.values():
        n = len(values)
        n_reward = max(1, int(round(reward_fraction * n)))
        n_validation = max(1, int(round(validation_fraction * n)))
        splits["reward_train"].extend(values[:n_reward])
        splits["validation_train"].extend(values[n_reward : n_reward + n_validation])
        splits["test"].extend(values[n_reward + n_validation :])
    return splits


def fingerprint_matrix(records: list[MolRecord], radius: int, n_bits: int):
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    x = np.zeros((len(records), n_bits), dtype="float32")
    y = np.array([record.label for record in records], dtype="int64")
    for i, record in enumerate(records):
        mol = Chem.MolFromSmiles(record.smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, x[i])
    return x, y


def positive_fraction(records: list[MolRecord]) -> float:
    return sum(record.label for record in records) / len(records) if records else 0.0


def safe_auc(y_true, y_score) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(int(v) for v in y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score) -> float | None:
    from sklearn.metrics import average_precision_score

    if len(set(int(v) for v in y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0.0 or den_b == 0.0:
        return None
    return float(num / (den_a * den_b))


def mean_abs_diff(a: list[float], b: list[float]) -> float | None:
    if not a or len(a) != len(b):
        return None
    return float(sum(abs(x - y) for x, y in zip(a, b)) / len(a))


def predict_probability(model, x) -> list[float]:
    if hasattr(model, "predict_proba"):
        return [float(v) for v in model.predict_proba(x)[:, 1]]
    scores = model.decision_function(x)
    return [1.0 / (1.0 + math.exp(-float(score))) for score in scores]


def write_split_csv(path: Path, splits: dict[str, list[MolRecord]]):
    fieldnames = ["split", "smiles", "label", "molecule_chembl_id", "pchembl_value", "scaffold"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, records in splits.items():
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "smiles": record.smiles,
                        "label": record.label,
                        "molecule_chembl_id": record.row.get("molecule_chembl_id", ""),
                        "pchembl_value": record.row.get("pchembl_value", ""),
                        "scaffold": record.scaffold,
                    }
                )


def dump_model(path: Path, payload: dict):
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--reward-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-validation-auc", type=float, default=0.60)
    parser.add_argument("--max-model-correlation", type=float, default=0.95)
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    target = args.target.upper()
    component_name = f"{target}_activity"
    out_dir = Path(args.output_dir or f"qsar_models/{target}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = make_records(read_rows(Path(args.input)))
    if len(records) < 30:
        raise RuntimeError(f"Too few usable labeled molecules: {len(records)}")

    splits = scaffold_split(records, args.reward_fraction, args.validation_fraction, args.seed)
    if not all(split_has_two_classes(values) for values in splits.values()):
        splits = stratified_fallback_split(records, args.reward_fraction, args.validation_fraction, args.seed)
        split_method = "stratified_random_fallback"
    else:
        split_method = "scaffold_disjoint"

    if not all(split_has_two_classes(values) for values in splits.values()):
        raise RuntimeError("Could not create train/validation/test splits with both classes")

    x_reward, y_reward = fingerprint_matrix(splits["reward_train"], args.radius, args.n_bits)
    x_validation, y_validation = fingerprint_matrix(splits["validation_train"], args.radius, args.n_bits)
    x_test, y_test = fingerprint_matrix(splits["test"], args.radius, args.n_bits)

    reward_model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    validation_model = SVC(
        C=3.0,
        gamma="scale",
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=args.seed,
    )

    reward_model.fit(x_reward, y_reward)
    validation_model.fit(x_validation, y_validation)

    reward_test = predict_probability(reward_model, x_test)
    validation_test = predict_probability(validation_model, x_test)

    summary = {
        "target": target,
        "component_name": component_name,
        "split_method": split_method,
        "n_total": len(records),
        "n_reward_train": len(splits["reward_train"]),
        "n_validation_train": len(splits["validation_train"]),
        "n_test": len(splits["test"]),
        "reward_train_positive_fraction": positive_fraction(splits["reward_train"]),
        "validation_train_positive_fraction": positive_fraction(splits["validation_train"]),
        "test_positive_fraction": positive_fraction(splits["test"]),
        "reward_model": "RandomForestClassifier",
        "validation_model": "SVC_RBF_probability",
        "reward_test_auc": safe_auc(y_test, reward_test),
        "validation_test_auc": safe_auc(y_test, validation_test),
        "reward_test_average_precision": safe_average_precision(y_test, reward_test),
        "validation_test_average_precision": safe_average_precision(y_test, validation_test),
        "reward_validation_prediction_pearson": pearson(reward_test, validation_test),
        "reward_validation_prediction_mean_abs_diff": mean_abs_diff(reward_test, validation_test),
        "min_validation_auc": args.min_validation_auc,
        "max_model_correlation": args.max_model_correlation,
    }
    summary["passed"] = bool(
        summary["validation_test_auc"] is not None
        and summary["validation_test_auc"] >= args.min_validation_auc
        and (
            summary["reward_validation_prediction_pearson"] is None
            or summary["reward_validation_prediction_pearson"] <= args.max_model_correlation
        )
    )

    common_payload = {
        "target": target,
        "component_name": component_name,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "source_csv": str(Path(args.input)),
        "split_method": split_method,
    }
    dump_model(out_dir / "reward_rf.pkl", {**common_payload, "model": reward_model, "role": "training_reward"})
    dump_model(out_dir / "validation_svm.pkl", {**common_payload, "model": validation_model, "role": "controller_validation"})
    write_split_csv(out_dir / "splits.csv", splits)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 2)


if __name__ == "__main__":
    main()
