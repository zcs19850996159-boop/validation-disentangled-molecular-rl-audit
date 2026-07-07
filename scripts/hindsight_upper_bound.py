#!/usr/bin/env python
"""
Hindsight/oracle upper-bound analysis for dynamic-weight controller runs.

This is a posthoc diagnostic, not a counterfactual simulator. It cannot answer
what molecules REINVENT would have generated under unseen weights. Instead, it
asks how much room is visible in the already observed trajectories:

1. Best observed run.
2. Oracle window switcher: at each time window, pick the best observed run.
3. Oracle molecule pool: pick the best observed molecules from a group.

If even these optimistic observed-data ceilings are close to static equal
weight, controller redesign is unlikely to be worth more compute in this setup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_model(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def to_float(value: str | None) -> float | None:
    try:
        if value in ("", None):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def gmean(values) -> float | None:
    vals = [max(0.0, float(v)) for v in values if v is not None and math.isfinite(float(v))]
    if not vals or any(v <= 0.0 for v in vals):
        return 0.0 if vals else None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


class ValidationScorer:
    def __init__(self, model_path: Path):
        self.payload = load_model(model_path)
        self.cache: dict[str, float | None] = {}

    def fingerprint(self, smiles: str):
        import numpy as np
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        arr = np.zeros((1, int(self.payload["n_bits"])), dtype="float32")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=int(self.payload["radius"]),
            nBits=int(self.payload["n_bits"]),
        )
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr

    def activity(self, smiles: str) -> float | None:
        if smiles in self.cache:
            return self.cache[smiles]
        x = self.fingerprint(smiles)
        if x is None:
            self.cache[smiles] = None
            return None
        model = self.payload["model"]
        if hasattr(model, "predict_proba"):
            value = float(model.predict_proba(x)[0, 1])
        else:
            raw = float(model.decision_function(x)[0])
            value = 1.0 / (1.0 + math.exp(-raw))
        self.cache[smiles] = value if math.isfinite(value) else None
        return self.cache[smiles]


class PropertyScorer:
    def __init__(self):
        self.sa_cache: dict[str, tuple[float, float] | None] = {}
        self.qed_cache: dict[str, float | None] = {}
        self.mw_score_cache: dict[str, float | None] = {}

    def exact_sa(self, smiles: str) -> tuple[float, float] | None:
        if smiles in self.sa_cache:
            return self.sa_cache[smiles]
        from sa_score_utils import exact_sa_raw_from_mol, mol_from_smiles, sa_reward_from_raw

        mol = mol_from_smiles(smiles)
        if mol is None:
            self.sa_cache[smiles] = None
            return None
        raw = exact_sa_raw_from_mol(mol)
        reward = sa_reward_from_raw(raw)
        self.sa_cache[smiles] = (reward, raw)
        return self.sa_cache[smiles]

    def qed(self, smiles: str) -> float | None:
        if smiles in self.qed_cache:
            return self.qed_cache[smiles]
        from rdkit import Chem, RDLogger
        from rdkit.Chem import QED

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
        mol = Chem.MolFromSmiles(smiles)
        self.qed_cache[smiles] = None if mol is None else float(QED.qed(mol))
        return self.qed_cache[smiles]

    def mw_score(self, smiles: str) -> float | None:
        if smiles in self.mw_score_cache:
            return self.mw_score_cache[smiles]
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors

        RDLogger.DisableLog("rdApp.warning")
        RDLogger.DisableLog("rdApp.error")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.mw_score_cache[smiles] = None
        else:
            mw = float(Descriptors.MolWt(mol))
            self.mw_score_cache[smiles] = 1.0 if 200.0 <= mw <= 500.0 else 0.0
        return self.mw_score_cache[smiles]


def parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise RuntimeError("--run must be NAME=CSV_PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def read_run(path: Path, name: str, setting: str, activity: ValidationScorer, props: PropertyScorer) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get("SMILES") or "").strip()
            if not smiles or row.get("SMILES_state") != "1":
                continue
            step = int(float(row.get("step") or 0))
            val_activity = activity.activity(smiles)
            if val_activity is None:
                continue

            item = {
                "run": name,
                "smiles": smiles,
                "step": step,
                "validation_activity": val_activity,
                "training_score": to_float(row.get("Score")),
            }

            if setting == "activity_sa":
                sa = props.exact_sa(smiles)
                if sa is None:
                    continue
                sa_reward, raw_sa = sa
                item.update(
                    {
                        "validation_sa": sa_reward,
                        "validation_raw_sa": raw_sa,
                        "primary": gmean([val_activity, sa_reward]),
                    }
                )
            elif setting == "drd2_qed_mw":
                qed = to_float(row.get("QED (raw)") or row.get("QED"))
                mw_score = to_float(row.get("Molecular weight"))
                if qed is None:
                    qed = props.qed(smiles)
                if mw_score is None:
                    mw_score = props.mw_score(smiles)
                if qed is None or mw_score is None:
                    continue
                item.update(
                    {
                        "validation_qed": qed,
                        "validation_mw_score": mw_score,
                        "primary": gmean([val_activity, qed, mw_score]),
                    }
                )
            else:
                raise ValueError(f"Unknown setting: {setting}")

            rows.append(item)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n_rows": 0}
    final_step = max(row["step"] for row in rows)
    last50 = [row for row in rows if row["step"] >= final_step - 49]
    final = [row for row in rows if row["step"] == final_step]
    by_primary = sorted(rows, key=lambda row: row["primary"], reverse=True)
    return {
        "n_rows": len(rows),
        "final_step": final_step,
        "all_mean_primary": mean([row["primary"] for row in rows]),
        "last50_mean_primary": mean([row["primary"] for row in last50]),
        "final_step_mean_primary": mean([row["primary"] for row in final]),
        "top10_mean_primary": mean([row["primary"] for row in by_primary[:10]]),
        "top100_mean_primary": mean([row["primary"] for row in by_primary[:100]]),
        "last50_mean_validation_activity": mean([row["validation_activity"] for row in last50]),
    }


def group_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("random"):
        return "random"
    if lowered.startswith("static"):
        return "static"
    return "real"


def window_table(runs: dict[str, list[dict]], window_size: int) -> dict[str, dict[int, float]]:
    table = {}
    for name, rows in runs.items():
        by_window = defaultdict(list)
        for row in rows:
            window = max(0, (int(row["step"]) - 1) // window_size)
            by_window[window].append(row["primary"])
        table[name] = {window: mean(values) for window, values in by_window.items()}
    return table


def oracle_window_switch(
    table: dict[str, dict[int, float]],
    candidate_names: list[str],
    baseline_names: list[str],
    last_n_windows: int | None = None,
) -> dict:
    candidate_windows = sorted({w for name in candidate_names for w in table.get(name, {})})
    if last_n_windows is not None:
        candidate_windows = candidate_windows[-last_n_windows:]

    oracle_values = []
    chosen = []
    baseline_values = []

    for window in candidate_windows:
        available = [
            (name, table[name][window])
            for name in candidate_names
            if window in table.get(name, {}) and table[name][window] is not None
        ]
        if not available:
            continue
        best_name, best_value = max(available, key=lambda item: item[1])
        oracle_values.append(best_value)
        chosen.append({"window": window, "run": best_name, "primary": best_value})

        base = [
            table[name][window]
            for name in baseline_names
            if window in table.get(name, {}) and table[name][window] is not None
        ]
        if base:
            baseline_values.append(mean(base))

    oracle_mean = mean(oracle_values)
    baseline_mean = mean(baseline_values)
    return {
        "n_windows": len(oracle_values),
        "oracle_mean_primary": oracle_mean,
        "baseline_mean_primary": baseline_mean,
        "delta_vs_baseline": None
        if oracle_mean is None or baseline_mean is None
        else oracle_mean - baseline_mean,
        "chosen": chosen,
    }


def pool_topk(runs: dict[str, list[dict]], names: list[str], ks: list[int]) -> dict:
    rows = []
    for name in names:
        rows.extend(runs.get(name, []))
    rows = sorted(rows, key=lambda row: row["primary"], reverse=True)
    result = {"n_rows": len(rows)}
    for k in ks:
        result[f"top{k}_mean_primary"] = mean([row["primary"] for row in rows[:k]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["activity_sa", "drd2_qed_mw"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    activity = ValidationScorer(Path(args.model))
    props = PropertyScorer()

    runs = {}
    for raw in args.run:
        name, path = parse_run_arg(raw)
        runs[name] = read_run(path, name, args.setting, activity, props)

    run_summaries = {name: summarize_rows(rows) for name, rows in runs.items()}
    names = list(runs)
    real_names = [name for name in names if group_name(name) == "real"]
    random_names = [name for name in names if group_name(name) == "random"]
    static_names = [name for name in names if group_name(name) == "static"]
    table = window_table(runs, args.window_size)

    best_observed = max(
        (
            (name, summary.get("last50_mean_primary"))
            for name, summary in run_summaries.items()
            if summary.get("last50_mean_primary") is not None
        ),
        key=lambda item: item[1],
    )

    result = {
        "setting": args.setting,
        "metric_note": (
            "primary is sqrt(validation_activity * validation_SA)"
            if args.setting == "activity_sa"
            else "primary is geometric mean(validation_activity, QED, MW_window_score)"
        ),
        "counterfactual_caveat": (
            "This is an observed-data upper bound, not a simulation of molecules "
            "that would have been generated under unseen weights."
        ),
        "groups": {
            "real": real_names,
            "random": random_names,
            "static": static_names,
        },
        "runs": run_summaries,
        "best_observed_run_by_last50_primary": {
            "run": best_observed[0],
            "last50_mean_primary": best_observed[1],
        },
        "oracle_window_switch": {
            "all_runs_vs_static_all_windows": oracle_window_switch(
                table,
                names,
                static_names,
            ),
            "real_runs_vs_static_all_windows": oracle_window_switch(
                table,
                real_names,
                static_names,
            ),
            "random_runs_vs_static_all_windows": oracle_window_switch(
                table,
                random_names,
                static_names,
            ),
            "all_runs_vs_static_last4_windows": oracle_window_switch(
                table,
                names,
                static_names,
                last_n_windows=4,
            ),
            "real_runs_vs_static_last4_windows": oracle_window_switch(
                table,
                real_names,
                static_names,
                last_n_windows=4,
            ),
            "random_runs_vs_static_last4_windows": oracle_window_switch(
                table,
                random_names,
                static_names,
                last_n_windows=4,
            ),
        },
        "oracle_molecule_pool": {
            "all": pool_topk(runs, names, [10, 100, 1000]),
            "real": pool_topk(runs, real_names, [10, 100, 1000]),
            "random": pool_topk(runs, random_names, [10, 100, 1000]),
            "static": pool_topk(runs, static_names, [10, 100, 1000]),
        },
    }

    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
