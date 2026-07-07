#!/usr/bin/env python
"""Triangulate RF reward, SVM validation, and Vina docking signals."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import pickle
import random
from pathlib import Path


def load_model(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def fingerprint(smiles: str, radius: int, n_bits: int):
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((1, n_bits), dtype="float32")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr[0])
    return arr


def predict_probability(payload: dict, smiles: str) -> float | None:
    x = fingerprint(smiles, radius=int(payload["radius"]), n_bits=int(payload["n_bits"]))
    if x is None:
        return None
    model = payload["model"]
    if hasattr(model, "predict_proba"):
        value = float(model.predict_proba(x)[0, 1])
    else:
        raw = float(model.decision_function(x)[0])
        value = 1.0 / (1.0 + math.exp(-raw))
    return value if math.isfinite(value) else None


def to_float(value: str | None) -> float | None:
    try:
        if value in ("", None):
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_vina_rows(path: Path, reward_payload: dict, validation_payload: dict) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("", None, "ok"):
                continue
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            vina_score = to_float(row.get("vina_score"))
            docking_reward = to_float(row.get("validation_score"))
            if docking_reward is None and vina_score is not None:
                docking_reward = -vina_score
            if docking_reward is None:
                continue
            rf = predict_probability(reward_payload, smiles)
            svm = predict_probability(validation_payload, smiles)
            if rf is None or svm is None:
                continue
            rows.append(
                {
                    "id": row.get("id", ""),
                    "smiles": smiles,
                    "label": row.get("label", ""),
                    "reward_rf": rf,
                    "validation_svm": svm,
                    "vina_score": vina_score,
                    "docking_reward": docking_reward,
                }
            )
    return rows


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return None
    return num / (da * db)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            result[indexed[k][0]] = rank
        i = j
    return result


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(ranks(a), ranks(b))


def percentile(values: list[float], q: float) -> float:
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def bootstrap_corr_ci(
    a: list[float],
    b: list[float],
    fn,
    iterations: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    values = []
    n = len(a)
    for _idx in range(iterations):
        indices = [rng.randrange(n) for _j in range(n)]
        value = fn([a[i] for i in indices], [b[i] for i in indices])
        if value is not None and math.isfinite(value):
            values.append(value)
    return {
        "iterations": iterations,
        "ci95_low": percentile(values, 0.025),
        "ci95_high": percentile(values, 0.975),
    }


def top_fraction_indices(values: list[float], fraction: float) -> set[int]:
    n = max(1, int(round(len(values) * fraction)))
    return {idx for idx, _value in sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:n]}


def pair_stats(
    rows: list[dict],
    x_key: str,
    y_key: str,
    top_fraction: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    x = [float(row[x_key]) for row in rows]
    y = [float(row[y_key]) for row in rows]
    top_x = top_fraction_indices(x, top_fraction)
    top_y = top_fraction_indices(y, top_fraction)
    union = top_x | top_y
    inter = top_x & top_y
    return {
        "x": x_key,
        "y": y_key,
        "n": len(rows),
        "pearson": pearson(x, y),
        "pearson_ci95": bootstrap_corr_ci(x, y, pearson, bootstrap_iterations, seed),
        "spearman": spearman(x, y),
        "spearman_ci95": bootstrap_corr_ci(x, y, spearman, bootstrap_iterations, seed + 1),
        "top_fraction": top_fraction,
        "top_fraction_size": len(top_x),
        "top_fraction_overlap_count": len(inter),
        "top_fraction_jaccard": len(inter) / len(union) if union else None,
    }


def write_scored_csv(path: Path, rows: list[dict]) -> None:
    fields = ["id", "label", "smiles", "reward_rf", "validation_svm", "vina_score", "docking_reward"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def write_scatter_svg(path: Path, rows: list[dict], x_key: str, y_key: str, title: str) -> None:
    width, height = 640, 480
    left, right, top, bottom = 72, 24, 38, 62
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-family="Arial" font-size="16">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width / 2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="12">{html.escape(x_key)}</text>',
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height / 2:.1f})" font-family="Arial" font-size="12">{html.escape(y_key)}</text>',
        f'<text x="{left}" y="{height-42}" text-anchor="start" font-family="Arial" font-size="10">{xmin:.3f}</text>',
        f'<text x="{width-right}" y="{height-42}" text-anchor="end" font-family="Arial" font-size="10">{xmax:.3f}</text>',
        f'<text x="{left-8}" y="{height-bottom}" text-anchor="end" font-family="Arial" font-size="10">{ymin:.3f}</text>',
        f'<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="Arial" font-size="10">{ymax:.3f}</text>',
    ]
    for row in rows:
        x = scale(float(row[x_key]), xmin, xmax, left, width - right)
        y = scale(float(row[y_key]), ymin, ymax, height - bottom, top)
        color = "#1f77b4" if row.get("label") == "active" else "#d95f02"
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.3" fill="{color}" fill-opacity="0.72"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-model", required=True)
    parser.add_argument("--validation-model", required=True)
    parser.add_argument("--vina-scores", required=True)
    parser.add_argument("--output-json", default="qsar_vina_triangulation_DRD2_7DFP.json")
    parser.add_argument("--output-csv", default="qsar_vina_triangulation_DRD2_7DFP_scored.csv")
    parser.add_argument("--plot-prefix", default="qsar_vina_triangulation")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=909)
    args = parser.parse_args()

    reward_payload = load_model(Path(args.reward_model))
    validation_payload = load_model(Path(args.validation_model))
    rows = read_vina_rows(Path(args.vina_scores), reward_payload, validation_payload)
    if len(rows) < 3:
        raise RuntimeError("Not enough scored rows for triangulation")

    pairs = {
        "rf_reward_vs_svm_validation": pair_stats(
            rows, "reward_rf", "validation_svm", args.top_fraction, args.bootstrap_iterations, args.seed
        ),
        "rf_reward_vs_vina_docking_reward": pair_stats(
            rows, "reward_rf", "docking_reward", args.top_fraction, args.bootstrap_iterations, args.seed + 100
        ),
        "svm_validation_vs_vina_docking_reward": pair_stats(
            rows, "validation_svm", "docking_reward", args.top_fraction, args.bootstrap_iterations, args.seed + 200
        ),
    }

    output_csv = Path(args.output_csv)
    write_scored_csv(output_csv, rows)
    plot_paths = {
        "rf_vs_svm": f"{args.plot_prefix}_RF_vs_SVM.svg",
        "rf_vs_vina": f"{args.plot_prefix}_RF_vs_Vina.svg",
        "svm_vs_vina": f"{args.plot_prefix}_SVM_vs_Vina.svg",
    }
    write_scatter_svg(Path(plot_paths["rf_vs_svm"]), rows, "reward_rf", "validation_svm", "RF reward vs SVM validation")
    write_scatter_svg(Path(plot_paths["rf_vs_vina"]), rows, "reward_rf", "docking_reward", "RF reward vs Vina reward")
    write_scatter_svg(Path(plot_paths["svm_vs_vina"]), rows, "validation_svm", "docking_reward", "SVM validation vs Vina reward")

    labels = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1

    result = {
        "n_scored": len(rows),
        "label_counts": labels,
        "vina_score_direction": "docking_reward = -vina_score, so higher is better",
        "pairs": pairs,
        "scored_csv": str(output_csv),
        "plots": plot_paths,
        "interpretation": (
            "RF/SVM are QSAR proxies; Vina is a docking signal that passed redocking "
            "but failed the practical active-vs-negative AUROC gate. Correlations "
            "therefore quantify signal alignment, not ground-truth validity."
        ),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
