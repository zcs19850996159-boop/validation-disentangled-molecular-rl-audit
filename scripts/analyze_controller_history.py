#!/usr/bin/env python
"""
Analyze DynamicWeightController JSONL history.

The goal is to distinguish a plausible late-stage controller trend from a
random walk. It produces a compact JSON summary plus dependency-free SVG curves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def linear_r2(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    slope = linear_slope(xs, ys)
    intercept = mean(ys) - slope * mean(xs)
    ss_tot = sum((y - mean(ys)) ** 2 for y in ys)
    if ss_tot == 0.0:
        return 1.0
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return max(0.0, 1.0 - ss_res / ss_tot)


def total_variation(values: list[float]) -> float:
    return sum(abs(curr - prev) for prev, curr in zip(values, values[1:]))


def split_windows(rows: list[dict]) -> dict[str, list[dict]]:
    n = len(rows)
    third = max(1, n // 3)
    return {
        "early": rows[:third],
        "middle": rows[third : 2 * third],
        "late": rows[2 * third :],
    }


def component_names(rows: list[dict]) -> list[str]:
    return list(rows[0]["new_weights"].keys()) if rows else []


def summarize(rows: list[dict]) -> dict:
    names = component_names(rows)
    xs = [float(row["validation_index"]) for row in rows]
    windows = split_windows(rows)

    summary: dict = {
        "n_validation_updates": len(rows),
        "first_training_step": rows[0]["training_step"],
        "last_training_step": rows[-1]["training_step"],
        "components": {},
        "controller_reward": {
            "mean": mean([float(row["controller_reward"]) for row in rows]),
            "stdev": stdev([float(row["controller_reward"]) for row in rows]),
            "positive_fraction": mean([1.0 if float(row["controller_reward"]) > 0.0 else 0.0 for row in rows]),
        },
    }

    for name in names:
        weights = [float(row["new_weights"][name]) for row in rows]
        validations = [float(row["validation_scores"].get(name, 0.0)) for row in rows]
        first_half = weights[: max(2, len(weights) // 2)]
        second_half = weights[len(weights) // 2 :]
        late_rows = windows["late"]
        late_xs = [float(row["validation_index"]) for row in late_rows]
        late_weights = [float(row["new_weights"][name]) for row in late_rows]
        late_validations = [float(row["validation_scores"].get(name, 0.0)) for row in late_rows]

        summary["components"][name] = {
            "initial_weight": float(rows[0]["old_weights"][name]),
            "final_weight": weights[-1],
            "min_weight": min(weights),
            "max_weight": max(weights),
            "overall_weight_slope_per_update": linear_slope(xs, weights),
            "overall_weight_r2": linear_r2(xs, weights),
            "late_weight_slope_per_update": linear_slope(late_xs, late_weights),
            "first_half_weight_total_variation": total_variation(first_half),
            "second_half_weight_total_variation": total_variation(second_half),
            "late_weight_range": max(late_weights) - min(late_weights),
            "initial_validation": validations[0],
            "final_validation": validations[-1],
            "late_validation_mean": mean(late_validations),
            "late_validation_slope_per_update": linear_slope(late_xs, late_validations),
            "window_mean_weights": {
                window_name: mean([float(row["new_weights"][name]) for row in window_rows])
                for window_name, window_rows in windows.items()
            },
            "window_mean_validation": {
                window_name: mean([float(row["validation_scores"].get(name, 0.0)) for row in window_rows])
                for window_name, window_rows in windows.items()
            },
        }

    return summary


def scale(values: list[float], low: float, high: float, size: float, invert: bool = False) -> list[float]:
    if high == low:
        coords = [size / 2.0 for _ in values]
    else:
        coords = [(value - low) / (high - low) * size for value in values]
    if invert:
        coords = [size - value for value in coords]
    return coords


def write_svg(path: Path, rows: list[dict], key: str, title: str):
    names = component_names(rows)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    width, height = 900, 420
    left, top, plot_w, plot_h = 70, 45, 760, 300
    xs_raw = [float(row["training_step"]) for row in rows]
    values_by_name = {
        name: [float(row[key][name] if key == "new_weights" else row[key].get(name, 0.0)) for row in rows]
        for name in names
    }
    all_values = [value for values in values_by_name.values() for value in values]
    y_min = min(all_values)
    y_max = max(all_values)
    margin = (y_max - y_min) * 0.08 or 0.1
    y_min -= margin
    y_max += margin
    xs = [left + value for value in scale(xs_raw, min(xs_raw), max(xs_raw), plot_w)]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="Arial" font-size="18" fill="#111">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<text x="{left}" y="{top + plot_h + 35}" font-family="Arial" font-size="12" fill="#333">training step</text>',
        f'<text x="10" y="{top + 15}" font-family="Arial" font-size="12" fill="#333">{key}</text>',
        f'<text x="{left - 55}" y="{top + 5}" font-family="Arial" font-size="11" fill="#666">{y_max:.3f}</text>',
        f'<text x="{left - 55}" y="{top + plot_h}" font-family="Arial" font-size="11" fill="#666">{y_min:.3f}</text>',
    ]

    for idx, name in enumerate(names):
        ys = [top + value for value in scale(values_by_name[name], y_min, y_max, plot_h, invert=True)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        color = colors[idx % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + 20 + idx * 22
        parts.append(f'<line x1="{left + plot_w + 20}" y1="{legend_y}" x2="{left + plot_w + 50}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + plot_w + 58}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#111">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.history))
    if not rows:
        raise RuntimeError("No controller history rows found")

    prefix = Path(args.output_prefix)
    summary = summarize(rows)
    prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_svg(prefix.with_suffix(".weights.svg"), rows, "new_weights", "Controller weight trajectory")
    write_svg(prefix.with_suffix(".validation.svg"), rows, "validation_scores", "Validation score trajectory")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
