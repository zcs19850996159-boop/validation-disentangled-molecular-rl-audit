#!/usr/bin/env python3
"""Render manuscript figures from frozen paper_data tables only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["pdf.fonttype"] = 42

FINAL_WIDTH_MM = 183
MM_PER_INCH = 25.4
FINAL_WIDTH_IN = FINAL_WIDTH_MM / MM_PER_INCH

COLORS = {
    "ink": "#272727",
    "neutral_dark": "#4D4D4D",
    "neutral": "#767676",
    "neutral_light": "#D3D3D3",
    "grid": "#E6E6E6",
    "blue": "#0F4D92",
    "blue_2": "#3775BA",
    "red": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "blue_pale": "#B9D2E8",
    "red_pale": "#E7B9B4",
}

COMPARATOR_LABELS = {
    "circular_mean": "Circular shift",
    "cross_seed": "Cross-seed replay",
    "frozen_equal": "Equal static",
    "frozen_tuned": "Tuned static",
}

ENDPOINT_LABELS = {
    "svm_activity_mean": "Controller SVM",
    "mpn_probability_all_valid_mean": "Untouched MPN, all",
    "mpn_probability_ad_mean": "Untouched MPN, AD",
    "qed_mean": "QED",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.07,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axis(ax: mpl.axes.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_source_data(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def plot_temporal_qualification(data_dir: Path, output_dir: Path, source_dir: Path) -> None:
    temporal = pd.read_csv(data_dir / "table_temporal_placebo_qualification.csv")
    matching = pd.read_csv(data_dir / "table_cross_seed_matching.csv")
    write_source_data(temporal, source_dir / "fig2_temporal_placebo_qualification.csv")
    write_source_data(matching, source_dir / "fig2_cross_seed_matching.csv")

    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 4.45), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], wspace=0.28, hspace=0.08)

    ax = fig.add_subplot(grid[0, 0])
    dev = temporal[temporal["audit_set"] == "development_frozen"].set_index("method")
    metrics = [
        ("lag1_acf_gap", "Lag-1 ACF"),
        ("delta_ks", "Delta KS"),
        ("decline_count_gap", "Declines"),
        ("component_action_gap", "Actions"),
        ("final_weight_l1_gap", "Weight drift"),
    ]
    methods = ["legacy_shuffle_history", "circular_shift_frozen_mean", "block_permutation"]
    labels = ["Legacy shuffle", "Circular shift", "Block permutation"]
    colors = [COLORS["neutral"], COLORS["blue"], COLORS["teal"]]
    markers = ["s", "o", "^" ]
    legacy = dev.loc["legacy_shuffle_history"]
    x = np.arange(len(metrics))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        values = np.array([dev.loc[method, metric] / legacy[metric] for metric, _ in metrics])
        ax.plot(x, values, color=color, marker=marker, markersize=4.2, linewidth=1.1, label=label, zorder=3)
    ax.axhline(1, color=COLORS["neutral_dark"], linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_xticks(x, [label for _, label in metrics], rotation=24, ha="right")
    ax.set_ylabel("Gap relative to legacy shuffle")
    ax.set_ylim(0, 1.22)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        fontsize=6.0,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    clean_axis(ax, "y")
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    conf = temporal[temporal["audit_set"] == "confirmatory_frozen"].set_index("method")
    frozen_methods = [
        "circular_shift_replicate_0",
        "circular_shift_replicate_1",
        "circular_shift_replicate_2",
        "circular_shift_frozen_mean",
    ]
    frozen_labels = ["Shift 0", "Shift 1", "Shift 2", "Three-shift mean"]
    frozen_metrics = [
        ("lag1_acf_gap", "ACF gap"),
        ("delta_ks", "Delta KS"),
        ("decline_count_gap", "Decline gap"),
        ("component_action_gap", "Action gap"),
        ("final_weight_l1_gap", "Weight gap"),
    ]
    y = np.arange(len(frozen_metrics))
    offsets = [-0.21, -0.07, 0.07, 0.21]
    markers = ["o", "s", "^", "D"]
    colors = [COLORS["blue_pale"], COLORS["blue_2"], COLORS["teal"], COLORS["blue"]]
    for method, label, offset, marker, color in zip(frozen_methods, frozen_labels, offsets, markers, colors):
        values = np.array([conf.loc[method, metric] for metric, _ in frozen_metrics])
        ax.scatter(values, y + offset, s=19, marker=marker, color=color, edgecolor="white", linewidth=0.35, label=label, zorder=3)
    ax.set_yticks(y, [label for _, label in frozen_metrics])
    ax.set_xlabel("Absolute discrepancy")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 2.1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    clean_axis(ax, "x")
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    match = matching[matching["component"] != "overall"]
    match_metrics = [
        ("checkpoint_mae", "Checkpoint MAE"),
        ("abs_score_mean_difference", "Mean score"),
        ("phase_mean_abs_gap", "Phase mean"),
        ("abs_slope_difference", "Stage slope"),
    ]
    component_labels = {"DRD2_activity": "DRD2", "Molecular weight": "MW", "QED": "QED"}
    component_colors = {"DRD2_activity": COLORS["blue"], "Molecular weight": COLORS["teal"], "QED": COLORS["violet"]}
    x = np.arange(len(match_metrics))
    width = 0.22
    for idx, component in enumerate(["DRD2_activity", "Molecular weight", "QED"]):
        row = match[match["component"] == component].iloc[0]
        values = [row[metric] for metric, _ in match_metrics]
        ax.bar(x + (idx - 1) * width, values, width=width, color=component_colors[component], label=component_labels[component], zorder=3)
    ax.set_xticks(x, [label for _, label in match_metrics], rotation=24, ha="right")
    ax.set_ylabel("Target-donor absolute gap")
    ax.legend(loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.0)
    clean_axis(ax, "y")
    panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    qed_rows = dev.loc[["legacy_shuffle_history", "circular_shift_frozen_mean", "block_permutation"]]
    y = np.arange(3)[::-1]
    source = qed_rows["source_declines_qed"].to_numpy(float)
    placebo = qed_rows["placebo_declines_qed"].to_numpy(float)
    for yi, left, right in zip(y, source, placebo):
        ax.plot([left, right], [yi, yi], color=COLORS["neutral_light"], linewidth=1.4, zorder=1)
    ax.scatter(source, y, color=COLORS["ink"], marker="o", s=24, label="Real trajectory", zorder=3)
    ax.scatter(placebo, y, color=[COLORS["neutral"], COLORS["blue"], COLORS["teal"]], marker="D", s=24, label="Placebo", zorder=3)
    ax.set_yticks(y, ["Legacy shuffle", "Circular shift", "Block permutation"])
    ax.set_xlabel("Mean QED decline events per seed")
    ax.set_xlim(2.2, 4.9)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS["ink"], label="Real trajectory"),
        Line2D([], [], marker="D", linestyle="", color=COLORS["blue"], label="Placebo"),
    ]
    ax.legend(handles=handles, loc="lower right")
    clean_axis(ax, "x")
    panel_label(ax, "d")

    save_figure(fig, output_dir / "fig2_temporal_placebo")


def plot_confirmatory_forest(data_dir: Path, output_dir: Path, source_dir: Path) -> None:
    all_contrasts = pd.read_csv(data_dir / "table_confirmatory_all_contrasts.csv")
    endpoints = list(ENDPOINT_LABELS)
    data = all_contrasts[all_contrasts["endpoint"].isin(endpoints)].copy()
    write_source_data(data, source_dir / "fig3_confirmatory_contrasts.csv")

    fig, axes = plt.subplots(2, 2, figsize=(FINAL_WIDTH_IN, 4.45), layout="constrained")
    comparator_order = ["circular_mean", "cross_seed", "frozen_equal", "frozen_tuned"]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["neutral"], COLORS["red"]]
    markers = ["o", "^", "s", "D"]
    for panel, (ax, endpoint) in enumerate(zip(axes.flat, endpoints)):
        rows = data[data["endpoint"] == endpoint].set_index("comparator").loc[comparator_order]
        y = np.arange(len(comparator_order))[::-1]
        means = rows["mean_difference"].to_numpy(float)
        lows = rows["ci95_low"].to_numpy(float)
        highs = rows["ci95_high"].to_numpy(float)
        ax.axvline(0, color=COLORS["ink"], linewidth=0.8, zorder=1)
        ax.axvline(0.015, color=COLORS["red"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        for yi, mean, low, high, color, marker in zip(y, means, lows, highs, colors, markers):
            ax.plot([low, high], [yi, yi], color=color, linewidth=1.25, zorder=2)
            ax.scatter(mean, yi, s=28, marker=marker, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_yticks(y, [COMPARATOR_LABELS[c] for c in comparator_order])
        ax.set_title(ENDPOINT_LABELS[endpoint], loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("Paired mean difference (real - comparator)")
        xmin = min(-0.03, lows.min() - 0.003)
        xmax = max(0.03, highs.max() + 0.003)
        ax.set_xlim(xmin, xmax)
        clean_axis(ax, "x")
        panel_label(ax, "abcd"[panel])
    axes[1, 1].text(
        0.0157,
        1.5,
        "+0.015 practical margin",
        color=COLORS["red"],
        fontsize=6.5,
        rotation=90,
        va="center",
        ha="left",
    )
    save_figure(fig, output_dir / "fig3_confirmatory_forest")


def plot_seed_differences(data_dir: Path, output_dir: Path, source_dir: Path) -> None:
    seed_data = pd.read_csv(data_dir / "figure_confirmatory_seed_differences.csv")
    summaries = pd.read_csv(data_dir / "table_confirmatory_all_contrasts.csv")
    endpoints = ["svm_activity_mean", "mpn_probability_all_valid_mean", "mpn_probability_ad_mean"]
    data = seed_data[(seed_data["comparator"] == "circular_mean") & seed_data["endpoint"].isin(endpoints)].copy()
    summary = summaries[(summaries["comparator"] == "circular_mean") & summaries["endpoint"].isin(endpoints)].copy()
    write_source_data(data, source_dir / "fig4_seed_differences.csv")
    write_source_data(summary, source_dir / "fig4_seed_difference_summaries.csv")

    fig, axes = plt.subplots(1, 3, figsize=(FINAL_WIDTH_IN, 2.75), layout="constrained")
    colors = [COLORS["blue"], COLORS["teal"], COLORS["violet"]]
    for panel, (ax, endpoint, color) in enumerate(zip(axes, endpoints, colors)):
        rows = data[data["endpoint"] == endpoint].sort_values("seed")
        stats = summary[summary["endpoint"] == endpoint].iloc[0]
        x = np.arange(len(rows)) + 1
        differences = rows["difference"].to_numpy(float)
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.axhline(0.015, color=COLORS["red"], linewidth=0.8, linestyle=(0, (3, 2)))
        ax.plot(x, differences, color=COLORS["neutral_light"], linewidth=0.75, zorder=1)
        ax.scatter(x, differences, s=23, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        mean_x = 12.8
        ax.errorbar(
            mean_x,
            stats["mean_difference"],
            yerr=[[stats["mean_difference"] - stats["ci95_low"]], [stats["ci95_high"] - stats["mean_difference"]]],
            fmt="D",
            color=COLORS["ink"],
            markersize=4.0,
            linewidth=1.2,
            capsize=2,
            zorder=4,
        )
        ax.set_xticks([1, 5, 10, mean_x], ["6101", "6105", "6110", "Mean"])
        ax.set_xlim(0.3, 13.6)
        ax.set_title(ENDPOINT_LABELS[endpoint], loc="left", fontweight="bold", pad=4)
        ax.set_xlabel("Paired seed")
        if panel == 0:
            ax.set_ylabel("Difference (real - circular mean)")
        clean_axis(ax, "y")
        panel_label(ax, "abc"[panel])
    save_figure(fig, output_dir / "fig4_seed_level_primary")


def plot_positive_control(data_dir: Path, output_dir: Path, source_dir: Path) -> None:
    data = pd.read_csv(data_dir / "figure_v5_paired_qed.csv").sort_values("seed")
    write_source_data(data, source_dir / "fig5_v5_positive_control.csv")

    fig = plt.figure(figsize=(FINAL_WIDTH_IN, 3.65), layout="constrained")
    grid = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.08)

    ax = fig.add_subplot(grid[0, 0])
    for _, row in data.iterrows():
        ax.plot([0, 1], [row["frozen_tail_qed"], row["real_tail_qed"]], color=COLORS["neutral_light"], linewidth=1.0)
        ax.scatter(0, row["frozen_tail_qed"], color=COLORS["neutral_dark"], marker="s", s=23, zorder=3)
        ax.scatter(1, row["real_tail_qed"], color=COLORS["blue"], marker="o", s=23, zorder=3)
    ax.set_xticks([0, 1], ["Frozen weight", "Adaptive controller"])
    ax.set_ylabel("Tail QED")
    ax.set_xlim(-0.35, 1.35)
    clean_axis(ax, "y")
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    gains = data["real_minus_frozen_qed"].to_numpy(float)
    x = np.arange(len(data))
    ax.axhline(0.05, color=COLORS["red"], linewidth=0.8, linestyle=(0, (3, 2)), label="Pass threshold (+0.05)")
    ax.bar(x, gains, color=COLORS["blue"], width=0.66, zorder=3)
    ax.axhline(gains.mean(), color=COLORS["ink"], linewidth=0.9, label=f"Mean = {gains.mean():.3f}")
    ax.set_xticks(x, data["seed"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("QED gain")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.5,
    )
    clean_axis(ax, "y")
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    x = np.arange(len(data))
    ax.bar(x - 0.18, data["frozen_final_qed_weight"], width=0.36, color=COLORS["neutral"], label="Frozen weight")
    ax.bar(x + 0.18, data["real_final_qed_weight"], width=0.36, color=COLORS["blue"], label="Final adaptive weight")
    ax.set_xticks(x, data["seed"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("QED scalarization weight")
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.9,
        handlelength=1.4,
    )
    clean_axis(ax, "y")
    panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    quality = pd.DataFrame(
        {
            "Validity": data["real_validity"] - data["frozen_validity"],
            "Uniqueness": data["real_uniqueness"] - data["frozen_uniqueness"],
        }
    )
    positions = [0, 1]
    for pos, column, color, marker in zip(positions, quality.columns, [COLORS["teal"], COLORS["violet"]], ["o", "D"]):
        values = quality[column].to_numpy(float)
        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), pos) + offsets, values, s=23, color=color, marker=marker, edgecolor="white", linewidth=0.35, zorder=3)
        ax.plot([pos - 0.22, pos + 0.22], [values.mean(), values.mean()], color=COLORS["ink"], linewidth=1.2, zorder=4)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.axhline(-0.02, color=COLORS["red"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks(positions, quality.columns)
    ax.set_ylabel("Difference (adaptive - frozen)")
    clean_axis(ax, "y")
    panel_label(ax, "d")

    save_figure(fig, output_dir / "fig5_end_to_end_positive_control")


def validate_outputs(output_dir: Path) -> dict[str, object]:
    expected_stems = [
        "fig2_temporal_placebo",
        "fig3_confirmatory_forest",
        "fig4_seed_level_primary",
        "fig5_end_to_end_positive_control",
    ]
    formats = ["svg", "pdf", "tiff", "png"]
    missing = [str(output_dir / f"{stem}.{suffix}") for stem in expected_stems for suffix in formats if not (output_dir / f"{stem}.{suffix}").exists()]
    svg_text_counts = {}
    for stem in expected_stems:
        svg = output_dir / f"{stem}.svg"
        if svg.exists():
            svg_text_counts[stem] = svg.read_text(encoding="utf-8").count("<text")
    return {
        "status": "PASS" if not missing and all(count > 0 for count in svg_text_counts.values()) else "FAIL",
        "missing": missing,
        "svg_text_elements": svg_text_counts,
        "final_width_mm": FINAL_WIDTH_MM,
        "tiff_dpi": 600,
        "png_dpi": 300,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("paper_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/rendered"))
    parser.add_argument("--source-dir", type=Path, default=Path("figures/source_data"))
    args = parser.parse_args()

    set_style()
    plot_temporal_qualification(args.data_dir, args.output_dir, args.source_dir)
    plot_confirmatory_forest(args.data_dir, args.output_dir, args.source_dir)
    plot_seed_differences(args.data_dir, args.output_dir, args.source_dir)
    plot_positive_control(args.data_dir, args.output_dir, args.source_dir)

    report = validate_outputs(args.output_dir)
    (args.output_dir / "render_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
