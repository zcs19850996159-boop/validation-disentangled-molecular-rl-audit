from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "neutral_black": "#272727",
    "delta_up": "#2E9E44",
    "delta_down": "#E53935",
    "red_soft": "#E9A6A1",
    "gold": "#D18F2F",
}


def apply_publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def save_pub(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    base = FIG_DIR / stem
    saved = []
    for ext, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
    ]:
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def load_json(name: str):
    with (ROOT / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    text_color: str = "#272727",
    fontsize: float = 7.0,
    weight: str = "normal",
    radius: float = 0.025,
    lw: float = 0.9,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight=weight,
        linespacing=1.15,
        zorder=3,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    rad: float = 0.0,
    lw: float = 1.15,
    style: str = "-|>",
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=10,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        shrinkA=4,
        shrinkB=4,
        zorder=1,
    )
    ax.add_patch(patch)


def make_fig1() -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.1, 4.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue = PALETTE["blue_main"]
    blue_soft = "#EAF1FA"
    teal = "#42949E"
    teal_soft = "#E6F3F4"
    lilac = "#F0F0FA"
    grey_line = PALETTE["neutral_mid"]

    ax.text(
        0.03,
        0.96,
        "Validation-disentangled diagnostic framework",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=PALETTE["neutral_black"],
    )
    ax.text(
        0.03,
        0.915,
        "Adaptive reward weighting in molecular generation",
        ha="left",
        va="top",
        fontsize=6.8,
        color=PALETTE["neutral_dark"],
    )

    # Main process boxes.
    reward_xy = (0.07, 0.625)
    reward_w, reward_h = 0.25, 0.15
    reward_box = FancyBboxPatch(
        reward_xy,
        reward_w,
        reward_h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=blue,
        facecolor=blue_soft,
        mutation_aspect=1,
        zorder=2,
    )
    ax.add_patch(reward_box)
    reward_cx = reward_xy[0] + reward_w / 2
    ax.text(
        reward_cx,
        reward_xy[1] + 0.113,
        "Training reward stream",
        ha="center",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=PALETTE["neutral_black"],
        zorder=3,
    )
    ax.text(
        reward_cx,
        reward_xy[1] + 0.078,
        "$R(x; w_t)=\\sum_{k} w_{k,t} c_k(x)$",
        ha="center",
        va="center",
        fontsize=5.5,
        color=PALETTE["neutral_black"],
        zorder=3,
    )
    ax.text(
        reward_cx,
        reward_xy[1] + 0.044,
        "scores training molecules",
        ha="center",
        va="center",
        fontsize=5.6,
        color=PALETTE["neutral_black"],
        zorder=3,
    )
    ax.text(
        reward_cx,
        reward_xy[1] + 0.023,
        "used for generator update",
        ha="center",
        va="center",
        fontsize=5.6,
        color=PALETTE["neutral_black"],
        zorder=3,
    )
    generator_box = draw_round_box(
        ax,
        (0.42, 0.655),
        0.23,
        0.12,
        "REINVENT4\nmolecular generator",
        facecolor="#DCE9F7",
        edgecolor=blue,
        fontsize=7.3,
        weight="bold",
    )
    molecules_box = draw_round_box(
        ax,
        (0.75, 0.655),
        0.20,
        0.12,
        "Generated molecules\nsampled from\ncurrent policy",
        facecolor="white",
        edgecolor=grey_line,
        fontsize=6.6,
    )
    val_batch_box = draw_round_box(
        ax,
        (0.74, 0.34),
        0.22,
        0.12,
        "Validation-only batch\nnot used for training reward",
        facecolor=teal_soft,
        edgecolor=teal,
        fontsize=6.4,
        weight="bold",
    )
    val_signal_box = draw_round_box(
        ax,
        (0.40, 0.34),
        0.25,
        0.12,
        "Independent validation signal\nSVM-QSAR / Vina / exact SA\nindependent proxy or candidate oracle",
        facecolor=teal_soft,
        edgecolor=teal,
        fontsize=5.8,
    )
    controller_box = draw_round_box(
        ax,
        (0.07, 0.34),
        0.25,
        0.12,
        "Weight controller\nupdates only from\nvalidation summaries",
        facecolor=lilac,
        edgecolor=PALETTE["baseline_dark"],
        fontsize=6.3,
        weight="bold",
    )
    weights_box = draw_round_box(
        ax,
        (0.07, 0.50),
        0.25,
        0.07,
        "Updated objective weights $w_t$",
        facecolor="white",
        edgecolor=PALETTE["baseline_dark"],
        fontsize=6.5,
    )

    # Training loop.
    arrow(ax, (0.32, 0.705), (0.42, 0.715), blue, lw=1.4)
    arrow(ax, (0.65, 0.715), (0.75, 0.715), blue, lw=1.4)
    ax.text(0.54, 0.80, "training loop", fontsize=6.3, color=blue, ha="center", va="center")

    # Validation-only controller loop.
    arrow(ax, (0.85, 0.655), (0.85, 0.46), teal, lw=1.35)
    arrow(ax, (0.74, 0.40), (0.65, 0.40), teal, lw=1.35)
    arrow(ax, (0.40, 0.40), (0.32, 0.40), teal, lw=1.35)
    arrow(ax, (0.195, 0.46), (0.195, 0.50), PALETTE["baseline_dark"], lw=1.15)
    arrow(ax, (0.195, 0.57), (0.195, 0.625), PALETTE["baseline_dark"], lw=1.15)
    ax.text(
        0.865,
        0.555,
        "checkpoint sampling",
        fontsize=6.3,
        color=teal,
        ha="left",
        va="center",
    )

    # Explicit disentanglement divider.
    ax.plot([0.04, 0.96], [0.292, 0.292], color=PALETTE["neutral_mid"], lw=0.75, ls=(0, (4, 4)), alpha=0.7)
    ax.text(
        0.955,
        0.305,
        "training reward and validation feedback remain separate",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["neutral_dark"],
    )

    # Diagnostic audit layer.
    audit = FancyBboxPatch(
        (0.045, 0.06),
        0.91,
        0.17,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=0.65,
        edgecolor="#C7C7C7",
        facecolor="#FAFAFA",
        zorder=0,
    )
    ax.add_patch(audit)
    ax.text(
        0.075,
        0.205,
        "Diagnostic audit layer",
        fontsize=6.7,
        fontweight="bold",
        ha="left",
        va="center",
        color=PALETTE["neutral_dark"],
    )
    chips = [
        ("random-validation\ncontrol", 0.12),
        ("constant-signal /\nfrozen-policy checks", 0.30),
        ("positive-control\nsensitivity ladder", 0.50),
        ("paired bootstrap\nand power analysis", 0.70),
        ("hindsight\nupper bound", 0.88),
    ]
    for label, cx in chips:
        draw_round_box(
            ax,
            (cx - 0.075, 0.085),
            0.15,
            0.075,
            label,
            facecolor="white",
            edgecolor=PALETTE["neutral_light"],
            fontsize=5.6,
            radius=0.018,
            lw=0.7,
        )
    arrow(ax, (0.50, 0.292), (0.50, 0.34), "#B8B8B8", lw=0.55, linestyle="--", alpha=0.65)
    arrow(ax, (0.195, 0.292), (0.195, 0.34), "#B8B8B8", lw=0.55, linestyle="--", alpha=0.65)
    arrow(ax, (0.85, 0.292), (0.85, 0.34), "#B8B8B8", lw=0.55, linestyle="--", alpha=0.65)

    # Small route labels.
    ax.text(0.37, 0.73, "generator update", fontsize=5.7, color=blue, ha="center")
    ax.text(0.225, 0.595, "updated weights", fontsize=5.7, color=PALETTE["baseline_dark"], ha="left")
    ax.text(0.225, 0.48, "weight update", fontsize=5.7, color=PALETTE["baseline_dark"], ha="left")
    ax.text(0.68, 0.415, "separate\nscoring", fontsize=5.7, color=teal, ha="center", va="center")

    return save_pub(fig, "fig1_framework_revised")


def nice_limits(values: np.ndarray, pad_fraction: float = 0.06) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi == lo:
        return lo - 1.0, hi + 1.0
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def make_fig2() -> list[Path]:
    rows = []
    with (ROOT / "qsar_vina_triangulation_DRD2_7DFP_scored.csv").open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    stats = load_json("qsar_vina_triangulation_DRD2_7DFP.json")["pairs"]
    panels = [
        (
            "a",
            "reward_rf",
            "validation_svm",
            "RF reward",
            "SVM validation",
            "rf_reward_vs_svm_validation",
        ),
        (
            "b",
            "reward_rf",
            "docking_reward",
            "RF reward",
            "-Vina reward",
            "rf_reward_vs_vina_docking_reward",
        ),
        (
            "c",
            "validation_svm",
            "docking_reward",
            "SVM validation",
            "-Vina reward",
            "svm_validation_vs_vina_docking_reward",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), constrained_layout=True)
    label_to_color = {"active": PALETTE["red_soft"], "decoy": PALETTE["blue_secondary"]}
    label_to_edge = {"active": PALETTE["delta_down"], "decoy": PALETTE["blue_main"]}

    for ax, (letter, xkey, ykey, xlabel, ylabel, skey) in zip(axes, panels):
        xs = np.array([float(row[xkey]) for row in rows])
        ys = np.array([float(row[ykey]) for row in rows])
        labels = np.array([row["label"] for row in rows])
        xlo, xhi = nice_limits(xs)
        ylo, yhi = nice_limits(ys)
        for group in ["decoy", "active"]:
            mask = labels == group
            ax.scatter(
                xs[mask],
                ys[mask],
                s=14,
                c=label_to_color[group],
                edgecolors=label_to_edge[group],
                linewidths=0.35,
                alpha=0.78,
                label="active" if group == "active" else "negative",
            )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#E8E8E8", linewidth=0.45, zorder=0)
        add_panel_label(ax, letter)
        st = stats[skey]
        annotation = (
            f"Pearson {st['pearson']:.3f}\n"
            f"Spearman {st['spearman']:.3f}\n"
            f"Top 10% {st['top_fraction_overlap_count']}/{st['top_fraction_size']}"
        )
        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": PALETTE["neutral_light"],
                "linewidth": 0.5,
                "alpha": 0.92,
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc="lower right", fontsize=6.5)
    fig.suptitle("RF/SVM/Vina signal triangulation", fontsize=8.0, fontweight="bold", x=0.06, ha="left")
    return save_pub(fig, "fig2_signal_triangulation")


def make_fig3() -> list[Path]:
    old_stats = load_json("controller_effect_statistics.json")["settings"]
    drd2_n10 = load_json("drd2_base_n10_interim_pair_summary.json")
    activity_sa = old_stats["activity_sa"]
    drd2_static_gap = old_stats["drd2_qed_mw"]["static_repeat_gap"]

    drd2_primary = {
        "deltas": drd2_n10["summary"]["last50"]["deltas"],
        "mean_delta": drd2_n10["summary"]["last50"]["mean_delta"],
        "paired_bootstrap_mean_ci95": drd2_n10["summary"]["last50"]["paired_bootstrap_mean_ci95"],
        "positive_count": drd2_n10["summary"]["last50"]["positive_count"],
        "n_paired_repeats": drd2_n10["n_paired_repeats"],
        "static_repeat_gap": drd2_static_gap,
    }
    rows = [
        ("DRD2/QED/MW", "SVM activity, last 50 steps", drd2_primary),
        ("Activity/SA", "joint validation, last 50 steps", activity_sa),
    ]

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )

    all_values = []
    for _, _, setting in rows:
        all_values.extend(setting["deltas"])
        ci = setting["paired_bootstrap_mean_ci95"]
        all_values.extend([ci["ci95_low"], ci["ci95_high"]])
        all_values.extend([setting["static_repeat_gap"], -setting["static_repeat_gap"]])

    ylo, yhi = nice_limits(np.array(all_values), 0.18)
    ax.axhline(0, color=PALETTE["neutral_dark"], linestyle="--", linewidth=0.8, zorder=1)

    for i, (name, metric, setting) in enumerate(rows):
        gap = setting["static_repeat_gap"]
        ax.fill_between(
            [i - 0.34, i + 0.34],
            [-gap, -gap],
            [gap, gap],
            color=PALETTE["baseline_soft"],
            alpha=0.25,
            linewidth=0,
            zorder=0,
        )
        deltas = np.array(setting["deltas"])
        jitter = np.linspace(-0.16, 0.16, len(deltas))
        colors = np.where(deltas >= 0, PALETTE["blue_main"], PALETTE["delta_down"])
        ax.scatter(
            np.full_like(deltas, i, dtype=float) + jitter,
            deltas,
            s=18 if len(deltas) > 5 else 22,
            c=colors,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ci = setting["paired_bootstrap_mean_ci95"]
        mean_delta = setting["mean_delta"]
        lo = ci["ci95_low"]
        hi = ci["ci95_high"]
        ax.errorbar(
            i,
            mean_delta,
            yerr=[[mean_delta - lo], [hi - mean_delta]],
            fmt="o",
            color=PALETTE["neutral_black"],
            ecolor=PALETTE["neutral_black"],
            elinewidth=1.1,
            capsize=3,
            markersize=4.0,
            zorder=4,
        )
        n = setting.get("n_paired_repeats", len(deltas))
        pos = int(np.sum(deltas > 0))
        ax.text(
            i,
            yhi - (yhi - ylo) * 0.04,
            f"mean {mean_delta:+.4f}\n{pos}/{n} positive",
            ha="center",
            va="top",
            fontsize=6.2,
            linespacing=1.15,
        )
        ax.text(
            i,
            ylo + (yhi - ylo) * 0.04,
            metric,
            ha="center",
            va="bottom",
            fontsize=5.9,
            color=PALETTE["neutral_dark"],
        )

    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Real - random validation delta")
    ax.set_ylim(ylo, yhi)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.45)
    add_panel_label(ax, "a")
    ax.set_title("Primary paired effects overlap zero", loc="left", fontsize=8.0, fontweight="bold")

    metric_labels = ["All rows", "Last 50", "Final step", "Top 10"]
    metric_keys = ["all", "last50", "final", "top10"]
    x = np.arange(len(metric_keys))
    means = np.array([drd2_n10["summary"][key]["mean_delta"] for key in metric_keys])
    ci_lows = np.array([drd2_n10["summary"][key]["paired_bootstrap_mean_ci95"]["ci95_low"] for key in metric_keys])
    ci_highs = np.array([drd2_n10["summary"][key]["paired_bootstrap_mean_ci95"]["ci95_high"] for key in metric_keys])
    colors = [PALETTE["neutral_mid"], PALETTE["blue_main"], PALETTE["gold"], PALETTE["neutral_mid"]]

    ax2.axhline(0, color=PALETTE["neutral_dark"], linestyle="--", linewidth=0.8, zorder=1)
    ax2.bar(x, means, color=colors, alpha=0.78, edgecolor="white", linewidth=0.6, zorder=2)
    ax2.errorbar(
        x,
        means,
        yerr=[means - ci_lows, ci_highs - means],
        fmt="none",
        ecolor=PALETTE["neutral_black"],
        elinewidth=1.0,
        capsize=3,
        zorder=4,
    )
    ax2.axhspan(-drd2_static_gap, drd2_static_gap, color=PALETTE["baseline_soft"], alpha=0.22, zorder=0)
    for i, mean in enumerate(means):
        va = "bottom" if mean >= 0 else "top"
        dy = 0.0013 if mean >= 0 else -0.0013
        ax2.text(i, mean + dy, f"{mean:+.4f}", ha="center", va=va, fontsize=5.8)
    y2lo, y2hi = nice_limits(np.concatenate([means, ci_lows, ci_highs, [-drd2_static_gap, drd2_static_gap]]), 0.18)
    ax2.set_ylim(y2lo, y2hi)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, rotation=25, ha="right")
    ax2.set_ylabel("DRD2/QED/MW delta")
    ax2.grid(True, axis="y", color="#E8E8E8", linewidth=0.45)
    add_panel_label(ax2, "b")
    ax2.set_title("DRD2 endpoint signal is unstable", loc="left", fontsize=8.0, fontweight="bold")

    return save_pub(fig, "fig4_paired_deltas")


def make_fig4() -> list[Path]:
    levels = load_json("positive_control_sensitivity_ladder.json")["levels"]
    labels = ["Strong", "Medium\n2-window", "Medium\n3-window", "Near-noise"]

    fig, ax = plt.subplots(figsize=(5.25, 3.05), constrained_layout=True)
    x = np.arange(len(levels))
    real = np.array([level["real"]["final_weight_target_a"] for level in levels])
    shuffled_mean = np.array([level["shuffled_control"]["final_weight_target_a_mean"] for level in levels])
    shuffled_min = np.array([level["shuffled_control"]["final_weight_target_a_min"] for level in levels])
    shuffled_max = np.array([level["shuffled_control"]["final_weight_target_a_max"] for level in levels])
    z_values = [level["sensitivity"]["z_vs_shuffled"] for level in levels]
    event_counts = [level["detectable_drop_events"] for level in levels]
    passes = [level["sensitivity"]["passes_positive_control"] for level in levels]

    ax.bar(
        x - 0.14,
        shuffled_max - shuffled_min,
        width=0.22,
        bottom=shuffled_min,
        color=PALETTE["neutral_light"],
        edgecolor=PALETTE["neutral_mid"],
        linewidth=0.7,
        zorder=1,
    )
    ax.scatter(
        x - 0.14,
        shuffled_mean,
        s=20,
        c=PALETTE["neutral_dark"],
        marker="s",
        zorder=3,
    )
    real_colors = [PALETTE["blue_main"] if p else PALETTE["red_soft"] for p in passes]
    real_edges = [PALETTE["blue_main"] if p else PALETTE["delta_down"] for p in passes]
    ax.scatter(
        x + 0.14,
        real,
        s=34,
        c=real_colors,
        edgecolors=real_edges,
        linewidths=0.9,
        zorder=4,
    )

    for i, (z, events) in enumerate(zip(z_values, event_counts)):
        ax.text(i, 0.895, f"z={z:.2f}", ha="center", va="bottom", fontsize=5.8)
        ax.text(i, 0.878, f"events={events}", ha="center", va="top", fontsize=5.7, color=PALETTE["neutral_dark"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final Target A weight")
    ax.set_ylim(0.46, 0.91)
    ax.grid(True, axis="y", color="#E8E8E8", linewidth=0.45)
    add_panel_label(ax, "a")
    ax.set_title("Only dense synthetic validation signals are reliably detected", loc="left", fontsize=8.0, fontweight="bold")
    ax.text(
        0.02,
        0.08,
        "Grey: shuffled-control range and mean\nBlue/red: real synthetic signal",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=PALETTE["neutral_dark"],
    )

    ax.annotate(
        "stable detection",
        xy=(0.14, real[0]),
        xytext=(0.55, 0.825),
        arrowprops={"arrowstyle": "-", "color": PALETTE["neutral_dark"], "linewidth": 0.7},
        fontsize=6.5,
        ha="left",
    )
    ax.annotate(
        "blind zone",
        xy=(2.14, real[2]),
        xytext=(2.27, 0.575),
        arrowprops={"arrowstyle": "-", "color": PALETTE["neutral_dark"], "linewidth": 0.7},
        fontsize=6.5,
        ha="left",
    )

    return save_pub(fig, "fig3_positive_control_ladder")


def main() -> None:
    apply_publication_style()
    outputs = []
    outputs.extend(make_fig1())
    outputs.extend(make_fig2())
    outputs.extend(make_fig3())
    outputs.extend(make_fig4())
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
