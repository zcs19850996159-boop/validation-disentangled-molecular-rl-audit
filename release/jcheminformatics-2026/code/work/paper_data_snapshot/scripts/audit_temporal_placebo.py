#!/usr/bin/env python
"""Offline audit of temporal placebo trajectories.

This script evaluates controller-input dynamics, not molecule quality.  It is
used before launching a new placebo arm to demonstrate that a temporal replay
preserves the validation process materially better than the legacy independent
history shuffle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Iterable

from temporal_placebo_common import automatic_shift_schedule, eligible_circular_shifts


def parse_component_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise RuntimeError("No component names were provided")
    return names


def parse_initial_weights(raw: str, names: list[str]) -> dict[str, float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if len(values) != len(names):
        raise RuntimeError("--initial-weights needs one value per component")
    return {name: float(value) for name, value in zip(names, values)}


def read_history(path: Path, names: list[str]) -> list[dict[str, float]]:
    trajectory: list[dict[str, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        scores = record.get("validation_scores", {})
        if not all(name in scores for name in names):
            raise RuntimeError(f"{path}:{line_number} is missing a requested score")
        trajectory.append({name: float(scores[name]) for name in names})
    if len(trajectory) < 2:
        raise RuntimeError(f"{path} needs at least two validation records")
    return trajectory


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator_left = sum((x - left_mean) ** 2 for x in left)
    denominator_right = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(denominator_left * denominator_right)
    return numerator / denominator if denominator else None


def ks_distance(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left = sorted(left)
    right = sorted(right)
    i = j = 0
    largest = 0.0
    while i < len(left) or j < len(right):
        if j == len(right) or (i < len(left) and left[i] <= right[j]):
            value = left[i]
        else:
            value = right[j]
        while i < len(left) and left[i] <= value:
            i += 1
        while j < len(right) and right[j] <= value:
            j += 1
        largest = max(largest, abs(i / len(left) - j / len(right)))
    return largest


def pair_names(names: list[str]) -> list[tuple[str, str]]:
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def simulate_final_weights(
    trajectory: list[dict[str, float]],
    names: list[str],
    deadband: float,
    action_step: float,
    initial_weights: dict[str, float],
    decrease_on_improvement: bool,
) -> tuple[dict[str, float], int, int, dict[str, int]]:
    weights = dict(initial_weights)
    initial_sum = sum(weights.values())
    action_checkpoints = 0
    component_actions = 0
    decline_counts = {name: 0 for name in names}
    for previous, current in zip(trajectory, trajectory[1:]):
        actions = {}
        for name in names:
            delta = current[name] - previous[name]
            if delta < -deadband:
                actions[name] = 1
                decline_counts[name] += 1
            elif decrease_on_improvement and delta > deadband:
                actions[name] = -1
            else:
                actions[name] = 0
        if any(action != 0 for action in actions.values()):
            action_checkpoints += 1
            component_actions += sum(action != 0 for action in actions.values())
            for name, action in actions.items():
                weights[name] *= math.exp(action_step * action)
            scale = initial_sum / sum(weights.values())
            for name in names:
                weights[name] *= scale
    return weights, action_checkpoints, component_actions, decline_counts


def metrics(
    trajectory: list[dict[str, float]],
    names: list[str],
    deadband: float,
    action_step: float,
    initial_weights: dict[str, float],
    decrease_on_improvement: bool,
) -> dict:
    values = {name: [record[name] for record in trajectory] for name in names}
    deltas = {
        name: [right - left for left, right in zip(series, series[1:])]
        for name, series in values.items()
    }
    final_weights, action_checkpoints, component_actions, decline_counts = simulate_final_weights(
        trajectory,
        names,
        deadband,
        action_step,
        initial_weights,
        decrease_on_improvement,
    )
    pairs = pair_names(names)
    return {
        "n_validation_points": len(trajectory),
        "lag1_autocorrelation": {
            name: pearson(series[:-1], series[1:]) for name, series in values.items()
        },
        "delta_summary": {
            name: {
                "mean": mean(series),
                "sd": statistics.stdev(series) if len(series) > 1 else 0.0,
                "q05": sorted(series)[max(0, math.ceil(len(series) * 0.05) - 1)],
                "q50": statistics.median(series),
                "q95": sorted(series)[min(len(series) - 1, math.ceil(len(series) * 0.95) - 1)],
            }
            for name, series in deltas.items()
        },
        "deltas": deltas,
        "decline_event_count": decline_counts,
        "controller_action_checkpoints": action_checkpoints,
        "controller_component_actions": component_actions,
        "final_weights": final_weights,
        "final_weight_l1_drift": sum(
            abs(final_weights[name] - initial_weights[name]) for name in names
        ),
        "score_correlations": {
            f"{left}|{right}": pearson(values[left], values[right]) for left, right in pairs
        },
        "delta_correlations": {
            f"{left}|{right}": pearson(deltas[left], deltas[right]) for left, right in pairs
        },
    }


def map_distance(left: dict[str, float | None], right: dict[str, float | None]) -> float | None:
    return mean(
        abs(float(left[key]) - float(right[key]))
        for key in left.keys() & right.keys()
        if left[key] is not None and right[key] is not None
    )


def dynamic_gap(candidate: dict, source: dict, names: list[str]) -> dict[str, float | None]:
    delta_ks = {
        name: ks_distance(candidate["deltas"][name], source["deltas"][name]) for name in names
    }
    return {
        "mean_abs_lag1_autocorrelation_gap": map_distance(
            candidate["lag1_autocorrelation"], source["lag1_autocorrelation"]
        ),
        "mean_delta_ks_distance": mean(delta_ks.values()),
        "mean_abs_decline_event_count_gap": mean(
            abs(candidate["decline_event_count"][name] - source["decline_event_count"][name])
            for name in names
        ),
        "abs_controller_action_checkpoint_gap": abs(
            candidate["controller_action_checkpoints"] - source["controller_action_checkpoints"]
        ),
        "abs_controller_component_action_gap": abs(
            candidate["controller_component_actions"] - source["controller_component_actions"]
        ),
        "final_weight_l1_gap": sum(
            abs(candidate["final_weights"][name] - source["final_weights"][name]) for name in names
        ),
        "mean_score_correlation_gap": map_distance(
            candidate["score_correlations"], source["score_correlations"]
        ),
        "mean_delta_correlation_gap": map_distance(
            candidate["delta_correlations"], source["delta_correlations"]
        ),
    }


def circular_trajectories(trajectory: list[dict[str, float]], shifts: list[int]) -> list[list[dict[str, float]]]:
    return [trajectory[shift:] + trajectory[:shift] for shift in shifts]


def block_permutation(trajectory: list[dict[str, float]], block_size: int, seed: int) -> list[dict[str, float]]:
    blocks = [trajectory[start : start + block_size] for start in range(0, len(trajectory), block_size)]
    if len(blocks) < 2:
        raise RuntimeError("block permutation needs at least two blocks")
    order = list(range(len(blocks)))
    random.Random(seed).shuffle(order)
    if order == list(range(len(blocks))):
        order = order[1:] + order[:1]
    return [record for block_index in order for record in blocks[block_index]]


def seed_label(path: Path, fallback_index: int) -> str:
    match = re.search(r"seed(\d+)", path.name)
    return match.group(1) if match else f"index{fallback_index}"


def mean_map(records: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = set().union(*(record.keys() for record in records))
    return {key: mean(record.get(key) for record in records) for key in keys}


def mean_variant_metrics(variants: list[dict]) -> dict:
    """Keep the per-shift average readable while gaps retain full distributions."""

    return {
        "n_validation_points": mean(item["n_validation_points"] for item in variants),
        "lag1_autocorrelation": mean_map(
            [item["lag1_autocorrelation"] for item in variants]
        ),
        "decline_event_count": mean_map(
            [item["decline_event_count"] for item in variants]
        ),
        "controller_action_checkpoints": mean(
            item["controller_action_checkpoints"] for item in variants
        ),
        "controller_component_actions": mean(
            item["controller_component_actions"] for item in variants
        ),
        "final_weights": mean_map([item["final_weights"] for item in variants]),
        "final_weight_l1_drift": mean(item["final_weight_l1_drift"] for item in variants),
        "score_correlations": mean_map([item["score_correlations"] for item in variants]),
        "delta_correlations": mean_map([item["delta_correlations"] for item in variants]),
    }


def series_slope(values: list[float]) -> float:
    """Least-squares score change per validation checkpoint."""

    indices = list(range(len(values)))
    index_mean = statistics.fmean(indices)
    value_mean = statistics.fmean(values)
    denominator = sum((index - index_mean) ** 2 for index in indices)
    if not denominator:
        return 0.0
    return sum(
        (index - index_mean) * (value - value_mean)
        for index, value in zip(indices, values)
    ) / denominator


def phase_profile(trajectory: list[dict[str, float]], names: list[str]) -> dict[str, dict]:
    """Summarize early, middle, and late validation-score phases."""

    phase_names = ("early", "middle", "late")
    profiles = {}
    for name in names:
        values = [record[name] for record in trajectory]
        phase_values = {phase: [] for phase in phase_names}
        for index, value in enumerate(values):
            phase_index = min(len(phase_names) - 1, index * len(phase_names) // len(values))
            phase_values[phase_names[phase_index]].append(value)
        profiles[name] = {
            "checkpoint_mean": statistics.fmean(values),
            "slope_per_checkpoint": series_slope(values),
            "phase_means": {
                phase: statistics.fmean(values) for phase, values in phase_values.items()
            },
        }
    return profiles


def donor_target_match(
    target_trajectory: list[dict[str, float]],
    donor_trajectory: list[dict[str, float]],
    target_metrics: dict,
    donor_metrics: dict,
    names: list[str],
) -> dict:
    """Quantify whether a pre-specified donor is phase-comparable to its target."""

    target_profile = phase_profile(target_trajectory, names)
    donor_profile = phase_profile(donor_trajectory, names)
    checkpoint_mean_difference = {
        name: target_profile[name]["checkpoint_mean"] - donor_profile[name]["checkpoint_mean"]
        for name in names
    }
    checkpoint_mae = {
        name: mean(
            abs(target[name] - donor[name])
            for target, donor in zip(target_trajectory, donor_trajectory)
        )
        for name in names
    }
    acf_gap = {
        name: abs(
            float(target_metrics["lag1_autocorrelation"][name])
            - float(donor_metrics["lag1_autocorrelation"][name])
        )
        for name in names
        if target_metrics["lag1_autocorrelation"][name] is not None
        and donor_metrics["lag1_autocorrelation"][name] is not None
    }
    delta_ks = {
        name: ks_distance(target_metrics["deltas"][name], donor_metrics["deltas"][name])
        for name in names
    }
    decline_difference = {
        name: target_metrics["decline_event_count"][name]
        - donor_metrics["decline_event_count"][name]
        for name in names
    }
    slope_difference = {
        name: target_profile[name]["slope_per_checkpoint"]
        - donor_profile[name]["slope_per_checkpoint"]
        for name in names
    }
    phase_mean_difference = {
        name: {
            phase: target_profile[name]["phase_means"][phase]
            - donor_profile[name]["phase_means"][phase]
            for phase in ("early", "middle", "late")
        }
        for name in names
    }
    phase_mean_abs_gap = {
        name: mean(abs(value) for value in phase_mean_difference[name].values())
        for name in names
    }
    return {
        "target_profile": target_profile,
        "donor_profile": donor_profile,
        "checkpoint_level": {
            "component_mean_difference_target_minus_donor": checkpoint_mean_difference,
            "component_checkpoint_mae": checkpoint_mae,
            "mean_abs_component_mean_difference": mean(
                abs(value) for value in checkpoint_mean_difference.values()
            ),
            "mean_checkpoint_mae": mean(checkpoint_mae.values()),
        },
        "lag1_autocorrelation": {
            "component_abs_gap": acf_gap,
            "mean_abs_gap": mean(acf_gap.values()),
        },
        "delta_distribution": {
            "component_ks_distance": delta_ks,
            "mean_ks_distance": mean(delta_ks.values()),
        },
        "decline_events": {
            "component_difference_target_minus_donor": decline_difference,
            "mean_abs_difference": mean(abs(value) for value in decline_difference.values()),
        },
        "phase_trends": {
            "component_slope_difference_target_minus_donor": slope_difference,
            "mean_abs_slope_difference": mean(abs(value) for value in slope_difference.values()),
            "component_phase_mean_difference_target_minus_donor": phase_mean_difference,
            "component_phase_mean_abs_gap": phase_mean_abs_gap,
            "mean_phase_mean_abs_gap": mean(phase_mean_abs_gap.values()),
        },
    }


def summarize_donor_matches(matches: list[dict], names: list[str]) -> dict:
    """Aggregate target-donor match diagnostics without selecting by similarity."""

    per_component = {
        name: {
            "mean_abs_score_mean_difference": mean(
                abs(match["matching"]["checkpoint_level"]
                    ["component_mean_difference_target_minus_donor"][name])
                for match in matches
            ),
            "mean_checkpoint_mae": mean(
                match["matching"]["checkpoint_level"]["component_checkpoint_mae"][name]
                for match in matches
            ),
            "mean_abs_lag1_autocorrelation_gap": mean(
                match["matching"]["lag1_autocorrelation"]["component_abs_gap"].get(name)
                for match in matches
            ),
            "mean_delta_ks_distance": mean(
                match["matching"]["delta_distribution"]["component_ks_distance"][name]
                for match in matches
            ),
            "mean_abs_decline_event_difference": mean(
                abs(match["matching"]["decline_events"]
                    ["component_difference_target_minus_donor"][name])
                for match in matches
            ),
            "mean_abs_slope_difference": mean(
                abs(match["matching"]["phase_trends"]
                    ["component_slope_difference_target_minus_donor"][name])
                for match in matches
            ),
            "mean_phase_mean_abs_gap": mean(
                match["matching"]["phase_trends"]["component_phase_mean_abs_gap"][name]
                for match in matches
            ),
        }
        for name in names
    }
    return {
        "n_target_donor_pairs": len(matches),
        "mean_abs_component_mean_difference": mean(
            match["matching"]["checkpoint_level"]["mean_abs_component_mean_difference"]
            for match in matches
        ),
        "mean_checkpoint_mae": mean(
            match["matching"]["checkpoint_level"]["mean_checkpoint_mae"] for match in matches
        ),
        "mean_abs_lag1_autocorrelation_gap": mean(
            match["matching"]["lag1_autocorrelation"]["mean_abs_gap"] for match in matches
        ),
        "mean_delta_ks_distance": mean(
            match["matching"]["delta_distribution"]["mean_ks_distance"] for match in matches
        ),
        "mean_abs_decline_event_difference": mean(
            match["matching"]["decline_events"]["mean_abs_difference"] for match in matches
        ),
        "mean_abs_slope_difference": mean(
            match["matching"]["phase_trends"]["mean_abs_slope_difference"]
            for match in matches
        ),
        "mean_phase_mean_abs_gap": mean(
            match["matching"]["phase_trends"]["mean_phase_mean_abs_gap"]
            for match in matches
        ),
        "per_component": per_component,
    }


def summarize_method(per_seed: list[dict]) -> dict:
    gap_keys = list(per_seed[0]["mean_dynamic_gap"].keys()) if per_seed else []
    component_names = list(per_seed[0]["source_metrics"]["decline_event_count"].keys()) if per_seed else []
    return {
        "n_seeds": len(per_seed),
        "mean_dynamic_gap": {
            key: mean(item["mean_dynamic_gap"][key] for item in per_seed) for key in gap_keys
        },
        "mean_source_decline_event_count": {
            name: mean(item["source_metrics"]["decline_event_count"][name] for item in per_seed)
            for name in component_names
        },
        "mean_placebo_decline_event_count": {
            name: mean(item["placebo_metrics"]["decline_event_count"][name] for item in per_seed)
            for name in component_names
        },
        "mean_source_action_checkpoints": mean(
            item["source_metrics"]["controller_action_checkpoints"] for item in per_seed
        ),
        "mean_source_component_actions": mean(
            item["source_metrics"]["controller_component_actions"] for item in per_seed
        ),
        "mean_placebo_action_checkpoints": mean(
            item["placebo_metrics"]["controller_action_checkpoints"] for item in per_seed
        ),
        "mean_placebo_component_actions": mean(
            item["placebo_metrics"]["controller_component_actions"] for item in per_seed
        ),
        "mean_source_final_weight_l1_drift": mean(
            item["source_metrics"]["final_weight_l1_drift"] for item in per_seed
        ),
        "mean_placebo_final_weight_l1_drift": mean(
            item["placebo_metrics"]["final_weight_l1_drift"] for item in per_seed
        ),
    }


def format_markdown(result: dict) -> str:
    frozen = result["frozen_circular"]
    cross_match = result["cross_seed_target_donor"]
    component_names = result["parameters"]["component_names"]
    lines = [
        "# Frozen Temporal Placebo Audit",
        "",
        "The formal circular placebo uses only replicate indices 0, 1, and 2. "
        f"The schedule seed is `{frozen['schedule_seed']}` and the eligible shift rule is "
        f"`ceil(T/4) <= s <= floor(3T/4)`; with T={frozen['history_length']}, "
        f"that is {frozen['eligible_shift_min']} through {frozen['eligible_shift_max']}.",
        "",
        "## Frozen Circular Schedule",
        "",
        "| Target seed | Circular replicate 0 | Circular replicate 1 | Circular replicate 2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for mapping in frozen["shift_mapping"]:
        selected = mapping["selected_shifts"]
        lines.append(
            f"| {mapping['seed']} | {selected['0']} | {selected['1']} | {selected['2']} |"
        )
    lines.extend(
        [
            "",
            "The three values in every row are distinct. They are selected before outcome analysis "
            "from the middle-half candidate set, not from the full set of non-zero offsets.",
            "",
            "## Replay-Source Dynamics",
            "",
            "All gaps in this table compare a replay with the trajectory it replays. "
            "For cross-seed replay, this is the donor, so zero confirms replay fidelity rather than "
            "target-donor phase matching.",
            "",
            "| Placebo | Seeds | Lag-1 ACF gap | Delta KS | Decline-count gap | Action-checkpoint gap | Final-weight L1 gap | Score-corr gap | Delta-corr gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, payload in result["methods"].items():
        summary = payload["summary"]
        gap = summary["mean_dynamic_gap"]
        value = lambda key: "NA" if gap[key] is None else f"{gap[key]:.4f}"
        lines.append(
            "| {name} | {n} | {lag} | {ks} | {declines} | {actions} | {weights} | {score_corr} | {delta_corr} |".format(
                name=name,
                n=summary["n_seeds"],
                lag=value("mean_abs_lag1_autocorrelation_gap"),
                ks=value("mean_delta_ks_distance"),
                declines=value("mean_abs_decline_event_count_gap"),
                actions=value("abs_controller_action_checkpoint_gap"),
                weights=value("final_weight_l1_gap"),
                score_corr=value("mean_score_correlation_gap"),
                delta_corr=value("mean_delta_correlation_gap"),
            )
        )
    lines.extend(
        [
            "",
            "| Placebo | Action checkpoints (source -> placebo) | Component actions (source -> placebo) | "
            + " | ".join(
                f"{component} declines (source -> placebo)" for component in component_names
            )
            + " | Weight L1 drift (source -> placebo) |",
            "| --- | ---: | ---: | "
            + " | ".join("---:" for _ in component_names)
            + " | ---: |",
        ]
    )
    for name, payload in result["methods"].items():
        summary = payload["summary"]
        declines = " | ".join(
            f"{summary['mean_source_decline_event_count'][component]:.2f} -> "
            f"{summary['mean_placebo_decline_event_count'][component]:.2f}"
            for component in component_names
        )
        lines.append(
            f"| {name} | {summary['mean_source_action_checkpoints']:.2f} -> "
            f"{summary['mean_placebo_action_checkpoints']:.2f} | "
            f"{summary['mean_source_component_actions']:.2f} -> "
            f"{summary['mean_placebo_component_actions']:.2f} | {declines} | "
            f"{summary['mean_source_final_weight_l1_drift']:.4f} -> "
            f"{summary['mean_placebo_final_weight_l1_drift']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`circular_shift_frozen_mean` is the formal per-seed mean over replicate indices 0, 1, "
            "and 2. The individual replica rows are retained so a favorable pooled result cannot "
            "mask a poor selected shift.",
            "",
            "## Cross-Seed Target-Donor Preflight",
            "",
            "Mapping is frozen as the supplied seed order circularly shifted by one: target i uses "
            "donor i+1 (mod n). It is one-to-one, never self-replays, and does not use any outcome "
            "or similarity-based donor selection.",
            "",
            "| Target seed | Donor seed | Mean score difference | Checkpoint MAE | ACF gap | Delta KS | Decline-count gap | Slope gap | Phase-mean gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair in cross_match["pairs"]:
        matching = pair["matching"]
        lines.append(
            f"| {pair['target_seed']} | {pair['donor_seed']} | "
            f"{matching['checkpoint_level']['mean_abs_component_mean_difference']:.4f} | "
            f"{matching['checkpoint_level']['mean_checkpoint_mae']:.4f} | "
            f"{matching['lag1_autocorrelation']['mean_abs_gap']:.4f} | "
            f"{matching['delta_distribution']['mean_ks_distance']:.4f} | "
            f"{matching['decline_events']['mean_abs_difference']:.4f} | "
            f"{matching['phase_trends']['mean_abs_slope_difference']:.4f} | "
            f"{matching['phase_trends']['mean_phase_mean_abs_gap']:.4f} |"
        )
    summary = cross_match["summary"]
    lines.extend(
        [
            "",
            "Across the frozen mapping: mean score difference "
            f"`{summary['mean_abs_component_mean_difference']:.4f}`, checkpoint MAE "
            f"`{summary['mean_checkpoint_mae']:.4f}`, ACF gap "
            f"`{summary['mean_abs_lag1_autocorrelation_gap']:.4f}`, Delta KS "
            f"`{summary['mean_delta_ks_distance']:.4f}`, decline-count gap "
            f"`{summary['mean_abs_decline_event_difference']:.4f}`, slope gap "
            f"`{summary['mean_abs_slope_difference']:.4f}`, and phase-mean gap "
            f"`{summary['mean_phase_mean_abs_gap']:.4f}`.",
            "",
            "| Component | Mean score difference | Checkpoint MAE | ACF gap | Delta KS | Decline-count gap | Slope gap | Phase-mean gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for component in component_names:
        values = summary["per_component"][component]
        lines.append(
            f"| {component} | {values['mean_abs_score_mean_difference']:.4f} | "
            f"{values['mean_checkpoint_mae']:.4f} | "
            f"{values['mean_abs_lag1_autocorrelation_gap']:.4f} | "
            f"{values['mean_delta_ks_distance']:.4f} | "
            f"{values['mean_abs_decline_event_difference']:.4f} | "
            f"{values['mean_abs_slope_difference']:.4f} | "
            f"{values['mean_phase_mean_abs_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Target-minus-donor signed component means, per-checkpoint values, and early/middle/late "
            "phase profiles are retained in the JSON output. Lower is better for all absolute-gap and "
            "distance columns in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-history", action="append", required=True)
    parser.add_argument("--shuffle-history", action="append", default=[])
    parser.add_argument("--component-names", required=True)
    parser.add_argument("--deadband", type=float, required=True)
    parser.add_argument("--action-step", type=float, required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--decrease-on-improvement", action="store_true")
    parser.add_argument(
        "--circular-schedule-seed",
        type=int,
        default=51001,
        help="Frozen seed used to order middle-half circular-shift candidates.",
    )
    parser.add_argument(
        "--circular-replicate-indices",
        default="0,1,2",
        help="Comma-separated frozen circular schedule indices, averaged within seed.",
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--block-seed", type=int, default=51000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument(
        "--frozen-manifest-output",
        default=None,
        help="Optional JSON manifest that records the frozen circular and donor mappings.",
    )
    args = parser.parse_args()

    names = parse_component_names(args.component_names)
    initial_weights = parse_initial_weights(args.initial_weights, names)
    real_paths = [Path(value) for value in args.real_history]
    shuffle_paths = [Path(value) for value in args.shuffle_history]
    if shuffle_paths and len(shuffle_paths) != len(real_paths):
        raise RuntimeError("Provide either zero or exactly one --shuffle-history per --real-history")

    trajectories = [read_history(path, names) for path in real_paths]
    if len({len(trajectory) for trajectory in trajectories}) != 1:
        raise RuntimeError("All real histories must have the same validation length for phase matching")
    history_length = len(trajectories[0])
    replicate_indices = [
        int(value) for value in args.circular_replicate_indices.split(",") if value.strip()
    ]
    if not replicate_indices or len(set(replicate_indices)) != len(replicate_indices):
        raise RuntimeError("--circular-replicate-indices must contain unique integers")
    eligible_shifts = eligible_circular_shifts(history_length)
    if any(index < 0 or index >= len(eligible_shifts) for index in replicate_indices):
        raise RuntimeError(
            "Each circular replicate index must select an eligible middle-half shift; "
            f"available indices are 0 through {len(eligible_shifts) - 1}"
        )

    metric_args = dict(
        names=names,
        deadband=args.deadband,
        action_step=args.action_step,
        initial_weights=initial_weights,
        decrease_on_improvement=args.decrease_on_improvement,
    )
    source_metrics = [metrics(trajectory, **metric_args) for trajectory in trajectories]
    methods: dict[str, dict] = {}

    if shuffle_paths:
        per_seed = []
        for index, (real_path, shuffle_path, source) in enumerate(
            zip(real_paths, shuffle_paths, source_metrics)
        ):
            placebo = metrics(read_history(shuffle_path, names), **metric_args)
            per_seed.append(
                {
                    "seed": seed_label(real_path, index),
                    "source_history": str(real_path),
                    "placebo_history": str(shuffle_path),
                    "source_metrics": source,
                    "placebo_metrics": placebo,
                    "mean_dynamic_gap": dynamic_gap(placebo, source, names),
                }
            )
        methods["legacy_shuffle_history"] = {"per_seed": per_seed, "summary": summarize_method(per_seed)}

    circular_by_replicate = {replicate_index: [] for replicate_index in replicate_indices}
    circular_frozen_mean = []
    shift_mapping = []
    for index, (real_path, trajectory, source) in enumerate(
        zip(real_paths, trajectories, source_metrics)
    ):
        schedule = automatic_shift_schedule(
            args.circular_schedule_seed,
            file_digest(real_path),
            history_length,
        )
        selected_shifts = {replicate_index: schedule[replicate_index] for replicate_index in replicate_indices}
        shift_mapping.append(
            {
                "seed": seed_label(real_path, index),
                "target_history": str(real_path),
                "target_history_sha256": file_digest(real_path),
                "eligible_shifts": eligible_shifts,
                "selected_shifts": {str(key): value for key, value in selected_shifts.items()},
            }
        )
        variants = []
        variant_gaps = []
        for replicate_index, shift in selected_shifts.items():
            placebo = metrics(trajectory[shift:] + trajectory[:shift], **metric_args)
            gap = dynamic_gap(placebo, source, names)
            variants.append(placebo)
            variant_gaps.append(gap)
            circular_by_replicate[replicate_index].append(
                {
                    "seed": seed_label(real_path, index),
                    "source_history": str(real_path),
                    "shift": shift,
                    "replicate_index": replicate_index,
                    "source_metrics": source,
                    "placebo_metrics": placebo,
                    "mean_dynamic_gap": gap,
                }
            )
        circular_frozen_mean.append(
            {
                "seed": seed_label(real_path, index),
                "source_history": str(real_path),
                "selected_shifts": {str(key): value for key, value in selected_shifts.items()},
                "source_metrics": source,
                "placebo_metrics": mean_variant_metrics(variants),
                "mean_dynamic_gap": {
                    key: mean(gap[key] for gap in variant_gaps) for key in variant_gaps[0]
                },
            }
        )
    for replicate_index, per_seed in circular_by_replicate.items():
        methods[f"circular_shift_replicate_{replicate_index}"] = {
            "per_seed": per_seed,
            "summary": summarize_method(per_seed),
        }
    methods["circular_shift_frozen_mean"] = {
        "per_seed": circular_frozen_mean,
        "summary": summarize_method(circular_frozen_mean),
    }
    frozen_circular = {
        "schedule_seed": args.circular_schedule_seed,
        "history_length": history_length,
        "eligible_shift_min": eligible_shifts[0],
        "eligible_shift_max": eligible_shifts[-1],
        "eligible_shifts": eligible_shifts,
        "replicate_indices": replicate_indices,
        "shift_mapping": shift_mapping,
    }

    cross_per_seed = []
    cross_target_donor_pairs = []
    for index, (real_path, target_trajectory, source) in enumerate(
        zip(real_paths, trajectories, source_metrics)
    ):
        donor_index = (index + 1) % len(trajectories)
        donor_path = real_paths[donor_index]
        donor_trajectory = trajectories[donor_index]
        donor = source_metrics[donor_index]
        matching = donor_target_match(
            target_trajectory,
            donor_trajectory,
            source,
            donor,
            names,
        )
        cross_target_donor_pairs.append(
            {
                "target_seed": seed_label(real_path, index),
                "donor_seed": seed_label(donor_path, donor_index),
                "target_history": str(real_path),
                "target_history_sha256": file_digest(real_path),
                "donor_history": str(donor_path),
                "donor_history_sha256": file_digest(donor_path),
                "matching": matching,
            }
        )
        cross_per_seed.append(
            {
                "seed": seed_label(real_path, index),
                "source_history": str(donor_path),
                "target_history": str(real_path),
                "source_metrics": donor,
                "placebo_metrics": donor,
                "mean_dynamic_gap": dynamic_gap(donor, donor, names),
                "target_vs_donor_dynamic_gap": dynamic_gap(donor, source, names),
            }
        )
    methods["cross_seed_phase_matched"] = {
        "per_seed": cross_per_seed,
        "summary": summarize_method(cross_per_seed),
        "note": "Zero source gap is expected: phase-matched replay exactly reuses its donor trajectory. "
        "Target-vs-donor variation is reported separately per seed and is not a replay artifact.",
    }
    cross_seed_target_donor = {
        "mapping_rule": "Supplied real histories in order: target i -> donor i + 1 (mod n)",
        "pairs": cross_target_donor_pairs,
        "summary": summarize_donor_matches(cross_target_donor_pairs, names),
    }

    block_per_seed = []
    for index, (real_path, trajectory, source) in enumerate(
        zip(real_paths, trajectories, source_metrics)
    ):
        candidate = metrics(
            block_permutation(trajectory, args.block_size, args.block_seed + index), **metric_args
        )
        block_per_seed.append(
            {
                "seed": seed_label(real_path, index),
                "source_history": str(real_path),
                "block_size": args.block_size,
                "block_seed": args.block_seed + index,
                "source_metrics": source,
                "placebo_metrics": candidate,
                "mean_dynamic_gap": dynamic_gap(candidate, source, names),
            }
        )
    methods["block_permutation"] = {
        "per_seed": block_per_seed,
        "summary": summarize_method(block_per_seed),
    }

    result = {
        "purpose": "Offline controller-input temporal placebo audit",
        "parameters": {
            "component_names": names,
            "deadband": args.deadband,
            "action_step": args.action_step,
            "initial_weights": initial_weights,
            "decrease_on_improvement": args.decrease_on_improvement,
            "circular_schedule_seed": args.circular_schedule_seed,
            "circular_replicate_indices": replicate_indices,
            "circular_eligible_shifts": eligible_shifts,
            "cross_seed_rule": "supplied target i -> donor i + 1 (mod n), same validation index",
            "block_size": args.block_size,
        },
        "methods": methods,
        "frozen_circular": frozen_circular,
        "cross_seed_target_donor": cross_seed_target_donor,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    Path(args.markdown_output).write_text(format_markdown(result))
    if args.frozen_manifest_output:
        manifest = {
            "schema_version": 1,
            "purpose": "Frozen temporal placebo plan",
            "controller": {
                "component_names": names,
                "deadband": args.deadband,
                "action_step": args.action_step,
                "initial_weights": initial_weights,
                "decrease_on_improvement": args.decrease_on_improvement,
            },
            "circular_shift": {
                **frozen_circular,
                "within_seed_analysis": "Mean placebo outcomes over replicate indices 0, 1, and 2 before cross-seed analysis.",
            },
            "cross_seed_phase_matched": {
                "mapping_rule": cross_seed_target_donor["mapping_rule"],
                "selection_uses_similarity_or_outcomes": False,
                "pairs": [
                    {
                        key: pair[key]
                        for key in (
                            "target_seed",
                            "donor_seed",
                            "target_history",
                            "target_history_sha256",
                            "donor_history",
                            "donor_history_sha256",
                        )
                    }
                    for pair in cross_target_donor_pairs
                ],
            },
        }
        Path(args.frozen_manifest_output).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
