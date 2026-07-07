#!/usr/bin/env python
"""Run a synthetic positive-control sensitivity ladder.

The existing positive control shows that the diagnostic pipeline detects a very
strong validation-weight relationship. This ladder estimates where that
detection starts to disappear as the synthetic validation trend approaches the
deadband/noise scale.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from positive_control_validation_sensitivity import load_controller_module, mean, run_controller


DEFAULT_LEVELS = [
    {"label": "strong_every_window", "event_drop": 0.035, "event_every": 1},
    {"label": "medium_every_2_windows", "event_drop": 0.035, "event_every": 2},
    {"label": "medium_every_3_windows", "event_drop": 0.035, "event_every": 3},
    {"label": "near_noise_every_window", "event_drop": 0.008, "event_every": 1},
]


def ladder_validation_series(
    n_updates: int,
    event_drop: float,
    event_every: int,
    target_b_noise: float,
) -> list[dict[str, float]]:
    """Create a stream with controlled frequency of detectable A drops."""

    import math

    series = []
    target_a = 0.95
    for idx in range(n_updates):
        series.append(
            {
                "Target_A": target_a,
                "Target_B": 0.50 + target_b_noise * math.sin(idx),
            }
        )
        if idx < n_updates - 1 and (idx + 1) % event_every == 0:
            target_a -= event_drop
    return series


def run_level(
    module,
    level: dict,
    updates: int,
    n_shuffles: int,
    seed: int,
    target_b_noise: float,
) -> dict:
    series = ladder_validation_series(
        updates,
        event_drop=float(level["event_drop"]),
        event_every=int(level["event_every"]),
        target_b_noise=target_b_noise,
    )
    real = run_controller(module, series)

    rng = random.Random(seed)
    shuffled = []
    for _idx in range(n_shuffles):
        shuffled_series = list(series)
        rng.shuffle(shuffled_series)
        shuffled.append(run_controller(module, shuffled_series))

    random_final_a = [row["final_weight_target_a"] for row in shuffled]
    random_action_a = [row["target_a_action_sum"] for row in shuffled]
    rand_mean = mean(random_final_a)
    rand_sd = statistics.stdev(random_final_a) if len(random_final_a) > 1 else 0.0
    delta = real["final_weight_target_a"] - rand_mean
    z = None if rand_sd == 0.0 else delta / rand_sd
    empirical_p = (
        (1 + sum(1 for value in random_final_a if value >= real["final_weight_target_a"]))
        / (len(random_final_a) + 1)
    )

    return {
        "label": str(level["label"]),
        "event_drop": float(level["event_drop"]),
        "event_every": int(level["event_every"]),
        "detectable_drop_events": sum(
            1
            for prev, curr in zip(series, series[1:])
            if curr["Target_A"] - prev["Target_A"] < -0.01
        ),
        "total_target_a_decline": series[0]["Target_A"] - series[-1]["Target_A"],
        "updates": updates,
        "n_shuffles": n_shuffles,
        "real": {key: value for key, value in real.items() if key != "history"},
        "shuffled_control": {
            "final_weight_target_a_mean": rand_mean,
            "final_weight_target_a_sd": rand_sd,
            "final_weight_target_a_min": min(random_final_a),
            "final_weight_target_a_max": max(random_final_a),
            "target_a_action_sum_mean": mean(random_action_a),
        },
        "sensitivity": {
            "delta_final_weight_target_a": delta,
            "z_vs_shuffled": z,
            "empirical_one_sided_p": empirical_p,
            "passes_positive_control": (
                real["final_weight_target_a"] > rand_mean
                and empirical_p <= 0.05
                and (z is not None and z >= 2.0)
            ),
        },
    }


def parse_levels(raw_levels: list[str] | None) -> list[dict]:
    if not raw_levels:
        return list(DEFAULT_LEVELS)
    parsed = []
    for raw in raw_levels:
        parts = raw.split(":")
        if len(parts) != 3:
            raise RuntimeError("--level must use LABEL:EVENT_DROP:EVENT_EVERY")
        label, event_drop, event_every = parts
        parsed.append({"label": label, "event_drop": float(event_drop), "event_every": int(event_every)})
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="03_dynamic_weight_controller.py")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--target-b-noise", type=float, default=0.003)
    parser.add_argument("--level", action="append", help="Sensitivity level as LABEL:EVENT_DROP:EVENT_EVERY")
    parser.add_argument("--output", default="positive_control_sensitivity_ladder.json")
    args = parser.parse_args()

    module = load_controller_module(Path(args.controller))
    levels = []
    for idx, level in enumerate(parse_levels(args.level)):
        levels.append(
            run_level(
                module,
                level=level,
                updates=args.updates,
                n_shuffles=args.n_shuffles,
                seed=args.seed + idx * 1009,
                target_b_noise=args.target_b_noise,
            )
        )

    result = {
        "purpose": (
            "Sensitivity ladder for the positive-control diagnostic. It estimates "
            "the synthetic effect scale at which validation-driven weight updates "
            "become indistinguishable from shuffled controls."
        ),
        "controller_mode": "deterministic",
        "controller_action_step": 0.10,
        "controller_reward_deadband": 0.01,
        "level_design": (
            "Signal strength is varied by the frequency of validation windows "
            "whose Target_A drop exceeds the controller deadband. This is more "
            "informative for the thresholded deterministic controller than merely "
            "shrinking a per-window slope."
        ),
        "levels": levels,
        "interpretation": (
            "Effects above the first failing level are detectable by this "
            "diagnostic; effects at or below failing levels lie near the "
            "current detection blind zone."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
