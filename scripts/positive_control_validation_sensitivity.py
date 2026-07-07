#!/usr/bin/env python
"""
Positive-control sensitivity test for validation-driven weight updates.

This does not simulate molecular generation. It is a deliberately synthetic
test of the detection/controller-analysis pipeline: when the validation stream
contains a strong, explicit signal that objective A is deteriorating, the
deterministic controller should consistently upweight A. Shuffled controls keep
the same marginal validation values but break the temporal relationship.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path


def load_controller_module(path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_weight_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import controller module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validation_series(
    n_updates: int,
    target_a_slope: float = 0.035,
    target_b_noise: float = 0.003,
    target_a_start: float = 0.95,
    target_b_center: float = 0.50,
) -> list[dict[str, float]]:
    """Positive-control stream: Target_A steadily deteriorates."""

    series = []
    for idx in range(n_updates):
        series.append(
            {
                "Target_A": target_a_start - target_a_slope * idx,
                "Target_B": target_b_center + target_b_noise * math.sin(idx),
            }
        )
    return series


def run_controller(module, series: list[dict[str, float]], history_path: str = "") -> dict:
    controller = module.DeterministicValidationController(
        component_names=["Target_A", "Target_B"],
        validation_fn=lambda _smiles: {},
        init_weights={"Target_A": 0.50, "Target_B": 0.50},
        action_step=0.10,
        reward_deadband=0.01,
        seed=17,
        history_path=history_path,
    )
    for idx, scores in enumerate(series, start=1):
        controller._step_count = idx * 15
        controller.step(
            training_component_scores=None,
            validation_scores=scores,
            num_validation_smiles=32,
        )

    history = controller.history()
    final = history[-1]["new_weights"]
    actions = [row["actions"] for row in history]
    return {
        "final_weight_target_a": float(final["Target_A"]),
        "final_weight_target_b": float(final["Target_B"]),
        "target_a_action_sum": sum(int(action["Target_A"]) for action in actions),
        "target_b_action_sum": sum(int(action["Target_B"]) for action in actions),
        "actions_applied": sum(1 for row in history if row.get("action_applied")),
        "history": history,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="03_dynamic_weight_controller.py")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--n-shuffles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--target-a-slope", type=float, default=0.035)
    parser.add_argument("--target-b-noise", type=float, default=0.003)
    parser.add_argument("--target-a-start", type=float, default=0.95)
    parser.add_argument("--output", default="positive_control_validation_sensitivity.json")
    parser.add_argument("--history", default="positive_control_real_history.jsonl")
    args = parser.parse_args()

    module = load_controller_module(Path(args.controller))
    series = validation_series(
        args.updates,
        target_a_slope=args.target_a_slope,
        target_b_noise=args.target_b_noise,
        target_a_start=args.target_a_start,
    )
    real = run_controller(module, series, history_path=args.history)

    rng = random.Random(args.seed)
    shuffled = []
    for idx in range(args.n_shuffles):
        shuffled_series = list(series)
        rng.shuffle(shuffled_series)
        shuffled.append(run_controller(module, shuffled_series))

    random_final_a = [row["final_weight_target_a"] for row in shuffled]
    random_action_a = [row["target_a_action_sum"] for row in shuffled]
    rand_mean = mean(random_final_a)
    rand_sd = statistics.stdev(random_final_a) if len(random_final_a) > 1 else 0.0
    z = None if rand_sd == 0.0 else (real["final_weight_target_a"] - rand_mean) / rand_sd
    empirical_p = (
        (1 + sum(1 for value in random_final_a if value >= real["final_weight_target_a"]))
        / (len(random_final_a) + 1)
    )

    result = {
        "purpose": (
            "Positive control: verify that the pipeline detects a strong synthetic "
            "validation signal when one exists."
        ),
        "counterfactual_caveat": (
            "This is not a molecular-generation experiment; it is a sensitivity "
            "test of validation-driven weight updates and random-control analysis."
        ),
        "controller_mode": "deterministic",
        "updates": args.updates,
        "n_shuffles": args.n_shuffles,
        "target_a_slope": args.target_a_slope,
        "target_b_noise": args.target_b_noise,
        "target_a_start": args.target_a_start,
        "real": {
            key: value for key, value in real.items() if key != "history"
        },
        "shuffled_control": {
            "final_weight_target_a_mean": rand_mean,
            "final_weight_target_a_sd": rand_sd,
            "final_weight_target_a_min": min(random_final_a),
            "final_weight_target_a_max": max(random_final_a),
            "target_a_action_sum_mean": mean(random_action_a),
        },
        "sensitivity": {
            "delta_final_weight_target_a": real["final_weight_target_a"] - rand_mean,
            "z_vs_shuffled": z,
            "empirical_one_sided_p": empirical_p,
            "passes_positive_control": (
                real["final_weight_target_a"] > rand_mean
                and empirical_p <= 0.05
                and (z is not None and z >= 2.0)
            ),
        },
        "synthetic_series": series,
    }

    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
