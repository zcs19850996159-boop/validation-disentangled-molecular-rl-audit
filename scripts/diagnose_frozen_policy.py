#!/usr/bin/env python
"""
Diagnose the initial action bias of DynamicWeightController.

This script freezes the policy at initialization: no policy-gradient update and
no weight update are performed. It feeds an existing validation-score trajectory
through the initialized policy and reports both exact sampled actions and the
underlying action probabilities.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTION_VALUES = [-1, 0, 1]


def load_controller_module(path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_weight_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import controller module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def load_validation_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "training_step": record.get("training_step"),
                    "validation_index": record.get("validation_index"),
                    "validation_scores": dict(record["validation_scores"]),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if len(rows) < 2:
        raise RuntimeError("Need at least two validation rows for a frozen-policy trajectory")
    return rows


def make_controller(module, component_names: list[str], weights: list[float], args, seed: int):
    return module.DynamicWeightController(
        component_names=component_names,
        validation_fn=lambda _smiles: {name: 0.0 for name in component_names},
        init_weights=dict(zip(component_names, weights)),
        action_step=args.action_step,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        reward_baseline_beta=args.reward_baseline_beta,
        standardize_advantage=args.standardize_advantage,
        reward_variance_beta=args.reward_variance_beta,
        reward_deadband=args.reward_deadband,
        validation_floor_penalty=args.validation_floor_penalty,
        device=args.device,
        seed=seed,
        history_path="",
    )


def logits_and_probs(controller, scores: dict[str, float]):
    torch = controller._torch
    with torch.no_grad():
        state = controller._state_tensor(scores)
        logits = controller._policy(state).view(len(controller.component_names), 3)
        probs = torch.softmax(logits, dim=-1)
    return (
        [[float(v) for v in row] for row in logits.detach().cpu().tolist()],
        [[float(v) for v in row] for row in probs.detach().cpu().tolist()],
    )


def net_actions(actions_by_component: list[dict[str, int]], component_names: list[str]) -> dict[str, int]:
    return {
        name: int(sum(int(row.get(name, 0)) for row in actions_by_component))
        for name in component_names
    }


def exact_trajectory(module, rows, component_names, weights, args, seed: int) -> dict[str, Any]:
    controller = make_controller(module, component_names, weights, args, seed)
    trajectory = []
    sampled_actions = []
    previous_scores = None

    for row in rows:
        scores = row["validation_scores"]
        controller._last_validation_scores = previous_scores

        if previous_scores is None and not args.include_first:
            actions = {name: 0 for name in component_names}
            sampled = False
            logits, probs = logits_and_probs(controller, scores)
        else:
            logits, probs = logits_and_probs(controller, scores)
            state = controller._state_tensor(scores)
            sampled_values, _log_prob, _entropy = controller._sample_action(state)
            actions = dict(zip(component_names, sampled_values))
            sampled = True
            sampled_actions.append(actions)

        trajectory.append(
            {
                "training_step": row["training_step"],
                "validation_index": row["validation_index"],
                "sampled": sampled,
                "actions": actions,
                "probs_by_component": {
                    name: dict(zip([str(v) for v in ACTION_VALUES], probs[idx]))
                    for idx, name in enumerate(component_names)
                },
                "logits_by_component": {
                    name: dict(zip([str(v) for v in ACTION_VALUES], logits[idx]))
                    for idx, name in enumerate(component_names)
                },
            }
        )
        previous_scores = dict(scores)

    return {
        "seed": seed,
        "n_sampled_states": len(sampled_actions),
        "net_actions": net_actions(sampled_actions, component_names),
        "trajectory": trajectory,
    }


def accumulate_component_stats(stats, component_names, probs, logits, actions=None):
    for idx, name in enumerate(component_names):
        if actions is None:
            stats[name]["states"] += 1
            for action_idx, action_value in enumerate(ACTION_VALUES):
                key = str(action_value)
                stats[name]["prob_sum"][key] += float(probs[idx][action_idx])
                stats[name]["logit_sum"][key] += float(logits[idx][action_idx])
        else:
            action = int(actions[idx])
            stats[name]["action_counts"][str(action)] += 1
            stats[name]["action_sum"] += action
            stats[name]["samples"] += 1


def normalize_stats(stats, component_names):
    result = {}
    for name in component_names:
        states = max(1, int(stats[name]["states"]))
        samples = max(1, int(stats[name]["samples"]))
        result[name] = {
            "mean_probs": {
                str(action): stats[name]["prob_sum"][str(action)] / states
                for action in ACTION_VALUES
            },
            "mean_logits": {
                str(action): stats[name]["logit_sum"][str(action)] / states
                for action in ACTION_VALUES
            },
            "sampled_action_counts": {
                str(action): int(stats[name]["action_counts"][str(action)])
                for action in ACTION_VALUES
            },
            "sampled_action_probs": {
                str(action): stats[name]["action_counts"][str(action)] / samples
                for action in ACTION_VALUES
            },
            "sampled_mean_action": stats[name]["action_sum"] / samples,
            "n_probability_states": states,
            "n_action_samples": int(stats[name]["samples"]),
        }
    return result


def distribution_summary(module, rows, component_names, weights, args, seeds: list[int]):
    import torch

    aggregate_stats = defaultdict(
        lambda: {
            "states": 0,
            "samples": 0,
            "prob_sum": defaultdict(float),
            "logit_sum": defaultdict(float),
            "action_counts": defaultdict(int),
            "action_sum": 0.0,
        }
    )
    per_seed = {}

    for seed in seeds:
        controller = make_controller(module, component_names, weights, args, seed)
        seed_stats = defaultdict(
            lambda: {
                "states": 0,
                "samples": 0,
                "prob_sum": defaultdict(float),
                "logit_sum": defaultdict(float),
                "action_counts": defaultdict(int),
                "action_sum": 0.0,
            }
        )
        previous_scores = None

        for row in rows:
            scores = row["validation_scores"]
            controller._last_validation_scores = previous_scores
            if previous_scores is None and not args.include_first:
                previous_scores = dict(scores)
                continue

            logits, probs = logits_and_probs(controller, scores)
            accumulate_component_stats(seed_stats, component_names, probs, logits)
            accumulate_component_stats(aggregate_stats, component_names, probs, logits)

            state = controller._state_tensor(scores)
            for _ in range(args.samples_per_state):
                sampled_values, _log_prob, _entropy = controller._sample_action(state)
                accumulate_component_stats(
                    seed_stats, component_names, probs, logits, sampled_values
                )
                accumulate_component_stats(
                    aggregate_stats, component_names, probs, logits, sampled_values
                )

            previous_scores = dict(scores)

        per_seed[str(seed)] = normalize_stats(seed_stats, component_names)

    return {
        "aggregate": normalize_stats(aggregate_stats, component_names),
        "per_seed": per_seed,
        "torch_version": torch.__version__,
    }


def validation_signal_summary(rows, component_names: list[str], include_first: bool) -> dict[str, Any]:
    values = {name: [] for name in component_names}
    deltas = {name: [] for name in component_names}
    previous = None
    for row in rows:
        scores = row["validation_scores"]
        for name in component_names:
            values[name].append(float(scores.get(name, 0.0)))
            if previous is not None or include_first:
                prev = float(previous.get(name, scores.get(name, 0.0))) if previous else 0.0
                deltas[name].append(float(scores.get(name, 0.0)) - prev)
        previous = scores

    summary = {}
    for name in component_names:
        vals = values[name]
        ds = deltas[name]
        summary[name] = {
            "mean_value": sum(vals) / len(vals),
            "min_value": min(vals),
            "max_value": max(vals),
            "mean_delta": sum(ds) / len(ds) if ds else 0.0,
            "positive_deltas": sum(1 for value in ds if value > 0),
            "negative_deltas": sum(1 for value in ds if value < 0),
            "zero_deltas": sum(1 for value in ds if value == 0),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="03_dynamic_weight_controller.py")
    parser.add_argument("--history", default="controller_history_drd2_qsar_fixed_seed17.jsonl")
    parser.add_argument("--component-names", default="DRD2_activity,QED,Molecular weight")
    parser.add_argument("--init-weights", default="0.34,0.33,0.33")
    parser.add_argument("--seeds", default="17,23,101,202,303")
    parser.add_argument("--exact-seed", type=int, default=17)
    parser.add_argument("--samples-per-state", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-first", action="store_true")
    parser.add_argument("--action-step", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--reward-baseline-beta", type=float, default=0.9)
    parser.add_argument("--standardize-advantage", action="store_true")
    parser.add_argument("--reward-variance-beta", type=float, default=0.9)
    parser.add_argument("--reward-deadband", type=float, default=0.0)
    parser.add_argument("--validation-floor-penalty", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    component_names = parse_csv_list(args.component_names)
    weights = parse_float_list(args.init_weights)
    if len(component_names) != len(weights):
        raise RuntimeError("--init-weights must have one value per component")

    seeds = [int(value) for value in parse_csv_list(args.seeds)]
    module = load_controller_module(Path(args.controller))
    rows = load_validation_rows(Path(args.history), limit=args.limit)

    output = {
        "diagnostic": "frozen_initial_policy",
        "controller": str(Path(args.controller)),
        "history": str(Path(args.history)),
        "component_names": component_names,
        "initial_weights": dict(zip(component_names, weights)),
        "include_first": bool(args.include_first),
        "n_validation_rows": len(rows),
        "n_sampled_states": len(rows) if args.include_first else len(rows) - 1,
        "samples_per_state": args.samples_per_state,
        "seeds": seeds,
        "exact_trajectory": exact_trajectory(
            module, rows, component_names, weights, args, args.exact_seed
        ),
        "distribution": distribution_summary(module, rows, component_names, weights, args, seeds),
        "validation_signal_summary": validation_signal_summary(
            rows, component_names, args.include_first
        ),
    }

    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
