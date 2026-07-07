#!/usr/bin/env python
"""
Run a pure DynamicWeightController constant-signal diagnostic.

This isolates the controller update rule from REINVENT sampling, QSAR noise, and
training dynamics. With a constant validation signal, a validation-driven
controller should not drift systematically.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def parse_scores(raw: str, component_names: list[str]) -> dict[str, float]:
    if not raw.strip():
        return {name: 0.0 for name in component_names}
    scores = {name: 0.0 for name in component_names}
    for item in raw.split(","):
        if not item.strip():
            continue
        name, value = item.split("=", 1)
        scores[name.strip()] = float(value)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="03_dynamic_weight_controller.py")
    parser.add_argument("--component-names", default="DRD2_activity,QED,Molecular weight")
    parser.add_argument("--init-weights", default="0.34,0.33,0.33")
    parser.add_argument("--scores", default="")
    parser.add_argument("--updates", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--action-step", type=float, default=0.15)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--reward-deadband", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    component_names = [name.strip() for name in args.component_names.split(",") if name.strip()]
    weights = [float(value.strip()) for value in args.init_weights.split(",") if value.strip()]
    if len(component_names) != len(weights):
        raise RuntimeError("--init-weights must have one value per component")

    scores = parse_scores(args.scores, component_names)
    module = load_controller_module(Path(args.controller))
    controller = module.DynamicWeightController(
        component_names=component_names,
        validation_fn=lambda _smiles: scores,
        init_weights=dict(zip(component_names, weights)),
        action_step=args.action_step,
        entropy_coef=args.entropy_coef,
        reward_deadband=args.reward_deadband,
        seed=args.seed,
        history_path="",
    )

    for idx in range(args.updates):
        controller._step_count = (idx + 1) * 20
        controller.step(
            training_component_scores=None,
            validation_scores=scores,
            num_validation_smiles=64,
        )

    history = controller.history()
    summary = {
        "component_names": component_names,
        "constant_scores": scores,
        "updates": args.updates,
        "seed": args.seed,
        "action_step": args.action_step,
        "entropy_coef": args.entropy_coef,
        "reward_deadband": args.reward_deadband,
        "initial_weights": dict(zip(component_names, weights)),
        "final_weights": history[-1]["new_weights"],
        "actions": [row["actions"] for row in history],
        "action_applied": [row.get("action_applied", True) for row in history],
        "rewards": [row["controller_reward"] for row in history],
    }

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
