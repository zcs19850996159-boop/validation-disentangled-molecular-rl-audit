#!/usr/bin/env python
"""Unit test for frozen-weight validation RNG isolation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("dynamic_weight_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SamplingModel:
    def sample(self, _input_smilies):
        import numpy as np
        import torch

        # Deliberately consumes every global RNG stream that validation must restore.
        token = f"{random.random():.8f}_{np.random.random():.8f}_{torch.rand(1).item():.8f}"
        return SimpleNamespace(
            smilies=[f"C{token}", f"N{token}"],
            states=["VALID", "VALID"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="03_dynamic_weight_controller.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    module = load_controller(Path(args.controller))
    names = ["Target A", "Target B"]
    initial_weights = {"Target A": 0.4, "Target B": 0.6}
    seed_everything(1701)
    before_controller = module.rng_state_hashes(module.capture_rng_state())
    controller = module.FrozenWeightValidationController(
        component_names=names,
        validation_fn=lambda smiles: {
            "Target A": float(len(smiles)) / 10.0 + random.random() * 0.0,
            "Target B": 0.7,
        },
        init_weights=initial_weights,
        validation_batch_size=2,
        validation_max_attempts=1,
        seed=99,
        history_path="",
    )
    after_controller = module.rng_state_hashes(module.capture_rng_state())
    assert before_controller == after_controller, "Frozen controller initialization changed training RNG"

    learning = SimpleNamespace(sampling_model=SamplingModel(), input_smilies=["C"])
    before_validation = module.rng_state_hashes(module.capture_rng_state())
    smiles, scores, rng_audit = controller.sample_and_observe_validation(learning)
    after_validation = module.rng_state_hashes(module.capture_rng_state())
    assert rng_audit["rng_state_unchanged"]
    assert before_validation == after_validation

    controller.begin_training_step()
    controller.step(
        training_component_scores={"Target A": 0.1, "Target B": 0.2},
        validation_scores=scores,
        num_validation_smiles=len(smiles),
        rng_audit=rng_audit,
    )
    record = controller.history()[0]
    assert controller.get_weights() == initial_weights
    assert not record["policy_updated"]
    assert not record["action_applied"]
    assert all(value == 0 for value in record["actions"].values())
    assert record["rng_state_hash_before_validation"] == record["rng_state_hash_after_validation"]

    Path(args.output).write_text(
        json.dumps(
            {
                "passed": True,
                "validation_smiles": smiles,
                "validation_scores": scores,
                "rng_state_hashes": before_validation,
                "controller_record": record,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
