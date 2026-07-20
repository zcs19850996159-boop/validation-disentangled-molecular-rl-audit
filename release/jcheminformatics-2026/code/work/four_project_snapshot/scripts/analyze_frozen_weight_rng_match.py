#!/usr/bin/env python
"""Verify that frozen-weight validation is transparent to training RNG and updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_trace(records: list[dict]) -> tuple[list[dict], list[dict]]:
    return (
        [record for record in records if record.get("trace_kind") == "training_score"],
        [record for record in records if record.get("trace_kind") == "agent_update"],
    )


def exact_match(left: list[dict], right: list[dict], fields: list[str]) -> dict:
    equal = len(left) == len(right)
    mismatches = []
    for index, (left_record, right_record) in enumerate(zip(left, right), start=1):
        for field in fields:
            if left_record.get(field) != right_record.get(field):
                equal = False
                mismatches.append({"training_step": index, "field": field})
    return {"passed": equal, "mismatches": mismatches}


def checkpoint_network_comparison(ordinary_path: Path, frozen_path: Path) -> dict:
    import torch

    ordinary_payload = torch.load(ordinary_path, map_location="cpu", weights_only=False)
    frozen_payload = torch.load(frozen_path, map_location="cpu", weights_only=False)
    ordinary_network = ordinary_payload["network"]
    frozen_network = frozen_payload["network"]
    ordinary_digest = hashlib.sha256()
    frozen_digest = hashlib.sha256()
    mismatches = []
    for name in sorted(set(ordinary_network) | set(frozen_network)):
        if name not in ordinary_network or name not in frozen_network:
            mismatches.append({"parameter": name, "reason": "missing"})
            continue
        ordinary = ordinary_network[name].detach().cpu().contiguous()
        frozen = frozen_network[name].detach().cpu().contiguous()
        for digest, value in ((ordinary_digest, ordinary), (frozen_digest, frozen)):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(repr(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        if not torch.equal(ordinary, frozen):
            mismatches.append({"parameter": name, "reason": "tensor_values_differ"})
    return {
        "passed": not mismatches,
        "ordinary_network_hash": ordinary_digest.hexdigest(),
        "frozen_network_hash": frozen_digest.hexdigest(),
        "raw_checkpoint_sha256": {
            "ordinary": hashlib.sha256(ordinary_path.read_bytes()).hexdigest(),
            "frozen": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        },
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-trace", required=True)
    parser.add_argument("--frozen-trace", required=True)
    parser.add_argument("--frozen-history", required=True)
    parser.add_argument("--ordinary-checkpoint", required=True)
    parser.add_argument("--frozen-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ordinary_scores, ordinary_updates = split_trace(read_jsonl(Path(args.ordinary_trace)))
    frozen_scores, frozen_updates = split_trace(read_jsonl(Path(args.frozen_trace)))
    frozen_history = read_jsonl(Path(args.frozen_history))

    score_check = exact_match(
        ordinary_scores,
        frozen_scores,
        [
            "training_step",
            "smilies",
            "smiles_states",
            "total_scores",
            "training_component_scores",
            "rng_state_hash_after_training_score",
        ],
    )
    update_check = exact_match(
        ordinary_updates,
        frozen_updates,
        [
            "training_step",
            "post_update_agent_state_hash",
            "rng_state_hash_after_agent_update",
        ],
    )
    validation_check = {
        "n_validation_events": len(frozen_history),
        "all_rng_hashes_unchanged": all(
            record.get("validation_rng_isolated")
            and record.get("rng_state_unchanged")
            and record.get("rng_state_hash_before_validation")
            == record.get("rng_state_hash_after_validation")
            for record in frozen_history
        ),
        "all_weights_fixed": all(
            record.get("old_weights") == record.get("new_weights")
            and not record.get("policy_updated")
            and not record.get("action_applied")
            and all(value == 0 for value in record.get("actions", {}).values())
            for record in frozen_history
        ),
    }
    checkpoint_check = checkpoint_network_comparison(
        Path(args.ordinary_checkpoint),
        Path(args.frozen_checkpoint),
    )
    passed = (
        score_check["passed"]
        and update_check["passed"]
        and validation_check["all_rng_hashes_unchanged"]
        and validation_check["all_weights_fixed"]
        and checkpoint_check["passed"]
    )
    result = {
        "passed": passed,
        "training_score_trace": score_check,
        "agent_update_trace": update_check,
        "frozen_validation": validation_check,
        "checkpoint_network": checkpoint_check,
        "final_agent_state_hash": {
            "ordinary": ordinary_updates[-1].get("post_update_agent_state_hash")
            if ordinary_updates
            else None,
            "frozen": frozen_updates[-1].get("post_update_agent_state_hash")
            if frozen_updates
            else None,
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("Frozen-weight RNG match test failed; inspect the JSON output")


if __name__ == "__main__":
    main()
