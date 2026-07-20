#!/usr/bin/env python
"""Temporal-structure-preserving placebo validator for controller replay.

The controller invokes this command once per validation checkpoint.  Unlike the
legacy ``shuffle_history`` control, this script advances through a precomputed
history trajectory using a small, atomically updated state file.  Thus the
controller receives a score vector whose adjacent changes and between-target
dependence come from a real trajectory, but which is unrelated to its current
validation molecules.

Modes
-----
``circular_shift``
    Replay one history after a non-zero circular shift.  This is the primary
    placebo: it preserves every within-trajectory transition except the single
    wrap-around boundary.
``cross_seed_phase_matched``
    Replay a donor seed at the same validation index.  The command fails rather
    than wrapping when the donor history is exhausted.
``block_permutation``
    Deterministically permute contiguous time blocks.  This preserves all
    within-block transitions and is intended as a supplementary placebo.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterator

from temporal_placebo_common import automatic_shift_schedule, eligible_circular_shifts


STATE_VERSION = 1


def parse_component_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise RuntimeError("No component names were provided")
    return names


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_history_scores(path: Path, component_names: list[str]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        scores = row.get("validation_scores", {})
        if not all(name in scores for name in component_names):
            raise RuntimeError(
                f"Missing requested validation score on line {line_number} of {path}"
            )
        records.append({name: float(scores[name]) for name in component_names})
    if not records:
        raise RuntimeError(f"No usable validation_scores found in {path}")
    return records


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_indices(args, history_length: int, history_digest: str) -> tuple[list[int], dict]:
    if args.mode == "circular_shift":
        eligible_shifts = eligible_circular_shifts(history_length)
        if args.shift == "auto":
            schedule = automatic_shift_schedule(
                args.schedule_seed,
                history_digest,
                history_length,
            )
            if not 0 <= args.shift_replicate_index < len(schedule):
                raise RuntimeError(
                    "--shift-replicate-index must select one of the "
                    f"{len(schedule)} frozen middle-half shifts"
                )
            shift = schedule[args.shift_replicate_index]
            shift_metadata = {
                "shift_rule": "seeded_permutation",
                "shift_replicate_index": args.shift_replicate_index,
            }
        else:
            if args.shift_replicate_index:
                raise RuntimeError("--shift-replicate-index is only valid with --shift auto")
            shift = int(args.shift)
            shift_metadata = {"shift_rule": "explicit"}
        if shift not in eligible_shifts:
            raise RuntimeError(
                "--shift must satisfy the frozen middle-half rule "
                f"[{eligible_shifts[0]}, {eligible_shifts[-1]}] for T={history_length}; got {shift}"
            )
        return (
            [(index + shift) % history_length for index in range(history_length)],
            {
                "shift": shift,
                "eligible_shift_min": eligible_shifts[0],
                "eligible_shift_max": eligible_shifts[-1],
                "eligible_shift_count": len(eligible_shifts),
                **shift_metadata,
            },
        )

    if args.mode == "cross_seed_phase_matched":
        return list(range(history_length)), {"phase_rule": "same_validation_index"}

    if args.mode == "block_permutation":
        if args.block_size < 1:
            raise RuntimeError("--block-size must be positive")
        blocks = [
            list(range(start, min(start + args.block_size, history_length)))
            for start in range(0, history_length, args.block_size)
        ]
        if len(blocks) < 2:
            raise RuntimeError("block_permutation needs at least two time blocks")
        order = list(range(len(blocks)))
        random.Random(args.block_seed).shuffle(order)
        if order == list(range(len(blocks))):
            order = order[1:] + order[:1]
        indices = [source_index for block_index in order for source_index in blocks[block_index]]
        return indices, {"block_size": args.block_size, "block_order": order}

    raise RuntimeError(f"Unsupported mode: {args.mode}")


def verify_frozen_manifest(
    args,
    history_path: Path,
    history_digest: str,
    mode_metadata: dict,
) -> dict:
    """Verify that an invocation follows a pre-frozen source and schedule plan."""

    if not args.frozen_manifest and not args.target_seed:
        return {}
    if not args.frozen_manifest or not args.target_seed:
        raise RuntimeError("--frozen-manifest and --target-seed must be supplied together")

    manifest_path = Path(args.frozen_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    target_seed = str(args.target_seed)
    verification = {
        "frozen_manifest_path": str(manifest_path),
        "frozen_manifest_sha256": file_digest(manifest_path),
        "target_seed": target_seed,
    }

    if args.mode == "circular_shift":
        circular = manifest.get("circular_shift", {})
        if int(circular.get("schedule_seed", -1)) != args.schedule_seed:
            raise RuntimeError("Circular schedule seed does not match the frozen manifest")
        for mapping in circular.get("shift_mapping", []):
            if str(mapping.get("seed")) != target_seed:
                continue
            if mapping.get("target_history_sha256") != history_digest:
                raise RuntimeError("Circular source history checksum does not match the frozen manifest")
            expected = mapping.get("selected_shifts", {}).get(
                str(args.shift_replicate_index)
            )
            if expected is None or int(expected) != int(mode_metadata["shift"]):
                raise RuntimeError("Circular shift does not match the frozen manifest")
            return verification
        raise RuntimeError(f"Target seed {target_seed} is absent from the frozen circular plan")

    if args.mode == "cross_seed_phase_matched":
        pairs = manifest.get("cross_seed_phase_matched", {}).get("pairs", [])
        for pair in pairs:
            if str(pair.get("target_seed")) != target_seed:
                continue
            if pair.get("donor_history_sha256") != history_digest:
                raise RuntimeError(
                    "Cross-seed donor history checksum does not match the frozen manifest"
                )
            verification["donor_seed"] = str(pair.get("donor_seed"))
            return verification
        raise RuntimeError(f"Target seed {target_seed} is absent from the frozen donor mapping")

    raise RuntimeError("The frozen manifest supports circular_shift and cross_seed_phase_matched only")


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


@contextlib.contextmanager
def state_lock(path: Path) -> Iterator[None]:
    """Serialize state advancement if an external runner retries concurrently."""

    import fcntl

    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def next_source_index(state_path: Path, metadata: dict, source_indices: list[int]) -> tuple[int, int]:
    with state_lock(state_path):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("metadata") != metadata:
                raise RuntimeError(
                    f"Replay state metadata does not match this invocation: {state_path}. "
                    "Use a fresh --replay-state path for each placebo trajectory."
                )
        else:
            state = {"metadata": metadata, "next_call_index": 0}

        call_index = int(state.get("next_call_index", 0))
        if args_mode := metadata["mode"]:
            if args_mode != "circular_shift" and call_index >= len(source_indices):
                raise RuntimeError(
                    "Replay history exhausted at validation index "
                    f"{call_index + 1}; refusing to wrap a non-circular placebo."
                )
        source_index = source_indices[call_index % len(source_indices)]
        state["next_call_index"] = call_index + 1
        atomic_write_json(state_path, state)
        return call_index, source_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--component-names", required=True)
    parser.add_argument(
        "--mode",
        choices=["circular_shift", "cross_seed_phase_matched", "block_permutation"],
        required=True,
    )
    parser.add_argument("--replay-state", required=True)
    parser.add_argument(
        "--schedule-seed",
        "--seed",
        dest="schedule_seed",
        type=int,
        default=51001,
        help="Frozen seed used only to order eligible circular shifts.",
    )
    parser.add_argument(
        "--shift",
        default="auto",
        help="Middle-half circular shift or 'auto' for the deterministic seeded schedule.",
    )
    parser.add_argument(
        "--shift-replicate-index",
        type=int,
        default=0,
        help="With --shift auto, select a distinct middle-half shift from the fixed schedule.",
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--block-seed", type=int, default=51000)
    parser.add_argument("--frozen-manifest", default=None)
    parser.add_argument("--target-seed", default=None)
    args = parser.parse_args()

    component_names = parse_component_names(args.component_names)
    history_path = Path(args.history).resolve()
    history = read_history_scores(history_path, component_names)
    history_digest = file_digest(history_path)
    source_indices, mode_metadata = build_source_indices(args, len(history), history_digest)
    manifest_metadata = verify_frozen_manifest(
        args,
        history_path,
        history_digest,
        mode_metadata,
    )
    metadata = {
        "state_version": STATE_VERSION,
        "mode": args.mode,
        "component_names": component_names,
        "history_path": str(history_path),
        "history_sha256": history_digest,
        "history_length": len(history),
        "schedule_seed": args.schedule_seed,
        "block_seed": args.block_seed if args.mode == "block_permutation" else None,
        **mode_metadata,
        **manifest_metadata,
    }
    call_index, source_index = next_source_index(
        Path(args.replay_state), metadata, source_indices
    )
    scores = history[source_index]
    rows = read_rows(Path(args.input))

    output_fields = [
        "smiles",
        *component_names,
        "status",
        "control_mode",
        "placebo_call_index",
        "placebo_source_index",
        "placebo_source_history",
        "placebo_shift",
        "placebo_block_size",
    ]
    with Path(args.output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            out = {
                "smiles": (row.get("smiles") or "").strip(),
                "status": "temporal_placebo",
                "control_mode": args.mode,
                "placebo_call_index": call_index,
                "placebo_source_index": source_index,
                "placebo_source_history": str(history_path),
                "placebo_shift": mode_metadata.get("shift", ""),
                "placebo_block_size": mode_metadata.get("block_size", ""),
            }
            out.update(scores)
            writer.writerow(out)


if __name__ == "__main__":
    main()
