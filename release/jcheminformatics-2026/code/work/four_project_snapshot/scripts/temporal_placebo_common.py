"""Shared deterministic schedule rules for temporal placebo controls."""

from __future__ import annotations

import hashlib
import math
import random


def eligible_circular_shifts(history_length: int) -> list[int]:
    """Return only middle-of-trajectory circular shifts.

    Near-zero offsets can preserve the same local training phase in a smooth
    trajectory.  The frozen schedule therefore admits only
    ceil(T / 4) <= shift <= floor(3T / 4).
    """

    if history_length < 2:
        raise RuntimeError("circular_shift requires at least two validation records")
    lower = max(1, math.ceil(history_length / 4))
    upper = min(history_length - 1, math.floor(3 * history_length / 4))
    shifts = list(range(lower, upper + 1))
    if not shifts:
        raise RuntimeError(
            "No eligible circular shifts remain after applying the middle-half rule"
        )
    return shifts


def automatic_shift_schedule(
    schedule_seed: int,
    history_digest: str,
    history_length: int,
) -> list[int]:
    """Return a reproducible permutation of the frozen eligible shift set."""

    digest = hashlib.sha256(f"{schedule_seed}:{history_digest}".encode()).digest()
    shifts = eligible_circular_shifts(history_length)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(shifts)
    return shifts
