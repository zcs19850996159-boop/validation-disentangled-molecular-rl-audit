#!/usr/bin/env python
"""Repeat-level bootstrap CIs and power analysis for controller effects."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


T_CRIT_TWO_SIDED_005 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

T_CRIT_ONE_SIDED_005 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    21: 1.721,
    22: 1.717,
    23: 1.714,
    24: 1.711,
    25: 1.708,
    26: 1.706,
    27: 1.703,
    28: 1.701,
    29: 1.699,
    30: 1.697,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def paired_bootstrap_ci(deltas: list[float], iterations: int, seed: int) -> dict:
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _idx in range(iterations):
        sample = [deltas[rng.randrange(n)] for _j in range(n)]
        means.append(statistics.mean(sample))
    return {
        "iterations": iterations,
        "mean": statistics.mean(deltas),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "bootstrap_sd": statistics.stdev(means) if len(means) > 1 else 0.0,
    }


def sign_test_pvalues(deltas: list[float]) -> dict:
    nonzero = [value for value in deltas if value != 0.0]
    n = len(nonzero)
    positives = sum(1 for value in nonzero if value > 0.0)
    if n == 0:
        return {"n_nonzero": 0, "positive_count": 0, "one_sided_positive": None, "two_sided": None}

    def binom_p(k: int) -> float:
        return math.comb(n, k) * (0.5 ** n)

    one_sided = sum(binom_p(k) for k in range(positives, n + 1))
    tail = min(
        sum(binom_p(k) for k in range(0, positives + 1)),
        sum(binom_p(k) for k in range(positives, n + 1)),
    )
    return {
        "n_nonzero": n,
        "positive_count": positives,
        "one_sided_positive": one_sided,
        "two_sided": min(1.0, 2.0 * tail),
    }


def t_power(effect: float, sigma: float, n: int, iterations: int, seed: int, one_sided: bool) -> float:
    if sigma <= 0.0 or n < 2:
        return 0.0
    rng = random.Random(seed)
    df = n - 1
    table = T_CRIT_ONE_SIDED_005 if one_sided else T_CRIT_TWO_SIDED_005
    critical = table.get(df, 1.645 if one_sided else 1.960)
    hits = 0
    for _idx in range(iterations):
        sample = [rng.gauss(effect, sigma) for _j in range(n)]
        sample_sd = statistics.stdev(sample)
        if sample_sd == 0.0:
            continue
        t_stat = statistics.mean(sample) / (sample_sd / math.sqrt(n))
        if one_sided:
            if t_stat > critical:
                hits += 1
        elif abs(t_stat) > critical:
            hits += 1
    return hits / iterations


def summarize_setting(
    name: str,
    metric: str,
    deltas: list[float],
    static_repeat_gap: float | None,
    bootstrap_iterations: int,
    power_iterations: int,
    seed: int,
) -> dict:
    observed_mean = statistics.mean(deltas)
    observed_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    effects = [
        ("observed_abs_mean", abs(observed_mean)),
        ("small_0.005", 0.005),
        ("practical_0.010", 0.010),
        ("practical_0.015", 0.015),
    ]
    if static_repeat_gap is not None:
        effects.append(("static_repeat_gap_abs", abs(static_repeat_gap)))

    power = []
    for idx, (label, effect) in enumerate(effects):
        power.append(
            {
                "effect_label": label,
                "effect": effect,
                "n": len(deltas),
                "sigma": observed_sd,
                "two_sided_alpha_0.05": t_power(
                    effect, observed_sd, len(deltas), power_iterations, seed + idx * 17, one_sided=False
                ),
                "one_sided_positive_alpha_0.05": t_power(
                    effect, observed_sd, len(deltas), power_iterations, seed + idx * 17 + 1, one_sided=True
                ),
            }
        )

    return {
        "metric": metric,
        "n_paired_repeats": len(deltas),
        "deltas": deltas,
        "mean_delta": observed_mean,
        "median_delta": statistics.median(deltas),
        "sd_delta": observed_sd,
        "static_repeat_gap": static_repeat_gap,
        "sign_test": sign_test_pvalues(deltas),
        "paired_bootstrap_mean_ci95": paired_bootstrap_ci(deltas, bootstrap_iterations, seed),
        "power_analysis": {
            "method": (
                "Monte Carlo one-sample paired t-test under normal paired deltas, "
                "using observed repeat-level SD as sigma."
            ),
            "iterations": power_iterations,
            "rows": power,
        },
    }


def drd2_qed_mw_deltas(path: Path) -> list[float]:
    payload = load_json(path)
    return [float(row["delta_last50"]) for row in payload["rows"]]


def activity_sa_deltas(path: Path) -> list[float]:
    payload = load_json(path)
    seeds = [17, 23, 101, 202, 303]
    deltas = []
    for seed in seeds:
        real = payload[f"deterministic_sa_fast_seed{seed}"]["last50_mean_validation_joint"]
        random_control = payload[f"random_deterministic_sa_fast_seed{seed}"]["last50_mean_validation_joint"]
        deltas.append(float(real) - float(random_control))
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drd2-qed-mw", default="deterministic_multigpu_pair_summary.json")
    parser.add_argument("--activity-sa", default="qsar_models_DRD2_drd2_activity_sa_fastgate_5repeat_posthoc.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=100000)
    parser.add_argument("--power-iterations", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", default="controller_effect_statistics.json")
    args = parser.parse_args()

    drd2_deltas = drd2_qed_mw_deltas(Path(args.drd2_qed_mw))
    activity_sa_payload = load_json(Path(args.activity_sa))
    activity_static_gap = (
        activity_sa_payload["static_equal_fast_rep2"]["last50_mean_validation_joint"]
        - activity_sa_payload["static_equal_fast"]["last50_mean_validation_joint"]
    )

    result = {
        "method_note": (
            "Bootstrap is performed over paired repeats, not over generated "
            "molecules, to avoid pseudoreplication within runs."
        ),
        "settings": {
            "drd2_qed_mw": summarize_setting(
                "drd2_qed_mw",
                "last50_mean_validation_activity",
                drd2_deltas,
                static_repeat_gap=0.008755583832278535,
                bootstrap_iterations=args.bootstrap_iterations,
                power_iterations=args.power_iterations,
                seed=args.seed,
            ),
            "activity_sa": summarize_setting(
                "activity_sa",
                "last50_mean_validation_joint",
                activity_sa_deltas(Path(args.activity_sa)),
                static_repeat_gap=activity_static_gap,
                bootstrap_iterations=args.bootstrap_iterations,
                power_iterations=args.power_iterations,
                seed=args.seed + 1000,
            ),
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
