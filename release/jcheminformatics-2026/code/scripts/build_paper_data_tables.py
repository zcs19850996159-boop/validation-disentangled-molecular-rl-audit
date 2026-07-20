#!/usr/bin/env python3
"""Build manuscript-ready data tables from frozen experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "work" / "paper_data_snapshot"
OUTPUT = ROOT / "paper_data"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(name: str, rows: list[dict], fieldnames: list[str] | None = None):
    path = OUTPUT / name
    if not rows:
        raise ValueError(f"No rows supplied for {name}")
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_rows() -> list[dict]:
    return [
        {
            "dataset_id": "confirmatory_alpha025",
            "designation": "MAIN",
            "role": "Primary confirmatory evidence",
            "n": "10 paired seeds; 70 verified arms",
            "source": "work/paper_data_snapshot/results/drd2_alpha025_confirmatory_final.json",
            "paper_use": "Primary contrasts, static comparators, quality endpoints",
            "status": "frozen_complete",
        },
        {
            "dataset_id": "confirmatory_seed_differences",
            "designation": "MAIN_AND_SUPPLEMENT",
            "role": "Seed-level analysis units",
            "n": "10 differences per comparator-endpoint",
            "source": "work/paper_data_snapshot/results/drd2_alpha025_confirmatory_final.paired_differences.csv",
            "paper_use": "Paired plots in main text; complete table in supplement",
            "status": "frozen_complete",
        },
        {
            "dataset_id": "temporal_placebo_development_audit",
            "designation": "MAIN",
            "role": "Legacy shuffle versus temporal-preserving placebo qualification",
            "n": "10 historical real trajectories",
            "source": "outputs/temporal_placebo_frozen_audit.json",
            "paper_use": "Demonstrate why iid history shuffle is inadequate",
            "status": "frozen_complete",
        },
        {
            "dataset_id": "confirmatory_frozen_placebo_audit",
            "designation": "MAIN_AND_SUPPLEMENT",
            "role": "Audit exact three shifts and cyclic donor mapping used in confirmatory runs",
            "n": "10 seeds; 3 circular shifts per seed; 10 donor pairs",
            "source": "work/paper_data_snapshot/drd2_alpha025_confirmatory_temporal_audit.json",
            "paper_use": "Compact qualification table in main text; per-seed details in supplement",
            "status": "frozen_complete",
        },
        {
            "dataset_id": "static_grid_development",
            "designation": "MAIN_AND_SUPPLEMENT",
            "role": "Independent development selection of tuned static weights",
            "n": "7 candidates x 3 development seeds",
            "source": "work/paper_data_snapshot/drd2_static_grid_development_v1_selection.json",
            "paper_use": "Selection rule and chosen 0.25/0.50/0.25 weights",
            "status": "frozen_complete",
        },
        {
            "dataset_id": "frozen_weight_rng_match",
            "designation": "MAIN_AND_SUPPLEMENT",
            "role": "Compute/RNG matching qualification",
            "n": "30 steps; 6 validation events",
            "source": "work/paper_data_snapshot/rng_match_test_30steps_cadence5_seed1701_v1/rng_match_summary.json",
            "paper_use": "Qualification statement in main text; hashes in supplement",
            "status": "passed",
        },
        {
            "dataset_id": "concurrent_worker_determinism",
            "designation": "SUPPLEMENT",
            "role": "Execution amendment reproducibility check",
            "n": "5 concurrent GPU worker slots",
            "source": "work/paper_data_snapshot/multigpu_determinism_smoke_v1_summary.json",
            "paper_use": "Disclose concurrent execution and exact parameter equivalence",
            "status": "passed",
        },
        {
            "dataset_id": "bindingdb_mpn_oracle",
            "designation": "MAIN_AND_SUPPLEMENT",
            "role": "Untouched final evaluation oracle qualification",
            "n": "75 train; 25 validation; 25 scaffold-disjoint test",
            "source": "work/paper_data_snapshot/qsar_models/DRD2_bindingdb_mpn/bindingdb_drd2_mpn_report.json",
            "paper_use": "Oracle gate and downstream generalization endpoint",
            "status": "passed_with_low_generated_ad_coverage",
        },
        {
            "dataset_id": "v5_end_to_end_positive_control",
            "designation": "MAIN",
            "role": "End-to-end pipeline qualification",
            "n": "5 paired seeds",
            "source": "work/paper_data_snapshot/results/positive_control_v5_qualification/qualification_result.json",
            "paper_use": "Show the complete pipeline detects a known molecular benefit",
            "status": "passed",
        },
        {
            "dataset_id": "controller_input_sensitivity_ladder",
            "designation": "SUPPLEMENT",
            "role": "Controller-input positive control",
            "n": "Synthetic signal ladder",
            "source": "work/four_project_snapshot",
            "paper_use": "Retain with narrowed label; not end-to-end evidence",
            "status": "legacy_complete",
        },
        {
            "dataset_id": "alpha025_two_seed_pilot",
            "designation": "SUPPLEMENT",
            "role": "Exploratory finding that motivated the sole confirmatory follow-up",
            "n": "2 paired repeats",
            "source": "work/four_project_snapshot",
            "paper_use": "Pilot-to-confirmatory chronology only",
            "status": "legacy_complete_not_confirmatory",
        },
        {
            "dataset_id": "legacy_activity_sa_gsk3b_docking",
            "designation": "SUPPLEMENT_OR_ARCHIVE",
            "role": "Earlier portability and exploratory analyses",
            "n": "Mixed legacy designs",
            "source": "work/four_project_snapshot",
            "paper_use": "Supporting context only; no primary efficacy claim",
            "status": "legacy_complete_design_not_compute_matched",
        },
    ]


def contrast_rows(final: dict) -> list[dict]:
    rows = []
    for comparator, endpoints in final["contrasts"].items():
        for endpoint, stats in endpoints.items():
            rows.append(
                {
                    "comparator": comparator,
                    "endpoint": endpoint,
                    "n": stats["n"],
                    "mean_difference": stats["mean"],
                    "median_difference": stats["median"],
                    "sample_sd": stats["sample_sd"],
                    "ci95_low": stats["bootstrap_95_ci"][0],
                    "ci95_high": stats["bootstrap_95_ci"][1],
                    "positive_count": stats["positive_count"],
                    "negative_count": stats["negative_count"],
                    "zero_count": stats["zero_count"],
                    "sign_test_p": stats["exact_sign_test_two_sided_p"],
                    "practical_threshold": stats["practical_threshold"],
                    "mean_exceeds_threshold": stats["mean_exceeds_practical_threshold"],
                }
            )
    return rows


def arm_summary_rows(final: dict) -> list[dict]:
    """Summarize absolute arm outcomes at the paired-seed level."""
    paired_path = SNAPSHOT / "results" / "drd2_alpha025_confirmatory_final.paired_differences.csv"
    with paired_path.open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))

    endpoints = [
        "svm_activity_mean",
        "mpn_probability_all_valid_mean",
        "mpn_probability_ad_mean",
        "ad_coverage",
        "qed_mean",
        "mw_target_score_mean",
        "validity",
        "uniqueness",
        "scaffold_diversity",
    ]
    comparators = ["circular_mean", "cross_seed", "frozen_equal", "frozen_tuned"]
    rows = []
    for endpoint in endpoints:
        real_values = [
            float(row["real"])
            for row in paired
            if row["comparator"] == "circular_mean" and row["endpoint"] == endpoint
        ]
        output = {
            "endpoint": endpoint,
            "real_mean": statistics.mean(real_values),
            "real_sd": statistics.stdev(real_values),
        }
        for comparator in comparators:
            values = [
                float(row["comparison"])
                for row in paired
                if row["comparator"] == comparator and row["endpoint"] == endpoint
            ]
            output[f"{comparator}_mean"] = statistics.mean(values)
            output[f"{comparator}_sd"] = statistics.stdev(values)
        rows.append(output)
    return rows


def temporal_rows(audit: dict, audit_set: str) -> list[dict]:
    rows = []
    for method, result in audit["methods"].items():
        summary = result["summary"]
        gap = summary["mean_dynamic_gap"]
        source_declines = summary["mean_source_decline_event_count"]
        placebo_declines = summary["mean_placebo_decline_event_count"]
        rows.append(
            {
                "audit_set": audit_set,
                "method": method,
                "n_seeds": summary["n_seeds"],
                "lag1_acf_gap": gap["mean_abs_lag1_autocorrelation_gap"],
                "delta_ks": gap["mean_delta_ks_distance"],
                "decline_count_gap": gap["mean_abs_decline_event_count_gap"],
                "action_checkpoint_gap": gap["abs_controller_action_checkpoint_gap"],
                "component_action_gap": gap["abs_controller_component_action_gap"],
                "final_weight_l1_gap": gap["final_weight_l1_gap"],
                "source_action_checkpoints": summary["mean_source_action_checkpoints"],
                "placebo_action_checkpoints": summary["mean_placebo_action_checkpoints"],
                "source_component_actions": summary["mean_source_component_actions"],
                "placebo_component_actions": summary["mean_placebo_component_actions"],
                "source_final_weight_l1_drift": summary["mean_source_final_weight_l1_drift"],
                "placebo_final_weight_l1_drift": summary["mean_placebo_final_weight_l1_drift"],
                "source_declines_activity": source_declines["DRD2_activity"],
                "placebo_declines_activity": placebo_declines["DRD2_activity"],
                "source_declines_qed": source_declines["QED"],
                "placebo_declines_qed": placebo_declines["QED"],
                "source_declines_mw": source_declines["Molecular weight"],
                "placebo_declines_mw": placebo_declines["Molecular weight"],
            }
        )
    return rows


def cross_seed_rows(audit: dict) -> list[dict]:
    summary = audit["cross_seed_target_donor"]["summary"]
    rows = [
        {
            "component": "overall",
            "n_pairs": summary["n_target_donor_pairs"],
            "checkpoint_mae": summary["mean_checkpoint_mae"],
            "abs_score_mean_difference": summary["mean_abs_component_mean_difference"],
            "lag1_acf_gap": summary["mean_abs_lag1_autocorrelation_gap"],
            "delta_ks": summary["mean_delta_ks_distance"],
            "abs_decline_event_difference": summary["mean_abs_decline_event_difference"],
            "phase_mean_abs_gap": summary["mean_phase_mean_abs_gap"],
            "abs_slope_difference": summary["mean_abs_slope_difference"],
        }
    ]
    for component, values in summary["per_component"].items():
        rows.append(
            {
                "component": component,
                "n_pairs": summary["n_target_donor_pairs"],
                "checkpoint_mae": values["mean_checkpoint_mae"],
                "abs_score_mean_difference": values["mean_abs_score_mean_difference"],
                "lag1_acf_gap": values["mean_abs_lag1_autocorrelation_gap"],
                "delta_ks": values["mean_delta_ks_distance"],
                "abs_decline_event_difference": values["mean_abs_decline_event_difference"],
                "phase_mean_abs_gap": values["mean_phase_mean_abs_gap"],
                "abs_slope_difference": values["mean_abs_slope_difference"],
            }
        )
    return rows


def static_rows(selection: dict) -> list[dict]:
    rows = []
    for label, values in selection["summary"].items():
        rows.append(
            {
                "candidate": label,
                "n_development_seeds": values["n"],
                "mean_last50_validation_activity": values["mean_last50_validation_activity"],
                "mean_final_step_qed": values["mean_final_step_qed"],
                "mean_final_step_mw_score": values["mean_final_step_mw_score"],
                "quality_eligible": values["quality_eligible"],
                "selected": label == selection["selected_label"],
            }
        )
    return rows


def study_design_rows() -> list[dict]:
    return [
        {
            "arm": "real_validation_controller",
            "runs": 10,
            "per_seed": 1,
            "weights": "dynamic; initial 0.34/0.33/0.33; alpha 0.25",
            "validation_input": "current-seed real SVM trajectory",
            "controller_actions": "enabled",
            "role": "tested adaptive policy",
        },
        {
            "arm": "circular_shift_replay",
            "runs": 30,
            "per_seed": 3,
            "weights": "dynamic; initial 0.34/0.33/0.33; alpha 0.25",
            "validation_input": "three frozen middle-half circular shifts",
            "controller_actions": "enabled",
            "role": "primary temporal-preserving placebo",
        },
        {
            "arm": "cross_seed_phase_matched_replay",
            "runs": 10,
            "per_seed": 1,
            "weights": "dynamic; initial 0.34/0.33/0.33; alpha 0.25",
            "validation_input": "cyclic next-seed donor at same checkpoint",
            "controller_actions": "enabled",
            "role": "mechanism-distinct robustness placebo",
        },
        {
            "arm": "frozen_equal",
            "runs": 10,
            "per_seed": 1,
            "weights": "fixed 0.34/0.33/0.33",
            "validation_input": "full compute/RNG-matched SVM validation",
            "controller_actions": "disabled; no renormalization",
            "role": "equal static comparator",
        },
        {
            "arm": "frozen_tuned",
            "runs": 10,
            "per_seed": 1,
            "weights": "fixed 0.25/0.50/0.25",
            "validation_input": "full compute/RNG-matched SVM validation",
            "controller_actions": "disabled; no renormalization",
            "role": "development-selected static comparator",
        },
    ]


def circular_mapping_rows(audit: dict) -> list[dict]:
    rows = []
    for item in audit["frozen_circular"]["shift_mapping"]:
        rows.append(
            {
                "seed": item["seed"],
                "replicate_0_shift": item["selected_shifts"]["0"],
                "replicate_1_shift": item["selected_shifts"]["1"],
                "replicate_2_shift": item["selected_shifts"]["2"],
                "eligible_shift_min": audit["frozen_circular"]["eligible_shift_min"],
                "eligible_shift_max": audit["frozen_circular"]["eligible_shift_max"],
                "target_history_sha256": item["target_history_sha256"],
            }
        )
    return rows


def donor_mapping_rows(audit: dict) -> list[dict]:
    rows = []
    for pair in audit["cross_seed_target_donor"]["pairs"]:
        rows.append(
            {
                "target_seed": pair["target_seed"],
                "donor_seed": pair["donor_seed"],
                "target_history_sha256": pair["target_history_sha256"],
                "donor_history_sha256": pair["donor_history_sha256"],
            }
        )
    return rows


def v5_rows(v5: dict) -> list[dict]:
    rows = []
    for seed, result in v5["paired"].items():
        frozen = result["arms"]["frozen"]
        real = result["arms"]["real"]
        rows.append(
            {
                "seed": seed,
                "frozen_tail_qed": frozen["molecule"]["untouched_qed"],
                "real_tail_qed": real["molecule"]["untouched_qed"],
                "real_minus_frozen_qed": result["real_minus_frozen_tail_qed"],
                "frozen_final_qed_weight": frozen["history"]["final_qed_weight"],
                "real_final_qed_weight": real["history"]["final_qed_weight"],
                "real_qed_actions_by_10": real["history"]["qed_actions_by_10"],
                "frozen_validity": frozen["molecule"]["validity"],
                "real_validity": real["molecule"]["validity"],
                "frozen_uniqueness": frozen["molecule"]["uniqueness"],
                "real_uniqueness": real["molecule"]["uniqueness"],
            }
        )
    return rows


def qualification_rows(final: dict, rng: dict, workers: dict, oracle: dict, v5: dict) -> list[dict]:
    mean_ad_coverage = sum(
        arm["ad_coverage"]
        for name, arm in final["arm_summaries"].items()
        if name.startswith("real_seed")
    ) / final["analysis"]["n_seeds"]
    return [
        {"qualification": "frozen_weight_rng_match", "metric": "overall_pass", "value": rng["passed"], "gate": "true"},
        {"qualification": "frozen_weight_rng_match", "metric": "validation_events", "value": rng["frozen_validation"]["n_validation_events"], "gate": "6"},
        {"qualification": "frozen_weight_rng_match", "metric": "rng_unchanged", "value": rng["frozen_validation"]["all_rng_hashes_unchanged"], "gate": "true"},
        {"qualification": "frozen_weight_rng_match", "metric": "network_hash_equal", "value": rng["checkpoint_network"]["passed"], "gate": "true"},
        {"qualification": "concurrent_worker_determinism", "metric": "overall_pass", "value": workers["passed"], "gate": "true"},
        {"qualification": "bindingdb_mpn", "metric": "test_auroc", "value": oracle["metrics"]["auroc"], "gate": ">=0.70"},
        {"qualification": "bindingdb_mpn", "metric": "test_average_precision", "value": oracle["metrics"]["average_precision"], "gate": ">=prevalence+0.10"},
        {"qualification": "bindingdb_mpn", "metric": "test_brier", "value": oracle["metrics"]["brier"], "gate": "<=0.25"},
        {"qualification": "bindingdb_mpn", "metric": "test_ece", "value": oracle["metrics"]["ece"], "gate": "reported_not_gated"},
        {"qualification": "bindingdb_mpn", "metric": "mean_real_generated_ad_coverage", "value": mean_ad_coverage, "gate": "reported_not_gated"},
        {"qualification": "v5_end_to_end", "metric": "overall_pass", "value": v5["qualification_passed"], "gate": "true"},
        {"qualification": "v5_end_to_end", "metric": "mean_qed_gain", "value": v5["summary"]["mean_qed_gain"], "gate": ">=0.050"},
        {"qualification": "v5_end_to_end", "metric": "positive_qed_gains", "value": sum(x > 0 for x in v5["summary"]["qed_gains"]), "gate": "5/5"},
        {"qualification": "v5_end_to_end", "metric": "mean_validity_difference", "value": v5["summary"]["mean_validity_difference"], "gate": ">=-0.02"},
        {"qualification": "v5_end_to_end", "metric": "mean_uniqueness_difference", "value": v5["summary"]["mean_uniqueness_difference"], "gate": ">=-0.02"},
    ]


def write_manifest(paths: list[Path]):
    rows = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"sha256": digest, "bytes": path.stat().st_size, "path": str(path.relative_to(ROOT))})
    write_csv("paper_data_manifest.csv", rows)


def main():
    OUTPUT.mkdir(exist_ok=True)
    final = load_json(SNAPSHOT / "results" / "drd2_alpha025_confirmatory_final.json")
    development_temporal = load_json(ROOT / "outputs" / "temporal_placebo_frozen_audit.json")
    confirmatory_temporal = load_json(SNAPSHOT / "drd2_alpha025_confirmatory_temporal_audit.json")
    static_selection = load_json(SNAPSHOT / "drd2_static_grid_development_v1_selection.json")
    rng = load_json(SNAPSHOT / "rng_match_test_30steps_cadence5_seed1701_v1" / "rng_match_summary.json")
    workers = load_json(SNAPSHOT / "multigpu_determinism_smoke_v1_summary.json")
    oracle = load_json(SNAPSHOT / "qsar_models" / "DRD2_bindingdb_mpn" / "bindingdb_drd2_mpn_report.json")
    v5 = load_json(SNAPSHOT / "results" / "positive_control_v5_qualification" / "qualification_result.json")

    write_csv("data_inventory.csv", inventory_rows())
    all_contrasts = contrast_rows(final)
    write_csv("table_confirmatory_all_contrasts.csv", all_contrasts)
    write_csv("table_confirmatory_arm_summaries.csv", arm_summary_rows(final))
    main_endpoints = {
        "svm_activity_mean",
        "mpn_probability_all_valid_mean",
        "mpn_probability_ad_mean",
        "qed_mean",
        "validity",
        "uniqueness",
        "scaffold_diversity",
    }
    write_csv("table_confirmatory_main_endpoints.csv", [row for row in all_contrasts if row["endpoint"] in main_endpoints])
    temporal = temporal_rows(development_temporal, "development_frozen")
    temporal.extend(temporal_rows(confirmatory_temporal, "confirmatory_frozen"))
    write_csv("table_temporal_placebo_qualification.csv", temporal)
    write_csv("table_cross_seed_matching.csv", cross_seed_rows(confirmatory_temporal))
    write_csv("table_study_design.csv", study_design_rows())
    write_csv("supplement_circular_shift_mapping.csv", circular_mapping_rows(confirmatory_temporal))
    write_csv("supplement_cross_seed_donor_mapping.csv", donor_mapping_rows(confirmatory_temporal))
    write_csv("table_static_grid_development.csv", static_rows(static_selection))
    write_csv("table_qualification_checks.csv", qualification_rows(final, rng, workers, oracle, v5))
    write_csv("figure_v5_paired_qed.csv", v5_rows(v5))

    copied_seed_rows = []
    paired_path = SNAPSHOT / "results" / "drd2_alpha025_confirmatory_final.paired_differences.csv"
    with paired_path.open(encoding="utf-8", newline="") as handle:
        copied_seed_rows.extend(csv.DictReader(handle))
    write_csv("figure_confirmatory_seed_differences.csv", copied_seed_rows)

    source_paths = [
        SNAPSHOT / "results" / "drd2_alpha025_confirmatory_final.json",
        paired_path,
        ROOT / "outputs" / "temporal_placebo_frozen_audit.json",
        SNAPSHOT / "drd2_alpha025_confirmatory_temporal_audit.json",
        SNAPSHOT / "drd2_static_grid_development_v1_selection.json",
        SNAPSHOT / "rng_match_test_30steps_cadence5_seed1701_v1" / "rng_match_summary.json",
        SNAPSHOT / "multigpu_determinism_smoke_v1_summary.json",
        SNAPSHOT / "qsar_models" / "DRD2_bindingdb_mpn" / "bindingdb_drd2_mpn_report.json",
        SNAPSHOT / "results" / "positive_control_v5_qualification" / "qualification_result.json",
    ]
    write_manifest(source_paths)


if __name__ == "__main__":
    main()
