#!/usr/bin/env python
"""Apply the frozen V5 end-to-end qualification rule."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SEEDS = (5201, 5202, 5203, 5204, 5205)


def summarize_csv(path: Path) -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    RDLogger.DisableLog("rdApp.warning"); RDLogger.DisableLog("rdApp.error")
    rows = list(csv.DictReader(path.open(newline="")))
    tail = [row for row in rows if 251 <= int(row["step"]) <= 300]
    valid = []
    for row in tail:
        smi = (row.get("SMILES") or "").strip(); mol = Chem.MolFromSmiles(smi) if smi else None
        if row.get("SMILES_state") == "1" and mol is not None:
            valid.append((Chem.MolToSmiles(mol, canonical=True), float(QED.qed(mol))))
    return {
        "tail_rows": len(tail), "valid_rows": len(valid),
        "untouched_qed": statistics.mean(x[1] for x in valid),
        "validity": len(valid) / len(tail),
        "uniqueness": len({x[0] for x in valid}) / len(valid),
    }


def summarize_history(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    return {
        "complete": [r["validation_index"] for r in rows] == list(range(1, 16)),
        "rng_ok": all(r.get("validation_rng_isolated") is True and r.get("rng_state_unchanged") is True and r.get("rng_state_hash_before_validation") == r.get("rng_state_hash_after_validation") for r in rows),
        "qed_actions_by_10": sum(r["actions"]["QED"] > 0 for r in rows if r["validation_index"] <= 10),
        "anti_qed_actions": sum(r["actions"]["Anti-QED"] != 0 for r in rows),
        "final_qed_weight": rows[-1]["new_weights"]["QED"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
    paired = {}; gains = []; validity = []; uniqueness = []; integrity = True; mechanism = True
    for seed in SEEDS:
        arms = {}
        for mode in ("frozen", "real"):
            stem = f"{mode}_s10_seed{seed}"
            arms[mode] = {"molecule": summarize_csv(a.root / f"{stem}_1.csv"), "history": summarize_history(a.root / f"{stem}.history.jsonl")}
        fm, rm = arms["frozen"]["molecule"], arms["real"]["molecule"]
        fh, rh = arms["frozen"]["history"], arms["real"]["history"]
        gain = rm["untouched_qed"] - fm["untouched_qed"]
        gains.append(gain); validity.append(rm["validity"] - fm["validity"]); uniqueness.append(rm["uniqueness"] - fm["uniqueness"])
        integrity = integrity and all(x["complete"] and x["rng_ok"] for x in (fh, rh)) and rh["anti_qed_actions"] == 0
        mechanism = mechanism and rh["qed_actions_by_10"] >= 6 and rh["final_qed_weight"] >= 0.80
        paired[str(seed)] = {"arms": arms, "real_minus_frozen_tail_qed": gain}
    checks = {
        "all_runs_complete_and_rng_isolated": integrity,
        "all_real_seeds_meet_action_and_weight_gate": mechanism,
        "all_five_qed_gains_positive": all(x > 0 for x in gains),
        "mean_qed_gain_at_least_0_050": statistics.mean(gains) >= 0.050,
        "mean_validity_noninferior": statistics.mean(validity) >= -0.02,
        "mean_uniqueness_noninferior": statistics.mean(uniqueness) >= -0.02,
    }
    result = {"protocol": "28_v5_positive_control_qualification_protocol.md", "seeds": list(SEEDS), "paired": paired, "summary": {"qed_gains": gains, "mean_qed_gain": statistics.mean(gains), "mean_validity_difference": statistics.mean(validity), "mean_uniqueness_difference": statistics.mean(uniqueness)}, "checks": checks, "qualification_passed": all(checks.values()), "main_confirmatory_authorized": all(checks.values())}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
