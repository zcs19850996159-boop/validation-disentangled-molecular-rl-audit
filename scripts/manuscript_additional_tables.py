#!/usr/bin/env python
"""Build manuscript tables for static/random/real and molecule quality checks.

The script is intentionally post-hoc: it reads logged REINVENT CSV files and
existing validation summaries. It does not retrain a generator or scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


DRD2_SEEDS = [17, 23, 101, 202, 303, 404, 505, 606, 707, 808]
ACTIVITY_SA_SEEDS = [17, 23, 101, 202, 303]


def drd2_qed_mw_runs() -> dict[str, list[tuple[str, str]]]:
    real = [("deterministic_db002_seed17", "drd2_qsar_deterministic_db002_1.csv")]
    random_runs = [("random_deterministic_db002_seed17", "drd2_qsar_random_deterministic_db002_1.csv")]
    for seed in DRD2_SEEDS[1:]:
        real.append((f"deterministic_mg_seed{seed}", f"drd2_qsar_deterministic_mg_seed{seed}_1.csv"))
        random_runs.append(
            (f"random_deterministic_mg_seed{seed}", f"drd2_qsar_random_deterministic_mg_seed{seed}_1.csv")
        )
    return {
        "static": [
            ("static_equal_seed17", "drd2_qsar_static_equal_1.csv"),
            ("static_equal_rep2", "drd2_qsar_static_equal_rep2_1.csv"),
            ("static_equal_rep3", "drd2_qsar_static_equal_rep3_1.csv"),
            ("static_equal_rep4", "drd2_qsar_static_equal_rep4_1.csv"),
            ("static_equal_rep5", "drd2_qsar_static_equal_rep5_1.csv"),
        ],
        "random": random_runs,
        "real": real,
    }


def activity_sa_runs() -> dict[str, list[tuple[str, str]]]:
    return {
        "static": [
            ("static_equal_fast", "drd2_activity_sa_fast_static_equal_1.csv"),
            ("static_equal_fast_rep2", "drd2_activity_sa_fast_static_equal_rep2_1.csv"),
            ("static_equal_fast_rep3", "drd2_activity_sa_fast_static_equal_rep3_1.csv"),
            ("static_equal_fast_rep4", "drd2_activity_sa_fast_static_equal_rep4_1.csv"),
            ("static_equal_fast_rep5", "drd2_activity_sa_fast_static_equal_rep5_1.csv"),
        ],
        "random": [
            (
                f"random_deterministic_sa_fast_seed{seed}",
                f"drd2_activity_sa_fast_random_deterministic_seed{seed}_1.csv",
            )
            for seed in ACTIVITY_SA_SEEDS
        ],
        "real": [
            (f"deterministic_sa_fast_seed{seed}", f"drd2_activity_sa_fast_deterministic_seed{seed}_1.csv")
            for seed in ACTIVITY_SA_SEEDS
        ],
    }


SETTINGS = {
    "DRD2/QED/MW": {
        "runs": drd2_qed_mw_runs,
        "posthoc": "manuscript_drd2_qed_mw_posthoc_with_static.json",
        "primary": "last50_mean_validation_activity",
        "top10": "top10_mean_validation_activity",
        "final": "final_step_mean_validation_activity",
        "all": "all_mean_validation_activity",
        "metric_label": "validation activity",
    },
    "DRD2 activity/SA": {
        "runs": activity_sa_runs,
        "posthoc": "qsar_models_DRD2_drd2_activity_sa_fastgate_5repeat_posthoc.json",
        "primary": "last50_mean_validation_joint",
        "top10": "top10_mean_validation_joint",
        "final": "final_step_mean_validation_joint",
        "all": "all_mean_validation_joint",
        "metric_label": "validation joint",
    },
}


def finite(values: list[float | None]) -> list[float]:
    result = []
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            result.append(parsed)
    return result


def safe_mean(values: list[float | None]) -> float | None:
    values = finite(values)
    return mean(values) if values else None


def safe_sd(values: list[float | None]) -> float | None:
    values = finite(values)
    return stdev(values) if len(values) > 1 else None


def f3(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def f4(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def f3_pm(metric: dict | None) -> str:
    if not isinstance(metric, dict):
        return "--"
    value = metric.get("mean")
    if value is None:
        return "--"
    sd = metric.get("sd")
    if sd is None:
        return f"{value:.3f}"
    return f"{value:.3f}$\\pm${sd:.3f}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_static_controller_comparison(root: Path) -> dict:
    result = {}
    for setting, spec in SETTINGS.items():
        posthoc = load_json(root / spec["posthoc"])
        runs = spec["runs"]()
        row = {
            "metric_label": spec["metric_label"],
            "n_runs": {group: len(items) for group, items in runs.items()},
            "groups": {},
        }
        for group, items in runs.items():
            names = [name for name, _path in items]
            summaries = [posthoc[name] for name in names if name in posthoc]
            group_summary = {}
            for label, key in [
                ("last50", spec["primary"]),
                ("top10", spec["top10"]),
                ("final", spec["final"]),
                ("all", spec["all"]),
            ]:
                values = [summary.get(key) for summary in summaries]
                group_summary[label] = {
                    "mean": safe_mean(values),
                    "sd": safe_sd(values),
                    "values": finite(values),
                }
            row["groups"][group] = group_summary
        real = row["groups"]["real"]["last50"]["mean"]
        random_control = row["groups"]["random"]["last50"]["mean"]
        static = row["groups"]["static"]["last50"]["mean"]
        row["deltas"] = {
            "real_minus_random_last50": None if real is None or random_control is None else real - random_control,
            "real_minus_static_last50": None if real is None or static is None else real - static,
            "random_minus_static_last50": None if random_control is None or static is None else random_control - static,
        }
        result[setting] = row
    return result


def canonicalize_reference(paths: list[Path], active_only: bool = False) -> set[str]:
    try:
        from rdkit import Chem, RDLogger
    except Exception:
        return set()

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    refs = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".smi":
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                smiles = line.strip().split()[0] if line.strip() else ""
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is not None:
                    refs.add(Chem.MolToSmiles(mol, canonical=True))
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if active_only:
                    label = (row.get("label") or row.get("Label") or row.get("activity_label") or "").strip()
                    if label not in {"1", "active", "Active", "ACTIVE"}:
                        continue
                smiles = (row.get("smiles") or row.get("SMILES") or "").strip()
                mol = Chem.MolFromSmiles(smiles) if smiles else None
                if mol is not None:
                    refs.add(Chem.MolToSmiles(mol, canonical=True))
    return refs


def import_rdkit():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    return Chem, DataStructs, AllChem, Crippen, Descriptors, Lipinski, QED, FilterCatalog, FilterCatalogParams, MurckoScaffold


def load_sa_scorer():
    try:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from scripts.sa_score_utils import exact_sa_raw_from_mol

        return exact_sa_raw_from_mol
    except Exception:
        return None


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def pairwise_diversity(fps) -> float | None:
    if len(fps) < 2:
        return None
    from rdkit import DataStructs

    sims = []
    for idx in range(1, len(fps)):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx]))
    return 1.0 - mean(sims) if sims else None


def make_pains_catalog(FilterCatalog, FilterCatalogParams):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def make_named_catalog(FilterCatalog, FilterCatalogParams, names: list[str]):
    params = FilterCatalogParams()
    added = False
    for name in names:
        catalog = getattr(FilterCatalogParams.FilterCatalogs, name, None)
        if catalog is None:
            continue
        params.AddCatalog(catalog)
        added = True
    return FilterCatalog(params) if added else None


def fps_from_smiles(smiles_set: set[str], Chem, AllChem):
    fps = []
    for smiles in smiles_set:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return fps


def mean_max_similarity(fps, reference_fps) -> float | None:
    if not fps or not reference_fps:
        return None
    from rdkit import DataStructs

    maxima = []
    for fp in fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, reference_fps)
        if sims:
            maxima.append(max(sims))
    return mean(maxima) if maxima else None


def summarize_run_quality(
    path: Path,
    reference_smiles: set[str],
    active_reference_smiles: set[str],
    sample_size: int,
    seed: int,
) -> dict:
    Chem, _DataStructs, AllChem, Crippen, Descriptors, Lipinski, QED, FilterCatalog, FilterCatalogParams, MurckoScaffold = import_rdkit()
    sa_raw_from_mol = load_sa_scorer()
    pains_catalog = make_pains_catalog(FilterCatalog, FilterCatalogParams)
    brenk_catalog = make_named_catalog(FilterCatalog, FilterCatalogParams, ["BRENK"])
    active_reference_fps = fps_from_smiles(active_reference_smiles, Chem, AllChem)
    rng = random.Random(seed)

    total_rows = 0
    logged_valid_rows = 0
    valid_records = []
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            total_rows += 1
            smiles = (row.get("SMILES") or "").strip()
            if row.get("SMILES_state") == "1":
                logged_valid_rows += 1
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            scaffold = (row.get("Scaffold") or "").strip()
            if not scaffold:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                except Exception:
                    scaffold = ""
            score = to_float(row.get("Score"))
            qed = to_float(row.get("QED (raw)") or row.get("QED"))
            if qed is None:
                qed = float(QED.qed(mol))
            mw = to_float(row.get("Molecular weight (raw)"))
            if mw is None:
                mw = float(Descriptors.MolWt(mol))
            sa_raw = None
            if sa_raw_from_mol is not None:
                try:
                    sa_raw = float(sa_raw_from_mol(mol))
                except Exception:
                    sa_raw = None
            if sa_raw is None:
                sa_raw = to_float(row.get("SA_score (raw)"))
            pains = bool(pains_catalog.HasMatch(mol))
            brenk = bool(brenk_catalog.HasMatch(mol)) if brenk_catalog is not None else False
            valid_records.append(
                {
                    "canonical": canonical,
                    "scaffold": scaffold,
                    "score": score,
                    "qed": qed,
                    "mw": mw,
                    "sa_raw": sa_raw,
                    "pains": pains,
                    "brenk": brenk,
                    "structural_alert": pains or brenk,
                    "logp": float(Crippen.MolLogP(mol)),
                    "tpsa": float(Descriptors.TPSA(mol)),
                    "hbd": float(Lipinski.NumHDonors(mol)),
                    "hba": float(Lipinski.NumHAcceptors(mol)),
                    "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                    "ring_count": float(Lipinski.RingCount(mol)),
                    "mol": mol,
                }
            )

    unique_by_smiles = {}
    for record in valid_records:
        unique_by_smiles.setdefault(record["canonical"], record)
    unique_records = list(unique_by_smiles.values())
    sampled_records = unique_records[:]
    if len(sampled_records) > sample_size:
        sampled_records = rng.sample(sampled_records, sample_size)
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(record["mol"], radius=2, nBits=2048)
        for record in sampled_records
    ]

    top100 = sorted(
        unique_records,
        key=lambda record: -1e9 if record["score"] is None else record["score"],
        reverse=True,
    )[:100]
    top100_fps = [
        AllChem.GetMorganFingerprintAsBitVect(record["mol"], radius=2, nBits=2048)
        for record in top100
    ]
    sampled_fps = [
        AllChem.GetMorganFingerprintAsBitVect(record["mol"], radius=2, nBits=2048)
        for record in sampled_records
    ]

    unique_scaffolds = {record["scaffold"] for record in unique_records if record["scaffold"]}
    scaffold_counts = Counter(record["scaffold"] for record in unique_records if record["scaffold"])
    top_scaffold_count = max(scaffold_counts.values()) if scaffold_counts else 0
    top10_scaffold_count = sum(count for _scaffold, count in scaffold_counts.most_common(10))
    novel_count = sum(1 for record in unique_records if record["canonical"] not in reference_smiles)
    return {
        "file": path.name,
        "total_rows": total_rows,
        "logged_validity": logged_valid_rows / total_rows if total_rows else None,
        "rdkit_validity": len(valid_records) / total_rows if total_rows else None,
        "n_valid": len(valid_records),
        "n_unique": len(unique_records),
        "uniqueness": len(unique_records) / len(valid_records) if valid_records else None,
        "novelty_vs_reference": novel_count / len(unique_records) if unique_records and reference_smiles else None,
        "n_unique_scaffolds": len(unique_scaffolds),
        "scaffold_diversity": len(unique_scaffolds) / len(unique_records) if unique_records else None,
        "internal_diversity_sampled": pairwise_diversity(fps),
        "internal_diversity_sample_size": len(fps),
        "top100_diversity": pairwise_diversity(top100_fps),
        "top100_size": len(top100_fps),
        "mean_qed": safe_mean([record["qed"] for record in valid_records]),
        "mean_sa_raw": safe_mean([record["sa_raw"] for record in valid_records]),
        "mean_mw": safe_mean([record["mw"] for record in valid_records]),
        "mean_logp": safe_mean([record["logp"] for record in valid_records]),
        "mean_tpsa": safe_mean([record["tpsa"] for record in valid_records]),
        "mean_hbd": safe_mean([record["hbd"] for record in valid_records]),
        "mean_hba": safe_mean([record["hba"] for record in valid_records]),
        "mean_rotatable_bonds": safe_mean([record["rotatable_bonds"] for record in valid_records]),
        "mean_ring_count": safe_mean([record["ring_count"] for record in valid_records]),
        "pains_fraction": mean([1.0 if record["pains"] else 0.0 for record in valid_records]) if valid_records else None,
        "brenk_fraction": mean([1.0 if record["brenk"] else 0.0 for record in valid_records]) if valid_records else None,
        "structural_alert_fraction": mean([1.0 if record["structural_alert"] else 0.0 for record in valid_records]) if valid_records else None,
        "top_scaffold_fraction": top_scaffold_count / len(unique_records) if unique_records else None,
        "top10_scaffold_coverage": top10_scaffold_count / len(unique_records) if unique_records else None,
        "mean_nn_similarity_to_active_reference": mean_max_similarity(sampled_fps, active_reference_fps),
        "top100_nn_similarity_to_active_reference": mean_max_similarity(top100_fps, active_reference_fps),
    }


def aggregate_run_quality(per_run: list[dict]) -> dict:
    keys = [
        "logged_validity",
        "rdkit_validity",
        "uniqueness",
        "novelty_vs_reference",
        "scaffold_diversity",
        "internal_diversity_sampled",
        "top100_diversity",
        "mean_qed",
        "mean_sa_raw",
        "mean_mw",
        "mean_logp",
        "mean_tpsa",
        "mean_hbd",
        "mean_hba",
        "mean_rotatable_bonds",
        "mean_ring_count",
        "pains_fraction",
        "brenk_fraction",
        "structural_alert_fraction",
        "top_scaffold_fraction",
        "top10_scaffold_coverage",
        "mean_nn_similarity_to_active_reference",
        "top100_nn_similarity_to_active_reference",
    ]
    out = {"n_runs": len(per_run), "n_total_rows": sum(int(run.get("total_rows", 0)) for run in per_run)}
    for key in keys:
        values = finite([run.get(key) for run in per_run])
        out[key] = {"mean": mean(values) if values else None, "sd": stdev(values) if len(values) > 1 else None}
    return out


def build_quality_tables(root: Path, sample_size: int) -> dict:
    reference_paths = [
        root / "qsar_models" / "DRD2" / "splits.csv",
        root / "data" / "chembl_validation" / "DRD2_chembl_validation.csv",
        root / "data" / "chembl_validation" / "DRD2_actives.smi",
        root / "data" / "chembl_validation" / "DRD2_negatives.smi",
    ]
    reference_smiles = canonicalize_reference(reference_paths)
    active_reference_paths = [
        root / "qsar_models" / "DRD2" / "splits.csv",
        root / "data" / "chembl_validation" / "DRD2_actives.smi",
        root / "data" / "chembl_validation" / "DRD2_chembl_validation.csv",
    ]
    active_reference_smiles = canonicalize_reference(active_reference_paths, active_only=True)
    if not active_reference_smiles:
        active_reference_smiles = canonicalize_reference([root / "data" / "chembl_validation" / "DRD2_actives.smi"])
    result = {
        "reference_size": len(reference_smiles),
        "active_reference_size": len(active_reference_smiles),
        "settings": {},
    }
    for setting, spec in SETTINGS.items():
        runs = spec["runs"]()
        setting_result = {}
        for group, items in runs.items():
            per_run = []
            for idx, (name, rel_path) in enumerate(items):
                path = root / rel_path
                if not path.exists():
                    continue
                summary = summarize_run_quality(
                    path,
                    reference_smiles,
                    active_reference_smiles,
                    sample_size,
                    seed=1000 + idx,
                )
                summary["run"] = name
                per_run.append(summary)
            setting_result[group] = {"per_run": per_run, "aggregate": aggregate_run_quality(per_run)}
        result["settings"][setting] = setting_result
    return result


def latex_static_table(comparison: dict) -> str:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Static, random-validation, and real-validation controller comparison.}",
        r"\label{tab:static_random_real}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{}p{0.20\linewidth}p{0.16\linewidth}ccccc@{}}",
        r"\toprule",
        r"Setting & Primary metric & Static & Random & Real & Real--Random & Real--Static\\",
        r"\midrule",
    ]
    for setting, row in comparison.items():
        static = row["groups"]["static"]["last50"]["mean"]
        random_control = row["groups"]["random"]["last50"]["mean"]
        real = row["groups"]["real"]["last50"]["mean"]
        lines.append(
            f"{setting} & {row['metric_label']} & {f4(static)} & {f4(random_control)} & "
            f"{f4(real)} & {f4(row['deltas']['real_minus_random_last50'])} & "
            f"{f4(row['deltas']['real_minus_static_last50'])}\\\\"
        )
    lines.extend(
        [
            r"\botrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            r"\raggedright",
            r"\footnotesize",
            r"Note: Static equal-weight values are means over five static repeats; controller values use ten paired repeats for DRD2/QED/MW and five paired repeats for DRD2 activity/SA. The practical rule is two-stage: real validation must beat random-validation control to demonstrate validation specificity and static equal weighting to demonstrate practical utility.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def latex_quality_table(quality: dict) -> str:
    metric_rows = [
        ("Validity", "rdkit_validity"),
        ("Uniqueness", "uniqueness"),
        ("Target-set novelty", "novelty_vs_reference"),
        ("Internal diversity", "internal_diversity_sampled"),
        ("Scaffold diversity", "scaffold_diversity"),
        ("Mean QED", "mean_qed"),
        ("Mean SA", "mean_sa_raw"),
        ("Mean MW", "mean_mw"),
        ("PAINS fraction", "pains_fraction"),
        ("Top-100 diversity", "top100_diversity"),
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Generated-molecule quality sanity checks.}",
        r"\label{tab:molecule_quality}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{}p{0.22\linewidth}p{0.22\linewidth}ccc@{}}",
        r"\toprule",
        r"Setting & Metric & Static & Random & Real\\",
        r"\midrule",
    ]
    for setting, setting_result in quality["settings"].items():
        first = True
        for label, key in metric_rows:
            values = []
            for group in ["static", "random", "real"]:
                metric = setting_result[group]["aggregate"].get(key, {})
                values.append(f3_pm(metric if isinstance(metric, dict) else None))
            setting_cell = setting if first else ""
            lines.append(f"{setting_cell} & {label} & {values[0]} & {values[1]} & {values[2]}\\\\")
            first = False
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\botrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            r"\raggedright",
            r"\footnotesize",
            r"Note: Values are run-level mean$\pm$SD. Target-set novelty is relative to the DRD2 QSAR split and ChEMBL validation reference molecules, not the full chemical universe or the REINVENT pretraining corpus. Internal diversity is the mean pairwise Morgan-fingerprint Tanimoto distance on up to 1000 unique molecules per run; top-100 diversity is computed after ranking by the logged scalar training score. Mean SA is the Ertl--Schuffenhauer synthetic-accessibility score, where lower is easier to synthesize.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def latex_extended_quality_table(quality: dict) -> str:
    metric_rows = [
        ("Mean logP", "mean_logp"),
        ("Mean TPSA", "mean_tpsa"),
        ("Mean HBD", "mean_hbd"),
        ("Mean HBA", "mean_hba"),
        ("Mean rotatable bonds", "mean_rotatable_bonds"),
        ("Mean ring count", "mean_ring_count"),
        ("Brenk alert fraction", "brenk_fraction"),
        ("Any alert fraction", "structural_alert_fraction"),
        ("Top scaffold fraction", "top_scaffold_fraction"),
        ("Top-10 scaffold coverage", "top10_scaffold_coverage"),
        ("Mean NN similarity to target actives", "mean_nn_similarity_to_active_reference"),
        ("Top-100 NN similarity to target actives", "top100_nn_similarity_to_active_reference"),
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Extended physicochemical and scaffold-distribution sanity checks.}",
        r"\label{tab:molecule_quality_extended}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{}p{0.22\linewidth}p{0.28\linewidth}ccc@{}}",
        r"\toprule",
        r"Setting & Metric & Static & Random & Real\\",
        r"\midrule",
    ]
    for setting, setting_result in quality["settings"].items():
        first = True
        for label, key in metric_rows:
            values = []
            for group in ["static", "random", "real"]:
                metric = setting_result[group]["aggregate"].get(key, {})
                values.append(f3_pm(metric if isinstance(metric, dict) else None))
            setting_cell = setting if first else ""
            lines.append(f"{setting_cell} & {label} & {values[0]} & {values[1]} & {values[2]}\\\\")
            first = False
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\botrule",
            r"\end{tabularx}",
            r"\vspace{2pt}",
            r"\raggedright",
            r"\footnotesize",
            r"Note: Values are run-level mean$\pm$SD. Nearest-neighbor (NN) similarity uses Morgan fingerprints and target-active reference molecules when available; if active-reference files are absent, the corresponding JSON fields are null. Scaffold enrichment is computed from unique generated molecules per run.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-prefix", default="manuscript")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    comparison = build_static_controller_comparison(root)
    (root / f"{args.output_prefix}_static_controller_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latex_parts = [latex_static_table(comparison)]

    if args.quality:
        quality = build_quality_tables(root, sample_size=args.sample_size)
        (root / f"{args.output_prefix}_molecule_quality_sanity.json").write_text(
            json.dumps(quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        latex_parts.append(latex_quality_table(quality))
        latex_parts.append(latex_extended_quality_table(quality))

    (root / f"{args.output_prefix}_additional_tables.tex").write_text(
        "\n\n".join(latex_parts) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
