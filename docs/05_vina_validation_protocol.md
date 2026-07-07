# Vina validation protocol

## Purpose

Before using docking as the controller validation signal, we must show that the
prepared docking protocol is at least directionally meaningful for the target.
The minimum sanity checks are:

- co-crystal ligand redocking should reproduce the crystallographic pose;
- known active molecules should dock better than decoys;
- `validation_score = -vina_score`, so higher validation score is better;
- DRD2 pilot should not start until this check passes.

## Locked receptor structures

These are the initial locked structures. If any one fails calibration, do not
quietly swap it during RL runs. Re-prepare it, rerun calibration, and record a
new protocol version.

| Target | PDB | Chain | Co-crystal ligand | Altloc rule | Use |
|---|---:|---:|---:|---:|---|
| DRD2 | 6CM4 | A | 8NU | A | Pilot target; D2 receptor bound to risperidone |
| GSK3B | 1Q5K | A | TMU | A | Expansion target |
| JNK3 | 4WHZ | A | 3NL | A | Expansion target |

Source pages:

- DRD2: https://www.rcsb.org/structure/6CM4
- GSK3B: https://www.rcsb.org/structure/1Q5K
- JNK3: https://www.rcsb.org/structure/4WHZ

## Grid rule

Use a deterministic rule for all targets:

- center: centroid of the co-crystal ligand heavy-atom coordinates;
- size: ligand coordinate span plus 12 Angstrom padding;
- minimum size: 22 Angstrom per axis;
- Vina defaults: `exhaustiveness=8`, `num_modes=1`, fixed seed per ligand.

Generate locked receptor JSON files:

```bash
python scripts/prepare_vina_receptors.py --targets DRD2,GSK3B,JNK3 --out-dir receptors
```

The generated `receptors/<TARGET>.json` files are part of the method record.
After calibration, do not change them inside an experiment series.

## Receptor preparation notes

The script strips the selected protein chain and co-crystal ligand from the PDB,
keeps blank altloc plus altloc `A`, and clears retained altloc markers before
receptor preparation. Receptor PDBQT generation uses Meeko
`mk_prepare_receptor.py` with fixed grid center/size, `--default_altloc A`, and
`--allow_bad_res`; the generated JSON records whether PDBQT creation succeeded
and points to the Meeko preparation log. Before using the receptor in a real
run, review:

- protonation states at the chosen docking pH;
- receptor hydrogens and Gasteiger charges;
- Meeko preparation logs, especially residues deleted by `--allow_bad_res`;
- retained cofactors/ions/waters, if any are scientifically justified;
- DRD2 binding-site protonation, especially GPCR conserved charged residues;
- whether the co-crystal ligand redocking pose is reasonable.

Run co-crystal ligand redocking before active-vs-decoy calibration:

```bash
python scripts/redock_cocrystal_ligand.py \
  --target DRD2 \
  --exhaustiveness 8 \
  --output redocking/DRD2_8NU_redocking_exh8.csv
```

Initial gate: direct heavy-atom RMSD <= 2 Angstrom in the receptor coordinate
frame. If this fails, inspect ligand/receptor preparation before spending time
on a larger active-vs-decoy calibration.

The redocking script also reports:

- symmetry-aware no-fit RMSD, used to rule out atom-index/symmetry artifacts;
- RDKit best-fit RMSD, only as a diagnostic, not as the docking pose gate;
- ligand formal charge;
- minimum margin between ligand coordinates and the Vina box boundary.

For DRD2/8NU, also test pH-dependent ligand preparation:

```bash
python scripts/redock_cocrystal_ligand.py \
  --target DRD2 \
  --exhaustiveness 8 \
  --protonate-ph 7.4 \
  --output redocking/DRD2_8NU_redocking_exh8_ph74.csv
```

If `6CM4/8NU` fails the redocking gate, run the same cross-check on an
independent DRD2 structure before abandoning docking. The first alternative
receptor is:

- `DRD2_7DFP`: PDB `7DFP`, chain `A`, ligand `SIP` spiperone.

```bash
python scripts/prepare_vina_receptors.py \
  --targets DRD2_7DFP \
  --out-dir receptors

python scripts/redock_cocrystal_ligand.py \
  --target DRD2_7DFP \
  --exhaustiveness 8 \
  --output redocking/DRD2_7DFP_SIP_redocking_exh8_sym.csv
```

Use this as a structure-level cross-check: if an alternative DRD2 structure
passes redocking, the failure is more likely structure/ligand specific than a
general failure of the Vina pipeline.

Audit whether deleted residues touch the co-crystal pocket:

```bash
python scripts/audit_receptor_prep.py \
  --target DRD2 \
  --pocket-cutoff 6.0 \
  --output receptor_audit/DRD2_receptor_pocket_audit.csv
```

## DRD2 calibration gate

Use known DRD2 actives from ChEMBL/ExCAPE-DB and an independent negative set.
DUD-E does not provide a direct DRD2 target file in its standard target list, so
the initial DRD2 gate uses ChEMBL low-activity binding records as negatives. For
targets with DUD-E coverage, DUD-E decoys or generated property-matched decoys
can be used as a stronger follow-up.

If actives are filtered toward a receptor-state-specific chemotype, the negative
set must be filtered or matched at the same time. Do not compare
antagonist-like DRD2 actives against the original unfiltered low-activity set:
that can inflate AUROC through analogue/property bias rather than true binding
discrimination. At minimum, match negatives on simple physicochemical
descriptors such as molecular weight, logP, TPSA, HBA/HBD, rotatable bonds,
heavy-atom count, and formal charge while avoiding the same Murcko scaffold and
high pairwise fingerprint similarity.

Fetch the initial DRD2 calibration set:

```bash
python scripts/fetch_chembl_validation_sets.py \
  --target DRD2 \
  --output-dir data/chembl_validation \
  --active-threshold 7.0 \
  --negative-threshold 5.0 \
  --max-per-class 256
```

Run a balanced sample first, e.g. 64 actives + 64 negatives:

```bash
python scripts/vina_calibration.py \
  --target DRD2 \
  --actives data/chembl_validation/DRD2_actives.smi \
  --decoys data/chembl_validation/DRD2_negatives.smi \
  --dock-command "python scripts/dock_with_vina.py --target {target} --input {input} --output {output}" \
  --sample-size 64 \
  --permutations 1000
```

Initial pass gate:

- AUROC >= 0.60 for active-vs-decoy separation;
- permutation p-value <= 0.10 for active mean validation score > decoy mean;
- inspect distribution plots before treating the signal as trustworthy.

For paper-quality runs, tighten this gate or report it transparently.

For an antagonist-state DRD2 follow-up using `DRD2_7DFP`, build a matched set
before running the 64+64 gate:

```bash
python scripts/build_matched_calibration_set.py \
  --input data/chembl_validation/DRD2_chembl217_validation.csv \
  --reference-pdb receptors/DRD2/DRD2_8NU_cocrystal_ligand.pdb \
  --reference-pdb receptors/DRD2_7DFP/DRD2_7DFP_SIP_cocrystal_ligand.pdb \
  --active-similarity-threshold 0.20 \
  --max-pair-tanimoto 0.45 \
  --n 64 \
  --output-prefix data/chembl_validation_matched/DRD2_7DFP_antagonist_matched_t020
```

Then run the same Vina calibration on the matched active/negative `.smi`
outputs. If either the AUROC gate or the p-value gate fails, do not use that
docking protocol as the controller validation oracle.

## Wall-clock budget

The calibration script reports `seconds_per_molecule`. Use it to set
`validate_every` and validation batch size. For the pilot, target:

- validation overhead <= 30-40% of total wall-clock time;
- validation batch size 16-64 depending on measured docking speed;
- `validate_every` increased if docking dominates training time.

If formal ablations with 5+ seeds become too slow, switch from synchronous Vina
to an async validation worker: the controller should consume the latest completed
validation result instead of blocking every RL step.

## Random validation control

Run a cheap control in the DRD2 pilot:

- cache Vina validation scores from real validation batches;
- shuffle scores across validation intervals or add strong noise before
  controller update;
- keep the same sampled molecules and same training reward.

This distinguishes response to a meaningful validation signal from response to
arbitrary noise.
