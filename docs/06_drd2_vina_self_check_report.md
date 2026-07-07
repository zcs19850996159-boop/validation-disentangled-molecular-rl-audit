# DRD2 Vina self-check report

## Status

Current DRD2 docking oracle is not ready to drive the RL controller.

The quick active-vs-negative check did not pass the pilot gate. This is useful:
it prevents us from running a controller experiment on a validation signal that
is probably too noisy in its current form.

Important interpretation caveat: the quick check used only 16 actives and 16
negatives. That sample size has low statistical power and a wide AUROC
uncertainty interval. Therefore, the result should be read as "we did not
observe a strong validation signal in this small sample", not as proof that the
receptor preparation is wrong. Direct geometric diagnostics, especially
co-crystal ligand redocking, should be prioritized before spending time on a
larger 64+64 calibration.

## Server outputs

Remote project directory:

```text
/root/autodl-tmp/four_project
```

Key generated files:

```text
receptors/DRD2.json
receptors/DRD2/DRD2_receptor.pdbqt
receptors/DRD2/DRD2_receptor.prep.log
redocking/DRD2_8NU_redocking_exh8.csv
redocking/DRD2_8NU_redocking_exh32.csv
redocking/DRD2_8NU_redocking_exh8_sym2.csv
redocking/DRD2_8NU_redocking_exh32_sym2.csv
redocking/DRD2_8NU_redocking_exh8_ph74_sym2.csv
redocking/DRD2_8NU_redocking_exh32_ph74_sym2.csv
receptor_audit/DRD2_receptor_pocket_audit.csv
receptors/DRD2_7DFP.json
redocking/DRD2_7DFP_SIP_redocking_exh8_sym.csv
redocking/DRD2_7DFP_SIP_redocking_exh32_sym.csv
redocking/DRD2_7DFP_SIP_redocking_exh8_ph74_sym.csv
vina_calibration_quick_7dfp/DRD2_7DFP_vina_calibration_scores.csv
vina_calibration_quick_7dfp/DRD2_7DFP_vina_calibration_summary.csv
data/chembl_validation_matched/DRD2_7DFP_antagonist_matched_t020_actives.smi
data/chembl_validation_matched/DRD2_7DFP_antagonist_matched_t020_negatives.smi
data/chembl_validation_matched/DRD2_7DFP_antagonist_matched_t020_audit.csv
data/chembl_validation_matched/DRD2_7DFP_antagonist_matched_t020_summary.json
vina_calibration_matched_7dfp_t020/DRD2_7DFP_vina_calibration_scores.csv
vina_calibration_matched_7dfp_t020/DRD2_7DFP_vina_calibration_summary.csv
data/chembl_validation/DRD2_actives.smi
data/chembl_validation/DRD2_negatives.smi
vina_calibration_quick/DRD2_vina_calibration_scores.csv
vina_calibration_quick/DRD2_vina_calibration_summary.csv
```

## Receptor preparation

Locked DRD2 receptor setup:

- PDB: `6CM4`
- protein chain: `A`
- co-crystal ligand: `8NU`
- altloc rule: keep blank altloc and altloc `A`
- grid center: `(9.925, 5.846, -9.582)`
- grid size: `(22.0, 22.0, 26.815)`
- Vina defaults: `exhaustiveness=8`, `num_modes=1`, fixed seed

Meeko generated a receptor PDBQT, but the preparation log reports 70 unique
deleted residues for DRD2 under `--allow_bad_res`. GSK3B had 0 and JNK3 had 1 in
the same preparation pass. This makes DRD2 receptor cleanup the first suspect.

Distance check against the DRD2 grid center:

- deleted residues within 8 Angstrom: 0
- deleted residues within 12 Angstrom: 1
- deleted residues within 16 Angstrom: 4
- closest deleted residues: `A:407` at 11.27 Angstrom, `A:394` at 12.18
  Angstrom, `A:406` at 13.55 Angstrom, `A:99` at 15.00 Angstrom

Pocket-contact audit against the co-crystal ligand is more informative than
grid-center distance:

- 6 Angstrom co-crystal pocket residues: 28
- deleted pocket residues within 6 Angstrom: 0
- key retained DRD2 pocket residues include `D114`, `S193`, `S194`, `S197`,
  `W386`, and `F389`
- closest deleted residues to the co-crystal ligand: `L407` at 6.94 Angstrom,
  `V406` at 7.47 Angstrom, `E99` at 9.33 Angstrom

This weakens the hypothesis that Meeko deleted a core orthosteric-contact
residue. DRD2 receptor preparation still needs review, but the current evidence
does not point to a missing central pocket side chain as the main cause.

## Co-crystal ligand redocking

Redocking script:

```bash
python scripts/redock_cocrystal_ligand.py \
  --target DRD2 \
  --exhaustiveness 8 \
  --output redocking/DRD2_8NU_redocking_exh8.csv
```

Result:

| Protonation | Exhaustiveness | Charge | Vina score | Direct RMSD | Symmetry no-fit RMSD | Symmetry best-fit RMSD | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| neutral | 8 | 0 | -12.228 | 2.333 Angstrom | 2.333 Angstrom | 1.706 Angstrom | fail |
| neutral | 32 | 0 | -12.228 | 2.333 Angstrom | 2.333 Angstrom | 1.706 Angstrom | fail |
| pH 7.4 | 8 | +1 | -12.649 | 2.363 Angstrom | 2.363 Angstrom | 1.720 Angstrom | fail |
| pH 7.4 | 32 | +1 | -12.669 | 2.375 Angstrom | 2.375 Angstrom | 1.724 Angstrom | fail |

The RMSD gate was 2 Angstrom. This is a borderline failure rather than a
catastrophic redocking failure, but increasing exhaustiveness from 8 to 32 did
not improve the pose. That points away from simple search-budget insufficiency
and toward receptor/ligand preparation details.

Symmetry/atom-mapping follow-up:

- RDKit symmetry-aware no-fit RMSD is identical to direct RMSD for the neutral
  and pH 7.4 variants.
- Therefore, the 2.33-2.38 Angstrom pose RMSD is not an atom-index/symmetry
  false positive.
- RDKit `GetBestRMS` after ligand fitting is below 2 Angstrom, which means the
  docked ligand shape is close to the crystal pose, but this is not the docking
  pose gate because it permits a ligand-level rigid-body fit after docking.

Ligand protonation follow-up:

- Open Babel default ligand preparation gives formal charge 0.
- Open Babel pH 7.4 preparation gives formal charge +1, matching the expected
  protonated amine state better.
- The pH 7.4 ligand scores slightly better but does not improve redocking RMSD.

Grid coverage follow-up:

- Crystal ligand minimum margin to the docking box boundary: about 5.08
  Angstrom.
- Redocked ligand minimum margin to the docking box boundary: about 5.04-5.09
  Angstrom.
- The current box covers the co-crystal and redocked poses with several
  Angstroms of margin, so box truncation is not the immediate explanation.

## Alternative DRD2 structure cross-check

To distinguish a `6CM4`-specific issue from a general DRD2/Vina pipeline issue,
we prepared and redocked an independent DRD2 antagonist-bound structure:

- PDB: `7DFP`
- title: human dopamine D2 receptor in complex with spiperone
- method/resolution: X-ray diffraction, 3.10 Angstrom
- construct note: engineered DRD2/soluble cytochrome B562 chimera with Fab
- ligand: `SIP`, spiperone
- RCSB page: https://www.rcsb.org/structure/7DFP

This is not a non-engineered wild-type receptor, but it is independent from
`6CM4`: different antagonist, different fusion partner, and different structure
determination experiment.

`DRD2_7DFP` receptor preparation:

- grid center: `(-93.057, -21.490, 213.490)`
- grid size: `(22.000, 22.000, 24.051)`
- Meeko deleted residues: 14
- deleted residues within 6 Angstrom of co-crystal ligand: 0

`7DFP/SIP` redocking:

| Protonation | Exhaustiveness | Charge | Vina score | Symmetry no-fit RMSD | Gate |
|---|---:|---:|---:|---:|---:|
| neutral | 8 | 0 | -10.868 | 0.610 Angstrom | pass |
| neutral | 32 | 0 | -10.901 | 0.620 Angstrom | pass |
| pH 7.4 | 8 | +1 | -11.177 | 0.810 Angstrom | pass |

This is a strong cross-check: the same receptor-prep, ligand-prep, Vina, and
RMSD scripts can reproduce an independent DRD2 antagonist pose well below the
2 Angstrom gate. Therefore, the `6CM4/8NU` 2.33 Angstrom redocking result should
not be interpreted as proof that the whole DRD2/Vina pipeline is broken.

Quick `DRD2_7DFP` active-vs-negative calibration using the same 16+16 ChEMBL
sample:

- active mean validation score: `9.808`
- negative mean validation score: `9.467`
- active median validation score: `9.480`
- negative median validation score: `9.480`
- AUROC: `0.570`
- permutation p-value: `0.159`
- seconds per molecule: `8.61`
- passed pilot statistical gate: `False`

This is slightly better than the `6CM4` quick check (`AUROC 0.543`, `p=0.308`)
but still does not provide a strong validation signal at `n=16+16`. Because
`7DFP` passes redocking cleanly, the weak active-vs-negative signal is now more
likely to reflect the current mixed ChEMBL calibration set, ligand-class
mismatch, or Vina's ranking limitations than a basic redocking/pipeline failure.

## Matched antagonist-like 7DFP calibration

To avoid analogue/property bias, the next `7DFP` calibration did not simply
filter actives while keeping the original negative set. Instead, actives were
selected for similarity to the antagonist co-crystal ligands `8NU` and `SIP`,
and each active was paired with a low-activity ChEMBL negative matched on
physicochemical descriptors while avoiding the same Murcko scaffold and high
pairwise Morgan fingerprint similarity.

Matched-set construction:

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

Matched-set audit summary:

- active pool: `256`
- negative pool: `223`
- matched pairs: `64`
- active mean MW/logP/TPSA: `431.57 / 3.985 / 60.745`
- negative mean MW/logP/TPSA: `424.71 / 3.867 / 60.466`
- active mean HBA/HBD/rotatable bonds/heavy atoms/formal charge:
  `4.531 / 1.234 / 6.578 / 30.594 / 0`
- negative mean HBA/HBD/rotatable bonds/heavy atoms/formal charge:
  `4.625 / 1.125 / 6.375 / 30.141 / 0`
- mean pair property distance: `1.070`
- mean pair Tanimoto: `0.191`
- warning: one matched negative pair had property distance `3.995`

Full matched 64+64 Vina calibration on `DRD2_7DFP`:

- attempted molecules: `64` actives + `64` negatives
- usable docked molecules: `63` actives + `64` negatives
- excluded molecule: `active_13`, `CHEMBL2204343`; Open Babel produced an
  empty ligand PDBQT for a silicon-containing compound, so this molecule was
  excluded before statistical scoring because its atom type was outside the
  current ligand-parameterization path
- active mean validation score: `10.177`
- negative mean validation score: `9.910`
- active median validation score: `10.032`
- negative median validation score: `9.8585`
- AUROC: `0.565`
- one-sided permutation p-value: `0.049`
- wall-clock: `1243.8` seconds
- seconds per attempted molecule: `9.79`
- passed pilot statistical gate: `False`

Interpretation: after chemotype filtering and property-matched negative control,
the direction of the mean score difference is favorable and the permutation test
is nominally significant, but the rank separation remains too weak
(`AUROC < 0.60`). These are not contradictory conclusions. The permutation
p-value asks whether the observed active-vs-negative mean gap is likely to be
pure noise; at `n=63+64`, the answer is probably no. AUROC asks how large and
practically useful the ranking effect is; here the effect is still close to
random ranking (`0.565` vs `0.5`). Therefore, the signal is plausibly real but
too weak to use as the controller's main validation oracle.

This is why the pilot gate requires both statistical evidence and an effect-size
threshold. If the gate used only `p < 0.05`, a weak but impractical signal could
be accepted merely because the calibration sample is large enough. The result is
more defensible than the earlier unmatched quick check because it controls the
main analogue-bias concern, but it still supports keeping docking as a side
branch rather than the DRD2 pilot blocker.

## Calibration data

Initial calibration data came from ChEMBL target `CHEMBL217`:

- actives: binding records with `pChEMBL >= 7.0`
- negatives: binding records with `pChEMBL <= 5.0`
- standard types: `IC50`, `KI`, `KD`, `EC50`
- assay type: `B`
- fetched counts: 256 actives, 223 negatives

DUD-E does not provide a direct DRD2 target file in its standard target list, so
ChEMBL low-activity records were used for the first quick gate.

## Receptor conformation note

The `6CM4` PDB header records the structure as "D2 dopamine receptor bound to
the atypical antipsychotic drug risperidone", published in Nature 2018
(`PMID 29466326`, `DOI 10.1038/NATURE25758`) at 2.87 Angstrom resolution. The
construct is an engineered D2 receptor/endolysin chimera. Because risperidone is
an atypical antipsychotic, this receptor should be treated as
antagonist/inactive-state biased for docking protocol design. The
inactive-state label is an inference from the bound ligand and pharmacology, not
something proved by the quick calibration run.

The current ChEMBL calibration set uses binding records only; it is not
stratified into antagonist, agonist, and partial-agonist chemotypes. A mismatch
between the inactive-like risperidone-bound receptor and an active set enriched
for agonist-like ligands could reduce AUROC even when receptor preparation is
technically acceptable.

Before the next calibration, split or tag the DRD2 calibration molecules by
known mechanism where possible. At minimum, separately inspect top active
chemotypes from ChEMBL and rerun calibration on antagonist-like actives if the
goal is to validate a risperidone-bound receptor oracle.

## Quick calibration result

Command shape:

```bash
python scripts/vina_calibration.py \
  --target DRD2 \
  --actives data/chembl_validation/DRD2_actives.smi \
  --decoys data/chembl_validation/DRD2_negatives.smi \
  --dock-command "python scripts/dock_with_vina.py --target {target} --input {input} --output {output}" \
  --sample-size 16 \
  --permutations 200 \
  --output-dir vina_calibration_quick
```

Result:

- usable docked molecules: 16 actives + 16 negatives
- active mean validation score: `9.700`
- negative mean validation score: `9.520`
- active median validation score: `9.594`
- negative median validation score: `9.933`
- AUROC: `0.543`
- permutation p-value: `0.308`
- passed: `False`

Pilot gate was `AUROC >= 0.60` and one-sided permutation `p <= 0.10`. This
small diagnostic run did not pass that gate, but because `n=16+16` is
underpowered, this is not enough by itself to diagnose the root cause.

## Wall-clock observation

The 32-molecule quick run took 271 seconds:

- mean: 8.47 seconds per molecule
- max: 22.73 seconds per molecule

The matched `DRD2_7DFP` 64+64 run took 1243.8 seconds for 128 attempted
molecules:

- mean: 9.79 seconds per attempted molecule
- usable docked molecules: 127/128

Rough synchronous validation costs at the same settings:

- batch 16: about 2.6 minutes
- batch 32: about 5.2 minutes
- batch 64: about 10.4 minutes

For any RL pilot, choose `validate_every` so Vina is not more than 30-40% of
wall-clock time, or move Vina validation into an async worker.

## Recommended next step

Do not run the DRD2 RL pilot with this Vina oracle yet.

Proceed with independent-QSAR validation fallback for the DRD2 controller pilot
so the main RL-controller work is not blocked by docking oracle calibration.

Keep docking as a side branch:

1. Keep `7DFP/SIP` as the stronger DRD2 docking-branch candidate because it
   passes redocking cleanly.
2. Treat the matched 64+64 result as directional but insufficient: p-value is
   favorable, but AUROC remains below the pilot gate.
3. If revisiting docking, first filter or repair unsupported/problematic
   molecules such as the silicon-containing `CHEMBL2204343`, then inspect score
   distributions and consider stronger antagonist labels or rescoring. Do not
   spend more RL time on this branch until active-vs-negative ranking improves.
4. Record `6CM4/8NU` as a structure-specific borderline redocking failure and
   `7DFP/SIP` as a redocking-pass but calibration-fail case.
