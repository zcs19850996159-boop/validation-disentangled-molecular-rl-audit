# Ablation plan: validation-driven RL weights in REINVENT4

## Goal

Move the validation-driven RL controller idea from adaptive loss weighting in
cross-lingual biomedical NER to multi-objective molecular generation in
REINVENT4.

The central constraint is strict signal separation:

- Training reward: the fast REINVENT4 scoring components used to update the
  agent policy.
- Controller validation signal: a separate validation-only evaluator used only
  to update objective weights. The controller must not use the same scored batch
  that trained the agent.

## Benchmarks

Use the standard molecular optimization targets:

- DRD2
- GSK3B
- JNK3

Each target should be run with the same prior, same RL budget, same random seeds,
same diversity filter, and same reporting scripts.

## Materials and methods lock

Before running the DRD2 pilot, lock the docking protocol described in
`05_vina_validation_protocol.md`.

Initial receptor set:

- DRD2: PDB 6CM4, chain A, co-crystal ligand 8NU.
- GSK3B: PDB 1Q5K, chain A, co-crystal ligand TMU.
- JNK3: PDB 4WHZ, chain A, co-crystal ligand 3NL.

Grid rule for all three targets:

- center: co-crystal ligand centroid;
- size: ligand span plus 12 Angstrom padding;
- minimum size: 22 Angstrom per axis;
- Vina defaults: `exhaustiveness=8`, `num_modes=1`, fixed seed per ligand.

Do not change receptor/grid settings inside an experiment series. If a target
fails calibration, create a new protocol version and rerun calibration before
using it as a validation signal.

Recommended multi-objective reward:

- Target activity score from a fast reward oracle or predictor.
- QED.
- SA or synthetic accessibility proxy.
- Optional molecular weight/logP range filters.
- Diversity filter kept identical across all arms.

Recommended controller validation signal:

- Primary: Vina docking on a validation-only batch for each target. Training can
  use cheap ExCAPE-DB QSAR classifiers, but the controller reward should come
  from a different oracle such as AutoDock Vina against DRD2/GSK3B/JNK3 target
  structures.
- Backup: independent QSAR ensemble per target, trained with a different
  algorithm and data split from the training reward oracle. For example, train
  reward with RandomForest on Morgan fingerprints and validate with SVM or a
  different scaffold split. This is cheaper but less persuasive than docking.

QSAR fallback self-check:

- Split the labeled source molecules by scaffold into three disjoint pools:
  reward-model training, validation-model training, and a common holdout test.
- Train the fast training reward as a RandomForest on Morgan fingerprints.
- Train the controller validation model as an SVM on a different scaffold split.
- Report both models on the common holdout and also report RF/SVM prediction
  correlation. The validation model must be useful, and the two predictors must
  not be numerically identical.
- First pilot gate: validation-model holdout AUROC >= 0.60 and RF/SVM
  prediction correlation <= 0.95. This is only a low-cost fallback gate; docking
  remains stronger evidence when it passes its own calibration.

Controller sanity checks before scaling:

- Constant-validation control: feed identical validation scores at every
  controller update. The controller must not drift under a no-change validation
  signal. If weights move systematically, fix the controller update rule before
  running more seeds.
- Random-validation control: feed shuffled or noisy validation-scale scores.
  The proposed dynamic controller should separate from this control in both
  late-window metrics and weight trajectories before expanding to more targets.

Why docking first: DRD2/GSK3B/JNK3 benchmark rewards are commonly ExCAPE-DB
activity classifiers. If controller validation uses the same classifier, the
method collapses into optimizing one proxy twice. Docking gives a clearer
validation-driven story: the controller learns weights using a slower, more
biophysically grounded signal that does not participate in agent reward updates.

Vina self-check gate:

- First run co-crystal ligand redocking. Use direct heavy-atom RMSD in the
  receptor coordinate frame with an initial gate of <= 2 Angstrom.
- Audit receptor-preparation side effects against co-crystal ligand contact
  residues, not only distance to the grid center.
- For each target, run known actives vs decoys before using docking as
  controller feedback.
- For DRD2 pilot, use ChEMBL/ExCAPE-DB actives and an independent negative set.
  DUD-E does not provide a direct DRD2 target file in its standard target list,
  so the first gate uses ChEMBL low-activity binding records as negatives; use
  DUD-E or generated property-matched decoys when target coverage is available.
- If actives are filtered by receptor-state or chemotype, match/filter the
  negative set at the same time. Otherwise a higher AUROC can reflect analogue
  or physicochemical-property bias rather than true docking discrimination.
- If the first DRD2 structure fails redocking, run an alternative-structure
  cross-check before dropping docking. Initial alternative: `DRD2_7DFP`
  (`7DFP`, spiperone-bound DRD2), which should be treated as an exploratory
  docking branch until active-vs-negative calibration passes.
- Require actives to score better than decoys, with AUROC >= 0.60 and a
  one-sided permutation p-value <= 0.10 as the first pilot-quality gate.
- If either gate fails, do not run RL pilot with that docking protocol. Use the
  independent-QSAR validation fallback for the controller pilot while docking
  remains a separate calibration branch.

## Arms

Run each arm on DRD2, GSK3B, and JNK3 with at least 5 seeds.

1. Static equal weights
   - Fixed weights, e.g. activity/QED/SA = 1/1/1 or normalized equivalent.
   - This is the main baseline.

2. Static tuned weights
   - Small grid over fixed weights, selected on a validation budget before final
     reporting.
   - This checks whether dynamic control beats careful static tuning.

3. Old bandit controller
   - The original heuristic rule: increase weights for objectives whose recent
     score drops and decrease weights for objectives that improve.
   - Include only as an ablation, not as the proposed method.

4. RL controller with independent validation signal
   - The proposed method.
   - Controller actions are per-objective {-1, 0, +1} multiplicative weight
     changes.
   - Controller reward is improvement in independent Vina validation metrics.
   - Validation molecules are sampled separately from the current policy and are
     not used for agent reward updates.

5. Leakage control
   - Same RL controller, but intentionally feed it the training reward batch as
     validation feedback.
   - This is not a valid method; it quantifies the overfitting/leakage risk and
     supports the paper argument.

6. Random validation control
   - Same RL controller, but shuffle cached validation scores across intervals
     or add strong noise before passing them to the controller.
   - For the QSAR fallback pilot, use `scripts/validate_random_control.py` in
     `shuffle_history` mode: it samples historical validation-score records and
     feeds them to the controller for the current validation batch.
   - This checks that gains are not just from injecting weight noise or from
     controller dynamics that are insensitive to the semantic content of the
     validation signal.
   - Compare both final molecule metrics and controller trajectories. If random
     validation produces similar weight trends and similar late-window metrics,
     tune the controller before scaling seeds or targets.

7. QSAR-validation backup controller
   - Same as arm 4, but validation uses an independent QSAR model instead of
     docking.
   - This is a practical low-cost fallback, but should be framed as weaker
     evidence than docking.

## Metrics

Primary decision metric:

- Late-window independent validation aggregate, averaged over seeds. The late
  window is the final 20% of training steps for formal runs, or the last
  50 steps for the current 300-step DRD2 pilot. For the single-target DRD2 QSAR
  pilot this is `last50_mean_validation_activity`; for full multi-objective
  runs this should be the same late-window mean computed on the independent
  validation aggregate across activity/QED/SA or the locked objective set.

Rationale: this metric asks whether the final generator has better stable
independent-validation performance. A single final step is too noisy, and an
all-row average mixes final quality with the dynamic controller's early learning
cost. All-row metrics remain useful for training-efficiency analysis, but they
are not the primary decision criterion.

Secondary metrics:

- All-row independent validation aggregate, used as a training-efficiency
  diagnostic.
- Final-step independent validation aggregate, reported only as a noisy
  endpoint diagnostic.
- Best validation activity per target.
- Mean top-10 and top-100 validation activity.
- Success rate above target-specific activity threshold.
- Number of unique valid molecules above threshold.
- QED, SA, molecular weight/logP distributions.
- Internal diversity and scaffold diversity.
- Novelty against training/reference molecules if reference sets are available.
- Controller weight trajectories.
- Controller reward trajectory and entropy.
- Wall-clock cost and validation-evaluator cost.

Statistical reporting:

- Report mean and standard deviation over seeds.
- Use paired seed comparisons where possible.
- Plot learning curves against both RL steps and wall-clock time.

## Minimal run matrix

For a first paper-quality sweep:

- 3 targets x 4 main arms x 5 seeds:
  static equal, static tuned, old bandit, RL independent validation.

Then add diagnostic controls:

- 3 targets x 2 diagnostic arms x 3 seeds:
  leakage control, random validation control.

Optional low-cost QSAR backup:

- 3 targets x 1 independent-QSAR validation arm x 3 seeds.

## DRD2 QSAR Controller Diagnostic Queue

Before expanding to more targets or formal multi-seed claims, use the DRD2 QSAR
fallback as a controller-mechanism diagnostic. These runs are not final
benchmark evidence; they decide whether the controller is actually sensitive to
meaningful validation feedback.

Locked reference run:

- `dynamic_fixed`: real independent QSAR validation after the no-signal drift
  fix, old hyperparameters.
- `random_fixed`: shuffled-history validation control after the same fix.

Single-variable diagnostics, each paired with a real-validation run and a
random-validation control:

- `stdadv`: set `REINVENT_CONTROLLER_STANDARDIZE_ADVANTAGE=1`.
  Tests whether component scale/variance is dominating the controller update.
- `step005`: set `REINVENT_CONTROLLER_ACTION_STEP=0.05`.
  Tests whether the old step size makes weight trajectories too exploratory.
- `ve30`: set `REINVENT_CONTROLLER_VALIDATE_EVERY=30`.
  Tests whether validation noise at the current update frequency is driving
  unstable actions.
- `ctrlseed23`: set `REINVENT_CONTROLLER_SEED=23` with old hyperparameters.
  This is a stochastic controller replicate, not a fully controlled REINVENT
  seed, because the current TOML configs do not expose a global REINVENT seed.

Optional variance check:

- `static_equal_rep2`: repeat the static equal-weight run as a stochastic
  baseline replicate.

Decision rule:

- If `dynamic_fixed` separates cleanly from `random_fixed`, keep the structural
  fix as the main controller change and treat the single-variable runs as
  sensitivity analyses.
- If `dynamic_fixed` still resembles `random_fixed`, use the single-variable
  diagnostics to choose the next controller change. Do not combine changes until
  one variable has shown a clear effect.

## Implementation notes

- Do not edit REINVENT4 core source files.
- Keep controller code in `03_dynamic_weight_controller.py` or a
  `reinvent_plugins` plugin package.
- Training reward config remains TOML-driven.
- Independent Vina validation can be connected with
  `REINVENT_VINA_COMMAND="python dock_with_vina.py --target {target} --input {input} --output {output}"`.
- Independent command-style validation can also be connected with
  `REINVENT_VALIDATION_COMMAND="python validate_qsar.py --input {input} --output {output}"`.
- The command must emit a CSV with one column per controller objective.
- Save generated SMILES, reward scores, validation scores, and controller
  histories for every seed.
- Measure Vina seconds per molecule during calibration and choose
  `validate_every` so validation overhead stays below roughly 30-40% of total
  wall-clock time.
- Pilot may use synchronous docking. Formal multi-target ablations may need an
  async validation worker if docking cost dominates RL training.
