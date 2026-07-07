# DRD2 QSAR fallback pilot report

## Status

The DRD2 docking branch is paused because `DRD2_7DFP` passed redocking but did
not pass active-vs-negative ranking calibration. The main controller pilot now
uses an independent-QSAR fallback:

- training reward: RandomForest QSAR model;
- controller validation signal: SVM QSAR model;
- model training pools: scaffold-disjoint;
- common self-check pool: scaffold-held-out test set.

This is weaker than a validated docking oracle, but it preserves the core
separation between training reward and controller feedback.

## Files

Remote project directory:

```text
/root/autodl-tmp/four_project
```

Key files:

```text
scripts/train_qsar_fallback.py
scripts/validate_qsar.py
scripts/score_qsar_json.py
scripts/analyze_qsar_pilot.py
scripts/analyze_controller_history.py
scripts/validate_random_control.py
scripts/simulate_controller_constant.py
03_dynamic_weight_controller.py
07_drd2_qsar_pilot.toml
08_drd2_qsar_static_equal.toml
10_drd2_qsar_random_validation.toml
qsar_models/DRD2/reward_rf.pkl
qsar_models/DRD2/validation_svm.pkl
qsar_models/DRD2/summary.json
controller_history_drd2_qsar_seed17.jsonl
drd2_qsar_pilot_1.csv
drd2_qsar_static_equal_1.csv
controller_history_drd2_qsar_random_seed17.jsonl
drd2_qsar_random_validation_1.csv
qsar_models/DRD2/drd2_qsar_pilot_posthoc.json
qsar_models/DRD2/drd2_qsar_pilot_posthoc_with_random.json
controller_history_drd2_qsar_seed17.summary.json
controller_history_drd2_qsar_seed17.weights.svg
controller_history_drd2_qsar_seed17.validation.svg
controller_history_drd2_qsar_random_seed17.summary.json
controller_history_drd2_qsar_random_seed17.weights.svg
controller_history_drd2_qsar_random_seed17.validation.svg
controller_constant_zero_after_fix.json
controller_constant_representative_after_fix.json
```

## QSAR fallback self-check

Command:

```bash
python scripts/train_qsar_fallback.py \
  --target DRD2 \
  --input data/chembl_validation/DRD2_chembl_validation.csv \
  --output-dir qsar_models/DRD2 \
  --seed 17 \
  --min-validation-auc 0.60 \
  --max-model-correlation 0.95
```

Result:

- total labeled molecules: `479`
- split method: scaffold-disjoint
- reward RF train size: `239`
- validation SVM train size: `120`
- common holdout size: `120`
- reward RF holdout AUROC: `0.961`
- validation SVM holdout AUROC: `0.941`
- reward RF holdout average precision: `0.958`
- validation SVM holdout average precision: `0.943`
- RF/SVM prediction Pearson correlation: `0.868`
- RF/SVM mean absolute prediction difference: `0.132`
- self-check passed: `true`

Interpretation: the validation SVM is strong enough for a low-cost pilot, and
the RF/SVM predictions are related but not identical. This supports using the
SVM as a controller feedback signal distinct from the RF training reward.

## Training-vs-validation signal independence

Artifact:

- `scripts/qsar_signal_independence.py`
- `scripts/remote_qsar_signal_independence.py`
- `qsar_signal_independence_DRD2.json`

Held-out test-set comparison between the training reward RF and validation SVM:

| Metric | Value |
|---|---:|
| Held-out molecules scored | 120 |
| Pearson correlation | 0.868 |
| Spearman correlation | 0.857 |
| Mean absolute prediction difference | 0.132 |
| Top-10% Jaccard overlap | 0.714 |

Interpretation:

The RF reward model and SVM validation model are algorithmically independent and
trained on disjoint scaffold splits, but they are not orthogonal signals. Their
held-out predictions are highly correlated. Therefore the QSAR fallback should
be described as an independent-model validation proxy, not as a genuinely
independent biological oracle. This weakens the strongest version of the
validation-driven story and is one reason the docking branch remains important
for any future positive claim.

## RF/SVM/Vina triangulation

Motivation:

The QSAR fallback preserves model-level separation between the training reward
and validation feedback, but it is still QSAR-on-QSAR. To connect this fallback
to the docking branch, score the already docked `DRD2_7DFP` matched calibration
molecules with the RF reward model and SVM validation model, then compare all
three signals.

Artifacts:

- `scripts/qsar_vina_triangulation.py`
- `scripts/remote_qsar_vina_triangulation.py`
- `qsar_vina_triangulation_DRD2_7DFP.json`
- `qsar_vina_triangulation_DRD2_7DFP_scored.csv`
- `qsar_vina_triangulation_RF_vs_SVM.svg`
- `qsar_vina_triangulation_RF_vs_Vina.svg`
- `qsar_vina_triangulation_SVM_vs_Vina.svg`

Scored calibration set:

- usable molecules: `127`
- active: `63`
- decoy: `64`
- Vina direction: `docking_reward = -vina_score`, so higher is better.

Triangulation matrix:

| Pair | Pearson, 95% bootstrap CI | Spearman, 95% bootstrap CI | Top-10% overlap |
|---|---:|---:|---:|
| RF reward vs SVM validation | 0.698 [0.601, 0.781] | 0.625 [0.536, 0.690] | 0/13, Jaccard 0.000 |
| RF reward vs Vina docking reward | 0.105 [-0.046, 0.252] | 0.036 [-0.121, 0.192] | 0/13, Jaccard 0.000 |
| SVM validation vs Vina docking reward | 0.106 [-0.073, 0.268] | 0.013 [-0.168, 0.197] | 1/13, Jaccard 0.040 |

Interpretation:

- On the docking calibration molecules, RF and SVM remain related but are less
  aligned than on the scaffold-held-out QSAR test set.
- The RF/SVM `top-10%` overlap is `0/13` even though their continuous scores are
  moderately correlated. This is not a contradiction: global rank/score
  correlation does not imply agreement on the highest-ranked generated
  candidates, which are often the molecules that matter most in molecular
  generation. This top-k disagreement supports treating the SVM validation proxy
  as more than a direct copy of the RF reward model.
- Both QSAR signals are only weakly aligned with Vina, and the bootstrap CIs for
  QSAR-vs-Vina correlations include zero.
- This supports two points at once: the QSAR fallback is not simply measuring
  the same thing as docking, but the current `DRD2_7DFP` docking calibration is
  not reliable enough to serve as the controller oracle.
- The paper should therefore frame the QSAR fallback as a pragmatic independent
  proxy, while treating docking as an important branch whose current DRD2
  calibration is insufficient for a strong biological-oracle claim.

## Positive-control sensitivity test

Motivation:

The negative controller results are only meaningful if the analysis pipeline can
detect a strong effect when one is deliberately present.

Artifact:

- `scripts/positive_control_validation_sensitivity.py`
- `scripts/positive_control_sensitivity_ladder.py`
- `positive_control_validation_sensitivity.json`
- `positive_control_sensitivity_ladder.json`
- `positive_control_real_history.jsonl`

Design:

Use a synthetic validation stream with two objectives. `Target_A` steadily
deteriorates, while `Target_B` stays approximately constant. The deterministic
controller should therefore upweight `Target_A`. Shuffled controls preserve the
same marginal validation values but break the temporal relationship.

Initial strong sanity-check result:

| Metric | Value |
|---|---:|
| Real final `Target_A` weight | 0.870 |
| Shuffled-control mean final `Target_A` weight | 0.719 |
| Delta | +0.151 |
| z-score vs shuffled controls | 6.677 |
| Empirical one-sided p-value | 0.0099 |
| Positive control passed | true |

Interpretation:

The controller/analysis pipeline is sensitive to a deliberately strong
validation-weight relationship. Therefore the DRD2 negative results should not
be dismissed simply as a totally insensitive detection pipeline. This positive
control does not prove molecular-generation efficacy; it only validates the
diagnostic machinery.

Manuscript note:

Use the sensitivity-ladder strong result below as the main-text positive-control
number. The initial strong sanity check above used fewer shuffled controls
(`100`), so its z-score and empirical p-value differ from the ladder run
(`1000` shuffled controls). Keep the initial sanity check as provenance or
supplementary material rather than presenting both strong-control numbers as
separate main-text findings.

Sensitivity ladder:

The strong positive control alone only proves detection of a large coherent
effect. The ladder varies the frequency of validation windows whose `Target_A`
drop exceeds the controller deadband (`0.01`). This is more informative than
only shrinking a per-window slope, because the deterministic controller acts
when a drop crosses the deadband.

| Synthetic level | Detectable drop events | Real final `Target_A` weight | Shuffled mean | Delta | z-score | Empirical p | Passed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strong, every window | 19 | 0.870 | 0.720 | +0.150 | 5.657 | 0.0010 | true |
| Medium, every 2 windows | 9 | 0.711 | 0.709 | +0.002 | 0.085 | 0.6364 | false |
| Medium, every 3 windows | 6 | 0.646 | 0.700 | -0.055 | -1.987 | 0.9880 | false |
| Near-noise, every window below deadband | 0 | 0.500 | 0.699 | -0.199 | -7.163 | 1.0000 | false |

Interpretation:

- The diagnostic pipeline reliably detects a coherent strong synthetic
  validation-weight relationship.
- Sparser medium-strength signals are not reliably distinguishable from
  shuffled controls under the current deterministic action rule.
- Below-deadband near-noise signals are in the detection blind zone.
- This supports a cautious negative conclusion: the molecular experiments do
  not rule out arbitrarily tiny effects, but the observed real-vs-random deltas
  are below the regime where this setup can stably detect a meaningful
  validation-driven controller effect.

## Dynamic controller run

Command shape:

```bash
export REINVENT_CONTROLLER_COMPONENTS="DRD2_activity,QED,Molecular weight"
export REINVENT_CONTROLLER_INIT_WEIGHTS="0.34,0.33,0.33"
export REINVENT_CONTROLLER_SEED=17
export REINVENT_CONTROLLER_VALIDATE_EVERY=20
export REINVENT_CONTROLLER_VALIDATION_BATCH=64
export REINVENT_CONTROLLER_DEVICE=cpu
export REINVENT_CONTROLLER_HISTORY_PATH=controller_history_drd2_qsar_seed17.jsonl
export REINVENT_VALIDATION_COMMAND="/root/miniconda3/envs/reinvent4/bin/python scripts/validate_qsar.py --input {input} --output {output} --model qsar_models/DRD2/validation_svm.pkl --component-names 'DRD2_activity,QED,Molecular weight'"

python 03_dynamic_weight_controller.py \
  -l drd2_qsar_dynamic_seed17.log \
  07_drd2_qsar_pilot.toml
```

Run result:

- exit code: `0`
- steps: `300`
- validation updates: `15`
- validation interval: every `20` steps
- validation batch size: `64`
- peak main memory: `2137.7 MiB`
- peak reserved GPU memory: `1492.0 MiB`

Controller trajectory:

- initial weights: `DRD2_activity=0.34`, `QED=0.33`, `Molecular weight=0.33`
- final weights: `DRD2_activity=0.448`, `QED=0.375`, `Molecular weight=0.177`
- weight ranges:
  - `DRD2_activity`: `0.305` to `0.459`
  - `QED`: `0.330` to `0.473`
  - `Molecular weight`: `0.166` to `0.346`
- initial validation scores:
  - `DRD2_activity=0.309`
  - `QED=0.563`
  - `Molecular weight=0.953`
- final validation scores:
  - `DRD2_activity=0.297`
  - `QED=0.724`
  - `Molecular weight=0.938`
- mean controller reward: `-0.00484`

Interpretation: the controller loop is technically working and produces
nontrivial, bounded weight changes. In this single seed, it shifted weight away
from molecular weight and toward activity/QED. Final sampled validation improved
QED strongly, kept molecular weight high, but did not improve validation
activity relative to the first validation batch.

## Controller trajectory diagnosis

Command:

```bash
python scripts/analyze_controller_history.py \
  --history controller_history_drd2_qsar_seed17.jsonl \
  --output-prefix controller_history_drd2_qsar_seed17
```

Trajectory diagnostics:

| Component | Overall weight slope/update | Overall R2 | Late weight slope/update | Late weight range | Early mean validation | Middle mean validation | Late mean validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRD2_activity | +0.00824 | 0.684 | +0.02220 | 0.098 | 0.305 | 0.293 | 0.291 |
| Molecular weight | -0.00832 | 0.626 | +0.00445 | 0.045 | 0.913 | 0.922 | 0.931 |
| QED | +0.00008 | 0.000 | -0.02666 | 0.143 | 0.617 | 0.673 | 0.694 |

Additional diagnostics:

- controller reward mean: `-0.00484`
- controller reward positive fraction: `0.333`
- DRD2_activity first-half weight total variation: `0.178`
- DRD2_activity second-half weight total variation: `0.171`
- QED first-half weight total variation: `0.213`
- QED second-half weight total variation: `0.309`

Interpretation: this is not a pure flat random walk because activity weight
trends upward and molecular-weight weight trends downward with moderate linear
fit. However, it also does not show convincing convergence within 15 validation
updates: second-half variation remains similar to first-half variation for
activity, and QED is more variable in the second half. Therefore the better
final-step validation activity should be treated as a noisy endpoint signal, not
as evidence that the controller had reached a stable superior policy. The
primary comparison should use late-window performance over more seeds.

## Random-validation control

Command shape:

```bash
export REINVENT_CONTROLLER_COMPONENTS="DRD2_activity,QED,Molecular weight"
export REINVENT_CONTROLLER_INIT_WEIGHTS="0.34,0.33,0.33"
export REINVENT_CONTROLLER_SEED=17
export REINVENT_CONTROLLER_VALIDATE_EVERY=20
export REINVENT_CONTROLLER_VALIDATION_BATCH=64
export REINVENT_CONTROLLER_DEVICE=cpu
export REINVENT_CONTROLLER_HISTORY_PATH=controller_history_drd2_qsar_random_seed17.jsonl
export REINVENT_VALIDATION_COMMAND="/root/miniconda3/envs/reinvent4/bin/python scripts/validate_random_control.py --input {input} --output {output} --history controller_history_drd2_qsar_seed17.jsonl --component-names 'DRD2_activity,QED,Molecular weight' --mode shuffle_history --seed 1701"

python 03_dynamic_weight_controller.py \
  -l drd2_qsar_random_validation_seed17.log \
  10_drd2_qsar_random_validation.toml
```

This control uses shuffled historical validation-score records. It preserves
the scale and marginal variation of the real validation signal but removes the
relationship between the current validation-only molecules and the controller
feedback.

Run result:

- exit code: `0`
- steps: `300`
- validation updates: `15`
- peak main memory: `2137.6 MiB`
- peak reserved GPU memory: `1008.0 MiB`

Random-control trajectory:

| Component | Real final weight | Random-control final weight | Real overall slope/update | Random overall slope/update |
|---|---:|---:|---:|---:|
| DRD2_activity | 0.448 | 0.473 | +0.00824 | +0.01063 |
| Molecular weight | 0.177 | 0.187 | -0.00832 | -0.00724 |
| QED | 0.375 | 0.340 | +0.00008 | -0.00339 |

Random-control trajectory diagnostics:

- DRD2_activity overall weight R2: `0.821`
- Molecular-weight overall weight R2: `0.556`
- QED overall weight R2: `0.169`
- DRD2_activity late weight range: `0.095`
- QED late weight range: `0.138`
- mean controller reward: `-0.0104`
- reward positive fraction: `0.333`

Interpretation: the random-validation control produces a weight trajectory that
is qualitatively similar to the real dynamic run: activity weight increases,
molecular-weight weight decreases, and late-window variation remains present.
This weakens the claim that the observed seed17 weight trend is specifically
driven by meaningful validation feedback. It does not invalidate the pipeline,
but it says the current controller update rule is not yet sufficiently
signal-sensitive to support a method claim.

## Constant-validation structural check

The next diagnostic isolated the controller from REINVENT, QSAR scoring, and
validation-batch sampling. We repeatedly called `DynamicWeightController.step()`
with fixed validation scores for 15 validation updates.

Command shape:

```bash
python scripts/simulate_controller_constant.py \
  --controller 03_dynamic_weight_controller.py \
  --output controller_constant_zero_after_fix.json
```

Before fixing the update rule, constant validation still caused substantial
weight drift:

| Case | Final DRD2_activity | Final QED | Final Molecular weight | Interpretation |
|---|---:|---:|---:|---|
| all-zero constant scores | 0.459 | 0.330 | 0.211 | drift despite zero reward |
| representative constant scores | 0.473 | 0.340 | 0.187 | same activity-up/MW-down pattern |
| all-zero, entropy disabled | 0.497 | 0.307 | 0.196 | entropy was not the root cause |
| all-zero, action step 0.05 | 0.380 | 0.333 | 0.287 | smaller drift, but still drift |

Root cause: `step()` sampled and applied a new action at every validation
interval even when there was no previous validation baseline or when
`controller_reward == 0`. Therefore exploration actions could move weights even
without any validation signal. This explains why the random-validation control
could mimic the real dynamic run's broad weight trend.

Fix applied in `03_dynamic_weight_controller.py`:

- add `reward_deadband`;
- do not apply a new action when there is no previous validation score;
- do not apply a new action when `abs(controller_reward) <= reward_deadband`;
- record `action_applied` in controller history.

After the fix:

| Case | Final DRD2_activity | Final QED | Final Molecular weight | Actions applied |
|---|---:|---:|---:|---:|
| all-zero constant scores | 0.340 | 0.330 | 0.330 | 0/15 |
| representative constant scores | 0.340 | 0.330 | 0.330 | 0/15 |

Interpretation: the structural no-signal drift is fixed. This does not prove
the controller is now effective; it only restores the necessary property that
weights should not move when validation feedback contains no change.

## Static equal-weight run

Command:

```bash
reinvent -l drd2_qsar_static_equal_seed17.log 08_drd2_qsar_static_equal.toml
```

Run result:

- exit code: `0`
- steps: `300`
- peak main memory: `2166.4 MiB`
- peak reserved GPU memory: `1276.0 MiB`

## Posthoc independent validation comparison

Command:

```bash
python scripts/analyze_qsar_pilot.py \
  --model qsar_models/DRD2/validation_svm.pkl \
  --run dynamic=drd2_qsar_pilot_1.csv \
  --run static_equal=drd2_qsar_static_equal_1.csv \
  --output qsar_models/DRD2/drd2_qsar_pilot_posthoc.json
```

Posthoc metrics use the independent validation SVM, not the RF training reward.
The CSV files contain REINVENT's reported valid sampled rows.

| Metric | Dynamic | Static equal | Random validation |
|---|---:|---:|---:|
| Rows analyzed | 18,941 | 18,945 | 18,944 |
| Unique SMILES | 18,920 | 18,926 | 18,920 |
| Unique scaffolds | 15,975 | 15,910 | 15,806 |
| All mean validation activity | 0.301 | 0.309 | 0.298 |
| Last-50-step mean validation activity (primary pilot metric) | 0.301 | 0.301 | 0.299 |
| Final-step mean validation activity | 0.353 | 0.288 | 0.325 |
| Top-10 mean validation activity | 0.940 | 0.939 | 0.919 |
| Top-100 mean validation activity | 0.850 | 0.856 | 0.829 |
| All mean training score | 0.523 | 0.551 | 0.515 |
| Last-50-step mean training score | 0.521 | 0.576 | 0.503 |
| Final-step mean training score | 0.536 | 0.554 | 0.478 |
| Final-step mean QED | 0.659 | 0.707 | 0.649 |
| Final-step mean MW score | 0.953 | 0.930 | 0.912 |

Interpretation: this single seed cannot support any algorithmic win/loss claim.
Under the pre-specified pilot primary metric, late-window validation activity is
essentially tied between dynamic and static, and the random-validation control is
only slightly lower. Static equal weights are better on mean training score and
slightly better on all-row validation activity. Dynamic is better on final-step
validation activity, but random validation also has a stronger final-step value
than static, reinforcing that final-step is a noisy diagnostic rather than a
decision metric. Top-10 activity is tied between dynamic and static, while
random validation is lower. This is useful as a healthy pilot and a controller
diagnostic, but it is not evidence of dynamic superiority.

## Next steps

1. Rerun dynamic and random-validation controls with only the structural
   no-signal fix applied, keeping the old hyperparameters. This isolates whether
   the fix alone separates real validation from random validation.
2. If the two controls remain similar, change one variable at a time. Suggested
   order:
   - `REINVENT_CONTROLLER_STANDARDIZE_ADVANTAGE=1`
   - then `REINVENT_CONTROLLER_ACTION_STEP=0.05`
   - then `REINVENT_CONTROLLER_VALIDATE_EVERY=30`
   - then test `REINVENT_CONTROLLER_VALIDATION_BATCH=128` only if wall-clock is
     acceptable.
3. After dynamic clearly separates from random validation, run at least two more
   DRD2 seeds for dynamic and static equal-weight arms, reserving budget for 5+
   seeds if this comparison becomes central.
4. Keep reporting random-validation controls alongside the proposed method; this
   is now essential for demonstrating true validation sensitivity.

## Overnight diagnostic queue

Queued after the fixed-controller real/random rerun:

| Label | Change from fixed controller | Purpose |
|---|---|---|
| `stdadv` | `REINVENT_CONTROLLER_STANDARDIZE_ADVANTAGE=1` | Test component-scale/variance dominance |
| `step005` | `REINVENT_CONTROLLER_ACTION_STEP=0.05` | Test whether action size is too exploratory |
| `ve30` | `REINVENT_CONTROLLER_VALIDATE_EVERY=30` | Test validation-frequency noise sensitivity |
| `ctrlseed23` | `REINVENT_CONTROLLER_SEED=23` | Stochastic controller replicate |
| `static_equal_rep2` | static equal rerun | Rough baseline variance check |

Each dynamic diagnostic except `static_equal_rep2` is paired with a
random-validation control using the same setting and a shuffled history from
that setting's real-validation run. These are mechanism diagnostics, not final
benchmark claims. The key question is whether real validation and random
validation produce meaningfully different weight trajectories and late-window
validation metrics.

## Fixed rerun and overnight results

Both remote queues completed successfully:

- fixed-controller rerun completed at `2026-07-05T01:15:11+08:00`;
- overnight controller diagnostic queue completed at
  `2026-07-05T05:58:33+08:00`.

Primary pilot metric:

| Run | Last-50 validation activity | All-row validation activity | Final-step validation activity | Top-10 validation activity |
|---|---:|---:|---:|---:|
| `dynamic_fixed` | 0.290 | 0.295 | 0.312 | 0.926 |
| `random_fixed` | 0.322 | 0.311 | 0.319 | 0.922 |
| `static_equal_seed17` | 0.301 | 0.309 | 0.288 | 0.939 |
| `static_equal_rep2` | 0.310 | 0.311 | 0.328 | 0.938 |
| `dynamic_stdadv` | 0.295 | 0.299 | 0.290 | 0.950 |
| `random_stdadv` | 0.315 | 0.308 | 0.348 | 0.941 |
| `dynamic_step005` | 0.308 | 0.308 | 0.331 | 0.931 |
| `random_step005` | 0.312 | 0.301 | 0.289 | 0.931 |
| `dynamic_ve30` | 0.304 | 0.301 | 0.313 | 0.935 |
| `random_ve30` | 0.300 | 0.300 | 0.284 | 0.936 |
| `dynamic_ctrlseed23` | 0.311 | 0.306 | 0.321 | 0.939 |
| `random_ctrlseed23` | 0.308 | 0.307 | 0.305 | 0.932 |

Controller-action diagnosis:

| Pair | Real minus random, last-50 validation | Exact action sequence matches |
|---|---:|---:|
| `fixed` | -0.032 | 14/15 |
| `stdadv` | -0.020 | 12/15 |
| `step005` | -0.004 | 4/15 |
| `ve30` | +0.004 | 4/10 |
| `ctrlseed23` | +0.003 | 2/15 |

Interpretation:

- The no-signal bug fix is necessary and successful, but it is not sufficient.
- In the fixed real-vs-random rerun, the random-validation control outperforms
  the real-validation controller on the primary pilot metric.
- The fixed real and random controllers used almost the same action sequence
  (`14/15` exact action matches), which means the controller is still dominated
  by stochastic policy sampling under the current update rule.
- `ACTION_STEP=0.05` stabilizes metrics and reduces real/random action-sequence
  identity, but it still does not produce clear validation-driven superiority.
- `VALIDATE_EVERY=30` and controller seed 23 give tiny positive real-minus-random
  deltas, but the effects are too small to support a mechanism claim.
- Static equal weights remain competitive across the two static replicates.

Decision: do not scale this controller to GSK3B/JNK3 yet. The next work should
target controller signal sensitivity and credit assignment, not more seeds or
more targets.

## Reward-deadband diagnostic

After inspecting the fixed-controller action sequence, the main failure mode
looked like excessive response to small, noisy reward changes. In
`dynamic_fixed`, activity actions had a positive net direction:

| Run | Net activity action | Net QED action | Net MW action | Final activity weight | Final QED weight | Final MW weight |
|---|---:|---:|---:|---:|---:|---:|
| `dynamic_fixed` | +6 | +2 | 0 | 0.519 | 0.276 | 0.205 |
| `random_fixed` | +4 | +2 | 0 | 0.444 | 0.319 | 0.237 |

Because weights are normalized, this default activity/QED-up tendency pushed MW
down even when MW actions were neutral. This explains why a validation-insensitive
controller can underperform static equal weights.

Implementation change:

- `reward_deadband` now gates both policy update and new action sampling.
- If `abs(controller_reward) <= reward_deadband`, the controller does not update
  the policy and does not sample or apply a new action.
- The history now records `policy_updated`.

Regression check:

| Case | Final DRD2_activity | Final QED | Final Molecular weight | Actions applied |
|---|---:|---:|---:|---:|
| all-zero constant scores | 0.340 | 0.330 | 0.330 | 0/15 |
| representative constant scores | 0.340 | 0.330 | 0.330 | 0/15 |

Diagnostic run: `REINVENT_CONTROLLER_REWARD_DEADBAND=0.02`, all other
hyperparameters kept as in `dynamic_fixed`.

| Run | Last-50 validation activity | All-row validation activity | Final-step validation activity | Top-10 validation activity | Actions applied |
|---|---:|---:|---:|---:|---:|
| `dynamic_deadband002` | 0.275 | 0.297 | 0.296 | 0.933 | 6/15 |
| `random_deadband002` | 0.286 | 0.301 | 0.269 | 0.907 | 3/15 |
| `dynamic_fixed` | 0.290 | 0.295 | 0.312 | 0.926 | 14/15 |
| `random_fixed` | 0.322 | 0.311 | 0.319 | 0.922 | 14/15 |
| `static_equal_seed17` | 0.301 | 0.309 | 0.288 | 0.939 | n/a |
| `static_equal_rep2` | 0.310 | 0.311 | 0.328 | 0.938 | n/a |

Action sequence comparison:

| Pair | Exact action sequence matches |
|---|---:|
| `fixed` real vs random | 14/15 |
| `deadband002` real vs random | 10/15 |

Interpretation:

- The deadband does what it was meant to do mechanically: action frequency drops
  from 14/15 to 6/15 for real validation and 3/15 for random validation.
- It also reduces exact action-sequence identity between real and random runs.
- However, it does not make real validation outperform random validation on the
  primary metric. `dynamic_deadband002` remains below `random_deadband002` and
  below both static equal-weight runs.
- Therefore the dominant issue is no longer just overreaction to tiny rewards.
  The controller still lacks reliable credit assignment from validation changes
  to objective-specific weight changes.

Decision: do not pursue more deadband tuning as the next main step. The next
mechanistic change should be per-objective credit assignment or a simpler
deterministic validation-response controller before returning to policy-gradient
experiments.

## Frozen-initial-policy diagnostic

Diagnostic artifact: `frozen_policy_diagnostic_drd2_seed17.json`, generated by
`scripts/diagnose_frozen_policy.py`.

Purpose: freeze the policy at initialization, perform no gradient update and no
weight update, and feed the real `dynamic_fixed` validation-score trajectory
through the initialized policy. This isolates initialization/action-sampling
effects from learned validation response.

Key result:

| Comparison | Exact action-sequence matches | Net activity action | Net QED action | Net MW action |
|---|---:|---:|---:|---:|
| `dynamic_fixed` real run | n/a | +6 | +2 | 0 |
| frozen seed17 policy | 13/14 vs `dynamic_fixed` | +5 | +2 | 0 |

Across five independent policy initializations (`17,23,101,202,303`), the
aggregate frozen action distribution was close to neutral for activity and QED:

| Component | P(-1) | P(0) | P(+1) | Mean sampled action |
|---|---:|---:|---:|---:|
| DRD2_activity | 0.335 | 0.329 | 0.336 | +0.001 |
| QED | 0.325 | 0.352 | 0.323 | -0.002 |
| Molecular weight | 0.365 | 0.297 | 0.337 | -0.028 |

Seed17 alone had slightly negative action probabilities for all three
components, but its single sampled trajectory still produced the same
activity/QED-up net direction as the actual RL run. Therefore the observed
`dynamic_fixed` action pattern is not evidence that the controller learned to
respond to validation content. It can be almost fully reproduced by the frozen
initial policy's sampling trajectory.

Interpretation:

- The dominant failure is not a deterministic architecture-wide positive bias
  toward activity/QED.
- The policy-gradient update is too weak or too poorly credited to move the
  action sequence away from the initial stochastic trajectory.
- More seeds or more targets would mainly measure this stochastic controller
  behavior, not a validation-driven mechanism.

Decision: add a deterministic validation-response sanity baseline before deeper
policy-gradient redesign. If the deterministic rule can beat static and random
controls, the validation signal is useful and the RL credit-assignment mechanism
is the likely failure point. If it also fails, the current DRD2 QSAR validation
signal may be too noisy or too weak for reactive dynamic weighting.

## Deterministic validation-response baseline

Implementation: `REINVENT_CONTROLLER_MODE=deterministic` in
`03_dynamic_weight_controller.py`.

Rule: at each independent validation point, if a component's validation score
drops by more than `REINVENT_CONTROLLER_REWARD_DEADBAND=0.02` relative to the
previous validation window, increase that component's weight by the same
multiplicative `action_step` used by the RL controller. Otherwise hold the
component weight. This removes policy-gradient sampling and tests whether a
simple validation-reactive mechanism has any useful signal.

Run artifacts:

- `controller_history_drd2_qsar_deterministic_db002_seed17.jsonl`
- `controller_history_drd2_qsar_random_deterministic_db002_seed17.jsonl`
- `qsar_models_DRD2_drd2_qsar_deterministic_db002_posthoc.json`

Primary and diagnostic metrics:

| Run | Last-50 validation activity | All-row validation activity | Final-step validation activity | Top-10 validation activity |
|---|---:|---:|---:|---:|
| `deterministic_db002` | 0.308 | 0.311 | 0.309 | 0.926 |
| `random_deterministic_db002` | 0.301 | 0.306 | 0.315 | 0.940 |
| `dynamic_fixed` | 0.290 | 0.295 | 0.312 | 0.926 |
| `random_fixed` | 0.322 | 0.311 | 0.319 | 0.922 |
| `dynamic_deadband002` | 0.275 | 0.297 | 0.296 | 0.933 |
| `random_deadband002` | 0.286 | 0.301 | 0.269 | 0.907 |
| `static_equal_seed17` | 0.301 | 0.309 | 0.288 | 0.939 |
| `static_equal_rep2` | 0.310 | 0.311 | 0.328 | 0.938 |

Action summaries:

| Run | Actions applied | Net activity action | Net QED action | Net MW action | Final activity weight | Final QED weight | Final MW weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| `deterministic_db002` | 8/15 | +4 | +3 | +2 | 0.391 | 0.327 | 0.281 |
| `random_deterministic_db002` | 9/15 | +3 | +5 | +5 | 0.276 | 0.362 | 0.362 |

Interpretation:

- The deterministic rule is better than the RL controller variants on the
  primary `last50_mean_validation_activity` metric.
- It also beats its own shuffled-validation control on the primary metric
  (`0.308` vs `0.301`), but the difference is only `0.007`.
- The two static equal-weight repeats differ by `0.009` on the same primary
  metric (`0.301` vs `0.310`). Therefore the observed deterministic real-vs-
  random gap is smaller than the seed-to-seed noise already observed for an
  unchanged static method.
- The current deterministic result is therefore within the empirical noise
  floor of this pilot. It should not yet be interpreted as evidence of a
  validation-driven signal.
- The deterministic rule also does not beat both static equal-weight replicates,
  and it loses to the random deterministic control on final-step and top-10
  validation activity.
- Therefore the current conclusion is not "dynamic weighting works", but rather:
  the deterministic real-vs-random difference is directionally interesting but
  statistically unresolved. More repeats are needed before deciding whether
  validation-reactive dynamic weighting has any practical value in this DRD2
  setting.

Next decision experiment:

- Run deterministic real-vs-random for at least five total repeats.
- Treat the existing `seed17` pair as repeat 1.
- Add four more repeats, labelled `23, 101, 202, 303`.
- Primary criterion: mean real-random difference in
  `last50_mean_validation_activity`.
- A convincing signal should be directionally consistent across repeats and
  should exceed roughly `0.01`, the observed same-method noise floor from the
  static repeats. If the differences flip sign or remain within this range, the
  DRD2 deterministic controller should be considered unresolved or negative.
- The TOML configs do not expose a global REINVENT seed; these are best treated
  as stochastic repeats rather than fully controlled seeds. `REINVENT_CONTROLLER_SEED`
  and the shuffled-validation seed are still recorded for traceability.

Execution status:

- `scripts/remote_deterministic_multiseed.py` was added to run the four
  additional repeats sequentially on the server.
- The queue was started on 2026-07-05 14:01:20 +08:00.
- It runs repeat labels `23, 101, 202, 303`, then combines them with the already
  completed repeat `17` into
  `qsar_models_DRD2_drd2_qsar_deterministic_multiseed_posthoc.json`.
- The queue was paused on 2026-07-05 14:26:07 +08:00 because the work will move
  to a multi-GPU server. The interrupted `seed23` real run had only 11/15
  validation updates and should be treated as incomplete, not as an analysis
  replicate.

Multi-GPU repeat results:

- `scripts/remote_deterministic_multigpu.py` was added to rerun the same
  deterministic real-vs-random repeats in parallel on the upgraded four-GPU
  server.
- The multi-GPU queue started on 2026-07-05 14:30:32 +08:00 and completed on
  2026-07-05 15:36:00 +08:00.
- All five paired repeats have complete controller histories: repeat `17` from
  the earlier single-pair run plus fresh multi-GPU repeats `23, 101, 202, 303`.
- The old interrupted sequential `seed23` partial run is excluded.
- Downloaded analysis artifacts:
  `qsar_models_DRD2_drd2_qsar_deterministic_multigpu_posthoc.json` and
  `deterministic_multigpu_pair_summary.json`.

Paired primary metric, `last50_mean_validation_activity`:

| Repeat | Real | Random validation control | Real - random |
|---:|---:|---:|---:|
| 17 | 0.308 | 0.301 | +0.007 |
| 23 | 0.310 | 0.308 | +0.002 |
| 101 | 0.303 | 0.309 | -0.006 |
| 202 | 0.308 | 0.293 | +0.015 |
| 303 | 0.307 | 0.298 | +0.009 |

Aggregate:

- Mean primary delta: `+0.0054`.
- Median primary delta: `+0.0071`.
- Standard deviation of primary delta: `0.0081`.
- Direction: `4/5` repeats positive.
- Approximate two-sided sign-test p-value for `4/5` positives: `0.375`.
- Static equal-weight repeat gap observed earlier: `0.0088`.

Interpretation:

- The deterministic validation-response controller shows weak positive
  directionality, but the mean primary delta is smaller than the empirical
  static-repeat noise floor (`0.0054 < 0.0088`) and below the pre-specified
  practical signal threshold of roughly `0.01`.
- The `4/5` positive direction is encouraging enough to keep as a diagnostic
  observation, but not strong enough to claim that validation-driven dynamic
  weighting is working in this DRD2 setting.
- The final-step metric is also positive in `4/5` repeats, but it is a
  secondary diagnostic metric with high variance and should not be used to
  override the primary metric.
- Top-10 validation activity does not show a useful advantage.

Decision:

The current DRD2/QED/MW deterministic controller result should be treated as
unresolved-to-negative, not as a successful dynamic-weighting result. Do not
scale this exact controller to GSK3B/JNK3 as a formal positive ablation yet.
The next useful experiment is either a stronger-conflict objective setting
where static equal weighting is less likely to be near-optimal, or a controller
redesign with explicit per-objective credit assignment before target expansion.

## Activity-vs-SA follow-up

Rationale:

The DRD2/QED/MW setting may have too weak a target conflict for reactive dynamic
weighting. Before redesigning the controller, run a stronger-conflict setting:
DRD2 activity versus synthetic accessibility.

Implementation:

- Training reward components:
  - `DRD2_activity`: existing RandomForest QSAR reward.
  - `SA_score`: cheap RDKit complexity proxy, exposed through
    `scripts/score_rdkit_property_json.py`.
- Independent validation components:
  - `DRD2_activity`: existing independent SVM QSAR validator.
  - `SA_score`: exact RDKit Contrib SA score converted to a 0..1 reward,
    where larger is synthetically easier.
- Controller mode: deterministic validation-response controller.
- Primary metric: `last50_mean_validation_joint`, the geometric mean of
  independent validation activity and exact validation SA.
- Paired repeats: `17, 23, 101, 202, 303`.
- Noise baseline: two static equal-weight repeats in the same activity-vs-SA
  setting.

Artifacts added:

- `13_drd2_activity_sa_dynamic.toml`
- `14_drd2_activity_sa_random_validation.toml`
- `15_drd2_activity_sa_static_equal.toml`
- `scripts/sa_score_utils.py`
- `scripts/score_rdkit_property_json.py`
- `scripts/analyze_activity_sa_pilot.py`
- `scripts/remote_activity_sa_multigpu.py`
- `scripts/remote_activity_sa_fastgate.py`
- `scripts/remote_activity_sa_fastgate_extend.py`

Execution status:

- Exact SA smoke test passed on the server: `SA exact smoke score: 0.8996`.
- At the initial activity-vs-SA launch, the server exposed two RTX 4080 GPUs,
  not four. Later runs used the detected GPU count rather than assuming a fixed
  device count.
- The first launch failed before training because the SA proxy used a
  non-portable RDKit API (`Mol.GetStereoAtoms`). This was fixed by switching to
  `Chem.FindMolChiralCenters`.
- The restarted `v2` queue began on 2026-07-05 16:04:50 +08:00.
- Current queue design uses the detected GPU count and runs:
  - five deterministic real-vs-random paired repeats;
  - two static equal-weight repeats;
  - posthoc analysis to
    `qsar_models_DRD2_drd2_activity_sa_v2_multigpu_posthoc.json`.
- This full queue was stopped before completion because it was too expensive
  for a still-uncertain follow-up setting. At stop time, the first two real
  runs had produced 10 validation records each and no random/static comparison
  had completed.

Fast gate:

- Added `scripts/remote_activity_sa_fastgate.py`.
- Added `scripts/remote_activity_sa_fastgate_extend.py` to bring the fast gate
  to five paired repeats without resuming the expensive 300-step queue.
- Purpose: decide quickly whether the activity-vs-SA setting is worth a full
  repeat run.
- Design:
  - five deterministic real-vs-random paired repeats: `17, 23, 101, 202, 303`;
  - two static equal-weight runs;
  - `max_steps=120`, `min_steps=20`, `batch_size=32`;
  - `validate_every=15`, `validation_batch=32`;
  - same primary metric:
    `last50_mean_validation_joint = sqrt(validation_activity * validation_SA)`.
- The fast gate was launched on 2026-07-05 16:30:28 +08:00.
- Remote smoke test passed: `SA smoke exact=0.8996 proxy=0.9145`.
- The fast gate completed on 2026-07-05 16:59:17 +08:00.
- The extension completed on 2026-07-05 18:26:58 +08:00.
- All ten controller histories completed `8/8` validation records.
- Posthoc artifacts:
  `qsar_models_DRD2_drd2_activity_sa_fastgate_posthoc.json`.
  `qsar_models_DRD2_drd2_activity_sa_fastgate_5repeat_posthoc.json`.

Fast gate 5-repeat primary results:

| Seed | Real last-50 joint | Random-control last-50 joint | Real-random delta |
|---:|---:|---:|---:|
| 17 | 0.489151 | 0.494644 | -0.005493 |
| 23 | 0.498324 | 0.497400 | +0.000924 |
| 101 | 0.501981 | 0.506187 | -0.004206 |
| 202 | 0.491498 | 0.489595 | +0.001903 |
| 303 | 0.499739 | 0.496547 | +0.003192 |

Aggregate:

- Mean paired primary delta: `-0.0007`.
- Median paired primary delta: `+0.0009`.
- Standard deviation of paired delta: `0.0039`.
- Direction: `3/5` repeats positive.
- Exact sign-test p-value: two-sided `1.0`, one-sided positive `0.5`.
- Static equal-weight repeat gap in the same fast-gate setting: `+0.0128`
  (`0.4890` vs `0.5018`).

Interpretation:

- The activity-vs-SA fast gate does not show a usable validation-driven signal
  after being extended to five paired repeats.
- The mean real-vs-random delta is slightly negative on the pre-specified
  primary metric.
- The observed real-vs-random effect size is far smaller than the two-run static
  repeat gap in the same fast-gate setting.
- A `3/5` positive direction is statistically uninformative and should not be
  treated as evidence of a controller effect.

Decision:

Do not resume the full activity-vs-SA repeat queue. The stronger-conflict
activity-vs-SA setting did not rescue the current deterministic controller after
five paired fast-gate repeats. The next useful work is not more repeats of this
controller, but either a task/oracle with a larger expected dynamic-weighting
ceiling or a very small offline controller redesign tested against logged
random-validation controls before any new expensive generation queue.

## Repeat-level uncertainty and power

Motivation:

The repeated runs are paired by seed, and molecules within a run are not
independent samples. Therefore uncertainty must be estimated over paired
repeats, not over generated molecules.

Artifact:

- `scripts/controller_effect_statistics.py`
- `controller_effect_statistics.json`

Paired repeat-level bootstrap:

| Setting | Primary metric | Mean real-random delta | Paired-delta SD | 95% bootstrap CI | Direction | Sign-test p |
|---|---|---:|---:|---:|---:|---:|
| DRD2/QED/MW | `last50_mean_validation_activity` | +0.0054 | 0.0081 | [-0.0014, +0.0113] | 4/5 positive | 0.375 |
| Activity-vs-SA | `last50_mean_validation_joint` | -0.0007 | 0.0039 | [-0.0038, +0.0022] | 3/5 positive | 1.000 |

Power analysis:

Monte Carlo paired t-test power was estimated with `n=5`, using the observed
paired-delta SD for each setting.

| Setting | Assumed true effect | Two-sided power | One-sided positive power |
|---|---:|---:|---:|
| DRD2/QED/MW | observed `0.0054` | 0.212 | 0.345 |
| DRD2/QED/MW | static-repeat gap `0.0088` | 0.457 | 0.637 |
| DRD2/QED/MW | practical `0.0100` | 0.558 | 0.733 |
| DRD2/QED/MW | practical `0.0150` | 0.867 | 0.957 |
| Activity-vs-SA | observed `0.0007` | 0.063 | 0.101 |
| Activity-vs-SA | small `0.0050` | 0.592 | 0.767 |
| Activity-vs-SA | practical `0.0100` | 0.987 | 0.999 |
| Activity-vs-SA | practical `0.0150` | 1.000 | 1.000 |

Interpretation:

- With five paired repeats, the DRD2/QED/MW experiment has low power for very
  small effects around `0.005`, moderate power around the static-repeat noise
  floor, and high power for larger practical effects around `0.015`.
- Activity-vs-SA has lower observed paired-delta variance in the fast gate, so a
  practical effect around `0.010` would likely have been detected. The observed
  effect is instead centered near zero.
- The appropriate conclusion is not that validation-driven control is
  universally ineffective. It is that, in the tested settings, any real effect
  is small relative to stochastic variability and below the practical effect
  scale needed to justify expensive target expansion.

## Hindsight/oracle upper-bound diagnostic

Motivation:

Before investing in a new controller mechanism, estimate whether the already
observed trajectories contain exploitable structure. This diagnostic asks: even
if an oracle could pick the best observed run at each time window after seeing
the outcomes, how much better would it be than static equal weight?

Important caveat:

This is an observed-data upper bound, not a counterfactual simulator. It cannot
tell what REINVENT would have generated under unseen weights. It only measures
how much headroom is visible in the trajectories we already ran.

Artifacts:

- `scripts/hindsight_upper_bound.py`
- `scripts/remote_hindsight_analysis.py`
- `hindsight_activity_sa_fastgate.json`
- `hindsight_activity_sa_fastgate_5repeat.json`
- `hindsight_drd2_qed_mw_multigpu.json`

Activity-vs-SA fast gate, five repeats:

| Oracle setting | Oracle window mean | Static window mean | Delta |
|---|---:|---:|---:|
| All observed runs, all windows | 0.510 | 0.495 | +0.015 |
| Real runs only, all windows | 0.506 | 0.495 | +0.011 |
| Random-control runs only, all windows | 0.507 | 0.495 | +0.012 |
| All observed runs, last 4 windows | 0.512 | 0.494 | +0.018 |
| Real runs only, last 4 windows | 0.507 | 0.494 | +0.014 |
| Random-control runs only, last 4 windows | 0.508 | 0.494 | +0.015 |

DRD2/QED/MW multigpu set:

| Oracle setting | Oracle window mean | Static window mean | Delta |
|---|---:|---:|---:|
| All observed runs, all windows | 0.539 | 0.531 | +0.008 |
| Real runs only, all windows | 0.537 | 0.531 | +0.006 |
| Random-control runs only, all windows | 0.538 | 0.531 | +0.007 |
| All observed runs, last 4 windows | 0.552 | 0.547 | +0.005 |
| Real runs only, last 4 windows | 0.550 | 0.547 | +0.003 |
| Random-control runs only, last 4 windows | 0.551 | 0.547 | +0.005 |

Molecule-pool check:

- Activity-vs-SA top-100 observed molecules:
  - real runs: `0.840`
  - random-control runs: `0.846`
  - static runs: `0.814`
- DRD2/QED/MW top-100 observed molecules:
  - real runs: `0.905`
  - random-control runs: `0.910`
  - static runs: `0.891`

Interpretation:

- The most optimistic observed-window oracle gains are small.
- Random-control trajectories provide nearly the same oracle upper bound as
  real validation trajectories.
- The molecule pools contain high-scoring candidates, but those candidates are
  not specific to meaningful validation-driven control; random-control runs
  produce comparable or better top pools.
- In the five-repeat activity-vs-SA hindsight analysis, the random-control-only
  oracle is slightly better than the real-validation-only oracle in both
  all-window and last-four-window settings. This argues against a hidden
  validation-specific opportunity in the current logged trajectories.
- Therefore the current evidence does not justify immediately building a more
  complex controller on the assumption that the validation signal contains a
  large unused dynamic-weighting opportunity.

Decision:

Pause controller redesign until there is a stronger reason to believe the task
has exploitable dynamic-weighting headroom. If continuing the project, the next
best step is not another reactive-controller variant, but either:

- a task with a clearly larger expected dynamic-weighting ceiling, such as
  selectivity or toxicity-vs-activity with independently validated objectives;
  or
- a very small offline contextual-bandit prototype tested first on logged
  trajectories with strict random-validation controls.

Speed note:

The DRD2 runs are GPU-enabled for REINVENT generation, but QSAR scoring is still
largely CPU/process-bound. The training reward uses `ExternalProcess` with
`scripts/score_qsar_json.py`, so each scoring call can repeatedly start Python,
load the pickled RF model, initialize RDKit/sklearn, and serialize scores. For
formal multi-seed ablations, the most useful speed optimization is to replace
the QSAR reward scorer with an in-process REINVENT plugin or monkeypatch-backed
cached scorer. Shorter `max_steps`, larger `validate_every`, and smaller
validation batches are useful for fast diagnostics, but they change
comparability with the 300-step pilot runs.
