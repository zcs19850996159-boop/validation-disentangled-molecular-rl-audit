# Manuscript framing outline

## Working title options

Preferred:

```text
Auditing Validation-Driven Reward Weighting for Multi-Objective Molecular RL:
A Diagnostic Framework and Two Negative Case Studies
```

Alternative:

```text
When Does Validation-Driven Dynamic Weighting Help?
A Systematic Negative-Result Study in Multi-Objective Molecular Generation
```

Use a title that signals a diagnostic/negative-result paper. Do not frame the
paper as a positive efficacy claim for a stronger molecular-generation method.

## Core thesis

Do not claim:

```text
We propose a controller that improves molecular generation.
```

Claim instead:

```text
We propose a validation-disentangled diagnostic framework for testing whether
validation-driven dynamic weighting provides exploitable signal in
multi-objective molecular generation.
```

Main conclusion:

```text
The validation-driven weighting loop is technically functional, but it does not
provide a reliable validation-specific advantage in the tested DRD2/QED/MW and
DRD2 activity/SA settings. Any effect in these settings is small relative to
stochastic variability and below the practical effect scale needed to justify
expensive target expansion.
```

## Contribution framing

The paper's contribution is not that dynamic weighting wins. The contribution is
a reusable audit framework for validation-driven controllers:

- training reward and validation feedback are explicitly separated;
- random-validation controls test whether the controller responds to meaningful
  feedback rather than arbitrary fluctuations;
- constant-signal and frozen-policy diagnostics catch controller drift and
  action-space bias;
- positive-control sensitivity tests estimate the detection regime;
- paired repeat-level bootstrap avoids molecule-level pseudoreplication;
- power analysis distinguishes small undetectable effects from practical
  effects;
- hindsight/oracle upper-bound analysis estimates whether logged trajectories
  contain hidden exploitable dynamic-weighting headroom.

## Recommended paper structure

### 1. Introduction

Problem:

- Static reward weighting is common in multi-objective molecular generation, but
  fixed weights can be brittle and task-dependent.
- Validation-driven dynamic weighting is attractive because it could adaptively
  shift effort toward objectives whose independent validation performance is
  lagging.
- The initial hypothesis was inspired by validation-driven RL control in another
  domain, but molecular generation has different noise, oracle, and sampling
  constraints.

End the introduction with the negative result upfront:

```text
Despite enforcing a separation between training reward and validation feedback,
we did not observe a validation-specific advantage over random-validation
controls in two DRD2 settings. We therefore present the method primarily as a
diagnostic framework and report two negative case studies.
```

### 2. Related Work

Cover:

- multi-objective molecular generation with static weights and Pareto-style
  methods;
- dynamic weighting and adaptive control in adjacent ML/RL settings;
- validation-driven or held-out-feedback methods.

Key positioning:

- Distinguish this work from validation-set approaches that use the same metric
  on a held-out split.
- Emphasize that this project tests a stricter principle: controller feedback
  should come from a signal separated from the training reward, ideally from a
  different oracle.
- Verify exact bibliographic details for QADD, CPRL, Abels et al., DynaOpt, Lu
  et al., APEX, and adaptive-control citations before submission.

### 3. Method

#### 3.1 Validation-disentangled training principle

Explain why the training reward and controller feedback must be separated. If
the same signal drives both molecule optimization and controller updates, the
controller can overfit the training proxy and lose the core validation-driven
meaning.

#### 3.2 Controller implementations

Describe both implementations:

- policy-gradient controller, including the action space and the no-signal bug
  that was later diagnosed;
- deterministic validation-response controller, including deadband and
  validation-drop logic.

#### 3.3 Diagnostic framework

Make this a major method section, not an afterthought:

- random-validation control;
- constant-signal test;
- frozen-policy diagnostic;
- positive-control sensitivity ladder;
- paired-run bootstrap;
- power analysis;
- hindsight/oracle upper-bound analysis.

This is the most reusable part of the paper.

### 4. Experimental Setup

Include:

- REINVENT4 setup and monkeypatch/plugin constraint;
- DRD2 target and two objective settings:
  - DRD2/QED/MW;
  - DRD2 activity/SA;
- QSAR fallback:
  - RF training reward;
  - SVM validation proxy;
  - scaffold-disjoint splitting;
- docking side branch:
  - redocking;
  - matched active/negative calibration;
  - why docking was paused as the controller oracle.

### 5. Results

Keep the discovery order. The order is part of the argument.

#### 5.1 Docking branch self-check

Report:

- 7DFP redocking passed;
- matched active-vs-negative calibration produced AUROC `0.565`, p `0.049`;
- this is directionally non-random but below the practical AUROC gate.

Conclusion:

```text
The current DRD2_7DFP docking calibration is insufficient as a reliable
controller oracle, but docking remains an important future biological-validation
branch.
```

#### 5.2 QSAR fallback and signal triangulation

Report:

- held-out RF/SVM Pearson `0.868`, Spearman `0.857`;
- docking-subset RF/SVM Pearson `0.698`, Spearman `0.625`;
- RF/Vina and SVM/Vina correlations near zero with CIs crossing zero.

Key interpretation:

- RF and SVM are related but not identical;
- RF/SVM top-10% overlap of `0/13` on the docking subset means overall score
  correlation does not imply agreement on top candidates;
- QSAR fallback is an independent-model proxy, not a true biological oracle.

#### 5.3 Policy-gradient controller diagnostics

Tell the bug-discovery story:

- real and random-validation trajectories were nearly indistinguishable;
- constant-signal tests exposed no-signal action drift;
- frozen-policy diagnostics helped localize action bias;
- the bug was fixed, but the corrected controller still lacked a reliable
  validation-specific advantage.

#### 5.4 DRD2/QED/MW paired result

Report:

- `n=5` paired repeats;
- mean real-random delta `+0.0054`;
- 95% paired-bootstrap CI `[-0.0014, +0.0113]`;
- `4/5` positive, sign-test p `0.375`;
- static-repeat noise floor around `0.0088`.

Interpretation:

```text
Direction is weakly positive, but the effect is below the empirical noise floor
and the CI overlaps zero.
```

#### 5.5 DRD2 activity/SA stronger-conflict follow-up

Report:

- `n=5` paired fast-gate repeats;
- mean real-random delta `-0.0007`;
- 95% paired-bootstrap CI `[-0.0038, +0.0022]`;
- `3/5` positive, sign-test p `1.000`.

Interpretation:

```text
The stronger-conflict setting did not rescue the controller.
```

#### 5.6 Positive-control sensitivity ladder

Use the ladder strong-control result in the main text:

- strong continuous signal: delta `+0.150`, z `5.657`, p `0.0010`;
- sparser medium signals fail;
- near-noise below-deadband signal fails.

Put the initial strong sanity check in supplementary/provenance:

- delta `+0.151`, z `6.677`, p `0.0099`;
- differs because it used fewer shuffled controls.

#### 5.7 Power analysis

Report:

- DRD2/QED/MW has low power for effects around `0.005`, but high power for
  effects around `0.015`;
- activity/SA would likely have detected practical effects around `0.010`, but
  the observed effect is centered near zero.

Conclusion:

```text
The study does not rule out tiny effects, but it does rule against a practical
effect large enough to justify target expansion under the tested settings.
```

#### 5.8 Hindsight/oracle upper-bound

Report:

- observed-window oracle gains are modest;
- random-control-only oracle is similar to or slightly better than
  real-validation-only oracle;
- top candidate pools are not validation-specific.

Conclusion:

```text
The logged trajectories do not show a large hidden validation-specific
opportunity that the current controller merely failed to exploit.
```

### 6. Discussion

Main discussion points:

- The negative result is not simply "the method failed." It narrows the claim:
  any effect under the tested settings is small relative to stochastic
  variability.
- Oracle quality is likely a ceiling:
  - QSAR fallback is not a biological oracle;
  - docking calibration was directional but not reliable enough;
  - QSAR/Vina alignment was weak.
- The diagnostic framework is the broader contribution. Future validation-driven
  controller papers should include random-validation controls and positive
  controls before making efficacy claims.

Use constructive language when discussing related work. The point is to raise
the evidentiary standard, not to attack prior methods.

### 7. Limitations

Include:

- single target family and DRD2 focus;
- `n=5` repeats;
- 300-step and fast-gate training budgets;
- QSAR fallback is not a true biological oracle;
- GSK3B/JNK3, longer runs, stronger biological oracles, and larger repeat counts
  remain future work.

### 8. Conclusion

Short version:

```text
We present a validation-disentangled diagnostic framework for auditing dynamic
reward weighting in molecular RL. Across two DRD2 settings, real validation did
not outperform random-validation controls by a reliable or practically useful
margin. Positive controls and power analysis show that the framework can detect
large coherent effects, while the molecular effects observed here are small and
not validation-specific. These results argue for stronger diagnostic standards
before claiming validation-driven controller efficacy in molecular generation.
```

## Suggested writing order

1. Write Method 3.3 first: diagnostic framework.
2. Write Results in discovery order, using the existing report artifacts.
3. Write Experimental Setup.
4. Write Discussion.
5. Write Introduction last, after the claims are calibrated.
