# V5 end-to-end positive-control qualification protocol

## Status and separation

V4 development completed with no eligible candidate because both slopes ended
at QED weight `0.777`, below the predeclared `0.80` mechanistic threshold.
Nevertheless, all four V4 real-minus-frozen molecular QED contrasts were
positive (`+0.069` to `+0.088`) and every other gate passed. V4 is retained as
a failed development selection and is not reclassified.

V5 is an independently frozen qualification using new seeds. It uses the same
mathematically known conflict task, `QED` versus `1-QED`, and fixes the staged
validation-loss slope at `0.10` before any V5 result exists. No V5 development
ladder or parameter selection is permitted.

## Frozen design

- Qualification seeds: `5201, 5202, 5203, 5204, 5205`.
- Arms: deterministic real and compute/RNG-matched frozen weight for every seed.
- Validation schedule: slope `0.10`, starts at checkpoint 5, six decline events.
- Initial weights: `0.50,0.50`; action step `0.25`; deadband `0.02`.
- Cadence: 20; validation batch: 64; training: 300 steps, batch 64.
- Strict deterministic CUDA and validation RNG restoration are mandatory.
- Molecular endpoint: untouched RDKit QED over valid molecules from steps 251-300.

The weighted geometric reward is `q^w (1-q)^(1-w)` and has optimum `q=w`, so
QED upweighting has a known beneficial direction. The fixed Anti-QED controller
reference remains `0.5`; the generator still receives the true `1-QED` conflict.

## Qualification rule

Qualification passes only if all conditions hold:

1. every arm completes all 15 checkpoints and every RNG audit passes;
2. every real seed has at least six QED-increasing actions by checkpoint 10;
3. every real seed ends with QED weight at least `0.80`;
4. real-minus-frozen untouched tail QED is positive for all five seeds;
5. the mean paired QED gain is at least `+0.050`;
6. mean real-minus-frozen validity and uniqueness are each at least `-0.02`.

Failure blocks the main DRD2 confirmatory experiment. Passing qualifies the
end-to-end generator/controller/evaluator path and authorizes sealing the
separate main confirmatory protocol; V5 molecules are never used in that test.
