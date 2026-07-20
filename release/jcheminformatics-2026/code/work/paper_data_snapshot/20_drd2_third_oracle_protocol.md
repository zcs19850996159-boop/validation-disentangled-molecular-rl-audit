# Untouched final oracle: BindingDB DRD2 graph neural network

## Frozen data source and labels

The oracle uses the article-curated BindingDB snapshot and the BindingDB
PDSP-Ki snapshot, both published on 2026-06-30.  Their downloaded SHA-256
values are respectively
`d2584d1519318d00ab5f46289da5ab3549affe732d598a5072f8777b6b3b5262` and
`8db133e55de26a6801e80ee94411cce08a9eee9b2eb9759fb58cc601e8b41847`.
No BindingDB ChEMBL-derived subset is used.

Select human dopamine D2 receptor measurements whose curated Swiss-Prot target
identifier includes `P14416`.  Use only numeric Ki, IC50, or Kd values in nM;
values qualified by `<`, `>`, `~`, or other non-numeric text are excluded.

At the canonical-SMILES level, aggregate repeated exact measurements by their
median pAffinity.  A molecule is active at median pAffinity >= 7.0 (<=100 nM)
and weak-binding/inactive at median pAffinity <= 6.0 (>=1,000 nM); the
one-log-unit interval between them is excluded.  This wider inactive boundary
is a data-feasibility amendment made before model fitting: the article-curated
DRD2 snapshot contains no median pAffinity <=5.0 molecules after target and
exact-measurement filtering.  Per-molecule measurement counts and median
values are retained for audit.

Before splitting, remove every molecule that is an exact canonical-SMILES
duplicate of any RF/SVM split molecule and remove every molecule sharing a
Bemis--Murcko scaffold with those split molecules.  The audit reports both
removal counts.

## Model and calibration

Use a fixed three-layer message-passing neural network over RDKit molecular
graphs (atom embedding, bond-type message embedding, mean graph pooling), not
Morgan fingerprints.  Split the remaining external data scaffold-disjointly
into train/validation/test sets with seed 2507.  Select the checkpoint by
validation AUROC; fit one scalar temperature on validation logits; evaluate the
held-out scaffold test set once.

The pre-launch calibration gate is scaffold-test AUROC >= 0.70, average
precision at least prevalence + 0.10, and Brier score <= 0.25.  Report AUROC,
AP, Brier, expected calibration error, class balance, checkpoint SHA-256,
RF/SVM correlations, and top-10% overlap.  A failed gate is reported as a
failed-oracle audit and cannot support practical-utility claims.

## Final use

The oracle is never used in static-grid selection, controller feedback,
controller hyperparameter selection, or placebo construction.  Its final
generated-molecule score is reported together with applicability-domain
coverage, defined by maximum ECFP4 Tanimoto similarity >=0.30 to the external
GNN training set.
