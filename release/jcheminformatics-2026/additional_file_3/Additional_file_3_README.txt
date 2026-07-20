Additional file 3: generated-molecule records
================================================

File: Additional_file_3_generated_molecule_records.csv.gz

Scope
-----
The file contains all 224,000 generated rows from frozen steps 251-300 for the
70 confirmatory runs (3,200 rows per run). It retains repeated molecules because
the confirmatory run-level means were computed over valid generated rows. Of
these rows, 221,015 are valid and they contain 217,423 distinct canonical SMILES
across all arms.

Columns
-------
run_id: Frozen run identifier.
arm: real, circular, cross_seed, frozen_equal, or frozen_tuned.
seed: Target training seed.
circular_replicate: Circular-shift replicate index; blank for other arms.
donor_seed: Cross-seed replay donor; blank for other arms.
step: REINVENT4 training step (251-300).
sample_index: Zero-based position within the 64-molecule step batch.
raw_smiles: SMILES recorded by REINVENT4.
valid: 1 when RDKit parsing and canonicalization succeeded; otherwise 0.
canonical_smiles: RDKit canonical SMILES; blank for invalid rows.
svm_activity: Frozen controller-feedback SVM probability, rescored on canonical SMILES.
mpn_probability: Frozen untouched MPN probability.
ad_covered: 1 when max ECFP4 Tanimoto to the MPN training set is at least 0.30.
max_training_tanimoto: Maximum ECFP4 Tanimoto to the MPN training set.
qed: RDKit QED.
molecular_weight: RDKit molecular weight in Da.
mw_target_score: 1 for molecular weight 200-500 Da inclusive; otherwise 0.
bemis_murcko_scaffold: RDKit Bemis-Murcko scaffold; [ACYCLIC] when applicable.

All score, applicability-domain, and structure-derived fields are blank for
invalid rows. The sidecar metadata JSON records source-model hashes, row counts,
and the SHA-256 checksum of the compressed CSV. Recomputed arm summaries matched
the frozen confirmatory JSON within 1e-8; this tolerance covers GPU
floating-point replay differences below 1e-8 and is much stricter than the
reported precision.
