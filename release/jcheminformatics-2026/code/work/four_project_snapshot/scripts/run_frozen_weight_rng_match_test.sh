#!/usr/bin/env bash
set -Eeuo pipefail

# A two-step integration test, not a confirmatory experiment. Both arms use
# the frozen-weight wrapper; only the second arm executes validation.

cd /root/autodl-tmp/four_project
PY=/root/miniconda3/envs/reinvent4/bin/python
ROOT=${1:-rng_match_test}
SEED=${REINVENT_RNG_MATCH_SEED:-1701}
STEPS=${REINVENT_RNG_MATCH_STEPS:-2}
VALIDATE_EVERY=${REINVENT_RNG_MATCH_VALIDATE_EVERY:-1}
VALIDATION_BATCH=${REINVENT_RNG_MATCH_VALIDATION_BATCH:-8}

if [ -e "${ROOT}" ]; then
  echo "Refusing to overwrite existing test directory: ${ROOT}" >&2
  exit 2
fi
mkdir -p "${ROOT}/ordinary" "${ROOT}/frozen"

"${PY}" - "${ROOT}" "${STEPS}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
steps = int(sys.argv[2])
source = Path("07_drd2_qsar_pilot_smoke.toml").read_text()

for arm in ("ordinary", "frozen"):
    text = source
    tb = root / arm / "tb"
    config_json = root / arm / "config.json"
    training_prefix = root / arm / "training"
    checkpoint = root / arm / "final.chkpt"
    text = re.sub(r'tb_logdir = ".*"', f'tb_logdir = "{tb}"', text)
    text = re.sub(r'json_out_config = ".*"', f'json_out_config = "{config_json}"', text)
    text = re.sub(r'summary_csv_prefix = ".*"', f'summary_csv_prefix = "{training_prefix}"', text)
    text = re.sub(r'chkpt_file = ".*"', f'chkpt_file = "{checkpoint}"', text)
    text = re.sub(r'min_steps = \d+', f'min_steps = {steps}', text)
    text = re.sub(r'max_steps = \d+', f'max_steps = {steps}', text)
    (root / arm / "run.toml").write_text(text)
PY

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export REINVENT_STRICT_DETERMINISM=1
export REINVENT_CONTROLLER_COMPONENTS="DRD2_activity,QED,Molecular weight"
export REINVENT_CONTROLLER_INIT_WEIGHTS="0.34,0.33,0.33"
export REINVENT_CONTROLLER_MODE=frozen_weight
export REINVENT_CONTROLLER_SEED="${SEED}"
export REINVENT_CONTROLLER_VALIDATION_BATCH="${VALIDATION_BATCH}"
export REINVENT_CONTROLLER_DEVICE=cpu

# Ordinary fixed-weight training: exact same wrapper/scoring path, no validation call.
unset REINVENT_VALIDATION_COMMAND
export REINVENT_CONTROLLER_VALIDATE_EVERY="$((STEPS + 1))"
export REINVENT_CONTROLLER_HISTORY_PATH="${ROOT}/ordinary/controller_history.jsonl"
export REINVENT_TRAINING_TRACE_PATH="${ROOT}/ordinary/training_trace.jsonl"
"${PY}" 03_dynamic_weight_controller.py --seed "${SEED}" -l "${ROOT}/ordinary/run.log" "${ROOT}/ordinary/run.toml"

# Frozen-weight wrapper: identical fixed weights, training config, seed, and wrapper path.
export REINVENT_CONTROLLER_VALIDATE_EVERY="${VALIDATE_EVERY}"
export REINVENT_CONTROLLER_HISTORY_PATH="${ROOT}/frozen/controller_history.jsonl"
export REINVENT_TRAINING_TRACE_PATH="${ROOT}/frozen/training_trace.jsonl"
export REINVENT_VALIDATION_COMMAND="${PY} scripts/validate_qsar.py --input {input} --output {output} --model qsar_models/DRD2/validation_svm.pkl --component-names 'DRD2_activity,QED,Molecular weight'"
"${PY}" 03_dynamic_weight_controller.py --seed "${SEED}" -l "${ROOT}/frozen/run.log" "${ROOT}/frozen/run.toml"

"${PY}" scripts/analyze_frozen_weight_rng_match.py \
  --ordinary-trace "${ROOT}/ordinary/training_trace.jsonl" \
  --frozen-trace "${ROOT}/frozen/training_trace.jsonl" \
  --frozen-history "${ROOT}/frozen/controller_history.jsonl" \
  --ordinary-checkpoint "${ROOT}/ordinary/final.chkpt" \
  --frozen-checkpoint "${ROOT}/frozen/final.chkpt" \
  --output "${ROOT}/rng_match_summary.json"

echo "PASS ${ROOT}/rng_match_summary.json"
