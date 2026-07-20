#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   run_frozen_weight_control.sh CONFIG PREFIX REINVENT_SEED CADENCE VALIDATION_BATCH
# The caller must use the same CONFIG, seed, cadence, batch size, component
# list, and initial weights as its paired real controller arm.

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 CONFIG PREFIX REINVENT_SEED CADENCE VALIDATION_BATCH" >&2
  exit 2
fi

cd /root/autodl-tmp/four_project
PY=/root/miniconda3/envs/reinvent4/bin/python
CONFIG=$1
PREFIX=$2
SEED=$3
CADENCE=$4
VALIDATION_BATCH=$5

if [ ! -f "${CONFIG}" ]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi
if [ -e "controller_history_${PREFIX}.jsonl" ]; then
  echo "Refusing to overwrite controller history for ${PREFIX}" >&2
  exit 2
fi

export REINVENT_CONTROLLER_MODE=frozen_weight
export REINVENT_CONTROLLER_SEED="${SEED}"
export REINVENT_CONTROLLER_VALIDATE_EVERY="${CADENCE}"
export REINVENT_CONTROLLER_VALIDATION_BATCH="${VALIDATION_BATCH}"
export REINVENT_CONTROLLER_COMPONENTS="${REINVENT_CONTROLLER_COMPONENTS:-DRD2_activity,QED,Molecular weight}"
export REINVENT_CONTROLLER_INIT_WEIGHTS="${REINVENT_CONTROLLER_INIT_WEIGHTS:-0.34,0.33,0.33}"
export REINVENT_CONTROLLER_DEVICE="${REINVENT_CONTROLLER_DEVICE:-cpu}"
export REINVENT_CONTROLLER_HISTORY_PATH="controller_history_${PREFIX}.jsonl"
export REINVENT_VALIDATION_COMMAND="${REINVENT_VALIDATION_COMMAND:-${PY} scripts/validate_qsar.py --input {input} --output {output} --model qsar_models/DRD2/validation_svm.pkl --component-names 'DRD2_activity,QED,Molecular weight'}"

"${PY}" 03_dynamic_weight_controller.py --seed "${SEED}" -l "${PREFIX}.log" "${CONFIG}"
"${PY}" scripts/analyze_controller_history.py \
  --history "controller_history_${PREFIX}.jsonl" \
  --output-prefix "controller_history_${PREFIX}"
