#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training
results=results_eval/offline-oracle/calibration
mkdir -p "$results"
checkpoint=/workspace/runs/pedal-offline-oracle-24ms-fromscratch-20260802/final-step-12000.torch
locked_cache=/workspace/cache/pedal-offline-oracle-step12000-limit32-full.pt

for threshold in 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9; do
  label=${threshold/./}
  python 4_eval_pedal.py DEVICE=cpu CHECKPOINT="$checkpoint" \
    MAESTRO_PATH=/root/maestro-v3.0.0 \
    'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
    'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
    SPLIT=validation LIMIT=32 FULL_RECORDING_INFERENCE=true \
    DECODER=regression EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
    CHUNK_SECS=5 OVERLAP_SECS=2 PREDICTION_CACHE="$locked_cache" \
    RESULTS_JSON="$results/threshold-${label}.json" \
    > "/workspace/pedal-offline-oracle-threshold-${label}.log" 2>&1
done

best_threshold=$(python - "$results" <<'PY'
import json, sys
from pathlib import Path
rows = []
for path in Path(sys.argv[1]).glob('threshold-*.json'):
    payload = json.loads(path.read_text())
    threshold = float(path.stem.split('-', 1)[1]) / 10.0
    if threshold > 1:
        threshold /= 10.0
    rows.append((payload['pedal_down']['f1'],
                 payload['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
                 threshold))
print(max(rows)[2])
PY
)

python 4_eval_pedal.py DEVICE=cpu \
  CHECKPOINT=/workspace/runs/pedal-offline-oracle-24ms-fromscratch-20260802/step-12000.torch \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
  'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
  SPLIT=validation FULL_RECORDING_INFERENCE=true \
  DECODER=regression EVENT_THRESHOLD="$best_threshold" NMS_RADIUS=3 \
  CHUNK_SECS=8 OVERLAP_SECS=4 \
  PREDICTION_CACHE=/workspace/cache/pedal-offline-oracle-step12000-full-validation.pt \
  RESULTS_JSON=results_eval/offline-oracle/step12000-full-validation-calibrated.json \
  > /workspace/pedal-offline-oracle-full-validation-calibrated.log 2>&1

echo "ORACLE_SELECTED_THRESHOLD=$best_threshold"
