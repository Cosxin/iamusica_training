#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

wait_pid="${WAIT_PID:-82263}"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done

maestro=/root/maestro-v3.0.0
mel24='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll24='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
mel10='/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5'
roll10='/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5'

run_control() {
  local name="$1" run="$2" mel="$3" roll="$4" nms="$5" final_step="$6"
  shift 6
  local checkpoints=("$@")
  local results="results_eval/$name" cache="/workspace/cache/$name"
  mkdir -p "$results/calibration" "$cache"

  for step in "${checkpoints[@]}"; do
    python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${step}.torch" \
      MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
      SPLIT=validation LIMIT=32 FULL_RECORDING_INFERENCE=true \
      DECODER=regression EVENT_THRESHOLD=0.3 NMS_RADIUS="$nms" \
      CHUNK_SECS=5 OVERLAP_SECS=2 \
      PREDICTION_CACHE="$cache/step${step}-locked32.pt" \
      RESULTS_JSON="$results/step${step}-locked32.json" \
      > "/workspace/${name}-step${step}-eval.log" 2>&1
  done

  local best_step
  best_step=$(python - "$results" <<'PY'
import json, re, sys
from pathlib import Path
rows=[]
for path in Path(sys.argv[1]).glob('step*-locked32.json'):
    match=re.fullmatch(r'step(\d+)-locked32\.json',path.name)
    if match:
        p=json.loads(path.read_text())
        rows.append((p['pedal_down']['f1'],
                     p['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
                     int(match.group(1))))
if not rows:
    raise SystemExit('no checkpoint results')
print(max(rows)[2])
PY
  )

  for threshold in 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9; do
    local label=${threshold/./}
    python 4_eval_pedal.py DEVICE=cpu CHECKPOINT="$run/step-${best_step}.torch" \
      MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
      SPLIT=validation LIMIT=32 FULL_RECORDING_INFERENCE=true \
      DECODER=regression EVENT_THRESHOLD="$threshold" NMS_RADIUS="$nms" \
      CHUNK_SECS=5 OVERLAP_SECS=2 \
      PREDICTION_CACHE="$cache/step${best_step}-locked32.pt" \
      RESULTS_JSON="$results/calibration/threshold-${label}.json" \
      > "/workspace/${name}-threshold-${label}.log" 2>&1
  done

  local best_threshold
  best_threshold=$(python - "$results/calibration" <<'PY'
import json, sys
from pathlib import Path
rows=[]
for path in Path(sys.argv[1]).glob('threshold-*.json'):
    label=path.stem.split('-',1)[1]
    threshold=float(label)/10
    if threshold>1: threshold/=10
    p=json.loads(path.read_text())
    rows.append((p['pedal_down']['f1'],
                 p['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
                 threshold))
print(max(rows)[2])
PY
  )

  python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${best_step}.torch" \
    MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
    SPLIT=validation FULL_RECORDING_INFERENCE=true \
    DECODER=regression EVENT_THRESHOLD="$best_threshold" NMS_RADIUS="$nms" \
    CHUNK_SECS=8 OVERLAP_SECS=4 \
    PREDICTION_CACHE="$cache/step${best_step}-full-validation.pt" \
    RESULTS_JSON="$results/step${best_step}-full-validation.json" \
    > "/workspace/${name}-full-validation.log" 2>&1

  python - "$results" "$best_step" "$best_threshold" "$final_step" <<'PY'
import json,sys
from pathlib import Path
root,step,threshold,exposure=Path(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4])
p=json.loads((root/f'step{step}-full-validation.json').read_text())
out={'selected_step':step,'formal_final_step':exposure,'selected_threshold':threshold,
     'selection_order':['locked32_down_f1','locked32_strict50_f1'],
     'full_validation':{'state_f1':p['pedal_state']['f1'],
       'down_f1':p['pedal_down']['f1'],
       'up50_f1':p['fixed_tolerance_diagnostic']['50ms']['pedal_up']['f1'],
       'strict50_f1':p['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1']}}
(root/'selection.json').write_text(json.dumps(out,indent=2)+'\n')
PY
}

# The physical batch is probed first. If 48 examples do not fit, reduce the
# batch while preserving total example exposure in the formal step count.
flatten_bs=48
if ! TRAIN_BS="$flatten_bs" ./runpod_pedal_offline_oracle_flatten_24ms.sh smoke \
    > /workspace/pedal-offline-oracle-flatten-24ms-smoke.log 2>&1; then
  flatten_bs=24
  TRAIN_BS="$flatten_bs" ./runpod_pedal_offline_oracle_flatten_24ms.sh smoke \
    > /workspace/pedal-offline-oracle-flatten-24ms-smoke-retry.log 2>&1
fi
TRAIN_BS="$flatten_bs" ./runpod_pedal_offline_oracle_flatten_24ms.sh formal \
  > /workspace/pedal-offline-oracle-flatten-24ms-formal.log 2>&1
flatten_final=$((12000 * 48 / flatten_bs))
run_control offline-oracle-flatten-24ms \
  /workspace/runs/pedal-offline-oracle-flatten-24ms-fromscratch-20260802 \
  "$mel24" "$roll24" 3 "$flatten_final" 1000 $((flatten_final/2)) "$flatten_final"

pooled10_bs=20
if ! TRAIN_BS="$pooled10_bs" ./runpod_pedal_offline_oracle_pooled_10ms.sh smoke \
    > /workspace/pedal-offline-oracle-pooled-10ms-smoke.log 2>&1; then
  pooled10_bs=10
  TRAIN_BS="$pooled10_bs" ./runpod_pedal_offline_oracle_pooled_10ms.sh smoke \
    > /workspace/pedal-offline-oracle-pooled-10ms-smoke-retry.log 2>&1
fi
TRAIN_BS="$pooled10_bs" ./runpod_pedal_offline_oracle_pooled_10ms.sh formal \
  > /workspace/pedal-offline-oracle-pooled-10ms-formal.log 2>&1
pooled10_final=$((12000 * 48 / pooled10_bs))
run_control offline-oracle-pooled-10ms \
  /workspace/runs/pedal-offline-oracle-pooled-10ms-fromscratch-20260802 \
  "$mel10" "$roll10" 7 "$pooled10_final" 2400 $((pooled10_final/2)) "$pooled10_final"

echo ORACLE_CONTROLS_COMPLETE
