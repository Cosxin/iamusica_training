#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

training_supervisor_pid="${TRAINING_SUPERVISOR_PID:-80155}"
run=/workspace/runs/pedal-edge-10ms-allocation-128x288-radius7-10ep-500ms-20260802
results=results_eval/edge-10ms-allocation-128x288
cache=/workspace/cache/edge-10ms-allocation-128x288
mel='/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5'
roll='/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5'

while kill -0 "$training_supervisor_pid" 2>/dev/null; do
  sleep 30
done
test -s "$run/step-120000.torch"
mkdir -p "$results/calibration" "$cache" /workspace/artifacts

for step in 1000 12000 24000 60000 120000; do
  python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${step}.torch" \
    MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" SPLIT=validation LIMIT=32 \
    DECODER=regression EVENT_THRESHOLD=0.3 NMS_RADIUS=3 \
    CHUNK_SECS=5 OVERLAP_SECS=2 \
    PREDICTION_CACHE="$cache/step${step}-limit32.pt" \
    RESULTS_JSON="$results/step${step}-limit32.json" \
    > "/workspace/pedal-edge-allocation-128x288-step${step}-eval.log" 2>&1
done

best_step=$(python - "$results" <<'PY'
import json, re, sys
from pathlib import Path
rows = []
for path in Path(sys.argv[1]).glob('step*-limit32.json'):
    match = re.fullmatch(r'step(\d+)-limit32\.json', path.name)
    if match is None:
        continue
    payload = json.loads(path.read_text())
    rows.append((payload['pedal_down']['f1'],
                 payload['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
                 int(match.group(1))))
print(max(rows)[2])
PY
)

for threshold in 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9; do
  label=${threshold/./}
  python 4_eval_pedal.py DEVICE=cpu CHECKPOINT="$run/step-${best_step}.torch" \
    MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" SPLIT=validation LIMIT=32 \
    DECODER=regression EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
    CHUNK_SECS=5 OVERLAP_SECS=2 \
    PREDICTION_CACHE="$cache/step${best_step}-limit32.pt" \
    RESULTS_JSON="$results/calibration/threshold-${label}.json" \
    > "/workspace/pedal-edge-allocation-128x288-threshold-${label}.log" 2>&1
done

best_threshold=$(python - "$results/calibration" <<'PY'
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

python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${best_step}.torch" \
  MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
  HDF5_ROLL_PATH="$roll" SPLIT=validation \
  DECODER=regression EVENT_THRESHOLD="$best_threshold" NMS_RADIUS=3 \
  CHUNK_SECS=5 OVERLAP_SECS=2 \
  PREDICTION_CACHE="$cache/step${best_step}-full-validation.pt" \
  RESULTS_JSON="$results/step${best_step}-full-validation.json" \
  > "/workspace/pedal-edge-allocation-128x288-full-validation.log" 2>&1

artifact="/workspace/artifacts/pedal-edge-10ms-allocation-128x288-step${best_step}.onnx"
python export_mobile_pedal_onnx.py \
  --checkpoint "$run/step-${best_step}.torch" --output "$artifact" --frames 201 \
  > "/workspace/pedal-edge-allocation-128x288-export.log" 2>&1

python - "$results" "$best_step" "$best_threshold" "$artifact" <<'PY'
import json, sys
from pathlib import Path
results, step, threshold, artifact = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
validation = json.loads((results / f'step{step}-full-validation.json').read_text())
payload = {
    'selected_step': step,
    'selected_threshold': threshold,
    'selection_order': ['locked32_down_f1', 'locked32_strict50_f1'],
    'threshold_selection_order': ['locked32_down_f1', 'locked32_strict50_f1'],
    'full_validation': {
        'state_f1': validation['pedal_state']['f1'],
        'down_f1': validation['pedal_down']['f1'],
        'up50_f1': validation['fixed_tolerance_diagnostic']['50ms']['pedal_up']['f1'],
        'strict50_f1': validation['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
    },
    'artifact': json.loads(Path(str(artifact) + '.json').read_text()),
}
(results / 'selection.json').write_text(json.dumps(payload, indent=2) + '\n')
PY

if python - "$results/selection.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))['full_validation']
raise SystemExit(0 if p['down_f1'] >= .75 and p['strict50_f1'] >= .70 else 1)
PY
then
  python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${best_step}.torch" \
    MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" SPLIT=test \
    DECODER=regression EVENT_THRESHOLD="$best_threshold" NMS_RADIUS=3 \
    CHUNK_SECS=5 OVERLAP_SECS=2 \
    PREDICTION_CACHE="$cache/step${best_step}-test.pt" \
    RESULTS_JSON="$results/step${best_step}-test.json" \
    > "/workspace/pedal-edge-allocation-128x288-test.log" 2>&1
else
  echo "ALLOCATION_TEST_SKIPPED validation gates not met"
fi
