#!/usr/bin/env bash
set -euo pipefail

sweep_pid=${1:?sweep PID required}
while kill -0 "$sweep_pid" 2>/dev/null; do sleep 30; done

cd /workspace/iamusica_training
maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
baseline=/workspace/runs/pedal-mobile-regression-v2c2-20260801/final-step-4000.torch
root=/workspace/runs/pedal-lookahead-sweep-20260801
mkdir -p "$root/d000"

for threshold in 0.4 0.5 0.6 0.7; do
  result="$root/d000/eval-validation-limit32-e${threshold/./}.json"
  python 4_eval_pedal.py \
    MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
    CHECKPOINT="$baseline" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
    STATE_THRESHOLD=0.5 EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
    RESULTS_JSON="$result"
done

read -r selected_run selected_checkpoint selected_threshold < <(python - "$root" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, 'd*', 'eval-validation-limit32-*.json')):
    with open(path) as stream:
        result = json.load(stream)
    rows.append({
        'run': os.path.basename(os.path.dirname(path)),
        'checkpoint': result['config']['CHECKPOINT'],
        'threshold': result['config']['EVENT_THRESHOLD'],
        'future_s': result['effective_future_context_secs'],
        'state_f1': result['pedal_state']['f1'],
        'down_f1': result['pedal_down']['f1'],
        'strict_f1': result['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
    })
rows.sort(key=lambda x: (x['future_s'], x['threshold']))
delayed = [row for row in rows if row['future_s'] > 0]
selected = max(delayed, key=lambda x: (x['down_f1'], x['strict_f1']))
with open(os.path.join(root, 'subset-summary-with-baseline.json'), 'w') as stream:
    json.dump(rows, stream, indent=2)
with open(os.path.join(root, 'selected.json'), 'w') as stream:
    json.dump(selected, stream, indent=2)
print(selected['run'], selected['checkpoint'], selected['threshold'])
PY
)

python 4_eval_pedal.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  CHECKPOINT="$baseline" SPLIT=validation \
  CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
  STATE_THRESHOLD=0.5 EVENT_THRESHOLD=0.6 NMS_RADIUS=3 \
  RESULTS_JSON="$root/d000/eval-validation-full-e060.json"

python 4_eval_pedal.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  CHECKPOINT="$selected_checkpoint" SPLIT=validation \
  CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
  STATE_THRESHOLD=0.5 EVENT_THRESHOLD="$selected_threshold" NMS_RADIUS=3 \
  RESULTS_JSON="$root/$selected_run/eval-validation-full-selected.json"

echo PEDAL_LOOKAHEAD_FULL_VALIDATION_COMPLETE
