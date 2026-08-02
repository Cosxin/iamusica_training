#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
baseline=/workspace/runs/pedal-mobile-regression-v2c2-20260801/final-step-4000.torch
root=/workspace/runs/pedal-lookahead-sweep-20260801
mkdir -p "$root"

# Baseline already has ~96 ms of local CNN future context. These delays add
# fixed recurrent future evidence. Steps are mildly increased for long delays
# to compensate for the shorter supervised portion of each 8-second chunk.
tags=(d012 d025 d050 d100 d300)
delays=(0.12 0.25 0.50 1.00 3.00)
steps=(2000 2100 2200 2400 3200)
thresholds=(0.4 0.5 0.6 0.7)

for i in "${!tags[@]}"; do
  tag=${tags[$i]}
  delay=${delays[$i]}
  max_steps=${steps[$i]}
  run="$root/$tag"

  if [[ ! -f "$run/final-step-$max_steps.torch" ]]; then
    python 6_train_mobile_pedal_regression.py \
      MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
      V1_CHECKPOINT=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch \
      INIT_CHECKPOINT="$baseline" OUTPUT_DIR="$run" \
      TRAIN_BS=4 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS="$delay" \
      MAX_STEPS="$max_steps" SAVE_EVERY="$max_steps" LOG_EVERY=50 \
      FREEZE_SHARED=false LR=0.00005
  fi

  checkpoint="$run/final-step-$max_steps.torch"
  for threshold in "${thresholds[@]}"; do
    result="$run/eval-validation-limit32-e${threshold/./}.json"
    if [[ ! -f "$result" ]]; then
      python 4_eval_pedal.py \
        MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
        CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
        CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
        STATE_THRESHOLD=0.5 EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
        RESULTS_JSON="$result"
    fi
  done
done

python - "$root" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, 'd*', 'eval-validation-limit32-*.json')):
    with open(path) as stream:
        result = json.load(stream)
    rows.append({
        'run': os.path.basename(os.path.dirname(path)),
        'threshold': result['config']['EVENT_THRESHOLD'],
        'future_s': result['effective_future_context_secs'],
        'state_f1': result['pedal_state']['f1'],
        'down_f1': result['pedal_down']['f1'],
        'strict_f1': result['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
    })
rows.sort(key=lambda x: (x['future_s'], x['threshold']))
with open(os.path.join(root, 'subset-summary.json'), 'w') as stream:
    json.dump(rows, stream, indent=2)
print(json.dumps(rows, indent=2))
PY

echo PEDAL_LOOKAHEAD_SUBSET_SWEEP_COMPLETE
