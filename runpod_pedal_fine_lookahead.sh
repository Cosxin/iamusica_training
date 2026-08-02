#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training
maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
init=/workspace/runs/pedal-mobile-regression-v2c2-20260801/final-step-4000.torch
root=/workspace/runs/pedal-fine-lookahead-20260801
mkdir -p "$root"

tags=(d048 d072 d096 d120 d144 d168 d192)
delays=(0.048 0.072 0.096 0.120 0.144 0.168 0.192)

for i in "${!tags[@]}"; do
  tag=${tags[$i]}
  delay=${delays[$i]}
  run="$root/$tag"
  if [[ ! -f "$run/final-step-6000.torch" ]]; then
    python 6_train_mobile_pedal_regression.py \
      MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
      V1_CHECKPOINT=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch \
      INIT_CHECKPOINT="$init" OUTPUT_DIR="$run" \
      TRAIN_BS=4 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS="$delay" \
      MAX_STEPS=6000 SAVE_EVERY=1000 LOG_EVERY=50 \
      FREEZE_SHARED=false LR=0.00005
  fi

  # Screen 2k/4k/6k checkpoints at the two useful coarse thresholds. Keeping
  # training and context comparisons in one run exposes under/over-training.
  for step in 2000 4000 6000; do
    checkpoint="$run/step-$step.torch"
    [[ $step == 6000 ]] && checkpoint="$run/final-step-6000.torch"
    for threshold in 0.6 0.7; do
      result="$run/eval-validation-limit32-s${step}-e${threshold/./}.json"
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
done

python - "$root" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, 'd*', 'eval-validation-limit32-*.json')):
    with open(path) as stream:
        r = json.load(stream)
    rows.append({
        'run': os.path.basename(os.path.dirname(path)),
        'step': r['checkpoint_step'],
        'threshold': r['config']['EVENT_THRESHOLD'],
        'future_s': r['effective_future_context_secs'],
        'state_f1': r['pedal_state']['f1'],
        'down_f1': r['pedal_down']['f1'],
        'up50_f1': r['fixed_tolerance_diagnostic']['50ms']['pedal_up']['f1'],
        'strict_f1': r['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
        'checkpoint': r['config']['CHECKPOINT'],
    })
rows.sort(key=lambda x: (x['future_s'], x['step'], x['threshold']))
selected = max(rows, key=lambda x: (x['down_f1'], x['strict_f1']))
with open(os.path.join(root, 'subset-summary.json'), 'w') as stream:
    json.dump(rows, stream, indent=2)
with open(os.path.join(root, 'selected.json'), 'w') as stream:
    json.dump(selected, stream, indent=2)
print(json.dumps(selected, indent=2))
PY

echo PEDAL_FINE_LOOKAHEAD_SWEEP_COMPLETE
