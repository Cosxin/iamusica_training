#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training
maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
v1=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch
baseline=/workspace/runs/pedal-fine-lookahead-20260801/d072/final-step-6000.torch
run=/workspace/runs/pedal-dual-timescale-quick-20260801-v1

test ! -e "$run"
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  V1_CHECKPOINT="$v1" INIT_CHECKPOINT="$baseline" OUTPUT_DIR="$run" \
  TRAIN_BS=4 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  MAX_STEPS=1000 SAVE_EVERY=250 LOG_EVERY=25 FREEZE_SHARED=true \
  MODEL_VARIANT=dual_timescale SLOW_DILATION=2 LR=0.00005

for step in 250 500 750 1000; do
  checkpoint="$run/step-$step.torch"
  [[ $step == 1000 ]] && checkpoint="$run/final-step-1000.torch"
  for threshold in 0.6 0.7; do
    python 4_eval_pedal.py \
      MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
      CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
      CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
      STATE_THRESHOLD=0.5 EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
      RESULTS_JSON="$run/eval-validation-limit32-s${step}-e${threshold/./}.json"
  done
done

python - "$run" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, "eval-validation-limit32-*.json")):
    with open(path) as stream:
        result = json.load(stream)
    rows.append({
        "step": result["checkpoint_step"],
        "threshold": result["config"]["EVENT_THRESHOLD"],
        "state_f1": result["pedal_state"]["f1"],
        "down_f1": result["pedal_down"]["f1"],
        "up50_f1": result["fixed_tolerance_diagnostic"]["50ms"]["pedal_up"]["f1"],
        "strict_f1": result["fixed_tolerance_diagnostic"]["50ms"]["down_and_up"]["f1"],
    })
rows.sort(key=lambda row: (row["step"], row["threshold"]))
selected = max(rows, key=lambda row: (row["down_f1"], row["strict_f1"]))
with open(os.path.join(root, "quick-summary.json"), "w") as stream:
    json.dump({"rows": rows, "selected": selected}, stream, indent=2)
print(json.dumps({"selected": selected}, indent=2))
PY

echo PEDAL_DUAL_TIMESCALE_QUICK_COMPLETE
