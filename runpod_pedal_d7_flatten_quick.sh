#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
init=/workspace/runs/pedal-dynamic-crf-formal-bs384-20260802/step-16000.torch
run=/workspace/runs/pedal-d7-flatten-formal-20260802

test ! -e "$run"

python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  EVENT_CACHE_PATH=/workspace/cache/pedal-event-train.pt \
  V1_CHECKPOINT=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch \
  INIT_CHECKPOINT="$init" OUTPUT_DIR="$run" MODEL_VARIANT=dynamic_crf_flatten \
  TRAIN_BS=384 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  DATALOADER_WORKERS=8 MAX_STEPS=600 SAVE_EVERY=100 LOG_EVERY=10 \
  EVENT_RADIUS=3 EVENT_POS_WEIGHT=5.0 \
  CONFIDENCE_WEIGHT=3.0 OFFSET_WEIGHT=2.0 \
  LR=0.00005 WEIGHT_DECAY=0.0003 CRF_WEIGHT=1.0 \
  TRANSITION_AUX_WEIGHT=0.0 ALIGN_STATE_TO_EXACT_EVENTS=false \
  TRAIN_TRANSITION_HEAD_ONLY=false FREEZE_FRONTEND=true \
  FREEZE_SHARED=false FREEZE_STATE=false AMP=true

for step in 100 300 600; do
  checkpoint="$run/step-${step}.torch"
  cache="/workspace/cache/pedal-d7-flatten-step${step}-limit32.pt"

  python 4_eval_pedal.py \
    MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
    CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=regression \
    EVENT_THRESHOLD=0.5 PREDICTION_CACHE="$cache" \
    RESULTS_JSON="$run/eval-regression-limit32-step-${step}.json"

  python 4_eval_pedal.py \
    MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
    CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=crf \
    CRF_CONFIDENCE_WEIGHT=1.0 PREDICTION_CACHE="$cache" \
    RESULTS_JSON="$run/eval-crf-alpha1-limit32-step-${step}.json"
done

python - "$run" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, "eval-*-limit32-step-*.json")):
    with open(path) as stream:
        result = json.load(stream)
    rows.append({
        "decoder": result["config"]["DECODER"],
        "step": result["checkpoint_step"],
        "state_f1": result["pedal_state"]["f1"],
        "down_f1": result["pedal_down"]["f1"],
        "up50_f1": result["fixed_tolerance_diagnostic"]["50ms"]["pedal_up"]["f1"],
        "strict_f1": result["fixed_tolerance_diagnostic"]["50ms"]["down_and_up"]["f1"],
    })
rows.sort(key=lambda row: (row["step"], row["decoder"]))
selected = max(rows, key=lambda row: (row["down_f1"], row["strict_f1"]))
with open(os.path.join(root, "quick-summary.json"), "w") as stream:
    json.dump({"rows": rows, "selected": selected}, stream, indent=2)
print(json.dumps({"rows": rows, "selected": selected}, indent=2))
PY

echo PEDAL_D7_FLATTEN_QUICK_COMPLETE
