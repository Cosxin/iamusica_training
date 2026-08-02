#!/usr/bin/env bash
set -euo pipefail
cd /workspace/iamusica_training

maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
run=/workspace/runs/pedal-d7c-transient-aux-formal-20260802
test ! -e "$run"

python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  EVENT_CACHE_PATH=/workspace/cache/pedal-event-train.pt \
  V1_CHECKPOINT=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch \
  INIT_CHECKPOINT=/workspace/runs/pedal-dynamic-crf-formal-bs384-20260802/step-16000.torch \
  OUTPUT_DIR="$run" MODEL_VARIANT=dynamic_crf_two_stream \
  TRAIN_BS=320 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  DATALOADER_WORKERS=8 MAX_STEPS=600 SAVE_EVERY=100 LOG_EVERY=10 \
  EVENT_RADIUS=3 EVENT_POS_WEIGHT=5.0 \
  CONFIDENCE_WEIGHT=3.0 OFFSET_WEIGHT=2.0 LR=0.00005 \
  WEIGHT_DECAY=0.0003 CRF_WEIGHT=1.0 TRANSITION_AUX_WEIGHT=0.0 \
  TRANSIENT_AUX_WEIGHT=1.0 ALIGN_STATE_TO_EXACT_EVENTS=false \
  FREEZE_FRONTEND=true FREEZE_SHARED=false FREEZE_STATE=false AMP=true

for step in 100 300 600; do
  checkpoint="$run/step-${step}.torch"
  cache="/workspace/cache/pedal-d7c-transient-aux-step${step}-limit32.pt"
  python 4_eval_pedal.py MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=regression EVENT_THRESHOLD=0.5 \
    PREDICTION_CACHE="$cache" \
    RESULTS_JSON="$run/eval-regression-limit32-step-${step}.json"
  python 4_eval_pedal.py MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" CHECKPOINT="$checkpoint" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=crf CRF_CONFIDENCE_WEIGHT=1.0 \
    PREDICTION_CACHE="$cache" \
    RESULTS_JSON="$run/eval-crf-alpha1-limit32-step-${step}.json"
done

selection=$(python - "$run" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(root, "eval-*-limit32-step-*.json")):
    result = json.load(open(path))
    fixed = result["fixed_tolerance_diagnostic"]["50ms"]
    rows.append({
        "decoder": result["config"]["DECODER"],
        "alpha": result["config"].get("CRF_CONFIDENCE_WEIGHT", 0.0),
        "step": result["checkpoint_step"],
        "state_f1": result["pedal_state"]["f1"],
        "down_f1": result["pedal_down"]["f1"],
        "up50_f1": fixed["pedal_up"]["f1"],
        "strict_f1": fixed["down_and_up"]["f1"],
    })
best = max(rows, key=lambda x: (x["down_f1"], x["strict_f1"]))
json.dump({"rows": sorted(rows, key=lambda x: (x["step"], x["decoder"])),
           "selected": best}, open(os.path.join(root, "quick-summary.json"), "w"),
          indent=2)
print(best["step"], best["decoder"], best["alpha"], best["down_f1"])
PY
)
read -r best_step best_decoder best_alpha best_down <<< "$selection"
echo "D7C_SELECTED step=$best_step decoder=$best_decoder alpha=$best_alpha down=$best_down"

if python - "$best_down" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.7157 else 1)
PY
then
  args=(DECODER="$best_decoder")
  [[ "$best_decoder" == crf ]] && args+=(CRF_CONFIDENCE_WEIGHT="$best_alpha")
  python 4_eval_pedal.py MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" CHECKPOINT="$run/step-${best_step}.torch" \
    SPLIT=validation CHUNK_SECS=5.0 OVERLAP_SECS=2.0 "${args[@]}" \
    PREDICTION_CACHE=/workspace/cache/pedal-d7c-selected-full.pt \
    RESULTS_JSON="$run/eval-selected-full-validation-step-${best_step}.json"
else
  echo D7C_FULL_EVAL_SKIPPED_NO_SUBSET_GAIN
fi
echo PEDAL_D7C_TRANSIENT_AUX_QUICK_COMPLETE
