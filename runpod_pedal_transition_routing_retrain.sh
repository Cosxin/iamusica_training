#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

# Close the frontend gate on its final checkpoint before starting the next
# experiment. Alpha 1 reuses the raw-output cache created by alpha 0.
frontend_run=/workspace/runs/pedal-dynamic-crf-unfrozen-bs64-ablation-20260802
for alpha in 0.0 1.0; do
  label=${alpha/./p}
  python 4_eval_pedal.py \
    MAESTRO_PATH=/root/maestro-v3.0.0 \
    'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
    'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
    CHECKPOINT="$frontend_run/step-12000.torch" SPLIT=validation LIMIT=32 \
    CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=crf \
    CRF_CONFIDENCE_WEIGHT="$alpha" \
    PREDICTION_CACHE=/workspace/cache/pedal-unfrozen-step12000-limit32.pt \
    RESULTS_JSON="$frontend_run/eval-alpha${label}-limit32-step-12000.json"
done

run=/workspace/runs/pedal-dynamic-crf-routing-formal-w0p1-20260802
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
  'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
  EVENT_CACHE_PATH=/workspace/cache/pedal-event-train.pt \
  V1_CHECKPOINT=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch \
  INIT_CHECKPOINT=/workspace/runs/pedal-dynamic-crf-formal-bs384-20260802/step-16000.torch \
  OUTPUT_DIR="$run" MODEL_VARIANT=dynamic_crf \
  TRAIN_BS=384 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  DATALOADER_WORKERS=8 MAX_STEPS=1000 SAVE_EVERY=100 LOG_EVERY=10 \
  EVENT_RADIUS=3 EVENT_POS_WEIGHT=5.0 \
  CONFIDENCE_WEIGHT=3.0 OFFSET_WEIGHT=2.0 \
  LR=0.00005 WEIGHT_DECAY=0.0003 CRF_WEIGHT=1.0 \
  TRANSITION_AUX_WEIGHT=0.1 ALIGN_STATE_TO_EXACT_EVENTS=true \
  TRAIN_TRANSITION_HEAD_ONLY=true FREEZE_FRONTEND=true \
  FREEZE_SHARED=true FREEZE_STATE=true AMP=true

for step in 100 300 600 1000; do
  cache="/workspace/cache/pedal-routing-w0p1-step${step}-limit32.pt"
  for alpha in 0.0 1.0; do
    label=${alpha/./p}
    python 4_eval_pedal.py \
      MAESTRO_PATH=/root/maestro-v3.0.0 \
      'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
      'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
      CHECKPOINT="$run/step-${step}.torch" SPLIT=validation LIMIT=32 \
      CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=crf \
      CRF_CONFIDENCE_WEIGHT="$alpha" PREDICTION_CACHE="$cache" \
      RESULTS_JSON="$run/eval-alpha${label}-limit32-step-${step}.json"
  done
done

# Select only on the locked 32-file subset, then touch full validation once.
selection=$(python - "$run" <<'PY'
import glob
import json
import os
import sys

run = sys.argv[1]
candidates = []
for path in glob.glob(os.path.join(run, "eval-alpha*-limit32-step-*.json")):
    payload = json.load(open(path))
    strict = payload["fixed_tolerance_diagnostic"]["50ms"]["down_and_up"]["f1"]
    candidates.append((payload["pedal_down"]["f1"], strict,
                       payload["checkpoint_step"],
                       payload["config"]["CRF_CONFIDENCE_WEIGHT"]))
best = max(candidates)
print(best[2], best[3])
PY
)
read -r best_step best_alpha <<< "$selection"
best_label=${best_alpha/./p}
echo "SELECTED_ROUTING_CHECKPOINT step=$best_step alpha=$best_alpha"
python 4_eval_pedal.py \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  'HDF5_MEL_PATH=/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
  'HDF5_ROLL_PATH=/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
  CHECKPOINT="$run/step-${best_step}.torch" SPLIT=validation \
  CHUNK_SECS=5.0 OVERLAP_SECS=2.0 DECODER=crf \
  CRF_CONFIDENCE_WEIGHT="$best_alpha" \
  PREDICTION_CACHE="/workspace/cache/pedal-routing-selected-step${best_step}-full-validation.pt" \
  RESULTS_JSON="$run/eval-selected-alpha${best_label}-full-validation-step-${best_step}.json"

echo PEDAL_TRANSITION_ROUTING_RETRAIN_AND_EVAL_COMPLETE
