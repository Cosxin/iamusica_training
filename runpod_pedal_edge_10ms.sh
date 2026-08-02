#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

mode="${1:-smoke}"
train_bs="${TRAIN_BS:-4}"
fast_size="${FAST_SIZE:-64}"
context_hidden="${CONTEXT_HIDDEN:-320}"
run_tag="${RUN_TAG:-radius7-10ep-500ms-20260802}"
maestro=/root/maestro-v3.0.0
mel='/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5'
roll='/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5'
events=/workspace/cache/pedal-event-train.pt

case "$mode" in
  smoke)
    output="${OUTPUT_DIR:-/workspace/runs/pedal-edge-10ms-${run_tag}-smoke-bs${train_bs}}"
    steps=20
    save_every=20
    warmup_steps=5
    ;;
  formal)
    output="${OUTPUT_DIR:-/workspace/runs/pedal-edge-10ms-${run_tag}}"
    steps="${MAX_STEPS:-24000}"
    save_every=1000
    warmup_steps=500
    ;;
  *)
    echo "usage: $0 [smoke|formal]" >&2
    exit 2
    ;;
esac

test -s "$mel"
test -s "$roll"

# A centered 2048-sample STFT sees ~64 ms of future waveform. A 430 ms event
# delay therefore remains just below the 500 ms end-to-end acoustic budget.
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  EVENT_CACHE_PATH="$events" OUTPUT_DIR="$output" \
  MODEL_VARIANT=edge_multirate_10ms FROM_SCRATCH=true \
  FAST_SIZE="$fast_size" CONTEXT_HIDDEN="$context_hidden" \
  CONTEXT_LAYERS=2 CONTEXT_POOL=5 \
  EVENT_DELAY_FRAMES=43 HEAD_HIDDEN=128 \
  DROPOUT=0.12 TRAIN_BS="$train_bs" TRAIN_BATCH_SECS=6.0 \
  DATALOADER_WORKERS=12 FREEZE_FRONTEND=false FREEZE_SHARED=false \
  FREEZE_STATE=false FRONTEND_LR_SCALE=1.0 STATE_WEIGHT=1.0 \
  EVENT_RADIUS=7 EVENT_POS_WEIGHT=5.0 CONFIDENCE_WEIGHT=3.0 \
  OFFSET_WEIGHT=2.0 ALIGN_STATE_TO_EXACT_EVENTS=true \
  FUTURE_CONTEXT_SECS=0.43 DELAY_STATE_WITH_EVENTS=false \
  LR=0.0007 WARMUP_STEPS="$warmup_steps" MIN_LR_RATIO=0.05 \
  WEIGHT_DECAY=0.0003 AMP=true \
  MAX_STEPS="$steps" SAVE_EVERY="$save_every" LOG_EVERY=10
