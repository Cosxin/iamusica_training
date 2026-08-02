#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

mode="${1:-smoke}"
train_bs="${TRAIN_BS:-20}"
case "$mode" in
  smoke)
    output="/workspace/runs/pedal-offline-oracle-pooled-10ms-smoke-bs${train_bs}-20260802"
    steps=20
    save_every=20
    ;;
  formal)
    output=/workspace/runs/pedal-offline-oracle-pooled-10ms-fromscratch-20260802
    # Match the 24 ms oracle's 12k x batch-48 sample exposure. A smaller batch
    # is required because each ten-second example has about 2.4x more frames.
    steps=$((12000 * 48 / train_bs))
    save_every=2400
    ;;
  *) echo "usage: $0 [smoke|formal]" >&2; exit 2 ;;
esac

python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  'HDF5_MEL_PATH=/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5' \
  'HDF5_ROLL_PATH=/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5' \
  EVENT_CACHE_PATH=/workspace/cache/pedal-event-train.pt OUTPUT_DIR="$output" \
  MODEL_VARIANT=offline_oracle FROM_SCRATCH=true \
  SHARED_HIDDEN=384 TOWER_HIDDEN=256 SHARED_LAYERS=2 \
  BIDIRECTIONAL_HIDDEN=224 DROPOUT=0.15 \
  TRAIN_BS="$train_bs" TRAIN_BATCH_SECS=10.0 DATALOADER_WORKERS=8 \
  FREEZE_FRONTEND=false FREEZE_SHARED=false FREEZE_STATE=false \
  FRONTEND_LR_SCALE=1.0 STATE_WEIGHT=1.0 \
  EVENT_RADIUS=7 EVENT_POS_WEIGHT=5.0 \
  CONFIDENCE_WEIGHT=3.0 OFFSET_WEIGHT=2.0 \
  ALIGN_STATE_TO_EXACT_EVENTS=true FUTURE_CONTEXT_SECS=0.0 \
  LR=0.001 WEIGHT_DECAY=0.0003 AMP=true \
  MAX_STEPS="$steps" SAVE_EVERY="$save_every" LOG_EVERY=10
