#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training
maestro=/root/maestro-v3.0.0
mel='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
roll='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
cache=/workspace/cache/pedal-event-train.pt
v1=/workspace/runs/pedal-mobile-v1-20260801/final-step-12000.torch
dual=/workspace/runs/pedal-dual-timescale-quick-20260801-v1/step-750.torch
root=/workspace/runs/pedal-dynamic-crf-fast-20260802
stage_a="$root/stage-a-head"
stage_b="$root/stage-b-e2e"

mkdir -p "$root"
test ! -e "$stage_a"
test ! -e "$stage_b"

# Stage A: learn input-dependent transition potentials without destabilizing
# the selected acoustic/state representation.
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  EVENT_CACHE_PATH="$cache" V1_CHECKPOINT="$v1" INIT_CHECKPOINT="$dual" \
  OUTPUT_DIR="$stage_a" MODEL_VARIANT=dynamic_crf \
  TRAIN_BS=128 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  DATALOADER_WORKERS=24 MAX_STEPS=128 SAVE_EVERY=64 LOG_EVERY=8 \
  FREEZE_SHARED=true FREEZE_STATE=true LR=0.0001 CRF_WEIGHT=1.0

# Stage B: end-to-end refinement at lower LR. Across both stages this sees
# 49,152 chunks, over twice the 24,000 chunks in the prior BS4/6000-step run.
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH="$maestro" HDF5_MEL_PATH="$mel" HDF5_ROLL_PATH="$roll" \
  EVENT_CACHE_PATH="$cache" V1_CHECKPOINT="$v1" \
  INIT_CHECKPOINT="$stage_a/final-step-128.torch" \
  OUTPUT_DIR="$stage_b" MODEL_VARIANT=dynamic_crf \
  TRAIN_BS=128 TRAIN_BATCH_SECS=8.0 FUTURE_CONTEXT_SECS=0.072 \
  DATALOADER_WORKERS=24 MAX_STEPS=256 SAVE_EVERY=64 LOG_EVERY=8 \
  FREEZE_SHARED=false FREEZE_STATE=false LR=0.00005 CRF_WEIGHT=1.0

echo PEDAL_DYNAMIC_CRF_FAST_TRAINING_COMPLETE
