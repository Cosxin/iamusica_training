#!/usr/bin/env bash
set -euo pipefail

# Prepared formal frozen-pedal run. This file is intentionally never launched by
# the data-preparation workflow; run it explicitly only after reviewing smoke logs.
cd /workspace/iamusica_training

MEL='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
RUN='/workspace/runs/pedal-frozen-v1'

test -s "$MEL"
test -s "$ROLL"
test -s models/keyup_framehead_step12000.torch
test ! -e "$RUN"

python 3_train_pedal.py \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  HDF5_MEL_PATH="$MEL" \
  HDF5_ROLL_PATH="$ROLL" \
  BASE_SNAPSHOT=models/keyup_framehead_step12000.torch \
  OUTPUT_DIR="$RUN" \
  TRAIN_BS=8 \
  TRAIN_BATCH_SECS=2.5 \
  DATALOADER_WORKERS=4 \
  MAX_STEPS=12000 \
  LOG_EVERY=10 \
  SAVE_EVERY=1000 \
  AMP=true
