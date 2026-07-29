#!/usr/bin/env bash
# Path A fine-tune: warm-start the shipped checkpoint and adapt it to aptX-codec
# audio. Low LR to preserve the pretrained 0.9675 onset skill while learning
# codec robustness. Bounded to MAX_STEPS with XV checkpoints so we get a loss
# curve + F1 progress within the lease.
set -e
cd /workspace/iamusica_training
CKPT=$(ls assets/*.torch | head -1)
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
echo "CKPT=$CKPT"
python 1_train_onsets_velocities.py \
  SNAPSHOT_INPATH="$CKPT" \
  MAESTRO_PATH=/root/maestro_extract/maestro-v3.0.0 \
  HDF5_MEL_PATH="/root/h5-aptx/$MEL" \
  HDF5_ROLL_PATH="/root/h5-aptx/$ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
  OUTPUT_DIR=/root/runs/A_aptx \
  LR_MAX=0.0008 MAX_STEPS=3000 XV_EVERY=1000 TRAIN_LOG_EVERY=10 \
  TRAINABLE_ONSETS=True
echo PATHA_FINETUNE_DONE
