#!/usr/bin/env bash
# Full mixed run: merge clean+aptX+SBC HDF5s, then a complete combined A+B
# fine-tune on the mix. Runs alone (no parallel render) so it can't OOM the
# 50GB container. Detached-friendly: one long job, frequent checkpoints.
set -e
cd /workspace/iamusica_training
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
mkdir -p /root/h5-mixed

if [ ! -f "/root/h5-mixed/$MEL" ]; then
  echo "=== MERGING MEL (clean+aptx+sbc) ==="
  python merge_h5.py "/root/h5-mixed/$MEL" \
      "/root/h5-clean/$MEL" "/root/h5-aptx/$MEL" "/root/h5-sbc/$MEL"
else echo "=== MEL already merged — skipping ==="; fi
if [ ! -f "/root/h5-mixed/$ROLL" ]; then
  echo "=== MERGING ROLL (clean+aptx+sbc) ==="
  python merge_h5.py "/root/h5-mixed/$ROLL" \
      "/root/h5-clean/$ROLL" "/root/h5-aptx/$ROLL" "/root/h5-sbc/$ROLL"
else echo "=== ROLL already merged — skipping ==="; fi

echo "=== LAUNCHING FULL MIXED A+B TRAINING ==="
CKPT=$(ls assets/*.torch | head -1)
python 1_train_onsets_velocities.py \
  SNAPSHOT_INPATH="$CKPT" \
  MAESTRO_PATH=/root/maestro_extract/maestro-v3.0.0 \
  HDF5_MEL_PATH="/root/h5-mixed/$MEL" \
  HDF5_ROLL_PATH="/root/h5-mixed/$ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
  OUTPUT_DIR=/root/runs/mixed_ABC \
  LR_MAX=0.0008 MAX_STEPS=5000 XV_EVERY=1000 TRAIN_LOG_EVERY=10 \
  TRAINABLE_ONSETS=True TRAINABLE_COMPONENTS=all ENABLE_FRAME_HEAD=True
echo "=== FULL_MIXED_TRAINING_DONE ==="
