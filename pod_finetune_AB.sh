#!/usr/bin/env bash
# Combined Path A + B fine-tune: ONE model, ONE run.
#  - A: warm-start the shipped checkpoint and adapt to aptX-codec audio
#  - B: enable the frame/release head (random-init, loaded non-strict) so the
#       same model also learns note duration/release.
# Loss now has 3 terms: onset, velocity, frame. Low LR preserves the pretrained
# onset/velocity skill while the frame head and codec adaptation are learned.
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
  OUTPUT_DIR=/root/runs/AB_aptx \
  LR_MAX=0.0008 MAX_STEPS=3000 XV_EVERY=1000 TRAIN_LOG_EVERY=10 \
  TRAINABLE_ONSETS=True TRAINABLE_COMPONENTS=all ENABLE_FRAME_HEAD=True
echo PATHAB_FINETUNE_DONE
