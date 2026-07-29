#!/usr/bin/env bash
# Baseline eval of the shipped checkpoint on clean + aptX test sets.
# Establishes the reference: reproduce ~0.9675 on clean, and measure how much
# the shipped model degrades on aptX-codec audio (the motivation for Path A).
set -e
cd /workspace/iamusica_training
CKPT=$(ls assets/*.torch | head -1)
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
GT=/root/maestro_extract/maestro-v3.0.0
echo "CKPT=$CKPT"

for V in clean aptx; do
  H5=/root/h5-$V
  echo "=== EVAL baseline on $V ==="
  python 2_eval_onsets_velocities.py \
    SNAPSHOT_INPATH="$CKPT" \
    MAESTRO_PATH="$GT" \
    HDF5_MEL_PATH="$H5/$MEL" \
    HDF5_ROLL_PATH="$H5/$ROLL" \
    DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
    RESULTS_JSON=/root/eval-baseline-$V.json \
    RUN_NAME=baseline_$V DATASET_VARIANT=$V
  echo "=== DONE $V ==="
done
echo ALL_BASELINE_EVALS_DONE
