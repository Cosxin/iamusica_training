#!/usr/bin/env bash
# Full eval matrix: fine-tuned (mixed step5000) model on clean/aptX/SBC test
# sets, plus the missing shipped-on-SBC baseline. Fine-tuned model has the frame
# head, so ENABLE_FRAME_HEAD=True when loading it.
set -e
cd /workspace/iamusica_training
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
GT=/root/maestro_extract/maestro-v3.0.0
FT=$(ls -t /root/runs/mixed_ABC/model_snapshots/*step=5000* 2>/dev/null | head -1)
SHIPPED=$(ls assets/*.torch | head -1)
echo "FT=$FT"

for V in sbc aptx clean; do
  echo "=== FINE-TUNED on $V ==="
  python 2_eval_onsets_velocities.py \
    SNAPSHOT_INPATH="$FT" MAESTRO_PATH="$GT" \
    HDF5_MEL_PATH="/root/h5-$V/$MEL" HDF5_ROLL_PATH="/root/h5-$V/$ROLL" \
    DEVICE=cuda ALLOW_PARTIAL_HDF5=True ENABLE_FRAME_HEAD=True \
    RESULTS_JSON=/root/eval-finetuned-$V.json RUN_NAME=finetuned_$V DATASET_VARIANT=$V
done

echo "=== SHIPPED baseline on SBC ==="
python 2_eval_onsets_velocities.py \
  SNAPSHOT_INPATH="$SHIPPED" MAESTRO_PATH="$GT" \
  HDF5_MEL_PATH="/root/h5-sbc/$MEL" HDF5_ROLL_PATH="/root/h5-sbc/$ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
  RESULTS_JSON=/root/eval-baseline-sbc.json RUN_NAME=baseline_sbc DATASET_VARIANT=sbc
echo EVAL_MATRIX_DONE
