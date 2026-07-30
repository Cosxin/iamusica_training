#!/usr/bin/env bash
# Extra research evals for the paper. Waits for the running matrix+offset evals
# to finish (avoid GPU/RAM contention), then runs:
#  1) ABLATION on SBC: aptX-only fine-tune (NO SBC in training, step2000) vs the
#     mixed fine-tune (WITH SBC) -> isolates the value of SBC augmentation.
#  2) MULTI-CHECKPOINT note-with-offset curve (mixed step 1000/3000/5000 on SBC)
#     -> shows the release head improving with training (the "eval chart").
# All models here have the frame head (ENABLE_FRAME_HEAD=True).
set -e
cd /workspace/iamusica_training
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
GT=/root/maestro_extract/maestro-v3.0.0
SBC_MEL="/root/h5-sbc/$MEL"; SBC_ROLL="/root/h5-sbc/$ROLL"

echo "=== waiting for matrix + offset evals to finish ==="
while ! grep -q EVAL_MATRIX_DONE /root/eval-matrix.log 2>/dev/null \
   || ! grep -q OFFSET_ALL_DONE /root/offset-all.log 2>/dev/null; do sleep 60; done
echo "=== running-evals done; starting research evals ==="

APTX2000=$(ls /root/runs/AB_aptx/model_snapshots/*step=2000*.torch | head -1)

# 1) ablation: aptX-only (no SBC in training) on SBC, onset/vel + offset
echo "=== ABLATION aptX-only(step2000) on SBC ==="
python 2_eval_onsets_velocities.py SNAPSHOT_INPATH="$APTX2000" MAESTRO_PATH="$GT" \
  HDF5_MEL_PATH="$SBC_MEL" HDF5_ROLL_PATH="$SBC_ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True ENABLE_FRAME_HEAD=True \
  RESULTS_JSON=/root/eval-aptxonly-sbc.json RUN_NAME=aptxonly_sbc DATASET_VARIANT=sbc
python eval_offset.py SNAPSHOT_INPATH="$APTX2000" MAESTRO_PATH="$GT" \
  HDF5_MEL_PATH="$SBC_MEL" HDF5_ROLL_PATH="$SBC_ROLL" \
  DATASET_VARIANT=sbc RESULTS_JSON=/root/offset-aptxonly-sbc.json

# 2) multi-checkpoint note-with-offset curve on SBC (subset for speed)
for STEP in 1000 3000 5000; do
  CK=$(ls /root/runs/mixed_ABC/model_snapshots/*step=${STEP}_*.torch | head -1)
  echo "=== MULTICKPT offset mixed step=$STEP on SBC ==="
  python eval_offset.py SNAPSHOT_INPATH="$CK" MAESTRO_PATH="$GT" \
    HDF5_MEL_PATH="$SBC_MEL" HDF5_ROLL_PATH="$SBC_ROLL" \
    DATASET_VARIANT=sbc LIMIT=60 RESULTS_JSON=/root/offset-mixed-step${STEP}-sbc.json
done
echo RESEARCH_EVALS_DONE
