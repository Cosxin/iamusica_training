#!/usr/bin/env bash
# The production training recipe, as measured on 2026-08-07.
#
# Run pod_bootstrap.sh first: it brings a bare pod to training-ready and GATES
# on reproducing the author's published 0.9675 before letting you spend GPU
# hours. Then run this.
#
# Every choice below is a measured one, on the deterministic xv split
# (validation[::5]), warm-started from the author's checkpoint, one variable at
# a time:
#
#   author's checkpoint, no training      0.96754   <- our XV reproduces it exactly
#   clean data, no extra head             0.96748   <- our fine-tune is faithful
#   augmented data, offset head           0.96637   <- THIS RECIPE, -0.0012
#   clean data, frame + offset heads      0.95598   <- frame head costs -0.0116
#
#   ENABLE_OFFSET_HEAD=True   sounding-off regression; the frame head is gone
#                             (NOTES_frame_head.md)
#   OFFSET_CAP_SECS=2.0       notes past 2 s are censored, not given an exact
#                             target the model cannot infer from 0.5 s of
#                             lookahead (the script default is already 2.0)
#   AUGMENTED data            pitch and reverb were the top two degraders in the
#                             ablation and cost ~0.001 F1 to train against; the
#                             Pi hears a room over Bluetooth, not close-mic'd
#                             MAESTRO. Only the TRAIN split is augmented, so the
#                             xv number stays comparable to the author's.
#   TRAIN_BS=80 x 2.5 s       = 200 audio-seconds/step, matching the author's
#                             40 x 5.0. BATCH_NORM is 0.95 (95% weight on the
#                             CURRENT batch), so a smaller product makes the
#                             running stats used at inference much noisier.
#                             This is not a free knob.
#   RANDOM_SEED=1234          upstream defaults to random.randint(0,1e7) and
#                             never logs it. Pin it or runs are incomparable.
set -uo pipefail
cd /workspace/iamusica_training || exit 1
export PYTHONUNBUFFERED=1

MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
H5=${H5:-/data/h5-aug}
MAESTRO=${MAESTRO:-/data/maestro-aug}
RUNS=${RUNS:-/data/runs/production}
OUT=${OUT:-/workspace/artifacts/production}
SNAP=assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch

mkdir -p "$OUT"
# Checkpoints are written to the EPHEMERAL container disk and the run only
# copies out at the end; a pod death at hour five would take everything with
# it. Mirror them to the persistent volume every 5 minutes instead.
( while true; do
    cp -f "$RUNS"/model_snapshots/*.torch "$OUT/" 2>/dev/null
    sleep 300
  done ) &
COPY_OUT=$!
trap 'kill $COPY_OUT 2>/dev/null' EXIT

echo "[prod] augmented data, sounding-off head, seed 1234, ${MAX_STEPS:-6000} steps"
python 1_train_onsets_velocities.py \
  SNAPSHOT_INPATH="$SNAP" \
  MAESTRO_PATH="$MAESTRO" \
  HDF5_MEL_PATH="$H5/$MEL" HDF5_ROLL_PATH="$H5/$ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
  OUTPUT_DIR="$RUNS" \
  ENABLE_OFFSET_HEAD=True OFFSET_CAP_SECS=2.0 \
  TRAINABLE_COMPONENTS=all TRAINABLE_ONSETS=True \
  TRAIN_BATCH_SECS=2.5 TRAIN_BS=80 \
  LR_MAX=0.0002 MAX_STEPS="${MAX_STEPS:-6000}" RANDOM_SEED=1234 \
  XV_EVERY="${XV_EVERY:-1000}" TRAIN_LOG_EVERY=50 \
  || { echo '[prod] FAILED train'; exit 1; }

cp -f "$RUNS"/model_snapshots/*.torch "$OUT/" 2>/dev/null
CKPT=$(ls -t "$OUT"/*.torch 2>/dev/null | head -1)
echo "[prod] best checkpoint: ${CKPT:-NONE}"

if [[ -n $CKPT ]]; then
  echo '[prod] ONNX export'
  python export_offset_onnx.py --checkpoint "$CKPT" \
    --output "$OUT/production.onnx" 2>&1 | tail -14 \
    || echo '[prod] export failed; checkpoint is safe'
fi
echo '[prod] === DONE ==='
