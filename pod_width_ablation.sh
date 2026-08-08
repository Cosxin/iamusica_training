#!/usr/bin/env bash
# P3/P4: onset target-width ablation. ONE GPU DAY, unattended.
#
# The pre-registered causal test (RESEARCH_PLAN_repeat_resolution.md). Every
# model that avoids the repeat-fusion problem changed several things at once:
# Kong changed targets AND frame rate AND architecture; Mobile-AMT those plus
# augmentation; Hu et al. target width AND causality. Nobody has varied target
# width alone. This does, warm-started from the author checkpoint on a fixed
# backbone, with everything else held constant.
#
#   P3 narrower targets -> better 75-125 ms recall, monotone in width, with
#      clean XV F1 within seed spread of the width-3 control
#   P4 width-1 may cost clean accuracy or timing robustness; if it does, the
#      claim becomes "width is a resolution/robustness dial", not "narrow wins"
#
# Decision threshold, pre-registered: beat 0.8485 recall at 75-125 ms on
# BM-REPEAT at comparable precision (>= 0.9844 - CI) or do not port.
#
#   tmux new -s abl
#   bash pod_width_ablation.sh 2>&1 | tee /workspace/ablation.log
set -uo pipefail

WS=/workspace
DATA=${DATA:-/data}
REPO=$WS/iamusica_training
H5=${H5:-$DATA/h5-clean}
RUNS=$DATA/runs/width_ablation
OUT=$WS/artifacts/width_ablation
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
SNAP=assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch
WIDTHS=${WIDTHS:-"3 2 1"}          # control FIRST: if it does not reproduce, stop
SEEDS=${SEEDS:-"1234 2025 777"}
STEPS=${STEPS:-1500}

mkdir -p "$OUT" "$RUNS"
cd "$REPO" || exit 1
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Checkpoints land on the EPHEMERAL container disk; mirror them continuously.
( while true; do
    find "$RUNS" -name '*.torch' -exec cp -f {} "$OUT/" \; 2>/dev/null
    sleep 300
  done ) &
trap 'kill %1 2>/dev/null' EXIT

# ---- preflight: never spend GPU hours on an unverified recipe -------------
log "recipe verification"
python verify_training_recipe.py || { log "RECIPE VERIFY FAILED"; exit 1; }

log "reproduction gate: the author checkpoint must score 0.9675 through THIS pipeline"
python 1_train_onsets_velocities.py \
  SNAPSHOT_INPATH="$SNAP" MAESTRO_PATH=/data/maestro/maestro-v3.0.0 \
  HDF5_MEL_PATH="$H5/$MEL" HDF5_ROLL_PATH="$H5/$ROLL" \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True OUTPUT_DIR="$DATA/runs/gate" \
  ENABLE_OFFSET_HEAD=False TRAINABLE_COMPONENTS=all \
  TRAIN_BS=80 TRAIN_BATCH_SECS=2.5 MAX_STEPS=1 XV_EVERY=1 \
  2>&1 | tee "$OUT/gate.log" | grep -E 'XV_SUMMARY' || true
GATE=$(grep -o '"best_f1_o": [0-9.]*' "$OUT/gate.log" | tail -1 | grep -o '[0-9.]*$')
python - "$GATE" <<'PY' || { echo "GATE FAILED - do not train"; exit 1; }
import sys
f1 = float(sys.argv[1])
print(f"[gate] {f1:.5f} vs published 0.9675 -> {'PASS' if abs(f1-0.9675)<0.002 else 'FAIL'}")
raise SystemExit(0 if abs(f1 - 0.9675) < 0.002 else 1)
PY

# ---- the ablation --------------------------------------------------------
for W in $WIDTHS; do
  for S in $SEEDS; do
    TAG="w${W}_s${S}"
    if [[ -f "$OUT/$TAG.done" ]]; then log "skip $TAG (done)"; continue; fi
    log "=== width=$W seed=$S ($STEPS steps) ==="
    python 1_train_onsets_velocities.py \
      SNAPSHOT_INPATH="$SNAP" MAESTRO_PATH=/data/maestro/maestro-v3.0.0 \
      HDF5_MEL_PATH="$H5/$MEL" HDF5_ROLL_PATH="$H5/$ROLL" \
      DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
      OUTPUT_DIR="$RUNS/$TAG" \
      TARGET_WIDTH_FRAMES="$W" \
      ENABLE_OFFSET_HEAD=False \
      TRAINABLE_COMPONENTS=all TRAINABLE_ONSETS=True \
      TRAIN_BATCH_SECS=2.5 TRAIN_BS=80 \
      LR_MAX=0.0002 MAX_STEPS="$STEPS" RANDOM_SEED="$S" \
      XV_EVERY=500 TRAIN_LOG_EVERY=50 \
      2>&1 | tee "$OUT/$TAG.log" | grep -E 'XV_SUMMARY|Traceback|Error' || true
    find "$RUNS/$TAG" -name '*.torch' -exec cp -f {} "$OUT/" \; 2>/dev/null
    # Export the best checkpoint so BM-REPEAT can score it off-pod.
    CK=$(ls -t "$RUNS/$TAG"/model_snapshots/*f1=*.torch 2>/dev/null | head -1)
    if [[ -n $CK ]]; then
      python export_offset_onnx.py --checkpoint "$CK" \
        --output "$OUT/$TAG.onnx" 2>&1 | tail -4
    fi
    touch "$OUT/$TAG.done"
    log "--- $TAG complete ---"
  done
done

log "=== ALL RUNS COMPLETE ==="
grep -H -o '"best_f1_o": [0-9.]*' "$OUT"/w*_s*.log 2>/dev/null | sed 's|.*/||' | sort
ls -la "$OUT"/*.onnx 2>/dev/null
