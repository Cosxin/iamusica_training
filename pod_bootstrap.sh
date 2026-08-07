#!/usr/bin/env bash
# One-shot pod bootstrap: bare RunPod container -> ready to train.
#
#   bash <(curl -sL <this file>)      # or scp it over and: bash pod_bootstrap.sh
#
# Brings up the environment, fetches MAESTRO, builds the clean HDF5, and then
# GATES on a correctness check before declaring success. Every phase writes a
# marker under $STATE and is skipped on re-run, so a dropped SSH link costs
# nothing -- just run it again.
#
# Takes roughly 75 min on an A4000-class pod:
#   deps 3 min | download 30 min | 0a 30 min | verify 6 min | gate 5 min
#
# When it finishes, train with:
#   bash pod_bootstrap.sh --print-train-cmd
#
# ── why the layout is the way it is ─────────────────────────────────────────
# /workspace is a NETWORK VOLUME with a ~20 GB quota. `df` reports the whole
# multi-petabyte cluster, which is actively misleading -- a MAESTRO download
# died at 19 GB with "Disk quota exceeded" after looking like it had room.
# The container disk at / has ~250 GB. So: bulk data on /data, and only small
# durable artifacts (checkpoints, logs, ONNX) on /workspace.
#
# The container disk is EPHEMERAL. Anything you want to survive the pod must be
# copied to /workspace the moment it exists, not at the end of the run.
set -uo pipefail

WS=/workspace
DATA=${DATA:-/data}
REPO=$WS/iamusica_training
GIT_URL=${GIT_URL:-https://github.com/Cosxin/iamusica_training.git}
GIT_REF=${GIT_REF:-main}
MAESTRO=$DATA/maestro/maestro-v3.0.0
H5=${H5:-$DATA/h5-clean}
OUT=$WS/artifacts
STATE=$WS/.bootstrap
MEL="MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
ROLL="MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
BASE_CKPT="assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "FAILED at phase: $1"; echo "$1" > "$STATE/FAILED"; exit 1; }
have() { [[ -f "$STATE/$1.done" ]]; }
mark() { touch "$STATE/$1.done"; log "=== $1 OK ==="; }

# ── --print-train-cmd: emit the ready-to-paste training command and exit ────
if [[ "${1:-}" == "--print-train-cmd" ]]; then
  cat <<CMD
cd $REPO && tmux new -s run
python 1_train_onsets_velocities.py \\
  SNAPSHOT_INPATH=$BASE_CKPT \\
  MAESTRO_PATH=$MAESTRO \\
  HDF5_MEL_PATH='$H5/$MEL' \\
  HDF5_ROLL_PATH='$H5/$ROLL' \\
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True OUTPUT_DIR=$DATA/runs/myrun \\
  TRAINABLE_COMPONENTS=all TRAINABLE_ONSETS=True \\
  TRAIN_BS=80 TRAIN_BATCH_SECS=2.5 \\
  LR_MAX=0.0002 MAX_STEPS=4000 XV_EVERY=1000 TRAIN_LOG_EVERY=50 \\
  2>&1 | tee $WS/train.log

# TRAIN_BS x TRAIN_BATCH_SECS must stay at 200 audio-seconds per step, which is
# what the author used (40 x 5.0). BATCH_NORM is 0.95, i.e. 95% weight on the
# CURRENT batch, so the running statistics used at inference are dominated by
# whatever the last few batches looked like. Training at 16 x 2.5 = 40 s/step
# made those statistics 5x noisier and measurably degraded the model.
#
# Checkpoints land on the ephemeral container disk. Copy them out as they
# appear, do not wait for the run to end:
#   watch -n300 'cp -f $DATA/runs/myrun/model_snapshots/*.torch $OUT/'
CMD
  exit 0
fi

mkdir -p "$DATA" "$STATE" "$OUT"

# ── 1. system + python deps ─────────────────────────────────────────────────
if ! have deps; then
  log "installing dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  # ffmpeg: audio decode for augmentation rendering and evaluation
  # tmux:   the SSH link to these pods drops; never run a long job without it
  apt-get install -y -qq ffmpeg tmux git rsync >/dev/null 2>&1 || fail deps
  # torch ships with the RunPod image -- do NOT reinstall it, pip will happily
  # replace a working CUDA build with a CPU wheel.
  python -c 'import torch; assert torch.cuda.is_available()' \
    || fail "deps (no CUDA torch in image)"
  pip install -q \
    omegaconf h5py pandas mido mir_eval coloredlogs parse \
    huggingface_hub[cli] av onnx onnxruntime websockets scipy \
    || fail deps
  mark deps
fi
log "torch $(python -c 'import torch;print(torch.__version__)')  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ── 2. repository ───────────────────────────────────────────────────────────
if ! have repo; then
  log "cloning $GIT_URL@$GIT_REF"
  rm -rf "$REPO"
  git clone -q "$GIT_URL" "$REPO" || fail repo
  git -C "$REPO" checkout -q "$GIT_REF" || fail repo
  [[ -s "$REPO/$BASE_CKPT" ]] || fail "repo (author checkpoint missing from assets/)"
  mark repo
fi
cd "$REPO" || fail cd

# ── 3. MAESTRO v3 ───────────────────────────────────────────────────────────
# The official Google download is slow and unresumable. The HuggingFace mirror
# is fast and resumes on its own, which matters because these links drop.
if ! have maestro; then
  log "downloading MAESTRO v3 (~120 GB) -> $DATA"
  mkdir -p "$DATA/maestro"
  # HF_HOME too, or the hub's own cache lands on the quota'd volume and the
  # download dies at 19 GB having written everything twice.
  export HF_HOME="$DATA/hf_home"
  HF_HUB_DISABLE_PROGRESS_BARS=1 hf download ddPn08/maestro-v3.0.0 \
    --repo-type dataset --local-dir "$DATA/maestro/hf" >/dev/null || fail maestro
  CSV=$(find "$DATA/maestro/hf" -name 'maestro-v3.0.0.csv' | head -1)
  [[ -n $CSV ]] || fail "maestro (csv not found in download)"
  ln -sfn "$(dirname "$CSV")" "$MAESTRO"
  WAVS=$(find -L "$MAESTRO" -name '*.wav' | wc -l)
  log "wav files: $WAVS (expect 1276)"
  [[ $WAVS -ge 1276 ]] || fail "maestro (incomplete: $WAVS/1276 wavs)"
  mark maestro
fi

# ── 4. 0a: log-mel + sustain-extended piano roll ────────────────────────────
# Defaults are exactly what the model wants (extendsus=True). Expect a stream
# of "onset/offset collision" and "beg==end" warnings -- the upstream README
# documents these as normal for MAESTRO and harmless, since rolls are never
# used for evaluation.
if ! have h5; then
  log "0a build -> $H5 (mel on GPU)"
  export OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc)
  # 0a's STFT honours DEVICE and defaults to cpu, which leaves the GPU at 0%
  # through the longest phase. CPU fallback so an unattended run can't strand.
  if ! python 0a_maestro_to_hdf5mel.py \
      MAESTRO_INPATH="$MAESTRO" OUTPUT_DIR="$H5" DEVICE=cuda; then
    log "GPU 0a failed; retrying on CPU from scratch"
    rm -rf "$H5"
    python 0a_maestro_to_hdf5mel.py \
      MAESTRO_INPATH="$MAESTRO" OUTPUT_DIR="$H5" DEVICE=cpu || fail h5
  fi
  [[ -s "$H5/$MEL" && -s "$H5/$ROLL" ]] || fail "h5 (outputs missing)"
  log "hdf5: $(du -sh "$H5" | cut -f1)"
  mark h5
fi

# ── 5. recipe self-check ────────────────────────────────────────────────────
# 14 assertions on the training code itself: that "frozen" components really
# have requires_grad off AND their BatchNorm buffers stop moving, that the
# offset head's censored targets are right, that loss weights are what we
# think. Catches the class of bug that silently costs accuracy and is only
# visible hours later. Cheap -- run it before every GPU lease.
if ! have verify; then
  log "verifying training recipe"
  python verify_training_recipe.py || fail verify
  mark verify
fi

# ── 6. THE GATE: does this pipeline reproduce the published number? ─────────
# Scores the author's pristine checkpoint through OUR cross-validation path
# with zero optimizer steps (XV runs before the first step, so MAX_STEPS=1 and
# XV_EVERY=1 measures the weights exactly as loaded). Both heads off so the
# state dict loads strictly.
#
# It must print 0.9675 / 0.9480 -- the numbers in the checkpoint's own
# filename. If it does, the HDF5 prep, mel, roll quantization, peak-picking,
# mir_eval call and threshold sweep are all faithful, and any accuracy change
# you measure afterwards is a real property of your training rather than an
# artifact of the data pipeline. Verified exact to four decimals 2026-08-07.
#
# This is the single highest-value five minutes on a fresh pod: it converts
# "our model scores 0.957, is that bad?" from an unanswerable question into a
# measurement, BEFORE you spend the GPU hours.
if ! have gate; then
  log "GATE: reproducing published baseline (expect 0.9675 / 0.9480)"
  python 1_train_onsets_velocities.py \
    SNAPSHOT_INPATH="$BASE_CKPT" MAESTRO_PATH="$MAESTRO" \
    HDF5_MEL_PATH="$H5/$MEL" HDF5_ROLL_PATH="$H5/$ROLL" \
    DEVICE=cuda ALLOW_PARTIAL_HDF5=True OUTPUT_DIR="$DATA/runs/gate" \
    ENABLE_OFFSET_HEAD=False \
    TRAINABLE_COMPONENTS=all TRAIN_BS=80 TRAIN_BATCH_SECS=2.5 \
    MAX_STEPS=1 XV_EVERY=1 2>&1 | tee "$WS/gate.log" | grep -E 'XV_SUMMARY'
  F1=$(grep -o '"best_f1_o": [0-9.]*' "$WS/gate.log" | tail -1 | grep -o '[0-9.]*$')
  [[ -n $F1 ]] || fail "gate (no XV_SUMMARY produced)"
  log "GATE onset F1 = $F1  (published 0.9675)"
  python - "$F1" <<'PYEOF' || fail "gate (baseline NOT reproduced -- do not train until this is understood)"
import sys
f1 = float(sys.argv[1])
ok = abs(f1 - 0.9675) < 0.002
print(f"[gate] {'PASS' if ok else 'FAIL'}: {f1:.4f} vs published 0.9675")
raise SystemExit(0 if ok else 1)
PYEOF
  mark gate
fi

log "READY. training command:"
bash "$0" --print-train-cmd
