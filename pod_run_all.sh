#!/usr/bin/env bash
# Unattended chain: MAESTRO -> augmented render -> 0a -> train -> eval -> export.
#
# Designed for a ~6 h window with nobody watching, so:
#  * every phase writes a DONE marker under $STATE and is skipped on re-run
#  * ONE rendered tree and ONE 0a pass (augmenting only the train split), rather
#    than three sequential 0a passes which alone would exceed the window
#  * training is the LAST thing that can fail; if anything upstream breaks the
#    log says exactly which phase and nothing silently trains on wrong data
#  * artifacts land on /workspace (persistent volume), never /root
#
#   tmux new -s run
#   bash pod_run_all.sh 2>&1 | tee -a /workspace/run_all.log
set -uo pipefail

# /workspace is a NETWORK VOLUME with a ~20 GB quota (df reports the whole
# 1.8 P cluster, which is misleading -- the download died at 19 GB with
# "Disk quota exceeded"). The container disk at / has 250 GB free, so all bulk
# data lives there and only small, durable artifacts go to /workspace.
# DATA is ephemeral: checkpoints are copied to $OUT as soon as they exist.
WS=/workspace
DATA=${DATA:-/data}
REPO=$WS/iamusica_training
MAESTRO=$DATA/maestro/maestro-v3.0.0
AUG_ROOT=$DATA/maestro-aug
H5=$DATA/h5-aug
RUNS=$DATA/runs/offset_head
OUT=$WS/artifacts
STATE=$WS/.bootstrap
mkdir -p "$DATA"
MEL="MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
ROLL="MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
BASE_CKPT="$REPO/assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch"
mkdir -p "$STATE" "$OUT"

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "FAILED at phase: $1"; echo "$1" > "$STATE/FAILED"; exit 1; }
done_with() { [[ -f "$STATE/$1.done" ]]; }
mark()      { touch "$STATE/$1.done"; log "=== $1 DONE ==="; }

cd "$REPO" || fail cd

# ── 1. MAESTRO ──────────────────────────────────────────────────────────────
if ! done_with maestro; then
  log "MAESTRO download -> $DATA (container disk, 250 GB)"
  mkdir -p "$DATA/maestro"
  # HF_HOME too: the hub cache would otherwise also land on the quota'd volume.
  export HF_HOME="$DATA/hf_home"
  HF_HUB_DISABLE_PROGRESS_BARS=1 hf download ddPn08/maestro-v3.0.0 \
    --repo-type dataset --local-dir "$DATA/maestro/hf" >/dev/null || fail maestro
  CSV=$(find "$DATA/maestro/hf" -name 'maestro-v3.0.0.csv' | head -1)
  [[ -n $CSV ]] || fail maestro
  ln -sfn "$(dirname "$CSV")" "$MAESTRO"
  WAVS=$(find -L "$MAESTRO" -name '*.wav' | wc -l)
  log "wav files: $WAVS (expect 1276)"
  [[ $WAVS -ge 1276 ]] || fail maestro
  mark maestro
fi

# ── 2. augmented render (train split only) ──────────────────────────────────
# Pitch and reverb were the only two terms the ablation found worth training
# against. Test/validation pass through clean so evaluation stays honest.
if ! done_with render; then
  log "augmentation render (train split; test/val clean)"
  python prep_augment_render.py \
    MAESTRO_INPATH="$MAESTRO" VARIANT=mixed SPLITS=train \
    OUTPUT_ROOT="$AUG_ROOT" JOBS="$(nproc)" || fail render
  # 0a asserts wav >= midi per file and dies on the first violation, hours
  # into the run. Check it here instead, where it costs seconds.
  python - "$AUG_ROOT" <<'PYEOF' || fail render_check
import csv, os, sys, wave, mido
root = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(root, "maestro-v3.0.0.csv"),
                                encoding="utf-8")))
# Only RENDERED files are checked. Clean pass-throughs are symlinks to the
# untouched dataset, and ~14 MAESTRO files legitimately have their MIDI meta
# end a few ms past the wav -- flagging those re-litigates the dataset instead
# of testing our own resampling. Tolerance covers float/tick rounding.
bad, checked = [], 0
for row in rows:
    wav = os.path.join(root, row["audio_filename"])
    mid = os.path.join(root, row["midi_filename"])
    if not (os.path.exists(wav) and os.path.exists(mid)):
        continue
    if os.path.islink(wav):
        continue
    checked += 1
    with wave.open(wav) as handle:
        seconds = handle.getnframes() / handle.getframerate()
    if seconds + 0.05 < mido.MidiFile(mid).length:
        bad.append(os.path.basename(wav))
print(f"[check] {len(bad)} of {checked} RENDERED files have wav shorter than midi")
if bad:
    print("[check] first:", bad[:5])
    raise SystemExit(1)
PYEOF
  mark render
fi

# ── 3. 0a: mel + sustain-extended roll ──────────────────────────────────────
# Default 0a args give exactly what the offset head needs (extendsus=True).
# KEEP $H5 on the volume: rebuilding costs hours.
if ! done_with h5; then
  # 0a's mel STFT honours DEVICE (see 0a:134/185). It defaults to cpu, which
  # left the GPU at 0% during the longest phase of the run. Use cuda and fall
  # back to cpu if it OOMs, so an unattended run cannot be stranded by it.
  log "0a build (mel on GPU, fallback to CPU)"
  export OMP_NUM_THREADS=$(nproc) MKL_NUM_THREADS=$(nproc)
  if ! python 0a_maestro_to_hdf5mel.py \
      MAESTRO_INPATH="$AUG_ROOT" OUTPUT_DIR="$H5" DEVICE=cuda; then
    log "GPU 0a failed; retrying on CPU from scratch"
    rm -rf "$H5"
    python 0a_maestro_to_hdf5mel.py \
      MAESTRO_INPATH="$AUG_ROOT" OUTPUT_DIR="$H5" DEVICE=cpu || fail h5
  fi
  [[ -s "$H5/$MEL" && -s "$H5/$ROLL" ]] || fail h5
  ls -la "$H5"
  # WAVs are only an input to 0a (training reads mels from the HDF5), so they
  # can go -- but the CSV and MIDI must STAY: the trainer reads the CSV to
  # enumerate splits and the MIDI for cross-validation ground truth. Deleting
  # the whole tree here cost a run with
  # "FileNotFoundError: /data/maestro-aug/maestro-v3.0.0.csv".
  du -sh "$DATA"/* 2>/dev/null
  find "$AUG_ROOT" -name '*.wav' -delete 2>/dev/null || true
  rm -rf "$DATA/hf_home"
  log "freed wavs (kept csv+midi); remaining:"; df -h "$DATA" | tail -1
  mark h5
fi

# ── 4. train ────────────────────────────────────────────────────────────────
# Warm-start from the author's baseline: our key-up checkpoint was never pushed
# and is gone, and the author's backbone measured BETTER anyway (0.9708 vs
# 0.9703 onset F1 -- production is a BatchNorm-drifted copy). TRAINABLE=all so
# the augmentation can actually reshape the backbone features it targets;
# onset+velocity losses stay active to anchor them.
if ! done_with train; then
  log "train"
  mkdir -p "$RUNS"
  # Resume is OPT-IN (RESUME=1). It used to be automatic "newest snapshot
  # wins", which silently turned a fresh experiment into a continuation: a run
  # meant to test a corrected augmentation recipe instead warm-started from the
  # previous run's damaged weights, so the fix was never actually measured.
  # Default is the author's baseline, which is what a new experiment wants.
  if [[ "${RESUME:-0}" == "1" ]]; then
    LATEST=$(ls -1t "$RUNS"/model_snapshots/*.torch 2>/dev/null | head -1)
  else
    LATEST=""
  fi
  SNAP=${SNAP_OVERRIDE:-${LATEST:-$BASE_CKPT}}
  STEPS=${MAX_STEPS:-12000}
  log "warm-start: $SNAP"
  log "max steps : $STEPS"
  [[ -s $SNAP ]] || fail train
  python 1_train_onsets_velocities.py \
    SNAPSHOT_INPATH="$SNAP" \
    MAESTRO_PATH="$AUG_ROOT" \
    HDF5_MEL_PATH="$H5/$MEL" \
    HDF5_ROLL_PATH="$H5/$ROLL" \
    DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
    OUTPUT_DIR="$RUNS" \
    ENABLE_OFFSET_HEAD=True \
    TRAINABLE_COMPONENTS=all TRAINABLE_ONSETS=True \
    TRAIN_BATCH_SECS=2.5 TRAIN_BS="${TRAIN_BS_OVERRIDE:-80}" \
    LR_MAX="${LR_MAX:-0.0002}" MAX_STEPS="$STEPS" \
    XV_EVERY="${XV_EVERY:-2000}" TRAIN_LOG_EVERY=50 \
    || fail train
  # Checkpoints live on the EPHEMERAL container disk; copy to the persistent
  # volume immediately. They are ~15 MB, well inside the 20 GB quota.
  find "$RUNS" -name '*.torch' -exec cp -f {} "$OUT/" \; 2>/dev/null || true
  mark train
fi

# ── 5. pick the best checkpoint ─────────────────────────────────────────────
CKPT=$(ls -t "$RUNS"/**/*.torch "$RUNS"/*.torch 2>/dev/null | head -1)
log "checkpoint: ${CKPT:-NONE}"
[[ -n $CKPT ]] || fail checkpoint

# ── 6. export ONNX ──────────────────────────────────────────────────────────
# The exporter detects which heads the checkpoint carries and adapts its output
# list, so it works for onset-only and onset+offset alike. A failure here does
# NOT lose the training run.
if ! done_with export; then
  log "ONNX export"
  python export_offset_onnx.py --checkpoint "$CKPT" \
    --output "$OUT/offset_head.onnx" 2>&1 | tail -12 \
    || log "export failed; checkpoint is safe"
  mark export
fi

# ── 7. evaluate ─────────────────────────────────────────────────────────────
# Repeated-note IOI is the product metric: whether a same-pitch reattack is
# detected as a second note. Note-with-offset F1 is deliberately NOT used --
# see NOTES_frame_head.md. Test split is clean (see phase 2).
if ! done_with eval; then
  log "eval: repeated-note IOI (the product metric)"
  python eval_repeats.py SNAPSHOT_INPATH="$CKPT" MAESTRO_PATH="$AUG_ROOT" \
    HDF5_MEL_PATH="$H5/$MEL" HDF5_ROLL_PATH="$H5/$ROLL" \
    DATASET_VARIANT=clean LIMIT=40 \
    RESULTS_JSON="$OUT/eval_repeats.json" 2>&1 | tail -8 \
    || log "eval_repeats failed"
  mark eval
fi

# ── 8. summary ──────────────────────────────────────────────────────────────
cp -f "$CKPT" "$OUT/" 2>/dev/null || true
{
  echo "=== RUN SUMMARY $(date -u +%FT%TZ) ==="
  echo "checkpoint : $CKPT"
  echo "artifacts  : $OUT"
  ls -la "$OUT"
  echo
  echo "--- onset F1 progression (checkpoint names) ---"
  ls -t "$RUNS"/**/*.torch "$RUNS"/*.torch 2>/dev/null | head -8
  echo
  for json in "$OUT"/eval_repeats.json; do
    [[ -s $json ]] && { echo "--- $(basename "$json") ---"; \
      python -c "import json,sys;d=json.load(open(sys.argv[1]));print({k:v for k,v in d.items() if not isinstance(v,list)})" "$json"; }
  done
} | tee "$OUT/SUMMARY.txt"

log "=== ALL PHASES COMPLETE ==="
