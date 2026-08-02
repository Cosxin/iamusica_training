#!/usr/bin/env bash
set -euo pipefail

LOG=/workspace/logs/pedal-prepare.log
ARCHIVE=/root/maestro_hf/maestro-v3.0.0.zip
MAESTRO=/root/maestro-v3.0.0
mkdir -p /workspace/logs
exec >>"$LOG" 2>&1

echo "extract_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd /root
if test -s "$MAESTRO/maestro-v3.0.0.csv" && \
   test "$(find "$MAESTRO" -type f -name '*.wav' | wc -l)" -eq 1276 && \
   test "$(find "$MAESTRO" -type f -name '*.midi' | wc -l)" -eq 1276; then
  echo "extract_already_complete=true"
else
  test -s "$ARCHIVE"
  python - "$ARCHIVE" /root <<'PY'
import os
import sys
import zipfile

archive, destination = sys.argv[1:]
with zipfile.ZipFile(archive) as source:
    entries = source.infolist()
    for index, entry in enumerate(entries, 1):
        target = os.path.join(destination, entry.filename)
        if entry.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        if os.path.isfile(target) and os.path.getsize(target) == entry.file_size:
            continue
        source.extract(entry, destination)
        if index == 1 or index % 100 == 0:
            print(f"extract_progress={index}/{len(entries)}", flush=True)
PY
fi
test -s "$MAESTRO/maestro-v3.0.0.csv"
test "$(find "$MAESTRO" -type f -name '*.wav' | wc -l)" -eq 1276
test "$(find "$MAESTRO" -type f -name '*.midi' | wc -l)" -eq 1276
echo "extract_ok=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if test -e "$ARCHIVE"; then
  rm -f -- "$ARCHIVE"
fi
echo "archive_removed_after_validation=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd /workspace/iamusica_training
LIMIT=$(python - "$MAESTRO/maestro-v3.0.0.csv" <<'PY'
import csv
import sys

seen = set()
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    for index, row in enumerate(csv.DictReader(stream), 1):
        seen.add(row["split"])
        if seen == {"train", "validation", "test"}:
            print(index)
            break
    else:
        raise SystemExit("all splits not found")
PY
)
echo "smoke_limit=$LIMIT"

mkdir -p /root/data/h5-pedal-smoke
python 0a_maestro_to_hdf5mel.py \
  MAESTRO_INPATH="$MAESTRO" \
  OUTPUT_DIR=/root/data/h5-pedal-smoke \
  LIMIT="$LIMIT"
MEL='/root/data/h5-pedal-smoke/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='/root/data/h5-pedal-smoke/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
python 3_train_pedal.py \
  MAESTRO_PATH="$MAESTRO" \
  HDF5_MEL_PATH="$MEL" \
  HDF5_ROLL_PATH="$ROLL" \
  BASE_SNAPSHOT=models/keyup_framehead_step12000.torch \
  OUTPUT_DIR=/workspace/runs/pedal-smoke \
  ALLOW_PARTIAL_HDF5=true TRAIN_BS=2 DATALOADER_WORKERS=0 \
  MAX_STEPS=2 LOG_EVERY=1 SAVE_EVERY=100 AMP=false
test -s /workspace/runs/pedal-smoke/smoke-step-2.torch
echo "pedal_smoke_ok=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p /root/data/h5-clean
python 0a_maestro_to_hdf5mel.py \
  MAESTRO_INPATH="$MAESTRO" \
  OUTPUT_DIR=/root/data/h5-clean
test -s /root/data/h5-clean/'MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
test -s /root/data/h5-clean/'MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
echo "PEDAL_FORMAL_TRAINING_READY=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
