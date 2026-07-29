#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/iamusica_training}"
MAESTRO="${MAESTRO:-/root/maestro_extract/maestro-v3.0.0}"
SMOKE_ROOT="${SMOKE_ROOT:-/root/codec_smoke}"
EXTRACT_PID="${EXTRACT_PID:-}"
CHECKPOINT="$REPO/assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch"
MEL_NAME='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL_NAME='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to overwrite existing smoke root: $SMOKE_ROOT" >&2
  exit 2
fi
if [[ -n "$EXTRACT_PID" ]]; then
  while kill -0 "$EXTRACT_PID" 2>/dev/null; do
    echo "Waiting for MAESTRO extraction PID $EXTRACT_PID..."
    sleep 30
  done
fi
test -s "$MAESTRO/maestro-v3.0.0.csv"
test -s "$CHECKPOINT"

LIMIT="$(python - "$MAESTRO/maestro-v3.0.0.csv" <<'PY'
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
        raise SystemExit("CSV does not contain every split")
PY
)"
echo "Using first $LIMIT files (minimum prefix containing every split)"

mkdir -p "$SMOKE_ROOT"
cd "$REPO"
git rev-parse HEAD > "$SMOKE_ROOT/git_sha.txt"
python prep_codec_render.py --selftest
python prep_codec_render.py MAESTRO_INPATH="$MAESTRO" VARIANT=aptx \
  OUTPUT_ROOT="$SMOKE_ROOT/maestro-aptx" LIMIT="$LIMIT"
python prep_codec_render.py MAESTRO_INPATH="$MAESTRO" VARIANT=sbc_low \
  OUTPUT_ROOT="$SMOKE_ROOT/maestro-sbc" LIMIT="$LIMIT"

for variant in clean aptx sbc; do
  case "$variant" in
    clean) input="$MAESTRO" ;;
    aptx) input="$SMOKE_ROOT/maestro-aptx" ;;
    sbc) input="$SMOKE_ROOT/maestro-sbc" ;;
  esac
  python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH="$input" \
    OUTPUT_DIR="$SMOKE_ROOT/h5-$variant" LIMIT="$LIMIT"
done

mkdir -p "$SMOKE_ROOT/h5-mixed"
python merge_codec_hdf5.py \
  --variant clean "$SMOKE_ROOT/h5-clean/$MEL_NAME" "$SMOKE_ROOT/h5-clean/$ROLL_NAME" \
  --variant aptx "$SMOKE_ROOT/h5-aptx/$MEL_NAME" "$SMOKE_ROOT/h5-aptx/$ROLL_NAME" \
  --variant sbc "$SMOKE_ROOT/h5-sbc/$MEL_NAME" "$SMOKE_ROOT/h5-sbc/$ROLL_NAME" \
  --out-mel "$SMOKE_ROOT/h5-mixed/$MEL_NAME" \
  --out-roll "$SMOKE_ROOT/h5-mixed/$ROLL_NAME" \
  --provenance "$SMOKE_ROOT/h5-mixed/provenance.json"

python 1_train_onsets_velocities.py \
  MAESTRO_PATH="$MAESTRO" \
  HDF5_MEL_PATH="$SMOKE_ROOT/h5-mixed/$MEL_NAME" \
  HDF5_ROLL_PATH="$SMOKE_ROOT/h5-mixed/$ROLL_NAME" \
  XV_HDF5_MEL_PATH="$SMOKE_ROOT/h5-clean/$MEL_NAME" \
  XV_HDF5_ROLL_PATH="$SMOKE_ROOT/h5-clean/$ROLL_NAME" \
  SNAPSHOT_INPATH="$CHECKPOINT" OUTPUT_DIR="$SMOKE_ROOT/train" \
  ALLOW_PARTIAL_HDF5=true \
  RANDOM_SEED=20260729 TRAIN_BS=1 DATALOADER_WORKERS=0 \
  TRAIN_BATCH_SECS=2 MAX_STEPS=2 XV_EVERY=999 TRAIN_LOG_EVERY=1

MODEL="$(find "$SMOKE_ROOT/train/model_snapshots" -type f -name '*.torch' |
  sort | tail -1)"
test -s "$MODEL"
python 2_eval_onsets_velocities.py \
  MAESTRO_PATH="$MAESTRO" \
  HDF5_MEL_PATH="$SMOKE_ROOT/h5-clean/$MEL_NAME" \
  HDF5_ROLL_PATH="$SMOKE_ROOT/h5-clean/$ROLL_NAME" \
  SNAPSHOT_INPATH="$MODEL" OUTPUT_DIR="$SMOKE_ROOT/eval-clean" \
  XV_TAKE_ONE_EVERY=1 'SEARCH_THRESHOLDS=[0.5]' 'SEARCH_SHIFTS=[-0.01]' \
  RESULTS_JSON="$SMOKE_ROOT/log/smoke-clean.json" \
  ALLOW_PARTIAL_HDF5=true \
  RUN_NAME=codec_smoke DATASET_VARIANT=clean

python -m pytest -q tests/test_codec_finetune.py
echo "SMOKE_E2E_OK model=$MODEL json=$SMOKE_ROOT/log/smoke-clean.json"
