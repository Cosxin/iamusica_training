#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/iamusica_training}"
MAESTRO="${MAESTRO:-/root/maestro_extract/maestro-v3.0.0}"
RUN_ROOT="${RUN_ROOT:-/root/full_pipeline}"

test -s "$RUN_ROOT/log/stage1-complete.txt"
cd "$REPO"
for variant in aptx; do
  short_name="${variant/_low/}"
  out="$RUN_ROOT/h5-$short_name"
  if [[ -e "$out" ]]; then
    echo "Refusing to overwrite existing output: $out" >&2
    exit 2
  fi
  mkdir -p "$out"
  echo "variant=$variant started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" |
    tee "$RUN_ROOT/log/stage2-$short_name-start.txt"
  python 0a_maestro_to_hdf5mel.py \
    MAESTRO_INPATH="$MAESTRO" OUTPUT_DIR="$out" \
    CODEC_VARIANT="$variant"
  {
    echo "variant=$variant completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    find "$out" -maxdepth 1 -type f -printf '%f %s bytes\n'
    df -h /
  } > "$RUN_ROOT/log/stage2-$short_name-complete.txt"
done
echo "FULL_STAGE2_OK"
