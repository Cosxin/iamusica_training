#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/workspace/iamusica_training}"
MAESTRO="${MAESTRO:-/root/maestro_extract/maestro-v3.0.0}"
RUN_ROOT="${RUN_ROOT:-/root/full_pipeline}"
OUT="$RUN_ROOT/h5-clean"

test -s "$MAESTRO/maestro-v3.0.0.csv"
test "$(find "$MAESTRO" -type f -name '*.wav' | wc -l)" -eq 1276
test "$(find "$MAESTRO" -type f -name '*.midi' | wc -l)" -eq 1276
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 2
fi

mkdir -p "$OUT" "$RUN_ROOT/log"
cd "$REPO"
{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python -c 'import torch; print("torch="+torch.__version__); print("cuda="+str(torch.version.cuda))'
  ffmpeg -version | head -1
  df -h /
} > "$RUN_ROOT/log/stage1-environment.txt"

python 0a_maestro_to_hdf5mel.py \
  MAESTRO_INPATH="$MAESTRO" OUTPUT_DIR="$OUT"

{
  echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  find "$OUT" -maxdepth 1 -type f -printf '%f %s bytes\n'
  df -h /
} > "$RUN_ROOT/log/stage1-complete.txt"
echo "FULL_STAGE1_OK"
