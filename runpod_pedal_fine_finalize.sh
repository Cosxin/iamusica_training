#!/usr/bin/env bash
set -euo pipefail

sweep_pid=${1:?sweep PID required}
while kill -0 "$sweep_pid" 2>/dev/null; do sleep 30; done

cd /workspace/iamusica_training
root=/workspace/runs/pedal-fine-lookahead-20260801
selected="$root/selected.json"
if [[ ! -f "$selected" ]]; then
  echo "Fine sweep ended without selected.json" >&2
  exit 1
fi

read -r run checkpoint threshold < <(python - "$selected" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
print(x['run'], x['checkpoint'], x['threshold'])
PY
)

python 4_eval_pedal.py \
  MAESTRO_PATH=/root/maestro-v3.0.0 \
  HDF5_MEL_PATH='/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5' \
  HDF5_ROLL_PATH='/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5' \
  CHECKPOINT="$checkpoint" SPLIT=validation \
  CHUNK_SECS=8.0 OVERLAP_SECS=4.0 DECODER=regression \
  STATE_THRESHOLD=0.5 EVENT_THRESHOLD="$threshold" NMS_RADIUS=3 \
  RESULTS_JSON="$root/$run/eval-validation-full-selected.json"

echo PEDAL_FINE_LOOKAHEAD_FULL_VALIDATION_COMPLETE
