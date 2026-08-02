#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training

preprocess_pid="${PREPROCESS_PID:-72605}"
mel='/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5'
roll='/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5'
run=/workspace/runs/pedal-edge-10ms-radius7-10ep-500ms-20260802

while kill -0 "$preprocess_pid" 2>/dev/null; do
  sleep 30
done

python - <<'PY'
import h5py
paths = (
    '/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5',
    '/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5',
)
with h5py.File(paths[0]) as mel, h5py.File(paths[1]) as roll:
    assert mel['data'].shape[1] == roll['data'].shape[1]
    assert len(mel['metadata']) == len(roll['metadata'])
    assert list(mel['metadata']) == list(roll['metadata'])
    assert (mel['data_idxs'][:] == roll['data_idxs'][:]).all()
    print({'mel_shape': mel['data'].shape, 'roll_shape': roll['data'].shape,
           'metadata': len(mel['metadata'])})
PY

python -m pytest tests/test_mobile_pedal.py tests/test_pedal.py -q

selected=''
for batch in 8 6 4 2; do
  if TRAIN_BS="$batch" ./runpod_pedal_edge_10ms.sh smoke \
      > "/workspace/pedal-edge-10ms-smoke-bs${batch}.log" 2>&1; then
    selected="$batch"
    break
  fi
done
test -n "$selected"
echo "EDGE_10MS_SELECTED_BATCH=$selected"

chunks=$(python - <<'PY' | tail -n 1
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestroChunks
metadata = MetaMAESTROv3('/root/maestro-v3.0.0', splits=['train'],
                         years=MetaMAESTROv3.ALL_YEARS)
base = MelMaestroChunks(
    '/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5',
    '/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5',
    600, 600, *(item[0] for item in metadata.data), with_oob=False,
    as_torch_tensors=True)
print(len(base))
PY
)
steps_per_epoch=$(( (chunks + selected - 1) / selected ))
target_steps=$(( steps_per_epoch * 10 ))
target_steps=$(( (target_steps + 999) / 1000 * 1000 ))
echo "EDGE_10MS_EPOCH_GEOMETRY chunks=$chunks batch=$selected steps_per_epoch=$steps_per_epoch target_epochs=10 target_steps=$target_steps"

TRAIN_BS="$selected" MAX_STEPS="$target_steps" ./runpod_pedal_edge_10ms.sh formal \
  > /workspace/pedal-edge-10ms-formal.log 2>&1

mkdir -p results_eval/edge-10ms /workspace/cache
checkpoints=$(printf '%s\n' 1000 \
  "$(( (steps_per_epoch + 999) / 1000 * 1000 ))" \
  "$(( (steps_per_epoch * 2 + 999) / 1000 * 1000 ))" \
  "$(( (steps_per_epoch * 5 + 999) / 1000 * 1000 ))" \
  "$target_steps" | sort -nu)
for step in $checkpoints; do
  checkpoint="$run/step-${step}.torch"
  python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$checkpoint" \
    MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" SPLIT=validation LIMIT=32 \
    DECODER=regression EVENT_THRESHOLD=0.3 NMS_RADIUS=3 CHUNK_SECS=5 OVERLAP_SECS=2 \
    PREDICTION_CACHE="/workspace/cache/pedal-edge-10ms-step${step}-limit32.pt" \
    RESULTS_JSON="results_eval/edge-10ms/step${step}-limit32.json" \
    > "/workspace/pedal-edge-10ms-step${step}-eval.log" 2>&1
done

python - <<'PY'
import json
import re
from pathlib import Path
for path in sorted(Path('results_eval/edge-10ms').glob('step*-limit32.json')):
    if re.fullmatch(r'step\d+-limit32\.json', path.name) is None:
        continue
    p = json.loads(path.read_text())
    print(path.name, {
        'state': p['pedal_state']['f1'],
        'down': p['pedal_down']['f1'],
        'up50': p['fixed_tolerance_diagnostic']['50ms']['pedal_up']['f1'],
        'strict50': p['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
    })
PY

best_step=$(python - <<'PY'
import json
import re
from pathlib import Path
rows = []
for path in Path('results_eval/edge-10ms').glob('step*-limit32.json'):
    match = re.fullmatch(r'step(\d+)-limit32\.json', path.name)
    if match is None:
        continue
    payload = json.loads(path.read_text())
    step = int(match.group(1))
    rows.append((payload['pedal_down']['f1'],
                 payload['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
                 step))
print(max(rows)[2])
PY
)
echo "EDGE_10MS_SELECTED_STEP=$best_step"
python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${best_step}.torch" \
  MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
  HDF5_ROLL_PATH="$roll" SPLIT=validation \
  DECODER=regression EVENT_THRESHOLD=0.3 NMS_RADIUS=3 CHUNK_SECS=5 OVERLAP_SECS=2 \
  PREDICTION_CACHE="/workspace/cache/pedal-edge-10ms-step${best_step}-full-validation.pt" \
  RESULTS_JSON="results_eval/edge-10ms/step${best_step}-full-validation.json" \
  > "/workspace/pedal-edge-10ms-step${best_step}-full-validation.log" 2>&1

mkdir -p /workspace/artifacts
artifact="/workspace/artifacts/pedal-edge-10ms-radius7-step${best_step}.onnx"
python export_mobile_pedal_onnx.py \
  --checkpoint "$run/step-${best_step}.torch" --output "$artifact" --frames 250 \
  > "/workspace/pedal-edge-10ms-step${best_step}-export.log" 2>&1
python - "$best_step" "$artifact" <<'PY'
import json, sys
from pathlib import Path

step, artifact = int(sys.argv[1]), Path(sys.argv[2])
validation = json.load(open(
    f'results_eval/edge-10ms/step{step}-full-validation.json'))
manifest = json.load(open(str(artifact) + '.json'))
payload = {
    'selected_step': step,
    'selection_order': ['locked32_pedal_down_f1', 'locked32_strict50_f1'],
    'event_threshold': 0.3,
    'nms_radius': 3,
    'full_validation': {
        'state_f1': validation['pedal_state']['f1'],
        'down_f1': validation['pedal_down']['f1'],
        'up50_f1': validation['fixed_tolerance_diagnostic']['50ms']['pedal_up']['f1'],
        'strict50_f1': validation['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'],
    },
    'artifact': manifest,
}
Path('results_eval/edge-10ms/selection.json').write_text(
    json.dumps(payload, indent=2) + '\n')
PY

if python - "$best_step" <<'PY'
import json, sys
p = json.load(open(f'results_eval/edge-10ms/step{sys.argv[1]}-full-validation.json'))
raise SystemExit(0 if (p['pedal_down']['f1'] >= .75 and
    p['fixed_tolerance_diagnostic']['50ms']['down_and_up']['f1'] >= .70) else 1)
PY
then
  python 4_eval_pedal.py DEVICE=cuda CHECKPOINT="$run/step-${best_step}.torch" \
    MAESTRO_PATH=/root/maestro-v3.0.0 HDF5_MEL_PATH="$mel" \
    HDF5_ROLL_PATH="$roll" SPLIT=test DECODER=regression EVENT_THRESHOLD=0.3 NMS_RADIUS=3 \
    CHUNK_SECS=5 OVERLAP_SECS=2 \
    PREDICTION_CACHE="/workspace/cache/pedal-edge-10ms-step${best_step}-test.pt" \
    RESULTS_JSON="results_eval/edge-10ms/step${best_step}-test.json" \
    > "/workspace/pedal-edge-10ms-step${best_step}-test.log" 2>&1
else
  echo "EDGE_10MS_TEST_SKIPPED deployment gates not met on full validation"
fi
