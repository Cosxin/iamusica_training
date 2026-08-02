#!/usr/bin/env bash
set -euo pipefail

cd /workspace/iamusica_training
wait_pid="${WAIT_PID:-82263}"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 20
done

baseline_run=/workspace/runs/pedal-edge-10ms-radius7-10ep-500ms-20260802
baseline_result=results_eval/edge-10ms/step120000-full-validation-threshold08.json
allocation_root=results_eval/edge-10ms-allocation-128x288
baseline_onnx=/workspace/artifacts/pedal-edge-10ms-baseline-step120000.onnx

test -s "$baseline_run/step-120000.torch"
test -s "$baseline_result"
test -s "$allocation_root/selection.json"
mkdir -p /workspace/artifacts

python export_mobile_pedal_onnx.py \
  --checkpoint "$baseline_run/step-120000.torch" \
  --output "$baseline_onnx" --frames 201 \
  > /workspace/pedal-edge-10ms-baseline-export.log 2>&1

python - "$baseline_result" "$allocation_root/selection.json" \
  "$baseline_onnx" "$allocation_root/production-decision.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

baseline_path, allocation_path, baseline_onnx, output = map(Path, sys.argv[1:])
baseline = json.loads(baseline_path.read_text())
allocation = json.loads(allocation_path.read_text())
fixed = baseline['fixed_tolerance_diagnostic']['50ms']

rows = {
    'edge_10ms_64x320': {
        'parameters': 1130605,
        'checkpoint_step': 120000,
        'threshold': 0.8,
        'state_f1': baseline['pedal_state']['f1'],
        'down_f1': baseline['pedal_down']['f1'],
        'up50_f1': fixed['pedal_up']['f1'],
        'strict50_f1': fixed['down_and_up']['f1'],
        'onnx_path': str(baseline_onnx),
        'onnx_sha256': hashlib.sha256(baseline_onnx.read_bytes()).hexdigest(),
    },
    'edge_10ms_128x288': {
        'parameters': 1048941,
        'checkpoint_step': allocation['selected_step'],
        'threshold': allocation['selected_threshold'],
        **allocation['full_validation'],
        'onnx_path': allocation['artifact']['onnx'],
        'onnx_sha256': allocation['artifact']['onnx_sha256'],
    },
}

def passes(row):
    return (row['state_f1'] >= .88 and row['down_f1'] >= .75 and
            row['strict50_f1'] >= .70)

for row in rows.values():
    row['passes_deployment_gate'] = passes(row)

eligible = [(row['down_f1'], row['strict50_f1'], name)
            for name, row in rows.items() if passes(row)]
best = max((row['down_f1'], row['strict50_f1'], name)
           for name, row in rows.items())[2]
selected = max(eligible)[2] if eligible else None
decision = {
    'gates': {'state_f1': .88, 'down_f1': .75, 'strict50_f1': .70},
    'candidates': rows,
    'best_research_candidate': best,
    'deployment_selected': selected,
    'deploy': selected is not None,
    'reason': ('full-validation gates passed' if selected else
               'no candidate passed all full-validation gates'),
    'test_policy': ('evaluate selected candidate once' if selected else
                    'do not touch test split'),
}
output.write_text(json.dumps(decision, indent=2) + '\n')
print(json.dumps(decision, indent=2))
PY

echo PRODUCTION_DECISION_COMPLETE
