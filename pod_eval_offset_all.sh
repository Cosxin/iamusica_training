#!/usr/bin/env bash
# Full note-with-offset (press+release) eval of the fine-tuned model on all
# three codecs. Reports onset F1 (sanity) + note-with-offset F1 + GT/pred
# duration medians. Runs alongside the onset/velocity matrix (light on RAM).
set -e
cd /workspace/iamusica_training
FT=$(ls /root/runs/mixed_ABC/model_snapshots/*step=5000*.torch | head -1)
MEL='MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5'
ROLL='MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5'
GT=/root/maestro_extract/maestro-v3.0.0
echo "FT=$FT"
for V in clean aptx sbc; do
  echo "=== OFFSET EVAL on $V ==="
  python eval_offset.py \
    SNAPSHOT_INPATH="$FT" MAESTRO_PATH="$GT" \
    HDF5_MEL_PATH="/root/h5-$V/$MEL" HDF5_ROLL_PATH="/root/h5-$V/$ROLL" \
    DATASET_VARIANT=$V RESULTS_JSON=/root/offset-finetuned-$V.json
done
echo OFFSET_ALL_DONE
