#!/usr/bin/env bash
# Build mel+roll HDF5s for clean / aptX / SBC via 0a, run concurrently.
# The aptX/SBC roots are incomplete (some WAVs never rendered), so first filter
# each variant CSV to only the rendered files (read from the ORIGINAL clean CSV,
# not the variant symlink). Plain mel (no codec) on pre-rendered 16k-mono WAVs.
set -e
cd /workspace/iamusica_training
ORIG=/root/maestro_extract/maestro-v3.0.0/maestro-v3.0.0.csv

for v in aptx sbc; do
  root=/workspace/maestro-$v
  python3 -c "
import csv, os
orig, root = '$ORIG', '$root'
rows = list(csv.DictReader(open(orig, newline='', encoding='utf-8')))
fields = list(rows[0].keys())
kept = [r for r in rows if os.path.exists(os.path.join(root, r['audio_filename']))]
out = os.path.join(root, 'maestro-v3.0.0.csv')
if os.path.islink(out) or os.path.exists(out): os.remove(out)
w = csv.DictWriter(open(out, 'w', newline='', encoding='utf-8'), fieldnames=fields)
w.writeheader(); w.writerows(kept)
print('CSVFILTER', root, 'kept', len(kept), 'of', len(rows))
"
done

export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
nohup python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH=/root/maestro_extract/maestro-v3.0.0 OUTPUT_DIR=/workspace/h5-clean > /root/0a-clean.log 2>&1 &
echo "clean_pid=$!"
nohup python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH=/workspace/maestro-aptx OUTPUT_DIR=/workspace/h5-aptx > /root/0a-aptx.log 2>&1 &
echo "aptx_pid=$!"
nohup python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH=/workspace/maestro-sbc OUTPUT_DIR=/workspace/h5-sbc > /root/0a-sbc.log 2>&1 &
echo "sbc_pid=$!"
echo LAUNCHED_ALL_3
