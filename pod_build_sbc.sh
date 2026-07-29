#!/usr/bin/env bash
# Filter the SBC variant CSV to rendered files, then 0a it into a mel+roll HDF5.
set -e
cd /workspace/iamusica_training
ORIG=/root/maestro_extract/maestro-v3.0.0/maestro-v3.0.0.csv
python3 -c "
import csv, os
orig, root = '$ORIG', '/root/maestro-sbc'
rows = list(csv.DictReader(open(orig, newline='', encoding='utf-8')))
fields = list(rows[0].keys())
kept = [r for r in rows if os.path.exists(os.path.join(root, r['audio_filename']))]
out = os.path.join(root, 'maestro-v3.0.0.csv')
if os.path.lexists(out): os.remove(out)
w = csv.DictWriter(open(out, 'w', newline='', encoding='utf-8'), fieldnames=fields)
w.writeheader(); w.writerows(kept)
print('CSVFILTER', root, 'kept', len(kept), 'of', len(rows))
"
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24
nohup python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH=/root/maestro-sbc OUTPUT_DIR=/root/h5-sbc > /root/0a-sbc.log 2>&1 &
echo "sbc_0a_pid=$!"
echo LAUNCHED_SBC_0A
