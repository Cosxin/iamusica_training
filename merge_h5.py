#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Concatenate several IncrementalHDF5 files (same height) into one.

Used to build a mixed clean+aptX+SBC training set: merge the per-variant mel
HDF5s into one mel HDF5, and the per-variant roll HDF5s into one roll HDF5, IN
THE SAME SOURCE ORDER so mel[i] and roll[i] stay aligned. Per-file metadata
(which carries the train/val/test split) is preserved verbatim, so the trainer's
split filtering keeps working. Appends one file at a time -> low memory.

Usage: python merge_h5.py OUT.h5 SRC1.h5 SRC2.h5 SRC3.h5
"""
import sys
import numpy as np
import h5py
from ov_piano.utils import IncrementalHDF5

CHUNKLEN = 333  # ~8s at hop384/16k, matches 0a


def merge(out_path, sources):
    with h5py.File(sources[0], "r") as h0:
        height = h0[IncrementalHDF5.DATA_NAME].shape[0]
    out = IncrementalHDF5(out_path, height, dtype=np.float32, compression="lzf",
                          data_chunk_length=CHUNKLEN,
                          metadata_chunk_length=CHUNKLEN, err_if_exists=True)
    total = 0
    for src in sources:
        with h5py.File(src, "r") as h:
            idxs = h[IncrementalHDF5.IDXS_NAME][:]           # (2, N)
            meta = h[IncrementalHDF5.METADATA_NAME]
            data = h[IncrementalHDF5.DATA_NAME]
            n = idxs.shape[1]
            for i in range(n):
                beg, end = int(idxs[0, i]), int(idxs[1, i])
                mat = data[:, beg:end]
                m = meta[i]
                m = m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else str(m)
                out.append(mat, m)
                total += 1
        print(f"[merge] +{n} from {src}  (total {total})", flush=True)
    out.close()
    print(f"[merge] DONE {out_path}: height={height} entries={total}", flush=True)


if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2:])
