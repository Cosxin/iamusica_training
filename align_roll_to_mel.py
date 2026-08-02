#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Align a standalone (IGNORE_MEL) roll HDF5 to a reference mel HDF5's per-file
lengths, so mel[i] and roll[i] have identical frame counts (== identical
data_idxs), which the trainer's MelMaps requires.

Reproduces 0a's pad/trim step (which only runs when mel+roll are built together):
per file, pad the roll with zeros or trim it to the mel's length.

Low memory: builds only a basename->(beg,end) index map of the source roll, then
reads one file's roll slice at a time (on disk) in the mel's file order.

Usage:
  python align_roll_to_mel.py MEL.h5 ROLL_SRC.h5 ROLL_OUT.h5
"""
import sys
from ast import literal_eval

import numpy as np
import h5py
from ov_piano.utils import IncrementalHDF5

CHUNKLEN = 333  # ~8s at hop384/16k, matches 0a


def base(md):
    md = md.decode("utf-8") if isinstance(md, (bytes, bytearray)) else str(md)
    try:
        return literal_eval(md)[0]
    except Exception:
        return md


def main(mel_path, roll_src, roll_out):
    with h5py.File(roll_src, "r") as hr:
        ridx = hr[IncrementalHDF5.IDXS_NAME][:]
        rmeta = hr[IncrementalHDF5.METADATA_NAME]
        height = hr[IncrementalHDF5.DATA_NAME].shape[0]
        idxmap = {base(rmeta[i]): (int(ridx[0, i]), int(ridx[1, i]))
                  for i in range(ridx.shape[1])}
    print(f"[align] source roll files={len(idxmap)} height={height}", flush=True)

    out = IncrementalHDF5(roll_out, height, dtype=np.float32, compression="lzf",
                          data_chunk_length=CHUNKLEN,
                          metadata_chunk_length=CHUNKLEN, err_if_exists=True)
    written = missing = padded = trimmed = 0
    with h5py.File(mel_path, "r") as hm, h5py.File(roll_src, "r") as hr:
        rdata = hr[IncrementalHDF5.DATA_NAME]
        midx = hm[IncrementalHDF5.IDXS_NAME][:]
        mmeta = hm[IncrementalHDF5.METADATA_NAME]
        for i in range(midx.shape[1]):
            md = mmeta[i]
            md = md.decode("utf-8") if isinstance(md, (bytes, bytearray)) else str(md)
            bn = base(md)
            if bn not in idxmap:
                missing += 1
                continue
            mlen = int(midx[1, i]) - int(midx[0, i])
            rb, re = idxmap[bn]
            roll = np.asarray(rdata[:, rb:re], dtype=np.float32)
            rlen = roll.shape[1]
            if rlen < mlen:
                roll = np.pad(roll, ((0, 0), (0, mlen - rlen)))
                padded += 1
            elif rlen > mlen:
                roll = roll[:, :mlen]
                trimmed += 1
            assert roll.shape[1] == mlen, "align failed"
            out.append(roll, md)
            written += 1
            if written % 200 == 0:
                print(f"[align] {written} files...", flush=True)
    out.close()
    print(f"[align] DONE written={written} missing={missing} "
          f"padded={padded} trimmed={trimmed} -> {roll_out}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
