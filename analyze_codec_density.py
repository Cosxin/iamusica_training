#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Test the 'codec penalty concentrates on fast-repeat-dense music' hypothesis.

Per test file, correlate the clean->SBC onset-F1 drop with that file's density
of fast same-pitch repeats (consecutive same-pitch pairs with IOI < THR, per
minute). Uses existing per-file F1 from the matrix JSONs + GT note timing.
No model inference.

Usage:
  python analyze_codec_density.py MAESTRO_PATH=... HDF5_MEL_PATH=... \
     HDF5_ROLL_PATH=... CLEAN_JSON=/root/eval-baseline-clean.json \
     SBC_JSON=/root/eval-baseline-sbc.json [IOI_THR=0.15] [RESULTS_JSON=...]
"""
import os
import json
from ast import literal_eval
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py
from omegaconf import OmegaConf

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import IncrementalHDF5
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestro
from ov_piano.eval import GtLoaderMaestro


@dataclass
class ConfDef:
    MAESTRO_PATH: str = ""
    HDF5_MEL_PATH: str = ""
    HDF5_ROLL_PATH: str = ""
    CLEAN_JSON: str = ""
    SBC_JSON: str = ""
    IOI_THR: float = 0.15
    RESULTS_JSON: Optional[str] = None


def key_base(s):
    return os.path.splitext(os.path.basename(str(s)))[0]


def load_f1(path):
    d = json.load(open(path))
    sel = d.get("protocol", {}).get("selected_threshold")
    e = None
    for t in d["test"]:
        if sel is not None and abs(t["threshold"] - sel) < 1e-9:
            e = t
    e = e or d["test"][0]
    return {key_base(fl["filename"]): fl["onsets"]["f1"] for fl in e["files"]}


if __name__ == "__main__":
    CONF = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    clean_f1 = load_f1(CONF.CLEAN_JSON)
    sbc_f1 = load_f1(CONF.SBC_JSON)
    THR = CONF.IOI_THR

    meta = MetaMAESTROv3(CONF.MAESTRO_PATH, splits=["test"],
                         years=MetaMAESTROv3.ALL_YEARS)
    with h5py.File(CONF.HDF5_MEL_PATH, "r") as h:
        present = {literal_eval(t.decode("utf-8"))[0]
                   for t in h[IncrementalHDF5.METADATA_NAME]}
    meta.data = [it for it in meta.data if os.path.basename(it[0]) in present]
    ds = MelMaestro(CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
                    *(x[0] for x in meta.data), as_torch_tensors=False)
    gts = GtLoaderMaestro(ds, meta)

    rows = []
    for i, (mel, roll, md) in enumerate(ds, 1):
        kb = key_base(md[0])
        if kb not in clean_f1 or kb not in sbc_f1:
            continue
        kev = gts(md)[0]
        gt_on = kev["onset"].to_numpy()
        gt_key = kev["key"].to_numpy().astype(int)
        fast = 0
        for k in np.unique(gt_key):
            gon = np.sort(gt_on[gt_key == k])
            if len(gon) < 2:
                continue
            iois = np.diff(gon)
            fast += int((iois < THR).sum())
        dur_min = max(float(md[3]) / 60.0, 1e-6)
        dens = fast / dur_min           # fast same-pitch repeats per minute
        drop = clean_f1[kb] - sbc_f1[kb]
        title = f"{md[4]} - {md[5]}" if len(md) > 5 else kb
        rows.append({"file": kb, "title": title, "dur_min": dur_min,
                     "fast_repeats": fast, "density_per_min": dens,
                     "clean_f1": clean_f1[kb], "sbc_f1": sbc_f1[kb],
                     "codec_drop": drop})
        if i % 40 == 0:
            print(f"...{i}/{len(ds)}", flush=True)

    dens = np.array([r["density_per_min"] for r in rows])
    drop = np.array([r["codec_drop"] for r in rows])
    r_pear = float(np.corrcoef(dens, drop)[0, 1])
    order = np.argsort(dens)
    lo_idx = order[:len(order)//2]
    hi_idx = order[len(order)//2:]
    lo_drop = float(np.mean(drop[lo_idx]))
    hi_drop = float(np.mean(drop[hi_idx]))

    print("\n" + "=" * 78)
    print(f"CODEC DROP vs FAST-REPEAT DENSITY  (n={len(rows)}, IOI<{THR}s, clean->SBC)")
    print("=" * 78)
    print(f"Pearson r(density, codec_drop) = {r_pear:+.3f}")
    print(f"mean codec_drop  LOW-density half = {lo_drop:+.4f}   "
          f"HIGH-density half = {hi_drop:+.4f}   (ratio {hi_drop/lo_drop:.2f}x)"
          if lo_drop else "")
    print("\nTop-12 densest pieces (most fast same-pitch repeats/min):")
    print(f"{'dens/min':>9}{'clean':>8}{'sbc':>8}{'drop':>8}  title")
    for j in np.argsort(-dens)[:12]:
        r = rows[j]
        print(f"{r['density_per_min']:>9.1f}{r['clean_f1']:>8.3f}"
              f"{r['sbc_f1']:>8.3f}{r['codec_drop']:>8.3f}  {r['title'][:52]}")

    result = {"n": len(rows), "ioi_thr": THR, "pearson_r": r_pear,
              "lo_density_mean_drop": lo_drop, "hi_density_mean_drop": hi_drop,
              "rows": rows}
    if CONF.RESULTS_JSON:
        json.dump(result, open(CONF.RESULTS_JSON, "w"), indent=2)
        print(f"\nwrote {CONF.RESULTS_JSON}", flush=True)
