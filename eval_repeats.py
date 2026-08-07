#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Repeated-note (re-trigger) eval, binned by inter-onset interval (IOI).

The product question: "how close can two SAME-PITCH notes be before the model
merges them into one?" This is NOT note-with-offset F1. We mine the GT for
consecutive same-pitch note pairs, bin them by IOI = onset2 - onset1, and for
each pair check whether the model resolves it as TWO distinct notes.

We decompose the two heads' contributions:
  * onset_retrigger : the onset head fired TWICE (a pred onset matches onset1
    and a DISTINCT pred onset matches onset2, each within TOL). Determines
    whether two notes exist at all.
  * clean_segment  : onset_retrigger AND note-1 is RELEASED before note-2's
    onset (pred offset_1 <= onset2 + GAP_TOL). Determines whether the release
    head opens a visible gap (LED blinks off-then-on) instead of overlapping.

Recall per IOI bin = fraction of GT repeat pairs that succeed.

Usage:
  python eval_repeats.py SNAPSHOT_INPATH=... MAESTRO_PATH=... \
      HDF5_MEL_PATH=... HDF5_ROLL_PATH=... [DATASET_VARIANT=clean] \
      [THRESHOLD=0.75] [LIMIT=0] [RESULTS_JSON=...]
"""
import os
import json
from ast import literal_eval
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from omegaconf import OmegaConf

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import IncrementalHDF5, load_model
from ov_piano.data.maestro import MetaMAESTROv3
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import (
    strided_inference, OnsetVelocityNmsDecoder)
from ov_piano.eval import GtLoaderMaestro


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = ""
    HDF5_MEL_PATH: str = ""
    HDF5_ROLL_PATH: str = ""
    SNAPSHOT_INPATH: str = ""
    DATASET_VARIANT: Optional[str] = None
    RESULTS_JSON: Optional[str] = None
    LIMIT: int = 0
    THRESHOLD: float = 0.75
    CONV1X1: tuple = (200, 200)
    LEAKY_RELU_SLOPE: float = 0.1
    INFERENCE_CHUNK_SIZE: float = 300.0
    INFERENCE_CHUNK_OVERLAP: float = 12.0
    TOLERANCE_SECS: float = 0.05           # onset-match tolerance
    GAP_TOL_SECS: float = 0.024            # release-before-next-onset slack (~1 frame)
    # IOI bin edges in seconds (upper-exclusive); last bin is [1.0, inf)
    BIN_EDGES: tuple = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)


def match_onsets(pred_on, gt_time, tol, used):
    """Nearest UNUSED pred onset within tol of gt_time. Returns index or -1."""
    best, best_d = -1, tol + 1e9
    for j, t in enumerate(pred_on):
        if j in used:
            continue
        d = abs(t - gt_time)
        if d <= tol and d < best_d:
            best, best_d = j, d
    return best


if __name__ == "__main__":
    CONF = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    (_, SR, WIN, HOP, MELS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(CONF.HDF5_MEL_PATH))
    SPF = HOP / SR
    CHUNK = round(CONF.INFERENCE_CHUNK_SIZE / SPF)
    OVL = round(CONF.INFERENCE_CHUNK_OVERLAP / SPF)
    OVL += OVL % 2
    key_beg, key_end = PIANO_MIDI_RANGE
    num_keys = key_end - key_beg
    TOL = CONF.TOLERANCE_SECS
    GAP = CONF.GAP_TOL_SECS
    edges = list(CONF.BIN_EDGES)

    from ov_piano.data.maestro import MelMaestro
    meta = MetaMAESTROv3(CONF.MAESTRO_PATH, splits=["test"],
                         years=MetaMAESTROv3.ALL_YEARS)
    with h5py.File(CONF.HDF5_MEL_PATH, "r") as h:
        present = {literal_eval(t.decode("utf-8"))[0]
                   for t in h[IncrementalHDF5.METADATA_NAME]}
    meta.data = [it for it in meta.data if os.path.basename(it[0]) in present]
    if CONF.LIMIT:
        meta.data = meta.data[:CONF.LIMIT]
    ds = MelMaestro(CONF.HDF5_MEL_PATH, CONF.HDF5_ROLL_PATH,
                    *(x[0] for x in meta.data), as_torch_tensors=False)
    gts = GtLoaderMaestro(ds, meta)
    print(f"[repeats] variant={CONF.DATASET_VARIANT} files={len(ds)} "
          f"tol={TOL}s gap_tol={GAP}s", flush=True)

    # Repeated-note IOI is an ONSET-retrigger metric: it asks whether a
    # same-pitch reattack is detected as a second note, which depends only on
    # the onset head. Release information is not needed and is not used.
    onset_only = True
    model = OnsetsAndVelocities(
        in_chans=2, in_height=MELS, out_height=num_keys,
        conv1x1head=tuple(CONF.CONV1X1), bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE, dropout_drop_p=0).to(CONF.DEVICE)
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True, strict=True)
    decoder = OnsetVelocityNmsDecoder(
        num_keys, nms_pool_ksize=3, gauss_conv_stddev=1,
        gauss_conv_ksize=11, vel_pad_left=1, vel_pad_right=1)
    print("[repeats] mode=ONSET_ONLY", flush=True)

    def infer(x):
        out = model(x)
        probs = F.pad(torch.sigmoid(out[0][-1]), (1, 0))
        vels = F.pad(torch.sigmoid(out[1]), (1, 0))
        return probs, vels

    # per-bin counters
    nb = len(edges) + 1
    tot = np.zeros(nb, int)
    onset_ok = np.zeros(nb, int)
    clean_ok = np.zeros(nb, int)

    def bin_of(ioi):
        for i, e in enumerate(edges):
            if ioi < e:
                return i
        return nb - 1

    for i, (mel, roll, md) in enumerate(ds, 1):
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            out = strided_inference(infer, tmel, CHUNK, OVL)
            del tmel
        notes = decoder(out[0], out[1], pthresh=CONF.THRESHOLD)
        notes = notes[notes["prob"] >= CONF.THRESHOLD]
        pr_on = notes["t_idx"].to_numpy() * SPF
        pr_off = np.full(len(pr_on), np.inf)   # no release info
        pr_key = (notes["key"].to_numpy() + key_beg).astype(int)

        kev = gts(md)[0]
        gt_on = kev["onset"].to_numpy()
        gt_off = kev["offset"].to_numpy()
        gt_key = kev["key"].to_numpy().astype(int)

        # group by pitch, find consecutive same-pitch pairs
        for k in np.unique(gt_key):
            gm = np.where(gt_key == k)[0]
            gon = np.sort(gt_on[gm])
            if len(gon) < 2:
                continue
            pm = np.where(pr_key == k)[0]
            p_on = np.sort(pr_on[pm])
            # keep pred offsets aligned to sorted onsets
            order = np.argsort(pr_on[pm])
            p_off = pr_off[pm][order]
            for a in range(len(gon) - 1):
                o1, o2 = gon[a], gon[a + 1]
                ioi = o2 - o1
                b = bin_of(ioi)
                tot[b] += 1
                used = set()
                j1 = match_onsets(p_on, o1, TOL, used)
                if j1 >= 0:
                    used.add(j1)
                j2 = match_onsets(p_on, o2, TOL, used)
                if j1 >= 0 and j2 >= 0 and j1 != j2:
                    onset_ok[b] += 1
                    # released before next onset (opens a gap)?
                    if p_off[j1] <= o2 + GAP:
                        clean_ok[b] += 1
        if i % 10 == 0 or i == len(ds):
            print(f"[{i}/{len(ds)}] pairs={tot.sum()} "
                  f"onset_retrig={onset_ok.sum()} clean={clean_ok.sum()}",
                  flush=True)

    labels = []
    lo = 0.0
    for e in edges:
        labels.append(f"{int(lo*1000)}-{int(e*1000)}ms")
        lo = e
    labels.append(f">{int(edges[-1]*1000)}ms")

    print("\n" + "=" * 72)
    print(f"REPEATED SAME-PITCH NOTES by IOI  (variant={CONF.DATASET_VARIANT})")
    print("=" * 72)
    ch = "n/a" if onset_only else "clean-segment"
    print(f"{'IOI bin':<14}{'#pairs':>8}{'onset-retrig':>16}{ch:>16}")
    bins_out = []
    for b in range(nb):
        n = int(tot[b])
        orr = onset_ok[b] / n if n else float('nan')
        crr = clean_ok[b] / n if n else float('nan')
        cstr = "     n/a" if onset_only else f"{crr:>15.3f}"
        print(f"{labels[b]:<14}{n:>8}{orr:>15.3f} {cstr}")
        bins_out.append({"bin": labels[b], "n": n,
                         "onset_retrigger_recall": (None if not n else orr),
                         "clean_segment_recall": (
                             None if (onset_only or not n) else crr),
                         "onset_ok": int(onset_ok[b]),
                         "clean_ok": (None if onset_only else int(clean_ok[b]))})
    result = {"variant": CONF.DATASET_VARIANT, "n_files": len(ds),
              "mode": "onset_only",
              "threshold": CONF.THRESHOLD, "tol_secs": TOL, "gap_tol_secs": GAP,
              "total_pairs": int(tot.sum()),
              "onset_retrigger_recall_overall": float(onset_ok.sum() / tot.sum()),
              "clean_segment_recall_overall": (
                  None if onset_only else float(clean_ok.sum() / tot.sum())),
              "bins": bins_out}
    cso = result['clean_segment_recall_overall']
    print(f"\nOVERALL onset-retrig={result['onset_retrigger_recall_overall']:.3f} "
          f"clean-segment={'n/a' if cso is None else f'{cso:.3f}'} "
          f"over {result['total_pairs']} pairs", flush=True)
    if CONF.RESULTS_JSON:
        with open(CONF.RESULTS_JSON, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {CONF.RESULTS_JSON}", flush=True)
