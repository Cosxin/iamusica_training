#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Note-with-offset (press+RELEASE) eval, using the trained frame/release head.

Standard O&V eval (2_eval) scores onset + onset-velocity only. This script
additionally scores note-with-OFFSET F1: it decodes onset+velocity+frame into
complete notes (OnsetVelocityFrameDecoder — onset starts a note, frame activity
releases it), then compares to the MIDI ground-truth note intervals with
mir_eval, both ignoring offsets (sanity vs 2_eval) and requiring offsets.

Only meaningful for a model trained with ENABLE_FRAME_HEAD=True.

Usage:
  python eval_offset.py SNAPSHOT_INPATH=... MAESTRO_PATH=... \
      HDF5_MEL_PATH=... HDF5_ROLL_PATH=... [DATASET_VARIANT=sbc] \
      [THRESHOLD=0.75] [LIMIT=0] [RESULTS_JSON=...]
"""
import os
import json
from ast import literal_eval
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import h5py
from omegaconf import OmegaConf
from mir_eval.transcription import precision_recall_f1_overlap as prf1o

from ov_piano import PIANO_MIDI_RANGE, HDF5PathManager
from ov_piano.utils import IncrementalHDF5, load_model
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.inference import strided_inference, OnsetVelocityFrameDecoder
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
    FRAME_OFF_THRESHOLD: float = 0.5
    RELEASE_DEBOUNCE_FRAMES: int = 3
    CONV1X1: tuple = (200, 200)
    LEAKY_RELU_SLOPE: float = 0.1
    INFERENCE_CHUNK_SIZE: float = 300.0
    INFERENCE_CHUNK_OVERLAP: float = 12.0
    TOLERANCE_SECS: float = 0.05
    OFFSET_RATIO: float = 0.2


def score(gt_on, gt_off, gt_key, pr_on, pr_off, pr_key, tol, offset_ratio):
    """mir_eval note F1, MIDI-key pitches (repo convention). Returns (P,R,F1)."""
    if len(pr_on) == 0 or len(gt_on) == 0:
        return 0.0, 0.0, 0.0
    ref_int = np.stack([gt_on, gt_off]).T
    est_int = np.stack([pr_on, pr_off]).T
    p, r, f1, _ = prf1o(
        ref_int, np.asarray(gt_key, dtype=float),
        est_int, np.asarray(pr_key, dtype=float),
        onset_tolerance=tol, pitch_tolerance=0.1,
        offset_ratio=offset_ratio, offset_min_tolerance=tol)
    return float(p), float(r), float(f1)


if __name__ == "__main__":
    CONF = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    (_, SR, WIN, HOP, MELS, FMIN, FMAX) = HDF5PathManager.parse_mel_hdf5_basename(
        os.path.basename(CONF.HDF5_MEL_PATH))
    SPF = HOP / SR                                    # seconds per frame
    CHUNK = round(CONF.INFERENCE_CHUNK_SIZE / SPF)
    OVL = round(CONF.INFERENCE_CHUNK_OVERLAP / SPF)
    OVL += OVL % 2                                    # must be even
    key_beg, key_end = PIANO_MIDI_RANGE
    num_keys = key_end - key_beg

    # test split, filtered to files present in the (possibly partial) HDF5
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
    print(f"[offset-eval] variant={CONF.DATASET_VARIANT} files={len(ds)} "
          f"thresh={CONF.THRESHOLD} offset_ratio={CONF.OFFSET_RATIO}", flush=True)

    model = OnsetsAndVelocities(
        in_chans=2, in_height=MELS, out_height=num_keys,
        conv1x1head=tuple(CONF.CONV1X1), bn_momentum=0,
        leaky_relu_slope=CONF.LEAKY_RELU_SLOPE, dropout_drop_p=0,
        enable_frame_head=True).to(CONF.DEVICE)
    load_model(model, CONF.SNAPSHOT_INPATH, eval_phase=True, strict=True)

    decoder = OnsetVelocityFrameDecoder(
        num_keys, frame_off_threshold=CONF.FRAME_OFF_THRESHOLD,
        release_debounce_frames=CONF.RELEASE_DEBOUNCE_FRAMES,
        nms_pool_ksize=3, gauss_conv_stddev=1, gauss_conv_ksize=11,
        vel_pad_left=1, vel_pad_right=1)

    def infer(x):
        probs, vels, frames = model(x)
        probs = F.pad(torch.sigmoid(probs[-1]), (1, 0))
        vels = F.pad(torch.sigmoid(vels), (1, 0))
        frames = F.pad(torch.sigmoid(frames), (1, 0))
        return probs, vels, frames

    on_f1s, off_f1s, per_file = [], [], []
    gt_durs, pred_durs = [], []
    for i, (mel, roll, md) in enumerate(ds, 1):
        with torch.no_grad():
            tmel = torch.from_numpy(mel).to(CONF.DEVICE).unsqueeze(0)
            onset_p, vel_p, frame_p = strided_inference(infer, tmel, CHUNK, OVL)
            del tmel
        notes = decoder(onset_p, vel_p, frame_p, pthresh=CONF.THRESHOLD)
        notes = notes[notes["prob"] >= CONF.THRESHOLD]
        pr_on = notes["onset_idx"].to_numpy() * SPF
        pr_off = notes["offset_idx"].to_numpy() * SPF
        pr_key = notes["key"].to_numpy() + key_beg
        kev = gts(md)[0]
        gt_on = kev["onset"].to_numpy()
        gt_off = kev["offset"].to_numpy()
        gt_key = kev["key"].to_numpy()
        _, _, on_f1 = score(gt_on, gt_off, gt_key, pr_on, pr_off, pr_key,
                             CONF.TOLERANCE_SECS, None)
        _, _, off_f1 = score(gt_on, gt_off, gt_key, pr_on, pr_off, pr_key,
                             CONF.TOLERANCE_SECS, CONF.OFFSET_RATIO)
        on_f1s.append(on_f1); off_f1s.append(off_f1)
        if len(gt_on): gt_durs.extend((gt_off - gt_on).tolist())
        if len(pr_on): pred_durs.extend((pr_off - pr_on).tolist())
        per_file.append({"file": md[0], "onset_f1": on_f1, "offset_f1": off_f1})
        if i % 10 == 0 or i == len(ds):
            print(f"[{i}/{len(ds)}] mean onset_f1={np.mean(on_f1s):.4f} "
                  f"note_w_offset_f1={np.mean(off_f1s):.4f}", flush=True)

    result = {"variant": CONF.DATASET_VARIANT, "n_files": len(ds),
              "threshold": CONF.THRESHOLD, "offset_ratio": CONF.OFFSET_RATIO,
              "onset_f1_mean": float(np.mean(on_f1s)),
              "note_with_offset_f1_mean": float(np.mean(off_f1s)),
              "per_file": per_file}
    result["gt_dur_median"] = float(np.median(gt_durs)) if gt_durs else None
    result["pred_dur_median"] = float(np.median(pred_durs)) if pred_durs else None
    print(f"[dur] GT median={np.median(gt_durs):.3f}s mean={np.mean(gt_durs):.3f}s | "
          f"PRED median={np.median(pred_durs):.3f}s mean={np.mean(pred_durs):.3f}s", flush=True)
    print(f"[offset-eval] DONE onset_f1={result['onset_f1_mean']:.4f} "
          f"note_with_offset_f1={result['note_with_offset_f1_mean']:.4f}", flush=True)
    if CONF.RESULTS_JSON:
        with open(CONF.RESULTS_JSON, "w") as f:
            json.dump(result, f, indent=2)
