#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Render pitch- and reverb-augmented MAESTRO for training.

The ablation (bluetooth_receiver/ablation/FINDINGS.md) measured what actually
degrades this model off-distribution:

    tune +12c   -0.56 recall  -4.45 precision  +62 spurious
    reverb hall -1.07 recall  -6.42 precision  +108 spurious
    vorbis 96k  -0.08 recall  -0.81 precision  +9 spurious   <- negligible

so pitch and room are the two terms worth training against; codec is not.
MAESTRO itself is close-mic'd, dry and exactly A=440, and the training pipeline
reads PRECOMPUTED log-mels from HDF5 -- by training time the waveform is gone,
so augmentation is structurally impossible on the fly and must be rendered
here, before 0a, exactly as prep_codec_render.py does for codecs.

PITCH is applied by RESAMPLING, which shifts pitch and stretches time together;
the paired MIDI is written out with its tick deltas scaled by the same factor,
so labels stay sample-aligned. That is exact and artifact-free, unlike a phase
vocoder -- and it avoids the TIME_SCALE correction that already produced one
false result in the ablation.

REVERB is convolution with an exponentially-decaying noise impulse response.
Onset times are unchanged, so labels are reused as-is (symlinked).

Deterministic: the per-file RNG is seeded from the file's relative path, so a
re-render reproduces the same augmentation.

  python prep_augment_render.py MAESTRO_INPATH=... VARIANT=pitch \\
      OUTPUT_ROOT=... [LIMIT=5] [JOBS=32] [PITCH_CENTS=50] [REVERB_WET=0.28]
"""
import csv
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys

import numpy as np

TARGET_SR = 16_000          # what 0a consumes; render straight to it
TARGET_CH = 1


def _run(cmd, stdin=None):
    proc = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True)
    return proc.stdout


def read_audio(path, sr=TARGET_SR):
    raw = _run(["ffmpeg", "-v", "error", "-threads", "1", "-i", path,
                "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32).copy()


def write_wav(path, samples, sr=TARGET_SR):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0.999:
        samples = samples * (0.999 / peak)
    _run(["ffmpeg", "-v", "error", "-threads", "1", "-y",
          "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
          "-c:a", "pcm_s16le", path],
         stdin=samples.astype(np.float32).tobytes())


def resample_ratio(samples, ratio):
    """Pitch * ratio, duration / ratio, via linear resampling.

    Reading the source faster (ratio > 1) raises pitch and shortens the file;
    the MIDI is scaled by the same factor downstream so the pair stays aligned.
    """
    n_out = int(round(len(samples) / ratio))
    if n_out < 2:
        return samples
    src = np.arange(len(samples), dtype=np.float64)
    dst = np.linspace(0.0, len(samples) - 1.0, n_out)
    return np.interp(dst, src, samples).astype(np.float32)


def reverb(samples, wet, decay_s, seed, sr=TARGET_SR):
    """Convolve with a decaying-noise IR. Onsets keep their times."""
    rng = np.random.default_rng(seed)
    length = int(decay_s * sr)
    impulse = rng.normal(0.0, 1.0, length) * np.exp(
        -np.arange(length) / (sr * decay_s / 5.0))
    impulse[0] += 1.0                      # keep the direct sound dominant
    impulse /= np.sqrt(np.sum(impulse ** 2))
    size = 1 << int(np.ceil(np.log2(len(samples) + length)))
    wet_signal = np.fft.irfft(
        np.fft.rfft(samples, size) * np.fft.rfft(impulse, size),
        size)[:len(samples)]
    return ((1.0 - wet) * samples + wet * wet_signal).astype(np.float32)


def scale_midi(src_path, dst_path, ratio):
    """Copy a MIDI with its timeline divided by `ratio`.

    Scales ABSOLUTE tick positions and re-derives deltas, rather than rounding
    each delta independently. Per-delta rounding looks harmless but is biased
    (a delta of 1 tick divided by 1.03 rounds back to 1, never to 0) and a
    MAESTRO file has ~100k events, so the bias accumulates: measured 0.7% of
    total duration, which desynchronised audio from labels and tripped 0a's
    "Wav isn't expected to be shorter than MIDI!" assertion. Accumulating
    absolute positions bounds the total error at one tick (~1 ms at 480 tpq /
    120 bpm), far below the 24 ms roll quantization.

    Per track, because in a format-1 file each track carries its own timeline.
    """
    import mido

    midi = mido.MidiFile(src_path)
    for track in midi.tracks:
        absolute = 0
        previous_scaled = 0
        for message in track:
            absolute += message.time
            scaled = int(round(absolute / ratio))
            message.time = scaled - previous_scaled
            previous_scaled = scaled
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    midi.save(dst_path)


def _augment_one(task):
    (wav_in, wav_out, midi_in, midi_out, variant, cents, wet, decay) = task
    try:
        # Seeded from the filename (not Python's randomized hash) so a re-render
        # after an interrupted run reproduces the same augmentation.
        seed = int(hashlib.sha256(
            os.path.basename(wav_in).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        if variant == "mixed":
            # HALF the training data stays clean. The first run used only 15%
            # and the model drifted globally conservative: recall fell 3-9
            # points on EVERY condition, including ones never augmented
            # (level-24dB lost 9.4 points and was never touched). A strong
            # clean anchor is what keeps augmentation additive rather than a
            # distribution shift.
            variant = rng.choices(
                ("pitch", "reverb", "both", "clean"),
                weights=(0.20, 0.18, 0.12, 0.50))[0]
        if variant == "clean":
            os.makedirs(os.path.dirname(wav_out), exist_ok=True)
            os.makedirs(os.path.dirname(midi_out), exist_ok=True)
            for src, dst in ((wav_in, wav_out), (midi_in, midi_out)):
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
            return (wav_out, 1.0, "ok")
        samples = read_audio(wav_in)
        ratio = 1.0
        if variant in ("pitch", "both"):
            # Sample where real recordings actually sit. A=441 is +4c, A=442
            # +8c, A=443 +12c; virtually everything real is inside +-15c, and
            # half a semitone is a broken speed transfer, not a concert pitch.
            #
            # The first run drew uniform(8, 50) cents -- mean |shift| ~29c --
            # so ~80% of augmented audio trained on offsets that essentially
            # never occur. The model learned exactly that distribution: +22 F1
            # at +50c but only +0.6 at +12c, the case that matters.
            #
            # Gaussian concentrates the mass near zero while still reaching the
            # tail; the clamp keeps a few large examples without dominating.
            shift = max(-cents, min(cents, rng.gauss(0.0, cents / 6.0)))
            ratio = 2.0 ** (shift / 1200.0)
            samples = resample_ratio(samples, ratio)
        if variant in ("reverb", "both"):
            samples = reverb(samples, rng.uniform(wet * 0.5, wet),
                             rng.uniform(decay * 0.6, decay), seed)
        # 0a asserts the wav is at least as long as the MIDI. MAESTRO already
        # runs a hair tighter than that, so pad a little silence rather than
        # rely on the resample landing on the safe side of the rounding.
        samples = np.concatenate(
            [samples, np.zeros(int(0.5 * TARGET_SR), dtype=np.float32)])
        write_wav(wav_out, samples)
        if abs(ratio - 1.0) > 1e-9:
            scale_midi(midi_in, midi_out, ratio)
        else:
            os.makedirs(os.path.dirname(midi_out), exist_ok=True)
            if not os.path.exists(midi_out):
                shutil.copy2(midi_in, midi_out)
        return (wav_out, ratio, "ok")
    except Exception as error:            # one bad file must not kill the run
        return (wav_out, None, f"ERROR: {error}")


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    in_root = args["MAESTRO_INPATH"].rstrip("/")
    variant = args["VARIANT"]
    out_root = args["OUTPUT_ROOT"].rstrip("/")
    if variant not in ("pitch", "reverb", "both", "mixed"):
        raise SystemExit("VARIANT must be pitch, reverb, both or mixed")
    # Which splits get augmented. Everything else is symlinked through
    # untouched, so ONE rendered tree yields ONE HDF5 that trains on augmented
    # audio while still evaluating against clean test files -- never augment
    # the set you score on.
    aug_splits = set(args.get("SPLITS", "train").split(","))
    cents = float(args.get("PITCH_CENTS", 50.0))
    wet = float(args.get("REVERB_WET", 0.28))
    decay = float(args.get("REVERB_DECAY", 1.4))
    limit = int(args.get("LIMIT", 0))
    jobs = int(args.get("JOBS", min(32, os.cpu_count() or 8)))

    csv_paths = glob.glob(os.path.join(in_root, "maestro-v*.csv"))
    if len(csv_paths) != 1:
        raise SystemExit(f"expected one MAESTRO csv in {in_root}, got {csv_paths}")
    with open(csv_paths[0], newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if limit:
        rows = rows[:limit]

    os.makedirs(out_root, exist_ok=True)
    # csv/json describe the tree; copy rather than symlink so the augmented
    # root is self-contained if the source is later unmounted.
    for extra in glob.glob(os.path.join(in_root, "*.csv")) + \
            glob.glob(os.path.join(in_root, "*.json")):
        destination = os.path.join(out_root, os.path.basename(extra))
        if not os.path.exists(destination):
            shutil.copy2(extra, destination)

    tasks = []
    for row in rows:
        wav_in = os.path.join(in_root, row["audio_filename"])
        midi_in = os.path.join(in_root, row["midi_filename"])
        if not (os.path.isfile(wav_in) and os.path.isfile(midi_in)):
            continue
        per_file = variant if row.get("split") in aug_splits else "clean"
        tasks.append((
            wav_in, os.path.join(out_root, row["audio_filename"]),
            midi_in, os.path.join(out_root, row["midi_filename"]),
            per_file, cents, wet, decay))
    augmented = sum(1 for task in tasks if task[4] != "clean")
    print(f"[augment] variant={variant} splits={sorted(aug_splits)} "
          f"files={len(tasks)} ({augmented} augmented, "
          f"{len(tasks) - augmented} passed through clean) jobs={jobs} "
          f"-> {out_root}", flush=True)

    ratios, errors = [], []
    with mp.Pool(jobs) as pool:
        for index, (wav_out, ratio, status) in enumerate(
                pool.imap_unordered(_augment_one, tasks, chunksize=1), 1):
            if status == "ok":
                ratios.append(ratio)
            else:
                errors.append({"file": os.path.relpath(wav_out, out_root),
                               "error": status})
            if index % 50 == 0 or index == len(tasks):
                print(f"[augment] {index}/{len(tasks)} "
                      f"errors={len(errors)}", flush=True)

    summary = {
        "variant": variant, "rendered": len(ratios), "errors": errors,
        "pitch_cents_max": cents, "reverb_wet_max": wet,
        "reverb_decay_max": decay,
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
    }
    with open(os.path.join(out_root, "augment_manifest.json"), "w",
              encoding="utf-8") as stream:
        json.dump(summary, stream, indent=1)
    print(f"[augment] done: {len(ratios)} ok, {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
