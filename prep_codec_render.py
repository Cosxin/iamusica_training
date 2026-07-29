#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Codec-augmentation renderer for path A (domain fine-tune).

Builds a parallel MAESTRO root whose WAVs have been passed through a Bluetooth
A2DP codec round-trip (aptX / SBC) so a model fine-tuned on it is robust to the
live phone->BT->Pi path. MIDI/CSV/JSON are symlinked unchanged, so `0a` can be
pointed straight at the variant root with no code changes.

CRITICAL correctness point: codecs add a constant algorithmic delay. If not
compensated, every note-onset label drifts by that delay relative to the
(unchanged) MIDI -- poison for a press-time-sensitive model. We measure the
per-file delay by cross-correlation against the clean signal and re-align, so
audio stays sample-accurate to the MIDI timeline.

I/O is done entirely through ffmpeg pipes (decode->f32le / encode-from-f32le), so
the only Python dep is numpy. The exact ffmpeg codec flags live in CODECS below
and MUST be validated with --selftest on a box with the real ffmpeg before a full
run (SBC bitpool flags in particular are ffmpeg-build-dependent).

Usage:
  python prep_codec_render.py --selftest                 # validate codecs+delay
  python prep_codec_render.py MAESTRO_INPATH=... VARIANT=aptx OUTPUT_ROOT=... \
                              [LIMIT=5] [CODEC_SR=44100]
"""
import os
import sys
import csv
import json
import glob
import subprocess
import multiprocessing as mp
import numpy as np
from scipy.signal import correlate, correlation_lags


# ############################################################################
# # CODEC SPECS  (validate flags with --selftest before trusting!)
# ############################################################################
# Each entry: ffmpeg args to ENCODE from raw f32le stereo into the codec's raw
# stream, and to DECODE that stream back to raw f32le stereo. {sr} is filled in.
CODECS = {
    # aptX: fixed 4:1, native ffmpeg codec, stereo. Matches the live-preferred link.
    "aptx": {
        "enc": ["-f", "aptx", "-"],
        "dec_infmt": ["-f", "aptx", "-ar", "{sr}", "-ac", "{ch}"],
        "note": "aptX 4:1, the codec the live Pi link negotiates/prefers",
    },
    # Low-quality SBC: worst-case robustness variant. bitrate knob emulates a low
    # bitpool. NOTE: exact rate-control flag is ffmpeg-build dependent -> selftest.
    "sbc_low": {
        "enc": ["-acodec", "sbc", "-b:a", "128k", "-f", "sbc", "-"],
        "dec_infmt": ["-f", "sbc"],
        "note": "SBC at reduced bitrate ~ low bitpool, worst-case artifacts",
    },
}


def _run(cmd, in_bytes=None, timeout=300):
    """Run a command, feed optional stdin bytes, return stdout bytes. Raises on
    error, or subprocess.TimeoutExpired if it runs longer than `timeout` s.
    The timeout guards against a pathological/huge file hanging a pool worker
    forever (real files finish in seconds); such files are then skipped as
    errors instead of stalling the whole run."""
    p = subprocess.run(cmd, input=in_bytes, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {' '.join(cmd)}\n"
                           f"{p.stderr.decode('utf-8', 'ignore')[-800:]}")
    return p.stdout


def decode_wav_to_pcm(path, sr, ch=2):
    """Load any audio file, resample to `sr`, return float32 array (n, ch)."""
    raw = _run(["ffmpeg", "-v", "error", "-threads", "1","-i", path, "-ar", str(sr),
                "-ac", str(ch), "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    return a.reshape(-1, ch)


def pcm_to_wav(pcm, path, sr, ch=2, out_sr=None, out_ch=None):
    """Write float32 (n, ch) PCM to a 16-bit wav via ffmpeg, optionally
    resampling to out_sr and/or downmixing to out_ch on the way out (ffmpeg does
    it). Storing 16k-mono (what 0a needs) makes the file ~5.5x smaller than
    44.1k-stereo, which matters a lot for write-bound parallel runs."""
    out_sr = out_sr or sr
    out_ch = out_ch or ch
    b = np.ascontiguousarray(pcm.reshape(-1), dtype="<f4").tobytes()
    _run(["ffmpeg", "-v", "error", "-threads", "1","-y", "-f", "f32le", "-ar", str(sr),
          "-ac", str(ch), "-i", "-", "-ar", str(out_sr), "-ac", str(out_ch),
          "-acodec", "pcm_s16le", path], in_bytes=b)


def codec_roundtrip(pcm, sr, spec, ch=2):
    """Encode PCM through the codec and decode back. Returns float32 (m, ch)."""
    b = np.ascontiguousarray(pcm.reshape(-1), dtype="<f4").tobytes()
    enc = [c.format(sr=sr) for c in spec["enc"]]
    stream = _run(["ffmpeg", "-v", "error", "-threads", "1","-f", "f32le", "-ar", str(sr),
                   "-ac", str(ch), "-i", "-"] + enc, in_bytes=b)
    dec_infmt = [c.format(sr=sr, ch=ch) for c in spec["dec_infmt"]]
    dec = _run(["ffmpeg", "-v", "error", "-threads", "1"] + dec_infmt +
               ["-i", "-", "-ar", str(sr), "-ac", str(ch),
                "-f", "f32le", "-acodec", "pcm_f32le", "-"], in_bytes=stream)
    return np.frombuffer(dec, dtype="<f4").astype(np.float32).reshape(-1, ch)


# ############################################################################
# # DELAY MEASUREMENT + ALIGNMENT  (validated in scratchpad/codec_align_core.py)
# ############################################################################
def measure_delay_samples(clean, coded, sr, max_lag_ms=60.0):
    """Integer sample delay d s.t. coded[n] ~= clean[n-d] (positive => coded lags)."""
    a = clean.mean(axis=1) if clean.ndim == 2 else clean
    b = coded.mean(axis=1) if coded.ndim == 2 else coded
    n = min(len(a), len(b))
    probe_n = min(n, sr * 30)
    probe_beg = max(0, (n - probe_n) // 2)
    probe_end = probe_beg + probe_n
    a = a[probe_beg:probe_end].astype(np.float64).copy()
    b = b[probe_beg:probe_end].astype(np.float64).copy()
    a -= a.mean(); b -= b.mean()
    max_lag = int(round(max_lag_ms * 1e-3 * sr))
    corr = correlate(b, a, mode="full", method="fft")
    lags = correlation_lags(len(b), len(a), mode="full")
    keep = np.abs(lags) <= max_lag
    corr, lags = corr[keep], lags[keep]
    return int(lags[np.argmax(corr)])


def align_and_fit(coded, d, target_len):
    """Shift coded back by d samples and pad/trim to exactly target_len rows."""
    if d > 0:
        coded = coded[d:]
    elif d < 0:
        coded = np.concatenate([np.zeros((-d,) + coded.shape[1:], coded.dtype), coded])
    if len(coded) >= target_len:
        return coded[:target_len]
    pad = np.zeros((target_len - len(coded),) + coded.shape[1:], coded.dtype)
    return np.concatenate([coded, pad])


def process_file(in_wav, out_wav, sr, spec, out_sr=None, out_ch=None):
    """Full clean->codec->align pipeline for one file. Returns measured delay.
    Codec + delay-align run at `sr` (the A2DP rate, where the artifacts live);
    the aligned result is written at out_sr/out_ch (default = same as input)."""
    clean = decode_wav_to_pcm(in_wav, sr)
    coded = codec_roundtrip(clean, sr, spec)
    d = measure_delay_samples(clean, coded, sr)
    aligned = align_and_fit(coded, d, len(clean))
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    pcm_to_wav(aligned, out_wav, sr, out_sr=out_sr, out_ch=out_ch)
    return d


# ############################################################################
# # SELFTEST: synth signal -> each codec -> verify delay recovery + degradation
# ############################################################################
def selftest(sr=44100):
    rng = np.random.default_rng(0)
    t = np.arange(sr * 3) / sr
    # tones + periodic clicks (sharp transients stress onset timing)
    sig = sum(np.exp(-3 * (t % 0.5)) * np.sin(2 * np.pi * f * t)
              for f in (220, 440, 660, 880))
    clicks = np.zeros_like(t); clicks[::sr // 4] = 1.0
    sig = 0.6 * sig + 0.4 * clicks
    pcm = np.stack([sig, sig * 0.95], axis=1).astype(np.float32)
    print(f"[selftest] sr={sr}  {len(pcm)} samples ({len(pcm)/sr:.1f}s)")
    ok = True
    for name, spec in CODECS.items():
        try:
            coded = codec_roundtrip(pcm, sr, spec)
            d = measure_delay_samples(pcm, coded, sr)
            aligned = align_and_fit(coded, d, len(pcm))
            # degradation after re-alignment (lower = better preserved timing)
            resid = float(np.abs(aligned - pcm).mean())
            corr = float(np.corrcoef(aligned.mean(1), pcm.mean(1))[0, 1])
            quality_ok = corr >= 0.9 and resid <= 0.1
            ok &= quality_ok
            status = "OK" if quality_ok else "FAIL_QUALITY"
            print(f"  {name:9s} {status}  delay={d:4d} smp "
                  f"({d/sr*1000:6.2f} ms)  "
                  f"resid={resid:.4f}  corr={corr:.4f}  [{spec['note']}]")
        except Exception as e:
            ok = False
            print(f"  {name:9s} FAIL  {e}")
    print("SELFTEST_OK" if ok else "SELFTEST_FAILED")
    return ok


# ############################################################################
# # PARALLEL WORKER (module-level so it is picklable by multiprocessing)
# ############################################################################
def _render_one(task):
    """Render one file. Returns (out_wav, delay_or_None, status).
    process_file already makes the output dir (exist_ok, race-safe)."""
    w, out_wav, sr, spec, out_sr, out_ch = task
    if os.path.exists(out_wav):
        return (out_wav, None, "skip")
    try:
        d = process_file(w, out_wav, sr, spec, out_sr=out_sr, out_ch=out_ch)
        return (out_wav, d, "ok")
    except Exception as e:  # keep one bad file from killing the whole run
        return (out_wav, None, f"ERROR: {e}")


# ############################################################################
# # MAIN
# ############################################################################
def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest(int(args.get("CODEC_SR", 44100))) else 1)

    in_root = args["MAESTRO_INPATH"].rstrip("/")
    variant = args["VARIANT"]
    out_root = args["OUTPUT_ROOT"].rstrip("/")
    sr = int(args.get("CODEC_SR", 44100))
    limit = int(args.get("LIMIT", 0))
    spec = CODECS[variant]

    if limit:
        csv_paths = glob.glob(os.path.join(in_root, "maestro-v*.csv"))
        if len(csv_paths) != 1:
            raise RuntimeError(
                f"Expected one MAESTRO CSV in {in_root}, got {csv_paths}")
        with open(csv_paths[0], newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))[:limit]
        wavs = [os.path.join(in_root, row["audio_filename"])
                for row in rows]
        missing = [path for path in wavs if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} CSV-selected WAVs are missing; "
                f"first: {missing[0]}")
    else:
        wavs = sorted(glob.glob(
            os.path.join(in_root, "**", "*.wav"), recursive=True))
    print(f"[render] variant={variant} sr={sr} files={len(wavs)} -> {out_root}")

    # symlink the non-audio bits (csv/json + all midi) so 0a sees a full root
    os.makedirs(out_root, exist_ok=True)
    for extra in glob.glob(os.path.join(in_root, "*.csv")) + \
            glob.glob(os.path.join(in_root, "*.json")):
        dst = os.path.join(out_root, os.path.basename(extra))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(extra), dst)
    for midi in glob.glob(os.path.join(in_root, "**", "*.midi"), recursive=True):
        rel = os.path.relpath(midi, in_root)
        dst = os.path.join(out_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(midi), dst)

    # PARALLEL render: each file is independent (own ffmpeg round-trip), so a
    # process pool turns this from single-core-serial into ~JOBS-way parallel.
    # Codecs + mel are CPU-bound; on a 96-core box this is a ~30x speedup.
    jobs = int(args.get("JOBS", min(32, os.cpu_count() or 8)))
    # Store compact 16k-mono by default (what 0a consumes) -> ~5.5x smaller
    # writes than 44.1k-stereo. OUTPUT_SR=0 keeps the codec rate/full stereo.
    out_sr = int(args.get("OUTPUT_SR", 16000))
    out_ch = int(args.get("OUTPUT_CH", 1))
    if out_sr == 0:
        out_sr, out_ch = sr, 2
    tasks = [(w, os.path.join(out_root, os.path.relpath(w, in_root)),
              sr, spec, out_sr, out_ch) for w in wavs]
    print(f"[render] {len(tasks)} files across {jobs} workers "
          f"-> {out_sr}Hz/{out_ch}ch", flush=True)

    delays, log, errors = [], [], []
    with mp.Pool(jobs) as pool:
        for i, (out_wav, d, status) in enumerate(
                pool.imap_unordered(_render_one, tasks, chunksize=1), 1):
            rel = os.path.relpath(out_wav, out_root)
            if status == "ok":
                delays.append(d)
                log.append({"file": rel, "delay_samples": d,
                            "delay_ms": d / sr * 1000})
            elif status.startswith("ERROR"):
                errors.append({"file": rel, "error": status})
                print(f"  !! {rel}: {status}", flush=True)
            if i % 50 == 0 or i == len(tasks):
                md = float(np.median(delays)) if delays else 0.0
                print(f"[{i}/{len(tasks)}] ok={len(delays)} err={len(errors)} "
                      f"skip={i - len(delays) - len(errors)} "
                      f"median_delay={md:.1f} smp", flush=True)

    # provenance for the paper
    prov = {"variant": variant, "codec_sr": sr, "spec": spec, "jobs": jobs,
            "n_files": len(wavs), "n_ok": len(delays), "n_errors": len(errors),
            "delay_samples_median": float(np.median(delays)) if delays else None,
            "delay_ms_median": float(np.median(delays) / sr * 1000) if delays else None,
            "errors": errors, "per_file": log}
    with open(os.path.join(out_root, f"_codec_provenance_{variant}.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(f"[render] done. ok={len(delays)} err={len(errors)} "
          f"provenance -> _codec_provenance_{variant}.json", flush=True)


if __name__ == "__main__":
    main()
