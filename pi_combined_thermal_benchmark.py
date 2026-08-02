#!/usr/bin/env python3
"""Sustained, cadence-matched note + pedal ONNX benchmark for Raspberry Pi."""

import argparse
import concurrent.futures
import importlib.util
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def temperature_c():
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0


def throttled():
    try:
        return subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def make_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def make_input(session, seed, frames):
    spec = session.get_inputs()[0]
    shape = tuple(frames if i == len(spec.shape) - 1 else
                  (1 if not isinstance(v, int) else v)
                  for i, v in enumerate(spec.shape))
    return spec.name, np.random.default_rng(seed).standard_normal(
        shape, dtype=np.float32)


def stats(values):
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--pedal", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--period", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--parallel-branches", action="store_true",
        help="Run independent note and pedal feature+ONNX branches concurrently, "
             "matching the production bridge.")
    parser.add_argument("--note-frames", type=int, default=105)
    parser.add_argument("--pedal-frames", type=int, default=250)
    parser.add_argument("--note-audio-seconds", type=float, default=2.5)
    parser.add_argument("--pedal-audio-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mel-module", type=Path,
        help="Optional production mel_np.py; when set, benchmark includes both "
             "384- and 160-sample feature paths with production window sizes.")
    args = parser.parse_args()

    note = make_session(args.note, args.threads)
    pedal = make_session(args.pedal, args.threads)
    mel_module = None
    audio = None
    if args.mel_module is not None:
        spec = importlib.util.spec_from_file_location("production_mel_np", args.mel_module)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load mel module: {args.mel_module}")
        mel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mel_module)
        if (args.note_audio_seconds <= 0 or args.pedal_audio_seconds <= 0 or
                args.pedal_audio_seconds > args.note_audio_seconds):
            parser.error("audio windows must satisfy 0 < pedal <= note")
        note_samples = round(args.note_audio_seconds * mel_module.SR)
        pedal_samples = round(args.pedal_audio_seconds * mel_module.SR)
        audio = np.random.default_rng(3).standard_normal(
            note_samples, dtype=np.float32)
        note_input = mel_module.logmel(audio, hop=384)[None]
        pedal_input = mel_module.logmel(audio[-pedal_samples:], hop=160)[None]
        note_name = note.get_inputs()[0].name
        pedal_name = pedal.get_inputs()[0].name
    else:
        note_name, note_input = make_input(note, 1, args.note_frames)
        pedal_name, pedal_input = make_input(pedal, 2, args.pedal_frames)
    for _ in range(3):
        note.run(None, {note_name: note_input})
        pedal.run(None, {pedal_name: pedal_input})

    note_ms, pedal_ms, note_mel_ms, pedal_mel_ms = [], [], [], []
    combined_ms, temperatures = [], []
    executor = (concurrent.futures.ThreadPoolExecutor(max_workers=2)
                if args.parallel_branches else None)

    def run_note():
        before = time.perf_counter()
        value = (mel_module.logmel(audio, hop=384)[None]
                 if mel_module is not None else note_input)
        after_mel = time.perf_counter()
        note.run(None, {note_name: value})
        after = time.perf_counter()
        return value, (after_mel - before) * 1000.0, (after - after_mel) * 1000.0

    def run_pedal():
        before = time.perf_counter()
        value = (mel_module.logmel(audio[-pedal_samples:], hop=160)[None]
                 if mel_module is not None else pedal_input)
        after_mel = time.perf_counter()
        pedal.run(None, {pedal_name: value})
        after = time.perf_counter()
        return value, (after_mel - before) * 1000.0, (after - after_mel) * 1000.0

    throttle_start = throttled()
    started = time.monotonic()
    deadline = started
    while time.monotonic() - started < args.duration:
        deadline += args.period
        before = time.perf_counter()
        if executor is not None:
            note_future = executor.submit(run_note)
            pedal_future = executor.submit(run_pedal)
            note_input, note_mel, note_inference = note_future.result()
            pedal_input, pedal_mel, pedal_inference = pedal_future.result()
        else:
            note_input, note_mel, note_inference = run_note()
            pedal_input, pedal_mel, pedal_inference = run_pedal()
        after = time.perf_counter()
        note_mel_ms.append(note_mel)
        note_ms.append(note_inference)
        pedal_mel_ms.append(pedal_mel)
        pedal_ms.append(pedal_inference)
        combined_ms.append((after - before) * 1000.0)
        temperatures.append(temperature_c())
        time.sleep(max(0.0, deadline - time.monotonic()))
    if executor is not None:
        executor.shutdown()

    payload = {
        "duration_s": time.monotonic() - started,
        "period_ms": args.period * 1000.0,
        "iterations": len(combined_ms),
        "threads_per_session": args.threads,
        "parallel_branches": args.parallel_branches,
        "includes_feature_extraction": mel_module is not None,
        "models": {"note": str(args.note), "pedal": str(args.pedal)},
        "audio_window_seconds": {
            "note": args.note_audio_seconds,
            "pedal": args.pedal_audio_seconds,
        },
        "input_shapes": {"note": list(note_input.shape),
                         "pedal": list(pedal_input.shape)},
        "note": stats(note_ms),
        "pedal": stats(pedal_ms),
        "note_mel": stats(note_mel_ms),
        "pedal_mel": stats(pedal_mel_ms),
        "combined": stats(combined_ms),
        "deadline_misses": int(sum(x > args.period * 1000.0
                                   for x in combined_ms)),
        "temperature_c": {"start": temperatures[0],
                          "max": max(temperatures),
                          "end": temperatures[-1]},
        "throttled_start": throttle_start,
        "throttled_end": throttled(),
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
