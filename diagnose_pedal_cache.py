#!/usr/bin/env python
"""Error autopsy for cached nine-channel dynamic-CRF pedal outputs."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ov_piano.crf import (
    state_unary_from_logit, transition_event_logits, viterbi_decode)
from ov_piano.data.maestro import MetaMAESTROv3
from ov_piano.eval import GtLoaderMaestro
from ov_piano.pedal import (
    decode_bounded_semi_markov_pedal_events,
    decode_dynamic_crf_pedal_events)


def reference_intervals(states, threshold=64):
    active, onset, intervals = False, None, []
    for timestamp, value in states[["ts", "val"]].itertuples(index=False):
        next_active = value >= threshold
        if next_active and not active:
            onset = float(timestamp)
        elif active and not next_active and onset is not None:
            intervals.append((onset, float(timestamp)))
            onset = None
        active = next_active
    return intervals


def decoded_intervals(events, seconds_per_frame):
    onset, intervals = None, []
    for frame, active, _ in events:
        timestamp = frame * seconds_per_frame
        if active and onset is None:
            onset = timestamp
        elif not active and onset is not None:
            intervals.append((onset, timestamp))
            onset = None
    return intervals


def matched_reference_downs(reference, estimated, tolerance):
    ref_index = est_index = 0
    matched = set()
    while ref_index < len(reference) and est_index < len(estimated):
        delta = estimated[est_index][0] - reference[ref_index][0]
        if abs(delta) <= tolerance:
            matched.add(ref_index)
            ref_index += 1
            est_index += 1
        elif delta < 0:
            est_index += 1
        else:
            ref_index += 1
    return matched


def matched_reference_times(reference, estimated, tolerance):
    """Return reference indices matched by a monotonic event-time alignment."""
    ref_index = est_index = 0
    matched = set()
    while ref_index < len(reference) and est_index < len(estimated):
        delta = estimated[est_index] - reference[ref_index]
        if abs(delta) <= tolerance:
            matched.add(ref_index)
            ref_index += 1
            est_index += 1
        elif delta < 0:
            est_index += 1
        else:
            ref_index += 1
    return matched


def evidence_near(outputs, timestamp, seconds_per_frame, active, radius_frames):
    center = int(round(timestamp / seconds_per_frame))
    left = max(0, center - radius_frames)
    right = min(outputs.shape[-1], center + radius_frames + 1)
    if left >= right:
        return None
    confidence_idx = 1 if active else 3
    transitions = outputs[5:9, left:right].transpose(0, 1).reshape(-1, 2, 2)
    down_odds, up_odds = transition_event_logits(transitions)
    odds = down_odds if active else up_odds
    return {
        "max_confidence": float(outputs[confidence_idx, left:right].max()),
        "max_transition_odds": float(odds.max()),
        "mean_state_probability": float(outputs[0, left:right].mean()),
    }


def coupled_path(outputs, alpha):
    eps = torch.finfo(outputs.dtype).eps
    state_logit = torch.logit(outputs[0].clamp(eps, 1.0 - eps))
    unary = state_unary_from_logit(state_logit).unsqueeze(0)
    transitions = outputs[5:9].transpose(0, 1).reshape(
        1, outputs.shape[-1], 2, 2).clone()
    transitions[0, :, 0, 1] += alpha * torch.logit(
        outputs[1].clamp(eps, 1.0 - eps))
    transitions[0, :, 1, 0] += alpha * torch.logit(
        outputs[3].clamp(eps, 1.0 - eps))
    return viterbi_decode(unary.float(), transitions.float())[0][0]


def local_peaks(confidence, threshold, radius):
    pooled = F.max_pool1d(
        confidence[None, None], 2 * radius + 1,
        stride=1, padding=radius)[0, 0]
    return ((confidence >= threshold) & (confidence >= pooled)).nonzero(
        as_tuple=False).flatten().tolist()


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "mean_absolute": float(np.abs(values).mean()),
        "q05_q25_q50_q75_q95": [
            float(value) for value in np.quantile(values, [.05, .25, .5, .75, .95])],
    }


def validated_cache_geometry(identity, split, limit, stride_seconds=None):
    """Validate diagnostic dataset selection and return the stitch stride."""
    if not isinstance(identity, dict):
        raise ValueError("prediction cache has no identity metadata")
    required = ("checkpoint", "split", "limit", "chunk_secs", "overlap_secs")
    missing = [field for field in required if field not in identity]
    if missing:
        raise ValueError(f"prediction cache identity missing fields: {missing}")
    if identity["split"] != split:
        raise ValueError(
            f"cache split {identity['split']!r} does not match {split!r}")
    if identity["limit"] != limit:
        raise ValueError(
            f"cache limit {identity['limit']!r} does not match {limit!r}")
    derived = float(identity["chunk_secs"]) - float(identity["overlap_secs"])
    if derived <= 0:
        raise ValueError("cache chunk geometry has a nonpositive stride")
    if stride_seconds is not None and not np.isclose(
            stride_seconds, derived, atol=1e-9, rtol=0):
        raise ValueError(
            f"requested stride {stride_seconds} does not match cache stride "
            f"{derived}")
    return derived


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--maestro", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--decoder", choices=("crf", "semi_markov"), default="crf")
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--event-penalty", type=float, default=0.0)
    parser.add_argument("--min-up-frames", type=int, default=1)
    parser.add_argument("--min-down-frames", type=int, default=1)
    parser.add_argument("--seconds-per-frame", type=float, default=0.024)
    # Normally derived from the cache's chunk/overlap identity. If supplied,
    # this acts as an assertion rather than an unchecked override.
    parser.add_argument("--stride-seconds", type=float)
    args = parser.parse_args()

    payload = torch.load(args.cache, map_location="cpu")
    identity = payload.get("identity")
    stride_seconds = validated_cache_geometry(
        identity, args.split, args.limit, args.stride_seconds)
    predictions = payload["predictions"]
    metadata = MetaMAESTROv3(
        args.maestro, splits=[args.split], years=MetaMAESTROv3.ALL_YEARS)
    if args.limit is not None:
        metadata.data = metadata.data[:args.limit]

    gap_edges = [-np.inf, 0.1, 0.15, 0.3, np.inf]
    gap_names = ["<100ms", "100-150ms", "150-300ms", ">=300ms"]
    gap_total = np.zeros(4, dtype=int)
    gap_missed = np.zeros(4, dtype=int)
    first_total = first_missed = 0
    event_modulo = []
    transition_offsets = []
    confidence_peak_offsets = []
    transition_peak_distance = []
    duration_edges = [-np.inf, 0.15, 0.3, 1.0, np.inf]
    duration_names = ["<150ms", "150-300ms", "300ms-1s", ">=1s"]
    duration_total = np.zeros(4, dtype=int)
    duration_missed_down = np.zeros(4, dtype=int)
    duration_missed_up = np.zeros(4, dtype=int)
    pair_outcomes = {name: 0 for name in (
        "both_matched", "down_only", "up_only", "both_missed")}
    evidence = {
        event: {outcome: {window: [] for window in (2, 6)}
                for outcome in ("matched", "missed")}
        for event in ("down", "up")}

    for md in metadata.data:
        data_md = (os.path.basename(md[0]), *md[1])
        outputs = predictions[data_md[0]]
        selected_boundaries = None
        if args.decoder == "crf":
            events = decode_dynamic_crf_pedal_events(outputs, args.alpha)
        else:
            events, selected_boundaries = decode_bounded_semi_markov_pedal_events(
                outputs, confidence_weight=args.alpha,
                transition_weight=args.transition_weight,
                state_weight=args.state_weight,
                candidate_threshold=args.candidate_threshold,
                min_up_frames=args.min_up_frames,
                min_down_frames=args.min_down_frames,
                event_penalty=args.event_penalty, return_selected=True)
        estimated = decoded_intervals(events, args.seconds_per_frame)
        midi_path = GtLoaderMaestro.get_metadata_path(data_md, metadata)
        _, sustain_states, _, _, _ = GtLoaderMaestro.get_midi_eventdata(midi_path)
        reference = reference_intervals(sustain_states)
        matched = matched_reference_downs(reference, estimated, 0.05)
        matched_ups = matched_reference_times(
            [end for _, end in reference], [end for _, end in estimated], 0.05)
        for index, (start, end) in enumerate(reference):
            down_ok, up_ok = index in matched, index in matched_ups
            pair_outcomes[
                "both_matched" if down_ok and up_ok else
                "down_only" if down_ok else "up_only" if up_ok else
                "both_missed"] += 1
            duration = end - start
            bucket = int(np.searchsorted(duration_edges, duration, side="right") - 1)
            bucket = min(max(bucket, 0), 3)
            duration_total[bucket] += 1
            duration_missed_down[bucket] += not down_ok
            duration_missed_up[bucket] += not up_ok
            for event, timestamp, active, ok in (
                    ("down", start, True, down_ok),
                    ("up", end, False, up_ok)):
                outcome = "matched" if ok else "missed"
                for window in (2, 6):
                    item = evidence_near(
                        outputs, timestamp, args.seconds_per_frame,
                        active, window)
                    if item is not None:
                        evidence[event][outcome][window].append(item)
        for index, (start, _) in enumerate(reference):
            if index == 0:
                first_total += 1
                first_missed += index not in matched
                continue
            gap = start - reference[index - 1][1]
            bucket = int(np.searchsorted(gap_edges, gap, side="right") - 1)
            bucket = min(max(bucket, 0), 3)
            gap_total[bucket] += 1
            gap_missed[bucket] += index not in matched

        event_modulo.extend([
            (frame * args.seconds_per_frame) % stride_seconds
            for frame, _, _ in events])
        if selected_boundaries is None:
            path = coupled_path(outputs, args.alpha)
            decoder_boundaries = [
                (frame, int(path[frame])) for frame in range(1, len(path))
                if path[frame] != path[frame - 1]]
        else:
            decoder_boundaries = selected_boundaries
        peaks_by_kind = {
            1: local_peaks(outputs[1], 0.3, 3),
            0: local_peaks(outputs[3], 0.3, 3),
        }
        for frame, active in decoder_boundaries:
            offset_channel = 2 if active else 4
            transition_offsets.append(float(outputs[offset_channel, frame]))
            peaks = peaks_by_kind[active]
            if peaks:
                transition_peak_distance.append(min(abs(frame - peak) for peak in peaks))
        for active, confidence_channel, offset_channel in ((1, 1, 2), (0, 3, 4)):
            confidence_peak_offsets.extend([
                float(outputs[offset_channel, frame])
                for frame in peaks_by_kind[active]])

    missed_total = int(gap_missed.sum() + first_missed)
    gap_rows = {}
    for name, total, missed in zip(gap_names, gap_total, gap_missed):
        gap_rows[name] = {
            "reference_downs": int(total),
            "missed_downs": int(missed),
            "miss_rate": float(missed / max(total, 1)),
            "share_of_all_missed": float(missed / max(missed_total, 1)),
        }
    modulo = np.asarray(event_modulo)
    histogram, edges = np.histogram(
        modulo, bins=np.linspace(0.0, stride_seconds, 31))
    seam_distance = np.minimum(modulo, stride_seconds - modulo)
    result = {
        "config": vars(args),
        "cache_identity": identity,
        "derived_stride_seconds": stride_seconds,
        "missed_down_by_preceding_gap": {
            "total_missed": missed_total,
            "first_interval": {"reference_downs": first_total,
                               "missed_downs": first_missed},
            "bins": gap_rows,
        },
        "misses_by_reference_interval_duration": {
            name: {
                "reference_intervals": int(total),
                "missed_downs": int(missed_down),
                "missed_ups": int(missed_up),
                "down_miss_rate": float(missed_down / max(total, 1)),
                "up_miss_rate": float(missed_up / max(total, 1)),
            }
            for name, total, missed_down, missed_up in zip(
                duration_names, duration_total, duration_missed_down,
                duration_missed_up)
        },
        "interval_boundary_match_outcomes_50ms": pair_outcomes,
        "reference_boundary_evidence": {
            event: {
                outcome: {
                    f"within_{window * args.seconds_per_frame * 1000:.0f}ms": {
                        field: distribution([item[field] for item in items])
                        for field in ("max_confidence", "max_transition_odds",
                                      "mean_state_probability")
                    }
                    for window, items in windows.items()
                }
                for outcome, windows in outcomes.items()
            }
            for event, outcomes in evidence.items()
        },
        "event_position_modulo_stitch_stride": {
            "events": int(len(modulo)),
            "within_50ms_of_seam": int((seam_distance <= 0.05).sum()),
            "within_100ms_of_seam": int((seam_distance <= 0.10).sum()),
            "histogram_100ms": histogram.tolist(),
            "bin_edges_seconds": edges.tolist(),
        },
        "offset_head_at_decoder_boundaries": {
            "decoder": args.decoder,
            "selected_boundaries": distribution(transition_offsets),
            "confidence_peaks": distribution(confidence_peak_offsets),
            "boundary_distance_to_nearest_same_kind_confidence_peak_frames":
                distribution(transition_peak_distance),
        },
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
