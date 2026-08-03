#!/usr/bin/env python
"""Evaluate direct CC64 state and pedal down/up events on MAESTRO."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import OmegaConf
import torch
from mir_eval.transcription import precision_recall_f1_overlap

from ov_piano import HDF5PathManager
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestro
from ov_piano.eval import GtLoaderMaestro
from ov_piano.inference import strided_inference
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.models.pedal import FrozenPedalModel
from ov_piano.models.mobile_pedal import (
    EdgePedalAMTRegression,
    MultirateEdgePedalAMTRegression,
    OfflinePedalAMTRegression, OfflinePedalAMTRegressionFlatten,
    StreamingPedalAMT, StreamingPedalAMTRegression,
    StreamingPedalAMTDualTimescale, StreamingPedalAMTDynamicCRF,
    StreamingPedalAMTDynamicCRFFlatten,
    StreamingPedalAMTDynamicCRFTwoStream)
from ov_piano.pedal import (
    decode_pedal_events, decode_regression_pedal_events,
    decode_bounded_semi_markov_pedal_events,
    decode_dynamic_crf_pedal_events)


@dataclass
class ConfDef:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAESTRO_PATH: str = "datasets/maestro/maestro-v3.0.0"
    HDF5_MEL_PATH: str = "datasets/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5"
    HDF5_ROLL_PATH: str = "datasets/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5"
    CHECKPOINT: str = "runs/pedal_frozen/step-12000.torch"
    SPLIT: str = "validation"
    LIMIT: Optional[int] = None
    CHUNK_SECS: float = 5.0
    OVERLAP_SECS: float = 2.0
    FULL_RECORDING_INFERENCE: bool = False
    # Offline oracle only. None means unrestricted future context. A finite
    # value bounds total CNN + recurrent lookahead using fixed-lag smoothing.
    OFFLINE_FUTURE_LAG_SECS: Optional[float] = None
    OFFLINE_LAG_WINDOW_BATCH: int = 128
    # None loads the training-time fixed event delay from the checkpoint.
    FUTURE_CONTEXT_SECS: Optional[float] = None
    STATE_THRESHOLD: float = 0.5
    EVENT_THRESHOLD: float = 0.3
    EVENT_TOLERANCE_SECS: float = 0.05
    NMS_RADIUS: int = 3
    DECODER: str = "event"
    # Add event-head log odds to the corresponding CRF transition potential.
    CRF_CONFIDENCE_WEIGHT: float = 0.0
    SEMI_MARKOV_CONFIDENCE_WEIGHT: float = 1.0
    SEMI_MARKOV_TRANSITION_WEIGHT: float = 1.0
    SEMI_MARKOV_STATE_WEIGHT: float = 1.0
    SEMI_MARKOV_CANDIDATE_THRESHOLD: float = 0.05
    SEMI_MARKOV_MIN_UP_FRAMES: int = 1
    SEMI_MARKOV_MIN_DOWN_FRAMES: int = 1
    SEMI_MARKOV_MAX_UP_FRAMES: Optional[int] = None
    SEMI_MARKOV_MAX_DOWN_FRAMES: Optional[int] = None
    SEMI_MARKOV_EVENT_PENALTY: float = 0.0
    MODEL_TYPE: str = "auto"
    # Optional reusable raw-output cache. Cache identity includes checkpoint,
    # model type, split, chunk geometry, and effective future context.
    PREDICTION_CACHE: Optional[str] = None
    RESULTS_JSON: str = "pedal-eval.json"


def pedal_intervals_from_states(states, threshold=64):
    active = False
    onset = None
    intervals = []
    for timestamp, value in states[["ts", "val"]].itertuples(index=False):
        next_active = value >= threshold
        if next_active and not active:
            onset = float(timestamp)
        elif active and not next_active and onset is not None:
            intervals.append((onset, float(timestamp)))
            onset = None
        active = next_active
    return intervals


def intervals_from_decoded(events, seconds_per_frame):
    onset = None
    intervals = []
    for frame, active, _ in events:
        timestamp = frame * seconds_per_frame
        if active and onset is None:
            onset = timestamp
        elif not active and onset is not None:
            intervals.append((onset, timestamp))
            onset = None
    return intervals


def decode_state_transitions(probabilities, threshold):
    active = False
    events = []
    for frame, value in enumerate(probabilities[0]):
        next_active = bool(value >= threshold)
        if next_active != active:
            events.append((frame, next_active, float(value)))
            active = next_active
    return events


def count_onset_matches(reference, estimated, tolerance):
    reference = sorted(start for start, _ in reference)
    estimated = sorted(start for start, _ in estimated)
    ref_index = est_index = matches = 0
    while ref_index < len(reference) and est_index < len(estimated):
        delta = estimated[est_index] - reference[ref_index]
        if abs(delta) <= tolerance:
            matches += 1
            ref_index += 1
            est_index += 1
        elif delta < 0:
            est_index += 1
        else:
            ref_index += 1
    return matches


def count_offset_matches(reference, estimated, tolerance):
    return count_onset_matches(
        [(end, end) for _, end in reference],
        [(end, end) for _, end in estimated], tolerance)


def count_fixed_interval_matches(reference, estimated, onset_tolerance,
                                 offset_tolerance):
    """Greedy monotonic matching for non-overlapping pedal intervals."""
    reference, estimated = sorted(reference), sorted(estimated)
    ref_index = est_index = matches = 0
    while ref_index < len(reference) and est_index < len(estimated):
        ref_on, ref_off = reference[ref_index]
        est_on, est_off = estimated[est_index]
        if est_on < ref_on - onset_tolerance:
            est_index += 1
        elif est_on > ref_on + onset_tolerance:
            ref_index += 1
        elif abs(est_off - ref_off) <= offset_tolerance:
            matches += 1
            ref_index += 1
            est_index += 1
        elif est_off < ref_off - offset_tolerance:
            est_index += 1
        else:
            ref_index += 1
    return matches


if __name__ == "__main__":
    conf = OmegaConf.merge(OmegaConf.structured(ConfDef()), OmegaConf.from_cli())
    (_, sample_rate, _, hop_size, mel_bins, _, _) = \
        HDF5PathManager.parse_mel_hdf5_basename(os.path.basename(conf.HDF5_MEL_PATH))
    seconds_per_frame = hop_size / sample_rate
    chunk_frames = round(conf.CHUNK_SECS / seconds_per_frame)
    overlap_frames = round(conf.OVERLAP_SECS / seconds_per_frame)
    overlap_frames += overlap_frames % 2

    saved = torch.load(conf.CHECKPOINT, map_location="cpu")
    model_conf = saved["config"]
    future_context_secs = (
        model_conf.get("FUTURE_CONTEXT_SECS", 0.0)
        if conf.FUTURE_CONTEXT_SECS is None else conf.FUTURE_CONTEXT_SECS)
    future_frames = round(future_context_secs / seconds_per_frame)
    if future_frames < 0:
        raise ValueError("FUTURE_CONTEXT_SECS must be nonnegative")
    model_type = conf.MODEL_TYPE
    if model_type == "auto":
        model_type = ("mobile_dynamic_crf_flatten" if
                      saved.get("model_type") == "mobile_dynamic_crf_flatten" else
                      "mobile_dynamic_crf_two_stream" if
                      saved.get("model_type") == "mobile_dynamic_crf_two_stream" else
                      "mobile_dual_timescale" if
                      saved.get("model_type") == "mobile_dual_timescale" else
                      "mobile_dynamic_crf" if
                      saved.get("model_type") == "mobile_dynamic_crf" else
                      saved.get("model_type") if saved.get("model_type") in
                      ("mobile_offline_oracle",
                       "mobile_offline_oracle_flatten") else
                      "mobile_edge_10ms" if
                      saved.get("model_type") == "mobile_edge_10ms" else
                      "mobile_edge_multirate_10ms" if
                      saved.get("model_type") == "mobile_edge_multirate_10ms" else
                      "mobile_regression" if saved.get("model_type") == "mobile_regression"
                      else "mobile" if "SHARED_HIDDEN" in model_conf else "frozen")
    if model_type == "mobile":
        model = StreamingPedalAMT(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"])
    elif model_type == "mobile_regression":
        model = StreamingPedalAMTRegression(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"])
    elif model_type == "mobile_offline_oracle":
        model = OfflinePedalAMTRegression(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            bidirectional_hidden=model_conf["BIDIRECTIONAL_HIDDEN"])
    elif model_type == "mobile_offline_oracle_flatten":
        model = OfflinePedalAMTRegressionFlatten(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            bidirectional_hidden=model_conf["BIDIRECTIONAL_HIDDEN"],
            projection_size=model_conf["ORACLE_PROJECTION_SIZE"])
    elif model_type == "mobile_edge_10ms":
        model = EdgePedalAMTRegression(
            mel_bins, frontend_size=model_conf.get("FRONTEND_SIZE", 512),
            shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"])
    elif model_type == "mobile_edge_multirate_10ms":
        model = MultirateEdgePedalAMTRegression(
            mel_bins, fast_size=model_conf["FAST_SIZE"],
            context_hidden=model_conf["CONTEXT_HIDDEN"],
            context_layers=model_conf["CONTEXT_LAYERS"],
            context_pool=model_conf["CONTEXT_POOL"],
            event_delay_frames=model_conf["EVENT_DELAY_FRAMES"],
            head_hidden=model_conf["HEAD_HIDDEN"],
            dropout=model_conf["DROPOUT"])
    elif model_type == "mobile_dual_timescale":
        model = StreamingPedalAMTDualTimescale(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            slow_dilation=model_conf.get("SLOW_DILATION", 2))
    elif model_type == "mobile_dynamic_crf":
        model = StreamingPedalAMTDynamicCRF(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            slow_dilation=model_conf.get("SLOW_DILATION", 2))
    elif model_type == "mobile_dynamic_crf_flatten":
        model = StreamingPedalAMTDynamicCRFFlatten(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            slow_dilation=model_conf.get("SLOW_DILATION", 2))
    elif model_type == "mobile_dynamic_crf_two_stream":
        model = StreamingPedalAMTDynamicCRFTwoStream(
            mel_bins, shared_hidden=model_conf["SHARED_HIDDEN"],
            tower_hidden=model_conf["TOWER_HIDDEN"],
            shared_layers=model_conf["SHARED_LAYERS"],
            dropout=model_conf["DROPOUT"],
            slow_dilation=model_conf.get("SLOW_DILATION", 2))
    elif model_type == "frozen":
        base = OnsetsAndVelocities(2, mel_bins, 88, enable_frame_head=True)
        model = FrozenPedalModel(
            base,
            model_conf["ADAPTER_CHANNELS"],
            model_conf["HIDDEN_SIZE"],
            model_conf["GRU_LAYERS"],
            model_conf["DROPOUT"],
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")
    if model_type == "mobile_edge_multirate_10ms":
        if future_frames != model.event_delay_frames:
            raise ValueError(
                "multirate event delay does not match FUTURE_CONTEXT_SECS")
        if (not conf.FULL_RECORDING_INFERENCE and
                overlap_frames // 2 <= model.event_delay_frames):
            raise ValueError(
                "multirate strided inference requires half the overlap to "
                "exceed EVENT_DELAY_FRAMES")
    missing, unexpected = model.load_state_dict(saved["model"], strict=False)
    allowed_missing = []
    if model_type == "mobile_dynamic_crf_two_stream":
        # D7b predates the diagnostic-only transient auxiliary head. It does
        # not affect the deployed forward path, so old checkpoints remain
        # exactly evaluable after D7c added that head.
        allowed_missing = [name for name in missing
                           if name.startswith("transient_head.")]
    if sorted(missing) != sorted(allowed_missing) or unexpected:
        raise ValueError(
            f"incompatible checkpoint: missing={missing}, "
            f"unexpected={unexpected}")
    model.to(conf.DEVICE).eval()
    # This entry point is evaluation-only.  Disabling autograd globally keeps
    # fixed-lag inference from retaining one graph per recurrent window across
    # an entire recording (which otherwise exhausts GPU memory on long files).
    torch.set_grad_enabled(False)

    metadata = MetaMAESTROv3(
        conf.MAESTRO_PATH, splits=[conf.SPLIT], years=MetaMAESTROv3.ALL_YEARS)
    if conf.LIMIT is not None:
        metadata.data = metadata.data[:conf.LIMIT]
    dataset = MelMaestro(
        conf.HDF5_MEL_PATH, conf.HDF5_ROLL_PATH,
        *(item[0] for item in metadata.data), as_torch_tensors=False)

    cache_identity = {
        "checkpoint": os.path.abspath(conf.CHECKPOINT),
        "model_type": model_type,
        "split": conf.SPLIT,
        "limit": conf.LIMIT,
        "chunk_secs": conf.CHUNK_SECS,
        "overlap_secs": conf.OVERLAP_SECS,
        "future_context_secs": future_context_secs,
        "full_recording_inference": conf.FULL_RECORDING_INFERENCE,
        "offline_future_lag_secs": conf.OFFLINE_FUTURE_LAG_SECS,
    }
    cached_predictions = {}
    cache_dirty = False
    uncached_since_flush = 0
    if conf.PREDICTION_CACHE and os.path.isfile(conf.PREDICTION_CACHE):
        payload = torch.load(conf.PREDICTION_CACHE, map_location="cpu")
        if payload.get("identity") != cache_identity:
            raise ValueError("prediction cache identity does not match evaluation")
        cached_predictions = payload["predictions"]

    event_matches = event_gt = event_pred = 0
    down_matches = down_gt = down_pred = 0
    fixed_tolerances = (0.025, 0.05, 0.10, 0.15, 0.20)
    up_matches = {tolerance: 0 for tolerance in fixed_tolerances}
    interval_matches = {tolerance: 0 for tolerance in fixed_tolerances}
    down_matches_by_tolerance = {tolerance: 0 for tolerance in fixed_tolerances}
    overlap_weighted = 0.0
    frame_tp = frame_fp = frame_fn = 0
    file_results = []

    for mel, roll, md in dataset:
        def probabilities_from_logits(logits):
            probabilities = logits.clone()
            if model_type in ("mobile_regression", "mobile_edge_10ms",
                              "mobile_edge_multirate_10ms",
                              "mobile_dual_timescale",
                              "mobile_dynamic_crf", "mobile_dynamic_crf_flatten",
                              "mobile_dynamic_crf_two_stream",
                              "mobile_offline_oracle",
                              "mobile_offline_oracle_flatten"):
                probabilities[:, (0, 1, 3)] = torch.sigmoid(
                    probabilities[:, (0, 1, 3)])
                probabilities[:, (2, 4)] = torch.tanh(
                    probabilities[:, (2, 4)])
                if model_type not in ("mobile_dynamic_crf",
                                      "mobile_dynamic_crf_flatten",
                                      "mobile_dynamic_crf_two_stream"):
                    probabilities = probabilities[:, :5]
            else:
                probabilities = torch.sigmoid(logits)
            if model_type == "frozen":
                probabilities = torch.nn.functional.pad(probabilities, (1, 0))
            return probabilities

        def inference(chunk):
            if (model_type in ("mobile_offline_oracle",
                               "mobile_offline_oracle_flatten") and
                    conf.OFFLINE_FUTURE_LAG_SECS is not None):
                lag_frames = round(
                    conf.OFFLINE_FUTURE_LAG_SECS / seconds_per_frame)
                logits = model.forward_fixed_lag(
                    chunk, lag_frames, conf.OFFLINE_LAG_WINDOW_BATCH)
            else:
                logits = model(chunk)
            return (probabilities_from_logits(logits),)

        cache_key = md[0]
        probabilities = cached_predictions.get(cache_key)
        if probabilities is None:
            tensor = torch.from_numpy(mel).unsqueeze(0).to(conf.DEVICE)
            original_frames = tensor.shape[-1]
            stride_frames = chunk_frames - overlap_frames
            remainder = original_frames % stride_frames
            tail_padding = 3 - remainder if 0 < remainder < 3 else 0
            # Right padding supplies real inference positions for delayed event
            # outputs near the end of a recording. State remains undelayed; only
            # confidence/offset event channels are shifted back onto target time.
            tensor = torch.nn.functional.pad(
                tensor, (0, tail_padding + future_frames))
            if (conf.FULL_RECORDING_INFERENCE and
                    model_type in ("mobile_offline_oracle",
                                   "mobile_offline_oracle_flatten")):
                def frontend_inference(chunk):
                    features = model.extract_frontend(chunk).transpose(1, 2)
                    return (features,)
                features = strided_inference(
                    frontend_inference, tensor, chunk_frames,
                    overlap_frames)[0].transpose(1, 2).to(conf.DEVICE)
                if conf.OFFLINE_FUTURE_LAG_SECS is None:
                    logits = model.forward_features(features)
                else:
                    total_lag_frames = round(
                        conf.OFFLINE_FUTURE_LAG_SECS / seconds_per_frame)
                    recurrent_lag_frames = (
                        total_lag_frames - model.frontend_future_frames)
                    if recurrent_lag_frames < 0:
                        raise ValueError(
                            "offline lag is shorter than frontend lookahead")
                    logits = model.forward_fixed_lag_features(
                        features, recurrent_lag_frames,
                        conf.OFFLINE_LAG_WINDOW_BATCH)
                raw_probabilities = probabilities_from_logits(logits)[0].cpu()
            elif conf.FULL_RECORDING_INFERENCE:
                raw_probabilities = inference(tensor)[0][0]
            else:
                raw_probabilities = strided_inference(
                    inference, tensor, chunk_frames, overlap_frames)[0][0]
            probabilities = raw_probabilities[:, :original_frames].clone()
            if future_frames:
                if model_type in ("mobile_dynamic_crf",
                                   "mobile_dynamic_crf_flatten",
                                   "mobile_dynamic_crf_two_stream"):
                    # Unary and transition potentials were jointly supervised
                    # at output t+D against the path at t. Shift them together.
                    probabilities[:] = raw_probabilities[
                        :, future_frames:future_frames + original_frames]
                else:
                    probabilities[1:] = raw_probabilities[
                        1:, future_frames:future_frames + original_frames]
            cached_predictions[cache_key] = probabilities
            cache_dirty = True
            uncached_since_flush += 1
            if conf.PREDICTION_CACHE and uncached_since_flush >= 8:
                os.makedirs(
                    os.path.dirname(os.path.abspath(conf.PREDICTION_CACHE)),
                    exist_ok=True)
                temporary = conf.PREDICTION_CACHE + f".tmp-{os.getpid()}"
                torch.save({"identity": cache_identity,
                            "predictions": cached_predictions}, temporary)
                os.replace(temporary, conf.PREDICTION_CACHE)
                uncached_since_flush = 0
        if conf.DECODER == "regression":
            events = decode_regression_pedal_events(
                probabilities[:5], conf.STATE_THRESHOLD, conf.EVENT_THRESHOLD,
                conf.NMS_RADIUS)
        elif conf.DECODER == "event":
            events = decode_pedal_events(
                probabilities, conf.STATE_THRESHOLD, conf.EVENT_THRESHOLD)
        elif conf.DECODER == "state":
            events = decode_state_transitions(
                probabilities, conf.STATE_THRESHOLD)
        elif conf.DECODER == "crf":
            if model_type not in ("mobile_dynamic_crf",
                                  "mobile_dynamic_crf_flatten",
                                  "mobile_dynamic_crf_two_stream"):
                raise ValueError("CRF decoder requires mobile_dynamic_crf")
            events = decode_dynamic_crf_pedal_events(
                probabilities, conf.CRF_CONFIDENCE_WEIGHT)
        elif conf.DECODER == "semi_markov":
            if model_type not in ("mobile_dynamic_crf",
                                  "mobile_dynamic_crf_flatten",
                                  "mobile_dynamic_crf_two_stream"):
                raise ValueError("semi-Markov decoder requires dynamic CRF outputs")
            events = decode_bounded_semi_markov_pedal_events(
                probabilities,
                confidence_weight=conf.SEMI_MARKOV_CONFIDENCE_WEIGHT,
                transition_weight=conf.SEMI_MARKOV_TRANSITION_WEIGHT,
                state_weight=conf.SEMI_MARKOV_STATE_WEIGHT,
                candidate_threshold=conf.SEMI_MARKOV_CANDIDATE_THRESHOLD,
                nms_radius=conf.NMS_RADIUS,
                min_up_frames=conf.SEMI_MARKOV_MIN_UP_FRAMES,
                min_down_frames=conf.SEMI_MARKOV_MIN_DOWN_FRAMES,
                max_up_frames=conf.SEMI_MARKOV_MAX_UP_FRAMES,
                max_down_frames=conf.SEMI_MARKOV_MAX_DOWN_FRAMES,
                event_penalty=conf.SEMI_MARKOV_EVENT_PENALTY)
        else:
            raise ValueError(f"unknown decoder: {conf.DECODER}")
        predicted = intervals_from_decoded(events, seconds_per_frame)

        midi_path = GtLoaderMaestro.get_metadata_path(md, metadata)
        _, sustain_states, _, _, _ = GtLoaderMaestro.get_midi_eventdata(midi_path)
        ground_truth = pedal_intervals_from_states(sustain_states)

        gt_state = torch.from_numpy(roll[-3] >= 64)
        pred_state = probabilities[0].cpu() >= conf.STATE_THRESHOLD
        length = min(len(gt_state), len(pred_state))
        gt_state, pred_state = gt_state[:length], pred_state[:length]
        frame_tp += int((gt_state & pred_state).sum())
        frame_fp += int((~gt_state & pred_state).sum())
        frame_fn += int((gt_state & ~pred_state).sum())

        event_gt += len(ground_truth)
        event_pred += len(predicted)
        down_gt += len(ground_truth)
        down_pred += len(predicted)
        down_matches += count_onset_matches(
            ground_truth, predicted, conf.EVENT_TOLERANCE_SECS)
        for tolerance in fixed_tolerances:
            down_matches_by_tolerance[tolerance] += count_onset_matches(
                ground_truth, predicted, tolerance)
            up_matches[tolerance] += count_offset_matches(
                ground_truth, predicted, tolerance)
            interval_matches[tolerance] += count_fixed_interval_matches(
                ground_truth, predicted, conf.EVENT_TOLERANCE_SECS, tolerance)
        if ground_truth and predicted:
            gt_file = np.asarray(ground_truth, dtype=float).reshape(-1, 2)
            pred_file = np.asarray(predicted, dtype=float).reshape(-1, 2)
            file_precision, _, _, file_overlap = precision_recall_f1_overlap(
                gt_file, np.full(len(gt_file), 64.0),
                pred_file, np.full(len(pred_file), 64.0),
                onset_tolerance=conf.EVENT_TOLERANCE_SECS,
                pitch_tolerance=0.1,
                offset_ratio=0.2,
                offset_min_tolerance=conf.EVENT_TOLERANCE_SECS,
            )
            file_matches = int(round(file_precision * len(predicted)))
            event_matches += file_matches
            overlap_weighted += file_overlap * file_matches
        file_results.append({
            "name": md[0], "gt_intervals": len(ground_truth),
            "pred_intervals": len(predicted),
        })
        print(json.dumps(file_results[-1]), flush=True)

    precision = event_matches / max(event_pred, 1)
    recall = event_matches / max(event_gt, 1)
    event_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    overlap = overlap_weighted / max(event_matches, 1)
    down_precision = down_matches / max(down_pred, 1)
    down_recall = down_matches / max(down_gt, 1)
    down_f1 = 2 * down_precision * down_recall / max(
        down_precision + down_recall, 1e-12)
    state_precision = frame_tp / max(frame_tp + frame_fp, 1)
    state_recall = frame_tp / max(frame_tp + frame_fn, 1)
    state_f1 = 2 * state_precision * state_recall / max(
        state_precision + state_recall, 1e-12)
    def prf(matches):
        precision = matches / max(event_pred, 1)
        recall = matches / max(event_gt, 1)
        return {"precision": precision, "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "matched": matches}
    offline_oracle = model_type in (
        "mobile_offline_oracle", "mobile_offline_oracle_flatten")
    if offline_oracle and conf.OFFLINE_FUTURE_LAG_SECS is None:
        recurrent_context_mode = "full_bidirectional_recording"
        recurrent_future_context_secs = None
    elif offline_oracle:
        recurrent_context_mode = "fixed_lag"
        recurrent_future_context_secs = conf.OFFLINE_FUTURE_LAG_SECS
    else:
        recurrent_context_mode = "causal"
        recurrent_future_context_secs = 0.0
    result = {
        "config": OmegaConf.to_container(conf),
        # FUTURE_CONTEXT_SECS is the explicit event-output delay.  It is not
        # the recurrent future scope of an offline bidirectional oracle.
        "effective_future_context_secs": future_context_secs,
        "recurrent_context_mode": recurrent_context_mode,
        "recurrent_future_context_secs": recurrent_future_context_secs,
        "checkpoint_step": saved["step"],
        "base_sha256": saved.get("base_sha256"),
        "files": len(file_results),
        "ground_truth_intervals": event_gt,
        "predicted_intervals": event_pred,
        "matched_intervals": event_matches,
        "pedal_state": {
            "precision": state_precision, "recall": state_recall, "f1": state_f1,
        },
        "pedal_event_onset_offset": {
            "precision": precision, "recall": recall, "f1": event_f1,
            "overlap": overlap,
        },
        "pedal_down": {
            "precision": down_precision, "recall": down_recall,
            "f1": down_f1, "matched": down_matches,
        },
        "fixed_tolerance_diagnostic": {
            f"{round(tolerance * 1000)}ms": {
                "pedal_down": prf(down_matches_by_tolerance[tolerance]),
                "pedal_up": prf(up_matches[tolerance]),
                "down_and_up": prf(interval_matches[tolerance]),
            } for tolerance in fixed_tolerances
        },
        "per_file_counts": file_results,
    }
    result_path = os.path.abspath(conf.RESULTS_JSON)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result_temporary = result_path + f".tmp-{os.getpid()}"
    with open(result_temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    os.replace(result_temporary, result_path)
    if conf.PREDICTION_CACHE and cache_dirty:
        os.makedirs(os.path.dirname(os.path.abspath(conf.PREDICTION_CACHE)),
                    exist_ok=True)
        temporary = conf.PREDICTION_CACHE + f".tmp-{os.getpid()}"
        torch.save({"identity": cache_identity,
                    "predictions": cached_predictions}, temporary)
        os.replace(temporary, conf.PREDICTION_CACHE)
    print(json.dumps(result, indent=2), flush=True)
