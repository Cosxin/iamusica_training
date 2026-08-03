"""CC64 target construction and event decoding utilities."""

import torch
import torch.nn.functional as F
import math

from ov_piano.crf import (
    bounded_semi_markov_decode, state_unary_from_logit,
    transition_event_logits, viterbi_decode)


def _smooth_impulses(impulses, radius=3):
    if radius <= 0:
        return impulses
    sigma = max(radius / 2.0, 0.5)
    offsets = torch.arange(
        -radius, radius + 1, device=impulses.device, dtype=impulses.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    candidates = []
    for offset, weight in zip(range(-radius, radius + 1), kernel):
        shifted = F.pad(impulses, (radius, radius))
        start = radius - offset
        candidates.append(shifted[..., start:start + impulses.shape[-1]] * weight)
    return torch.stack(candidates).amax(dim=0)


def pedal_targets_from_roll(roll, threshold=64, event_radius=3):
    """Return state/down/up targets aligned to the O&V ``T-1`` outputs.

    The repository roll layout ends in sustain, soft, and sostenuto rows. Using
    adjacent original roll frames preserves the state immediately before a
    sampled chunk and avoids inventing an event at its left boundary.
    """
    if roll.ndim != 3 or roll.shape[1] < 3 or roll.shape[-1] < 2:
        raise ValueError("roll must have shape (batch, channels>=3, time>=2)")
    sustain = roll[:, -3].float()
    previous = sustain[:, :-1] >= threshold
    current = sustain[:, 1:] >= threshold
    down = (~previous & current).float()
    up = (previous & ~current).float()
    state = current.float()
    return torch.stack((
        state,
        _smooth_impulses(down, event_radius),
        _smooth_impulses(up, event_radius),
    ), dim=1)


def decode_pedal_events(probabilities, state_threshold=0.5,
                        event_threshold=0.3):
    """Decode one ``(3, T)`` probability map into frame-indexed transitions."""
    if probabilities.ndim != 2 or probabilities.shape[0] != 3:
        raise ValueError("probabilities must have shape (3, time)")
    state, down, up = probabilities
    active = False
    events = []
    for frame in range(probabilities.shape[-1]):
        wants_down = down[frame] >= event_threshold and state[frame] >= state_threshold
        wants_up = up[frame] >= event_threshold and state[frame] < state_threshold
        if not active and bool(wants_down):
            events.append((frame, True, float(down[frame])))
            active = True
        elif active and bool(wants_up):
            events.append((frame, False, float(up[frame])))
            active = False
    return events


def exact_event_targets(state, down_times, up_times, chunk_start_seconds,
                        seconds_per_frame, frames, radius=3,
                        align_state_to_events=False):
    """Build state/confidence/offset targets from exact CC64 timestamps.

    Offsets are signed fractions of a frame relative to the nearest frame and
    are supervised only inside each event's triangular confidence window.
    """
    result = torch.zeros(5, frames, dtype=torch.float32)
    if align_state_to_events:
        # The confidence peak uses the nearest frame, whereas the historical
        # roll state can change on a different quantized frame. Reconstructing
        # state from the same exact events gives the CRF and auxiliary event
        # losses one internally consistent transition time base.
        events = []
        for active, times in ((True, down_times), (False, up_times)):
            events.extend(
                ((float(timestamp) - chunk_start_seconds) / seconds_per_frame,
                 active)
                for timestamp in times)
        active = False
        cursor = 0
        for position, next_active in sorted(events):
            center = int(round(position))
            if center < 0:
                active = next_active
                continue
            if center >= frames:
                break
            result[0, cursor:center] = float(active)
            active = next_active
            cursor = center
        result[0, cursor:] = float(active)
    else:
        result[0] = torch.as_tensor(state, dtype=torch.float32)[:frames]
    for channel, times in ((1, down_times), (3, up_times)):
        for timestamp in times:
            position = (float(timestamp) - chunk_start_seconds) / seconds_per_frame
            center = int(round(position))
            if center < -radius or center >= frames + radius:
                continue
            for frame in range(max(0, center-radius), min(frames, center+radius+1)):
                confidence = max(0.0, 1.0 - abs(frame-position)/(radius+1))
                if confidence > result[channel, frame]:
                    result[channel, frame] = confidence
                    result[channel+1, frame] = position - frame
    return result


def decode_regression_pedal_events(outputs, state_threshold=0.5,
                                   event_threshold=0.5, nms_radius=3):
    """Peak/NMS decoder with offset refinement and alternating transitions."""
    if outputs.ndim != 2 or outputs.shape[0] != 5:
        raise ValueError("outputs must have shape (5, time)")
    state = outputs[0]
    candidates = []
    for active, confidence_idx, offset_idx in ((True, 1, 2), (False, 3, 4)):
        confidence = outputs[confidence_idx]
        for frame in range(len(confidence)):
            left, right = max(0, frame-nms_radius), min(len(confidence), frame+nms_radius+1)
            if confidence[frame] < event_threshold or confidence[frame] < confidence[left:right].max():
                continue
            refined = frame + float(outputs[offset_idx, frame].clamp(-1, 1))
            refined = max(0.0, min(float(len(confidence) - 1), refined))
            candidates.append((refined, active, float(confidence[frame])))
    candidates.sort(key=lambda item: item[0])
    active = bool(state[0] >= state_threshold)
    events = []
    for timestamp, next_active, score in candidates:
        if next_active != active:
            events.append((timestamp, next_active, score))
            active = next_active
    return events


def decode_dynamic_crf_pedal_events(outputs, transition_confidence_weight=0.0):
    """Decode a nine-channel Dynamic CRF output and refine transition times.

    Channels 0--4 contain state probability, down confidence/offset, and up
    confidence/offset. Channels 5--8 are unnormalized transition potentials in
    row-major ``UP->UP, UP->DOWN, DOWN->UP, DOWN->DOWN`` order.
    """
    if outputs.ndim != 2 or outputs.shape[0] != 9:
        raise ValueError("outputs must have shape (9, time)")
    eps = torch.finfo(outputs.dtype).eps
    state_logit = torch.logit(outputs[0].clamp(eps, 1.0 - eps))
    unary = state_unary_from_logit(state_logit).unsqueeze(0)
    transitions = outputs[5:9].transpose(0, 1).reshape(
        1, outputs.shape[-1], 2, 2).clone()
    if transition_confidence_weight:
        down_logit = torch.logit(outputs[1].clamp(eps, 1.0 - eps))
        up_logit = torch.logit(outputs[3].clamp(eps, 1.0 - eps))
        transitions[0, :, 0, 1] += transition_confidence_weight * down_logit
        transitions[0, :, 1, 0] += transition_confidence_weight * up_logit
    path, _ = viterbi_decode(unary.float(), transitions.float())
    path = path[0]
    events = []
    for frame in range(1, len(path)):
        previous, current = int(path[frame-1]), int(path[frame])
        if previous == current:
            continue
        active = current == 1
        confidence_idx, offset_idx = ((1, 2) if active else (3, 4))
        refined = frame + float(outputs[offset_idx, frame].clamp(-1, 1))
        refined = max(0.0, min(float(outputs.shape[-1] - 1), refined))
        events.append((refined, active, float(outputs[confidence_idx, frame])))
    return events


def decode_bounded_semi_markov_pedal_events(
        outputs, confidence_weight=1.0, transition_weight=1.0,
        state_weight=1.0, candidate_threshold=0.05, nms_radius=3,
        min_up_frames=1, min_down_frames=1,
        max_up_frames=None, max_down_frames=None, event_penalty=0.0,
        return_selected=False):
    """Decode alternating pedal intervals with segmental dynamic programming.

    Candidate boundaries are local confidence peaks. The score combines state
    evidence accumulated over each complete interval with gauge-invariant CRF
    transition odds and event-head log odds at its boundary.
    """
    if outputs.ndim != 2 or outputs.shape[0] != 9:
        raise ValueError("outputs must have shape (9, time)")
    if nms_radius < 0:
        raise ValueError("nms_radius must be nonnegative")
    eps = torch.finfo(outputs.dtype).eps
    state_logit = torch.logit(outputs[0].clamp(eps, 1.0 - eps))
    unary = state_weight * state_unary_from_logit(state_logit)
    transitions = outputs[5:9].transpose(0, 1).reshape(
        outputs.shape[-1], 2, 2)
    down_odds, up_odds = transition_event_logits(transitions)
    confidence = torch.stack((outputs[3], outputs[1]), dim=-1)
    confidence_logit = torch.logit(confidence.clamp(eps, 1.0 - eps))
    transition_odds = torch.stack((up_odds, down_odds), dim=-1)
    boundary = (transition_weight * transition_odds +
                confidence_weight * confidence_logit)

    pooled = F.max_pool1d(
        confidence.transpose(0, 1).unsqueeze(0),
        2 * nms_radius + 1, stride=1, padding=nms_radius)[0].transpose(0, 1)
    candidates = ((confidence >= candidate_threshold) &
                  (confidence >= pooled))
    path, _, selected = bounded_semi_markov_decode(
        unary.float(), boundary.float(), candidates,
        min_duration=(min_up_frames, min_down_frames),
        max_duration=(max_up_frames, max_down_frames),
        event_penalty=event_penalty)
    events = []
    for frame, state in selected:
        active = state == 1
        confidence_idx, offset_idx = ((1, 2) if active else (3, 4))
        refined = frame + float(outputs[offset_idx, frame].clamp(-1, 1))
        refined = max(0.0, min(float(outputs.shape[-1] - 1), refined))
        events.append((refined, active, float(outputs[confidence_idx, frame])))
    return (events, selected) if return_selected else events
