"""Batched two-state input-dependent linear-chain CRF utilities."""

from collections import deque

import numpy as np
import torch


def _validate(unary, transitions, targets=None):
    if unary.ndim != 3 or unary.shape[-1] != 2:
        raise ValueError("unary must have shape (batch, time, 2)")
    if unary.shape[1] < 1:
        raise ValueError("CRF sequence must contain at least one frame")
    if transitions.shape != (*unary.shape[:2], 2, 2):
        raise ValueError("transitions must have shape (batch, time, 2, 2)")
    if targets is not None:
        if targets.shape != unary.shape[:2]:
            raise ValueError("targets must have shape (batch, time)")
        if targets.dtype != torch.long:
            raise ValueError("targets must use torch.long")


def crf_log_partition(unary, transitions):
    """Return log partition for a two-state, input-dependent chain.

    ``transitions[:, t, i, j]`` scores state i at t-1 transitioning to
    state j at t. The transition slice at t=0 is intentionally unused.
    """
    _validate(unary, transitions)
    alpha = unary[:, 0]
    for frame in range(1, unary.shape[1]):
        scores = alpha.unsqueeze(-1) + transitions[:, frame]
        alpha = torch.logsumexp(scores, dim=1) + unary[:, frame]
    return torch.logsumexp(alpha, dim=1)


def crf_path_score(unary, transitions, targets):
    """Score labelled state paths without normalizing them."""
    _validate(unary, transitions, targets)
    score = unary[:, 0].gather(1, targets[:, :1]).squeeze(1)
    for frame in range(1, unary.shape[1]):
        previous, current = targets[:, frame - 1], targets[:, frame]
        transition = transitions[:, frame]
        score = score + transition[
            torch.arange(len(targets), device=targets.device), previous, current]
        score = score + unary[:, frame].gather(
            1, current.unsqueeze(1)).squeeze(1)
    return score


def crf_nll(unary, transitions, targets, reduction="mean"):
    """Globally normalized negative log-likelihood."""
    losses = crf_log_partition(unary, transitions) - crf_path_score(
        unary, transitions, targets)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"unknown reduction: {reduction}")


def viterbi_decode(unary, transitions):
    """Return MAP state paths and their scores using batched backpointers."""
    _validate(unary, transitions)
    # Full recordings contain tens of thousands of frames. For the common
    # single-recording CPU decode, tiny per-frame torch kernels dominate the
    # actual two-state DP. Scalar NumPy arithmetic is exactly equivalent and
    # substantially faster; keep the batched torch path for training/GPU use.
    if unary.device.type == "cpu" and unary.shape[0] == 1:
        unary_np = unary.detach().numpy()[0]
        transition_np = transitions.detach().numpy()[0]
        frames = unary_np.shape[0]
        backpointers = np.empty((frames, 2), dtype=np.int8)
        score_up = unary_np[0, 0]
        score_down = unary_np[0, 1]
        for frame in range(1, frames):
            up_from_up = score_up + transition_np[frame, 0, 0]
            up_from_down = score_down + transition_np[frame, 1, 0]
            down_from_up = score_up + transition_np[frame, 0, 1]
            down_from_down = score_down + transition_np[frame, 1, 1]
            backpointers[frame, 0] = int(up_from_down > up_from_up)
            backpointers[frame, 1] = int(down_from_down > down_from_up)
            score_up = max(up_from_up, up_from_down) + unary_np[frame, 0]
            score_down = max(down_from_up, down_from_down) + unary_np[frame, 1]
        current = int(score_down > score_up)
        path = np.empty(frames, dtype=np.int64)
        path[-1] = current
        for frame in range(frames - 1, 0, -1):
            current = int(backpointers[frame, current])
            path[frame - 1] = current
        return (torch.from_numpy(path).unsqueeze(0),
                unary.new_tensor([max(score_up, score_down)]))
    scores = unary[:, 0]
    backpointers = []
    for frame in range(1, unary.shape[1]):
        candidates = scores.unsqueeze(-1) + transitions[:, frame]
        best_scores, previous = candidates.max(dim=1)
        scores = best_scores + unary[:, frame]
        backpointers.append(previous)
    final_scores, final_states = scores.max(dim=1)
    paths = [final_states]
    current = final_states
    batch = torch.arange(len(unary), device=unary.device)
    for previous in reversed(backpointers):
        current = previous[batch, current]
        paths.append(current)
    return torch.stack(list(reversed(paths)), dim=1), final_scores


def bounded_semi_markov_decode(unary, boundary, candidates,
                               min_duration=(1, 1),
                               max_duration=(None, None),
                               event_penalty=0.0):
    """Decode one alternating binary path by scoring complete intervals.

    ``unary[t, s]`` scores frame ``t`` in state ``s``. ``boundary[t, s]``
    scores a transition into state ``s`` at frame ``t`` and
    ``candidates[t, s]`` determines which acoustic peaks may be boundaries.
    Consecutive boundaries must alternate states. Duration bounds apply to
    complete interior segments only; the first and last segments are censored
    by the recording boundary and are intentionally left unbounded.

    The return value is a one-dimensional state path, its score, and the
    selected ``(frame, new_state)`` boundaries. This is a segmental Viterbi
    recurrence (a two-state bounded semi-Markov model), not a greedy peak
    filter.
    """
    if unary.ndim != 2 or unary.shape[-1] != 2:
        raise ValueError("unary must have shape (time, 2)")
    if boundary.shape != unary.shape or candidates.shape != unary.shape:
        raise ValueError("boundary and candidates must match unary")
    if unary.shape[0] < 1:
        raise ValueError("sequence must contain at least one frame")
    if len(min_duration) != 2 or len(max_duration) != 2:
        raise ValueError("duration bounds must have two entries")
    for state in range(2):
        if min_duration[state] < 1:
            raise ValueError("minimum durations must be positive")
        if (max_duration[state] is not None and
                max_duration[state] < min_duration[state]):
            raise ValueError("maximum duration precedes minimum duration")

    # Decoding is deliberately CPU/NumPy: recordings are long, the candidate
    # graph is sparse, and launching tiny tensor kernels for every edge is much
    # slower than scalar arithmetic.
    unary_np = unary.detach().cpu().numpy().astype(np.float64, copy=False)
    boundary_np = boundary.detach().cpu().numpy().astype(np.float64, copy=False)
    candidates_np = candidates.detach().cpu().numpy().astype(bool, copy=False)
    frames = len(unary_np)
    prefix = np.concatenate((np.zeros((1, 2), dtype=np.float64),
                             np.cumsum(unary_np, axis=0)), axis=0)

    nodes = [(frame, state) for frame in range(1, frames)
             for state in range(2) if candidates_np[frame, state]]
    scores = np.full(len(nodes), -np.inf, dtype=np.float64)
    previous = np.full(len(nodes), -1, dtype=np.int64)
    by_state = tuple(
        [index for index, (_, node_state) in enumerate(nodes)
         if node_state == state]
        for state in range(2))
    add_cursor = [0, 0]
    eligible = [deque(), deque()]
    for index, (frame, state) in enumerate(nodes):
        old_state = 1 - state
        # The initial segment is left-censored, so no duration bound applies.
        scores[index] = (prefix[frame, old_state] + unary_np[frame, state] +
                         boundary_np[frame, state] - event_penalty)
        state_nodes = by_state[old_state]
        while add_cursor[old_state] < len(state_nodes):
            old_index = state_nodes[add_cursor[old_state]]
            old_frame, _ = nodes[old_index]
            if old_frame > frame - min_duration[old_state]:
                break
            value = scores[old_index] - prefix[old_frame + 1, old_state]
            while eligible[old_state] and eligible[old_state][-1][1] <= value:
                eligible[old_state].pop()
            eligible[old_state].append((old_index, value))
            add_cursor[old_state] += 1
        maximum = max_duration[old_state]
        if maximum is not None:
            oldest = frame - maximum
            while (eligible[old_state] and
                   nodes[eligible[old_state][0][0]][0] < oldest):
                eligible[old_state].popleft()
        if eligible[old_state]:
            old_index, old_value = eligible[old_state][0]
            candidate_score = (
                old_value + prefix[frame, old_state] +
                unary_np[frame, state] + boundary_np[frame, state] -
                event_penalty)
            if candidate_score > scores[index]:
                scores[index] = candidate_score
                previous[index] = old_index

    # A recording may contain no complete interval at all.
    best_score = float(prefix[frames].max())
    best_index = -1
    best_constant_state = int(prefix[frames].argmax())
    for index, (frame, state) in enumerate(nodes):
        total = scores[index] + prefix[frames, state] - prefix[frame + 1, state]
        if total > best_score:
            best_score, best_index = float(total), index

    if best_index < 0:
        path = np.full(frames, best_constant_state, dtype=np.int64)
        return (torch.from_numpy(path).to(unary.device),
                unary.new_tensor(best_score), [])

    selected = []
    index = best_index
    while index >= 0:
        selected.append(nodes[index])
        index = int(previous[index])
    selected.reverse()
    initial_state = 1 - selected[0][1]
    path = np.full(frames, initial_state, dtype=np.int64)
    for frame, state in selected:
        path[frame:] = state
    return (torch.from_numpy(path).to(unary.device),
            unary.new_tensor(best_score), selected)


def state_unary_from_logit(state_logits):
    """Convert one binary state logit to symmetric UP/DOWN CRF potentials."""
    return torch.stack((-0.5 * state_logits, 0.5 * state_logits), dim=-1)


def transition_event_logits(transitions):
    """Return gauge-invariant DOWN and UP event odds from 2x2 potentials.

    At an UP frame, the relevant DOWN evidence is the score advantage of
    ``UP->DOWN`` over ``UP->UP``. At a DOWN frame, UP evidence is likewise
    ``DOWN->UP`` relative to ``DOWN->DOWN``. Supervising these differences
    avoids assigning meaning to an arbitrary common offset in a CRF row.
    """
    if transitions.ndim < 3 or transitions.shape[-2:] != (2, 2):
        raise ValueError("transitions must end in shape (2, 2)")
    down = transitions[..., 0, 1] - transitions[..., 0, 0]
    up = transitions[..., 1, 0] - transitions[..., 1, 1]
    return down, up
