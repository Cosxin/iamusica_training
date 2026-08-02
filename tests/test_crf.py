import itertools

import torch

from ov_piano.crf import (
    bounded_semi_markov_decode, crf_log_partition, crf_nll, crf_path_score,
    state_unary_from_logit, transition_event_logits, viterbi_decode)


def enumerate_scores(unary, transitions):
    scores, paths = [], []
    for states in itertools.product((0, 1), repeat=unary.shape[0]):
        score = unary[0, states[0]]
        for frame in range(1, len(states)):
            score = score + transitions[frame, states[frame-1], states[frame]]
            score = score + unary[frame, states[frame]]
        scores.append(score)
        paths.append(states)
    return torch.stack(scores), paths


def test_transition_event_logits_are_relative_row_odds():
    transitions = torch.tensor([[[[1.0, 3.5], [-2.0, 4.0]]]])
    down, up = transition_event_logits(transitions)
    assert down.tolist() == [[2.5]]
    assert up.tolist() == [[-6.0]]
    # A common row offset is a CRF gauge transformation and must not change
    # the supervised transition evidence.
    shifted = transitions + torch.tensor([[[7.0, 7.0], [-3.0, -3.0]]])
    shifted_down, shifted_up = transition_event_logits(shifted)
    assert torch.equal(down, shifted_down)
    assert torch.equal(up, shifted_up)


def test_crf_partition_and_viterbi_match_brute_force():
    torch.manual_seed(7)
    unary = torch.randn(2, 5, 2, dtype=torch.float64)
    transitions = torch.randn(2, 5, 2, 2, dtype=torch.float64)
    paths, scores = viterbi_decode(unary, transitions)
    partitions = crf_log_partition(unary, transitions)
    for batch in range(2):
        brute_scores, brute_paths = enumerate_scores(
            unary[batch], transitions[batch])
        assert torch.allclose(partitions[batch], torch.logsumexp(brute_scores, 0))
        index = int(brute_scores.argmax())
        assert tuple(paths[batch].tolist()) == brute_paths[index]
        assert torch.allclose(scores[batch], brute_scores[index])


def test_crf_nll_has_finite_gradients_and_matches_path_score():
    torch.manual_seed(11)
    unary = torch.randn(3, 9, 2, requires_grad=True)
    transitions = torch.randn(3, 9, 2, 2, requires_grad=True)
    targets = torch.randint(0, 2, (3, 9))
    loss = crf_nll(unary, transitions, targets)
    expected = (crf_log_partition(unary, transitions) -
                crf_path_score(unary, transitions, targets)).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(unary.grad).all()
    assert torch.isfinite(transitions.grad).all()


def test_symmetric_binary_unary_preserves_logit_difference():
    logits = torch.tensor([[-2.0, 0.0, 3.0]])
    unary = state_unary_from_logit(logits)
    assert torch.equal(unary[..., 1] - unary[..., 0], logits)


def test_partition_probabilities_sum_to_one_and_gradients_are_marginals():
    """This directly audits that forward DP sums every possible state path."""
    torch.manual_seed(19)
    frames = 4
    unary = torch.randn(1, frames, 2, dtype=torch.float64, requires_grad=True)
    transitions = torch.randn(
        1, frames, 2, 2, dtype=torch.float64, requires_grad=True)
    log_z = crf_log_partition(unary, transitions)[0]
    unary_grad, transition_grad = torch.autograd.grad(
        log_z, (unary, transitions))

    brute_scores, paths = enumerate_scores(unary[0].detach(), transitions[0].detach())
    probabilities = torch.exp(brute_scores - log_z.detach())
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))

    expected_unary = torch.zeros_like(unary[0])
    expected_transition = torch.zeros_like(transitions[0])
    for probability, path in zip(probabilities, paths):
        for frame, state in enumerate(path):
            expected_unary[frame, state] += probability
        for frame in range(1, frames):
            expected_transition[frame, path[frame-1], path[frame]] += probability
    assert torch.allclose(unary_grad[0], expected_unary)
    assert torch.allclose(transition_grad[0], expected_transition)
    assert torch.equal(transition_grad[0, 0], torch.zeros(2, 2, dtype=torch.float64))


def test_nll_is_invariant_to_per_frame_common_score_offsets():
    torch.manual_seed(23)
    unary = torch.randn(2, 7, 2, dtype=torch.float64)
    transitions = torch.randn(2, 7, 2, 2, dtype=torch.float64)
    targets = torch.randint(0, 2, (2, 7))
    expected = crf_nll(unary, transitions, targets, reduction="none")

    unary_offsets = torch.randn(2, 7, 1, dtype=torch.float64)
    transition_offsets = torch.randn(2, 7, 1, 1, dtype=torch.float64)
    transition_offsets[:, 0] = 0
    actual = crf_nll(
        unary + unary_offsets, transitions + transition_offsets,
        targets, reduction="none")
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_long_high_magnitude_chain_is_finite_in_float32():
    torch.manual_seed(29)
    unary = 25 * torch.randn(3, 1000, 2)
    transitions = 25 * torch.randn(3, 1000, 2, 2)
    targets = torch.randint(0, 2, (3, 1000))
    losses = crf_nll(unary, transitions, targets, reduction="none")
    paths, scores = viterbi_decode(unary, transitions)
    assert torch.isfinite(losses).all()
    assert torch.isfinite(scores).all()
    assert paths.shape == targets.shape


def test_crf_rejects_empty_sequences():
    try:
        crf_log_partition(torch.empty(2, 0, 2), torch.empty(2, 0, 2, 2))
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("empty CRF sequence accepted")


def test_bounded_semi_markov_scores_complete_alternating_intervals():
    # State evidence alone prefers DOWN throughout, but the paired UP and DOWN
    # boundaries make a short re-pedaling gap globally optimal.
    unary = torch.tensor([[0., 1.]] * 8)
    boundary = torch.full((8, 2), -100.0)
    candidates = torch.zeros(8, 2, dtype=torch.bool)
    boundary[3, 0], candidates[3, 0] = 4.0, True
    boundary[5, 1], candidates[5, 1] = 4.0, True
    path, score, selected = bounded_semi_markov_decode(
        unary, boundary, candidates, min_duration=(2, 1))
    assert selected == [(3, 0), (5, 1)]
    assert path.tolist() == [1, 1, 1, 0, 0, 1, 1, 1]
    assert score.item() == 14.0


def test_bounded_semi_markov_enforces_duration_on_interior_segments_only():
    unary = torch.zeros(7, 2)
    boundary = torch.full((7, 2), -100.0)
    candidates = torch.zeros(7, 2, dtype=torch.bool)
    boundary[2, 0], candidates[2, 0] = 5.0, True
    boundary[3, 1], candidates[3, 1] = 5.0, True
    boundary[5, 1], candidates[5, 1] = 4.0, True
    _, _, selected = bounded_semi_markov_decode(
        unary, boundary, candidates, min_duration=(2, 1))
    # The one-frame UP segment 2->3 is illegal. The initial and final censored
    # segments do not need to satisfy the minimum duration.
    assert selected == [(2, 0), (5, 1)]


def test_bounded_semi_markov_matches_exhaustive_candidate_paths():
    torch.manual_seed(41)
    unary = torch.randn(5, 2, dtype=torch.float64)
    boundary = torch.randn(5, 2, dtype=torch.float64)
    candidates = torch.ones(5, 2, dtype=torch.bool)
    candidates[0] = False
    minimum, maximum, penalty = (1, 2), (3, 4), 0.3
    path, score, selected = bounded_semi_markov_decode(
        unary, boundary, candidates, minimum, maximum, penalty)

    nodes = [(frame, state) for frame in range(1, 5) for state in range(2)]
    brute = []
    for constant_state in (0, 1):
        states = torch.full((5,), constant_state, dtype=torch.long)
        brute.append((float(unary[:, constant_state].sum()), states, []))
    for mask in range(1, 1 << len(nodes)):
        chosen = [node for bit, node in enumerate(nodes) if mask & (1 << bit)]
        if any(chosen[index][0] == chosen[index - 1][0] or
               chosen[index][1] == chosen[index - 1][1]
               for index in range(1, len(chosen))):
            continue
        if any(not (minimum[old_state] <= frame - old_frame <=
                    maximum[old_state])
               for (old_frame, old_state), (frame, _) in
               zip(chosen, chosen[1:])):
            continue
        initial = 1 - chosen[0][1]
        states = torch.full((5,), initial, dtype=torch.long)
        for frame, state in chosen:
            states[frame:] = state
        value = unary[torch.arange(5), states].sum()
        if chosen:
            value = value + sum(boundary[frame, state]
                                for frame, state in chosen)
            value = value - penalty * len(chosen)
        brute.append((float(value), states, chosen))
    brute_score, brute_path, brute_selected = max(brute, key=lambda item: item[0])
    assert abs(score.item() - brute_score) < 1e-10
    assert torch.equal(path.cpu(), brute_path)
    assert selected == brute_selected
