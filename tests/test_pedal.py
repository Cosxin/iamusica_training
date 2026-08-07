import torch

from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.models.pedal import FrozenPedalModel
from ov_piano.pedal import (
    decode_bounded_semi_markov_pedal_events,
    decode_pedal_events, decode_regression_pedal_events,
    decode_dynamic_crf_pedal_events,
    exact_event_targets, pedal_targets_from_roll)


def test_pedal_targets_preserve_boundary_and_transitions():
    roll = torch.zeros(1, 259, 10)
    roll[:, -3, :3] = 127
    roll[:, -3, 5:8] = 127
    state, down, up = pedal_targets_from_roll(roll, event_radius=0)[0]
    assert state.tolist() == [1, 1, 0, 0, 1, 1, 1, 0, 0]
    assert down.nonzero().flatten().tolist() == [4]
    assert up.nonzero().flatten().tolist() == [2, 7]


def test_smoothed_targets_remain_probabilities():
    roll = torch.zeros(2, 259, 20)
    roll[:, -3, 10:] = 127
    targets = pedal_targets_from_roll(roll, event_radius=3)
    assert targets.shape == (2, 3, 19)
    assert torch.all((targets >= 0) & (targets <= 1))
    assert targets[:, 1].amax().item() == 1


def test_frozen_pedal_model_shapes_and_gradients():
    base = OnsetsAndVelocities(2, 229, 88)
    model = FrozenPedalModel(base, adapter_channels=32, hidden_size=16,
                             gru_layers=1, dropout=0)
    model.train()
    output = model(torch.rand(2, 229, 32))
    assert output.shape == (2, 3, 31)
    output.sum().backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert any(parameter.grad is not None for parameter in model.head.parameters())
    assert not model.base.training


def test_decoder_requires_state_consistency():
    probabilities = torch.zeros(3, 8)
    probabilities[0, 2:6] = 0.9
    probabilities[1, 2] = 0.8
    probabilities[2, 6] = 0.7
    assert decode_pedal_events(probabilities) == [
        (2, True, probabilities[1, 2].item()),
        (6, False, probabilities[2, 6].item()),
    ]


def test_exact_targets_preserve_fractional_event_time():
    target = exact_event_targets(
        torch.zeros(12), [0.055], [0.151], 0.0, 0.024, 12, radius=2)
    assert target.shape == (5, 12)
    assert target[1].argmax().item() == 2
    assert abs(target[2, 2].item() - (0.055 / 0.024 - 2)) < 1e-6
    assert target[3].argmax().item() == 6


def test_exact_targets_can_align_state_to_event_centers():
    # The supplied roll state changes early at frame 2. Exact event alignment
    # instead changes at round(2.6)=3, and reconstructs the state at a chunk
    # beginning after a previous down event.
    roll_state = torch.tensor([0, 0, 1, 1, 1, 0, 0], dtype=torch.float32)
    target = exact_event_targets(
        roll_state, [0.026], [0.052], 0.0, 0.01, 7, radius=1,
        align_state_to_events=True)
    assert target[0].tolist() == [0, 0, 0, 1, 1, 0, 0]
    continued = exact_event_targets(
        torch.zeros(4), [0.02], [0.13], 0.10, 0.01, 4, radius=1,
        align_state_to_events=True)
    assert continued[0].tolist() == [1, 1, 1, 0]


def test_regression_decoder_refines_and_alternates_events():
    outputs = torch.zeros(5, 12)
    outputs[1, 2], outputs[2, 2] = 0.9, 0.25
    outputs[1, 3] = 0.7  # suppressed by NMS
    outputs[3, 7], outputs[4, 7] = 0.8, -0.2
    events = decode_regression_pedal_events(outputs, event_threshold=0.5,
                                            nms_radius=2)
    assert [(round(t, 2), active) for t, active, _ in events] == [
        (2.25, True), (6.8, False)]


def test_regression_decoder_clamps_offsets_to_recording_bounds():
    outputs = torch.zeros(5, 4)
    outputs[1, 0], outputs[2, 0] = 1.0, -1.0
    outputs[3, 3], outputs[4, 3] = 1.0, 1.0
    events = decode_regression_pedal_events(
        outputs, event_threshold=0.5, nms_radius=1)
    assert [(timestamp, active) for timestamp, active, _ in events] == [
        (0.0, True), (3.0, False)]


def test_dynamic_crf_decoder_uses_matrix_order_and_refines_transitions():
    outputs = torch.zeros(9, 7)
    outputs[0] = 0.5
    outputs[0, 0] = 0.01  # establish an UP start without fixing it externally
    # Default preference is to remain in the current state.
    outputs[5] = 4.0   # UP -> UP
    outputs[6] = -4.0  # UP -> DOWN
    outputs[7] = -4.0  # DOWN -> UP
    outputs[8] = 4.0   # DOWN -> DOWN
    outputs[6, 2] = 20.0  # force pedal down at frame 2
    outputs[7, 5] = 20.0  # force pedal up at frame 5
    outputs[1, 2], outputs[2, 2] = 0.91, 0.25
    outputs[3, 5], outputs[4, 5] = 0.82, -0.4
    events = decode_dynamic_crf_pedal_events(outputs)
    assert [(round(t, 2), active) for t, active, _ in events] == [
        (2.25, True), (4.6, False)]
    assert [round(score, 2) for _, _, score in events] == [0.91, 0.82]


def test_dynamic_crf_decoder_can_couple_event_confidence_to_transitions():
    outputs = torch.zeros(9, 7)
    outputs[0] = 0.5
    outputs[0, 0] = 0.01
    outputs[5] = 4.0
    outputs[6] = -4.0
    outputs[7] = -4.0
    outputs[8] = 4.0
    outputs[1, 2] = 1.0
    outputs[3, 5] = 1.0
    assert decode_dynamic_crf_pedal_events(outputs) == []
    events = decode_dynamic_crf_pedal_events(
        outputs, transition_confidence_weight=1.0)
    assert [(timestamp, active) for timestamp, active, _ in events] == [
        (2.0, True), (5.0, False)]


def test_semi_markov_decoder_pairs_short_gap_boundaries():
    outputs = torch.zeros(9, 9)
    outputs[0] = 0.9  # State evidence wants one uninterrupted DOWN segment.
    outputs[1] = outputs[3] = 0.001
    outputs[3, 3], outputs[4, 3] = 0.99, 0.25
    outputs[1, 5], outputs[2, 5] = 0.99, -0.25
    events = decode_bounded_semi_markov_pedal_events(
        outputs, transition_weight=0.0, state_weight=0.1,
        candidate_threshold=0.05, nms_radius=1, min_up_frames=2)
    assert [(round(timestamp, 2), active) for timestamp, active, _ in events] == [
        (3.25, False), (4.75, True)]

    returned_events, selected = decode_bounded_semi_markov_pedal_events(
        outputs, transition_weight=0.0, state_weight=0.1,
        candidate_threshold=0.05, nms_radius=1, min_up_frames=2,
        return_selected=True)
    assert returned_events == events
    assert selected == [(3, 0), (5, 1)]
