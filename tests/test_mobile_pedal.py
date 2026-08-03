import torch

from export_mobile_pedal_onnx import build_model
from ov_piano.models.mobile_pedal import (
    MBConv, CausalTemporalBlock, CausalTransientBranch, PedalTower,
    EdgePedalAMTRegression,
    MultirateEdgePedalAMTRegression,
    OfflinePedalAMTRegression, OfflinePedalAMTRegressionFlatten,
    StreamingPedalAMT,
    StreamingPedalAMTRegression, StreamingPedalAMTDualTimescale,
    StreamingPedalAMTDynamicCRF, StreamingPedalAMTDynamicCRFFlatten,
    StreamingPedalAMTDynamicCRFTwoStream)


def test_edge_10ms_shape_frequency_grid_and_causality():
    torch.manual_seed(7)
    model = EdgePedalAMTRegression(
        mel_bins=33, channels=(8, 12, 16), frontend_size=24,
        shared_hidden=20, tower_hidden=12, shared_layers=1,
        dropout=0.0).eval()
    inputs = torch.randn(2, 33, 31)
    outputs = model(inputs)
    assert outputs.shape == (2, 5, 31)
    assert model.frontend.remaining_bins == 9
    changed = inputs.clone()
    changed[:, :, 18:] += torch.randn_like(changed[:, :, 18:])
    changed_outputs = model(changed)
    torch.testing.assert_close(outputs[:, :, :18], changed_outputs[:, :, :18])


def test_multirate_edge_shape_causality_and_delay_lane():
    torch.manual_seed(11)
    model = MultirateEdgePedalAMTRegression(
        mel_bins=33, channels=(4, 6, 8), fast_size=12,
        context_hidden=16, context_layers=1, context_pool=4,
        event_delay_frames=7, head_hidden=10, dropout=0.0).eval()
    inputs = torch.randn(2, 33, 37)
    outputs = model(inputs)
    assert outputs.shape == (2, 5, 37)
    assert model.frontend.remaining_bins == 9
    assert model.event_delay_frames == 7
    changed = inputs.clone()
    changed[:, :, 23:] += torch.randn_like(changed[:, :, 23:])
    changed_outputs = model(changed)
    torch.testing.assert_close(outputs[:, :, :23], changed_outputs[:, :, :23])


def test_exporter_reconstructs_multirate_checkpoint_architecture():
    saved = {
        "model_type": "mobile_edge_multirate_10ms",
        "config": {
            "FAST_SIZE": 12, "CONTEXT_HIDDEN": 16, "CONTEXT_LAYERS": 1,
            "CONTEXT_POOL": 4, "EVENT_DELAY_FRAMES": 7,
            "HEAD_HIDDEN": 10, "DROPOUT": 0.0,
        },
    }
    model = build_model(saved, mel_bins=33)
    assert isinstance(model, MultirateEdgePedalAMTRegression)
    assert model.event_delay_frames == 7
    assert model.context_pool == 4
    assert model(torch.randn(1, 33, 37)).shape == (1, 5, 37)


def test_mbconv_shapes_and_frequency_stride():
    block = MBConv(8, 16, expansion=2, frequency_stride=2)
    output = block(torch.randn(2, 8, 31, 17))
    assert output.shape == (2, 16, 16, 17)


def test_streaming_pedal_shapes_gradients_and_budget():
    model = StreamingPedalAMT(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    inputs = torch.randn(2, 32, 41)
    output = model(inputs)
    assert output.shape == (2, 3, 41)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    production = StreamingPedalAMT()
    assert 3_000_000 <= production.parameter_count <= 8_000_000


def test_model_rejects_wrong_mel_shape():
    model = StreamingPedalAMT(mel_bins=32, channels=(8, 12, 16, 24),
                              shared_hidden=32, tower_hidden=16,
                              shared_layers=1)
    try:
        model(torch.randn(2, 31, 20))
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong mel shape accepted")


def test_regression_model_loads_v1_and_emits_five_channels():
    v1 = StreamingPedalAMT(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    v2 = StreamingPedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    copied = v2.load_v1(v1.state_dict())
    assert copied
    assert torch.equal(v2.towers[1].head.weight[0], v1.towers[1].head.weight[0])
    assert v2(torch.randn(2, 32, 41)).shape == (2, 5, 41)


def test_offline_oracle_is_parameter_matched_and_reads_future():
    torch.manual_seed(13)
    streaming = StreamingPedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=2, dropout=0).eval()
    offline = OfflinePedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=2, dropout=0,
        bidirectional_hidden=20).eval()
    inputs = torch.randn(2, 32, 41)
    changed = inputs.clone()
    changed[:, :, 30:] = torch.randn_like(changed[:, :, 30:])
    with torch.no_grad():
        before = offline(inputs)
        after = offline(changed)
    assert before.shape == (2, 5, 41)
    assert not torch.equal(before[:, :, :20], after[:, :, :20])
    relative_difference = (
        abs(offline.parameter_count - streaming.parameter_count) /
        streaming.parameter_count)
    assert relative_difference < 0.15


def test_flatten_oracle_is_budget_matched_and_retains_frequency_grid():
    pooled = OfflinePedalAMTRegression()
    flattened = OfflinePedalAMTRegressionFlatten(
        projection_size=160, bidirectional_hidden=191)
    relative_delta = (
        abs(flattened.parameter_count - pooled.parameter_count) /
        pooled.parameter_count)
    assert relative_delta < 0.001
    assert flattened.frontend_frequency_bins == 15
    assert flattened.frequency_projection.in_features == 160 * 15
    output = flattened(torch.randn(1, 229, 31))
    assert output.shape == (1, 5, 31)


def test_offline_fixed_lag_shared_respects_stacked_future_bound():
    torch.manual_seed(14)
    model = OfflinePedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=2, dropout=0,
        bidirectional_hidden=20).eval()
    first = torch.randn(1, 18, 24)
    second = first.clone()
    second[:, 10:] = torch.randn_like(second[:, 10:])
    with torch.no_grad():
        first_output = model._fixed_lag_shared(first, future_frames=4,
                                               window_batch=3)
        second_output = model._fixed_lag_shared(second, future_frames=4,
                                                window_batch=3)
    # Frame 5 may read through frame 9 but never frame 10.
    assert torch.equal(first_output[:, 5], second_output[:, 5])
    assert not torch.equal(first_output[:, 6], second_output[:, 6])


def test_offline_chunked_whole_context_matches_single_gru_call():
    torch.manual_seed(15)
    model = OfflinePedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=2, dropout=0,
        bidirectional_hidden=20).eval()
    features = torch.randn(2, 31, 24)
    with torch.no_grad():
        expected, _ = model.shared(features)
        actual = model._whole_context_shared(features, chunk_frames=7)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_eval_pedal_tower_chunking_matches_single_gru_call():
    torch.manual_seed(16)
    tower = PedalTower(24, 20, dropout=0, output_size=2).eval()
    features = torch.randn(2, 31, 24)
    with torch.no_grad():
        expected_features, _ = tower.gru(features)
        expected = tower.head(tower.norm(expected_features)).transpose(1, 2)
        actual = tower(features, chunk_frames=7).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_offline_fixed_lag_rejects_less_than_frontend_context():
    model = OfflinePedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0,
        bidirectional_hidden=20)
    try:
        model.forward_fixed_lag(torch.randn(1, 32, 20), 4)
    except ValueError as error:
        assert "frontend" in str(error)
    else:
        raise AssertionError("impossible total future lag was accepted")


def test_offline_feature_split_matches_forward():
    torch.manual_seed(15)
    model = OfflinePedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0,
        bidirectional_hidden=20).eval()
    inputs = torch.randn(1, 32, 20)
    with torch.no_grad():
        expected = model(inputs)
        actual = model.forward_features(model.extract_frontend(inputs))
    assert torch.equal(actual, expected)


def test_causal_temporal_block_does_not_read_future():
    torch.manual_seed(3)
    block = CausalTemporalBlock(8, kernel_size=9, dilation=2, dropout=0).eval()
    first = torch.randn(2, 8, 41)
    second = first.clone()
    second[:, :, 25:] = torch.randn_like(second[:, :, 25:])
    with torch.no_grad():
        first_output = block(first)
        second_output = block(second)
    assert torch.equal(first_output[:, :, :25], second_output[:, :, :25])


def test_dual_timescale_warm_starts_regression_and_emits_five_channels():
    baseline = StreamingPedalAMTRegression(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    model = StreamingPedalAMTDualTimescale(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    initialized = model.load_regression(baseline.state_dict())
    assert initialized
    assert all(name.startswith("dual_timescale.") for name in initialized)
    assert torch.equal(model.shared.weight_ih_l0, baseline.shared.weight_ih_l0)
    assert model(torch.randn(2, 32, 41)).shape == (2, 5, 41)


def test_dynamic_crf_warm_starts_dual_model_and_emits_transition_matrix():
    baseline = StreamingPedalAMTDualTimescale(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    model = StreamingPedalAMTDynamicCRF(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0)
    initialized = model.load_dual_timescale(baseline.state_dict())
    assert initialized
    assert all(name.startswith("transition_head.") for name in initialized)
    output = model(torch.randn(2, 32, 41))
    assert output.shape == (2, 9, 41)
    assert output[:, 5:].transpose(1, 2).reshape(2, 41, 2, 2).shape == (
        2, 41, 2, 2)


def test_d7_flatten_warm_start_is_functionally_exact():
    torch.manual_seed(17)
    baseline = StreamingPedalAMTDynamicCRF(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0).eval()
    model = StreamingPedalAMTDynamicCRFFlatten(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0).eval()
    model.load_dynamic_crf(baseline.state_dict())
    inputs = torch.randn(2, 32, 41)
    with torch.no_grad():
        expected = baseline(inputs)
        actual = model(inputs)
    assert actual.shape == (2, 9, 41)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_transient_branch_is_causal_and_preserves_time():
    torch.manual_seed(19)
    branch = CausalTransientBranch(
        frequency_bins=16, channels=4, output_size=8, dropout=0).eval()
    first = torch.randn(2, 1, 16, 41)
    second = first.clone()
    second[:, :, :, 25:] = torch.randn_like(second[:, :, :, 25:])
    with torch.no_grad():
        first_output = branch(first)
        second_output = branch(second)
    assert first_output.shape == (2, 41, 8)
    assert torch.equal(first_output[:, :25], second_output[:, :25])


def test_d7b_two_stream_warm_start_is_functionally_exact():
    torch.manual_seed(23)
    baseline = StreamingPedalAMTDynamicCRF(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0).eval()
    model = StreamingPedalAMTDynamicCRFTwoStream(
        mel_bins=32, channels=(8, 12, 16, 24), shared_hidden=32,
        tower_hidden=16, shared_layers=1, dropout=0,
        transient_start_bin=16, transient_channels=4,
        transient_hidden=8).eval()
    model.load_dynamic_crf(baseline.state_dict())
    inputs = torch.randn(2, 32, 41)
    with torch.no_grad():
        expected = baseline(inputs)
        actual, auxiliary = model.forward_with_transient(inputs)
    assert actual.shape == (2, 9, 41)
    assert auxiliary.shape == (2, 4, 41)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
