#!/usr/bin/env python
"""Export a mobile pedal checkpoint and verify ONNX Runtime parity."""

import argparse
import hashlib
import json

import numpy as np
import torch

from ov_piano.models.mobile_pedal import (
    EdgePedalAMTRegression, MultirateEdgePedalAMTRegression,
    StreamingPedalAMT, StreamingPedalAMTRegression)


REGRESSION_MODEL_TYPES = {
    "mobile_regression", "mobile_edge_10ms", "mobile_edge_multirate_10ms",
}


def build_model(saved, mel_bins=229):
    conf = saved["config"]
    model_type = saved.get("model_type", "mobile")
    if model_type == "mobile_edge_multirate_10ms":
        return MultirateEdgePedalAMTRegression(
            mel_bins, fast_size=conf["FAST_SIZE"],
            context_hidden=conf["CONTEXT_HIDDEN"],
            context_layers=conf["CONTEXT_LAYERS"],
            context_pool=conf["CONTEXT_POOL"],
            event_delay_frames=conf["EVENT_DELAY_FRAMES"],
            head_hidden=conf["HEAD_HIDDEN"], dropout=conf["DROPOUT"])
    if model_type == "mobile_edge_10ms":
        return EdgePedalAMTRegression(
            mel_bins, frontend_size=conf.get("FRONTEND_SIZE", 512),
            shared_hidden=conf["SHARED_HIDDEN"],
            tower_hidden=conf["TOWER_HIDDEN"],
            shared_layers=conf["SHARED_LAYERS"], dropout=conf["DROPOUT"])
    model_class = (StreamingPedalAMTRegression if
                   model_type == "mobile_regression" else StreamingPedalAMT)
    return model_class(
        mel_bins, shared_hidden=conf["SHARED_HIDDEN"],
        tower_hidden=conf["TOWER_HIDDEN"],
        shared_layers=conf["SHARED_LAYERS"], dropout=conf["DROPOUT"])


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, logmel):
        outputs = self.model(logmel)
        if outputs.shape[1] == 5:
            return (torch.sigmoid(outputs[:, 0]), torch.sigmoid(outputs[:, 1]),
                    torch.tanh(outputs[:, 2]), torch.sigmoid(outputs[:, 3]),
                    torch.tanh(outputs[:, 4]))
        probabilities = torch.sigmoid(outputs)
        return probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=209)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu")
    model_type = saved.get("model_type", "mobile")
    model = build_model(saved)
    model.load_state_dict(saved["model"], strict=True)
    # Set the wrapper itself to eval. torch.onnx.export restores the root
    # module's training state after tracing; leaving the wrapper in train mode
    # would recursively put the already-evaluated child back into train mode
    # before the parity checks (changing BatchNorm behavior).
    wrapper = Wrapper(model).eval()
    torch.manual_seed(0)
    example = torch.randn(1, 229, args.frames)
    output_names = (["pedal_state", "pedal_down_confidence", "pedal_down_offset",
                     "pedal_up_confidence", "pedal_up_offset"] if
                    model_type in REGRESSION_MODEL_TYPES else
                    ["pedal_state", "pedal_down", "pedal_up"])
    torch.onnx.export(
        wrapper, example, args.output, opset_version=17,
        input_names=["logmel"], output_names=output_names,
        dynamic_axes={
            # The live bridge always runs batch 1. Keeping batch fixed avoids
            # ONNX GRU's dynamic-batch hidden-state ambiguity while preserving
            # the arbitrary chunk lengths required by streaming inference.
            "logmel": {2: "time"},
            **{name: {1: "time"} for name in output_names},
        })
    import onnxruntime as ort
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    for frames in (97, args.frames, 311):
        candidate = torch.randn(1, 229, frames)
        expected = [value.detach().numpy() for value in wrapper(candidate)]
        actual = session.run(None, {"logmel": candidate.numpy()})
        for reference, observed in zip(expected, actual):
            np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=2e-4)
    manifest = {
        "checkpoint": args.checkpoint, "checkpoint_sha256": sha256(args.checkpoint),
        "onnx": args.output, "onnx_sha256": sha256(args.output),
        "model_type": model_type,
        "outputs": output_names,
        "dynamic_time_parity_frames": [97, args.frames, 311],
    }
    if model_type == "mobile_edge_multirate_10ms":
        manifest.update({
            "event_delay_frames": saved["config"]["EVENT_DELAY_FRAMES"],
            "context_pool": saved["config"]["CONTEXT_POOL"],
            "future_context_secs": saved["config"]["FUTURE_CONTEXT_SECS"],
        })
    with open(args.output + ".json", "w") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))
