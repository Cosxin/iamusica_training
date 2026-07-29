#!/usr/bin/env python
"""Export the frame-enabled model with a stable live-inference contract."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import load_model


class LiveInferenceWrapper(torch.nn.Module):
    """Map raw model logits to aligned onset/velocity/frame probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, logmel):
        onset_stages, velocity_logits, frame_logits = self.model(logmel)
        onset = F.pad(torch.sigmoid(onset_stages[-1]), (1, 0))
        velocity = F.pad(torch.sigmoid(velocity_logits), (1, 0))
        frame = F.pad(torch.sigmoid(frame_logits), (1, 0))
        return onset, velocity, frame


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--example-frames", type=int, default=401)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    model = OnsetsAndVelocities(
        in_chans=2, in_height=229, out_height=88,
        conv1x1head=(200, 200), bn_momentum=0,
        leaky_relu_slope=0.1, dropout_drop_p=0,
        enable_frame_head=True)
    load_model(model, args.checkpoint, eval_phase=True, to_cpu=True,
               strict=True)
    wrapper = LiveInferenceWrapper(model).eval()
    torch.manual_seed(0)
    example = torch.randn(1, 229, args.example_frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        wrapper, example, args.output,
        input_names=["logmel"],
        output_names=["onset", "velocity", "frame"],
        dynamic_axes={
            "logmel": {0: "batch", 2: "time"},
            "onset": {0: "batch", 2: "time"},
            "velocity": {0: "batch", 2: "time"},
            "frame": {0: "batch", 2: "time"},
        },
        opset_version=args.opset)

    import onnxruntime as ort

    expected = [value.detach().numpy() for value in wrapper(example)]
    session = ort.InferenceSession(
        args.output, providers=["CPUExecutionProvider"])
    output_names = [item.name for item in session.get_outputs()]
    if output_names != ["onset", "velocity", "frame"]:
        raise RuntimeError(f"Unexpected ONNX outputs: {output_names}")
    actual = session.run(None, {"logmel": example.numpy()})
    for name, reference, candidate in zip(
            output_names, expected, actual):
        if reference.shape != candidate.shape:
            raise RuntimeError(
                f"{name}: shape mismatch {reference.shape} != "
                f"{candidate.shape}")
        np.testing.assert_allclose(
            candidate, reference, rtol=args.rtol, atol=args.atol)
        if candidate.shape[-1] != args.example_frames:
            raise RuntimeError(f"{name}: time alignment was not preserved")

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "checkpoint": {
            "path": os.path.abspath(args.checkpoint),
            "sha256": sha256(args.checkpoint),
        },
        "onnx": {
            "path": os.path.abspath(args.output),
            "sha256": sha256(args.output),
            "opset": args.opset,
        },
        "contract": {
            "input": {"name": "logmel", "shape": ["batch", 229, "time"]},
            "outputs": [
                {"name": name, "shape": ["batch", 88, "time"],
                 "domain": "probability"}
                for name in output_names
            ],
            "time_alignment": "left-padded by one frame after T-1 model output",
        },
    }
    manifest_path = args.output + ".json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
