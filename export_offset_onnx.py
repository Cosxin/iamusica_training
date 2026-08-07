#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Export an O&V checkpoint to ONNX, adapting to whichever heads it carries.

The head configuration is DETECTED FROM THE CHECKPOINT rather than hardcoded.
Detection makes the strict load correct by construction: the model is built
with exactly the submodules the state dict contains, so a mismatch is a real
error rather than a configuration slip. Legacy frame-head checkpoints are
rejected with a pointer to the commit that can still export them; see
NOTES_frame_head.md.

Alignment: every head's raw map is T-1 wide (the model differences the mel
along time), so each is left-padded by one frame to line up with the input
grid. Probability heads are padded with 0 after sigmoid; the offset head is a
REGRESSION output (log1p seconds remaining) and is padded with log1p(0) == 0,
which reads as "ends now" -- and must NOT be passed through a sigmoid.

  python export_offset_onnx.py --checkpoint X.torch --output Y.onnx
"""
import argparse
import hashlib
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ov_piano.models.ov import OnsetsAndVelocities  # noqa: E402
from ov_piano.utils import load_model  # noqa: E402

MELS = 229
KEYS = 88


def detect_offset_head(checkpoint_path):
    """Does this checkpoint carry the sounding-off regression head?

    Also rejects legacy frame-head checkpoints outright rather than silently
    dropping their weights: the frame head no longer exists in this model (see
    NOTES_frame_head.md), so such a checkpoint cannot be exported from here.
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    # Checkpoints are saved either as a bare state dict or wrapped.
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state \
                and isinstance(state[key], dict):
            state = state[key]
            break
    keys = list(state.keys())
    if any(k.startswith("frame_stage.") for k in keys):
        raise SystemExit(
            "This checkpoint has a frame head, which was removed (it cost "
            "-1.15 onset F1; see NOTES_frame_head.md). To export it, check "
            "out commit aee486d.")
    return any(k.startswith("offset_stage.") for k in keys)


class LiveInferenceWrapper(torch.nn.Module):
    """Raw logits -> aligned probability heads + optional offset regression."""

    def __init__(self, model, has_offset):
        super().__init__()
        self.model = model
        self.has_offset = has_offset

    def forward(self, logmel):
        outputs = self.model(logmel, trainable_onsets=False)
        onset_stages, velocity = outputs[0], outputs[1]
        result = [
            F.pad(torch.sigmoid(onset_stages[-1]), (1, 0)),
            F.pad(torch.sigmoid(velocity), (1, 0)),
        ]
        if self.has_offset:
            result.append(F.pad(outputs[-1], (1, 0)))
        return tuple(result)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--example-frames", type=int, default=401)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    has_offset = detect_offset_head(args.checkpoint)
    print(f"[export] sounding-off head: {has_offset}", flush=True)

    model = OnsetsAndVelocities(
        in_chans=2, in_height=MELS, out_height=KEYS,
        enable_offset_head=has_offset)
    load_model(model, args.checkpoint, eval_phase=True)
    model.eval()
    wrapper = LiveInferenceWrapper(model, has_offset).eval()

    names = ["onset", "velocity"] + (["offset"] if has_offset else [])

    example = torch.randn(1, MELS, args.example_frames)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with torch.no_grad():
        reference = wrapper(example)
    torch.onnx.export(
        wrapper, (example,), args.output,
        input_names=["logmel"],
        output_names=names,
        dynamic_axes={n: {0: "batch", 2: "time"}
                      for n in ["logmel"] + names},
        opset_version=args.opset)

    import onnxruntime as ort

    session = ort.InferenceSession(args.output,
                                   providers=["CPUExecutionProvider"])
    got = session.run(None, {"logmel": example.numpy()})
    worst = max(float(torch.max(torch.abs(torch.from_numpy(a) - b)))
                for a, b in zip(got, reference))
    print(json.dumps({
        "checkpoint": os.path.abspath(args.checkpoint),
        "output": os.path.abspath(args.output),
        "sha256": sha256(args.output),
        "bytes": os.path.getsize(args.output),
        "offset_head": has_offset,
        "outputs": [o.name for o in session.get_outputs()],
        "max_abs_diff_vs_torch": worst,
        "parity_ok": worst <= args.atol,
    }, indent=1))
    if worst > args.atol:
        raise SystemExit(f"ONNX parity failed: {worst} > {args.atol}")


if __name__ == "__main__":
    main()
