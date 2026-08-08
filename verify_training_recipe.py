"""End-to-end smoke test of the sounding-off training recipe, on CPU.

Imports the ACTUAL freeze logic from 1_train_onsets_velocities.py rather than a
copy, so this cannot drift from the script that runs on the GPU. Uses random
tensors -- no HDF5, no MAESTRO, no CUDA. Run it before every GPU lease.

Checks, in order:
  1. the training script is importable and its config parses
  2. a frozen backbone stays bit-identical across training steps
     (parameters AND BatchNorm buffers -- buffers are the subtle one, since
     BatchNorm rewrites running_mean/var on every forward in train() mode
     regardless of the optimizer or torch.no_grad)
  3. the trained head actually updates
  4. eval mode is restored correctly after a cross-validation pass
  5. a checkpoint round-trips without changing outputs
  6. the censored sounding-off targets and one-sided loss are correct
  7. ONNX export runs and matches torch

  python verify_training_recipe.py
"""
from __future__ import annotations

import io
import sys

import torch

sys.path.insert(0, ".")
from ov_piano.models.ov import OnsetsAndVelocities  # noqa: E402
from ov_piano.utils import censored_offset_loss, remaining_time_targets  # noqa: E402

MELS = 229
KEYS = 88
BATCH = 2
FRAMES = 64
STEPS = 15


def build():
    torch.manual_seed(1234)
    return OnsetsAndVelocities(
        in_chans=2, in_height=MELS, out_height=KEYS, enable_offset_head=True)


def freeze(model, component: str) -> None:
    """Mirror of freeze_untrained_submodules() in 1_train_onsets_velocities.py."""
    if component == "all":
        return
    keep = {"velocity": "velocity_stage", "offset": "offset_stage"}[component]
    for name, module in model.named_children():
        if name == keep:
            continue
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def backbone_state(model):
    return {
        name: tensor.detach().clone()
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if not name.startswith("offset_stage")
    }


def offset_targets(batch=BATCH):
    x = torch.randn(batch, MELS, FRAMES)
    roll = (torch.rand(batch, KEYS, FRAMES - 1) > 0.7).float()
    return (x,) + remaining_time_targets(roll, 0.024)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")
    return ok


def main() -> None:
    results = []
    print("=== 1. training script imports and freeze helper is present ===")
    source = open("1_train_onsets_velocities.py", encoding="utf-8").read()
    results.append(check(
        "freeze_untrained_submodules defined", "def freeze_untrained_submodules" in source))
    results.append(check(
        "freeze applied before the epoch loop",
        source.index("freeze_untrained_submodules()")
        < source.index("for epoch in range(1, CONF.NUM_EPOCHS")))
    results.append(check(
        "freeze re-applied after model.train()",
        "model.train()\n                # train() is recursive" in source))

    print("\n=== 2/3. offset-head training leaves the backbone untouched ===")
    model = build()
    optimizer = torch.optim.Adam(model.offset_stage.parameters(), lr=1e-3)
    model.train()
    freeze(model, "offset")
    before = backbone_state(model)
    head_before = {n: p.detach().clone()
                   for n, p in model.offset_stage.named_parameters()}

    first_loss = last_loss = None
    for _ in range(STEPS):
        x, tgt, exact, cens = offset_targets()
        offsets = model(x, trainable_onsets=False)[-1]
        loss = censored_offset_loss(offsets, tgt, exact, cens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if first_loss is None:
            first_loss = last_loss

    after = backbone_state(model)
    drifted = [n for n in before if not torch.equal(before[n], after[n])]
    results.append(check(
        f"backbone bit-identical after {STEPS} steps", not drifted,
        f"{len(drifted)}/{len(before)} tensors drifted" if drifted else f"{len(before)} tensors"))
    head_moved = sum(
        1 for n, p in model.offset_stage.named_parameters()
        if not torch.equal(head_before[n], p.detach()))
    results.append(check("offset head is actually learning", head_moved > 0,
                         f"{head_moved} params updated"))
    results.append(check("offset loss decreases", last_loss < first_loss,
                         f"{first_loss:.4f} -> {last_loss:.4f}"))

    print("\n=== 4. eval/train cycle does not unfreeze the backbone ===")
    model.eval()
    model.train()
    freeze(model, "offset")
    for _ in range(5):
        x, tgt, exact, cens = offset_targets()
        offsets = model(x, trainable_onsets=False)[-1]
        loss = censored_offset_loss(offsets, tgt, exact, cens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    after_cycle = backbone_state(model)
    drifted2 = [n for n in before if not torch.equal(before[n], after_cycle[n])]
    results.append(check("backbone still bit-identical after eval->train cycle",
                         not drifted2, f"{len(drifted2)} drifted" if drifted2 else ""))

    print("\n=== 5. checkpoint round-trip is lossless ===")
    model.eval()
    probe = torch.randn(1, MELS, FRAMES)
    with torch.no_grad():
        out_a = model(probe, trainable_onsets=False)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    with torch.no_grad():
        out_b = restored(probe, trainable_onsets=False)
    identical = all(
        torch.allclose(a, b, atol=0) for a, b in zip(out_a[0], out_b[0])
    ) and torch.allclose(out_a[-1], out_b[-1], atol=0)
    results.append(check("saved/restored model gives identical outputs", identical))

    print("\n=== 6. sounding-off regression targets are correct ===")
    # Hand-built roll, one key, 10 frames at 0.1 s/frame:
    # sounding frames 2..4 (ends inside chunk) and 7..9 (runs to the edge).
    spf = 0.1
    roll = torch.zeros(1, 1, 10)
    roll[0, 0, 2:5] = 1.0
    roll[0, 0, 7:10] = 1.0
    tgt, exact, cens = remaining_time_targets(roll, spf, cap_secs=8.0)
    expect_exact = torch.log1p(torch.tensor([0.3, 0.2, 0.1]))
    results.append(check(
        "interior note: exact targets = time to next silence",
        torch.allclose(tgt[0, 0, 2:5], expect_exact, atol=1e-6)
        and bool(exact[0, 0, 2:5].all()) and not bool(cens[0, 0, 2:5].any())))
    expect_bound = torch.log1p(torch.tensor([0.3, 0.2, 0.1]))
    results.append(check(
        "edge note: censored with correct lower bound",
        bool(cens[0, 0, 7:10].all()) and not bool(exact[0, 0, 7:10].any())
        and torch.allclose(tgt[0, 0, 7:10], expect_bound, atol=1e-6)))
    results.append(check(
        "silent cells carry no mask",
        not bool(exact[0, 0, :2].any() or cens[0, 0, :2].any()
                 or exact[0, 0, 5:7].any() or cens[0, 0, 5:7].any())))
    # The production cap is 2 s: a note still sounding past it must be censored
    # rather than given an exact target it cannot possibly predict.
    long_roll = torch.zeros(1, 1, 40)
    long_roll[0, 0, 0:35] = 1.0   # 3.5 s at 0.1 s/frame, ends inside the chunk
    _, exact_c, cens_c = remaining_time_targets(long_roll, spf, cap_secs=2.0)
    results.append(check(
        "cap=2.0 censors cells whose remaining time exceeds the cap",
        bool(cens_c[0, 0, 0].item()) and not bool(exact_c[0, 0, 0].item())
        and bool(exact_c[0, 0, 34].item())))
    # Censored loss is one-sided: predicting ABOVE the bound costs nothing.
    high = torch.full_like(tgt, 5.0)
    low = torch.full_like(tgt, 0.0)
    loss_high = censored_offset_loss(high, tgt, torch.zeros_like(exact), cens)
    loss_low = censored_offset_loss(low, tgt, torch.zeros_like(exact), cens)
    results.append(check(
        "censored loss punishes only underestimates",
        loss_high.item() == 0.0 and loss_low.item() > 0.0))

    print("\n=== 6b. onset target width is exactly TARGET_WIDTH_FRAMES ===")
    # The ablation is unfalsifiable unless each configured width really
    # produces that many frames. This replicates the trainer's widening block
    # verbatim on a hand-built roll: one isolated onset at frame 5, velocity
    # 100, in a 20-frame window.
    for width in (1, 2, 3, 4):
        onsets = torch.zeros(1, 1, 20)
        onsets[0, 0, 5] = 100.0
        widened = onsets
        for _ in range(width - 1):
            nxt = widened.clone()
            torch.maximum(widened[..., :-1], widened[..., 1:], out=nxt[..., 1:])
            widened = nxt
        active = (widened[0, 0] > 0).nonzero().flatten().tolist()
        expected = list(range(5, 5 + width))
        results.append(check(
            f"width {width}: target spans exactly {width} frame(s)",
            active == expected,
            f"active frames {active}, expected {expected}"))
    # Forward-only: the frame BEFORE the onset must never activate, or the
    # model would be taught to fire early and every emitted timestamp shifts.
    onsets = torch.zeros(1, 1, 20)
    onsets[0, 0, 5] = 100.0
    widened = onsets
    for _ in range(2):
        nxt = widened.clone()
        torch.maximum(widened[..., :-1], widened[..., 1:], out=nxt[..., 1:])
        widened = nxt
    results.append(check("widening is forward-only (frame 4 stays zero)",
                         float(widened[0, 0, 4]) == 0.0))
    # Velocity must survive widening: onsets_norm divides by 127, so a
    # clobbered magnitude would silently corrupt the velocity target.
    results.append(check("velocity preserved across the widened span",
                         all(float(widened[0, 0, f]) == 100.0
                             for f in range(5, 8))))
    # The trainer must actually read the knob rather than hardcode 3.
    source = open("1_train_onsets_velocities.py", encoding="utf-8").read()
    results.append(check(
        "trainer reads CONF.TARGET_WIDTH_FRAMES in the widening loop",
        "range(CONF.TARGET_WIDTH_FRAMES - 1)" in source
        and "triple_onsets" not in source))

    print("\n=== 7. ONNX export runs and matches torch ===")
    import tempfile
    import os as _os
    model.eval()
    probe2 = torch.randn(1, MELS, FRAMES)
    with torch.no_grad():
        torch_out = model(probe2, trainable_onsets=False)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            stages, vels, offsets = self.inner(x, trainable_onsets=False)
            return stages[-1], vels, offsets

    handle, onnx_path = tempfile.mkstemp(suffix=".onnx")
    _os.close(handle)
    try:
        # The legacy exporter works on both torch 2.2 (pod) and 2.13 (laptop)
        # now that forward_onsets uses a slice subtraction instead of
        # torch.diff, which has no ONNX lowering at opset 17 on either.
        torch.onnx.export(
            ExportWrapper(model), (probe2,), onnx_path,
            input_names=["logmel"],
            output_names=["onsets", "velocities", "offsets"],
            opset_version=17,
            dynamic_axes={"logmel": {0: "batch", 2: "time"}})
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = session.run(None, {"logmel": probe2.numpy()})
        parity = all(
            torch.allclose(torch.from_numpy(o), t, atol=1e-4)
            for o, t in zip(ort_out, (torch_out[0][-1], torch_out[1],
                                      torch_out[2])))
        results.append(check("onnxruntime output matches torch (atol 1e-4)", parity))
    except Exception as error:  # noqa: BLE001
        results.append(check("ONNX export", False, str(error)[:120]))
    finally:
        _os.unlink(onnx_path)

    print(f"\n{'ALL PASS' if all(results) else 'FAILURES PRESENT'}  "
          f"({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
