# The frame head: removed, and not worth retesting

The binary per-frame "is this key sounding" head (`frame_stage`,
`ENABLE_FRAME_HEAD`) has been removed from this repository. This note records
why, so nobody re-adds it hoping for a different answer.

## What it cost

Measured 2026-08-07 on the deterministic cross-validation split
(`validation[::5]`, seed-independent), warm-starting every run from the
author's published checkpoint. One variable changed at a time:

| run | data | frame head | offset head | onset F1 | vs author |
|---|---|---|---|---|---|
| author's checkpoint, no training | — | — | — | **0.96754** | — |
| control | clean | off | off | 0.96748 | −0.0001 |
| production | augmented | off | **on** | 0.96637 | −0.0012 |
| — | clean | **on** | on | 0.95598 | **−0.0116** |
| v3 | augmented | **on** | on | 0.9578 | −0.0097 |

Turning the frame head on costs **1.15 points of onset F1**, immediately — the
damage is present by step 1000 and does not recover with more training. Turning
it off recovers 90% of that, leaving −0.12, which is within run-to-run noise.

The mechanism: a randomly-initialised head with λ=1.0 and `pos_weight=8` pushes
large early gradients into the **shared trunk**, dragging the backbone off an
optimum that took the author 43,500 steps to reach.

It also broke calibration. The frame-head runs peaked at decode threshold
**0.70**, the bottom edge of `XV_THRESHOLDS`, meaning their true optimum may
have been outside the measurable range. Without the frame head the peak returns
to **0.725** — exactly the author's threshold.

## Why it was redundant anyway

Both heads were built from the *same* tensor: the sustain-extended roll
(`extendsus=True`), which already encodes `max(key_up, pedal_release)`. So the
frame mask ("is it sounding now") and the offset regression ("seconds until it
stops sounding") describe the **same event** in two encodings. Their falling
edge and countdown-to-zero mark the same instant.

And nothing consumed it at inference: the bridge's deadline decoder reads only
the offset output. The frame map was pure training cost with no consumer.

## What removing it gives up

The per-frame sounding **mask**. This matters more than it looks:

`remaining_time_targets()` computes its loss only on sounding cells —
`exact_mask` and `censored_mask` are both `& binary`, and their union is exactly
the sounding cells. Where a key is silent the offset head is **completely
unconstrained**, so you cannot recover an envelope by thresholding the offset
map. It was never trained to say "not sounding".

Consequence for the bridge: it must be **onset-driven**, not mask-driven. On a
detected onset, read `offset[pitch, onset_frame]` to schedule an off; while the
note is believed sounding, re-read on later frames and take latest-wins. That
keeps every read in-distribution, since those are exactly the cells the head was
trained on.

## Related: why note-with-offset F1 is not the metric

`eval_offset.py` (deleted with the frame head) scored mir_eval
note-with-offset F1. That was already known to be the wrong target — the
product metric is **repeated-note IOI**, because the bottleneck a listener
hears is onset re-trigger, not release accuracy. `eval_repeats.py` measures
that and is onset-only by construction.

## Also changed: OFFSET_CAP_SECS 8.0 → 2.0

Notes still sounding past the cap are marked censored rather than given an exact
target. At 8s the head was asked to name an exact duration for notes lasting
many seconds, which is not inferable: at the onset frame the model sees only
**0.5s** of future audio. It learned to emit a near-constant prior instead —
0.87–0.96s regardless of the truth, Pearson r = 0.318, and 3051ms error on
notes over 2.5s versus 236–248ms under 1.2s. That is an information limit, not
a capacity limit; a bigger backbone cannot invent the answer.

## Recovering it

Nothing is lost — it is in git history. Commit `aee486d` ("Key-up release head
for LED") carries the frame head, its training recipe and the shipped
`keyup_framehead` checkpoint. The LED model deployed to the Pi before
2026-08-07 uses it: `frame_stage` trained on a key-up (non-sustain-extended)
roll, ~0.605 key-up F1. If that model ever needs rebuilding, check out
`aee486d` rather than re-adding the head here.
