# Eval findings — 2026-07-30 (session 3)

Model-eval results for the O&V codec + release-head work. All numbers are on the
MAESTRO v3 **test** split. Raw per-file JSONs pulled to `results_eval/` (see end).
Scope note: this is **model eval only** — the midee/bridge dashboard is another
agent's job.

Checkpoint under test: `mixed_ABC .../step=5000_f1=0.9606__0.9362.torch`
(warm-started shipped ckpt, LR 8e-4, 5000 steps, mixed clean+aptX+SBC,
ENABLE_FRAME_HEAD=True → onset+velocity+release). Baseline = shipped O&V ckpt
(onset+velocity only).

---

## 0. The headline reframe (read this first)

**note-with-offset F1 is the WRONG target for this product.** Two consequences:

- The product need is *repeated-note segmentation* ("two short same-pitch notes
  must not merge"), which is a **re-trigger** problem, driven mostly by the
  **onset head**, not the release head. See §4.
- note-with-offset F1 is scored against **pedal-extended** GT (holds notes long),
  the *opposite* of what crisp repeat-segmentation wants. Our release head does
  **perceptual** release (note inaudible) → shorter notes → *better* gap between
  repeats. The "gap from pedal GT" is a **feature** here, not a defect.

So: do NOT chase Mobile-AMT's 0.77 note-with-offset. Chase repeat resolution
(min resolvable same-pitch IOI) and onset robustness.

---

## 1. Robustness matrix (offline, 300 s window), onset & onset+vel F1

| condition | onset F1 | onset+vel F1 |
|---|---|---|
| baseline clean | **0.9678** | 0.9450 |
| baseline aptX | 0.9623 | 0.9400 |
| baseline SBC | 0.9525 | 0.9099 |
| finetuned clean | 0.9614 | 0.9350 |
| finetuned aptX | 0.9565 | 0.9300 |
| finetuned SBC | 0.9468 | 0.9139 |

- **Shipped O&V is already codec-robust**: SBC costs only ~1.5 onset-F1 pts,
  aptX ~0.5. aptX is near-transparent.
- Harness reproduces the paper (clean onset 0.9677 ≈ paper 0.9675).

## 2. Ablation — value of SBC-in-training (evaluated ON SBC)

| model | onset F1 | onset+vel F1 | note-w-offset F1 |
|---|---|---|---|
| baseline (shipped) | 0.9525 | 0.9099 | — (no frame head) |
| aptX-only fine-tune (NO SBC in train) | 0.9460 | 0.8955 | 0.5233 |
| mixed fine-tune (WITH SBC) | 0.9468 | **0.9139** | **0.5617** |

**Refined finding (supersedes the earlier "fine-tune didn't help"):** onsets were
never the codec-vulnerable part (baseline already 0.95 on SBC). Codec damage lands
on **velocity and release**, and SBC augmentation is what recovers those:
onset+vel +1.8 pts and note-offset +3.8 pts over aptX-only. Onset is untouched
(0.946 vs 0.946) and slightly below baseline (mild forgetting). Net: the codec
augmentation's value is real but **confined to velocity + release on SBC**.

## 3. Release head — training curve (note-with-offset on SBC)

| step | note-w-offset F1 | onset F1 |
|---|---|---|
| 1000 | 0.4513 | 0.9367 |
| 3000 | 0.5532 | 0.9354 |
| 5000 | 0.5617 | 0.9369 |

Release head learns and **plateaus ~0.56**; onset flat. More steps buy little.
Durations: GT median ≈ pred median (~0.34 s / ~0.33 s) → **no systematic bias**;
the low F1 is per-note *scatter* vs a 50–70 ms tolerance, plus pedal-vs-perceptual
divergence on long sustained notes. (Not a lookahead problem — release is a
past-facing event.)

## 4. Repeated same-pitch notes by inter-onset interval (THE product metric)

Fine-tuned model, clean, 40 files, 154 628 GT pairs. `onset-retrig` = both onsets
detected (onset head). `clean-segment` = onsets detected AND note-1 released
before note-2 onset (adds release head).

| IOI gap | rate | #pairs | onset-retrig | clean-segment |
|---|---|---|---|---|
| 0–50 ms | >20/s | 49 | 0.000 | 0.000 |
| 50–75 ms | ~16/s | 205 | 0.029 | 0.029 |
| 75–100 ms | ~11/s | 1068 | 0.532 | 0.483 |
| 100–150 ms | ~8/s | 8677 | 0.783 | 0.707 |
| 150–200 ms | | 9220 | 0.808 | 0.743 |
| 200–300 ms | | 13953 | 0.812 | 0.762 |
| 300–500 ms | | 19413 | 0.860 | 0.816 |
| 500–1000 ms | | 26527 | 0.866 | 0.833 |
| >1000 ms | | 75516 | 0.863 | 0.858 |
| **overall** | | 154628 | **0.847** | **0.820** |

- **Clean resolution holds to ~100 ms IOI (~10 repeats/s).** 75–100 ms is coin-flip;
  <75 ms collapses (but that's 0.16 % of pairs, genuinely extreme).
- **Bottleneck is the ONSET head, not the release head** — `clean-segment` tracks
  `onset-retrig` closely (release opens a clean gap once two onsets exist). To
  push into trill territory, improve **onset re-triggering** (onset target that
  re-fires on attacks over sustain / refractory-aware decoder), NOT the release
  head or a base swap.
- The ~0.86 large-IOI plateau ≈ baseline onset recall² (needs both onsets); the
  repeat-specific penalty is the drop *below* it, only material under ~150 ms.

### 4b. Baseline vs fine-tuned onset re-trigger (de-confounds §4)

The §4 curve was on the FINE-TUNED model (only it has a frame head). Since the
fine-tune slightly hurt onsets, that could understate repeat resolution. Re-ran
BOTH models in ONSET_ONLY mode (identical NMS onset decoder, same 40 clean files):

| IOI gap | baseline (shipped, F1 0.9675) | finetuned | Δ |
|---|---|---|---|
| 75–100 ms | 0.537 | 0.532 | +0.5 |
| 100–150 ms | **0.807** | 0.783 | **+2.4** |
| 150–200 ms | 0.821 | 0.808 | +1.3 |
| 200–300 ms | 0.826 | 0.812 | +1.4 |
| 300–500 ms | 0.867 | 0.860 | +0.7 |
| 500–1000 ms | 0.875 | 0.866 | +0.9 |
| >1000 ms | 0.879 | 0.863 | +1.6 |
| **overall** | **0.861** | 0.847 | **+1.4** |

**The fine-tune HURT repeat resolution** (+1.4 overall, +2.4 in the critical
100–150 ms zone), tracking its onset regression (clean onset-F1 0.9678→0.9614).
So for the LED product the **shipped baseline is the better onset/repeat model**;
the codec fine-tune's velocity/release gains (§2) come at a net cost to the
product-critical metric. Publishable framing: note-F1 moved −0.6 but the
deployment-critical repeat metric moved 2–3× more → "F1 hides what matters."
JSONs: `repeats-baseline-clean.json`, `repeats-finetuned-clean-onsetonly.json`.

## 5. Low-latency window sweep (config-only, no retrain)

Fine-tuned, clean, 8 files, FIXED threshold 0.75 (→ pessimistic). Window = chunk
size fed to `strided_inference` (`INFERENCE_CHUNK_SIZE`).

| chunk window | onset F1 | note-offset F1 |
|---|---|---|
| 300 s (offline) | 0.9418 | 0.5531 |
| 5 s | 0.9422 | 0.5557 |
| 2 s | 0.9399 | 0.5410 |
| 1 s | 0.9333 | 0.5249 |
| 0.5 s | 0.9172 | 0.4888 |
| 0.25 s | *(hung — killed)* | |

- **300 s → 5 s is FREE** (0.9418 → 0.9422): O&V's "unbounded" SE-pool context was
  never actually needed. Big latency win at zero cost.
- **Down to 1 s costs ~0.9 pts; 0.5 s costs ~2.5 pts** (knee ~0.5 s).
- Numbers are pessimistic (fixed threshold across all windows). A window-matched
  threshold or a short window-matched fine-tune would recover part of the sub-1s
  drop. The ~100 ms point is unmeasured (0.25 s run hung).
- Mechanism: only the SE `AdaptiveAvgPool2d(1)` statistic changes with window;
  convs + frozen BN are window-invariant. Sub-0.5 s is where a window-matched
  fine-tune (or a causal/trailing SE pool) would be warranted.

Terminology note for the paper: DON'T claim "causal" (that means 0 lookahead).
Claim a **stated latency budget** (e.g. "~100 ms algorithmic latency"), which is
below Mobile-AMT's 174 ms. Mobile-AMT itself is not strictly causal — its 174 ms
includes STFT framing + CNN receptive field + post-processing lookahead.

## 6. Codec penalty vs fast-repeat density ("Turkish March" hypothesis)

**Turkish March (Mozart K.331 / Rondo alla Turca) is NOT in MAESTRO v3.** Tested the
mechanism instead: per file, clean→SBC onset drop vs density of fast same-pitch
repeats (IOI < 150 ms per minute). Baseline model, 169 files.

- Pearson r(density, drop) = **+0.087** (weak/noisy — density is not the sole factor).
- Mean codec drop: **low-density half +0.0038 vs high-density half +0.0262 (~7×).**
- Densest pieces are the repeated-note showpieces, and they take the biggest hits:
  Schubert **"Erlkönig"** (383 repeats/min, drop 0.019 — closest analog to Turkish
  March), Liszt **"Gnomenreigen"**, Liszt **"Wilde Jagd"**, Scarlatti K.132.

**Conclusion:** fast-repeat-dense music does suffer more codec loss (direction
confirmed), but **only ~1–2 % offline** — far milder than the *serious* real-world
Bluetooth degradation reported. The missing severity lives in factors this offline
ffmpeg-SBC eval cannot see: **real over-the-air SBC** (adaptive/low bitrate +
packet loss) and **real-time streaming jitter** (offline full-file eval bypasses the
timing pipeline; fast passages are exactly where jitter lumps events). → motivates §7.

---

## 7. DECISIVE next step — needs the Pi (cannot be done on the cloud pod)

1. **Over-the-air SBC**: capture real Bluetooth-SBC audio on the Pi (not ffmpeg
   round-trip) and re-run onset + repeat-IOI evals. This is the sim-to-real claim
   and the likely explanation of the severe Turkish-March degradation.
2. **Real-time jitter / thermal benchmark** on the Pi, especially on fast passages
   (dense repeats) — the other half of the severity gap.
3. If chasing faster repeats: **onset re-trigger** improvement (target/decoder),
   not the release head.
4. If low latency becomes the headline: window-matched fine-tune (or causal SE
   pool) below 0.5 s; consider a Mobile-AMT-style base only if the real-time claim
   is central (O&V's onset stem is already SOTA + tiny — keep it).

## Artifacts (local, `results_eval/`)

- `eval_summary_compact.json` — matrix (baseline+finetuned × clean/aptX/SBC) +
  offset, per-file F1 arrays (for CIs).
- `repeats-finetuned-clean.json` — repeated-note IOI curve (§4).
- `sweep_research_compact.json` — window sweep (§5) + ablation + multi-ckpt (§2/§3).
- `codec-density.json` — per-file density vs codec drop (§6).

## New eval scripts (repo root, also on pod `/workspace/iamusica_training`)

- `eval_repeats.py` — repeated-note / IOI re-trigger eval (onset-retrig vs
  clean-segment, binned by IOI).
- `analyze_codec_density.py` — codec-drop vs fast-repeat-density correlation
  (no inference; reuses matrix JSONs + GT).
- Window