# Pedal transcription — final evaluation record

Date: 2026-08-03. Pod terminated after this record was written.

**Outcome: no pedal model passed the deployment gates. Nothing was deployed to
the Raspberry Pi. The test split was never touched.** Pedal checkpoints were
deliberately NOT preserved — every candidate either failed its gate or was a
refuted research control, and all are reproducible from the configs below.

This document is the durable output of the pedal work. The LED product does not
depend on any of it (see §9).

---

## 1. Gates and verdict

| Gate (full validation) | Target | Best achieved | Pass? |
|---|---:|---:|:--:|
| Pedal state F1 | >= 0.88 | 0.95273 | yes |
| Pedal down F1 @50 ms | >= 0.75 | 0.75105 (offline oracle, NOT deployable) | n/a |
| Strict down+up F1 @50 ms | >= 0.70 | 0.69739 (offline oracle, NOT deployable) | no |
| **Best deployable (causal, 494 ms)** | — | **down 0.66461 / strict 0.56861** | **no** |

The only configuration that reached the down gate was the **offline
bidirectional oracle**, which sees the entire recording and is by construction
not deployable. It still missed the strict-interval gate by 0.0026. The best
causal, Pi-deployable model fell 0.085 short on down.

Production decision recorded at the time:

```json
{"best_research_candidate": "edge_10ms_128x288", "deployment_selected": null,
 "deploy": false, "reason": "no candidate passed all full-validation gates",
 "test_policy": "do not touch test split"}
```

---

## 2. Headline results

All numbers MAESTRO **v3 validation** (never test). `locked32` = deterministic
32-file screening subset; `full` = all 137 validation recordings. Format:
`state / down / kong_onoff / up50 / strict50`.

`kong_onoff` = `pedal_event_onset_offset` (50 ms onset, offset_ratio 0.2) — the
**only** offset-inclusive metric comparable to published work.

### Offline oracles (research controls, not deployable)

| Model | Scope | thr | Metrics |
|---|---|---:|---|
| **Pooled 24 ms (best overall)** | locked32 | 0.3 | 0.96763 / 0.78126 / 0.72649 / 0.90390 / 0.70973 |
| Pooled 24 ms | locked32 | 0.8 | 0.96763 / **0.80310** / 0.76252 / 0.90529 / **0.74902** |
| **Pooled 24 ms** | **full** | 0.3 | 0.95273 / 0.72360 / 0.67337 / 0.87046 / 0.65766 |
| **Pooled 24 ms** | **full** | **0.8** | **0.95273 / 0.75105 / 0.70980 / 0.86773 / 0.69739** |
| Flatten 24 ms, 12k | locked32 | 0.3 | 0.96296 / 0.74437 / 0.68430 / 0.88486 / 0.66802 |
| Flatten 24 ms, 12k | full | 0.75 | 0.94597 / 0.72641 / 0.67879 / 0.85507 / 0.66542 |
| Flatten 24 ms, 60k-equiv | locked32 | 0.3 | 0.96762 / 0.77543 / 0.71774 / 0.89215 / 0.70321 |
| Flatten 24 ms, 60k-equiv | locked32 | 0.85 | 0.96762 / 0.80178 / 0.75702 / 0.90515 / 0.74695 |
| **Pooled 10 ms, 12k** | locked32 | 0.3 | 0.96195 / 0.71333 / 0.63895 / 0.85980 / 0.61799 |

> NOTE: the flatten-extended run has **no full-validation pass** in the results
> tree — only locked32 milestones and calibration. Its locked32 numbers are
> therefore not directly comparable to the pooled oracle's full-validation
> numbers, and locked32 runs optimistic by roughly 0.03-0.05 on down.

### Causal streaming models (Pi-deployable)

| Model | Params | Scope | thr | Metrics |
|---|---:|---|---:|---|
| Edge 10 ms 128/288 | 1,048,941 | full | 0.75 | 0.90793 / 0.66461 / 0.60606 / 0.77510 / 0.56861 |
| Edge 10 ms 64/320 | 1,130,605 | full | 0.8 | 0.90843 / 0.65911 / 0.60370 / 0.76345 / 0.57663 |
| Edge 10 ms 128/288 | | locked32 | 0.75 | 0.93734 / 0.72196 / 0.66630 / 0.82430 / 0.62663 |

Allocation arm (128/288) is 7.2% smaller and marginally better on event F1
(+0.006 down, +0.012 up50) but worse on strict intervals (-0.008). A wash.

---

## 3. Falsification record — six hypotheses, all null or negative

This is the main scientific content. Every plausible cause of the gap to Kong
was tested with a controlled experiment; none explained it.

| Hypothesis | Experiment | Result |
|---|---|---|
| Streaming context is the bottleneck | fixed-lag sweep, unlimited vs 500 ms | **0.044 only** |
| Parameter allocation is wrong | 64/320 vs 128/288, matched total params | **wash** (+0.006/-0.008) |
| Frequency mean-pooling destroys register info | flatten vs pooled, matched params AND exposure | **worse** (-0.025 down) |
| Undertrained | 12k -> 60k-equiv (~10 -> ~50 raw passes) | **+0.03, saturates ~30k** |
| Decoder is limiting | threshold / CRF / semi-Markov / state-transition | **all within 0.03** |
| Time resolution too coarse | 24 ms vs 10 ms hop, matched arch + steps | **worse** (-0.031 down) |

### Why frequency pooling winning is not a fluke

The sustain pedal lifts **every damper simultaneously** — its acoustic signature
is a broadband rise in sympathetic resonance, not a register-specific event. So
frequency-translation invariance is the physically correct inductive bias, and
mean-pooling encodes it for free while a flatten projection must learn it from
data. Pooling is also the cheaper primitive (a free 15x dimensionality
reduction). **For pedal, pooling is both faster and more accurate.**

### The decisive diagnostic: frame parity, event deficit

| Metric | Ours (full val) | Kong (test) | Delta |
|---|---:|---:|---:|
| Frame / state F1 | **0.95273** | 0.9425 | **+0.010** |
| Event onset (down) F1 | 0.75105 | 0.9186 | **-0.168** |
| Event onset+offset F1 | 0.70980 | 0.8658 | **-0.156** |

We **match Kong on frame occupancy and lose heavily on events.** The model knows
when the pedal is down as well as theirs does; it cannot localise the transition
to within 50 ms. The deficit is transition localisation, not perception. Both
comparable metrics degrade by a similar amount, which argues against a
timing/pairing-specific cause and for a general evidence deficit at boundaries.

---

## 4. Latency-accuracy frontier (the publishable figure)

Pooled 24 ms oracle, one checkpoint, evaluated with truncated future context.
locked32, threshold 0.3.

| Total future context | state | down | up50 | strict50 |
|---:|---:|---:|---:|---:|
| 250 ms | 0.95143 | 0.64085 | 0.72682 | 0.46865 |
| 500 ms | 0.95910 | 0.73765 | 0.85036 | 0.62520 |
| 1 s | 0.96361 | 0.76602 | 0.87818 | 0.67556 |
| 3 s | 0.96663 | 0.78060 | 0.89920 | 0.70534 |
| unlimited | 0.96763 | 0.78126 | 0.90390 | 0.70973 |

**Pedal-down needs ~500 ms - 1 s of future context and saturates by 1 s**, where
Mobile-AMT does *notes* fine at 174 ms. That contrast is the most interesting
result the pedal work produced. Caveat: trained full-context, evaluated
truncated, so these are mild lower bounds for a matched fixed-lag model.

---

## 5. Decoder ablations (all CPU, from cached predictions)

Dynamic-CRF checkpoint, step 16000, locked32.

| Decoder | down | up50 | strict50 |
|---|---:|---:|---:|
| regression, NMS r3 | 0.66681 | 0.83417 | 0.56431 |
| regression, NMS r1 | 0.65531 | 0.82798 | 0.55167 |
| semi-Markov, best (tw0.5/sw0.1) | 0.70793 | 0.89364 | 0.63418 |
| semi-Markov, NMS r3 | 0.70835 | 0.89041 | 0.63510 |
| semi-Markov, NMS r1 | 0.70633 | 0.89062 | 0.63331 |

State-transition decoder vs event heads (pooled oracle, **full validation**):

| Decoder | down | up50 | strict50 | predicted intervals |
|---|---:|---:|---:|---:|
| regression (event heads) @0.8 | **0.75105** | 0.86773 | **0.69739** | 45,018 |
| state @0.6 | 0.74311 | 0.88313 | 0.68532 | 50,638 |
| state @0.5 | 0.73574 | **0.88575** | 0.67795 | 50,704 |

Ground truth = 51,519 intervals. **The event heads under-detect by ~12.6%**
(45,018 vs 51,519) while the state channel implies 98% coverage. Decomposed:
event heads P 0.805 / R 0.704; state@0.6 P 0.750 / **R 0.737**. The state channel
recovers +3.3 points of recall the event heads never propose, but pays for it in
precision and timing (it has no sub-frame refinement — pure frame quantisation).

Kong reaches 0.9186 with a **plain threshold decoder**, no CRF, no Viterbi. That
is the strongest external confirmation that decoding is not the lever.

---

## 6. Kong comparison — protocol caveats (must accompany any citation)

Kong et al. 2021, "High-resolution Piano Transcription with Pedals by Regressing
Onset and Offset Times" (TASLP). Verified from paper + `bytedance/piano_transcription`.

**Published pedal metrics (Table VI, MAESTRO test):** frame P 94.30 / R 94.42 /
**F1 94.25**; event onset **F1 91.86**; event onset+offset **F1 86.58**.

| Difference | Kong | Ours |
|---|---|---|
| Dataset | MAESTRO **v2.0.0** | MAESTRO **v3.0.0** |
| Split reported | **test** | **validation** (test sealed) |
| Hop / fmin | 10 ms / 30 Hz | 24 ms / 50 Hz (oracle) |
| GT binarisation | CC64 **> 64** | CC64 **>= 64** |
| Augmentation | hinted, undocumented | none |
| STFT window | **2048 (128 ms)** | **2048 (128 ms)** — identical |

**Architecture:** `Regress_pedal_CRNN` instantiates **three fully independent**
`AcousticModelCRnn8Dropout` towers (onset / offset / frame). Zero shared
backbone — only the log-mel is shared. **No fusion layers**; outputs are combined
solely by threshold rules at decode time. (Their *note* model does have fusion
GRUs; they deliberately omitted it for pedal.)

Per-tower budget ≈ 4.62M (conv ≈483k / fc5 ≈1.379M / biGRU ≈2.759M), so
**≈13.86M total**. Our oracle is a single 3.27M model — but note Kong's
**per-branch** capacity is only ~1.4x ours. The 12x headline is task
replication, not scale.

**`up50` and `strict50` have NO Kong counterpart.** Kong publishes no standalone
pedal-offset F1, and no fixed-50 ms strict metric (theirs uses offset_ratio 0.2,
which is far more lenient). Never table those against Kong.

---

## 7. Remaining untested hypotheses

Not eliminated — never cleanly tested:

1. **Capacity** — 3.27M single model vs Kong's 3 x 4.62M. Entangled with (2);
   separating them needs two controlled runs.
2. **Task specialisation** — three independent towers vs one shared trunk with
   small heads. Kong's pedal-onset tower alone exceeds our entire oracle.
3. **Augmentation** — their paper attributes discrepancies to "different data
   augmentation strategies" without detail. Downgraded as an explanation: it
   would be expected to move frame and event metrics together, but we are
   already *ahead* on frame.

Cheapest remaining decoder idea (CPU-only): re-run the semi-Markov decoder with
candidate boundaries seeded from **state transitions** instead of confidence
peaks — D8 inherited the event heads' 12.6% detection deficit by construction.

---

## 8. Key-up observability finding (transfers to the LED product)

Measured on MAESTRO validation, 137 recordings, **639,425 notes**:

| Measure | Value |
|---|---:|
| **Key-ups occurring while pedal is DOWN** | **412,292 / 639,425 = 64.5%** |
| Onsets while pedal down | 59.6% |
| Mean pedal duty cycle (by time) | 64.5% |
| Median note duration, key-up GT | **0.095 s** |
| Median note duration, pedal-extended GT | **0.410 s** (4.3x longer) |
| Per-file % key-ups under pedal | min 0.0 / median 68.4 / max 99.0 |

**Two-thirds of key-ups are acoustically occluded** — the damper never falls, so
the primary release cue is largely absent. If those were wholly undetectable,
recall would cap at 35.5%, bounding note-with-offset F1 at ~0.524. The shipped
key-up model scores **0.605**, i.e. **above that bound**, so it is already
recovering occluded releases from residual cues.

**The per-file range 0.0% - 99.0% explains the original product complaint** that
release thresholds needed retuning per piece: the observability regime itself
changes drastically between sparse and heavily-pedalled repertoire. That is a
mechanism, not a tuning failure.

Suggested (untested) follow-ups: stratify key-up errors by pedal state (~30
lines, no training — `GtLoaderMaestro.get_midi_eventdata()` returns key events
and sustain states in one call); and use pedal state as a *conditioning* signal
for key-up decoding rather than a replacement (pedal state is a single global
scalar and carries no per-key information, so it can never substitute).

---

## 9. LED product status — unaffected by all of the above

The product path is onset + velocity + key-up, all already deployed and local.

| Metric | Value | Note |
|---|---:|---|
| Onset F1 | 0.9675 (paper) / 0.9677 reproduced | O&V EUSIPCO 2023 shipped checkpoint |
| Onset+velocity F1 | 0.9480 (paper) / 0.9450 reproduced | same checkpoint, stricter paired metric |
| Note-with-offset F1 vs KEY-UP GT | 0.605 clean / 0.587 SBC | our trained key-up head |
| Repeat re-trigger (clean-segment) | 0.859 | the actual product metric |

Baselines (onset F1): Kong 0.9672, Onsets & Frames 0.9480. Our onset number is
the O&V paper's own result reproduced — **not** an improvement over it.

**Caveat on the onset numbers:** measured with `INFERENCE_CHUNK_SIZE = 300 s`
(`2_eval_onsets_velocities.py:146`), i.e. near-whole-file context. The Pi bridge
streams ~2.5 s. A prior receptive-field study found onset peaks 99.2% identical
between 0.19-0.58 s past context and 1.0 s, and 100% identical beyond ~0.77 s,
so the deployed number is very likely close — but this is inferred from
prediction identity, **not** a measured F1 at deployment window. To close it:
re-run `2_eval_onsets_velocities.py` with chunk/overlap matched to the bridge.

Onset/velocity is frozen by design. The one fine-tune attempt (codec
augmentation) **hurt** onsets (0.9678 -> 0.9614 clean) and repeat resolution;
conclusion recorded then was that the shipped baseline is the better product
model.

---

## 10. Reproduction

Everything is reproducible from configs in the repo; no pedal checkpoint was
retained.

**Best pedal model (offline oracle, 24 ms pooled), 12k steps from scratch:**

```
python 6_train_mobile_pedal_regression.py \
  MAESTRO_PATH=<maestro-v3> \
  HDF5_MEL_PATH=<...stft=2048w384h_mel=229(50-8000).h5> \
  HDF5_ROLL_PATH=<...roll_quant=0.024_midivals=128_extendsus=True.h5> \
  EVENT_CACHE_PATH=<cache>/pedal-event-train.pt DATALOADER_WORKERS=8 \
  MODEL_VARIANT=offline_oracle SHARED_HIDDEN=384 TOWER_HIDDEN=256 \
  SHARED_LAYERS=2 BIDIRECTIONAL_HIDDEN=224 DROPOUT=0.15 \
  STATE_WEIGHT=1.0 EVENT_POS_WEIGHT=5.0 CONFIDENCE_WEIGHT=3.0 \
  OFFSET_WEIGHT=2.0 EVENT_RADIUS=3 ALIGN_STATE_TO_EXACT_EVENTS=true \
  FROM_SCRATCH=true FREEZE_FRONTEND=false FREEZE_SHARED=false \
  FREEZE_STATE=false FRONTEND_LR_SCALE=1.0 LR=0.001 WEIGHT_DECAY=0.0003 \
  AMP=true TRAIN_BS=48 TRAIN_BATCH_SECS=10.0 MAX_STEPS=12000
```

Eval: `4_eval_pedal.py ... SPLIT=validation FULL_RECORDING_INFERENCE=true
DECODER=regression EVENT_THRESHOLD=0.8 NMS_RADIUS=3`.

**10 ms variant:** identical except the two HDF5 paths (`stft=2048w160h`,
`quant=0.01`) and `EVENT_RADIUS=7` (preserves the ±70 ms absolute target window
and the positive-frame ratio after the hop change — critical, otherwise the
class imbalance shifts 2.4x and the comparison is confounded).

**Deployable edge model:** `MODEL_VARIANT=edge_multirate_10ms FAST_SIZE=128
CONTEXT_HIDDEN=288 CONTEXT_LAYERS=2 CONTEXT_POOL=5 EVENT_DELAY_FRAMES=43
HEAD_HIDDEN=128 TRAIN_BS=8 TRAIN_BATCH_SECS=6.0 MAX_STEPS=120000
FUTURE_CONTEXT_SECS=0.43` (= 494 ms total lookahead incl. 64 ms STFT half-window).

**Selection protocol used throughout:** screen preregistered checkpoints on
locked32 @ thr 0.3, rank by down F1 then strict50, calibrate threshold on the
same locked32 cache only, then one full-validation pass. Test touched only if
gates pass — they never did.

---

## 11. Operational notes for any future pod

- RunPod's SSH proxy is PTY-only; `scp` reports exit 0 while failing. Transfer
  via PTY SSH + base64 + SHA-256 verification.
- **Multi-line bash sent through the PowerShell -> ssh.exe -> PTY pipeline gets
  silently corrupted** (heredocs and `\` continuations both). Not CRLF (verified
  zero `\r` bytes). Symptom: commands run with *no arguments*, or vanish with no
  log. Use a single flat command, or `python3 -c` with
  `subprocess.Popen([...], start_new_session=True)` and an explicit argument
  list — no shell parsing at all. This cost several hours of misdiagnosis.
- `4_eval_pedal.py` validates prediction-cache identity (checkpoint, model type,
  split, limit, chunk geometry, future context). Never rename a cache to reuse it.
- A local tolerance-sweep patch to `4_eval_pedal.py` (adds 25 ms tolerance and a
  per-tolerance `pedal_down` sweep) was written and syntax-verified locally but
  **never applied on the pod** — the sweep data was not collected.
