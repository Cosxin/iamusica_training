# Streaming sustain-pedal transcription: master research record

Last updated: 2026-08-03 (Asia/Shanghai)

This is the authoritative narrative, decision record, metric ledger, and
reproducibility index for the CC64 work. It supersedes the former proposal,
technical-review, three-day progress ledger, RunPod handoff, and D10 status
report. Machine-readable JSON, checkpoints, caches, logs, and source remain the
authoritative evidence behind the summarized numbers.

The note-onset, velocity, and physical-key-up line is intentionally separate.
Its research record remains in `EVAL_FINDINGS_2026-07-30.md`, and its rebuild
recipe remains in `REBUILD.md`.

## One-picture summary

```text
MAESTRO audio + exact MIDI CC64 timestamps
                 |
                 +--> 10 ms, 229-bin log-mel ---------------------------+
                 |                                                       |
                 |        deployable 500 ms streaming model              |
                 |                                                       v
                 |  causal spectral frontend --> fast 100 fps lane (128)
                 |              |                         |
                 |              +--> pool x5 --> 2x GRU(288) at 20 fps
                 |                                        |
                 |  fast feature(t) + context(t+43) ------+
                 |                     |
                 |                     +--> current state
                 |                     +--> down confidence + sub-frame dt
                 |                     +--> up confidence + sub-frame dt
                 |                     +--> peak/NMS + alternating decoder
                 |                                      |
                 |                                      +--> timestamped CC64
                 |
                 +--> offline controls
                        24 ms pooled biGRU (context oracle)
                        24 ms frequency-flattened biGRU (register control)
                        10 ms pooled biGRU (time-resolution control)

CC64 --> Pi bridge audio-time scheduler --> WebSocket/InputBus
     --> LivePerformanceBus --> sampled-piano sustain and release
```

The selected production candidate is not a larger version of the old
frame-classifier. It is a separate, 1,048,941-parameter, causal multirate
pedal model with exact-timestamp event regression. The existing note/key-up
model remains independent and bit-identical.

## Research question and defensible claim

The product question is whether sustain-pedal down/up boundaries can be
transcribed accurately on a Raspberry Pi 4 using no more than 500 ms of
acoustic future context. The research question is which limitation dominates:

1. structured event routing/decoding;
2. register-aware acoustic evidence;
3. 24 ms versus 10 ms temporal sampling;
4. bounded streaming context versus offline bidirectional context.

The experiments support a bounded claim, not a generic SOTA claim:

- Input-dependent CRF transitions were capable enough, but event confidence
  was routed outside the Viterbi score. Confidence fusion recovered a material
  fraction of the loss.
- A correct bounded semi-Markov decoder produced only a small precision-led
  gain and did not solve sub-100 ms flutter. Decoder mathematics cannot create
  absent acoustic candidates.
- Ordinary frequency flattening, frontend unfreezing, and a high-frequency
  transient branch each produced small gains or null results. These are useful
  controlled baselines but not a publishable novelty by themselves.
- Whole-recording bidirectional context materially helps, but calibrated full
  validation still only narrowly reaches the down gate and narrowly misses the
  strict interval gate. Context is structural, but not the sole explanation.
- The active 10 ms multirate candidate tests the joint hypothesis that fine
  event sampling and parameter reallocation toward frequency-preserving fast
  evidence can improve streaming performance while staying Pi-deployable.

The potential publication contribution is therefore the measured decomposition
of streaming pedal transcription error under an edge-compute constraint,
together with a deployable multirate exact-event model and matched context,
frequency, and hop-resolution controls. Any stronger novelty claim waits for
the final controlled results.

## Protocol, data, and gates

- Dataset: MAESTRO v3, 1,276 recordings; the manifest contains exactly
  962/137/177 train/validation/test recordings and split separation is
  preserved.
- 10 ms cache: mel `(229, 71,723,913)`, roll `(259, 71,723,913)`.
- Cache validation: equal global frame count, exact metadata identity, and
  exact `data_idxs` equality.
- A fresh pod-side audit found 1,276 WAVs and 1,276 MIDIs, matching `(2, 1276)`
  mel/roll index tables, and a 962-entry training event cache whose unique
  recording names exactly equal its event-map keys.
- Formal train split: 95,307 non-overlapping six-second chunks.
- Checkpoint/threshold selection uses a fixed 32-file validation subset.
- Deployment decisions use all 137 validation recordings.
- Test is touched once only after a configuration, checkpoint, threshold, and
  decoder pass full validation. Failed candidates do not consume test data.
- Primary metrics: state F1, down F1 at 50 ms, up F1 at 50 ms, and strict
  down+up interval F1 at 50 ms. Literature-compatible relative-offset metrics
  remain diagnostics and are never substituted for strict product metrics.
- Full-validation gates: state `>=0.88`, down `>=0.75`, strict50 `>=0.70`.
- Pi gate: complete note+pedal feature extraction and inference must meet the
  500 ms cadence without thermal throttling. Float acceptance precedes QAT or
  post-training quantization.

## Experiment ledger

Scores are state/down/up50/strict50 unless a row explicitly omits a metric.
“Locked32” is screening evidence, not a deployment result.

| Line / experiment | Scope | Score | Conclusion |
|---|---|---|---|
| Frozen note-stem pedal v1 | test | `0.88410/0.42746/-/0.31584` | Small train/test gap; event underfitting, not classic overfit |
| Direct-logmel Mobile-AMT-style v1, 3.19M | validation | `0.91392/0.55552/-/0.49252` | Direct acoustics help; frame classification remains insufficient |
| Exact-timestamp regression v2 | validation | `0.91615/0.57816/-/0.51646` | Sub-frame regression helps but misses gates |
| Positive-lookahead 72 ms lineage | locked32 | `0.93609/0.68346/0.84993/0.60855` | Best fine-delay screen; subset optimism is large |
| Same 72 ms lineage | full validation | `0.90762/0.60867/0.79445/0.53850` | Required full-validation correction |
| Dynamic CRF, 16k, uncoupled | locked32 | `0.94503/0.65017/0.84046/0.55477` | State improves while exact events regress |
| Same checkpoint, regression decoder | locked32 | `0.94503/0.66681/0.83417/0.56431` | CRF path suppressed event-head evidence |
| Same checkpoint, confidence-coupled CRF | locked32 | `0.94503/0.70573/0.89217/0.63410` | Routing, not transition-head capacity, was limiting |
| Confidence-coupled CRF | full validation | `0.90821/0.63664/0.84631/0.56881` | Real gain, still far below gates |
| D4 8 s/4 s stitching control | locked32 | `0.94569/0.70747/0.89339/0.63624` | Window seams/context add only about 0.002 strict F1 |
| D5 directly supervised transition routing | full validation | `0.90821/0.63062/0.83858/0.56198` | Causal evidence for diagnosis; double fusion hurts |
| D6 matched frontend unfreeze | full validation | `0.91159/0.63803/0.84990/0.57159` | Only `+0.00138` down over frozen alpha-1 winner |
| D7 flatten projection | locked32 | `0.94563/0.71160/0.89319/0.64106` | Required register-aware baseline; small gain |
| D7b two-stream transient branch | locked32 | `0.94482/0.70712/0.89229/0.63531` | No meaningful gain; branch engagement weak |
| D7c transient auxiliary routing | locked32 | `0.94503/0.70680/0.89261/0.63489` | Transient branch learns its task but main result remains flat |
| D8 bounded semi-Markov | locked32 | `0.94503/0.70835/0.89041/0.63510` | `+0.00262` down, `-0.00176` up; flutter not solved |
| Offline pooled biGRU, threshold 0.3 | full validation | `0.95273/0.72360/0.87046/0.65766` | Whole-recording context helps; locked32 was optimistic |
| Offline pooled biGRU, calibrated 0.8 | full validation | `0.95273/0.75105/0.86773/0.69739` | Down gate +0.00105; strict gate -0.00261 even offline |
| D10 10 ms 64/320 baseline, calibrated 0.8 | full validation | `0.90843/0.65911/0.76345/0.57663` | Pi-safe but does not pass quality gates |
| D10 10 ms 128/288 allocation, step 24k, thr 0.3 | locked32 | `0.92351/0.55399/0.52502/0.28379` | All metrics exceed matched 24k baseline; endpoint pending |
| D10 10 ms 128/288 allocation, step 60k, thr 0.3 | locked32 | `0.93034/0.59295/0.57947/0.33834` | Early advantage reversed; matched baseline is `0.93514/0.62475/0.64893/0.41433` |
| D10 10 ms 128/288 allocation, calibrated 0.75 | full validation | `0.90793/0.66461/0.77510/0.56861` | Smaller model has best down/up but lower strict50; does not pass quality gates |

### Context curve from one offline oracle

The 24 ms pooled oracle was trained once with full bidirectional context and
evaluated with its recurrent future truncated. These are diagnostic lower
bounds for each lag because of train/eval mismatch:

| Total future context | State | Down | Up50 | Strict50 |
|---:|---:|---:|---:|---:|
| 250 ms | 0.95143 | 0.64085 | 0.72682 | 0.46865 |
| 500 ms | 0.95910 | 0.73765 | 0.85036 | 0.62520 |
| 1 s | 0.96361 | 0.76602 | 0.87818 | 0.67556 |
| 3 s | 0.96663 | 0.78060 | 0.89920 | 0.70534 |
| unlimited | 0.96763 | 0.78126 | 0.90390 | 0.70973 |

Most context gain saturates by about one second. The deployable 500 ms point
requires matched training before it can be treated as a final ceiling.

## What the diagnostics established

### Decoder and routing

- The original CRF lattice used state unary plus four dynamic transition
  potentials but excluded the directly supervised confidence heads.
- Adding down/up confidence logits to the corresponding transitions increased
  full-validation down F1 from `0.60867` to `0.63664` in the mature lineage.
- Direct transition supervision recovered similar evidence and showed that
  applying confidence fusion again double-counts it.
- Exact forward-algorithm tests enumerate every path, verify `log Z`, verify
  path probabilities sum to one, compare Viterbi to exhaustive maximization,
  and match autograd marginals to brute force.

### Flutter and semi-Markov structure

- Gaps below 150 ms account for 37.2% of missed downs, motivating the test.
- The bounded semi-Markov recurrence was verified exhaustively against all
  legal candidate paths and uses prefix-summed interval evidence plus monotone
  predecessor queues.
- It improved 100–300 ms errors but increased sub-100 ms and total missed
  downs (`3409 -> 3468`). The small F1 gain was precision-led.
- NMS radius 1 was worse than radius 3. NMS is per event kind and was not the
  mechanism suppressing opposite-kind flutter pairs.

### Acoustic representation

- The old mean-pooled MBConv representation discarded register position.
- Controlled unfreezing, flatten projection, transient bypass, and direct
  transient supervision did not produce the missing tenth of down F1.
- These null/small results bound those particular modifications; they do not
  prove acoustics are solved. The active 10 ms model changes both temporal
  sampling and parameter allocation while remaining edge-deployable.

### Evaluation correctness

- Exact event targets use nearest-frame centers plus signed sub-frame offsets;
  state/event time bases are aligned for models that jointly train them.
- Offset loss is masked to supervised nearest-event frames.
- Delayed outputs discard the unsupervised prefix in training, shift event
  channels back in evaluation, and reject invalid target spans in the bridge.
- Prediction caches validate checkpoint/model/split/limit/chunk/future-context
  identity and save atomically every eight newly inferred recordings.
- Long whole-recording GRUs use exact directional chunking with hidden-state
  carry, verified equal to single-call recurrence.
- New training checkpoints save to a same-directory temporary path and are
  atomically promoted. The already-running formal D10 process predates that
  change, so its overlapping 60k evaluator retries complete deserialization.

## Production candidate: D10 allocation arm

- Class: `MultirateEdgePedalAMTRegression`.
- Parameters: 1,048,941 (`FAST_SIZE=128`, `CONTEXT_HIDDEN=288`, two GRU
  layers), smaller than the 1,130,605 64/320 baseline.
- Input: 16 kHz, centered 2048-sample STFT, 229 mel bins, 160-sample hop.
- Explicit event delay: 43 frames = 430 ms; centered-STFT half-window adds
  64 ms, for about 494 ms worst-case acoustic lookahead.
- Targets: +/-70 ms confidence triangles (`EVENT_RADIUS=7`), positive weight
  5, state BCE, down/up confidence BCE, masked Smooth-L1 offsets.
- Training: all modules from scratch, AdamW, LR `7e-4`, 500-step warmup,
  cosine decay to 0.05x, weight decay `3e-4`, AMP, batch 8, six-second chunks.
- Formal budget: 120,000 steps = 10.07 measured epochs. This is the production
  candidate run, not the 20-step smoke test.
- Selection checkpoints: 1k, 12k, 24k, 60k, and 120k; rank by locked32 down
  F1, then strict50. Calibrate threshold on locked32; evaluate full validation
  once; test only if down/strict gates pass.

Matched step-24k evidence shows the allocation change is not trading one event
direction for the other:

| Metric | 64/320 baseline | 128/288 allocation | Delta |
|---|---:|---:|---:|
| State | 0.90675 | 0.92351 | +0.01676 |
| Down | 0.49328 | 0.55399 | +0.06071 |
| Up50 | 0.43385 | 0.52502 | +0.09117 |
| Strict50 | 0.20104 | 0.28379 | +0.08275 |

At step 60k the allocation arm trails the matched baseline by
state/down/up50/strict50 `-0.00480/-0.03180/-0.06946/-0.07599`. This rules out
claiming a monotonic allocation win from the 24k screen. At 120k, checkpoint
selection and locked32 calibration selected threshold 0.75. Full-validation
state/down/up50/strict50 is `0.90793/0.66461/0.77510/0.56861`, versus
`0.90843/0.65911/0.76345/0.57663` for the 64/320 baseline. The allocation arm
uses 7.2% fewer parameters and slightly improves down/up F1, but loses strict
interval F1. Neither candidate passes the preregistered down or strict50 gate;
therefore test remains sealed and no model is deployed.

## Raspberry Pi and live-path evidence

The rejected 6.58M full-rate-GRU prototype took roughly 1.0/1.9/4.4 seconds
at 50/100/250 frames on the Pi and was removed before GPU training.

The selected multirate architecture passed progressively stronger gates:

- Random-initialization allocation ONNX parity at 97/201/311 frames; SHA-256
  `79614afeb22d3c4f1f7254105abb921d2667dc634282e97a8127e1b729dda068`.
- Complete note+pedal production-path preflight at 60 iterations:
  median 363.65 ms, p95 393.02 ms, max 405.63 ms, zero 500 ms misses,
  peak 71.09 C, no throttling, RSS 162.36 MB.
- Ten-minute full production-path run on the 64/320 graph, including both
  distinct mel transforms and parallel branches: 1,200 cycles,
  median 372.78 ms, p95 396.11 ms, p99 410.10 ms, max 437.95 ms,
  zero deadline misses, max 79.85 C, `throttled=0x0`, RSS 163.5 MB.
- ORT sessions use two threads each; OpenBLAS and OMP are restricted to one
  thread to avoid oversubscription.

The bridge implements dual-hop feature extraction, five-output regression
decode, audio-time scheduling, WebSocket CC64 emission, and delayed-span
validity checks. Midee now routes all InputBus pedal sources through
`LivePerformanceBus`, so Pi pedal down defers sampled-note release and Pi pedal
up releases it. Focused evidence: 16 Python bridge/mel/timing tests, TypeScript
typecheck, and 28 protocol/input/performance tests. Final live verification
waits for the trained selected ONNX to be deployed atomically.

The eight legacy timing checks were hardened from boolean-returning helpers
(which pytest only warned about) into real assertions. They now pass both as
part of the 16-test focused pytest suite and through the direct 8/8 timing
runner, so a failed deadline/geometry check will fail CI rather than merely
print a warning. A fresh Windows verification also passes TypeScript typecheck
and the 28 focused protocol/input/performance tests. The unrelated full Midee
suite currently executes 662 tests successfully but has five pre-test import
failures caused by the existing jsdom canvas/Windows `file:///@solid-refresh`
test environment; those failures do not exercise the pedal path and are not
counted as pedal evidence.

Read-only deployment audit on 2026-08-03 found Pi `CommieX` active at the
expected MAC/IP and `live-audio-led-bridge.service` healthy, but deliberately
still running the old note-only command. No `~/pi-a2m/pedal-edge-10ms.onnx`
exists. The active bridge SHA-256 is
`ce592684be292790de9e809300f45496cb4492ae908e0f88474924bf9a68521f`;
the staged local dual-model bridge SHA-256 is
`297499da37cd750dded56301d7d95aaf8f66ce9c72d60274f5415a336080f883`.
This is the intended safe state until the cross-architecture quality decision
authorizes an atomic model/bridge/service promotion.

## Reproducibility map

Keep these entry points; they retain research or production value:

| Purpose | Entry point |
|---|---|
| Dataset preparation and validation | `runpod_prepare_pedal.sh`, `align_roll_to_mel.py` |
| Frozen pedal baseline | `3_train_pedal.py`, `runpod_pedal_formal.sh` |
| Exact regression and all later variants | `6_train_mobile_pedal_regression.py` |
| Common evaluation/cache/decoders | `4_eval_pedal.py`, `diagnose_pedal_cache.py` |
| ONNX export | `export_mobile_pedal_onnx.py` |
| Lookahead controls | `runpod_pedal_lookahead_sweep.sh`, `runpod_pedal_lookahead_finalize.sh`, `runpod_pedal_fine_lookahead.sh`, `runpod_pedal_fine_finalize.sh` |
| CRF/routing controls | `runpod_pedal_dynamic_crf_fast.sh`, `runpod_pedal_transition_routing_retrain.sh` |
| Acoustic D6/D7 controls | `runpod_pedal_dual_timescale_quick.sh`, `runpod_pedal_d7_flatten_quick.sh`, `runpod_pedal_d7b_two_stream_quick.sh`, `runpod_pedal_d7c_transient_aux_quick.sh` |
| D10 baseline/formal pipeline | `runpod_pedal_edge_10ms.sh`, `runpod_pedal_edge_10ms_e2e.sh` |
| Allocation selection/export | `runpod_pedal_edge_allocation_eval.sh` |
| Cross-architecture release/no-release decision | `runpod_pedal_production_finalize.sh` |
| Offline context/frequency/hop controls | `runpod_pedal_offline_oracle_24ms.sh`, `runpod_pedal_offline_oracle_calibrate.sh`, `runpod_pedal_offline_oracle_flatten_24ms.sh`, `runpod_pedal_offline_oracle_pooled_10ms.sh`, `runpod_pedal_oracle_controls_e2e.sh` |
| Pi combined benchmark | `pi_combined_thermal_benchmark.py` |
| Final evidence manifest | `collect_pedal_evidence.py` |
| Mathematical/model tests | `tests/test_crf.py`, `tests/test_pedal.py`, `tests/test_mobile_pedal.py`, `tests/test_diagnose_pedal_cache.py` |

The consolidated authoritative RunPod suite for those four files passes
`44 passed` in 5.62 seconds on the current implementation. The Windows host's
default Python lacks PyTorch, so it is not treated as the training-test
environment.

`collect_pedal_evidence.py` atomically writes a machine-readable inventory of
metric JSONs, ONNX artifacts, source hashes, git state, dependency versions,
and mel/roll identity. Its two platform-independent unit tests pass locally;
the final manifest will be generated on the pod after the queued controls so
it captures their outputs as well.

Remote evidence roots on the active pod are `/workspace/runs`,
`/workspace/cache`, `/workspace/artifacts`, and
`/workspace/iamusica_training/results_eval`. Pod addresses, PIDs, and one-off
wait helpers are operational state, not durable research documentation.

Repository cleanup leaves this file as the sole pedal-research narrative.
`EVAL_FINDINGS_2026-07-30.md` and `REBUILD.md` remain because they document the
separate onset/velocity and key-up line of research; `README.md` now points
here. Obsolete pedal notes, resume files, completed one-off launchers, Python
bytecode, and pytest caches were removed. Reproducible experiment entry points
and raw metric JSON remain.

## Authoritative artifacts already locked

- D10 baseline full-validation threshold-0.8:
  state/down/up50/strict50 `0.90843/0.65911/0.76345/0.57663`.
- Offline pooled oracle threshold-0.3 JSON SHA-256:
  `d5c4276e5c5e69916e467033489e45e6f3dd156c779860798a629e52df8e6366`.
- Offline pooled oracle calibrated threshold-0.8 JSON SHA-256:
  `370c227606b9b39100fd91366610ed1e01e6fd123d9993bb4b52273920519bfc`.
- Offline full-validation prediction cache SHA-256:
  `e61f4ee87b7a5bb0300ca5fcb66e89077cfe5fd45f4830f53e15ca74dd90de24`.
- Pi ten-minute model-only JSON:
  `results_eval/edge-10ms/pi-combined-thermal-10min.json`; its metadata
  correctly says feature extraction is excluded. The stronger full-path
  measurement above is currently recorded in this master report and remote
  logs and must be copied into the final artifact bundle.
- Allocation step-60k locked32 JSON SHA-256:
  `1a30876a1e64f8675221f2d92a8bb95dcd16e9ee39f6b46b56f679a1a4744277`;
  prediction-cache SHA-256:
  `201747fbe06a55d32ac4787aaa9508fda61d20996ab12cf6e3d4bc575b08c54e`.
- Allocation step-120k checkpoint SHA-256:
  `31efd2cd8a8471a5e273b26a1b9a241b883570a3b2f4534e88e8fe16ff5fecbb`;
  exported five-output ONNX SHA-256:
  `fb60d0227f20ae173bb62e57760028e7e6d7d46eacc01d23950f2b14d2b90d7d`.
- Baseline step-120k checkpoint SHA-256:
  `4c2f65de520f1ea8b4e75901a62332f2e3dba340ce788d13c4a40009d75a517e`;
  exported ONNX SHA-256:
  `eadcabb0c59d25942ac71b458c98299201146e901a0e8d845c2224237190f364`.
- Offline pooled-oracle step-12k checkpoint SHA-256:
  `5d2157eabcb54906c32b129aecc57066f762c3effc6757edaf267e81dd466d8a`.
- All three checkpoints above are present under local
  `models/pedal_checkpoints/` and independently hash-verified against RunPod.

## Pending evidence and stop conditions

1. Run the matched 24 ms flatten and 10 ms pooled offline controls. These
   separate register preservation from temporal sampling before any
   publication claim.
2. Preserve final JSON, prediction-cache/checkpoint/ONNX hashes, dependency
   versions, and the honest go/no-go decision here.

Deployment and live CC64 verification are intentionally stopped: the production
decision is `deploy=false`, so touching test or promoting a graph to the Pi
would violate the preregistered gate policy.

Do not prolong a failed architecture merely because GPU time remains. Do not
claim deployment, SOTA, live synthesis, or publication readiness from subset
scores, random-weight Pi timing, or implementation-only tests.
