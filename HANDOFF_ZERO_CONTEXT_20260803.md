# Zero-context handoff: streaming piano pedal transcription

Last authoritative update: 2026-08-03 02:05 Asia/Shanghai
(2026-08-02 18:05 UTC on RunPod).

This document is deliberately self-contained. A replacement agent should be
able to resume the work by reading only this file. Treat live filesystem,
process, and result JSON state as authoritative if anything below is stale.

## 1. Mission and non-negotiable product constraints

The user wants an end-to-end, research-quality sustain-pedal transcription
system integrated with the existing streaming piano product:

1. design and implement a stronger but edge-sized model;
2. prepare and validate MAESTRO data;
3. train, select, calibrate, and evaluate models without test leakage;
4. benchmark the actual note+pedal production path on Raspberry Pi;
5. synthesize pedal down/up as MIDI CC64 in the browser/piano path only if the
   quality and Pi gates pass;
6. preserve reproducible scripts, checkpoints, aggregate metrics, hashes, and
   an honest research report;
7. transfer all durable artifacts locally before the RunPod is terminated;
8. push reproducible source/docs to GitHub, but do not accidentally publish
   private checkpoints or raw per-recording artifacts.

The deployable system is causal/fixed-lag streaming. The bridge already has a
roughly 500 ms acoustic lookahead budget. Offline bidirectional models are
oracles/controls, not deployable candidates. Do not silently relax this.

The held-out test split is sealed. Exactly one final frozen candidate may earn
the single test evaluation after checkpoint, decoder, and threshold are
selected on validation and all preregistered gates pass. That candidate may be
an offline paper model; test eligibility and Pi deployability are separate.
No result available when this rule was written had passed, so test had not
been touched and no new pedal model had been deployed to the Pi.

## 2. Critical live status: read before touching RunPod

At 2026-08-02 18:05 UTC:

- Active base-control trainer: PID `90771`, parent shell `90770`.
- It is training the 24 ms frequency-flattened offline oracle from scratch.
- Progress at the snapshot: step `2670 / 12000`, about 121 steps/minute.
- GPU: RTX A4500, 20,470 MiB total, 14,765 MiB used, 98% utilization,
  75 C, about 191 W.
- Base run directory:
  `/workspace/runs/pedal-offline-oracle-flatten-24ms-fromscratch-20260802`
- Base log:
  `/workspace/pedal-offline-oracle-flatten-24ms-formal.log`
- Expected base checkpoint:
  `/workspace/runs/pedal-offline-oracle-flatten-24ms-fromscratch-20260802/step-12000.torch`

The old queue supervisor, PID `85582`, is intentionally in Linux state `T`
(SIGSTOP). It was frozen without stopping its child trainer. **Do not send
SIGCONT to PID 85582.** If resumed, it will evaluate the base model and then
start the postponed 10 ms pooled control, competing with the new schedule.

The old evidence collector PID `87198` is harmlessly waiting for PID 85582 to
exit. Leave both alone until the new extended run has completed and all
artifacts are collected. They may then be terminated explicitly.

The new autonomous supervisor is PID `91721` (restarted after the reporting
rules below were preregistered):

```text
/workspace/iamusica_training/runpod_pedal_flatten_extended_e2e.sh
SHA-256 d1e2fc384f999059be8544cbeaa36ef7d288704221051b1fc0f97e239cca7e3e
```

It is already launched under `nohup` and will:

1. wait for the atomically saved/deserializable 12k base checkpoint;
2. evaluate the exact 12k matched-exposure result on locked32;
3. calibrate threshold over 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9;
4. run all 137 validation recordings once at the selected threshold;
5. start a separate extended training stage initialized from the 12k weights;
6. train 48,000 additional steps at batch 48 (about 6.6 hours at the measured
   rate), preserving 12k as an independent controlled result;
7. save every 6,000 additional steps;
8. evaluate additional steps 6k, 18k, 30k, and 48k (total-equivalent exposure
   18k, 30k, 42k, and 60k);
9. calibrate the winner and run one full-validation evaluation;
10. write a completion marker.

The new supervisor status file is the quickest source of truth:

```bash
cat /workspace/pedal-flatten-extended.status
```

Expected stages are:

```text
WAITING_FOR_BASE
BASE_12K_EVAL
EXTENDED_TRAIN
EXTENDED_EVAL
COMPLETE
```

On failure it contains `FAILED`, UTC time, exit code, and shell line number.
The supervisor log is:

```text
/workspace/pedal-flatten-extended-supervisor.log
```

Extended training output and log:

```text
/workspace/runs/pedal-offline-oracle-flatten-24ms-extended-from12k-20260803
/workspace/pedal-offline-oracle-flatten-24ms-extended-formal.log
```

Extended aggregate results:

```text
/workspace/iamusica_training/results_eval/offline-oracle-flatten-24ms-extended
```

The 12k matched result will be in:

```text
/workspace/iamusica_training/results_eval/offline-oracle-flatten-24ms
```

### Fast monitoring commands

RunPod SSH requires a PTY through its proxy. From Windows PowerShell:

```powershell
$key = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
@(
  'date -u',
  'cat /workspace/pedal-flatten-extended.status',
  'tail -3 /workspace/pedal-offline-oracle-flatten-24ms-formal.log 2>/dev/null',
  'tail -3 /workspace/pedal-offline-oracle-flatten-24ms-extended-formal.log 2>/dev/null',
  'ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd --sort=-%cpu | head -16',
  'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader',
  'exit'
) | C:\Windows\System32\OpenSSH\ssh.exe -tt -i $key `
  '<POD_ID>@ssh.runpod.io'
```

Direct SSH identity (no password is stored in this repository):

```text
ssh <POD_ID>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## 3. Why training was extended and how to interpret it

Kong et al.'s official training recipe uses a separate pedal CRNN trained for
300,000 iterations at batch 12 on 10-second segments: 3.6 million examples.
Their repository reports 571,589 indexed training segments, so this is about
6.3 indexed-segment epochs. In raw audio-duration exposure it is roughly 62
passes over MAESTRO v2's 161.3 training hours. Their training took about one
week on a V100.

Our 12k x batch-48 base control sees 576,000 examples: about one Kong-style
indexed-segment epoch or roughly ten raw-duration passes. It is the necessary
matched-exposure control against our pooled 24 ms oracle, but it is far below
Kong's total exposure. The user explicitly requested that the 12k result be
retained and that this model continue training for at least six hours.

The extended stage loads 12k model weights with `INIT_CHECKPOINT`, but the
current trainer does not restore optimizer/scheduler state. It is therefore an
explicit second optimization stage, not bit-exact continuation. It resets
AdamW, uses LR `5e-4`, 200 warmup steps, cosine decay to 0.1x, weight decay
`3e-4`, batch 48, AMP, and 48,000 additional steps. This choice is intentional:
preserve the controlled 12k result, then measure whether substantially more
exposure closes the acoustic gap without running seven hours at constant
`1e-3`.

At 121 steps/minute, 48k additional steps take about 6 hours 37 minutes. Base
training had about 77 minutes remaining at the snapshot. Base calibration/full
validation may take 15-30 minutes. End-to-end completion was therefore
expected roughly 8-8.5 hours after the snapshot.

## 4. Workspace and repository map

Local workspace root:

```text
D:\code\android\pi
```

Important repositories/directories:

```text
D:\code\android\pi\iamusica_training  # training/evaluation research repo
D:\code\android\pi\midee              # browser UI + versioned Pi bridge copy
D:\code\android\pi\bluetooth_receiver # original bridge working directory; not a Git repo
D:\code\android\pi\.publish\iamusica_training # code-only publication worktree
```

RunPod training checkout:

```text
/workspace/iamusica_training
```

Authoritative research overview before this handoff:

```text
PEDAL_RESEARCH_MASTER.md
```

This handoff supersedes it only for live process/schedule state. Final results
must be merged back into both documents, or this handoff should be renamed as
the single final operational record.

## 5. Git state and public publishing policy

### Training repository

Original working tree branch: `codec-offset-finetune`.

- Broad local commit: `b9dbbc5 Add reproducible streaming pedal transcription pipeline`.
- It includes source plus raw results and two older key-up checkpoints.
- It is intentionally **local only**. A public push was rejected because it
  would publish model binaries/raw artifacts without an informed release
  decision. Do not retry or work around that rejection.
- There are unrelated pre-existing deleted files under `assets/`. They belong
  to the user. Do not restore, stage, or otherwise alter them.

Safe code-only publication worktree:

```text
D:\code\android\pi\.publish\iamusica_training
branch: cosxin/pedal-reproduction
commit: 4bcbc7e Add reproducible streaming pedal transcription pipeline
```

It is pushed only to the user's `Cosxin/iamusica_training` fork. The former
PR-associated branch `agent/pedal-reproduction` was deleted after its commit
was preserved under the new name. Upstream draft PR 2 was closed without merge:

```text
https://github.com/andres-fr/iamusica_training/pull/2
```

After final metrics:

1. copy updated source/docs and compact aggregate JSONs into the code-only
   worktree;
2. exclude `.torch`, `.onnx`, caches, raw per-recording JSON, and large logs;
3. commit and push a second scoped commit only to
   `Cosxin/iamusica_training:cosxin/pedal-reproduction`;
4. do not reopen or create an upstream PR unless the user explicitly asks.

### Midee/browser/bridge repository

Branch: `cosxin/pi-pedal-sustain`.

```text
commit 0a554ce Integrate Pi evaluation and pedal sustain path
```

Pushed only to `Cosxin/midee`. The former PR-associated branch
`agent/cite-onsets-velocities` was deleted after preserving its commit under
the new name. Upstream draft PR 3 was closed without merge:

```text
https://github.com/aayushdutt/midee/pull/3
```

The authoritative tested bridge copy is versioned at:

```text
midee/pi_bridge/live_audio/
```

It contains the Python bridge, mel extraction, systemd service template,
requirements, README, and focused tests. The original working bridge remains
under `bluetooth_receiver`.

## 6. Dataset preparation and validation

Dataset is MAESTRO v3.0.0 on RunPod:

```text
/root/maestro-v3.0.0
```

Direct audit already completed:

- 1,276 aligned WAV/MIDI recordings.
- Splits: 962 train, 137 validation, 177 test.
- Test remains sealed.

24 ms features:

```text
/root/data/h5-clean/MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5
/root/data/h5-clean/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5
```

10 ms features:

```text
/root/data/h5-10ms/MAESTROv3_logmel_sr=16000_stft=2048w160h_mel=229(50-8000).h5
/root/data/h5-10ms/MAESTROv3_roll_quant=0.01_midivals=128_extendsus=True.h5
```

Feature facts:

- audio sample rate 16 kHz;
- centered STFT window 2048 samples (128 ms; half-window future is 64 ms);
- 229 mel bins from 50 to 8,000 Hz;
- 24 ms hop uses 384 samples;
- 10 ms hop uses 160 samples;
- 24 ms mel shape `(229, 71723913)`;
- 24 ms roll shape `(259, 71723913)`;
- mel/roll `data_idxs` are exactly equal with shape `(2, 1276)`;
- metadata arrays are exactly equal;
- event cache contains exactly the 962 unique train names.

Exact-event cache:

```text
/workspace/cache/pedal-event-train.pt
```

Training targets are pedal state plus confidence triangles around exact down
and up transitions, plus signed sub-frame offset regression. Offset loss is
masked to supervised nearest-event frames. This is not ordinary frame-only
classification.

## 7. Model families and architecture decisions

### Deployable reduced streaming model (completed)

Class: `MultirateEdgePedalAMTRegression`.

```text
229-bin 10 ms log-mel
  -> frequency-preserving causal fast frontend (fast size 128)
  -> two paths
       fast lane: exact 10 ms transient features delayed 43 frames
       context lane: causal pool x5 -> 2-layer GRU width 288 -> repeat to 10 ms
  -> state head from context
  -> down confidence + down offset + up confidence + up offset
     from [delayed fast, current context]
```

Parameters: 1,048,941. Baseline 64/320 model has 1,130,605 parameters, so the
allocation model is 7.2% smaller.

Lookahead: 43 x 10 ms explicit delay = 430 ms, plus the centered-STFT
half-window of 64 ms, about 494 ms worst-case acoustic lookahead.

Formal training: 120,000 steps, batch 8, 6-second chunks, from scratch, AMP,
AdamW, LR 7e-4 with warmup/cosine. Training completed successfully. Final
single-minibatch loss was 2.06194; do not treat a single minibatch as a smooth
convergence statistic.

### Matched 10 ms streaming baseline (completed)

Fast size 64, context GRU 320. Same 120k exposure and evaluation protocol.
It is slightly larger and has slightly higher strict interval F1, but slightly
lower down/up event F1.

### 24 ms pooled offline oracle (completed)

Same mature acoustic frontend/towers as the original streaming regression
model, but the shared causal GRU is replaced by a two-layer bidirectional GRU.
This isolates unlimited future context. It established that future context is
important but does not fully explain the SOTA gap.

### Active 24 ms flatten offline control

Class: `OfflinePedalAMTRegressionFlatten`.

```text
229-bin 24 ms log-mel
  -> CNN/MBConv channels (32, 64, 96, 160), frequency downsample x16
  -> retain 15 frequency positions instead of averaging them
  -> flatten 160 x 15 at each time
  -> learned projection to 160
  -> 2-layer bidirectional GRU, width 191 per direction
  -> projection/context normalization and three regression towers
  -> state, down confidence/offset, up confidence/offset
```

Parameters: 3,270,277, all trainable. The 191-wide biGRU and projection size
160 were selected to match the pooled 224-wide oracle within about 0.1% total
parameters. This is a controlled test of whether preserving register/frequency
location adds value; it is offline and not a Pi model.

Base 12k config:

- from scratch;
- batch 48, ten-second chunks;
- 12,000 steps, checkpoint each 1,000;
- LR 1e-3 constant, AdamW weight decay 3e-4;
- AMP, eight data workers;
- confidence weight 3, offset weight 2, state weight 1;
- event radius 3 frames = +/-72 ms at 24 ms hop;
- all frontend/shared/state modules unfrozen;
- exact event/state alignment enabled.

### Prior dynamic CRF / decoder work

The dynamic CRF has input-dependent 2x2 transition potentials. Its forward
algorithm sums every legal state path; Viterbi finds the maximum-probability
path. Exhaustive tiny-lattice tests verify `log Z`, total path probability 1,
Viterbi equality to brute force, and autograd marginals.

The important diagnosis was routing, not transition-head capacity: the CRF
transition lattice initially ignored directly supervised confidence heads.
Confidence-coupled routing improved results, but decoder mathematics could not
manufacture missing acoustic event evidence.

A bounded semi-Markov interval decoder was also implemented/tested. It helped
some 100-300 ms errors but did not solve sub-100 ms flutter. NMS radius 1 was
worse than radius 3. These are publishable negative/diagnostic results, not the
current training path.

## 8. Evaluation protocol, metric meanings, and gates

Selection protocol:

1. evaluate preregistered checkpoints on a deterministic locked subset of 32
   validation recordings (`locked32`) at threshold 0.3;
2. select by pedal-down F1, then strict50 F1;
3. calibrate threshold on that same locked32 cache only;
4. evaluate all 137 validation recordings for the frozen selected
   checkpoint/threshold and label every full-validation run by its exact
   checkpoint, threshold, decoder, and cache identity;
5. touch test only if all gates pass.

Full-validation gates:

```text
state F1    >= 0.88
pedal-down  >= 0.75
strict50    >= 0.70
```

Metric interpretation:

- `state F1` measures frame occupancy (pedal held/not held). Long easy spans
  dominate it, so it can be high despite poor event boundaries.
- `down F1` matches pedal-down events under the evaluator's event tolerance.
- `up50 F1` requires pedal-up within 50 ms absolute error.
- `strict50 F1` requires both the corresponding down and up boundaries of an
  interval to be within 50 ms. It can be lower than either marginal event F1
  because both must be correct for the same interval.

### Preregistered decisions for the active flatten experiment

These decisions were frozen on 2026-08-03 before either the 12k flatten result
or extended flatten result existed. They override any ambiguous wording
elsewhere in this file.

#### A. Sealed test applies to offline paper candidates too

Test is **not reserved exclusively for a deployable model**. There are two
conceptual tracks:

- deployment track: causal/Pi-eligible models must pass the deployment gates;
- offline paper track: an offline model may earn one final test evaluation as
  a research result even though it can never be deployed.

The current streaming candidates failed validation, so none is eligible for
test. For the remaining offline track, validation must first select exactly
one frozen flatten checkpoint and threshold. That single offline candidate is
authorized for one test evaluation only if its full-validation down F1 is at
least 0.75 and strict50 F1 is at least 0.70. Passing does not authorize testing
both 12k and extended checkpoints: validation selects one. If no offline
candidate passes, test stays sealed. The test result is a paper number, not a
deployment authorization. Never use it to tune or choose another model.

#### B. The extended flatten run is not an architecture-matched comparison

The only controlled pooled-vs-flatten architecture comparison is:

```text
pooled 24 ms oracle at 12k, batch 48
versus
flatten 24 ms oracle at 12k, batch 48
```

The 60k-equivalent flatten endpoint has roughly five times the exposure of the
12k pooled oracle. It may answer only **does more exposure help this flatten
model?** It must not be used to attribute a gain to flattening/frequency
preservation. An extended pooled 24 ms oracle at matched exposure is not part
of the currently running six-hour job. It becomes a required follow-up before
any paper claim that an extended flatten architecture beats pooling. If compute
is unavailable, report 12k as the controlled architecture result and the
extended run only as an unmatched learning/exposure diagnostic.

#### C. Effect-size rule for “flatten worked”

Use the earlier acoustic-frontend bar, fixed before results:

- primary architecture-success criterion at matched 12k exposure:
  full-validation pedal-down F1 improvement of at least **+0.020 absolute**
  over the independently calibrated 12k pooled oracle;
- non-degradation guard: strict50 F1 may not fall by more than 0.010 absolute;
- state F1 must remain at least 0.88.

The calibrated 12k pooled reference is down `0.751048820659436`, strict50
`0.697390637786548`, state `0.952734225358917`. Therefore the numerical
matched-control success thresholds are down at least `0.771048820659436`,
strict50 at least `0.687390637786548`, and state at least `0.88`.

For the unmatched extended run, “more exposure helped flatten” uses the same
+0.020 down-F1 improvement and -0.010 strict50 guard relative to the calibrated
12k flatten result. Even if satisfied, this does not establish an architecture
effect until a pooled model receives matched exposure.

#### D. Kong-compatible and internal metrics

The headline reporting quadruple is now:

```text
state / down50 / Kong onset+offset(50 ms, offset_ratio 0.2) / strict50
```

The JSON mapping is:

```text
state     -> pedal_state.f1
down50    -> pedal_down.f1
Kong O+O  -> pedal_event_onset_offset.f1
strict50  -> fixed_tolerance_diagnostic["50ms"].down_and_up.f1
```

`up50 = fixed_tolerance_diagnostic["50ms"].pedal_up.f1` remains a useful
internal diagnostic, but it is no longer the third headline metric and must
never be placed beside a Kong number as though it were published by Kong.

Kong reports pedal-onset F1 at 50 ms (`0.9186`) and joint onset+offset F1 with
50 ms onset tolerance and offset tolerance `max(50 ms, 0.2 * interval
duration)` (`0.8658`). Kong does not report standalone pedal-offset F1 or our
fixed-50 ms strict interval F1. Therefore:

- our `pedal_down.f1` is the intended onset-F1 comparison;
- our `pedal_event_onset_offset.f1` is the intended joint comparison;
- `up50` has no Kong counterpart;
- `strict50` is harder than Kong's duration-relative joint metric and is not
  numerically comparable to `0.8658`.

Other protocol caveats must accompany even the two comparable columns: Kong
uses MAESTRO v2, a roughly 20M-parameter dedicated CRNN, 10 ms frames,
bidirectional full-track context, 300k x batch-12 training, and reports final
test results. Our active flatten model uses MAESTRO v3, 24 ms frames, 3.27M
parameters, and validation until it earns the one authorized test touch.

#### E. Honest validation-use rule

“Full validation exactly once” was inaccurate and must not appear in a paper.
Validation is a repeatedly used decision/development set. Existing pooled and
10 ms experiments legitimately evaluated distinct `(checkpoint, threshold)`
configurations on all 137 files, including uncalibrated and calibrated runs.

The actual rule is:

- prediction inference should be cached once per checkpoint/model/split/chunk
  geometry and may be decoded at multiple thresholds;
- every reported row must identify checkpoint, threshold, decoder, cache, and
  result path;
- locked32 is used for checkpoint/threshold selection in the active protocol;
- after that selection, run one all-137 evaluation for that frozen active
  configuration;
- separate preregistered candidates (for example 12k flatten and extended
  flatten) may each have a full-validation result;
- do not tune on an all-137 result and then portray a later result on the same
  validation set as an untouched holdout estimate.

Test, not validation, is the irreversible single-touch evaluation set.

Prediction caches validate checkpoint hash/model type/split/limit/chunk
geometry/future-context identity and save atomically. Do not reuse a stale
cache by manually changing filenames.

## 9. Key experiment ledger and metric provenance

New headline rows are ordered
`state / down50 / Kong onset+offset / strict50`. Historical diagnostic rows
that explicitly say **legacy order** remain
`state / down50 / up50 diagnostic / strict50`; do not compare their third
number to Kong.

### Mature 24 ms lineage and decoder/acoustic diagnostics

| Experiment | Scope | Legacy-order metrics | Evidence path | Conclusion |
| --- | --- | --- | --- | --- |
| Regression decoder on mature checkpoint | locked32 | `0.94503/0.66681/0.83417/0.56431` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | supervised event heads held evidence CRF path ignored |
| Confidence-coupled CRF | locked32 | `0.94503/0.70573/0.89217/0.63410` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | routing was limiting |
| Confidence-coupled CRF | full validation | `0.90821/0.63664/0.84631/0.56881` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | real gain, still below gates |
| D4 8s/4s stitch control | locked32 | `0.94569/0.70747/0.89339/0.63624` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | seams/context add only about 0.002 strict |
| D5 transition supervision | full validation | `0.90821/0.63062/0.83858/0.56198` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | double fusion hurts |
| D6 frontend unfreeze | full validation | `0.91159/0.63803/0.84990/0.57159` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | only +0.00138 down |
| D7 flatten projection | locked32 | `0.94563/0.71160/0.89319/0.64106` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | required register-aware baseline; small gain |
| D7b two-stream transient | locked32 | `0.94482/0.70712/0.89229/0.63531` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | no meaningful gain |
| D7c transient auxiliary | locked32 | `0.94503/0.70680/0.89261/0.63489` | `PEDAL_RESEARCH_MASTER.md` only; raw JSON absent from current inventory | branch learned auxiliary task, main result flat |
| D8 bounded semi-Markov | locked32 | `0.94503/0.70835/0.89041/0.63510` | remote `results_eval/semi_markov/tw0p5_sw0p25.json`; verify before citation | tiny precision-led gain; flutter unsolved |

The missing raw provenance for legacy D3-D7 rows is now an explicit handoff
defect, not hidden certainty. Do not place those rows in a final paper table
until the generating JSON/log is restored and hashed, or the experiment is
reproduced. Their use here is diagnostic continuity only.

### Context oracle and latency curve

24 ms pooled offline oracle at threshold 0.3, full validation, headline order:
`0.952734/0.723600/0.673372/0.657655`.

Calibrated threshold 0.8, full validation, headline order:
`0.952734/0.751049/0.709800/0.697391`.

The corresponding internal up50 diagnostics are `0.870460` and `0.867729`;
they are not Kong-comparable.

Authoritative local extracted JSONs:

```text
artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-full-validation-threshold03.json
artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-full-validation-calibrated.json
```

Remote originals are the same relative paths under
`/workspace/iamusica_training/`.

It narrowly passes down by 0.00105 and misses strict50 by 0.00261.

One full-context training run was evaluated with truncated future state. This
table is an internal **legacy-order** diagnostic (`state/down/up50/strict50`),
not a Kong comparison:

| Total future | State | Down | Up50 | Strict50 |
| ---: | ---: | ---: | ---: | ---: |
| 250 ms | 0.95143 | 0.64085 | 0.72682 | 0.46865 |
| 500 ms | 0.95910 | 0.73765 | 0.85036 | 0.62520 |
| 1 s | 0.96361 | 0.76602 | 0.87818 | 0.67556 |
| 3 s | 0.96663 | 0.78060 | 0.89920 | 0.70534 |
| unlimited | 0.96763 | 0.78126 | 0.90390 | 0.70973 |

Source mapping:

```text
250 ms   -> artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-limit32-lag250ms.json
500 ms   -> artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-limit32-lag500ms.json
1 s      -> artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-limit32-lag1000ms.json
3 s      -> artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-limit32-lag3000ms.json
unlimited-> artifacts/runpod_handoff/core/results_eval/offline-oracle/step12000-limit32-full.json
```

These truncated points are mild lower bounds because the model was trained
full-context, not at each fixed lag. Most context benefit saturates by about
one second.

### Formal 10 ms streaming results

Baseline 64/320, threshold 0.8, full validation, headline order:

```text
0.908427 / 0.659105 / 0.603695 / 0.576625
```

Internal baseline up50 diagnostic: `0.763446`.

Source:

```text
artifacts/runpod_handoff/core/results_eval/edge-10ms/step120000-full-validation-threshold08.json
```

Reduced allocation 128/288, threshold 0.75, full validation, headline order:

```text
0.907928 / 0.664606 / 0.606055 / 0.568614
```

Internal allocation up50 diagnostic: `0.775096`.

Sources:

```text
results_eval/edge-10ms-allocation-128x288/step120000-full-validation.json
artifacts/runpod_handoff/core/results_eval/edge-10ms-allocation-128x288/step120000-full-validation.json
```

Important filename disambiguation: the similarly named baseline file
`artifacts/runpod_handoff/core/results_eval/edge-10ms/step120000-full-validation.json`
is the uncalibrated threshold-0.3 run, with
state/down/Kong-O+O/up50/strict50
`0.908427/0.583801/0.470879/0.639353/0.395563`. It is not the source of the
quoted calibrated baseline or allocation result.

Production decision:

```json
{
  "best_research_candidate": "edge_10ms_128x288",
  "deployment_selected": null,
  "deploy": false,
  "reason": "no candidate passed all full-validation gates",
  "test_policy": "do not touch test split"
}
```

The smaller allocation model improves down by about 0.0055 and up50 by about
0.0117, but loses about 0.0080 strict50. It is a useful edge Pareto point, not
a release candidate under the current gates.

## 10. Raspberry Pi and live synthesis state

The Pi is not the current bottleneck.

Completed evidence:

- random-initialization allocation ONNX dynamic-time parity verified at 97,
  201, and 311 frames;
- complete note+pedal production-path preflight, 60 iterations:
  median 363.65 ms, p95 393.02 ms, max 405.63 ms, zero deadline misses,
  no throttling;
- ten-minute complete production-path run with 64/320 graph:
  median 372.78 ms, p95 396.11 ms, p99 410.10 ms, max 437.95 ms,
  zero deadline misses, max 79.85 C, no throttling.

The Pi currently runs the old note-only service. This is intentional and safe.
No pedal graph was promoted because validation gates failed. Do not deploy a
failed candidate merely to demonstrate CC64.

The browser/bridge implementation already supports predicted pedal down/up:

- bridge validates five-output model contract;
- predicted state/confidence/offsets become MIDI CC64 down/up events;
- UI/piano path sustains currently held/released notes appropriately;
- ordinary Pi-stream decoded notes still print and play;
- key-up support remains from the newer note model;
- active old service is left unchanged until a pedal model passes.

If a future model passes, deployment must use hash/output-contract checks and
atomic promotion, followed by a real model-generated CC64 sustain/release test.
Until then, deployment/live CC64 verification is a stopped branch, not missing
implementation.

Pi access instructions live in `.agents/skills/pi-access/SKILL.md`. Any agent
accessing the Pi must read and follow that skill first. Last known Pi IP was
`192.168.10.220`, user `ltq`; rediscover rather than blindly trust the IP.

## 11. Local checkpoints and transferred artifacts

Local checkpoint directory:

```text
D:\code\android\pi\iamusica_training\models\pedal_checkpoints
```

Already downloaded and independently SHA-256 verified against RunPod:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `allocation-128x288-step120000.torch` | 12,677,226 | `31efd2cd8a8471a5e273b26a1b9a241b883570a3b2f4534e88e8fe16ff5fecbb` |
| `baseline-64x320-step120000.torch` | 13,657,194 | `4c2f65de520f1ea8b4e75901a62332f2e3dba340ce788d13c4a40009d75a517e` |
| `offline-oracle-pooled24-step12000.torch` | 39,418,005 | `5d2157eabcb54906c32b129aecc57066f762c3effc6757edaf267e81dd466d8a` |
| `allocation-128x288-step120000.onnx` | 4,230,033 | `fb60d0227f20ae173bb62e57760028e7e6d7d46eacc01d23950f2b14d2b90d7d` |
| `baseline-64x320-step120000.onnx` | 4,556,684 | `eadcabb0c59d25942ac71b458c98299201146e901a0e8d845c2224237190f364` |

Manifest:

```text
models/pedal_checkpoints/CHECKPOINTS.md
```

The `.torch`/`.onnx` files are narrowly ignored by Git to prevent accidental
public release.

Core completed handoff archive:

```text
artifacts/runpod_handoff/pedal-core-handoff-20260803.tar.gz
SHA-256 de3e74c8dc3e120121da9a8bf94a21d448b13b1eff469d4df4e2ffea53db3a5a
```

It has been safely path-audited and extracted under:

```text
artifacts/runpod_handoff/core
```

It contains baseline/allocation/offline-oracle aggregate result trees, both
streaming ONNX files, run configs, and formal/evaluation logs.

Compact allocation results were also copied into:

```text
results_eval/edge-10ms-allocation-128x288/
```

Remote primary artifacts:

```text
/workspace/runs/pedal-edge-10ms-allocation-128x288-radius7-10ep-500ms-20260802/step-120000.torch
/workspace/runs/pedal-edge-10ms-radius7-10ep-500ms-20260802/step-120000.torch
/workspace/runs/pedal-offline-oracle-24ms-fromscratch-20260802/step-12000.torch
/workspace/artifacts/pedal-edge-10ms-allocation-128x288-step120000.onnx
/workspace/artifacts/pedal-edge-10ms-baseline-step120000.onnx
```

Pending transfer after completion:

- flatten 12k checkpoint and selected result/cache/logs;
- extended flatten selected checkpoint plus final/milestone checkpoints judged
  research-relevant;
- extended aggregate JSONs and prediction-cache hashes;
- final evidence manifest;
- dependency versions and final source/Git hashes.

RunPod's SSH proxy rejects ordinary non-PTY SCP with `Your SSH client doesn't
support PTY` while returning misleading exit status 0. Do not trust SCP exit
code alone. The successful fallback is PTY SSH + Base64 markers, local decode,
then remote/local SHA-256 comparison. Existing checkpoints were transferred
that way. Never declare transfer complete merely because `scp -O` said 0;
verify the destination file exists, size matches, and SHA matches.

## 12. Tests and correctness evidence

Authoritative RunPod focused suite:

```bash
python -m pytest \
  tests/test_crf.py \
  tests/test_pedal.py \
  tests/test_mobile_pedal.py \
  tests/test_diagnose_pedal_cache.py -q
```

Result: 44 passed.

Evidence collector focused tests: 2 passed. Copied bridge tests: 16 passed.
Focused Midee tests: 28 passed. Focused TypeScript typecheck passed.

Full Midee test command reached 662 passing tests but had five pre-test import
failures caused by pre-existing jsdom canvas and Windows
`file:///@solid-refresh` behavior; these are unrelated to the pedal changes.
Do not misreport the full suite as completely green.

Local Windows Python currently lacks PyTorch, so model test collection fails
with `ModuleNotFoundError: torch`. Do not install a heavyweight duplicate stack
merely for handoff; use the authoritative RunPod environment. A later local
collector-test attempt hit sandbox temp-directory permissions and created
cache directories; those exact generated directories were removed.

Atomic checkpoint save was tested. ONNX parity/output shape checks passed for
dynamic time lengths. Data identity checks passed as described above.

## 13. Research interpretation to preserve

Do not claim that a larger decoder is the answer. The strongest findings are:

1. routing supervised event confidence into structured transitions matters;
2. decoder changes recover evidence but do not create missing evidence;
3. flutter errors were not primarily an NMS-radius artifact;
4. marginal unfreezing/two-stream additions to a mature 24 ms model did not
   close the gap;
5. full-context biGRU materially improves pedal events, with most context gain
   by about one second;
6. 10 ms sampling helps localization, but the current small streaming model
   still does not meet event gates;
7. the active flatten control isolates whether frequency/register information
   plus much longer training is the missing acoustic factor;
8. state F1 is not evidence that event timing is solved;
9. Kong-level comparison must disclose that Kong uses a much larger dedicated
   end-to-end CRNN, full bidirectional context, 10 ms resolution, and about
   3.6 million training examples.

The publication angle is a constrained streaming-SOTA study with controlled
context, hop, frequency pooling, parameter allocation, and structured-decoder
ablations. Ordinary Viterbi/CRF/semi-Markov components alone are not novel.
Novelty must be framed around the causal fixed-lag problem, explicitly delayed
fast/context fusion, and controlled evidence about where the performance gap
comes from. Wait for flatten/extended results before making a strong claim.

## 14. Recovery playbook

### If status remains `WAITING_FOR_BASE`

Check base progress and GPU. If PID 90771 is alive and the log advances, do
nothing. At roughly 121 steps/min, estimate remaining time from `(12000-step)`.

### If status is `BASE_12K_EVAL`

Base training succeeded. Monitor the named eval logs and GPU. Do not start a
second trainer. The full-validation log ends in a JSON object when complete.

### If status is `EXTENDED_TRAIN`

Use:

```bash
tail -5 /workspace/pedal-offline-oracle-flatten-24ms-extended-formal.log
find /workspace/runs/pedal-offline-oracle-flatten-24ms-extended-from12k-20260803 \
  -maxdepth 1 -name 'step-*.torch' -printf '%f %s\n' | sort -V
```

The local step count is additional exposure. Add 12,000 for total-equivalent
step. Expected checkpoints: 6k, 12k, 18k, 24k, 30k, 36k, 42k, 48k.

### If status is `EXTENDED_EVAL`

Training completed. Monitor:

```text
/workspace/offline-oracle-flatten-24ms-extended-step*-eval.log
/workspace/offline-oracle-flatten-24ms-extended-threshold-*.log
/workspace/offline-oracle-flatten-24ms-extended-full-validation.log
```

### If status is `FAILED`

Read the supervisor log and the stage-specific log. Do not delete any run.
The training script refuses to overwrite existing output directories by
default. If a partial extended stage must resume, select the latest atomically
saved checkpoint, start a **new** continuation output directory with
`INIT_CHECKPOINT`, document another optimizer reset, and maintain cumulative
exposure explicitly. Do not set `ALLOW_EXISTING_OUTPUT_DIR=true` casually.

### If the pod restarted

All listed PIDs are stale. Inspect files first. Check status, completion marker,
checkpoints, and logs. Relaunch only the missing stage. Do not blindly rerun the
entire orchestrator if its output directory already exists.

## 15. Ordered next actions for the replacement agent

1. Read this entire file before running commands.
2. Check `/workspace/pedal-flatten-extended.status`, relevant logs, process
   table, and GPU.
3. Let PID 91721 work autonomously if healthy.
4. When 12k base evaluation lands, record its selected threshold and full
   headline `state/down50/Kong-onset+offset/strict50` metrics plus the separate
   up50 diagnostic in `PEDAL_RESEARCH_MASTER.md` and this handoff. Attach the
   exact checkpoint, threshold, decoder, prediction-cache, and result path.
5. Apply the matched 12k architecture rule: compare 12k flatten only to the
   independently calibrated 12k pooled oracle. “Flatten worked” requires
   down improvement >=0.020, strict50 degradation <=0.010, and state >=0.88.
6. When extended evaluation lands, compare it primarily to 12k flatten to
   answer whether more exposure helped this architecture. Do not attribute an
   extended-vs-pooled-12k difference to flattening. If an architecture claim is
   desired, schedule a pooled 24 ms extension with matched exposure tomorrow.
7. Compare only our down50 and `pedal_event_onset_offset` with Kong's 0.9186
   and 0.8658, respectively, and state every dataset/model/context/exposure
   caveat in section 8D. Never place up50 or strict50 beside a Kong value as a
   like-for-like result.
8. Decide whether longer training materially helped. Examine validation trends,
   not final minibatch loss alone.
9. Download flatten base/extended checkpoints and all compact artifacts to the
   local checkpoint/handoff directories. Verify remote/local SHA-256.
10. Update `models/pedal_checkpoints/CHECKPOINTS.md` with origins, sizes, hashes,
   architecture, and selected threshold.
11. Generate/rerun `collect_pedal_evidence.py` after final docs/source are in
   their desired state. The previously queued collector waits on the frozen
   old supervisor and should not be trusted as final without inspection.
12. Build one final non-public handoff archive containing selected checkpoints,
    ONNX where applicable, compact JSONs, configs, logs, manifests, and hashes.
13. Transfer and verify that archive locally.
14. Update `PEDAL_RESEARCH_MASTER.md` into a final coherent report; remove
    language saying completed items are pending.
15. Copy final source/docs and compact aggregate metrics into the code-only
    publication worktree, commit, and push only the existing
    `Cosxin/iamusica_training:cosxin/pedal-reproduction` branch. Do not create
    an upstream PR. Do not publish binaries/raw per-recording data without
    explicit user approval.
16. Leave the old 10 ms pooled control postponed unless the user explicitly
    wants it tomorrow. The user asked to spend this session on at least six
    hours of flatten training.
17. Once all remote artifacts are durable locally, terminate the intentionally
    stopped old supervisor PID 85582 and old waiting collector PID 87198. Never
    resume PID 85582.
18. If the one selected offline paper candidate passes down >=0.75 and strict50
    >=0.70 on full validation, freeze its configuration and authorize the one
    test evaluation even though it cannot deploy. Do not test both 12k and
    extended variants. If it fails, leave test sealed.
19. Do not deploy to the Pi unless a causal candidate passes all gates. If none passes,
    document `deploy=false`, preserve the old note-only service, and consider
    the deployment branch correctly stopped.
20. Perform the completion audit below before telling the user the pod is safe
    to terminate.

## 16. Completion audit and pod termination rule

Do not say `SAFE TO TERMINATE POD` until every applicable item is proven:

- base 12k flatten checkpoint exists, loads, and is locally hash-verified;
- base 12k locked32 calibration/full validation is complete and local;
- extended training ran at least six measured hours or reached the planned
  48k additional steps without failure;
- extended milestone selection/calibration/full validation is complete;
- selected extended checkpoint(s) are local and hash-verified;
- aggregate JSONs, configs, relevant logs, ONNX artifacts if exported, and
  evidence manifest are local and hash-verified;
- final report accurately distinguishes validation/test and causal/offline;
- test remains sealed unless gates passed;
- Pi deployment state matches the gate decision;
- scoped reproducibility source/docs are pushed and draft PR state verified;
- no important result exists only on `/workspace`;
- the final local archive can be enumerated and its hash is recorded.

If a control is intentionally postponed by the user, state that explicitly; do
not pretend it completed. The original broad goal is not complete merely
because GPU credits are nearly exhausted.

## 17. Files worth reading after this handoff

Only after reading this file, use these as implementation detail:

```text
PEDAL_RESEARCH_MASTER.md
runpod_pedal_flatten_extended_e2e.sh
runpod_pedal_oracle_controls_e2e.sh
runpod_pedal_offline_oracle_flatten_24ms.sh
runpod_pedal_offline_oracle_pooled_10ms.sh
6_train_mobile_pedal_regression.py
4_eval_pedal.py
collect_pedal_evidence.py
ov_piano/models/mobile_pedal.py
ov_piano/crf.py
ov_piano/pedal.py
diagnose_pedal_cache.py
models/pedal_checkpoints/CHECKPOINTS.md
```

The user specifically requested a zero-context handoff. If new work changes
processes, paths, metrics, hashes, or decisions, update this document before
ending the session.
