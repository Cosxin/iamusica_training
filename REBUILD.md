# Rebuild recipe — key-up release head for the LED product (2026-07-30)

Everything needed to reproduce the **key-up frame head** (the LED model) from the
shipped O&V baseline. Full results/analysis: `EVAL_FINDINGS_2026-07-30.md`.
Trained checkpoint committed at `models/keyup_framehead_step12000.torch`
(onset F1 frozen at baseline 0.9675; key-up note-w-offset 0.605 clean / 0.587 SBC).

## The idea

- LED needs **key-up** offsets (finger off key), NOT pedal-extended sustain.
- Train ONLY the frame head (`TRAINABLE_COMPONENTS=frame`) on a **key-up** roll
  (`MIDI_SUS_EXTEND=False`), warm-started from the shipped baseline, so onsets +
  velocity + stem stay frozen at baseline quality.
- Decoder: onset -> LED on (re-trigger every onset); frame prob < thresh -> LED off.
  Two fixed thresholds, no per-song tuning. Replaces the old energy-decay heuristic.

## Prereqs

- torch 2.2 / cu121 env, MAESTRO v3 extracted at `$MAESTRO`.
- Clean mel HDF5 already built (0a, no codec):
  `MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5`
- Shipped baseline ckpt: `assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch`

## 1. Build the key-up roll (roll only, no mel recompute)

```bash
# raw MIDI roll with sustain OFF -> key-up note-offs
python 0a_maestro_to_hdf5mel.py MAESTRO_INPATH=$MAESTRO OUTPUT_DIR=h5-keyup \
    IGNORE_MEL=True MIDI_SUS_EXTEND=False
# GOTCHA: IGNORE_MEL skips 0a's roll->mel length alignment, so the standalone
# roll's per-file frame counts differ from the mel -> trainer asserts
# "Unequal data indexes". Fix: align each file's roll to the mel length:
python align_roll_to_mel.py \
    <clean_mel>.h5 \
    h5-keyup/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=False.h5 \
    h5-keyup/keyup_roll_ALIGNED.h5
mv h5-keyup/keyup_roll_ALIGNED.h5 \
   h5-keyup/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=False.h5
```

## 2. Train the key-up frame head (frozen baseline)

```bash
python 1_train_onsets_velocities.py \
  SNAPSHOT_INPATH=assets/OnsetsAndVelocities_2023_03_04_09_53_53.289step=43500_f1=0.9675__0.9480.torch \
  MAESTRO_PATH=$MAESTRO \
  HDF5_MEL_PATH=<clean_mel>.h5 \
  HDF5_ROLL_PATH=h5-keyup/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=False.h5 \
  DEVICE=cuda ALLOW_PARTIAL_HDF5=True \
  OUTPUT_DIR=runs/framehead_keyup \
  ENABLE_FRAME_HEAD=True TRAINABLE_COMPONENTS=frame \
  LR_MAX=0.001 MAX_STEPS=12000 XV_EVERY=2000
```
`TRAINABLE_COMPONENTS=frame` => optimizer holds only `frame_stage` params and the
loss is frame-only (verified `1_train_onsets_velocities.py` ~line 520). Onset stem
runs under `no_grad` -> fast (~0.4 s/step, ~80 min on A5000). Onset F1 in the
checkpoint names stays pinned at baseline, confirming the freeze.

## 3. Evaluate against KEY-UP GT

`eval_offset.py` has `KEYUP_GT` (default True): a `KeyUpGtLoader` runs the MIDI
state machine with `sus_thresh=ten_thresh=127` so pedals never register ->
offsets = key-up. Without it you'd score against pedal GT and the key-up model
looks artificially bad (~0.25).

```bash
python eval_offset.py SNAPSHOT_INPATH=<keyup_ckpt> MAESTRO_PATH=$MAESTRO \
  HDF5_MEL_PATH=<clean_mel>.h5 \
  HDF5_ROLL_PATH=h5-keyup/MAESTROv3_roll_quant=0.024_midivals=128_extendsus=False.h5 \
  KEYUP_GT=True DATASET_VARIANT=clean RESULTS_JSON=out.json
# repeated-note / IOI resolution (the product metric):
python eval_repeats.py SNAPSHOT_INPATH=<keyup_ckpt> MAESTRO_PATH=$MAESTRO \
  HDF5_MEL_PATH=<clean_mel>.h5 HDF5_ROLL_PATH=<any_aligned_roll>.h5 \
  DATASET_VARIANT=clean LIMIT=40 RESULTS_JSON=repeats.json
```

## Key results (see EVAL_FINDINGS_2026-07-30.md, results/)

- key-up head vs KEY-UP GT: **0.605** clean / **0.587** SBC (onset 0.9674, frozen).
- old pedal head vs KEY-UP GT: 0.234 (why the retrain mattered).
- repeat resolution: clean-segment **0.859** ~= onset-retrig 0.861 (no eaten repeats).
- median key-hold ~86 ms -> bridge should enforce a min LED on-time (~80-120 ms).

## Abandoned / notes

- **Pedal head**: binary frame head caps at ~0.55-0.58 vs pedal GT (0.545 frozen /
  0