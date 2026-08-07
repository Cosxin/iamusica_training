# NIME 2027 paper plan

Date written: 2026-08-03. Target venue: NIME 2027 (New Interfaces for Musical
Expression), 22-25 June 2027, Devinci Higher Education, Paris, France.

## IMPORTANT: date correction

**NIME 2026 already happened.** Title/abstract deadline was 5 Feb 2026, final
submission 12 Feb 2026, conference ran 24-26 June 2026 — all in the past as of
today (2026-08-03). There is no 2026 submission to target. This plan targets
**NIME 2027**.

**NIME 2027 dates are confirmed for the conference itself (22-25 June 2027)
but the Call for Papers / submission deadlines are NOT yet published.** The
schedule below assumes NIME 2027 follows the same ~4.5-month lead pattern as
2026:

| Milestone | NIME 2026 (actual) | NIME 2027 (**estimated, unconfirmed**) |
|---|---|---|
| Title/abstract deadline | 5 Feb 2026 | ~early Feb 2027 |
| Final submission deadline | 12 Feb 2026 | ~mid Feb 2027 |
| Acceptance notification | 3 Apr 2026 | ~early Apr 2027 |
| Camera-ready deadline | 30 Apr 2026 | ~late Apr 2027 |
| Conference | 24-26 Jun 2026 | **22-25 Jun 2027 (confirmed)** |

**Action item: re-check nime2027.org / nime.org for the actual CFP once it
posts (historically a few months before the deadline, so likely late 2026),
and correct this table before committing to hard internal deadlines.**

---

## Why NIME over the MIR venues previously considered

ISMIR / EUSIPCO / TASLP judge primarily on transcription accuracy against
benchmarks (Kong et al., etc.) — the pedal work never cleared those gates, and
per the current project decision, pedal research is being dropped if F1 doesn't
improve. NIME's core subject is exactly what remains: a **built physical
interface** (Bluetooth audio -> Pi 4 -> per-key LED). NIME explicitly welcomes
systems papers evaluated on design rationale, engineering tradeoffs, and
demonstrated function — not SOTA-beating benchmarks. This is a better fit for
where the project actually is.

---

## CRITICAL DEPENDENCY: patent sequencing

This is not a side note — it gates the writing timeline.

- Double-blind CMT submission (Feb 2027 est.) is confidential peer review;
  reasonable assumption it does not itself constitute public disclosure — but
  this assumption should be **confirmed with patent counsel**, not taken on
  faith.
- **Camera-ready publication (~Apr 2027 est., if accepted) is genuine public
  disclosure** — full author names, full paper, indexed and citable.
- Separately, pre-existing public disclosures already exist (closed-but-public
  PRs on `andres-fr/iamusica_training` and `aayushdutt/midee`, and the public
  `Cosxin/iamusece_training` fork) — see prior conversation record. These
  already affect the novelty clock for absolute-novelty jurisdictions
  regardless of this paper.
- **Hard rule for this plan: file the provisional patent application before
  camera-ready, not after.** Ideally before final submission, to avoid relying
  on the "double-blind = not disclosure" assumption at all.

---

## Scope: what the paper is about

**In scope** (matches the current, working, deployed system):
- Acoustic (non-MIDI) piano audio -> Bluetooth -> Raspberry Pi 4 -> per-key LED
- Real-time streaming onset + velocity detection (O&V architecture)
- The key-up/release head: the actual technical contribution story —
  energy-decay heuristic tried and failed (needed per-song retuning) ->
  replaced with a trained release head -> fixes repeated-note merging
- Streaming latency engineering: bounded lookahead, receptive-field analysis,
  hop/past context tuning for real-time inference on commodity embedded
  hardware (the `ov-receptive-field` work: RTF 0.35 -> 0.18, 2x headroom)
- System architecture rationale: why Bluetooth audio (vs wired mic, vs MIDI)
  decouples the sound source from the device

**Out of scope:**
- Pedal transcription research (separate track, gated on its own decision)
- Any claim of beating published transcription SOTA (we don't, and that's not
  the point of this venue)

---

## Draft structure (Long paper, 6000 words, NIME 2-column LaTeX template)

1. **Introduction / motivation** — existing LED key-guide products require
   MIDI/digital pianos (cite Piano LED Plus's explicit incompatibility with
   acoustic pianos, already verified). Acoustic pianos have no equivalent.
   State the gap plainly.
2. **Related work** — MIDI-based LED guides (Piano LED Plus, i-Piano,
   PartyKeys, open-source `Piano-LED-Visualizer`); monophonic audio-to-LED
   (tuners, `piano-visualizer`'s FFT mic mode — confirmed monophonic,
   melody-only, no chords, no Bluetooth); neural AMT literature (O&V, Kong,
   Mobile-AMT) as the acoustic-transcription lineage this builds on.
3. **System architecture** — full pipeline diagram: audio source -> Bluetooth
   -> Pi 4 mel/inference -> onset+velocity+key-up heads -> decoder -> LED
   driver. State the ~2.5s streaming context budget and why.
4. **The key-up problem and solution** — this is the paper's spine. State the
   product failure mode (eaten repeated notes), the naive fix (energy decay)
   and why it failed empirically (needed per-song threshold retuning), the
   learned fix (frozen O&V backbone + trained release head via
   `TRAINABLE_COMPONENTS=frame`), and why key-up (not pedal-extended) is the
   correct ground truth for a keyboard-mirroring display.
5. **Evaluation** — three tiers, honest about what each does and doesn't show:
   - Quantitative: onset F1 (0.9675/0.9677), key-up note-with-offset F1
     (0.605 clean / 0.587 SBC), repeat re-trigger resolution (0.859
     clean-segment) — with the explicit framing from `EVAL_FINDINGS_2026-07-30`
     that note-with-offset F1 is the wrong target and repeat resolution is the
     real product metric.
   - Systems: Pi 4 latency/thermal benchmarks (already measured — RTF,
     compute headroom, streaming hop/past tuning).
   - Observability finding: 64.5% of key-ups occur while the sustain pedal is
     down (measured this session on MAESTRO validation, 639k notes) — explains
     per-piece threshold sensitivity, a genuine empirical contribution.
6. **Discussion / limitations** — the deployment-window caveat (offline
   300s-context eval vs streaming ~2.5s reality; note the receptive-field
   study's 99.2-100% peak-agreement finding, and flag that a direct F1 at
   deployment window has not yet been measured — see open items below).
7. **Conclusion.**

---

## Open work items before drafting can start

Ranked by what most changes the paper's strength, cheapest first:

1. **Deployment-window onset F1** — re-run `2_eval_onsets_velocities.py` with
   `INFERENCE_CHUNK_SIZE`/`INFERENCE_CHUNK_OVERLAP` matched to the actual
   bridge's ~2.5s window (1.5s past + 0.5s target + 0.5s lookahead), instead of
   the current 300s offline number. No training, one eval run. Closes the gap
   flagged in the last venue-comparison discussion.
2. **Key-up-vs-pedal-state stratification** — bucket existing key-up
   detections by ground-truth pedal state at eval time (~30 lines, reuses
   `GtLoaderMaestro.get_midi_eventdata()`, no training). Would upgrade the
   64.5%-occlusion finding from a suggestive statistic into a real result
   (e.g. "recall 0.80 pedal-up vs 0.42 pedal-down").
3. **Pedal-extended-GT comparison point** — run the same key-up checkpoint
   with `KEYUP_GT=False` to get the standard-convention number alongside 0.605,
   giving readers a calibration point against the wider note-with-offset
   literature.
4. **A user-facing demo/qualitative section** — NIME reviewers respond well to
   video/demo evidence of the system working live on a real piano. Doesn't
   need to be a formal user study; a described demo session with screenshots
   or a linked video would materially strengthen the systems-paper case.
5. **Provisional patent filed** — see dependency above. Not a writing task,
   but blocks the safe camera-ready timeline.

---

## Phased schedule (from today, assuming estimated 2027 dates)

| Phase | Window (est.) | Work |
|---|---|---|
| 1. IP sequencing | Aug-Sep 2026 | Consult patent counsel; file provisional |
| 2. Close open items 1-3 above | Sep-Oct 2026 | Cheap evals, no training required |
| 3. Demo/qualitative evidence | Oct-Nov 2026 | Record real-piano demo session |
| 4. First draft | Nov-Dec 2026 | Full draft against structure above |
| 5. Internal review + anonymization pass | Dec 2026-Jan 2027 | Check against NIME's strict anonymization rules (max 125-char abstract, no self-identifying references, no linked project names) |
| 6. Confirm real NIME 2027 CFP dates | as soon as published (likely late 2026) | Replace estimated dates in this doc |
| 7. Title/abstract submission | ~Feb 2027 (est.) | CMT |
| 8. Final PDF submission | ~Feb 2027 (est.), ~1 week later | CMT |
| 9. Camera-ready (if accepted) | ~Apr 2027 (est.) | **Provisional must be filed before this point** |
| 10. Conference | 22-25 Jun 2027 (confirmed) | — |

This is comfortable runway relative to today (six-plus months to submission),
consistent with the "plenty of time" framing — the only correction was the
target year.
