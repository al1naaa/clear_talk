# ClearTalk multilingual - round 4: the large model in noise

Rounds 1-2 are closed (tests A and B opened once each). Round 3 (the small model with Kazakh broadcast speech) was
not trained, and its test set C was never opened; it is used here as test D. This round continues training the
large model of round 2 so that it copes better with background noise.

Goal: the user asked on 2026-10-15 that the model work well "on short and long recordings, with noise and without,
mixed speech and so on" ("модель должна хорошо работать и на коротких и на длинных аудиозаписях, и с шумом и без,
смешанная речь и тд"). The robustness check on validation (`data/ml/results/robustness.json`) found the app's
weakest points: babble (other people talking) - at 10 dB SNR the error on mixed recordings rises from 8.2% to
38.7%, mostly extra words from the background voices; and white noise - 21.7% at 10 dB, mostly wrong words.
Method: the round-2 decoder LoRA, trained further on the same training examples with background noise added to
half of them. The encoder (the part that hears) stays frozen, so the model can learn to ignore background voices
and to be careful in noise, not to hear better.

## Checklist

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | done | no new speech data; new: the noise sources (synthetic white and pink noise; babble from training-role utterances) |
| 2 | EDA | done | the robustness check on validation (see Goal) |
| 3 | Problem contract | approved-gate | see "Problem contract" below |
| 4 | Data cleaning | done | rules of round 1, unchanged |
| 5 | Data engineering | done | noise augmentation (cleartalk_ml/noise.py): 50% of examples, white/pink 0-20 dB or babble of 3-5 talkers 3-20 dB, seeded per example |
| 6 | Train / val / test split | done | roles of round 1; validation noise and test noise use talkers disjoint from the training noise; test D frozen (2,694 units) |
| 7 | Feature engineering | done | unchanged (128-band log-mel of the noisy audio) |
| 8 | Preprocessing | done | unchanged targets: the transcript of the main speech only |
| 9 | Baseline model | done | the round-2 large model: one-pass WER 26.3% on the noisy robustness validation recordings (white 20/10 dB, babble 10/5 dB) |
| 10 | Model training | done | decoder LoRA continued from round-2 step 600 with noise on half the examples; 450 steps (0.5 epoch); Figure 10 |
| 11 | Hyperparameter tuning | done | learning rate 1e-4 kept from round 2; the checkpoint (150 / 300 / 450) chosen on noisy validation among those meeting the clean guard |
| 12 | Model evaluation | done | not promoted: step 150 26.7% vs 26.3% (+0.4 points, 95% -0.6 … +1.6); the round-2 model stays (Figure 12) |
| 13 | Error analysis | done | hallucination audit, background turns, loops, region-first and overlap transcripts; five app options measured (see Gate C) |
| 14 | Final test | done | test D opened once (app mode): all 26.8% → 20.8% (-6.0 points, 95% -7.7 … -4.7); babble 10 dB 34.4% → 21.0%; clean 13.1% → 13.0% (Figure 14) |
| 15 | Deployment | done | profile options set by the checks; the app run in the browser (café demo, background list); CPU 41-s demo ≈ 2.7 min |
| 16 | Monitoring + retraining plan | done | see "Monitoring and retraining plan" below |

## Delegation (2026-10-15)

The user asked for the project to be improved further, explicitly for noisy recordings ("давай все улучшай еще …
и с шумом и без"). Her standing rule applies: pass the pipeline stops yourself, the final test included, and write
everything down for her review. No new data were downloaded for this round (the offer of the ESC-10 noise clips,
about 140 MB, is still open and unanswered; they are not used).

## Problem contract (step 3) - what changes against round 2

- **Objective:** lower WER of the large model in noise (white noise and babble) with clean speech no worse.
- **Starting point:** the round-2 adapter (`models/multilingual/runs/large-lr1.5e-4/step-00600`), trained further
  (warm start), learning rate 1e-4, half an epoch (450 optimizer steps of 16 examples), evaluation every 150 steps.
- **Baselines:** the round-2 model (what this round must beat); for reference the untouched starting model.
- **Validation:** one-pass transcription through the app's decoding of the robustness recordings
  (`cleartalk_ml.robustness`: 100 mixed recordings and 150 short phrases) in the noisy conditions white 20 dB,
  white 10 dB, babble 10 dB and babble 5 dB; the clean validation sets of round 1 as guards.
- **Pre-registered rules (written before training):**
  1. *Checkpoint* - of the evaluated checkpoints (steps 150, 300, 450), the one with the lowest pooled WER over the
     noisy robustness recordings among those that meet the clean guard of rule 2(b); if none meets it, round 4
     keeps the round-2 model.
  2. *Promotion* - the new model replaces the round-2 model in the app (`cleartalk-large`) if on validation (a) its
     pooled WER over the noisy robustness recordings is at least 10% lower, relative, than the round-2 model's,
     with the 95% bootstrap interval of the difference below zero, and (b) on the clean sets - mixed recordings in
     one pass, FLEURS Kazakh, Russian and English, KSC2, Golos (language given) - none is more than 0.5 points worse.
  3. *Report, not a rule:* every noise condition and every clean set, before and after.
- **Test D (frozen before any training of this round):** test C - 600 KSC2 utterances (100 per source), 100 FLEURS
  Kazakh, 80 FLEURS Russian, 18 FLEURS English, 100 Golos - each clean, with white noise 10 dB and with babble 10 dB
  (2,694 units). No mixed recordings (no unused English test audio is left); mixed recordings in noise are
  validation-only evidence. Opened once, after the decisions above.
- **Known caveats:** noise is added, not recorded (real rooms, microphones and street noise were not available);
  babble is read speech of the same corpora; SNR is measured against the whole recording.

## Gate A and Gate B report (steps 1-8)

- **What was done.** The app was measured on validation recordings with added noise (figure 02). A noise generator
  for training was written and checked: half of the examples get white, pink or babble noise at a seeded SNR, and
  the babble talkers of training, validation and test are three disjoint sets of training-role utterances
  (figure 07). Test D was frozen.
- **What was found.** Babble is the hardest condition (extra words from the background voices), then white noise
  (wrong words). Long conversations behave like the mixed recordings; short phrases are the weakest clean set.
- **Decisions and why.** Warm start from the round-2 adapter (the clean behaviour is already good and must be kept);
  half of the examples noisy (the other half keeps the clean behaviour); half an epoch at a lower learning rate
  (a refinement, not a new model); encoder frozen (6 GB of graphics memory).
- **Known weaknesses kept in view.** Added, not recorded, noise; read speech as babble; test D has no mixed recordings.
- **What comes next (steps 9-12).** Training, then the checkpoint and promotion rules on validation.
- **The question - "may I continue to training?"** Delegated by the user on 2026-10-15 (see "Delegation").

- Gate A: approved 2026-10-15 (round 4; delegated: the user asked for the model to be improved for noisy recordings and for every stop to be passed and recorded)

Model rationale:
- traits: the speech-to-text task of rounds 1-2; 14,400 training examples per epoch, half of them with added noise; inputs are 30 s windows of 128-band log-mel
- baseline: the round-2 model through the app's decoding (what this round must beat); dummy = empty transcript (100% WER)
- candidates: the round-2 decoder LoRA (rank 16) trained further with noise augmentation, learning rate 1e-4, 450 steps, encoder frozen
- ruled out: LoRA in the encoder (does not fit 6 GB); a separate denoising model before recognition (needs a download, and enhancement artefacts often hurt Whisper); training from the starting model again (would throw away round 2)
- metric: pooled one-pass WER over the noisy robustness validation recordings (primary); clean validation sets as guards

- Gate B: approved 2026-10-15 (round 4; delegated as above; recorded after figures 02 and 07 were written)

## Gate C report (steps 9-13)

- **What was done.** (9) Baseline: the round-2 large model through the app's decoding. (10) Its decoder LoRA trained
  further for 450 steps with noise on half of the examples. (11) The checkpoint chosen on noisy validation among those
  meeting the clean guard. (12) The promotion rule applied. (13) Error analysis of the app's validation transcripts
  (hallucination audit, background turns, repetition loops, region-first and overlap transcripts), and from it five
  app options, each measured on validation against the app as it then stood, with a rule written before measuring.
  Test D was not touched.
- **What was found.**
  - Training in noise did not help: the chosen checkpoint (step 150) gave
    26.3% → 26.7% on the noisy robustness
    recordings (+0.4 points, 95% -0.6 … +1.6). The round-2 model stays.
  - The commonest invented text in noise was not inside the speech but around it: short turns where only background
    talkers were audible got a speaker label of their own and a few invented words; one recogniser loop wrote "Senate"
    213 times; and in long conversations the speaker labels cut sentences, so words at the cuts were missed or guessed.
  - Adopted, in order: repetition loops cut back to three occurrences (no clean set changed); background turns set
    aside (25.2% → 20.7%, -4.5 points, 95% -5.5 … -3.5; the half not used to
    choose its thresholds -4.6 points, 95% -6.3 … -3.2); region-first transcription, third variant
    (20.7% → 17.8%, -2.9 points, 95% -3.8 … -2.0; clean conversations
    11.8% → 8.2%).
  - Not adopted: region-first first variant (clean mixed recordings +1.3
    points), second variant (turns holding two people's utterances 2.2%, limit 2 points);
    overlap separation first variant (one voice written twice: robustness recordings 17.8% → 30.0%);
    the narrower cross-check (-0.2 points, 95% -0.6 … +0.3).
  - Overlap separation, second variant: not adopted: on the held-out overlap recordings cpWER 33.7% → 16.0% (-17.7 points, 95% -25.0 … -11.5), but robustness recordings 17.8% → 29.0%, extra words 2.1% → 12.4%; guards mixed 8.2% → 8.5%, long 8.2% → 11.6%, short 11.8% → 11.8%.
  - The app now, against the app when this round began (validation, language detected per turn): babble 10 dB
    mixed recordings 38.7% → 22.4%, conversations 30.2% → 14.9%;
    clean conversations 11.8% → 8.2%. Short Kazakh phrases got slightly worse in some noise
    conditions (white 20 dB 24.7% → 25.9%, babble 5 dB
    44.8% → 48.2%); the rules guarded clean recordings only.
- **Decisions and why.** Each option was adopted or not by its rule as written; the profile options now on:
  background, region (models/multilingual/profile.json), plus the loop cut in every mode. Three variants
  were tried for region-first and two for overlap separation; every variant, its cause of failure and its rule are in
  "Decisions", and none was changed after its own measurement.
- **What comes next (steps 14-16).** The one-time final test on test D, as decided before opening it: the app as this
  round began against the app with everything adopted, language detected automatically, on 898 never-seen utterances
  clean, with white noise at 10 dB and with babble at 10 dB; then the deployment check and the monitoring plan.
- **The question - "may I open test D?"** Delegated in advance by the user (see "Delegation"): the stops are passed
  without her, the final test included, and everything is written down for her review.

- Gate C: approved 2026-10-16 (round 4; delegated as above)

## Final test (step 14)

Test D was opened once on 2026-10-16, after Gate C, in the app mode decided before opening it (the noise-trained model
was not promoted): the app as round 4 began ("auto-before": per-turn, no loop cut, no background stretches set aside,
no region-first transcription) against the final app ("auto+background+region" with the loop cut), language detected
automatically, 2,694 units (898 utterances × clean / white noise 10 dB / babble 10 dB). Results:
`data/ml/results/test-final-D.json`; transcripts: `data/ml/results/hyp-testD/`.

| Condition | before | after | difference (95% interval) |
|---|---|---|---|
| clean | 13.1% | 13.0% | -0.1 points, 95% -0.3 … +0.0 |
| white noise 10 dB | 32.9% | 28.3% | -4.5 points, 95% -9.2 … -1.6 |
| babble 10 dB | 34.4% | 21.0% | -13.3 points, 95% -15.5 … -11.2 |
| all | 26.8% | 20.8% | -6.0 points, 95% -7.7 … -4.7 |

- Extra words per reference word 6.9% → 2.5%.
- Largest changes, all in babble: Kazakh read sentences 57.9% → 21.9%, Russian read sentences 35.9% → 12.5%, Russian short phrases 44.6% → 33.5%, KSC2 28.1% → 22.4%, English 18.4% → 9.7%.
- Clean sets: unchanged on FLEURS and Golos, KSC2 15.1% → 14.9%: the protections do no harm where there is no noise.
- Test D holds one person per recording, so it measures the protections on single utterances; the gains in conversations
  (region-first transcription) rest on validation only.

## Deployment (step 15)

- The options are read from `models/multilingual/profile.json` (written by the checks): `transcribe_by_region: true`,
  `hide_background_turns: true`, `separate_overlaps: false`, `cross_check: false`; the loop cut is part of the code.
  The packaged models archive carries this profile; a computer that already has the models needs only the new
  `profile.json` and the new code.
- Run in the browser on the Acer: the café version of the demo (`data/demo/conversation_cafe.wav`, the six FLEURS
  sentences with babble of other FLEURS training talkers at 10 dB) gives six clean turns, speakers numbered 1-6; the
  app as round 4 began wrote one invented Kazakh sentence and cut two sentences (24.0% → 10.7% WER on this recording).
  On a babble 5 dB validation recording the list of set-aside stretches appeared under the transcript; one of the two
  was a garbled piece of a real sentence - the reason they are listed rather than deleted.
- CPU (the Mac's path): the 41-s demo takes about 2.7 minutes with 6 threads on the Acer's processor in both settings.

## Monitoring and retraining plan (step 16)

- **What to watch on real recordings** (check examples, `python -m cleartalk_ml.real`): WER of the app; the share of
  set-aside stretches that hold real words (open the list under the transcript); turns holding two people's words
  (region-first); repetition-loop flags.
- **When to act.** If set-aside stretches hold real words in more than 1 in 10 recordings, raise
  `ABOVE_FLOOR_DB` or switch `hide_background_turns` off in the profile (no retraining needed). If real conversations
  show turns with two people's words, switch `transcribe_by_region` off. Both are one line in `profile.json`.
- **Retraining.** Training in noise did not help with 450 steps of half-noisy examples of read speech; a retry needs
  real noisy conversational speech (not available offline here) and a new frozen test set, since test D is spent.
- **Overlap.** Separating simultaneous voices helps on real overlaps (cpWER 33.7% → 16.0% on validation) but invents
  words in ordinary and noisy conversations; a future attempt needs a better decision of when two people really talk
  at once (for example an overlap detector trained for it), measured with the same rule.

## Wrap-up (Gate D)

- Round 4 did not change the model and changed the app: three protections adopted on validation, each by a rule written
  before measuring; five options or variants measured and not adopted, each recorded with its cause. The one-time test
  confirmed the change on never-seen data: 26.8% → 20.8% overall, 34.4% → 21.0% in babble,
  clean unchanged.
- Test sets: A (round 1), B (round 2) and D (= test C with noise, round 4) are spent; no frozen test set is left.

- Gate D: approved 2026-10-16 (round 4 wrap-up; delegated as above)

## Decisions

- 2026-10-15 · Test D was built and frozen before any training of this round (fingerprint in `.guard_state.json`).
- 2026-10-15 · Rule for the narrower cross-check, written before it is measured (it will be measured after this round's decision, with the large model the app then uses): 'cross_check: kazakh' - a turn the detector sent to large-v3-turbo as English is taken over by our large model only if our model wrote Kazakh (Kazakh-only letters). Adopted if on the robustness validation recordings (mixed recordings and short phrases; clean, white noise 10 dB, babble 10 dB) the pooled WER is lower with the 95% bootstrap interval of the difference below zero, and English FLEURS validation through the pipeline is not more than 0.5 points worse. The broader version was measured on 2026-10-15 and not adopted (see the round-2 log).
- 2026-10-15 · Hallucination audit (validation transcripts, cleartalk_ml/hallucinations.py): the commonest invented text is (1) a short Kazakh phrase written as English after a confident wrong language decision (3 of 50 clean, 8 of 58 in babble 5 dB), and (2) words guessed at the edges of turns: in the 7-minute conversations 77 of 262 utterances (29%; 58% in babble) were cut into several turns where the speaker labels flipped, each piece was recognised on its own, and the recogniser missed or invented words at the cut (clean conversations: 3.4% extra words against 0.8% on the short mixed recordings). Repetition loops are rare.
- 2026-10-15 · Rule for 'transcribe_by_region', written before it is measured: the recogniser gets every stretch of continuous speech whole and its sentences are given to speakers by time. Adopted if on the robustness validation recordings (per-turn setting; mixed recordings, short phrases, conversations; every measured condition) (a) the pooled WER is lower with the 95% bootstrap interval below zero, (b) clean mixed recordings, clean short phrases and clean conversations are each not more than 0.5 points worse, and (c) on the clean mixed recordings and conversations the share of turns holding words of more than one utterance rises by no more than 2 points.
- 2026-10-15 · People talking at once. 100 validation recordings in which a second utterance starts while the first is still running (20-80% of the shorter one spoken over; cleartalk_ml/overlap.py). Recordings 0-49 (and mixed validation recordings 100-149 as stretches without overlap) are used only to set the overlap detector; 50-99 only to measure. On one Russian + Kazakh probe the mixture gave 34% WER and the separated voices 0% and 12%. The first detector (both separated tracks loud at once for ≥ 1 s) was rejected on the calibration recordings before any measurement: at that setting it found 25 of 50 overlaps and fired on 53 of 188 stretches without overlap (the separator often spreads one voice over both tracks). The detector used instead, set on the same calibration stretches before any measurement (the separated tracks were saved and measured offline): a stretch is separated when the speaker labels show two people talking at least 0.5 s each in it (30 of 48 stretches with overlap, 2 of 226 without; the other 18 overlaps the speaker labels did not see, and they stay as they are) and the quieter separated track's loud frames reach within 12 dB of the louder track's (all 30; one of the two others had -16.7 dB). Every sentence written from a separated track must also come from the louder track at that moment (else it is leakage of the other voice and is dropped), and if nothing survives the stretch is recognised as usual. Rule for 'separate_overlaps', written before it is measured: adopted if (a) on overlap recordings 50-99 the cpWER is lower than the current app pipeline's with the 95% bootstrap interval of the difference below zero; (b) on the robustness validation recordings (every condition) the pooled WER is not more than 0.3 points higher; (c) clean mixed recordings, clean short phrases and clean conversations are each not more than 0.5 points worse; (d) invented words (insertions per reference word) on the robustness recordings rise by no more than 0.2 points.
- 2026-10-15 · Background voices. In the app's validation transcripts with babble, many extra words come from short turns where only background talkers were audible: they pass speech detection, get a speaker label of their own and the recogniser writes a few invented words ("episode one of the first season"). A turn was counted as such when dropping it removes errors of its recording: 1,092 of 3,927 turns (clean 23, white 10 dB 98, babble 10 dB 525, babble 5 dB 436). They are short (median 1.5 s against 3.6 s), quieter than the recording's speech (median -7 dB against -1 dB) and barely above its background level (median 7.6 dB against 16 dB). Rule for setting them aside (backend/background.py), chosen on one half of the robustness validation recordings (split by a hash of the recording id) with the per-turn transcripts: a turn of at most 3 s whose loud frames are at least 5 dB below the recording's speech level and at most 10 dB above its background level (20th percentile of 50 ms frames). On the other half: 28.36% → 22.74% WER (-5.6 points, 95% interval -8.6 … -3.5), clean recordings unchanged (no clean turn set aside), babble 10 dB mixed 39.9% → 24.5%, conversations 32.1% → 22.1%. Set-aside turns are not deleted: the app lists them under the transcript, faded, and leaves them out of the text and the exports. Rule for 'hide_background_turns', written before the pipeline is measured with it: adopted if, against the app as it then stands, on all robustness validation recordings (every condition) the pooled WER is lower with the 95% interval below zero, the same holds on the half not used to choose the thresholds, and clean mixed recordings, clean short phrases and clean conversations are each not more than 0.2 points worse.
- 2026-10-15 · Repetition loops (backend/loops.py), fixed before measuring: a word or phrase of up to six words that follows itself more than three times in a row is cut back to three occurrences, and the turn is flagged for checking; it applies to every mode. Measured on the app's stored validation transcripts (a pure text step, so the stored transcripts give its exact effect): clean recordings unchanged (one turn changed, itself a loop: "я не знаю" six times); white 10 dB conversations 25.9% → 23.6%; babble 10 dB short English phrases 46.8% → 14.9% (one loop of 213 extra words: "Senate Senate …"); babble 5 dB mixed recordings 67.4% → 61.4%, short Russian phrases 76.4% → 67.4%. No set got worse. From here on every comparison reads both sides as the app now writes them (cleartalk_ml.evaluate.without_loops).
- 2026-10-15 · 'hide_background_turns' adopted (scripts/check_option.py background; data/ml/results/option-background.json). Against the per-turn app: all robustness validation recordings 25.2% → 20.7% (-4.5 points, 95% -5.5 … -3.5); the half not used to choose the thresholds 25.8% → 21.2% (-4.6, 95% -6.3 … -3.2); clean mixed recordings, clean short phrases and clean conversations unchanged (8.15%, 11.76%, 11.79%); by condition clean 10.3% → 10.3%, white 20 dB 12.4% → 12.4%, white 10 dB 22.7% → 22.6%, babble 10 dB 32.7% → 22.4%, babble 5 dB 55.9% → 41.6%.
- 2026-10-15 · Region-first transcription, first variant: not adopted. On all robustness validation recordings 25.2% → 19.2% (-6.0 points, 95% -7.3 … -4.7), clean conversations 11.8% → 9.5%, babble 10 dB conversations 30.2% → 18.2%, babble 10 dB mixed recordings 38.7% → 23.4%, and turns holding more than one utterance 0 → 9 of 593 (within the 2-point limit); but clean mixed recordings 8.2% → 9.5% (+1.3, limit +0.5), so rule (b) fails. Cause, read in the transcripts: the speech detector splits one sentence at breath pauses of a few hundred milliseconds, and each piece got its own language decision and recogniser ("Please treat the site with all of the dignity, solemn…" / "The tea and respect. It deserves." / "Do not make jokes …"); the per-turn pipeline does not, because speaker turns already close pauses up to 0.6 s. Second variant: stretches of speech are joined across pauses of at most 0.6 s (the same gap the speaker turns use) up to 24 s (backend/speakers.joined_regions). Rule for it, written before it is measured, the same as for the first: against the app as it then stands, (a) pooled WER on all robustness validation recordings lower with the 95% interval below zero, (b) clean mixed recordings, clean short phrases and clean conversations each not more than 0.5 points worse, (c) on the clean mixed recordings and conversations the share of turns holding more than one utterance rises by no more than 2 points. It is measured after the background check and before the overlap check.
- 2026-10-15 · Region-first, second variant: not adopted (data/ml/results/option-region-v2.json; transcripts kept as "auto+background+region-v2"). Against the app with background stretches set aside: all robustness validation recordings 20.7% → 18.1% (-2.7 points, 95% -3.6 … -1.6); clean mixed recordings 8.15% → 8.51%, clean short phrases unchanged, clean conversations 11.8% → 8.2% (guards met); but turns holding more than one utterance 0 of 677 → 12 of 545 (2.2%, limit 2 points), so rule (c) fails. The 12 turns join utterances of two different people with a pause of 0.3-0.6 s between them ("… positive comments" after a Kazakh sentence of another speaker).
- 2026-10-15 · Found while preparing the overlap check, before it was measured: as written, 'separate_overlaps' switched the whole recording to region-first transcription, so its check would have credited it with the gains (and the attribution errors) of an option that failed its own rule. Changed before any overlap measurement: the separator now works only on stretches of continuous speech in which the speaker labels show two people (each at least 0.5 s); where it finds two voices, their sentences replace the turns of that stretch; everything else is transcribed exactly as without the option. The overlap rule written above is unchanged. (The first overlap run was stopped after its baseline transcripts only; the candidate had not started.)
- 2026-10-15 · Region-first, third variant: a pause of at most 0.6 s is closed only when the speaker labels give the same person on both sides of it (backend/speakers.joined_regions with "same"), so utterances of two different people are no longer joined across a short pause, while one person's breath pauses still are. Rule, written before it is measured: the same three conditions (a)-(c) as for the first two variants. This is the last variant; if it fails, region-first transcription stays off.
- 2026-10-15 · Region-first, third variant: adopted ('transcribe_by_region'; data/ml/results/option-region.json). Against the app with background stretches set aside: all robustness validation recordings 20.7% → 17.8% (-2.9 points, 95% -3.8 … -2.0); clean mixed recordings 8.15% → 8.23%, clean short phrases unchanged (11.76%), clean conversations 11.8% → 8.2%; by condition clean 10.3% → 8.7%, white 20 dB 12.4% → 11.7%, white 10 dB 22.6% → 17.9%, babble 10 dB 22.4% → 18.6%, babble 5 dB 41.6% → 38.9%; turns holding more than one utterance 0 of 677 → 10 of 559 (1.8%, within 2 points).
- 2026-10-16 · The overlap check ran out of graphics memory on the 7-minute conversations with babble (the separator was run on very long stretches; memory spilled into shared memory, the run slowed about 30-fold overnight and stopped at babble 10 dB). Before re-running, without looking at any guard result: memory is released after each separation (no effect on the output), and stretches longer than 30 s are not separated (backend/overlap.MAX_SECONDS; every overlap recording is at most 20.4 s, so the overlap measurement is unaffected; a minute of "two people" is background chatter rather than an interruption). The candidate's conversation transcripts made without the limit were set aside (*.before-30s-limit.jsonl) and are redone; the rule is unchanged.
- 2026-10-16 · 'separate_overlaps', first variant: not adopted (data/ml/results/option-overlap-v1.json; transcripts kept as "auto+background+overlap-v1+region"). On the held-out overlap recordings the cpWER fell from 33.7% to 16.0% (-17.7 points, 95% -25.0 … -11.5; missing words 251 → 59), but on the robustness recordings the pooled WER rose from 17.8% to 30.0% and extra words from 2.1% to 13.3% of the reference words; clean conversations 8.2% → 14.0%. Cause, read in the transcripts: where one person spoke and the speaker labels showed two (quick turn-taking), the separator spread that one voice over both tracks and the same sentence was written twice, for two speakers ("Включи мне на смотрешке передачу рэн тв." for Speaker 1 and Speaker 4 at the same time).
- 2026-10-16 · 'separate_overlaps', second variant, fixed before it is measured: if the texts of the two separated tracks share at least half of the shorter one's words (in order), the stretch carries one voice twice and is recognised as usual (backend/overlap.same_words; two people talking at once do not say the same words). Rule unchanged ((a)-(d) above). Measured after the narrower cross-check, against the app as it then stands. This is the last overlap variant.
- 2026-10-16 · Narrower cross-check ('cross_check: kazakh'): not adopted (data/ml/results/option-kk.json). Against the app with background stretches set aside and region-first transcription: mixed recordings and short phrases in clean, white 10 dB and babble 10 dB 16.77% → 16.61% (-0.16 points, 95% -0.64 … +0.28, the interval includes zero); English FLEURS validation through the pipeline 6.06% → 6.12%.
- 2026-10-16 · 'separate_overlaps', second variant: not adopted (data/ml/results/option-overlap-v2.json; transcripts kept as "auto+background+overlap-v2+region"). Held-out overlap recordings: cpWER 33.7% → 16.0% (-17.7 points, 95% -25.0 … -11.5; 63 separated turns); robustness recordings 17.8% → 29.0% (limit +0.3), extra words 2.1% → 12.4% of the reference words (limit +0.2), clean conversations 8.2% → 11.6% (limit +0.5). The duplicate check removed part of the damage; the rest is mostly background talkers written as the "second voice" in noisy recordings and garbled copies of one voice. As announced, this was the last overlap variant: Conversation mode keeps not separating simultaneous voices. Final app options: background stretches set aside, region-first transcription (third variant), repetition loops cut.
- 2026-10-15 · Order of the remaining option checks (each against the options adopted before it, data/ml/results/app-options.json): region-first transcription (first variant), background, region-first (second variant), region-first (third variant), overlap (first variant), the narrower cross-check, overlap (second variant).
- 2026-10-15 · Use of test D if the noise-trained model is not promoted (written before test D is opened, while the selection was still running; the noisy validation WER of every checkpoint was already above the round-2 model's): test D then measures the app itself, language detected automatically, once: the app as it was when this round began (none of this round's protections; cleartalk_ml.evaluate "auto-before") against the app with everything adopted on validation in this round. Reported per condition (clean, white 10 dB, babble 10 dB) and per set with paired bootstrap intervals, and the rate of extra words; no decision depends on it. Test D holds single utterances only (no mixed recordings, no conversations), so it measures the protections where one person speaks.

## Figures
- Figure 02_app_in_noise_validation.png: The app (language detected per turn, both decoding options on) on validation recordings with added noise. Mixed recordings: clean 8.2%, white-20 11.5%, white-10 21.7%, babble-10 38.7%, babble-5 67.4%; 7-minute conversations: clean 11.8%, white 10 dB 25.9%, babble 10 dB 30.2%. In babble most of the added errors are extra words (the background voices are written down); in white noise they are wrong words. Short phrases are the weakest already when clean (Kazakh and Russian about 19-21%): mostly names and joined words, and a few Kazakh phrases detected as English with confidence.
- Figure 07_noisy_training_example.png: One mixed training recording as the model sees it: clean, with white noise and with babble at 10 dB SNR. Of the 14,400 training examples 7,176 get noise (white 1,498, pink 1,392, babble 4,286); SNR uniform in 0-20 dB (white, pink) or 3-20 dB (babble of 3-5 talkers). Babble comes from 3,000 training utterances; the validation noise and the test noise use other talkers. The target is always the transcript of the main speech.
- Figure 10_learning_curves_noise.png: Greedy decoding on fixed validation subsets; step 0 is the round-2 model. Mixed recordings with babble at 10 dB: 23.2% -> 22.8% at step 450; the same recordings clean: 9.2% -> 10.0%. FLEURS kk/ru/en: 10.5% -> 10.7%, 4.7% -> 4.7%, 10.9% -> 11.1%. Validation loss clean 0.387 -> 0.395, babble 0.630 -> 0.619.
- Figure 12_noise_decision.png: Pooled one-pass WER over the noisy robustness validation recordings: round-2 model 26.3%; checkpoints step 150 26.7%, step 300 29.0%, step 450 26.6%. Chosen: step 150 (+2% relative, difference +0.4 points, 95% -0.6 to +1.6); by condition white-20 14.6% -> 14.4%, white-10 28.8% -> 31.3%, babble-10 21.7% -> 21.1%, babble-5 39.9% -> 39.7%. Clean guard: mixed 9.8% -> 9.9%, fleurs-kk 10.6% -> 10.4%, fleurs-ru 5.3% -> 5.4%, fleurs-en 6.9% -> 6.7%, ksc2-kk 10.3% -> 10.3%, golos-ru 14.2% -> 14.1%. Promotion: no.
- Step 14 final test: unit_wer=0.2079 (touch 1, 2026-10-16)
- Figure 14_final_test_d.png: The one-time final test on test D (2694 units: 898 never-seen utterances, each clean, with white noise at 10 dB and with babble at 10 dB), app before -> after this round's protections, language detected automatically: all 26.8% -> 20.8% (-6.0 points, 95% -7.7 to -4.7); clean 13.1% -> 13.0% (-0.1, -0.3 to +0.0); white noise 10 dB 32.9% -> 28.3% (-4.5, -9.2 to -1.6); babble 10 dB 34.4% -> 21.0% (-13.3, -15.5 to -11.2). Extra words per reference word 6.9% -> 2.5%. Largest change: Kazakh read sentences in babble 57.9% -> 21.9%. Clean sets unchanged within 0.3 points.
