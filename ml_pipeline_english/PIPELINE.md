# ClearTalk ASR adaptation - ML pipeline log

Goal: a trained ClearTalk model that lowers word errors on the text the app actually produces -
Whisper small.en transcribing each SepFormer-separated voice from a noisy two-speaker recording.
Method: LoRA fine-tuning (PEFT) of Whisper small.en on separated LibriSpeech speech.

Notebooks (open with `.venv-train/bin/marimo edit ml_pipeline_english/<file>`):
`01_eda.py` (steps 1-3), `02_prep.py` (steps 4-8), `03_model.py` (steps 9-12), `04_eval.py` (steps 13-16).

## Checklist

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | done | 37,329 utterances (train 28,539 / val 3,231 / test 5,559); 0 speakers shared between roles; profile: no leakage suspects; 8.0% lack punctuated text |
| 2 | EDA | done | v4-validation baseline 36.5% WER (Whisper norm.); errors track separation quality (r = -0.62), 70% substitutions; train utterances much longer than val/test (median 14.0 s vs 5.6 s) |
| 3 | Problem contract | approved-gate | see "Problem contract" below |
| 4 | Data cleaning | done | train pool 28,539 → 24,482 utterances (demo speakers, missing/mismatched punctuated text, 1-16.5 s); eval sets unfiltered |
| 5 | Data engineering | done (build running) | frozen seeded plans: 8,000 train mixtures + 3,000 single utterances, V1 300+300, T2 300+400; every mixture separated by the app's SepFormer; tracks < -5 dB SI-SDR dropped from training targets (8.4%) |
| 6 | Train / val / test split | done | official speaker-disjoint subsets, 0 shared speakers; test frozen in .guard_state.json (796 units: T1 96 + T2 300 mixtures + T3 400 clean) |
| 7 | Feature engineering | done | Whisper's fixed 80-band log-mel (no fitted statistics); audio decoded exactly as the app decodes saved tracks |
| 8 | Preprocessing | done | Whisper BPE targets from punctuated text (median 37 tokens, max 78); 49% keep timestamps; processor saved with the model |
| 9 | Baseline model | done | dummy (empty text) 100% WER; pretrained small.en, app decoding: V1 mixtures 28.9%, V2 (v4) mixtures 36.5%, V3 clean 3.0% |
| 10 | Model training | done | run `lr1.5e-4`, 1 epoch = 1,102 steps on 17,632 tracks (float32, effective batch 16). Monitor subset, greedy: val mixtures 29.4% → 21.2%, clean 2.0% → 2.2%, training subset 10.5% → 7.3%; val loss 0.976 → 0.781. Relative change train -30% vs validation -28%: no overfitting signal |
| 11 | Hyperparameter tuning | done | LR pilots, 100 steps each, same data/seed: 5e-4 → 24.68% val WER, 1.5e-4 → 22.81% (clean 1.89% vs 1.70%); 1.5e-4 continued. Training length chosen by validation checkpoints |
| 12 | Model evaluation | done | app-path validation of every checkpoint; step 1102 selected. V1 mixtures 28.87% → 21.77% WER (-24.6% relative, bootstrap 95% -12.9…-1.9 points), V2 (v4) 36.47% → 30.71%, V3 clean 2.99% → 2.79%. Promotion rule passed on validation |
| 13 | Error analysis | done | validation tracks: better on 225, same on 279, worse on 96 of 600; insertions 853 → 415 (leaked second voice no longer transcribed); failed separations (≤ -5 dB) stay above 100% WER; gain concentrated at 5-15 dB SNR, almost none at 0 dB |
| 14 | Final test | done (once) | T2 new mixtures 26.16% → 20.74% (-5.41 points, 95% -7.63…-3.65); T1 v4 mixtures 33.25% → 29.58% (-3.67, 95% -5.61…-1.88); T3 clean 2.58% → 2.34% (-0.24, 95% -0.55…+0.04). Legacy normalizer on T1: 33.79% → 29.85% (v4 refiner: 33.69%) |
| 15 | Deployment | done | merged LoRA exported to CTranslate2 float16 (`models/whisper-small.en-cleartalk`), SHA-256 pinned in `models/active_profile.json`, default per the validation promotion rule; the service-loaded artifact reproduces 20/20 validated transcripts; UI and API checked on the demo |
| 16 | Monitoring + retraining plan | done | see "Monitoring and retraining plan" below |

## Delegation (2026-10-09)

The user was away during steps 1-12 and explicitly delegated Gates A and B on 2026-10-09
("Проходи A и B сама"): proceed after recording artifacts here, the user reviews afterwards.
Gate C (before the one-time final test) still requires the user's explicit approval.

## Problem contract (step 3)

- **Target:** the words spoken by ONE speaker. Training target = LibriSpeech-PC transcript
  (punctuated, cased, so the app keeps readable output). Scoring reference = official
  LibriSpeech transcript.
- **Prediction unit:** one SepFormer-separated track (8 kHz, saved as PCM16 exactly as the app does,
  decoded to 16 kHz for Whisper) from a two-speaker noisy recording of at most 20 s (the app limit).
  Secondary unit: one clean single-speaker utterance (regression check: "Transcribe only" mode).
- **Population:** English read speech (LibriSpeech audiobooks); synthetic mixtures of two different
  speakers, gain ±6 dB, onset delay 0 / 25 / 50 % of the shorter utterance, ESC-10 environmental or
  synthetic noise at 0-15 dB SNR, mild synthetic reverb in half the cases.
- **Information available at prediction time:** only the uploaded audio. Clean sources and the
  separation score (SI-SDR) are used at training time only - to pair each separated track with its
  transcript and to drop failed separations from training targets - never as model input.
- **Objective:** fewer word errors on separated tracks, without making clean single-speaker
  transcription worse.
- **Metric:** pooled WER = (S + D + I) / reference words with Whisper's EnglishTextNormalizer;
  the two tracks of a mixture are scored with the cheaper of the two speaker assignments. Supporting:
  S / D / I breakdown, the old "legacy" normalizer for comparison with the v4 reports, clean-speech WER.
- **Model selection:** validation only. Early stopping on validation WER (greedy decoding); the app's
  decoding policy (faster-whisper int8, beam 5, VAD) is then measured on validation for baseline and
  candidate. The test sets are used once, at step 14.
- **Pre-registered promotion rule (fine-tuned model becomes the app default):** on validation mixtures
  (a) WER at least 5 % lower relative to the pretrained baseline, (b) 95 % bootstrap interval of the
  WER difference over mixtures entirely below zero, (c) clean single-speaker validation WER no more
  than 0.5 percentage points worse. Otherwise the baseline stays the default.
- **Constraints:** runs in the existing app on CPU through faster-whisper (CTranslate2 int8), same
  architecture so the same cost as today; LoRA keeps trainable weights to a few percent; trained on
  this Mac (Apple M5, MPS) in hours; nothing is uploaded anywhere.
- **Frozen evaluation sets (defined now, before any training):**
  - Validation: V1 = 300 new mixtures (200 dev-clean, 100 dev-other subset); V2 = the 48 v4
    validation mixtures (continuity); V3 = 300 clean dev-clean utterances.
  - Test: T1 = the 96 v4 test mixtures (test-other, continuity with the 33.79 % report);
    T2 = 300 new mixtures (150 test-clean, 150 test-other); T3 = 400 clean test-clean utterances.

## Monitoring and retraining plan (step 16)

The app runs locally and keeps no usage statistics, so monitoring means re-measuring, not dashboards.

- **What can drift.** Input: real microphones, accents, spontaneous speech, more than two speakers, noise types
  outside ESC-10/synthetic, sampling rates and codecs. Output: a rising share of empty tracks, very long
  single segments, or text that repeats the other track (leaked voice).
- **How to check.** Keep a small consented set of real two-speaker recordings with human transcripts (even
  20-30 clips). After any change of separator, decoding policy or weights, run
  `python -m cleartalk_asr.evaluate --set val --model baseline` and the same for the fine-tuned model, then
  `scripts/activate_model.py --check`. The app-path validation numbers in this file are the reference.
- **When to retrain.** (a) validation WER of the fine-tuned model through the app path rises above 23.0%
  (reference 21.77%) or its advantage over the pretrained model falls under 5% relative; (b) clean-speech WER
  exceeds the pretrained model by more than 0.5 points; (c) the separator or the 8 kHz pipeline changes - the
  model is adapted to this exact separator's artefacts; (d) real-recording WER is clearly worse than the
  pretrained model on the consented set.
- **How to retrain.** Reuse this pipeline from step 1: new data go through `cleartalk_asr.mixtures plan/build`
  (new plan files, new seeds), `02_prep.py`, `cleartalk_asr.train`, app-path validation of checkpoints,
  `scripts/select_checkpoint.py`, `scripts/activate_model.py`. The current test sets are spent: a new final
  claim needs a new frozen test set (other LibriSpeech speakers, or better, real recordings).
- **Known weak spots to target next.** 0 dB SNR and failed separations (WER stays ~50%+); names and rare words;
  one coarse segment per track; English read speech only.

## Gates

- Gate A: approved 2026-10-09 (delegated: the user authorized passing Gates A and B unattended on 2026-10-09; contract above is for her review)

Model rationale:
- traits: speech-to-text sequence task; 37,329 utterances (not small); repeated entities = speakers (official speaker-disjoint split, no datetime); inputs are SepFormer-separated 8 kHz tracks with residual interference; 2.5% of validation words unseen in training text
- baseline: dummy = empty transcript (100% WER); pretrained Whisper small.en through the app's decoding (V1 28.9%, V3 clean 3.0%)
- candidates: Whisper small.en + LoRA (rank 16, alpha 32, dropout 0.05) on all attention and feed-forward projections of encoder and decoder; learning rate chosen on validation from {1.5e-4, 5e-4} by 100-step pilots on the full training set (the winner is resumed as the main run); training length chosen by validation WER with early stopping; float32 on MPS
- ruled out: full fine-tuning of 242 M weights (forgetting/overfitting risk, too slow on MPS); training an ASR model from scratch (needs thousands of hours); larger Whisper models (3-6x slower on CPU in the app); another spectral refiner (v0.4: +1.4 dB but no WER gain); a text-only correction model (errors are acoustic: 70% substitutions over 308 different word pairs, plus insertions from the residual second voice); reduced precision on MPS (corrupts SDPA attention)
- metric: pooled WER (Whisper normalizer) on validation two-speaker mixtures (V1) is primary; clean single-speaker WER (V3) is a guard; S/D/I and the legacy normalizer are supporting

- Gate B: approved 2026-10-09 (delegated as above). Recorded while the frozen training plan is still being rendered (≈4,000 of 19,000 tracks done); steps 4-8 rules and figures are final, their counts are refreshed when the build completes.

- Gate C: approved 2026-10-12 (user, after reviewing the validation report: "ну давай да сделай все до конца")

- Gate D: approved 2026-10-12 (wrap-up; the user asked on 2026-10-12 to finish everything after the final test)

## Wrap-up (Gate D)

- **Result.** A LoRA-adapted Whisper small.en (6.5 M trained parameters, 2.6%) lowers word errors on separated
  two-speaker speech on speakers never seen in training: final test 26.16% → 20.74% (new mixtures) and
  33.25% → 29.58% (v4 mixtures), with clean single-speaker speech unchanged or slightly better (2.58% → 2.34%).
  Validation told the same story (28.87% → 21.77%), so the validation-based choices generalized.
- **Compared with v0.4.** On the same 96 v4 test mixtures and the same legacy metric, the spectral refiner
  changed 33.79% to 33.69%; adapting the recognizer gives 29.85%.
- **Not overfit, as far as measured.** Speaker-disjoint roles, one epoch, relative WER change -30% on the
  training subset vs -28% on validation, validation loss falling to the end, and a test set scored once.
- **Limits.** Synthetic mixtures of read audiobooks; ~20-30% WER is still high in absolute terms; no gain when
  the separator fails (0 dB SNR); one coarse timed segment per track; English only; real meetings untested.
- **Stronger ready-made systems exist** (e.g. BUT-FIT/DiCoW v3.3 reports 9.7% tcpWER on Libri2Mix with a
  0.9 B-parameter target-speaker Whisper plus a diarizer); not benchmarked here on the same audio.

## Decisions

- 2026-10-09 · Splits = official LibriSpeech subsets (speaker-disjoint by construction): train-clean-100 →
  train, dev-clean + local dev-other subset → validation, test-clean + test-other → test. Keeps the
  test comparable with the published v4 numbers; a random group split would move test-other speakers
  into training.
- 2026-10-09 · Speakers 198 and 5703 (the packaged demo) are excluded from training so the demo stays
  an unseen example.
- 2026-10-09 · Primary metric uses Whisper's normalizer. On v4 validation it changes WER only from
  36.3 % (legacy) to 36.5 %, so formatting is not where the errors are; both are reported.
- 2026-10-09 · Training durations will be re-weighted toward the validation/test duration profile
  (train median 14.0 s vs 5.6 s) while keeping some long examples for 20 s app uploads.

- 2026-10-09 · Data-build bug caught before any training: track rows briefly pointed `path` at the clean
  LibriSpeech source instead of the separated WAV (a dict merge overwrote it). Fixed, rows repaired
  deterministically, `load_tracks` now refuses rows that do not point at an existing separated WAV, a
  regression test was added, and the baseline evaluation that had read clean audio was discarded and rerun.
- 2026-10-09 · Numerical check on Apple MPS (torch 2.8): half-precision SDPA attention corrupts Whisper
  (same batch: loss 0.576 fp32 vs 1.284 bf16 / 1.278 fp16; greedy decoding emitted invalid token ids).
  Eager attention in half precision is correct (0.574) and fp32 SDPA is correct. All training and
  evaluation therefore run in float32 with SDPA; no reduced precision on MPS.
- 2026-10-09 · fp16 SepFormer on MPS was tested for speed and rejected: slower than fp32 on this Mac.

- 2026-10-09 · Reproducibility check: on the 48 v4 validation mixtures the new evaluation code gives
  36.33% WER with the legacy normalizer (440 errors / 1,211 words) - exactly the v4 report's baseline.
  Whisper-normalizer WER on the same audio is 36.47%.
- 2026-10-09 · Memory: float32 training needs ~3.3 GB of MPS memory per example; micro-batch 2 ×
  accumulation 8 (effective batch 16) is used while the separator still runs, micro-batch 4 × 4 after.
- 2026-10-09 · Smoke test of the full training loop (2 tiny steps, evaluation, best/last checkpoints,
  resume state) passed; 6,488,064 of 248,222,208 parameters are trainable (2.6%).

- 2026-10-09 · First training attempt with micro-batch 4 ran out of MPS memory after the step-0 evaluation;
  restarted with micro-batch 2 × accumulation 8 (same effective batch) and smaller fixed monitoring subsets
  (200 validation mixtures, 60 clean, 70 training units) because a full greedy evaluation took 12 min.
- 2026-10-09 · Monitoring caveat: at step 0 (pretrained model) the training subset already scores 10.5% vs 29.4%
  on validation, because only well-separated training mixtures pass the step-5 filter. The overfitting check is
  therefore the CHANGE of the train/validation gap from step 0, not its absolute size.
- 2026-10-09 · Transfer check on validation through the app path (int8, beam 5, VAD) for the 5e-4 pilot at step
  100: V1 mixtures 28.87% → 22.79% WER (insertions 808 → 433), clean 2.99% → 2.82%. Validation only.

- 2026-10-09 · MPS caching allocator grew the training process to 23 GB (swap 9 GB, steps slowed 14.5 → 23 s);
  fixed by releasing the MPS cache every 5 steps and PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=1.0/0.7 (resumed from the
  step-200 state; nothing lost). The Mac was also on battery in Low Power Mode (~20 s/step).
- 2026-10-09 ~10:45 · Paused at the user's request (laptop closed). Saved state: step 200, best = step 200
  (monitor subset: 21.99% val mixtures, 1.99% clean). Training resumes from the last checkpoint (`--save-every 25` added).

- 2026-10-12 · Training resumed from the step-200 state (deterministic: step-250 loss reproduced to 4 digits).
  An evaluation at step 400 ran out of MPS memory under the 1.0 watermark; resumed from the step-375 state with
  evaluation batch 16 and watermark 1.2. Nothing else changed; the run finished the full epoch.
- 2026-10-12 · The greedy monitor on 200 mixtures is noisy (step 400 looked worse: 23.9% vs 22.0% at step 200,
  while through the app path on all 300 mixtures step 400 is better: 22.38% vs 22.78%). Final checkpoint
  selection therefore uses app-path validation of every saved checkpoint (`scripts/select_checkpoint.py`).

- 2026-10-12 · Checkpoint selection on app-path validation (300 mixtures / 300 clean): step 200 22.78% / 2.92%,
  400 22.38% / 2.89%, 600 21.95% / 2.82%, 800 22.31% / 2.91%, 1000 21.95% / 2.98%, 1102 21.77% / 2.79%;
  pretrained 28.87% / 2.99%. Step 1102 exported; the exported model.bin has the same SHA-256 as the validated file.
- 2026-10-12 · Where the gain is (validation): SNR 5 dB 35.6% → 19%, 10 dB 16% → 10%, 15 dB 13.5% → 9.5%, but
  0 dB 50% → 48%: when the separator fails the recognizer cannot recover the words. Insertions 808 → 394.
- 2026-10-12 · Behaviour change to disclose: the fine-tuned model returns one timed segment per track (its
  training targets were single-segment), where the pretrained model split tracks into several segments.
- 2026-10-12 · App smoke check with the exported model (demo recording, speakers excluded from training): both
  recognizers run through the service; the profile's default is `cleartalk` per the validation promotion rule.

- 2026-10-12 · Project folder moved to ~/Desktop/nlp-project/cleartalk at the user's request (virtual-environment
  script paths rewritten; both test suites pass from the new location).
- 2026-10-12 · Real-data pilot (after the final test, no model change): 28 two-speaker overlapping fragments cut
  from six AMI test meetings (CC BY 4.0), 1,051 reference words, scored through the app pipeline.
  Pretrained 29.2% WER vs ClearTalk 34.1%; without one laughter fragment 29.2% vs 27.6%; fragments better/same/
  worse 9/7/12; 95% bootstrap of the difference -8 to +23 points. Conclusion: the gain measured on audiobook
  mixtures is NOT confirmed on spontaneous meeting speech, and the fine-tuned model can loop on laughter
  ("heh" ×70). The decoding policy uses a single temperature, so faster-whisper's compression-ratio fallback
  never triggers; a fallback or repetition guard is the first thing to try, validated on validation data.
- 2026-10-19 · Repetition guard in the English mode (no model change). The multilingual work on the Acer added
  `backend/loops.py` (a word or phrase of up to six words repeated more than three times in a row is cut back to
  three) to the shared `backend/transcription.py`, so the policy signature in `models/active_profile.json` was
  re-pinned on the Mac. Applied to the stored validation transcripts (900 tracks): ClearTalk changes on 2 tracks,
  track-level WER 15.0% → 14.2%; pretrained changes on 6, 19.7% → 17.0%; v4 validation and the ClearTalk test
  transcripts are unchanged. The final-test numbers above were measured without the guard and stay as recorded.

## Figures
- Figure 01_corpus_overview.png: Training has 100.6 h from 251 speakers; validation 6.3 h and test 10.7 h use different speakers. dev-other is only a 528-utterance local subset, so validation leans on dev-clean.
- Figure 01_missing_punctuated_text.png: 8.0% of utterances have no punctuated transcript (worst: train-clean-100 at 8.8%). These cannot be training targets without losing punctuation, but they keep their official text for scoring.
- Figure 02_duration_by_role.png: Training utterances are long (median 14.0 s, 6% over 16 s) while validation/test are shorter (median 5.6 s). A two-speaker mixture must fit Whisper's 30 s window, so training sources are capped and short ones kept.
- Figure 02_length_and_speaking_rate.png: Median speaking rate is 2.73 words/s in train, 2.79 in validation and 2.69 in test: the reading pace is similar, so the main difference between roles is length, not speaking style.
- Figure 02_gender_balance.png: Female share of speakers is {'train': 0.5, 'val': 0.49, 'test': 0.51}. The roles are balanced, so a gender skew cannot explain a validation/test difference.
- Figure 02_vocabulary_and_oov.png: The training text has 33,798 word types with a Zipfian tail; 2.52% of validation and 2.62% of test words never occur in training. Those words must come from Whisper's pretrained knowledge, which is one reason to adapt Whisper gently (LoRA) instead of retraining it.
- Figure 02_baseline_error_profile_validation.png: On the 48 v4 validation mixtures the current app has 36.5% WER with Whisper normalization (36.3% with the old rule). Errors correlate with separation quality (r = -0.62) and are mostly substitutions (313 vs 48 deletions, 88 insertions): the recognizer mishears artefact-laden speech, which is what fine-tuning on separated audio targets.
- Figure 02_top_substitutions_validation.png: 313 substituted words fall into 308 distinct pairs and the top 15 cover only 6%: errors are spread across acoustically similar words, not one systematic spelling problem a text rule could fix.
- Figure 04_cleaning_funnel.png: 28,539 training utterances → 24,482 usable (85.8%). The biggest cuts are missing punctuated text and punctuated text whose words differ from the official transcript (e.g. 'misses' for 'missus', dropped phrases); validation/test utterances are not filtered.
- Figure 07_logmel_clean_vs_separated.png: Whisper's input features are fixed log-mel spectrograms (no fitted statistics, so nothing can leak from validation/test). The separated track loses everything above 4 kHz (top quarter of the bands is empty, mean level change +0.13) and carries residual interference: this is the domain shift the fine-tuning has to absorb.
- Figure 05_separation_quality_and_filter.png: 8.6% of 16,000 separated training tracks fall below -5 dB. On validation the pretrained recognizer already has 113% WER on such tracks: their transcript is not in the audio (the separator failed), so training on them would teach the model to invent words. They are dropped from training targets only; evaluation keeps every track.
- Figure 06_split_and_duration_matching.png: 17,632 training tracks, 648 validation units and 796 frozen test units (mixtures or single utterances). Speaker overlap between roles: {'train&val': 0, 'train&test': 0, 'val&test': 0}. Re-weighting moved the training median duration to 11.9 s (raw corpus 14.0 s) toward validation/test.
- Figure 08_target_token_lengths.png: Targets are Whisper BPE tokens of the punctuated transcript: median 40, max 84 tokens, far below the 440-token limit, so nothing is truncated. 50% of training tracks keep segment timestamps so the app's timed segments still work.
- Figure 09_baselines.png: An empty transcript scores 100% WER. The current app (pretrained Whisper small.en on separated tracks) has 28.9% WER on 300 new validation mixtures, 36.5% on the 48 v4 validation mixtures, and 3.0% on clean single-speaker speech. The gap between clean and separated speech is what fine-tuning tries to close.
- Figure 11_learning_rate_pilots.png: Identical 100-step runs that differ only in learning rate, compared on validation mixtures: lr1.5e-4 → 22.8%, lr5e-4 → 24.7%. lr1.5e-4 was continued as the full run; the test sets played no part.
- Figure 10_learning_curves.png: Validation WER on mixtures goes from 29.4% (step 0 = pretrained) to 21.2% at step 1102; clean-speech WER 2.0% → 2.2%. The training subset is easier from the start (10.5% at step 0, because failed separations are filtered out of training), so the overfitting check is the relative change: training -30% vs validation -28%. The gap does not open and validation loss keeps falling (0.976 → 0.781): no sign of overfitting within one epoch.
- Figure 12_validation_comparison.png: Through the app's exact decoding, validation WER on mixtures is 28.9% → 21.8% (difference -7.1 points, 95% bootstrap interval -12.9 to -1.9 over 300 mixtures); v4 validation 36.5% → 30.7%; clean speech 3.0% → 2.8%. The right panel shows where the gain comes from.
- Step 14 final test: unit_wer=0.1519 (touch 1, 2026-10-12)
- Figure 13_error_analysis_validation.png: On 600 separated validation tracks ClearTalk makes fewer errors on 225, the same on 279 and more on 96. Substitutions 1656 → 1409, deletions 416 → 379, insertions 853 → 415: the main gain is no longer transcribing the leaked second voice. Tracks where the separator failed (SI-SDR ≤ -5 dB) stay at 107% WER (113% pretrained): the recognizer cannot recover words that are not in the audio.
- Figure 14_final_test.png: T1 · v4 test mixtures: 33.25% → 29.58% WER over 3,080 words (difference -3.67 points, 95% bootstrap -5.61 to -1.88); T2 · new test mixtures: 26.16% → 20.74% WER over 9,367 words (difference -5.41 points, 95% bootstrap -7.63 to -3.65); T3 · clean single speaker: 2.58% → 2.34% WER over 7,494 words (difference -0.24 points, 95% bootstrap -0.55 to +0.04). Same frozen audio for both recognizers, reported as measured; the model was fixed on validation before this run.
