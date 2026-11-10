# Noise robustness follow-up - 2026-11-10

Status: approved preparation completed through step 5; step 6 awaits a new independent test source.
No new training, model selection or final-test scoring has occurred.
The completed rounds remain unchanged. This follow-up has no inherited gate approvals.

## Steps

| Step | Status | Evidence |
|---|---|---|
| 1. Inspect available data | done | `data_profile.json`, `inventory.json`; 26,183 recipe occurrences, 9 fields |
| 2. EDA | done | figures below; all referenced multilingual audio missing on this computer |
| 3. Prediction contract | approved-gate | user approval recorded below |
| 4. Cleaning | done | 8,364 to 8,340 training recordings; 18 too long/short and 6 demo recordings removed; validation unchanged |
| 5. Data engineering | done | 16,680 training recipe views, exactly 50% clean; 4,476 validation views; source-linked seeds and hashes |
| 6. Split | in progress | official train/dev source groups disjoint; new independent test source still required |
| 7. Features | todo | fixed Whisper log-mel features |
| 8. Preprocessing | todo | reproducible train-only augmentation, frozen validation recipes |
| 9. Baseline | todo | current deployed pipeline; empty-transcript reference |
| 10. Train | todo | bounded LoRA experiment after Gate B |
| 11. Tune | todo | validation only; small declared search |
| 12. Evaluate | todo | paired word errors, clean-speech guard |
| 13. Error analysis | todo | validation failures by language, noise and speaker |
| 14. Final test | todo | new, untouched set after Gate C |
| 15. Deploy | todo | promote only if all acceptance rules pass |
| 16. Monitor | todo | track word errors and false text on silent/noise-only inputs |

## Inventory findings

The following inventory describes the state before restoration. After approval, all 12 FLEURS train/dev
archives and tables were restored and their original SHA-256 checksums matched (`restore_receipt.json`).
The prepared FLEURS subset has zero train/dev utterance, sentence-group or normalized-text intersections.
The wider old mixture recipes still reference absent KSC2/Golos audio and remain unchanged.

- 18,671 distinct training and 745 validation utterances appear in saved mixture recipes. None of their audio files is present. The full multilingual utterance manifest and trainable base weights are also absent; the converted inference weights are installed.
- Exact occurrence rows are not duplicated. Reusing an utterance in multiple mixtures is expected, but these mixtures are not statistically independent.
- No shared train/validation utterance IDs; one shared lowercased, whitespace-normalized transcript. Restore the full manifest and audit sentence identity before cleaning; zero shared IDs does not prove zero leakage.
- Speaker IDs are missing throughout. Speaker-disjoint performance cannot be claimed. Earlier results cannot become a fresh test for this experiment.
- The generic profile labels unique text targets as multiclass/imbalanced. That is a profiler limitation: the task is audio-to-text sequence prediction, not sentence classification. Text, language, IDs and durations are audit metadata, never recognizer input features.
- EDA balance plots describe language composition across recipe occurrences. Their majority-class percentages are not speech recognition accuracy. Encoded-language correlations are descriptive only.
- Historical noise fine-tuning failed its promotion criteria. It remains rejected; no favorable checkpoint is selected from the test results.

## Proposed contract for approval

Input: an audio recording; primary target: the words spoken by the intended foreground speaker, in the original Russian or Kazakh script. For two-speaker mode, both foreground voices are targets, including their simultaneous words. A quieter second foreground speaker must not be removed as background noise.

Unit: an audio window up to 30 seconds, evaluated within its source-recording group. English is a regression check. Population: read and conversational speech with cafe/traffic/stationary noise and reverberation; natural RU/KK overlap is a separate evaluation slice, not inferred from sequential language-switching performance.

At inference the system receives audio and optional user language/speaker settings only. Reference transcripts and source waveforms are evaluation material only.

Metrics: pooled WER and CER per language/noise condition; permutation-invariant speaker-attributed WER for two foreground voices; SI-SDR improvement when clean stems exist; false inserted words on noise-only clips; processing time. Do not report `100 - WER` as accuracy.

Acceptance proposal: at least 10% relative noisy-validation WER reduction versus the current deployed pipeline, paired 95% confidence interval entirely below zero, and no clean-language degradation above 0.5 percentage points. Report overlap separately and do not promote a noise improvement that suppresses the second speaker. Bootstrap by original recording/speaker where available, never by repeated augmented copies.

Candidate: a small encoder-and-decoder LoRA adaptation, conservative learning rate, dropout and validation-based early stopping. Keep at least half of training examples clean; generate augmentation only after source/noise splits are fixed. Compare with the deployed decoder-only adaptation. Avoid blindly adding denoising or aggressive suppression, which can erase consonants and quieter voices. This is a hypothesis, not a measured improvement.

Resource plan: restore the exact public source corpus files and training base first, with disk-size checks; use a free Colab GPU if available. Converted CTranslate2 inference files are not a resumable training checkpoint. No paid compute and no automatic production replacement.

- Gate A: approved 2026-11-10

User response: "Да, готовь данные по этому плану". Next phase restores and audits inputs, documents exclusions, fixes source/noise splits and prepares augmentation. Gate B is required before fitting. A new independent test strategy is required before Gate C because previous test results have already been inspected.

Restoration starts with FLEURS train/dev for kk/ru/en, pinned to the recorded source revision and checked against the original SHA-256 manifest. Previously used test audio is not downloaded. Broader broadcast/noise/overlap coverage remains required for promotion. The chat workspace links its active `ml_pipeline` to this experiment so its existing profile and figures can be checked by the gate hook.

## Figures

Preparation output lives in `data/ml/noise_followup_20261110/`. `cleaning_summary.json` records every
exclusion and `recipe_summary.json` pins the recipes. Noise has not yet been rendered for fitting.
Synthetic views of one recording are not independent examples. The original tests have already been used;
no guard fingerprint is invented for a nonexistent new test. Model training is not authorized by Gate A.
- Figure 01_missingness.png: 1 of 3 columns have missing values; worst is speaker_id at 100.0%.
- Figure 02_target_balance.png: language=ru is 34.1% of 26,183 rows, so a majority-class dummy scores 34.1%; the minority share is 32.9%.
- Figure 02_distributions.png: Histograms of 1 numeric features; 0 are strongly skewed (none), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with language: duration_s (0.37); anything above 0.95 is a leakage suspect.
- Figure 02_audio_availability.png: Saved train/validation recipes reference audio that must be restored before training. Manifest completeness does not prove that the recordings are available.
- Figure 04_cleaning_funnel.png: Training 8364 to 8340; validation 1119 unchanged. Exclusions: {'missing_audio': 0, 'empty_transcript': 0, 'duration_outside_1_30_seconds': 18, 'demo': 6, 'validation_sentence': 0, 'validation_text': 0, 'duplicate_training_utterance': 0}.
- Figure 05_augmentation_plan.png: Counts: {'clean': 8340, 'white': 2806, 'pink': 2753, 'echo': 2781}. Every source has one clean and one augmented view. Synthetic noise and a mild echo are a preparation pilot; real noise and two-voice separation remain separate checks.
