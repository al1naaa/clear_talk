# ClearTalk multilingual - round 3: the small model with Kazakh broadcast speech

Rounds 1 (`../PIPELINE.md`) and 2 (`../round2/PIPELINE.md`) are closed; their test sets A and B were each opened
once. This round retrains the SMALL multilingual model with more Kazakh speech and has its own frozen test set C.

Goal: the small model (the app's stand-in recognizer for every language when the large models are not installed -
on the Mac, for example, where everything runs on the processor) should write Kazakh outside read speech - radio,
podcasts, talk shows, parliament, TV news - much better than the round-1 model, without losing anything else.
Method: the round-1 recipe unchanged (multilingual Whisper small, LoRA rank 16 on all attention and feed-forward
projections, learning rate 5e-4, one epoch) with Kazakh broadcast speech from the KSC2 training part added.

## Checklist

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | todo | |
| 2 | EDA | todo | |
| 3 | Problem contract | approved-gate | see "Problem contract" below |
| 4 | Data cleaning | todo | |
| 5 | Data engineering | todo | |
| 6 | Train / val / test split | todo | |
| 7 | Feature engineering | todo | |
| 8 | Preprocessing | todo | |
| 9 | Baseline model | todo | |
| 10 | Model training | todo | |
| 11 | Hyperparameter tuning | todo | |
| 12 | Model evaluation | todo | |
| 13 | Error analysis | todo | |
| 14 | Final test | todo | |
| 15 | Deployment | todo | |
| 16 | Monitoring + retraining plan | todo | |

## Delegation (2026-10-14)

After round 1 the user asked for all four improvements on offer ("все улучшения сделай … хорошо и качественно"),
one of them "more Kazakh data for the small model". The data needed a download above the 15-20 GB she had
approved, so she was asked first: the question said that the KSC2 archive can only be read from its beginning,
that about 30 GB would have to be transferred with a few GB kept, and that another night of training of the small
model would follow. On 2026-10-14 she answered "да, скачай казахские данные". Her standing rule from the start of
the project applies: pass the pipeline stops yourself, the final test included, and write everything down for her
review. As before, nothing is recorded as approved before its figures and tables exist on disk.

## Problem contract (step 3) - what changes against round 1

Target, prediction unit, population, information at prediction time and the metric are those of round 1
(`../PIPELINE.md`, "Problem contract").

- **Objective:** lower Kazakh WER of the small model on KSC2 - above all its broadcast sources - with read Kazakh,
  Russian, English and mixed recordings no worse than the round-1 model.
- **Data:** the round-1 training epoch unchanged, plus the KSC2 training part's broadcast sources (parliament,
  podcasts, radio, talk shows, TV news; at most 3,000 utterances per source were kept from the stream): 500 single
  utterances per source per epoch, and every kept broadcast utterance offered to the builder of the mixed
  training recordings next to the 6,000 crowdsourced ones (new frozen recipes, plan "train3"). The round-1
  composition is kept in the code as "base"; earlier runs draw exactly what they drew before.
- **Baselines:** the round-1 small model (what this round must beat); untouched Whisper small; reference points:
  the Kazakh turbo fine-tune and our large model of round 2 on the Kazakh sets.
- **Hyperparameters:** those of round 1, learning rate 5e-4 carried over (no pilots): only the data change, so the
  comparison with round 1 is "same model, same settings, more Kazakh speech".
- **Checkpoint:** of the shortlisted checkpoints (the last one and the greedy monitor's best, as in rounds 1 and 2),
  the one with the lowest pooled Kazakh validation WER through the app's decoding among those that meet guards
  (b) and (c) of rule 1 below; if none meets them, the one with the lowest pooled Kazakh WER (rule 1 then fails).
- **Pre-registered rules (written before any training of this round):**
  1. *Promotion* - the new model replaces the round-1 model as the app's small recognizer (`small_recognizer`) if
     on validation (a) its pooled Kazakh WER (FLEURS kk + KSC2) is at least 10% lower, relative, than the round-1
     model's, with the 95% bootstrap interval of the difference below zero, (b) its one-pass WER on mixed
     recordings is not more than 0.5 points worse, and (c) FLEURS Russian, FLEURS English and Golos are each not
     more than 0.5 points worse.
  2. *Routes* - rule 2 of round 1, re-applied with the new model taking part (each language to the installed
     recognizer with the lowest pooled validation WER).
  3. *Report, not a rule:* KSC2 per source, round-1 model against the new one.
- **Frozen evaluation sets:**
  - Validation: the round-1 validation sets, unchanged.
  - Test C (new, frozen before any training of this round, `.guard_state.json` in this folder): 898 units from
    test-role utterances that neither test A nor test B used - 100 per KSC2 source (600), FLEURS Kazakh 100,
    FLEURS Russian 80, Golos 100, FLEURS English 18 (all that is left). No mixed recordings: each needs English.
    The English guard catches a collapse, not a small change; for English and mixed recordings the round's
    evidence is validation only.
- **Known caveats:** no speaker ids - TV anchors, radio hosts and members of parliament may speak in both the
  KSC2 training and test folders, which would flatter broadcast results; the official split is followed and this
  limit is stated, not solved.

## Decisions

- 2026-10-14 · Test C was built and frozen before the new data were added to the manifest or any training of this
  round began (898 units, fingerprint in `.guard_state.json`; shared with tests A and B, validation or training: 0).
- 2026-10-15 · Round 3 was NOT trained. The stream of 2026-10-14 reached the KSC2 training part's broadcast sources only at 34 GB of the archive: within the approved transfer only parliament speech was reached, and since audio and transcripts are scattered in the archive, only 1,080 parliament utterances have both (2.1 h). Reaching podcasts, radio, talk shows and TV news would need the archive read again from the start (about 75 GB); the user was told and did not ask for it, and the project moved on to other improvements. Test C was never opened; on 2026-10-15 it was handed to round 4 (the large model in noise) as test D.

## Figures
