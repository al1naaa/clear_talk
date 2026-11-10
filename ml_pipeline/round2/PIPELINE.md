# ClearTalk multilingual - round 2: one large recognizer that follows language changes

Round 1 (`../PIPELINE.md`) adapted the small Whisper and is closed: its test set was opened once. Its own
retraining plan says a new model starts this pipeline again with a new frozen test set. This is that second pass.

Goal: one large recognizer that writes Kazakh, Russian and English and keeps writing correctly when the language
changes inside a recording - including inside one turn, which the app's turn-by-turn pipeline cannot handle.
Method: LoRA on the **decoder only** of the Kazakh fine-tune of Whisper large-v3-turbo
(shyngys879/kazakh-whisper-large-v3-turbo, Apache 2.0). The encoder stays frozen, so the training fits the laptop's
6 GB graphics card.

Notebooks: `r2_prep.py` (steps 1-8, what is new in this round), `r2_model.py` (steps 9-12), `r2_eval.py` (steps 13-16).

## Checklist

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | done | no new speech data: the 44,514 utterances inspected in round 1 (no leakage suspects, no speaker ids); new in this round are the starting model and the second test set |
| 2 | EDA | done | starting points on validation: Kazakh turbo fine-tune kk 12.2% (FLEURS) / 9.4% (KSC2), ru 7.3%, en 9.7%; untouched large-v3-turbo ru 4.0%, en 5.6%; mixed recordings in one pass: small 60.0%, large-v3-turbo 47.7%, our round-1 model 16.1% |
| 3 | Problem contract | approved-gate | see "Problem contract" above (what changes against round 1) |
| 4 | Data cleaning | done | cleaning rules of round 1, unchanged (20,767 training utterances kept) |
| 5 | Data engineering | done | training recipes of round 1, unchanged (9,000 mixed recordings per epoch); 170 new seeded recipes for the mixed recordings of test B, built only from unused test utterances |
| 6 | Train / val / test split | done | roles of round 1; test B frozen: 1,020 units (170 per set), none shared with test A, validation or training |
| 7 | Feature engineering | done | Whisper's fixed log-mel spectrogram with 128 bands × 3000 frames (the large model's input); nothing fitted |
| 8 | Preprocessing | done | targets with the large model's tokenizer: median 45 tokens single / 99 mixed, longest 231, none cut; same format as round 1 |
| 9 | Baseline model | done | empty transcript 100% WER; the baseline of this round, the starting model untouched, through the app's decoding: mixed recordings in one pass 30.9%; fleurs-kk 12.2%, ksc2-kk 9.4%, fleurs-ru 7.3%, golos-ru 16.8%, fleurs-en 9.7%. Reference points: untouched large-v3-turbo mixed 47.7%, fleurs-kk 26.2%, ksc2-kk 46.6%, fleurs-ru 4.0%, golos-ru 21.0%, fleurs-en 5.6%; our small model of round 1 mixed 16.1%, fleurs-kk 24.2%, ksc2-kk 41.6%, fleurs-ru 9.5%, golos-ru 15.8%, fleurs-en 7.4%; the app's pipeline with the routes of round 1, mixed 10.8% |
| 10 | Model training | done | run `large-lr1.5e-4`, one epoch = 900 steps over 14,400 examples (float32, 4 examples x 4 accumulated = 16 per step, about 16 s per step; decoder-only LoRA, 2,129,920 of 811,008,000 parameters). Greedy monitor: validation mixed 56.4% → 9.1%, training subset 53.2% → 10.9%; validation loss 0.636 → 0.393 (lowest 0.387 at step 600), training-subset loss 0.654 → 0.222 |
| 11 | Hyperparameter tuning | done | LR pilots, 80 steps each, same data/seed/schedule: 0.00015 → 11.8% validation mixed WER, 0.0005 → 17.8% validation mixed WER; `large-lr1.5e-4` continued. Training length chosen by validation checkpoints |
| 12 | Model evaluation | done | app-path validation of the shortlisted checkpoints (step 600: 9.8%, step 900: 9.9% on mixed recordings); step 600 selected. One-pass mixed: starting model 30.9% → ours 9.8% (-21.1 points, bootstrap 95% -24.1…-18.0); our small model 16.1% (-6.3 points, -7.3…-5.3). Language given, starting model → ours: fleurs-kk 12.2% → 10.6%, ksc2-kk 9.4% → 10.3%, fleurs-ru 7.3% → 5.3%, golos-ru 16.8% → 14.2%, fleurs-en 9.7% → 6.9%. Rule 1 routes: {'kk': 'cleartalk-large', 'ru': 'cleartalk-large', 'en': 'turbo'}; rule 2 recognizer for the mixed setting: cleartalk-large. Kazakh KSC2 is +0.9 points against the starting model (reported, see Gate C) |
| 13 | Error analysis | done | validation only. On 300 mixed recordings our large model makes fewer errors than the starting model on 202 (same 66, more 32) and than our small model on 216 (same 40, more 44). Starting model → ours: missing words 2,416 → 170, wrong words 859 → 798, extra words 114 → 106; our small model had 1,366 wrong words - the acoustic gap this round was meant to close. By place in the recording the starting model collapses after the first change of language (utterance 1 15.1%, utterance 2 46.1%, utterance 3+ 32.8%); ours is level (utterance 1 9.6%, utterance 2 9.4%, utterance 3+ 11.4%). English words written in Cyrillic: 3.9% → 0.2%. Kazakh KSC2 by source, starting model → ours: crowdsourced 6.2% → 6.9%, parliament 7.6% → 8.0%, podcasts 16.8% → 17.8%, radio 20.5% → 19.1%, talkshow 13.2% → 16.3%, tv_news 4.3% → 5.4% - worse on crowdsourced, parliament, podcasts, talkshow, tv_news, better on radio: the loss on KSC2 is spread over the sources, not one failure. The language our model detects for a mixed recording is its first utterance's in only 34.3% of cases, yet WER is the same either way (10.5% vs 9.4%): the token does not steer what it writes. Reading the worst recordings: names (Arly Velasquez), numbers written as digits where the reference spells them out, hesitation sounds kept in KSC2 references, and joined or split compounds (Super-G) |
| 14 | Final test | done (once) | against the starting model: mixed 30.0% → 9.7% (-20.3 points, 95% -24.2…-16.3); fleurs-kk 12.2% → 11.3% (-0.8 points, 95% -1.8…+0.0); ksc2-kk 9.8% → 10.3% (+0.5 points, 95% -0.5…+1.7); fleurs-ru 6.9% → 5.2% (-1.7 points, 95% -2.6…-0.8); golos-ru 20.8% → 17.3% (-3.5 points, 95% -6.1…-1.2); fleurs-en 9.1% → 6.6% (-2.5 points, 95% -3.5…-1.5). Other systems on the same audio: untouched large-v3-turbo mixed 43.6%, fleurs-kk 23.3%, ksc2-kk 49.5%, fleurs-ru 3.7%, golos-ru 26.4%, fleurs-en 5.4%; our small model of round 1 mixed 16.4%, fleurs-kk 22.5%, ksc2-kk 45.6%, fleurs-ru 10.8%, golos-ru 20.0%, fleurs-en 7.3%; Whisper small mixed 59.9%, fleurs-kk 74.4%, ksc2-kk 102.7%, fleurs-ru 9.9%, golos-ru 39.3%, fleurs-en 7.6%; app pipeline, language per turn mixed 9.1%; app pipeline, mixed setting mixed 8.3% |
| 15 | Deployment | done | the validated checkpoint folder was copied as `models/multilingual/whisper-large-cleartalk-multilingual` (same SHA-256), signatures of 5 recognizers pinned in `models/multilingual/profile.json`; routes {'kk': 'cleartalk-large', 'ru': 'cleartalk-large', 'en': 'turbo'}, recognizer for the mixed setting cleartalk-large; the service-loaded model reproduces 20/20 mixed and 20/20 single validated transcripts; on the CPU (int8) 6 of those units took 203.6 s for 114.1 s of audio with WER 10.5% (graphics card: 11.5%); the API processed the three-language demo with the routes, with the mixed setting and with our large model alone (6, 6 and 6 turns) |
| 16 | Monitoring + retraining plan | done | see "Monitoring and retraining plan" below |

## Delegation (2026-10-13, evening)

After the round-1 results the user was offered four improvements and answered: "все улучшения сделай - и большую
модель можешь, и остальное, хорошо и качественно прям главное сделай". The offer for this one read: train the
decoder of the Kazakh turbo model on the mixed recordings, one night on the Acer, with a new check and a new test
set built from unused recordings - "this is the permission for a new training run". Her standing rule from the
start of the project still applies: pass the pipeline stops yourself, the final test included, but write
everything down for her review. So, as in round 1, every stop is reported here first and then recorded as
delegated; nothing is recorded before its figures and tables exist on disk.

## Why this candidate (facts from round 1, validation only)

- On mixed-language validation recordings in one pass: Whisper small 60.0%, large-v3-turbo 47.7%, our small
  LoRA model 16.1%. The app's turn-by-turn pipeline with ready large models gets 10.8%, but only because every
  language change in the assembled recordings is also a pause and a speaker change.
- The remaining errors of our small model are wrong words (substitutions), not lost or invented phrases: a
  decoding setting cannot fix them; a stronger acoustic model can. The Kazakh turbo fine-tune has that acoustic
  model (Kazakh FLEURS 12.2%, KSC2 9.4% on validation) but was tuned for Kazakh only (Russian 7.3%, English 9.7%
  against 4.0% and 5.6% for the untouched large-v3-turbo).
- A probe without training showed that a decoder-only LoRA of this model needs 3.6-4.4 GB and about one second per
  example on the GTX 1660 Ti: one epoch of 14,400 examples is one night.

## Problem contract (step 3) - what changes against round 1

Target, prediction unit, population, information at prediction time and the metric are those of round 1
(`../PIPELINE.md`, "Problem contract"): the words spoken in a window of at most 30 s, each part in its own
language and script; pooled WER after the common normalizer, through the app's decoding.

- **Objective:** a lower one-pass WER on mixed recordings than every installed recognizer, without losing the
  starting model's Kazakh and while recovering Russian and English towards the untouched large-v3-turbo.
- **Baselines:** the starting model untouched (Kazakh turbo fine-tune); untouched large-v3-turbo; our round-1
  small model; the app's pipeline with the current routes.
- **Model selection:** validation only; the checkpoint by the app's decoding on the mixed validation recordings.
- **Pre-registered rules (decided before any training of this round):**
  1. *Routes* - unchanged rule 2 of round 1: each language goes to the installed recognizer with the lowest pooled
     validation WER over that language's sets; the new model simply takes part.
  2. *Recognizer for mixed speech* - the installed recognizer with the lowest one-pass WER on the mixed validation
     recordings becomes the app's `mixed_recognizer`: the model that writes every turn when the user says the
     languages change inside turns. A tie goes to the smaller model.
  3. *Report, not a rule:* the new model's WER per language against its untouched starting model. If Kazakh gets
     more than 0.5 points worse, it is said plainly in the report and the model card.
- **Constraints:** GTX 1660 Ti 6 GB, float32, encoder frozen; nothing uploaded; the exported model must run through
  faster-whisper on a CPU as well (1.6 GB, the size of large-v3-turbo).
- **Frozen evaluation sets:**
  - Validation: the round-1 validation sets, unchanged (they were never a test).
  - Test B (new, frozen before any training of this round): 170 units per set - FLEURS kk / ru / en, KSC2, Golos and
    mixed recordings from new seeded recipes - drawn only from test-role utterances that the first final test never
    transcribed. 1,020 units; `.guard_state.json` in this folder.
- **Known caveat carried over:** the starting model's authors trained it on KSC2; whether the official KSC2 Test
  folders (our KSC2 validation and test) were excluded is not stated. KSC2 numbers of this model family may be
  optimistic; FLEURS is the cleaner Kazakh measure.

## Gate A and Gate B report (steps 1-8)

Both stops are reported together because the data of this round are round 1's and were inspected, cleaned and
split there; what had to be done again is short.

- **What was done.** The starting points were laid side by side on validation (figure 02). A second test set was
  drawn from test utterances that the first final test never transcribed, and frozen (figure 06). The training
  examples were turned into the large model's features (128-band log-mel, figure 07) and tokens (figure 08).
- **What was found.** The Kazakh fine-tune is strong where our small model is weak (Kazakh) and behind the
  untouched large-v3-turbo on Russian and English. No ready model follows a language change in one pass: 47.7% and
  60.0% WER on mixed validation recordings. Our round-1 small model does (16.1%) but with a weak acoustic model.
- **Decisions and why.**
  - Start from the Kazakh fine-tune, not from the untouched large-v3-turbo: Kazakh acoustics cannot be learned by
    a decoder-only LoRA with a frozen encoder, while Russian and English wording can be recovered by the decoder
    from an encoder that still carries them (7.3% and 9.7% are damaged, not lost).
  - LoRA in the decoder only: with the encoder frozen no encoder activations are kept for gradients, so the model
    fits 6 GB (3.6-4.4 GB measured) at about one second per example.
  - Same 14,400 training examples as round 1: the comparison with the small model is then "same data, other model".
  - Test B has 170 units per set (347 unused FLEURS English test utterances exist).
- **Known weaknesses kept in view.** Those of round 1 (no speaker ids; every language change in a mixed recording is
  also a speaker change). New: the starting model's authors trained on KSC2 and do not say whether its official
  Test folders were held out, so its KSC2 validation and test numbers may be optimistic; with a frozen encoder
  the model cannot become better at hearing, only at writing.
- **What comes next (steps 9-12).** Baselines on validation; two 80-step learning-rate pilots; one epoch;
  checkpoint choice and the two pre-registered rules on validation.
- **The question - "may I continue to training?"** Delegated by the user on 2026-10-13 (see "Delegation").

- Gate A: approved 2026-10-13 (round 2; delegated: the user approved this new training run and all its stops on 2026-10-13)

Model rationale:
- traits: speech-to-text sequence task in three languages; 14,400 training examples per epoch from 20,767 utterances (not small); no datetime; repeated entities = sentences (kept inside one role) and speakers (ids not published); inputs are 30 s windows of 128-band log-mel, 62% of them holding several languages
- baseline: dummy = empty transcript (100% WER); the untouched starting model (Kazakh turbo fine-tune) and the untouched large-v3-turbo through the app's decoding; reference points: our round-1 small model and the app's pipeline with the current routes
- candidates: Kazakh turbo fine-tune + LoRA (rank 16, alpha 32, dropout 0.05) on the attention and feed-forward projections of the 4 decoder layers only (2.1 M of 811 M parameters), encoder frozen, float32; learning rate chosen on validation from {1.5e-4, 5e-4} by 80-step pilots with identical data and seed; training length by validation with early stopping
- ruled out: LoRA in the encoder of a large model (gradients through 32 encoder layers do not fit 6 GB); full fine-tuning (same, and it would forget); starting from the untouched large-v3-turbo (its Kazakh, 26% WER, is an encoder weakness a frozen encoder cannot repair); half precision on this card (slower than float32, measured in round 1); a text-only corrector (cannot recover a phrase the recognizer dropped)
- metric: one-pass pooled WER on mixed validation recordings through the app's decoding (primary); WER per language against the untouched starting model and against the current routes (guards and routing rule); CER and errors per language inside mixed recordings (supporting)

- Gate B: approved 2026-10-13 (round 2; delegated as above; recorded after the round-2 figures 02, 06, 07 and 08 were written)

## Gate C report (steps 9-12)

- **What was done.** (9) Baselines on validation through the app's decoding: an empty transcript, the starting model
  untouched (the baseline of this round), and as reference points the untouched large-v3-turbo, Whisper small, our
  small model of round 1 and the app's pipeline. (10) One epoch of decoder-only LoRA training. (11) Two learning-rate
  pilots that differ only in the rate. (12) The shortlisted checkpoints were exported and scored through the app's
  decoding; the checkpoint and both pre-registered rules were decided on validation only. Test B was not touched.
- **What was found.**
  - The starting model hears Kazakh well but does not follow a language change: on mixed recordings in one pass it
    gets 30.9% of the words wrong, and 2,416 of its 3,389
    errors are missing words (inside mixed recordings: Kazakh 22.5%, Russian 17.3%, English 43.4%).
  - Our large model: mixed recordings 30.9% → **9.8%** (difference -21.1
    points, 95% bootstrap -24.1 to -18.0, 300 recordings); missing words
    2,416 → 170. Inside mixed recordings: Kazakh 12.2%, Russian 10.1%, English 8.2%.
  - Against our small model of round 1 (same training data, other model): 16.1% → 9.8%,
    -6.3 points (interval -7.3 to -5.3). The small model's Kazakh inside mixed recordings was
    33.8%; the large model's is 12.2%. English inside mixed recordings
    is the one place where the small model is not behind (7.6% vs 8.2%).
  - With the language given, starting model → ours: Kazakh FLEURS 12.2% → 10.6%,
    Russian FLEURS 7.3% → 5.3%, Russian Golos
    16.8% → 14.2%, English FLEURS
    9.7% → 6.9%. Russian and English were recovered only part of the
    way to the untouched large-v3-turbo (4.0% and 5.6% on FLEURS).
  - **Kazakh broadcast and read speech (KSC2) got worse: 9.4% → 10.3%,
    +0.9 points - more than the 0.5 points the contract asked to report plainly.** The starting model's authors
    trained it on KSC2 and do not say whether these official test folders were held out, so its number may be optimistic; ours was
    trained on one KSC2 source only. Either way the contract's aim "without losing the starting model's Kazakh" is met on read
    sentences (FLEURS) and not met on KSC2.
  - The app's pipeline on the same mixed recordings: routes of round 1 10.8%; the new routes
    9.9%; the "languages change inside turns" setting (every turn to our large model)
    **8.2%**. The setting is ahead of the per-language routes because the language detector calls some
    Kazakh and Russian turns English; such a turn then goes to large-v3-turbo with an English token and is written in English,
    while our model writes what it hears whatever the token says.
  - Our large model names the language badly: it labels Kazakh speech as Kazakh for 24.4% of FLEURS and
    20.5% of KSC2 utterances (the starting model: 100.0% and
    95.2%). The text is still Kazakh - WER with automatic language 10.4%
    against 10.6% with the language given. As in round 1, the language token was left out of the loss on
    purpose. In the app the label of a turn written by our models is read from the text, and the detector is large-v3-turbo.
  - Training fit and validation: the training-subset loss falls to the end (0.654 → 0.222) while the
    validation loss is flat from step 200 (0.394, lowest 0.387 at step 600, 0.393 at the
    end): the gap grows, so more passes over the same data would not help. Validation WER did not get worse inside the epoch
    (mixed, greedy: 10.7% at step 200, 9.1% at the end). Most of the gain came in the first 80
    steps (56.4% → 11.8%): the model only had to learn to keep writing after a language change.
- **Decisions and why.**
  - Learning rate 0.00015: lower validation WER on mixed recordings after 80 steps
    (0.00015: 11.8%, 0.0005: 17.8%).
  - Checkpoint: step 600 by the app's decoding on the mixed validation recordings
    (step 600: 9.8%, step 900: 9.9%). The two are within noise of each
    other; the rule (lower WER) was applied as written.
  - Rule 1 (routes, lowest pooled validation WER per language): **{'kk': 'cleartalk-large', 'ru': 'cleartalk-large', 'en': 'turbo'}**. Kazakh: our large model 10.5%, the starting model 11.0%, our small model 31.7%, untouched large-v3-turbo 35.1%, Whisper small 87.2%. Russian: our large model 7.3%, untouched large-v3-turbo 7.9%, our small model 10.9%, Whisper small 17.3%.
    English: untouched large-v3-turbo 5.6%, our large model 6.9%, Whisper small 7.4%, our small model 7.4%. For Kazakh our model wins the pooled comparison (better on FLEURS, worse on KSC2) by a small margin.
  - Rule 2 (recognizer for the mixed setting, lowest one-pass WER on mixed recordings): **cleartalk-large**
    (our large model 9.8%, our small model 16.1%, untouched large-v3-turbo 47.7%, Whisper small 60.0%).
  - The small recognizer that stands in when no large model is installed stays the round-1 model (rule 1 of round 1).
- **What comes next (steps 13-16).** Error analysis on validation; the one-time final test on test B (1,020 units that no
  system has transcribed: our large model, its starting model, large-v3-turbo, our small model, Whisper small and the app's
  pipeline in both settings on the same audio); deployment check, including the model on a CPU; monitoring plan.
- **The question - "may I open test B?"** Delegated in advance by the user on 2026-10-13 (see "Delegation": the stops of
  this pipeline are passed without her, the final test included, and everything is written down for her review).

- Gate C: approved 2026-10-14 (round 2; delegated: the user authorized the stops of this training run, the one-time final test included, on 2026-10-13; this report is for her review)

## Monitoring and retraining plan (step 16)

The app runs locally and keeps no usage statistics, so monitoring means re-measuring, not dashboards.

- **What can drift.** Input: live conversation instead of read sentences, one person changing language inside a
  sentence (never measured: every language change in our mixed recordings is also a change of speaker), other
  microphones and rooms, phone audio, accents. Output: turns written in the wrong language, a rising share of turns
  flagged CHECK, repeated words, empty turns, a Kazakh or Russian turn written in English after the language
  detector called it English.
- **How to check.** The app now keeps consented real recordings with corrected transcripts ("Save as check
  example", stored in `data/real/` on the computer only); `python -m cleartalk_ml.real` scores every way of
  transcribing on them. Ten to fifteen minutes of real mixed Kazakh-Russian-English speech is the most valuable
  measurement still missing. After any change of weights, routes or decoding policy run
  `python -m cleartalk_ml.evaluate --system <name>` on the validation sets and `python scripts/check_round2_deployment.py`.
  The validation numbers in this file are the reference.
- **When to retrain or re-route.** (a) The large model's one-pass validation WER on mixed recordings rises above
  10.8% (reference 9.8%); (b) on the real check examples the
  per-language routes or the mixed setting are clearly worse than a ready model alone; (c) a better ready model for a
  language appears - re-run `scripts/select_round2.py`, which re-applies both rules on validation; (d) Kazakh on real
  broadcast-like speech is worse than with the starting model (validation KSC2 already is, by +0.9 points):
  route Kazakh back to the starting model by hand in `models/multilingual/profile.json` or retrain with Kazakh
  broadcast data.
- **How to retrain.** Reuse this pipeline from step 1: new data through `cleartalk_ml.corpora`, new seeded recipes,
  `scripts/run_training.py --base kazakh-turbo`, `scripts/select_round2.py`. Both test sets are spent: a new final
  claim needs a third frozen test set built the way `cleartalk_ml.round2` built test B.
- **Known weak spots to target next.** Kazakh broadcast speech (a frozen encoder cannot learn to hear better; the
  decoder adapters cost a little there); English, still behind the untouched large-v3-turbo; the model's own language
  token, which is unreliable (the app reads the label from the text instead); language-detector mistakes in the
  per-turn setting - sending uncertain turns to the recognizer of the mixed setting is the obvious next experiment
  and needs its own validation; real language switching inside a sentence.

## Wrap-up (Gate D)

- **Result.** The Kazakh fine-tune of Whisper large-v3-turbo with LoRA in the decoder only (2,129,920 trained
  parameters, 0.3%), one epoch on a laptop GPU, against its untouched starting model on test B, opened once:
  mixed 30.0% → 9.7% (-20.3 points, 95% -24.2…-16.3); fleurs-kk 12.2% → 11.3% (-0.8 points, 95% -1.8…+0.0); ksc2-kk 9.8% → 10.3% (+0.5 points, 95% -0.5…+1.7); fleurs-ru 6.9% → 5.2% (-1.7 points, 95% -2.6…-0.8); golos-ru 20.8% → 17.3% (-3.5 points, 95% -6.1…-1.2); fleurs-en 9.1% → 6.6% (-2.5 points, 95% -3.5…-1.5).
- **Against our small model of round 1 (same training data).** mixed 16.4% → 9.7% (-6.7 points, 95% -7.9…-5.5); fleurs-kk 22.5% → 11.3% (-11.1 points, 95% -13.6…-8.7); ksc2-kk 45.6% → 10.3% (-35.3 points, 95% -38.6…-32.1); fleurs-ru 10.8% → 5.2% (-5.6 points, 95% -7.3…-3.9); golos-ru 20.0% → 17.3% (-2.7 points, 95% -6.0…+0.6); fleurs-en 7.3% → 6.6% (-0.6 points, 95% -1.5…+0.2).
- **Against the untouched large-v3-turbo.** mixed 43.6% → 9.7% (-33.9 points, 95% -37.3…-30.1); fleurs-kk 23.3% → 11.3% (-11.9 points, 95% -14.3…-9.5); ksc2-kk 49.5% → 10.3% (-39.2 points, 95% -47.6…-31.8); fleurs-ru 3.7% → 5.2% (+1.5 points, 95% +0.6…+2.6); golos-ru 26.4% → 17.3% (-9.1 points, 95% -14.5…-3.9); fleurs-en 5.4% → 6.6% (+1.2 points, 95% +0.5…+2.1).
- **The app's pipeline on the mixed test recordings.** Language per turn with the new routes 9.1%; the
  "languages change inside turns" setting 8.3%; our large model in one pass
  9.7%.
- **Rules (decided on validation, before the test).** Routes {'kk': 'cleartalk-large', 'ru': 'cleartalk-large', 'en': 'turbo'}; recognizer for the
  mixed setting cleartalk-large.
- **Test agrees with validation?** Against the starting model the change has the same direction on
  mixed, fleurs-kk, ksc2-kk, fleurs-ru, golos-ru, fleurs-en. Sets where the whole 95% interval is
  below zero (better): mixed, fleurs-ru, golos-ru, fleurs-en; above zero (worse): none.
- **Language naming on the test.** Our large model's own language token is right for
  100.0% of fleurs-en, 24.1% of fleurs-kk, 98.2% of fleurs-ru, 74.7% of golos-ru, 20.0% of ksc2-kk. The app does not rely on it.
- **Not overfit, as far as measured.** Roles from the corpora's official parts, no shared utterance or FLEURS
  sentence, one epoch, validation WER not rising, a second test set frozen before training and scored once.
- **Limits.** Mixed recordings are assembled from single-language utterances (a language change is also a speaker
  change); read sentences and short phrases dominate; no speaker ids, so speaker overlap between roles cannot be
  excluded; 170 test units per set, so intervals are wide; the starting model's authors trained on KSC2 and do not
  say whether its test folders were held out, so comparisons with it on KSC2 may be tilted in its favour;
  overlapping voices are not handled in Conversation mode.

- Gate D: approved 2026-10-14 (round 2 wrap-up; delegated: the user asked on 2026-10-13 for every stop of this training run, the final test included, to be passed unattended and recorded for her review)

## Decisions

- 2026-10-13 · Test B is smaller than test A (170 instead of 300 per set) because FLEURS English has only 347
  test utterances left unused. Intervals will be wider; the effects looked for here are large.
- 2026-10-13 · More Kazakh training speech for the SMALL model (the other improvement the user chose) is not part
  of this round: the only open source of Kazakh broadcast speech is the 81 GB KSC2 archive, which can only be read
  from its beginning, so reaching those folders needs a download well above the 10-20 GB that was quoted to her.
  It waits for her answer.
- 2026-10-14 · Learning-rate pilots (80 steps each, greedy validation subset of mixed recordings): 1.5e-4 gave 11.8% WER, 5e-4 gave 17.8%; the starting model 56.4%. 1.5e-4 was continued as the main run (the rule: lower validation WER, a tie to the lower rate).
- 2026-10-14 · The evaluation at the end of the second pilot took 2 h 8 min instead of 9 min (23:43 to 01:51). The laptop did not sleep (no sleep events in the system log); the power source changed twice at 23:44-23:45, and during training the graphics card showed 5.5-5.95 GB of 6 GB in use. The likely cause is graphics memory: training leaves several gigabytes cached, the evaluation allocates blocks of other sizes, the total passes the card's memory and Windows spills into shared memory without an error. The pilot's numbers are not affected (only the time). Fix, applied at step 125 of the main run by stopping it right after a saved state and resuming: cached blocks are released before, during and after every evaluation, and evaluation batches are 4 instead of 8. Neither changes what is trained or any measured number. After the restart the card showed 4.75 GB in use.
- 2026-10-14 · The checkpoints shortlisted for the app-path validation were the last one (step 900) and the one the greedy monitor liked best (step 600), as in round 1. On the greedy monitor step 900 was 0.1 points lower on mixed recordings than step 600 - less than the monitor's minimum improvement, so its best stayed step 600; through the app's decoding step 600 is 0.1 points ahead. The difference is noise; step 600 is used because the rule (lower WER through the app's decoding) says so.
- 2026-10-15 · Two decoding options of the app, tested after Gate D on VALIDATION only (no test set is left for mixed recordings: test B is spent and test C has none). Rules written before running them: (1) 'mixed_skip_detection' - in the 'languages change inside turns' setting the language detector (large-v3-turbo, one extra encoder pass per turn) is not run and our large model chooses its own token; adopted if its WER on the 300 mixed validation recordings is at most 0.3 points above the setting as validated (8.16%); the reason is speed on a computer without a graphics card. (2) 'uncertain_to_mixed' - with the language detected per turn, a turn whose detection is uncertain (share below 0.6, or shorter than 2 s) goes to the recognizer of the mixed setting instead of its language's route; adopted if its WER on the mixed validation recordings is lower than the validated per-turn pipeline (9.87%) with the 95% bootstrap interval of the difference below zero, and on the 394 English FLEURS validation utterances through the pipeline it is not more than 0.5 points worse.
- 2026-10-15 · Results of the two decoding options (validation): (1) the mixed setting without the language detector 8.16% -> 8.23% on the 300 mixed recordings (+0.07 points, 95% -0.03 to +0.18) - within the 0.3-point allowance: ADOPTED (one large encoder pass less per turn). (2) uncertain turns to the recognizer of the mixed setting: per-turn pipeline 9.87% -> 7.82% on the mixed recordings (-2.05 points, 95% -3.02 to -1.19); English FLEURS through the pipeline 6.13% -> 6.12%: ADOPTED. Both are now on in models/multilingual/profile.json ('options'). These are validation numbers only; the test B numbers above were measured before.
- 2026-10-15 · A third decoding option, from the robustness check (validation): short Kazakh phrases were sometimes detected as English with confidence, routed to large-v3-turbo and written as invented English. Option 'cross_check': with the language detected per turn, a turn routed to large-v3-turbo is also written by our large model; if our model heard Kazakh or Russian (not the detected language), its text is kept. Rule written before measuring: adopted if on the robustness validation recordings (mixed recordings and short phrases; clean, white noise 10 dB, babble 10 dB) the pooled WER is lower with the 95% bootstrap interval of the difference below zero, and English FLEURS validation through the pipeline is not more than 0.5 points worse.
- 2026-10-15 · Result of 'cross_check' (validation): robustness recordings 23.45% -> 22.18% (-1.27 points, 95% -4.63 to +0.63; 750 units) - the interval includes zero, so by the rule it is NOT adopted (English FLEURS 6.12% -> 6.36%). By set: short Kazakh phrases improved (clean 18.6% -> 13.7%, white noise 10 dB 35.4% -> 24.4%) and short English phrases in babble (46.8% -> 17.4%), but short English phrases got worse when clean (5.8% -> 7.8%) and in white noise (13.2% -> 15.2%): our model writes some English phrases in Cyrillic, and the check then replaced a correct English transcript. The option stays off.

## Figures
- Figure 02_starting_points_validation.png: Where this round starts, on validation. The Kazakh turbo fine-tune is far ahead on Kazakh (FLEURS 12.2%, KSC2 9.4%) but behind the untouched large-v3-turbo on Russian (7.3% vs 4.0%) and English (9.7% vs 5.6%); it was never scored on Golos or on mixed recordings (it is routed Kazakh only). On mixed recordings in one pass: Whisper small (untouched) 60.0%, Our small model (round 1) 16.1%, large-v3-turbo (untouched) 47.7%. The large models have the acoustic quality; what they lack is the habit of following a language change, which the decoder decides - the reason to train the decoder only.
- Figure 06_test_b.png: Test B: 1,020 units, 170 per set, frozen by fingerprint before any training of this round. Every unit comes from test-role utterances that the first final test never transcribed; shared with test A, validation or training: 0. It is smaller than test A (300 per set) because only 347 unused FLEURS English test utterances are left. Cleaning rules, training examples and validation sets are unchanged from round 1.
- Figure 07_logmel_large_model.png: The large model's input is Whisper's fixed log-mel spectrogram with 128 bands × 3000 frames (the small model uses 80 bands). It is a fixed formula with no statistics estimated from our data, so nothing can leak from validation or test. The training examples themselves are the 14,400 of round 1.
- Figure 08_target_tokens_large_model.png: Targets with the large model's tokenizer (the same vocabulary as large-v3-turbo; token ids of the special tokens differ from the small model's): median 45 tokens for a single utterance, 99 for a mixed recording, longest 231; 0 reach the limit. The target format is round 1's: one language token per window, half of the examples with timestamps, language and task tokens left out of the loss.
- Figure 09_baselines.png: An empty transcript scores 100% WER. The baseline of this round is the starting model untouched (the Kazakh turbo fine-tune) through the app's decoding: fleurs-kk 12.2%, ksc2-kk 9.4%, fleurs-ru 7.3%, golos-ru 16.8%, fleurs-en 9.7%; mixed recordings in one pass 30.9% (words of each language inside them: Kazakh 23%, Russian 17%, English 43%). Reference points on mixed recordings: large-v3-turbo (untouched) 47.7%, Whisper small (untouched) 60.0%, Our small model (round 1) 16.1%, App pipeline (round-1 routes) 10.8%. The starting model hears Kazakh well but, in a recording that changes language, keeps writing in the language it started with: that habit belongs to the decoder, which is the part this round trains.
- Figure 11_learning_rate_pilots.png: Identical 80-step runs of the large model that differ only in learning rate, compared on a fixed validation subset of mixed recordings (greedy decoding): 0.00015 → 11.8%, 0.0005 → 17.8%; FLEURS kk/ru/en after the pilot: 0.00015: 10.5/6.0/11.1; 0.0005: 10.3/4.3/11.4. large-lr1.5e-4 was continued as the full run; test B played no part.
- Figure 10_learning_curves.png: Greedy decoding on fixed subsets. Mixed recordings: validation WER 56.4% (step 0 = the untouched starting model) → 9.1% at step 900; the training subset 53.2% → 10.9%. Relative change: training -79% vs validation -84%; validation loss 0.636 → 0.393 (lowest 0.387 at step 600; training-subset loss 0.654 → 0.222). Single-language subsets: fleurs-en 17.2% → 11.1%, fleurs-kk 11.5% → 10.9%, fleurs-ru 6.3% → 4.5%, golos-ru 17.6% → 9.9%, ksc2-kk 7.7% → 9.1%. A training curve that keeps falling while the validation curve turns up would mean overfitting.
- Figure 12_validation_comparison.png: Through the app's exact decoding, one-pass validation WER on mixed recordings: starting model 30.9% → our large model 9.8% (step 600; difference -21.1 points, 95% bootstrap -24.1 to -18.0); against our small model of round 1 (16.1%): -6.3 points, interval -7.3 to -5.3, 300 recordings. With the language given, starting model → our large model: fleurs-kk 12.2% → 10.6%, ksc2-kk 9.4% → 10.3%, fleurs-ru 7.3% → 5.3%, golos-ru 16.8% → 14.2%, fleurs-en 9.7% → 6.9%. The app's pipeline on the same recordings: Pipeline, round-1 routes 10.8%, Pipeline, new routes 9.9%, Pipeline, mixed setting 8.2%. Rule 1 (routes, lowest pooled validation WER per language): {'kk': 'cleartalk-large', 'ru': 'cleartalk-large', 'en': 'turbo'}. Rule 2 (recognizer for the mixed setting, lowest one-pass WER on these recordings): cleartalk-large.
- Step 14 final test: unit_wer=0.0892 (touch 1, 2026-10-14)
- Figure 13_error_analysis_validation.png: On the mixed validation recordings (one pass, automatic language) our large model makes fewer errors than the starting model on 202 of 300 recordings (the same on 66, more on 32); than the our small (round 1) on 216 of 300 recordings (the same on 40, more on 44). Kinds of errors, starting model → our small → our large: wrong words 859 → 1,366 → 798, missing words 2,416 → 272 → 170, extra words 114 → 128 → 106. By place in the recording: utterance 1 15% → 17% → 10%, utterance 2 46% → 15% → 9%, utterance 3+ 33% → 16% → 11%. Words written in the other alphabet: Kazakh 0.0% → 0.0% → 0.0%, Russian 0.1% → 0.1% → 0.1%, English 3.9% → 0.0% → 0.2%. Kazakh KSC2 by source: crowdsourced 6% → 29% → 7%, parliament 8% → 31% → 8%, podcasts 17% → 59% → 18%, radio 20% → 70% → 19%, talkshow 13% → 58% → 16%, tv_news 4% → 38% → 5%. The language our large model detects for a mixed recording is its first utterance's in 34% of cases (WER 10.5% then, 9.4% otherwise); on single utterances it names the language correctly for 24% of fleurs-kk, 99% of fleurs-ru, 100% of fleurs-en.
- Figure 14_final_test_b.png: Final test on test B, opened once, the same frozen audio for every system (1020 units, 170 per set). Our large model against its untouched starting model: mixed 30.0% → 9.7% (-20.3 points, 95% bootstrap -24.2 to -16.3); fleurs-kk 12.2% → 11.3% (-0.8 points, 95% bootstrap -1.8 to +0.0); ksc2-kk 9.8% → 10.3% (+0.5 points, 95% bootstrap -0.5 to +1.7); fleurs-ru 6.9% → 5.2% (-1.7 points, 95% bootstrap -2.6 to -0.8); golos-ru 20.8% → 17.3% (-3.5 points, 95% bootstrap -6.1 to -1.2); fleurs-en 9.1% → 6.6% (-2.5 points, 95% bootstrap -3.5 to -1.5). Against our small model of round 1: mixed 16.4% → 9.7% (-6.7 points, 95% bootstrap -7.9 to -5.5); fleurs-kk 22.5% → 11.3% (-11.1 points, 95% bootstrap -13.6 to -8.7); ksc2-kk 45.6% → 10.3% (-35.3 points, 95% bootstrap -38.6 to -32.1); fleurs-ru 10.8% → 5.2% (-5.6 points, 95% bootstrap -7.3 to -3.9); golos-ru 20.0% → 17.3% (-2.7 points, 95% bootstrap -6.0 to +0.6); fleurs-en 7.3% → 6.6% (-0.6 points, 95% bootstrap -1.5 to +0.2). Against the untouched large-v3-turbo: mixed 43.6% → 9.7% (-33.9 points, 95% bootstrap -37.3 to -30.1); fleurs-kk 23.3% → 11.3% (-11.9 points, 95% bootstrap -14.3 to -9.5); ksc2-kk 49.5% → 10.3% (-39.2 points, 95% bootstrap -47.6 to -31.8); fleurs-ru 3.7% → 5.2% (+1.5 points, 95% bootstrap +0.6 to +2.6); golos-ru 26.4% → 17.3% (-9.1 points, 95% bootstrap -14.5 to -3.9); fleurs-en 5.4% → 6.6% (+1.2 points, 95% bootstrap +0.5 to +2.1). Other points: Whisper small (untouched) mixed 59.9%, fleurs-kk 74.4%, ksc2-kk 102.7%, fleurs-ru 9.9%, golos-ru 39.3%, fleurs-en 7.6%; App pipeline, language per turn mixed 9.1%; App pipeline, mixed setting mixed 8.3%. Reported as measured; the checkpoint, the routes and the recognizer for the mixed setting were fixed on validation before this run.
