import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log

log.set_step(4, "done",
             "training utterances 20,979 → 20,767 (99.0%): same sentence held out 166, longer than 30 s 18, audio/text length "
             "mismatch 16, shorter than 1 s 6, demo 6; validation/test lose only the 98 empty transcripts; plain transcripts "
             "(Golos, KSC2) get sentence form as targets")
log.set_step(5, "done",
             "frozen seeded recipes of mixed-language recordings: 10,000 train / 300 validation / 400 test, 2-4 utterances "
             "each (mean 2.5), median 23.3 s, neighbouring parts always in different languages, each role built from its own "
             "utterances only; recipe checks passed (unique ids, order, ≤ 29 s)")
log.set_step(6, "done",
             "official corpus parts; KSC2 official Test divided into validation and test halves by a seeded draw inside every "
             "source; 0 shared utterances and 0 shared FLEURS sentences between roles; 25,340 training examples per epoch, "
             "2,219 validation units, 3,878 test units frozen in .guard_state.json")
log.set_step(7, "done", "Whisper's fixed log-mel spectrogram, 80 bands × 3000 frames per 30 s window; no fitted statistics")
log.set_step(8, "done",
             "Whisper BPE targets (median 44 tokens single, 99 mixed, longest 231, none cut); 50% keep timestamps, one timed "
             "segment per utterance; one language token per mixed recording (first language in 70%)")

text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
report = '''## Gate B report (steps 4-8)

- **What was done.** (4) Cleaning by five rules, each with before/after counts. (5) Mixed-language recordings
  were assembled from frozen seeded recipes. (6) Roles were fixed from the corpora's official parts, checked for
  overlap, and the test units were frozen by fingerprint. (7) Features: Whisper's own log-mel spectrogram.
  (8) Targets: Whisper tokens, half of them with timestamps.
- **What was found.** Cleaning removes 1% of training utterances; the largest group (166) are sentences that
  also occur in validation or test. An early version of the speaking-rate rule would have removed 425 ordinary
  short Golos commands, so only impossible rates are removed now (16). A mixed recording holds 2.5 utterances on
  average because FLEURS sentences are long (10-13 s). No target reaches the token limit.
- **Decisions and why.**
  - Plain transcripts get sentence form (capital letter, full stop) so that the model learns one writing style
    instead of switching style with the corpus; inner punctuation is not invented.
  - Per epoch: every cleaned FLEURS training utterance (8,340), 3,000 Golos and 4,000 KSC2 utterances (seeded
    samples) and 10,000 mixed recordings. Sampling, not "everything": Golos and KSC2 would otherwise outweigh
    English, which has only FLEURS.
  - A mixed recording is given ONE language token (as the app does after detecting one language per window):
    the first part's language in 70% of recordings, another language it contains otherwise. The loss ignores
    the language and task tokens, so the model is not trained to guess them.
  - Validation and test are never filtered for quality; only the 98 Golos test recordings with an empty
    transcript are left out because they cannot be scored.
- **Known weaknesses kept in view.** Speaker overlap between roles cannot be excluded (no speaker ids). Every
  language change in a mixed recording is also a speaker and microphone change. KSC2 training speech is only
  the crowdsourced source, while KSC2 validation/test also hold broadcast, parliament, podcasts and talk shows:
  those are a deliberate out-of-domain check.
- **What comes next (steps 9-12).** Baselines on validation (empty transcript, Whisper small, the ready large
  models, the app's Conversation pipeline); LoRA training with learning-rate pilots; checkpoint choice by the
  app's decoding on validation; validation comparison with bootstrap intervals.
- **The question - "may I continue to training?"** Delegated in advance by the user on 2026-10-13.

Model rationale:
- traits: speech-to-text sequence task in three languages; 25,340 training examples per epoch from 20,767 utterances (not small); no datetime; repeated entities = sentences (kept inside one role) and speakers (ids not published); inputs are 30 s log-mel windows, 39% of them holding several languages; 22.7% / 22.1% / 14.7% of validation words (kk/ru/en) unseen in training text
- baseline: dummy = empty transcript (100% WER); untouched Whisper small through the app's decoding (the model being adapted); reference points: Whisper large-v3-turbo, the Kazakh turbo fine-tune and the app's Conversation pipeline built from them
- candidates: multilingual Whisper small + LoRA (rank 16, alpha 32, dropout 0.05) on all attention and feed-forward projections of encoder and decoder, float32; learning rate chosen on validation from {1.5e-4, 5e-4} by 100-step pilots with identical data and seed (the winner is resumed as the main run); training length chosen by validation with early stopping
- ruled out: full fine-tuning of 244 M weights (forgets the languages and domains not in our data, overfits read speech, does not fit 6 GB comfortably); training a recognizer from scratch (needs thousands of hours); LoRA on large-v3-turbo (gradients do not fit 6 GB); half precision on the GTX 1660 Ti unless a same-batch loss check agrees with float32 and it is faster (in CTranslate2 float16 was 3.5 times slower on this card); a text-only corrector for language switches (the errors are acoustic and lexical); a random split of utterances (the same sentence, and possibly speaker, on both sides)
- metric: pooled WER on mixed validation recordings through the app's decoding with automatic language (primary); WER per language on FLEURS validation with the language given (guards); CER, errors per language inside mixed recordings, language detection accuracy and other-domain WER (supporting)

- Gate B: approved 2026-10-13 (delegated as above; recorded after the step 4-8 figures and counts were written)

'''
assert marker in text and "## Gate B report" not in text
log.LOG.write_text(text.replace(marker, report + marker, 1), encoding="utf-8")
log.note("Machine check: PyTorch 2.8.0 with CUDA 12.6 runs on the GTX 1660 Ti after the NVIDIA driver update to the "
         "current Game Ready driver (the 2020 driver could not run it); no reboot was needed.")
log.note("CTranslate2 on the GTX 1660 Ti, same 18 validation utterances: float32, float16 and int8_float32 give identical "
         "transcripts; int8_float32 is the fastest (about 15x real time for large-v3-turbo, 2.3 GB) and float16 the slowest "
         "(2.6x). The app and all validation scoring on this laptop use int8_float32 on the graphics card; on a CPU, int8.")
log.note("The Kazakh fine-tune's tokenizer settings were written by a newer transformers release than the pinned one and "
         "do not load; its vocabulary and added tokens are identical to openai/whisper-large-v3-turbo (checked token by "
         "token), so the converted model carries the original tokenizer file. Its weights are unchanged.")
log.note("Speaker labels of Conversation mode (not part of the trained model): clustering threshold 0.70 chosen on 40 "
         "assembled validation conversations (0.2% of speech time given to the wrong speaker, speaker count right in 97.5%); "
         "on the other 40: 0.0% and 100%. Speakers in these conversations always differ in language or gender - an easy "
         "case; same-gender speakers of one language could not be tested without speaker ids.")
print("logged")
