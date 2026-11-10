import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log

log.set_step(1, "done",
             "44,514 utterances in one table (FLEURS kk/ru/en 11,761; Golos ru 18,780; KSC2 kk 13,973), 85.9 h; no audio file "
             "missing; 98 empty transcripts (Golos test); 43 longer than 30 s; profile: no leakage suspects; no corpus has speaker ids")
log.set_step(2, "done",
             "FLEURS sentence ids shared between parts: 0; identical sentences in two parts: 159 (repeated Golos commands and KSC2 "
             "sentences); 74% of transcripts are plain lower-case text; almost no natural language mixing (Latin words 0.4%); "
             "22.7% / 22.1% / 14.7% of validation words (kk/ru/en) unseen in training text; ready models on FLEURS validation: "
             "turbo kk 26.2% / ru 4.0% / en 5.6% WER")
log.set_step(3, "approved-gate", "see \"Problem contract\" below")

text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
contract = '''## Problem contract (step 3)

- **Target:** the words spoken in the recording, each part written in its own language and script - Kazakh
  and Russian in Cyrillic, English in Latin letters - in the order they were spoken. Training target = the
  corpus transcript (FLEURS as published, with capitals and punctuation; Golos and KSC2 are published as plain
  lower-case text and are put in sentence form: capital first letter, full stop). Scoring reference = the
  published transcript.
- **Prediction unit:** one audio window of at most 30 s (Whisper's window): either one utterance in one
  language, or a mixed recording - two to four utterances in different languages one after another.
- **Population:** read sentences (FLEURS, three languages), Kazakh read and broadcast speech (KSC2), short
  Russian phrases (Golos), and mixed recordings assembled from them. Not covered: free conversation, one person
  changing language in the middle of a sentence, voices talking at the same time, telephone audio.
- **Information available at prediction time:** the audio only. The app either takes the language from the
  user (it is then given to the model as its language token) or detects it from the same audio; a mixed
  recording gets one language token although it holds several languages.
- **Objective:** fewer word errors than the untouched Whisper small on Kazakh and on mixed recordings, without
  making Russian or English worse.
- **Metric:** pooled WER = (substitutions + deletions + insertions) / reference words after one normalizer for
  all three languages (lower case, punctuation removed, ё→е). Primary: WER on the mixed validation recordings
  through the app's decoding with automatic language detection. Guards: WER per language on FLEURS validation
  with the language given. Supporting: character error rate, errors per language inside mixed recordings,
  language detection accuracy, WER on the other-domain sets (KSC2 broadcast, Golos phrases).
- **Model selection:** validation only. During training a small fixed validation subset is decoded greedily;
  the final checkpoint is chosen by the app's decoding on the full validation sets. Test sets are used once,
  at step 14.
- **Pre-registered promotion rules (decided before any training):**
  1. *Our model replaces Whisper small as the app's small recognizer* if, on validation, (a) its WER on mixed
     recordings is at least 10% lower relative to Whisper small and the 95% bootstrap interval of the
     difference lies entirely below zero, and (b) no language's FLEURS WER is more than 0.5 percentage points
     worse than Whisper small.
  2. *Which recognizer the app uses for each language when the user lets the app choose* ("routes"): for each
     language, the installed recognizer with the lowest pooled validation WER over that language's sets
     (FLEURS + KSC2 for Kazakh, FLEURS + Golos for Russian, FLEURS for English). Our model gets a route only
     where it is the best; otherwise it stays selectable by hand. Test results never enter either rule.
- **Constraints:** trainable overnight on one GTX 1660 Ti with 6 GB - Whisper small (244 M parameters) with
  LoRA; runs in the app through faster-whisper on a CPU (Mac) or a graphics card (Acer); nothing is uploaded.
- **Frozen evaluation sets (defined before any training):**
  - Validation: FLEURS dev (kk 369, ru 356, en 394 utterances); Golos validation (a seeded sample of 400);
    KSC2 (a seeded sample of 400 from the validation half of the official Test folders); 300 mixed recordings
    built from validation utterances.
  - Test: FLEURS test (kk 856, ru 775, en 647); Golos test (seeded sample of 600); KSC2 (seeded sample of 600
    from the test half of the official Test folders); 400 mixed recordings built from test utterances.

## Gate A report (steps 1-3)

- **What was done.** All downloaded speech was unpacked and listed in one table: 44,514 utterances, 85.9 hours,
  three languages, three corpora. The table was profiled, seven figures were drawn, and three ready recognizers
  were measured on FLEURS validation through the app's decoding.
- **What was found.** (1) The data are sound: no missing audio, 98 empty transcripts, 43 over-long utterances.
  (2) FLEURS parts share no sentence; 159 identical sentences occur in two parts of Golos or KSC2 (repeated
  commands and read sentences) and must leave training. (3) Two writing styles: FLEURS has capitals and
  punctuation, Golos and KSC2 do not. (4) There is no natural language mixing in the corpora, so mixed speech
  has to be assembled, and every language change will also be a speaker change. (5) No corpus has speaker ids:
  speaker overlap between parts cannot be checked. (6) Only the beginning of KSC2 is on disk: its complete
  official Test folders (six sources, 9,351 utterances) and 4,622 crowdsourced Train utterances; one source
  (tv_news) is published with doubled file extensions and was first read as empty. (7) Whisper large-v3-turbo
  is already strong in Russian (4.0% WER) and English (5.6%) and weak in Kazakh (26.2%).
- **Decisions and why.** Roles follow the corpora's official parts (no random split: the same sentence and
  possibly the same speaker would land on both sides). KSC2 has no Dev folders in the downloaded part, so its
  official Test utterances are divided once, by a seeded draw inside every source, into a validation half and
  a test half - both outside training. The FLEURS test part stays unopened until step 14; ready models are
  compared on validation. WER is reported with CER because Kazakh words are long and inflected.
- **What comes next (steps 4-8).** Cleaning rules with before/after counts; assembling mixed-language
  recordings from frozen seeded recipes; fixing the roles and freezing the test set by fingerprint; Whisper's
  fixed log-mel features; transcripts as Whisper tokens.
- **The question - "may I continue?"** Delegated in advance by the user on 2026-10-13 (see "Delegation").

- Gate A: approved 2026-10-13 (delegated: the user authorized passing every gate unattended on 2026-10-13; the contract and this report are for her review)

'''
assert marker in text and "## Problem contract" not in text
log.LOG.write_text(text.replace(marker, contract + marker, 1), encoding="utf-8")
print("logged")
