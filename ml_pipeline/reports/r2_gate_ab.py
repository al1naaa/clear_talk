"""Round 2: write steps 1-8 and the reports of stops A and B into ml_pipeline/round2/PIPELINE.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log
from cleartalk_ml import round2

log.use(round2.R2)
log.set_step(1, "done", "no new speech data: the 44,514 utterances inspected in round 1 (no leakage suspects, no speaker ids); "
                        "new in this round are the starting model and the second test set")
log.set_step(2, "done", "starting points on validation: Kazakh turbo fine-tune kk 12.2% (FLEURS) / 9.4% (KSC2), ru 7.3%, en 9.7%; "
                        "untouched large-v3-turbo ru 4.0%, en 5.6%; mixed recordings in one pass: small 60.0%, large-v3-turbo 47.7%, "
                        "our round-1 model 16.1%")
log.set_step(3, "approved-gate", "see \"Problem contract\" above (what changes against round 1)")
log.set_step(4, "done", "cleaning rules of round 1, unchanged (20,767 training utterances kept)")
log.set_step(5, "done", "training recipes of round 1, unchanged (9,000 mixed recordings per epoch); 170 new seeded recipes for the mixed "
                        "recordings of test B, built only from unused test utterances")
log.set_step(6, "done", "roles of round 1; test B frozen: 1,020 units (170 per set), none shared with test A, validation or training")
log.set_step(7, "done", "Whisper's fixed log-mel spectrogram with 128 bands × 3000 frames (the large model's input); nothing fitted")
log.set_step(8, "done", "targets with the large model's tokenizer: median 45 tokens single / 99 mixed, longest 231, none cut; same "
                        "format as round 1")

text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
report = '''## Gate A and Gate B report (steps 1-8)

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

'''
assert marker in text and "## Gate A and Gate B report" not in text
log.LOG.write_text(text.replace(marker, report + marker, 1), encoding="utf-8")
print("logged")
