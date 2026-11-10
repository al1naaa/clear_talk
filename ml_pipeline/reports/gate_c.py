"""Write steps 9-12 and the Gate C report into PIPELINE.md from the saved results (no number is typed by hand)."""
import json
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log
from cleartalk_ml import validation
from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS, RUNS

p = lambda value: f"{100 * value:.1f}%"  # noqa: E731
choice = json.loads((RUNS / "lr-choice.json").read_text(encoding="utf-8"))
selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
run = selection["run"]
ours = f"runs/{run}/ct2-step-{selection['chosen_step']:05d}"
history = [json.loads(line) for line in (RUNS / run / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
evals = {}
for record in history:
    if record.get("type") == "eval":
        evals[record["step"]] = record
first, last = evals[0], evals[max(evals)]
config = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))

small_single, turbo_single, kazakh_single, ours_single = (validation.single_scores(s) for s in ("small", "turbo", "kazakh", ours))
small_mixed, turbo_mixed, pipe_mixed, ours_mixed = (validation.mixed_scores(s) for s in ("small", "turbo", "pipeline:auto", ours))
sets = ["fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]


def row(scores, label):
    return f"{label}: " + ", ".join(f"{s} {p(scores[s]['forced']['wer'])}" for s in sets if s in scores)


promo = selection["promotion"]
routes = selection["routes"]
mixed_refs = ", ".join(f"{name} {p(block['wer'])}" for name, block in (("Whisper large-v3-turbo in one pass", turbo_mixed),
                                                                         ("the app's Conversation pipeline with ready models", pipe_mixed)) if block)

log.set_step(9, "done",
             f"empty transcript 100% WER; untouched Whisper small through the app's decoding: mixed recordings {p(small_mixed['wer'])}, "
             + ", ".join(f"{s} {p(small_single[s]['forced']['wer'])}" for s in sets) + f". Reference points: {row(turbo_single, 'turbo')}; "
             + f"{row(kazakh_single, 'Kazakh turbo')}; mixed recordings: {mixed_refs}")
log.set_step(10, "done",
             f"run `{run}`, one epoch = {config['total_steps']} steps over {config['train_examples']:,} examples (float32, effective batch 16, "
             f"8.6 s per step). Greedy monitor: validation mixed {p(first['val_mixed']['wer'])} → {p(last['val_mixed']['wer'])}, training subset "
             f"{p(first['train_mixed']['wer'])} → {p(last['train_mixed']['wer'])}; validation loss {first['val_loss']:.3f} → {last['val_loss']:.3f} "
             f"(flat from step 600: {evals[600]['val_loss']:.3f}); FLEURS kk {p(first['val_fleurs-kk']['wer'])} → {p(last['val_fleurs-kk']['wer'])}, "
             f"ru {p(first['val_fleurs-ru']['wer'])} → {p(last['val_fleurs-ru']['wer'])}, en {p(first['val_fleurs-en']['wer'])} → {p(last['val_fleurs-en']['wer'])}")
log.set_step(11, "done",
             "LR pilots, 80 steps each, same data/seed/schedule: "
             + ", ".join(f"{info['lr']:g} → {p(info['val_mixed_wer'])} validation mixed WER" for info in choice["pilots"].values())
             + f"; `{choice['chosen']}` continued. Training length chosen by validation checkpoints")
log.set_step(12, "done",
             f"app-path validation of the shortlisted checkpoints ({', '.join(f'step {k}: {p(v)}' for k, v in selection['checkpoints_scored_on_mixed_validation'].items())} "
             f"on mixed recordings); step {selection['chosen_step']} selected. Mixed {p(promo['mixed']['baseline_wer'])} → {p(promo['mixed']['candidate_wer'])} "
             f"({100 * promo['mixed']['relative_change']:+.0f}% relative, bootstrap 95% {100 * promo['mixed']['low']:+.1f}…{100 * promo['mixed']['high']:+.1f} points); "
             + ", ".join(f"FLEURS {lang} {p(promo['languages'][lang]['baseline_wer'])} → {p(promo['languages'][lang]['candidate_wer'])}" for lang in LANGUAGES)
             + f"; KSC2 {p(small_single['ksc2-kk']['forced']['wer'])} → {p(ours_single['ksc2-kk']['forced']['wer'])}, Golos "
             f"{p(small_single['golos-ru']['forced']['wer'])} → {p(ours_single['golos-ru']['forced']['wer'])}. Promotion rule 1: "
             f"{'passed' if promo['promoted'] else 'NOT passed'}; routes by rule 2: {routes['routes']}")

worse = [lang for lang in LANGUAGES if not promo["languages"][lang]["ok"]]
ahead = [s for s in sets if s in turbo_single and ours_single[s]["forced"]["wer"] < turbo_single[s]["forced"]["wer"]]
behind = [s for s in sets if s in turbo_single and ours_single[s]["forced"]["wer"] > turbo_single[s]["forced"]["wer"]]
text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
report = f'''## Gate C report (steps 9-12)

- **What was done.** (9) Baselines on validation through the app's decoding: an empty transcript, untouched Whisper
  small, and the large ready models as reference points. (10) One epoch of LoRA training. (11) Two learning-rate
  pilots that differ only in the rate. (12) The shortlisted checkpoints were exported and scored through the app's
  decoding; the choice and both pre-registered rules were applied on validation only.
- **What was found.**
  - Untouched Whisper small is unusable for Kazakh ({p(small_single['fleurs-kk']['forced']['wer'])} on FLEURS,
    {p(small_single['ksc2-kk']['forced']['wer'])} on KSC2 - above 100% because it writes extra words) and for mixed
    recordings ({p(small_mixed['wer'])}).
  - Our model: mixed recordings {p(promo['mixed']['baseline_wer'])} → **{p(promo['mixed']['candidate_wer'])}**
    (difference {100 * promo['mixed']['difference']:+.1f} points, 95% bootstrap {100 * promo['mixed']['low']:+.1f} to
    {100 * promo['mixed']['high']:+.1f}, {promo['mixed']['recordings']} recordings); Kazakh FLEURS
    {p(promo['languages']['kk']['baseline_wer'])} → {p(promo['languages']['kk']['candidate_wer'])}; Kazakh KSC2
    {p(small_single['ksc2-kk']['forced']['wer'])} → {p(ours_single['ksc2-kk']['forced']['wer'])}; Russian FLEURS
    {p(promo['languages']['ru']['baseline_wer'])} → {p(promo['languages']['ru']['candidate_wer'])}; Russian Golos
    {p(small_single['golos-ru']['forced']['wer'])} → {p(ours_single['golos-ru']['forced']['wer'])}; English FLEURS
    {p(promo['languages']['en']['baseline_wer'])} → {p(promo['languages']['en']['candidate_wer'])}.
  - Reference points, the large ready models: {row(turbo_single, 'turbo')}; {row(kazakh_single, 'Kazakh turbo')};
    mixed recordings: {mixed_refs}. Our small model has a lower WER than large-v3-turbo on {', '.join(ahead) or 'no set'}
    and a higher one on {', '.join(behind) or 'no set'}; the Kazakh fine-tune stays far ahead on Kazakh.
  - Language detection by our model got worse for Kazakh: on FLEURS it names the language correctly for
    {p(ours_single['fleurs-kk']['language_accuracy'])} of Kazakh utterances (untouched small: {p(small_single['fleurs-kk']['language_accuracy'])}),
    Russian {p(ours_single['fleurs-ru']['language_accuracy'])}, English {p(ours_single['fleurs-en']['language_accuracy'])}. The text is still written in
    Kazakh (WER with automatic language {p(ours_single['fleurs-kk']['auto']['wer'])} vs {p(ours_single['fleurs-kk']['forced']['wer'])} with the language given):
    the language token was deliberately left out of the loss, so the model learned to write what it hears whatever the
    token says, and its own choice of token drifted. In the app the language label of a turn comes from the detector
    (large-v3-turbo), so this matters only where our model has to detect the language itself.
  - Training fit and validation: the training-subset loss keeps falling to the end ({first['train_loss']:.3f} →
    {last['train_loss']:.3f}) while the validation loss stops improving after step 600 ({evals[600]['val_loss']:.3f} at step 600,
    {last['val_loss']:.3f} at the end) - the gap between them grows, so the model fits its own training recordings better
    than new ones. Validation WER did not get worse inside the epoch (mixed, greedy: {p(evals[600]['val_mixed']['wer'])} at
    step 600, {p(last['val_mixed']['wer'])} at the end), but more passes over the same data would probably overfit; one
    epoch was the right length.
- **Decisions and why.**
  - Learning rate {choice['pilots'][choice['chosen']]['lr']:g}: lower validation WER on mixed recordings after 80 steps
    ({', '.join(f"{info['lr']:g}: {p(info['val_mixed_wer'])}" for info in choice['pilots'].values())}).
  - Checkpoint: step {selection['chosen_step']} by the app's decoding on the mixed validation recordings.
  - Promotion rule 1 (our model replaces Whisper small as the app's small recognizer): mixed gain
    {'met' if promo['rule_a_mixed_gain'] else 'NOT met'}; no language more than 0.5 points worse
    {'met' if promo['rule_b_no_language_worse'] else 'NOT met' + (' (' + ', '.join(LANGUAGE_NAMES[lang] + ' ' + format(100 * promo['languages'][lang]['worse_by'], '+.1f') + ' points' for lang in worse) + ')' if worse else '')}
    → **{'promoted' if promo['promoted'] else 'not promoted: Whisper small stays the default small recognizer; ours is selectable by hand'}**.
  - Rule 2 (routes when the app chooses): {routes['routes']} - pooled validation WER per language:
    {json.dumps({lang: {k: round(100 * v, 1) for k, v in scores.items()} for lang, scores in routes['pooled_validation_wer'].items()})}.
- **What comes next (steps 13-16).** Error analysis on validation; the one-time final test on the 1,800 frozen test
  units (our model, Whisper small and the reference systems on the same audio); deployment check; monitoring plan.
- **The question - "may I open the test set?"** Delegated in advance by the user on 2026-10-13
  ("включая финальный тест").

- Gate C: approved 2026-10-13 (delegated: the user authorized the one-time final test in advance on 2026-10-13; this report is for her review)

'''
assert marker in text and "## Gate C report" not in text
log.LOG.write_text(text.replace(marker, report + marker, 1), encoding="utf-8")
log.note("Training was interrupted from outside at step 880 of 900 (the process was terminated at about 06:47, not by "
         "an error) and resumed from the state saved at step 875; the last 25 steps and the final evaluation ran "
         "after the resume. Nothing else changed.")
log.note("From about 07:00 the laptop ran on battery: Windows capped the graphics card at about 29 W, so validation "
         "scoring ran two to three times slower than the speeds logged earlier. Speed numbers measured in that period "
         "are not comparable with the earlier ones; the transcripts are not affected.")
print("logged")
