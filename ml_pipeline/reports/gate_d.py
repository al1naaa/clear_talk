"""Write steps 13-16, the monitoring plan and the wrap-up into PIPELINE.md from the saved results.

Also adds the final-test numbers to the exported model's card (the card is what the app shows).
Run once, after the final test, the step 13/14 figures and the deployment check exist.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import analysis
from cleartalk_ml import pipeline_log as log
from cleartalk_ml.export import EXPORT
from cleartalk_ml.paths import LANGUAGE_NAMES, LANGUAGES, RESULTS

p = lambda value: f"{100 * value:.1f}%"  # noqa: E731
selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
final = json.loads((RESULTS / "test-final.json").read_text(encoding="utf-8"))
deploy = json.loads((RESULTS / "deployment-check.json").read_text(encoding="utf-8"))
ours = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
sets = ["mixed", "fleurs-kk", "fleurs-ru", "fleurs-en", "ksc2-kk", "golos-ru"]


def score(system, set_name):
    for item in final["scores"]:
        if item["system"] == system and item["set"] == set_name:
            return item["wer_auto"] if set_name == "mixed" else item["wer_forced"]
    return None


def line(set_name):
    block = final["cleartalk_vs_small"][set_name]
    return (f"{set_name} {p(score('small', set_name))} → {p(score('cleartalk', set_name))} ({100 * block['difference']:+.1f} points, "
            f"95% {100 * block['low']:+.1f}…{100 * block['high']:+.1f})")


moves = analysis.better_same_worse("mixed-val", "small", ours)
kinds = {name: analysis.edit_kinds("mixed-val", system) for name, system in (("small", "small"), ("ours", ours))}
sources = analysis.ksc2_by_source(ours)
small_sources = analysis.ksc2_by_source("small")
promo = selection["promotion"]
references = "; ".join(f"{name} " + ", ".join(f"{s} {p(score(key, s))}" for s in sets if score(key, s) is not None)
                       for key, name in (("turbo", "Whisper large-v3-turbo"), ("kazakh", "Kazakh turbo fine-tune"), ("pipeline:auto", "app pipeline with ready models")))
worse_on_test = [s for s in sets if final["cleartalk_vs_small"][s]["low"] > 0]
better_on_test = [s for s in sets if final["cleartalk_vs_small"][s]["high"] < 0]

log.set_step(13, "done",
             f"validation: on {moves['units']} mixed recordings better on {moves['better']}, same on {moves['same']}, worse on {moves['worse']}; "
             f"missing words {kinds['small']['deletions']:,} → {kinds['ours']['deletions']:,} (the untouched model drops whole utterances after a "
             f"language change), extra words {kinds['small']['insertions']:,} → {kinds['ours']['insertions']:,}, wrong words "
             f"{kinds['small']['substitutions']:,} → {kinds['ours']['substitutions']:,}; KSC2 by source: "
             + ", ".join(f"{s} {p(small_sources[s]['wer'])} → {p(sources[s]['wer'])}" for s in sorted(sources)))
log.set_step(14, "done (once)", "; ".join(line(s) for s in sets) + f". Reference points: {references}")
log.set_step(15, "done",
             f"the validated checkpoint folder was copied as `models/multilingual/{EXPORT.name}` (same SHA-256), signatures of "
             f"{len(deploy['signatures_verified'])} recognizers pinned in `models/multilingual/profile.json`; routes {deploy['profile_routes']}; the "
             f"service-loaded model reproduces {deploy['validated_transcripts_reproduced']['mixed']} mixed and "
             f"{deploy['validated_transcripts_reproduced']['single']} single validated transcripts; the API processed the three-language demo "
             f"with the routes and with our model alone ({deploy['api_demo']['routes']['turns']} and {deploy['api_demo']['cleartalk']['turns']} turns)")
log.set_step(16, "done", "see \"Monitoring and retraining plan\" below")

text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
report = f'''## Monitoring and retraining plan (step 16)

The app runs locally and keeps no usage statistics, so monitoring means re-measuring, not dashboards.

- **What can drift.** Input: live conversation instead of read sentences, one person changing language inside a
  sentence, other microphones and rooms, phone audio, accents, speakers of the same gender and language (the
  speaker labels were checked only on clearly different voices). Output: turns written in the wrong language,
  a rising share of turns flagged CHECK, repeated words, empty turns.
- **How to check.** Keep a small consented set of real recordings with human transcripts - 20-30 clips per
  language, and above all real mixed Kazakh-Russian-English speech, which no open corpus provides. After any change
  of weights, routes or decoding policy run `python -m cleartalk_ml.evaluate --system <name>` on the validation sets
  and `python scripts/check_multilingual_deployment.py`. The validation numbers in this file are the reference.
- **When to retrain or re-route.** (a) Our model's validation WER on mixed recordings rises above
  {p(promo['mixed']['candidate_wer'] * 1.1)} (reference {p(promo['mixed']['candidate_wer'])}); (b) any language's FLEURS validation WER
  becomes more than 0.5 points worse than untouched Whisper small; (c) a better ready model for a language appears -
  re-run `scripts/select_multilingual.py`, which re-applies the routing rule on validation; (d) WER on the consented
  real set is clearly worse than the ready models'.
- **How to retrain.** Reuse this pipeline from step 1: new data go through `cleartalk_ml.corpora`,
  `cleartalk_ml.mixed plan` (new recipes, new seeds), `ml_pipeline/02_prep.py`, `scripts/run_training.py`,
  `scripts/select_multilingual.py`. The test units are spent: a new final claim needs a new frozen test set.
- **Known weak spots to target next.** Kazakh outside read speech (broadcast, parliament, talk shows); English, the
  only language with a single small source in training; real language switching; a larger base model (LoRA on
  large-v3-turbo needs a 16 GB GPU).

## Wrap-up (Gate D)

- **Result.** A LoRA-adapted multilingual Whisper small (6,488,064 trained parameters, 2.6%), trained for one epoch
  on a laptop GPU, against the untouched model on the frozen test audio, opened once: {"; ".join(line(s) for s in sets)}.
- **Where it stands against ready large models.** {references}. Routes chosen on validation for the app:
  {selection['routes']['routes']}.
- **Promotion rule 1 (decided on validation, before the test).** Mixed gain {'met' if promo['rule_a_mixed_gain'] else 'not met'};
  no language more than 0.5 points worse {'met' if promo['rule_b_no_language_worse'] else 'not met'} →
  {"our model is the app's small recognizer" if promo["promoted"] else "Whisper small stays the default small recognizer; ours is selectable by hand"}.
- **Test agrees with validation?** Sets where our model is better on the test with the whole interval below zero:
  {', '.join(better_on_test) or 'none'}; worse with the whole interval above zero: {', '.join(worse_on_test) or 'none'}.
- **Not overfit, as far as measured.** Roles from the corpora's official parts, no shared utterance or FLEURS
  sentence, one epoch, training and validation curves falling together, validation loss flat at the end, and a
  test set scored once.
- **Limits.** Mixed recordings are assembled from single-language utterances (a language change is also a speaker
  change); read sentences and short phrases dominate; no speaker ids, so speaker overlap between roles cannot be
  excluded; 300 test units per set; overlapping voices are not handled in Conversation mode.

- Gate D: approved 2026-10-13 (wrap-up; delegated: the user asked on 2026-10-13 for every stop, the final test included, to be passed unattended and recorded for her review)

'''
assert marker in text and "## Wrap-up (Gate D)" not in text
log.LOG.write_text(text.replace(marker, report + marker, 1), encoding="utf-8")

# The app shows the model card: add the one-time test numbers to it.
card_path = EXPORT / "model_card.json"
card = json.loads(card_path.read_text(encoding="utf-8"))
names = {"mixed": "mixed-language recordings", "fleurs-kk": "Kazakh read sentences", "fleurs-ru": "Russian read sentences",
         "fleurs-en": "English read sentences", "ksc2-kk": "Kazakh broadcast and read speech", "golos-ru": "short Russian phrases"}
card["test"] = {s: {"baseline_wer": score("small", s), "cleartalk_wer": score("cleartalk", s),
                    "units": final["cleartalk_vs_small"][s]["units"]} for s in sets}
tail = card["evidence"][-1]
card["evidence"] = [e for e in card["evidence"][:-1] if not e.startswith("Final test")]
card["evidence"] += [f"Final test, {names[s]} ({final['cleartalk_vs_small'][s]['units']}): {p(score('small', s))} Whisper small → "
                     f"{p(score('cleartalk', s))} ClearTalk." for s in sets]
card["evidence"].append(tail)
card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("logged; model card updated;", LANGUAGE_NAMES[LANGUAGES[0]])
