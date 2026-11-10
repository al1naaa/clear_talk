"""Round 2: write steps 14-16, the monitoring plan and the wrap-up into round2/PIPELINE.md from the saved results.

Also adds the final-test numbers to the exported model's card (the card is what the app shows).
Run once, after the final test on test B, its figure and the deployment check exist.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log
from cleartalk_ml import round2
from cleartalk_ml.paths import BASES, MODELS, RESULTS

log.use(round2.R2)
p = lambda value: f"{100 * value:.1f}%"  # noqa: E731
pts = lambda value: f"{100 * value:+.1f}"  # noqa: E731
selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
final = json.loads((RESULTS / "test-final-B.json").read_text(encoding="utf-8"))
deploy = json.loads((RESULTS / "deployment-check-round2.json").read_text(encoding="utf-8"))
export = MODELS / BASES["kazakh-turbo"]["export"]
sets = ["mixed", "fleurs-kk", "ksc2-kk", "fleurs-ru", "golos-ru", "fleurs-en"]
names = {"mixed": "mixed-language recordings", "fleurs-kk": "Kazakh read sentences", "fleurs-ru": "Russian read sentences",
         "fleurs-en": "English read sentences", "ksc2-kk": "Kazakh broadcast and read speech", "golos-ru": "short Russian phrases"}
compare = final["cleartalk_large_minus"]
validation = selection["report_against_starting_model"]
rule2 = selection["rule_2_mixed_recognizer"]


def score(system, set_name):
    for item in final["scores"]:
        if item["system"] == system and item["set"] == set_name:
            return item["wer_auto"] if set_name == "mixed" else item["wer_forced"]
    return None


def line(other, set_name):
    block = compare[other][set_name]
    return (f"{set_name} {p(score(other, set_name))} → {p(score('cleartalk-large', set_name))} ({pts(block['difference'])} points, "
            f"95% {pts(block['low'])}…{pts(block['high'])})")


def sure(other, better=True):
    return [s for s in sets if s in compare[other] and (compare[other][s]["high"] < 0 if better else compare[other][s]["low"] > 0)]


accuracy = {item["set"]: item["language_accuracy"] for item in final["scores"] if item["system"] == "cleartalk-large" and "language_accuracy" in item}
others = "; ".join(f"{label} " + ", ".join(f"{s} {p(score(key, s))}" for s in sets if score(key, s) is not None)
                   for key, label in (("turbo", "untouched large-v3-turbo"), ("cleartalk", "our small model of round 1"), ("small", "Whisper small"),
                                      ("pipeline:auto", "app pipeline, language per turn"), ("pipeline:mixed", "app pipeline, mixed setting")))
# Does the test tell the same story as validation, set by set (same sign of the change against the starting model)?
val_change = {"mixed": rule2["large_minus_starting_model"]["difference"], **{s: validation[s]["change"] for s in sets if s != "mixed"}}
agree = [s for s in sets if (val_change[s] < 0) == (compare["kazakh"][s]["difference"] < 0)]
disagree = [s for s in sets if s not in agree]
cpu = deploy.get("cpu_check") or {}
api = deploy["api_demo"]

log.set_step(14, "done (once)", "against the starting model: " + "; ".join(line("kazakh", s) for s in sets) + f". Other systems on the same audio: {others}")
log.set_step(15, "done",
             f"the validated checkpoint folder was copied as `models/multilingual/{export.name}` (same SHA-256), signatures of "
             f"{len(deploy['signatures_verified'])} recognizers pinned in `models/multilingual/profile.json`; routes {deploy['profile_routes']}, recognizer for "
             f"the mixed setting {deploy['served_mixed_recognizer']}; the service-loaded model reproduces {deploy['validated_transcripts_reproduced']['mixed']} "
             f"mixed and {deploy['validated_transcripts_reproduced']['single']} single validated transcripts"
             + (f"; on the CPU (int8) {cpu['units']} of those units took {cpu['processing_seconds']} s for {cpu['audio_seconds']} s of audio with WER "
                f"{p(cpu['wer_cpu_int8'])} (graphics card: {p(next(v for k, v in cpu.items() if k.startswith('wer_') and k != 'wer_cpu_int8'))})" if cpu else "")
             + f"; the API processed the three-language demo with the routes, with the mixed setting and with our large model alone "
             f"({api['routes']['turns']}, {api['mixed_setting']['turns']} and {api['cleartalk_large']['turns']} turns)")
log.set_step(16, "done", "see \"Monitoring and retraining plan\" below")

text = log.LOG.read_text(encoding="utf-8")
marker = "## Decisions"
report = f'''## Monitoring and retraining plan (step 16)

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
  {p(rule2['one_pass_mixed_validation_wer']['cleartalk-large'] * 1.1)} (reference {p(rule2['one_pass_mixed_validation_wer']['cleartalk-large'])}); (b) on the real check examples the
  per-language routes or the mixed setting are clearly worse than a ready model alone; (c) a better ready model for a
  language appears - re-run `scripts/select_round2.py`, which re-applies both rules on validation; (d) Kazakh on real
  broadcast-like speech is worse than with the starting model (validation KSC2 already is, by {pts(validation['ksc2-kk']['change'])} points):
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
  {"; ".join(line("kazakh", s) for s in sets)}.
- **Against our small model of round 1 (same training data).** {"; ".join(line("cleartalk", s) for s in sets)}.
- **Against the untouched large-v3-turbo.** {"; ".join(line("turbo", s) for s in sets)}.
- **The app's pipeline on the mixed test recordings.** Language per turn with the new routes {p(score("pipeline:auto", "mixed"))}; the
  "languages change inside turns" setting {p(score("pipeline:mixed", "mixed"))}; our large model in one pass
  {p(score("cleartalk-large", "mixed"))}.
- **Rules (decided on validation, before the test).** Routes {selection['rule_1_routes']['routes']}; recognizer for the
  mixed setting {rule2['chosen']}.
- **Test agrees with validation?** Against the starting model the change has the same direction on
  {', '.join(agree) or 'no set'}{' and the opposite direction on ' + ', '.join(disagree) if disagree else ''}. Sets where the whole 95% interval is
  below zero (better): {', '.join(sure('kazakh')) or 'none'}; above zero (worse): {', '.join(sure('kazakh', False)) or 'none'}.
- **Language naming on the test.** Our large model's own language token is right for
  {', '.join(f"{p(value)} of {name}" for name, value in sorted(accuracy.items()))}. The app does not rely on it.
- **Not overfit, as far as measured.** Roles from the corpora's official parts, no shared utterance or FLEURS
  sentence, one epoch, validation WER not rising, a second test set frozen before training and scored once.
- **Limits.** Mixed recordings are assembled from single-language utterances (a language change is also a speaker
  change); read sentences and short phrases dominate; no speaker ids, so speaker overlap between roles cannot be
  excluded; 170 test units per set, so intervals are wide; the starting model's authors trained on KSC2 and do not
  say whether its test folders were held out, so comparisons with it on KSC2 may be tilted in its favour;
  overlapping voices are not handled in Conversation mode.

- Gate D: approved 2026-10-14 (round 2 wrap-up; delegated: the user asked on 2026-10-13 for every stop of this training run, the final test included, to be passed unattended and recorded for her review)

'''
assert marker in text and "## Wrap-up (Gate D)" not in text
log.LOG.write_text(text.replace(marker, report + marker, 1), encoding="utf-8")

# The app shows the model card: add the one-time test numbers to it.
card_path = export / "model_card.json"
card = json.loads(card_path.read_text(encoding="utf-8"))
card["test_b"] = {s: {"baseline_wer": score("kazakh", s), "cleartalk_wer": score("cleartalk-large", s), "units": compare["kazakh"][s]["units"]} for s in sets}
tail = card["evidence"][-1]
card["evidence"] = [e for e in card["evidence"][:-1] if not e.startswith("Final test")]
card["evidence"] += [f"Final test, {names[s]} ({compare['kazakh'][s]['units']}): {p(score('kazakh', s))} Kazakh Whisper large-v3-turbo → "
                     f"{p(score('cleartalk-large', s))} ClearTalk large." for s in sets]
card["evidence"].append(f"Final test, {names['mixed']} ({compare['cleartalk']['mixed']['units']}): {p(score('cleartalk', 'mixed'))} ClearTalk small → "
                        f"{p(score('cleartalk-large', 'mixed'))} ClearTalk large.")
card["evidence"].append(tail)
card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("logged; model card updated")
