"""Round 4: write the Gate C report (steps 9-13) into ml_pipeline/round4/PIPELINE.md from the saved results.

    python ml_pipeline/round4/reports/r4_gate_c.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LOG = ROOT / "ml_pipeline" / "round4" / "PIPELINE.md"
RESULTS = ROOT / "data" / "ml" / "results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def p(value):
    return f"{100 * value:.1f}%"


def d(block):
    return f"{100 * block['difference']:+.1f} points, 95% {100 * block['low']:+.1f} … {100 * block['high']:+.1f}"


def main():
    selection = load("selection-round4.json")
    background, region = load("option-background.json"), load("option-region.json")
    region_v2, overlap_v1, overlap, kazakh = load("option-region-v2.json"), load("option-overlap-v1.json"), load("option-overlap-v2.json"), load("option-kk.json")
    by_region_v1 = load("by-region.json")
    final = load("robustness-final.json")
    flags = json.loads((RESULTS / "app-options.json").read_text(encoding="utf-8"))["flags"]
    rows = {(r["condition"], r["set"]): r for r in final["rows"]}
    noisy_mixed, long_babble, long_clean = rows[("babble-10", "mixed")], rows[("babble-10", "long")], rows[("clean", "long")]

    overlap_line = (f"adopted: on the held-out overlap recordings cpWER {p(overlap['overlap_cpwer']['before'])} → {p(overlap['overlap_cpwer']['after'])} "
                    f"({d(overlap['overlap_cpwer'])}); robustness recordings {p(overlap['all']['before'])} → {p(overlap['all']['after'])}; "
                    f"extra words {p(overlap['insertions']['before'])} → {p(overlap['insertions']['after'])}"
                    if overlap["adopted"] else
                    f"not adopted: on the held-out overlap recordings cpWER {p(overlap['overlap_cpwer']['before'])} → {p(overlap['overlap_cpwer']['after'])} "
                    f"({d(overlap['overlap_cpwer'])}), but robustness recordings {p(overlap['all']['before'])} → {p(overlap['all']['after'])}, "
                    f"extra words {p(overlap['insertions']['before'])} → {p(overlap['insertions']['after'])}; guards "
                    + ", ".join(f"{k} {p(v['before'])} → {p(v['after'])}" for k, v in overlap["clean_guards"].items()))
    report = f"""## Gate C report (steps 9-13)

- **What was done.** (9) Baseline: the round-2 large model through the app's decoding. (10) Its decoder LoRA trained
  further for 450 steps with noise on half of the examples. (11) The checkpoint chosen on noisy validation among those
  meeting the clean guard. (12) The promotion rule applied. (13) Error analysis of the app's validation transcripts
  (hallucination audit, background turns, repetition loops, region-first and overlap transcripts), and from it five
  app options, each measured on validation against the app as it then stood, with a rule written before measuring.
  Test D was not touched.
- **What was found.**
  - Training in noise did not help: the chosen checkpoint (step {selection['chosen_step']}) gave
    {p(selection['baseline_noisy_wer'])} → {p(selection['baseline_noisy_wer'] + selection['noisy']['difference'])} on the noisy robustness
    recordings ({d(selection['noisy'])}). The round-2 model stays.
  - The commonest invented text in noise was not inside the speech but around it: short turns where only background
    talkers were audible got a speaker label of their own and a few invented words; one recogniser loop wrote "Senate"
    213 times; and in long conversations the speaker labels cut sentences, so words at the cuts were missed or guessed.
  - Adopted, in order: repetition loops cut back to three occurrences (no clean set changed); background turns set
    aside ({p(background['all']['before'])} → {p(background['all']['after'])}, {d(background['all'])}; the half not used to
    choose its thresholds {d(background['held_out_half'])}); region-first transcription, third variant
    ({p(region['all']['before'])} → {p(region['all']['after'])}, {d(region['all'])}; clean conversations
    {p(region['clean_guards']['long']['before'])} → {p(region['clean_guards']['long']['after'])}).
  - Not adopted: region-first first variant (clean mixed recordings +{100 * (by_region_v1['clean_guards']['mixed']['after'] - by_region_v1['clean_guards']['mixed']['before']):.1f}
    points), second variant (turns holding two people's utterances {100 * region_v2['attribution']['after']['share']:.1f}%, limit 2 points);
    overlap separation first variant (one voice written twice: robustness recordings {p(overlap_v1['all']['before'])} → {p(overlap_v1['all']['after'])});
    the narrower cross-check ({d(kazakh['all'])}).
  - Overlap separation, second variant: {overlap_line}.
  - The app now, against the app when this round began (validation, language detected per turn): babble 10 dB
    mixed recordings {p(noisy_mixed['before'])} → {p(noisy_mixed['after'])}, conversations {p(long_babble['before'])} → {p(long_babble['after'])};
    clean conversations {p(long_clean['before'])} → {p(long_clean['after'])}. Short Kazakh phrases got slightly worse in some noise
    conditions (white 20 dB {p(rows[('white-20', 'short-kk')]['before'])} → {p(rows[('white-20', 'short-kk')]['after'])}, babble 5 dB
    {p(rows[('babble-5', 'short-kk')]['before'])} → {p(rows[('babble-5', 'short-kk')]['after'])}); the rules guarded clean recordings only.
- **Decisions and why.** Each option was adopted or not by its rule as written; the profile options now on:
  {', '.join(flags) or 'none'} (models/multilingual/profile.json), plus the loop cut in every mode. Three variants
  were tried for region-first and two for overlap separation; every variant, its cause of failure and its rule are in
  "Decisions", and none was changed after its own measurement.
- **What comes next (steps 14-16).** The one-time final test on test D, as decided before opening it: the app as this
  round began against the app with everything adopted, language detected automatically, on 898 never-seen utterances
  clean, with white noise at 10 dB and with babble at 10 dB; then the deployment check and the monitoring plan.
- **The question - "may I open test D?"** Delegated in advance by the user (see "Delegation"): the stops are passed
  without her, the final test included, and everything is written down for her review.

- Gate C: approved 2026-10-16 (round 4; delegated as above)

"""
    text = LOG.read_text(encoding="utf-8")
    if "## Gate C report (steps 9-13)" in text:
        start = text.index("## Gate C report (steps 9-13)")
        end = text.index("## Decisions")
        text = text[:start] + report + text[end:]
    else:
        text = text.replace("## Decisions", report + "## Decisions", 1)
    text = text.replace("| 13 | Error analysis | todo | |",
                        "| 13 | Error analysis | done | hallucination audit, background turns, loops, region-first and overlap transcripts; five app options measured (see Gate C) |")
    LOG.write_text(text, encoding="utf-8")
    print("Gate C report written")


if __name__ == "__main__":
    main()
