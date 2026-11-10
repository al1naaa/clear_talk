"""Measure the English cross-check of the per-turn pipeline on validation and apply the rule written before measuring.

    python scripts/check_cross_check.py

Rule (ml_pipeline/round2/PIPELINE.md, "Decisions"): adopted if, on the robustness validation recordings (mixed
recordings and short phrases; clean, white noise 10 dB, babble 10 dB), pooled WER is lower than without it with the
95% bootstrap interval of the difference below zero, and English FLEURS validation utterances through the pipeline
are not more than 0.5 points worse. If adopted, "cross_check" is switched on in models/multilingual/profile.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONDITIONS = ("clean", "white-10", "babble-10")


def main():
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("any", "kazakh"), default="any", help='"kazakh": take over only turns our model wrote in Kazakh')
    args = parser.parse_args()
    system = "pipeline:auto-crosscheck" if args.variant == "any" else "pipeline:auto-crosscheck-kk"
    name = system.split(":", 1)[1]

    from cleartalk_ml import robustness, validation
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.evaluate import bootstrap_difference, details, pick_device, read_hypotheses, run_units, units_of
    from cleartalk_ml.paths import MODELS, RESULTS
    from cleartalk_ml.text import pooled

    keep_awake()
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
    before, after = [], []
    for condition in CONDITIONS:
        units = robustness.units(plan, condition)
        units = units[units.set != "long"].reset_index(drop=True)
        path = robustness.hyp_file(condition, name, backend)
        run_units(system, units, path, device, precision)
        old, new = read_hypotheses(robustness.hyp_file(condition, "auto", backend)), read_hypotheses(path)
        for set_name, group in units.groupby("set"):
            kind = "auto" if set_name == "mixed" else "forced"
            before.append(details(group, old, kind).assign(unit_id=lambda f, c=condition: c + ":" + f.unit_id))
            after.append(details(group, new, kind).assign(unit_id=lambda f, c=condition: c + ":" + f.unit_id))
    before, after = pd.concat(before, ignore_index=True), pd.concat(after, ignore_index=True)
    english = units_of("fleurs-val", language="en")
    english_path = RESULTS / "hyp" / f"fleurs-val__pipeline-{name}-r2__{backend}.jsonl"
    run_units(system, english, english_path, device, precision)
    old_en, _ = validation.hypotheses("fleurs-val", "pipeline-auto-safe-r2")
    en_before, en_after = details(english, old_en, "forced"), details(english, read_hypotheses(english_path), "forced")
    wer = lambda f: pooled(f.to_dict("records"))["wer"]  # noqa: E731
    robust = {"before": wer(before), "after": wer(after), **bootstrap_difference(before, after)}
    eng = {"before": wer(en_before), "after": wer(en_after), **bootstrap_difference(en_before, en_after)}
    adopted = robust["high"] < 0 and eng["after"] - eng["before"] <= 0.005
    profile_path = MODELS / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.setdefault("options", {})["cross_check"] = (args.variant if args.variant != "any" else True) if adopted else False
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"robustness_recordings": robust, "english_fleurs": eng, "adopted": bool(adopted), "profile_options": profile["options"]}
    (RESULTS / ("cross-check.json" if args.variant == "any" else "cross-check-kazakh.json")).write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print("CROSS_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
