"""Measure the two decoding options of the app on validation and apply the rules written before measuring.

    python scripts/check_decoding_options.py

Rules (ml_pipeline/round2/PIPELINE.md, "Decisions"):
1. mixed_skip_detection - adopted if WER on the mixed validation recordings is at most 0.3 points above the
   "languages change inside turns" setting as validated.
2. uncertain_to_mixed - adopted if WER on the mixed validation recordings is lower than the validated per-turn
   pipeline with the 95% bootstrap interval below zero, and English FLEURS validation utterances through the
   pipeline are not more than 0.5 points worse.
Adopted options are written to models/multilingual/profile.json ("options"); transcripts are cached.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from cleartalk_ml import validation
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.evaluate import bootstrap_difference, details, pick_device, run_units, units_of
    from cleartalk_ml.paths import MODELS, RESULTS
    from cleartalk_ml.text import pooled

    keep_awake()
    device, precision = pick_device()
    backend = f"{device}-{precision}"

    def path(set_name, name):
        return RESULTS / "hyp" / f"{set_name}__{name}__{backend}.jsonl"

    mixed = units_of("mixed-val")
    english = units_of("fleurs-val", language="en")
    jobs = [("pipeline:mixed-fast", mixed, "mixed-val", "pipeline-mixed-fast-r2"),
            ("pipeline:auto-safe", mixed, "mixed-val", "pipeline-auto-safe-r2"),
            ("pipeline:auto-validated", english, "fleurs-val", "pipeline-auto-validated-r2"),
            ("pipeline:auto-safe", english, "fleurs-val", "pipeline-auto-safe-r2")]
    for system, units, set_name, name in jobs:
        run_units(system, units, path(set_name, name), device, precision)

    def frame(set_name, name, units, kind):
        hyp, _ = validation.hypotheses(set_name, name)
        return details(units, hyp, kind)

    validated_mixed = frame("mixed-val", "pipeline-mixed-r2", mixed, "auto")
    fast_mixed = frame("mixed-val", "pipeline-mixed-fast-r2", mixed, "auto")
    validated_auto = frame("mixed-val", "pipeline-auto-r2", mixed, "auto")
    safe_auto = frame("mixed-val", "pipeline-auto-safe-r2", mixed, "auto")
    validated_en = frame("fleurs-val", "pipeline-auto-validated-r2", english, "forced")
    safe_en = frame("fleurs-val", "pipeline-auto-safe-r2", english, "forced")
    wer = lambda f: pooled(f.to_dict("records"))["wer"]  # noqa: E731

    option1 = {"validated_wer": wer(validated_mixed), "candidate_wer": wer(fast_mixed),
               **bootstrap_difference(validated_mixed, fast_mixed)}
    option1["adopted"] = option1["candidate_wer"] - option1["validated_wer"] <= 0.003
    option2 = {"mixed": {"validated_wer": wer(validated_auto), "candidate_wer": wer(safe_auto), **bootstrap_difference(validated_auto, safe_auto)},
               "english": {"validated_wer": wer(validated_en), "candidate_wer": wer(safe_en), **bootstrap_difference(validated_en, safe_en)}}
    option2["adopted"] = (option2["mixed"]["high"] < 0 and option2["english"]["candidate_wer"] - option2["english"]["validated_wer"] <= 0.005)

    profile_path = MODELS / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["options"] = {"mixed_skip_detection": bool(option1["adopted"]), "uncertain_to_mixed": bool(option2["adopted"])}
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"mixed_skip_detection": option1, "uncertain_to_mixed": option2, "profile_options": profile["options"]}
    (RESULTS / "decoding-options.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print("OPTIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
