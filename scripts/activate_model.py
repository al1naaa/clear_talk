"""Decide on validation whether the fine-tuned recognizer becomes the app default, then register it.

    .venv-train/bin/python scripts/activate_model.py            # decide + write profile/model card
    .venv-train/bin/python scripts/activate_model.py --check    # print the decision only

Inputs are the app-engine validation results written by ``python -m cleartalk_asr.evaluate``:
``data/asr/results/{val,v4val}-{baseline,finetuned}-app.json``. The pre-registered rule
(ml_pipeline_english/PIPELINE.md, step 3) is applied unchanged; test results never enter this decision.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleartalk_asr.evaluate import RESULTS, bootstrap_difference, eval_tracks, score  # noqa: E402
from cleartalk_asr.export import sha256  # noqa: E402
from cleartalk_asr.paths import EXPORT  # noqa: E402

MIN_RELATIVE_GAIN = 0.05
MAX_CLEAN_WORSENING = 0.005  # absolute WER


def details(set_name, label):
    saved = json.loads((RESULTS / f"{set_name}-{label}-app.json").read_text())
    frame = eval_tracks(set_name)
    hypotheses = {row["track_id"]: row["hypothesis"] for row in saved["tracks"]}
    metrics, per_unit = score(frame, [hypotheses[t] for t in frame.track_id])
    return metrics, per_unit


def decide():
    base, base_units = details("val", "baseline")
    tuned, tuned_units = details("val", "finetuned")
    pairs = lambda units: units[units.kind == "pair"]  # noqa: E731
    interval = bootstrap_difference(pairs(base_units), pairs(tuned_units))
    relative = (base["pairs"]["wer"] - tuned["pairs"]["wer"]) / base["pairs"]["wer"]
    clean_change = tuned["singles"]["wer"] - base["singles"]["wer"]
    v4_base, _ = details("v4val", "baseline")
    v4_tuned, _ = details("v4val", "finetuned")
    gates = {"relative_wer_reduction_at_least_5_percent": relative >= MIN_RELATIVE_GAIN,
             "bootstrap_95_interval_below_zero": interval["high"] < 0,
             "clean_speech_not_worse_than_0.5_points": clean_change <= MAX_CLEAN_WORSENING}
    return {
        "rule": "promote if all gates pass (pre-registered 2026-10-09, step 3)",
        "gates": gates, "promoted": all(gates.values()),
        "validation": {"baseline_wer": base["pairs"]["wer"], "cleartalk_wer": tuned["pairs"]["wer"],
                       "mixtures": base["pairs"]["mixtures"], "reference_words": base["pairs"]["reference_words"],
                       "relative_reduction": relative, "bootstrap": interval},
        "validation_clean": {"baseline_wer": base["singles"]["wer"], "cleartalk_wer": tuned["singles"]["wer"],
                             "utterances": base["singles"]["utterances"], "change": clean_change},
        "validation_v4": {"baseline_wer": v4_base["pairs"]["wer"], "cleartalk_wer": v4_tuned["pairs"]["wer"],
                          "mixtures": v4_base["pairs"]["mixtures"]},
    }


def register(decision):
    files = {name: sha256(EXPORT / name) for name in ("model.bin", "config.json", "tokenizer.json")}
    profile_path = ROOT / "models/active_profile.json"
    profile = json.loads(profile_path.read_text())
    profile["recognizers"]["cleartalk"]["files_sha256"] = files
    profile["default_model"] = "cleartalk" if decision["promoted"] else "baseline"
    profile.pop("note", None)
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    card_path = EXPORT / "model_card.json"
    card = json.loads(card_path.read_text())
    card.update(validation=decision["validation"], validation_clean=decision["validation_clean"],
                validation_v4=decision["validation_v4"], promotion={"promoted": decision["promoted"], "gates": decision["gates"]})
    card_path.write_text(json.dumps(card, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    decision = decide()
    print(json.dumps(decision, indent=2))
    if not args.check:
        (RESULTS / "promotion.json").write_text(json.dumps(decision, indent=2) + "\n")
        register(decision)
        print("default model:", "cleartalk" if decision["promoted"] else "baseline")


if __name__ == "__main__":
    main()
