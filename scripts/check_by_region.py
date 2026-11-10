"""Measure "recognise whole stretches of speech, attach speakers by time" on validation and apply the pre-registered rule.

    python scripts/check_by_region.py

Rule (ml_pipeline/round4/PIPELINE.md, "Decisions", written before measuring): adopted if, on the robustness validation
recordings with the language detected per turn (mixed recordings, short phrases, 7-minute conversations; every
measured noise condition), (a) the pooled WER is lower than the current pipeline's with the 95% bootstrap interval
of the difference below zero; (b) clean mixed recordings, clean short phrases and clean conversations are each not
more than 0.5 points worse; (c) on the clean mixed recordings and conversations, turns that hold words of more than
one utterance (a sentence given to the wrong person) rise by no more than 2% of the turns. If adopted,
"transcribe_by_region" is switched on in models/multilingual/profile.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mixed_turns(plan, hyp, units):
    """(turns, turns spanning more than one utterance by at least 0.3 s each) on mixed recordings and conversations."""
    import soundfile as sf

    from cleartalk_ml.mixed import load_plan

    spans = {recipe["mix_id"]: [(p["start"], p["end"]) for p in recipe["parts"]] for recipe in load_plan("val")}
    for item in plan["long"]:
        spans[item["unit_id"]] = []
        for part in item["parts"]:
            info = sf.info(str(ROOT / part["path"]))
            spans[item["unit_id"]].append((part["start"], part["start"] + info.frames / info.samplerate))
    total = several = 0
    for unit in units.itertuples():
        if unit.set not in ("mixed", "long"):
            continue
        for turn in hyp[unit.unit_id]["turns"]:
            total += 1
            several += sum(1 for s, e in spans[unit.unit_id] if min(e, turn["end"]) - max(s, turn["start"]) > 0.3) > 1
    return total, several


def main():
    import pandas as pd

    from cleartalk_ml import robustness
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.evaluate import bootstrap_difference, details, pick_device, run_units
    from cleartalk_ml.evaluate import read_hypotheses as read_raw
    from cleartalk_ml.evaluate import without_loops

    def read_hypotheses(path):  # both sides as the app now writes them (repetition loops cut)
        return without_loops(read_raw(path))
    from cleartalk_ml.paths import MODELS, RESULTS
    from cleartalk_ml.text import pooled

    keep_awake()
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
    before, after, clean = [], [], {}
    for condition in robustness.CONDITIONS:
        units = robustness.units(plan, condition)
        path = robustness.hyp_file(condition, "auto-region", backend)
        run_units("pipeline:auto-region", units, path, device, precision)
        old, new = read_hypotheses(robustness.hyp_file(condition, "auto", backend)), read_hypotheses(path)
        for set_name, group in units.groupby("set"):
            kind = "auto" if set_name in ("mixed", "long") else "forced"
            a = details(group, old, kind).assign(unit_id=lambda f, c=condition: c + ":" + f.unit_id)
            b = details(group, new, kind).assign(unit_id=lambda f, c=condition: c + ":" + f.unit_id)
            before.append(a)
            after.append(b)
            if condition == "clean":
                clean[set_name] = {"before": pooled(a.to_dict("records"))["wer"], "after": pooled(b.to_dict("records"))["wer"]}
        if condition == "clean":
            attribution = {"before": mixed_turns(plan, old, units), "after": mixed_turns(plan, new, units)}
    before, after = pd.concat(before, ignore_index=True), pd.concat(after, ignore_index=True)
    wer = lambda f: pooled(f.to_dict("records"))["wer"]  # noqa: E731
    overall = {"before": wer(before), "after": wer(after), **bootstrap_difference(before, after)}
    def clean_short(frame):
        return frame[frame.unit_id.str.startswith("clean:") & ~frame.unit_id.str.contains("mixed-val|long-")]

    short_before, short_after = wer(clean_short(before)), wer(clean_short(after))
    guards = {"mixed": clean["mixed"], "long": clean["long"], "short": {"before": short_before, "after": short_after}}
    for block in guards.values():
        block["ok"] = block["after"] - block["before"] <= 0.005
    (t0, s0), (t1, s1) = attribution["before"], attribution["after"]
    attribution_ok = s1 / max(t1, 1) - s0 / max(t0, 1) <= 0.02
    adopted = overall["high"] < 0 and all(block["ok"] for block in guards.values()) and attribution_ok
    profile_path = MODELS / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.setdefault("options", {})["transcribe_by_region"] = bool(adopted)
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {"overall": overall, "clean_guards": guards, "clean_by_set": clean,
               "attribution": {"before": {"turns": t0, "several_utterances": s0}, "after": {"turns": t1, "several_utterances": s1}, "ok": attribution_ok},
               "adopted": bool(adopted), "profile_options": profile["options"]}
    (RESULTS / "by-region.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print("BY_REGION_DONE", flush=True)


if __name__ == "__main__":
    main()
