"""Measure one Conversation-mode option on validation against the app as it now stands, and apply its rule.

    python scripts/check_option.py region        # recognise whole stretches of speech (breath pauses closed)
    python scripts/check_option.py background    # set aside short quiet stretches of background voices or noise
    python scripts/check_option.py overlap       # separate stretches where two people talk at once
    python scripts/check_option.py kk            # narrower cross-check: our model takes over "English" turns it heard as Kazakh
    python scripts/check_option.py text          # round 5: our model writes every turn; English turns go to the English route

The rules were written in ml_pipeline/round4/PIPELINE.md ("Decisions") before measuring. Options are measured one
after another; each is compared with the options adopted so far (data/ml/results/app-options.json), and an adopted
option is switched on in models/multilingual/profile.json.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE_KEYS = {"region": "transcribe_by_region", "background": "hide_background_turns", "overlap": "separate_overlaps", "kk": "cross_check",
                "text": "route_by_text"}


def several_utterances(plan, hyp, units):
    """(turns, turns holding at least 0.3 s of more than one reference utterance) on clean mixed recordings and conversations."""
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

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("option", choices=("region", "background", "overlap", "kk", "text"))
    args = parser.parse_args()

    from cleartalk_ml import overlap, robustness
    from cleartalk_ml.app_options import setting_name
    from cleartalk_ml.awake import keep_awake
    from cleartalk_ml.evaluate import bootstrap_difference, details, pick_device, run_units, units_of
    from cleartalk_ml.evaluate import read_hypotheses as read_raw
    from cleartalk_ml.evaluate import without_loops

    def read_hypotheses(path):  # both sides as the app now writes them (repetition loops cut)
        return without_loops(read_raw(path))
    from cleartalk_ml.paths import MODELS, RESULTS
    from cleartalk_ml.text import pooled

    keep_awake()
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    state_path = RESULTS / "app-options.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        by_region = json.loads((RESULTS / "by-region.json").read_text(encoding="utf-8"))
        state = {"flags": [], "history": [{"option": "region (first variant, scripts/check_by_region.py)", "adopted": by_region["adopted"]}]}
    current, candidate = setting_name(state["flags"]), setting_name(state["flags"] + [args.option])
    wer = lambda f: pooled(f.to_dict("records"))["wer"]  # noqa: E731
    interval = lambda a, b: {"before": wer(a), "after": wer(b), **bootstrap_difference(a, b)}  # noqa: E731

    def robust(conditions, with_long=True):
        rplan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
        before, after = [], []
        for condition in conditions:
            units = robustness.units(rplan, condition)
            if not with_long:
                units = units[units.set != "long"].reset_index(drop=True)
            files = {}
            for name in (current, candidate):
                files[name] = robustness.hyp_file(condition, name, backend)
                run_units(f"pipeline:{name}", units, files[name], device, precision)
            old, new = read_hypotheses(files[current]), read_hypotheses(files[candidate])
            for set_name, group in units.groupby("set"):
                kind = "auto" if set_name in ("mixed", "long") else "forced"
                for rows, out in ((old, before), (new, after)):
                    out.append(details(group, rows, kind).assign(condition=condition, set=set_name,
                                                                 unit_id=lambda f, c=condition: c + ":" + f.unit_id))
        return pd.concat(before, ignore_index=True), pd.concat(after, ignore_index=True)

    def clean_guards(before, after, limit):
        guards = {}
        for name, pick in (("mixed", lambda f: f.set == "mixed"), ("long", lambda f: f.set == "long"), ("short", lambda f: f.set.str.startswith("short"))):
            a, b = before[(before.condition == "clean") & pick(before)], after[(after.condition == "clean") & pick(after)]
            if len(a):
                guards[name] = {"before": wer(a), "after": wer(b), "ok": wer(b) - wer(a) <= limit}
        return guards

    summary = {"option": args.option, "compared_with": current, "measured": candidate}
    if args.option == "region":
        before, after = robust(robustness.CONDITIONS)
        summary["all"] = interval(before, after)
        summary["clean_guards"] = clean_guards(before, after, 0.005)
        summary["by_condition"] = {c: {"before": wer(before[before.condition == c]), "after": wer(after[after.condition == c])} for c in robustness.CONDITIONS}
        rplan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
        units = robustness.units(rplan, "clean")
        shares = {}
        for name in (current, candidate):
            turns, several = several_utterances(rplan, read_hypotheses(robustness.hyp_file("clean", name, backend)), units)
            shares[name] = {"turns": turns, "several_utterances": several, "share": several / max(turns, 1)}
        summary["attribution"] = {"before": shares[current], "after": shares[candidate],
                                  "ok": shares[candidate]["share"] - shares[current]["share"] <= 0.02}
        adopted = summary["all"]["high"] < 0 and all(g["ok"] for g in summary["clean_guards"].values()) and summary["attribution"]["ok"]
    elif args.option == "background":
        before, after = robust(robustness.CONDITIONS)
        half = lambda f: f[[zlib.crc32(u.split(":", 1)[1].encode()) % 2 == 0 for u in f.unit_id]]  # noqa: E731
        summary["all"] = interval(before, after)
        summary["held_out_half"] = interval(half(before), half(after))
        summary["clean_guards"] = clean_guards(before, after, 0.002)
        summary["by_condition"] = {c: {"before": wer(before[before.condition == c]), "after": wer(after[after.condition == c])} for c in robustness.CONDITIONS}
        adopted = summary["all"]["high"] < 0 and summary["held_out_half"]["high"] < 0 and all(g["ok"] for g in summary["clean_guards"].values())
    elif args.option == "overlap":
        plan = overlap.build()
        held_out = {"seed": plan["seed"], "recordings": plan["recordings"][50:]}   # 0-49 set the detector
        units = overlap.units(held_out)
        files = {}
        for name in (current, candidate):
            files[name] = RESULTS / "hyp-overlap" / f"{name}__{backend}.jsonl"
            run_units(f"pipeline:{name}", units, files[name], device, precision)
        cp_old = overlap.cp_frame(held_out, read_hypotheses(files[current]))
        cp_new = overlap.cp_frame(held_out, read_hypotheses(files[candidate]))
        summary["overlap_cpwer"] = interval(cp_old, cp_new)
        summary["overlap_cpwer"]["separated_turns"] = sum(1 for row in read_hypotheses(files[candidate]).values() for t in row["turns"] if t.get("overlap"))
        before, after = robust(robustness.CONDITIONS)
        summary["all"] = interval(before, after)
        summary["all"]["ok"] = summary["all"]["after"] - summary["all"]["before"] <= 0.003
        summary["clean_guards"] = clean_guards(before, after, 0.005)
        rate = lambda f: float(f.insertions.sum() / max(f.reference_words.sum(), 1))  # noqa: E731
        summary["insertions"] = {"before": rate(before), "after": rate(after), "ok": rate(after) - rate(before) <= 0.002}
        adopted = (summary["overlap_cpwer"]["high"] < 0 and summary["all"]["ok"] and summary["insertions"]["ok"]
                   and all(g["ok"] for g in summary["clean_guards"].values()))
    else:  # kk, text
        if args.option == "text":
            before, after = robust(robustness.CONDITIONS)
            summary["clean_guards"] = clean_guards(before, after, 0.005)
            summary["by_condition"] = {c: {"before": wer(before[before.condition == c]), "after": wer(after[after.condition == c])} for c in robustness.CONDITIONS}
        else:
            before, after = robust(("clean", "white-10", "babble-10"), with_long=False)
        summary["all"] = interval(before, after)
        english = units_of("fleurs-val", language="en")
        files = {}
        for name in (current, candidate):
            files[name] = RESULTS / "hyp" / f"fleurs-val__pipeline-{name}-r2__{backend}.jsonl"
            if name == "auto" and not files[name].is_file():
                files[name] = RESULTS / "hyp" / f"fleurs-val__pipeline-auto-safe-r2__{backend}.jsonl"   # the same setting, measured earlier
            run_units(f"pipeline:{name}", english, files[name], device, precision)
        en_before, en_after = details(english, read_hypotheses(files[current]), "forced"), details(english, read_hypotheses(files[candidate]), "forced")
        summary["english_fleurs"] = interval(en_before, en_after)
        adopted = summary["all"]["high"] < 0 and summary["english_fleurs"]["after"] - summary["english_fleurs"]["before"] <= 0.005
        if args.option == "text":
            adopted = adopted and all(g["ok"] for g in summary["clean_guards"].values())

    profile_path = MODELS / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    value = ("kazakh" if args.option == "kk" else True) if adopted else False
    profile.setdefault("options", {})[PROFILE_KEYS[args.option]] = value
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if adopted:
        state["flags"] = sorted(set(state["flags"]) | {args.option})
    state["history"].append({"option": args.option, "adopted": bool(adopted)})
    state_path.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    summary.update(adopted=bool(adopted), profile_options=profile["options"], app_flags=state["flags"])
    (RESULTS / f"option-{args.option}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print("OPTION_DONE", args.option, flush=True)


if __name__ == "__main__":
    main()
