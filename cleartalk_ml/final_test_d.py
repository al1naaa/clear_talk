"""Round 4, step 14: the one-time final test on test D.

    python -m cleartalk_ml.final_test_d

Requires Gate C in ml_pipeline/round4/PIPELINE.md. Test D (cleartalk_ml.round4) is test C - 898 units no system has
transcribed - clean, with white noise at 10 dB and with babble at 10 dB.

If the noise-trained model was promoted on validation, the test compares models, through the app's decoding:

* ``noise``  - the validation-selected model of round 4 (language given AND automatic);
* ``round2`` - the large model of round 2, which it must beat (language given);
* ``kazakh`` - the untouched starting model, a reference point (language given).

If it was not (the round-2 model stays), the test compares the app itself, language detected automatically, as
decided before test D was opened (see the round-4 log):

* ``app-before`` - the app as it was when this round began (none of this round's protections);
* ``app-final``  - the app with everything adopted on validation in this round (cleartalk_ml.app_options).

The noisy audio is rendered from the frozen recipes only now. guard.final_test verifies the fingerprint, refuses a
second run and logs the round-4 model's pooled WER; transcripts are appended to data/ml/results/hyp-testD/ as they
are produced, so an interrupted run continues with the same command and still counts as one run.
"""

import argparse
import json
import re
import subprocess
import sys

from . import round4, splits
from .evaluate import bootstrap_difference, details, pick_device, read_hypotheses, run_units
from .paths import RESULTS, ROOT
from .text import pooled

HYP = RESULTS / "hyp-testD"
OUT = RESULTS / "test-final-D.json"
GROUPS = {"first": ("noise",), "second": ("round2", "kazakh")}
FORCED_ONLY = {"noise": False, "round2": True, "kazakh": True}


def promoted() -> bool:
    return bool(json.loads((RESULTS / "selection-round4.json").read_text(encoding="utf-8")).get("promoted"))


def app_systems():
    from .app_options import adopted_flags, setting_name

    return {"app-before": "pipeline:auto-before", "app-final": f"pipeline:{setting_name(adopted_flags())}"}


def run_app(units):
    device, precision = pick_device()
    for name, system in app_systems().items():
        print(f"== {name} ({system})", flush=True)
        run_units(system, units, path_of(name, f"{device}-{precision}"), device, precision)


def app_table(units):
    device, precision = pick_device()
    hyp = {name: read_hypotheses(path_of(name, f"{device}-{precision}")) for name in app_systems()}
    by = {}
    for column in ("condition", "set"):
        by[column] = {}
        for key, group in units.groupby(column):
            a, b = details(group, hyp["app-before"], "auto"), details(group, hyp["app-final"], "auto")
            by[column][key] = {"units": int(len(group)), "before": pooled(a.to_dict("records"))["wer"],
                               "final": pooled(b.to_dict("records"))["wer"], **bootstrap_difference(a, b)}
    a, b = details(units, hyp["app-before"], "auto"), details(units, hyp["app-final"], "auto")
    rate = lambda f: float(f.insertions.sum() / max(f.reference_words.sum(), 1))  # noqa: E731
    return {"systems": app_systems(), "all": {"before": pooled(a.to_dict("records"))["wer"], "final": pooled(b.to_dict("records"))["wer"],
                                               **bootstrap_difference(a, b)},
            "insertions_per_word": {"before": rate(a), "final": rate(b)}, **by}


def systems():
    selection = json.loads((RESULTS / "selection-round4.json").read_text(encoding="utf-8"))
    round2 = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    return {"noise": f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}",
            "round2": f"runs/{round2['run']}/ct2-step-{round2['chosen_step']:05d}", "kazakh": "kazakh"}


def gate_c_approved() -> bool:
    text = (round4.R4 / "PIPELINE.md").read_text(encoding="utf-8")
    return any(re.search(r"\bgate\s*c\b", line, re.I) and re.search(r"\bapproved\b", line, re.I)
               and not re.search(r"\b(not|pending|awaiting|todo)\b", line, re.I) for line in text.splitlines())


def path_of(name, backend):
    return HYP / f"{name}__{backend}.jsonl"


def run_group(group, units):
    device, precision = pick_device()
    names = systems()
    for name in GROUPS[group]:
        print(f"== {name}", flush=True)
        run_units(names[name], units, path_of(name, f"{device}-{precision}"), device, precision, forced_only=FORCED_ONLY[name])


def table(units):
    device, precision = pick_device()
    hyp = {name: read_hypotheses(path_of(name, f"{device}-{precision}")) for name in GROUPS["first"] + GROUPS["second"]}
    rows, comparisons = [], {}
    for set_name, group in units.groupby("set"):
        for name, rows_of in hyp.items():
            score = pooled(details(group, rows_of, "forced").to_dict("records"))
            rows.append({"system": name, "set": set_name, "units": int(len(group)), "wer": score["wer"], "words": score["reference_words"]})
        comparisons[set_name] = {other: bootstrap_difference(details(group, hyp[other], "forced"), details(group, hyp["noise"], "forced"))
                                 for other in ("round2", "kazakh")}
    by_condition = {}
    for condition, group in units.groupby("condition"):
        by_condition[condition] = {name: pooled(details(group, rows_of, "forced").to_dict("records"))["wer"] for name, rows_of in hyp.items()}
        by_condition[condition]["noise_minus_round2"] = bootstrap_difference(details(group, hyp["round2"], "forced"), details(group, hyp["noise"], "forced"))
    return {"scores": rows, "noise_minus": comparisons, "by_condition": by_condition}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=sorted(GROUPS))
    args = parser.parse_args()
    if not gate_c_approved():
        raise SystemExit("The final test needs Gate C in ml_pipeline/round4/PIPELINE.md")
    units = round4.test_d_units()
    if args.worker:
        run_group(args.worker, units)
        return
    from .paths import PIPELINE

    sys.path.insert(0, str(PIPELINE))
    import guard

    if OUT.exists():
        raise SystemExit(f"The final test on test D was already run once; see {OUT}")
    from .awake import keep_awake

    keep_awake()

    if not promoted():
        def transcribe_app(features):
            round4.render(units, round4.build_plan())   # the frozen noise recipes become audio only now
            run_app(units)
            device, precision = pick_device()
            final = read_hypotheses(path_of("app-final", f"{device}-{precision}"))
            return [final[unit_id]["auto"] for unit_id in features.unit_id]

        score = guard.final_test(transcribe_app, units, target="reference", metric_fn=splits.unit_wer,
                                 state_dir=round4.R4, pipeline_md=round4.R4 / "PIPELINE.md")
        result = {"mode": "app", "app_final_wer_all_units": score, "units": int(len(units)), **app_table(units)}
        OUT.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
        print(f"final test done: the final app's pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)
        print("FINAL_TEST_D_DONE", flush=True)
        return

    def transcribe_all(features):
        round4.render(units, round4.build_plan())   # the frozen noise recipes become audio only now
        helper = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "cleartalk_ml.final_test_d", "--worker", "second"], cwd=ROOT)
        run_group("first", units)
        if helper.wait() != 0:
            raise RuntimeError("The second process did not finish; run the same command again to continue")
        device, precision = pick_device()
        ours = read_hypotheses(path_of("noise", f"{device}-{precision}"))
        return [ours[unit_id]["auto"] for unit_id in features.unit_id]

    score = guard.final_test(transcribe_all, units, target="reference", metric_fn=splits.unit_wer,
                             state_dir=round4.R4, pipeline_md=round4.R4 / "PIPELINE.md")
    result = {"noise_model_wer_all_units_automatic_language": score, "units": int(len(units)), "systems": systems(), **table(units)}
    OUT.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"final test done: round-4 model pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)
    print("FINAL_TEST_D_DONE", flush=True)


if __name__ == "__main__":
    main()
