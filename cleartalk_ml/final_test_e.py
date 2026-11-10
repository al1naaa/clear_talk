"""Round 5, step 14: the one-time final test on test E - the app as round 5 began against the app after it.

    python -m cleartalk_ml.final_test_e

Requires Gate C in ml_pipeline/round5/PIPELINE.md. Test E (cleartalk_ml.round5) holds utterances no test used so far:
single utterances (clean, white 10 dB, babble 10 dB), mixed Kazakh/Russian recordings, recordings where two people talk
at once, and 5-minute conversations (clean and babble 10 dB). Both arms run the app's Conversation pipeline with the
language detected automatically:

* ``before`` - the app as round 5 began: round-4 options (background stretches set aside, region-first transcription,
  loop cut) with the round-2 large model;
* ``after``  - the app with everything adopted in round 5 (cleartalk_ml.app_options) and the model the round-5
  training rule left in place.

The audio is rendered from the frozen recipes only now. guard.final_test verifies the fingerprint, refuses a second run
and logs the after-arm's pooled WER; transcripts are appended to data/ml/results/hyp-testE/ as they are produced, so an
interrupted run continues with the same command and still counts as one run.
"""

import json
import os
import re
import sys

from . import overlap as overlap_measure
from . import round5, splits
from .app_options import adopted_flags, setting_name
from .evaluate import bootstrap_difference, details, pick_device, read_hypotheses, run_units
from .paths import RESULTS
from .text import pooled

HYP = RESULTS / "hyp-testE"
OUT = RESULTS / "test-final-E.json"
BEFORE_FLAGS = ["background", "region"]


def promoted() -> bool:
    path = RESULTS / "selection-round5.json"
    return path.is_file() and bool(json.loads(path.read_text(encoding="utf-8")).get("promoted"))


def arms() -> dict:
    return {"before": f"pipeline:{setting_name(BEFORE_FLAGS)}", "after": f"pipeline:{setting_name(adopted_flags())}"}


def round2_model() -> str:
    selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
    return f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"


def gate_c_approved() -> bool:
    text = (round5.R5 / "PIPELINE.md").read_text(encoding="utf-8")
    return any(re.search(r"\bgate\s*c\b", line, re.I) and re.search(r"\bapproved\b", line, re.I)
               and not re.search(r"\b(not|pending|awaiting|todo)\b", line, re.I) for line in text.splitlines())


def path_of(arm, backend):
    return HYP / f"{arm}__{backend}.jsonl"


def run_arms(units):
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    for arm, system in arms().items():
        print(f"== {arm} ({system})", flush=True)
        if arm == "before" and promoted():
            os.environ["CLEARTALK_EVAL_OVERRIDE"] = f"cleartalk-large={round2_model()}"   # the model round 5 began with
        try:
            run_units(system, units, path_of(arm, backend), device, precision)
        finally:
            os.environ.pop("CLEARTALK_EVAL_OVERRIDE", None)


def table(units, plan):
    device, precision = pick_device()
    backend = f"{device}-{precision}"
    hyp = {arm: read_hypotheses(path_of(arm, backend)) for arm in arms()}
    out = {"arms": arms(), "model_promoted": promoted(), "parts": {}}
    rate = lambda f: float(f.insertions.sum() / max(f.reference_words.sum(), 1))  # noqa: E731
    wer = lambda f: pooled(f.to_dict("records"))["wer"]  # noqa: E731
    for (part, condition), group in units.groupby(["part", "condition"]):
        kind = "forced" if part == "single" else "auto"
        a, b = details(group, hyp["before"], kind), details(group, hyp["after"], kind)
        block = {"units": int(len(group)), "before": wer(a), "after": wer(b), **bootstrap_difference(a, b),
                 "insertions_before": rate(a), "insertions_after": rate(b)}
        if part == "overlap":
            recordings = {"seed": plan["seed"], "recordings": plan["overlap"]}
            for arm in ("before", "after"):
                rows = {key.split("::", 1)[1]: row for key, row in hyp[arm].items() if key.startswith(condition + "::testE-overlap")}
                frame = overlap_measure.cp_frame(recordings, rows)
                block[f"cpwer_{arm}"] = float(frame[["substitutions", "deletions", "insertions"]].sum().sum() / max(frame.reference_words.sum(), 1))
                block[f"_cp_{arm}"] = frame
            block["cpwer_difference"] = bootstrap_difference(block.pop("_cp_before"), block.pop("_cp_after"))
        out["parts"][f"{part}@{condition}"] = block
    for condition, group in units.groupby("condition"):
        a, b = details(group, hyp["before"], "auto"), details(group, hyp["after"], "auto")
        out.setdefault("conditions", {})[condition] = {"before": wer(a), "after": wer(b), **bootstrap_difference(a, b)}
    a, b = details(units, hyp["before"], "auto"), details(units, hyp["after"], "auto")
    out["all"] = {"before": wer(a), "after": wer(b), **bootstrap_difference(a, b), "insertions_before": rate(a), "insertions_after": rate(b)}
    singles = units[units.part == "single"]
    out["single_sets"] = {}
    for set_name, group in singles.groupby("set"):
        sa, sb = details(group, hyp["before"], "forced"), details(group, hyp["after"], "forced")
        out["single_sets"][set_name] = {"units": int(len(group)), "before": wer(sa), "after": wer(sb)}
    return out


def main():
    if not gate_c_approved():
        raise SystemExit("The final test needs Gate C in ml_pipeline/round5/PIPELINE.md")
    from .paths import PIPELINE

    sys.path.insert(0, str(PIPELINE))
    import guard

    if OUT.exists():
        raise SystemExit(f"The final test on test E was already run once; see {OUT}")
    from .awake import keep_awake

    keep_awake()
    plan = round5.build_plan()
    units = round5.units(plan)

    def transcribe_all(features):
        round5.render(plan)   # the frozen recipes become audio only now
        run_arms(units)
        device, precision = pick_device()
        after = read_hypotheses(path_of("after", f"{device}-{precision}"))
        return [after[unit_id]["auto"] for unit_id in features.unit_id]

    score = guard.final_test(transcribe_all, units, target="reference", metric_fn=splits.unit_wer,
                             state_dir=round5.R5, pipeline_md=round5.R5 / "PIPELINE.md")
    result = {"after_wer_all_units": score, "units": int(len(units)), **table(units, plan)}
    OUT.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"final test done: the round-5 app's pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)
    print("FINAL_TEST_E_DONE", flush=True)


if __name__ == "__main__":
    main()
