"""Round 2, step 14: the one-time final test of the large multilingual model on test B.

    python -m cleartalk_ml.final_test_b

Requires Gate C in ml_pipeline/round2/PIPELINE.md. Test B is the second frozen test set (cleartalk_ml.round2):
1,020 units that no system has transcribed. Every system transcribes the same audio through the app's
decoding path:

* ``cleartalk-large`` - the validation-selected, exported large model (all sets; language given AND automatic);
* ``kazakh``          - the untouched starting model, the baseline of this round (all sets);
* ``turbo``, ``cleartalk`` (our small model of round 1), ``small`` - reference points (all sets);
* ``pipeline:auto``, ``pipeline:mixed`` - the app's Conversation pipeline with the profile written by
  scripts/select_round2.py: language detected per turn, and the "languages change inside turns" setting
  (mixed recordings only).

For every system except ours, single utterances are transcribed with the language given only; mixed recordings
always use automatic language.

``guard.final_test`` verifies the frozen fingerprint, refuses a second run and logs our model's pooled WER in
the round's PIPELINE.md. Transcripts are appended to data/ml/results/hyp-testB/ as they are produced, so a run
interrupted by sleep or power loss continues with the same command and still counts as one run.
Two processes share the graphics card while each holds at most one large model; the pipelines, which hold
several, run alone at the end.
"""

import argparse
import json
import re
import subprocess
import sys

from . import round2, splits
from .evaluate import bootstrap_difference, details, mixed_by_language, pick_device, read_hypotheses, run_units
from .paths import BASES, LANGUAGES, MODELS, RESULTS, ROOT
from .text import pooled

HYP = RESULTS / "hyp-testB"
OUT = RESULTS / "test-final-B.json"
OURS = BASES["kazakh-turbo"]["export"]
SMALL_OURS = BASES["small"]["export"]
MIXED_ONLY = ("mixed",)
# system -> (sets it transcribes, language given only)
PLAN = {
    "cleartalk-large": (None, False),
    "kazakh": (None, True),
    "turbo": (None, True),
    "cleartalk": (None, True),
    "small": (None, True),
    "pipeline:auto": (MIXED_ONLY, False),
    "pipeline:mixed": (MIXED_ONLY, False),
}
FOLDER = {"cleartalk-large": OURS, "cleartalk": SMALL_OURS}
GROUPS = {"first": ("cleartalk-large", "cleartalk", "small"), "second": ("kazakh", "turbo"), "last": ("pipeline:auto", "pipeline:mixed")}
COMPARED_WITH = ("kazakh", "cleartalk", "turbo")


def gate_c_approved() -> bool:
    text = (round2.R2 / "PIPELINE.md").read_text(encoding="utf-8")
    return any(re.search(r"\bgate\s*c\b", line, re.I) and re.search(r"\bapproved\b", line, re.I)
               and not re.search(r"\b(not|pending|awaiting|todo)\b", line, re.I) for line in text.splitlines())


def path_of(system, backend):
    return HYP / f"{system.replace(':', '-')}__{backend}.jsonl"


def units_for(system, units):
    sets = PLAN[system][0]
    return units if sets is None else units[units.set.isin(sets)].reset_index(drop=True)


def run_group(group, units):
    device, precision = pick_device()
    for system in GROUPS[group]:
        print(f"== {system}", flush=True)
        run_units(FOLDER.get(system, system), units_for(system, units), path_of(system, f"{device}-{precision}"), device, precision,
                  forced_only=PLAN[system][1])


def hypotheses(system):
    device, precision = pick_device()
    return read_hypotheses(path_of(system, f"{device}-{precision}"))


def table(units):
    """Scores of every system on every set of test B, and our large model against three others with intervals."""
    rows, by_language = [], {}
    mixed = units[units.set == "mixed"]
    for system in PLAN:
        hyp = hypotheses(system)
        for set_name, group in units_for(system, units).groupby("set"):
            if any(u not in hyp for u in group.unit_id):
                continue
            kinds = ("forced",) if PLAN[system][1] and set_name != "mixed" else ("forced", "auto")
            block = {"system": system, "set": set_name, "units": int(len(group))}
            for kind in kinds:
                score = pooled(details(group, hyp, kind).to_dict("records"))
                block[f"wer_{kind}"], block[f"cer_{kind}"] = score["wer"], score["cer"]
                block[f"words_{kind}"] = score["reference_words"]
            if set_name != "mixed" and not PLAN[system][1]:
                block["language_accuracy"] = sum(hyp[u].get("detected") == lang for u, lang in zip(group.unit_id, group.language)) / len(group)
            rows.append(block)
        if len(mixed) and all(u in hyp for u in mixed.unit_id):
            by_language[system] = mixed_by_language(mixed, hyp)
    ours, comparisons = hypotheses("cleartalk-large"), {}
    for other in (*COMPARED_WITH, "pipeline:auto", "pipeline:mixed"):
        theirs = hypotheses(other)
        comparisons[other] = {}
        for set_name, group in units_for(other, units).groupby("set"):
            kind = "auto" if set_name == "mixed" else "forced"
            comparisons[other][set_name] = bootstrap_difference(details(group, theirs, kind), details(group, ours, kind))
    return {"scores": rows, "cleartalk_large_minus": comparisons, "mixed_by_language": by_language}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=sorted(GROUPS), help="internal: transcribe one group of systems")
    args = parser.parse_args()
    if not gate_c_approved():
        raise SystemExit("The final test needs Gate C in ml_pipeline/round2/PIPELINE.md")
    units = round2.test_b_units()
    if args.worker:
        run_group(args.worker, units)
        return
    from .paths import PIPELINE

    sys.path.insert(0, str(PIPELINE))
    import guard

    if OUT.exists():
        raise SystemExit(f"The final test on test B was already run once; see {OUT}")
    from .awake import keep_awake
    from .mixed import render_split

    keep_awake()

    def transcribe_all(features):
        render_split(round2.MIXED_SPLIT)  # the frozen recipes become audio only now
        helper = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "cleartalk_ml.final_test_b", "--worker", "second"], cwd=ROOT)
        run_group("first", units)
        if helper.wait() != 0:
            raise RuntimeError("The second process did not finish; run the same command again to continue")
        run_group("last", units)
        ours = hypotheses("cleartalk-large")
        return [ours[unit_id]["auto"] for unit_id in features.unit_id]

    score = guard.final_test(transcribe_all, units, target="reference", metric_fn=splits.unit_wer,
                             state_dir=round2.R2, pipeline_md=round2.R2 / "PIPELINE.md")
    result = {"cleartalk_large_wer_all_units_automatic_language": score, "units": int(len(units)), **table(units),
              "languages": list(LANGUAGES), "profile": json.loads((MODELS / "profile.json").read_text(encoding="utf-8"))["routes"]}
    OUT.write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"final test done: ClearTalk large pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)
    print("FINAL_TEST_B_DONE", flush=True)


if __name__ == "__main__":
    main()
