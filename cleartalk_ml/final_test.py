"""Step 14: the one-time final test of the multilingual model.

    python -m cleartalk_ml.final_test

Requires Gate C in ml_pipeline/PIPELINE.md. Every system transcribes the same frozen test audio through
the app's decoding path:

* ``cleartalk`` - the validation-selected, exported model (all sets; language given AND automatic);
* ``small``     - untouched Whisper small, the baseline it is compared with (all sets);
* ``turbo``, ``kazakh``, ``pipeline:auto`` - the large ready models as reference points
  (the Kazakh model only on Kazakh sets; the app's Conversation pipeline only on mixed recordings).

For every system except ours, single utterances are transcribed with the language given only (their
language detection was measured on validation; skipping the second pass keeps the test to about two
hours). Mixed recordings always use automatic language. Our model is compared with Whisper small on
exactly these two conditions.

``guard.final_test`` verifies the frozen fingerprint, refuses a second run and logs our model's
pooled WER. Transcripts are appended to data/ml/results/hyp-test/ as they are produced, so a run
interrupted by sleep or power loss continues with the same command and still counts as one run.
Two processes share the graphics card; the order inside each keeps their combined memory under 6 GB.
"""

import argparse
import json
import re
import subprocess
import sys

from . import splits
from .evaluate import bootstrap_difference, details, mixed_by_language, pick_device, read_hypotheses, run_units
from .export import EXPORT
from .paths import LANGUAGES, MODELS, PIPELINE, RESULTS, ROOT
from .text import pooled

HYP = RESULTS / "hyp-test"
OURS = EXPORT.relative_to(MODELS).as_posix()
# system -> (sets it transcribes, language given only)
PLAN = {
    "cleartalk": (None, False),
    "small": (None, True),
    "turbo": (None, True),
    "kazakh": (("fleurs-kk", "ksc2-kk"), True),
    "pipeline:auto": (("mixed",), False),
}
# While the first process holds a small model (0.5 GB) the second holds large-v3-turbo (2.3 GB); when the
# first moves on to the large models (up to 4.7 GB for the pipeline) the second is on Whisper small.
GROUPS = {"first": ("cleartalk", "kazakh", "pipeline:auto"), "second": ("turbo", "small")}


def gate_c_approved() -> bool:
    text = (PIPELINE / "PIPELINE.md").read_text(encoding="utf-8")
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
        name = OURS if system == "cleartalk" else system
        print(f"== {system}", flush=True)
        run_units(name, units_for(system, units), path_of(system, f"{device}-{precision}"), device, precision,
                  forced_only=PLAN[system][1])


def hypotheses(system):
    device, precision = pick_device()
    return read_hypotheses(path_of(system, f"{device}-{precision}"))


def table(units):
    """Scores of every system on every test set, and our model against Whisper small with intervals."""
    rows, comparisons, by_language = [], {}, {}
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
        mixed = units[units.set == "mixed"]
        if len(mixed) and PLAN[system][0] in (None, ("mixed",)) and all(u in hyp for u in mixed.unit_id):
            by_language[system] = mixed_by_language(mixed, hyp)
    ours, small = hypotheses("cleartalk"), hypotheses("small")
    for set_name, group in units.groupby("set"):
        kind = "auto" if set_name == "mixed" else "forced"
        comparisons[set_name] = bootstrap_difference(details(group, small, kind), details(group, ours, kind))
    return {"scores": rows, "cleartalk_vs_small": comparisons, "mixed_by_language": by_language}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=sorted(GROUPS), help="internal: transcribe one group of systems")
    args = parser.parse_args()
    if not gate_c_approved():
        raise SystemExit("The final test needs Gate C in ml_pipeline/PIPELINE.md")
    units = splits.eval_units("test")
    if args.worker:
        run_group(args.worker, units)
        return
    sys.path.insert(0, str(PIPELINE))
    import guard

    out = RESULTS / "test-final.json"
    if out.exists():
        raise SystemExit(f"The final test was already run once; see {out}")
    from .mixed import render_split

    def transcribe_all(features):
        render_split("test")  # the frozen recipes become audio only now
        helper = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "cleartalk_ml.final_test", "--worker", "second"], cwd=ROOT)
        run_group("first", units)
        if helper.wait() != 0:
            raise RuntimeError("The second process did not finish; run the same command again to continue")
        ours = hypotheses("cleartalk")
        return [ours[unit_id]["auto"] for unit_id in features.unit_id]

    score = guard.final_test(transcribe_all, units, target="reference", metric_fn=splits.unit_wer,
                             state_dir=PIPELINE, pipeline_md=PIPELINE / "PIPELINE.md")
    result = {"cleartalk_wer_all_units_automatic_language": score, "units": int(len(units)), **table(units),
              "languages": list(LANGUAGES)}
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"final test done: ClearTalk pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)
    print("FINAL_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
