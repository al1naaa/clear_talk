"""The robustness table of the app before and after this round's protections (validation, language detected per turn).

    python scripts/robustness_final.py

Reads the stored validation transcripts only (no recognition): "before" is the app as round 4 began (stored per-turn
transcripts, repetition loops not cut), "after" is the app with every option adopted in this round
(cleartalk_ml.app_options) and the loop cut. Writes data/ml/results/robustness-final.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from cleartalk_ml import robustness
    from cleartalk_ml.app_options import adopted_flags, setting_name
    from cleartalk_ml.evaluate import details, pick_device, read_hypotheses, without_loops
    from cleartalk_ml.paths import RESULTS
    from cleartalk_ml.text import pooled

    device, precision = pick_device()
    backend = f"{device}-{precision}"
    final = setting_name(adopted_flags())
    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
    rows = []
    for condition in robustness.CONDITIONS:
        units = robustness.units(plan, condition)
        before = read_hypotheses(robustness.hyp_file(condition, "auto", backend))
        after = without_loops(read_hypotheses(robustness.hyp_file(condition, final, backend)))
        for set_name, group in units.groupby("set"):
            kind = "auto" if set_name in ("mixed", "long") else "forced"
            a, b = details(group, before, kind), details(group, after, kind)
            rows.append({"condition": condition, "set": set_name, "units": int(len(group)),
                         "before": pooled(a.to_dict("records"))["wer"], "after": pooled(b.to_dict("records"))["wer"],
                         "insertions_before": int(a.insertions.sum()), "insertions_after": int(b.insertions.sum()),
                         "reference_words": int(a.reference_words.sum())})
    out = {"final_setting": final, "flags": adopted_flags(), "rows": rows}
    (RESULTS / "robustness-final.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['condition']:10s} {row['set']:9s} {100 * row['before']:5.1f}% -> {100 * row['after']:5.1f}%")


if __name__ == "__main__":
    main()
