"""What kind of invented text the app produces, counted on VALIDATION transcripts (no test data).

    python -m cleartalk_ml.hallucinations

Four kinds are counted, per condition of the robustness check (cleartalk_ml.robustness) and per set:

* ``wrong_alphabet`` - the speech was Kazakh or Russian but the text came out in Latin letters (or English speech
  came out in Cyrillic): almost always an invented "translation" after a wrong language decision;
* ``loops`` - the same one to four words repeated four or more times in a row;
* ``too_fast`` - a turn with more than 25 characters per second of audio: more words than anyone can say;
* ``where_nobody_spoke`` - (mixed recordings, whose utterance times are known) a turn that does not overlap any
  utterance by at least half: text written over a pause or over background noise.
"""

import json
import re

import pandas as pd

from .paths import RESULTS

LATIN = re.compile(r"[a-z]")
CYRILLIC = re.compile(r"[а-яёәғқңөұүһі]")


def alphabet(text: str) -> str:
    text = text.lower()
    latin, cyrillic = len(LATIN.findall(text)), len(CYRILLIC.findall(text))
    if latin + cyrillic == 0:
        return "none"
    return "latin" if latin > cyrillic else "cyrillic"


def loops(text: str) -> int:
    words = text.lower().split()
    for size in range(1, 5):
        for start in range(0, max(0, len(words) - size * 4 + 1)):
            chunk = words[start:start + size]
            if all(words[start + size * k:start + size * (k + 1)] == chunk for k in range(1, 4)):
                return 1
    return 0


def audit(backend="cuda-int8_float32", setting="auto"):
    from . import robustness
    from .evaluate import read_hypotheses
    from .mixed import load_plan

    plan = json.loads(robustness.PLAN.read_text(encoding="utf-8"))
    spans = {recipe["mix_id"]: [(p["start"], p["end"]) for p in recipe["parts"]] for recipe in load_plan("val")}
    rows = []
    for condition in robustness.CONDITIONS:
        path = robustness.hyp_file(condition, setting, backend)
        if not path.is_file():
            continue
        hyp = read_hypotheses(path)
        units = robustness.units(plan, condition)
        for unit in units.itertuples():
            row = hyp.get(unit.unit_id)
            if row is None:
                continue
            turns = row.get("turns", [])
            expected = "latin" if unit.language == "en" else "cyrillic"
            wrong = sum(1 for t in turns if unit.kind == "single" and alphabet(t["text"]) not in (expected, "none"))
            fast = sum(1 for t in turns if len(t["text"]) / max(t["end"] - t["start"], 0.3) > 25)
            nobody = 0
            if unit.unit_id in spans:
                for t in turns:
                    overlap = sum(max(0.0, min(e, t["end"]) - max(s, t["start"])) for s, e in spans[unit.unit_id])
                    nobody += overlap < 0.5 * max(t["end"] - t["start"], 1e-3)
            rows.append({"condition": condition, "set": unit.set, "turns": len(turns), "wrong_alphabet": wrong,
                         "loops": sum(loops(t["text"]) for t in turns), "too_fast": fast, "where_nobody_spoke": nobody})
    frame = pd.DataFrame(rows)
    return frame.groupby(["condition", "set"]).sum(numeric_only=True)


def main():
    pd.set_option("display.width", 200)
    table = audit()
    print(table.to_string())
    (RESULTS / "hallucinations.json").write_text(table.reset_index().to_json(orient="records", indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
