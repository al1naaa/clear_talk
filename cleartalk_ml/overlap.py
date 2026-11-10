"""Two people talking at once: validation recordings with overlapping speech, and an order-free error measure.

    python -m cleartalk_ml.overlap build      # freeze the recipes and render the audio (CPU)

Each recording holds two VALIDATION utterances (any of the three languages) at similar loudness; the second starts
while the first is still running, so 20-80% of the shorter one is spoken over. Nothing of the test sets is used.

When two people speak at once there is no single right order of their words, so the measure is cpWER
("concatenated minimum-permutation WER", as in the CHiME-6 challenge): the hypothesis text of every speaker label
is joined in time order, each reference speaker is matched to the hypothesis speaker that fits best, and the errors
of the best matching are counted. It punishes words given to the wrong person as well as wrong words.
"""

import argparse
import itertools
import json
import sys

import numpy as np

from .paths import AUDIO, PLANS, SAMPLE_RATE

SEED = 2026101560
COUNT = 100
PLAN = PLANS / "overlap-val.json"
FOLDER = AUDIO / "overlap-val"
SEP = " ||| "


def build():
    import soundfile as sf

    from .corpora import read_audio
    from .splits import eval_units

    if PLAN.is_file():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
    else:
        rng = np.random.default_rng(SEED)
        singles = eval_units("val")
        singles = singles[(singles.kind == "single") & singles.duration.between(3, 14)].sort_values("unit_id").reset_index(drop=True)
        recordings = []
        for index in range(COUNT):
            first, second = singles.iloc[rng.choice(len(singles), size=2, replace=False)].to_dict("records")
            shorter = min(first["duration"], second["duration"])
            overlap = float(rng.uniform(0.2, 0.8)) * shorter
            start = max(0.3, first["duration"] - overlap)
            levels = [float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))]
            recordings.append({"unit_id": f"overlap-val-{index:03d}", "duration": round(max(first["duration"], start + second["duration"]) + 0.3, 3),
                               "parts": [{"unit_id": first["unit_id"], "path": first["path"], "language": first["language"], "start": 0.0,
                                          "level_db": levels[0], "reference": first["reference"]},
                                         {"unit_id": second["unit_id"], "path": second["path"], "language": second["language"],
                                          "start": round(start, 3), "level_db": levels[1], "reference": second["reference"]}],
                               "overlap_seconds": round(overlap, 3)})
        plan = {"seed": SEED, "recordings": recordings}
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    for item in plan["recordings"]:
        path = FOLDER / f"{item['unit_id']}.wav"
        if path.is_file():
            continue
        out = np.zeros(int(round(item["duration"] * SAMPLE_RATE)), dtype=np.float32)
        for part in item["parts"]:
            audio = read_audio(part["path"]).astype(np.float32)
            audio = audio / (float(np.sqrt(np.mean(audio ** 2))) or 1.0) * 0.06 * 10 ** (part["level_db"] / 20)
            begin = int(round(part["start"] * SAMPLE_RATE))
            out[begin:begin + len(audio)] += audio[: max(0, len(out) - begin)]
        peak = float(np.max(np.abs(out)))
        out = out * (0.95 / peak) if peak > 0.95 else out
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, out, SAMPLE_RATE, subtype="PCM_16")
    return plan


def units(plan):
    import pandas as pd

    from .paths import ROOT

    rows = [{"unit_id": item["unit_id"], "set": "overlap", "kind": "mixed", "language": "mixed",
             "reference": SEP.join(p["reference"] for p in item["parts"]), "languages": "+".join(p["language"] for p in item["parts"]),
             "duration": item["duration"], "path": (FOLDER / f"{item['unit_id']}.wav").relative_to(ROOT).as_posix()} for item in plan["recordings"]]
    return pd.DataFrame(rows)


def cp_counts(references, turns) -> dict:
    """Errors of the best matching between reference speakers and hypothesis speaker labels."""
    from .text import counts

    by_speaker = {}
    for turn in sorted(turns, key=lambda t: t["start"]):
        by_speaker.setdefault(turn["speaker"], []).append(turn["text"])
    hypotheses = [" ".join(texts) for texts in by_speaker.values()]
    while len(hypotheses) < len(references):
        hypotheses.append("")
    best = None
    for chosen in itertools.permutations(range(len(hypotheses)), len(references)):
        rows = [counts(ref, hypotheses[h]) for ref, h in zip(references, chosen)]
        extra = [counts("", hypotheses[h]) for h in range(len(hypotheses)) if h not in chosen]   # words of unmatched labels
        errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows + extra)
        if best is None or errors < best[0]:
            best = (errors, rows + extra)
    total = {key: sum(r[key] for r in best[1]) for key in ("substitutions", "deletions", "insertions", "reference_words")}
    total["errors"] = best[0]
    return total


def cp_frame(plan, hyp):
    import pandas as pd

    rows = []
    for item in plan["recordings"]:
        row = hyp.get(item["unit_id"])
        if row is None:
            continue
        rows.append({"unit_id": item["unit_id"], **cp_counts([p["reference"] for p in item["parts"]], row.get("turns", []))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("build",))
    parser.parse_args()
    plan = build()
    print(len(plan["recordings"]), "recordings; overlap seconds median",
          round(float(np.median([r["overlap_seconds"] for r in plan["recordings"]])), 2))


if __name__ == "__main__":
    sys.exit(main())
