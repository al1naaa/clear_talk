"""Re-score the frozen Test D and two English test sets using the full local transfer."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleartalk_asr.augment import render
from cleartalk_asr.mixtures import si_sdr
from cleartalk_asr.text import pair_counts
from cleartalk_ml.text import counts, pooled


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def score_noise():
    data = ROOT / "data/ml"
    with (data / "plans/testD-units.csv").open(encoding="utf-8") as stream:
        units = list(csv.DictReader(stream))
    expected = {r["unit_id"] for r in units}
    assert len(expected) == len(units) == 2694
    summary = []
    archived = json.loads((data / "results/test-final-D.json").read_text())
    for system, field in (("app-before", "before"), ("app-final", "final")):
        rows = records(data / f"results/hyp-testD/{system}__cuda-int8_float32.jsonl")
        predictions = {r["utt_id"]: r for r in rows}
        assert len(predictions) == len(rows) and expected == predictions.keys()
        groups = defaultdict(list)
        for unit in units:
            row = predictions[unit["unit_id"]]
            score = counts(unit["reference"], row["auto"])
            score["correct_label"] = row["detected"] == unit["language"]
            groups[unit["language"], unit["condition"]].append(score)
        for (language, condition), group in sorted(groups.items()):
            summary.append(
                dict(
                    system=system,
                    language=language,
                    condition=condition,
                    n=len(group),
                    exact_language_labels=sum(r["correct_label"] for r in group),
                    **pooled(group),
                )
            )
        for condition in archived["condition"]:
            joined = [
                r for (_, c), rows in groups.items() if c == condition for r in rows
            ]
            assert (
                abs(pooled(joined)["wer"] - archived["condition"][condition][field])
                < 1e-10
            )
    return summary


def acoustics(recipe_id, recipe):
    split = "test" if recipe else "v4test"
    separated = []
    for i in range(2):
        track, rate = sf.read(ROOT / f"data/asr/mixtures/{split}/{recipe_id}-{i}.wav")
        assert rate == 8000
        separated.append(track)
    if recipe:
        rebuilt, target, _ = render(recipe)
        stored, rate = sf.read(
            ROOT / f"data/asr/mixtures/test/{recipe_id}-mix.wav", dtype="int16"
        )
        assert rate == 16000 and np.max(np.abs(stored.astype(int) - rebuilt)) <= 1
        mixture = resample_poly(stored.astype(float) / 32768, 1, 2)
        target = resample_poly(target.astype(float), 1, 2, axis=0)
    else:
        with np.load(
            ROOT / f"archive/evaluation/v4/asr_test/audio/{recipe_id}.npz"
        ) as cache:
            mixture, target = (
                cache["mixture"].astype(float),
                cache["target"].astype(float),
            )
    assert target.shape == (len(mixture), 2) and all(
        t.shape == mixture.shape for t in separated
    )
    before = np.mean([si_sdr(mixture, t) for t in target.T])
    after = max(
        np.mean([si_sdr(separated[i ^ swap], t) for i, t in enumerate(target.T)])
        for swap in (0, 1)
    )
    return float(before), float(after)


def score_pairs():
    path = ROOT / "data/asr"
    test = json.loads((path / "results/test-final.json").read_text())
    recipes = {
        r["id"]: r for r in records(path / "plans/test.jsonl") if r["kind"] == "pair"
    }
    summary = []
    for split, expected in (("test", 300), ("v4test", 96)):
        sums = defaultdict(int)
        signal = []
        for idx, (role, kind) in enumerate(zip(test["set"], test["kind"], strict=True)):
            if role != split or kind != "pair":
                continue
            refs = test["reference"][idx].split(" ||| ")
            assert len(refs) == 2
            for model in ("baseline", "cleartalk"):
                hypotheses = test["hyp_" + model][idx].split(" ||| ")
                score = pair_counts(refs, hypotheses)
                sums[model] += score["errors"]
            sums["words"] += score["reference_words"]
            rid = test["unit_id"][idx].split(":", 1)[1]
            signal.append(acoustics(rid, recipes[rid] if split == "test" else None))
        assert len(signal) == expected
        before, after = np.mean(signal, axis=0)
        summary.append(
            dict(
                split=split,
                n=expected,
                **sums,
                baseline_cpwer=sums["baseline"] / sums["words"],
                cleartalk_cpwer=sums["cleartalk"] / sums["words"],
                mix_si_sdr_db=before,
                separated_si_sdr_db=after,
                si_sdri_db=after - before,
            )
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    result = dict(noise=score_noise(), overlap=score_pairs())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
