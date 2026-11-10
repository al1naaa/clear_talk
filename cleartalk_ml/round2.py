"""Round 2: a second frozen test set and the base models of the second round of training.

The first test set ("test A", ml_pipeline/.guard_state.json) was opened once and is spent. A new model
needs test audio that no system has transcribed, so "test B" is drawn from the test-role utterances
that test A did not use: not among its single utterances and not inside its mixed recordings.
Mixed recordings of test B are assembled from the same unused utterances by new seeded recipes.

    python -m cleartalk_ml.round2            # build the recipes of test B and freeze its units
"""

import json
import sys

import numpy as np
import pandas as pd

from .paths import LANGUAGES, PIPELINE, PLANS

R2 = PIPELINE / "round2"
PER_SET = 170               # units per set; FLEURS English has 347 unused test utterances
SEED = 2026101320
MIXED_SPLIT = "testB"       # data/ml/plans/mixed-testB.jsonl, audio in data/ml/audio/mixed/testB/
MIX_POOL = {"golos": 600, "ksc2": 600}
SEP = " ||| "


def used_by_test_a() -> set:
    """Utterances whose audio was transcribed in the first final test."""
    from .mixed import load_plan
    from .splits import EVAL_MIXED, eval_units

    units = eval_units("test")
    used = set(units[units.kind == "single"].unit_id)
    for recipe in load_plan("test")[:EVAL_MIXED["test"]]:
        used |= {part["utt_id"] for part in recipe["parts"]}
    return used


def unused_test_table() -> pd.DataFrame:
    """Kept test-role utterances that test A never touched."""
    from .corpora import load_manifest
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "test") & table.keep & ~table.utt_id.isin(used_by_test_a())]
    return table.sort_values("utt_id").reset_index(drop=True)


def _draw(frame, count, seed):
    frame = frame.sort_values("utt_id")
    if len(frame) <= count:
        return frame
    return frame.iloc[np.sort(np.random.default_rng(seed).permutation(len(frame))[:count])]


def build_mixed_plan():
    """Seeded recipes of the mixed recordings of test B (written once, then frozen)."""
    from .mixed import make_plan, plan_path, save_plan

    if plan_path(MIXED_SPLIT).is_file():
        return plan_path(MIXED_SPLIT)
    table = unused_test_table()
    pool = [frame if corpus not in MIX_POOL else _draw(frame, MIX_POOL[corpus], SEED + 100 + len(corpus))
            for corpus, frame in table.groupby("corpus")]
    return save_plan(MIXED_SPLIT, make_plan(pd.concat(pool), PER_SET, SEED, "mixed-testB"))


def test_b_units() -> pd.DataFrame:
    """Scoring units of test B, in the same shape as cleartalk_ml.splits.eval_units."""
    from .mixed import plan_table

    table = unused_test_table()
    rows = []
    for index, ((corpus, language), frame) in enumerate(sorted(table.groupby(["corpus", "language"]), key=lambda item: item[0])):
        for row in _draw(frame, PER_SET, SEED + index).itertuples():
            rows.append({"unit_id": row.utt_id, "set": f"{corpus}-{language}", "kind": "single", "language": language,
                         "path": row.path, "duration": round(float(row.duration), 3), "reference": row.text_raw,
                         "languages": language})
    for row in plan_table(MIXED_SPLIT).itertuples():
        rows.append({"unit_id": row.mix_id, "set": "mixed", "kind": "mixed", "language": "mixed", "path": row.path,
                     "duration": round(float(row.duration), 3),
                     "reference": SEP.join(text for _, text in row.segments), "languages": row.languages})
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def check_fresh(units) -> dict:
    """Test B must share nothing with test A, validation or training."""
    from .corpora import load_manifest
    from .mixed import load_plan
    from .splits import cleaned, eval_units, role_table

    table = cleaned(load_manifest())
    role_of = dict(zip(table.utt_id, table.role))
    inside = {part["utt_id"] for recipe in load_plan(MIXED_SPLIT) for part in recipe["parts"]}
    mine = set(units[units.kind == "single"].unit_id) | inside
    val = eval_units("val")
    val_used = set(val[val.kind == "single"].unit_id) | {p["utt_id"] for r in load_plan("val") for p in r["parts"]}
    train_used = set(role_table("train", "single").utt_id) | {p["utt_id"] for r in load_plan("train") for p in r["parts"]}
    result = {"not test role": sum(role_of[u] != "test" for u in mine), "shared with test A": len(mine & used_by_test_a()),
              "shared with validation": len(mine & val_used), "shared with training": len(mine & train_used)}
    if any(result.values()):
        raise ValueError(f"Test B is not fresh: {result}")
    return result


def freeze(units) -> dict:
    """Record the fingerprint of test B in ml_pipeline/round2/.guard_state.json (as guard.split does)."""
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    R2.mkdir(parents=True, exist_ok=True)
    state = R2 / guard.STATE_FILE
    fingerprint = guard._fingerprint(units)
    if state.exists():
        current = json.loads(state.read_text(encoding="utf-8"))
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test B is already frozen")
        return current
    record = {"test_fingerprint": fingerprint, "n_test": int(len(units)), "test_touches": 0, "duplicates_dropped": 0,
              "split": "test-role utterances not used by the first final test (see cleartalk_ml/round2.py)",
              "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    units.to_csv(PLANS / "testB-units.csv", index=False, encoding="utf-8")
    return record


def main():
    build_mixed_plan()
    units = test_b_units()
    print(units.groupby("set").agg(units=("unit_id", "size"), hours=("duration", lambda s: round(s.sum() / 3600, 2))).to_string())
    print(check_fresh(units))
    print(freeze(units))
    print({language: int((units.language == language).sum()) for language in LANGUAGES})


if __name__ == "__main__":
    main()
