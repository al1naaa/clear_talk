"""Round 3: the small model again, with Kazakh broadcast speech added - and a third frozen test set ("test C").

Tests A and B are spent (each was opened once). Test C is drawn from test-role utterances that neither of them
used: not among their single utterances and not inside their mixed recordings. The question of this round is
Kazakh outside read speech, so test C is weighted towards KSC2: 100 utterances from each of its six sources,
plus guards for read Kazakh, Russian and English.

Only 18 FLEURS English test utterances are left unused (tests A and B took the rest, partly inside their mixed
recordings), so test C has an English guard of 18 utterances - enough to catch a collapse, not a small change -
and no mixed recordings (every mixed recording needs English). For English and mixed recordings the round's
guards are validation only; this was decided when test C was built, before any training of round 3.

    python -m cleartalk_ml.round3            # build the recipes of test C and freeze its units
"""

import json
import sys

import numpy as np
import pandas as pd

from .paths import LANGUAGES, PIPELINE, PLANS

R3 = PIPELINE / "round3"
SEED = 2026101430
PER_SOURCE = 100            # KSC2 utterances per source (six sources)
PER_SET = {"fleurs-kk": 100, "fleurs-ru": 80, "fleurs-en": 18, "golos-ru": 100}   # FLEURS: 253 kk, 187 ru, 18 en unused


def used_by_tests_a_and_b() -> set:
    """Utterances whose audio was transcribed in either of the first two final tests."""
    from .mixed import load_plan
    from .round2 import MIXED_SPLIT as B_SPLIT
    from .round2 import used_by_test_a

    used = set(used_by_test_a())
    b_units = pd.read_csv(PLANS / "testB-units.csv")
    used |= set(b_units[b_units.kind == "single"].unit_id)
    for recipe in load_plan(B_SPLIT):
        used |= {part["utt_id"] for part in recipe["parts"]}
    return used


def unused_test_table() -> pd.DataFrame:
    from .corpora import load_manifest
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "test") & table.keep & ~table.utt_id.isin(used_by_tests_a_and_b())]
    return table.sort_values("utt_id").reset_index(drop=True)


def _draw(frame, count, seed):
    frame = frame.sort_values("utt_id")
    if len(frame) <= count:
        return frame
    return frame.iloc[np.sort(np.random.default_rng(seed).permutation(len(frame))[:count])]


def single_rows() -> pd.DataFrame:
    table = unused_test_table()
    picked = []
    ksc2 = table[table.corpus == "ksc2"]
    for index, (_, frame) in enumerate(sorted(ksc2.groupby("domain"), key=lambda item: item[0])):
        picked.append(_draw(frame, PER_SOURCE, SEED + index))
    for index, (name, count) in enumerate(sorted(PER_SET.items())):
        corpus, language = name.split("-")
        picked.append(_draw(table[(table.corpus == corpus) & (table.language == language)], count, SEED + 50 + index))
    return pd.concat(picked).sort_values("utt_id")


def test_c_units() -> pd.DataFrame:
    """Scoring units of test C, in the same shape as cleartalk_ml.splits.eval_units (plus the KSC2 source)."""
    rows = [{"unit_id": row.utt_id, "set": f"{row.corpus}-{row.language}", "kind": "single", "language": row.language,
             "path": row.path, "duration": round(float(row.duration), 3), "reference": row.text_raw, "languages": row.language,
             "domain": row.domain if row.corpus == "ksc2" else ""} for row in single_rows().itertuples()]
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def build_training_plan():
    """Frozen recipes of the round-3 mixed training recordings (plan "train3"), from the round-3 training pool."""
    import os

    from .mixed import make_plan, plan_path, save_plan
    from .splits import COMPOSITIONS, role_table

    if plan_path("train3").is_file():
        return plan_path("train3")
    previous = os.environ.get("CLEARTALK_DATA")
    os.environ["CLEARTALK_DATA"] = "kazakh-broadcast"
    try:
        pool = role_table("train", "mix")
    finally:
        if previous is None:
            os.environ.pop("CLEARTALK_DATA")
        else:
            os.environ["CLEARTALK_DATA"] = previous
    return save_plan("train3", make_plan(pool, COMPOSITIONS["kazakh-broadcast"]["mixed"], SEED + 7, "mixed-train3"))


def check_fresh(units) -> dict:
    """Test C must share nothing with tests A and B, validation or training (of any composition)."""
    from .corpora import load_manifest
    from .mixed import load_plan, plan_path
    from .splits import cleaned, eval_units

    table = cleaned(load_manifest())
    role_of = dict(zip(table.utt_id, table.role))
    mine = set(units.unit_id)
    val = eval_units("val")
    val_used = set(val[val.kind == "single"].unit_id) | {p["utt_id"] for r in load_plan("val") for p in r["parts"]}
    train_used = set(table[table.role == "train"].utt_id)
    for split in ("train", "train3"):
        if plan_path(split).is_file():
            train_used |= {p["utt_id"] for r in load_plan(split) for p in r["parts"]}
    result = {"not test role": sum(role_of[u] != "test" for u in mine), "shared with tests A and B": len(mine & used_by_tests_a_and_b()),
              "shared with validation": len(mine & val_used), "shared with training": len(mine & train_used)}
    if any(result.values()):
        raise ValueError(f"Test C is not fresh: {result}")
    return result


def freeze(units) -> dict:
    """Record the fingerprint of test C in ml_pipeline/round3/.guard_state.json (as guard.split does)."""
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    R3.mkdir(parents=True, exist_ok=True)
    state = R3 / guard.STATE_FILE
    fingerprint = guard._fingerprint(units)
    if state.exists():
        current = json.loads(state.read_text(encoding="utf-8"))
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test C is already frozen")
        return current
    record = {"test_fingerprint": fingerprint, "n_test": int(len(units)), "test_touches": 0, "duplicates_dropped": 0,
              "split": "test-role utterances used by neither of the first two final tests (see cleartalk_ml/round3.py)",
              "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    units.to_csv(PLANS / "testC-units.csv", index=False, encoding="utf-8")
    return record


def main():
    units = test_c_units()
    print(units.groupby(["set", "domain"]).agg(units=("unit_id", "size"), minutes=("duration", lambda s: round(s.sum() / 60, 1))).to_string())
    print(check_fresh(units))
    print(freeze(units))
    print({language: int((units.language == language).sum()) for language in LANGUAGES})


if __name__ == "__main__":
    main()
