"""Round 4: the large model trained to cope with background noise - and its frozen test set ("test D").

Test D is test C (898 units that no system has transcribed; round 3 was not trained and never opened it) heard in
three ways: clean, with white noise at 10 dB SNR and with babble at 10 dB SNR. The noise recipes (seeds and babble
talkers) are frozen here before any training of this round; the noisy audio is rendered only for the final test.
Babble talkers of test D are training-role utterances that neither the training noise (cleartalk_ml.noise) nor the
validation noise (cleartalk_ml.robustness) uses.

    python -m cleartalk_ml.round4            # freeze test D
"""

import json
import sys

import numpy as np
import pandas as pd

from .paths import AUDIO, PIPELINE, PLANS

R4 = PIPELINE / "round4"
PLAN = PLANS / "testD.json"
FOLDER = AUDIO / "testD"
SEED = 2026101550
CONDITIONS = {"clean": None, "white-10": ("white", 10.0), "babble-10": ("babble", 10.0)}


def build_plan():
    if PLAN.is_file():
        return json.loads(PLAN.read_text(encoding="utf-8"))
    from .corpora import load_manifest
    from .noise import talker_pool
    from .robustness import PLAN as ROBUST
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "train") & table.keep & table.duration.between(3, 20)].sort_values("utt_id")
    taken = set(talker_pool()) | set(json.loads(ROBUST.read_text(encoding="utf-8"))["babble_talkers"])
    free = table[~table.path.isin(taken)]
    talkers = free.path.iloc[np.sort(np.random.default_rng(SEED).permutation(len(free))[:300])].tolist()
    plan = {"seed": SEED, "talkers": talkers}
    PLAN.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return plan


def test_d_units() -> pd.DataFrame:
    from .paths import ROOT
    from .round3 import test_c_units

    base = test_c_units()
    frames = []
    for condition in CONDITIONS:
        frame = base.copy()
        frame["condition"] = condition
        frame["source_unit"] = frame.unit_id
        frame["unit_id"] = condition + "::" + frame.unit_id
        frame["set"] = frame.set + "@" + condition
        if condition != "clean":
            frame["path"] = [(FOLDER / condition / f"{unit}.wav").relative_to(ROOT).as_posix() for unit in frame.source_unit]
        frames.append(frame)
    return pd.concat(frames).sort_values("unit_id").reset_index(drop=True)


def render(units, plan):
    """Write the noisy audio of test D (only at the final test)."""
    import soundfile as sf

    from .corpora import read_audio
    from .noise import add
    from .paths import ROOT

    clean_paths = dict(zip(units[units.condition == "clean"].source_unit, units[units.condition == "clean"].path))
    for number, (condition, spec) in enumerate(CONDITIONS.items()):
        if spec is None:
            continue
        for index, unit in enumerate(sorted(clean_paths)):
            out = FOLDER / condition / f"{unit}.wav"
            if out.is_file():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            audio = read_audio(clean_paths[unit])
            noisy = add(audio, {"kind": spec[0], "snr": spec[1], "seed": plan["seed"] + 100000 * number + index}, plan["talkers"])
            sf.write(out.with_suffix(".tmp"), noisy, 16000, subtype="PCM_16", format="WAV")
            out.with_suffix(".tmp").replace(out)
    return ROOT


def freeze(units) -> dict:
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    R4.mkdir(parents=True, exist_ok=True)
    state = R4 / guard.STATE_FILE
    fingerprint = guard._fingerprint(units)
    if state.exists():
        current = json.loads(state.read_text(encoding="utf-8"))
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test D is already frozen")
        return current
    record = {"test_fingerprint": fingerprint, "n_test": int(len(units)), "test_touches": 0, "duplicates_dropped": 0,
              "split": "test C (never opened) clean, with white noise 10 dB and with babble 10 dB (see cleartalk_ml/round4.py)",
              "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    units.to_csv(PLANS / "testD-units.csv", index=False, encoding="utf-8")
    return record


def main():
    plan = build_plan()
    units = test_d_units()
    print(units.groupby("condition").size().to_dict(), "babble talkers:", len(plan["talkers"]))
    print(freeze(units))


if __name__ == "__main__":
    main()
