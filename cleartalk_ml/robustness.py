"""How the app copes with noise, very short recordings and long recordings - measured on VALIDATION audio.

    python -m cleartalk_ml.robustness build     # freeze the recipes and render the audio (CPU)
    python -m cleartalk_ml.robustness run       # transcribe through the app's pipeline (graphics card), cached
    python -m cleartalk_ml.robustness report    # the table

Recordings (all from validation-role utterances; no test audio is touched):

* ``mixed`` - the first 100 mixed validation recordings (two to four utterances in different languages);
* ``short`` - 150 single utterances of at most 4 s: 50 Kazakh (KSC2), 50 Russian (Golos), 50 English (the
  shortest FLEURS ones);
* ``long`` - 6 assembled conversations of about seven minutes, utterances of the three languages one after
  another with pauses of 0.3-1.5 s, so the app's turn finding, language detection and memory handling are
  exercised on long audio.

Conditions: clean; white noise at 20 and 10 dB SNR; babble (four other people talking at once) at 10 and 5 dB SNR.
Long conversations get clean, white 10 dB and babble 10 dB. SNR is measured against the RMS of the whole clean
recording, pauses included. Babble is made of TRAINING-role utterances, never of validation or test speech.
Everything is seeded and frozen in data/ml/plans/robustness.json.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .paths import AUDIO, LANGUAGES, PLANS, RESULTS, SAMPLE_RATE

SEED = 2026101501
PLAN = PLANS / "robustness.json"
FOLDER = AUDIO / "robustness"
CONDITIONS = {"clean": None, "white-20": ("white", 20), "white-10": ("white", 10), "babble-10": ("babble", 10), "babble-5": ("babble", 5)}
LONG_CONDITIONS = ("clean", "white-10", "babble-10")
LONG_COUNT, LONG_SECONDS = 6, 420
SEP = " ||| "


def build():
    """Freeze which recordings, which babble talkers and which noise seeds; then render the audio."""
    from .corpora import load_manifest
    from .splits import cleaned, eval_units

    if PLAN.is_file():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
    else:
        rng = np.random.default_rng(SEED)
        table = cleaned(load_manifest())
        val = eval_units("val")
        mixed = val[val.kind == "mixed"].sort_values("unit_id").head(100)
        singles = val[val.kind == "single"]
        short = []
        for set_name, language in (("ksc2-kk", "kk"), ("golos-ru", "ru"), ("fleurs-en", "en")):
            pool = singles[(singles.set == set_name) & (singles.duration <= 4.0)].sort_values("unit_id")
            if len(pool) < 50:   # FLEURS has few utterances this short: take its shortest
                pool = singles[singles.set == set_name].sort_values(["duration", "unit_id"]).head(50)
            short += pool.iloc[np.sort(rng.permutation(len(pool))[:50])].to_dict("records")
        # Long conversations: validation utterances of the three languages in turn.
        by_language = {language: singles[singles.language == language].sort_values("unit_id").to_dict("records") for language in LANGUAGES}
        order = {language: list(rng.permutation(len(items))) for language, items in by_language.items()}
        longs = []
        for index in range(LONG_COUNT):
            parts, cursor = [], 0.0
            while cursor < LONG_SECONDS:
                language = LANGUAGES[len(parts) % 3] if rng.random() < 0.7 else str(rng.choice(LANGUAGES))
                if not order[language]:
                    order[language] = list(rng.permutation(len(by_language[language])))
                item = by_language[language][order[language].pop()]
                gap = float(rng.uniform(0.3, 1.5)) if parts else 0.5
                parts.append({"unit_id": item["unit_id"], "path": item["path"], "language": language, "start": round(cursor + gap, 3),
                              "reference": item["reference"]})
                cursor += gap + float(item["duration"])
            longs.append({"unit_id": f"long-{index:02d}", "parts": parts, "duration": round(cursor + 0.5, 3)})
        talkers = table[(table.role == "train") & table.keep & table.duration.between(4, 20)].sort_values("utt_id")
        babble = talkers.iloc[np.sort(rng.permutation(len(talkers))[:400])].path.tolist()
        plan = {"seed": SEED, "mixed": mixed[["unit_id", "path", "duration", "reference", "languages"]].to_dict("records"),
                "short": [{k: item[k] for k in ("unit_id", "path", "duration", "reference", "language", "set")} for item in short],
                "long": longs, "babble_talkers": babble}
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    render(plan)
    return plan


def _clean_audio(kind, item):
    from .corpora import read_audio

    if kind != "long":
        return read_audio(item["path"]).astype(np.float32)
    out = np.zeros(int(round(item["duration"] * SAMPLE_RATE)), dtype=np.float32)
    for part in item["parts"]:
        audio = read_audio(part["path"]).astype(np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2))) or 1.0
        audio = audio * (0.08 / rms)     # every talker at the same loudness
        start = int(round(part["start"] * SAMPLE_RATE))
        audio = audio[: max(0, len(out) - start)]
        out[start:start + len(audio)] = audio
    return out


def _noise(kind, length, rng, talkers):
    from .corpora import read_audio

    if kind == "white":
        return rng.standard_normal(length).astype(np.float32)
    mix = np.zeros(length, dtype=np.float32)
    for path in rng.choice(talkers, size=4, replace=False):
        voice = read_audio(path).astype(np.float32)
        voice = voice / (float(np.sqrt(np.mean(voice ** 2))) or 1.0)
        reps = int(np.ceil(length / max(len(voice), 1)))
        voice = np.tile(voice, reps)
        offset = int(rng.integers(0, max(1, len(voice) - length)))
        mix += voice[offset:offset + length]
    return mix


def with_noise(clean, condition, rng, talkers):
    spec = CONDITIONS[condition]
    if spec is None:
        return clean
    kind, snr = spec
    noise = _noise(kind, len(clean), rng, talkers)
    speech_rms = float(np.sqrt(np.mean(clean ** 2))) or 1e-4
    noise *= speech_rms / (float(np.sqrt(np.mean(noise ** 2))) or 1.0) / (10 ** (snr / 20))
    out = clean + noise
    peak = float(np.max(np.abs(out)))
    return out * (0.95 / peak) if peak > 0.95 else out


def render(plan):
    import soundfile as sf

    talkers = plan["babble_talkers"]
    for kind in ("mixed", "short", "long"):
        for index, item in enumerate(plan[kind]):
            conditions = LONG_CONDITIONS if kind == "long" else tuple(CONDITIONS)
            clean = None
            for number, condition in enumerate(conditions):
                path = FOLDER / condition / f"{item['unit_id']}.wav"
                if path.is_file():
                    continue
                clean = _clean_audio(kind, item) if clean is None else clean
                rng = np.random.default_rng(plan["seed"] + 1000 * number + index + (500000 if kind == "long" else 0))
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path.with_suffix(".tmp"), with_noise(clean, condition, rng, talkers), SAMPLE_RATE, subtype="PCM_16", format="WAV")
                path.with_suffix(".tmp").replace(path)
        print(f"rendered {kind}", flush=True)


def units(plan, condition):
    """Scoring units of one condition, in the shape cleartalk_ml.evaluate expects."""
    from .paths import ROOT

    rows = []
    for item in plan["mixed"]:
        rows.append({"unit_id": item["unit_id"], "set": "mixed", "kind": "mixed", "language": "mixed", "reference": item["reference"],
                     "languages": item["languages"], "duration": item["duration"]})
    for item in plan["short"]:
        rows.append({"unit_id": item["unit_id"], "set": f"short-{item['language']}", "kind": "single", "language": item["language"],
                     "reference": item["reference"], "languages": item["language"], "duration": item["duration"]})
    if condition in LONG_CONDITIONS:
        for item in plan["long"]:
            rows.append({"unit_id": item["unit_id"], "set": "long", "kind": "mixed", "language": "mixed",
                         "reference": SEP.join(part["reference"] for part in item["parts"]),
                         "languages": "+".join(part["language"] for part in item["parts"]), "duration": item["duration"]})
    frame = pd.DataFrame(rows)
    frame["path"] = [(FOLDER / condition / f"{unit}.wav").relative_to(ROOT).as_posix() for unit in frame.unit_id]
    return frame


SETTINGS = {"auto": "pipeline:auto", "mixed": "pipeline:mixed"}


def hyp_file(condition, setting, backend):
    return RESULTS / "hyp-robust" / f"{condition}__{setting}__{backend}.jsonl"


def run(settings=("auto", "mixed")):
    from .awake import keep_awake
    from .evaluate import pick_device, run_units

    keep_awake()
    plan = build()
    device, precision = pick_device()
    for condition in CONDITIONS:
        frame = units(plan, condition)
        for setting in settings:
            chosen = frame if setting == "auto" else frame[frame.set.isin(["mixed", "long"])].reset_index(drop=True)
            run_units(SETTINGS[setting], chosen, hyp_file(condition, setting, f"{device}-{precision}"), device, precision)
    print("ROBUSTNESS_DONE", flush=True)


def table(backend="cuda-int8_float32"):
    from .evaluate import details, read_hypotheses
    from .text import pooled

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    rows = []
    for condition in CONDITIONS:
        frame = units(plan, condition)
        for setting in SETTINGS:
            path = hyp_file(condition, setting, backend)
            if not path.is_file():
                continue
            hyp = read_hypotheses(path)
            for set_name, group in frame.groupby("set"):
                if any(u not in hyp for u in group.unit_id):
                    continue
                kind = "auto" if set_name in ("mixed", "long") else "forced"
                score = pooled(details(group, hyp, kind).to_dict("records"))
                rows.append({"condition": condition, "setting": setting, "set": set_name, "units": len(group), "wer": score["wer"],
                             "words": score["reference_words"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("build", "run", "report"))
    args = parser.parse_args()
    if args.action == "build":
        plan = build()
        print({kind: len(plan[kind]) for kind in ("mixed", "short", "long")}, "long minutes:",
              [round(item["duration"] / 60, 1) for item in plan["long"]])
    elif args.action == "run":
        run()
    else:
        frame = table()
        pd.set_option("display.width", 200)
        print(frame.pivot_table(index=["setting", "set"], columns="condition", values="wer").mul(100).round(1).to_string())
        out = RESULTS / "robustness.json"
        out.write_text(frame.to_json(orient="records", indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
