"""Round 5: the fifth frozen test set ("test E"), built only from test-role utterances that no test so far used.

    python -m cleartalk_ml.round5            # freeze test E (recipes only; the audio is rendered at the final test)

Tests A, B and D are spent. Still unused among the test-role utterances: 3,457 KSC2 (Kazakh crowdsourced, TV news,
podcasts, radio, talkshow, parliament), 144 FLEURS Kazakh, 103 FLEURS Russian and 9,175 Golos Russian phrases; no
English is left, so English stays measured on validation only. Test E holds four parts, each frozen here before any
work of round 5:

* ``single``  - one utterance per recording (KSC2 by domain, FLEURS kk/ru, Golos), clean, white noise 10 dB, babble 10 dB;
* ``mixed``   - 2-4 utterances in Kazakh and Russian one after another (20-30 s), clean and babble 10 dB;
* ``overlap`` - two utterances, the second starting while the first is still running (20-80% spoken over), clean and babble 10 dB;
* ``talk``    - 5-minute conversations of utterances with short pauses (0.1-1.2 s) and occasional interruptions, clean and babble 10 dB.

Babble talkers are training-role utterances that neither the training noise, the validation noise nor test D uses.
"""

import json
import sys

import numpy as np
import pandas as pd

from .paths import AUDIO, PIPELINE, PLANS, ROOT, SAMPLE_RATE

R5 = PIPELINE / "round5"
PLAN = PLANS / "testE.json"
UNITS = PLANS / "testE-units.csv"
FOLDER = AUDIO / "testE"
SEED = 2026101800
SEP = " ||| "
NOISY = {"white-10": ("white", 10.0), "babble-10": ("babble", 10.0)}
SINGLE_COUNTS = {"ksc2": 360, "fleurs-kk": 144, "fleurs-ru": 103, "golos": 150}
COUNTS = {"mixed": 80, "overlap": 80, "talk": 6}
TALK_SECONDS = 300


def unused_test_table() -> pd.DataFrame:
    from .corpora import load_manifest
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "test") & table.keep & table.exists].copy()
    used = set()
    for name in ("test-units.csv", "testB-units.csv", "testC-units.csv"):
        frame = pd.read_csv(PLANS / name)
        used |= set(frame["unit_id"])
    for name in ("mixed-test.jsonl", "mixed-testB.jsonl"):
        for line in (PLANS / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                used |= {part["utt_id"] for part in json.loads(line)["parts"]}
    return table[~table.utt_id.isin(used)].sort_values("utt_id").reset_index(drop=True)


def babble_talkers(rng) -> list[str]:
    from .corpora import load_manifest
    from .noise import talker_pool
    from .robustness import PLAN as ROBUST
    from .round4 import PLAN as TEST_D
    from .splits import cleaned

    table = cleaned(load_manifest())
    table = table[(table.role == "train") & table.keep & table.duration.between(3, 20)].sort_values("utt_id")
    taken = set(talker_pool()) | set(json.loads(ROBUST.read_text(encoding="utf-8"))["babble_talkers"])
    taken |= set(json.loads(TEST_D.read_text(encoding="utf-8"))["talkers"])
    free = table[~table.path.isin(taken)]
    return free.path.iloc[np.sort(rng.permutation(len(free))[:300])].tolist()


def build_plan() -> dict:
    if PLAN.is_file():
        return json.loads(PLAN.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    table = unused_test_table()
    table["key"] = [("golos" if c == "golos" else "ksc2" if c == "ksc2" else f"{c}-{lang}") for c, lang in zip(table.corpus, table.language)]
    row = lambda r: {"utt_id": r["utt_id"], "path": r["path"], "language": r["language"], "reference": r["text_raw"],  # noqa: E731
                     "duration": round(float(r["duration"]), 3), "domain": r["domain"]}
    taken = set()

    # single utterances; KSC2 spread over its domains (natural broadcast speech included)
    singles = []
    for key, count in SINGLE_COUNTS.items():
        pool = table[(table.key == key) & table.duration.between(1.0, 25.0)]
        if key == "ksc2":
            per = []
            domains = sorted(pool.domain.unique())
            for domain in domains:
                part = pool[pool.domain == domain]
                take = min(len(part), count // len(domains))
                per.append(part.iloc[np.sort(rng.permutation(len(part))[:take])])
            chosen = pd.concat(per)
            rest = pool[~pool.utt_id.isin(chosen.utt_id)]
            chosen = pd.concat([chosen, rest.iloc[np.sort(rng.permutation(len(rest))[:count - len(chosen)])]])
        else:
            chosen = pool.iloc[np.sort(rng.permutation(len(pool))[:min(count, len(pool))])]
        singles += [row(r) for r in chosen.to_dict("records")]
    taken |= {s["utt_id"] for s in singles}

    pool = table[~table.utt_id.isin(taken) & table.duration.between(2.0, 14.0)].reset_index(drop=True)
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    cursor = 0

    def take(n=1, language=None):
        nonlocal cursor
        out = []
        while len(out) < n:
            r = pool.iloc[cursor % len(pool)]
            cursor += 1
            if r["utt_id"] in taken or (language and r["language"] != language):
                continue
            taken.add(r["utt_id"])
            out.append(row(r.to_dict()))
        return out

    # mixed recordings: Kazakh and Russian alternate, 20-30 s
    mixed = []
    for index in range(COUNTS["mixed"]):
        language = "kk" if rng.random() < 0.5 else "ru"
        parts, cursor_s = [], float(rng.uniform(0.2, 0.6))
        while cursor_s < 20.0 and len(parts) < 4:
            utt = take(1, language)[0]
            parts.append({**utt, "start": round(cursor_s, 3), "level_db": float(rng.uniform(-3, 3))})
            cursor_s += utt["duration"] + float(rng.uniform(0.3, 1.5))
            language = "ru" if language == "kk" else "kk"
        mixed.append({"unit_id": f"testE-mixed-{index:03d}", "parts": parts, "duration": round(cursor_s + 0.3, 3)})

    # two people at once
    overlaps = []
    for index in range(COUNTS["overlap"]):
        first, second = (take(1, "kk" if rng.random() < 0.5 else "ru")[0] for _ in range(2))   # Kazakh and Russian equally often
        shorter = min(first["duration"], second["duration"])
        start = max(0.3, first["duration"] - float(rng.uniform(0.2, 0.8)) * shorter)
        overlaps.append({"unit_id": f"testE-overlap-{index:03d}", "duration": round(max(first["duration"], start + second["duration"]) + 0.3, 3),
                         "parts": [{**first, "start": 0.0, "level_db": float(rng.uniform(-3, 3))},
                                   {**second, "start": round(start, 3), "level_db": float(rng.uniform(-3, 3))}],
                         "overlap_seconds": round(first["duration"] - start if start < first["duration"] else 0.0, 3)})

    # conversations: short pauses, about one in ten turns starts before the previous one ends
    talks = []
    for index in range(COUNTS["talk"]):
        parts, cursor_s = [], 0.5
        while cursor_s < TALK_SECONDS:
            utt = take(1, "kk" if rng.random() < 0.5 else "ru")[0]
            parts.append({**utt, "start": round(cursor_s, 3), "level_db": float(rng.uniform(-3, 3))})
            gap = float(rng.uniform(-1.5, -0.3)) if rng.random() < 0.1 else float(rng.uniform(0.1, 1.2))
            cursor_s = max(cursor_s + 0.5, cursor_s + utt["duration"] + gap)
        talks.append({"unit_id": f"testE-talk-{index:02d}", "parts": parts, "duration": round(max(p["start"] + p["duration"] for p in parts) + 0.5, 3)})

    plan = {"seed": SEED, "single": singles, "mixed": mixed, "overlap": overlaps, "talk": talks, "talkers": babble_talkers(rng)}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    return plan


def units(plan=None) -> pd.DataFrame:
    """Scoring units of test E in the shape cleartalk_ml.evaluate expects (one row per recording and condition)."""
    plan = plan or build_plan()
    rows = []
    for item in plan["single"]:
        corpus = "golos" if "golos" in item["utt_id"] else "ksc2" if item["utt_id"].startswith("ksc2") else "fleurs"
        for condition in ("clean",) + tuple(NOISY):
            rows.append({"unit_id": f"{condition}::{item['utt_id']}", "part": "single", "condition": condition, "set": f"{corpus}-{item['language']}@{condition}",
                         "kind": "single", "language": item["language"], "reference": item["reference"], "languages": item["language"],
                         "duration": item["duration"], "domain": item["domain"],
                         "path": (FOLDER / condition / f"{item['utt_id']}.wav").relative_to(ROOT).as_posix()})
    for part in ("mixed", "overlap", "talk"):
        for item in plan[part]:
            for condition in ("clean", "babble-10"):
                rows.append({"unit_id": f"{condition}::{item['unit_id']}", "part": part, "condition": condition, "set": f"{part}@{condition}",
                             "kind": "mixed", "language": "mixed", "reference": SEP.join(p["reference"] for p in item["parts"]),
                             "languages": "+".join(dict.fromkeys(p["language"] for p in item["parts"])), "duration": item["duration"], "domain": "",
                             "path": (FOLDER / condition / f"{item['unit_id']}.wav").relative_to(ROOT).as_posix()})
    return pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)


def render(plan=None):
    """Write the audio of test E (only at the final test)."""
    import soundfile as sf

    from .corpora import read_audio
    from .noise import add

    plan = plan or build_plan()
    jobs = [(item["utt_id"], [{"path": item["path"], "start": 0.0, "level_db": 0.0}], item["duration"] + 0.0, list(("clean",) + tuple(NOISY)))
            for item in plan["single"]]
    jobs += [(item["unit_id"], item["parts"], item["duration"], ["clean", "babble-10"]) for part in ("mixed", "overlap", "talk") for item in plan[part]]
    for number, (name, parts, duration, conditions) in enumerate(jobs):
        clean = None
        for k, condition in enumerate(conditions):
            out = FOLDER / condition / f"{name}.wav"
            if out.is_file():
                continue
            if clean is None:
                if len(parts) == 1 and parts[0]["level_db"] == 0.0:
                    clean = read_audio(parts[0]["path"]).astype(np.float32)
                else:
                    clean = np.zeros(int(round(duration * SAMPLE_RATE)), dtype=np.float32)
                    for p in parts:
                        audio = read_audio(p["path"]).astype(np.float32)
                        audio = audio / (float(np.sqrt(np.mean(audio ** 2))) or 1.0) * 0.06 * 10 ** (p["level_db"] / 20)
                        begin = int(round(p["start"] * SAMPLE_RATE))
                        clean[begin:begin + len(audio)] += audio[: max(0, len(clean) - begin)]
                    peak = float(np.max(np.abs(clean)))
                    clean = clean * (0.95 / peak) if peak > 0.95 else clean
            audio = clean if condition == "clean" else add(clean, {"kind": NOISY[condition][0], "snr": NOISY[condition][1],
                                                                  "seed": plan["seed"] + 1000003 * (k + 1) + number}, plan["talkers"])
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out.with_suffix(".tmp"), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
            out.with_suffix(".tmp").replace(out)


def check_fresh(plan) -> dict:
    """Nothing of test E was used by a test so far, by validation or by training."""
    from .corpora import load_manifest
    from .splits import cleaned

    ids = [s["utt_id"] for s in plan["single"]] + [p["utt_id"] for part in ("mixed", "overlap", "talk") for item in plan[part] for p in item["parts"]]
    roles = cleaned(load_manifest()).set_index("utt_id").role
    return {"utterances": len(ids), "distinct": len(set(ids)), "all_test_role": bool(all(roles.get(i) == "test" for i in ids)),
            "unused_before": bool(set(ids) <= set(unused_test_table().utt_id))}


def freeze(frame) -> dict:
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    R5.mkdir(parents=True, exist_ok=True)
    state = R5 / guard.STATE_FILE
    fingerprint = guard._fingerprint(frame)
    if state.exists():
        current = json.loads(state.read_text(encoding="utf-8"))
        if current["test_fingerprint"] != fingerprint:
            raise ValueError("A different test E is already frozen")
        return current
    record = {"test_fingerprint": fingerprint, "n_test": int(len(frame)), "test_touches": 0, "duplicates_dropped": 0,
              "split": "test E: unused test-role utterances; single / mixed / overlap / talk, clean and noisy (see cleartalk_ml/round5.py)",
              "created": pd.Timestamp.today().date().isoformat()}
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    frame.to_csv(UNITS, index=False, encoding="utf-8")
    return record


def main():
    plan = build_plan()
    frame = units(plan)
    print(frame.groupby(["part", "condition"]).size().to_dict())
    print(check_fresh(plan))
    print(freeze(frame))


if __name__ == "__main__":
    sys.exit(main())
