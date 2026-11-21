"""Independent test recordings for the frozen final pipeline and paired reference.

Choose additional files after the six-item exploratory control subset, using
the same fixed order; hold out whole KSD recording groups and LibriSpeech readers
where available. Never use these controls in training or checkpoint selection.
"""

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pitch_control_inputs import GOLOS, ksd_subset, verify
from scripts.external_pilot import external, read_audio, seed
from scripts.validate_pitch_asr import INCUMBENT, score_sets, summary


def controls():
    pilot = external()
    blocked = {u for units in pilot.values() for u, _, _ in units}
    data = ROOT / "data/ml/noise_controls"
    metadata = ksd_subset()

    def group(key):
        return str(Path(key).parent / Path(key).stem.split("_")[0])

    seen = {group(u) for u, _, _ in pilot["kk"]}
    sets = {"kk": [], "ru": [], "en": []}
    for row in sorted(metadata, key=lambda r: seed(r["wav"].removeprefix("dataset/"))):
        uid = row["wav"].removeprefix("dataset/")
        file = data / "ksd" / (hashlib.sha256(uid.encode()).hexdigest()[:20] + ".wav")
        if group(uid) in seen or not file.is_file() or not row["text"].strip():
            continue
        try:
            audio = read_audio(file)
        except sf.LibsndfileError:
            continue
        if 2 <= len(audio) / 16000 <= 15:
            seen.add(group(uid))
            sets["kk"].append((uid, audio, row["text"]))
        if len(sets["kk"]) == 18:
            break
    verify(GOLOS)
    table = pq.read_table(GOLOS).to_pylist()
    for idx in sorted(range(len(table)), key=seed):
        uid = f"golos-{idx}"
        row = table[idx]
        if uid in blocked or not row["transcription"].strip():
            continue
        x = read_audio(io.BytesIO(row["audio"]["bytes"]))
        if 2 <= len(x) / 16000 <= 15:
            sets["ru"].append((uid, x, row["transcription"]))
        if len(sets["ru"]) == 18:
            break
    seen = {uid.split("-")[0] for uid, _, _ in pilot["en"]}
    rows = [
        json.loads(s)
        for s in (ROOT / "data/asr/plans/test.jsonl").read_text().splitlines()
    ]
    sources = sorted(
        (r["sources"][0] for r in rows if r["kind"] == "single"),
        key=lambda s: seed(s["utt_id"]),
    )
    for src in sources:
        if str(src["speaker"]) in seen or not 2 <= src["duration"] <= 15:
            continue
        seen.add(str(src["speaker"]))
        sets["en"].append(
            (src["utt_id"], read_audio(ROOT / src["path"]), src["text_ls"])
        )
        if len(sets["en"]) == 18:
            break
    assert all(len(rs) == 18 for rs in sets.values()) and not blocked.intersection(
        u for rs in sets.values() for u, _, _ in rs
    )
    expected = {
        "kk": "4ea0b7fe4f4d70e5ce44dbb9492dc0227e9175c17e5abc9383876a39da6612f8",
        "ru": "b2f801299e4de8ec53cec4176ad88a0644dc0df0f20b3bafe45d0b4abbb4a0a6",
        "en": "b9c4ff3135bd91e8433758f2af677c4a3fa165c33b4ff96a01849b4645883f82",
    }
    for language, rows in sets.items():
        order = "\n".join(uid for uid, _, _ in rows) + "\n"
        if hashlib.sha256(order.encode()).hexdigest() != expected[language]:
            raise ValueError(
                f"{language} control recordings differ from the frozen inventory"
            )
    return sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "models/pitch-asr")
    parser.add_argument("--threads", type=int, default=4)
    opt = parser.parse_args()
    step = json.loads((opt.work / "selection.json").read_text())["selected_step"]
    sets = controls()
    summary_by_step = {}
    for s in dict.fromkeys((0, step)):
        rs = score_sets(
            sets,
            opt.work / f"ct2-{s:03d}" if s else INCUMBENT,
            opt.work / "holdout",
            s,
            opt.threads,
        )
        summary_by_step[s] = summary(rs)
    (opt.work / "external-noise.json").write_text(
        json.dumps(dict(selected_step=step, systems=summary_by_step), indent=2) + "\n"
    )
