"""Small fixed-seed noise enhancement and two-voice control experiment on local public audio.

Run in the full transfer tree. Only aggregate metrics belong in the public repo.
"""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import istft, resample_poly, stft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pitch_control_inputs import GOLOS, ksd_subset, verify
from backend.config import Settings
from backend.conversation import ConversationPipeline
from scripts.pitch_scoring import signal_scores, stream_errors
from cleartalk_ml.text import counts, pooled

SEED = 20261121
LANGS = ("kk", "ru", "en")
BINS = np.linspace(-15, 40, 56)
FRAMES = dict(fs=16000, nperseg=512, noverlap=384)


def read_audio(source):
    x, rate = sf.read(source, dtype="float32")
    if x.ndim == 2:
        x = x.mean(axis=1)
    return resample_poly(x, 16000, rate).astype("float32")


def seed(key):
    return int.from_bytes(hashlib.sha256(f"{SEED}/{key}".encode()).digest()[:8], "big")


def noisy(x, key, db):
    rng = np.random.default_rng(seed(key))
    n = rng.standard_normal(len(x))
    return (x + n * np.linalg.norm(x) / (np.linalg.norm(n) * 10 ** (db / 20))).astype(
        "float32"
    )


def features(x):
    _, _, z = stft(x, **FRAMES)
    power = abs(z) ** 2
    floor = np.maximum(np.percentile(power, 20, axis=1, keepdims=True), 1e-12)
    bins = np.digitize(10 * np.log10(np.maximum(power, 1e-12) / floor), BINS)
    return z, bins


def fit(rows):
    cross = np.zeros(len(BINS) + 1)
    energy = np.zeros_like(cross)
    for r in rows:
        x = read_audio(ROOT / r["path"])
        y = noisy(x, r["utt_id"], seed(r["utt_id"]) % 21)
        clean_z = stft(x, **FRAMES)[2]
        z, bins = features(y)
        cross += np.bincount(
            bins.ravel(), (clean_z * z.conj()).real.ravel(), minlength=len(cross)
        )
        energy += np.bincount(bins.ravel(), (abs(z) ** 2).ravel(), minlength=len(cross))
    return np.clip(cross / np.maximum(energy, 1e-12), 0.12, 1.0)


def enhance(x, weights):
    z, bins = features(x)
    return istft(z * weights[bins], **FRAMES)[1][: len(x)].astype("float32")


def external():
    control = ROOT / "data/ml/noise_controls"
    meta = ksd_subset()
    by_group = {}
    for row in meta:
        member = row["wav"].removeprefix("dataset/")
        file = (
            control
            / "ksd"
            / (hashlib.sha256(member.encode()).hexdigest()[:20] + ".wav")
        )
        if file.is_file():
            group = str(Path(member).parent / Path(member).stem.split("_")[0])
            try:
                duration = sf.info(file).duration
            except sf.LibsndfileError:
                print("Excluded undecodable KSD source", file.name, flush=True)
                continue
            if 2 <= duration <= 15 and row["text"].strip():
                by_group.setdefault(group, (member, file, row["text"]))
    ks = [
        (key, read_audio(file), ref)
        for key, file, ref in sorted(by_group.values(), key=lambda v: seed(v[0]))[:6]
    ]
    verify(GOLOS)
    table = pq.read_table(GOLOS).to_pylist()
    ru = []
    for i in sorted(range(len(table)), key=seed):
        x = read_audio(io.BytesIO(table[i]["audio"]["bytes"]))
        if 2 <= len(x) / 16000 <= 15 and table[i]["transcription"].strip():
            ru.append((f"golos-{i}", x, table[i]["transcription"]))
        if len(ru) == 6:
            break
    english = []
    for line in (ROOT / "data/asr/plans/test.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "single":
            s = r["sources"][0]
            english.append((s["utt_id"], s["path"], s["text_ls"]))
    en = [
        (key, read_audio(ROOT / path), ref)
        for key, path, ref in sorted(english, key=lambda v: seed(v[0]))[:6]
    ]
    assert all(len(v) == 6 for v in (ks, ru, en))
    return dict(zip(LANGS, (ks, ru, en)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with (
        ROOT / "data/ml/noise_followup_20261110/development_manifest.csv"
    ).open() as f:
        rows = list(csv.DictReader(f))

    def select(role, n):
        return [
            r
            for lang in LANGS
            for r in sorted(
                (
                    r
                    for r in rows
                    if r["language"] == lang
                    and r["split"] == role
                    and not r["excluded_reason"]
                    and 2 <= float(r["duration"]) <= 15
                ),
                key=lambda r: seed(r["utt_id"]),
            )[:n]
        ]

    train, val = select("train", 64), select("val", 4)
    assert (
        len(train) == 192
        and len(val) == 12
        and not ({r["group"] for r in train} & {r["group"] for r in val})
    )
    weights = fit(train)
    np.save(args.work / "gain.npy", weights)
    pipe = ConversationPipeline(Settings(cpu_threads=4))
    import torch

    torch.set_num_threads(4)
    cache = args.work / "predictions.jsonl"
    saved = {
        tuple(r["key"]): r
        for r in (
            map(json.loads, cache.read_text().splitlines()) if cache.exists() else []
        )
    }

    def store(key, data):
        record = dict(key=key, **data)
        with cache.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        saved[key] = record
        print(key, flush=True)
        return record

    def evaluate(stage, sets, models):
        metrics = {}
        for name in models:
            bins = defaultdict(list)
            for lang, units in sets.items():
                for uid, x, ref in units:
                    for condition in ("clean", "white-10"):
                        key = (stage, name, uid, condition)
                        if key not in saved:
                            y = x if condition == "clean" else noisy(x, uid, 10)
                            if name == "gain":
                                y = enhance(y, weights)
                            turns = pipe.transcribe(y, language="auto", speakers=1)[
                                "turns"
                            ]
                            hyp = " ".join(t["text"] for t in turns)
                            store(
                                key,
                                dict(
                                    hyp=hyp,
                                    labels=sorted({t["language"] for t in turns}),
                                ),
                            )
                        bins[lang, condition].append(counts(ref, saved[key]["hyp"]))
            metrics[name] = {
                lang + "_" + cond: pooled(rs) for (lang, cond), rs in bins.items()
            }
        return metrics

    val_sets = {
        l: [
            (r["utt_id"], read_audio(ROOT / r["path"]), r["text_raw"])
            for r in val
            if r["language"] == l
        ]
        for l in LANGS
    }
    valid = evaluate("validation", val_sets, ["bypass", "gain"])

    def noisy_wer(name):
        rs = [valid[name][l + "_white-10"] for l in LANGS]
        return sum(r["errors"] for r in rs) / sum(r["reference_words"] for r in rs)

    good_clean = all(
        valid["gain"][l + "_clean"]["wer"]
        <= valid["bypass"][l + "_clean"]["wer"] + 0.005
        for l in LANGS
    )
    selected = (
        "gain"
        if good_clean and noisy_wer("gain") <= 0.9 * noisy_wer("bypass")
        else "bypass"
    )
    decision = dict(validation=valid, selected=selected, train_n=192, val_n=12)
    (args.work / "selection.json").write_text(json.dumps(decision, indent=2))
    controls = external()
    tests = evaluate("controls", controls, list(dict.fromkeys(("bypass", selected))))
    results = []
    for (kid, k, kr), (rid, r, rr) in zip(controls["kk"], controls["ru"], strict=True):
        key = ("overlap", kid, rid, "white-10")
        if key not in saved:
            x, y = k / np.sqrt(np.mean(k * k)), r / np.sqrt(np.mean(r * r))
            onset = int(0.3 * min(len(x), len(y)))
            n = max(len(x), onset + len(y))
            sources = np.zeros((2, n), dtype="float32")
            sources[0, : len(x)] = x * 0.07
            sources[1, onset : onset + len(y)] = y * 0.07
            mixture = noisy(sources.sum(axis=0), kid + rid, 10)
            out = pipe.transcribe_two_voices(mixture)
            turns = out["turns"]
            errors, words = stream_errors((kr, rr), turns)
            if out.get("two_voices"):
                estimates = out["_source_audio"]
            else:
                # Report the acoustic separator even when the transcript gate uses the mixture.
                estimates = pipe.separator().split(mixture)[0]
            mix_score, pit = signal_scores(mixture, sources, estimates)
            store(
                key,
                dict(
                    errors=errors,
                    words=words,
                    emitted=bool(out.get("two_voices")),
                    si_sdri_db=pit - mix_score,
                ),
            )
        results.append(saved[key])
    signal = dict(
        pairs=len(results),
        errors=sum(t["errors"] for t in results),
        words=sum(t["words"] for t in results),
        emitted=sum(t["emitted"] for t in results),
        si_sdri_db=float(np.mean([t["si_sdri_db"] for t in results])),
    )
    (args.work / "external.json").write_text(
        json.dumps(dict(**decision, test=tests, overlap=signal), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
