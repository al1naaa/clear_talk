"""Plan, render and separate the ClearTalk ASR datasets.

    python -m cleartalk_asr.mixtures plan
    python -m cleartalk_asr.mixtures build --set train --shard 0 --num-shards 4 --device mps

Every recipe is planned up front with fixed seeds (``data/asr/plans/*.jsonl``) and rendered
deterministically, so the data can be rebuilt bit-for-bit. Each two-speaker mixture is
processed the way the app processes an upload: 16 kHz PCM16 → 8 kHz → frozen SepFormer →
per-track peak normalization → 8 kHz PCM16 WAV. The clean sources decide which separated
track belongs to which transcript; they are never model inputs.
"""

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import augment
from .paths import DATA, MIXTURES, ROOT, SEPARATOR

PLANS = DATA / "plans"
SYNTHETIC = ("white", "pink", "brown", "band")
MAX_MIXTURE_SECONDS = 20.0  # the app rejects longer uploads
SEED = 20261009
# Planned sizes (frozen in PIPELINE.md before any training).
SIZES = {"train_pairs": 8000, "train_single": 3000, "val_pairs": (200, 100), "val_single": 300,
         "test_pairs": (150, 150), "test_single": 400}
NOISE_GROUPS = {"train": {"train"}, "val": {"asr_val", "signal_val"}, "test": {"test"}}


# ----------------------------------------------------------------------------- step 4: cleaning
def _words(text):
    return [w.strip("'") for w in re.sub(r"[^a-z0-9' ]", " ", str(text).lower()).split() if w.strip("'")]


def clean(utterances: pd.DataFrame):
    """Return (training pool, evaluation pool, report). Rules are documented in PIPELINE.md."""
    report = {}
    train = utterances[utterances.role == "train"]
    report["train_total"] = len(train)
    train = train[~train.demo_speaker]
    report["after_excluding_demo_speakers"] = len(train)
    train = train[train.has_pc]
    report["after_requiring_punctuated_text"] = len(train)
    agree = [_words(a) == _words(b) for a, b in zip(train.text, train.text_ls)]
    train = train[agree]
    report["after_requiring_pc_words_equal_official"] = len(train)
    train = train[(train.duration >= 1.0) & (train.duration <= 16.5)]
    report["after_duration_1_to_16.5s"] = len(train)
    evaluation = utterances[utterances.role != "train"]
    report["evaluation_total"] = len(evaluation)
    return train.reset_index(drop=True), evaluation.reset_index(drop=True), report


# ----------------------------------------------------------------------------- planning
def _source(row):
    return {"utt_id": row.utt_id, "path": row.path, "speaker": row.speaker, "subset": row.subset,
            "duration": round(float(row.duration), 3), "text": row.text if isinstance(row.text, str) else None,
            "text_ls": row.text_ls}


def _noise(index, rng, groups):
    if index % 2 == 0:
        clips = [c for c in augment.noise_clips() if c["group"] in groups]
        return "esc10", clips[int(rng.integers(len(clips)))]["path"]
    return SYNTHETIC[(index // 2) % 4], None


def _offsets(durations, delay_fraction, delayed):
    offsets = [0.0, 0.0]
    offsets[delayed] = round(delay_fraction * min(durations), 3)
    return offsets


def duration_weights(train, evaluation):
    """Importance weights that make sampled training durations resemble validation/test ones."""
    bins = np.arange(0, 17.5, 1.0)
    eval_hist, _ = np.histogram(evaluation.duration.clip(upper=17), bins=bins, density=True)
    train_hist, _ = np.histogram(train.duration, bins=bins, density=True)
    ratio = np.where(train_hist > 0, eval_hist / np.maximum(train_hist, 1e-9), 0)
    index = np.clip(np.digitize(train.duration, bins) - 1, 0, len(ratio) - 1)
    weights = ratio[index]
    return weights / weights.sum()


def plan_train(train, evaluation, rng):
    weights = duration_weights(train, evaluation)
    rows = list(train.itertuples())
    recipes = []

    def draw(exclude_speaker=None):
        while True:
            if rng.random() < 0.5:
                row = rows[int(rng.choice(len(rows), p=weights))]
            else:
                row = rows[int(rng.integers(len(rows)))]
            if row.speaker != exclude_speaker:
                return row

    for index in range(SIZES["train_pairs"]):
        first = draw()
        second = draw(first.speaker)
        durations = [first.duration, second.duration]
        delay, delayed = float(rng.choice([0.0, 0.25, 0.5])), int(rng.integers(2))
        offsets = _offsets(durations, delay, delayed)
        if max(o + d for o, d in zip(offsets, durations)) > MAX_MIXTURE_SECONDS:
            offsets = [0.0, 0.0]  # sources are at most 16.5 s, so no delay always fits
        noise, noise_path = _noise(index, rng, NOISE_GROUPS["train"])
        recipes.append({"id": f"train-pair-{index:05d}", "set": "train", "kind": "pair",
                        "sources": [_source(first), _source(second)], "offsets_seconds": offsets,
                        "gains_db": [0.0, round(float(rng.uniform(-6, 6)), 2)],
                        "snr_db": round(float(rng.uniform(-2, 18)), 2), "noise": noise, "noise_path": noise_path,
                        "reverb": bool(rng.random() < 0.5), "seed": int(rng.integers(2 ** 31))})
    for index in range(SIZES["train_single"]):
        row = rows[int(rng.integers(len(rows)))]
        noisy = rng.random() < 0.5
        noise, noise_path = _noise(index, rng, NOISE_GROUPS["train"]) if noisy else ("none", None)
        recipes.append({"id": f"train-single-{index:05d}", "set": "train", "kind": "single",
                        "sources": [_source(row)], "offsets_seconds": [0.0], "gains_db": [0.0],
                        "snr_db": round(float(rng.uniform(5, 25)), 2), "noise": noise, "noise_path": noise_path,
                        "reverb": bool(rng.random() < 0.3), "seed": int(rng.integers(2 ** 31))})
    return recipes


def plan_eval(evaluation, role, rng):
    """Whole utterances, each used once per set; conditions cycle like the v4 evaluation."""
    subsets = {"val": ("dev-clean", "dev-other"), "test": ("test-clean", "test-other")}[role]
    counts = SIZES[f"{role}_pairs"]
    recipes = []
    used = set()
    for subset, count in zip(subsets, counts):
        pool = evaluation[(evaluation.subset == subset) & evaluation.duration.between(2.0, 12.0)]
        pool = pool.sample(frac=1.0, random_state=int(rng.integers(2 ** 31))).reset_index(drop=True)
        by_speaker = {s: list(g.itertuples()) for s, g in pool.groupby("speaker")}
        for index in range(count):
            speakers = sorted(s for s, rows in by_speaker.items() if any(r.utt_id not in used for r in rows))
            pair = rng.choice(speakers, size=2, replace=False)
            chosen = []
            for speaker in pair:
                row = next(r for r in by_speaker[speaker] if r.utt_id not in used)
                used.add(row.utt_id)
                chosen.append(row)
            n = len(recipes)
            noise, noise_path = _noise(n, rng, NOISE_GROUPS[role])
            delay = (0.0, 0.25, 0.5)[n % 3]
            recipes.append({"id": f"{role}-pair-{n:04d}", "set": role, "kind": "pair",
                            "sources": [_source(r) for r in chosen],
                            "offsets_seconds": _offsets([r.duration for r in chosen], delay, 1),
                            "gains_db": [0.0, round(float(rng.uniform(-6, 6)), 2)],
                            "snr_db": float((0, 5, 10, 15)[(n // 2) % 4]), "noise": noise, "noise_path": noise_path,
                            "reverb": bool(rng.random() < 0.5), "seed": int(rng.integers(2 ** 31))})
    clean_subset = {"val": "dev-clean", "test": "test-clean"}[role]
    pool = evaluation[(evaluation.subset == clean_subset) & (evaluation.duration <= MAX_MIXTURE_SECONDS)
                      & ~evaluation.utt_id.isin(used)]
    for n, row in enumerate(pool.sample(SIZES[f"{role}_single"], random_state=int(rng.integers(2 ** 31))).itertuples()):
        recipes.append({"id": f"{role}-single-{n:04d}", "set": role, "kind": "single", "sources": [_source(row)],
                        "offsets_seconds": [0.0], "gains_db": [0.0], "snr_db": None, "noise": "none",
                        "noise_path": None, "reverb": False, "seed": int(rng.integers(2 ** 31))})
    return recipes


def write_plans():
    utterances = pd.read_csv(DATA / "utterances.csv.gz", dtype={"speaker": str, "chapter": str})
    train, evaluation, report = clean(utterances)
    rng = np.random.default_rng(SEED)
    PLANS.mkdir(parents=True, exist_ok=True)
    plans = {"train": plan_train(train, evaluation, rng), "val": plan_eval(evaluation, "val", rng),
             "test": plan_eval(evaluation, "test", rng)}
    for name, recipes in plans.items():
        path = PLANS / f"{name}.jsonl"
        if path.exists():
            raise FileExistsError(f"{path} is frozen; delete it deliberately to re-plan")
        path.write_text("".join(json.dumps(r) + "\n" for r in recipes))
    (PLANS / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


# ----------------------------------------------------------------------------- building
def si_sdr(estimate, target):
    estimate = estimate - estimate.mean()
    target = target - target.mean()
    projection = (estimate @ target) / max(float(target @ target), 1e-12) * target
    noise = estimate - projection
    return float(10 * np.log10(max(float(projection @ projection), 1e-12) / max(float(noise @ noise), 1e-12)))


def _source_fields(source):
    """Source metadata for a track row; its audio path is kept as ``source_path``."""
    fields = {key: value for key, value in source.items() if key != "path"}
    fields["source_path"] = source["path"]
    return fields


def _peak_pcm16(signal):
    signal = np.asarray(signal, dtype=np.float32).copy()
    peak = float(np.max(np.abs(signal)))
    if peak > 1e-6:
        signal *= 0.95 / peak
    return signal


def _claim(claims, recipe_id):
    """Atomic work claim, so several workers (MPS + CPU) can share one plan."""
    try:
        os.close(os.open(claims / recipe_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def build(set_name, worker, device, threads, release_stale_claims=False):
    import torch
    from speechbrain.inference.separation import SepformerSeparation

    torch.set_num_threads(threads)
    recipes = [json.loads(line) for line in (PLANS / f"{set_name}.jsonl").read_text().splitlines()]
    out_dir = MIXTURES / set_name
    claims = out_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    done = set(load_tracks(set_name).get("recipe_id", pd.Series(dtype=str)))
    if release_stale_claims:
        # A claim without a finished row belongs to an interrupted worker.
        for claim in claims.iterdir():
            if claim.name not in done:
                claim.unlink()
    rows_path = out_dir / f"tracks-{worker}.jsonl"
    separator = None
    started, processed = time.monotonic(), 0
    with rows_path.open("a") as rows_file:
        for recipe in recipes:
            if recipe["id"] in done or not _claim(claims, recipe["id"]):
                continue
            pcm, clean_sources, spans = augment.render(recipe)
            mixture16 = pcm.astype(np.float32) / 32768
            rows = []
            if recipe["kind"] == "single":
                path = out_dir / f"{recipe['id']}.wav"
                sf.write(path, pcm, augment.RATE, subtype="PCM_16")
                source = recipe["sources"][0]
                rows.append({**_source_fields(source), "track_id": recipe["id"], "path": path.relative_to(ROOT).as_posix(),
                             "sample_rate": augment.RATE, "si_sdr_db": None, "span": spans[0]})
            else:
                if separator is None:
                    separator = SepformerSeparation.from_hparams(
                        source=str(SEPARATOR), savedir=str(SEPARATOR), run_opts={"device": device})
                    separator.mods.eval()
                mixture8 = resample_poly(mixture16, 8000, augment.RATE).astype(np.float32)
                with torch.inference_mode():
                    separated = separator.separate_batch(torch.from_numpy(mixture8)[None].to(device))[0].cpu().numpy()
                if separated.shape != (len(mixture8), 2) or not np.isfinite(separated).all():
                    raise RuntimeError(f"Invalid separator output for {recipe['id']}")
                references = [resample_poly(clean_sources[:, i], 8000, augment.RATE)[:len(mixture8)] for i in range(2)]
                scores = [[si_sdr(separated[:, k], references[i]) for k in range(2)] for i in range(2)]
                direct = scores[0][0] + scores[1][1]
                swapped = scores[0][1] + scores[1][0]
                assignment = [0, 1] if direct >= swapped else [1, 0]
                if set_name != "train":
                    sf.write(out_dir / f"{recipe['id']}-mix.wav", pcm, augment.RATE, subtype="PCM_16")
                for i, k in enumerate(assignment):
                    path = out_dir / f"{recipe['id']}-{i}.wav"
                    sf.write(path, _peak_pcm16(separated[:, k]), 8000, subtype="PCM_16")
                    rows.append({**_source_fields(recipe["sources"][i]), "track_id": f"{recipe['id']}-{i}",
                                 "path": path.relative_to(ROOT).as_posix(), "sample_rate": 8000,
                                 "si_sdr_db": round(scores[i][k], 3), "other_si_sdr_db": round(scores[1 - i][k], 3),
                                 "span": spans[i], "separated_index": k})
            for row in rows:
                row.update(recipe_id=recipe["id"], set=set_name, kind=recipe["kind"], noise=recipe["noise"],
                           snr_db=recipe["snr_db"], reverb=recipe["reverb"],
                           offsets_seconds=recipe["offsets_seconds"], gains_db=recipe["gains_db"],
                           mixture_seconds=round(len(pcm) / augment.RATE, 3))
                rows_file.write(json.dumps(row) + "\n")
            rows_file.flush()
            processed += 1
            if processed % 100 == 0:
                print(f"{set_name} {worker}: {processed} recipes in {time.monotonic() - started:.0f}s", flush=True)
    print(f"{set_name} {worker}: finished {processed} new recipes in {time.monotonic() - started:.0f}s", flush=True)


def repair_track_paths(set_name):
    """One-time fix for rows written before 2026-10-09 03:40, whose ``path`` held the clean source."""
    prefix = (MIXTURES / set_name).relative_to(ROOT).as_posix() + "/"
    for rows_path in sorted((MIXTURES / set_name).glob("tracks-*.jsonl")):
        fixed = []
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            if not row["path"].startswith(prefix):
                row["source_path"] = row["path"]
                row["path"] = f"{prefix}{row['track_id']}.wav"
            fixed.append(json.dumps(row) + "\n")
        rows_path.write_text("".join(fixed))


def load_tracks(set_name) -> pd.DataFrame:
    rows = []
    prefix = (MIXTURES / set_name).relative_to(ROOT).as_posix() + "/"
    for path in sorted((MIXTURES / set_name).glob("tracks-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if any(not row["path"].startswith(prefix) or not (ROOT / row["path"]).is_file() for row in rows):
        raise ValueError(f"{set_name}: a track row does not point to its separated WAV file")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--set", choices=("train", "val", "test"), required=True)
    build_parser.add_argument("--worker", default="w0", help="unique name per concurrent worker")
    build_parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    build_parser.add_argument("--threads", type=int, default=2)
    build_parser.add_argument("--release-stale-claims", action="store_true",
                              help="only when no other worker is running: free claims of unfinished recipes")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(write_plans(), indent=2))
    else:
        build(args.set, args.worker, args.device, args.threads, args.release_stale_claims)


if __name__ == "__main__":
    main()
