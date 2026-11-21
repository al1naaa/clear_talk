"""Choose a checkpoint using only automatic, three-language validation transcripts.

Render identical views for every candidate and keep the incumbent unless a
candidate reduces macro noisy WER by at least 10% relative, the paired 95% interval is below zero,
and stays within 0.5 percentage points of each language's clean WER.
"""

import argparse
import hashlib
import tempfile
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import Settings
from backend.conversation import ConversationPipeline
from backend.multilingual import restricted_language
from cleartalk_ml.text import counts, pooled
from scripts.train_three_language_asr import (
    CHECKPOINTS,
    LANGS,
    babble,
    contaminate,
    rank,
    split_rows,
    wave,
)

INCUMBENT = ROOT / "models/multilingual/whisper-large-cleartalk-multilingual"


def convert(step, work):
    from ctranslate2.converters.transformers import WhisperLoader
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration

    out = work / f"ct2-{step:03d}"
    if (out / "complete").exists():
        return out
    model = WhisperForConditionalGeneration.from_pretrained(
        ROOT / "models/training-round2", dtype=torch.float32
    )
    model = PeftModel.from_pretrained(
        model, work / f"step-{step:03d}"
    ).merge_and_unload()
    spec = WhisperLoader().get_model_spec(model)
    spec.register_vocabulary(json.loads((INCUMBENT / "vocabulary.json").read_text()))
    spec.validate()
    spec.optimize(quantization="float16")
    out.mkdir(exist_ok=True)
    spec._serialize(str(out / "model.bin"))
    for f in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "vocabulary.json",
    ):
        shutil.copyfile(INCUMBENT / f, out / f)
    (out / "complete").write_text(f"{step}\n")
    return out


def validate_sets():
    samples = split_rows("val", 12, "asr-val")
    return {
        l: [(r["utt_id"], wave(r), r["text_raw"]) for r in rs]
        for l, rs in samples.items()
    }


def noise_pool():
    import csv

    with (
        ROOT / "data/ml/noise_followup_20261110/development_manifest.csv"
    ).open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["split"] == "train"
            and not r["excluded_reason"]
            and 3 <= float(r["duration"]) <= 15
        ]
    # Training babble used the first 160 ranked non-foreground items. Use the opposite end.
    pool = sorted(rows, key=lambda r: rank(r["utt_id"], "babble"))[-64:]
    train = split_rows("train", 192, "asr-fit")
    foreground = {r["utt_id"] for rs in train.values() for r in rs}
    return [r for r in pool if r["utt_id"] not in foreground]


def condition_audio(x, uid, condition, background):
    if condition == "clean":
        return x
    rng = np.random.default_rng(int(rank(uid, condition)[:16], 16))
    noise = (
        rng.standard_normal(len(x))
        if condition == "white-10"
        else babble(len(x), background, rng)
    )
    return contaminate(x, noise, 10)


class AuditPipeline(ConversationPipeline):
    def __init__(self, folder, threads, detector_cache=None):
        super().__init__(Settings(cpu_threads=threads))
        self.folder = folder
        self.used_candidate = False
        self.detector_cache = detector_cache
        self.detector_votes = []

    def recognizer(self, key):
        if key == "cleartalk-large":
            self.used_candidate = True
            if key not in self._recognizers:
                self._recognizers[key] = WhisperModel(
                    str(self.folder),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.settings.cpu_threads,
                    local_files_only=True,
                )
            return self._recognizers[key]
        model = super().recognizer(key)
        if (
            self.detector_cache
            and key == "turbo"
            and not getattr(model, "_cached_language", False)
        ):
            original = model.detect_language
            cache = self.detector_cache
            cache.mkdir(exist_ok=True, parents=True)

            def detect(audio, **options):
                digest = hashlib.sha256(
                    audio.tobytes() + repr(sorted(options.items())).encode()
                ).hexdigest()
                path = cache / (digest + ".json")
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    result = original(audio, **options)
                    with tempfile.NamedTemporaryFile(
                        "w", dir=cache, delete=False
                    ) as out:
                        json.dump(result, out)
                    os.replace(out.name, path)
                self.detector_votes.append(restricted_language(result[2])[0])
                return result

            model.detect_language = detect
            model._cached_language = True
        return model


def read_rows(path):
    if not path.exists():
        return {}
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    keyed = {(r["uid"], r["condition"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate result key")
    return keyed


def score_sets(sets, folder, cache_dir, step, threads):
    torch.set_num_threads(threads)
    cache_dir.mkdir(exist_ok=True, parents=True)
    path = cache_dir / f"{step}.jsonl"
    saved = read_rows(path)
    baseline = read_rows(cache_dir / "0.jsonl") if step else {}
    pipe = AuditPipeline(folder, threads, cache_dir / "turbo-detector")
    bg = noise_pool()
    for lang, items in sets.items():
        for uid, x, ref in items:
            for c in ("clean", "white-10", "babble-10"):
                key = uid, c
                if key in saved:
                    continue
                if key in baseline and not baseline[key]["used_candidate"]:
                    row = baseline[key]
                else:
                    pipe.used_candidate = False
                    pipe.detector_votes.clear()
                    result = pipe.transcribe(
                        condition_audio(x, uid, c, bg), language="auto", speakers=1
                    )
                    turns = result["turns"]
                    hyp = " ".join(t["text"] for t in turns)
                    row = dict(
                        uid=uid,
                        language=lang,
                        condition=c,
                        used_candidate=pipe.used_candidate,
                        exact_lid=sorted({t["language"] for t in turns}) == [lang],
                        **counts(ref, hyp),
                    )
                    if cache_dir.name == "holdout":
                        row.update(
                            hypothesis=hyp, detector_votes=list(pipe.detector_votes)
                        )
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                saved[key] = row
                print(step, lang, uid, c, len(saved), flush=True)
    expected = {
        (uid, c)
        for rs in sets.values()
        for uid, _, _ in rs
        for c in ("clean", "white-10", "babble-10")
    }
    assert set(saved) == expected
    return saved


def summary(rows):
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["language"], row["condition"]].append(row)
    result = {}
    for (language, condition), rows in groups.items():
        group = dict(
            n=len(rows), exact_lid=sum(r["exact_lid"] for r in rows), **pooled(rows)
        )
        if all("detector_votes" in r for r in rows):
            votes = [vote for row in rows for vote in row["detector_votes"]]
            group.update(
                detector_calls=len(votes),
                detector_correct=sum(v == language for v in votes),
                clips_with_detector_calls=sum(
                    bool(row["detector_votes"]) for row in rows
                ),
            )
        result[language + "_" + condition] = group
    return result


def macro_noisy(summary):
    return np.mean(
        [summary[l + "_" + c]["wer"] for l in LANGS for c in ("white-10", "babble-10")]
    )


def paired_interval(base, new):
    rng = np.random.default_rng(20261122)
    differences = []
    for l in LANGS:
        ids = sorted({uid for (uid, c), r in base.items() if r["language"] == l})
        draws = rng.integers(len(ids), size=(4000, len(ids)))
        for c in ("white-10", "babble-10"):
            errors = np.array(
                [
                    sum(
                        new[u, c][k] - base[u, c][k]
                        for k in ("insertions", "deletions", "substitutions")
                    )
                    for u in ids
                ]
            )
            words = np.array([base[u, c]["reference_words"] for u in ids])
            differences.append(errors[draws].sum(axis=1) / words[draws].sum(axis=1))
    return np.quantile(np.mean(differences, axis=0), [0.025, 0.975]).tolist()


def select(work, cache):
    by_step = {step: read_rows(cache / f"{step}.jsonl") for step in (0, *CHECKPOINTS)}
    assert len(by_step[0]) == 108 and all(
        rs.keys() == by_step[0].keys() for rs in by_step.values()
    )
    scores = {step: summary(rs) for step, rs in by_step.items()}
    decisions = {}
    eligible = [0]
    for step, rs in by_step.items():
        macro = float(macro_noisy(scores[step]))
        ci = paired_interval(by_step[0], rs)
        clean_safe = all(
            scores[step][l + "_clean"]["wer"] <= scores[0][l + "_clean"]["wer"] + 0.005
            for l in LANGS
        )
        decisions[step] = dict(
            macro_noisy_wer=macro, delta_ci95=ci, clean_safe=clean_safe
        )
        if step and clean_safe and ci[1] < 0 and macro <= 0.90 * macro_noisy(scores[0]):
            eligible.append(step)
    winner = min(eligible, key=lambda step: decisions[step]["macro_noisy_wer"])
    clean_safe_steps = [step for step in CHECKPOINTS if decisions[step]["clean_safe"]]
    best = (
        min(clean_safe_steps, key=lambda step: decisions[step]["macro_noisy_wer"])
        if clean_safe_steps
        else None
    )
    result = dict(
        selected_step=winner,
        best_clean_safe_trained_step=best,
        promotion_rule="At least 10% relative macro noisy WER reduction, paired 95% interval below zero and clean WER within 0.5 pp in every language.",
        steps=decisions,
        metrics=scores,
    )
    (work / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=(0, *CHECKPOINTS))
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--work", type=Path, default=ROOT / "models/pitch-asr")
    opt = parser.parse_args()
    cache = opt.work / "validation"
    if opt.select:
        print(select(opt.work, cache)["selected_step"])
    else:
        assert opt.step is not None
        cache.mkdir(parents=True, exist_ok=True)
        with FileLock(cache / f"{opt.step}.lock"):
            folder = convert(opt.step, opt.work) if opt.step else INCUMBENT
            score_sets(validate_sets(), folder, cache, opt.step, opt.threads)
